/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/function.h>

#include <algorithm>
#include <cmath>
#include <iterator>
#include <memory>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "flashinfer/trtllm/common/cudaUtils.h"
#include "flashinfer/trtllm/fused_moe/DevKernel.h"
#include "flashinfer/trtllm/fused_moe/RoutingKernel.h"
#include "flashinfer/trtllm/fused_moe/runner.h"
#include "tvm_ffi_utils.h"

// Routing/workspace support shared by the full TRTLLM launcher and the smaller
// Prims BF16 module. This must not instantiate or call the cubin GEMM Runner.
namespace flashinfer {
namespace btg = batchedGemm::trtllm::gen;
using tensorrt_llm::kernels::trtllmgen_moe::MoE::ActivationType;
using tensorrt_llm::kernels::trtllmgen_moe::Routing::RoutingMethodType;
using tvm::ffi::Array;
using tvm::ffi::Optional;

enum class RoutingInputMode {
  FromLogits,          // Mode 1: Compute routing from logits
  PackedPrecomputed,   // Mode 2: Pre-computed with packed (score << 16 | id) format
  UnpackedPrecomputed  // Mode 3: Pre-computed with separate topk_ids and topk_weights
};

// Validate routing_replay_out tensor properties.
// NOTE: dim0 is only bounded from below. The routing kernels write one replay row per
// token unconditionally (DeepSeek launches numBlocks == num_tokens and writes row
// blockIdx.x; the custom and llama4 kernels write row tokenIdx), so a buffer with fewer
// rows than tokens is written past its end. Oversized buffers stay legal: with CUDA
// graphs the buffer is pre-allocated at maximum batch size and reused across steps with
// varying num_tokens.
inline void validate_routing_replay_out(TensorView const& replay, TensorView const& hidden_states,
                                        int64_t top_k) {
  TVM_FFI_ICHECK(replay.device().device_type == kDLCUDA)
      << "routing_replay_out must be a CUDA tensor";
  TVM_FFI_ICHECK(replay.device().device_id == hidden_states.device().device_id)
      << "routing_replay_out must be on the same device as hidden_states";
  TVM_FFI_ICHECK(replay.ndim() == 2) << "routing_replay_out must be 2D [num_tokens, top_k]";
  TVM_FFI_ICHECK(replay.size(1) == top_k) << "routing_replay_out dim1 must equal top_k";
  TVM_FFI_ICHECK(replay.size(0) >= hidden_states.size(0))
      << "routing_replay_out dim0 must be >= num_tokens (" << hidden_states.size(0) << "), got "
      << replay.size(0) << "; the routing kernel writes one replay row per token";
  TVM_FFI_ICHECK((replay.dtype() == DLDataType{kDLInt, 16, 1}))
      << "routing_replay_out must be int16 dtype";
  TVM_FFI_ICHECK(replay.IsContiguous())
      << "routing_replay_out must be contiguous (packed row-major)";
}

inline ActivationType validateAndCastActivationType(int64_t act_type) {
  TVM_FFI_ICHECK(act_type >= 0 && act_type < static_cast<int64_t>(ActivationType::InvalidType))
      << "Invalid activation type: " << act_type;
  return static_cast<ActivationType>(act_type);
}

// Utility function to compute the next power of two
inline int32_t nextPowerOfTwo(float value) {
  int32_t n = static_cast<int32_t>(std::ceil(value));
  if (n <= 1) return 1;

  // If n is already a power of 2, return it
  if ((n & (n - 1)) == 0) return n;

  // Find the next power of 2
  n--;
  n |= n >> 1;
  n |= n >> 2;
  n |= n >> 4;
  n |= n >> 8;
  n |= n >> 16;
  n++;

  return n;
}

inline std::set<int32_t> computeSelectedTileN(std::vector<int32_t> const& supported_tile_nums,
                                              int64_t const num_tokens, int64_t const top_k,
                                              int64_t const num_local_experts) {
  TVM_FFI_ICHECK(!supported_tile_nums.empty()) << "supported_tile_nums must not be empty.";
  float const avg_tokens_per_expert = static_cast<float>(num_tokens * top_k) / num_local_experts;
  // NOTE: This differs from Python AutoTuner bucketing:
  // - AutoTuner maps raw num_tokens with last_positive_power_of_2 (round-down).
  // - Here we map derived avg_tokens_per_expert and use nextPowerOfTwo (round-up).
  // Because they round different quantities in different directions, cache bucket and runtime
  // tile candidates can diverge; launcher-side tactic resolution handles that mismatch.
  // assume supported_tile_nums is sorted
  int32_t tile_tokens_dim = std::clamp(nextPowerOfTwo(avg_tokens_per_expert),
                                       supported_tile_nums.front(), supported_tile_nums.back());
  auto it = std::find(supported_tile_nums.begin(), supported_tile_nums.end(), tile_tokens_dim);
  FLASHINFER_CHECK(
      it != supported_tile_nums.end(), "computeSelectedTileN expected exact tile ", tile_tokens_dim,
      " in supported_tile_nums (size=", supported_tile_nums.size(),
      "). Please keep supported_tile_nums as a dense power-of-2 ladder for this launcher.");

  // Candidate tile set centered on the heuristic tile.
  // This function returns nearby candidates (not a single final tile):
  //   center, +1, +2, and -1 neighbors when available.
  // Final tile choice is made later (autotuner-provided tile if valid, otherwise fallback policy).
  std::set<int32_t> selected_tile_nums;
  selected_tile_nums.insert(tile_tokens_dim);
  if (std::next(it) != supported_tile_nums.end()) {
    selected_tile_nums.insert(*std::next(it));
    if (std::next(std::next(it)) != supported_tile_nums.end()) {
      selected_tile_nums.insert(*std::next(std::next(it)));
    }
  }
  if (it != supported_tile_nums.begin()) {
    selected_tile_nums.insert(*std::prev(it));
  }

  return selected_tile_nums;
}

inline int64_t selectDefaultTileN(std::vector<int32_t> const& supported_tile_nums,
                                  int64_t const num_tokens, int64_t const top_k,
                                  int64_t const num_local_experts) {
  auto selected = computeSelectedTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
  TVM_FFI_ICHECK(!selected.empty()) << "No selected tile_N candidates for current MoE input.";
  return *selected.begin();
}

// Resolve the (tile_N, config) pair passed from Python side, applying fallback logic
// when tile_N is -1.
inline std::pair<int64_t, int64_t> resolveMoeTileAndConfig(
    Array<int64_t> const& config_index, std::vector<int32_t> const& supported_tile_nums,
    int64_t const num_tokens, int64_t const top_k, int64_t const num_local_experts) {
  // Python side convention: tactic is [tile_N, config]
  TVM_FFI_ICHECK(config_index.size() == 2)
      << "Invalid tactic, expected to be [tile_N, config], but got array of size "
      << config_index.size();
  const int64_t tile_N = config_index[0];
  const int64_t config = config_index[1];

  if (tile_N == -1 || config == -1) {
    // Use fallback tactic
    auto const default_tile_N =
        selectDefaultTileN(supported_tile_nums, num_tokens, top_k, num_local_experts);
    return {default_tile_N, -1};
  }

  return {tile_N, config};
}

// Validate the FC1 bias tensor against the selected BiasType. Currently only
// BiasType::None and BiasType::Mn are exercised from the flashinfer MoE path
inline void check_gemm1_bias_mn(Optional<TensorView> const& gemm1_bias,
                                batchedGemm::gemm::BiasType bias_type, int32_t num_tokens,
                                int32_t top_k, int32_t intermediate_size,
                                bool allow_bias_with_none_type = false) {
  if (bias_type == batchedGemm::gemm::BiasType::None) {
    TVM_FFI_ICHECK(allow_bias_with_none_type || !gemm1_bias.has_value())
        << "gemm1_bias must not be provided when gemm1_bias_type is None.";
    return;
  }
  TVM_FFI_ICHECK(bias_type == batchedGemm::gemm::BiasType::Mn)
      << "flashinfer MoE only supports gemm1_bias_type in {None, Mn}; got "
      << static_cast<int64_t>(bias_type);
  TVM_FFI_ICHECK(gemm1_bias.has_value())
      << "gemm1_bias must be provided when gemm1_bias_type is Mn";
  auto const& bias = gemm1_bias.value();
  TVM_FFI_ICHECK_EQ(bias.dtype(), dl_bfloat16) << "gemm1_bias must be bfloat16.";
  TVM_FFI_ICHECK_EQ(bias.ndim(), 3)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
  TVM_FFI_ICHECK_EQ(bias.size(0), num_tokens)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
  TVM_FFI_ICHECK_EQ(bias.size(1), top_k)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
  TVM_FFI_ICHECK_EQ(bias.size(2), 2 * intermediate_size)
      << "gemm1_bias must have shape [num_tokens, top_k, 2 * intermediate_size].";
}

class MoeRoutingLauncher {
 protected:
  Optional<TensorView> routing_logits;
  Optional<TensorView> routing_bias;
  TensorView hidden_states;
  TensorView gemm1_weights;
  Optional<TensorView> gemm1_bias;
  Optional<TensorView> output1_scales_scalar;
  Optional<TensorView> output1_scales_gate_scalar;
  TensorView gemm2_weights;
  Optional<TensorView> output2_scales_scalar;
  Optional<TensorView> per_token_scales;
  Tensor per_token_scales_fc2;
  bool use_per_channel_scaling_gemm1{false};
  bool use_per_channel_scaling_gemm2{false};

  int64_t tile_tokens_dim{};
  int64_t routing_method_type{};
  bool use_shuffled_weight{};
  batchedGemm::gemm::MatrixLayout weight_layout{batchedGemm::gemm::MatrixLayout::MajorK};
  batchedGemm::gemm::BiasType gemm1_bias_type{batchedGemm::gemm::BiasType::None};

  std::tuple<int, int> device_version;
  std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs> args;
  tensorrt_llm::kernels::trtllmgen_moe::MoE::MoEWorkspace workspace;

  btg::Dtype mDtypeAct{btg::Dtype::Bfloat16};
  btg::Dtype mDtypeWeights{btg::Dtype::Bfloat16};
  btg::Dtype mRoutingBiasDtype{
      btg::Dtype::Bfloat16};  // Dtype for expert weights in routing, based on routing bias
  btg::Dtype mRoutingLogitsDtype{btg::Dtype::Bfloat16};
  bool norm_topk_prob{true};
  ActivationType activation_type{ActivationType::Swiglu};

  // Optional routing replay output: [num_tokens, top_k] int16 tensor
  Optional<TensorView> routing_replay_out;

  int64_t intermediate_size_factor{2};

 public:
  // Constructor that initializes all TensorView members
  MoeRoutingLauncher(const Optional<TensorView>& routing_logits,
                     const Optional<TensorView>& routing_bias, const TensorView& hidden_states,
                     const TensorView& gemm1_weights, const Optional<TensorView>& gemm1_bias,
                     const Optional<TensorView>& output1_scales_scalar,
                     const Optional<TensorView>& output1_scales_gate_scalar,
                     const TensorView& gemm2_weights,
                     const Optional<TensorView>& output2_scales_scalar,
                     const Optional<TensorView>& per_token_scales,
                     RoutingInputMode routing_input_mode = RoutingInputMode::FromLogits)
      : routing_input_mode_(routing_input_mode),
        routing_logits(routing_logits),
        routing_bias(routing_bias),
        hidden_states(hidden_states),
        gemm1_weights(gemm1_weights),
        gemm1_bias(gemm1_bias),
        output1_scales_scalar(output1_scales_scalar),
        output1_scales_gate_scalar(output1_scales_gate_scalar),
        gemm2_weights(gemm2_weights),
        output2_scales_scalar(output2_scales_scalar),
        per_token_scales(per_token_scales),
        tile_tokens_dim{},
        routing_method_type{},
        use_shuffled_weight{},
        weight_layout{batchedGemm::gemm::MatrixLayout::MajorK},
        mDtypeAct{btg::Dtype::Bfloat16},
        mDtypeWeights{btg::Dtype::Bfloat16},
        activation_type{ActivationType::Swiglu},
        intermediate_size_factor{2} {}

 public:
  void set_routing_replay_out(const Optional<TensorView>& replay_out) {
    routing_replay_out = replay_out;
  }

 protected:
  // Initialize common data necessary for later.
  // May throw exception from TVM_FFI_ICHECK.
  void init_common(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
                   int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
                   int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
                   bool norm_topk_prob = true);

  // Routing logits [num_tokens, num_experts]
  void check_routing_logits() const {
    if (routing_logits.has_value()) {
      // Check shape
      TVM_FFI_ICHECK_EQ(routing_logits.value().ndim(), 2) << "routing_logits must be 2D.";
      TVM_FFI_ICHECK_EQ(routing_logits.value().size(0), hidden_states.size(0))
          << "routing_logits and hidden_states must have the same number of tokens.";
      TVM_FFI_ICHECK_EQ(routing_logits.value().size(1), args->num_experts)
          << "routing_logits dim1 must match num_experts.";

      // Check dtype
      TVM_FFI_ICHECK(routing_logits.value().dtype() == dl_float32 ||
                     routing_logits.value().dtype() == dl_bfloat16)
          << "routing_logits must be float or bfloat16.";
    }
  }

  // Empty placeholder tensors may still carry a non-null data_ptr.
  static bool has_precomputed(TensorView const& tensor) {
    return tensor.ndim() == 2 && tensor.size(0) > 0;
  }

  static btg::Dtype expert_weights_dtype(TensorView const& weights) {
    return weights.dtype() == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  }

  bool is_unpacked_routing() const {
    return routing_input_mode_ == RoutingInputMode::UnpackedPrecomputed;
  }

  int32_t* unpacked_expert_ids(TensorView const& indices) const {
    return is_unpacked_routing() ? static_cast<int32_t*>(const_cast<void*>(indices.data_ptr()))
                                 : nullptr;
  }

  virtual int32_t* precomputed_expert_ids() const { return nullptr; }

  RoutingInputMode routing_input_mode_;

  // Routing bias [num_experts]
  void check_routing_bias_shape() const {
    if (routing_bias.has_value()) {
      TVM_FFI_ICHECK(routing_bias.value().dtype() == dl_bfloat16 ||
                     routing_bias.value().dtype() == dl_float32)
          << "routing_bias must be bfloat16 or float.";
      TVM_FFI_ICHECK_EQ(routing_bias.value().ndim(), 1) << "routing_bias must be 1D.";
      TVM_FFI_ICHECK_EQ(routing_bias.value().size(0), args->num_experts)
          << "routing_bias has incorrect shape.";
    }
  }

  // Hidden states [num_tokens, hidden_size]
  void check_hidden_states_shape() const {
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
    TVM_FFI_ICHECK_EQ(hidden_states.size(1), args->intermediate_size)
        << "hidden_states has incorrect shape.";
  }

  // GEMM2's N, mirroring getGemm2OutputHiddenSize() in trtllm_fused_moe_runner.cu:
  // hidden_size_output narrows the GEMM only when the caller also supplied valid dims. Without
  // them a caller may hand in an `output` narrower than hidden_size (the pre-existing NVFP4
  // contract) while the FC2 weights still describe the full padded hidden_size, so GEMM2 keeps
  // the full N there and finalize truncates instead. Tactic selection must use the same value.
  //
  // With valid dims this is roundUp(valid_hidden_size, 128) -- see moe_hidden_size_output() -- and
  // is deliberately WIDER than the caller-visible output (valid_hidden_size). GPT-OSS: 2944 here,
  // 2880 output width.
  int32_t gemm2_output_hidden_size() const {
    return args->valid_hidden_size.has_value()
               ? args->hidden_size_output.value_or(args->hidden_size)
               : args->hidden_size;
  }

  // GEMM1 or GEMM2 weights [num_experts, M, K] or [num_experts, K/block_k, M, block_k]
  //
  // Only called for the launchers whose gemm2 weights actually follow `weight_layout`. That
  // member describes GEMM1; on the FP4 / MXINT4 routed paths GEMM2's weights do not share it
  // (they arrive 3-D where weight_layout says BlockMajorK), so calling this for their gemm2
  // would reject valid inputs. See the PR discussion for why the reviewer-requested extension
  // to those launchers needs a per-GEMM layout first.
  void check_weights_shape(std::string which_weights) const {
    TensorView weights = (which_weights == "gemm1") ? gemm1_weights : gemm2_weights;
    if (which_weights != "gemm1" && which_weights != "gemm2") {
      TVM_FFI_LOG_AND_THROW(InternalError) << "Internal error: which_weights = " << which_weights;
    }

    int64_t Mn = 0, K = 0;
    if (weight_layout == batchedGemm::gemm::MatrixLayout::MajorK) {
      // MajorK [num_experts, M, K]
      TVM_FFI_ICHECK_EQ(weights.ndim(), 3) << which_weights << " weights must be [experts, M, K].";
      Mn = weights.size(1);
      K = weights.size(2);
    } else if (weight_layout == batchedGemm::gemm::MatrixLayout::BlockMajorK) {
      // BlockMajorK [num_experts, K/block_k, M, block_k]
      TVM_FFI_ICHECK_EQ(weights.ndim(), 4)
          << which_weights << " weights must be [experts, K/block_k, M, block_k].";
      Mn = weights.size(2);
      int64_t block_k = weights.size(3);
      K = weights.size(1) * block_k;
    } else {
      TVM_FFI_LOG_AND_THROW(NotImplementedError)
          << "Unsupported weight_layout: " << (int)weight_layout;
    }
    if (which_weights == "gemm1") {
      // Gated MoE activations (e.g. Swiglu/Geglu) pack gate+up projections in GEMM1,
      // so Mn = 2 * intermediate_size and must be even.
      if (intermediate_size_factor == 2) {
        TVM_FFI_ICHECK_EQ(Mn % 2, 0) << which_weights << " weights Mn dimension must be even.";
      }
      // Non-gated activations (e.g. Relu2) use a single projection in GEMM1,
      // so Mn = intermediate_size. This check covers both gated and non-gated cases.
      TVM_FFI_ICHECK_EQ(args->intermediate_size * intermediate_size_factor, Mn)
          << "intermediate_size has incorrect shape.";
      TVM_FFI_ICHECK_EQ(K, hidden_states.size(1))
          << which_weights << " weights K dimension must be equal to hidden_size.";
    } else if (which_weights == "gemm2") {
      // The FC2 weights are laid out for GEMM2's N -- roundUp(valid_hidden_size, 128) when valid
      // dims are given -- NOT for the (narrower) caller output width; see
      // gemm2_output_hidden_size(). Comparing against the output width instead would reject
      // correctly shaped GPT-OSS weights (2944 vs 2880).
      TVM_FFI_ICHECK_EQ(Mn, gemm2_output_hidden_size())
          << which_weights << " weights M dimension must be equal to GEMM2's output hidden size.";
      // GEMM2 always consumes the post-activation hidden of size intermediate_size.
      TVM_FFI_ICHECK_EQ(K, args->intermediate_size)
          << which_weights << " weights K dimension must be equal to intermediate_size.";
    }
    if (args->num_fused_shared_experts > 0) {
      TVM_FFI_ICHECK_EQ(weights.size(0), args->local_num_experts + args->num_fused_shared_experts)
          << which_weights
          << " weights dim 0 must be local_num_experts + num_fused_shared_experts.";
    }
  }

  void check_optional_per_expert_float_tensor(Optional<TensorView> const& tensor,
                                              std::string const& tensor_name) const {
    if (!tensor.has_value()) {
      return;
    }
    auto const& value = tensor.value();
    TVM_FFI_ICHECK(value.device().device_type == kDLCUDA)
        << tensor_name << " must be a CUDA tensor.";
    TVM_FFI_ICHECK(value.device().device_id == hidden_states.device().device_id)
        << tensor_name << " must be on the same device as hidden_states.";
    TVM_FFI_ICHECK_EQ(value.dtype(), dl_float32) << tensor_name << " must be float32.";
    TVM_FFI_ICHECK_EQ(value.ndim(), 1) << tensor_name << " must be 1D.";
    // The batched GEMM indexes per-expert tensors by local batch entry, and fused
    // shared experts occupy the rows after the routed local experts, so the tensor
    // must cover local_num_experts + num_fused_shared_experts rows.
    TVM_FFI_ICHECK_EQ(value.size(0), args->local_num_experts + args->num_fused_shared_experts)
        << tensor_name << " must have shape [local_num_experts + num_fused_shared_experts].";
    TVM_FFI_ICHECK(value.IsContiguous()) << tensor_name << " must be contiguous.";
  }

  void check_routing_common() const {
    TVM_FFI_ICHECK(args->top_k > 0 && args->top_k <= args->num_experts)
        << "top_k must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_num_experts > 0 && args->local_num_experts <= args->num_experts)
        << "local_num_experts must be between 1 and num_experts";
    TVM_FFI_ICHECK(args->local_expert_offset >= 0 &&
                   args->local_expert_offset + args->local_num_experts <= args->num_experts)
        << "expert offset and count must be within valid range";

    check_routing_logits();

    if (routing_bias.has_value()) {
      check_routing_bias_shape();
    }
  }

  // Routing phase workspace tensors (allocated in prepare_routing() or prepare_routing_common())
  Tensor num_tokens_per_expert;
  Tensor total_num_padded_tokens;
  Tensor expanded_idx_to_permuted_idx;
  Tensor permuted_idx_to_token_idx;
  Tensor permuted_idx_to_expanded_idx;
  // Launcher-owned routing weights, returned by run() on the do_finalize=false
  // rows. Stays empty when a derived launcher borrows a caller-supplied buffer
  // instead, in which case only workspace.expert_weights points at it and the
  // caller must substitute its own buffer for that slot.
  Tensor expert_weights;
  Tensor expert_indexes;
  Tensor expert_count_histogram;
  Tensor cta_idx_xy_to_batch_idx;
  Tensor cta_idx_xy_to_mn_limit;
  Tensor num_non_exiting_ctas;

  void* permuted_idx_to_expanded_idx_ptr() const {
    return permuted_idx_to_expanded_idx.defined() ? permuted_idx_to_expanded_idx.data_ptr()
                                                  : nullptr;
  }

  void prepare_routing_common() {
    int32_t const totalExpertsPerToken = args->top_k + args->num_fused_shared_experts;
    int32_t const totalNumExperts = args->num_experts + args->num_fused_shared_experts;

    // Allocate routing phase workspace tensors
    num_tokens_per_expert = alloc_tensor({totalNumExperts}, dl_int32, hidden_states.device());
    int32_t max_num_padded_tokens =
        tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxPermutedPaddedCount(
            args->num_tokens, totalExpertsPerToken, totalNumExperts, tile_tokens_dim);

    total_num_padded_tokens = alloc_tensor({1}, dl_int32, hidden_states.device());

    expanded_idx_to_permuted_idx =
        alloc_tensor({args->num_tokens * totalExpertsPerToken}, dl_int32, hidden_states.device());

    // WAR: the routed batched-GEMM kernels read one int32 past the end of the route map.
    // TODO: drop the +1 once the fixed kernel cubins land.
    permuted_idx_to_token_idx =
        alloc_tensor({max_num_padded_tokens + 1}, dl_int32, hidden_states.device());

    if (gemm1_bias_type == batchedGemm::gemm::BiasType::Mn) {
      permuted_idx_to_expanded_idx =
          alloc_tensor({max_num_padded_tokens}, dl_int32, hidden_states.device());
    }

    expert_indexes =
        alloc_tensor({args->num_tokens, totalExpertsPerToken}, dl_int32, hidden_states.device());

    // expert_weights allocation should be done by derived class since data type could vary

    int64_t const size_of_expert_count_histogram = std::max(totalNumExperts * 2, 256 * 2);
    expert_count_histogram = alloc_tensor({size_of_expert_count_histogram},
                                          dl_int32,  // 256 is the max number of threads per block
                                                     // and max number of experts
                                          hidden_states.device());

    int32_t max_num_ctas = tensorrt_llm::kernels::trtllmgen_moe::Routing::getMaxNumCtasInBatchDim(
        args->num_tokens, totalExpertsPerToken, totalNumExperts, tile_tokens_dim);

    cta_idx_xy_to_batch_idx = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());

    cta_idx_xy_to_mn_limit = alloc_tensor({max_num_ctas}, dl_int32, hidden_states.device());

    num_non_exiting_ctas = alloc_tensor({1}, dl_int32, hidden_states.device());

    workspace.total_num_padded_tokens = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.total_max_padded_tokens = max_num_padded_tokens;
    workspace.ProjUpTileN = tile_tokens_dim;
    workspace.routing_expert_indexes = static_cast<int*>(expert_indexes.data_ptr());
    workspace.permuted_idx_size = static_cast<int*>(total_num_padded_tokens.data_ptr());
    workspace.expanded_idx_to_permuted_idx =
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr());
    workspace.permuted_idx_to_token_idx = static_cast<int*>(permuted_idx_to_token_idx.data_ptr());
    workspace.permuted_idx_to_expanded_idx = static_cast<int*>(permuted_idx_to_expanded_idx_ptr());
    // workspace.expert_weights will be set by derived class after expert_weights allocation
    workspace.cta_idx_xy_to_batch_idx = static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr());
    workspace.cta_idx_xy_to_mn_limit = static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr());
    workspace.num_non_exiting_ctas = static_cast<int*>(num_non_exiting_ctas.data_ptr());
  }

  void check_moe_common(bool allow_bias_with_none_type = false) const {
    // Hidden states [num_tokens, hidden_size]
    TVM_FFI_ICHECK_EQ(hidden_states.ndim(), 2) << "hidden_states must be 2D.";
    check_gemm1_bias_mn(gemm1_bias, gemm1_bias_type, args->num_tokens, args->top_k,
                        args->intermediate_size, allow_bias_with_none_type);
  }

  // MoE computation phase workspace tensors (allocated in prepare_moe() or prepare_moe_common())
  Tensor gemm1_output;
  Tensor activation_output;
  Tensor gemm2_output;
  Tensor workspace_fc1;
  Tensor workspace_fc2;
  Tensor output;
  int64_t moe_tactic{-1};

 public:
  virtual ~MoeRoutingLauncher() = default;
  virtual void check_routing() const = 0;
  virtual void prepare_routing() = 0;
  virtual void check_moe() const = 0;
  virtual void prepare_moe(int64_t& moe_tactic) = 0;
};

inline void MoeRoutingLauncher::init_common(
    std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
    int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
    int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
    bool norm_topk_prob) {
  // Check devicearchitecture: Blackwell (SM 10.x) required
  auto device = hidden_states.device().device_id;
  int major = 0, minor = 0;
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  TVM_FFI_ICHECK(major == 10 || major == 12)
      << "MoE kernel requires SM 10.x or SM 12.x architecture. Current device has SM " << major
      << minor;
  this->device_version = std::make_tuple(major, minor);

  args->routing_logits = routing_logits.has_value() ? routing_logits.value().data_ptr() : nullptr;
  args->routing_bias = routing_bias.has_value() ? routing_bias.value().data_ptr() : nullptr;
  args->hidden_states = hidden_states.data_ptr();
  args->gemm1_weights = gemm1_weights.data_ptr();
  args->gemm2_weights = gemm2_weights.data_ptr();
  args->gemm1_bias = gemm1_bias.has_value() ? gemm1_bias.value().data_ptr() : nullptr;
  auto bias_type_enum = static_cast<batchedGemm::gemm::BiasType>(gemm1_bias_type);
  args->gemm1_bias_type = bias_type_enum;

  // Fused shared experts do not yet support expert parallelism (EP).
  //
  // The routing kernel assigns each fused shared expert the global id
  // (num_experts + k) and the permutation pipeline maps a global expert id to a
  // weight-tensor row as (global_id - local_expert_offset) (see
  // include/flashinfer/trtllm/fused_moe/RoutingKernel.cuh). A shared expert
  // therefore lands at its intended local row (local_num_experts + k) only when
  // local_expert_offset + local_num_experts == num_experts, i.e. when every
  // routed expert is local to this rank. Under EP (local_expert_offset > 0, or
  // local_num_experts < num_experts) this silently produces wrong results, so
  // reject it explicitly until the kernel learns to map the shared-expert id
  // range independently of the routed-expert window.
  TVM_FFI_ICHECK(args->num_fused_shared_experts == 0 ||
                 (args->local_expert_offset == 0 && args->local_num_experts == args->num_experts))
      << "Fused shared experts (num_fused_shared_experts > 0) are currently only supported "
         "without expert parallelism, i.e. local_expert_offset == 0 and "
         "local_num_experts == num_experts. Got num_fused_shared_experts="
      << args->num_fused_shared_experts << ", local_expert_offset=" << args->local_expert_offset
      << ", local_num_experts=" << args->local_num_experts << ", num_experts=" << args->num_experts
      << ".";

  this->args = std::move(args);
  this->tile_tokens_dim = tile_tokens_dim;
  this->routing_method_type = routing_method_type;
  this->use_shuffled_weight = use_shuffled_weight;
  TVM_FFI_ICHECK(0 <= weight_layout && weight_layout <= 2)
      << "the value of weight_layout is not recognized";
  this->weight_layout = static_cast<batchedGemm::gemm::MatrixLayout>(weight_layout);
  this->activation_type = activation_type;
  this->intermediate_size_factor = isGatedActivation(activation_type) ? 2 : 1;
  this->norm_topk_prob = norm_topk_prob;
  this->gemm1_bias_type = bias_type_enum;
}

// Staged MoE: routing + external GEMM workspace + finalize (no cubin BMM).
// Dtype-specific behavior is provided by subclasses (e.g. Bf16StagedMoeLauncher).
class StagedMoeLauncher : public MoeRoutingLauncher {
 public:
  StagedMoeLauncher(Optional<TensorView> const& routing_logits,
                    Optional<TensorView> const& routing_bias, TensorView const& expert_indices,
                    TensorView const& expert_weights, TensorView const& hidden_states,
                    TensorView const& gemm1_weights, TensorView const& gemm2_weights)
      : MoeRoutingLauncher(routing_logits, routing_bias, hidden_states, gemm1_weights,
                           Optional<TensorView>(), Optional<TensorView>(), Optional<TensorView>(),
                           gemm2_weights, Optional<TensorView>(), Optional<TensorView>()),
        expert_indices(expert_indices),
        expert_weights(expert_weights) {}

  void init(std::unique_ptr<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>&& args,
            int64_t tile_tokens_dim, int64_t routing_method_type, bool use_shuffled_weight,
            int64_t weight_layout, ActivationType activation_type, int64_t gemm1_bias_type,
            bool norm_topk_prob = true) {
    MoeRoutingLauncher::init_common(std::move(args), tile_tokens_dim, routing_method_type,
                                    use_shuffled_weight, weight_layout, activation_type,
                                    gemm1_bias_type, norm_topk_prob);
  }

  virtual DLDataType gemm_buffer_dtype() const = 0;
  virtual void configure_routing_dtypes() = 0;
  virtual bool routing_use_deep_seek_fp8() const { return false; }
  virtual std::vector<int32_t> supported_tile_nums() const { return {8, 16, 32, 64, 128}; }
  virtual void prepare_external_gemm_buffers() {
    int32_t max_num_padded_tokens = workspace.total_max_padded_tokens;
    auto const buf_dtype = gemm_buffer_dtype();
    gemm1_output = alloc_tensor({max_num_padded_tokens, args->intermediate_size}, buf_dtype,
                                hidden_states.device());
    gemm2_output =
        alloc_tensor({max_num_padded_tokens, args->hidden_size}, buf_dtype, hidden_states.device());
    workspace.gemm1_output = gemm1_output.data_ptr();
    workspace.gemm2_output = gemm2_output.data_ptr();
  }

  void check_routing() const override {
    MoeRoutingLauncher::check_routing_common();
    if (expert_indices.ndim() == 2 && expert_indices.size(0) > 0) {
      TVM_FFI_ICHECK_EQ(expert_indices.ndim(), 2) << "expert_indices must be 2D.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(0), hidden_states.size(0))
          << "expert_indices and hidden_states must have same number of tokens.";
      TVM_FFI_ICHECK_EQ(expert_indices.size(1), args->top_k)
          << "expert_indices dim1 must match top_k.";
      TVM_FFI_ICHECK_EQ(expert_indices.dtype(), dl_int32) << "expert_indices must be int32.";
    }
  }

  void prepare_routing() override {
    MoeRoutingLauncher::prepare_routing_common();
    configure_routing_dtypes();

    bool has_precomputed_indices = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    if (has_precomputed_indices) {
      workspace.routing_expert_indexes =
          static_cast<int*>(const_cast<void*>(expert_indices.data_ptr()));
    }
    bool has_precomputed_weights = expert_weights.ndim() == 2 && expert_weights.size(0) > 0;
    if (has_precomputed_weights) {
      // Caller-owned precomputed weights are consumed directly by the routing kernel.
      // Slot 0 in collect_routing_outputs() is intentionally undefined in this
      // mode; Python substitutes the caller tensor for downstream use.
      workspace.expert_weights = const_cast<void*>(expert_weights.data_ptr());
    } else {
      // Routing output weights have their own fixed ABI and do not inherit the
      // logits dtype. In particular, DeepSeekV3 accepts FP32 logits but emits
      // BF16 weights; allocating this buffer as FP32 mislabels packed BF16 data
      // and makes the staged finalize path consume the wrong values.
      MoeRoutingLauncher::expert_weights = alloc_tensor(
          {args->num_tokens, args->top_k}, expert_weight_storage_dtype(), hidden_states.device());
      workspace.expert_weights = MoeRoutingLauncher::expert_weights.data_ptr();
    }
  }

  void check_moe() const override {
    MoeRoutingLauncher::check_moe_common();
    TVM_FFI_ICHECK(weight_layout == batchedGemm::gemm::MatrixLayout::MajorK ||
                   weight_layout == batchedGemm::gemm::MatrixLayout::BlockMajorK)
        << "Staged MoE: weight_layout must be MajorK or BlockMajorK";
    check_weights_shape("gemm1");
    check_weights_shape("gemm2");
    TVM_FFI_ICHECK_EQ(args->intermediate_size % 128, 0)
        << "the second dimension of weights must be a multiple of 128.";
  }

  // Staged path allocates GEMM buffers directly; cubin BMM workspace is unused.
  void prepare_moe(int64_t& /*moe_tactic*/) override {}

  void run_routing_kernel(bool enable_pdl, bool use_routing_scales_on_input = false) {
    tensorrt_llm::kernels::trtllmgen_moe::Routing::Runner routing_runner(tile_tokens_dim);
    cudaStream_t routing_stream = get_stream(hidden_states.device());
    bool const has_precomputed_indices = expert_indices.ndim() == 2 && expert_indices.size(0) > 0;
    bool const has_precomputed_weights = expert_weights.ndim() == 2 && expert_weights.size(0) > 0;
    int32_t* expert_ids_param =
        has_precomputed_indices && has_precomputed_weights
            ? static_cast<int32_t*>(const_cast<void*>(expert_indices.data_ptr()))
            : nullptr;
    void* routing_logits_ptr = has_precomputed_indices ? nullptr : args->routing_logits;
    int16_t* replay_ptr = nullptr;
    if (routing_replay_out.has_value()) {
      replay_ptr = reinterpret_cast<int16_t*>(routing_replay_out.value().data_ptr());
    }
    routing_runner.run(
        routing_logits_ptr, args->routing_bias, args->num_tokens, args->num_experts, args->top_k,
        args->num_fused_shared_experts, args->n_group, args->topk_group, args->local_expert_offset,
        args->local_num_experts, args->routed_scaling_factor, workspace.routing_expert_indexes,
        static_cast<int*>(expert_count_histogram.data_ptr()),
        static_cast<int*>(total_num_padded_tokens.data_ptr()),
        static_cast<int*>(expanded_idx_to_permuted_idx.data_ptr()),
        workspace.permuted_idx_to_expanded_idx,
        static_cast<int*>(permuted_idx_to_token_idx.data_ptr()), expert_ids_param,
        workspace.expert_weights, static_cast<int*>(num_tokens_per_expert.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_batch_idx.data_ptr()),
        static_cast<int*>(cta_idx_xy_to_mn_limit.data_ptr()),
        static_cast<int*>(num_non_exiting_ctas.data_ptr()), args->mDtypeElt, mRoutingBiasDtype,
        use_routing_scales_on_input, routing_use_deep_seek_fp8(),
        static_cast<RoutingMethodType>(routing_method_type), routing_stream, mRoutingLogitsDtype,
        norm_topk_prob, replay_ptr, enable_pdl);
  }

  Tensor expert_weights_tensor() const { return MoeRoutingLauncher::expert_weights; }

  Array<Tensor> collect_routing_outputs() const {
    Array<Tensor> result;
    result.push_back(expert_weights_tensor());
    result.push_back(expanded_idx_to_permuted_idx);
    result.push_back(permuted_idx_to_token_idx);
    result.push_back(cta_idx_xy_to_batch_idx);
    result.push_back(cta_idx_xy_to_mn_limit);
    result.push_back(num_non_exiting_ctas);
    result.push_back(total_num_padded_tokens);
    result.push_back(gemm1_output);
    result.push_back(gemm2_output);
    return result;
  }

 protected:
  virtual DLDataType expert_weight_storage_dtype() const { return gemm_buffer_dtype(); }

 private:
  TensorView expert_indices;
  TensorView expert_weights;
};

class Bf16StagedMoeLauncher : public StagedMoeLauncher {
 public:
  using StagedMoeLauncher::StagedMoeLauncher;

  DLDataType gemm_buffer_dtype() const override { return dl_bfloat16; }

  void configure_routing_dtypes() override {
    args->mDtypeElt = btg::Dtype::Bfloat16;
    args->mUseDeepSeekFp8 = false;
    mDtypeAct = btg::Dtype::Bfloat16;
    mDtypeWeights = btg::Dtype::Bfloat16;

    auto const routing_bias_dtype =
        routing_bias.has_value() ? routing_bias.value().dtype() : dl_bfloat16;
    mRoutingBiasDtype = routing_bias_dtype == dl_bfloat16 ? btg::Dtype::Bfloat16 : btg::Dtype::Fp32;

    auto const routing_logits_dtype =
        routing_logits.has_value() ? routing_logits.value().dtype() : dl_bfloat16;
    mRoutingLogitsDtype =
        routing_logits_dtype == dl_float32 ? btg::Dtype::Fp32 : btg::Dtype::Bfloat16;
  }
};

inline void run_staged_moe_finalize(btg::Dtype dtype_elt, btg::Dtype dtype_exp_w,
                                    bool use_deep_seek_fp8, TensorView gemm2_output,
                                    TensorView output, TensorView expert_weights,
                                    TensorView expanded_idx_to_permuted_idx,
                                    TensorView total_num_padded_tokens, int64_t num_tokens,
                                    int64_t num_experts, int64_t top_k, int64_t hidden_size,
                                    bool enable_pdl, bool use_routing_scales_on_input) {
  moe::dev::finalize::Data finalizeData;
  finalizeData.mDtypeElt = dtype_elt;
  finalizeData.mDtypeExpW = dtype_exp_w;
  finalizeData.mUsePdl = enable_pdl;
  finalizeData.mUseDeepSeekFp8 = use_deep_seek_fp8;
  finalizeData.inPtr = const_cast<void*>(gemm2_output.data_ptr());
  finalizeData.outPtr = output.data_ptr();
  finalizeData.inDqSfsPtr = nullptr;
  finalizeData.outDqSfsPtr = nullptr;
  finalizeData.expertWeightsPtr =
      use_routing_scales_on_input ? nullptr : const_cast<void*>(expert_weights.data_ptr());
  finalizeData.expandedIdxToPermutedIdx =
      static_cast<int32_t*>(const_cast<void*>(expanded_idx_to_permuted_idx.data_ptr()));
  finalizeData.numTokens = static_cast<int32_t>(num_tokens);
  finalizeData.numExperts = static_cast<int32_t>(num_experts);
  finalizeData.topK = static_cast<int32_t>(top_k);
  finalizeData.hiddenDim = static_cast<int32_t>(hidden_size);
  finalizeData.hiddenDimPadded = static_cast<int32_t>(hidden_size);
  finalizeData.totalNumPaddedTokens =
      static_cast<int32_t*>(const_cast<void*>(total_num_padded_tokens.data_ptr()));

  cudaStream_t stream = get_stream(output.device());
  moe::dev::finalize::run(finalizeData, stream);
  sync_check_cuda_error(stream);
}

inline btg::Dtype expert_weights_dtype(TensorView const& expert_weights) {
  if (expert_weights.dtype() == dl_float32) {
    return btg::Dtype::Fp32;
  }
  TVM_FFI_ICHECK_EQ(expert_weights.dtype(), dl_bfloat16)
      << "Staged MoE finalize: expert_weights must be bfloat16 or float32.";
  return btg::Dtype::Bfloat16;
}

inline std::unique_ptr<StagedMoeLauncher> make_bf16_staged_launcher(
    Optional<TensorView> const& routing_logits, Optional<TensorView> const& routing_bias,
    TensorView const& expert_indices, TensorView const& expert_weights,
    TensorView const& hidden_states, TensorView const& gemm1_weights,
    TensorView const& gemm2_weights, int64_t num_experts, int64_t top_k, Optional<int64_t> n_group,
    Optional<int64_t> topk_group, int64_t intermediate_size, int64_t local_expert_offset,
    int64_t local_num_experts, Optional<double> routed_scaling_factor, int64_t routing_method_type,
    bool use_shuffled_weight, int64_t weight_layout, int64_t tile_tokens_dim,
    int64_t activation_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out) {
  auto const num_tokens = hidden_states.size(0);
  auto const hidden_size = hidden_states.size(1);
  auto const activation = validateAndCastActivationType(activation_type);

  auto args = std::make_unique<tensorrt_llm::kernels::trtllmgen_moe::MoE::MoERunnerArgs>();
  args->num_tokens = num_tokens;
  args->num_experts = num_experts;
  args->hidden_size = hidden_size;
  args->hidden_size_output = hidden_size;
  args->top_k = top_k;
  args->n_group = n_group.value_or(0);
  args->topk_group = topk_group.value_or(0);
  args->routed_scaling_factor = routed_scaling_factor.value_or(1.0);
  args->local_expert_offset = local_expert_offset;
  args->local_num_experts = local_num_experts;
  args->intermediate_size = intermediate_size;
  args->do_finalize = true;
  args->output = nullptr;
  args->output_scale = nullptr;
  args->mDtypeOut = btg::Dtype::Bfloat16;
  args->mDtypeExpW = btg::Dtype::Bfloat16;

  auto launcher = std::make_unique<Bf16StagedMoeLauncher>(
      routing_logits, routing_bias, expert_indices, expert_weights, hidden_states, gemm1_weights,
      gemm2_weights);
  launcher->init(std::move(args), tile_tokens_dim, routing_method_type, use_shuffled_weight,
                 weight_layout, activation, static_cast<int64_t>(batchedGemm::gemm::BiasType::None),
                 norm_topk_prob);
  launcher->set_routing_replay_out(routing_replay_out);
  return launcher;
}

inline Array<Tensor> trtllm_moe_run_routing(
    Optional<TensorView> const& routing_logits, Optional<TensorView> const& routing_bias,
    TensorView const& expert_indices, TensorView const& expert_weights,
    TensorView const& hidden_states, TensorView const& gemm1_weights,
    TensorView const& gemm2_weights, int64_t num_experts, int64_t top_k, Optional<int64_t> n_group,
    Optional<int64_t> topk_group, int64_t intermediate_size, int64_t local_expert_offset,
    int64_t local_num_experts, Optional<double> routed_scaling_factor, int64_t routing_method_type,
    bool use_shuffled_weight, int64_t weight_layout, bool enable_pdl, Array<int64_t> moe_tactic,
    int64_t activation_type, bool norm_topk_prob, Optional<TensorView> routing_replay_out) {
  TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_bfloat16)
      << "Staged MoE routing (BF16): hidden_states must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_bfloat16)
      << "Staged MoE routing (BF16): gemm1_weights must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_bfloat16)
      << "Staged MoE routing (BF16): gemm2_weights must be bfloat16.";

  if (routing_replay_out.has_value()) {
    validate_routing_replay_out(routing_replay_out.value(), hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  std::vector<int32_t> supported_tile_nums{8, 16, 32, 64, 128, 256};
  auto const [tile_N, config] = resolveMoeTileAndConfig(moe_tactic, supported_tile_nums, num_tokens,
                                                        top_k, local_num_experts);
  (void)config;

  auto launcher = make_bf16_staged_launcher(
      routing_logits, routing_bias, expert_indices, expert_weights, hidden_states, gemm1_weights,
      gemm2_weights, num_experts, top_k, n_group, topk_group, intermediate_size,
      local_expert_offset, local_num_experts, routed_scaling_factor, routing_method_type,
      use_shuffled_weight, weight_layout, tile_N, activation_type, norm_topk_prob,
      routing_replay_out);

  launcher->check_routing();
  launcher->prepare_routing();
  launcher->run_routing_kernel(enable_pdl);
  launcher->check_moe();
  launcher->prepare_external_gemm_buffers();

  return launcher->collect_routing_outputs();
}

// Allocate and expose routing/GEMM buffers without submitting routing kernels.
inline Array<tvm::ffi::Any> trtllm_moe_prepare_routing(
    Optional<Tensor> const& routing_logits, Optional<Tensor> const& routing_bias,
    Tensor const& expert_indices, Tensor const& expert_weights, Tensor const& hidden_states,
    Tensor const& gemm1_weights, Tensor const& gemm2_weights, int64_t num_experts, int64_t top_k,
    Optional<int64_t> n_group, Optional<int64_t> topk_group, int64_t intermediate_size,
    int64_t local_expert_offset, int64_t local_num_experts, Optional<double> routed_scaling_factor,
    int64_t routing_method_type, bool use_shuffled_weight, int64_t weight_layout, bool enable_pdl,
    Array<int64_t> moe_tactic, int64_t activation_type, bool norm_topk_prob,
    Optional<Tensor> routing_replay_out) {
  TVM_FFI_ICHECK_EQ(hidden_states.dtype(), dl_bfloat16)
      << "Staged MoE routing (BF16): hidden_states must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm1_weights.dtype(), dl_bfloat16)
      << "Staged MoE routing (BF16): gemm1_weights must be bfloat16.";
  TVM_FFI_ICHECK_EQ(gemm2_weights.dtype(), dl_bfloat16)
      << "Staged MoE routing (BF16): gemm2_weights must be bfloat16.";

  if (routing_replay_out.has_value()) {
    auto const replay = routing_replay_out.value();
    validate_routing_replay_out(replay, hidden_states, top_k);
  }

  auto const num_tokens = hidden_states.size(0);
  std::vector<int32_t> supported_tile_nums{8, 16, 32, 64, 128, 256};
  auto const [tile_N, config] = resolveMoeTileAndConfig(moe_tactic, supported_tile_nums, num_tokens,
                                                        top_k, local_num_experts);
  (void)config;

  auto as_view = [](Optional<Tensor> const& tensor) -> Optional<TensorView> {
    if (tensor.has_value()) {
      auto const owner = tensor.value();
      return TensorView(owner);
    }
    return Optional<TensorView>();
  };
  std::shared_ptr<StagedMoeLauncher> launcher = make_bf16_staged_launcher(
      as_view(routing_logits), as_view(routing_bias), expert_indices, expert_weights, hidden_states,
      gemm1_weights, gemm2_weights, num_experts, top_k, n_group, topk_group, intermediate_size,
      local_expert_offset, local_num_experts, routed_scaling_factor, routing_method_type,
      use_shuffled_weight, weight_layout, tile_N, activation_type, norm_topk_prob,
      as_view(routing_replay_out));

  launcher->check_routing();
  launcher->prepare_routing();
  launcher->check_moe();
  launcher->prepare_external_gemm_buffers();

  // TensorView only borrows shape/stride metadata. Keep owning Tensor handles
  // for every input until the deferred callable and workspace are released.
  Array<Tensor> inputs{expert_indices, expert_weights, hidden_states, gemm1_weights, gemm2_weights};
  if (routing_logits.has_value()) inputs.push_back(routing_logits.value());
  if (routing_bias.has_value()) inputs.push_back(routing_bias.value());
  if (routing_replay_out.has_value()) inputs.push_back(routing_replay_out.value());
  auto const device = hidden_states.device();
  auto const stream = get_stream(device);
  auto submit = tvm::ffi::Function::FromTyped(
      [launcher, inputs, device, stream, enable_pdl](TensorView stream_anchor) {
        (void)inputs;
        // A tensor argument installs the caller's framework stream in TVM-FFI.
        TVM_FFI_ICHECK(stream_anchor.device().device_type == device.device_type &&
                       stream_anchor.device().device_id == device.device_id)
            << "Prepared MoE routing must be submitted on the preparation device.";
        TVM_FFI_ICHECK(get_stream(device) == stream)
            << "Prepared MoE routing must be submitted on the preparation stream.";
        launcher->run_routing_kernel(enable_pdl);
      });
  return {launcher->collect_routing_outputs(), submit};
}

inline void trtllm_moe_run_finalize(TensorView gemm2_output, TensorView output,
                                    TensorView expert_weights,
                                    TensorView expanded_idx_to_permuted_idx,
                                    TensorView total_num_padded_tokens, int64_t num_tokens,
                                    int64_t num_experts, int64_t top_k, int64_t hidden_size,
                                    bool enable_pdl, bool use_routing_scales_on_input) {
  TVM_FFI_ICHECK_EQ(gemm2_output.dtype(), dl_bfloat16)
      << "Staged MoE finalize (BF16): gemm2_output must be bfloat16.";
  TVM_FFI_ICHECK_EQ(output.dtype(), dl_bfloat16)
      << "Staged MoE finalize (BF16): output must be bfloat16.";

  run_staged_moe_finalize(btg::Dtype::Bfloat16, expert_weights_dtype(expert_weights),
                          /*use_deep_seek_fp8=*/false, gemm2_output, output, expert_weights,
                          expanded_idx_to_permuted_idx, total_num_padded_tokens, num_tokens,
                          num_experts, top_k, hidden_size, enable_pdl, use_routing_scales_on_input);
}

}  // namespace flashinfer
