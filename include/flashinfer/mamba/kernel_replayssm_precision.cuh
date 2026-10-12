/*
 * Copyright (c) 2026 by FlashInfer team.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef FLASHINFER_MAMBA_KERNEL_REPLAYSSM_PRECISION_CUH_
#define FLASHINFER_MAMBA_KERNEL_REPLAYSSM_PRECISION_CUH_

#include "kernel_checkpointing_ssu.cuh"

namespace flashinfer::mamba::checkpointing {

// Correctness-oriented SIMT implementation of the same factorization, not a
// sequential-decode fallback. MASK bits retain FP32 readout (1), scaled B in
// replay (2), and scaled CB (4). MASK=0 controls for the change of implementation.
// It retains the ring layout and host-owned acceptance/checkpoint bookkeeping.
template <bool PRESERVE>
__device__ __forceinline__ float precision_operand(float v) {
  if constexpr (PRESERVE) return v;
  return __bfloat162float(__float2bfloat16_rn(v));
}

__device__ __forceinline__ float precision_warp_sum(float v) {
#pragma unroll
  for (int offset = 16; offset > 0; offset /= 2) v += __shfl_down_sync(0xffffffff, v, offset);
  return __shfl_sync(0xffffffff, v, 0);
}

template <int MAX_WINDOW>
__device__ __forceinline__ void precision_history_decay(CheckpointingSsuParams const& p,
                                                        int64_t slot, int head, int start,
                                                        int count, float a, float (&dt)[MAX_WINDOW],
                                                        float (&cum)[MAX_WINDOW]) {
  auto const* src = static_cast<float const*>(p.dt_cache) + slot * p.dt_cache_stride_seq +
                    head * p.dt_cache_stride_head;
  float sum = 0.f;
  for (int j = 0; j < count; ++j) {
    dt[j] = src[(start + j) % p.ring_buffer_len];
    sum += dt[j];
    cum[j] = sum * a;
  }
}

template <typename T, typename I, int DIM, int DSTATE, int MAX_WINDOW, int MASK>
__device__ __forceinline__ void precision_replay_row(CheckpointingSsuParams const& p, int64_t slot,
                                                     int head, int group, int row, int start,
                                                     int count, float const (&dt)[MAX_WINDOW],
                                                     float const (&cum)[MAX_WINDOW],
                                                     float2 (&values)[DSTATE / 64]) {
  int const lane = threadIdx.x;
  auto const* state = static_cast<T const*>(p.state) + slot * p.state_stride_seq +
                      (int64_t(head) * DIM + row) * DSTATE;
  auto const* x = static_cast<I const*>(p.x_cache) + slot * p.x_cache_stride_seq +
                  head * p.x_cache_stride_head + row;
  auto const* b = static_cast<I const*>(p.B_cache) + slot * p.B_cache_stride_seq +
                  group * p.B_cache_stride_group;
  float const total = count ? cum[count - 1] : 0.f;
#pragma unroll
  for (int k = 0; k < DSTATE / 64; ++k) {
    int const n = lane * 2 + k * 64;
    float2 v = toFloat2(state + n);
    v.x *= __expf(total);
    v.y *= __expf(total);
    for (int j = 0; j < count; ++j) {
      int const pos = (start + j) % p.ring_buffer_len;
      float const coeff = __expf(total - cum[j]) * dt[j];
      float const xv = toFloat(x[pos * p.x_cache_stride_pos]);
      float2 const bv = toFloat2(b + pos * p.B_cache_stride_pos + n);
      v.x = fmaf(xv, precision_operand<(MASK & 2) != 0>(bv.x * coeff), v.x);
      v.y = fmaf(xv, precision_operand<(MASK & 2) != 0>(bv.y * coeff), v.y);
    }
    values[k] = v;
  }
}

template <typename T, int DSTATE, int ROUNDS>
__device__ __forceinline__ void precision_store_row(T* state, int64_t offset,
                                                    float2 const (&values)[DSTATE / 64],
                                                    int64_t seed) {
  uint32_t random[4];
#pragma unroll
  for (int k = 0; k < DSTATE / 64; ++k) {
    int64_t const index = offset + threadIdx.x * 2 + k * 64;
    if constexpr (ROUNDS > 0 && std::is_same_v<T, __half>) {
      // Identical mapping for every precision MASK, but not the MMA kernel's mapping.
      *reinterpret_cast<uint32_t*>(state + index) =
          stochastic_round_pair_with_philox_refresh<ROUNDS>(values[k].x, values[k].y, k, seed,
                                                            index, random);
    } else {
      *reinterpret_cast<Pair<T>*>(state + index) = pack_float2<T>(values[k]);
    }
  }
}

template <typename I, int DSTATE>
__device__ __forceinline__ float precision_cb_dot(I const* c, I const* b) {
  float dot = 0.f;
#pragma unroll
  for (int n = threadIdx.x; n < DSTATE; n += 32) dot = fmaf(toFloat(c[n]), toFloat(b[n]), dot);
  return precision_warp_sum(dot);
}

template <typename I, typename DT, typename W, typename A, typename T, typename Index, int DIM,
          int DSTATE, int NPREDICTED, int MAX_WINDOW, int MASK, int ROUNDS, bool PDL>
__global__ void checkpointing_precision_kernel(CheckpointingSsuParams p) {
  static_assert(DSTATE % 64 == 0 && DIM % 4 == 0);
  int const lane = threadIdx.x, row = blockIdx.x * 4 + threadIdx.y;
  int const seq = blockIdx.y, head = blockIdx.z, group = head / (p.nheads / p.ngroups);
  auto const* indices = static_cast<Index const*>(p.state_batch_indices);
  int64_t const slot = indices ? int64_t(indices[seq]) : seq;
  if (slot == p.pad_slot_id) return;
  int64_t outer = seq;
  int length = NPREDICTED;
  if (p.cu_seqlens) {
    auto const* cu = static_cast<int32_t const*>(p.cu_seqlens);
    outer = cu[seq];
    length = cu[seq + 1] - outer;
    if (length <= 0) return;
  }
  if constexpr (PDL) cudaGridDependencySynchronize();
  int const start = static_cast<int32_t const*>(p.ring_start)[slot];
  int const count = static_cast<int32_t const*>(p.prev_num_accepted)[slot];
  assert(count >= 0 && count <= MAX_WINDOW && length <= NPREDICTED);
  assert(start >= 0 && start < p.ring_buffer_len);
  bool const checkpoint = count + length > MAX_WINDOW;
  float const a = toFloat(static_cast<A const*>(p.A)[head]);
  float const bias = p.dt_bias ? toFloat(static_cast<W const*>(p.dt_bias)[head]) : 0.f;
  float const d = p.D ? toFloat(static_cast<W const*>(p.D)[head]) : 0.f;
  auto const* x = static_cast<I const*>(p.x) + outer * p.x_stride_seq + head * DIM + row;
  auto const* b = static_cast<I const*>(p.B) + outer * p.B_stride_seq + group * DSTATE;
  auto const* c = static_cast<I const*>(p.C) + outer * p.C_stride_seq + group * DSTATE;
  auto const* raw_dt = static_cast<DT const*>(p.dt) + outer * p.dt_stride_seq + head;
  float dt[NPREDICTED], cum[NPREDICTED], old_dt[MAX_WINDOW], old_cum[MAX_WINDOW];
  float sum = 0.f;
  for (int t = 0; t < length; ++t) {
    float v = toFloat(raw_dt[t * p.dt_stride_token]) + bias;
    dt[t] = p.dt_softplus ? thresholded_softplus(v) : v;
    sum += a * dt[t];
    cum[t] = sum;
  }
  precision_history_decay(p, slot, head, start, count, a, old_dt, old_cum);
  float const total = count ? old_cum[count - 1] : 0.f;
  float2 values[DSTATE / 64];
  precision_replay_row<T, I, DIM, DSTATE, MAX_WINDOW, MASK>(
      p, slot, head, group, row, start, checkpoint ? count : 0, old_dt, old_cum, values);
  auto* old_x =
      static_cast<I*>(p.x_cache) + slot * p.x_cache_stride_seq + head * p.x_cache_stride_head + row;
  auto* old_b =
      static_cast<I*>(p.B_cache) + slot * p.B_cache_stride_seq + group * p.B_cache_stride_group;
  for (int t = 0; t < length; ++t) {
    auto const* ct = c + t * p.C_stride_token;
    float y = 0.f;
#pragma unroll
    for (int k = 0; k < DSTATE / 64; ++k) {
      float2 const cv = toFloat2(ct + lane * 2 + k * 64);
      y = fmaf(precision_operand<(MASK & 1) != 0>(values[k].x), cv.x, y);
      y = fmaf(precision_operand<(MASK & 1) != 0>(values[k].y), cv.y, y);
    }
    y = precision_warp_sum(y) * __expf(cum[t]) * (checkpoint ? 1.f : __expf(total));
    for (int j = 0; j <= t; ++j) {
      float const cb = precision_cb_dot<I, DSTATE>(ct, b + j * p.B_stride_token) *
                       __expf(cum[t] - cum[j]) * dt[j];
      y = fmaf(precision_operand<(MASK & 4) != 0>(cb), toFloat(x[j * p.x_stride_token]), y);
    }
    if (!checkpoint) {
      for (int j = 0; j < count; ++j) {
        int const pos = (start + j) % p.ring_buffer_len;
        float const cb = precision_cb_dot<I, DSTATE>(ct, old_b + pos * p.B_cache_stride_pos) *
                         __expf(cum[t] + total - old_cum[j]) * old_dt[j];
        y = fmaf(precision_operand<(MASK & 4) != 0>(cb), toFloat(old_x[pos * p.x_cache_stride_pos]),
                 y);
      }
    }
    if (lane == 0) {
      y = fmaf(d, toFloat(x[t * p.x_stride_token]), y);
      if (p.z) {
        float const z = toFloat(static_cast<I const*>(
            p.z)[outer * p.z_stride_seq + t * p.z_stride_token + head * DIM + row]);
        y *= z / (1.f + __expf(-z));
      }
      static_cast<I*>(
          p.output)[outer * p.out_stride_seq + t * p.out_stride_token + head * DIM + row] =
          static_cast<I>(y);
    }
  }
  if (checkpoint) {
    int64_t const offset = slot * p.state_stride_seq + (int64_t(head) * DIM + row) * DSTATE;
    precision_store_row<T, DSTATE, ROUNDS>(static_cast<T*>(p.state), offset, values,
                                           ROUNDS > 0 ? *p.rand_seed : 0);
  }
  // Appends cannot overlap the committed prefix: ring length >= window + T.
  for (int t = 0; t < length; ++t) {
    int const pos = (start + count + t) % p.ring_buffer_len;
    if (lane == 0) old_x[pos * p.x_cache_stride_pos] = x[t * p.x_stride_token];
    if (row == 0) {
      if (lane == 0)
        static_cast<float*>(
            p.dt_cache)[slot * p.dt_cache_stride_seq + head * p.dt_cache_stride_head + pos] = dt[t];
      if (head % (p.nheads / p.ngroups) == 0) {
        for (int n = lane; n < DSTATE; n += 32)
          old_b[pos * p.B_cache_stride_pos + n] = b[t * p.B_stride_token + n];
      }
    }
  }
  if constexpr (PDL) cudaTriggerProgrammaticLaunchCompletion();
}

}  // namespace flashinfer::mamba::checkpointing
#endif
