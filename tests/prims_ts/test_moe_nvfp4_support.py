# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    DType,
    RouteImpl,
    TileScheduler,
)
from flashinfer.prims_ts.moe.config_mapper import (
    _DType,
    _make_json_moe_config_pair,
    map_trtllm_nvfp4_moe_tactic,
    valid_prims_ts_nvfp4_moe_tactics,
)
from flashinfer.tllm_enums import ActivationType
from flashinfer.utils import is_sm100a_supported


_requires_non_sm107_prims_ts = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7),
    reason="NVFP4 Prims-TS kernels support SM100 and SM103, not SM107",
)


def _find_bs1_ldgsts_persistent_pair():
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        num_tokens=1,
        top_k=8,
        num_local_experts=256,
    )

    for tactic in tactics:
        pair = _make_json_moe_config_pair(
            tile_n=tactic[0],
            moe_config_index=tactic[1],
            activation_type=int(ActivationType.Swiglu),
            dtype_a=int(_DType.E2M1),
            dtype_b=int(_DType.E2M1),
            fc1_dtype_c=int(_DType.E2M1),
            fc2_dtype_c=int(_DType.BF16),
            dtype_label="NVFP4xNVFP4",
        )
        if pair is None:
            continue
        fc1 = pair.fc1.cfg.kwargs
        fc2 = pair.fc2.cfg.kwargs
        if (
            fc1["tile_n"] == 8
            and fc1["tile_k"] == 512
            and fc1["cluster_m"] == 1
            and fc1["route_act"] == int(RouteImpl.LDGSTS)
            and fc1["route_sfs_act"] == int(RouteImpl.LDGSTS)
            and fc1["tile_scheduler"] == int(TileScheduler.PERSISTENT)
            and fc1["use_clc_fast_drain"] == 1
            and fc1["use_work_throttle"] == 1
            and fc2["tile_n"] == 8
            and fc2["tile_k"] == 512
            and fc2["cluster_m"] == 1
            and fc2["num_stages_a"] == 4
            and fc2["tile_scheduler"] == int(TileScheduler.PERSISTENT)
            and fc2["use_clc_fast_drain"] == 0
            and fc2["use_work_throttle"] == 0
        ):
            return pair

    return None


def test_nvfp4_search_contains_bs1_ldgsts_persistent_pair():
    assert _find_bs1_ldgsts_persistent_pair() is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 PrimsTS kernels require Blackwell SM100A+",
)
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize(
    (
        "stage",
        "problem_n",
        "problem_k",
        "num_experts",
        "top_k",
        "early_exit_max_token_ctas",
        "dtype_c_override",
        "seed",
    ),
    (
        pytest.param(
            "fc1", 4096, 7168, 16, 8, 16, int(DType.BF16), 42, id="fc1-mainloop"
        ),
        pytest.param("fc1", 256, 512, 2, 1, 4, None, 123, id="fc1-quantized-output"),
        pytest.param("fc2", 7168, 2048, 16, 8, 16, None, 42, id="fc2"),
    ),
)
def test_bs1_deepseek_persistent_pair_gpu_correctness(
    stage,
    problem_n,
    problem_k,
    num_experts,
    top_k,
    early_exit_max_token_ctas,
    dtype_c_override,
    seed,
):
    """Exercise the BS=1 tactic, including skipped persistent work tiles."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = _find_bs1_ldgsts_persistent_pair()
    assert pair is not None
    cfg = dict(pair.fc1.cfg.kwargs if stage == "fc1" else pair.fc2.cfg.kwargs)
    if dtype_c_override is not None:
        # The standalone random-input reference overstates E2M1 quantization
        # error for DeepSeek's large K. Use BF16 here to isolate the exact FC1
        # LDGSTS/persistent mainloop; the second FC1 row covers its real E2M1
        # output epilogue with the same tile and pipeline schedule.
        cfg["dtype_c"] = dtype_c_override

    assert reference_check(
        num_experts=num_experts,
        num_tokens=1,
        top_k=top_k,
        problem_n=problem_n,
        problem_k=problem_k,
        seed=seed,
        early_exit_max_token_ctas=early_exit_max_token_ctas,
        **cfg,
    )


def test_nvfp4_tile128_fused_tmem_allocation_has_no_standalone_sf_columns():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    pair = map_trtllm_nvfp4_moe_tactic(
        [128, 1],
        num_tokens=8192,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )
    cfg = pair.fc1.cfg.build()
    _, _, _, tmem_allocator = _build_schedule_validate(cfg)

    assert cfg.fuse_sf_copy_to_mma
    assert tmem_allocator.total_tmem_columns == cfg.tmem_required_cols == 320


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 PrimsTS kernels require Blackwell SM100A+",
)
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize(
    ("problem_k", "dtype_c_override"),
    (
        pytest.param(512, None, id="quantized-output"),
        pytest.param(1024, int(DType.BF16), id="two-k-tiles"),
    ),
)
def test_nvfp4_tile128_fused_fc1_gpu_correctness(problem_k, dtype_c_override):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = map_trtllm_nvfp4_moe_tactic(
        [128, 1],
        num_tokens=8192,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )
    cfg = dict(pair.fc1.cfg.kwargs)
    assert cfg["use_clc_fast_drain"]
    if dtype_c_override is not None:
        cfg["dtype_c"] = dtype_c_override

    assert reference_check(
        num_experts=2,
        num_tokens=257,
        top_k=1,
        problem_n=256,
        problem_k=problem_k,
        seed=123,
        early_exit_max_token_ctas=16,
        **cfg,
    )


def test_nvfp4_bs256_has_matching_gen_tile32_pair():
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        num_tokens=256,
        top_k=8,
        num_local_experts=256,
        activation_type=int(ActivationType.Swiglu),
    )

    for tactic in tactics:
        pair = map_trtllm_nvfp4_moe_tactic(
            tactic,
            num_tokens=256,
            top_k=8,
            num_local_experts=256,
            activation_type=int(ActivationType.Swiglu),
        )
        fc1 = pair.fc1.cfg.build()
        fc2 = pair.fc2.cfg.build()
        if (
            fc1.tile_n == 32
            and fc1.tile_k == 512
            and fc1.route_act == int(RouteImpl.LDGSTS)
            and fc1.route_sfs_act == int(RouteImpl.LDGSTS)
            and fc1.num_stages_a == 4
            and fc2.tile_n == 32
            and fc2.tile_k == 512
            and fc2.cluster_m == 2
            and fc2.num_stages_a == 4
            and fc2.use_unroll_loop_2x_for_mma
            and fc1.tile_scheduler == int(TileScheduler.PERSISTENT)
            and fc1.use_clc_fast_drain
            and fc2.tile_scheduler == int(TileScheduler.PERSISTENT)
            and fc2.use_clc_fast_drain
        ):
            break
    else:
        pytest.fail("missing the Gen-matched NVFP4 BS=256 tile-32 config pair")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize("stage", ["fc1", "fc2"])
@pytest.mark.parametrize("expert_start,expert_count", [(0, 1), (2, 3), (6, 1)])
def test_nvfp4_resident_weights_and_scales_bypass_poisoned_staging(
    stage,
    expert_start,
    expert_count,
):
    """Remap packed weights and R128c4 scales for uneven resident ranges."""
    from dataclasses import replace
    import cutlass
    import cuda.bindings.driver as cuda_drv
    from cutlass.cute.runtime import make_ptr
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _build_launch_io,
        _compile_for_launch,
        _launch_arg_tuple,
    )

    pair = map_trtllm_nvfp4_moe_tactic(
        [8, 0],
        num_tokens=127,
        top_k=2,
        num_local_experts=7,
        activation_type=int(ActivationType.Swiglu),
    )
    config = dict((pair.fc1 if stage == "fc1" else pair.fc2).cfg.kwargs)
    config["dtype_c"] = int(DType.BF16)
    io = _build_launch_io(
        num_experts=7,
        num_tokens=127,
        top_k=2,
        problem_n=256,
        problem_k=512,
        **config,
    )
    stream = cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)
    output = io["_keepalive"][4]
    output.zero_()
    _compile_for_launch(io, stream)(*_launch_arg_tuple(io, stream))
    expected = output.clone()
    weights, scales = io["_keepalive"][0], io["_keepalive"][2]
    stop = expert_start + expert_count
    local_weights = weights.view(7, -1)[expert_start:stop].clone()
    local_scales = scales.view(7, -1)[expert_start:stop].clone()
    weights.view(7, -1)[expert_start:stop].zero_()
    scales.view(torch.uint8).view(7, -1)[expert_start:stop].fill_(0x7F)
    io["cfg"] = replace(
        io["cfg"],
        local_expert_start=expert_start,
        local_expert_count=expert_count,
    )
    io["local_weight_dp"] = make_ptr(
        cutlass.Float4E2M1FN,
        local_weights.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    io["local_sf_dp"] = make_ptr(
        cutlass.Float8E4M3FN,
        local_scales.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=32,
    )
    output.zero_()
    _compile_for_launch(io, stream)(*_launch_arg_tuple(io, stream))
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("activation", [ActivationType.Swiglu, ActivationType.Relu2])
@pytest.mark.parametrize("tokens", [1, 16960])
def test_per_token_nvfp4_requantizes_bf16_fc1_output(activation, tokens):
    """Both GEMMs need dynamic scales, including non-gated Nemotron experts."""
    pair = map_trtllm_nvfp4_moe_tactic(
        [-1, -1],
        activation_type=int(activation),
        num_tokens=tokens,
        top_k=2,
        num_local_experts=7,
        use_per_token_sf_b=True,
    )
    fc1, fc2 = pair.fc1.cfg.build(), pair.fc2.cfg.build()
    assert not fc1.has_epilogue_quant
    assert fc1.dtype_c == int(DType.BF16)
    assert fc1.has_per_token_sf_b
    assert fc2.has_per_token_sf_b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize("tile_n", [8, 128])
def test_nvfp4_relu2_dequantizes_global_scale_before_squaring(tile_n):
    """Ignoring the expert scale before ReLU2 causes full-model overflow."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    pair = map_trtllm_nvfp4_moe_tactic(
        [tile_n, -1],
        activation_type=int(ActivationType.Relu2),
        num_tokens=192,
        top_k=2,
        num_local_experts=8,
        use_per_token_sf_b=True,
    )
    common = dict(
        num_experts=8,
        num_tokens=192,
        top_k=2,
        problem_n=256,
        problem_k=512,
        return_output=True,
        **pair.fc1.cfg.kwargs,
    )
    ok, unscaled = reference_check(scale_gate_value=1.0, **common)
    ok_scaled, scaled = reference_check(scale_gate_value=0.5, **common)
    assert ok and ok_scaled
    torch.testing.assert_close(scaled, unscaled * 0.25, rtol=0, atol=0)


def test_per_token_nvfp4_excludes_unvalidated_tile256_pairs():
    """Stale or fresh tuning must not choose the large-prefill NaN tactic."""
    tactics = valid_prims_ts_nvfp4_moe_tactics(
        activation_type=int(ActivationType.Relu2),
        num_tokens=16960,
        top_k=22,
        num_local_experts=512,
        use_per_token_sf_b=True,
    )
    assert tactics
    assert all(tile_n <= 128 for tile_n, _ in tactics)
    with pytest.raises(ValueError, match="Unsupported Prims-TS NVFP4 tile_N=256"):
        map_trtllm_nvfp4_moe_tactic(
            [256, 2],
            activation_type=int(ActivationType.Relu2),
            num_tokens=16960,
            top_k=22,
            num_local_experts=512,
            use_per_token_sf_b=True,
        )
