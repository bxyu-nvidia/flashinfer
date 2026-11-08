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
from flashinfer.tllm_enums import ActivationType, WeightLayout
from flashinfer.utils import is_sm100a_supported


_requires_non_sm107_prims_ts = pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (10, 7),
    reason="NVFP4 Prims-TS kernels support SM100 and SM103, not SM107",
)


_requires_sm100_prims_ts = pytest.mark.skipif(
    torch.cuda.is_available() and not is_sm100a_supported(torch.device("cuda")),
    reason="NVFP4 Prims-TS kernels require Blackwell SM100A+",
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
def test_per_token_nvfp4_requantizes_bf16_fc1_output(monkeypatch, activation, tokens):
    """The opt-out retains BF16 FC1 output followed by dynamic quantization."""
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "0")
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


@pytest.mark.parametrize("tile_k", [64, 128, 192, 384])
def test_direct_nvfp4_rejects_unvalidated_k_tiles(monkeypatch, tile_k):
    """Unsupported K tiles must fail before launching the persistent kernel."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        validate_config,
    )

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    cfg = map_trtllm_nvfp4_moe_tactic(
        [128, 0],
        activation_type=int(ActivationType.Relu2),
        num_tokens=16960,
        top_k=22,
        num_local_experts=512,
        use_per_token_sf_b=True,
    ).fc1.cfg.build()
    cfg.tile_k = tile_k
    compute_warp_layout(cfg)
    with pytest.raises(ValueError, match="tile_K 256 or 512"):
        validate_config(cfg)


def test_swizzled_nvfp4_scales_reject_non_power_of_two_blocks(monkeypatch):
    """XOR swizzling must not overlap rows with a three-block SF pitch."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        validate_config,
    )

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    cfg = map_trtllm_nvfp4_moe_tactic(
        [64, 0],
        activation_type=int(ActivationType.Relu2),
        num_tokens=16960,
        top_k=22,
        num_local_experts=512,
        use_per_token_sf_b=True,
    ).fc1.cfg.build()
    cfg.tile_k = 192
    compute_warp_layout(cfg)
    with pytest.raises(ValueError, match="power-of-two block count"):
        validate_config(cfg)


@pytest.mark.parametrize("tactic", [[64, 0], [64, 3], [128, 0]])
def test_direct_nvfp4_register_budget_rejects_overcommit(monkeypatch, tactic):
    """Extra epilogue warps must not deadlock while borrowing registers."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        validate_config,
    )

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    cfg = map_trtllm_nvfp4_moe_tactic(
        tactic,
        activation_type=int(ActivationType.Relu2),
        num_tokens=16960,
        top_k=22,
        num_local_experts=512,
        use_per_token_sf_b=True,
    ).fc1.cfg.build()
    assert cfg.use_deferred_nvfp4_scale
    compute_warp_layout(cfg)
    validate_config(cfg)
    cfg.num_load_sfb_warps = 8 if tactic[0] == 64 else 1
    cfg.epilogue_regs = 96 if tactic[0] == 64 else 160
    compute_warp_layout(cfg)
    with pytest.raises(ValueError, match="register budgets exceed"):
        validate_config(cfg)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize("sf_layout", [1, 3])
@pytest.mark.parametrize("cols", [32, 168])
@pytest.mark.parametrize("defer_token", [False, True])
@_requires_sm100_prims_ts
@pytest.mark.parametrize("defer_experts", [False, True])
@pytest.mark.parametrize("vectorized,offset", [(False, 0), (True, 0), (True, 1)])
def test_direct_nvfp4_scale_metadata_matches_power_of_two_reference(
    sf_layout, cols, defer_token, defer_experts, vectorized, offset
):
    """Routing skips and row padding must preserve each FP4 block's scale."""
    from flashinfer.prims_ts.moe.nvfp4_output import normalize_nvfp4_output_scales

    if defer_experts and not defer_token:
        pytest.skip("Expert deferral also requires token metadata")
    torch.manual_seed(71)
    tile_n = 64 if sf_layout == 1 else 128
    capacity = 3 * tile_n + 5
    exponent = torch.randint(-30, 31, (capacity, cols), device="cuda")
    mantissa = torch.randint(8, 16, (capacity, cols), device="cuda").float() / 8
    raw = mantissa * torch.pow(2.0, exponent)
    raw[0] = 0
    limits = (torch.arange(4, device="cuda", dtype=torch.int32) + 1) * tile_n
    limits[0] = tile_n // 2
    limits[1] = tile_n
    padded_rows = torch.tensor([capacity - 2], device="cuda", dtype=torch.int32)
    physical_rows = torch.arange(capacity, device="cuda")
    valid = (physical_rows < padded_rows[0]) & (
        physical_rows < limits[physical_rows // tile_n]
    )
    alignment = 8 if sf_layout == 1 else 128
    rows = (capacity + alignment - 1) // alignment * alignment
    scales = torch.full((rows * cols,), 0x55, dtype=torch.uint8, device="cuda")
    if offset:
        raw_storage = torch.empty(raw.numel() + offset, device="cuda")
        raw = raw_storage[offset:].view_as(raw).copy_(raw)
        scale_storage = torch.full(
            (scales.numel() + offset,), 0x55, device="cuda", dtype=torch.uint8
        )
        scales = scale_storage[offset:]
    input_scales = torch.pow(2.0, torch.arange(53, device="cuda") % 17 - 8)
    route_map = ((physical_rows * 17 + 3) % 53).to(torch.int32)
    route_map[~valid] = 1000000
    deferred_kwargs = (
        dict(input_token_scales=input_scales, route_map=route_map)
        if defer_token
        else {}
    )
    tile_experts = torch.tensor([2, 0, 1, 2], device="cuda", dtype=torch.int32)
    gate_scales = torch.tensor([0.37, 1.125, 2.3], device="cuda")
    output_scales = torch.tensor([3.7, 0.125, 1.3], device="cuda")
    if defer_experts:
        deferred_kwargs.update(
            tile_experts=tile_experts,
            expert_gate_scales=gate_scales,
            expert_output_scales=output_scales,
        )
    token_scales = normalize_nvfp4_output_scales(
        raw,
        scales,
        limits,
        padded_rows,
        tile_n=tile_n,
        sf_layout=sf_layout,
        vectorized=vectorized,
        **deferred_kwargs,
    )
    row = torch.arange(capacity, device="cuda")[:, None]
    col = torch.arange(cols, device="cuda")[None, :]
    if sf_layout == 1:
        index = (row // 8 * (cols // 4) + col // 4) * 32 + row % 8 * 4 + col % 4
    else:
        index = (row // 128 * (cols // 4) + col // 4) * 512
        index += ((row % 32) * 4 + (row % 128) // 32) * 4 + col % 4
    expected_token = torch.pow(
        2.0, torch.ceil(torch.log2(raw.double().amax(1) / 448))
    ).float()
    expected_token[0] = 0
    expected_folded = expected_token[valid]
    if defer_token:
        token_factor = input_scales[route_map[valid].long()]
        expected_folded = expected_folded * token_factor
        if defer_experts:
            expert = tile_experts[(physical_rows[valid] // tile_n).long()]
            expected_folded = expected_folded * gate_scales[expert]
        expected_folded = expected_folded * token_factor
        if defer_experts:
            expected_folded = expected_folded * gate_scales[expert]
            expected_folded = expected_folded * output_scales[expert]
    torch.testing.assert_close(token_scales[valid], expected_folded, rtol=1e-6, atol=0)
    expected = (
        torch.where(expected_token[:, None] > 0, raw / expected_token[:, None], 0)
        .to(torch.float8_e4m3fn)
        .float()
    )
    actual = scales.view(torch.float8_e4m3fn).float()[index]
    torch.testing.assert_close(actual[valid], expected[valid], rtol=0, atol=0)
    assert torch.all(scales[index[~valid]] == 0x55)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize("tactic", [[64, 0], [64, 3]])
@_requires_sm100_prims_ts
def test_nvfp4_compact_scale_swizzle_preserves_fc1_reference(monkeypatch, tactic):
    """Swizzled staging changes neither routed scale values nor the activation."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "0")
    cfg = map_trtllm_nvfp4_moe_tactic(
        tactic,
        activation_type=int(ActivationType.Relu2),
        num_tokens=192,
        top_k=2,
        num_local_experts=8,
        use_per_token_sf_b=True,
    ).fc1.cfg.kwargs
    ok = reference_check(
        num_experts=8,
        num_tokens=192,
        top_k=2,
        problem_n=256,
        problem_k=1024,
        **(cfg | {"use_swizzled_sfb": 1}),
    )
    assert ok


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize("tactic", [[64, 0], [128, 0]])
@_requires_sm100_prims_ts
@pytest.mark.parametrize(
    "weight_layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK]
)
@pytest.mark.parametrize("defer_token", [False, True])
@pytest.mark.parametrize("wide_output", [False, True])
@pytest.mark.parametrize("prerouted", [False, True])
def test_direct_nvfp4_persistent_tiles_preserve_per_token_scales(
    monkeypatch, tactic, weight_layout, defer_token, wide_output, prerouted
):
    """Faster warps must not overwrite scales still read by another epilogue."""
    from dataclasses import replace

    import cutlass
    import cuda.bindings.driver as driver
    from cutlass.cute.runtime import make_ptr
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _compile_for_launch,
        _launch_arg_tuple,
        prepare_tensors,
    )

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEFER_NVFP4_TOKEN_SCALE", "0")
    if prerouted and not wide_output:
        pytest.skip("Pre-routed scale copies target the wide deferred FC1 path")
    if wide_output and (tactic[0] != 128 or not defer_token):
        pytest.skip("Wide output candidate targets deferred tile_N 128")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "0")
    cfg = map_trtllm_nvfp4_moe_tactic(
        tactic,
        activation_type=int(ActivationType.Relu2),
        num_tokens=4103,
        top_k=2,
        num_local_experts=8,
        use_per_token_sf_b=True,
        weight_layout=int(weight_layout),
    ).fc1.cfg.kwargs
    io = prepare_tensors(
        num_experts=8, num_tokens=4103, top_k=2, problem_n=2688, problem_k=4096, **cfg
    )
    token_sf = torch.pow(2.0, torch.arange(4103, device="cuda") % 17 - 8)
    if wide_output:
        token_sf *= 1.0 + 0.2 * torch.sin(torch.arange(4103, device="cuda").float())
    io["per_token_sf_b_dp"] = make_ptr(
        cutlass.Float32,
        token_sf.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    gate_scales = torch.linspace(0.37, 1.83, 8, device="cuda")
    output_scales = torch.linspace(0.21, 2.37, 8, device="cuda")
    if wide_output:
        for key, values in (
            ("scale_gate_dp", gate_scales),
            ("scale_c_dp", output_scales),
        ):
            io[key] = make_ptr(
                cutlass.Float32,
                values.data_ptr(),
                cutlass.AddressSpace.gmem,
                assumed_align=16,
            )
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    baseline = _compile_for_launch(io, stream)
    baseline(*_launch_arg_tuple(io, stream))
    reference = io["_keepalive"][4].T.double()
    capacity = io["total_padded_tokens"]
    packed = torch.zeros((capacity, 1344), dtype=torch.uint8, device="cuda")
    scales = torch.full((capacity, 168), float("nan"), device="cuda")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    direct_cfg = map_trtllm_nvfp4_moe_tactic(
        tactic,
        activation_type=int(ActivationType.Relu2),
        num_tokens=4103,
        top_k=2,
        num_local_experts=8,
        use_per_token_sf_b=True,
        weight_layout=int(weight_layout),
    ).fc1.cfg.build()
    direct_cfg = replace(direct_cfg, defer_nvfp4_token_scale=int(defer_token))
    if wide_output:
        stages = {
            key: 3
            for key in (
                "num_stages_a",
                "num_stages_b",
                "num_stages_smem_sfa",
                "num_stages_smem_sfb",
                "num_stages_tmem_sfa",
                "num_stages_tmem_sfb",
            )
        }
        direct_cfg = replace(
            direct_cfg,
            **stages,
            packed_nvfp4_epilogue=1,
            vectorize_routed_sfb=int(not prerouted),
            prerouted_sfb_ldgsts=int(prerouted),
            stage_nvfp4_scales=int(prerouted),
            static_persistent_clusters=(
                torch.cuda.get_device_properties(0).multi_processor_count // 2
                if prerouted
                else 0
            ),
            defer_nvfp4_global_scales=1,
            gather_regs=64,
            route_act=2,
            route_sfs_act=2,
            fuse_operand_sf_loads=1,
            fuse_sf_copy_to_mma=1,
            tile_k=512,
            num_load_sfb_warps=4,
            num_load_sfa_warps=1,
            num_epilogue_warps=8,
            epi_tile_n=64,
            epilogue_regs=96 if prerouted else 112,
        )
    direct_io = dict(
        io,
        cfg=direct_cfg,
        c0_dp=make_ptr(
            cutlass.Float4E2M1FN,
            packed.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        ),
        sf_c_dp=make_ptr(
            cutlass.Float32,
            scales.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        ),
    )
    if prerouted:
        from flashinfer.prims_ts.moe.routed_scales import (
            preroute_nvfp4_activation_scales,
        )

        prepared_scales = preroute_nvfp4_activation_scales(
            io["_keepalive"][3],
            io["_keepalive"][8],
            io["_keepalive"][7],
            torch.tensor([capacity], dtype=torch.int32, device="cuda"),
            capacity=capacity,
            scale_columns=4096 // 16,
        )
        direct_io["sfb_dp"] = make_ptr(
            cutlass.Float8E4M3FN,
            prepared_scales.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        )
    direct = _compile_for_launch(direct_io, stream)

    def launch():
        direct(
            *_launch_arg_tuple(
                direct_io, driver.CUstream(torch.cuda.current_stream().cuda_stream)
            )
        )

    launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    levels = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device="cuda",
    )
    for _ in range(3):
        graph.replay()
        values = levels[torch.stack((packed & 15, packed >> 4), -1).long()].flatten(1)
        actual = values.double() * scales.double().repeat_interleave(16, dim=1)
        if defer_token:
            route_map = io["_keepalive"][8]
            actual *= token_sf[route_map[:capacity].long()].double().square()[:, None]
        if wide_output:
            tile_experts = io["_keepalive"][6]
            expert = (
                tile_experts[torch.arange(capacity, device="cuda") // 128]
                .long()
                .clamp(0, 7)
            )
            actual *= gate_scales[expert].double().square()[:, None]
            actual *= output_scales[expert].double()[:, None]
        valid = reference.abs().amax(1) > 0
        assert torch.isfinite(actual[valid]).all()
        row_rmse = (
            (actual[valid] - reference[valid]).square().mean(1)
            / reference[valid].square().mean(1)
        ).sqrt()
        assert row_rmse.max().item() < 0.2

    if weight_layout != WeightLayout.MajorK or not hasattr(
        direct_cfg, "local_expert_count"
    ):
        return

    # A resident range must use local pointers, including its FP8 weight scales.
    expected_packed = packed.clone()
    expected_scales = scales.clone()
    weights, weight_scales = io["_keepalive"][0], io["_keepalive"][2]
    local_weights = weights.view(8, -1)[4:].clone()
    local_scales = weight_scales.view(8, -1)[4:].clone()
    weights.view(8, -1)[4:].zero_()
    weight_scales.view(torch.uint8).view(8, -1)[4:].fill_(0x7F)
    direct_io.update(
        cfg=replace(direct_cfg, local_expert_start=4, local_expert_count=4),
        local_weight_dp=make_ptr(
            cutlass.Float4E2M1FN,
            local_weights.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=16,
        ),
        local_sf_dp=make_ptr(
            cutlass.Float8E4M3FN,
            local_scales.data_ptr(),
            cutlass.AddressSpace.gmem,
            assumed_align=32,
        ),
    )
    packed.zero_()
    scales.fill_(float("nan"))
    _compile_for_launch(direct_io, stream)(*_launch_arg_tuple(direct_io, stream))
    torch.testing.assert_close(packed[valid], expected_packed[valid], rtol=0, atol=0)
    torch.testing.assert_close(scales[valid], expected_scales[valid], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA GPU required")
@_requires_non_sm107_prims_ts
@pytest.mark.parametrize("tactic", [[64, 0], [128, 0]])
@_requires_sm100_prims_ts
def test_direct_nvfp4_output_matches_core_reference(monkeypatch, tactic):
    """The core harness must allocate and decode the deferred FP32 block scales."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import reference_check

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    cfg = map_trtllm_nvfp4_moe_tactic(
        tactic,
        activation_type=int(ActivationType.Relu2),
        num_tokens=33,
        top_k=2,
        num_local_experts=8,
        use_per_token_sf_b=True,
    ).fc1.cfg.kwargs
    assert cfg["use_deferred_nvfp4_scale"]
    assert reference_check(
        num_experts=8, num_tokens=33, top_k=2, problem_n=256, problem_k=1024, **cfg
    )


@pytest.mark.parametrize("flag", ["1", "true", "yes", "on"])
def test_direct_nvfp4_keeps_four_over_six_quantization_policy(monkeypatch, flag):
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", flag)
    cfg = map_trtllm_nvfp4_moe_tactic(
        [128, 0],
        activation_type=int(ActivationType.Relu2),
        num_tokens=16960,
        top_k=22,
        num_local_experts=512,
        use_per_token_sf_b=True,
    ).fc1.cfg.build()
    assert not cfg.use_deferred_nvfp4_scale


@pytest.mark.parametrize("gather_registers", [48, 64])
def test_direct_nvfp4_register_budget_counts_inactive_warp_groups(
    monkeypatch, gather_registers
):
    """Fused loads must account for inactive leader warp groups."""
    from dataclasses import replace
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        compute_warp_layout,
        validate_config,
    )

    monkeypatch.setenv("FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1", "1")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEFER_NVFP4_TOKEN_SCALE", "1")
    cfg = map_trtllm_nvfp4_moe_tactic(
        [128, 0],
        activation_type=int(ActivationType.Relu2),
        num_tokens=16960,
        top_k=22,
        num_local_experts=512,
        use_per_token_sf_b=True,
    ).fc1.cfg.build()
    stages = {
        key: 2
        for key in (
            "num_stages_a",
            "num_stages_b",
            "num_stages_smem_sfa",
            "num_stages_smem_sfb",
            "num_stages_tmem_sfa",
            "num_stages_tmem_sfb",
        )
    }
    cfg = replace(
        cfg,
        **stages,
        tile_n=256,
        mma_n=256,
        prerouted_sfb_ldgsts=0,
        stage_nvfp4_scales=0,
        vectorize_routed_sfb=1,
        num_stages_tmem_acc=1,
        num_epilogue_warps=16,
        epi_tile_n=64,
        epilogue_regs=88,
        gather_regs=gather_registers,
    )
    compute_warp_layout(cfg)
    if gather_registers == 64:
        with pytest.raises(ValueError, match="register budgets exceed"):
            validate_config(cfg)
    else:
        validate_config(cfg)


@_requires_sm100_prims_ts
@pytest.mark.parametrize("columns", [32, 64, 256])
@pytest.mark.parametrize("offset", [0, 4])
def test_prerouted_nvfp4_scale_tiles_mask_padding_and_replay(columns, offset):
    """Invalid route IDs must remain unread, including partial final tiles."""
    from flashinfer.prims_ts.moe.routed_scales import preroute_nvfp4_activation_scales

    capacity = 389
    source = torch.randint(
        0, 256, (7 * columns + offset,), dtype=torch.uint8, device="cuda"
    )[offset:].view(7, columns)
    rows = torch.arange(capacity, device="cuda")
    limits = torch.tensor([13, 128, 383, 387], dtype=torch.int32, device="cuda")
    padded = torch.tensor([387], dtype=torch.int32, device="cuda")
    valid = (rows < limits[rows // 128]) & (rows < padded[0])
    route = (rows % 7).int()
    route[~valid] = 1000000000
    reference = torch.zeros((512, columns), dtype=torch.uint8, device="cuda")
    reference[:capacity][valid] = source[route[valid].long()]
    row = torch.arange(512, device="cuda")[:, None]
    col = torch.arange(columns, device="cuda")[None, :]
    index = (row // 128 * (columns // 4) + col // 4) * 512
    index += ((row % 32) * 4 + (row % 128) // 32) * 4 + col % 4
    kwargs = dict(capacity=capacity, scale_columns=columns)
    preroute_nvfp4_activation_scales(source, route, limits, padded, **kwargs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = preroute_nvfp4_activation_scales(
            source, route, limits, padded, **kwargs
        )
    graph.replay()
    assert torch.equal(output[index], reference)
    output.fill_(0x55)
    graph.replay()
    assert torch.equal(output[index], reference)
