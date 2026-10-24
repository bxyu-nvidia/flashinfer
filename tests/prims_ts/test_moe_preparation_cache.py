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

from dataclasses import asdict
from unittest.mock import Mock

import pytest
import torch

from flashinfer.prims_ts.moe import compile_cache, config_mapper, tensor_adapter
from flashinfer.tllm_enums import ActivationType, WeightLayout


@pytest.fixture(autouse=True)
def clear_caches():
    caches = (
        config_mapper._build_config_cached,
        config_mapper._bf16_config_pair_cached,
        config_mapper._block_major_k_bytes_cached,
        config_mapper._immutable_config_spec_cached,
        compile_cache._bf16_spec_runtime_config_cached,
        compile_cache._bf16_runtime_config_cached,
    )
    for cache in caches:
        cache.cache_clear()
    yield
    for cache in caches:
        cache.cache_clear()


@pytest.fixture
def config_module():
    from flashinfer.prims_ts.batched_gemm import batched_gemm_config

    return batched_gemm_config


def test_config_build_reuses_work_without_sharing_mutable_config(
    monkeypatch, config_module
):
    build = Mock(wraps=config_module.make_config)
    monkeypatch.setattr(config_module, "make_config", build)
    spec = config_mapper.PrimsTsConfigSpec({"tile_n": 128, "tile_k": 64})
    first = spec.build()
    first.tile_n = 256  # Assembly or a caller may mutate its own config.
    second = config_mapper.PrimsTsConfigSpec({"tile_k": 64, "tile_n": 128}).build()
    assert second.tile_n == 128
    assert build.call_count == 1
    spec.kwargs["tile_k"] = 128
    assert spec.build().tile_k == 128
    assert build.call_count == 2


@pytest.fixture
def map_pair(monkeypatch):
    def resolve(**kwargs):
        cfg = config_mapper.PrimsTsConfigSpec({"tile_n": kwargs["tile_n"]})
        fc = config_mapper.PrimsTsMoeGemmConfig(cfg, 0)
        return config_mapper.PrimsTsGemmPair(kwargs["tile_n"], 0, fc, fc)

    resolve = Mock(side_effect=resolve)
    monkeypatch.setattr(config_mapper, "_required_json_moe_config_pair", resolve)
    return resolve


def test_mapping_reuses_work_without_sharing_option_dicts(map_pair):
    first = config_mapper.map_trtllm_bf16_moe_tactic([128, 0])
    first.fc1.cfg.kwargs["tile_n"] = 256
    second = config_mapper.map_trtllm_bf16_moe_tactic((128, 0))
    assert second.fc1.cfg.kwargs["tile_n"] == 128
    assert second.fc2.cfg.kwargs["tile_n"] == 128
    assert first.fc2.cfg.kwargs["tile_n"] == 128
    assert map_pair.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        {"activation_type": int(ActivationType.Relu2)},
        {"num_tokens": 64},
        {"top_k": 2},
        {"num_local_experts": 4},
        {"fc1_has_bias": True},
        {"fc2_has_bias": True},
        {"has_gemm1_alpha": True},
        {"has_gemm1_beta": True},
        {"has_gemm1_clamp_limit": True},
        {"enable_pdl": True},
        {"weight_layout": int(WeightLayout.BlockMajorK)},
    ],
)
def test_mapping_revalidates_changed_options(map_pair, change):
    config_mapper.map_trtllm_bf16_moe_tactic([128, 0])
    config_mapper.map_trtllm_bf16_moe_tactic([128, 0], **change)
    assert map_pair.call_count == 2


def test_mapping_revalidates_debug_mode_and_tactic(map_pair, monkeypatch):
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    config_mapper.map_trtllm_bf16_moe_tactic([128, 0])
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "1")
    config_mapper.map_trtllm_bf16_moe_tactic([128, 0])
    config_mapper.map_trtllm_bf16_moe_tactic([64, 0])
    config_mapper.map_trtllm_bf16_moe_tactic([64, 1])
    assert map_pair.call_count == 4


def test_mapping_does_not_cache_failure(map_pair):
    map_pair.side_effect = ValueError("invalid schedule")
    for _ in range(2):
        with pytest.raises(ValueError, match="invalid schedule"):
            config_mapper.map_trtllm_bf16_moe_tactic([128, 0])
    assert map_pair.call_count == 2


def test_runtime_config_and_hash_match_uncached_and_do_not_share_mutations(
    monkeypatch, config_module
):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    original = batched_gemm_run._runtime_config
    normalize = Mock(wraps=original)
    original_hash = compile_cache.stable_config_hash
    serialize = Mock(wraps=original_hash)
    monkeypatch.setattr(batched_gemm_run, "_runtime_config", normalize)
    monkeypatch.setattr(compile_cache, "stable_config_hash", serialize)
    cfg = config_module.make_config(tile_k=64, use_unroll_loop_2x_for_mma=1)
    for k, unroll in ((128, 1), (192, 0)):
        expected = original(cfg, k)
        first, key = compile_cache.get_bf16_runtime_config(cfg, k)
        assert asdict(first) == asdict(expected)
        assert key == original_hash(expected)
        assert first.use_unroll_loop_2x_for_mma == unroll
        first.tile_k = 512
        second, second_key = compile_cache.get_bf16_runtime_config(cfg, k)
        assert asdict(second) == asdict(expected)
        assert second_key == key
    assert normalize.call_count == 2
    assert serialize.call_count == 2
    cfg.tile_k = 128
    changed, _ = compile_cache.get_bf16_runtime_config(cfg, 128)
    assert changed.tile_k == 128
    assert changed.use_unroll_loop_2x_for_mma == 0
    assert normalize.call_count == 3


def test_runtime_config_failure_is_not_cached(config_module):
    cfg = config_module.make_config(use_deepseek_fp8=1, tile_k=64)
    for _ in range(2):
        with pytest.raises(ValueError, match="requires tile_k=128"):
            compile_cache.get_bf16_runtime_config(cfg, 128)
    assert compile_cache._bf16_runtime_config_cached.cache_info().currsize == 0


@pytest.mark.parametrize("layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK])
@pytest.mark.parametrize("use_spec", [False, True, "immutable"])
@pytest.mark.parametrize("fc", ["fc1", "fc2"])
@pytest.mark.parametrize("swap_ab", [False, True])
def test_launch_io_refreshes_tensor_pointers_and_shapes(
    monkeypatch, config_module, fc, swap_ab, use_spec, layout
):
    import cutlass.cute.runtime

    # No GPU launches: use real CPU tensors and record each bound address.
    make_ptr = Mock(side_effect=lambda dtype, address, space, **kw: address)
    monkeypatch.setattr(cutlass.cute.runtime, "make_ptr", make_ptr)
    options = dict(
        dtype_a=3,
        dtype_b=3,
        dtype_c=3,
        mma_k=16,
        tile_k=64,
        batch_mode=0 if swap_ab else 1,
        transpose_mma_output=int(swap_ab),
        weight_layout=int(layout),
    )

    cfg = (
        config_mapper.PrimsTsConfigSpec(options)
        if use_spec
        else config_module.make_config(**options)
    )

    if use_spec == "immutable":
        cfg = config_mapper._immutable_config_spec_cached(
            tuple(sorted(options.items()))
        )

    weight_shape = (2, 128, 128) if layout == WeightLayout.MajorK else (2, 2, 128, 64)

    def inputs(tokens):
        return dict(
            fc=fc,
            cfg=cfg,
            hidden_states=torch.empty((tokens, 128), dtype=torch.bfloat16),
            gemm1_weights=torch.empty(weight_shape, dtype=torch.bfloat16),
            gemm2_weights=torch.empty(weight_shape, dtype=torch.bfloat16),
            gemm1_output=torch.empty((tokens, 128), dtype=torch.bfloat16),
            gemm2_output=torch.empty((tokens, 128), dtype=torch.bfloat16),
            tile_idx=torch.zeros(4, dtype=torch.int32),
            mn_limit=torch.zeros(4, dtype=torch.int32),
            route_map=torch.zeros(tokens, dtype=torch.int32),
            num_non_exiting_ctas=torch.ones(1, dtype=torch.int32),
            total_num_padded_tokens=torch.tensor([tokens], dtype=torch.int32),
            routed_token_capacity=tokens,
            activation_type=int(ActivationType.Identity),
            num_experts=2,
            num_tokens=tokens,
            top_k=1,
            intermediate_size=128,
            hidden_size=128,
        )

    old_inputs, new_inputs = inputs(2), inputs(4)
    optional_params = Mock(wraps=tensor_adapter._make_gemm1_oa_ptrs)
    monkeypatch.setattr(tensor_adapter, "_make_gemm1_oa_ptrs", optional_params)
    old = tensor_adapter.build_bf16_launch_io(**old_inputs)
    assert make_ptr.call_count == 8
    if use_spec == "immutable":

        def unexpected_layout(_self):
            pytest.fail("Warm immutable launch must use its precomputed weight layout")

        for name in ("block_major_k_bytes", "weight_dtype_tma_bits"):
            monkeypatch.setattr(
                config_module.BatchedGemmConfig, name, property(unexpected_layout)
            )
    new = tensor_adapter.build_bf16_launch_io(**new_inputs)
    assert make_ptr.call_count == 16
    for name in (
        "bias_dp",
        "scale_gate_dp",
        "gemm1_alpha_dp",
        "gemm1_beta_dp",
        "gemm1_clamp_limit_dp",
        "per_token_sf_a_dp",
        "per_token_sf_b_dp",
    ):
        assert new[name] is new["scale_c_dp"]
    assert new["sf_c_dp"] is new["c0_dp"]
    assert new["act_dp"] is new["b_dp" if swap_ab else "a_dp"]
    activation = "hidden_states" if fc == "fc1" else "gemm1_output"
    weights = "gemm1_weights" if fc == "fc1" else "gemm2_weights"
    output = "gemm1_output" if fc == "fc1" else "gemm2_output"
    for pointer, tensor in (
        ("a_dp", weights if swap_ab else activation),
        ("b_dp", activation if swap_ab else weights),
        ("c0_dp", output),
        ("route_map_dp", "route_map"),
        ("tile_idx_dp", "tile_idx"),
    ):
        assert new[pointer] == new_inputs[tensor].data_ptr()
        assert new[pointer] != old[pointer]
    assert new["shape"] == ((128, 4, 128, 2, 4) if swap_ab else (4, 128, 128, 2, 4))
    assert old["shape"] == ((128, 2, 128, 2, 2) if swap_ab else (2, 128, 128, 2, 2))
    assert new["config_hash"] == old["config_hash"]
    optional_params.assert_not_called()
    # The prebound tuple must track the shared kernel ABI and live IO values.
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _launch_arg_tuple

    for io in (old, new):
        assert (*io["_launch_args"], 123) == _launch_arg_tuple(io, 123)
    if fc == "fc1":
        for name in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit"):
            with pytest.raises(ValueError, match="disabled"):
                tensor_adapter.build_bf16_launch_io(
                    **new_inputs, **{name: torch.ones(2, dtype=torch.float32)}
                )
    cache = (
        compile_cache._bf16_spec_runtime_config_cached
        if use_spec
        else compile_cache._bf16_runtime_config_cached
    )
    assert cache.cache_info().misses == 1
    # A hit must still run input validation.
    new_inputs["gemm1_weights" if fc == "fc1" else "gemm2_weights"] = torch.empty(1)
    error = (
        "invalid MajorK shape" if layout == WeightLayout.MajorK else "rank-4 storage"
    )
    with pytest.raises(ValueError, match=error):
        tensor_adapter.build_bf16_launch_io(**new_inputs)


@pytest.mark.parametrize("use_dummy", [False, True])
def test_optional_activation_parameters_keep_live_pointers_and_validation(use_dummy):
    import cutlass
    from types import SimpleNamespace

    cfg = SimpleNamespace(has_gemm1_alpha=1, has_gemm1_beta=0, has_gemm1_clamp_limit=0)
    alpha = torch.ones(2, dtype=torch.float32)
    make_ptr = Mock(side_effect=lambda dtype, address, space, **kw: (dtype, address))
    dummy = object() if use_dummy else None
    args = dict(
        make_ptr=make_ptr,
        cutlass=cutlass,
        cfg=cfg,
        fc="fc1",
        dummy_data_ptr=123,
        num_experts=2,
        device=torch.device("cpu"),
        gemm1_alpha=alpha,
        gemm1_beta=None,
        gemm1_clamp_limit=None,
        dummy_float_ptr=dummy,
    )
    result = tensor_adapter._make_gemm1_oa_ptrs(**args)
    assert result[0] == (cutlass.Float32, alpha.data_ptr())
    assert result[3] is alpha
    assert make_ptr.call_count == (1 if use_dummy else 3)
    if use_dummy:
        assert result[1] is result[2] is dummy
    args["gemm1_alpha"] = torch.ones(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="must be float32"):
        tensor_adapter._make_gemm1_oa_ptrs(**args)


@pytest.mark.parametrize("layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK])
@pytest.mark.parametrize("activation", [0, 1, 3])
def test_spec_runtime_matches_materialized_config_and_isolates_mutations(
    monkeypatch, layout, activation
):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run

    spec = config_mapper.PrimsTsConfigSpec(
        dict(
            dtype_a=3,
            dtype_b=3,
            dtype_c=3,
            tile_k=64,
            mma_k=16,
            weight_layout=int(layout),
            act_kind=activation,
            use_unroll_loop_2x_for_mma=1,
        )
    )
    normalize = Mock(wraps=batched_gemm_run._runtime_config)
    monkeypatch.setattr(batched_gemm_run, "_runtime_config", normalize)
    for k in (128, 192):
        expected, expected_key = compile_cache.get_bf16_runtime_config(spec.build(), k)
        first, key = compile_cache.get_bf16_runtime_config(spec, k)
        assert asdict(first) == asdict(expected)
        assert key == expected_key
        first.tile_k = 512
        first.num_mma_warps = 99  # Simulate kernel assembly modifying warp layout.
        second, key = compile_cache.get_bf16_runtime_config(spec, k)
        assert asdict(second) == asdict(expected)
        assert key == expected_key
    assert normalize.call_count == 4  # Once per K for each of the two paths.
    spec.kwargs["tile_k"] = 128
    changed, changed_key = compile_cache.get_bf16_runtime_config(spec, 128)
    expected, expected_key = compile_cache.get_bf16_runtime_config(spec.build(), 128)
    assert changed.tile_k == 128
    assert asdict(changed) == asdict(expected)
    assert changed_key == expected_key


def test_spec_validation_and_launch_hit_skip_build_and_field_scan(monkeypatch):
    spec = config_mapper.PrimsTsConfigSpec(
        dict(
            dtype_a=3,
            dtype_b=3,
            dtype_c=3,
            mma_k=16,
            tile_k=64,
            weight_layout=int(WeightLayout.BlockMajorK),
        )
    )
    block_bytes = spec.block_major_k_bytes
    _, expected_key = compile_cache.get_bf16_runtime_config(spec, 128)

    def unexpected(*args, **kwargs):
        pytest.fail("Warm spec path must not build or scan a materialized config")

    monkeypatch.setattr(config_mapper.PrimsTsConfigSpec, "build", unexpected)
    monkeypatch.setattr(config_mapper, "_build_config_cached", unexpected)
    monkeypatch.setattr(compile_cache, "_build_config_cached", unexpected)
    monkeypatch.setattr(compile_cache, "_config_field_names", unexpected)
    for _ in range(2):
        assert spec.block_major_k_bytes == block_bytes
        _, key = compile_cache.get_bf16_runtime_config(spec, 128)
        assert key == expected_key


def test_spec_layout_metadata_tracks_mutated_options():
    spec = config_mapper.PrimsTsConfigSpec(
        dict(
            dtype_a=3,
            dtype_b=3,
            mma_k=16,
            tile_k=64,
            weight_layout=int(WeightLayout.BlockMajorK),
        )
    )
    before = spec.block_major_k_bytes
    assert before == spec.build().block_major_k_bytes
    spec.kwargs["tile_k"] = 128
    assert spec.block_major_k_bytes == spec.build().block_major_k_bytes
    assert config_mapper._block_major_k_bytes_cached.cache_info().misses == 2


def test_spec_runtime_failure_is_not_cached():
    spec = config_mapper.PrimsTsConfigSpec(dict(use_deepseek_fp8=1, tile_k=64))
    for _ in range(2):
        with pytest.raises(ValueError, match="requires tile_k=128"):
            compile_cache.get_bf16_runtime_config(spec, 128)
    assert compile_cache._bf16_spec_runtime_config_cached.cache_info().currsize == 0


def test_internal_specs_are_immutable_and_public_mutations_are_isolated(map_pair):
    from dataclasses import FrozenInstanceError

    pair = config_mapper._resolve_bf16_moe_tactic([128, 0])
    spec = pair.fc1.cfg
    assert isinstance(spec, config_mapper._ImmutablePrimsTsConfigSpec)
    assert config_mapper._resolve_bf16_moe_tactic([128, 0]) is pair
    with pytest.raises(FrozenInstanceError):
        spec.block_major_k_bytes = 999
    with pytest.raises(TypeError):
        spec.config_items[0] = ("tile_n", 256)
    public = config_mapper.map_trtllm_bf16_moe_tactic([128, 0])
    public.fc1.cfg.kwargs["tile_n"] = 256
    public.fc1.cfg.kwargs["tile_k"] = 128
    assert spec.build().tile_n == 128
    assert config_mapper._resolve_bf16_moe_tactic([128, 0]) is pair
    assert map_pair.call_count == 1


def test_internal_warm_preparation_does_not_sort_build_or_recompute_layout(
    map_pair, monkeypatch, config_module
):
    pair = config_mapper._resolve_bf16_moe_tactic([128, 0])
    expected, expected_key = compile_cache.get_bf16_runtime_config(pair.fc1.cfg, 128)
    expected_fields = asdict(expected)
    block_bytes = pair.fc1.cfg.block_major_k_bytes

    def unexpected(*args, **kwargs):
        pytest.fail("Warm immutable preparation must reuse keys and layout metadata")

    monkeypatch.setattr(config_mapper, "sorted", unexpected, raising=False)
    monkeypatch.setattr(compile_cache, "sorted", unexpected, raising=False)
    monkeypatch.setattr(config_mapper, "_build_config_cached", unexpected)
    monkeypatch.setattr(compile_cache, "_build_config_cached", unexpected)
    monkeypatch.setattr(
        config_module.BatchedGemmConfig, "block_major_k_bytes", property(unexpected)
    )
    for _ in range(2):
        same_pair = config_mapper._resolve_bf16_moe_tactic([128, 0])
        assert same_pair is pair
        assert same_pair.fc1.cfg.block_major_k_bytes == block_bytes
        launch_cfg, key = compile_cache.get_bf16_runtime_config(same_pair.fc1.cfg, 128)
        assert asdict(launch_cfg) == expected_fields
        assert key == expected_key
        launch_cfg.tile_n = 256
        launch_cfg.num_mma_warps = 99


@pytest.mark.parametrize("layout", [WeightLayout.MajorK, WeightLayout.BlockMajorK])
@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("k", [128, 192])
def test_immutable_specs_match_runtime_config_hash_and_layout(layout, swap_ab, k):
    options = dict(
        dtype_a=3,
        dtype_b=3,
        dtype_c=3,
        mma_k=16,
        tile_k=64,
        batch_mode=0 if swap_ab else 1,
        transpose_mma_output=int(swap_ab),
        weight_layout=int(layout),
        use_unroll_loop_2x_for_mma=1,
    )
    public = config_mapper.PrimsTsConfigSpec(options)
    spec = config_mapper._immutable_config_spec_cached(tuple(sorted(options.items())))
    original, original_key = compile_cache.get_bf16_runtime_config(public.build(), k)
    actual, key = compile_cache.get_bf16_runtime_config(spec, k)
    assert asdict(actual) == asdict(original)
    assert key == original_key
    assert spec.weight_layout == original.weight_layout
    assert spec.weight_dtype_tma_bits == original.weight_dtype_tma_bits
    assert spec.block_major_k_bytes == original.block_major_k_bytes
    assert actual.use_unroll_loop_2x_for_mma == (1 if k == 128 else 0)


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize(
    "replacement", [None, "routing", "activation", "resize_routing"]
)
def test_fc2_reuses_only_matching_live_fc1_pointers(monkeypatch, swap_ab, replacement):
    import cutlass.cute.runtime

    # Distinct wrapper identities let this test detect real reuse, not just
    # equality of addresses returned by independently constructed wrappers.
    make_ptr = Mock(
        side_effect=lambda dtype, address, space, **kw: (
            dtype,
            address,
            kw["assumed_align"],
        )
    )
    monkeypatch.setattr(cutlass.cute.runtime, "make_ptr", make_ptr)
    options = dict(
        dtype_a=3,
        dtype_b=3,
        dtype_c=3,
        mma_k=16,
        tile_k=64,
        batch_mode=0 if swap_ab else 1,
        transpose_mma_output=int(swap_ab),
        weight_layout=int(WeightLayout.BlockMajorK),
    )
    spec = config_mapper._immutable_config_spec_cached(tuple(sorted(options.items())))
    args = dict(
        cfg=spec,
        hidden_states=torch.empty(2, 128, dtype=torch.bfloat16),
        gemm1_weights=torch.empty(2, 2, 128, 64, dtype=torch.bfloat16),
        gemm2_weights=torch.empty(2, 2, 128, 64, dtype=torch.bfloat16),
        gemm1_output=torch.empty(2, 128, dtype=torch.bfloat16),
        gemm2_output=torch.empty(2, 128, dtype=torch.bfloat16),
        tile_idx=torch.zeros(4, dtype=torch.int32),
        mn_limit=torch.zeros(4, dtype=torch.int32),
        route_map=torch.zeros(2, dtype=torch.int32),
        num_non_exiting_ctas=torch.ones(1, dtype=torch.int32),
        total_num_padded_tokens=torch.tensor([2], dtype=torch.int32),
        routed_token_capacity=2,
        activation_type=int(ActivationType.Identity),
        num_experts=2,
        num_tokens=2,
        top_k=1,
        intermediate_size=128,
        hidden_size=128,
    )
    first = tensor_adapter.build_bf16_launch_io(fc="fc1", **args)
    assert make_ptr.call_count == 8
    if replacement == "routing":
        args["route_map"] = torch.zeros(2, dtype=torch.int32)
    elif replacement == "resize_routing":
        args["route_map"].resize_(8192)
        assert args["route_map"].data_ptr() != first["_routing_data_ptrs"][1]
    elif replacement == "activation":
        args["gemm1_output"] = torch.empty(2, 128, dtype=torch.bfloat16)
    second = tensor_adapter.build_bf16_launch_io(fc="fc2", _reuse_fc1_io=first, **args)
    assert make_ptr.call_count == 11 + (
        4
        if replacement in ("routing", "resize_routing")
        else 1
        if replacement == "activation"
        else 0
    )
    activation_dp = second["b_dp" if swap_ab else "a_dp"]
    assert activation_dp[1] == args["gemm1_output"].data_ptr()
    assert (activation_dp is first["c0_dp"]) == (replacement != "activation")
    for name in ("tile_idx", "route_map", "mn_limit", "num_non_exiting_ctas"):
        assert second[name + "_dp"][1] == args[name].data_ptr()
        assert (second[name + "_dp"] is first[name + "_dp"]) == (
            replacement not in ("routing", "resize_routing")
        )
    assert second["c0_dp"][1] == args["gemm2_output"].data_ptr()
    assert second["cfg"] is not first["cfg"]
    # Reuse does not bypass checks on the current inputs, even if the address
    # remains unchanged and would otherwise match the saved pointer.
    args["route_map"] = args["route_map"].as_strided((2,), (0,))
    assert args["route_map"].data_ptr() == second["route_map_dp"][1]
    with pytest.raises(ValueError, match="route_map must be contiguous"):
        tensor_adapter.build_bf16_launch_io(fc="fc2", _reuse_fc1_io=first, **args)


def test_spec_runtime_copy_avoids_generic_copy_and_preserves_private_fields(
    monkeypatch,
):
    spec = config_mapper._immutable_config_spec_cached((("tile_k", 64),))
    first, first_key = compile_cache.get_bf16_runtime_config(spec, 128)
    expected = asdict(first)

    # Warm cache should only duplicate the already-normalized field dictionary.
    def unexpected(*args, **kwargs):
        pytest.fail("Warm spec launch must not use generic copy.copy")

    monkeypatch.setattr(compile_cache, "copy", unexpected)
    second, second_key = compile_cache.get_bf16_runtime_config(spec, 128)
    assert first_key == second_key
    assert asdict(second) == expected
    assert first.__dict__ is not second.__dict__
    first.tile_k = 512
    first.num_mma_warps = 99
    third, _ = compile_cache.get_bf16_runtime_config(spec, 128)
    assert asdict(third) == expected
