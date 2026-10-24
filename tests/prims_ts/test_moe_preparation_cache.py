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


@pytest.mark.parametrize("fc", ["fc1", "fc2"])
@pytest.mark.parametrize("swap_ab", [False, True])
def test_launch_io_refreshes_tensor_pointers_and_shapes(
    monkeypatch, config_module, fc, swap_ab
):
    import cutlass.cute.runtime

    # No GPU launches: use real CPU tensors and record each bound address.
    monkeypatch.setattr(
        cutlass.cute.runtime, "make_ptr", lambda dtype, address, space, **kw: address
    )
    cfg = config_module.make_config(
        dtype_a=3,
        dtype_b=3,
        dtype_c=3,
        mma_k=16,
        tile_k=64,
        batch_mode=0 if swap_ab else 1,
        transpose_mma_output=int(swap_ab),
        weight_layout=int(WeightLayout.MajorK),
    )

    def inputs(tokens):
        return dict(
            fc=fc,
            cfg=cfg,
            hidden_states=torch.empty((tokens, 128), dtype=torch.bfloat16),
            gemm1_weights=torch.empty((2, 128, 128), dtype=torch.bfloat16),
            gemm2_weights=torch.empty((2, 128, 128), dtype=torch.bfloat16),
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
    old = tensor_adapter.build_bf16_launch_io(**old_inputs)
    new = tensor_adapter.build_bf16_launch_io(**new_inputs)
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
    assert compile_cache._bf16_runtime_config_cached.cache_info().misses == 1
    # A hit must still run input validation.
    new_inputs["gemm1_weights" if fc == "fc1" else "gemm2_weights"] = torch.empty(1)
    with pytest.raises(ValueError, match="invalid MajorK shape"):
        tensor_adapter.build_bf16_launch_io(**new_inputs)
