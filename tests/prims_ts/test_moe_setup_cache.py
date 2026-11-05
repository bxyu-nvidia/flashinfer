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

import gc
import importlib
from unittest.mock import Mock
import weakref

import pytest
import torch

from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode


@pytest.fixture
def backend():
    module = importlib.import_module("flashinfer.fused_moe.backends.prims_ts.bf16_op")
    module._BF16_TUNING_CONFIG_CACHE.clear()
    module._empty_routing_placeholder.cache_clear()
    yield module
    module._BF16_TUNING_CONFIG_CACHE.clear()
    module._empty_routing_placeholder.cache_clear()


def runner(backend, *, num_experts=4, hidden_size=128):
    return backend.PrimsTsBf16MoERunner(
        object(),
        top_k=2,
        num_local_experts=2,
        num_experts=num_experts,
        hidden_size=hidden_size,
        intermediate_size=128,
    )


def inputs(tokens=2, *, mode=RoutingInputMode.FromLogits):
    logits = mode == RoutingInputMode.FromLogits
    packed = mode == RoutingInputMode.PackedPrecomputed
    return MoeRunnerInputs(
        output=torch.empty(tokens, 128, dtype=torch.bfloat16),
        routing_logits=torch.empty(tokens, 4) if logits else None,
        topk_ids=torch.empty(0 if logits else (tokens, 2), dtype=torch.int32),
        expert_weights=torch.empty(0 if logits or packed else (tokens, 2)),
        hidden_states=torch.empty(tokens, 128, dtype=torch.bfloat16),
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )


def get_config(
    backend, model, call_inputs, *, mode=RoutingInputMode.FromLogits, limit=8192
):
    return backend._get_bf16_tuning_config(
        model,
        call_inputs,
        tune_max_num_tokens=limit,
        routing_input_mode=mode,
    )


@pytest.mark.parametrize("mode", list(RoutingInputMode))
def test_cached_config_matches_builder_and_reuses_across_token_counts(
    backend, monkeypatch, mode
):
    model = runner(backend)
    original = model._make_tuning_config
    expected = original(
        inputs(mode=mode),
        routing_input_mode=mode,
        tune_max_num_tokens=8192,
        use_cuda_graph=True,
        use_cold_l2_cache=True,
    )
    build = Mock(wraps=original)
    monkeypatch.setattr(model, "_make_tuning_config", build)
    first = get_config(backend, model, inputs(mode=mode), mode=mode)
    second = get_config(backend, model, inputs(7, mode=mode), mode=mode)
    assert first == expected
    assert second == expected
    assert build.call_count == 1
    assert first is not second
    assert first.dynamic_tensor_specs[0] is not second.dynamic_tensor_specs[0]
    first.use_cuda_graph = False
    first.dynamic_tensor_specs[0].input_idx = ()
    assert get_config(backend, model, inputs(mode=mode), mode=mode) == expected


def test_tuning_key_separates_experts_modes_limits_and_presence(backend):
    model = runner(backend)
    base = get_config(backend, model, inputs())
    more_experts = get_config(backend, runner(backend, num_experts=8), inputs())
    unpacked = get_config(
        backend,
        model,
        inputs(mode=RoutingInputMode.UnpackedPrecomputed),
        mode=RoutingInputMode.UnpackedPrecomputed,
    )
    larger_limit = get_config(backend, model, inputs(), limit=16384)
    no_expert_weights = inputs()
    no_expert_weights.expert_weights = None
    fewer_inputs = get_config(backend, model, no_expert_weights)
    assert base.tensor_initializers != more_experts.tensor_initializers
    assert base.tensor_initializers != unpacked.tensor_initializers
    assert (
        base.dynamic_tensor_specs[0].gen_tuning_buckets
        != larger_limit.dynamic_tensor_specs[0].gen_tuning_buckets
    )
    assert (
        base.dynamic_tensor_specs[0].input_idx
        != fewer_inputs.dynamic_tensor_specs[0].input_idx
    )
    assert len(backend._BF16_TUNING_CONFIG_CACHE) == 5


def test_tuning_cache_retains_no_request_tensors_or_runner(backend):
    model, call_inputs = runner(backend), inputs()
    refs = [weakref.ref(model)] + [
        weakref.ref(t) for t in call_inputs.to_list() if t is not None
    ]
    get_config(backend, model, call_inputs)
    del model, call_inputs
    gc.collect()
    assert all(ref() is None for ref in refs)


def test_tuning_cache_retries_failed_build(backend, monkeypatch):
    model = runner(backend)
    build = Mock(side_effect=ValueError("bad tuning config"))
    monkeypatch.setattr(model, "_make_tuning_config", build)
    for _ in range(2):
        with pytest.raises(ValueError, match="bad tuning config"):
            get_config(backend, model, inputs())
    assert build.call_count == 2
    assert not backend._BF16_TUNING_CONFIG_CACHE


def test_tuning_cache_is_bounded_and_evicts_least_recently_used(backend, monkeypatch):
    monkeypatch.setattr(backend, "_BF16_TUNING_CONFIG_CACHE_SIZE", 2)
    model = runner(backend)
    build = Mock(wraps=model._make_tuning_config)
    monkeypatch.setattr(model, "_make_tuning_config", build)
    for limit in (1024, 2048, 1024, 4096, 1024):
        get_config(backend, model, inputs(), limit=limit)
    assert build.call_count == 3
    assert len(backend._BF16_TUNING_CONFIG_CACHE) == 2
    get_config(backend, model, inputs(), limit=2048)
    assert build.call_count == 4


def test_empty_placeholders_reuse_only_matching_dtype(backend):
    device = torch.device("cpu")
    first = backend._empty_routing_placeholder(device, torch.int32)
    assert backend._empty_routing_placeholder(device, torch.int32) is first
    other = backend._empty_routing_placeholder(device, torch.bfloat16)
    assert first.numel() == other.numel() == 0
    assert first.dtype == torch.int32
    assert other.dtype == torch.bfloat16
    assert first is not other


def test_empty_placeholders_partition_by_cuda_device(backend, monkeypatch):
    # Exercise device keys without needing two physical GPUs.
    empty = Mock(side_effect=lambda *args, **kwargs: object())
    monkeypatch.setattr(torch, "empty", empty)
    first = backend._empty_routing_placeholder(torch.device("cuda:0"), torch.int32)
    second = backend._empty_routing_placeholder(torch.device("cuda:1"), torch.int32)
    assert first is not second
    assert (
        backend._empty_routing_placeholder(torch.device("cuda:0"), torch.int32) is first
    )
    assert empty.call_count == 2
    assert empty.call_args_list[0].kwargs["device"] == torch.device("cuda:0")
    assert empty.call_args_list[1].kwargs["device"] == torch.device("cuda:1")
