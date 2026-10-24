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

import logging
from unittest.mock import Mock

import pytest
import torch

import flashinfer.autotuner.autotuner as module
from flashinfer.autotuner import (
    AutoTuner,
    DynamicTensorSpec,
    TunableRunner,
    TuningConfig,
)
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
from flashinfer.prims_ts.moe.runner import PrimsTsBf16MoERunner


class SelectionRunner(TunableRunner):
    _cache_inference_selection = True

    def get_valid_tactics(self, inputs, profile):
        return [1, 2]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        raise AssertionError("Selection must not execute kernels")


@pytest.fixture
def state(monkeypatch, caplog):
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_TIMER", "cuda_event")
    monkeypatch.delenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", raising=False)
    caplog.set_level(logging.WARNING, logger=module.logger.name)
    return AutoTuner(), SelectionRunner(), [torch.empty(16, 8)], TuningConfig()


def choose(state, **kwargs):
    tuner, runner, inputs, config = state
    return tuner.choose_one("test::selection", [runner], config, inputs, **kwargs)[1]


def key(state):
    tuner, runner, inputs, config = state
    return tuner._get_cache_key(
        "test::selection",
        runner,
        tuple(x.size() for x in inputs),
        config,
        runner.get_cache_key_extras(inputs),
    )


def test_fallback_winner_replacement_and_removal(state, monkeypatch):
    tuner = state[0]
    search = Mock(wraps=tuner.search_cache)
    monkeypatch.setattr(tuner, "search_cache", search)
    assert choose(state) == choose(state) == -1
    assert search.call_count == 1
    tuner.profiling_cache[key(state)] = ((2, 3), None)
    assert choose(state) == choose(state) == (2, 3)
    assert search.call_count == 2
    tuner.profiling_cache[key(state)] = ((4, 5), None)
    assert choose(state) == (4, 5)
    del tuner.profiling_cache[key(state)]
    assert choose(state) == -1
    assert search.call_count == 4


def test_load_file_and_clear_invalidate_fallback(state, tmp_path):
    tuner, runner, _, _ = state
    assert choose(state) == -1
    import json

    path = tmp_path / "tactics.json"
    path.write_text(json.dumps({key(state).file_key: [type(runner).__name__, [2, 3]]}))
    tuner.load_configs(str(path))
    assert choose(state) == (2, 3)
    tuner.clear_cache()
    assert not tuner._inference_selections
    assert choose(state) == -1


def test_debug_key_is_not_built_when_disabled(state, monkeypatch, caplog):
    # A runner without selection caching still avoids the duplicate debug key.
    monkeypatch.setattr(SelectionRunner, "_cache_inference_selection", False)
    build = Mock(wraps=AutoTuner._get_cache_key)
    monkeypatch.setattr(AutoTuner, "_get_cache_key", build)
    assert choose(state) == -1
    assert build.call_count == 1
    caplog.set_level(logging.DEBUG, logger=module.logger.name)
    assert choose(state) == -1
    assert build.call_count == 3


def test_shape_config_and_runner_changes_do_not_alias(state, monkeypatch):
    tuner, runner, inputs, _ = state
    tuner.profiling_cache[key(state)] = (7, None)
    assert choose(state) == 7
    assert choose((tuner, runner, [torch.empty(32, 8)], TuningConfig())) == -1
    runner.variant = 1
    assert choose(state) == -1
    del runner.variant
    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(16, 32, 64),
                map_to_tuning_buckets=lambda n: 32,
            ),
        )
    )
    assert choose((tuner, runner, inputs, config)) == -1
    assert choose(state) == 7


def test_bucket_overrides_and_skip_take_effect(state):
    tuner = state[0]
    assert choose(state) == -1
    config = TuningConfig(
        dynamic_tensor_specs=(
            DynamicTensorSpec(
                input_idx=(0,),
                dim_idx=(0,),
                gen_tuning_buckets=(16, 32, 64),
                map_to_tuning_buckets=lambda n: n,
            ),
        )
    )
    configured = (*state[:3], config)
    tuner.profiling_cache[key(configured)] = (3, None)
    assert choose(configured) == 3
    tuner._get_override_stack().append(((32, 64), True, None))
    try:
        assert choose(configured) == -1
    finally:
        tuner._get_override_stack().pop()
    tuner._get_skip_ops_stack().append(frozenset({"test::selection"}))
    try:
        assert choose(configured) == -1
    finally:
        tuner._get_skip_ops_stack().pop()
    assert choose(configured) == 3


def test_tuning_never_reuses_serving_selection(state, monkeypatch):
    tuner = state[0]
    assert choose(state) == -1
    tuner.is_tuning_mode = True
    generate = Mock(side_effect=RuntimeError("entered tuning"))
    monkeypatch.setattr(tuner, "_generate_optimization_profiles", generate)
    with pytest.raises(RuntimeError, match="entered tuning"):
        choose(state)
    assert not tuner._inference_selections


@pytest.mark.parametrize("source", ["managed", "legacy", "validator", "plain_dict"])
def test_dynamic_sources_bypass_memo(state, monkeypatch, source):
    tuner, runner, _, _ = state
    assert choose(state) == -1
    if source == "managed":
        tuner._managed_cache = object()
    elif source == "legacy":
        monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    elif source == "validator":
        monkeypatch.setattr(
            SelectionRunner, "validate_tactic", lambda *a: True, raising=False
        )
    else:
        tuner.profiling_cache = dict(tuner.profiling_cache)
    search = Mock(side_effect=[(True, 0, 2, None), (True, 0, 3, None)])
    monkeypatch.setattr(tuner, "search_cache", search)
    assert choose(state) == 2
    assert choose(state) == 3


def test_selection_cache_is_bounded_and_does_not_retain_inputs(state):
    import gc
    import weakref

    tuner, runner, _, config = state
    for count in range(1, 1030):
        inputs = [torch.empty(count, 1)]
        assert choose((tuner, runner, inputs, config)) == -1
    ref = weakref.ref(inputs[0])
    del inputs
    gc.collect()
    assert ref() is None
    assert len(tuner._inference_selections) == 1024


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d.__setitem__("a", 2),
        lambda d: d.__delitem__("a"),
        lambda d: d.update(a=2),
        lambda d: d.setdefault("b", 2),
        lambda d: d.pop("a"),
        lambda d: d.popitem(),
        lambda d: d.clear(),
        lambda d: d.__ior__({"a": 2}),
    ],
)
def test_all_config_mutations_advance_generation(mutation):
    mapping = module._VersionedConfigDict()
    mapping["a"] = 1
    old = mapping.version
    mutation(mapping)
    assert mapping.version > old


def test_prims_runner_hash_preserved_and_invalidated(monkeypatch):
    runner = PrimsTsBf16MoERunner(
        object(),
        top_k=2,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=256,
    )
    reference = TunableRunner.__hash__
    compute = Mock(wraps=reference)
    monkeypatch.setattr(TunableRunner, "__hash__", lambda self: compute(self))
    assert hash(runner) == hash(runner) == reference(runner)
    assert compute.call_count == 1
    runner.top_k = 4
    assert hash(runner) == reference(runner)
    assert compute.call_count == 2
    runner.set_cache_key_static_extras(enable_pdl=True)
    assert hash(runner) == reference(runner)
    runner.extra = 3
    assert hash(runner) == reference(runner)
    del runner.extra
    assert hash(runner) == reference(runner)
    before = compute.call_count
    runner.example_cache = {}
    assert hash(runner) == reference(runner)
    assert compute.call_count == before


def test_prims_extras_reused_and_separate_optional_inputs():
    runner = PrimsTsBf16MoERunner(
        object(),
        top_k=2,
        num_local_experts=4,
        hidden_size=128,
        intermediate_size=256,
    )
    inputs = [None] * len(MoeRunnerInputs._FIELDS)
    first = runner.get_cache_key_extras(inputs)
    assert runner.get_cache_key_extras(inputs) is first
    assert dict(first)["activation_type"] == int(runner.activation_type)
    for name in (
        "per_token_scale",
        "gemm1_lora_delta",
        "routing_logits",
        "expert_weights",
    ):
        inputs[MoeRunnerInputs.idx(name)] = torch.empty(0)
        assert dict(runner.get_cache_key_extras(inputs))[name] is True
    runner.set_cache_key_static_extras(enable_pdl=True)
    assert dict(runner.get_cache_key_extras(inputs))["enable_pdl"] is True


def test_measurement_partition_changes_invalidate_selection(state):
    from types import SimpleNamespace

    tuner = state[0]
    tuner.profiling_cache[key(state)] = (3, None)
    assert choose(state) == 3
    tuner._v2_local.measure = SimpleNamespace(
        use_cuda_graph=None,
        cold_l2=None,
        manifest_fields=lambda: {"timer": "cuda_event"},
    )
    assert choose(state) == -1
    tuner._winner_cache()[key(state)] = (5, None)
    assert choose(state) == 5
    tuner._v2_local.measure = None
    assert choose(state) == 3


def test_selection_reuses_equivalent_new_prims_runners(state, monkeypatch):
    tuner, _, _, config = state
    moe_op = object()
    inputs = [None] * len(MoeRunnerInputs._FIELDS)
    inputs[MoeRunnerInputs.idx("hidden_states")] = torch.empty(16, 128)
    search = Mock(wraps=tuner.search_cache)
    monkeypatch.setattr(tuner, "search_cache", search)
    for _ in range(3):
        runner = PrimsTsBf16MoERunner(
            moe_op,
            top_k=2,
            num_local_experts=4,
            hidden_size=128,
            intermediate_size=256,
        )
        runner.set_cache_key_static_extras(enable_pdl=True)
        assert choose((tuner, runner, inputs, config)) == -1
    assert search.call_count == 1


def test_runner_metadata_cache_keeps_runners_independent():
    from flashinfer.prims_ts.moe.runner import _bf16_runner_metadata

    _bf16_runner_metadata.cache_clear()
    args = dict(top_k=2, num_local_experts=4, hidden_size=128, intermediate_size=256)
    first = PrimsTsBf16MoERunner(object(), **args)
    second = PrimsTsBf16MoERunner(object(), **args)
    assert _bf16_runner_metadata.cache_info().misses == 1
    assert _bf16_runner_metadata.cache_info().hits == 1
    first.top_k = 4
    first.set_cache_key_static_extras(enable_pdl=True)
    assert second.top_k == 2
    assert second._cache_key_static_extras == ()
    third = PrimsTsBf16MoERunner(object(), **(args | {"top_k": 3}))
    assert third.top_k == 3
    assert _bf16_runner_metadata.cache_info().misses == 2
    assert _bf16_runner_metadata.cache_info().maxsize == 256
    hash(first)
    first.__init__(second.moe_op, **args)
    assert hash(first) == TunableRunner.__hash__(first)
