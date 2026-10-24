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

from contextlib import nullcontext
import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    utils = importlib.import_module("flashinfer.prims_ts.utils")
    support = importlib.import_module("flashinfer.prims_ts.moe.support")
    mapper = importlib.import_module("flashinfer.prims_ts.moe.config_mapper")
    runner = importlib.import_module("flashinfer.prims_ts.moe.runner")
    backend = importlib.import_module("flashinfer.fused_moe.backends.prims_ts.bf16_op")
    monkeypatch.setattr(utils, "_PRIMS_TS_AVAILABLE", False)
    monkeypatch.setattr(support, "_SUPPORTED_PRIMS_TS_DEVICES", set())
    return SimpleNamespace(
        utils=utils, support=support, mapper=mapper, runner=runner, backend=backend
    )


def test_dependency_success_cached_but_missing_dependencies_retried(
    modules, monkeypatch
):
    find = Mock(side_effect=[None, object(), object(), object(), object()])
    monkeypatch.setattr(modules.utils.importlib.util, "find_spec", find)
    assert not modules.utils.is_prims_ts_available()
    assert modules.utils.is_prims_ts_available()
    assert modules.utils.is_prims_ts_available()
    assert find.call_count == 5


def test_dependency_import_failure_retried(modules, monkeypatch):
    find = Mock(side_effect=ModuleNotFoundError("cutlass"))
    monkeypatch.setattr(modules.utils.importlib.util, "find_spec", find)
    assert not modules.utils.is_prims_ts_available()
    assert not modules.utils.is_prims_ts_available()
    assert find.call_count == 2


def test_device_success_is_per_device_and_failures_retry(modules, monkeypatch):
    available = Mock(return_value=True)
    capability = Mock(side_effect=[(10, 0), (9, 0), (10, 3)])
    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(modules.support, "get_compute_capability", capability)
    check = modules.support._device_supports_prims_ts
    assert check(torch.device("cuda:0"))
    assert check(torch.device("cuda:0"))
    assert not check(torch.device("cuda:1"))
    assert check(torch.device("cuda:1"))
    assert check(torch.device("cuda:1"))
    assert not check(torch.device("cpu"))
    assert capability.call_count == available.call_count == 3


def test_unindexed_device_tracks_current_device(modules, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    current = Mock(side_effect=[0, 1])
    capability = Mock(side_effect=[(10, 0), (9, 0)])
    monkeypatch.setattr(torch.cuda, "current_device", current)
    monkeypatch.setattr(modules.support, "get_compute_capability", capability)
    assert modules.support._device_supports_prims_ts(torch.device("cuda"))
    assert not modules.support._device_supports_prims_ts(torch.device("cuda"))
    assert capability.call_args_list[1].args == (torch.device("cuda:1"),)


@pytest.fixture
def invocation(modules, monkeypatch):
    from flashinfer.tllm_enums import ActivationType, RoutingMethodType, WeightLayout

    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda *args: nullcontext())
    monkeypatch.setattr(modules.support, "is_prims_ts_available", lambda: True)
    monkeypatch.setattr(
        modules.support, "_device_supports_prims_ts", lambda device: True
    )
    monkeypatch.setattr(modules.backend, "_get_moe_op", lambda: object())
    cls = modules.runner.PrimsTsBf16MoERunner
    monkeypatch.setattr(cls, "set_cache_key_static_extras", lambda self, **kwargs: None)
    execute = Mock(return_value=[])
    monkeypatch.setattr(cls, "_forward_prepared", execute)

    def pair(block_bytes, index):
        cfg = SimpleNamespace(block_major_k_bytes=block_bytes)
        fc = SimpleNamespace(cfg=cfg)
        return SimpleNamespace(tile_n=128, moe_config_index=index, fc1=fc, fc2=fc)

    default_pair, selected_pair = pair(128, 0), pair(128, 1)
    mapping = Mock(
        side_effect=lambda tactic, **kwargs: default_pair
        if tuple(tactic) == (-1, -1)
        else selected_pair
    )
    monkeypatch.setattr(modules.mapper, "_resolve_bf16_moe_tactic", mapping)
    check = Mock(wraps=modules.support._check_prims_ts_bf16_call)
    monkeypatch.setattr(modules.backend, "_check_prims_ts_bf16_call", check)
    monkeypatch.setattr(modules.runner, "_check_prims_ts_bf16_call", check)
    args = dict(
        routing_logits=torch.empty(2, 4),
        routing_bias=None,
        topk_ids=None,
        expert_weights=None,
        hidden_states=torch.empty(2, 128, dtype=torch.bfloat16),
        gemm1_weights=torch.empty(2, 1, 128, 64, dtype=torch.bfloat16),
        gemm2_weights=torch.empty(2, 1, 128, 64, dtype=torch.bfloat16),
        gemm1_lora_delta=None,
        num_experts=4,
        top_k=2,
        n_group=1,
        topk_group=1,
        intermediate_size=128,
        local_expert_offset=0,
        local_num_experts=2,
        routed_scaling_factor=1.0,
        routing_method_type=int(RoutingMethodType.DeepSeekV3),
        use_shuffled_weight=True,
        weight_layout=int(WeightLayout.BlockMajorK),
        activation_type=int(ActivationType.Relu2),
        enable_pdl=False,
    )
    return SimpleNamespace(
        args=args,
        execute=execute,
        check=check,
        mapping=mapping,
        default_pair=default_pair,
        selected_pair=selected_pair,
    )


@pytest.mark.parametrize("tactic", [-1, [-1, -1], [128, 1]])
def test_wrapper_checks_call_once_and_validates_selected_tactic(
    modules, invocation, monkeypatch, tactic
):
    tuner = SimpleNamespace(choose_one=lambda *a, **kw: (None, tactic))
    monkeypatch.setattr(modules.backend.AutoTuner, "get", lambda: tuner)
    output = modules.backend.prims_ts_bf16_moe_op(**invocation.args)
    assert output[0].shape == (2, 128)
    assert invocation.check.call_count == 1
    is_default = tactic == -1 or tactic == [-1, -1]
    assert invocation.mapping.call_count == (1 if is_default else 2)
    assert invocation.execute.call_args.args[1] is (
        invocation.default_pair if is_default else invocation.selected_pair
    )


def test_wrapper_rejects_selected_tactic_with_incompatible_storage(
    modules, invocation, monkeypatch
):
    tuner = SimpleNamespace(choose_one=lambda *a, **kw: (None, [128, 1]))
    monkeypatch.setattr(modules.backend.AutoTuner, "get", lambda: tuner)
    invocation.selected_pair.fc1.cfg.block_major_k_bytes = 64
    with pytest.raises(RuntimeError, match="expects 64 bytes"):
        modules.backend.prims_ts_bf16_moe_op(**invocation.args)
    invocation.execute.assert_not_called()


def test_direct_forward_revalidates_changed_inputs_and_configuration(
    modules, invocation
):
    from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs
    from flashinfer.tllm_enums import ActivationType

    model = modules.runner.PrimsTsBf16MoERunner(
        object(),
        top_k=2,
        num_local_experts=2,
        hidden_size=128,
        intermediate_size=128,
        activation_type=ActivationType.Relu2,
        weight_layout=invocation.args["weight_layout"],
    )
    args = dict(invocation.args)
    call_inputs = MoeRunnerInputs(
        output=torch.empty(2, 128),
        routing_logits=args["routing_logits"],
        topk_ids=None,
        expert_weights=None,
        hidden_states=args["hidden_states"],
        hidden_states_scale=None,
        gemm1_lora_delta=None,
        per_token_scale=None,
    )
    model.forward(call_inputs.to_list(), **args)
    assert invocation.check.call_count == 1
    assert invocation.execute.call_count == 1
    call_inputs.per_token_scale = torch.ones(2)
    with pytest.raises(RuntimeError, match="per-token output scaling"):
        model.forward(call_inputs.to_list(), **args)
    call_inputs.per_token_scale = None
    args["gemm1_weights"] = torch.empty(2, 1, 128, 32, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="expects 128 bytes"):
        model.forward(call_inputs.to_list(), **args)
    assert invocation.check.call_count == 3
    assert invocation.execute.call_count == 1
