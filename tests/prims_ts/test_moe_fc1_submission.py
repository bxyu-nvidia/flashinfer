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
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import tvm_ffi

from flashinfer.prims_ts.moe import runner as module
from flashinfer.prims_ts.moe import compile_cache, config_mapper
from flashinfer.tllm_enums import ActivationType, WeightLayout


@pytest.mark.parametrize("warm", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_fc1_submission_overlaps_setup_and_preserves_cold_path(
    monkeypatch, warm, finalize
):
    import cutlass.cute.runtime

    # Run the real orchestration and launch adapter with CPU tensors, replacing
    # only CUDA stream/launch operations and native routing/finalize.
    driver = ModuleType("cuda.bindings.driver")
    driver.CUstream = lambda stream: stream
    bindings = ModuleType("cuda.bindings")
    bindings.driver = driver
    cuda = ModuleType("cuda")
    cuda.bindings = bindings
    for name, obj in (
        ("cuda", cuda),
        ("cuda.bindings", bindings),
        ("cuda.bindings.driver", driver),
    ):
        monkeypatch.setitem(sys.modules, name, obj)
    monkeypatch.setattr(
        torch.cuda, "current_stream", lambda **kw: SimpleNamespace(cuda_stream=0)
    )
    monkeypatch.setattr(torch.cuda.nvtx, "range", lambda name: nullcontext())
    monkeypatch.setattr(
        cutlass.cute.runtime, "make_ptr", lambda dtype, address, space, **kw: address
    )
    events = []
    hidden = torch.ones(2, 128, dtype=torch.bfloat16)
    output = torch.empty_like(hidden)
    fc1_output, fc2_output = torch.zeros_like(hidden), torch.zeros_like(hidden)
    expert_weights = torch.ones(2, dtype=torch.float32)
    expanded = torch.arange(2, dtype=torch.int32)
    routing_values = [
        expert_weights,
        expanded,
        torch.arange(2, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.zeros(4, dtype=torch.int32),
        torch.ones(1, dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        fc1_output,
        fc2_output,
    ]
    routing_array = tvm_ffi.Array(routing_values)
    original_getitem = tvm_ffi.Array.__getitem__

    def getitem(self, index):
        events.append(("extract", index))
        return original_getitem(self, index)

    monkeypatch.setattr(tvm_ffi.Array, "__getitem__", getitem)

    original_config = compile_cache.get_bf16_runtime_config

    def runtime_config(*args):
        events.append("runtime_config")
        return original_config(*args)

    monkeypatch.setattr(module, "get_bf16_runtime_config", runtime_config)
    monkeypatch.setattr(compile_cache, "get_bf16_runtime_config", runtime_config)

    def launch_fc1(io):
        events.append("fc1")
        assert io["c0_dp"] == fc1_output.data_ptr()
        assert io["act_dp"] == hidden.data_ptr()
        fc1_output.copy_(hidden + 1)

    def launch_fc2(io):
        events.append("fc2")
        assert io["c0_dp"] == fc2_output.data_ptr()
        assert io["act_dp"] == fc1_output.data_ptr()
        fc2_output.copy_(fc1_output * 2)

    def find(*args):
        events.append("find")
        return launch_fc1 if warm else None

    def compile_kernel(key, fc, io, stream):
        events.append("compile_" + fc)
        return launch_fc1 if fc == "fc1" else launch_fc2

    def route(*args):
        events.append("routing")
        return routing_array

    def finish(gemm2, result, weights, indices, *args):
        events.append("finalize")
        assert weights.data_ptr() == expert_weights.data_ptr()
        assert indices.data_ptr() == expanded.data_ptr()
        result.copy_(gemm2)

    monkeypatch.setattr(module, "find_compiled_gemm", find)
    monkeypatch.setattr(module, "get_compiled_gemm", compile_kernel)
    monkeypatch.setattr(module, "_routed_token_capacity", lambda *args: 2)
    model = SimpleNamespace(
        hidden_size=128,
        intermediate_size=128,
        top_k=1,
        num_local_experts=2,
        activation_type=ActivationType.Identity,
        moe_op=SimpleNamespace(
            trtllm_moe_run_routing=route, trtllm_moe_run_finalize=finish
        ),
        _launch_args=lambda io, stream: (io,),
    )
    inputs = SimpleNamespace(
        hidden_states=hidden,
        output=output,
        routing_logits=None,
        topk_ids=None,
        expert_weights=None,
    )
    spec = config_mapper._immutable_config_spec_cached(
        tuple(
            sorted(
                dict(
                    dtype_a=3,
                    dtype_b=3,
                    dtype_c=3,
                    mma_k=16,
                    tile_k=64,
                    weight_layout=int(WeightLayout.MajorK),
                ).items()
            )
        )
    )
    fc = config_mapper.PrimsTsMoeGemmConfig(spec, 0)
    pair = config_mapper.PrimsTsGemmPair(128, 0, fc, fc)
    result = module.PrimsTsBf16MoERunner._forward_prepared(
        model,
        inputs,
        pair,
        routing_bias=None,
        gemm1_weights=torch.empty(2, 128, 128, dtype=torch.bfloat16),
        gemm2_weights=torch.empty(2, 128, 128, dtype=torch.bfloat16),
        num_experts=2,
        n_group=1,
        topk_group=1,
        local_expert_offset=0,
        routed_scaling_factor=1.0,
        routing_method_type=0,
        use_shuffled_weight=True,
        weight_layout=int(WeightLayout.MajorK),
        enable_pdl=False,
        do_finalize=finalize,
    )
    # Submit routing before independent host preparation, allowing GPU routing
    # to run while the runtime config and warm executable are looked up.
    assert events[:3] == ["routing", "runtime_config", "find"]
    split = events.index("fc1")
    assert [e for e in events[:split] if isinstance(e, tuple)] == [
        ("extract", slice(2, 8))
    ]
    assert [e for e in events[split:] if isinstance(e, tuple)] == [
        ("extract", slice(None, 2)),
        ("extract", slice(8, 9)),
    ]
    assert events.index("fc2") > split
    assert events.count("runtime_config") == 2
    assert ("compile_fc1" in events) == (not warm)
    if finalize:
        assert result == []
        torch.testing.assert_close(output, torch.full_like(output, 4))
    else:
        assert len(result) == 3
        torch.testing.assert_close(result[0], torch.full_like(fc2_output, 4))
        assert result[1].data_ptr() == expert_weights.data_ptr()
        assert result[2].data_ptr() == expanded.data_ptr()
