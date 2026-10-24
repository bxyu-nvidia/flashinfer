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

"""Native deferred-routing lifetime and stream regression tests (SM100)."""

import gc

import pytest
import torch

from flashinfer.tllm_enums import ActivationType, RoutingMethodType, WeightLayout


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("precomputed", [False, True])
def test_prepared_routing_owns_inputs_and_uses_preparation_stream(precomputed):
    from flashinfer.fused_moe.core import get_trtllm_moe_sm100_module
    from flashinfer.utils import is_sm100a_supported

    if not is_sm100a_supported(torch.device("cuda")):
        pytest.skip("Requires SM100 routing kernels")
    module = get_trtllm_moe_sm100_module().moe_op
    assert hasattr(module, "trtllm_moe_prepare_routing"), (
        "Rebuild the TRTLLM MoE binary with the prepared-routing API"
    )
    stream = torch.cuda.Stream()
    other_stream = torch.cuda.Stream()
    tokens, experts, top_k, hidden, intermediate = 32, 8, 2, 1024, 512
    with torch.cuda.stream(stream):
        logits = torch.randn(tokens, experts, device="cuda", dtype=torch.bfloat16)
        if precomputed:
            weights, indices = torch.topk(logits.float().softmax(-1), top_k, dim=-1)
            weights = (weights / weights.sum(-1, keepdim=True)).contiguous()
            indices = indices.to(torch.int32)
        else:
            weights = torch.empty(0, device="cuda", dtype=torch.bfloat16)
            indices = torch.empty(0, device="cuda", dtype=torch.int32)
        args = [
            None if precomputed else logits,
            None,
            indices,
            weights,
            torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16),
            torch.empty(
                experts, intermediate, hidden, device="cuda", dtype=torch.bfloat16
            ),
            torch.empty(
                experts, hidden, intermediate, device="cuda", dtype=torch.bfloat16
            ),
            experts,
            top_k,
            None,
            None,
            intermediate,
            0,
            experts,
            None,
            int(RoutingMethodType.Renormalize),
            True,
            int(WeightLayout.MajorK),
            False,
            [128, 0],
            int(ActivationType.Identity),
            True,
            None,
        ]
        expected = module.trtllm_moe_run_routing(*args)
        buffers, submit = module.trtllm_moe_prepare_routing(*args)
        # Seed a returned counter: prepare exposes buffers without starting
        # routing, and submission must write through those very same buffers.
        counter = buffers[6]
        counter.fill_(-1)
        pointer = counter.data_ptr()
        anchor = torch.empty(0, device="cuda")
    stream.synchronize()
    assert counter.item() == -1
    # The callback must retain input storage and TensorView shape metadata.
    del args, logits, indices, weights
    gc.collect()
    with (
        torch.cuda.stream(other_stream),
        pytest.raises(RuntimeError, match="preparation stream"),
    ):
        submit(anchor)
    assert counter.item() == -1
    with torch.cuda.stream(stream):
        submit(anchor)
    stream.synchronize()
    assert counter.data_ptr() == pointer
    torch.testing.assert_close(counter, expected[6])
    torch.testing.assert_close(buffers[5], expected[5])
    if not precomputed:
        torch.testing.assert_close(buffers[0], expected[0])
    # Permutation order can vary; verify the semantic token mapping instead.
    with torch.cuda.stream(stream):
        expanded_to_permuted = buffers[1].long()
        assert torch.all(expanded_to_permuted >= 0)
        actual_tokens = buffers[2][expanded_to_permuted]
        expected_tokens = torch.arange(tokens, device="cuda").repeat_interleave(top_k)
        torch.testing.assert_close(actual_tokens.long(), expected_tokens)
