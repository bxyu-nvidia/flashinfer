# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in BF16 latent-expert prefill using persistent Blackwell projections."""

import torch

from .retile_bf16_moe import make_retiled_jobs
from .routed_bf16_gemm1_ws import make_bf16_gemm2_plan, make_routed_gemm1_plan
from .shared.inputs import MoeRunnerInputs, RoutingInputMode

_hits = set()
_MAX_CHUNK_TOKENS = 69632


def run_nemotron_prefill(runner, inputs, kwargs):
    """Return None for unsupported calls, preserving the ordinary runner."""
    data = MoeRunnerInputs.from_list(inputs)
    x = data.hidden_states
    w1, w2 = kwargs["gemm1_weights"], kwargs["gemm2_weights"]
    mode = kwargs.get("routing_input_mode", RoutingInputMode.FromLogits)
    precomputed = (
        mode == RoutingInputMode.UnpackedPrecomputed
        and data.topk_ids is not None
        and data.expert_weights is not None
        and data.topk_ids.dtype == torch.int32
        and data.expert_weights.dtype == torch.bfloat16
        and tuple(data.topk_ids.shape) == (x.shape[0], 22)
        and tuple(data.expert_weights.shape) == (x.shape[0], 22)
        and data.topk_ids.is_contiguous()
        and data.expert_weights.is_contiguous()
    )
    if not (
        x.shape[0] >= 4096
        and x.shape[1] == 1024
        and runner.hidden_size_output == 1024
        and runner.intermediate_size == 2688
        and runner.num_experts == 512
        and runner.num_local_experts == 128
        and runner.top_k == 22
        and int(runner.activation_type) == 6
        and int(runner.weight_layout) == 2
        and runner.use_shuffled_weight
        and (
            precomputed
            or (
                mode == RoutingInputMode.FromLogits
                and kwargs["routing_method_type"] == 2
            )
        )
        and x.is_contiguous()
        and kwargs.get("norm_topk_prob", True)
        and (kwargs["n_group"] or 0) <= 1
        and data.gemm1_lora_delta is None
        and w1.dtype == w2.dtype == x.dtype == torch.bfloat16
        and tuple(w1.shape) == (128, 16, 2688, 64)
        and tuple(w2.shape) == (128, 42, 1024, 64)
        and w1.is_contiguous()
        and w2.is_contiguous()
        and all(
            kwargs.get(k) is None
            for k in (
                "gemm1_bias",
                "gemm2_bias",
                "gemm1_alpha",
                "gemm1_beta",
                "gemm1_clamp_limit",
            )
        )
        and torch.cuda.get_device_capability(x.device) == (10, 0)
    ):
        return None
    if kwargs["do_finalize"] and x.shape[0] > _MAX_CHUNK_TOKENS:
        # Bound expert staging storage without changing per-token arithmetic.
        chunks = (x.shape[0] + _MAX_CHUNK_TOKENS - 1) // _MAX_CHUNK_TOKENS
        chunk_size = (x.shape[0] + chunks - 1) // chunks
        for start in range(0, x.shape[0], chunk_size):
            end = min(start + chunk_size, x.shape[0])
            chunk_inputs = data.to_list()
            for name in (
                "output",
                "routing_logits",
                "topk_ids",
                "expert_weights",
                "hidden_states",
            ):
                tensor = getattr(data, name)
                if tensor is not None and tensor.numel():
                    chunk_inputs[MoeRunnerInputs.idx(name)] = tensor[start:end]
            chunk_kwargs = kwargs
            if kwargs.get("routing_replay_out") is not None:
                chunk_kwargs = kwargs | {
                    "routing_replay_out": kwargs["routing_replay_out"][start:end]
                }
            assert run_nemotron_prefill(runner, chunk_inputs, chunk_kwargs) is not None
        _hits.add((x.device.index, x.shape[0]))
        return [data.output]
    from .core import get_trtllm_moe_sm100_module, _torch_view_of_ffi_tensor

    native = get_trtllm_moe_sm100_module()
    empty = torch.empty(0, device=x.device, dtype=torch.int32)
    buffers = native.trtllm_moe_run_routing(
        data.routing_logits,
        kwargs["routing_bias"],
        data.topk_ids if precomputed else empty,
        data.expert_weights if precomputed else empty,
        x,
        w1,
        w2,
        runner.num_experts,
        runner.top_k,
        kwargs["n_group"],
        kwargs["topk_group"],
        runner.intermediate_size,
        kwargs["local_expert_offset"],
        runner.num_local_experts,
        kwargs["routed_scaling_factor"],
        kwargs["routing_method_type"],
        True,
        int(runner.weight_layout),
        kwargs["enable_pdl"],
        [128, -1],
        int(runner.activation_type),
        True,
        kwargs.get("routing_replay_out"),
    )
    buffers = list(buffers)
    if precomputed:
        buffers[0] = data.expert_weights
    weights, permutation, route, experts, limits, tiles, total, y1, y2 = (
        _torch_view_of_ffi_tensor(t) for t in buffers
    )
    make_routed_gemm1_plan(
        x,
        w1,
        y1,
        route,
        experts,
        limits,
        tiles,
        block_n=256,
        block_t=128,
        stages=4,
        load_warps=8,
        block_k=64,
        store_t=64,
        epilogue="16x256b",
    )()
    jobs = make_retiled_jobs(
        torch.empty(runner.num_experts, device=x.device, dtype=torch.int32),
        kwargs["local_expert_offset"],
        runner.num_local_experts,
        # Reuse one metadata specialization across full chunks and short tails.
        max(131072, x.shape[0]),
        y1.shape[0],
        native_tiles=(experts, limits, tiles),
    )
    make_bf16_gemm2_plan(
        y1,
        w2,
        y2,
        route,
        experts,
        limits,
        tiles,
        block_n=256,
        block_t=256,
        stages=3,
        load_warps=2,
        group_t=1,
        block_k=64,
        job_metadata=jobs,
        store_t=64,
        epilogue="16x256b",
        store_buffers=2,
    )()
    _hits.add((x.device.index, x.shape[0]))
    if kwargs["do_finalize"]:
        native.trtllm_moe_run_finalize(
            y2,
            data.output,
            weights,
            permutation,
            total,
            x.shape[0],
            runner.num_experts,
            runner.top_k,
            runner.hidden_size,
            kwargs["enable_pdl"],
            False,
        )
        return [data.output]
    return [y2, weights, permutation]
