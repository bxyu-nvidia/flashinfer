# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Opt-in integration of experimental GB200 BF16 expert projections."""

import functools

import torch

from .routed_bf16_gemm1_ws import make_bf16_gemm2_plan, make_routed_gemm1_plan

_seen = set()
_miss_seen = set()


@functools.lru_cache(maxsize=1)
def _native_module():
    from ..jit import setup_cubin_loader
    from ..jit.fused_moe import gen_trtllm_gen_fused_moe_sm100_module

    spec = gen_trtllm_gen_fused_moe_sm100_module()
    if not spec.jit_library_path.exists():
        raise RuntimeError("Build the experimental native BF16 stage helper first")
    module = spec.load()
    setup_cubin_loader(str(spec.jit_library_path))
    return module


def run_experimental_bf16_moe(runner, inputs, kwargs, tactic, mode):
    """Return None outside the validated specialization, otherwise finalized output."""
    from .core import (
        allocate_trtllm_moe_canonical_routing,
        canonicalize_trtllm_moe_routing_,
        _torch_view_of_ffi_tensor,
    )
    from .shared.inputs import RoutingInputMode

    output, logits, _, _, hidden, _, lora, _ = inputs
    if mode == "gemm2" and (hidden.shape[0] < 32768 or runner.num_local_experts != 128):
        return None
    if not isinstance(tactic, int):
        tactic = tuple(tactic)
    weights1, weights2 = kwargs["gemm1_weights"], kwargs["gemm2_weights"]
    if not (
        mode in ("gemm1", "gemm2", "both")
        and logits is not None
        and kwargs["routing_input_mode"] == RoutingInputMode.FromLogits
        and hidden.shape[0] >= 4096
        and hidden.shape[1] == 1024
        and runner.intermediate_size == 2688
        and runner.num_experts == 512
        and runner.num_local_experts in (128, 256)
        and runner.top_k == 22
        and int(runner.activation_type) == 6
        and int(runner.weight_layout) == 2
        and runner.use_shuffled_weight
        and weights1.dtype == weights2.dtype == hidden.dtype == torch.bfloat16
        and weights1.shape[-1] == weights2.shape[-1] == 64
        and kwargs["do_finalize"]
        and kwargs["norm_topk_prob"]
        and (kwargs["n_group"] or 0) <= 1
        and kwargs["routing_method_type"] == 2
        and lora is None
        and all(
            kwargs[k] is None
            for k in ("gemm1_alpha", "gemm1_beta", "gemm1_clamp_limit")
        )
        and isinstance(tactic, (tuple, list))
        and int(tactic[0]) == 128
        and torch.cuda.get_device_capability(hidden.device) == (10, 0)
    ):
        identity = (hidden.device.index, hidden.shape[0], str(tactic), mode)
        if hidden.shape[0] >= 4096 and identity not in _miss_seen:
            _miss_seen.add(identity)
            print(
                "EXPERIMENTAL_BF16_FALLBACK",
                {
                    "mode": mode,
                    "tokens": hidden.shape[0],
                    "tactic": str(tactic),
                    "tactic_type": type(tactic).__name__,
                    "routing_mode": kwargs["routing_input_mode"],
                    "norm_topk": kwargs["norm_topk_prob"],
                    "activation": int(runner.activation_type),
                    "layout": int(runner.weight_layout),
                    "routing": kwargs["routing_method_type"],
                    "hidden": hidden.shape[1],
                    "intermediate": runner.intermediate_size,
                },
                flush=True,
            )
        return None

    rows = runner.moe_op.trtllm_get_valid_moe_factorizations(
        runner.dtype_act,
        runner.dtype_weights,
        runner.fp8_quantization_type,
        runner.top_k,
        runner.hidden_size,
        runner.hidden_size_output,
        runner.intermediate_size,
        runner.num_local_experts,
        runner.activation_type,
        runner.use_shuffled_weight,
        runner.weight_layout,
        runner.use_per_token_scaling,
        hidden.shape[0],
        False,
    )
    factors = next(
        (
            tuple(map(int, row[2:4]))
            for row in rows
            if int(row[0]) == 128 and int(row[1]) == int(tactic[1])
        ),
        None,
    )
    if factors is None:
        return None
    identity = (hidden.device.index, hidden.shape[0], mode)
    if identity not in _seen:
        _seen.add(identity)
        print(
            "EXPERIMENTAL_BF16_MOE",
            mode,
            "device",
            hidden.device.index,
            "tokens",
            hidden.shape[0],
            "experts",
            runner.num_local_experts,
            flush=True,
        )
    native = _native_module()
    canonical = allocate_trtllm_moe_canonical_routing(
        logits, top_k=runner.top_k, tile_n=128
    )
    canonicalize_trtllm_moe_routing_(
        canonical,
        logits,
        kwargs["routing_bias"],
        hidden,
        top_k=runner.top_k,
        n_group=kwargs["n_group"],
        topk_group=kwargs["topk_group"],
        local_expert_offset=kwargs["local_expert_offset"],
        local_num_experts=runner.num_local_experts,
        routed_scaling_factor=kwargs["routed_scaling_factor"],
        routing_method_type=kwargs["routing_method_type"],
        use_routing_scales_on_input=False,
        use_deep_seek_fp8=False,
        norm_topk_prob=True,
        enable_pdl=kwargs["enable_pdl"],
    )
    scratch = canonical.scratch
    metadata = (
        scratch[1],
        scratch[2],
        scratch[3],
        canonical.expert_weights,
        scratch[4],
        scratch[0],
        scratch[5],
        scratch[6],
        scratch[7],
    )
    body_inputs = list(inputs)
    body_inputs[1:4] = [None, canonical.packed_router_scratch, canonical.expert_weights]
    body_kwargs = kwargs | {"routing_input_mode": RoutingInputMode.PackedPrecomputed}
    workspace = tuple(
        _torch_view_of_ffi_tensor(t)
        for t in runner.forward(
            body_inputs,
            tactic=tactic,
            do_preparation=True,
            da_routing_metadata=metadata,
            **body_kwargs,
        )
    )
    for stage, (x, weights, y, ws) in enumerate(
        (
            (hidden, weights1, workspace[0], workspace[2]),
            (workspace[0], weights2, workspace[1], workspace[3]),
        ),
        1,
    ):
        use_new = (mode == "both" or mode == f"gemm{stage}") and (
            stage == 1 or (hidden.shape[0] >= 32768 and runner.num_local_experts == 128)
        )
        if use_new:
            plan = make_routed_gemm1_plan if stage == 1 else make_bf16_gemm2_plan
            tuning = dict(block_n=256, block_t=128, stages=4, load_warps=8, block_k=64)
            if stage == 2:
                from .retile_bf16_moe import make_retiled_jobs

                jobs = make_retiled_jobs(
                    scratch[4][: runner.num_experts],
                    kwargs["local_expert_offset"],
                    runner.num_local_experts,
                    hidden.shape[0],
                    workspace[0].shape[0],
                )
                tuning.update(
                    block_t=256, stages=3, load_warps=2, group_t=1, job_metadata=jobs
                )
            plan(
                x,
                weights,
                y,
                scratch[3],
                scratch[5],
                scratch[6],
                scratch[7],
                **tuning,
            )()
        else:
            native.trtllm_moe_run_bf16_gemm_stage(
                x,
                weights,
                y,
                scratch[3],
                scratch[5],
                scratch[6],
                scratch[7],
                scratch[1],
                ws,
                stage,
                hidden.shape[0],
                runner.top_k,
                128,
                factors[stage - 1],
                kwargs["enable_pdl"],
            )
    native.trtllm_moe_run_finalize(
        workspace[1],
        output,
        canonical.expert_weights,
        scratch[2],
        scratch[1],
        hidden.shape[0],
        runner.num_experts,
        runner.top_k,
        runner.hidden_size,
        kwargs["enable_pdl"],
        False,
    )
    replay_out = kwargs["routing_replay_out"]
    if replay_out is not None:
        replay_out.copy_(canonical.routing_replay_ids)
    return [output]
