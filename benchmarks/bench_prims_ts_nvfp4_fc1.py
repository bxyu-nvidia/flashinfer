# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Compare Nemotron Prims FC1, including NVFP4 scale preparation/normalization.

Run with --tokens 16960 33920. For NCU, use --profile-from-start off and pass
--profile to this script; the capture contains all three NVFP4 kernels.
Routing is balanced synthetic top-22 routing over 512 experts, with a shared
random token permutation by default. This measures GPU
FC1 time, not serving latency or the complete MoE/model.
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import cuda.bindings.driver as driver
import cutlass
import torch
from cutlass.cute.runtime import make_ptr
from flashinfer.prims_ts.batched_gemm import batched_gemm_run as gemm
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config
from flashinfer.prims_ts.moe.config_mapper import (
    map_trtllm_bf16_moe_tactic,
    map_trtllm_nvfp4_moe_tactic,
)
from flashinfer.prims_ts.moe.nvfp4_output import normalize_nvfp4_output_scales
from flashinfer.prims_ts.moe.routed_scales import preroute_nvfp4_activation_scales
from flashinfer.testing import bench_gpu_time_with_cupti
from flashinfer.tllm_enums import ActivationType


def _prepare_scale_rows(
    scales, *, row_source_indices, logical_cols, layout, num_experts, sf_vec_size
):
    """Bound setup memory while avoiding scalar GPU copies in tensor preparation."""
    rows = row_source_indices.numel()
    columns = (logical_cols + sf_vec_size - 1) // sf_vec_size
    col = torch.arange(columns, device=scales.device)[None, :]

    def offset(row):
        if layout == 2:
            return row * columns + col
        if layout == 1:
            return (row // 8 * (columns // 4) + col // 4) * 32 + row % 8 * 4 + col % 4
        return (
            (row // 128 * (columns // 4) + col // 4) * 512
            + (row % 32 * 4 + row % 128 // 32) * 4
            + col % 4
        )

    result = torch.empty_like(scales)
    for start in range(0, rows * num_experts, 8192):
        dest = torch.arange(
            start, min(start + 8192, rows * num_experts), device=scales.device
        )[:, None]
        source = (
            dest // rows * rows + row_source_indices[(dest % rows).flatten()][:, None]
        )
        result[offset(dest)] = scales[offset(source)]
    return result


def benchmark(tokens, args):
    torch.manual_seed(71)
    stream = driver.CUstream(torch.cuda.current_stream().cuda_stream)
    shape = dict(
        num_experts=512, num_tokens=tokens, top_k=22, problem_n=2688, problem_k=1024
    )
    options = dict(
        activation_type=int(ActivationType.Relu2),
        num_tokens=tokens,
        top_k=22,
        num_local_experts=512,
    )
    bf16 = map_trtllm_bf16_moe_tactic([128, 0], **options).fc1.cfg.kwargs
    nvfp4 = map_trtllm_nvfp4_moe_tactic(
        [128, 0], use_per_token_sf_b=True, **options
    ).fc1.cfg.kwargs
    if not nvfp4.get("prerouted_sfb_ldgsts"):
        raise ValueError("This comparison requires the selected large-prefill config")
    nvfp4 = dict(
        nvfp4,
        static_persistent_clusters=(
            torch.cuda.get_device_properties(0).multi_processor_count
            // nvfp4["cluster_m"]
        ),
    )
    with patch.object(gemm, "_permute_mma_sf_rows", _prepare_scale_rows):
        bf_io = gemm.prepare_tensors(**shape, **bf16)
        nv_io = gemm.prepare_tensors(**shape, **nvfp4)
    if args.routing == "shuffled":
        permutation = torch.randperm(tokens, device="cuda", dtype=torch.int32)
        for io in (bf_io, nv_io):
            routes = io["_keepalive"][8]
            routes.copy_(permutation[routes.long()])

    capacity = nv_io["total_padded_tokens"]
    packed = torch.empty((capacity, 1344), device="cuda", dtype=torch.uint8)
    raw = torch.empty((capacity, 168), device="cuda", dtype=torch.float32)
    scale_bytes = torch.full((capacity * 168,), 0x55, device="cuda", dtype=torch.uint8)
    input_token_sf = torch.ones(tokens, device="cuda")
    expert_sf = torch.ones(512, device="cuda")
    padded = torch.tensor([capacity], device="cuda", dtype=torch.int32)
    limits, tile_experts, routes = (
        nv_io["_keepalive"][7],
        nv_io["_keepalive"][6],
        nv_io["_keepalive"][8],
    )
    source_sf = nv_io["_keepalive"][3]

    def pointer(dtype, tensor):
        return make_ptr(
            dtype, tensor.data_ptr(), cutlass.AddressSpace.gmem, assumed_align=16
        )

    def prepare_scales():
        return preroute_nvfp4_activation_scales(
            source_sf, routes, limits, padded, capacity=capacity, scale_columns=64
        )

    prepared = prepare_scales()
    nv_io.update(
        sfb_dp=pointer(cutlass.Float8E4M3FN, prepared),
        c0_dp=pointer(cutlass.Float4E2M1FN, packed),
        sf_c_dp=pointer(cutlass.Float32, raw),
        per_token_sf_b_dp=pointer(cutlass.Float32, input_token_sf),
    )
    nv_kernel = gemm._compile_for_launch(nv_io, stream)

    def launch_kernel(io):
        nv_kernel(
            *gemm._launch_arg_tuple(
                io, driver.CUstream(torch.cuda.current_stream().cuda_stream)
            )
        )

    def output_scales():
        return normalize_nvfp4_output_scales(
            raw,
            scale_bytes,
            limits,
            padded,
            tile_n=128,
            sf_layout=3,
            input_token_scales=input_token_sf,
            route_map=routes,
            tile_experts=tile_experts,
            expert_gate_scales=expert_sf,
            expert_output_scales=expert_sf,
        )

    def pipeline():
        routed = prepare_scales()
        launch_kernel(dict(nv_io, sfb_dp=pointer(cutlass.Float8E4M3FN, routed)))
        return output_scales()

    pipeline()
    torch.cuda.synchronize()
    valid = (
        torch.arange(capacity, device="cuda")
        < limits[torch.arange(capacity, device="cuda") // 128]
    )
    assert torch.isfinite(raw[valid]).all()
    gold_packed, gold_raw = packed[valid].clone(), raw[valid].clone()
    if args.profile:
        for _ in range(50):
            pipeline()
        flush = torch.empty(384 * 1024 * 1024, device="cuda", dtype=torch.uint8)
        flush.zero_()
        torch.cuda.synchronize()
        torch.cuda.profiler.start()
        pipeline()
        torch.cuda.synchronize()
        torch.cuda.profiler.stop()
        return dict(tokens=tokens, profiled=True)

    operations, keepalive = [], []
    for stages, throttle in [(6, 1), (7, 1), (8, 1), (8, 0)]:
        config = dict(
            bf16,
            num_stages_a=stages,
            num_stages_b=stages,
            epi_tile_n=64,
            use_work_throttle=throttle,
        )
        io = dict(bf_io, cfg=make_config(**config))
        fn = gemm._compile_for_launch(io, stream)
        operations.append(
            (
                f"bf16_s{stages}_throttle{throttle}",
                lambda fn=fn, io=io: fn(
                    *gemm._launch_arg_tuple(
                        io, driver.CUstream(torch.cuda.current_stream().cuda_stream)
                    )
                ),
            )
        )
        keepalive.append((fn, io))
    operations.extend(
        [
            ("nvfp4_kernel", lambda: launch_kernel(nv_io)),
            ("nvfp4_complete", pipeline),
            ("nvfp4_input_scales_isolated", prepare_scales),
            ("nvfp4_output_scales_isolated", output_scales),
        ]
    )
    warm_end = time.monotonic() + 5
    while time.monotonic() < warm_end:
        for _, operation in operations:
            for _ in range(16):
                operation()
        torch.cuda.synchronize()
    rounds = {name: [] for name, _ in operations}
    for index in range(args.rounds):
        ordered = operations if index % 2 == 0 else operations[::-1]
        for name, operation in ordered:
            values = bench_gpu_time_with_cupti(
                operation,
                repeat_iters=args.repeat_iters,
                dry_run_iters=3,
                use_cuda_graph=True,
                cold_l2_cache=True,
                aggregate_op=sum,
            )
            rounds[name].append(statistics.median(values))
    packed.fill_(255)
    raw.fill_(float("nan"))
    launch_kernel(nv_io)
    torch.cuda.synchronize()
    assert torch.equal(gold_packed, packed[valid])
    assert torch.equal(gold_raw, raw[valid])
    milliseconds = {name: statistics.median(values) for name, values in rounds.items()}
    best_bf16 = min(
        value for name, value in milliseconds.items() if name.startswith("bf16")
    )
    return dict(
        **shape,
        routing=args.routing,
        milliseconds=milliseconds,
        rounds_ms=rounds,
        kernel_speedup=best_bf16 / milliseconds["nvfp4_kernel"],
        complete_speedup=best_bf16 / milliseconds["nvfp4_complete"],
        fresh_output_verified=True,
        phase_note="Isolated helper times have different cache state; do not sum them",
        gpu=torch.cuda.get_device_name(),
        nvfp4_config=nvfp4,
        boundary=(
            "BF16 FC1 vs NVFP4 input scale preparation + fused FC1 "
            "+ output scale metadata"
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16960, 33920])
    parser.add_argument("--routing", choices=("cyclic", "shuffled"), default="shuffled")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeat-iters", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if min(args.tokens) < 8192 or min(args.rounds, args.repeat_iters) <= 0:
        parser.error("Use at least 8192 tokens and positive measurement counts")
    os.environ["FLASHINFER_PRIMS_TS_FUSED_NVFP4_FC1"] = "1"
    os.environ["FLASHINFER_PRIMS_TS_DEFER_NVFP4_TOKEN_SCALE"] = "1"
    torch.set_num_threads(4)
    results = [benchmark(tokens, args) for tokens in args.tokens]
    result = json.dumps(results, indent=2)
    print(result)
    if args.output:
        args.output.write_text(result + "\n")


if __name__ == "__main__":
    main()
