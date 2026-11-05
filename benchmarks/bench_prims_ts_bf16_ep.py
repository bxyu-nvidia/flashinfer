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

"""BF16 ReLU2 Prims EP probe: all-gather -> local experts -> reduce-scatter.

Launch with torch.distributed.run; timings exclude the common pre-replay barrier.
Synthetic request-specific router biases model mixed, uneven expert demand.
Timings use CUDA events around captured graphs and MAX over ranks per sample.
GEMM FLOPs = 4 * routed tokens across all ranks * hidden * intermediate.

Example (from the FlashInfer checkout, with an activated development venv):
    torchrun --standalone --nproc-per-node=4 benchmarks/bench_prims_ts_bf16_ep.py \
        --output /tmp/prims-ep.json
"""

import argparse
import gc
import importlib.util
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokens", type=int, nargs="+", default=[16960, 33920])
    parser.add_argument("--requests", type=int, nargs="+", default=[4, 8])
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--router-bias", type=float, default=1.5)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(
        "nccl", device_id=torch.device("cuda", torch.cuda.current_device())
    )
    torch.set_num_threads(1)
    torch.manual_seed(20261104 + rank)

    from flashinfer import ActivationType, RoutingMethodType
    from flashinfer.fused_moe import WeightLayout, prims_ts_bf16_routed_moe
    from flashinfer.prims_ts.moe import runner as runner_module
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_bf16_moe_tactic

    fi_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(fi_root / "benchmarks"))
    spec = importlib.util.spec_from_file_location(
        "existing_moe_bench",
        fi_root / "benchmarks/bench_trtllm_gen_fused_moe_autotuner.py",
    )
    bench = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = bench
    spec.loader.exec_module(bench)

    E, H, I, K = 512, 1024, 2688, 22
    assert E % world == 0
    L = E // world
    offset = rank * L
    a = torch.randn(L, I, H, device="cuda", dtype=torch.bfloat16) * 0.02
    b = torch.randn(L, H, I, device="cuda", dtype=torch.bfloat16) * 0.02
    w1, w2 = bench._shuffle_bf16_block_major_k(
        a, b, activation_type=int(ActivationType.Relu2)
    )
    common = dict(
        gemm1_weights=w1,
        gemm2_weights=w2,
        num_experts=E,
        top_k=K,
        n_group=None,
        topk_group=None,
        intermediate_size=I,
        local_expert_offset=offset,
        local_num_experts=L,
        routed_scaling_factor=None,
        routing_method_type=int(RoutingMethodType.Renormalize),
        use_shuffled_weight=True,
        weight_layout=int(WeightLayout.BlockMajorK),
        do_finalize=True,
        enable_pdl=True,
        activation_type=int(ActivationType.Relu2),
        tune_max_num_tokens=max(args.tokens) * world,
    )

    original = runner_module.PrimsTsBf16MoERunner._forward_prepared
    active_pair = None

    def forward(self, moe_inputs, pair, **kwargs):
        return original(self, moe_inputs, active_pair, **kwargs)

    runner_module.PrimsTsBf16MoERunner._forward_prepared = forward

    def pair(index):
        return map_trtllm_bf16_moe_tactic(
            [128, index],
            activation_type=int(ActivationType.Relu2),
            weight_layout=int(WeightLayout.BlockMajorK),
            enable_pdl=True,
        )

    def invoke(x, ids, weights):
        return prims_ts_bf16_routed_moe(
            topk_ids=(ids, weights), hidden_states=x, **common
        )

    def inputs(tokens, requests):
        generator = torch.Generator(device="cuda").manual_seed(20261104 + tokens + rank)
        x = torch.randn(
            tokens, H, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        scores = torch.randn(tokens, E, device="cuda", generator=generator)
        # Triangular length weights give each request a different prompt/chunk size.
        lengths = [
            tokens * (j + 1) // (requests * (requests + 1) // 2)
            for j in range(requests)
        ]
        lengths[-1] += tokens - sum(lengths)
        start = 0
        for j, length in enumerate(lengths):
            hot = (torch.arange(64, device="cuda") + (rank * requests + j) * 37) % E
            scores[start : start + length, hot] += args.router_bias
            start += length
        values, ids = scores.topk(K, dim=-1)
        return (x, ids.int(), values.softmax(dim=-1)), lengths

    def gathered(local):
        result = [
            torch.empty(
                (world * v.shape[0], *v.shape[1:]), dtype=v.dtype, device="cuda"
            )
            for v in local
        ]
        for out, value in zip(result, local, strict=True):
            dist.all_gather_into_tensor(out, value)
        return result

    def reference(x, ids, weights):
        result = torch.zeros_like(x, dtype=torch.float32)
        for expert in range(L):
            rows, slots = torch.where(ids == offset + expert)
            if rows.numel():
                z = (x[rows].float() @ a[expert].float().T).relu().square().bfloat16()
                y = (z.float() @ b[expert].float().T).bfloat16()
                result.index_add_(0, rows, y.float() * weights[rows, slots, None])
        return result.bfloat16()

    active_pair = pair(6)
    small, _ = inputs(63, 4)
    full = gathered(small)
    actual = invoke(*full)
    expected = reference(*full)
    local_error = (actual.float() - expected.float()).norm() / expected.float().norm()
    assert float(local_error) < 0.015, float(local_error)
    expected_rs = torch.empty_like(small[0])
    actual_rs = torch.empty_like(small[0])
    dist.reduce_scatter_tensor(expected_rs, expected)
    dist.reduce_scatter_tensor(actual_rs, actual)
    error = (
        actual_rs.float() - expected_rs.float()
    ).norm() / expected_rs.float().norm()
    dist.all_reduce(error, op=dist.ReduceOp.MAX)
    assert float(error) < 0.015, float(error)
    if rank == 0:
        print("DISTRIBUTED_REFERENCE_REL_L2", float(error), flush=True)
    del full, small, actual, expected, actual_rs, expected_rs

    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)

    def sample(graph):
        flush.zero_()
        dist.barrier()
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        value = torch.tensor(
            start.elapsed_time(end), device="cuda", dtype=torch.float64
        )
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        return float(value)

    results = []
    for tokens in args.tokens:
        for requests in args.requests:
            local, lengths = inputs(tokens, requests)
            full = gathered(local)
            local_routes = ((full[1] >= offset) & (full[1] < offset + L)).sum()
            route_counts = [torch.empty_like(local_routes) for _ in range(world)]
            dist.all_gather(route_counts, local_routes)
            # Tune one graph at a time: routing scratch is sized for M * top_k.
            trial = []
            gold = None
            for index in range(8):
                active_pair = pair(index)
                for _ in range(3):
                    invoke(*full)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = invoke(*full)
                graph.replay()
                torch.cuda.synchronize()
                if gold is None:
                    gold = output.clone()
                else:
                    assert torch.equal(gold, output), "pipeline output differs"
                for _ in range(80):
                    graph.replay()
                torch.cuda.synchronize()
                trial.append(statistics.median(sample(graph) for _ in range(12)))
                del graph, output
                gc.collect()
                torch.cuda.empty_cache()
            baseline = min(range(4), key=lambda j: trial[j])
            candidate = min(range(4, 8), key=lambda j: trial[j])
            graphs, outputs = [], []
            for chosen in (baseline, candidate):
                active_pair = pair(chosen)
                invoke(*full)
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = invoke(*full)
                graphs.append(graph)
                outputs.append(output)
            for _ in range(30):
                for graph in graphs:
                    graph.replay()
            torch.cuda.synchronize()
            kernel_samples = [[], []]
            for repeat in range(args.iterations):
                for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                    kernel_samples[arm].append(sample(graphs[arm]))

            ep_graphs, ep_outputs = [], []
            for chosen in (baseline, candidate):
                active_pair = pair(chosen)
                rs = torch.empty_like(local[0])

                def ep_step(full=full, local=local, rs=rs):
                    for out, value in zip(full, local, strict=True):
                        dist.all_gather_into_tensor(out, value)
                    partial = invoke(*full)
                    dist.reduce_scatter_tensor(rs, partial)

                for _ in range(3):
                    ep_step()
                torch.cuda.synchronize()
                dist.barrier()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    ep_step()
                ep_graphs.append(graph)
                ep_outputs.append(rs)
            for graph in ep_graphs:
                graph.replay()
            torch.cuda.synchronize()
            assert torch.equal(ep_outputs[0], ep_outputs[1]), "EP output differs"
            for _ in range(30):
                for graph in ep_graphs:
                    graph.replay()
            torch.cuda.synchronize()
            ep_samples = [[], []]
            for repeat in range(args.iterations):
                for arm in (0, 1) if repeat % 2 == 0 else (1, 0):
                    ep_samples[arm].append(sample(ep_graphs[arm]))
            medians = [
                list(map(statistics.median, values))
                for values in (kernel_samples, ep_samples)
            ]
            row = dict(
                tokens_per_worker=tokens,
                requests_per_worker=requests,
                request_lengths=lengths,
                global_tokens=tokens * world,
                routes_per_rank=[int(v) for v in route_counts],
                tactics=[baseline, candidate],
                tuning_trial_ms=trial,
                kernel_ms=medians[0],
                kernel_aggregate_tflops=[
                    4 * sum(int(v) for v in route_counts) * H * I / ms / 1e9
                    for ms in medians[0]
                ],
                ep_ms=medians[1],
                kernel_speedup=medians[0][0] / medians[0][1],
                ep_speedup=medians[1][0] / medians[1][1],
                kernel_samples_ms=kernel_samples,
                ep_samples_ms=ep_samples,
            )
            results.append(row)
            if rank == 0:
                print("RESULT", json.dumps(row), flush=True)
                Path(args.output).write_text(
                    json.dumps(
                        dict(
                            args=vars(args),
                            world=world,
                            hardware=torch.cuda.get_device_name(),
                            dtype="bfloat16",
                            activation="relu2",
                            shape=dict(
                                global_experts=E,
                                local_experts=L,
                                hidden=H,
                                intermediate=I,
                                top_k=K,
                            ),
                            timer="CUDA events; median of per-sample rank MAX",
                            exact_baseline_candidate_outputs=True,
                            torch=torch.__version__,
                            nccl=torch.cuda.nccl.version(),
                            reference_relative_l2=float(error),
                            results=results,
                        ),
                        indent=2,
                    )
                )
            del (
                graphs,
                outputs,
                ep_graphs,
                ep_outputs,
                full,
                local,
                graph,
                output,
                rs,
                gold,
                ep_step,
            )
            gc.collect()
            torch.cuda.empty_cache()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
