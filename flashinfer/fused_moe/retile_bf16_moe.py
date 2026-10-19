# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Build larger GEMM jobs over native 128-row-padded expert activations."""

from dataclasses import dataclass

import triton
import triton.language as tl


@triton.jit
def _counts_from_tiles(EXPERTS, LIMITS, NUM_TILES, COUNTS, BLOCK: tl.constexpr):
    expert = tl.program_id(0)
    idx = tl.arange(0, BLOCK)
    live = idx < tl.load(NUM_TILES)
    owner = tl.load(EXPERTS + idx, live, other=-1)
    limit = tl.load(LIMITS + idx, live, other=0)
    rows = tl.minimum(tl.maximum(limit - idx * 128, 0), 128)
    tl.store(COUNTS + expert, tl.sum(tl.where(owner == expert, rows, 0)))


@triton.jit
def _retile_metadata(
    COUNTS,
    STARTS,
    EXPERTS,
    LIMITS,
    NUM_JOBS,
    OFFSET: tl.constexpr,
    E: tl.constexpr,
    TILE: tl.constexpr,
    BLOCK_E: tl.constexpr,
    BLOCK_JOBS: tl.constexpr,
):
    expert = tl.program_id(0)
    ids = tl.arange(0, BLOCK_E)
    counts = tl.load(COUNTS + OFFSET + ids, ids < E, other=0)
    native_rows = tl.cdiv(counts, 128) * 128
    native_offsets = tl.cumsum(native_rows) - native_rows
    jobs = tl.cdiv(counts, TILE)
    job_offsets = tl.cumsum(jobs) - jobs
    start = tl.sum(tl.where(ids == expert, native_offsets, 0))
    job_start = tl.sum(tl.where(ids == expert, job_offsets, 0))
    count = tl.sum(tl.where(ids == expert, counts, 0))
    num = tl.cdiv(count, TILE)
    j = tl.arange(0, BLOCK_JOBS)
    tl.store(STARTS + job_start + j, start + j * TILE, j < num)
    tl.store(EXPERTS + job_start + j, expert, j < num)
    tl.store(LIMITS + job_start + j, start + count, j < num)
    if expert == 0:
        tl.store(NUM_JOBS, tl.sum(jobs))


@dataclass
class RetiledJobs:
    starts: object
    experts: object
    limits: object
    count: object
    counts: object
    local_offset: int
    local_experts: int
    max_tokens: int
    tile: int
    native_tiles: object = None

    def prepare(self):
        if self.native_tiles is not None:
            experts, limits, num_tiles = self.native_tiles
            _counts_from_tiles[(self.local_experts,)](
                experts,
                limits,
                num_tiles,
                self.counts,
                triton.next_power_of_2(experts.numel()),
                num_warps=4,
            )
        return _retile_metadata[(self.local_experts,)](
            self.counts,
            self.starts,
            self.experts,
            self.limits,
            self.count,
            self.local_offset,
            self.local_experts,
            self.tile,
            triton.next_power_of_2(self.local_experts),
            triton.next_power_of_2(triton.cdiv(self.max_tokens, self.tile)),
            num_warps=4,
        )


def make_retiled_jobs(
    counts,
    local_offset,
    local_experts,
    max_tokens,
    capacity_rows,
    tile=256,
    native_tiles=None,
):
    """Allocate metadata once; prepare() updates it without host synchronization."""
    import torch

    assert tile == 256
    assert local_offset >= 0 and local_offset + local_experts <= counts.numel()
    if native_tiles is not None:
        # The custom FromLogits router does not publish num_tokens_per_expert.
        counts = torch.empty(local_experts, device=counts.device, dtype=torch.int32)
        local_offset = 0
    capacity = triton.cdiv(capacity_rows, tile) + local_experts
    buffers = [
        torch.empty(capacity, device=counts.device, dtype=torch.int32) for _ in range(3)
    ]
    count = torch.empty((), device=counts.device, dtype=torch.int32)
    return RetiledJobs(
        *buffers,
        count,
        counts,
        local_offset,
        local_experts,
        max_tokens,
        tile,
        native_tiles,
    )
