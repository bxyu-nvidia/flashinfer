# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental fused routed BF16 GEMM1 with native TRT-LLM routing metadata."""

import triton
import triton.language as tl


@triton.jit
def _routed_bf16_gemm1(
    X,
    W,
    Y,
    ROUTE,
    EXPERT,
    LIMIT,
    NUM_TILES,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BN: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    GROUP_T: tl.constexpr,
):
    pid = tl.program_id(0)
    nt = tl.load(NUM_TILES) * (128 // BT)
    nf: tl.constexpr = triton.cdiv(N, BN)
    group = pid // (GROUP_T * nf)
    start_t = group * GROUP_T
    group_t = tl.minimum(nt - start_t, GROUP_T)
    if start_t < nt:
        tile_t = start_t + pid % (GROUP_T * nf) % group_t
        tile_f = pid % (GROUP_T * nf) // group_t
        if tile_f < nf:
            route_tile = tile_t // (128 // BT)
            expert = tl.load(EXPERT + route_tile).to(tl.int64)
            limit = tl.load(LIMIT + route_tile)
            rows = tile_t * BT + tl.arange(0, BT)
            tokens = tl.load(ROUTE + rows, rows < limit, other=-1).to(tl.int64)
            features = tile_f * BN + tl.arange(0, BN)
            physical_features = (
                features // 32 * 32 + features % 4 * 8 + features % 32 // 4
            )
            ks = tl.arange(0, BK)
            acc = tl.zeros((BN, BT), tl.float32)
            for kb in range(triton.cdiv(K, BK)):
                k = kb * BK + ks
                w = tl.load(
                    W
                    + expert * N * K
                    + (k[None, :] // 64) * N * 64
                    + physical_features[:, None] * 64
                    + k[None, :] % 64,
                    (features[:, None] < N) & (k[None, :] < K),
                    other=0,
                )
                x = tl.load(
                    X + tokens[None, :] * K + k[:, None],
                    (tokens[None, :] >= 0) & (tokens[None, :] < M) & (k[:, None] < K),
                    other=0,
                )
                acc = tl.dot(w, x, acc)
            acc = tl.maximum(acc, 0)
            value = (acc * acc).to(Y.dtype.element_ty)
            tl.store(
                Y + rows[None, :].to(tl.int64) * N + features[:, None],
                value,
                (rows[None, :] < limit) & (features[:, None] < N),
            )


def routed_bf16_gemm1(
    hidden_states,
    weights,
    output,
    route,
    expert,
    limit,
    num_tiles,
    *,
    block_n=128,
    block_t=64,
    block_k=64,
    group_t=4,
    num_warps=4,
    num_stages=3,
):
    """Launch fused gather, expert up-projection, and squared ReLU.

    Weights use TRT-LLM BlockMajorK [expert, K/64, N, 64]. Routing is padded
    to 128 rows per tile. The caller owns all allocations and route planning.
    Output has native permuted-row order, directly consumable by GEMM2.
    """
    assert block_t in (32, 64, 128)
    m, k = hidden_states.shape
    n = weights.shape[-2]
    tiles = triton.cdiv(output.shape[0], block_t)
    grid = (triton.cdiv(tiles, group_t) * group_t * triton.cdiv(n, block_n),)
    return _routed_bf16_gemm1[grid](
        hidden_states,
        weights,
        output,
        route,
        expert,
        limit,
        num_tiles,
        m,
        n,
        k,
        block_n,
        block_t,
        block_k,
        group_t,
        num_warps=num_warps,
        num_stages=num_stages,
    )
