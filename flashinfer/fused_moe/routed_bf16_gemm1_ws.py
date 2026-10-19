# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Experimental persistent, warp-specialized Blackwell routed GEMM1.

Consumes TRT-LLM BlockMajorK weights and routing metadata without staging a
global expert-major activation tensor. TMA gather, MMA, and the fused Relu2
epilogue run in independent warp partitions with a double-buffered accumulator.
"""

from collections import namedtuple
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.blackwell import TensorDescriptor
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    allocate_tensor_memory,
    get_tmem_reg_layout,
    tcgen05_mma,
    tcgen05_commit,
    tma,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    mbarrier,
    fence_async_shared,
)


_Pipeline = namedtuple(
    "_Pipeline",
    [
        "xd",
        "wd",
        "yd",
        "route",
        "expert",
        "limit",
        "tile_count",
        "starts",
        "xb",
        "wb",
        "load_free",
        "load_done",
        "acc",
        "acc_free",
        "acc_done",
    ],
)


@gluon.jit
def _coordinates(
    linear, token_tiles, N: gl.constexpr, BN: gl.constexpr, GROUP_T: gl.constexpr
):
    nf: gl.constexpr = (N + BN - 1) // BN
    first = linear // (GROUP_T * nf) * GROUP_T
    size = gl.minimum(GROUP_T, token_tiles - first)
    inner = linear % (GROUP_T * nf)
    return first + inner % size, inner // size


@gluon.jit
def _load_partition(
    p,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    S: gl.constexpr,
    GROUP_T: gl.constexpr,
    BK: gl.constexpr,
    STAGE: gl.constexpr,
):
    nt = gl.load(p.tile_count)
    if BT <= 128:
        nt *= 128 // BT
    total = nt * ((N + BN - 1) // BN)
    gather_layout: gl.constexpr = gl.SliceLayout(
        0, gl.BlockedLayout([1, 4], [32, 1], [1, gl.num_warps()], [1, 0])
    )
    r = gl.arange(0, BT, gather_layout)
    seq = 0
    for linear in range(gl.program_id(0), total, gl.num_programs(0)):
        token_tile, feature_tile = _coordinates(linear, nt, N, BN, GROUP_T)
        if BT <= 128:
            rt = token_tile // (128 // BT)
            row_start = token_tile * BT
        else:
            rt = token_tile
            row_start = gl.load(p.starts + token_tile)
        expert = gl.load(p.expert + rt)
        limit = gl.load(p.limit + rt)
        if STAGE == 1:
            tokens = gl.load(p.route + row_start + r, row_start + r < limit, other=-1)
        for kb in range((K + BK - 1) // BK):
            slot = seq % S
            phase = ((seq // S) & 1) ^ 1
            mbarrier.wait(p.load_free.index(slot), phase)
            done = p.load_done.index(slot)
            mbarrier.expect(done, (BT + BN) * BK * 2)
            if STAGE == 1:
                tma.async_gather(p.xd, tokens, kb * BK, done, p.xb.index(slot))
            else:
                tma.async_copy_global_to_shared(
                    p.xd, [row_start, kb * BK], done, p.xb.index(slot)
                )
            tma.async_copy_global_to_shared(
                p.wd,
                [expert * (K // 64) + kb * (BK // 64), feature_tile * BN, 0],
                done,
                p.wb.index(slot),
            )
            seq += 1


@gluon.jit
def _mma_partition(
    p,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    S: gl.constexpr,
    GROUP_T: gl.constexpr,
    BK: gl.constexpr,
):
    BM: gl.constexpr = min(BN, 128)
    A: gl.constexpr = min(2, 512 // ((BN // BM) * BT))
    nt = gl.load(p.tile_count)
    if BT <= 128:
        nt *= 128 // BT
    total = nt * ((N + BN - 1) // BN)
    seq = 0
    tile_seq = 0
    for _linear in range(gl.program_id(0), total, gl.num_programs(0)):
        acc_slot = tile_seq % A
        acc_phase = ((tile_seq // A) & 1) ^ 1
        mbarrier.wait(p.acc_free.index(acc_slot), acc_phase)
        for kb in range((K + BK - 1) // BK):
            slot = seq % S
            phase = (seq // S) & 1
            mbarrier.wait(p.load_done.index(slot), phase)
            for sub in gl.static_range(BK // 64):
                x = p.xb.index(slot).slice(sub * 64, 64, 1).permute((1, 0))
                for chunk in gl.static_range(BN // BM):
                    w = (
                        p.wb.index(slot)
                        .reshape((BK // 64 * BN, 64))
                        .slice(sub * BN + chunk * BM, BM, 0)
                    )
                    accum = p.acc.index(acc_slot * (BN // BM) + chunk)
                    tcgen05_mma(
                        w,
                        x,
                        accum,
                        use_acc=(kb != 0) | (sub != 0),
                        mbarriers=[p.load_free.index(slot)],
                        mbarrier_preds=[
                            (sub == BK // 64 - 1) & (chunk == BN // BM - 1)
                        ],
                    )
            seq += 1
        tcgen05_commit(p.acc_done.index(acc_slot))
        tile_seq += 1  # noqa: SIM113 - Gluon does not support enumerate.


@gluon.jit
def _epilogue_partition(
    p,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    S: gl.constexpr,
    GROUP_T: gl.constexpr,
    BK: gl.constexpr,
    EPILOGUE: gl.constexpr,
    STAGE: gl.constexpr,
    STORE_BUFFERS: gl.constexpr,
):
    nt = gl.load(p.tile_count)
    if BT <= 128:
        nt *= 128 // BT
    total = nt * ((N + BN - 1) // BN)
    BM: gl.constexpr = min(BN, 128)
    ET: gl.constexpr = p.yd.block_shape[0]
    A: gl.constexpr = min(2, 512 // ((BN // BM) * BT))
    tmem_layout: gl.constexpr = TensorMemoryLayout([BM, ET], col_stride=1)
    reg_layout: gl.constexpr = get_tmem_reg_layout(
        gl.float32, [BM, ET], tmem_layout, gl.num_warps(), instr_variant=EPILOGUE
    )
    out = gl.allocate_shared_memory(gl.bfloat16, [STORE_BUFFERS, ET, BM], p.yd.layout)
    store_seq = 0
    tile_seq = 0
    for linear in range(gl.program_id(0), total, gl.num_programs(0)):
        token_tile, feature_tile = _coordinates(linear, nt, N, BN, GROUP_T)
        if BT <= 128:
            row_start = token_tile * BT
            row_limit = 2147483647
        else:
            row_start = gl.load(p.starts + token_tile)
            row_limit = gl.load(p.limit + token_tile)
        slot = tile_seq % A
        phase = (tile_seq // A) & 1
        mbarrier.wait(p.acc_done.index(slot), phase)
        for chunk in gl.static_range(BN // BM):
            for part in gl.static_range(BT // ET):
                if BT <= 128 or row_start + part * ET < row_limit:
                    values = (
                        p.acc.index(slot * (BN // BM) + chunk)
                        .slice(part * ET, ET)
                        .load(reg_layout)
                    )
                    values = (
                        values.reshape(BM // 32, 4, 8, ET)
                        .permute((0, 2, 1, 3))
                        .reshape(BM, ET)
                    )
                    if STAGE == 1:
                        values = gl.maximum(values, 0)
                        values = values * values
                    values = values.to(gl.bfloat16).permute((1, 0))
                    # Keep other stores in flight while reusing this buffer.
                    tma.store_wait(STORE_BUFFERS - 1)
                    store_buffer = out.index(store_seq % STORE_BUFFERS)
                    store_buffer.store(values)
                    if BT <= 128 and chunk == BN // BM - 1 and part == BT // ET - 1:
                        mbarrier.arrive(p.acc_free.index(slot))
                    fence_async_shared()
                    tma.async_copy_shared_to_global(
                        p.yd,
                        [row_start + part * ET, feature_tile * BN + chunk * BM],
                        store_buffer,
                    )
                    store_seq += 1
        if BT > 128:
            mbarrier.arrive(p.acc_free.index(slot))
        tile_seq += 1  # noqa: SIM113 - Gluon does not support enumerate.
    tma.store_wait(0)


@gluon.jit
def _routed_gemm1_ws(
    xd,
    wd,
    yd,
    route,
    expert,
    limit,
    tile_count,
    starts,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    S: gl.constexpr,
    GROUP_T: gl.constexpr,
    BK: gl.constexpr,
    LOAD_WARPS: gl.constexpr,
    EPILOGUE: gl.constexpr,
    STAGE: gl.constexpr = 1,
    STORE_BUFFERS: gl.constexpr = 1,
):
    xb = gl.allocate_shared_memory(gl.bfloat16, [S, BT, BK], xd.layout)
    wb = gl.allocate_shared_memory(gl.bfloat16, [S, BK // 64, BN, 64], wd.layout)
    load_free = gl.allocate_shared_memory(gl.int64, [S, 1], mbarrier.MBarrierLayout())
    load_done = gl.allocate_shared_memory(gl.int64, [S, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(S):
        mbarrier.init(load_free.index(i), count=1)
        mbarrier.init(load_done.index(i), count=1)
    BM: gl.constexpr = min(BN, 128)
    A: gl.constexpr = min(2, 512 // ((BN // BM) * BT))
    acc_layout: gl.constexpr = TensorMemoryLayout([BM, BT], col_stride=1)
    acc = allocate_tensor_memory(gl.float32, [A * (BN // BM), BM, BT], acc_layout)
    acc_free = gl.allocate_shared_memory(gl.int64, [A, 1], mbarrier.MBarrierLayout())
    acc_done = gl.allocate_shared_memory(gl.int64, [A, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(A):
        mbarrier.init(acc_free.index(i), count=1)
        mbarrier.init(acc_done.index(i), count=1)
    p = _Pipeline(
        xd,
        wd,
        yd,
        route,
        expert,
        limit,
        tile_count,
        starts,
        xb,
        wb,
        load_free,
        load_done,
        acc,
        acc_free,
        acc_done,
    )
    gl.warp_specialize(
        [
            (
                _epilogue_partition,
                (p, N, K, BN, BT, S, GROUP_T, BK, EPILOGUE, STAGE, STORE_BUFFERS),
            ),
            (_load_partition, (p, N, K, BN, BT, S, GROUP_T, BK, STAGE)),
            (_mma_partition, (p, N, K, BN, BT, S, GROUP_T, BK)),
        ],
        [LOAD_WARPS, 1],
        [64, 24],
    )


@gluon.jit
def _bf16_gemm2_ws(
    xd,
    wd,
    yd,
    route,
    expert,
    limit,
    tile_count,
    starts,
    N: gl.constexpr,
    K: gl.constexpr,
    BN: gl.constexpr,
    BT: gl.constexpr,
    S: gl.constexpr,
    GROUP_T: gl.constexpr,
    BK: gl.constexpr,
    LOAD_WARPS: gl.constexpr,
    EPILOGUE: gl.constexpr,
    STORE_BUFFERS: gl.constexpr = 1,
):
    _routed_gemm1_ws(
        xd,
        wd,
        yd,
        route,
        expert,
        limit,
        tile_count,
        starts,
        N,
        K,
        BN,
        BT,
        S,
        GROUP_T,
        BK,
        LOAD_WARPS,
        EPILOGUE,
        STAGE=2,
        STORE_BUFFERS=STORE_BUFFERS,
    )


def make_routed_gemm1_plan(
    hidden_states,
    weights,
    output,
    route,
    expert,
    limit,
    num_tiles,
    *,
    block_n=128,
    block_t=128,
    stages=4,
    load_warps=4,
    group_t=4,
    num_ctas=None,
    block_k=64,
    epilogue="32x32b",
    store_t=128,
    epilogue_warps=4,
    store_buffers=1,
    stage=1,
    job_metadata=None,
    include_planning=True,
):
    """Prepare descriptors and return a graph-capturable, allocation-free launch."""
    assert block_t in (32, 64, 128, 256)
    assert block_t <= 128 or job_metadata is not None
    starts = num_tiles
    if job_metadata is not None:
        expert, limit, num_tiles, starts = (
            job_metadata.experts,
            job_metadata.limits,
            job_metadata.count,
            job_metadata.starts,
        )
    assert block_n in (64, 128, 256, 512)
    _, k = hidden_states.shape
    n = weights.shape[-2]
    assert block_k in (64, 128, 256) and k % 64 == 0
    xl = gl.NVMMASharedLayout.get_default_for([block_t, block_k], gl.bfloat16)
    wl = gl.NVMMASharedLayout.get_default_for([block_k // 64, block_n, 64], gl.bfloat16)
    bm = min(block_n, 128)
    assert (block_n // bm) * block_t <= 512
    assert store_buffers in (1, 2)
    assert store_t in (32, 64, 128)
    assert epilogue_warps in (4, 8)
    et = min(block_t, store_t)
    yl = gl.NVMMASharedLayout.get_default_for([et, bm], gl.bfloat16)
    xd = TensorDescriptor.from_tensor(
        hidden_states, [1 if stage == 1 else block_t, block_k], xl
    )
    wd = TensorDescriptor.from_tensor(
        weights.view(-1, n, 64), [block_k // 64, block_n, 64], wl
    )
    yd = TensorDescriptor.from_tensor(output, [et, bm], yl)
    if num_ctas is None:
        import torch

        num_ctas = torch.cuda.get_device_properties(
            hidden_states.device
        ).multi_processor_count

    kernel = _routed_gemm1_ws if stage == 1 else _bf16_gemm2_ws

    def launch(*_ignored):
        if job_metadata is not None and include_planning:
            job_metadata.prepare()
        return kernel[(num_ctas,)](
            xd,
            wd,
            yd,
            route,
            expert,
            limit,
            num_tiles,
            starts,
            n,
            k,
            block_n,
            block_t,
            stages,
            group_t,
            block_k,
            load_warps,
            epilogue,
            num_warps=epilogue_warps,
            STORE_BUFFERS=store_buffers,
        )

    return launch


def make_bf16_gemm2_plan(
    hidden_states, weights, output, route, expert, limit, num_tiles, **kwargs
):
    """Prepare a native-layout BF16 expert down-projection without activation."""
    return make_routed_gemm1_plan(
        hidden_states,
        weights,
        output,
        route,
        expert,
        limit,
        num_tiles,
        stage=2,
        **kwargs,
    )
