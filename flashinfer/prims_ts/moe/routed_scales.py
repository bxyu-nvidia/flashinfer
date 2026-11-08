# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Prepare reusable activation scale tiles for routed NVFP4 GEMMs."""

from functools import cache
import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Constexpr
from cutlass.experimental import primitives as prims


@cute.kernel
def _kernel(
    source: cute.Tensor,
    output: cute.Tensor,
    route: cute.Tensor,
    limits: cute.Tensor,
    padded: cute.Tensor,
    capacity: Int32,
    columns: Constexpr,
    vectorized: Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    tile, _, _ = cute.arch.block_idx()
    row = tile * 128 + (tid % 4) * 32 + tid // 4
    end = limits[tile]
    if end > padded[0]:
        end = padded[0]
    valid = (row < capacity) & (row < end)
    token = Int32(0)
    if valid:
        token = route[row]
    for block in cutlass.range_constexpr(columns // 16):
        values = cute.make_rmem_tensor((4,), Int32)
        for q in cutlass.range_constexpr(4):
            values[q] = Int32(0)
        if valid:
            if cutlass.const_expr(vectorized):
                loaded = prims.inline_ptx_hl(
                    "ld.global.v4.b32 { {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];",
                    write_only_types=[Int32] * 4,
                    read_only_args=[
                        source.iterator + Int64(token) * (columns // 4) + block * 4
                    ],
                )
                for q in cutlass.range_constexpr(4):
                    values[q] = loaded[q]
            else:
                for q in cutlass.range_constexpr(4):
                    values[q] = source[Int64(token) * (columns // 4) + block * 4 + q]
        for q in cutlass.range_constexpr(4):
            offset = (Int64(tile) * (columns // 4) + block * 4 + q) * 128 + tid
            output[offset] = values[q]


@cute.jit
def _launch(
    source: cute.Tensor,
    output: cute.Tensor,
    route: cute.Tensor,
    limits: cute.Tensor,
    padded: cute.Tensor,
    capacity: Int32,
    columns: Constexpr,
    vectorized: Constexpr,
    stream,
):
    _kernel(
        source, output, route, limits, padded, capacity, columns, vectorized
    ).launch(
        grid=[cute.ceil_div(capacity, 128), 1, 1], block=[128, 1, 1], stream=stream
    )


@cache
def _get(columns, arch, vectorized):
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    fake = lambda: cute.runtime.make_fake_compact_tensor(
        Int32, (cute.sym_int(),), assumed_align=4
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return build_and_load_cute_dsl_kernel(
        "preroute_sf_vector",
        f"{columns}_{arch[0]}{arch[1]}_vec{int(vectorized)}",
        lambda: cute.compile(
            _launch,
            fake(),
            fake(),
            fake(),
            fake(),
            fake(),
            Int32(1),
            columns,
            vectorized,
            stream,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def preroute_nvfp4_activation_scales(
    scales: torch.Tensor,
    route_map: torch.Tensor,
    tile_limits: torch.Tensor,
    padded_rows: torch.Tensor,
    *,
    capacity: int,
    scale_columns: int,
) -> torch.Tensor:
    """Copy linear input scales to routed R128c4 tiles, zeroing padding.

    Only scale bytes are copied. The original packed activation elements remain
    compact and are gathered by the GEMM.
    """
    if scale_columns % 16 or scales.data_ptr() % 4:
        raise ValueError("Routed NVFP4 scales require aligned, complete scale words")
    tiles = (capacity + 127) // 128
    if tile_limits.numel() < tiles or route_map.numel() < capacity:
        raise ValueError("Routing metadata must cover the scale tile capacity")
    output = torch.empty(
        tiles * 128 * scale_columns, dtype=torch.uint8, device=scales.device
    )
    if tiles:
        fn = _get(
            scale_columns,
            torch.cuda.get_device_capability(scales.device),
            scales.data_ptr() % 16 == 0,
        )
        fn(
            scales.view(-1).view(torch.int32),
            output.view(torch.int32),
            route_map,
            tile_limits,
            padded_rows,
            capacity,
        )
    return output
