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

"""
Scale metadata normalization for direct NVFP4 FC1 output.
"""

from functools import cache

import torch
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Int64, Float32, Constexpr
from cutlass.experimental import primitives as prims


@cute.kernel
def _normalize_scales_kernel(
    raw: cute.Tensor,
    scales: cute.Tensor,
    token: cute.Tensor,
    tile_limits: cute.Tensor,
    padded_rows: cute.Tensor,
    input_token_scales: cute.Tensor,
    route_map: cute.Tensor,
    tile_experts: cute.Tensor,
    expert_gate_scales: cute.Tensor,
    expert_output_scales: cute.Tensor,
    capacity: Int32,
    cols: Constexpr,
    layout: Constexpr,
    tile_n: Constexpr,
    defer_token_scale: Constexpr,
    defer_expert_scales: Constexpr,
    vectorized: Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tid % 32
    physical_row = cta * 8 + tid // 32
    row = Int32(-1)
    if physical_row < capacity:
        if physical_row < padded_rows[0]:
            if physical_row < tile_limits[physical_row // tile_n]:
                row = physical_row
    if (row >= 0) & (row < capacity):
        width = cutlass.const_expr(4 if vectorized else 1)
        groups = cutlass.const_expr(cute.ceil_div(cols, 32 * width))
        local = cute.make_rmem_tensor((groups, width), Float32)
        amax = Float32(0.0)
        for j in cutlass.range_constexpr(groups):
            col = lane * width + j * 32 * width
            for q in cutlass.range_constexpr(width):
                local[j, q] = Float32(0.0)
            if col < cols:
                if cutlass.const_expr(vectorized):
                    values = prims.inline_ptx_hl(
                        "ld.global.v4.f32 { {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];",
                        write_only_types=[Float32] * 4,
                        read_only_args=[raw.iterator + Int64(row) * cols + col],
                    )
                    for q in cutlass.range_constexpr(4):
                        local[j, q] = values[q]
                else:
                    local[j, 0] = raw[Int64(row) * cols + col]
            for q in cutlass.range_constexpr(width):
                amax = cute.math.max(amax, local[j, q])
        for mask in (16, 8, 4, 2, 1):
            other = prims.shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=amax.bitcast(Int32),
                offset=mask,
                mask_and_clamp=0x1F,
                kind=prims.Shfl.BFLY,
                return_value_and_is_valid=False,
            ).bitcast(Float32)
            amax = cute.math.max(amax, other)
        # A power-of-two row scale preserves each block's E4M3 mantissa.
        target = amax * Float32(1.0 / 448.0)
        bits = target.bitcast(Int32)
        exp_bits = bits & Int32(0x7F800000)
        if (bits & Int32(0x007FFFFF)) != 0:
            exp_bits = exp_bits + Int32(0x00800000)
        global_sf = exp_bits.bitcast(Float32)
        inv = Float32(0.0)
        if global_sf > Float32(0.0):
            inv = cute.math.rcp(global_sf, approx=True, ftz=True)
        if lane == 0:
            output_token_sf = global_sf
            if cutlass.const_expr(defer_token_scale):
                input_sf = input_token_scales[route_map[row]]
                output_token_sf = output_token_sf * input_sf
                if cutlass.const_expr(defer_expert_scales):
                    expert = tile_experts[row // tile_n]
                    gate = expert_gate_scales[expert]
                    output_token_sf = output_token_sf * gate
                output_token_sf = output_token_sf * input_sf
                if cutlass.const_expr(defer_expert_scales):
                    output_token_sf = output_token_sf * gate
                    output_token_sf = output_token_sf * expert_output_scales[expert]
            token[row] = output_token_sf
        for j in cutlass.range_constexpr(groups):
            col = lane * width + j * 32 * width
            if col < cols:
                idx = Int64(0)
                if cutlass.const_expr(layout == 1):
                    idx = (
                        (Int64(row // 8) * cute.ceil_div(cols, 4) + col // 4) * 32
                        + (row % 8) * 4
                        + col % 4
                    )
                else:
                    idx = (
                        (Int64(row // 128) * cute.ceil_div(cols, 4) + col // 4) * 512
                        + ((row % 32) * 4 + (row % 128) // 32) * 4
                        + col % 4
                    )
                if cutlass.const_expr(vectorized):
                    packed = prims.inline_ptx_hl(
                        "{ .reg .b16 lo, hi; "
                        "cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0}; "
                        "cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2}; "
                        "mov.b32 {$w0}, {lo, hi}; }",
                        write_only_types=[Int32],
                        read_only_args=[
                            local[j, 0] * inv,
                            local[j, 1] * inv,
                            local[j, 2] * inv,
                            local[j, 3] * inv,
                        ],
                    )
                    prims.inline_ptx_hl(
                        "st.global.b32 [{$r0}], {$r1};",
                        read_only_args=[scales.iterator + idx, packed],
                    )
                else:
                    scales[idx] = (local[j, 0] * inv).to(cutlass.Float8E4M3FN)


@cute.jit
def _normalize_scales(
    raw: cute.Tensor,
    scales: cute.Tensor,
    token: cute.Tensor,
    tile_limits: cute.Tensor,
    padded_rows: cute.Tensor,
    input_token_scales: cute.Tensor,
    route_map: cute.Tensor,
    tile_experts: cute.Tensor,
    expert_gate_scales: cute.Tensor,
    expert_output_scales: cute.Tensor,
    capacity: Int32,
    cols: Constexpr,
    layout: Constexpr,
    tile_n: Constexpr,
    defer_token_scale: Constexpr,
    defer_expert_scales: Constexpr,
    vectorized: Constexpr,
    stream,
):
    _normalize_scales_kernel(
        raw,
        scales,
        token,
        tile_limits,
        padded_rows,
        input_token_scales,
        route_map,
        tile_experts,
        expert_gate_scales,
        expert_output_scales,
        capacity,
        cols,
        layout,
        tile_n,
        defer_token_scale,
        defer_expert_scales,
        vectorized,
    ).launch(grid=[cute.ceil_div(capacity, 8), 1, 1], block=[256, 1, 1], stream=stream)


@cute.kernel
def _transpose_kernel(
    raw: cute.Tensor,
    scales: cute.Tensor,
    token: cute.Tensor,
    tile_limits: cute.Tensor,
    padded_rows: cute.Tensor,
    input_token_scales: cute.Tensor,
    route_map: cute.Tensor,
    tile_experts: cute.Tensor,
    expert_gate_scales: cute.Tensor,
    expert_output_scales: cute.Tensor,
    capacity: Int32,
    cols: Constexpr,
    layout: Constexpr,
    tile_n: Constexpr,
    defer_token_scale: Constexpr,
    defer_expert_scales: Constexpr,
    vectorized: Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    cta, _, _ = cute.arch.block_idx()
    lane = tid % 16
    block_start = cta * 128
    valid_count = Int32(0)
    if block_start < capacity:
        if block_start < padded_rows[0]:
            valid_count = tile_limits[cta] - block_start
            if valid_count > Int32(128):
                valid_count = Int32(128)
            if valid_count > capacity - block_start:
                valid_count = capacity - block_start
            if valid_count > padded_rows[0] - block_start:
                valid_count = padded_rows[0] - block_start
            if valid_count < Int32(0):
                valid_count = Int32(0)
    dense = valid_count >= Int32(16)
    col_groups = cutlass.const_expr(cols // 4)
    pitch = cutlass.const_expr(col_groups if col_groups % 2 else col_groups + 1)
    smem = cutlass.memory.SmemAllocator()
    staged = smem.allocate_tensor(
        Int32, cute.make_layout((128, pitch), stride=(pitch, 1)), byte_alignment=16
    )
    tile_gate = Float32(1.0)
    tile_output = Float32(1.0)
    if cutlass.const_expr(defer_expert_scales):
        if valid_count > 0:
            expert = tile_experts[cta]
            tile_gate = expert_gate_scales[expert]
            tile_output = expert_output_scales[expert]
    for row_step in cutlass.range(cute.ceil_div(valid_count, 16)):
        row_offset = row_step * 16 + tid // 16
        row = block_start + row_offset
        if row_offset < valid_count:
            prefetched_input = Float32(1.0)
            if cutlass.const_expr(defer_token_scale):
                if lane == 0:
                    prefetched_input = input_token_scales[route_map[row]]
            width = cutlass.const_expr(4 if vectorized else 1)
            groups = cutlass.const_expr(cute.ceil_div(cols, 16 * width))
            local = cute.make_rmem_tensor((groups, width), Float32)
            amax = Float32(0.0)
            for j in cutlass.range_constexpr(groups):
                col = lane * width + j * 16 * width
                for q in cutlass.range_constexpr(width):
                    local[j, q] = Float32(0.0)
                if col < cols:
                    if cutlass.const_expr(vectorized):
                        values = prims.inline_ptx_hl(
                            "ld.global.v4.f32 { {$w0}, {$w1}, {$w2}, {$w3} }, [{$r0}];",
                            write_only_types=[Float32] * 4,
                            read_only_args=[raw.iterator + Int64(row) * cols + col],
                        )
                        for q in cutlass.range_constexpr(4):
                            local[j, q] = values[q]
                    else:
                        local[j, 0] = raw[Int64(row) * cols + col]
                for q in cutlass.range_constexpr(width):
                    amax = cute.math.max(amax, local[j, q])
            # Each 16-lane row has its own mask, including partial tiles.
            amax = prims.inline_ptx_hl(
                "redux.sync.max.u32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Int32],
                read_only_args=[
                    amax.bitcast(Int32),
                    Int32(0xFFFF) << ((tid % 32) // 16 * 16),
                ],
            ).bitcast(Float32)
            # A power-of-two row scale preserves each block's E4M3 mantissa.
            target = amax * Float32(1.0 / 448.0)
            bits = target.bitcast(Int32)
            exp_bits = bits & Int32(0x7F800000)
            if (bits & Int32(0x007FFFFF)) != 0:
                exp_bits = exp_bits + Int32(0x00800000)
            global_sf = exp_bits.bitcast(Float32)
            inv = Float32(0.0)
            if global_sf > Float32(0.0):
                inv = cute.math.rcp(global_sf, approx=True, ftz=True)
            if lane == 0:
                output_token_sf = global_sf
                if cutlass.const_expr(defer_token_scale):
                    input_sf = prefetched_input
                    output_token_sf = output_token_sf * input_sf
                    if cutlass.const_expr(defer_expert_scales):
                        gate = tile_gate
                        output_token_sf = output_token_sf * gate
                    output_token_sf = output_token_sf * input_sf
                    if cutlass.const_expr(defer_expert_scales):
                        output_token_sf = output_token_sf * gate
                        output_token_sf = output_token_sf * tile_output
                token[row] = output_token_sf
            for j in cutlass.range_constexpr(groups):
                col = lane * width + j * 16 * width
                if col < cols:
                    idx = Int64(0)
                    if cutlass.const_expr(layout == 1):
                        idx = (
                            (Int64(row // 8) * cute.ceil_div(cols, 4) + col // 4) * 32
                            + (row % 8) * 4
                            + col % 4
                        )
                    else:
                        idx = (
                            (Int64(row // 128) * cute.ceil_div(cols, 4) + col // 4)
                            * 512
                            + ((row % 32) * 4 + (row % 128) // 32) * 4
                            + col % 4
                        )
                    if cutlass.const_expr(vectorized):
                        packed = prims.inline_ptx_hl(
                            "{ .reg .b16 lo, hi; "
                            "cvt.rn.satfinite.e4m3x2.f32 lo, {$r1}, {$r0}; "
                            "cvt.rn.satfinite.e4m3x2.f32 hi, {$r3}, {$r2}; "
                            "mov.b32 {$w0}, {lo, hi}; }",
                            write_only_types=[Int32],
                            read_only_args=[
                                local[j, 0] * inv,
                                local[j, 1] * inv,
                                local[j, 2] * inv,
                                local[j, 3] * inv,
                            ],
                        )
                        if dense:
                            physical_slot = (row % 32) * 4 + (row % 128) // 32
                            staged[physical_slot, col // 4] = packed
                        else:
                            prims.inline_ptx_hl(
                                "st.global.b32 [{$r0}], {$r1};",
                                read_only_args=[scales.iterator + idx, packed],
                            )
                    else:
                        scales[idx] = (local[j, 0] * inv).to(cutlass.Float8E4M3FN)
    if dense:
        cute.arch.barrier(barrier_id=0, number_of_threads=256)
        physical_slot = tid % 128
        logical_slot = (physical_slot % 4) * 32 + physical_slot // 4
        if logical_slot < valid_count:
            for group_step in cutlass.range_constexpr(cute.ceil_div(col_groups, 2)):
                group = group_step * 2 + tid // 128
                if group < col_groups:
                    packed = staged[physical_slot, group]
                    idx = (Int64(cta) * col_groups + group) * 512 + physical_slot * 4
                    prims.inline_ptx_hl(
                        "st.global.b32 [{$r0}], {$r1};",
                        read_only_args=[scales.iterator + idx, packed],
                    )


@cute.jit
def _transpose_launch(
    raw: cute.Tensor,
    scales: cute.Tensor,
    token: cute.Tensor,
    tile_limits: cute.Tensor,
    padded_rows: cute.Tensor,
    input_token_scales: cute.Tensor,
    route_map: cute.Tensor,
    tile_experts: cute.Tensor,
    expert_gate_scales: cute.Tensor,
    expert_output_scales: cute.Tensor,
    capacity: Int32,
    cols: Constexpr,
    layout: Constexpr,
    tile_n: Constexpr,
    defer_token_scale: Constexpr,
    defer_expert_scales: Constexpr,
    vectorized: Constexpr,
    stream,
):
    _transpose_kernel(
        raw,
        scales,
        token,
        tile_limits,
        padded_rows,
        input_token_scales,
        route_map,
        tile_experts,
        expert_gate_scales,
        expert_output_scales,
        capacity,
        cols,
        layout,
        tile_n,
        defer_token_scale,
        defer_expert_scales,
        vectorized,
    ).launch(
        grid=[cute.ceil_div(capacity, 128), 1, 1], block=[256, 1, 1], stream=stream
    )


@cache
def _get_normalizer(
    cols: int,
    layout: int,
    tile_n: int,
    arch: tuple[int, int],
    defer_token_scale: bool,
    defer_expert_scales: bool,
    vectorized: bool,
):
    from flashinfer.jit.cute_dsl_core import build_and_load_cute_dsl_kernel

    fake = lambda dtype: cute.runtime.make_fake_compact_tensor(
        dtype,
        (cute.sym_int(),),
        assumed_align=1 if dtype == cutlass.Float8E4M3FN else 4,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    transpose = vectorized and layout == 3 and tile_n == 128 and cols <= 256
    launch = _transpose_launch if transpose else _normalize_scales
    return build_and_load_cute_dsl_kernel(
        "prims_nvfp4_output",
        f"normalize_{cols}_{layout}_{tile_n}_{arch[0]}{arch[1]}_token{int(defer_token_scale)}_expert{int(defer_expert_scales)}_vector{int(vectorized)}",
        lambda: cute.compile(
            launch,
            fake(Float32),
            fake(cutlass.Float8E4M3FN),
            fake(Float32),
            fake(Int32),
            fake(Int32),
            fake(Float32),
            fake(Int32),
            fake(Int32),
            fake(Float32),
            fake(Float32),
            Int32(1),
            cols,
            layout,
            tile_n,
            defer_token_scale,
            defer_expert_scales,
            vectorized,
            stream,
            options="--enable-tvm-ffi",
        ),
        extra_key_files=(__file__,),
    )


def normalize_nvfp4_output_scales(
    block_scales: torch.Tensor,
    output_scales: torch.Tensor,
    tile_limits: torch.Tensor,
    padded_rows: torch.Tensor,
    *,
    tile_n: int,
    sf_layout: int,
    input_token_scales: torch.Tensor | None = None,
    route_map: torch.Tensor | None = None,
    tile_experts: torch.Tensor | None = None,
    expert_gate_scales: torch.Tensor | None = None,
    expert_output_scales: torch.Tensor | None = None,
    vectorized: bool = True,
) -> torch.Tensor:
    """Normalize block scales without reading FC1's packed FP4 elements.

    Physical row order coalesces writes to the swizzled scale buffer. Tile limits
    exclude routing padding and unused capacity. Power-of-two row scales preserve
    E4M3 mantissas; very small relative block scales can become subnormal or zero.
    """
    if block_scales.dtype != torch.float32 or block_scales.ndim != 2:
        raise ValueError("FC1 block scales must be a float32 matrix")
    if output_scales.dtype not in (torch.uint8, torch.float8_e4m3fn):
        raise ValueError("FC1 output scales must store E4M3 bytes")
    for tensor in (block_scales, output_scales, tile_limits, padded_rows):
        if not tensor.is_cuda or tensor.device != block_scales.device:
            raise ValueError("FC1 scales and routing metadata must share a CUDA device")
        if not tensor.is_contiguous():
            raise ValueError("FC1 scales and routing metadata must be contiguous")
    if sf_layout not in (1, 3):
        raise ValueError("FC1 output scales require R8c4 or R128c4 layout")
    if tile_n not in (64, 128):
        raise ValueError("FC1 scale normalization requires tile_N 64 or 128")
    if tile_limits.dtype != torch.int32 or tile_limits.ndim != 1:
        raise ValueError("FC1 tile limits must be an int32 vector")
    if padded_rows.dtype != torch.int32 or padded_rows.numel() != 1:
        raise ValueError("FC1 padded row count must be an int32 scalar")
    defer_token_scale = input_token_scales is not None
    if defer_token_scale != (route_map is not None):
        raise ValueError("Deferred token normalization requires scales and route map")
    if defer_token_scale:
        if input_token_scales.dtype != torch.float32 or input_token_scales.ndim != 1:
            raise ValueError("Deferred input token scales must be a float32 vector")
        if route_map.dtype != torch.int32 or route_map.ndim != 1:
            raise ValueError("Deferred token route map must be an int32 vector")
        for tensor in (input_token_scales, route_map):
            if (
                not tensor.is_cuda
                or tensor.device != block_scales.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "Deferred token metadata must be contiguous on the scales device"
                )
        if route_map.numel() < block_scales.shape[0]:
            raise ValueError("Deferred token route map must cover routed capacity")
    expert_metadata = (tile_experts, expert_gate_scales, expert_output_scales)
    defer_expert_scales = any(tensor is not None for tensor in expert_metadata)
    if defer_expert_scales:
        if not defer_token_scale or any(tensor is None for tensor in expert_metadata):
            raise ValueError(
                "Deferred expert scales require all expert and token metadata"
            )
        for tensor, dtype in zip(
            expert_metadata, (torch.int32, torch.float32, torch.float32), strict=False
        ):
            if (
                tensor.dtype != dtype
                or tensor.ndim != 1
                or tensor.device != block_scales.device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    "Deferred expert metadata has invalid dtype, shape or device"
                )
        if tile_experts.numel() < tile_limits.numel():
            raise ValueError("Deferred expert IDs must cover routing tiles")
        if expert_gate_scales.numel() != expert_output_scales.numel():
            raise ValueError("Deferred expert scale vectors must have equal length")
    capacity, cols = block_scales.shape
    if cols % 4:
        raise ValueError("FC1 scale columns must be a multiple of four")
    if tile_limits.numel() < (capacity + tile_n - 1) // tile_n:
        raise ValueError("FC1 tile limits are too small for the routed capacity")
    row_tile = 8 if sf_layout == 1 else 128
    required = ((capacity + row_tile - 1) // row_tile) * row_tile * cols
    if output_scales.numel() < required:
        raise ValueError("FC1 output scale buffer is too small")
    token_scales = torch.empty(
        capacity, dtype=torch.float32, device=block_scales.device
    )
    if capacity == 0:
        return token_scales
    vectorized = bool(
        vectorized
        and block_scales.data_ptr() % 16 == 0
        and output_scales.data_ptr() % 4 == 0
    )
    fn = _get_normalizer(
        cols,
        sf_layout,
        tile_n,
        torch.cuda.get_device_capability(block_scales.device),
        defer_token_scale,
        defer_expert_scales,
        vectorized,
    )
    fn(
        block_scales.view(-1),
        output_scales.view(torch.float8_e4m3fn).view(-1),
        token_scales,
        tile_limits,
        padded_rows.view(-1),
        input_token_scales if defer_token_scale else block_scales.view(-1),
        route_map if defer_token_scale else tile_limits,
        tile_experts if defer_expert_scales else tile_limits,
        expert_gate_scales if defer_expert_scales else block_scales.view(-1),
        expert_output_scales if defer_expert_scales else block_scales.view(-1),
        capacity,
    )
    return token_scales
