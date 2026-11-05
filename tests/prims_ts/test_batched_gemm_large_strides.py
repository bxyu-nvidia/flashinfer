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

"""Encode large GEMM tensor maps without allocating or accessing large matrices."""

import pytest


@pytest.mark.parametrize("swap_ab", [False, True])
@pytest.mark.parametrize("block_major_k", [False, True])
@pytest.mark.parametrize("rows", [151424, 3001216])
def test_large_operand_descriptor_strides(swap_ab, block_major_k, rows):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Prims-TS requires SM100-family GPU")

    import cutlass
    import cutlass.cute as cute
    from cuda.bindings import driver as cuda_driver
    from cutlass.cute.runtime import from_dlpack, make_ptr
    from cutlass.experimental import cuda

    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _make_operand_layouts,
    )
    from flashinfer.tllm_enums import WeightLayout

    @cute.kernel
    def record_strides(
        a_desc: cutlass.GridConstant[cuda.TensorMap],
        b_desc: cutlass.GridConstant[cuda.TensorMap],
        output: cute.Tensor,
        a_stride: cutlass.Int64,
        b_stride: cutlass.Int64,
    ):
        output[0] = a_stride
        output[1] = b_stride

    @cute.jit
    def encode(
        ptr,
        output,
        m: cutlass.Int32,
        n: cutlass.Int32,
        k: cutlass.Int32,
        cfg: cutlass.Constexpr,
        stream,
    ):
        a, b = _make_operand_layouts(m, n, k, cutlass.Int32(128), cfg)
        a_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=cute.make_tensor(ptr, a),
            box_dims=(64,) + (1,) * (cute.rank(a) - 1),
            stride_order=tuple(range(cute.rank(a))),
            swizzle=cuda.TensorMapSwizzle.s128b,
            tma_format=cuda.TensorMapDataFormat.DEFAULT,
        )
        b_desc = cuda.create_tensor_map_tiled_from_tensor(
            tensor=cute.make_tensor(ptr, b),
            box_dims=(64,) + (1,) * (cute.rank(b) - 1),
            stride_order=tuple(range(cute.rank(b))),
            swizzle=cuda.TensorMapSwizzle.s128b,
            tma_format=cuda.TensorMapDataFormat.DEFAULT,
        )
        record_strides(a_desc, b_desc, output, a.stride[-1], b.stride[-1]).launch(
            grid=(1, 1, 1), block=(1, 1, 1), stream=stream
        )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        batch_mode=int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
        transpose_mma_output=int(swap_ab),
        route_act=0,
        # FC2 swapAB BF16 bypasses B OOB even when enabled. For non-swapAB,
        # disable A OOB to exercise the symmetric flat activation descriptor.
        use_tma_oob_opt=int(swap_ab),
        weight_layout=int(
            WeightLayout.BlockMajorK if block_major_k else WeightLayout.MajorK
        ),
    )
    m, n, k = (1024, rows, 2688) if swap_ab else (rows, 1024, 2688)
    # Tensor-map encoding does not access the backing memory. Only the small
    # output tensor is accessed by record_strides; no GEMM/TMA loads run.
    backing = torch.empty(64, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(2, device="cuda", dtype=torch.int64)
    ptr = make_ptr(
        cutlass.BFloat16,
        backing.data_ptr(),
        cutlass.AddressSpace.gmem,
        assumed_align=16,
    )
    output_view = from_dlpack(output)
    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)
    fn = cute.compile(encode, ptr, output_view, m, n, k, cfg, stream)
    fn(ptr, output_view, m, n, k, stream)
    assert output.tolist() == [m * k, n * k]
