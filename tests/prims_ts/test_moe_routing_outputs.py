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

import gc

import pytest
import torch
import tvm_ffi

from flashinfer.prims_ts.moe.runner import _torch_views_of_ffi_tensors


@pytest.mark.parametrize("container", [list, tuple, iter])
def test_python_containers_preserve_torch_objects_and_optional_outputs(container):
    values = [torch.arange(6).reshape(2, 3), None, torch.empty(0)]
    actual = _torch_views_of_ffi_tensors(container(values))
    assert all(
        result is original for result, original in zip(actual, values, strict=True)
    )


def test_ffi_array_extracts_once_without_per_element_indexing(monkeypatch):
    values = [torch.arange(4) for _ in range(9)]
    array = tvm_ffi.Array(values)
    original_getitem = tvm_ffi.Array.__getitem__
    original_len = tvm_ffi.Array.__len__
    indices = []
    lengths = []

    def getitem(self, index):
        indices.append(index)
        return original_getitem(self, index)

    def length(self):
        lengths.append(True)
        return original_len(self)

    def unexpected_iteration(self):
        pytest.fail("Routing outputs must avoid FFI Array's per-element iterator")

    monkeypatch.setattr(tvm_ffi.Array, "__getitem__", getitem)
    monkeypatch.setattr(tvm_ffi.Array, "__len__", length)
    monkeypatch.setattr(tvm_ffi.Array, "__iter__", unexpected_iteration)
    actual = _torch_views_of_ffi_tensors(array)
    assert indices == [slice(None)]
    assert len(lengths) == 1
    for result, original in zip(actual, values, strict=True):
        assert result.data_ptr() == original.data_ptr()
        torch.testing.assert_close(result, original)


def test_ffi_array_repeated_conversion_preserves_order_views_and_ownership():
    backing = torch.arange(24).reshape(4, 6)
    view = backing[:, ::2]
    empty = torch.empty(0, dtype=torch.bfloat16)
    values = [view, None, empty, torch.tensor([7], dtype=torch.int32)]
    array = tvm_ffi.Array(values)
    first = _torch_views_of_ffi_tensors(array)
    second = _torch_views_of_ffi_tensors(array)
    assert first[1] is second[1] is None
    assert second[0].data_ptr() == view.data_ptr()
    assert second[0].stride() == view.stride()
    assert second[2].shape == empty.shape
    assert second[2].dtype == empty.dtype
    assert second[3].dtype == torch.int32
    expected = view.clone()
    del backing, view, empty, values, array
    gc.collect()
    torch.testing.assert_close(second[0], expected)
    first[0][0, 0] = -1
    assert second[0][0, 0].item() == -1
    assert second[3].item() == 7


def test_legacy_ffi_tensor_conversion_remains_repeatable():
    original = torch.arange(4)
    value = tvm_ffi.from_dlpack(original)
    first = _torch_views_of_ffi_tensors([value])[0]
    second = _torch_views_of_ffi_tensors([value])[0]
    assert first.data_ptr() == second.data_ptr() == original.data_ptr()
    torch.testing.assert_close(first, second)


def test_empty_ffi_array():
    assert _torch_views_of_ffi_tensors(tvm_ffi.Array([])) == []
