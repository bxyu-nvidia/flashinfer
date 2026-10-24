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

from flashinfer.prims_ts.batched_gemm import batched_gemm_run
from flashinfer.prims_ts.moe import compile_cache


def test_compiled_gemm_cache_is_partitioned_by_device_and_arch(monkeypatch, request):
    compiled = []
    target = [(0, "sm100a")]

    def fake_compile(io, stream):
        result = object()
        compiled.append((io, stream, result))
        return result

    monkeypatch.setattr(compile_cache, "_compile_target_key", lambda _io: target[0])
    monkeypatch.setattr(batched_gemm_run, "_compile_for_launch", fake_compile)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    io = {"cfg": object()}
    assert compile_cache.find_compiled_gemm("cfg", "fc1", io) is None
    assert not compiled
    first = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    assert compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream") is first

    assert compile_cache.find_compiled_gemm("cfg", "fc1", io) is first
    assert compile_cache.find_compiled_gemm("other", "fc1", io) is None
    assert compile_cache.find_compiled_gemm("cfg", "fc2", io) is None
    target[0] = (1, "sm100a")
    assert compile_cache.find_compiled_gemm("cfg", "fc1", io) is None
    second = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")
    target[0] = (1, "sm103a")
    assert compile_cache.find_compiled_gemm("cfg", "fc1", io) is None
    third = compile_cache.get_compiled_gemm("cfg", "fc1", io, "stream")

    assert first is not second
    assert second is not third
    assert len(compiled) == 3
