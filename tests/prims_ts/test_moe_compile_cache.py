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
    monkeypatch.setenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", "1")
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


def _cache_io(tokens=16, address=0, dtype=None, alignment=16):
    import cutlass
    import cutlass.cute as cute

    ptr = cute.runtime.make_ptr(
        dtype or cutlass.BFloat16,
        address,
        cutlass.AddressSpace.gmem,
        assumed_align=alignment,
    )
    names = (
        "a_dp",
        "b_dp",
        "sfa_dp",
        "sfb_dp",
        "c0_dp",
        "sf_c_dp",
        "tile_idx_dp",
        "route_map_dp",
        "mn_limit_dp",
        "num_non_exiting_ctas_dp",
        "act_dp",
        "per_token_sf_a_dp",
        "per_token_sf_b_dp",
        "bias_dp",
        "scale_c_dp",
        "scale_gate_dp",
        "gemm1_alpha_dp",
        "gemm1_beta_dp",
        "gemm1_clamp_limit_dp",
    )
    return {
        **dict.fromkeys(names, ptr),
        "shape": (128, tokens, 128, 2, tokens),
        "cfg": object(),
        "launch_early_exit_max_token_ctas": tokens,
    }


def test_disk_key_reuses_runtime_shapes_and_addresses():
    first = compile_cache._gemm_disk_key("cfg", "fc1", _cache_io())
    assert compile_cache._gemm_disk_key("cfg", "fc1", _cache_io(3228, 32)) == first
    assert compile_cache._gemm_disk_key("resident_cfg", "fc1", _cache_io()) != first
    assert compile_cache._gemm_disk_key("cfg", "fc2", _cache_io()) != first


def test_disk_key_separates_pointer_abi_and_debug_options(monkeypatch):
    import cutlass

    first = compile_cache._gemm_disk_key("cfg", "fc1", _cache_io())
    assert (
        compile_cache._gemm_disk_key("cfg", "fc1", _cache_io(dtype=cutlass.Float32))
        != first
    )
    assert compile_cache._gemm_disk_key("cfg", "fc1", _cache_io(alignment=4)) != first
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "1")
    assert compile_cache._gemm_disk_key("cfg", "fc1", _cache_io()) != first


def test_native_disk_cache_reloads_without_recompiling(monkeypatch, tmp_path, request):
    from pathlib import Path
    from unittest.mock import Mock
    import cutlass.cute as cute
    from flashinfer.jit import cute_dsl_core

    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    monkeypatch.setattr(compile_cache, "_compile_target_key", lambda _: (0, "sm100a"))
    monkeypatch.setattr(cute_dsl_core, "_get_compile_arch", lambda: "sm100a")
    monkeypatch.setattr(cute_dsl_core, "_get_cute_dsl_version", lambda: "test")
    source = ["source1"]
    monkeypatch.setattr(compile_cache, "_prims_gemm_source_hash", lambda: source[0])

    class NativeKernel:
        def export_to_c(self, directory, name):
            (Path(directory) / f"{name}.o").write_bytes(b"kernel")

    loaded = object()

    class Module:
        def __getattr__(self, name):
            return loaded

    compiler = Mock(return_value=NativeKernel())
    monkeypatch.setattr(batched_gemm_run, "_compile_for_launch", compiler)
    loader = Mock(return_value=Module())
    monkeypatch.setattr(cute.runtime, "load_module", loader)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)

    assert compile_cache.get_compiled_gemm("cfg", "fc1", _cache_io(), None) is loaded
    assert compiler.call_count == 1
    assert len(list(tmp_path.rglob("*.o"))) == 1
    compile_cache._COMPILED_GEMM_CACHE.clear()
    assert (
        compile_cache.get_compiled_gemm("cfg", "fc1", _cache_io(3228, 32), None)
        is loaded
    )
    assert compiler.call_count == 1
    assert loader.call_count >= 2
    assert all(
        call.kwargs.get("enable_tvm_ffi", False) is False
        for call in loader.call_args_list
    )

    source[0] = "source2"
    compile_cache._COMPILED_GEMM_CACHE.clear()
    compile_cache.get_compiled_gemm("cfg", "fc1", _cache_io(), None)
    assert compiler.call_count == 2


def test_native_gemm_disk_reload_uses_current_buffers(monkeypatch, tmp_path, request):
    """A persisted launcher must accept new pointers and token counts without JIT."""
    import pytest
    import torch

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("Prims GEMM requires Blackwell SM100/SM103")
    import cuda.bindings.driver as cuda
    from flashinfer import ActivationType
    from flashinfer.prims_ts.moe.config_mapper import map_trtllm_bf16_moe_tactic

    monkeypatch.setenv("CUTE_DSL_CACHE_DIR", str(tmp_path))
    monkeypatch.delenv("FLASHINFER_CUTE_DSL_DISABLE_CACHE", raising=False)
    compile_cache._COMPILED_GEMM_CACHE.clear()
    request.addfinalizer(compile_cache._COMPILED_GEMM_CACHE.clear)
    pair = map_trtllm_bf16_moe_tactic(
        [128, 2], activation_type=int(ActivationType.Relu2)
    )
    cfg = pair.fc1.cfg.kwargs
    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    raw_compiler = batched_gemm_run._compile_for_launch
    native = []

    def compile_once(io, stream):
        native.append(raw_compiler(io, stream))
        return native[-1]

    monkeypatch.setattr(batched_gemm_run, "_compile_for_launch", compile_once)
    io = batched_gemm_run._build_launch_io(
        num_experts=2, num_tokens=129, top_k=1, problem_n=256, problem_k=128, **cfg
    )
    key = compile_cache.stable_config_hash(io["cfg"])
    first = compile_cache.get_compiled_gemm(key, "fc1", io, stream)
    assert len(native) == 1
    assert len(list(tmp_path.rglob("*.o"))) == 1

    current_io = batched_gemm_run._build_launch_io(
        num_experts=2, num_tokens=131, top_k=1, problem_n=256, problem_k=128, **cfg
    )
    output = next(
        t
        for t in current_io["_keepalive"]
        if t.data_ptr() == current_io["c0_dp"]._pointer
    )
    arguments = batched_gemm_run._launch_arg_tuple(current_io, stream)
    native[0](*arguments)
    torch.cuda.synchronize()
    expected = output.clone()
    assert torch.count_nonzero(expected)
    output.zero_()
    compile_cache._COMPILED_GEMM_CACHE.clear()

    def forbidden(*args):
        pytest.fail("A warmed native artifact must not invoke cute.compile")

    monkeypatch.setattr(batched_gemm_run, "_compile_for_launch", forbidden)
    current_key = compile_cache.stable_config_hash(current_io["cfg"])
    assert key == current_key
    second = compile_cache.get_compiled_gemm(current_key, "fc1", current_io, stream)
    second(*arguments)
    torch.cuda.synchronize()
    assert torch.equal(output, expected)
    assert first is not second
