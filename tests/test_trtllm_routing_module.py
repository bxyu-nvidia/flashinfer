# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""CPU-only checks for the Prims routing module build and dispatch boundary."""

import ast
import functools
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock


ROOT = Path(__file__).resolve().parents[1]


def load_function(relative_path, name, namespace):
    # Exercise the real build/dispatch functions without importing CUDA or DSL.
    tree = ast.parse((ROOT / relative_path).read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name
    )
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), relative_path, "exec"),
        namespace,
    )
    return namespace[name]


def test_routing_build_excludes_gemm_stack(tmp_path):
    fetch = Mock(return_value=b"checksums or header")
    spec = Mock(
        side_effect=lambda name, sources, **kw: SimpleNamespace(
            name=name, sources=sources, **kw
        )
    )
    ns = {
        "JitSpec": object,
        "get_artifact": fetch,
        "get_meta_hash": Mock(return_value="hash"),
        "ArtifactPath": SimpleNamespace(TRTLLM_GEN_BMM="bmm"),
        "CheckSumHash": SimpleNamespace(TRTLLM_GEN_BMM="checksum"),
        "BMM_EXPORT_HEADERS": ["Enums.h", "trtllm/gen/DtypeDecl.h"],
        "jit_env": SimpleNamespace(
            FLASHINFER_CSRC_DIR=ROOT / "csrc", FLASHINFER_CUBIN_DIR=tmp_path
        ),
        "ensure_symlink": Mock(),
        "verify_symlinked_headers": Mock(),
        "current_compilation_context": SimpleNamespace(
            get_nvcc_flags_list=Mock(return_value=["arch-flags"])
        ),
        "gen_jit_spec": spec,
    }
    result = load_function(
        "flashinfer/jit/fused_moe.py", "gen_trtllm_gen_routing_module", ns
    )()
    assert result.name == "trtllm_gen_routing"
    assert all(source.is_file() for source in result.sources)
    names = {source.name for source in result.sources}
    assert {
        "trtllm_moe_staged_routing_binding.cu",
        "trtllm_fused_moe_dev_kernel.cu",
        "trtllm_fused_moe_routing_runner.cu",
    } <= names
    assert (
        not {
            "trtllm_fused_moe_kernel_launcher.cu",
            "trtllm_fused_moe_runner.cu",
            "trtllm_batched_gemm_runner.cu",
            "quantization.cu",
        }
        & names
    )
    assert all(
        call.args[0] == "bmm/checksums.txt" or "/include/" in call.args[0]
        for call in fetch.call_args_list
    )


def test_bf16_dispatch_reuses_standalone_native_module(monkeypatch):
    native = SimpleNamespace(trtllm_gen_routing=Mock())
    build = Mock(return_value=native)
    decorate = Mock(side_effect=lambda *a, **kw: lambda f: f)
    ns = {
        "functools": functools,
        "gen_trtllm_gen_routing_module": lambda: SimpleNamespace(build_and_load=build),
        "register_custom_op": decorate,
        "torch": SimpleNamespace(Tensor=object),
        "Optional": importlib.import_module("typing").Optional,
        "SimpleNamespace": SimpleNamespace,
    }
    getter = load_function(
        "flashinfer/fused_moe/trtllm_gen_routing.py",
        "get_trtllm_gen_routing_module",
        ns,
    )
    # Fake parent packages prevent importing FlashInfer (and hence CUDA).
    for name in ("flashinfer", "flashinfer.fused_moe"):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    module = ModuleType("flashinfer.fused_moe.trtllm_gen_routing")
    module.get_trtllm_gen_routing_module = getter
    monkeypatch.setitem(sys.modules, module.__name__, module)
    core = ModuleType("flashinfer.fused_moe.core")
    core.get_trtllm_moe_sm100_module = Mock(
        side_effect=AssertionError("loaded full GEMM module")
    )
    monkeypatch.setitem(sys.modules, core.__name__, core)
    get_op = load_function(
        "flashinfer/fused_moe/backends/prims_ts/bf16_op.py", "_get_moe_op", {}
    )
    assert get_op() is native
    assert get_op() is native
    build.assert_called_once()
    core.get_trtllm_moe_sm100_module.assert_not_called()
    # The existing public standalone operation still forwards to the same binary.
    getter().trtllm_gen_routing(*range(22))
    native.trtllm_gen_routing.assert_called_once_with(*range(22))
