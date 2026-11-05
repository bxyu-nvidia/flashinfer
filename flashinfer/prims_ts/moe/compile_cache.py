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
Compile cache helpers for Prims-TS batched GEMM launches.
"""

from __future__ import annotations

import functools
import hashlib
import os
from pathlib import Path
import tempfile
import json
from copy import copy
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from flashinfer.jit.cute_dsl_core import (
    JitSpecCuteDsl,
    _get_compile_arch,
    _hash_source_files,
    cute_dsl_cache_disabled,
)
from flashinfer.prims_ts.utils import (
    configure_cute_dsl_cache_dir,
    get_prims_ts_compile_options,
)

from .config_mapper import (
    PrimsTsConfigSpec,
    _ImmutablePrimsTsConfigSpec,
    _build_config_cached,
)


if TYPE_CHECKING:
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import BatchedGemmConfig


@functools.lru_cache(maxsize=8)
def _config_field_names(config_type: type[BatchedGemmConfig]) -> tuple[str, ...]:
    return tuple(field.name for field in fields(config_type) if field.init)


def get_bf16_runtime_config(
    cfg: BatchedGemmConfig | PrimsTsConfigSpec | _ImmutablePrimsTsConfigSpec,
    in_hidden: int,
) -> tuple[BatchedGemmConfig, str]:
    """Reuse scalar BF16 launch configuration and its compilation cache key.

    Tensor pointers, routing metadata, and buffers remain per-call. Copies keep
    kernel assembly's in-place warp-layout updates out of the cached template.
    """
    if isinstance(cfg, _ImmutablePrimsTsConfigSpec):
        template, config_hash = _bf16_spec_runtime_config_cached(
            cfg.config_items, in_hidden
        )
        return _copy_bf16_template(template), config_hash

    if isinstance(cfg, PrimsTsConfigSpec):
        # Use the original options directly: no build/copy followed by a scan
        # of every derived config field just to locate the cached result.
        template, config_hash = _bf16_spec_runtime_config_cached(
            tuple(sorted(cfg.kwargs.items())), in_hidden
        )
        return _copy_bf16_template(template), config_hash

    names = _config_field_names(type(cfg))
    template, config_hash = _bf16_runtime_config_cached(
        type(cfg), tuple(getattr(cfg, name) for name in names), in_hidden
    )
    return copy(template), config_hash


@dataclass(frozen=True, slots=True)
class Bf16LaunchMetadata:
    """Scalar launch properties; mutable kernel configs remain compilation-only."""

    config_hash: str
    is_swap_ab: bool
    has_gated_epilogue: bool
    has_bias_m: bool
    has_gemm1_alpha: bool
    has_gemm1_beta: bool
    has_gemm1_clamp_limit: bool
    use_early_exit: bool
    tile_m: int
    tile_n: int
    cluster_m: int


def get_bf16_launch_metadata(
    cfg: _ImmutablePrimsTsConfigSpec, in_hidden: int
) -> Bf16LaunchMetadata:
    """Return immutable launch metadata without copying a runtime config."""
    return _bf16_launch_metadata_cached(cfg.config_items, in_hidden)


@functools.lru_cache(maxsize=1024)
def _bf16_launch_metadata_cached(
    config_items: tuple[tuple[str, int], ...], in_hidden: int
) -> Bf16LaunchMetadata:
    cfg, key = _bf16_spec_runtime_config_cached(config_items, in_hidden)
    return Bf16LaunchMetadata(
        key,
        cfg.is_swap_ab,
        cfg.has_gated_epilogue,
        cfg.has_bias_m,
        bool(cfg.has_gemm1_alpha),
        bool(cfg.has_gemm1_beta),
        bool(cfg.has_gemm1_clamp_limit),
        bool(cfg.use_early_exit),
        cfg.tile_m,
        cfg.tile_n,
        cfg.cluster_m,
    )


def _copy_bf16_template(template: BatchedGemmConfig) -> BatchedGemmConfig:
    # make_config returns a plain, dict-backed dataclass of scalar fields.
    # Avoid copy.copy's pickle/reconstruction dispatch on every GEMM launch.
    # Keep the instance and its field dictionary private to this invocation.
    cfg = object.__new__(type(template))
    cfg.__dict__ = template.__dict__.copy()
    return cfg


@functools.lru_cache(maxsize=1024)
def _bf16_spec_runtime_config_cached(
    config_items: tuple[tuple[str, int], ...], in_hidden: int
) -> tuple[BatchedGemmConfig, str]:
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _runtime_config

    cfg = _runtime_config(copy(_build_config_cached(config_items)), in_hidden)
    return cfg, stable_config_hash(cfg)


@functools.lru_cache(maxsize=1024)
def _bf16_runtime_config_cached(
    config_type: type[BatchedGemmConfig], config_values: tuple[int, ...], in_hidden: int
) -> tuple[BatchedGemmConfig, str]:
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _runtime_config

    cfg = config_type(
        **dict(zip(_config_field_names(config_type), config_values, strict=True))
    )
    cfg = _runtime_config(cfg, in_hidden)
    return cfg, stable_config_hash(cfg)


def stable_config_hash(cfg: Any, **constexpr_args: Any) -> str:
    """Return a stable cache key for compile-time TS kernel arguments."""
    if is_dataclass(cfg):
        payload = asdict(cfg)
    elif hasattr(cfg, "__dict__"):
        payload = dict(cfg.__dict__)
    else:
        payload = repr(cfg)
    return json.dumps(
        {
            "cfg": payload,
            "constexpr_args": constexpr_args,
        },
        sort_keys=True,
        default=str,
    )


@functools.cache
def get_compile_options() -> str:
    return get_prims_ts_compile_options()


_COMPILED_GEMM_CACHE: dict[tuple[str, str, str, int, str], Any] = {}


def _compile_target_key(io: dict) -> tuple[int, str]:
    """Return the CUDA device and CuTe-DSL architecture for this compilation."""
    import torch

    device_index = None
    for value in io.get("_keepalive", ()):
        if isinstance(value, torch.Tensor) and value.device.type == "cuda":
            device_index = value.device.index
            break
    if device_index is None:
        device_index = torch.cuda.current_device()

    return int(device_index), _get_compile_arch()


def _compiled_gemm_key(cfg_hash: str, fc1_or_fc2: str, io: dict) -> tuple:
    device_index, compile_arch = _compile_target_key(io)
    return (
        cfg_hash,
        fc1_or_fc2,
        get_compile_options(),
        device_index,
        compile_arch,
    )


def find_compiled_gemm(cfg_hash: str, fc1_or_fc2: str, io: dict) -> Any | None:
    """Look up an existing GEMM without requiring routing buffers or compiling."""
    return _COMPILED_GEMM_CACHE.get(_compiled_gemm_key(cfg_hash, fc1_or_fc2, io))


class _PrimsGemmJitSpec(JitSpecCuteDsl):
    """Persist the native pointer/shape ABI without changing hot-path launches."""

    def __init__(self, kernel_name: str, compile_fn: Any, source_sha256: str):
        super().__init__(
            "prims_ts_batched_gemm", kernel_name, compile_fn, source_sha256
        )
        # Image warmup already collects CUTE_DSL_CACHE_DIR, including objects.
        self.module_dir = Path(configure_cute_dsl_cache_dir()) / self.module_dir_name
        self.object_path = self.module_dir / f"{self.kernel_name}.o"

    def _export_compiled_kernel(self, object_path: Path) -> None:
        with tempfile.TemporaryDirectory(
            prefix="tmp.", dir=self.module_dir
        ) as directory:
            self._compiled_kernel.export_to_c(directory, self.symbol)
            os.replace(Path(directory) / f"{self.symbol}.o", object_path)

    def _load_from_disk(self) -> Any:
        import cutlass.cute as cute

        module = cute.runtime.load_module(str(self.object_path))
        return getattr(module, self.symbol)


@functools.cache
def _prims_gemm_source_hash() -> str:
    from flashinfer.jit import cute_dsl_core

    sources = tuple(Path(__file__).resolve().parents[1].rglob("*.py"))
    return _hash_source_files((*sources, cute_dsl_core.__file__))


def _gemm_argument_type(value: Any) -> Any:
    if isinstance(value, (tuple, list)):
        return [type(value).__name__, [_gemm_argument_type(v) for v in value]]
    if value is None:
        return None
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int32" if -(1 << 31) <= value < (1 << 31) else "int64"
    if isinstance(value, float):
        return "float32"
    if hasattr(value, "__get_mlir_types__"):
        return [str(t) for t in value.__get_mlir_types__()]
    raise TypeError(f"Unsupported Prims GEMM cache argument type: {type(value)}")


def _gemm_disk_key(cfg_hash: str, fc: str, io: dict) -> str:
    from cutlass._mlir import ir
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _compile_arg_tuple,
        _generate_line_info_enabled,
    )

    # cfg is covered by cfg_hash; stream addresses and runtime values are not
    # specializations. Pointer dtype/address space/alignment remain part of the ABI.
    with ir.Context(), ir.Location.unknown():
        arguments = [_gemm_argument_type(v) for v in _compile_arg_tuple(io, None)[:-2]]
    key = json.dumps(
        (cfg_hash, fc, get_compile_options(), _generate_line_info_enabled(), arguments),
        sort_keys=True,
    )
    return hashlib.sha256(key.encode()).hexdigest()


def get_compiled_gemm(cfg_hash: str, fc1_or_fc2: str, io: dict, stream: Any) -> Any:
    """Compile and cache a TS GEMM for the supplied launch IO.

    ``cfg_hash`` includes only the normalized config and other constexpr
    operands. Runtime problem shape and IO objects are passed to
    ``cute.compile`` to establish argument types, but the generated kernel body
    does not depend on their values and they must not be part of the cache key.
    """

    key = _compiled_gemm_key(cfg_hash, fc1_or_fc2, io)
    if key in _COMPILED_GEMM_CACHE:
        return _COMPILED_GEMM_CACHE[key]
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _compile_for_launch

    if cute_dsl_cache_disabled():
        compiled = _compile_for_launch(io, stream)
    else:
        spec = _PrimsGemmJitSpec(
            _gemm_disk_key(cfg_hash, fc1_or_fc2, io),
            lambda: _compile_for_launch(io, stream),
            _prims_gemm_source_hash(),
        )
        compiled = spec.build_and_load()
    _COMPILED_GEMM_CACHE[key] = compiled
    return compiled
