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
import json
from copy import copy
from dataclasses import asdict, fields, is_dataclass
from typing import TYPE_CHECKING, Any

from flashinfer.jit.cute_dsl_core import _get_compile_arch
from flashinfer.prims_ts.utils import get_prims_ts_compile_options

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


def get_compiled_gemm(cfg_hash: str, fc1_or_fc2: str, io: dict, stream: Any) -> Any:
    """Compile and cache a TS GEMM for the supplied launch IO.

    ``cfg_hash`` includes only the normalized config and other constexpr
    operands. Runtime problem shape and IO objects are passed to
    ``cute.compile`` to establish argument types, but the generated kernel body
    does not depend on their values and they must not be part of the cache key.
    """

    device_index, compile_arch = _compile_target_key(io)
    key = (
        cfg_hash,
        fc1_or_fc2,
        get_compile_options(),
        device_index,
        compile_arch,
    )
    if key in _COMPILED_GEMM_CACHE:
        return _COMPILED_GEMM_CACHE[key]
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _compile_for_launch

    compiled = _compile_for_launch(io, stream)
    _COMPILED_GEMM_CACHE[key] = compiled
    return compiled
