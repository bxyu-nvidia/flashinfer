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

from types import ModuleType, SimpleNamespace
from unittest.mock import Mock
import sys

import pytest

from flashinfer.prims_ts.moe import config_mapper


@pytest.fixture
def builder(monkeypatch):
    # Schedule construction is CPU work, but importing its implementation needs
    # CUTLASS. Mock that boundary to test cache behavior independently of kernels.
    module_name = "flashinfer.prims_ts.batched_gemm.batched_gemm_kernel"
    module = ModuleType(module_name)
    build = Mock(return_value=SimpleNamespace())
    module.build_batched_gemm_task_manager = build
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.delenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", raising=False)
    config_mapper._validate_task_schedule_cached.cache_clear()
    yield build
    config_mapper._validate_task_schedule_cached.cache_clear()


def validate(kwargs=None, **dimensions):
    config_mapper._ensure_config_task_manager_buildable(
        config_mapper.PrimsTsConfigSpec(
            kwargs if kwargs is not None else {"tile_n": 128, "tile_k": 64}
        ),
        dtype_label="BF16",
        fc="FC1",
        tile_n=128,
        moe_config_index=0,
        **dimensions,
    )


def test_reuses_success_with_reordered_config_and_normalized_dimensions(builder):
    validate()
    validate({"tile_k": 64, "tile_n": 128}, num_tokens=128, top_k=1, num_experts=2)
    builder.assert_called_once_with(
        num_experts=2, num_tokens=128, top_k=1, verbose=False, tile_k=64, tile_n=128
    )


def test_mutated_config_is_revalidated(builder):
    config = {"tile_n": 128, "tile_k": 64}
    validate(config)
    config["tile_k"] = 128
    validate(config)
    assert builder.call_count == 2
    assert builder.call_args.kwargs["tile_k"] == 128


@pytest.mark.parametrize(
    "changed", [{"num_tokens": 256}, {"top_k": 2}, {"num_experts": 4}]
)
def test_dimensions_are_revalidated(builder, changed):
    validate()
    validate(**changed)
    assert builder.call_count == 2


def test_debug_validation_cannot_reuse_production_result(builder, monkeypatch):
    validate()
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "1")
    validate()
    validate()
    assert builder.call_count == 2
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    validate()
    assert builder.call_count == 2


def test_failed_validation_is_retried(builder):
    builder.side_effect = [ValueError("invalid schedule"), SimpleNamespace()]
    with pytest.raises(ValueError, match="invalid schedule"):
        validate()
    validate()
    validate()
    assert builder.call_count == 2


def test_cache_evicts_old_successes(builder):
    cache = config_mapper._validate_task_schedule_cached
    capacity = cache.cache_info().maxsize
    for num_tokens in range(1, capacity + 2):
        validate(num_tokens=num_tokens)
    assert cache.cache_info().currsize == capacity
    validate(num_tokens=1)
    assert builder.call_count == capacity + 2
