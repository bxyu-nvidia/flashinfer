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

import pytest
import torch

from flashinfer.jit import cute_dsl_core


@pytest.fixture(autouse=True)
def clear_arch_cache(monkeypatch):
    monkeypatch.delenv("CUTE_DSL_ARCH", raising=False)
    cute_dsl_core._get_compile_arch_cached.cache_clear()
    yield
    cute_dsl_core._get_compile_arch_cached.cache_clear()


def test_arch_cache_tracks_current_device(monkeypatch):
    current = [0]
    queries = []

    def capability(device):
        queries.append(device)
        return [(10, 0), (9, 0), (8, 0)][device]

    monkeypatch.setattr(torch.cuda, "current_device", lambda: current[0])
    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    for device, expected in [(0, "sm100a"), (1, "sm90a"), (2, "sm80"), (0, "sm100a")]:
        current[0] = device
        assert cute_dsl_core._get_compile_arch() == expected
        assert cute_dsl_core._get_compile_arch() == expected
    assert queries == [0, 1, 2]


def test_arch_override_changes_without_cuda(monkeypatch):
    def unexpected_cuda_call(*args):
        pytest.fail("An explicit architecture must not query CUDA")

    monkeypatch.setattr(torch.cuda, "current_device", unexpected_cuda_call)
    monkeypatch.setattr(torch.cuda, "get_device_capability", unexpected_cuda_call)
    normalized = []
    sanitize = cute_dsl_core.sanitize_symbol_name

    def record_normalization(arch):
        normalized.append(arch)
        return sanitize(arch)

    monkeypatch.setattr(cute_dsl_core, "sanitize_symbol_name", record_normalization)
    for arch, expected in [
        ("sm_100a", "sm100a"),
        ("sm_103a", "sm103a"),
        ("sm_100a", "sm100a"),
    ]:
        monkeypatch.setenv("CUTE_DSL_ARCH", arch)
        assert cute_dsl_core._get_compile_arch() == expected
        assert cute_dsl_core._get_compile_arch() == expected
    assert normalized == ["sm100a", "sm103a"]


@pytest.mark.parametrize("reset", [None, ""])
def test_removing_override_restores_device_target(monkeypatch, reset):
    queries = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def capability(device):
        queries.append(device)
        return (10, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    assert cute_dsl_core._get_compile_arch() == "sm100a"
    monkeypatch.setenv("CUTE_DSL_ARCH", "sm_103a")
    assert cute_dsl_core._get_compile_arch() == "sm103a"
    if reset is None:
        monkeypatch.delenv("CUTE_DSL_ARCH")
    else:
        monkeypatch.setenv("CUTE_DSL_ARCH", reset)
    assert cute_dsl_core._get_compile_arch() == "sm100a"
    assert queries == [0]


def test_failed_detection_is_retried(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)

    def capability(device):
        calls.append(device)
        if len(calls) == 1:
            raise RuntimeError("Device unavailable")
        return (10, 0)

    monkeypatch.setattr(torch.cuda, "get_device_capability", capability)
    with pytest.raises(RuntimeError, match="Device unavailable"):
        cute_dsl_core._get_compile_arch()
    assert cute_dsl_core._get_compile_arch() == "sm100a"
    assert calls == [0, 0]
