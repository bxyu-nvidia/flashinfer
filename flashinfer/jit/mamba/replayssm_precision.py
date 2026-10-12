"""Opt-in CUDA precision controls for ReplaySSM accuracy experiments.

Copyright (c) 2026 by FlashInfer team. Licensed under the Apache License, Version 2.0.
"""

import functools
import os
import warnings

import torch

PRECISION_MODES = {
    "native": -1,
    "bf16": 0,
    "readout": 1,
    "reconstruction": 2,
    "readout-reconstruction": 3,
    "cb": 4,
    "readout-cb": 5,
    "reconstruction-cb": 6,
    "all": 7,
}


@functools.cache
def replayssm_precision_mask():
    """Freeze the precision setting at first use; change it in a new process."""
    mode = os.environ.get("FLASHINFER_REPLAYSSM_PRECISION", "native")
    if mode not in PRECISION_MODES:
        raise ValueError(
            f"Invalid FLASHINFER_REPLAYSSM_PRECISION={mode!r}; "
            f"choose one of {tuple(PRECISION_MODES)}"
        )
    mask = PRECISION_MODES[mode]
    if mask >= 0:
        warnings.warn(
            f"ReplaySSM precision diagnostic {mode!r}: using the slower SIMT CUDA "
            "path for both monolith and two-kernel requests, including prefix "
            "materialization. Not a throughput-optimized kernel.",
            stacklevel=2,
        )
    return mask


def precision_suffix():
    mask = replayssm_precision_mask()
    return f"_precision_v1_m{mask}" if mask >= 0 else ""


def precision_flags(state_dtype, input_dtype, dim, dstate):
    mask = replayssm_precision_mask()
    if mask < 0:
        return []
    if (
        state_dtype not in (torch.float16, torch.float32)
        or input_dtype != torch.bfloat16
    ):
        raise ValueError(
            "ReplaySSM precision diagnostic needs FP16/FP32 state and BF16 inputs"
        )
    if dim % 4 or dstate % 64:
        raise ValueError(
            "ReplaySSM precision diagnostic needs dim % 4 == dstate % 64 == 0"
        )
    return [f"-DFLASHINFER_REPLAYSSM_PRECISION_MASK={mask}"]
