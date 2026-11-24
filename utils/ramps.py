# Copyright (c) 2018, Curious AI Ltd. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Functions for ramping hyperparameters up or down

Each function takes the current training step or epoch, and the
ramp length in the same format, and returns a multiplier between
0 and 1.
"""


import numpy as np


def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:               
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length


def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))


def resolve_rampup_unit(unit, rampup_length, default_unit="auto"):
    """Resolve explicit 'iter' | 'epoch' rampup unit with backward-compatible fallback."""
    if unit is not None and str(unit).strip() != "":
        normalized = str(unit).strip().lower()
        if normalized not in ("iter", "epoch"):
            raise ValueError(f"consistency_rampup_unit must be 'iter' or 'epoch', got {unit!r}")
        return normalized
    if default_unit in ("iter", "epoch"):
        return default_unit
    # Legacy heuristic preserved only when explicit consistency_rampup_unit is omitted
    return "iter" if float(rampup_length) > 100.0 else "epoch"


def compute_rampup_factor(iter_num, epoch, rampup_length, unit="epoch"):
    """Evaluate sigmoid_rampup on the explicitly configured axis ('iter' or 'epoch')."""
    if float(rampup_length) <= 0.0:
        return 1.0
    resolved_unit = resolve_rampup_unit(unit, rampup_length)
    axis_val = float(iter_num) if resolved_unit == "iter" else float(epoch)
    return sigmoid_rampup(axis_val, float(rampup_length))
