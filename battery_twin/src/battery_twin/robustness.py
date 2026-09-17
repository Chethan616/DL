"""Controlled sensor corruption for the required robustness experiments."""

from __future__ import annotations

import copy

import numpy as np

from .features import WindowSample


def corrupt_samples(
    samples: list[WindowSample],
    noise_std: float = 0.0,
    missing_rate: float = 0.0,
    drift_per_cycle: float = 0.0,
    seed: int = 7,
) -> list[WindowSample]:
    """Return corrupted copies without mutating the original evidence set.

    Noise and drift are applied to the normalized voltage/current/temperature
    channels only. Missing samples are represented by a zeroed value and a
    zero point mask, preserving the model's irregular-time contract.
    """

    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0, 1)")
    rng = np.random.default_rng(seed)
    output = []
    for sample in samples:
        item = copy.deepcopy(sample)
        active = item.point_mask.astype(bool)
        if noise_std:
            noise = rng.normal(0.0, noise_std, size=item.x[:, :, :3].shape).astype(np.float32)
            item.x[:, :, :3] += noise * item.point_mask[:, :, None]
        if drift_per_cycle:
            cycle_positions = np.arange(item.x.shape[0], dtype=np.float32)[:, None, None]
            item.x[:, :, :3] += drift_per_cycle * cycle_positions * item.point_mask[:, :, None]
        if missing_rate:
            drop = (rng.random(item.point_mask.shape) < missing_rate) & active
            item.point_mask[drop] = 0.0
            item.x[drop] = 0.0
        output.append(item)
    return output
