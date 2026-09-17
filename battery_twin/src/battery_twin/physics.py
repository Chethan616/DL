"""Physics-informed and consistency losses supported by the first POC."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype)
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def physics_losses(
    soh_pred: torch.Tensor,
    rul_pred: torch.Tensor,
    soh_true: torch.Tensor,
    rul_true: torch.Tensor,
    cycle_mask: torch.Tensor,
    eol_soh: float = 70.0,
    rul_scale: float = 250.0,
) -> dict[str, torch.Tensor]:
    """Return auditable data, physics, and consistency components.

    The monotonic terms are soft because capacity regeneration can appear after
    rest. The curve term compares predicted and measured long-term differences;
    it is a shape constraint, not a substitute for held-out testing.
    """

    valid = cycle_mask.bool()
    data_soh = _masked_mean((soh_pred - soh_true).abs(), valid)
    data_rul = _masked_mean(((rul_pred - rul_true) / float(rul_scale)).abs(), valid)
    l_data = data_soh + data_rul

    pair_mask = valid[:, 1:] & valid[:, :-1]
    soh_step = soh_pred[:, 1:] - soh_pred[:, :-1]
    rul_step = rul_pred[:, 1:] - rul_pred[:, :-1]
    true_soh_step = soh_true[:, 1:] - soh_true[:, :-1]
    soh_monotonic = _masked_mean(torch.relu(soh_step), pair_mask)
    rul_monotonic = _masked_mean(torch.relu(rul_step), pair_mask)
    curve_residual = _masked_mean((soh_step - true_soh_step).abs(), pair_mask) / 100.0
    nonnegative_consistency = _masked_mean(torch.relu(torch.as_tensor(eol_soh, device=soh_pred.device) - soh_pred) * 0.0, valid)
    # `nonnegative_consistency` is kept as an explicit zero-valued hook: the
    # bounded head already enforces SOH in [0,100], while a future EOL-aware
    # RUL/SOH relation can be added without changing the trainer interface.
    l_phys = soh_monotonic + rul_monotonic
    l_cons = curve_residual + nonnegative_consistency
    return {
        "data": l_data,
        "physics": l_phys,
        "consistency": l_cons,
        "soh_monotonic": soh_monotonic.detach(),
        "rul_monotonic": rul_monotonic.detach(),
        "curve_residual": curve_residual.detach(),
    }


def total_loss(losses: dict[str, torch.Tensor], physics_weight: float, consistency_weight: float) -> torch.Tensor:
    return losses["data"] + physics_weight * losses["physics"] + consistency_weight * losses["consistency"]


class LossTracker:
    """Aggregate scalar losses for reproducible training logs."""

    def __init__(self):
        self.values: dict[str, list[float]] = {}

    def add(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            self.values.setdefault(key, []).append(float(value))

    def mean(self) -> dict[str, float]:
        return {key: sum(values) / max(len(values), 1) for key, values in self.values.items()}
