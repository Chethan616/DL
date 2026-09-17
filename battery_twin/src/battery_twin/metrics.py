"""Metrics and physical-validity checks."""

from __future__ import annotations

import numpy as np


def mae(y_true, y_pred) -> float:
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mape(y_true, y_pred, epsilon: float = 1e-6) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs((true - pred) / np.maximum(np.abs(true), epsilon))) * 100.0)


def r2(y_true, y_pred) -> float:
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum((true - np.mean(true)) ** 2)
    return float(1.0 - np.sum((true - pred) ** 2) / denominator) if denominator > 1e-12 else 0.0


def degradation_violation_rate(soh_sequence) -> float:
    """Fraction of adjacent valid pairs where SOH increases."""
    values = np.asarray(soh_sequence, dtype=float)
    if values.size < 2:
        return 0.0
    return float(np.mean(np.diff(values) > 0.0))


def capacity_curve_residual(soh_true, soh_pred) -> float:
    """Normalized residual between predicted and measured SOH changes.

    This is the auditable first-stage capacity-curve residual used in the
    project POC. It is deliberately data-supported and does not claim to be
    an electrochemical SEI residual.
    """
    true = np.asarray(soh_true, dtype=float)
    pred = np.asarray(soh_pred, dtype=float)
    if true.size < 2 or pred.size < 2:
        return 0.0
    length = min(true.size, pred.size)
    return float(np.mean(np.abs(np.diff(pred[:length]) - np.diff(true[:length]))) / 100.0)


def soh_rul_consistency_violation_rate(soh_pred, rul_pred) -> float:
    """Fraction of adjacent predictions violating long-term degradation order.

    In the declared cycle order both SOH and RUL should not increase. This is
    a diagnostic rate, not a hard training constraint, because real batteries
    can show local capacity-regeneration effects.
    """
    soh = np.asarray(soh_pred, dtype=float)
    rul = np.asarray(rul_pred, dtype=float)
    length = min(soh.size, rul.size)
    if length < 2:
        return 0.0
    return float(np.mean((np.diff(soh[:length]) > 0.0) | (np.diff(rul[:length]) > 0.0)))


def metric_summary(soh_true, soh_pred, rul_true, rul_pred) -> dict[str, float]:
    return {
        "soh_mae": mae(soh_true, soh_pred),
        "soh_rmse": rmse(soh_true, soh_pred),
        "soh_mape_percent": mape(soh_true, soh_pred),
        "soh_r2": r2(soh_true, soh_pred),
        "rul_mae_cycles": mae(rul_true, rul_pred),
        "rul_rmse_cycles": rmse(rul_true, rul_pred),
        "soh_bounds_violation_rate": float(np.mean((np.asarray(soh_pred) < 0) | (np.asarray(soh_pred) > 100))),
        "rul_negative_rate": float(np.mean(np.asarray(rul_pred) < 0)),
        "soh_degradation_violation_rate": degradation_violation_rate(soh_pred),
        "soh_rul_consistency_violation_rate": soh_rul_consistency_violation_rate(soh_pred, rul_pred),
        "capacity_curve_residual": capacity_curve_residual(soh_true, soh_pred),
    }
