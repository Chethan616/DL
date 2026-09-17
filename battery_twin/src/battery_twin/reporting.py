"""Small, reproducible figures for the faculty evidence package."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def plot_dataset_audit(records_by_cell, output_dir: str | Path) -> list[str]:
    """Write capacity, sensor-range, and missingness plots from real records."""

    import matplotlib.pyplot as plt

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    fig, ax = plt.subplots(figsize=(9, 5))
    for cell, records in records_by_cell.items():
        records = sorted(records, key=lambda record: record.cycle_id)
        ax.plot([r.cycle_id for r in records], [r.soh for r in records], marker=".", linewidth=1.2, label=cell)
    ax.axhline(70.0, color="#b64e3f", linestyle="--", linewidth=1, label="EOL = 70% SOH")
    ax.set(xlabel="Discharge cycle", ylabel="SOH (%)", title="NASA-first capacity degradation")
    ax.grid(alpha=0.22)
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    path = output_dir / "capacity_fade.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    temperatures = np.concatenate([r.temperature[r.mask] for rs in records_by_cell.values() for r in rs])
    currents = np.concatenate([r.current[r.mask] for rs in records_by_cell.values() for r in rs])
    axes[0].hist(temperatures, bins=30, color="#1967a3", alpha=0.84)
    axes[0].set(xlabel="Temperature", ylabel="Samples", title="Temperature distribution")
    axes[1].hist(currents, bins=30, color="#c47d1f", alpha=0.84)
    axes[1].set(xlabel="Current", ylabel="Samples", title="Current distribution")
    fig.tight_layout()
    path = output_dir / "sensor_distributions.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    paths.append(str(path))
    return paths


def plot_model_comparison(runs: list[dict], output_dir: str | Path) -> str | None:
    """Plot mean SOH/RUL MAE by model when model metrics are available."""

    import matplotlib.pyplot as plt

    if not runs:
        return None
    grouped: dict[str, list[dict]] = {}
    for run in runs:
        for row in run.get("models", []):
            grouped.setdefault(row["model"], []).append(row["metrics"])
    if not grouped:
        return None
    names = list(grouped)
    soh = [np.mean([item["soh_mae"] for item in grouped[name]]) for name in names]
    rul = [np.mean([item["rul_mae_cycles"] for item in grouped[name]]) for name in names]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].bar(names, soh, color="#1967a3")
    axes[0].set_ylabel("SOH MAE (percentage points)")
    axes[0].set_title("Held-out SOH error")
    axes[1].bar(names, rul, color="#c47d1f")
    axes[1].set_ylabel("RUL MAE (cycles)")
    axes[1].set_title("Held-out RUL error")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    path = Path(output_dir) / "model_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return str(path)
