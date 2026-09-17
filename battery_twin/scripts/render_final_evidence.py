"""Render CSV summaries and faculty-ready figures from evidence_results.json."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=0))


def main() -> int:
    parser = argparse.ArgumentParser(description="Render final battery twin evidence tables and plots")
    parser.add_argument("--input", type=Path, default=Path("outputs/final_evidence_v2/evidence_results.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/final_evidence_v2"))
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    grouped = defaultdict(list)
    for row in result["model_comparison"]:
        grouped[row["model"]].append(row)
    summary_rows = []
    for model, rows in grouped.items():
        soh_mean, soh_std = mean_std([row["soh_mae"] for row in rows])
        rul_mean, rul_std = mean_std([row["rul_mae_cycles"] for row in rows])
        r2_mean, r2_std = mean_std([row["soh_r2"] for row in rows])
        summary_rows.append({"model": model, "soh_mae_mean": soh_mean, "soh_mae_std": soh_std, "rul_mae_mean_cycles": rul_mean, "rul_mae_std_cycles": rul_std, "soh_r2_mean": r2_mean, "soh_r2_std": r2_std})
    with (args.output_dir / "final_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0]))
        writer.writeheader()
        writer.writerows(summary_rows)

    import matplotlib.pyplot as plt
    names = [row["model"] for row in summary_rows]
    soh = [row["soh_mae_mean"] for row in summary_rows]
    soh_err = [row["soh_mae_std"] for row in summary_rows]
    rul = [row["rul_mae_mean_cycles"] for row in summary_rows]
    rul_err = [row["rul_mae_std_cycles"] for row in summary_rows]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].bar(names, soh, yerr=soh_err, capsize=4, color="#1967a3")
    axes[0].set(title="Held-out SOH error", ylabel="MAE (percentage points)")
    axes[1].bar(names, rul, yerr=rul_err, capsize=4, color="#c47d1f")
    axes[1].set(title="Held-out RUL error", ylabel="MAE (cycles)")
    for ax in axes:
        ax.tick_params(axis="x", labelrotation=35)
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(args.output_dir / "final_model_comparison.png", dpi=170)
    plt.close(fig)

    ablations = result["ablations"]
    ab_grouped = defaultdict(list)
    for row in ablations:
        ab_grouped[row["variant"]].append(row)
    ab_names = list(ab_grouped)
    frozen = [np.mean([row["frozen_metrics"]["soh_mae"] for row in ab_grouped[name]]) for name in ab_names]
    adapted = [np.mean([row["adapted_metrics"]["soh_mae"] for row in ab_grouped[name]]) for name in ab_names]
    x = np.arange(len(ab_names))
    fig, ax = plt.subplots(figsize=(11, 4.8))
    ax.bar(x - 0.18, frozen, width=0.36, label="Frozen / before update", color="#8ea1a2")
    ax.bar(x + 0.18, adapted, width=0.36, label="After variant", color="#718c45")
    ax.set_xticks(x, ab_names, rotation=30, ha="right")
    ax.set_ylabel("Post-shift SOH MAE")
    ax.set_title("Adaptation and ablation comparison")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "ablation_comparison.png", dpi=170)
    plt.close(fig)

    robustness = result["robustness"]
    ro_grouped = defaultdict(list)
    for row in robustness:
        ro_grouped[row["scenario"]].append(row)
    ro_names = list(ro_grouped)
    ro_frozen = [np.mean([row["frozen_metrics"]["soh_mae"] for row in ro_grouped[name]]) for name in ro_names]
    ro_adapted = [np.mean([row["adapted_metrics"]["soh_mae"] for row in ro_grouped[name]]) for name in ro_names]
    x = np.arange(len(ro_names))
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x - 0.18, ro_frozen, width=0.36, label="Frozen", color="#8ea1a2")
    ax.bar(x + 0.18, ro_adapted, width=0.36, label="Adaptive replay", color="#1967a3")
    ax.set_xticks(x, ro_names, rotation=25, ha="right")
    ax.set_ylabel("SOH MAE")
    ax.set_title("Robustness under sensor corruption")
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "robustness_comparison.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"summary": str((args.output_dir / 'final_summary.csv').resolve()), "figures": ["final_model_comparison.png", "ablation_comparison.png", "robustness_comparison.png"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
