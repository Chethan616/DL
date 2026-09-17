"""Run the NASA-first model comparison and adaptive proof of concept."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery_twin.config import TwinConfig
from battery_twin.data import CycleRecord, assign_soh_rul, audit_records, load_nasa_directory
from battery_twin.features import FeatureStats, make_window_samples
from battery_twin.metrics import metric_summary
from battery_twin.models import build_model
from battery_twin.reporting import plot_dataset_audit, plot_model_comparison
from battery_twin.training import adapt_stream, evaluate_model, fit_model, predict_samples, seed_everything


def synthetic_records(seed: int = 7) -> dict[str, list[CycleRecord]]:
    """Create a smoke-test corpus with NASA-like fields and controlled drift."""

    rng = np.random.default_rng(seed)
    result: dict[str, list[CycleRecord]] = {}
    for cell_index, cell_id in enumerate(("B0005", "B0006", "B0007", "B0018")):
        records = []
        cycles = 52 + cell_index * 3
        for cycle in range(1, cycles + 1):
            points = 80 + (cycle * 7) % 30
            timestamp = np.cumsum(rng.uniform(0.8, 1.2, points))
            t = np.linspace(0, 1, points)
            degradation = 0.0016 * cycle * (1.0 + 0.08 * cell_index)
            regeneration = 0.006 * np.exp(-((cycle - 14) / 5) ** 2)
            capacity = 2.0 * (1.0 - degradation + regeneration) + rng.normal(0, 0.004)
            voltage = 4.15 - 0.68 * t - 0.0008 * cycle + rng.normal(0, 0.008, points)
            current = -2.0 + 0.05 * np.sin(8 * t) + rng.normal(0, 0.015, points)
            temperature = 24.0 + 2.0 * np.sin(np.pi * t) + 0.01 * cycle + rng.normal(0, 0.05, points)
            mask = np.ones(points, dtype=bool)
            if cycle % 11 == 0:
                mask[::17] = False
            records.append(CycleRecord(cell_id, cycle, "discharge", timestamp, voltage, current, temperature, capacity, np.diff(timestamp, prepend=timestamp[0]), mask))
        result[cell_id] = records
    return assign_soh_rul(result)


def _save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=float), encoding="utf-8")


def _make_model(config: TwinConfig, model_name: str):
    return build_model(
        model_name,
        input_features=5,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_transformer_layers,
        dropout=config.dropout,
        rul_scale=config.rul_scale,
    )


def _adaptation_metrics(model, train_samples, initial_samples, stream_samples, config, device, epochs):
    frozen = predict_samples(model, stream_samples, device)
    adaptive_model = copy.deepcopy(model)
    adaptation = adapt_stream(adaptive_model, initial_samples, stream_samples, config, device)
    soh_after = np.asarray([item["soh"] for item in adaptation["predictions_after"]], dtype=float)
    rul_after = np.asarray([item["rul"] for item in adaptation["predictions_after"]], dtype=float)
    adaptive_metrics = metric_summary(frozen["soh_true"], soh_after, frozen["rul_true"], rul_after)
    frozen_metrics = metric_summary(frozen["soh_true"], frozen["soh"], frozen["rul_true"], frozen["rul"])
    full_retrain_model = copy.deepcopy(model)
    full_retrain_history = fit_model(
        full_retrain_model,
        train_samples + initial_samples + stream_samples,
        config,
        use_physics=True,
        epochs=max(1, epochs),
        device=device,
    )
    full_retrain = predict_samples(full_retrain_model, stream_samples, device)
    full_retrain_metrics = metric_summary(
        full_retrain["soh_true"], full_retrain["soh"], full_retrain["rul_true"], full_retrain["rul"]
    )
    return {
        "frozen": frozen_metrics,
        "full_retrain": full_retrain_metrics,
        "adaptive_replay": adaptive_metrics,
        "events": adaptation["events"],
        "drift_events": adaptation["drift_events"],
        "full_retrain_epochs": len(full_retrain_history),
        "adaptive_update_steps": int(adaptation["drift_events"] * config.adaptation_steps),
    }


def run_once(records_by_cell, config: TwinConfig, epochs: int, seed: int, device: str):
    seed_everything(seed)
    cells = list(records_by_cell)
    holdout = cells[-1]
    train_records = {cell: records_by_cell[cell] for cell in cells[:-1]}
    test_records = {holdout: records_by_cell[holdout]}
    stats = FeatureStats.fit(record for records in train_records.values() for record in records)
    train_samples = make_window_samples(train_records, stats, config.history_cycles, config.max_points_per_cycle)
    test_samples = make_window_samples(test_records, stats, config.history_cycles, config.max_points_per_cycle)
    initial_count = max(config.drift_warmup, len(test_samples) // 4)
    initial_samples, stream_samples = test_samples[:initial_count], test_samples[initial_count:]

    model_rows = []
    for model_name in ("dnn", "gru", "transformer", "pinn", "adaptive"):
        model = _make_model(config, model_name)
        use_physics = model_name in {"pinn", "adaptive"}
        history = fit_model(model, train_samples, config, use_physics=use_physics, epochs=epochs, device=device)
        metrics = evaluate_model(model, test_samples, device)
        row = {"model": model_name, "seed": seed, "use_physics": use_physics, "metrics": metrics, "final_training": history[-1]}
        if model_name == "adaptive":
            row["online_comparison"] = _adaptation_metrics(model, train_samples, initial_samples, stream_samples, config, device, epochs)
        model_rows.append(row)
    return {
        "seed": seed,
        "holdout_cell": holdout,
        "train_cells": cells[:-1],
        "train_windows": len(train_samples),
        "test_windows": len(test_samples),
        "models": model_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the adaptive battery twin proof of concept")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "nasa_poc")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke-test", action="store_true")
    args = parser.parse_args()
    if args.smoke_test:
        records = synthetic_records()
        status = "synthetic_smoke_only"
    else:
        try:
            records = load_nasa_directory(args.data_dir)
        except FileNotFoundError as exc:
            print(str(exc))
            print("Download the official archive first: python scripts/download_nasa.py --extract")
            return 2
        assign_soh_rul(records)
        status = "nasa_pcoe" if args.epochs >= 40 and len(args.seeds) >= 3 else "nasa_pcoe_preliminary"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = TwinConfig(epochs=args.epochs)
    _save_json(args.output_dir / "dataset_audit.json", {
        "status": status,
        "protocol": {
            "eol_soh": config.eol_soh,
            "nominal_capacity_ah": config.nominal_capacity_ah,
            "epochs": args.epochs,
            "seeds": args.seeds,
            "holdout_rule": "last cell held out; B0018 for the default four-cell order",
        },
        **audit_records(records),
    })
    plot_dataset_audit(records, args.output_dir)
    runs = [run_once(records, config, args.epochs, seed, args.device) for seed in args.seeds]
    _save_json(args.output_dir / "model_metrics.json", {"status": status, "runs": runs})
    plot_model_comparison(runs, args.output_dir)
    print(json.dumps({"status": status, "output_dir": str(args.output_dir.resolve()), "runs": len(runs)}, indent=2))
    if status == "synthetic_smoke_only":
        print("Smoke output is only a pipeline check; do not cite it as NASA evidence.")
    elif status == "nasa_pcoe_preliminary":
        print("Preliminary NASA output: use the full 3-seed, tuned protocol before making final claims.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
