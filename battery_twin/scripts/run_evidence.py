"""Run the final comparison, ablations, robustness, and target checks."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery_twin.config import TwinConfig
from battery_twin.data import assign_soh_rul, load_nasa_directory
from battery_twin.features import FeatureStats, make_window_samples
from battery_twin.metrics import metric_summary
from battery_twin.models import build_model
from battery_twin.robustness import corrupt_samples
from battery_twin.training import adapt_stream, evaluate_model, fit_model, predict_samples, seed_everything


MODEL_NAMES = ("dnn", "gru", "transformer", "pinn", "adaptive")


def model_from_config(name: str, config: TwinConfig):
    return build_model(
        name,
        input_features=5,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_transformer_layers,
        dropout=config.dropout,
        rul_scale=config.rul_scale,
    )


def stream_metrics(model, stream_samples, device):
    pred = predict_samples(model, stream_samples, device)
    return metric_summary(pred["soh_true"], pred["soh"], pred["rul_true"], pred["rul"])


def online_variant(
    base_model,
    initial_samples,
    stream_samples,
    config,
    device,
    *,
    train_size=0,
    use_physics=True,
    use_replay=True,
    use_drift=True,
):
    model = copy.deepcopy(base_model)
    before = stream_metrics(model, stream_samples, device)
    initial_before = stream_metrics(model, initial_samples, device)
    started = time.perf_counter()
    adaptation = adapt_stream(
        model,
        initial_samples,
        stream_samples,
        config,
        device,
        use_physics=use_physics,
        use_replay=use_replay,
        use_drift=use_drift,
        fixed_update_every=8,
    )
    adaptation_seconds = time.perf_counter() - started
    after = stream_metrics(model, stream_samples, device)
    initial_after = stream_metrics(model, initial_samples, device)
    improvement = (before["soh_mae"] - after["soh_mae"]) / max(before["soh_mae"], 1e-8)
    forgetting = (initial_after["soh_mae"] - initial_before["soh_mae"]) / max(initial_before["soh_mae"], 1e-8)
    full_steps = max(1, config.epochs * int(np.ceil((train_size + len(initial_samples) + len(stream_samples)) / config.batch_size)))
    update_steps = adaptation["drift_events"] * config.adaptation_steps
    return {
        "frozen_metrics": before,
        "adapted_metrics": after,
        "improvement_fraction": improvement,
        "prior_domain_forgetting_fraction": forgetting,
        "adaptation_seconds": adaptation_seconds,
        "adaptation_update_steps": int(update_steps),
        "estimated_full_retrain_steps": int(full_steps),
        "work_ratio": float(update_steps / full_steps),
        "drift_events": int(adaptation["drift_events"]),
        "events": adaptation["events"],
    }


def load_best_config(grid_path: Path, epochs: int) -> tuple[TwinConfig, dict]:
    if not grid_path.exists():
        return TwinConfig(epochs=epochs), {"source": "default; tuning file not found"}
    grid = json.loads(grid_path.read_text(encoding="utf-8"))
    values = grid.get("best_hyperparameters", {})
    config = TwinConfig(**values, epochs=epochs)
    return config, {"source": str(grid_path), "best_hyperparameters": values, "grid_combinations_run": grid.get("combinations_run")}


def mean_std(rows: list[dict], key: str) -> dict[str, float]:
    values = np.asarray([row[key] for row in rows], dtype=float)
    return {"mean": float(values.mean()), "std": float(values.std(ddof=0))}


def main() -> int:
    parser = argparse.ArgumentParser(description="Run final faculty evidence experiments")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--grid", type=Path, default=ROOT / "outputs" / "tuning" / "grid.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "final_evidence")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    records = assign_soh_rul(load_nasa_directory(args.data_dir))
    cells = list(records)
    holdout = cells[-1]
    train_records = {cell: records[cell] for cell in cells[:-1]}
    test_records = {holdout: records[holdout]}
    config, config_meta = load_best_config(args.grid, args.epochs)
    stats = FeatureStats.fit(record for values in train_records.values() for record in values)
    train_samples = make_window_samples(train_records, stats, config.history_cycles, config.max_points_per_cycle)
    test_samples = make_window_samples(test_records, stats, config.history_cycles, config.max_points_per_cycle)
    initial_count = max(config.drift_warmup, len(test_samples) // 4)
    initial_samples, stream_samples = test_samples[:initial_count], test_samples[initial_count:]

    comparison_rows = []
    ablation_rows = []
    robustness_rows = []
    for seed in args.seeds:
        seed_everything(seed)
        trained = {}
        for name in MODEL_NAMES:
            model = model_from_config(name, config)
            history = fit_model(model, train_samples, config, use_physics=name in {"pinn", "adaptive"}, epochs=args.epochs, device=args.device)
            trained[name] = model
            comparison_rows.append({"seed": seed, "model": name, **evaluate_model(model, test_samples, args.device), "epochs": args.epochs})

        no_physics_base = model_from_config("adaptive", config)
        fit_model(no_physics_base, train_samples, config, use_physics=False, epochs=args.epochs, device=args.device)
        trained["no_physics_base"] = no_physics_base

        proposed = trained["adaptive"]
        for variant, base_name, kwargs in (
            ("proposed_adaptive", "adaptive", {"use_physics": True, "use_replay": True, "use_drift": True}),
            ("no_physics", "no_physics_base", {"use_physics": False, "use_replay": True, "use_drift": True}),
            ("no_transformer", "dnn", {"use_physics": True, "use_replay": True, "use_drift": True}),
            ("no_replay", "adaptive", {"use_physics": True, "use_replay": False, "use_drift": True}),
            ("no_drift_detection", "adaptive", {"use_physics": True, "use_replay": True, "use_drift": False}),
        ):
            result = online_variant(trained[base_name], initial_samples, stream_samples, config, args.device, train_size=len(train_samples), **kwargs)
            ablation_rows.append({"seed": seed, "variant": variant, **{key: value for key, value in result.items() if key != "events"}, "events": result["events"]})

        full_model = copy.deepcopy(proposed)
        started = time.perf_counter()
        fit_model(full_model, train_samples + initial_samples + stream_samples, config, use_physics=True, epochs=args.epochs, device=args.device)
        full_seconds = time.perf_counter() - started
        full_metrics = stream_metrics(full_model, stream_samples, args.device)
        ablation_rows.append({
            "seed": seed,
            "variant": "full_retraining",
            "frozen_metrics": stream_metrics(proposed, stream_samples, args.device),
            "adapted_metrics": full_metrics,
            "improvement_fraction": 0.0,
            "prior_domain_forgetting_fraction": 0.0,
            "adaptation_seconds": full_seconds,
            "adaptation_update_steps": config.epochs * int(np.ceil(len(train_samples + initial_samples + stream_samples) / config.batch_size)),
            "estimated_full_retrain_steps": config.epochs * int(np.ceil(len(train_samples + initial_samples + stream_samples) / config.batch_size)),
            "work_ratio": 1.0,
            "drift_events": 0,
            "events": [],
        })

        scenarios = {
            "clean": {"noise_std": 0.0, "missing_rate": 0.0, "drift_per_cycle": 0.0},
            "voltage_current_temperature_noise": {"noise_std": 0.05, "missing_rate": 0.0, "drift_per_cycle": 0.0},
            "missing_samples": {"noise_std": 0.0, "missing_rate": 0.10, "drift_per_cycle": 0.0},
            "sensor_drift": {"noise_std": 0.0, "missing_rate": 0.0, "drift_per_cycle": 0.01},
        }
        for scenario, corruption in scenarios.items():
            corrupted_initial = corrupt_samples(initial_samples, seed=seed, **corruption)
            corrupted_stream = corrupt_samples(stream_samples, seed=seed + 100, **corruption)
            frozen_pred = predict_samples(proposed, corrupted_stream, args.device)
            frozen_metrics = metric_summary(frozen_pred["soh_true"], frozen_pred["soh"], frozen_pred["rul_true"], frozen_pred["rul"])
            adapted_model = copy.deepcopy(proposed)
            started = time.perf_counter()
            adaptation = adapt_stream(adapted_model, corrupted_initial, corrupted_stream, config, args.device, use_physics=True, use_replay=True, use_drift=True)
            seconds = time.perf_counter() - started
            adapted_metrics = stream_metrics(adapted_model, corrupted_stream, args.device)
            robustness_rows.append({
                "seed": seed,
                "scenario": scenario,
                "frozen_metrics": frozen_metrics,
                "adapted_metrics": adapted_metrics,
                "adaptation_seconds": seconds,
                "drift_events": adaptation["drift_events"],
            })

    proposed_rows = [row for row in ablation_rows if row["variant"] == "proposed_adaptive"]
    acceptance = {
        "adaptation_improvement_fraction": mean_std(proposed_rows, "improvement_fraction"),
        "work_ratio": mean_std(proposed_rows, "work_ratio"),
        "prior_domain_forgetting_fraction": mean_std(proposed_rows, "prior_domain_forgetting_fraction"),
        "drift_events": mean_std(proposed_rows, "drift_events"),
    }
    acceptance["targets"] = {
        "minimum_improvement_fraction": 0.10,
        "maximum_work_ratio": 0.10,
        "maximum_forgetting_fraction": 0.05,
        "requires_nonzero_drift_events": True,
    }
    acceptance["passed"] = {
        "adaptation_improvement": acceptance["adaptation_improvement_fraction"]["mean"] >= 0.10,
        "computational_cost": acceptance["work_ratio"]["mean"] <= 0.10,
        "previous_domain_forgetting": acceptance["prior_domain_forgetting_fraction"]["mean"] <= 0.05,
        "drift_detection_triggered": acceptance["drift_events"]["mean"] > 0,
    }
    acceptance["overall"] = all(acceptance["passed"].values())
    result = {
        "status": "final_evidence_run",
        "protocol": {
            "dataset": "NASA PCoE B0005/B0006/B0007 train, B0018 held out",
            "seeds": args.seeds,
            "epochs": args.epochs,
            "history_cycles": config.history_cycles,
            "max_points_per_cycle": config.max_points_per_cycle,
            "train_windows": len(train_samples),
            "test_windows": len(test_samples),
            "initial_windows": len(initial_samples),
            "stream_windows": len(stream_samples),
        },
        "configuration": config_meta,
        "model_comparison": comparison_rows,
        "ablations": ablation_rows,
        "robustness": robustness_rows,
        "acceptance": acceptance,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "evidence_results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "final_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({key for row in comparison_rows for key in row})
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(comparison_rows)
    with (args.output_dir / "ablation_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["seed", "variant", "improvement_fraction", "work_ratio", "prior_domain_forgetting_fraction", "drift_events", "adaptation_seconds"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(ablation_rows)
    print(json.dumps({"output_dir": str(args.output_dir.resolve()), "acceptance": acceptance}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
