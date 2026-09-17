"""Run the faculty-requested hyperparameter grid for the adaptive model."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery_twin.config import TwinConfig
from battery_twin.data import assign_soh_rul, load_nasa_directory
from battery_twin.features import FeatureStats, make_window_samples
from battery_twin.models import build_model
from battery_twin.training import evaluate_model, fit_model, seed_everything


GRID = {
    "hidden_size": (64, 128),
    "learning_rate": (1e-4, 5e-4, 1e-3),
    "physics_weight": (0.1, 1.0, 10.0),
    "replay_size": (128, 512, 2048),
}


def combinations():
    names = tuple(GRID)
    for values in itertools.product(*(GRID[name] for name in names)):
        yield dict(zip(names, values))


def make_model(config: TwinConfig):
    return build_model(
        "adaptive",
        input_features=5,
        hidden_size=config.hidden_size,
        num_heads=config.num_heads,
        num_layers=config.num_transformer_layers,
        dropout=config.dropout,
        rul_scale=config.rul_scale,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Tune the adaptive battery twin")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs" / "tuning" / "grid.json")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--limit", type=int, default=0, help="Run only the first N combinations for a quick check; 0 runs all 54")
    args = parser.parse_args()
    records = assign_soh_rul(load_nasa_directory(args.data_dir))
    cells = list(records)
    if len(cells) < 4:
        raise ValueError("The committed NASA tuning protocol expects four cells")
    train_cells, validation_cell, test_cell = cells[:-2], cells[-2], cells[-1]
    stats = FeatureStats.fit(record for cell in train_cells for record in records[cell])
    # Tuning uses a smaller declared cycle representation for speed. The best
    # configuration is refit below with the full default representation.
    train_samples = make_window_samples({cell: records[cell] for cell in train_cells}, stats, history_cycles=8, max_points=128)
    validation_samples = make_window_samples({validation_cell: records[validation_cell]}, stats, history_cycles=8, max_points=128)
    rows = []
    combinations_to_run = list(combinations())
    if args.limit:
        combinations_to_run = combinations_to_run[: args.limit]
    for index, values in enumerate(combinations_to_run, start=1):
        seed_metrics = []
        for seed in args.seeds:
            seed_everything(seed)
            config = TwinConfig(
                hidden_size=values["hidden_size"],
                learning_rate=values["learning_rate"],
                physics_weight=values["physics_weight"],
                replay_size=values["replay_size"],
                epochs=args.epochs,
                history_cycles=8,
                max_points_per_cycle=128,
                batch_size=64,
            )
            model = make_model(config)
            fit_model(model, train_samples, config, use_physics=True, epochs=args.epochs)
            seed_metrics.append({"seed": seed, "metrics": evaluate_model(model, validation_samples)})
        scores = [row["metrics"]["soh_mae"] / 100.0 + row["metrics"]["rul_mae_cycles"] / config.rul_scale for row in seed_metrics]
        mean_metrics = {key: sum(row["metrics"][key] for row in seed_metrics) / len(seed_metrics) for key in seed_metrics[0]["metrics"]}
        std_metrics = {key: float(__import__("numpy").std([row["metrics"][key] for row in seed_metrics])) for key in seed_metrics[0]["metrics"]}
        rows.append({"index": index, "hyperparameters": values, "seed_metrics": seed_metrics, "mean_validation_metrics": mean_metrics, "std_validation_metrics": std_metrics, "selection_score": sum(scores) / len(scores)})
        print(f"[{index}/{len(combinations_to_run)}] mean_score={sum(scores) / len(scores):.5f} {values}")
    rows.sort(key=lambda row: row["selection_score"])
    best_values = rows[0]["hyperparameters"]
    best_config = TwinConfig(**best_values, epochs=args.epochs)
    # Final test fit uses train + validation cells after selection. The test
    # cell remains untouched until this one final evaluation.
    final_stats = FeatureStats.fit(record for cell in train_cells + [validation_cell] for record in records[cell])
    final_train = make_window_samples({cell: records[cell] for cell in train_cells + [validation_cell]}, final_stats)
    final_test = make_window_samples({test_cell: records[test_cell]}, final_stats)
    final_seed_metrics = []
    for seed in args.seeds:
        seed_everything(seed)
        best_model = make_model(best_config)
        fit_model(best_model, final_train, best_config, use_physics=True, epochs=args.epochs)
        final_seed_metrics.append({"seed": seed, "metrics": evaluate_model(best_model, final_test)})
    final_metrics = {key: sum(row["metrics"][key] for row in final_seed_metrics) / len(final_seed_metrics) for key in final_seed_metrics[0]["metrics"]}
    final_metrics_std = {key: float(__import__("numpy").std([row["metrics"][key] for row in final_seed_metrics])) for key in final_seed_metrics[0]["metrics"]}
    result = {
        "grid_definition": GRID,
        "combinations_requested": 54,
        "combinations_run": len(rows),
        "seeds": args.seeds,
        "tuning_representation": {"history_cycles": 8, "max_points_per_cycle": 128},
        "protocol": {"train_cells": train_cells, "validation_cell": validation_cell, "test_cell": test_cell},
        "best_hyperparameters": best_values,
        "final_test_metrics": final_metrics,
        "final_test_metrics_std": final_metrics_std,
        "final_test_seed_metrics": final_seed_metrics,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"output": str(args.output.resolve()), "best": best_values, "test": final_metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
