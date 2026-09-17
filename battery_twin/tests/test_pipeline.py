import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from battery_twin.config import TwinConfig
from battery_twin.data import CycleRecord, assign_soh_rul
from battery_twin.features import FeatureStats, make_window_samples
from battery_twin.models import AdaptiveBatteryTwin, build_model
from battery_twin.physics import physics_losses
from battery_twin.robustness import corrupt_samples
from battery_twin.training import DriftDetector, ReplayBuffer


def make_records():
    records = {}
    for cell in ("B0005", "B0006"):
        cell_records = []
        for cycle in range(1, 7):
            n = 12 + cycle
            time = np.cumsum(np.ones(n))
            values = np.linspace(4.1, 3.0, n)
            cell_records.append(CycleRecord(cell, cycle, "discharge", time, values, -np.ones(n), 24 + np.zeros(n), 2.0 - 0.02 * cycle, np.diff(time, prepend=time[0]), np.ones(n, dtype=bool)))
        records[cell] = cell_records
    return assign_soh_rul(records)


def test_labels_and_windows_are_leakage_safe():
    records = make_records()
    stats = FeatureStats.fit(records["B0005"])
    samples = make_window_samples({"B0005": records["B0005"]}, stats, history_cycles=4, max_points=16)
    assert len(samples) == 6
    assert samples[-1].target_cycle == 6
    assert samples[-1].target_soh < samples[0].target_soh
    assert samples[-1].x.shape == (4, 16, 5)
    assert samples[-1].point_mask.shape == (4, 16)


def test_model_outputs_are_bounded_and_shaped():
    model = AdaptiveBatteryTwin(hidden_size=32, num_heads=4, num_layers=1)
    x = torch.randn(2, 4, 16, 5)
    point_mask = torch.ones(2, 4, 16)
    cycle_mask = torch.ones(2, 4)
    output = model(x, point_mask, cycle_mask)
    assert output["soh"].shape == (2, 4)
    assert output["rul"].shape == (2, 4)
    assert bool(torch.all((output["soh"] >= 0) & (output["soh"] <= 100)))
    assert bool(torch.all(output["rul"] >= 0))


def test_physics_loss_detects_soh_increase():
    soh_pred = torch.tensor([[90.0, 92.0, 91.0]])
    rul_pred = torch.tensor([[10.0, 9.0, 8.0]])
    soh_true = torch.tensor([[90.0, 91.0, 90.5]])
    rul_true = torch.tensor([[10.0, 9.0, 8.0]])
    losses = physics_losses(soh_pred, rul_pred, soh_true, rul_true, torch.ones(1, 3))
    assert losses["soh_monotonic"].item() > 0


def test_replay_buffer_preserves_bounded_history_and_detector_triggers():
    records = make_records()
    stats = FeatureStats.fit(records["B0005"])
    samples = make_window_samples({"B0005": records["B0005"]}, stats, history_cycles=4, max_points=16)
    buffer = ReplayBuffer(max_size=3)
    buffer.extend(samples)
    assert len(buffer) == 3
    detector = DriftDetector(warmup=2, quantile=0.5, alpha=1.0, cooldown=0)
    detector.calibrate([0.1, 0.1])
    assert detector.update(0.1) is False
    assert detector.update(0.5) is True


def test_model_factory_has_required_comparison_models():
    for name in ("dnn", "gru", "transformer", "pinn", "adaptive"):
        model = build_model(name, input_features=5, hidden_size=32, num_heads=4, num_layers=1)
        assert isinstance(model, torch.nn.Module)


def test_robustness_corruption_does_not_mutate_source():
    records = make_records()
    stats = FeatureStats.fit(records["B0005"])
    samples = make_window_samples({"B0005": records["B0005"]}, stats, history_cycles=4, max_points=16)
    original = samples[2].x.copy()
    corrupted = corrupt_samples(samples[2:3], noise_std=0.1, missing_rate=0.25, drift_per_cycle=0.01)
    assert np.array_equal(samples[2].x, original)
    assert np.sum(corrupted[0].point_mask == 0) > 0
