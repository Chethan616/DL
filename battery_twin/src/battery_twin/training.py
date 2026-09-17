"""Training, evaluation, and sequential adaptation utilities."""

from __future__ import annotations

from dataclasses import dataclass
import copy
import random
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TwinConfig
from .features import WindowDataset, WindowSample
from .metrics import metric_summary
from .physics import LossTracker, physics_losses, total_loss


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    result = {}
    for key, value in batch.items():
        result[key] = value.to(device) if isinstance(value, torch.Tensor) else value
    return result


def _forward_loss(model, batch: dict, config: TwinConfig, use_physics: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    output = model(batch["x"], batch["point_mask"], batch["cycle_mask"])
    losses = physics_losses(
        output["soh"],
        output["rul"],
        batch["soh_history"],
        batch["rul_history"],
        batch["cycle_mask"],
        eol_soh=config.eol_soh,
        rul_scale=config.rul_scale,
    )
    if use_physics:
        objective = total_loss(losses, config.physics_weight, config.consistency_weight)
    else:
        objective = losses["data"]
    return objective, losses


def fit_model(
    model: torch.nn.Module,
    samples: list[WindowSample],
    config: TwinConfig,
    use_physics: bool = False,
    epochs: int | None = None,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    """Fit one model and return a scalar training history."""

    if not samples:
        raise ValueError("Cannot train without window samples")
    device = torch.device(device)
    model.to(device)
    model.train()
    loader = DataLoader(WindowDataset(samples), batch_size=config.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    for epoch in range(epochs or config.epochs):
        tracker = LossTracker()
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            objective, losses = _forward_loss(model, batch, config, use_physics)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            tracker.add({"objective": objective, **losses})
        row = tracker.mean()
        row["epoch"] = float(epoch + 1)
        history.append(row)
    return history


@torch.no_grad()
def predict_samples(model: torch.nn.Module, samples: list[WindowSample], device: str | torch.device = "cpu") -> dict[str, np.ndarray | list[str] | list[int]]:
    if not samples:
        return {"soh": np.array([]), "rul": np.array([]), "soh_true": np.array([]), "rul_true": np.array([]), "cell_id": [], "cycle": []}
    device = torch.device(device)
    model.to(device)
    model.eval()
    loader = DataLoader(WindowDataset(samples), batch_size=256, shuffle=False)
    soh_hat, rul_hat, soh_true, rul_true, cells, cycles = [], [], [], [], [], []
    for raw_batch in loader:
        batch = _batch_to_device(raw_batch, device)
        output = model(batch["x"], batch["point_mask"], batch["cycle_mask"])
        last = batch["cycle_mask"].sum(dim=1).long().clamp_min(1) - 1
        row = torch.arange(last.shape[0], device=device)
        soh_hat.extend(output["soh"][row, last].detach().cpu().numpy().tolist())
        rul_hat.extend(output["rul"][row, last].detach().cpu().numpy().tolist())
        soh_true.extend(batch["target_soh"].detach().cpu().numpy().tolist())
        rul_true.extend(batch["target_rul"].detach().cpu().numpy().tolist())
        cells.extend(list(batch["cell_id"]))
        cycles.extend(batch["target_cycle"].detach().cpu().numpy().astype(int).tolist())
    return {
        "soh": np.asarray(soh_hat, dtype=float),
        "rul": np.asarray(rul_hat, dtype=float),
        "soh_true": np.asarray(soh_true, dtype=float),
        "rul_true": np.asarray(rul_true, dtype=float),
        "cell_id": cells,
        "cycle": cycles,
    }


def evaluate_model(model: torch.nn.Module, samples: list[WindowSample], device: str | torch.device = "cpu") -> dict[str, float]:
    predictions = predict_samples(model, samples, device)
    return metric_summary(
        predictions["soh_true"], predictions["soh"], predictions["rul_true"], predictions["rul"]
    )


class ReplayBuffer:
    """Recent samples plus evenly spaced historical representatives."""

    def __init__(self, max_size: int = 512):
        self.max_size = int(max_size)
        self._items: list[WindowSample] = []

    def add(self, sample: WindowSample) -> None:
        self._items.append(sample)
        if len(self._items) > self.max_size:
            indices = np.linspace(0, len(self._items) - 1, self.max_size).round().astype(int)
            self._items = [self._items[index] for index in indices]

    def extend(self, samples: Iterable[WindowSample]) -> None:
        for sample in samples:
            self.add(sample)

    def sample(self) -> list[WindowSample]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class DriftDetector:
    """Residual-based detector with a calibration threshold and cooldown."""

    def __init__(self, warmup: int = 8, quantile: float = 0.90, alpha: float = 0.25, cooldown: int = 3):
        self.warmup = int(warmup)
        self.quantile = float(quantile)
        self.alpha = float(alpha)
        self.cooldown = int(cooldown)
        self.reference: list[float] = []
        self.threshold: float | None = None
        self.ewma: float | None = None
        self._cooldown_left = 0

    def calibrate(self, errors: Iterable[float]) -> None:
        values = [float(value) for value in errors if np.isfinite(value)]
        self.reference = values
        if values:
            self.threshold = float(np.quantile(values, self.quantile))
            self.ewma = float(np.mean(values))

    def update(self, error: float) -> bool:
        error = float(error)
        if not np.isfinite(error):
            return False
        if self.ewma is None:
            self.ewma = error
        else:
            self.ewma = self.alpha * error + (1.0 - self.alpha) * self.ewma
        if len(self.reference) < self.warmup:
            self.reference.append(error)
            if len(self.reference) == self.warmup:
                self.threshold = float(np.quantile(self.reference, self.quantile))
            return False
        if self._cooldown_left:
            self._cooldown_left -= 1
            return False
        triggered = self.threshold is not None and self.ewma > self.threshold
        if triggered:
            self._cooldown_left = self.cooldown
        return bool(triggered)


def _sample_error(prediction: dict[str, np.ndarray], index: int, rul_scale: float) -> float:
    soh_err = abs(float(prediction["soh"][index]) - float(prediction["soh_true"][index])) / 100.0
    rul_err = abs(float(prediction["rul"][index]) - float(prediction["rul_true"][index])) / float(rul_scale)
    return soh_err + rul_err


def _adapt_steps(model, samples: list[WindowSample], config: TwinConfig, device: torch.device, use_physics: bool = True) -> None:
    if not samples:
        return
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.adaptation_learning_rate, weight_decay=config.weight_decay)
    loader = DataLoader(WindowDataset(samples), batch_size=min(config.batch_size, len(samples)), shuffle=True)
    for _ in range(config.adaptation_steps):
        for raw_batch in loader:
            batch = _batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            objective, losses = _forward_loss(model, batch, config, use_physics=use_physics)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()


def adapt_stream(
    model: torch.nn.Module,
    initial_samples: list[WindowSample],
    stream_samples: list[WindowSample],
    config: TwinConfig,
    device: str | torch.device = "cpu",
    use_physics: bool = True,
    use_replay: bool = True,
    use_drift: bool = True,
    fixed_update_every: int = 8,
) -> dict:
    """Run ordered replay adaptation and return a transparent event log."""

    device = torch.device(device)
    model.to(device)
    detector = DriftDetector(config.drift_warmup, config.drift_quantile, cooldown=config.drift_cooldown)
    trajectory_detector = DriftDetector(config.drift_warmup, config.drift_quantile, cooldown=config.drift_cooldown)
    buffer = ReplayBuffer(config.replay_size if use_replay else 1)
    if use_replay:
        buffer.extend(initial_samples[-min(len(initial_samples), config.replay_size):])
    reference_predictions = predict_samples(model, initial_samples, device)
    detector.calibrate([_sample_error(reference_predictions, i, config.rul_scale) for i in range(len(initial_samples))])
    def signature(sample: WindowSample) -> np.ndarray:
        valid_cycles = np.flatnonzero(sample.cycle_mask > 0)
        if valid_cycles.size == 0:
            return np.zeros(3, dtype=float)
        last = int(valid_cycles[-1])
        valid_points = sample.point_mask[last].astype(bool)
        return sample.x[last, valid_points, :3].mean(axis=0) if np.any(valid_points) else np.zeros(3, dtype=float)
    reference_signatures = np.asarray([signature(sample) for sample in initial_samples], dtype=float)
    reference_mean = reference_signatures.mean(axis=0) if len(reference_signatures) else np.zeros(3, dtype=float)
    reference_distances = np.mean(np.abs(reference_signatures - reference_mean), axis=1) if len(reference_signatures) else [0.0]
    trajectory_detector.calibrate(reference_distances)
    events = []
    before = []
    after = []
    for index, sample in enumerate(stream_samples, start=1):
        current = predict_samples(model, [sample], device)
        before.append({"soh": float(current["soh"][0]), "rul": float(current["rul"][0])})
        error = _sample_error(current, 0, config.rul_scale)
        feature_shift = float(np.mean(np.abs(signature(sample) - reference_mean)))
        residual_trigger = detector.update(error) if use_drift else False
        trajectory_trigger = trajectory_detector.update(feature_shift) if use_drift else False
        triggered = (residual_trigger or trajectory_trigger) if use_drift else (index % max(fixed_update_every, 1) == 0)
        buffer.add(sample)
        if triggered:
            _adapt_steps(model, buffer.sample() if use_replay else [sample], config, device, use_physics=use_physics)
        updated = predict_samples(model, [sample], device)
        after.append({"soh": float(updated["soh"][0]), "rul": float(updated["rul"][0])})
        events.append({
            "cell_id": sample.cell_id,
            "cycle": sample.target_cycle,
            "error_before": error,
            "ewma": detector.ewma,
            "threshold": detector.threshold,
            "feature_shift": feature_shift,
            "feature_threshold": trajectory_detector.threshold,
            "drift_triggered": bool(triggered),
            "replay_size": len(buffer),
            "use_replay": bool(use_replay),
            "use_drift": bool(use_drift),
        })
    return {"events": events, "predictions_before": before, "predictions_after": after, "drift_events": sum(e["drift_triggered"] for e in events)}
