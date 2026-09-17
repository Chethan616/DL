"""Irregular-time-aware cycle features and window datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import CycleRecord


FEATURE_NAMES = ("voltage", "current", "temperature", "delta_t", "cycle_index")


@dataclass
class FeatureStats:
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, records: Iterable[CycleRecord]) -> "FeatureStats":
        rows = []
        for record in records:
            valid = record.mask.astype(bool)
            rows.append(_cycle_matrix(record)[valid])
        if not rows:
            raise ValueError("Cannot fit feature statistics without records")
        matrix = np.concatenate(rows, axis=0)
        mean = np.nanmean(matrix, axis=0)
        std = np.nanstd(matrix, axis=0)
        std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
        return cls(mean=mean.astype(np.float32), std=std.astype(np.float32))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        return ((matrix - self.mean) / self.std).astype(np.float32)


def _cycle_matrix(record: CycleRecord) -> np.ndarray:
    n = record.n_points
    cycle_index = np.full(n, float(record.cycle_id), dtype=np.float64)
    return np.column_stack((record.voltage, record.current, record.temperature, record.delta_t, cycle_index))


def encode_cycle(record: CycleRecord, stats: FeatureStats, max_points: int = 256) -> tuple[np.ndarray, np.ndarray]:
    """Encode by selecting measured points and padding with a mask.

    No interpolation is performed. If a cycle is longer than `max_points`,
    evenly spaced measured rows are selected and their real `delta_t` stays in
    the feature vector.
    """

    matrix = _cycle_matrix(record)
    valid = record.mask.astype(bool)
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        indices = np.arange(matrix.shape[0])
    if indices.size > max_points:
        pick = np.linspace(0, indices.size - 1, max_points).round().astype(int)
        indices = indices[pick]
    selected = stats.transform(matrix[indices])
    features = np.zeros((max_points, selected.shape[1]), dtype=np.float32)
    mask = np.zeros(max_points, dtype=np.float32)
    length = min(selected.shape[0], max_points)
    features[:length] = selected[:length]
    mask[:length] = 1.0
    return features, mask


@dataclass
class WindowSample:
    x: np.ndarray
    point_mask: np.ndarray
    cycle_mask: np.ndarray
    soh_history: np.ndarray
    rul_history: np.ndarray
    target_soh: float
    target_rul: float
    cell_id: str
    target_cycle: int


class WindowDataset(Dataset):
    def __init__(self, samples: list[WindowSample]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str | int]:
        sample = self.samples[index]
        return {
            "x": torch.from_numpy(sample.x),
            "point_mask": torch.from_numpy(sample.point_mask),
            "cycle_mask": torch.from_numpy(sample.cycle_mask),
            "soh_history": torch.from_numpy(sample.soh_history),
            "rul_history": torch.from_numpy(sample.rul_history),
            "target_soh": torch.tensor(sample.target_soh, dtype=torch.float32),
            "target_rul": torch.tensor(sample.target_rul, dtype=torch.float32),
            "cell_id": sample.cell_id,
            "target_cycle": sample.target_cycle,
        }


def make_window_samples(
    records_by_cell: dict[str, list[CycleRecord]],
    stats: FeatureStats,
    history_cycles: int = 12,
    max_points: int = 256,
) -> list[WindowSample]:
    samples: list[WindowSample] = []
    for cell_id, records in records_by_cell.items():
        records = sorted(records, key=lambda item: item.cycle_id)
        encoded = [encode_cycle(record, stats, max_points) for record in records]
        for end in range(len(records)):
            start = max(0, end - history_cycles + 1)
            selected = list(range(start, end + 1))
            cycle_mask = np.zeros(history_cycles, dtype=np.float32)
            x = np.zeros((history_cycles, max_points, len(FEATURE_NAMES)), dtype=np.float32)
            point_mask = np.zeros((history_cycles, max_points), dtype=np.float32)
            soh_history = np.zeros(history_cycles, dtype=np.float32)
            rul_history = np.zeros(history_cycles, dtype=np.float32)
            offset = history_cycles - len(selected)
            for local, source_index in enumerate(selected, start=offset):
                x[local], point_mask[local] = encoded[source_index]
                soh_history[local] = records[source_index].soh
                rul_history[local] = records[source_index].rul
                cycle_mask[local] = 1.0
            samples.append(WindowSample(
                x=x,
                point_mask=point_mask,
                cycle_mask=cycle_mask,
                soh_history=soh_history,
                rul_history=rul_history,
                target_soh=float(records[end].soh),
                target_rul=float(records[end].rul),
                cell_id=cell_id,
                target_cycle=int(records[end].cycle_id),
            ))
    return samples
