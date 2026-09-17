"""NASA PCoE battery data loading and leakage-safe labels."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.io import loadmat


@dataclass
class CycleRecord:
    """One usable discharge cycle with irregular samples preserved."""

    cell_id: str
    cycle_id: int
    cycle_type: str
    timestamp: np.ndarray
    voltage: np.ndarray
    current: np.ndarray
    temperature: np.ndarray
    capacity: float
    delta_t: np.ndarray
    mask: np.ndarray
    soh: float = float("nan")
    rul: float = float("nan")

    @property
    def n_points(self) -> int:
        return int(self.voltage.size)

    def summary(self) -> dict[str, Any]:
        """Return JSON-friendly audit information, not raw arrays."""

        valid = self.mask.astype(bool)
        def finite_range(values: np.ndarray) -> list[float | None]:
            v = np.asarray(values, dtype=float)[valid]
            v = v[np.isfinite(v)]
            return [float(v.min()), float(v.max())] if v.size else [None, None]

        return {
            "cell_id": self.cell_id,
            "cycle_id": self.cycle_id,
            "cycle_type": self.cycle_type,
            "n_points": self.n_points,
            "capacity_ah": float(self.capacity),
            "soh": None if not np.isfinite(self.soh) else float(self.soh),
            "rul": None if not np.isfinite(self.rul) else float(self.rul),
            "voltage_range": finite_range(self.voltage),
            "current_range": finite_range(self.current),
            "temperature_range": finite_range(self.temperature),
            "missing_fraction": float(1.0 - np.mean(valid)) if valid.size else 1.0,
        }


def _unwrap(value: Any) -> Any:
    """Unwrap MATLAB singleton arrays without destroying vectors."""

    while isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.item()
    if isinstance(value, np.ndarray) and value.size == 1 and value.dtype == object:
        return _unwrap(value.reshape(-1)[0])
    return value


def _field(obj: Any, name: str) -> Any:
    """Read a field from scipy MATLAB structs, dicts, or numpy records."""

    obj = _unwrap(obj)
    if obj is None:
        return None
    if isinstance(obj, dict):
        if name in obj:
            return obj[name]
        lower = {str(k).lower(): k for k in obj}
        return obj.get(lower.get(name.lower()))
    if isinstance(obj, np.void) and obj.dtype.names:
        lower = {str(k).lower(): k for k in obj.dtype.names}
        key = lower.get(name.lower())
        return obj[key] if key else None
    for candidate in (name, name.lower(), name.upper()):
        if hasattr(obj, candidate):
            return getattr(obj, candidate)
    return None


def _first_field(obj: Any, names: Iterable[str]) -> Any:
    for name in names:
        value = _field(obj, name)
        if value is not None:
            return value
    return None


def _numeric_array(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    value = _unwrap(value)
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return None
    return arr if arr.size else None


def _scalar(value: Any) -> float | None:
    arr = _numeric_array(value)
    if arr is None:
        return None
    finite = arr[np.isfinite(arr)]
    return float(finite[0]) if finite.size else None


def _text(value: Any) -> str:
    value = _unwrap(value)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        value = value.reshape(-1)[0]
    return str(value).strip()


def _find_cell_struct(mat: dict[str, Any], requested: str | None = None) -> Any:
    if requested and requested in mat:
        return mat[requested]
    candidates = []
    for key, value in mat.items():
        if key.startswith("__"):
            continue
        if _field(value, "cycle") is not None:
            candidates.append(value)
    if not candidates:
        raise ValueError("No MATLAB struct containing a 'cycle' field was found")
    return candidates[0]


def _normalise_time(time: np.ndarray | None, n: int) -> np.ndarray:
    if time is None or time.size != n or not np.isfinite(time).all():
        return np.arange(n, dtype=np.float64)
    time = time.astype(np.float64, copy=True)
    # NASA records are usually seconds from cycle start. Sorting preserves the
    # measured gaps and avoids negative delta_t if a file is not ordered.
    order = np.argsort(time, kind="stable")
    return time[order]


def _make_record(cell_id: str, cycle_id: int, raw_cycle: Any) -> CycleRecord | None:
    cycle_type = _first_field(raw_cycle, ("type", "Type"))
    cycle_type = _text(cycle_type).lower() if cycle_type is not None else ""
    if cycle_type != "discharge":
        return None
    payload_candidate = _first_field(raw_cycle, ("data", "Data"))
    payload = raw_cycle if payload_candidate is None else payload_candidate
    voltage = _numeric_array(_first_field(payload, ("Voltage_measured", "voltage")))
    current = _numeric_array(_first_field(payload, ("Current_measured", "current")))
    temperature = _numeric_array(_first_field(payload, ("Temperature_measured", "temperature")))
    timestamp = _numeric_array(_first_field(payload, ("Time", "time", "Timestamp", "timestamp")))
    capacity = _scalar(_first_field(payload, ("Capacity", "capacity")))
    if voltage is None or current is None or temperature is None or capacity is None:
        return None
    n = min(voltage.size, current.size, temperature.size)
    if timestamp is not None:
        n = min(n, timestamp.size)
    voltage, current, temperature = voltage[:n], current[:n], temperature[:n]
    timestamp = _normalise_time(timestamp[:n] if timestamp is not None else None, n)
    # If sorting was needed, apply the same order to sensor arrays.
    original_time = _numeric_array(_first_field(payload, ("Time", "time", "Timestamp", "timestamp")))
    if original_time is not None and original_time.size >= n and np.isfinite(original_time[:n]).all():
        order = np.argsort(original_time[:n], kind="stable")
        voltage, current, temperature = voltage[order], current[order], temperature[order]
    finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(temperature) & np.isfinite(timestamp)
    delta_t = np.diff(timestamp, prepend=timestamp[:1] if timestamp.size else np.array([0.0]))
    delta_t[~np.isfinite(delta_t)] = 0.0
    delta_t = np.maximum(delta_t, 0.0)
    # Keep the row shape and expose invalid samples through mask. The feature
    # builder masks them instead of interpolating over the measurement.
    return CycleRecord(
        cell_id=cell_id,
        cycle_id=int(cycle_id),
        cycle_type=cycle_type,
        timestamp=timestamp,
        voltage=np.nan_to_num(voltage, nan=0.0),
        current=np.nan_to_num(current, nan=0.0),
        temperature=np.nan_to_num(temperature, nan=0.0),
        capacity=float(capacity),
        delta_t=np.nan_to_num(delta_t, nan=0.0),
        mask=finite,
    )


def load_nasa_mat(path: str | Path, cell_id: str | None = None) -> list[CycleRecord]:
    """Load discharge cycles from a NASA PCoE MATLAB file.

    The parser accepts the common NASA v5 `.mat` layout and raises a focused
    error for v7.3 files, which need an HDF5 reader rather than silent failure.
    """

    path = Path(path)
    cell_id = cell_id or path.stem
    try:
        mat = loadmat(path, squeeze_me=True, struct_as_record=False)
    except NotImplementedError as exc:
        raise ValueError(f"{path.name} appears to be MATLAB v7.3; use an HDF5 reader") from exc
    root = _find_cell_struct(mat, cell_id)
    cycles = _field(root, "cycle")
    cycles = np.atleast_1d(cycles)
    records: list[CycleRecord] = []
    for raw_index, raw_cycle in enumerate(cycles, start=1):
        record = _make_record(cell_id, raw_index, raw_cycle)
        if record is not None:
            # The MATLAB operation index includes charge and impedance
            # records. The project target is discharge-cycle RUL, so expose a
            # contiguous aging-cycle index while retaining chronological order.
            record.cycle_id = len(records) + 1
            records.append(record)
    if not records:
        raise ValueError(f"No discharge cycles with measured capacity found in {path}")
    return records


def load_nasa_directory(directory: str | Path, cells: Iterable[str] = ("B0005", "B0006", "B0007", "B0018")) -> dict[str, list[CycleRecord]]:
    """Load the committed four-cell NASA subset from a directory."""

    directory = Path(directory)
    result: dict[str, list[CycleRecord]] = {}
    missing: list[str] = []
    for cell in cells:
        path = directory / f"{cell}.mat"
        if not path.exists():
            missing.append(str(path))
            continue
        result[cell] = load_nasa_mat(path, cell)
    if missing:
        missing_text = ", ".join(missing)
        raise FileNotFoundError(f"Missing NASA files: {missing_text}")
    return result


def assign_soh_rul(records_by_cell: dict[str, list[CycleRecord]], nominal_capacity_ah: float = 2.0, eol_soh: float = 70.0) -> dict[str, list[CycleRecord]]:
    """Assign labels after the EOL rule is fixed, without changing inputs."""

    for cell, records in records_by_cell.items():
        records.sort(key=lambda item: item.cycle_id)
        capacities = np.asarray([r.capacity for r in records], dtype=float)
        # The denominator is the declared nominal capacity. If a file has a
        # different protocol, the caller must pass that value explicitly.
        soh = 100.0 * capacities / float(nominal_capacity_ah)
        eol_indices = np.flatnonzero(soh <= eol_soh)
        eol_position = int(eol_indices[0]) if eol_indices.size else len(records) - 1
        for index, record in enumerate(records):
            record.soh = float(soh[index])
            record.rul = float(max(eol_position - index, 0))
    return records_by_cell


def audit_records(records_by_cell: dict[str, list[CycleRecord]]) -> dict[str, Any]:
    """Create a compact dataset audit suitable for JSON and faculty review."""

    all_records = [record for records in records_by_cell.values() for record in records]
    capacities = np.asarray([r.capacity for r in all_records], dtype=float)
    temperatures = np.concatenate([r.temperature[r.mask] for r in all_records if np.any(r.mask)]) if all_records else np.array([])
    currents = np.concatenate([r.current[r.mask] for r in all_records if np.any(r.mask)]) if all_records else np.array([])
    return {
        "cells": {cell: len(records) for cell, records in records_by_cell.items()},
        "total_cycles": len(all_records),
        "capacity_range_ah": [float(np.min(capacities)), float(np.max(capacities))] if capacities.size else [None, None],
        "temperature_range": [float(np.min(temperatures)), float(np.max(temperatures))] if temperatures.size else [None, None],
        "current_range": [float(np.min(currents)), float(np.max(currents))] if currents.size else [None, None],
        "missing_fraction": float(np.mean([1.0 - np.mean(r.mask) for r in all_records])) if all_records else None,
        "cycle_summaries": [r.summary() for r in all_records[:12]],
    }
