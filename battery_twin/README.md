# Adaptive Physics-Informed Battery Digital Twin

This folder is the executable NASA-first proof of concept described in
`D:\DL\adaptive_battery_twin_analysis.html`.

The code is deliberately staged:

1. Parse NASA PCoE battery `.mat` files into cycle records.
2. Build leakage-safe, irregular-time-aware cycle windows.
3. Train comparable DNN, GRU, Transformer, PINN, and adaptive models.
4. Add soft degradation and SOH/RUL consistency losses.
5. Reveal later-life cycles sequentially and test replay-buffer adaptation.

The raw NASA archive is not copied into the repository. Put the extracted
files (`B0005.mat`, `B0006.mat`, `B0007.mat`, and `B0018.mat`) in
`data/raw/`, or point the runner to another directory with `--data-dir`.

## Quick start

From `D:\DL\battery_twin`:

```powershell
python -m pip install -e .
python scripts/run_poc.py --smoke-test --output-dir outputs/smoke
```

The smoke test uses deterministic synthetic cycles only to validate the
pipeline. It is not a scientific result and must not be used in the faculty
comparison table.

When the NASA files are available:

```powershell
python scripts/run_poc.py --data-dir data/raw --output-dir outputs/nasa_poc --epochs 40
```

The runner writes a dataset audit and model metrics. It uses cell-aware
leave-one-cell-out evaluation where the held-out cell is never used for
training. The adaptive comparison reports frozen, full-retrain, and replay
updates separately.

Each run also writes `capacity_fade.png`, `sensor_distributions.png`, and
`model_comparison.png` for the faculty evidence package.

The presentation and reflection material is in `faculty_evidence.md` and
`video_reflection_script.md`.

Render the final CSV summary and faculty-ready comparison figures with:

```powershell
python scripts/render_final_evidence.py --input outputs/final_evidence_v2/evidence_results.json --output-dir outputs/final_evidence_v2
```

The faculty hyperparameter grid is executable as a separate run:

```powershell
python scripts/tune.py --data-dir data/raw --output outputs/tuning/grid.json --epochs 40
```

The full grid is 54 combinations: hidden size 64/128, learning rate
0.0001/0.0005/0.001, physics weight 0.1/1/10, and replay size 128/512/2048.
Use `--limit 2` first to check the setup without waiting for the complete grid.

## Evidence boundary

The first implementation uses only measurements the NASA records can support:
capacity-derived SOH/RUL labels, soft long-term degradation consistency, and
capacity-curve residuals. The SEI-growth PDE from the EM-PINN paper is not
silently invented here; it is reserved for a later extension after its exact
variables and parameters are verified against the available data.
