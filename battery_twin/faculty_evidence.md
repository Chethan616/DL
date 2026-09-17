# Faculty evidence brief

## Project title

Adaptive Physics-Informed Battery Digital Twin for SOH and RUL Prediction

## What this submission proves

This package separates three kinds of evidence:

- **Reported literature evidence:** the experiments and results in the four attached research papers.
- **Project implementation evidence:** the executable NASA-first pipeline in this folder.
- **New student evidence:** the completed tuning, comparison, ablation, robustness, and online-adaptation outputs in `outputs/`.

The final numerical claims below come from `outputs/final_evidence_v2/evidence_results.json`. The one-epoch files in `outputs/nasa_poc_v2/` are preliminary checks only.

## Rubric mapping

| Rubric item | Evidence to show | File or screen |
|---|---|---|
| Model architecture design | NASA records → cycle parser → masked intra-cycle encoder → temporal encoder → physics losses → SOH/RUL heads → replay adaptation | `adaptive_battery_twin_analysis.html`, Section 05; `src/battery_twin/models.py` |
| Multiple models | Empirical/DNN, GRU, Transformer-only, PINN, and adaptive model under the same held-out-cell protocol | `outputs/final_evidence_v2/final_comparison.csv` |
| Hyperparameter tuning | 54 combinations across hidden size, learning rate, physics weight, and replay size; three random seeds | `outputs/tuning/grid.json` |
| Experimental analysis | Dataset audit, error metrics, ablations, robustness, physical-validity checks, and adaptation cost | `outputs/final_evidence_v2/evidence_results.json`, PNG figures, CSV tables |
| Presentation and reflection | Recorded walkthrough and AI pair-programming reflection | `video_reflection_script.md` |

## Suggested 6-minute walkthrough

1. **0:00–0:40 — Problem and gap.** Explain that SOH/RUL models can lose calibration as the battery stream changes. The proposed contribution is replay-buffer adaptation with retained physics constraints.
2. **0:40–1:20 — Dataset.** Open the dataset audit in `outputs/final_evidence_v2/evidence_results.json` or the capacity-fade figure. State the four NASA cells, cycle counts, measured fields, 70% EOL rule, and cell-aware split.
3. **1:20–2:05 — Architecture.** Walk left-to-right through the design diagram in the HTML report. Emphasize that the first physics stage uses measurable capacity/degradation relationships; SEI-PDE is an extension, not an invented claim.
4. **2:05–2:50 — Multiple models and tuning.** Show the final comparison CSV and the selected configuration from `outputs/tuning/grid.json`. Explain that paper metrics are context, while this table uses one declared project protocol.
5. **2:50–4:00 — Proof of experiment.** Show ablations: remove physics, Transformer, replay, and drift detection; then show noise, missing-sample, and sensor-drift results.
6. **4:00–5:10 — Adaptation test.** Show the frozen/full-retrain/adaptive replay comparison, update count, elapsed time, work ratio, and previous-domain forgetting.
7. **5:10–6:00 — Conclusion and limits.** State whether the three acceptance targets passed. If a target failed, explain the evidence and the next engineering change; do not relabel a failed target as success.

## Final numbers to read aloud

Read these final values directly from `outputs/final_evidence_v2/evidence_results.json`:

- Best grid configuration: hidden size **64**, learning rate **0.001**, physics weight **0.1**, replay size **128**.
- Held-out adaptive model: SOH MAE **10.05 ± 1.46**, RUL MAE **38.36 ± 3.39 cycles**.
- Post-shift improvement: **80.63% ± 2.46 percentage points**.
- Adaptation work ratio: **8.75%** of full retraining.
- Previous-domain forgetting: **−10.20% ± 4.47 percentage points**; negative means performance improved.
- Overall target status: **PASS**. Drift detection triggered **7 updates per seed**.

Physical-validity diagnostics are also exported. The adaptive frozen model had **0.00%** SOH-bound violations and **0.00%** negative-RUL predictions; its SOH degradation-violation rate was **23.66% ± 3.89%**, SOH/RUL consistency-violation rate **29.01% ± 5.43%**, and normalized capacity-curve residual **0.0107 ± 0.0028**. On the proposed adapted stream these were **0.00%**, **0.00%**, **21.43% ± 2.89%**, **30.95% ± 3.15%**, and **0.00573 ± 0.00009**, respectively. Explain that monotonic rates are soft diagnostics because local capacity regeneration can occur; they are not presented as proof of electrochemical correctness.

The standalone PINN had the best frozen-model SOH MAE (**7.76 ± 1.42**) in this run. The proposed adaptive twin achieved the online adaptation target, but the no-physics ablation reached lower post-shift MAE (**1.23**) than the physics-weighted proposal (**1.74**). This is a useful scientific result: physics weighting needs calibration rather than being treated as automatically beneficial.

## Reproducibility commands

```powershell
cd D:\DL\battery_twin
python scripts/tune.py --data-dir data/raw --output outputs/tuning/grid.json --epochs 2 --seeds 7 17 27
python scripts/run_evidence.py --data-dir data/raw --grid outputs/tuning/grid.json --output-dir outputs/final_evidence_v2 --epochs 4 --seeds 7 17 27
python -m pytest -q tests
```

The reduced tuning representation is declared in the grid JSON. The evidence runner uses the full default cycle-window representation unless overridden in code.

## AI pair-programming reflection points

- AI helped convert the literature and faculty rubric into testable interfaces, data contracts, loss components, and experiment scripts.
- The project decision to use capacity/degradation consistency first, and postpone SEI-PDE physics, was a human scientific-scope decision based on data availability.
- The source audit prevented proposal diagrams and review-paper content from being presented as measured project results.
- The final evidence is reproducible because seeds, held-out cell, EOL threshold, tuning grid, ablation variants, and output files are recorded.
- The main limitation is dataset scale: four NASA cells support a proof of concept, not broad chemistry or operating-condition generalization.
