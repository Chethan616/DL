# Faculty video and AI pair-programming reflection script

## Recording setup

Record the screen with the HTML report open, then show the project folder and the generated output files. Keep the final `evidence_results.json` visible when reading metrics.

## Script

**Opening — 30 seconds**

“Our project is an adaptive physics-informed battery digital twin. It estimates state of health and remaining useful life from battery measurements, then updates selected model parameters when later data shows a change in behavior. The important claim is not simply Transformer plus PINN. The claim we test is whether replay-based incremental adaptation improves later-stream error with less work than full retraining while preserving physical consistency.”

**Dataset and labels — 45 seconds**

“We use the NASA PCoE battery subset B0005, B0006, B0007, and B0018. The loader extracts discharge cycles, voltage, current, temperature, timestamps, and capacity. SOH is capacity divided by nominal capacity times 100. RUL is the declared EOL cycle minus the current discharge cycle, with EOL fixed at 70% SOH. The split is cell-aware so a held-out battery does not leak into training.”

**Architecture — 55 seconds**

“Each cycle is encoded with masked pooling over the measured voltage, current, temperature, time-gap, and cycle-index features. A temporal encoder models the history across cycles. Two heads predict SOH and RUL. The physics stage applies soft degradation and RUL consistency penalties plus a capacity-curve residual. We preserve irregular timing with `delta_t` and masks instead of blindly interpolating missing samples.”

**Experiments — 90 seconds**

“We compare a DNN, GRU, Transformer-only model, PINN without adaptation, and the proposed adaptive model. The tuning grid contains 54 combinations of hidden size, learning rate, physics weight, and replay-buffer size, evaluated over three seeds. We then remove physics, the temporal Transformer, replay, and drift detection one at a time. Robustness tests add sensor noise, missing samples, and sensor drift.”

**Adaptation and targets — 60 seconds**

“The online test reveals the held-out stream in time order. We compare a frozen model, periodic full retraining, and replay-buffer adaptation. We report post-shift MAE, update time, estimated work ratio, and forgetting on the initial domain. The acceptance targets are at least 10% post-shift MAE improvement, at most 10% of full-retraining work, and at most 5% previous-domain degradation. The result shown here is the recorded pass/fail status, including any failed target.”

“The physical-validity table shows zero SOH-bound and negative-RUL violations. The soft monotonic diagnostics are nonzero, so I present them as limitations and calibration targets rather than claiming perfect electrochemical validity.”

**AI pair-programming reflection — 45 seconds**

“AI helped structure the implementation and tests, but it did not decide what counted as scientific evidence. I used the attached papers to verify the distinction between reported experiments and proposed work. I also rejected an unsupported detailed electrochemical equation for the first NASA implementation and kept the SEI-PDE formulation as a documented extension. I checked the code with unit tests, a real NASA parser run, seeded experiments, and explicit leakage rules.”

**Closing — 20 seconds**

“The deliverable is an executable, auditable proof of concept. Its strongest contribution is tested adaptive updating, not a collection of familiar model names. The remaining limitation is that four NASA cells are appropriate for a course project proof of concept but not a universal battery-health claim.”

## Reflection checklist for the final upload

- Show the design diagram.
- Show all five model names.
- Show the 54-combination grid count and three seeds.
- Show at least one ablation and one robustness figure.
- Show frozen, full-retrain, and adaptive results.
- Read the target status exactly as recorded in `evidence_results.json`.
- Mention one limitation and one next experiment.
