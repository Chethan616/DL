import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { FileBlob, PresentationFile } from "file:///C:/Users/cheth/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const SOURCE = "D:/DL/phase2_output/dl_phase2_complete_v2.pptx";
const BUILD_DIR = "D:/DL/.codex-ppt-build";
const CANDIDATE = path.join(BUILD_DIR, "dl_phase2_complete_v3_candidate.pptx");
const FINAL = "D:/DL/phase2_output/dl_phase2_complete_v3.pptx";
const EVIDENCE = "D:/DL/battery_twin/outputs/final_evidence_v2/evidence_results.json";

const NAVY = "#0B2545";
const NAVY_2 = "#13355E";
const BLUE = "#1B4A73";
const TEAL = "#00A896";
const TEAL_DARK = "#028090";
const PAPER = "#FFFFFF";
const INK = "#0B2545";
const MUTED = "#5B6B79";
const LIGHT = "#F4F8FA";
const LINE = "#D7E3E9";
const PALE_BLUE = "#E8F2F7";
const GOOD = "#147D64";
const AMBER = "#B87800";
const BODY = "Calibri";
const HEAD = "Cambria";

const deck = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
const evidence = JSON.parse(await fs.readFile(EVIDENCE, "utf-8"));

function fmt(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function meanStd(rows, key) {
  const values = rows.map((row) => Number(row[key]));
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const variance = values.reduce((a, b) => a + (b - mean) ** 2, 0) / values.length;
  return [mean, Math.sqrt(variance)];
}

function addText(slide, text, left, top, width, height, opts = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position: { left, top, width, height },
    fill: "none",
    line: { fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: opts.typeface ?? BODY,
    fontSize: opts.fontSize ?? 18,
    bold: opts.bold ?? false,
    italic: opts.italic ?? false,
    color: opts.color ?? INK,
    autoFit: "shrinkText",
  };
  return shape;
}

function addRect(slide, left, top, width, height, fill, radius = false, lineFill = fill) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position: { left, top, width, height },
    fill,
    line: { fill: lineFill, width: lineFill === "none" ? 0 : 1 },
  });
}

function addFooter(slide, page, dark = false) {
  addText(slide, "Adaptive PINN Battery Digital Twin", 58, 688, 360, 18, { fontSize: 11, color: dark ? "#8FA9BE" : MUTED });
  addText(slide, `${page} / 31`, 1135, 688, 85, 18, { fontSize: 11, color: dark ? "#8FA9BE" : MUTED });
}

function addHeading(slide, kicker, title, subtitle = "", dark = false) {
  addText(slide, kicker.toUpperCase(), 58, 42, 500, 20, { fontSize: 12, bold: true, color: dark ? TEAL : TEAL_DARK });
  addText(slide, title, 58, 75, 1060, 55, { typeface: HEAD, fontSize: 31, bold: true, color: dark ? PAPER : INK });
  if (subtitle) addText(slide, subtitle, 58, 135, 1110, 42, { fontSize: 15, color: dark ? "#C7D8E5" : MUTED });
}

function addTable(slide, values, left, top, width, height, options = {}) {
  const rows = values.length;
  const columns = values[0].length;
  const table = slide.tables.add({ rows, columns, left, top, width, height, values, columnWidths: options.columnWidths });
  table.styleOptions = { headerRow: true, bandedRows: true };
  table.borders.assign({ style: "solid", fill: LINE, width: 1 });
  for (let r = 0; r < rows; r += 1) {
    for (let c = 0; c < columns; c += 1) {
      const cell = table.getCell(r, c);
      cell.fill = r === 0 ? (options.headerFill ?? NAVY_2) : (r % 2 === 0 ? LIGHT : PAPER);
      cell.text.style = {
        typeface: BODY,
        fontSize: r === 0 ? (options.headerSize ?? 12) : (options.bodySize ?? 12),
        bold: r === 0 || (options.boldFirstColumn && c === 0),
        color: r === 0 ? PAPER : (options.bodyColor ?? INK),
        autoFit: "shrinkText",
      };
    }
  }
  return table;
}

function addFormulaCard(slide, label, formula, body, left, top, width, height, opts = {}) {
  const dark = opts.dark ?? false;
  const fill = dark ? NAVY_2 : LIGHT;
  const line = dark ? NAVY_2 : LINE;
  addRect(slide, left, top, width, height, fill, true, line);
  addText(slide, label.toUpperCase(), left + 18, top + 13, width - 36, 17, { fontSize: 10, bold: true, color: dark ? TEAL : TEAL_DARK });
  addText(slide, formula, left + 18, top + 39, width - 36, opts.formulaHeight ?? 40, { typeface: HEAD, fontSize: opts.formulaSize ?? 21, bold: true, color: dark ? PAPER : INK });
  if (body) addText(slide, body, left + 18, top + (opts.bodyTop ?? 86), width - 36, height - (opts.bodyTop ?? 86) - 12, { fontSize: opts.bodySize ?? 12, color: dark ? "#C7D8E5" : MUTED });
}

// 23 / 31 — Equations actively used by the project
{
  const slide = deck.slides.add();
  slide.background.fill = NAVY;
  addHeading(slide, "Phase II · Formula audit", "Equations used in our NASA prototype", "These are the target definitions and input contract that are active in the first reproducible implementation.", true);
  addFormulaCard(slide, "State of health", "SOHₖ = 100 × Cₖ / C_nom", "Measured discharge capacity Cₖ becomes the supervised health label. Capacity is not fed as a future deployment input.", 58, 210, 550, 150, { dark: true, formulaSize: 25 });
  addFormulaCard(slide, "Remaining useful life", "RULₖ = k_EOL − k", "RUL is measured in discharge cycles remaining from the current cycle k.", 638, 210, 585, 150, { dark: true, formulaSize: 25 });
  addFormulaCard(slide, "End of life", "k_EOL = min{k : SOHₖ ≤ 70%}", "Primary project threshold. Every RUL result is interpreted against this declared EOL rule.", 58, 390, 550, 150, { dark: true, formulaSize: 21 });
  addFormulaCard(slide, "Deployed input vector", "xₖ = [V, I, T, Δt, k]", "Voltage, current, temperature, measured time-gap, and cycle index. Irregular timing is preserved with Δt and masks.", 638, 390, 585, 150, { dark: true, formulaSize: 22 });
  addText(slide, "Status: implemented and evaluated on NASA B0005/B0006/B0007 → B0018", 58, 602, 1000, 24, { fontSize: 16, bold: true, color: TEAL });
  addFooter(slide, 23, true);
  slide.speakerNotes.textFrame.setText("Implementation source: D:/DL/battery_twin/src/battery_twin/data.py, features.py, models.py, and metrics.py. The EOL threshold is the student-selected primary POC rule, not a universal battery standard.");
}

// 24 / 31 — Active loss and diagnostics
{
  const slide = deck.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Formula audit", "The active training objective and physical diagnostics", "The first-stage POC uses capacity/degradation consistency that the NASA records can support; it does not claim an SEI-PDE residual.");
  addRect(slide, 58, 200, 1165, 88, NAVY, true, NAVY);
  addText(slide, "L_total = L_data + λ_phys L_phys + λ_cons L_cons", 90, 221, 760, 34, { typeface: HEAD, fontSize: 27, bold: true, color: PAPER });
  addText(slide, "selected: λ_phys = 0.1 · λ_cons = 1 · RUL scale = 250 cycles", 90, 260, 1000, 18, { fontSize: 13, color: "#C7D8E5" });
  addFormulaCard(slide, "Data loss", "L_data = mean|ŝoh−soh| + mean|r̂ul−rul|/250", "Joint SOH and RUL regression with valid-cycle masks.", 58, 320, 550, 120, { formulaSize: 18, bodyTop: 82 });
  addFormulaCard(slide, "Soft physics loss", "L_phys = mean ReLU(Δŝoh) + mean ReLU(Δr̂ul)", "Penalizes predicted SOH or RUL increases across ordered cycles; soft because real cells can show local regeneration.", 638, 320, 585, 120, { formulaSize: 17, bodyTop: 82 });
  addFormulaCard(slide, "Curve consistency", "L_cons = mean|Δŝoh−Δsoh| / 100", "Capacity-curve shape residual between predicted and measured SOH changes.", 58, 462, 550, 120, { formulaSize: 19, bodyTop: 82 });
  addFormulaCard(slide, "Bounded heads", "ŝoh = 100·σ(z₁),  r̂ul = 250·σ(z₂)", "Output contract: SOH ∈ [0,100] and RUL ∈ [0,250]. The nonnegative-consistency term is an explicit zero-valued future hook.", 638, 462, 585, 120, { formulaSize: 19, bodyTop: 82 });
  addText(slide, "Diagnostics reported: SOH bounds violation · negative RUL rate · degradation violation rate · SOH/RUL consistency violation · capacity-curve residual", 58, 615, 1165, 30, { fontSize: 13, italic: true, color: MUTED });
  addFooter(slide, 24);
  slide.speakerNotes.textFrame.setText("Exact implementation source: D:/DL/battery_twin/src/battery_twin/physics.py and metrics.py. `nonnegative_consistency` is deliberately zero-valued because the bounded output head already enforces SOH bounds; it is not presented as an active extra constraint.");
}

// 25 / 31 — Base paper equations 1–5
{
  const slide = deck.slides.add();
  slide.background.fill = NAVY;
  addHeading(slide, "0.base_paper.pdf · equations (1–5)", "Base paper: general PINN formulation", "Transcribed from the methodological review, pp. 4–5. These are representative framework equations, not one new end-to-end model.", true);
  addFormulaCard(slide, "(1) Neural approximation", "uθ(x,t) = NNθ(x,t)", "A network represents a battery state variable such as concentration, voltage, temperature, or SOC.", 58, 205, 550, 105, { dark: true, formulaSize: 25, bodyTop: 69, bodySize: 11 });
  addFormulaCard(slide, "(2) Governing operator", "𝒩[uθ](x,t; λ) = 0", "𝒩 is the physical operator derived from electrochemical, ECM, aging, or thermal equations; λ denotes physical parameters.", 638, 205, 585, 105, { dark: true, formulaSize: 24, bodyTop: 69, bodySize: 11 });
  addFormulaCard(slide, "(3) Residual", "rθ(x,t) = 𝒩[uθ](x,t; λ)", "Automatic differentiation evaluates the physical-law residual at a point.", 58, 332, 550, 105, { dark: true, formulaSize: 23, bodyTop: 69, bodySize: 11 });
  addFormulaCard(slide, "(4) Physics loss", "L_phys = (1/Nₚ) Σ |rθ(xᵢ,tᵢ)|²", "Residuals are averaged over Nₚ physics collocation points.", 638, 332, 585, 105, { dark: true, formulaSize: 22, bodyTop: 69, bodySize: 11 });
  addFormulaCard(slide, "(5) Composite PINN loss", "L_PINN = w_d L_data + w_p L_phys + w_b L_BC/IC + w_r L_reg", "Weights balance data fit, governing physics, boundary/initial conditions, and optional regularization.", 58, 459, 1165, 118, { dark: true, formulaSize: 20, bodyTop: 78, bodySize: 12 });
  addText(slide, "Project mapping: our L_data / L_phys / L_cons is a smaller NASA-supported specialization of this review framework.", 58, 610, 1160, 24, { fontSize: 14, bold: true, color: TEAL });
  addFooter(slide, 25, true);
  slide.speakerNotes.textFrame.setText("Source: D:/DL/0.base_paper.pdf, pp. 4–5, equations (1)–(5). The review presents this as a general methodological foundation and surveys different implementations.");
}

// 26 / 31 — Base paper equations 6–9
{
  const slide = deck.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "0.base_paper.pdf · equations (6–9)", "Base paper: electrochemical constraints", "These equations are reviewed as physics sources for PINNs; they are not activated in the first NASA POC because the required internal variables are not observed.");
  addFormulaCard(slide, "(6) Solid-phase diffusion", "∂cₛ/∂t = (Dₛ/r²) ∂/∂r (r² ∂cₛ/∂r)", "Fick-type diffusion in a spherical active-material particle; cₛ is concentration and Dₛ is solid-phase diffusivity.", 58, 205, 550, 150, { formulaSize: 20 });
  addFormulaCard(slide, "(7) Particle boundary conditions", "∂cₛ/∂r|₀ = 0,  −Dₛ∂cₛ/∂r|Rₛ = Jₙ", "Symmetry at the particle center and surface-flux condition at radius Rₛ.", 638, 205, 585, 150, { formulaSize: 19 });
  addFormulaCard(slide, "(8) Butler–Volmer kinetics", "jₙ = i₀[exp(αₐFη/RT) − exp(−α𝚌Fη/RT)]", "Interfacial reaction-current relation; i₀ is exchange current density and η is overpotential.", 58, 385, 550, 150, { formulaSize: 18 });
  addFormulaCard(slide, "(9) Charge conservation", "∇·iₛ = −aₛjₙ,   ∇·iₑ = aₛjₙ", "Solid/electrolyte current-density balance with specific interfacial area aₛ.", 638, 385, 585, 150, { formulaSize: 21 });
  addRect(slide, 58, 570, 1165, 62, PALE_BLUE, true, PALE_BLUE);
  addText(slide, "Implementation status", 82, 589, 180, 18, { fontSize: 13, bold: true, color: TEAL_DARK });
  addText(slide, "Reviewed for Stage 2 extension only · not claimed as an executed SEI/electrochemical experiment", 278, 589, 880, 18, { fontSize: 14, color: INK });
  addFooter(slide, 26);
  slide.speakerNotes.textFrame.setText("Source: D:/DL/0.base_paper.pdf, p. 5, equations (6)–(9). The review explicitly discusses diffusion, boundary conditions, Butler–Volmer kinetics, and charge conservation as candidate electrochemical constraints.");
}

// 27 / 31 — Base paper equations 10–16
{
  const slide = deck.slides.add();
  slide.background.fill = NAVY;
  addHeading(slide, "0.base_paper.pdf · equations (10–16)", "Base paper: ECM state evolution and aging laws", "These equations connect macroscopic circuit state, capacity fade, and monotonic degradation constraints.", true);
  addFormulaCard(slide, "(10) Coulomb counting", "dz/dt = −ηI(t)/Qₙ", "z is SOC, I is applied current, Qₙ is nominal capacity, and η is Coulombic efficiency.", 58, 200, 550, 105, { dark: true, formulaSize: 24, bodyTop: 70 });
  addFormulaCard(slide, "(11) RC polarization", "dVₚ/dt = −Vₚ/(RₚCₚ) + I/Cₚ", "First-order equivalent-circuit polarization branch.", 638, 200, 585, 105, { dark: true, formulaSize: 22, bodyTop: 70 });
  addFormulaCard(slide, "(12) Terminal voltage", "Vₜ = U_OC(z) − IR₀ − Vₚ", "Open-circuit voltage plus ohmic and polarization terms.", 58, 327, 550, 105, { dark: true, formulaSize: 24, bodyTop: 70 });
  addFormulaCard(slide, "(13) Capacity-based SOH", "SOH(k) = Q(k)/Q₀", "The base review’s capacity-health equation is the source for our percentage form SOHₖ = 100Qₖ/Q_nom.", 638, 327, 585, 105, { dark: true, formulaSize: 24, bodyTop: 70 });
  addFormulaCard(slide, "(14–15) Capacity fade", "Q(k)=Q₀−ΔQ(k)   or   Q(k)=Q₀−akᵇ", "General accumulated loss and a semi-empirical cycle-dependent approximation.", 58, 454, 550, 105, { dark: true, formulaSize: 20, bodyTop: 70 });
  addFormulaCard(slide, "(16) Monotonic aging", "dQ/dk ≤ 0   or   Q(k+1) ≤ Q(k)", "The basis for soft degradation constraints; local regeneration can motivate a soft rather than hard penalty.", 638, 454, 585, 105, { dark: true, formulaSize: 21, bodyTop: 70 });
  addText(slide, "Project mapping: we use the capacity-health and monotonicity ideas, but not the ECM internal state equations in the first POC.", 58, 610, 1160, 24, { fontSize: 14, bold: true, color: TEAL });
  addFooter(slide, 27, true);
  slide.speakerNotes.textFrame.setText("Source: D:/DL/0.base_paper.pdf, pp. 7–8, equations (10)–(16). Equation (13) directly motivates the project SOH label; equations (10)–(12) and (14)–(16) are reviewed physical formulations and constraints.");
}

// 28 / 31 — Base paper equations 17–19 and scope boundary
{
  const slide = deck.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "0.base_paper.pdf · equations (17–19)", "Base paper: thermal and reaction-rate constraints", "The review extends the PINN equation map to coupled thermal fields and temperature-dependent kinetics.");
  addFormulaCard(slide, "(17) Lumped thermal model", "C_th dT/dt = Q_gen − hAₛ(T−T_amb)", "Cell temperature T evolves from heat generation, convection, and ambient temperature.", 58, 215, 550, 145, { formulaSize: 21 });
  addFormulaCard(slide, "(18) Distributed heat conduction", "ρcₚ ∂T/∂t = ∇·(k_th∇T) + q_gen", "Spatial thermal field equation with density ρ, heat capacity cₚ, and conductivity k_th.", 638, 215, 585, 145, { formulaSize: 21 });
  addFormulaCard(slide, "(19) Arrhenius reaction rate", "k_rxn(T) = A_r exp(−E_a/(R_gT))", "Temperature-dependent reaction-rate constant with activation energy E_a.", 58, 390, 550, 145, { formulaSize: 23 });
  addRect(slide, 638, 390, 585, 145, NAVY, true, NAVY);
  addText(slide, "Why these are not in the first run", 664, 412, 490, 20, { fontSize: 15, bold: true, color: TEAL });
  addText(slide, "The NASA cycle records expose voltage, current, temperature, time, and capacity, but not every internal concentration, flux, heat-source, or reaction parameter required to claim these residuals.", 664, 446, 515, 62, { fontSize: 15, color: PAPER });
  addRect(slide, 58, 570, 1165, 62, PALE_BLUE, true, PALE_BLUE);
  addText(slide, "Scientific boundary", 82, 589, 180, 18, { fontSize: 13, bold: true, color: TEAL_DARK });
  addText(slide, "Equations (1)–(19) are the review’s core numbered map; only measurable capacity/degradation constraints are active in our Stage 1 implementation.", 278, 589, 900, 18, { fontSize: 14, color: INK });
  addFooter(slide, 28);
  slide.speakerNotes.textFrame.setText("Source: D:/DL/0.base_paper.pdf, p. 8, equations (17)–(19). The review is methodological; the student project deliberately postpones thermal-PDE and reaction-rate residuals until the required variables and parameters are available.");
}

// 29 / 31 — Comprehensive final model results
{
  const slide = deck.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Final results", "Measured model comparison under the held-out-cell protocol", "Mean ± standard deviation over seeds 7, 17, and 27; B0005/B0006/B0007 train and B0018 held out.");
  const rows = evidence.model_comparison;
  const names = ["dnn", "gru", "transformer", "pinn", "adaptive"];
  const labels = { dnn: "DNN", gru: "GRU", transformer: "Transformer-only", pinn: "PINN without adaptation", adaptive: "Adaptive twin" };
  const summary = names.map((name) => {
    const selected = rows.filter((row) => row.model === name);
    const [sohMae, sohMaeStd] = meanStd(selected, "soh_mae");
    const [sohRmse, sohRmseStd] = meanStd(selected, "soh_rmse");
    const [sohMape, sohMapeStd] = meanStd(selected, "soh_mape_percent");
    const [r2] = meanStd(selected, "soh_r2");
    const [rulMae, rulMaeStd] = meanStd(selected, "rul_mae_cycles");
    return [labels[name], `${fmt(sohMae)} ± ${fmt(sohMaeStd)}`, `${fmt(sohRmse)} ± ${fmt(sohRmseStd)}`, `${fmt(sohMape)} ± ${fmt(sohMapeStd)}%`, fmt(r2), `${fmt(rulMae)} ± ${fmt(rulMaeStd)}`];
  });
  addTable(slide, [["Model", "SOH MAE", "SOH RMSE", "SOH MAPE", "SOH R²", "RUL MAE (cycles)"], ...summary], 58, 200, 1165, 300, { columnWidths: [290, 175, 175, 180, 120, 225], headerSize: 12, bodySize: 12, boldFirstColumn: true });
  addRect(slide, 58, 535, 1165, 70, NAVY, true, NAVY);
  addText(slide, "Readout", 82, 554, 120, 18, { fontSize: 14, bold: true, color: TEAL });
  addText(slide, "Standalone PINN has the lowest mean frozen SOH MAE (7.76 ± 1.42). The adaptive twin’s value is the later-stream improvement and low update cost, not a claim of the best frozen-cell score.", 208, 551, 980, 30, { fontSize: 14, color: PAPER });
  addFooter(slide, 29);
  slide.speakerNotes.textFrame.setText("Results source: D:/DL/battery_twin/outputs/final_evidence_v2/evidence_results.json and its rendered summaries. The four training epochs are intentionally retained as the executed POC protocol; longer training and broader validation are next experiments.");
}

// 30 / 31 — Acceptance, ablation, robustness results
{
  const slide = deck.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Evidence", "Adaptation targets, ablations, and robustness results", "All required experiment families were executed; the acceptance table is the direct proof-of-concept decision record.");
  const a = evidence.acceptance;
  const acceptance = [
    ["Acceptance measure", "Mean ± SD", "Target", "Status"],
    ["Post-shift improvement", `${fmt(a.adaptation_improvement_fraction.mean * 100)} ± ${fmt(a.adaptation_improvement_fraction.std * 100)}%`, "≥ 10%", "PASS"],
    ["Adaptation work ratio", `${fmt(a.work_ratio.mean * 100)} ± ${fmt(a.work_ratio.std * 100)}%`, "≤ 10%", "PASS"],
    ["Prior-domain forgetting", `${fmt(a.prior_domain_forgetting_fraction.mean * 100)} ± ${fmt(a.prior_domain_forgetting_fraction.std * 100)}%`, "≤ 5%", "PASS*"],
    ["Drift events", `${fmt(a.drift_events.mean, 0)} ± ${fmt(a.drift_events.std, 0)}`, "> 0", "PASS"],
  ];
  addTable(slide, acceptance, 58, 195, 565, 235, { columnWidths: [245, 150, 100, 70], headerSize: 11, bodySize: 11, boldFirstColumn: true });
  const robust = evidence.robustness;
  const scenarios = ["clean", "voltage_current_temperature_noise", "missing_samples", "sensor_drift"];
  const scenarioLabels = { clean: "Clean", voltage_current_temperature_noise: "V/I/T noise", missing_samples: "Missing samples", sensor_drift: "Sensor drift" };
  const robustTable = scenarios.map((scenario) => {
    const selected = robust.filter((row) => row.scenario === scenario);
    const [frozen] = meanStd(selected.map((row) => ({ v: row.frozen_metrics.soh_mae })), "v");
    const [adapted] = meanStd(selected.map((row) => ({ v: row.adapted_metrics.soh_mae })), "v");
    const [events] = meanStd(selected, "drift_events");
    return [scenarioLabels[scenario], fmt(frozen), fmt(adapted), fmt(events, 0)];
  });
  addTable(slide, [["Scenario", "Frozen MAE", "Adapted MAE", "Events"], ...robustTable], 658, 195, 565, 235, { columnWidths: [190, 125, 125, 80], headerSize: 11, bodySize: 11, boldFirstColumn: true });
  const ablationRows = evidence.ablations;
  const variants = ["proposed_adaptive", "no_physics", "no_transformer", "no_replay", "no_drift_detection", "full_retraining"];
  const labels = { proposed_adaptive: "Proposed", no_physics: "No physics", no_transformer: "No Transformer", no_replay: "No replay", no_drift_detection: "No drift detector", full_retraining: "Full retraining" };
  const ablationLine = variants.map((variant) => {
    const selected = ablationRows.filter((row) => row.variant === variant);
    const [after] = meanStd(selected.map((row) => ({ v: row.adapted_metrics.soh_mae })), "v");
    return `${labels[variant]} ${fmt(after)}`;
  }).join("  ·  ");
  addRect(slide, 58, 455, 1165, 94, NAVY, true, NAVY);
  addText(slide, "Ablation post-shift SOH MAE", 82, 475, 260, 19, { fontSize: 14, bold: true, color: TEAL });
  addText(slide, ablationLine, 82, 510, 1085, 22, { fontSize: 13, color: PAPER });
  const adaptiveRows = ablationRows.filter((row) => row.variant === "proposed_adaptive").map((row) => row.adapted_metrics);
  const [bounds] = meanStd(adaptiveRows.map((r) => ({ v: r.soh_bounds_violation_rate })), "v");
  const [negativeRul] = meanStd(adaptiveRows.map((r) => ({ v: r.rul_negative_rate })), "v");
  const [curve, curveStd] = meanStd(adaptiveRows.map((r) => ({ v: r.capacity_curve_residual })), "v");
  addText(slide, `Physical checks: SOH bounds ${fmt(bounds * 100)}% · negative RUL ${fmt(negativeRul * 100)}% · capacity-curve residual ${fmt(curve, 5)} ± ${fmt(curveStd, 5)} · *negative forgetting means prior-domain MAE improved`, 58, 580, 1165, 28, { fontSize: 12, italic: true, color: MUTED });
  addFooter(slide, 30);
  slide.speakerNotes.textFrame.setText("Acceptance, ablation, robustness, and physical-validity source: D:/DL/battery_twin/outputs/final_evidence_v2/evidence_results.json. PASS* is interpreted with absolute forgetting magnitude: the measured negative value means the prior-domain error improved rather than degraded.");
}

// 31 / 31 — Review and literature evidence boundary
{
  const slide = deck.slides.add();
  slide.background.fill = NAVY;
  addHeading(slide, "Phase II · Evidence boundary", "What the base review and prior papers actually prove", "The project result is new evidence from the executed NASA protocol; literature values remain context because protocols are not directly equivalent.", true);
  addTable(slide, [
    ["Source", "Experiment / formula focus", "Reported result or evidence"],
    ["0.base_paper.pdf", "Methodological review: PINN residuals, electrochemical/ECM/aging/thermal equations, uncertainty, deployment", "No single new end-to-end model or one reproducible score; it synthesizes reviewed methods."],
    ["Applied Energy paper", "Multi-task PINN + simplified Transformer; CV charging features; sensitivity and transfer learning", "SOH MAPE 0.75%; degradation-path deviation ≈0.01; RUL MAE 104 cycles in reported cases."],
    ["Energy EM-PINN", "SEI-layer-growth PDE with PDE, BC, IC, and SOH losses; XJTU/MIT/HUST; six data-volume scenarios", "Reported MAPE as low as 0.873%."],
    ["Scientific Reports Bat-T-GNN", "Cycle-aware patches, time-aware convolution, Transformer, dynamic GNN, PINN-RUL; ablations and corruption tests", "Reported NASA values include RMSE 0.014, MAE 0.007, MAPE 0.008; robustness evidence is central."],
    ["Complex & Intelligent Systems", "PINN + physics-guided augmentation + momentum contrastive learning on B0005/B0006/B0007/B0018", "Compared CNN, BPINN, Informer, XGBoost-ARIMA; added ablation, SHAP, parameter, time, and latency analysis."],
  ], 58, 190, 1165, 355, { columnWidths: [205, 520, 440], headerSize: 11, bodySize: 10, boldFirstColumn: true });
  addRect(slide, 58, 580, 1165, 54, BLUE, true, BLUE);
  addText(slide, "Faculty wording", 82, 597, 160, 18, { fontSize: 13, bold: true, color: TEAL });
  addText(slide, "Our defensible novelty claim is replay-buffer adaptation with drift detection and physics-preserving updates; it remains a project claim until replicated beyond this POC.", 258, 597, 920, 18, { fontSize: 13, color: PAPER });
  addFooter(slide, 31, true);
  slide.speakerNotes.textFrame.setText("Sources: D:/DL/0.base_paper.pdf, 1.pdf, 2.pdf, 3.pdf, 4.pdf. External references: Applied Energy https://www.sciencedirect.com/science/article/pii/S0306261925011572; Energy https://www.sciencedirect.com/science/article/pii/S0360544225053964; Scientific Reports https://www.nature.com/articles/s41598-025-28505-5; Complex & Intelligent Systems https://doi.org/10.1007/s40747-025-02194-z. Do not numerically rank the reported values against the project table without harmonizing datasets, labels, splits, and metric definitions.");
}

await fs.mkdir(BUILD_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL), { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(CANDIDATE);

const skillDir = "C:/Users/cheth/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const { finalizePresentation } = await import(`file://${skillDir}/container_tools/artifact_tool_utils.mjs`);
const referenceSha256 = createHash("sha256").update(await fs.readFile("D:/DL/dl.pptx")).digest("hex");
const stagingDir = path.join(BUILD_DIR, "finalizer_v3");
await fs.mkdir(stagingDir, { recursive: true });
const requiredTables = [14, 16, 17, 18, 19, 20, 29, 30, 31];
const result = await finalizePresentation({
  explicitTotalSlideCount: 31,
  requiredNativeTableOwnerSlides: requiredTables,
  workspaceDir: "D:/DL",
  candidatePath: CANDIDATE,
  finalPath: FINAL,
  pythonExecutable: "C:/Users/cheth/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: `${skillDir}/container_tools/inspect_presentation_package_integrity.py`,
  layoutValidatorPath: `${skillDir}/container_tools/inspect_presentation_layout_geometry.py`,
  layoutArgs: ["--expected-slide-size-emu", "12192000,6858000", "--validate-bullet-geometry", "--validate-heading-fit", ...requiredTables.flatMap((n) => ["--require-native-table-slide", String(n)])],
  requiredNativeTableOwnerSlides: requiredTables,
  fontPolicy: { basis: "reference", families: [BODY, HEAD], referencePath: "D:/DL/dl.pptx", referenceSha256 },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "dl_phase2_complete_v3.validation.json"),
});
console.log(JSON.stringify({ final: FINAL, result }, null, 2));
