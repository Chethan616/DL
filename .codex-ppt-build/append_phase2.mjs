import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { FileBlob, PresentationFile } from "file:///C:/Users/cheth/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const SOURCE = "D:/DL/dl.pptx";
const BUILD_DIR = "D:/DL/.codex-ppt-build";
const CANDIDATE = path.join(BUILD_DIR, "dl_phase2_candidate.pptx");
const FINAL = "D:/DL/phase2_output/dl_phase2_complete.pptx";
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

const p = await PresentationFile.importPptx(await FileBlob.load(SOURCE));
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
  addText(slide, `${page} / 21`, 1135, 688, 85, 18, { fontSize: 11, color: dark ? "#8FA9BE" : MUTED });
}

function addHeading(slide, kicker, title, subtitle = "", dark = false) {
  addText(slide, kicker.toUpperCase(), 58, 42, 500, 20, { fontSize: 12, bold: true, color: dark ? TEAL : TEAL_DARK });
  addText(slide, title, 58, 75, 1060, 55, { typeface: HEAD, fontSize: 31, bold: true, color: dark ? PAPER : INK });
  if (subtitle) addText(slide, subtitle, 58, 135, 1110, 42, { fontSize: 15, color: dark ? "#C7D8E5" : MUTED });
}

function addBullet(slide, number, title, body, left, top, width, dark = false) {
  addRect(slide, left, top + 2, 24, 24, TEAL, true);
  addText(slide, String(number), left + 7, top + 4, 14, 16, { fontSize: 11, bold: true, color: PAPER });
  addText(slide, title, left + 40, top, width - 40, 24, { fontSize: 17, bold: true, color: dark ? PAPER : INK });
  addText(slide, body, left + 40, top + 28, width - 40, 48, { fontSize: 14, color: dark ? "#C7D8E5" : MUTED });
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

function addPill(slide, label, value, left, top, width, fill = PALE_BLUE, valueColor = INK) {
  addRect(slide, left, top, width, 70, fill, true, fill);
  addText(slide, label.toUpperCase(), left + 16, top + 12, width - 32, 16, { fontSize: 10, bold: true, color: TEAL_DARK });
  addText(slide, value, left + 16, top + 32, width - 32, 26, { typeface: HEAD, fontSize: 22, bold: true, color: valueColor });
}

function addNode(slide, label, body, left, top, width, height, active = false) {
  addRect(slide, left, top, width, height, active ? NAVY_2 : LIGHT, true, active ? NAVY_2 : LINE);
  addRect(slide, left + 15, top + 15, 28, 28, active ? TEAL : TEAL_DARK, true);
  addText(slide, "•", left + 23, top + 14, 12, 20, { fontSize: 20, bold: true, color: PAPER });
  addText(slide, label, left + 15, top + 52, width - 30, 22, { fontSize: 15, bold: true, color: active ? PAPER : INK });
  addText(slide, body, left + 15, top + 78, width - 30, height - 90, { fontSize: 11, color: active ? "#C7D8E5" : MUTED });
}

function addConnector(slide, left, top, width) {
  addRect(slide, left, top, width, 4, TEAL, false, TEAL);
}

// 12 / 21 — Phase II divider
{
  const slide = p.slides.add();
  slide.background.fill = NAVY;
  addHeading(slide, "Phase II · Review 2", "Model Design & Experimental Study", "Completed implementation, evaluation evidence, and faculty-ready proof of concept", true);
  addText(slide, "NASA-first evidence bundle", 58, 235, 500, 26, { fontSize: 18, bold: true, color: TEAL });
  addText(slide, "54 hyperparameter combinations · 3 random seeds · 18 ablation rows · 12 robustness rows", 58, 270, 850, 28, { fontSize: 19, color: PAPER });
  addRect(slide, 58, 355, 1090, 3, TEAL, false, TEAL);
  addText(slide, "Course rubric focus", 58, 405, 250, 24, { fontSize: 14, bold: true, color: "#9FC6E0" });
  addText(slide, "Model architecture · Multiple models · Hyperparameter tuning · Experimental analysis · Presentation and AI reflection", 58, 438, 1060, 48, { fontSize: 18, color: PAPER });
  addFooter(slide, 12, true);
  slide.speakerNotes.textFrame.setText("Phase II section appended to the supplied deck. Evidence source: D:/DL/battery_twin/outputs/final_evidence_v2/evidence_results.json.");
}

// 13 / 21 — What was built
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Deliverable", "The project moved from proposal to tested prototype", "The original deck described the direction. This section records the executable implementation and measured evidence.");
  addBullet(slide, 1, "Data contract", "NASA B0005, B0006, B0007, and B0018 are parsed into ordered discharge-cycle records with voltage, current, temperature, time-gap, and capacity.", 70, 220, 520);
  addBullet(slide, 2, "Model stack", "DNN, GRU, Transformer-only, PINN, and adaptive Transformer-plus-physics models run under the same held-out-cell protocol.", 70, 340, 520);
  addBullet(slide, 3, "Online learning", "A drift detector triggers low-cost replay-buffer updates as the held-out stream arrives in time order.", 70, 460, 520);
  addBullet(slide, 4, "Faculty evidence", "The run exports final comparison tables, ablations, robustness results, physical diagnostics, figures, and a reflection script.", 665, 220, 520);
  addBullet(slide, 5, "What remains a proposal", "SEI-PDE physics and broad multi-chemistry validation remain extensions because the NASA POC does not expose every required electrochemical variable.", 665, 340, 520);
  addBullet(slide, 6, "Evidence discipline", "Reported paper metrics stay separate from new project metrics. The project claims only what the executed protocol measures.", 665, 460, 520);
  addFooter(slide, 13);
  slide.speakerNotes.textFrame.setText("Project implementation files: D:/DL/battery_twin/src/battery_twin and D:/DL/battery_twin/scripts. The source audit in the HTML report distinguishes proposal claims, literature evidence, and new student evidence.");
}

// 14 / 21 — Dataset and protocol
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Dataset", "NASA cells and a leakage-safe evaluation protocol", "The official NASA PCoE repository provides the first committed dataset for this proof of concept.");
  const audit = [
    ["Item", "Project choice", "Why it matters"],
    ["Cells", "B0005 / B0006 / B0007 train · B0018 held out", "Battery-level separation tests transfer to a new cell."],
    ["Labels", "SOH = capacity / nominal capacity × 100 · EOL = 70% SOH", "RUL = EOL cycle − current cycle."],
    ["Inputs", "Voltage · current · temperature · delta_t · cycle index", "Measured timing is retained with masks."],
    ["Protocol", "504 train windows · 132 test windows · 33 initial · 99 stream", "Later cycles arrive sequentially for adaptation."],
    ["Seeds", "7 · 17 · 27", "Final tables report mean ± standard deviation."],
  ];
  addTable(slide, audit, 58, 205, 1165, 270, { columnWidths: [170, 430, 565], headerSize: 12, bodySize: 12, boldFirstColumn: true });
  addPill(slide, "EOL rule", "70% SOH", 58, 520, 200, PALE_BLUE, TEAL_DARK);
  addPill(slide, "Held-out cell", "B0018", 280, 520, 200, PALE_BLUE, TEAL_DARK);
  addPill(slide, "Cycles parsed", "636 total", 502, 520, 200, PALE_BLUE, TEAL_DARK);
  addPill(slide, "Order", "Time-aware", 724, 520, 200, PALE_BLUE, TEAL_DARK);
  addPill(slide, "Metric unit", "RUL in cycles", 946, 520, 230, PALE_BLUE, TEAL_DARK);
  addFooter(slide, 14);
  slide.speakerNotes.textFrame.setText("Dataset source: https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/. Local audit and labels are implemented in data.py and features.py. The 636-cycle count is the parser audit across the four selected NASA files.");
}

// 15 / 21 — Architecture
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Architecture", "From measured cycle to adaptive digital-twin state", "The first-stage physics model uses relationships supported by the NASA measurements.");
  const nodes = [
    ["NASA records", "Cycle-level voltage, current, temperature, time, capacity", false],
    ["Masked encoder", "Measured points + delta_t, no blind interpolation", false],
    ["Temporal model", "History across cycles with Transformer context", false],
    ["SOH / RUL heads", "Bounded SOH 0–100 and non-negative RUL", false],
    ["Physics losses", "Soft degradation, RUL order, capacity-curve residual", false],
    ["Replay adaptation", "Drift trigger, recent + representative history", true],
  ];
  const left = 58;
  const top = 225;
  const gap = 18;
  const width = 178;
  nodes.forEach((node, i) => {
    const x = left + i * (width + gap);
    addNode(slide, node[0], node[1], x, top, width, 170, node[2]);
    if (i < nodes.length - 1) addConnector(slide, x + width + 4, top + 83, gap - 8);
  });
  addRect(slide, 58, 455, 1165, 112, NAVY, true, NAVY);
  addText(slide, "Training objective", 84, 478, 210, 22, { fontSize: 15, bold: true, color: TEAL });
  addText(slide, "L_total = L_data + λ_phys L_phys + λ_cons L_cons", 300, 472, 520, 34, { typeface: HEAD, fontSize: 24, bold: true, color: PAPER });
  addText(slide, "The SEI-growth PDE formulation remains a Stage 2 extension until every variable and parameter is available and verified.", 84, 522, 1080, 25, { fontSize: 14, color: "#C7D8E5" });
  addFooter(slide, 15);
  slide.speakerNotes.textFrame.setText("Implementation sources: models.py, physics.py, training.py. The methodology review and EM-PINN paper inform the physics discussion, but the student POC does not claim to reproduce the SEI-PDE experiment.");
}

// 16 / 21 — Tuning and baselines
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Tuning", "The selected configuration came from the complete grid", "Every requested combination ran across seeds 7, 17, and 27 before the final evidence pass.");
  const grid = evidence.configuration.best_hyperparameters;
  addTable(slide, [
    ["Grid dimension", "Values tested"],
    ["Hidden size", "64 · 128"],
    ["Learning rate", "0.0001 · 0.0005 · 0.001"],
    ["Physics weight", "0.1 · 1 · 10"],
    ["Replay size", "128 · 512 · 2048"],
    ["Total combinations", "54 × 3 seeds = 162 runs"],
  ], 58, 210, 520, 280, { columnWidths: [220, 300], headerSize: 13, bodySize: 13, boldFirstColumn: true });
  addRect(slide, 650, 210, 573, 280, NAVY, true, NAVY);
  addText(slide, "Selected configuration", 682, 235, 430, 24, { fontSize: 15, bold: true, color: TEAL });
  addText(slide, `Hidden size  ${grid.hidden_size}\nLearning rate  ${grid.learning_rate}\nPhysics weight  ${grid.physics_weight}\nReplay size  ${grid.replay_size}`, 682, 280, 480, 140, { typeface: HEAD, fontSize: 24, bold: true, color: PAPER });
  addText(slide, "The final evidence runner then used the selected configuration on the full default cycle-window representation.", 682, 440, 480, 34, { fontSize: 14, color: "#C7D8E5" });
  addPill(slide, "Grid status", "54 / 54 complete", 58, 540, 250, PALE_BLUE, GOOD);
  addPill(slide, "Model count", "5 comparisons", 330, 540, 230, PALE_BLUE, TEAL_DARK);
  addPill(slide, "Evidence mode", "Held-out cell", 580, 540, 230, PALE_BLUE, TEAL_DARK);
  addPill(slide, "Output", "JSON + CSV + PNG", 830, 540, 260, PALE_BLUE, TEAL_DARK);
  addFooter(slide, 16);
  slide.speakerNotes.textFrame.setText("Tuning source: D:/DL/battery_twin/outputs/tuning/grid.json. The grid JSON records combinations_requested = 54, combinations_run = 54, and seeds = [7, 17, 27].");
}

// 17 / 21 — Final model table
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Model comparison", "The standalone PINN led frozen SOH accuracy", "All model rows use the same B0018 held-out evaluation protocol and four training epochs.");
  const rows = evidence.model_comparison;
  const summary = ["dnn", "gru", "transformer", "pinn", "adaptive"].map((name) => {
    const selected = rows.filter((row) => row.model === name);
    const [sohMean, sohStd] = meanStd(selected, "soh_mae");
    const [rulMean, rulStd] = meanStd(selected, "rul_mae_cycles");
    const [r2Mean] = meanStd(selected, "soh_r2");
    const label = name === "dnn" ? "DNN" : name === "gru" ? "GRU" : name === "transformer" ? "Transformer-only" : name === "pinn" ? "PINN without adaptation" : "Adaptive twin";
    return [label, `${fmt(sohMean)} ± ${fmt(sohStd)}`, `${fmt(rulMean)} ± ${fmt(rulStd)}`, fmt(r2Mean)];
  });
  addTable(slide, [["Model", "SOH MAE", "RUL MAE (cycles)", "SOH R²"], ...summary], 58, 205, 1165, 285, { columnWidths: [360, 260, 300, 245], headerSize: 14, bodySize: 14, boldFirstColumn: true });
  const adaptive = summary.find((row) => row[0] === "Adaptive twin");
  const pinn = summary.find((row) => row[0] === "PINN without adaptation");
  addRect(slide, 58, 525, 560, 80, NAVY, true, NAVY);
  addText(slide, "Adaptive twin", 82, 545, 180, 18, { fontSize: 14, bold: true, color: TEAL });
  addText(slide, `${adaptive[1]} SOH MAE  ·  ${adaptive[2]} RUL MAE`, 82, 570, 500, 22, { fontSize: 17, bold: true, color: PAPER });
  addRect(slide, 650, 525, 573, 80, PALE_BLUE, true, PALE_BLUE);
  addText(slide, "Best frozen SOH row", 675, 545, 210, 18, { fontSize: 14, bold: true, color: TEAL_DARK });
  addText(slide, `PINN: ${pinn[1]} SOH MAE`, 675, 570, 480, 22, { fontSize: 17, bold: true, color: INK });
  addFooter(slide, 17);
  slide.speakerNotes.textFrame.setText("Metrics source: D:/DL/battery_twin/outputs/final_evidence_v2/evidence_results.json and final_summary.csv. Paper metrics are not numerically ranked against this table because datasets, splits, labels, and metric scales differ.");
}

// 18 / 21 — Ablations and adaptation
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Ablations", "Replay adaptation delivered the low-cost online update", "The ablation table shows which components mattered in the ordered B0018 stream.");
  const ablationRows = evidence.ablations;
  const variants = ["proposed_adaptive", "no_physics", "no_transformer", "no_replay", "no_drift_detection", "full_retraining"];
  const labels = { proposed_adaptive: "Proposed adaptive", no_physics: "No physics", no_transformer: "No Transformer", no_replay: "No replay", no_drift_detection: "No drift detection", full_retraining: "Full retraining" };
  const ablationTable = variants.map((variant) => {
    const r = ablationRows.filter((row) => row.variant === variant);
    const [improvement] = meanStd(r, "improvement_fraction");
    const [work] = meanStd(r, "work_ratio");
    const [forgetting] = meanStd(r, "prior_domain_forgetting_fraction");
    const [events] = meanStd(r, "drift_events");
    const after = r.map((row) => row.adapted_metrics.soh_mae).reduce((a, b) => a + b, 0) / r.length;
    return [labels[variant], fmt(after), variant === "full_retraining" ? "Reference" : `${fmt(improvement * 100)}%`, `${fmt(work * 100)}%`, variant === "full_retraining" ? "—" : fmt(events, 0)];
  });
  addTable(slide, [["Variant", "Post-shift SOH MAE", "Improvement", "Work", "Events"], ...ablationTable], 58, 200, 1165, 310, { columnWidths: [320, 220, 220, 180, 225], headerSize: 13, bodySize: 13, boldFirstColumn: true });
  addPill(slide, "Target 1", "80.63% improvement", 58, 545, 265, PALE_BLUE, GOOD);
  addPill(slide, "Target 2", "8.75% work", 340, 545, 230, PALE_BLUE, GOOD);
  addPill(slide, "Target 3", "−10.20% forgetting", 590, 545, 265, PALE_BLUE, GOOD);
  addPill(slide, "Detector", "7 updates / seed", 875, 545, 250, PALE_BLUE, GOOD);
  addFooter(slide, 18);
  slide.speakerNotes.textFrame.setText("Ablation source: evidence_results.json and ablation_summary.csv. Negative forgetting means the initial-domain SOH MAE improved after adaptation. Full retraining is the cost reference.");
}

// 19 / 21 — Robustness and physical validity
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Robustness", "The adaptive state remained useful under sensor corruption", "Robustness scenarios use the same trained proposal and corrupt the ordered stream before adaptation.");
  const robust = evidence.robustness;
  const scenarios = ["clean", "voltage_current_temperature_noise", "missing_samples", "sensor_drift"];
  const scenarioLabels = { clean: "Clean ordered stream", voltage_current_temperature_noise: "Voltage/current/temp noise", missing_samples: "10% missing samples", sensor_drift: "Sensor drift" };
  const robustTable = scenarios.map((scenario) => {
    const r = robust.filter((row) => row.scenario === scenario);
    const [frozen] = meanStd(r.map((row) => ({ value: row.frozen_metrics.soh_mae })), "value");
    const [adapted] = meanStd(r.map((row) => ({ value: row.adapted_metrics.soh_mae })), "value");
    const [events] = meanStd(r, "drift_events");
    return [scenarioLabels[scenario], fmt(frozen), fmt(adapted), fmt(events, 0)];
  });
  addTable(slide, [["Scenario", "Frozen SOH MAE", "Adaptive SOH MAE", "Updates"], ...robustTable], 58, 195, 650, 250, { columnWidths: [270, 140, 150, 90], headerSize: 12, bodySize: 12, boldFirstColumn: true });
  const adaptiveRows = evidence.ablations.filter((row) => row.variant === "proposed_adaptive").map((row) => row.adapted_metrics);
  const physical = [
    ["Diagnostic", "Mean ± SD"],
    ["SOH bounds violations", `${fmt(meanStd(adaptiveRows.map((r) => ({ v: r.soh_bounds_violation_rate })), "v")[0] * 100)}%`],
    ["Negative RUL", `${fmt(meanStd(adaptiveRows.map((r) => ({ v: r.rul_negative_rate })), "v")[0] * 100)}%`],
    ["SOH degradation violation", `${fmt(meanStd(adaptiveRows.map((r) => ({ v: r.soh_degradation_violation_rate })), "v")[0] * 100)}%`],
    ["SOH/RUL consistency violation", `${fmt(meanStd(adaptiveRows.map((r) => ({ v: r.soh_rul_consistency_violation_rate })), "v")[0] * 100)}%`],
    ["Capacity-curve residual", fmt(meanStd(adaptiveRows.map((r) => ({ v: r.capacity_curve_residual })), "v")[0], 5)],
  ];
  addTable(slide, physical, 750, 195, 473, 250, { columnWidths: [300, 173], headerSize: 12, bodySize: 12, boldFirstColumn: true });
  addRect(slide, 58, 490, 1165, 88, NAVY, true, NAVY);
  addText(slide, "Interpretation", 82, 511, 180, 18, { fontSize: 14, bold: true, color: TEAL });
  addText(slide, "Hard bounds stayed at zero. Soft monotonic diagnostics stayed nonzero, so the result supports a measured proof of concept with calibration limits rather than perfect electrochemical validity.", 82, 536, 1078, 28, { fontSize: 15, color: PAPER });
  addFooter(slide, 19);
  slide.speakerNotes.textFrame.setText("Robustness and physical-validity source: evidence_results.json. The monotonic rates are diagnostics because real cells can show local capacity regeneration. Capacity-curve residual is the first-stage data-supported residual, not an SEI-PDE residual.");
}

// 20 / 21 — Faculty proof of experiment
{
  const slide = p.slides.add();
  slide.background.fill = PAPER;
  addHeading(slide, "Phase II · Faculty evidence", "Proof of experiment mapped to the course rubric", "The deck now points to concrete artifacts a reviewer can inspect.");
  addTable(slide, [
    ["Rubric criterion", "What to show", "Artifact"],
    ["Model architecture design", "Data → masked encoder → temporal model → physics losses → heads → replay update", "HTML report · Section 05"],
    ["Multiple models", "DNN, GRU, Transformer-only, PINN, adaptive twin", "final_summary.csv"],
    ["Hyperparameter tuning", "54 combinations across three seeds", "tuning/grid.json"],
    ["Experimental analysis", "Ablations, robustness, physical checks, acceptance table", "evidence_results.json + PNG figures"],
    ["Presentation and reflection", "6-minute walkthrough and AI pair-programming reflection", "video_reflection_script.md"],
  ], 58, 200, 1165, 300, { columnWidths: [250, 545, 370], headerSize: 12, bodySize: 12, boldFirstColumn: true });
  addRect(slide, 58, 535, 1165, 64, PALE_BLUE, true, PALE_BLUE);
  addText(slide, "Recording checklist", 82, 550, 180, 18, { fontSize: 14, bold: true, color: TEAL_DARK });
  addText(slide, "Show the design diagram · show all five models · open the 54-combination grid · show one ablation and one robustness figure · read PASS status exactly", 278, 550, 900, 24, { fontSize: 14, color: INK });
  addFooter(slide, 20);
  slide.speakerNotes.textFrame.setText("Faculty rubric source: D:/DL/RUBRICS FOR COURSE BASED DESIGN PROJECT - FALL 26-27.pdf and the supplied Phase II screenshots. The actual recording must be made and uploaded by the student; this slide and video_reflection_script.md prepare the evidence.");
}

// 21 / 21 — Final status and reflection
{
  const slide = p.slides.add();
  slide.background.fill = NAVY;
  addHeading(slide, "Phase II · Final status", "A tested adaptive POC with a clear next experiment", "The strongest evidence is the low-cost online update. The main limitation is the small four-cell dataset.", true);
  addRect(slide, 58, 215, 520, 250, NAVY_2, true, NAVY_2);
  addText(slide, "What the run supports", 88, 242, 380, 22, { fontSize: 16, bold: true, color: TEAL });
  addBullet(slide, 1, "Adaptation benefit", "80.63% lower post-shift SOH MAE than frozen.", 88, 288, 430, true);
  addBullet(slide, 2, "Low update cost", "8.75% of full-retraining work.", 88, 368, 430, true);
  addRect(slide, 635, 215, 588, 250, PAPER, true, PAPER);
  addText(slide, "What the run does not prove", 665, 242, 450, 22, { fontSize: 16, bold: true, color: TEAL_DARK });
  addText(slide, "• Universal chemistry or temperature generalization\n• Perfect monotonic degradation behavior\n• SEI-PDE validity without additional variables\n• A direct numerical ranking against the four papers", 665, 292, 500, 120, { fontSize: 17, color: INK });
  addRect(slide, 58, 520, 1165, 72, BLUE, true, BLUE);
  addText(slide, "Next experiment", 84, 540, 170, 20, { fontSize: 14, bold: true, color: TEAL });
  addText(slide, "Calibrate physics weights, add longer training, and repeat leave-one-cell-out validation before making a broader scientific claim.", 260, 540, 900, 22, { fontSize: 16, color: PAPER });
  addFooter(slide, 21, true);
  slide.speakerNotes.textFrame.setText("Closing reflection: AI helped translate the proposal and rubric into code, tests, and evidence tables. Human scientific judgment kept unsupported SEI-PDE claims out of the first NASA implementation. Manual video recording remains the only presentation step not automated here.");
}

await fs.mkdir(BUILD_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL), { recursive: true });
await (await PresentationFile.exportPptx(p)).save(CANDIDATE);

const expectedSlideSizeEmu = "12192000,6858000";
const skillDir = "C:/Users/cheth/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const { finalizePresentation } = await import(`file://${skillDir}/container_tools/artifact_tool_utils.mjs`);
const referenceSha256 = createHash("sha256").update(await fs.readFile(SOURCE)).digest("hex");
const stagingDir = path.join(BUILD_DIR, "finalizer");
await fs.mkdir(stagingDir, { recursive: true });
const requirements = {
  explicitTotalSlideCount: 21,
  requiredNativeTableOwnerSlides: [14, 16, 17, 18, 19, 20],
};
const result = await finalizePresentation({
  ...requirements,
  workspaceDir: "D:/DL",
  candidatePath: CANDIDATE,
  finalPath: FINAL,
  pythonExecutable: "C:/Users/cheth/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe",
  integrityValidatorPath: `${skillDir}/container_tools/inspect_presentation_package_integrity.py`,
  layoutValidatorPath: `${skillDir}/container_tools/inspect_presentation_layout_geometry.py`,
  layoutArgs: ["--expected-slide-size-emu", expectedSlideSizeEmu, "--validate-bullet-geometry", "--validate-heading-fit", ...requirements.requiredNativeTableOwnerSlides.flatMap((n) => ["--require-native-table-slide", String(n)])],
  requiredNativeTableOwnerSlides: requirements.requiredNativeTableOwnerSlides,
  fontPolicy: { basis: "reference", families: [BODY, HEAD], referencePath: SOURCE, referenceSha256 },
  verifyArtifactToolImport: true,
  receiptPath: path.join(stagingDir, "dl_phase2_complete.validation.json"),
});
console.log(JSON.stringify({ final: FINAL, candidate: CANDIDATE, result }, null, 2));
