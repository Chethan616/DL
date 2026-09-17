import fs from "node:fs/promises";
import path from "node:path";
import { createHash } from "node:crypto";
import { FileBlob, PresentationFile } from "file:///C:/Users/cheth/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs";

const SOURCE = "D:/DL/phase2_output/dl_phase2_complete.pptx";
const BUILD_DIR = "D:/DL/.codex-ppt-build";
const CANDIDATE = path.join(BUILD_DIR, "dl_phase2_complete_v2_candidate.pptx");
const FINAL = "D:/DL/phase2_output/dl_phase2_complete_v2.pptx";
const NAVY = "#0B2545";
const BLUE = "#1B4A73";
const TEAL = "#00A896";
const PAPER = "#FFFFFF";
const MUTED = "#C7D8E5";
const BODY = "Calibri";
const HEAD = "Cambria";

const deck = await PresentationFile.importPptx(await FileBlob.load(SOURCE));

// Make the rubric focus explicit on the existing Phase II deliverable slide.
const focusSlide = deck.slides.getItem(12);
deck.resolve("sh/q50nydsj").text.replace(
  "The project moved from proposal to tested prototype",
  "Phase II focus: DL model design and hyperparameter tuning",
);
deck.resolve("sh/bq9orito").text.replace(
  "The original deck described the direction. This section records the executable implementation and measured evidence.",
  "The required work is explicit: design the DL models, tune their hyperparameters, deploy multiple models, and compare results.",
);

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
    color: opts.color ?? NAVY,
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

const videoSlide = deck.slides.add();
videoSlide.background.fill = NAVY;
addText(videoSlide, "PHASE II · PRESENTATION", 58, 42, 500, 20, { fontSize: 12, bold: true, color: TEAL });
addText(videoSlide, "Video demonstration", 58, 75, 1060, 55, { typeface: HEAD, fontSize: 34, bold: true, color: PAPER });
addText(videoSlide, "Empty submission slide for the Moodle, Stream, or Drive video link", 58, 138, 1050, 30, { fontSize: 17, color: MUTED });
addRect(videoSlide, 150, 260, 980, 165, BLUE, true, TEAL);
addText(videoSlide, "VIDEO LINK", 200, 294, 250, 22, { fontSize: 14, bold: true, color: TEAL });
addText(videoSlide, "PASTE VIDEO URL HERE", 200, 335, 820, 40, { typeface: HEAD, fontSize: 30, bold: true, color: PAPER });
addText(videoSlide, "Replace this placeholder with the actual uploaded video URL before submitting.", 200, 480, 900, 28, { fontSize: 16, color: MUTED });
addText(videoSlide, "No external video URL was present in the supplied files, so this slide intentionally keeps the link field open.", 200, 520, 900, 24, { fontSize: 13, italic: true, color: "#9FC6E0" });
addText(videoSlide, "Adaptive PINN Battery Digital Twin", 58, 688, 360, 18, { fontSize: 11, color: "#8FA9BE" });
addText(videoSlide, "22 / 22", 1135, 688, 85, 18, { fontSize: 11, color: "#8FA9BE" });
videoSlide.speakerNotes.textFrame.setText("Before faculty submission, replace PASTE VIDEO URL HERE with the actual Moodle, Stream, Drive, or other approved video URL. The user did not provide a video URL in the request.");

await fs.mkdir(BUILD_DIR, { recursive: true });
await fs.mkdir(path.dirname(FINAL), { recursive: true });
await (await PresentationFile.exportPptx(deck)).save(CANDIDATE);

const skillDir = "C:/Users/cheth/.codex/plugins/cache/openai-primary-runtime/presentations/26.904.11930/skills/presentations";
const { finalizePresentation } = await import(`file://${skillDir}/container_tools/artifact_tool_utils.mjs`);
const referenceSha256 = createHash("sha256").update(await fs.readFile("D:/DL/dl.pptx")).digest("hex");
const stagingDir = path.join(BUILD_DIR, "finalizer_v2");
await fs.mkdir(stagingDir, { recursive: true });
const requiredTables = [14, 16, 17, 18, 19, 20];
const result = await finalizePresentation({
  explicitTotalSlideCount: 22,
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
  receiptPath: path.join(stagingDir, "dl_phase2_complete_v2.validation.json"),
});
console.log(JSON.stringify({ final: FINAL, result }, null, 2));
