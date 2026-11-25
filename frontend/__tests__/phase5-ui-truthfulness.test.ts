/**
 * Phase 5 UI Truthfulness, Research/Production Separation & Preference Alignment Tests
 *
 * Verifies every Phase 5 contract from driveguard_final_implementation_plan.md:
 * 1. Zero fabricated ONNX readiness or false CNN-arbitration claims in production UI.
 * 2. No-op FaultInjectionToolbar and ExperimentSelector removed from Live Monitor;
 *    isolated Synthetic Testbench exists and is linked from Sidebar.
 * 3. Operational State and Drowsiness State are clearly separated in Monitor Inspector.
 * 4. Hardware console explicitly declares SIMULATED HARDWARE CONSOLE with zero fake
 *    "UART LINK: 115200 BAUD [ACTIVE]" badge.
 * 5. Dashboard shows independent Browser AI & Python AI calibration status and aligned
 *    sensitivity presets (1.20s / 1.50s / 1.80s) with no "MEDIUM" or "Using Fallback Defaults".
 * 6. Project Overview counts strictly the 4 spatial zones (excluding metadata keys).
 * 7. Settings and Onboarding both standardize on RELAXED | STANDARD | HIGH.
 * 8. Benchmarks, Defense Guide, and Help have zero raw unrendered $...$ LaTeX in JSX
 *    (preserving intentional LaTeX export string in benchmarks) and truthful uvicorn command.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

const FRONTEND_ROOT = path.resolve(__dirname, "..");

function readFrontend(relPath: string): string {
  const fullPath = path.join(FRONTEND_ROOT, relPath);
  assert.ok(fs.existsSync(fullPath), `Expected file to exist: ${relPath}`);
  return fs.readFileSync(fullPath, "utf-8");
}

type TestFn = () => void;
const tests: Array<{ name: string; fn: TestFn }> = [];
function test(name: string, fn: TestFn) {
  tests.push({ name, fn });
}

test("1. SystemHealthPanel reports truthful Browser WASM / Geometric Authoritative status without fake ONNX claims", () => {
  const src = readFrontend("components/dashboard/system-health-panel.tsx");
  assert.equal(src.includes("MobileNetV3 ONNX Ready"), false, "Must not claim MobileNetV3 ONNX Ready");
  assert.equal(src.includes("ONNX model runtime verified"), false, "Must not claim ONNX model runtime verified");
  assert.equal(src.includes("Edge Cascade ONNX pipeline"), false, "Must not hardcode Edge Cascade ONNX pipeline");
  assert.ok(src.includes("WebAssembly"), "Must perform a real WebAssembly capability check");
  assert.ok(src.includes("Geometric Authoritative"), "Must label production pipeline as Geometric Authoritative");
});

test("2. Live Monitor & InspectorWorkspace do not mount no-op FaultInjectionToolbar or ExperimentSelector", () => {
  const inspector = readFrontend("components/monitor/inspector-workspace.tsx");
  const monitor = readFrontend("components/monitor/unified-live-monitor.tsx");

  assert.equal(inspector.includes("FaultInjectionToolbar"), false, "InspectorWorkspace must not reference FaultInjectionToolbar");
  assert.equal(inspector.includes("ExperimentSelector"), false, "InspectorWorkspace must not reference ExperimentSelector");
  assert.equal(monitor.includes("activeFault"), false, "UnifiedLiveMonitor must not maintain unused activeFault state");
  assert.equal(monitor.includes("FaultInjectionToolbar"), false, "UnifiedLiveMonitor must not reference FaultInjectionToolbar");
  assert.equal(monitor.includes("ExperimentSelector"), false, "UnifiedLiveMonitor must not reference ExperimentSelector");

  assert.ok(
    inspector.includes("/dashboard/synthetic-testbench"),
    "InspectorWorkspace should link users to the isolated Synthetic Testbench"
  );
  assert.ok(
    inspector.includes("Operational State") && inspector.includes("Drowsiness State"),
    "InspectorWorkspace must display Operational State and Drowsiness State separately"
  );
  assert.ok(
    inspector.includes("GEOMETRIC_AUTHORITATIVE"),
    "InspectorWorkspace setup tab should truthfully display GEOMETRIC_AUTHORITATIVE mode"
  );
});

test("3. Isolated Synthetic Testbench exists with clear non-live banner and Sidebar link", () => {
  const testbench = readFrontend("app/dashboard/synthetic-testbench/page.tsx");
  const sidebar = readFrontend("components/sidebar.tsx");

  assert.ok(
    testbench.includes("SYNTHETIC SIMULATION TESTBENCH"),
    "Synthetic testbench must carry a prominent SYNTHETIC SIMULATION TESTBENCH banner"
  );
  assert.ok(
    testbench.includes("NOT LIVE TELEMETRY"),
    "Synthetic testbench must explicitly state NOT LIVE TELEMETRY"
  );
  assert.ok(
    testbench.includes("FaultInjectionToolbar"),
    "Synthetic testbench must mount FaultInjectionToolbar"
  );
  assert.ok(
    sidebar.includes("/dashboard/synthetic-testbench"),
    "Sidebar must link to /dashboard/synthetic-testbench"
  );
});

test("4. Hardware console explicitly declares SIMULATED HARDWARE CONSOLE without fake active UART badge", () => {
  const hardware = readFrontend("app/dashboard/hardware/page.tsx");

  assert.equal(
    hardware.includes("UART LINK: 115200 BAUD [ACTIVE]"),
    false,
    "Hardware console must not display a fake active UART link badge"
  );
  assert.ok(
    hardware.includes("SIMULATED HARDWARE CONSOLE"),
    "Hardware console must display SIMULATED HARDWARE CONSOLE badge"
  );
  assert.ok(
    hardware.includes("No Physical Microcontroller Connected"),
    "Hardware console must display an explicit no-physical-hardware disclaimer"
  );
});

test("5. Dashboard displays independent Browser AI & Python AI calibration badges and aligned sensitivity thresholds", () => {
  const dashboard = readFrontend("app/dashboard/page.tsx");

  assert.equal(dashboard.includes('"MEDIUM"'), false, "Dashboard must not use obsolete MEDIUM sensitivity value");
  assert.equal(dashboard.includes("Using Fallback Defaults"), false, "Dashboard must not display Using Fallback Defaults");
  assert.ok(dashboard.includes("Browser AI"), "Dashboard must show Browser AI calibration badge");
  assert.ok(dashboard.includes("Python AI"), "Dashboard must show Python AI calibration badge");
  assert.ok(dashboard.includes("CALIBRATION REQUIRED"), "Dashboard must show CALIBRATION REQUIRED when uncalibrated");
  assert.ok(
    dashboard.includes("1.20") && dashboard.includes("1.50") && dashboard.includes("1.80"),
    "Dashboard must reflect exact HIGH (1.20s), STANDARD (1.50s), and RELAXED (1.80s) thresholds"
  );
});

test("6. Project Overview counts strictly the 4 spatial zones and reflects Decision 1 architecture", () => {
  const overview = readFrontend("app/dashboard/project-overview/page.tsx");

  assert.ok(
    overview.includes("CENTER") &&
      overview.includes("LEFT_MIRROR") &&
      overview.includes("RIGHT_MIRROR") &&
      overview.includes("DASHBOARD"),
    "Project Overview must explicitly whitelist the 4 spatial zones"
  );
  assert.ok(
    overview.includes("GEOMETRIC_AUTHORITATIVE"),
    "Project Overview must document GEOMETRIC_AUTHORITATIVE production pipeline"
  );
});

test("7. Driver Preferences Form and Onboarding Wizard standardize on RELAXED | STANDARD | HIGH without false CNN copy", () => {
  const settings = readFrontend("components/settings/driver-preferences-form.tsx");
  const onboarding = readFrontend("components/onboarding/onboarding-wizard.tsx");

  for (const [label, src] of [
    ["driver-preferences-form.tsx", settings],
    ["onboarding-wizard.tsx", onboarding],
  ] as const) {
    assert.ok(src.includes('"RELAXED"'), `${label} must include RELAXED preset`);
    assert.ok(src.includes('"STANDARD"'), `${label} must include STANDARD preset`);
    assert.ok(src.includes('"HIGH"'), `${label} must include HIGH preset`);
    assert.equal(src.includes('"MEDIUM"'), false, `${label} must not reference obsolete MEDIUM preset`);
    assert.equal(
      src.includes("Stage-2 CNN arbitration"),
      false,
      `${label} must not contain false Stage-2 CNN arbitration copy`
    );
  }

  assert.ok(
    settings.includes("1.80s") && settings.includes("1.50s") && settings.includes("1.20s"),
    "Settings form must show exact microsleep thresholds for RELAXED/STANDARD/HIGH"
  );
});

test("8. Benchmarks, Defense Guide, and Help have zero raw unrendered LaTeX and accurate runtime documentation", () => {
  const benchmarks = readFrontend("app/dashboard/benchmarks/page.tsx");
  const defense = readFrontend("app/dashboard/defense-guide/page.tsx");
  const help = readFrontend("app/dashboard/help/page.tsx");

  // Strip the intentional copy-to-clipboard LaTeX thesis snippet before checking rendered JSX
  const benchmarksJsx = benchmarks.replace(/const latexTableSnippet = `[\s\S]*?`;/, "");
  assert.equal(/\$[^$\n]+\$/.test(benchmarksJsx), false, "benchmarks/page.tsx JSX must not contain raw unrendered $...$ LaTeX");
  assert.ok(benchmarks.includes("GEOMETRIC_AUTHORITATIVE"), "benchmarks/page.tsx must clarify production GEOMETRIC_AUTHORITATIVE mode");

  assert.equal(/\$[^$\n]+\$/.test(defense), false, "defense-guide/page.tsx must not contain raw unrendered $...$ LaTeX");
  assert.ok(defense.includes("GEOMETRIC_AUTHORITATIVE"), "defense-guide/page.tsx must clarify production GEOMETRIC_AUTHORITATIVE mode");

  // Ignore PowerShell $env: when checking for raw LaTeX $...$
  const helpWithoutEnv = help.replace(/\$env:[A-Z0-9_]+/g, "");
  assert.equal(/\$[^$\n]+\$/.test(helpWithoutEnv), false, "help/page.tsx must not contain raw unrendered $...$ LaTeX");
  assert.equal(help.includes("MobileNetV3 ONNX Cascade:"), false, "help/page.tsx must not falsely claim ONNX cascade preflight verification");
  assert.equal(help.includes("python run.py"), false, "help/page.tsx must not reference nonexistent python run.py command");
  assert.ok(
    help.includes("uvicorn backend.app.main:app"),
    "help/page.tsx must document the working uvicorn backend.app.main:app command"
  );
  assert.ok(
    help.includes("/dashboard/synthetic-testbench"),
    "help/page.tsx must document the Synthetic Testbench portal"
  );
});

let passed = 0;
let failed = 0;
for (const t of tests) {
  try {
    t.fn();
    console.log(`  PASS  ${t.name}`);
    passed++;
  } catch (err) {
    console.error(`  FAIL  ${t.name}`);
    console.error("       ", err instanceof Error ? err.message : err);
    failed++;
  }
}

console.log(`\nPhase 5 UI Truthfulness Summary: ${passed}/${tests.length} passed, ${failed} failed.`);
if (failed > 0) {
  process.exit(1);
}
