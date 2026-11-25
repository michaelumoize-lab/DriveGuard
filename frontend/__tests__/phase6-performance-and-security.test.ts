/**
 * Phase 6 Verification Tests — Frontend Performance, Real d(EAR)/dt,
 * Sensitivity Wiring, Mirror Suppression, Constant-Time JWT & Proxy Health Routing.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import {
  createBackendAccessToken,
  verifyBackendAccessToken,
} from "../lib/backend-auth";
import {
  isPublicHealthSubPath,
  resolveFastApiUpstreamPath,
} from "../lib/api/proxy-target";
import { computeEarDerivative } from "../components/engineering/kinematics-oscilloscope";
import {
  BrowserDetectionEngine,
  DecisionEngine,
  SENSITIVITY_PRESETS,
  normalizeSensitivityLevel,
  DrowsinessState,
  ReasonCode,
  NormalizedLandmark,
} from "../lib/ai";

const FRONTEND_ROOT = path.resolve(__dirname, "..");

function readFrontend(relPath: string): string {
  return fs.readFileSync(path.join(FRONTEND_ROOT, relPath), "utf-8");
}

type TestFn = () => void;
const tests: Array<{ name: string; fn: TestFn }> = [];
function test(name: string, fn: TestFn) {
  tests.push({ name, fn });
}

test("1. useCamera throttles diagnostics to 1 Hz (<= 2 Hz), uses requestVideoFrameCallback, and removes opticalQuality dep", () => {
  const src = readFrontend("hooks/use-camera.ts");

  assert.ok(
    src.includes("requestVideoFrameCallback"),
    "useCamera must use requestVideoFrameCallback for real video FPS measurement"
  );
  assert.ok(
    src.includes("setInterval(emitDiagnostics, 1000)"),
    "useCamera must throttle hardware diagnostics state updates to 1 Hz (1000 ms)"
  );
  assert.equal(
    src.includes("requestAnimationFrame(trackFps)"),
    false,
    "useCamera must not call setHardwareDiagnostics inside a 60 FPS requestAnimationFrame loop"
  );
  // Ensure opticalQuality is NOT in the diagnostics useEffect dependency array
  assert.equal(
    /\[cameraState,\s*idealFps,\s*idealHeight,\s*idealWidth,\s*facingMode,\s*resolution,\s*opticalQuality\]/.test(src),
    false,
    "useCamera diagnostics useEffect must NOT depend on opticalQuality"
  );
});

test("2. KinematicsOscilloscope removes idle 60 FPS rAF loop and computes genuine finite-difference d(EAR)/dt", () => {
  const oscSrc = readFrontend("components/engineering/kinematics-oscilloscope.tsx");
  const inspSrc = readFrontend("components/monitor/inspector-workspace.tsx");

  assert.equal(
    oscSrc.includes("requestAnimationFrame"),
    false,
    "KinematicsOscilloscope must not run an idle 60 FPS requestAnimationFrame canvas loop"
  );
  assert.equal(
    inspSrc.includes("-0.15"),
    false,
    "InspectorWorkspace must not pass hardcoded -0.15 dEarDt to KinematicsOscilloscope"
  );

  // Genuine ΔEAR / Δt = (0.18 - 0.28) / 0.100s = -1.00 / s
  const deriv = computeEarDerivative(0.28, 1000, 0.18, 1100);
  assert.ok(Math.abs(deriv - -1.0) < 1e-6, `Expected -1.0, got ${deriv}`);
  // Gap > 1000ms returns 0
  assert.equal(computeEarDerivative(0.28, 1000, 0.18, 2500), 0);
  // Duplicate / tiny dt returns 0
  assert.equal(computeEarDerivative(0.28, 1000, 0.18, 1000), 0);
});

test("3. verifyBackendAccessToken uses crypto.timingSafeEqual and rejects tampered tokens", () => {
  const src = readFrontend("lib/backend-auth.ts");
  assert.ok(src.includes("crypto.timingSafeEqual"), "verifyBackendAccessToken must use crypto.timingSafeEqual");

  const token = createBackendAccessToken({ userId: "driver_sec_1", role: "USER" }, 120);
  const verified = verifyBackendAccessToken(token);
  assert.ok(verified !== null && verified.sub === "driver_sec_1");

  // Tamper signature (same length and different length)
  const parts = token.split(".");
  const flipped = parts[2].slice(0, -1) + (parts[2].endsWith("a") ? "b" : "a");
  assert.equal(verifyBackendAccessToken(`${parts[0]}.${parts[1]}.${flipped}`), null);
  assert.equal(verifyBackendAccessToken(`${parts[0]}.${parts[1]}.short`), null);
});

test("4. BFF /api/v1/[...path] forwards /api/v1/health* to FastAPI root /health* routes", () => {
  assert.equal(isPublicHealthSubPath("health"), true);
  assert.equal(isPublicHealthSubPath("health/liveness"), true);
  assert.equal(isPublicHealthSubPath("health/readiness"), true);
  assert.equal(isPublicHealthSubPath("sessions"), false);

  assert.equal(resolveFastApiUpstreamPath("health"), "/health");
  assert.equal(resolveFastApiUpstreamPath("health/liveness"), "/health/liveness");
  assert.equal(resolveFastApiUpstreamPath("health/readiness"), "/health/readiness");
  assert.equal(resolveFastApiUpstreamPath("sessions/start"), "/api/v1/sessions/start");
  assert.equal(resolveFastApiUpstreamPath("livekit/token"), "/api/v1/livekit/token");
});

test("5. SENSITIVITY_PRESETS monotonicity, bounds, and BrowserDetectionEngine feature invariance", () => {
  const high = SENSITIVITY_PRESETS.HIGH;
  const std = SENSITIVITY_PRESETS.STANDARD;
  const rel = SENSITIVITY_PRESETS.RELAXED;

  assert.ok(high.sustainedClosureThresholdMs < std.sustainedClosureThresholdMs);
  assert.ok(std.sustainedClosureThresholdMs < rel.sustainedClosureThresholdMs);
  assert.ok(high.perclosWarningThresholdPct < std.perclosWarningThresholdPct);
  assert.ok(std.perclosWarningThresholdPct < rel.perclosWarningThresholdPct);
  assert.equal(high.recoveryCriticalMs, 2000.0);
  assert.equal(std.recoveryCriticalMs, 2000.0);
  assert.equal(rel.recoveryCriticalMs, 2000.0);
  assert.equal(normalizeSensitivityLevel("HIGHWAY"), "STANDARD");

  // Read canonical fixtures generated by Python harness to test BrowserDetectionEngine invariance
  const fixturePath = path.resolve(FRONTEND_ROOT, "../scripts/validation/parity_fixtures.json");
  const parsed = JSON.parse(fs.readFileSync(fixturePath, "utf-8"));
  const closureSeq = parsed.sequences.sustained_closure_presets;

  const featureSnapshots: Record<string, string[]> = {};
  const firstCriticalMs: Record<string, number | null> = {};

  for (const level of ["HIGH", "STANDARD", "RELAXED"] as const) {
    const engine = new BrowserDetectionEngine({
      profile: parsed.calibrated_profile,
      isMirrored: false,
      sensitivityLevel: level,
      mirrorSuppression: true,
    });
    const snaps: string[] = [];
    let critTs: number | null = null;

    for (const f of closureSeq.frames) {
      const { features, decision } = engine.evaluateFrameLandmarks(
        f.landmarks as NormalizedLandmark[],
        f.timestamp_ms,
        640,
        480,
        new Float32Array(f.facial_matrix_flat)
      );
      snaps.push(
        JSON.stringify([
          features.ear.compensatedEar,
          features.mar.mar,
          features.perclos.perclosPct,
          features.headPose.pitchDeg,
          features.headPose.yawDeg,
          features.headPose.rollDeg,
        ])
      );
      if (critTs === null && decision.state === DrowsinessState.CRITICAL_WARNING) {
        critTs = f.timestamp_ms;
      }
    }
    featureSnapshots[level] = snaps;
    firstCriticalMs[level] = critTs;
  }

  assert.deepEqual(featureSnapshots.HIGH, featureSnapshots.STANDARD);
  assert.deepEqual(featureSnapshots.STANDARD, featureSnapshots.RELAXED);
  assert.ok(firstCriticalMs.HIGH !== null && firstCriticalMs.STANDARD !== null && firstCriticalMs.RELAXED !== null);
  assert.ok(firstCriticalMs.HIGH! < firstCriticalMs.STANDARD! && firstCriticalMs.STANDARD! < firstCriticalMs.RELAXED!);

  // Verify DecisionEngine mirrorSuppression toggle directly
  const decOff = new DecisionEngine({ sensitivityLevel: "HIGH", mirrorSuppressionEnabled: false });
  const r = decOff.evaluate({
    timestampMs: 5000,
    activeClosureDurationMs: 0,
    activeOffRoadDwellMs: 2600,
    isMirrorCheckContext: true,
    isDashboardCheckContext: false,
    isNoddingMicrosleep: false,
    blinkCount60s: 10,
    blinkCount30s: 5,
    meanBlinkDurationMs60s: 180,
    prolongedBlinkCount60s: 0,
    severeBlinkCount60s: 0,
    meanRcr60s: 1.0,
    yawnCount60s: 0,
    yawnCount30s: 0,
    activeYawnDurationMs: 0,
    perclosPct: 2,
    perclosCoveragePct: 90,
    perclosIsReliable: true,
    effectivePerclosPct: 2,
    hasFace: true,
    faceLostDurationMs: 0,
    isEyeClosed: false,
    completedBlink: null,
    downwardDwellMs: 0,
  });
  assert.equal(r.state, DrowsinessState.DROWSINESS_WARNING);
  assert.equal(r.primaryReason?.code, ReasonCode.PERSISTENT_DISTRACTION);
  assert.equal(r.isMirrorCheckContext, false);
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

console.log(`\nPhase 6 Performance & Security Summary: ${passed}/${tests.length} passed, ${failed} failed.`);
if (failed > 0) {
  process.exit(1);
}
