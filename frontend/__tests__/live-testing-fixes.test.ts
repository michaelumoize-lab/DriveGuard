/**
 * DriveGuard — Regression Tests for Live-Testing Fixes:
 *   1. Local Python AI WebSocket telemetry resilience & HUD labels
 *   2. Browser AI OpenGL -> OpenCV head-pose basis conjugation (yaw/pitch/roll)
 *   3. Silent, race-free audio teardown on Stop / Pause / Mode-Switch / Unmount
 *
 * Run with: npx tsx __tests__/live-testing-fixes.test.ts
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";

import {
  convertOpenGLRotationToOpenCV,
  estimateHeadPose,
  extractEulerYXZ,
  isValidRotationMatrix,
} from "../lib/ai/head-pose";
import { DecisionEngine } from "../lib/ai/decision-engine";
import { ConfidenceTier } from "../lib/ai/confidence";
import {
  DrowsinessState,
  NormalizedLandmark,
  PoseZone,
  ReasonCode,
  TemporalMetrics,
} from "../lib/ai/types";
import { DriveGuardAudioEngine } from "../lib/audio/audio-engine";

type TestFn = () => void;
const tests: { name: string; fn: TestFn }[] = [];
function test(name: string, fn: TestFn) {
  tests.push({ name, fn });
}

function makeDummyLandmarks(): NormalizedLandmark[] {
  const arr: NormalizedLandmark[] = [];
  for (let i = 0; i < 468; i++) {
    arr.push({ x: 0.5, y: 0.5, z: 0.0 });
  }
  return arr;
}

/**
 * Pack a 3x3 rotation matrix R and translation (tx, ty, tz) into a
 * column-major 4x4 Float32Array (exact MediaPipe Tasks OpenGL format).
 */
function packColumnMajor4x4(R: number[][], tx: number, ty: number, tz: number): Float32Array {
  return new Float32Array([
    R[0][0], R[1][0], R[2][0], 0,
    R[0][1], R[1][1], R[2][1], 0,
    R[0][2], R[1][2], R[2][2], 0,
    tx,      ty,      tz,      1,
  ]);
}

/**
 * Pack a 3x3 rotation matrix R and translation (tx, ty, tz) into a
 * row-major 4x4 Float32Array (OpenCV format).
 */
function packRowMajor4x4(R: number[][], tx: number, ty: number, tz: number): Float32Array {
  return new Float32Array([
    R[0][0], R[0][1], R[0][2], tx,
    R[1][0], R[1][1], R[1][2], ty,
    R[2][0], R[2][1], R[2][2], tz,
    0,       0,       0,       1,
  ]);
}

function rotY(deg: number): number[][] {
  const rad = (deg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [
    [c, 0, s],
    [0, 1, 0],
    [-s, 0, c],
  ];
}

function rotX(deg: number): number[][] {
  const rad = (deg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [
    [1, 0, 0],
    [0, c, -s],
    [0, s, c],
  ];
}

function rotZ(deg: number): number[][] {
  const rad = (deg * Math.PI) / 180;
  const c = Math.cos(rad);
  const s = Math.sin(rad);
  return [
    [c, -s, 0],
    [s, c, 0],
    [0, 0, 1],
  ];
}

// ---------------------------------------------------------------------------
// ISSUE 2: MediaPipe Tasks OpenGL -> OpenCV Head Pose Conjugation
// ---------------------------------------------------------------------------

test("1. OpenGL -> OpenCV conjugation preserves SO(3) orthogonality and det=+1", () => {
  const R_gl = rotY(-25);
  const R_cv = convertOpenGLRotationToOpenCV(R_gl);
  assert.equal(isValidRotationMatrix(R_cv, 1e-6), true, "Conjugated matrix must remain valid SO(3)");
});

test("2. Browser AI OpenGL matrix: driver looking RIGHT 25° yields yaw ≈ -25° & RIGHT_MIRROR", () => {
  // In MediaPipe OpenGL (+X right, +Y up, +Z out of nose toward camera),
  // driver turning to driver's RIGHT turns the nose [0,0,+1] toward -X => R_y(-25°).
  const R_gl = rotY(-25);
  const m = packColumnMajor4x4(R_gl, 0.0, 0.0, -50.0);
  const pose = estimateHeadPose(makeDummyLandmarks(), 640, 480, 0, 0, null, false, null, m);

  assert.equal(pose.isValid, true);
  assert.ok(
    Math.abs(pose.yawDeg - -25) < 0.5,
    `Expected yawDeg ≈ -25° when driver looks right, got ${pose.yawDeg.toFixed(2)}°`
  );
  assert.equal(pose.zone, PoseZone.RIGHT_MIRROR, `Expected RIGHT_MIRROR, got ${pose.zone}`);
});

test("3. Browser AI OpenGL matrix: driver looking LEFT 25° yields yaw ≈ +25° & LEFT_MIRROR", () => {
  // Driver turning to driver's LEFT turns the nose [0,0,+1] toward +X => R_y(+25°).
  const R_gl = rotY(25);
  const m = packColumnMajor4x4(R_gl, 0.0, 0.0, -50.0);
  const pose = estimateHeadPose(makeDummyLandmarks(), 640, 480, 0, 0, null, false, null, m);

  assert.equal(pose.isValid, true);
  assert.ok(
    Math.abs(pose.yawDeg - 25) < 0.5,
    `Expected yawDeg ≈ +25° when driver looks left, got ${pose.yawDeg.toFixed(2)}°`
  );
  assert.equal(pose.zone, PoseZone.LEFT_MIRROR, `Expected LEFT_MIRROR, got ${pose.zone}`);
});

test("4. Browser AI OpenGL matrix: driver nodding DOWN 20° yields pitch ≈ -20° & DASHBOARD", () => {
  // In OpenGL (+Y up, +Z out of nose), nodding DOWN tilts nose [0,0,+1] toward -Y => R_x(+20°).
  const R_gl = rotX(20);
  const m = packColumnMajor4x4(R_gl, 0.0, 0.0, -50.0);
  const pose = estimateHeadPose(makeDummyLandmarks(), 640, 480, 0, 0, null, false, null, m);

  assert.equal(pose.isValid, true);
  assert.ok(
    Math.abs(pose.pitchDeg - -20) < 0.5,
    `Expected pitchDeg ≈ -20° when driver nods down, got ${pose.pitchDeg.toFixed(2)}°`
  );
  assert.equal(pose.zone, PoseZone.DASHBOARD, `Expected DASHBOARD, got ${pose.zone}`);
});

test("5. Browser AI OpenGL matrix: roll conjugation matches OpenCV C @ R_gl @ C identically", () => {
  const R_gl = rotZ(15);
  const expectedCv = convertOpenGLRotationToOpenCV(R_gl);
  const [, , expectedRawRoll] = extractEulerYXZ(expectedCv);

  const m = packColumnMajor4x4(R_gl, 0.0, 0.0, -50.0);
  const pose = estimateHeadPose(makeDummyLandmarks(), 640, 480, 0, 0, null, false, null, m);

  assert.equal(pose.isValid, true);
  assert.ok(
    Math.abs(pose.rollDeg - expectedRawRoll) < 0.5,
    `Expected rollDeg ≈ ${expectedRawRoll.toFixed(2)}°, got ${pose.rollDeg.toFixed(2)}°`
  );
});

test("6. Row-major OpenCV matrix (tz = +900 > 0) is NOT falsely conjugated", () => {
  // In OpenCV (+X right, +Y down, +Z into back of head), driver looking RIGHT rotates
  // back-of-head [0,0,+1] toward +X => R_y(+25°), rawYaw = +25°, yaw = -rawYaw = -25° => RIGHT_MIRROR.
  const R_cv = rotY(25);
  const m = packRowMajor4x4(R_cv, 0.0, 0.0, 900.0);
  const pose = estimateHeadPose(makeDummyLandmarks(), 640, 480, 0, 0, null, false, null, m);

  assert.equal(pose.isValid, true);
  assert.ok(
    Math.abs(pose.yawDeg - -25) < 0.5,
    `Expected row-major OpenCV yawDeg ≈ -25°, got ${pose.yawDeg.toFixed(2)}°`
  );
  assert.equal(pose.zone, PoseZone.RIGHT_MIRROR);
});

// ---------------------------------------------------------------------------
// ISSUE 3: Silent, Race-Free Audio Teardown
// ---------------------------------------------------------------------------

test("7. DriveGuardAudioEngine.resetSilent() cancels active CRITICAL loop without playing recovery chime", () => {
  const engine = new DriveGuardAudioEngine();
  let oscillatorsCreated = 0;

  // Inject minimal fake AudioContext to track oscillator synthesis and timer cleanup
  const fakeCtx = {
    state: "running",
    currentTime: 10.0,
    destination: {},
    createGain: () => ({
      gain: {
        value: 1,
        setValueAtTime: () => {},
        linearRampToValueAtTime: () => {},
        exponentialRampToValueAtTime: () => {},
      },
      connect: () => {},
      disconnect: () => {},
    }),
    createBiquadFilter: () => ({
      type: "lowpass",
      frequency: {
        value: 1000,
        setValueAtTime: () => {},
        linearRampToValueAtTime: () => {},
        exponentialRampToValueAtTime: () => {},
      },
      Q: {
        value: 1,
        setValueAtTime: () => {},
      },
      connect: () => {},
      disconnect: () => {},
    }),
    createOscillator: () => {
      oscillatorsCreated += 1;
      return {
        type: "sine",
        frequency: {
          setValueAtTime: () => {},
          linearRampToValueAtTime: () => {},
          exponentialRampToValueAtTime: () => {},
        },
        connect: () => {},
        start: () => {},
        stop: () => {},
        disconnect: () => {},
      };
    },
  };

  (engine as any).ctx = fakeCtx;
  (engine as any).masterGain = fakeCtx.createGain();
  (engine as any).unlocked = true;

  // Enter CRITICAL state — starts repeating criticalTimerId loop
  engine.setRiskState("CRITICAL", "evt_test_crit_1", Date.now());
  const oscAfterCritical = oscillatorsCreated;
  assert.ok(oscAfterCritical > 0, "CRITICAL state should synthesize alert oscillators");
  assert.notEqual(
    (engine as any).criticalTimerId,
    null,
    "CRITICAL state should arm repeating criticalTimerId"
  );
  assert.equal((engine as any).currentRiskState, "CRITICAL");

  // Silent reset must stop loop & reset state WITHOUT synthesizing a recovery chime
  engine.resetSilent();
  assert.equal(
    oscillatorsCreated,
    oscAfterCritical,
    "resetSilent() must NOT synthesize a recovery chime"
  );
  assert.equal((engine as any).criticalTimerId, null, "criticalTimerId must be cleared");
  assert.equal((engine as any).currentRiskState, "NORMAL", "currentRiskState must reset to NORMAL");
  assert.equal((engine as any).lastStateTimestampMs, null, "lastStateTimestampMs must reset");
  assert.equal((engine as any).currentTier, null, "currentTier must reset");
});

test("8. UnifiedLiveMonitor gates audio behind isMonitoringAudioAllowedRef and calls resetSilent() on teardown", () => {
  const src = fs.readFileSync(
    path.resolve(__dirname, "../components/monitor/unified-live-monitor.tsx"),
    "utf-8"
  );

  assert.ok(
    src.includes("const isMonitoringAudioAllowedRef = useRef<boolean>(false);"),
    "UnifiedLiveMonitor must track isMonitoringAudioAllowedRef"
  );
  assert.ok(
    src.includes("audioEngineRef.current?.resetSilent();"),
    "Unmount cleanup must call audioEngineRef.current?.resetSilent() via ref (never stale closure)"
  );
  const resetSilentCalls = (src.match(/\.resetSilent\(\)/g) || []).length;
  assert.ok(
    resetSilentCalls >= 5,
    `Expected resetSilent() across stop/pause/switch/start/unmount (>=5 calls), found ${resetSilentCalls}`
  );
});

// ---------------------------------------------------------------------------
// ISSUE 1: Local Python WS Resilience & HUD Truthfulness
// ---------------------------------------------------------------------------

test("9. Local Python hook preserves status poller during WS reconnect and exposes reconnectWebSocket", () => {
  const hookSrc = fs.readFileSync(
    path.resolve(__dirname, "../hooks/use-python-local-engine.ts"),
    "utf-8"
  );

  assert.ok(
    hookSrc.includes("closeWebSocket(false);"),
    "connectWebSocket() must pass clearStatusPoller=false so fallback poller survives WS retries"
  );
  assert.ok(
    hookSrc.includes("/api/v1/local/status"),
    "Hook must include /api/v1/local/status fallback telemetry poller"
  );
  assert.ok(
    hookSrc.includes("reconnectWebSocket,"),
    "Hook must export reconnectWebSocket for manual retry"
  );

  const hudSrc = fs.readFileSync(
    path.resolve(__dirname, "../components/monitor/live-telemetry-hud.tsx"),
    "utf-8"
  );
  assert.ok(
    hudSrc.includes("Local Python AI (127.0.0.1:8000)"),
    "HUD waiting state must show runtime-accurate Local Python AI copy instead of LiveKit copy"
  );
});

// ---------------------------------------------------------------------------
// ISSUE 4: DecisionEngine Cooldown primaryReason Consistency (Python & TS)
// ---------------------------------------------------------------------------

test("10. DecisionEngine cooldown retains elevated reason over DASHBOARD_CHECK_CONTEXT and prefers recent warning reason on step-down", () => {
  const de = new DecisionEngine();
  const baseMetrics = (tsMs: number, overrides: Partial<TemporalMetrics> = {}): TemporalMetrics => ({
    timestampMs: tsMs,
    activeClosureDurationMs: 0,
    activeOffRoadDwellMs: 0,
    isMirrorCheckContext: false,
    isDashboardCheckContext: false,
    isNoddingMicrosleep: false,
    blinkCount60s: 4,
    blinkCount30s: 2,
    meanBlinkDurationMs60s: 180,
    prolongedBlinkCount60s: 0,
    severeBlinkCount60s: 0,
    meanRcr60s: 1.0,
    yawnCount60s: 0,
    yawnCount30s: 0,
    activeYawnDurationMs: 0,
    perclosPct: 4.0,
    effectivePerclosPct: 4.0,
    perclosCoveragePct: 100.0,
    perclosIsReliable: true,
    hasFace: true,
    faceLostDurationMs: 0,
    isEyeClosed: false,
    completedBlink: null,
    downwardDwellMs: 0,
    ...overrides,
  });
  const goodConf = {
    overallConfidence: 1.0,
    tier: ConfidenceTier.HIGH,
    isReliable: true,
    headPoseViable: true,
    ocularViable: true,
    yawnViable: true,
    isHeadPoseAvailable: true,
    isOcularAvailable: true,
    isYawnAvailable: true,
    degradedReasons: [],
  };

  // 1. Trigger CRITICAL via NODDING_MICROSLEEP at t=1000ms
  const rCrit = de.evaluate(
    baseMetrics(1000, { isNoddingMicrosleep: true, activeClosureDurationMs: 600, isEyeClosed: true }),
    goodConf
  );
  assert.equal(rCrit.state, DrowsinessState.CRITICAL_WARNING);
  assert.equal(rCrit.primaryReason?.code, ReasonCode.NODDING_MICROSLEEP);

  // 2. Eyes reopen during dashboard glance at t=1500ms (still in 2000ms CRITICAL cooldown)
  // Must NOT leak DASHBOARD_CHECK_CONTEXT as primaryReason while CRITICAL_WARNING is held!
  const rHoldDash = de.evaluate(
    baseMetrics(1500, { isDashboardCheckContext: true, activeOffRoadDwellMs: 500 }),
    goodConf
  );
  assert.equal(rHoldDash.state, DrowsinessState.CRITICAL_WARNING);
  assert.equal(
    rHoldDash.primaryReason?.code,
    ReasonCode.NODDING_MICROSLEEP,
    "Cooldown-held CRITICAL_WARNING must retain elevated reason instead of DASHBOARD_CHECK_CONTEXT"
  );

  // 3. Step down to DROWSINESS_WARNING driven by PROLONGED_BLINK across continuous frames (>= 2000ms cooldown)
  for (let ts = 1700; ts <= 3900; ts += 200) {
    de.evaluate(
      baseMetrics(ts, { prolongedBlinkCount60s: 3, meanBlinkDurationMs60s: 650 }),
      goodConf
    );
  }
  const rWarn = de.evaluate(
    baseMetrics(4100, { prolongedBlinkCount60s: 3, meanBlinkDurationMs60s: 650 }),
    goodConf
  );
  assert.equal(rWarn.state, DrowsinessState.DROWSINESS_WARNING);
  assert.equal(rWarn.primaryReason?.code, ReasonCode.PROLONGED_BLINK);

  // 4. Eyes normal during DROWSINESS_WARNING cooldown at t=4500ms:
  // primaryReason must show PROLONGED_BLINK (most recent elevated reason), NOT stale NODDING_MICROSLEEP,
  // while historicalReason still preserves NODDING_MICROSLEEP for AlertStateManager.
  const rWarnHold = de.evaluate(baseMetrics(4500), goodConf);
  assert.equal(rWarnHold.state, DrowsinessState.DROWSINESS_WARNING);
  assert.equal(
    rWarnHold.primaryReason?.code,
    ReasonCode.PROLONGED_BLINK,
    "Held DROWSINESS_WARNING should display most recent warning reason (PROLONGED_BLINK)"
  );
  assert.equal(
    rWarnHold.historicalReason?.code,
    ReasonCode.NODDING_MICROSLEEP,
    "historicalReason must preserve highest unresolved alert reason for AlertStateManager"
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

console.log(`\nLive-Testing Fixes Regression Summary: ${passed}/${tests.length} passed, ${failed} failed.`);
if (failed > 0) {
  process.exit(1);
}
