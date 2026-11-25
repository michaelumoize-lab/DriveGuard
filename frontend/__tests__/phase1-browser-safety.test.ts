/**
 * DriveGuard Phase 1 Browser AI Safety & Mathematical Correctness Test Suite
 *
 * Verifies every Phase 1 acceptance criterion:
 * 1. profile-normalizer.ts: snake_case & camelCase zones, null safety, _poseMetadata unpacking, feasibility gate
 * 2. head-pose.ts: Float32Array(16) direct intake + 3D rigid Procrustes H = Q^T P accuracy (<= 1.0° on +25° yaw, -20° pitch, +15° roll)
 * 3. Uncalibrated Browser AI gate: 100 frames emit CALIBRATION_REQUIRED + INSUFFICIENT_DATA with 0 NORMAL frames
 * 4. Mirror dwell cap: LEFT_MIRROR at 2500ms suppressed (NORMAL), at 3600ms triggers DROWSINESS_WARNING (PERSISTENT_DISTRACTION)
 * 5. Alert storm elimination: 120 consecutive CRITICAL frames (10s @ 12fps) emit <= 2 alerts with unique eventIds
 * 6. Session finalization idempotency: queueSessionEnd() + stop() fires exactly 1 /api/sessions/end with real blink rate & finalizeToken
 * 7. Telemetry adapter truthfulness: real thresholds, PERCLOS coverage, GEOMETRIC_AUTHORITATIVE, NOT_APPLICABLE CNN
 */

import assert from "node:assert/strict";
import {
  AlertStateManager,
  BrowserDetectionEngine,
  CANONICAL_3D_FACE_POINTS,
  convertBrowserTelemetryToFrameTelemetry,
  DecisionEngine,
  DriverFeatureExtractor,
  DrowsinessState,
  estimateHeadPose,
  LEFT_EYE_INDICES,
  ManagedAlertState,
  matMul3x3,
  normalizeDriverProfile,
  NormalizedLandmark,
  OperationalState,
  PoseZone,
  ReasonCode,
  RIGHT_EYE_INDICES,
  RIGID_SOLVEPNP_INDICES,
  SOLVEPNP_LANDMARK_INDICES,
  TemporalAnalyzer,
} from "../lib/ai";
import { resolveEffectiveDisplayState } from "../types/telemetry";

let passed = 0;
let failed = 0;

async function test(name: string, fn: () => void | Promise<void>) {
  try {
    await fn();
    passed++;
    console.log(`  [PASS] ${name}`);
  } catch (err) {
    failed++;
    console.error(`  [FAIL] ${name}:`, err instanceof Error ? err.message : err);
  }
}

function buildRotationYXZ(yawDeg: number, pitchDeg: number, rollDeg: number): number[][] {
  const psi = (yawDeg * Math.PI) / 180.0;
  const theta = (pitchDeg * Math.PI) / 180.0;
  const phi = (rollDeg * Math.PI) / 180.0;

  const Ry = [
    [Math.cos(psi), 0, Math.sin(psi)],
    [0, 1, 0],
    [-Math.sin(psi), 0, Math.cos(psi)],
  ];
  const Rx = [
    [1, 0, 0],
    [0, Math.cos(theta), -Math.sin(theta)],
    [0, Math.sin(theta), Math.cos(theta)],
  ];
  const Rz = [
    [Math.cos(phi), -Math.sin(phi), 0],
    [Math.sin(phi), Math.cos(phi), 0],
    [0, 0, 1],
  ];
  return matMul3x3(matMul3x3(Ry, Rx), Rz);
}

/**
 * Construct synthetic 478 MediaPipe landmarks rotated by exact driver-centric (yaw, pitch, roll)
 * with configurable open/closed eye aperture (EAR).
 */
function makeSyntheticFaceLandmarks(options: {
  driverYawDeg?: number;
  driverPitchDeg?: number;
  driverRollDeg?: number;
  earTarget?: number;
  frameWidth?: number;
  frameHeight?: number;
} = {}): NormalizedLandmark[] {
  const {
    driverYawDeg = 0,
    driverPitchDeg = 0,
    driverRollDeg = 0,
    earTarget = 0.29,
    frameWidth = 640,
    frameHeight = 480,
  } = options;

  const lm: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0.0,
    visibility: 1.0,
  }));

  // Inverse of unmirrored driver-centric mapping in head-pose.ts:
  // yaw = -camYaw, pitch = -camPitch, roll = camRoll
  const R = buildRotationYXZ(-driverYawDeg, -driverPitchDeg, driverRollDeg);
  const scale = 1.8; // px per mm

  for (let i = 0; i < SOLVEPNP_LANDMARK_INDICES.length; i++) {
    const lmIdx = SOLVEPNP_LANDMARK_INDICES[i];
    const [px, py, pz] = CANONICAL_3D_FACE_POINTS[i];
    const qx = R[0][0] * px + R[0][1] * py + R[0][2] * pz;
    const qy = R[1][0] * px + R[1][1] * py + R[1][2] * pz;
    const qz = R[2][0] * px + R[2][1] * py + R[2][2] * pz;
    lm[lmIdx] = {
      x: 0.5 + (qx * scale) / frameWidth,
      y: 0.5 + (qy * scale) / frameHeight,
      z: (qz * scale) / frameWidth,
      visibility: 1.0,
    };
  }

  // Place clean eye contours with exact target EAR while preserving rotated 3D outer eye corners (idx 33 & 263)
  const aspectRatio = frameWidth / frameHeight;
  const horizNorm = 0.06; // horizontal eye span in normalized x
  const horizAspect = horizNorm * aspectRatio;
  const vertNorm = earTarget * horizAspect;

  const setEyeAnchoredAtIndex = (
    indices: readonly number[],
    anchorPos: 0 | 3,
    dirToOtherCorner: 1 | -1
  ) => {
    const anchor = lm[indices[anchorPos]];
    const otherX = anchor.x + dirToOtherCorner * horizNorm;
    const midX = (anchor.x + otherX) / 2;
    const cy = anchor.y;
    const z = anchor.z;
    // Preserve exact 3D-rotated outer eye corner (33 at pos 0, 263 at pos 3)
    const otherPos = anchorPos === 0 ? 3 : 0;
    lm[indices[otherPos]] = { x: otherX, y: cy, z, visibility: 1 };
    lm[indices[1]] = { x: midX - dirToOtherCorner * (horizNorm / 6), y: cy - vertNorm / 2, z, visibility: 1 };
    lm[indices[2]] = { x: midX + dirToOtherCorner * (horizNorm / 6), y: cy - vertNorm / 2, z, visibility: 1 };
    lm[indices[4]] = { x: midX + dirToOtherCorner * (horizNorm / 6), y: cy + vertNorm / 2, z, visibility: 1 };
    lm[indices[5]] = { x: midX - dirToOtherCorner * (horizNorm / 6), y: cy + vertNorm / 2, z, visibility: 1 };
  };

  setEyeAnchoredAtIndex(LEFT_EYE_INDICES, 0, 1);
  setEyeAnchoredAtIndex(RIGHT_EYE_INDICES, 3, -1);

  // Neutral closed mouth landmarks for MAR
  lm[61] = { x: 0.45, y: 0.62, z: 0, visibility: 1 };
  lm[291] = { x: 0.55, y: 0.62, z: 0, visibility: 1 };
  lm[13] = { x: 0.50, y: 0.615, z: 0, visibility: 1 };
  lm[14] = { x: 0.50, y: 0.625, z: 0, visibility: 1 };
  lm[81] = { x: 0.48, y: 0.615, z: 0, visibility: 1 };
  lm[178] = { x: 0.48, y: 0.625, z: 0, visibility: 1 };
  lm[311] = { x: 0.52, y: 0.615, z: 0, visibility: 1 };
  lm[402] = { x: 0.52, y: 0.625, z: 0, visibility: 1 };

  return lm;
}

function makeCalibratedPrismaProfile() {
  return {
    userId: "user_phase1",
    engineType: "BROWSER",
    isCalibrated: true,
    calibrationMode: "4_ZONE",
    centerEarMedian: 0.29,
    centerEarMadScaled: 0.016,
    neutralPitchDeg: 0.0,
    neutralYawDeg: 0.0,
    calibrationReliability: 0.92,
    zoneBaselines: {
      CENTER: {
        ear_median: 0.29,
        ear_mad_scaled: 0.016,
        pitch_median_deg: 0.0,
        yaw_median_deg: 0.0,
        sample_count: 60,
      },
      LEFT_MIRROR: {
        ear_median: 0.265,
        ear_mad_scaled: 0.018,
        pitch_median_deg: -1.0,
        yaw_median_deg: 24.0,
        sample_count: 55,
      },
      RIGHT_MIRROR: {
        ear_median: 0.262,
        ear_mad_scaled: 0.018,
        pitch_median_deg: -1.0,
        yaw_median_deg: -24.0,
        sample_count: 55,
      },
      DASHBOARD: {
        ear_median: 0.255,
        ear_mad_scaled: 0.019,
        pitch_median_deg: -12.0,
        yaw_median_deg: 0.0,
        sample_count: 50,
      },
      _poseMetadata: {
        neutral_rot_mat: [
          [1, 0, 0],
          [0, 1, 0],
          [0, 0, 1],
        ],
        calibration_is_mirrored: false,
      },
    },
  };
}

async function main() {
  console.log("Running DriveGuard Phase 1 Browser AI Safety & Math Tests...\n");

  await test("1. profile-normalizer handles Prisma snake_case zones, null inputs, and feasibility constraints", () => {
    const ok = normalizeDriverProfile(makeCalibratedPrismaProfile());
    assert.equal(ok.isCalibrated, true);
    assert.equal(ok.isFeasible, true);
    assert.ok(ok.profile);
    assert.equal(ok.profile.calibrationIsMirrored, false);
    assert.ok(ok.profile.neutralRotMat);

    const center = ok.zoneThresholds.get(PoseZone.CENTER);
    assert.ok(center);
    assert.ok(Number.isFinite(center.closureThreshold));
    assert.ok(center.closureThreshold > 0.08 && center.closureThreshold < center.reopenThreshold);
    assert.ok(center.reopenThreshold < center.earMedian);

    // Null / uncalibrated / null zoneBaselines never throw
    assert.equal(normalizeDriverProfile(null).isCalibrated, false);
    assert.equal(normalizeDriverProfile(undefined).isCalibrated, false);
    assert.equal(
      normalizeDriverProfile({ isCalibrated: false, zoneBaselines: null }).isCalibrated,
      false
    );
    // Infeasible tiny EAR baseline is rejected cleanly
    assert.equal(
      normalizeDriverProfile({
        isCalibrated: true,
        centerEarMedian: 0.13,
        centerEarMadScaled: 0.02,
        zoneBaselines: null,
      }).isFeasible,
      false
    );
  });

  await test("2. head-pose accepts Float32Array(16) directly and 3D Procrustes matches ground truth within <= 1.0 deg", () => {
    const lmNeutral = makeSyntheticFaceLandmarks({
      driverYawDeg: 0,
      driverPitchDeg: 0,
      driverRollDeg: 0,
    });

    // Build Float32Array(16) column-major matrix for driver yaw = +25°, pitch = -20°, roll = +15°
    const R = buildRotationYXZ(-25, 20, 15);
    const colMajor = new Float32Array([
      R[0][0], R[1][0], R[2][0], 0,
      R[0][1], R[1][1], R[2][1], 0,
      R[0][2], R[1][2], R[2][2], 0,
      10, -5, 450, 1,
    ]);

    const poseFromF32 = estimateHeadPose(
      lmNeutral,
      640,
      480,
      0,
      0,
      null,
      false,
      null,
      colMajor
    );
    assert.equal(poseFromF32.isValid, true);
    assert.ok(
      Math.abs(poseFromF32.yawDeg - 25.0) <= 0.5,
      `Float32Array yaw expected 25, got ${poseFromF32.yawDeg}`
    );
    assert.ok(
      Math.abs(poseFromF32.pitchDeg - -20.0) <= 0.5,
      `Float32Array pitch expected -20, got ${poseFromF32.pitchDeg}`
    );
    assert.ok(
      Math.abs(poseFromF32.rollDeg - 15.0) <= 0.5,
      `Float32Array roll expected 15, got ${poseFromF32.rollDeg}`
    );

    // 3D rigid Procrustes fallback (no facialTransformMatrix) on non-neutral rotated 3D landmarks
    const lmRotated = makeSyntheticFaceLandmarks({
      driverYawDeg: 25,
      driverPitchDeg: -20,
      driverRollDeg: 15,
    });
    const poseFrom3D = estimateHeadPose(lmRotated, 640, 480, 0, 0, null, false, null, null);
    assert.equal(poseFrom3D.isValid, true);
    assert.ok(
      Math.abs(poseFrom3D.yawDeg - 25.0) <= 1.0,
      `Procrustes yaw expected 25±1, got ${poseFrom3D.yawDeg}`
    );
    assert.ok(
      Math.abs(poseFrom3D.pitchDeg - -20.0) <= 1.0,
      `Procrustes pitch expected -20±1, got ${poseFrom3D.pitchDeg}`
    );
    assert.ok(
      Math.abs(poseFrom3D.rollDeg - 15.0) <= 1.0,
      `Procrustes roll expected 15±1, got ${poseFrom3D.rollDeg}`
    );
  });

  await test("3. Uncalibrated BrowserDetectionEngine emits CALIBRATION_REQUIRED + INSUFFICIENT_DATA across 100 frames (0 NORMAL)", () => {
    const engine = new BrowserDetectionEngine({
      driverId: "uncal_driver",
      profile: null,
    });
    const lmOpen = makeSyntheticFaceLandmarks({ earTarget: 0.29 });
    let normalCount = 0;

    for (let i = 1; i <= 100; i++) {
      const { decision, telemetry } = engine.evaluateFrameLandmarks(lmOpen, i * 80, 640, 480);
      const rawState: DrowsinessState = decision.state;
      if (rawState === DrowsinessState.NORMAL) normalCount++;
      assert.equal(decision.operationalState, OperationalState.CALIBRATION_REQUIRED);
      assert.equal(rawState, DrowsinessState.INSUFFICIENT_DATA);
      assert.equal(telemetry.operationalState, OperationalState.CALIBRATION_REQUIRED);

      const frame = convertBrowserTelemetryToFrameTelemetry(telemetry);
      assert.equal(resolveEffectiveDisplayState(frame, { isConnected: true }), "CALIBRATION_REQUIRED");
      assert.equal(frame.state, "MONITORING_UNAVAILABLE");
      assert.equal(frame.alert_state, null);
    }

    assert.equal(normalCount, 0);
    engine.dispose();
  });

  await test("4. Mirror dwell suppression caps at 3000ms (2500ms -> NORMAL, 3600ms -> DROWSINESS_WARNING PERSISTENT_DISTRACTION)", () => {
    const extractor = new DriverFeatureExtractor({
      profile: makeCalibratedPrismaProfile(),
      isMirrored: false,
    });
    const temporal = new TemporalAnalyzer();
    const decisionEngine = new DecisionEngine();

    const lmMirror = makeSyntheticFaceLandmarks({
      driverYawDeg: 24, // LEFT_MIRROR
      driverPitchDeg: 0,
      earTarget: 0.28,
    });

    let at2500Decision = null;
    let at3600Decision = null;

    for (let t = 100; t <= 3800; t += 100) {
      const features = extractor.extract(lmMirror, t, 640, 480);
      const tm = temporal.update(features);
      const dec = decisionEngine.evaluate(tm, {
        overallConfidence: 0.95,
        ocularViable: true,
        headPoseViable: true,
        yawnViable: true,
        isReliable: true,
        isOcularAvailable: true,
        isCalibrated: true,
        isCalibrating: false,
      });

      if (t === 2600) {
        // dwell = 2500ms (from t=100)
        assert.equal(tm.isMirrorCheckContext, true);
        at2500Decision = dec;
      }
      if (t === 3700) {
        // dwell = 3600ms (from t=100)
        assert.equal(tm.isMirrorCheckContext, false);
        at3600Decision = dec;
      }
    }

    assert.ok(at2500Decision);
    assert.equal(at2500Decision.state, DrowsinessState.NORMAL);
    assert.ok(at2500Decision.reasons.some((r) => r.code === ReasonCode.MIRROR_CHECK_CONTEXT));

    assert.ok(at3600Decision);
    assert.equal(at3600Decision.state, DrowsinessState.DROWSINESS_WARNING);
    assert.ok(at3600Decision.reasons.some((r) => r.code === ReasonCode.PERSISTENT_DISTRACTION));
  });

  await test("5. Alert storm regression: 120 consecutive CRITICAL frames (10s @ 12 FPS) emit <= 2 alerts with unique eventIds", () => {
    const emittedEventIds: string[] = [];
    const engine = new BrowserDetectionEngine({
      driverId: "storm_driver",
      profile: makeCalibratedPrismaProfile(),
      onSafetyAlert: (_decision, _telemetry, transition) => {
        if (transition) emittedEventIds.push(transition.eventId);
      },
    });

    const lmOpen = makeSyntheticFaceLandmarks({ earTarget: 0.29 });
    const lmClosed = makeSyntheticFaceLandmarks({ earTarget: 0.11 });

    // 5 open frames to establish baseline
    for (let i = 1; i <= 5; i++) {
      engine.evaluateFrameLandmarks(lmOpen, i * 83, 640, 480);
    }

    // 20 frames (1.66s) to enter CRITICAL_WARNING, then 120 consecutive CRITICAL frames (~10s @ 12 FPS)
    let ts = 500;
    for (let i = 0; i < 140; i++) {
      ts += 83;
      engine.evaluateFrameLandmarks(lmClosed, ts, 640, 480);
    }

    assert.ok(
      emittedEventIds.length >= 1 && emittedEventIds.length <= 2,
      `Expected 1-2 emitted critical alerts over 10s, got ${emittedEventIds.length}`
    );
    assert.equal(
      new Set(emittedEventIds).size,
      emittedEventIds.length,
      "Each emitted alert episode must have a unique eventId"
    );
    engine.dispose();
  });

  await test("6. Session end idempotency: queueSessionEnd() + stop() posts /api/sessions/end exactly once with blink rate & finalizeToken", async () => {
    const originalFetch = globalThis.fetch;
    const postedEndPayloads: Record<string, unknown>[] = [];
    globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = typeof input === "string" ? input : input.toString();
      if (url.includes("/api/sessions/end") && init?.body) {
        postedEndPayloads.push(JSON.parse(String(init.body)));
      }
      return new Response(JSON.stringify({ success: true }), { status: 200 });
    }) as typeof fetch;

    try {
      const engine = new BrowserDetectionEngine({
        driverId: "session_driver",
        sessionCode: "DRV-TEST-PHASE1",
        profile: makeCalibratedPrismaProfile(),
      });

      const lmOpen = makeSyntheticFaceLandmarks({ earTarget: 0.29 });
      const lmClosed = makeSyntheticFaceLandmarks({ earTarget: 0.12 });

      // Simulate 1 clean blink so totalBlinks === 1 and blinkCount60s === 1
      let ts = 100;
      for (let i = 0; i < 3; i++) {
        ts += 80;
        engine.evaluateFrameLandmarks(lmOpen, ts, 640, 480);
      }
      for (let i = 0; i < 2; i++) {
        ts += 80;
        engine.evaluateFrameLandmarks(lmClosed, ts, 640, 480);
      }
      for (let i = 0; i < 4; i++) {
        ts += 80;
        engine.evaluateFrameLandmarks(lmOpen, ts, 640, 480);
      }

      const firstQueued = engine.queueSessionEnd(12);
      const secondQueued = engine.queueSessionEnd(12);
      engine.stop();

      // Wait briefly for async flush microtask
      await new Promise((r) => setTimeout(r, 30));

      assert.equal(firstQueued, true);
      assert.equal(secondQueued, false);
      assert.equal(postedEndPayloads.length, 1, `Expected 1 /api/sessions/end POST, got ${postedEndPayloads.length}`);
      assert.equal(postedEndPayloads[0].sessionCode, "DRV-TEST-PHASE1");
      assert.ok(typeof postedEndPayloads[0].finalizeToken === "string");
      assert.equal(postedEndPayloads[0].totalBlinks, 1);
      assert.ok(
        Number(postedEndPayloads[0].averageBlinkRate) > 0,
        `Expected non-zero averageBlinkRate, got ${postedEndPayloads[0].averageBlinkRate}`
      );
      engine.dispose();
    } finally {
      globalThis.fetch = originalFetch;
    }
  });

  await test("7. Telemetry adapter reports truthful geometric mode and real metrics", () => {
    const engine = new BrowserDetectionEngine({
      driverId: "adapter_driver",
      profile: makeCalibratedPrismaProfile(),
    });
    const lmOpen = makeSyntheticFaceLandmarks({ earTarget: 0.29 });
    const { telemetry } = engine.evaluateFrameLandmarks(lmOpen, 1000, 640, 480);
    const frame = convertBrowserTelemetryToFrameTelemetry(telemetry);

    assert.equal(frame.engine_type, "BROWSER");
    assert.equal(frame.operational_state, "MONITORING_ACTIVE");
    assert.equal(frame.cascade.arbitration_winner, "GEOMETRIC_AUTHORITATIVE");
    assert.equal(frame.cascade.cnn_execution_mode, "NOT_APPLICABLE");
    assert.equal(frame.cascade.onnx_inference_executed, false);
    assert.ok(
      frame.eye_metrics.ear_closure_threshold > 0.12 &&
        frame.eye_metrics.ear_closure_threshold <= 0.22
    );
    assert.equal(frame.eye_metrics.total_blink_count, 0);
    engine.dispose();
  });

  await test("8. AlertStateManager preserves alert priority and emits verified recovery", () => {
    const mgr = new AlertStateManager();
    const warn = mgr.update({
      timestampMs: 1000,
      operationalState: OperationalState.MONITORING_ACTIVE,
      state: DrowsinessState.DROWSINESS_WARNING,
      severity: 2,
      reasons: [{ code: ReasonCode.PERCLOS_ELEVATED, message: "PERCLOS high" }],
      primaryReason: { code: ReasonCode.PERCLOS_ELEVATED, message: "PERCLOS high" },
      isMirrorCheckContext: false,
      alertActive: true,
      unresolvedAlertState: DrowsinessState.DROWSINESS_WARNING,
      historicalReason: { code: ReasonCode.PERCLOS_ELEVATED, message: "PERCLOS high" },
      isAlertRecovering: false,
      summaryText: "",
    });
    assert.ok(warn);
    assert.equal(warn.newState, ManagedAlertState.WARNING);

    // Same state inside cooldown -> null (no chatter)
    const dup = mgr.update({
      timestampMs: 1100,
      operationalState: OperationalState.MONITORING_ACTIVE,
      state: DrowsinessState.DROWSINESS_WARNING,
      severity: 2,
      reasons: [],
      primaryReason: null,
      isMirrorCheckContext: false,
      alertActive: true,
      unresolvedAlertState: DrowsinessState.DROWSINESS_WARNING,
      historicalReason: null,
      isAlertRecovering: false,
      summaryText: "",
    });
    assert.equal(dup, null);

    // Escalate to CRITICAL immediately
    const crit = mgr.update({
      timestampMs: 1200,
      operationalState: OperationalState.MONITORING_ACTIVE,
      state: DrowsinessState.CRITICAL_WARNING,
      severity: 3,
      reasons: [{ code: ReasonCode.SUSTAINED_EYE_CLOSURE, message: "Microsleep" }],
      primaryReason: { code: ReasonCode.SUSTAINED_EYE_CLOSURE, message: "Microsleep" },
      isMirrorCheckContext: false,
      alertActive: true,
      unresolvedAlertState: DrowsinessState.CRITICAL_WARNING,
      historicalReason: { code: ReasonCode.SUSTAINED_EYE_CLOSURE, message: "Microsleep" },
      isAlertRecovering: false,
      summaryText: "",
    });
    assert.ok(crit);
    assert.equal(crit.newState, ManagedAlertState.CRITICAL);

    // Verified recovery clears alert
    const rec = mgr.update({
      timestampMs: 4000,
      operationalState: OperationalState.MONITORING_ACTIVE,
      state: DrowsinessState.NORMAL,
      severity: 0,
      reasons: [{ code: ReasonCode.ALERT_NORMAL, message: "Recovered" }],
      primaryReason: { code: ReasonCode.ALERT_NORMAL, message: "Recovered" },
      isMirrorCheckContext: false,
      alertActive: false,
      unresolvedAlertState: null,
      historicalReason: null,
      isAlertRecovering: false,
      summaryText: "",
    });
    assert.ok(rec);
    assert.equal(rec.isRecovery, true);
    assert.equal(rec.newState, ManagedAlertState.NORMAL);
  });

  console.log(`\nPhase 1 Browser Safety Tests Complete: ${passed} passed, ${failed} failed.`);
  if (failed > 0) {
    process.exit(1);
  }
}

main();
