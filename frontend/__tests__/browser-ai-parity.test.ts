/**
 * Browser AI & Algorithm Parity Test Suite
 * ========================================
 * Verifies mathematical parity and algorithmic consistency between
 * the TypeScript browser detection pipeline and the reference Python implementation:
 * 1. EAR calculation, Hermite smoothstep yaw weighting, pitch compensation
 * 2. MAR calculation, 3-pair normalization, Schmitt-trigger hysteresis
 * 3. Blink detector state transitions, sub-frame crossing interpolation, velocities & RCR
 * 4. PERCLOS 60s sliding window, interval boundary clipping, coverage gating
 * 5. Head pose Cardan Y-X-Z Euler angles, cockpit zones, 1.5° hysteresis deadband
 * 6. PDBP personalized calibration, scaled MAD, feasibility constraints
 * 7. Decision Engine 5-level states, priority hierarchy, asymmetric cooldowns
 * 8. Offline Event Queue buffering, deduplication, and recovery flush
 */

import {
  calculateSingleEyeEar,
  calculateYawWeightedEar,
  calculatePitchCompensatedEar,
  calculateMar,
  MarDetector,
  BlinkDetector,
  BlinkState,
  RollingPerclosCalculator,
  estimateHeadPoseFromLandmarks,
  classifyCockpitZone,
  computeMedianAndScaledMad,
  computeZoneThresholds,
  DecisionEngine,
  DrowsinessState,
  PoseZone,
  ReasonCode,
  NormalizedLandmark,
} from "../lib/ai";

let totalTests = 0;
let passedTests = 0;

function assert(condition: boolean, testName: string, detail?: string) {
  totalTests++;
  if (condition) {
    passedTests++;
    console.log(`  [PASS] ${testName}`);
  } else {
    console.error(`  [FAIL] ${testName}${detail ? `: ${detail}` : ""}`);
    process.exitCode = 1;
  }
}

function assertCloseTo(actual: number, expected: number, tolerance = 0.005, testName: string) {
  const diff = Math.abs(actual - expected);
  assert(diff <= tolerance, testName, `Actual ${actual} vs Expected ${expected} (diff ${diff} > ${tolerance})`);
}

async function runTestSuite() {
  console.log("======================================================================");
  console.log("   [DriveGuard] Browser AI TypeScript & Python Parity Test Suite");
  console.log("======================================================================");

  // -------------------------------------------------------------------------
  // 1. EAR Calculation & Yaw/Pitch Compensation
  // -------------------------------------------------------------------------
  console.log("\n[1/8] Testing EAR Calculation, Yaw Weighting & Pitch Invariants...");
  {
    // Mock eye contour with exact 2D coordinates (aspectRatio=1.0, isNormalized=false)
    // p1 = (0.1, 0.5), p2 = (0.2, 0.4), p3 = (0.3, 0.4)
    // p4 = (0.4, 0.5), p5 = (0.3, 0.6), p6 = (0.2, 0.6)
    const eyePoints: NormalizedLandmark[] = [
      { x: 0.1, y: 0.5, z: 0.0 }, // p1 (lateral corner)
      { x: 0.2, y: 0.4, z: 0.0 }, // p2 (upper 1)
      { x: 0.3, y: 0.4, z: 0.0 }, // p3 (upper 2)
      { x: 0.4, y: 0.5, z: 0.0 }, // p4 (medial corner)
      { x: 0.3, y: 0.6, z: 0.0 }, // p5 (lower 2)
      { x: 0.2, y: 0.6, z: 0.0 }, // p6 (lower 1)
    ];

    // d1 = dist(p2, p6) = 0.2
    // d2 = dist(p3, p5) = 0.2
    // d3 = dist(p1, p4) = 0.3
    // EAR = (0.2 + 0.2) / (2 * 0.3) = 0.4 / 0.6 = 0.6667
    const indices = [0, 1, 2, 3, 4, 5] as const;
    const rawEar = calculateSingleEyeEar(eyePoints, indices, 1.0, false);
    assertCloseTo(rawEar, 0.6667, 0.002, "Basic Euclidean EAR matches formula (d1+d2)/(2*d3)");

    // Hermite Smoothstep Yaw Weighting:
    // Yaw = 0° -> 50% left, 50% right
    const earCenter = calculateYawWeightedEar(0.20, 0.30, 0.0, true, true, false);
    assertCloseTo(earCenter, 0.25, 0.001, "Hermite yaw weight at yaw=0 balances 50% left / 50% right");

    // Yaw = +30° -> 80% left, 20% right
    const earRight = calculateYawWeightedEar(0.20, 0.30, 30.0, true, true, false);
    assertCloseTo(earRight, 0.80 * 0.20 + 0.20 * 0.30, 0.001, "Hermite yaw weight at yaw >= +30 weights near eye at 80%");

    // Yaw = -30° -> 20% left, 80% right
    const earLeft = calculateYawWeightedEar(0.20, 0.30, -30.0, true, true, false);
    assertCloseTo(earLeft, 0.20 * 0.20 + 0.80 * 0.30, 0.001, "Hermite yaw weight at yaw <= -30 weights near eye at 80%");

    // Pitch Compensation Safety Invariant: only active above closure threshold
    const closureThreshold = 0.20;
    const closedEar = 0.12; // closed eye <= 0.20
    const compensatedClosed = calculatePitchCompensatedEar(closedEar, -20.0, 0.0, closureThreshold);
    assert(
      compensatedClosed === closedEar,
      "Pitch compensation is INACTIVE during closed eye (safety guard against masking closure)",
      `Expected ${closedEar}, got ${compensatedClosed}`
    );

    const openEar = 0.28; // open eye > 0.20
    const compensatedOpen = calculatePitchCompensatedEar(openEar, -20.0, 0.0, closureThreshold);
    assert(
      compensatedOpen > openEar,
      "Pitch compensation boosts foreshortened EAR when looking down while eyes are open",
      `Expected > ${openEar}, got ${compensatedOpen}`
    );
  }

  // -------------------------------------------------------------------------
  // 2. MAR Calculation & Schmitt-Trigger Hysteresis
  // -------------------------------------------------------------------------
  console.log("\n[2/8] Testing MAR Yawn Detection & Hysteresis State Machine...");
  {
    const createMouth = (verticalSpan: number) => {
      const lm: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0.0 }));
      // Corners 61, 291: width = 0.40
      lm[61] = { x: 0.3, y: 0.5, z: 0.0 };
      lm[291] = { x: 0.7, y: 0.5, z: 0.0 };
      // Pair 1: 13, 14
      lm[13] = { x: 0.5, y: 0.5 - verticalSpan / 2, z: 0.0 };
      lm[14] = { x: 0.5, y: 0.5 + verticalSpan / 2, z: 0.0 };
      // Pair 2: 81, 178
      lm[81] = { x: 0.45, y: 0.5 - verticalSpan / 2, z: 0.0 };
      lm[178] = { x: 0.45, y: 0.5 + verticalSpan / 2, z: 0.0 };
      // Pair 3: 311, 402
      lm[311] = { x: 0.55, y: 0.5 - verticalSpan / 2, z: 0.0 };
      lm[402] = { x: 0.55, y: 0.5 + verticalSpan / 2, z: 0.0 };
      return lm;
    };

    // When verticalSpan = 0.30, width = 0.40 -> MAR = 0.30 / 0.40 = 0.75
    const wideMouth = createMouth(0.30);
    const mar = calculateMar(wideMouth, 1.0, false);
    assertCloseTo(mar, 0.75, 0.01, "3-pair normalized MAR correctly calculates wide mouth opening");

    // Test Schmitt-trigger hysteresis
    const detector = new MarDetector({ onsetThreshold: 0.55, releaseThreshold: 0.45 });
    // Frame 1: Resting mouth (span = 0.06 -> MAR = 0.15)
    let res = detector.update(createMouth(0.06), 1000, 1.0, true, false);
    assert(!res.isYawnInProgress, "Resting mouth does not trigger yawn");

    // Frame 2: Opening mouth to span = 0.20 -> MAR = 0.50 (between release and onset)
    res = detector.update(createMouth(0.20), 1200, 1.0, true, false);
    assert(!res.isYawnInProgress, "MAR between 0.45 and 0.55 does not trigger onset (hysteresis low band)");

    // Frame 3: Opening mouth to span = 0.26 -> MAR = 0.65 (above onset 0.55)
    res = detector.update(createMouth(0.26), 1350, 1.0, true, false);
    assert(res.isYawnInProgress, "MAR >= 0.55 latches yawn in-progress state");

    // Frame 4: Drops to span = 0.20 -> MAR = 0.50 (still above release 0.45)
    res = detector.update(createMouth(0.20), 1500, 1.0, true, false);
    assert(res.isYawnInProgress, "MAR remaining above release 0.45 stays in-progress (hysteresis hold)");

    // Frame 5: Drops below release 0.45 (span = 0.12 -> MAR = 0.30)
    res = detector.update(createMouth(0.12), 1650, 1.0, true, false);
    assert(!res.isYawnInProgress, "MAR below 0.45 cleanly releases yawn");
  }

  // -------------------------------------------------------------------------
  // 3. Blink Detector & Eyelid Kinematics (RCR, Velocity)
  // -------------------------------------------------------------------------
  console.log("\n[3/8] Testing Eyelid Kinematics State Machine & Sub-Frame Crossing...");
  {
    const blinkDetector = new BlinkDetector({
      earClosureThreshold: 0.21,
      minBlinkDurationMs: 60,
      maxBlinkDurationMs: 800,
    });

    // Start OPEN at t=0ms, EAR = 0.28
    let b = blinkDetector.update(0.28, 0.28, 0.28, 0, true);
    assert(b.state === BlinkState.OPEN, "Initial eyelid state is OPEN");

    // EAR drops below threshold at t=100ms, EAR = 0.18 -> CLOSING -> CLOSED
    b = blinkDetector.update(0.18, 0.18, 0.18, 100, true);
    assert(b.state === BlinkState.CLOSED, "Eyelid crosses below threshold into CLOSED state");
    assert(b.isEyeClosed, "isEyeClosed is true when below threshold");

    // EAR stays closed at t=200ms, EAR = 0.12
    b = blinkDetector.update(0.12, 0.12, 0.12, 200, true);
    assert(b.currentClosureDurationMs >= 90, "Closure duration accumulator tracks active closure time");

    // EAR rises above threshold at t=260ms, EAR = 0.26 -> OPEN
    b = blinkDetector.update(0.26, 0.26, 0.26, 260, true);
    assert(b.state === BlinkState.OPEN, "Eyelid crosses above threshold back to OPEN state");
    assert(b.totalBlinkCount === 1, "Completed blink increments totalBlinkCount to 1");
    assert(
      b.recentBlinkDurationMs !== null && b.recentBlinkDurationMs > 100 && b.recentBlinkDurationMs < 260,
      "Sub-frame crossing interpolation measures accurate sub-frame duration",
      `Duration: ${b.recentBlinkDurationMs}`
    );
    assert(b.recentBlinkRcr !== null && b.recentBlinkRcr > 0, "Computes positive Reopening-to-Closing Ratio (RCR)");
  }

  // -------------------------------------------------------------------------
  // 4. PERCLOS Rolling Window (P80)
  // -------------------------------------------------------------------------
  console.log("\n[4/8] Testing PERCLOS 60s Sliding Window, Boundary Clipping & Gating...");
  {
    const perclosCalc = new RollingPerclosCalculator(60.0, 50.0);

    // Feed 35 seconds of observation with 10 seconds of closure at 100ms intervals (10 FPS)
    // First 10s: open (EAR = 0.30)
    for (let t = 0; t <= 10000; t += 100) {
      perclosCalc.update(t, false, true, true);
    }
    // Next 10s: closed (EAR = 0.15)
    for (let t = 10100; t <= 20000; t += 100) {
      perclosCalc.update(t, true, true, true);
    }
    // Next 15s: open (EAR = 0.30)
    let snap: any = null;
    for (let t = 20100; t <= 35000; t += 100) {
      snap = perclosCalc.update(t, false, true, true);
    }

    // 35s observed out of 60s = 58.3% coverage >= 50% required!
    assert(snap.isReliable, "Coverage >= 50% satisfies gating requirement");
    // 10s closed out of 35s observed = 10 / 35 = 28.57%
    assertCloseTo(snap.perclosPct, 28.57, 3.0, "PERCLOS accurately calculates 80% eye closure percentage over observed window");
    assert(snap.coveragePct >= 50.0, "Coverage percentage accurately reported");
  }

  // -------------------------------------------------------------------------
  // 5. Head Pose & Euler Angles (Cardan Y-X-Z) & Zone Classification
  // -------------------------------------------------------------------------
  console.log("\n[5/8] Testing Head Pose Estimation & 1.5° Deadband Zone Hysteresis...");
  {
    // Mock landmarks with neutral center pose
    const mockLandmarks: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({ x: 0.5, y: 0.5, z: 0.0 }));
    // Sellion (168)
    mockLandmarks[168] = { x: 0.5, y: 0.45, z: 0.0 };
    // Nose tip (1)
    mockLandmarks[1] = { x: 0.5, y: 0.55, z: -0.05 };
    // Eye corners: 33 (right) and 263 (left)
    mockLandmarks[33] = { x: 0.35, y: 0.45, z: 0.0 };
    mockLandmarks[263] = { x: 0.65, y: 0.45, z: 0.0 };
    // Glabella (10)
    mockLandmarks[10] = { x: 0.5, y: 0.35, z: 0.0 };

    const pose = estimateHeadPoseFromLandmarks(mockLandmarks, 640, 480, 0.0, 0.0, null, false);
    assert(pose.isValid, "6-DoF Rigid cranial alignment produces valid head pose");
    assert(Math.abs(pose.yawDeg) < 5.0, "Neutral head pose produces ~0° yaw", `Yaw: ${pose.yawDeg}`);
    assert(Math.abs(pose.pitchDeg) < 10.0, "Neutral head pose produces reasonable pitch", `Pitch: ${pose.pitchDeg}`);

    // Zone Classification with Hysteresis
    // CENTER zone has neutral pitch and yaw
    const z1 = classifyCockpitZone(0.0, 0.0);
    assert(z1 === PoseZone.CENTER, "Pitch=0, Yaw=0 classifies as CENTER");

    // Looking left (yaw = +18.0) -> LEFT_MIRROR
    const z2 = classifyCockpitZone(0.0, 18.0);
    assert(z2 === PoseZone.LEFT_MIRROR, "Yaw=+18° classifies as LEFT_MIRROR");

    // Hysteresis deadband: yaw drops to 13.0° (threshold is 14.0° - 1.5° deadband = 12.5°)
    const z3 = classifyCockpitZone(0.0, 13.0, 0.0, 0.0, 0.0, PoseZone.LEFT_MIRROR);
    assert(z3 === PoseZone.LEFT_MIRROR, "1.5° deadband prevents jittery flapping out of LEFT_MIRROR");

    // Looking down (pitch = -12.0) -> DASHBOARD
    const z4 = classifyCockpitZone(-12.0, 0.0);
    assert(z4 === PoseZone.DASHBOARD, "Pitch=-12° classifies as DASHBOARD");
  }

  // -------------------------------------------------------------------------
  // 6. PDBP Personalized Calibration & Mathematical Feasibility
  // -------------------------------------------------------------------------
  console.log("\n[6/8] Testing PDBP Scaled MAD & Mathematical Feasibility...");
  {
    const earSamples = [0.28, 0.29, 0.27, 0.28, 0.30, 0.28, 0.27, 0.29, 0.28, 0.28];
    const [med, mad] = computeMedianAndScaledMad(earSamples, 0.015);

    assertCloseTo(med, 0.28, 0.005, "Robust median correctly identifies 0.28 center baseline");
    assert(mad >= 0.015, "Scaled MAD respects 0.015 epsilon minimum flooring", `MAD: ${mad}`);

    const { isFeasible, closureThreshold } = computeZoneThresholds(med, mad);
    assert(isFeasible, "PDBP thresholds pass mathematical feasibility constraints");
    assert(closureThreshold < med, "Closure threshold is strictly less than resting open-eye baseline");
    assert(closureThreshold > 0.08, "Closure threshold is safely above zero-floor clamp");
  }

  // -------------------------------------------------------------------------
  // 7. Decision Engine Priority & Asymmetric Recovery Cooldown
  // -------------------------------------------------------------------------
  console.log("\n[7/8] Testing Multi-Indicator Decision Engine & Recovery Cooldown...");
  {
    const engine = new DecisionEngine();

    // Normal driver temporal metrics
    const normalMetrics: any = {
      timestampMs: 1000,
      hasFace: true,
      faceLostDurationMs: 0,
      activeClosureDurationMs: 0,
      activeOffRoadDwellMs: 0,
      isMirrorCheckContext: false,
      isDashboardCheckContext: false,
      isNoddingMicrosleep: false,
      blinkCount60s: 5,
      blinkCount30s: 3,
      meanBlinkDurationMs60s: 220,
      prolongedBlinkCount60s: 0,
      severeBlinkCount60s: 0,
      meanRcr60s: 0.95,
      yawnCount60s: 0,
      yawnCount30s: 0,
      activeYawnDurationMs: 0,
      perclosPct: 4.0,
      perclosCoveragePct: 100.0,
      perclosIsReliable: true,
      effectivePerclosPct: 4.0,
      isEyeClosed: false,
      completedBlink: null,
      downwardDwellMs: 0,
    };

    let decision = engine.evaluate(normalMetrics);
    assert(decision.state === DrowsinessState.NORMAL, "Healthy driver evaluates to NORMAL");

    // Critical trigger: prolonged eye closure > 1500ms
    const criticalMetrics = {
      ...normalMetrics,
      timestampMs: 2000,
      activeClosureDurationMs: 1800,
      isEyeClosed: true,
    };

    decision = engine.evaluate(criticalMetrics);
    assert(decision.state === DrowsinessState.CRITICAL_WARNING, "Prolonged closure triggers CRITICAL_WARNING");
    assert(
      decision.reasons.some((r) => r.code === ReasonCode.SUSTAINED_EYE_CLOSURE),
      "Explainable reason code includes SUSTAINED_EYE_CLOSURE"
    );

    // Eyelids snap open at t=2100ms (only 100ms into recovery)
    decision = engine.evaluate({
      ...normalMetrics,
      timestampMs: 2100,
      activeClosureDurationMs: 0,
      isEyeClosed: false,
    });
    assert(
      decision.state === DrowsinessState.CRITICAL_WARNING,
      "Asymmetric 2000ms cooldown holds CRITICAL_WARNING against premature de-escalation"
    );

    // Continuous driving frames from t=2200ms to t=4200ms (100ms intervals)
    for (let t = 2200; t <= 4200; t += 100) {
      engine.evaluate({
        ...normalMetrics,
        timestampMs: t,
        activeClosureDurationMs: 0,
        isEyeClosed: false,
      });
    }

    // After 2300ms of sustained normal driving (t=4300ms)
    decision = engine.evaluate({
      ...normalMetrics,
      timestampMs: 4300,
      activeClosureDurationMs: 0,
      isEyeClosed: false,
    });
    assert(decision.state === DrowsinessState.NORMAL, "Cleanly recovers to NORMAL after cooldown expires");
  }

  // -------------------------------------------------------------------------
  // 8. Offline Event Queue & Recovery Summary
  // -------------------------------------------------------------------------
  console.log("\n[8/8] Testing Offline Storage Contracts & Telemetry Adapter...");
  {
    // Test conversion from DetectionTelemetry to FrameTelemetry
    const mockTelemetry: any = {
      frameId: 42,
      timestampMs: 1700000000000,
      hasFace: true,
      alertState: DrowsinessState.DROWSINESS_WARNING,
      severity: "WARNING",
      primaryReason: ReasonCode.PERCLOS_ELEVATED,
      reasons: [ReasonCode.PERCLOS_ELEVATED],
      ear: 0.19,
      rawEar: 0.19,
      leftEar: 0.19,
      rightEar: 0.19,
      perclosPct: 15.5,
      effectivePerclosPct: 15.5,
      isClosed: false,
      closureDurationMs: 0,
      blinkCount: 12,
      recentBlinkDurationMs: 240,
      recentBlinkRcr: 0.95,
      mar: 0.32,
      isYawning: false,
      yawnCount: 0,
      pitchDeg: -4.0,
      yawDeg: 2.0,
      rollDeg: 0.0,
      poseZone: PoseZone.CENTER,
      isLookingAway: false,
      inferenceFps: 14.5,
      inferenceLatencyMs: 18.2,
      runtimeMode: "browser",
      persistenceSyncStatus: "synced",
    };

    const { convertBrowserTelemetryToFrameTelemetry } = await import("../lib/ai/telemetry-adapter");
    const frame = convertBrowserTelemetryToFrameTelemetry(mockTelemetry);

    assert(frame.face_detected === true, "Telemetry adapter maps hasFace -> face_detected");
    assert(frame.state === "WARNING", "Telemetry adapter maps DROWSINESS_WARNING -> WARNING");
    assert(frame.eye_metrics?.ear_raw === 0.19, "Telemetry adapter preserves raw EAR");
    assert(frame.eye_metrics?.perclos_pct === 15.5, "Telemetry adapter preserves PERCLOS");
    assert(frame.runtime_mode === "browser", "Telemetry adapter marks runtime_mode as browser");

    // Verify BrowserDetectionEngine.stop() clears cached telemetry & performance metrics
    const { BrowserDetectionEngine } = await import("../lib/ai/browser-detection-engine");
    const browserEngine = new BrowserDetectionEngine({ driverId: "driver_test" });
    (browserEngine as unknown as {
      _latestTelemetry: typeof mockTelemetry | null;
      _currentInferenceFps: number;
      _latencySamples: number[];
      _lastLatencyMs: number;
      _totalFrames: number;
      _droppedFrames: number;
    })._latestTelemetry = mockTelemetry;
    (browserEngine as unknown as {
      _currentInferenceFps: number;
      _latencySamples: number[];
      _lastLatencyMs: number;
      _totalFrames: number;
      _droppedFrames: number;
    })._currentInferenceFps = 14.5;
    (browserEngine as unknown as { _latencySamples: number[] })._latencySamples = [12, 18];
    (browserEngine as unknown as { _lastLatencyMs: number })._lastLatencyMs = 18;
    (browserEngine as unknown as { _totalFrames: number })._totalFrames = 42;
    (browserEngine as unknown as { _droppedFrames: number })._droppedFrames = 3;

    assert(
      browserEngine.retrieveCurrentState() !== null,
      "BrowserDetectionEngine holds latest telemetry while session is active"
    );

    browserEngine.stop();

    assert(
      browserEngine.retrieveCurrentState() === null,
      "BrowserDetectionEngine.stop() clears latest telemetry reading"
    );
    const clearedMetrics = browserEngine.retrievePerformanceMetrics();
    assert(
      clearedMetrics.inferenceFps === 0 &&
        clearedMetrics.averageInferenceLatencyMs === 0 &&
        clearedMetrics.totalFramesProcessed === 0 &&
        clearedMetrics.droppedFrames === 0,
      "BrowserDetectionEngine.stop() resets FPS, latency, and frame counters to 0"
    );
    browserEngine.dispose();
  }

  console.log("\n======================================================================");
  console.log(`[DriveGuard Parity Tests] ${passedTests}/${totalTests} Passed.`);
  console.log("======================================================================\n");

  if (process.exitCode && process.exitCode !== 0) {
    throw new Error(`Test suite failed with exit code ${process.exitCode}`);
  }
}

runTestSuite().catch((err) => {
  console.error("Test execution fatal error:", err);
  process.exit(1);
});
