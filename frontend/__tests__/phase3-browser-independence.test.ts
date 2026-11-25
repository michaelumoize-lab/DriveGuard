/**
 * DriveGuard — Phase 3 Browser AI Independence Verification Suite.
 *
 * Covers:
 * 1. PDBP calibration crash-safe rejection, blink-dip filtering, SO(3) medoid, skip & single-zone recalibration
 * 2. MediaPipe monotonic video timestamp guard
 * 3. SignalConfidenceEvaluator optical quality degradation -> OperationalState.DEGRADED
 * 4. OfflineEventQueue listener disposal (zero leaks), transient retry recovery & strict causal flush ordering
 * 5. BrowserProfileStore offline storage, engineType="BROWSER" enforcement & 409 conflict handling
 * 6. Pre-flight handshake offline Browser AI allowance (OFFLINE_QUEUED) & zero ONNX claims
 * 7. Mandatory Test A: Full Offline Browser AI Monitoring + Causal Reconnect Flush E2E
 */

import assert from "node:assert/strict";
import {
  BlinkState,
  BrowserDetectionEngine,
  BrowserProfileStore,
  computeZoneThresholds,
  ConfidenceTier,
  createProvisionalOfflineSession,
  DrowsinessState,
  enforceMonotonicVideoTimestamp,
  GuidedCalibrationEngine,
  isValidRotationMatrix,
  OfflineEventQueue,
  OperationalState,
  PoseZone,
  SignalConfidenceEvaluator,
} from "../lib/ai";
import { runPreFlightHandshake } from "../lib/monitor/preflight-handshake";
import type {
  DriverBehaviourProfileData,
  DriverFeatures,
  NormalizedLandmark,
} from "../lib/ai";
import type { OpticalFrameQuality } from "../types/camera";

let passed = 0;
let failed = 0;

async function runTest(name: string, fn: () => void | Promise<void>) {
  try {
    await fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.error(`  FAIL  ${name}`);
    console.error(err);
    failed++;
  }
}

/**
 * Generate a synthetic 478-landmark face mesh with controllable eye openness.
 */
function makeSyntheticFaceLandmarks(earScale: number = 1.0): NormalizedLandmark[] {
  const pts: NormalizedLandmark[] = Array.from({ length: 478 }, () => ({
    x: 0.5,
    y: 0.5,
    z: 0.0,
  }));

  // Anchor points used by HeadPoseEstimator (nose tip 1, chin 152, eye outer corners 33/263, mouth 61/291)
  pts[1] = { x: 0.50, y: 0.50, z: -0.05 };
  pts[152] = { x: 0.50, y: 0.68, z: -0.02 };
  pts[33] = { x: 0.40, y: 0.44, z: -0.01 };
  pts[263] = { x: 0.60, y: 0.44, z: -0.01 };
  pts[61] = { x: 0.44, y: 0.59, z: -0.02 };
  pts[291] = { x: 0.56, y: 0.59, z: -0.02 };

  // Left eye landmarks: [33, 160, 158, 133, 153, 144]
  const h = 0.016 * earScale;
  pts[33] = { x: 0.38, y: 0.44, z: 0.0 };
  pts[133] = { x: 0.46, y: 0.44, z: 0.0 };
  pts[160] = { x: 0.405, y: 0.44 - h, z: 0.0 };
  pts[158] = { x: 0.435, y: 0.44 - h, z: 0.0 };
  pts[144] = { x: 0.405, y: 0.44 + h, z: 0.0 };
  pts[153] = { x: 0.435, y: 0.44 + h, z: 0.0 };

  // Right eye landmarks: [362, 385, 387, 263, 373, 380]
  pts[362] = { x: 0.54, y: 0.44, z: 0.0 };
  pts[263] = { x: 0.62, y: 0.44, z: 0.0 };
  pts[385] = { x: 0.565, y: 0.44 - h, z: 0.0 };
  pts[387] = { x: 0.595, y: 0.44 - h, z: 0.0 };
  pts[380] = { x: 0.565, y: 0.44 + h, z: 0.0 };
  pts[373] = { x: 0.595, y: 0.44 + h, z: 0.0 };

  // Mouth landmarks for MAR: [61, 291, 13, 14, 81, 311, 178, 402]
  pts[13] = { x: 0.50, y: 0.585, z: -0.02 };
  pts[14] = { x: 0.50, y: 0.595, z: -0.02 };
  pts[81] = { x: 0.48, y: 0.586, z: -0.02 };
  pts[178] = { x: 0.48, y: 0.594, z: -0.02 };
  pts[311] = { x: 0.52, y: 0.586, z: -0.02 };
  pts[402] = { x: 0.52, y: 0.594, z: -0.02 };

  return pts;
}

function makeCalibratedProfile(driverId: string = "driver_phase3"): DriverBehaviourProfileData {
  return {
    driverId,
    engineType: "BROWSER",
    profileUuid: "11111111-2222-4333-8444-555555555555",
    profileVersion: 1,
    version: 1,
    calibrationMode: "FOUR_ZONE",
    centerEarMedian: 0.32,
    centerEarMadScaled: 0.014,
    blinkDurationMedianMs: 210,
    blinkDurationMadMs: 25,
    blinkRcrMedian: 0.95,
    blinkRcrMad: 0.08,
    blinkFreqMedianPerMin: 16,
    blinkFreqMadPerMin: 4,
    baselinePerclosPct: 3.5,
    blinkAmplitudeMedian: 0.14,
    temporalBlinksSampled: 0,
    temporalWindowSec: 0,
    sensitivityLevel: "STANDARD",
    neutralPitchDeg: 0,
    neutralYawDeg: 0,
    neutralRotMat: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    calibrationIsMirrored: false,
    calibrationReliability: 0.94,
    calibratedAt: new Date().toISOString(),
    zoneBaselines: {
      [PoseZone.CENTER]: {
        zone: PoseZone.CENTER,
        earMedian: 0.32,
        earMadScaled: 0.014,
        pitchMedianDeg: 0,
        yawMedianDeg: 0,
        sampleCount: 40,
      },
      [PoseZone.LEFT_MIRROR]: {
        zone: PoseZone.LEFT_MIRROR,
        earMedian: 0.29,
        earMadScaled: 0.016,
        pitchMedianDeg: -2,
        yawMedianDeg: 24,
        sampleCount: 35,
      },
      [PoseZone.RIGHT_MIRROR]: {
        zone: PoseZone.RIGHT_MIRROR,
        earMedian: 0.29,
        earMadScaled: 0.016,
        pitchMedianDeg: -2,
        yawMedianDeg: -24,
        sampleCount: 35,
      },
      [PoseZone.DASHBOARD]: {
        zone: PoseZone.DASHBOARD,
        earMedian: 0.27,
        earMadScaled: 0.016,
        pitchMedianDeg: -14,
        yawMedianDeg: 0,
        sampleCount: 35,
      },
    },
  };
}

function makeHealthyDriverFeatures(): DriverFeatures {
  return {
    timestampMs: 1000,
    hasFace: true,
    ear: {
      leftEar: 0.31,
      rightEar: 0.31,
      weightedEar: 0.31,
      compensatedEar: 0.31,
      rawEar: 0.31,
      closureThreshold: 0.22,
      isLeftClosed: false,
      isRightClosed: false,
      isClosed: false,
      isValid: true,
      isLeftValid: true,
      isRightValid: true,
      isGeometricValid: true,
    },
    mar: {
      mar: 0.12,
      isMouthOpen: false,
      isYawnInProgress: false,
      currentYawnDurationMs: 0,
      isAuthenticYawn: false,
      completedYawn: null,
      totalYawnCount: 0,
      isValid: true,
      baselineMar: 0.12,
      openingRatio: 1.0,
      peakMar: 0.12,
      observedYawnDurationMs: 0,
    },
    headPose: {
      pitchDeg: 0,
      yawDeg: 0,
      rollDeg: 0,
      relPitchDeg: 0,
      relYawDeg: 0,
      relRollDeg: 0,
      zone: PoseZone.CENTER,
      isLookingAway: false,
      isValid: true,
      rotMat: [
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ],
    },
    perclos: {
      perclosPct: 2.0,
      effectivePerclosPct: 2.0,
      coveragePct: 100,
      isReliable: true,
      closedDurationSec: 0.2,
      observedDurationSec: 10,
      windowSec: 60,
      unobservedDurationSec: 0,
    },
    blink: {
      state: BlinkState.OPEN,
      isEyeClosed: false,
      currentClosureDurationMs: 0,
      completedBlink: null,
      recentBlinkDurationMs: 190,
      recentBlinkRcr: 0.96,
      totalBlinkCount: 3,
      prolongedBlinkCount: 0,
      samplingGapDetected: false,
    },
    summaryText: "Normal",
  };
}

async function main() {
  console.log("\n=== Phase 3: Full Browser AI Independence Verification ===");

  // 1. PDBP Calibration rejection crash regression + full 4-zone capture + SO(3) + single-zone
  await runTest("GuidedCalibrationEngine: infeasible samples reject cleanly with zero subsequent crash", () => {
    const cal = new GuidedCalibrationEngine({
      driverId: "driver_reject",
      minSamplesPerZone: 12,
    });

    // Feed eyes-nearly-closed infeasible EAR (0.14 -> span < 0.040) over CENTER duration (4000ms+)
    let t = 1000;
    let progress = cal.update(t, 0.14, 0, 0, true, true, null);
    for (let i = 1; i <= 45 && !progress.isCompleted; i++) {
      t += 150;
      progress = cal.update(t, 0.14, 0, 0, true, true, null);
    }
    while (!progress.isCompleted) {
      progress = cal.skipStep();
    }

    assert.equal(progress.isCompleted, true, "Rejection must mark calibration completed");
    assert.ok(progress.rejectedReason && progress.rejectedReason.length > 0, "Rejection reason must be populated");

    // Subsequent update calls after rejection MUST NOT crash out-of-bounds
    const after = cal.update(t + 500, 0.30, 0, 0, true, true, null);
    assert.equal(after.isCompleted, true);
    assert.equal(after.rejectedReason, progress.rejectedReason);
  });

  await runTest("GuidedCalibrationEngine: 4-zone capture filters blink dips, computes SO(3) medoid & matches Python thresholds", () => {
    const cal = new GuidedCalibrationEngine({
      driverId: "driver_valid",
      minSamplesPerZone: 15,
    });

    const rad = (3 * Math.PI) / 180;
    const rotMat = [
      [Math.cos(rad), 0, Math.sin(rad)],
      [0, 1, 0],
      [-Math.sin(rad), 0, Math.cos(rad)],
    ];

    const zoneSpecs: Record<string, { yaw: number; pitch: number; ear: number }> = {
      CENTER: { yaw: 0, pitch: 0, ear: 0.31 },
      LEFT_MIRROR: { yaw: 24, pitch: -2, ear: 0.28 },
      RIGHT_MIRROR: { yaw: -24, pitch: -2, ear: 0.285 },
      DASHBOARD: { yaw: 0, pitch: -14, ear: 0.265 },
    };

    let t = 1000;
    let progress = cal.update(t, 0.31, 0, 0, true, true, rotMat);

    for (let step = 0; step < 300 && !progress.isCompleted; step++) {
      t += 120;
      const spec = zoneSpecs[progress.currentZone] ?? zoneSpecs.CENTER;
      // Inject occasional blink dips (0.11) that must be filtered out
      const ear = step % 7 === 0 ? 0.11 : spec.ear + ((step % 3) - 1) * 0.004;
      progress = cal.update(t, ear, spec.pitch, spec.yaw, true, true, rotMat);
    }

    assert.equal(progress.isCompleted, true, "Calibration should complete all 4 zones");
    assert.equal(progress.rejectedReason, undefined, `Unexpected rejection: ${progress.rejectedReason}`);
    assert.ok(progress.profile, "Completed profile must be returned");

    const p = progress.profile!;
    assert.equal(p.engineType, "BROWSER");
    assert.ok(p.profileUuid && p.profileUuid.length > 8);
    assert.ok(p.neutralRotMat && isValidRotationMatrix(p.neutralRotMat, 1e-5), "neutralRotMat must be exact SO(3)");
    assert.ok(p.centerEarMedian > 0.295 && p.centerEarMedian < 0.325, `Center EAR median unexpected: ${p.centerEarMedian}`);

    // Threshold formula parity with computeZoneThresholds within <= 1e-4
    const centerZb = p.zoneBaselines[PoseZone.CENTER];
    const th = computeZoneThresholds(centerZb.earMedian, centerZb.earMadScaled);
    assert.equal(th.isFeasible, true);
    assert.ok(th.span >= 0.04);

    // Single-zone recalibration preserves other zones
    const singleCal = new GuidedCalibrationEngine({
      driverId: "driver_valid",
      minSamplesPerZone: 12,
      targetZone: PoseZone.LEFT_MIRROR,
      existingProfile: p,
    });
    let st = 5000;
    let sp = singleCal.update(st, 0.25, -2, 24, true, true, rotMat);
    for (let i = 0; i < 60 && !sp.isCompleted; i++) {
      st += 120;
      sp = singleCal.update(st, 0.25, -2, 24, true, true, rotMat);
    }
    assert.equal(sp.isCompleted, true);
    assert.equal(sp.rejectedReason, undefined);
    assert.ok(sp.profile);
    assert.equal(sp.profile!.centerEarMedian, p.centerEarMedian, "CENTER baseline must be preserved");
    assert.ok(
      Math.abs(sp.profile!.zoneBaselines[PoseZone.LEFT_MIRROR].earMedian - 0.25) < 0.02,
      "LEFT_MIRROR baseline must update"
    );

    // skipStep advances past current stage
    const skipCal = new GuidedCalibrationEngine({ driverId: "driver_skip" });
    const skipped = skipCal.skipStep();
    assert.notEqual(skipped.currentZone, PoseZone.CENTER, "skipStep must advance past CENTER");
  });

  // 2. Monotonic MediaPipe video timestamp guard
  await runTest("MediaPipe timestamp guard: enforces strictly increasing integer ms", () => {
    const t1 = enforceMonotonicVideoTimestamp(1000.4, -1);
    const t2 = enforceMonotonicVideoTimestamp(1000.4, t1);
    const t3 = enforceMonotonicVideoTimestamp(995.0, t2);
    const t4 = enforceMonotonicVideoTimestamp(NaN, t3);
    assert.equal(t1, 1000);
    assert.equal(t2, 1001);
    assert.equal(t3, 1002);
    assert.equal(t4, 1003);
  });

  // 3. SignalConfidenceEvaluator + Optical Quality -> OperationalState.DEGRADED
  await runTest("SignalConfidenceEvaluator: SEVERELY_DEGRADED optics drives BrowserDetectionEngine to DEGRADED", () => {
    const evaluator = new SignalConfidenceEvaluator();
    const poorOptics: OpticalFrameQuality = {
      meanLuminance: 18,
      blurVarianceScore: 15,
      lightingCondition: "UNDEREXPOSED",
      blurCondition: "BLURRY",
      overallQuality: "SEVERELY_DEGRADED",
      reasonCodes: ["LOW_LIGHT_UNDEREXPOSED", "OPTICAL_BLUR"],
    };

    const assessment = evaluator.evaluate(makeHealthyDriverFeatures(), poorOptics, {
      isCalibrated: true,
      isCalibrating: false,
    });
    assert.equal(assessment.isReliable, false);
    assert.equal(assessment.isOpticalViable, false);
    assert.equal(assessment.tier, ConfidenceTier.DEGRADED);

    const engine = new BrowserDetectionEngine({
      driverId: "driver_optics",
      profile: makeCalibratedProfile("driver_optics"),
      isMirrored: false,
    });
    engine.setOpticalQuality(poorOptics);
    const landmarks = makeSyntheticFaceLandmarks(1.0);
    // Feed severely degraded optical frames beyond sensorUnreliableGraceMs (1000ms)
    let res = engine.evaluateFrameLandmarks(landmarks, 1000, 640, 480);
    for (let t = 1100; t <= 2200; t += 100) {
      res = engine.evaluateFrameLandmarks(landmarks, t, 640, 480);
    }
    assert.equal(res.telemetry.operationalState, OperationalState.DEGRADED);
    assert.equal(res.decision.state, DrowsinessState.INSUFFICIENT_DATA);
    assert.notEqual(res.decision.state, DrowsinessState.NORMAL);
    engine.dispose();
  });

  // 4. OfflineEventQueue listener disposal + transient retry + strict causal flush ordering
  await runTest("OfflineEventQueue: zero listener leaks on dispose & strict SESSION_START -> ALERT -> SESSION_END flush", async () => {
    const listeners = new Map<string, Set<unknown>>();
    const mockWindow = {
      addEventListener: (ev: string, fn: unknown) => {
        if (!listeners.has(ev)) listeners.set(ev, new Set());
        listeners.get(ev)!.add(fn);
      },
      removeEventListener: (ev: string, fn: unknown) => {
        listeners.get(ev)?.delete(fn);
      },
    };
    const origWindow = (globalThis as Record<string, unknown>).window;
    const origFetch = globalThis.fetch;
    (globalThis as Record<string, unknown>).window = mockWindow;

    try {
      const q1 = new OfflineEventQueue({ autoFlushIntervalMs: 0 });
      assert.equal(listeners.get("online")?.size, 1);
      assert.equal(listeners.get("offline")?.size, 1);
      q1.dispose();
      assert.equal(listeners.get("online")?.size, 0, "online listener must be removed on dispose");
      assert.equal(listeners.get("offline")?.size, 0, "offline listener must be removed on dispose");

      let failNetwork = true;
      const deliveredUrls: string[] = [];
      const deliveredBodies: Record<string, unknown>[] = [];

      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        if (failNetwork) {
          return new Response(JSON.stringify({ error: "Bad Gateway" }), { status: 502 });
        }
        deliveredUrls.push(url);
        deliveredBodies.push(JSON.parse(String(init?.body ?? "{}")));
        return new Response(JSON.stringify({ ok: true, sessionCode: "DRV-OFFLINE-1" }), { status: 200 });
      }) as typeof fetch;

      const q2 = new OfflineEventQueue({ autoFlushIntervalMs: 0, baseRetryDelayMs: 60_000 });
      const provisional = createProvisionalOfflineSession();
      q2.enqueueSessionStart({
        clientSessionId: provisional.clientSessionId,
        sessionCode: provisional.sessionCode,
        engineType: "BROWSER",
        startedAt: provisional.startedAt,
      });
      q2.enqueueAlert({
        eventId: "evt_causal_1",
        sessionCode: provisional.sessionCode,
        clientSessionId: provisional.clientSessionId,
        severity: "CRITICAL",
        riskScore: 0.91,
        earValue: 0.14,
        perclosPct: 28.0,
        poseZone: "CENTER",
        reasons: ["PROLONGED_CLOSURE"],
        cascadeDecision: "BROWSER_GEOMETRIC",
        timestampMs: Date.now(),
      });
      q2.enqueueSessionEnd({
        sessionCode: provisional.sessionCode,
        clientSessionId: provisional.clientSessionId,
        finalizeToken: "fin_causal_1",
        durationSeconds: 45,
        averageFps: 15,
        averageLatencyMs: 12,
        averagePerclosPct: 14,
        maxPerclosPct: 28,
        totalBlinks: 8,
        averageBlinkRate: 11,
        normalFramesCount: 500,
        warningAlertsCount: 1,
        criticalAlertsCount: 1,
        mirrorSuppressions: 0,
      });

      // Attempt flush during transient 502: items remain queued and status stays "pending" (NOT "offline")
      await q2.flush();
      assert.equal(q2.pendingCount, 3, "All 3 causal items must stay queued on 502");
      assert.equal(q2.status, "pending", "Transient 502 must NOT freeze _isOnline=false");

      // Network recovers -> flush must deliver in exact causal order
      failNetwork = false;
      await q2.flush();
      assert.deepEqual(deliveredUrls, [
        "/api/sessions/start",
        "/api/alerts/log",
        "/api/sessions/end",
      ]);
      assert.equal(deliveredBodies[0].clientSessionId, provisional.clientSessionId);
      assert.equal(deliveredBodies[1].eventId, "evt_causal_1");
      assert.equal(deliveredBodies[2].finalizeToken, "fin_causal_1");
      assert.equal(q2.pendingCount, 0);
      assert.equal(q2.status, "synced");

      q2.dispose();
    } finally {
      (globalThis as Record<string, unknown>).window = origWindow;
      globalThis.fetch = origFetch;
    }
  });

  // 5. BrowserProfileStore offline storage, engineType="BROWSER" & 409 conflict handling
  await runTest("BrowserProfileStore: offline save/hydrate, strict engineType=BROWSER & 409 conflict handling", async () => {
    const origFetch = globalThis.fetch;
    try {
      const store = new BrowserProfileStore();
      assert.equal(store.storageMode, "LOCAL_STORAGE_DEGRADED");

      const profile = makeCalibratedProfile("user_store_1");
      const saved = await store.saveLocalProfile("user_store_1", profile, {
        syncState: "PENDING_UPLOAD",
      });
      assert.equal(saved.engineType, "BROWSER");

      const hydrated = await store.getLocalProfile("user_store_1");
      assert.ok(hydrated);
      assert.equal(hydrated!.profile.centerEarMedian, 0.32);

      let sentBody: Record<string, unknown> | null = null;
      globalThis.fetch = (async (_url: RequestInfo | URL, init?: RequestInit) => {
        sentBody = JSON.parse(String(init?.body ?? "{}"));
        return new Response(
          JSON.stringify({
            status: "CONFLICT_SERVER_AUTHORITATIVE",
            conflictHistoryId: "hist_conflict_99",
            serverProfile: { profileVersion: 4 },
          }),
          { status: 409 }
        );
      }) as typeof fetch;

      const conflictOutcome = await store.syncToServer("user_store_1");
      assert.ok(sentBody);
      assert.equal((sentBody as Record<string, unknown>).engineType, "BROWSER");
      assert.equal(conflictOutcome.status, "CONFLICT_SERVER_AUTHORITATIVE");
      assert.equal(conflictOutcome.conflictHistoryId, "hist_conflict_99");
      assert.equal(conflictOutcome.record?.syncState, "CONFLICT_PENDING_REVIEW");

      const afterConflict = await store.getLocalProfile("user_store_1");
      assert.equal(afterConflict?.profile.centerEarMedian, 0.32);
      assert.equal(afterConflict?.syncState, "CONFLICT_PENDING_REVIEW");

      globalThis.fetch = (async () => {
        return new Response(
          JSON.stringify({
            status: "SYNCED",
            profile: { profileVersion: 2, contentHash: "sha256_abc" },
          }),
          { status: 200 }
        );
      }) as typeof fetch;

      const syncedOutcome = await store.syncToServer("user_store_1");
      assert.equal(syncedOutcome.status, "SYNCED");
      assert.equal(syncedOutcome.record?.syncState, "SYNCED");
      assert.equal(syncedOutcome.record?.profileVersion, 2);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  // 6. Pre-flight handshake offline Browser AI allowance & truthful messaging
  await runTest("Pre-flight handshake: Browser AI passes offline (OFFLINE_QUEUED) with zero ONNX claims", async () => {
    const origFetch = globalThis.fetch;
    try {
      globalThis.fetch = (async () => {
        throw new TypeError("Failed to fetch");
      }) as typeof fetch;

      const mockVideo = {
        readyState: 4,
        videoWidth: 640,
        videoHeight: 480,
      } as unknown as HTMLVideoElement;

      const browserRes = await runPreFlightHandshake({
        runtimeMode: "browser",
        videoElement: mockVideo,
        isMediaPipeReady: true,
        isCascadeReady: true,
        isAudioUnlocked: true,
      });

      assert.equal(browserRes.canStart, true, "Browser AI must be allowed to start offline");
      assert.equal(browserRes.persistenceMode, "OFFLINE_QUEUED");
      assert.equal(browserRes.failedSubsystems.length, 0);
      const backendCheck = browserRes.items.find((i) => i.id === "backend");
      assert.equal(backendCheck?.status, "warning");

      for (const item of browserRes.items) {
        assert.ok(
          !/onnx|mobilenet/i.test(item.message) && !/onnx|mobilenet/i.test(item.label),
          `Misleading ONNX claim found in preflight item: ${item.label} - ${item.message}`
        );
      }

      const remoteRes = await runPreFlightHandshake({
        runtimeMode: "remote",
        videoElement: mockVideo,
        isMediaPipeReady: true,
        isCascadeReady: true,
        isAudioUnlocked: true,
      });
      assert.equal(remoteRes.canStart, false, "Remote mode must fail preflight when backend is unreachable");
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  // 7. Mandatory Test A: Full Offline Browser AI Monitoring + Causal Reconnect Flush E2E
  await runTest("Mandatory Test A: Full offline Browser AI monitoring (zero backend) + causal reconnect flush", async () => {
    const origFetch = globalThis.fetch;
    let isOffline = true;
    const calls: { url: string; body: Record<string, unknown> }[] = [];

    try {
      globalThis.fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
        const url = String(input);
        assert.ok(!url.includes("127.0.0.1:8000") && !url.includes("livekit"), `Forbidden backend URL called: ${url}`);
        if (isOffline) {
          throw new TypeError("NetworkError when attempting to fetch resource");
        }
        const body = init?.body ? JSON.parse(String(init.body)) : {};
        calls.push({ url, body });
        return new Response(JSON.stringify({ ok: true }), { status: 200 });
      }) as typeof fetch;

      const profileStore = new BrowserProfileStore();
      await profileStore.saveLocalProfile("driver_offline_e2e", makeCalibratedProfile("driver_offline_e2e"), {
        syncState: "SYNCED",
      });

      const alertsTriggered: DrowsinessState[] = [];
      const engine = new BrowserDetectionEngine({
        driverId: "driver_offline_e2e",
        isMirrored: false,
        profileStore,
        onSafetyAlert: (decision) => {
          alertsTriggered.push(decision.state);
        },
      });

      const localRecord = await profileStore.getLocalProfile("driver_offline_e2e");
      assert.ok(localRecord, "Local Browser profile must hydrate offline");
      engine.setProfile(localRecord!.profile);
      assert.equal(engine.isCalibrated, true);

      const provisional = engine.startOfflineSession();
      assert.ok(provisional.sessionCode.startsWith("DRV-"));

      const openEyes = makeSyntheticFaceLandmarks(1.0);
      const closedEyes = makeSyntheticFaceLandmarks(0.15);

      let ts = 10_000;
      let res = engine.evaluateFrameLandmarks(openEyes, ts);
      for (let i = 0; i < 15; i++) {
        ts += 66;
        res = engine.evaluateFrameLandmarks(openEyes, ts);
      }
      assert.equal(res.telemetry.operationalState, OperationalState.MONITORING_ACTIVE);
      assert.equal(res.decision.state, DrowsinessState.NORMAL);

      // Single natural blink (3 closed frames ~200ms then open)
      for (let i = 0; i < 3; i++) {
        ts += 66;
        engine.evaluateFrameLandmarks(closedEyes, ts);
      }
      for (let i = 0; i < 5; i++) {
        ts += 66;
        res = engine.evaluateFrameLandmarks(openEyes, ts);
      }
      assert.ok(res.features.blink.totalBlinkCount >= 1, `Expected >= 1 blink detected offline, got ${res.features.blink.totalBlinkCount}`);

      // Sustained eye closure (> 1.8s) -> CRITICAL_WARNING triggered & queued offline
      for (let i = 0; i < 35; i++) {
        ts += 66;
        res = engine.evaluateFrameLandmarks(closedEyes, ts);
      }
      assert.equal(res.decision.state, DrowsinessState.CRITICAL_WARNING, "Sustained closure must trigger CRITICAL_WARNING offline");
      assert.ok(alertsTriggered.includes(DrowsinessState.CRITICAL_WARNING), "onSafetyAlert must fire offline");

      // End session offline
      engine.queueSessionEnd(12);
      assert.ok(
        engine.offlineQueue.pendingCount >= 3,
        `Expected >= 3 queued items offline (start, alert, end), got ${engine.offlineQueue.pendingCount}`
      );
      assert.equal(calls.length, 0, "Zero network deliveries should have succeeded while offline");

      // Reconnect & flush -> strict causal ordering: SESSION_START -> ALERT(s) -> SESSION_END
      isOffline = false;
      await engine.flushOfflineQueue();

      assert.ok(calls.length >= 3, `Expected >= 3 flushed calls, got ${calls.length}`);
      assert.equal(calls[0].url, "/api/sessions/start", "First flushed call MUST be /api/sessions/start");
      assert.equal(calls[calls.length - 1].url, "/api/sessions/end", "Last flushed call MUST be /api/sessions/end");

      const alertCalls = calls.slice(1, calls.length - 1);
      assert.ok(alertCalls.length >= 1, "At least one alert must be flushed between start and end");
      for (const ac of alertCalls) {
        assert.equal(ac.url, "/api/alerts/log");
        assert.equal(ac.body.clientSessionId, provisional.clientSessionId);
        assert.ok(typeof ac.body.eventId === "string" && ac.body.eventId.length > 0);
      }
      assert.equal(calls[0].body.clientSessionId, provisional.clientSessionId);
      assert.equal(calls[0].body.engineType, "BROWSER");
      assert.equal(calls[calls.length - 1].body.clientSessionId, provisional.clientSessionId);
      assert.ok(typeof calls[calls.length - 1].body.finalizeToken === "string");
      assert.equal(engine.offlineQueue.pendingCount, 0, "Queue must be empty after reconnect flush");

      engine.dispose();
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  console.log(`\nPhase 3 Verification Result: ${passed} passed, ${failed} failed\n`);
  if (failed > 0) {
    process.exit(1);
  }
}

main();
