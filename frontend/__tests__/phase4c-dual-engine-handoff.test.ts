/**
 * DriveGuard — Phase 4C Verification Suite:
 * Mandatory Test C (Dual-Engine Profile Isolation & Conflict-Aware Reconciliation)
 * Mandatory Test D (Exclusive Two-Phase Camera Handoff: BROWSER <-> PYTHON_LOCAL)
 * + Loopback Python Telemetry Normalization.
 */

import assert from "node:assert/strict";
import { BrowserProfileStore, PoseZone } from "../lib/ai";
import type { DriverBehaviourProfileData } from "../lib/ai";
import {
  executeTwoPhaseCameraHandoff,
  normalizeLocalTelemetryPayload,
  reconcilePythonProfileWithServer,
  stopAndVerifyBrowserMediaStream,
} from "../lib/monitor/engine-adapter";

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

function makeBrowserProfile(driverId: string, centerEar: number): DriverBehaviourProfileData {
  return {
    driverId,
    engineType: "BROWSER",
    profileUuid: "browser-uuid-1111-4222-8333-444444444444",
    profileVersion: 1,
    version: 1,
    calibrationMode: "FOUR_ZONE",
    centerEarMedian: centerEar,
    centerEarMadScaled: 0.015,
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
    calibratedAt: "2026-11-24T12:00:00Z",
    zoneBaselines: {
      [PoseZone.CENTER]: {
        zone: PoseZone.CENTER,
        earMedian: centerEar,
        earMadScaled: 0.015,
        pitchMedianDeg: 0,
        yawMedianDeg: 0,
        sampleCount: 40,
      },
    },
  };
}

async function main() {
  console.log("\n=== Phase 4C: Normalized Adapter, Mandatory Test C & Mandatory Test D ===");

  // Mandatory Test C: Dual-Engine Profile Isolation & Conflict-Aware Reconciliation
  await runTest("Mandatory Test C: BROWSER and PYTHON profiles stay strictly isolated online, offline, and across conflict reconciliation", async () => {
    // Simulated PostgreSQL dual-engine table keyed by (userId, engineType)
    const pgRows = new Map<string, Record<string, any>>();
    const pgHistory: Record<string, any>[] = [];
    let isNetworkOffline = false;

    // Simulated Python local ProfileStore on 127.0.0.1:8000
    let pythonLocalStoredProfile: Record<string, any> | null = {
      driver_id: "driver_dual_c",
      engine_type: "PYTHON",
      profile_uuid: "python-uuid-aaaa-4bbb-8ccc-dddddddddddd",
      profile_version: 1,
      base_server_version: null,
      is_calibrated: true,
      calibration_mode: "FOUR_ZONE",
      center_ear_median: 0.275,
      blink_duration_median_ms: 225,
      blink_duration_mad_ms: 30,
      blink_rcr_median: 0.92,
      blink_rcr_mad: 0.12,
      neutral_pitch_deg: -1.5,
      neutral_yaw_deg: 0.5,
      calibration_reliability: 0.93,
      zone_baselines: {
        CENTER: {
          zone: "CENTER",
          ear_median: 0.275,
          ear_mad_scaled: 0.016,
          dynamic_closure_threshold: 0.20,
          sample_count: 45,
        },
      },
    };

    const origFetch = globalThis.fetch;
    const mockFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);

      // Local loopback Python ProfileStore stays available even when internet/Neon is offline!
      if (url.startsWith("http://127.0.0.1:8000/api/v1/local/profile?")) {
        return new Response(
          JSON.stringify({
            driver_id: "driver_dual_c",
            engine_type: "PYTHON",
            found: Boolean(pythonLocalStoredProfile),
            profile: pythonLocalStoredProfile,
          }),
          { status: 200 }
        );
      }

      if (url === "http://127.0.0.1:8000/api/v1/local/profile/reconcile") {
        const body = JSON.parse(String(init?.body ?? "{}"));
        assert.equal(body.remote_profile.engine_type, "PYTHON", "Python local reconcile must ONLY receive PYTHON");
        return new Response(
          JSON.stringify({
            action: "LOCAL_AHEAD_PENDING_UPLOAD",
            archived_previous: false,
            profile: pythonLocalStoredProfile,
          }),
          { status: 200 }
        );
      }

      // Next.js / PostgreSQL routes fail when offline
      if (isNetworkOffline) {
        throw new TypeError("NetworkError: Neon / Next.js unreachable");
      }

      if (url.startsWith("/api/calibration/profile")) {
        const u = new URL(url, "http://localhost:3000");
        const eng = (u.searchParams.get("engineType") || "BROWSER").toUpperCase();
        const row = pgRows.get(`driver_dual_c:${eng}`) ?? null;
        return new Response(JSON.stringify({ profile: row }), { status: 200 });
      }

      if (url === "/api/calibration/save") {
        const body = JSON.parse(String(init?.body ?? "{}"));
        const eng = String(body.engineType || "").toUpperCase();
        assert.ok(eng === "BROWSER" || eng === "PYTHON", `Invalid engineType: ${eng}`);
        const key = `driver_dual_c:${eng}`;
        const existing = pgRows.get(key);

        if (
          existing &&
          typeof body.baseServerVersion === "number" &&
          body.baseServerVersion < existing.profileVersion
        ) {
          const histId = `hist_${pgHistory.length + 1}`;
          pgHistory.push({ id: histId, engineType: eng, superseded: body });
          return new Response(
            JSON.stringify({
              status: "CONFLICT_SERVER_AUTHORITATIVE",
              conflictHistoryId: histId,
              serverProfile: existing,
            }),
            { status: 409 }
          );
        }

        const nextVer = existing ? existing.profileVersion + 1 : 1;
        const saved = { ...body, engineType: eng, isCalibrated: true, profileVersion: nextVer };
        pgRows.set(key, saved);
        return new Response(JSON.stringify({ status: "SYNCED", profile: saved }), { status: 200 });
      }

      throw new Error(`Unexpected fetch URL: ${url}`);
    }) as typeof fetch;

    globalThis.fetch = mockFetch;

    try {
      // 1. Calibrate & sync Browser AI (centerEar = 0.345) -> writes strictly engineType="BROWSER"
      const browserStore = new BrowserProfileStore();
      await browserStore.saveLocalProfile("driver_dual_c", makeBrowserProfile("driver_dual_c", 0.345), {
        syncState: "PENDING_UPLOAD",
      });
      const browserSync = await browserStore.syncToServer("driver_dual_c");
      assert.equal(browserSync.status, "SYNCED");
      assert.equal(pgRows.get("driver_dual_c:BROWSER")?.centerEarMedian, 0.345);
      assert.equal(pgRows.has("driver_dual_c:PYTHON"), false, "Browser calibration must NEVER touch PYTHON row");

      // 2. Reconcile Python Local AI (centerEar = 0.275) -> writes strictly engineType="PYTHON"
      const pySync = await reconcilePythonProfileWithServer({
        driverId: "driver_dual_c",
        fetchImpl: mockFetch,
      });
      assert.equal(pySync.status, "UPLOADED_LOCAL");
      assert.equal(pgRows.get("driver_dual_c:PYTHON")?.centerEarMedian, 0.275);
      assert.equal(
        pgRows.get("driver_dual_c:BROWSER")?.centerEarMedian,
        0.345,
        "Python calibration must NEVER overwrite BROWSER row"
      );

      // 3. Go OFFLINE -> both Browser and Python still load their own distinct local profiles
      isNetworkOffline = true;
      const offlineBrowser = await browserStore.getLocalProfile("driver_dual_c");
      assert.equal(offlineBrowser?.profile.centerEarMedian, 0.345);

      const offlinePy = await reconcilePythonProfileWithServer({
        driverId: "driver_dual_c",
        fetchImpl: mockFetch,
      });
      assert.equal(offlinePy.status, "OFFLINE_LOCAL_ONLY");
      assert.equal(offlinePy.localProfile?.center_ear_median, 0.275);

      // 4. Simulate concurrent update on server (PYTHON v3) while local Python based on v1 edits offline
      isNetworkOffline = false;
      pgRows.set("driver_dual_c:PYTHON", {
        ...pgRows.get("driver_dual_c:PYTHON"),
        profileVersion: 3,
        centerEarMedian: 0.29,
      });
      pythonLocalStoredProfile = {
        ...pythonLocalStoredProfile!,
        base_server_version: 1,
        center_ear_median: 0.26,
      };

      const conflictRes = await reconcilePythonProfileWithServer({
        driverId: "driver_dual_c",
        fetchImpl: mockFetch,
      });
      assert.equal(conflictRes.status, "CONFLICT_PENDING_REVIEW");
      assert.ok(conflictRes.conflictHistoryId, "Conflict must archive superseded revision in DriverProfileHistory");
      assert.equal(pgHistory.length, 1);
      // BROWSER row still untouched!
      assert.equal(pgRows.get("driver_dual_c:BROWSER")?.centerEarMedian, 0.345);
    } finally {
      globalThis.fetch = origFetch;
    }
  });

  // Mandatory Test D: Exclusive Two-Phase Camera Handoff (BROWSER -> PYTHON_LOCAL -> BROWSER)
  await runTest("Mandatory Test D: Two-phase exclusive camera handoff enforces track ended + 200ms settle before next engine acquires hardware", async () => {
    const timeline: string[] = [];
    let getUserMediaCalls = 0;
    let pythonCameraOpened = false;

    // Mock browser MediaStream + track
    let currentBrowserTrack = {
      readyState: "live" as "live" | "ended",
      stop() {
        this.readyState = "ended";
        timeline.push("BROWSER_TRACK_ENDED");
      },
    };
    let currentBrowserStream = {
      getTracks: () => [currentBrowserTrack],
    } as unknown as MediaStream;

    const browserAdapter = {
      engineId: "BROWSER" as const,
      stop: async () => {
        timeline.push("BROWSER_STOP");
        return true;
      },
      releaseCamera: async () => {
        const ended = stopAndVerifyBrowserMediaStream(currentBrowserStream);
        assert.equal(ended, true, "All browser MediaStreamTracks must report readyState === 'ended'");
        return ended;
      },
      start: async () => {
        assert.equal(pythonCameraOpened, false, "Browser MUST NOT open webcam while Python holds camera!");
        getUserMediaCalls++;
        currentBrowserTrack = {
          readyState: "live",
          stop() {
            this.readyState = "ended";
            timeline.push("BROWSER_TRACK_ENDED");
          },
        };
        currentBrowserStream = {
          getTracks: () => [currentBrowserTrack],
        } as unknown as MediaStream;
        timeline.push("BROWSER_ACQUIRE_WEBCAM");
        return true;
      },
    };

    const pythonLocalAdapter = {
      engineId: "PYTHON_LOCAL" as const,
      stop: async () => {
        timeline.push("PYTHON_LOCAL_STOP");
        return true;
      },
      releaseCamera: async () => {
        pythonCameraOpened = false;
        timeline.push("PYTHON_LOCAL_CAMERA_RELEASED");
        return true;
      },
      start: async () => {
        // Strict safety invariant: browser track MUST already be ended before OpenCV opens!
        assert.equal(
          currentBrowserTrack.readyState,
          "ended",
          "Browser MediaStreamTrack MUST be 'ended' before Python OpenCV opens!"
        );
        pythonCameraOpened = true;
        timeline.push("PYTHON_LOCAL_START_OPENCV");
        return true;
      },
    };

    const mockSleep = async (ms: number) => {
      timeline.push(`SETTLE_${ms}MS`);
    };

    // 1. Switch BROWSER -> PYTHON_LOCAL
    const h1 = await executeTwoPhaseCameraHandoff(browserAdapter, pythonLocalAdapter, {
      settleDelayMs: 200,
      autoStartTarget: true,
      sleepImpl: mockSleep,
    });
    assert.equal(h1.previousReleased, true);
    assert.equal(h1.targetStarted, true);
    assert.equal(h1.settleDelayMs, 200);
    assert.equal(getUserMediaCalls, 0, "Switching to PYTHON_LOCAL must NEVER call getUserMedia");
    assert.deepEqual(timeline, [
      "BROWSER_STOP",
      "BROWSER_TRACK_ENDED",
      "SETTLE_200MS",
      "PYTHON_LOCAL_START_OPENCV",
    ]);

    // 2. Switch PYTHON_LOCAL -> BROWSER
    timeline.length = 0;
    const h2 = await executeTwoPhaseCameraHandoff(pythonLocalAdapter, browserAdapter, {
      settleDelayMs: 200,
      autoStartTarget: true,
      sleepImpl: mockSleep,
    });
    assert.equal(h2.previousReleased, true);
    assert.equal(h2.targetStarted, true);
    assert.equal(getUserMediaCalls, 1);
    assert.deepEqual(timeline, [
      "PYTHON_LOCAL_STOP",
      "PYTHON_LOCAL_CAMERA_RELEASED",
      "SETTLE_200MS",
      "BROWSER_ACQUIRE_WEBCAM",
    ]);

    // 3. Failure safety guard: if previous camera fails to release, target MUST NOT start
    let targetCalled = false;
    const stuckHandoff = await executeTwoPhaseCameraHandoff(
      {
        engineId: "PYTHON_LOCAL",
        stop: async () => true,
        releaseCamera: async () => false,
      },
      {
        engineId: "BROWSER",
        start: async () => {
          targetCalled = true;
          return true;
        },
      },
      { settleDelayMs: 200, autoStartTarget: true, sleepImpl: mockSleep }
    );
    assert.equal(stuckHandoff.previousReleased, false);
    assert.equal(stuckHandoff.targetStarted, false);
    assert.equal(targetCalled, false, "Target engine MUST NOT acquire camera if release failed");
  });

  await runTest("normalizeLocalTelemetryPayload maps Python loopback WS message to canonical FrameTelemetry", () => {
    const normalized = normalizeLocalTelemetryPayload({
      frame_id: 42,
      timestamp_ms: 1732400000123,
      fps: 19.5,
      inference_latency_ms: 9.8,
      face_detected: true,
      operational_state: "MONITORING_ACTIVE",
      state: "WARNING",
      risk_score: 0.66,
      pose_zone: "CENTER",
      reason_codes: ["PERCLOS_ELEVATED"],
      eye_metrics: {
        ear_left: 0.19,
        ear_right: 0.18,
        ear_raw: 0.185,
        ear_normalized: 0.185,
        ear_closure_threshold: 0.21,
        perclos_pct: 18.4,
        perclos_reliable: true,
        perclos_coverage_pct: 92.0,
        total_blink_count: 14,
      },
      head_pose: { pitch_deg: -2.1, yaw_deg: 1.4, roll_deg: 0.3, zone: "CENTER" },
      mouth_metrics: { mar: 0.14, is_yawning: false },
      optical_quality: { luminance: 132, blur_variance: 220, status: "OPTIMAL" },
      cascade: {
        arbitration_winner: "GEOMETRIC_AUTHORITATIVE",
        cnn_execution_mode: "SHADOW_DISABLED",
        onnx_inference_executed: false,
      },
    });

    assert.equal(normalized.engine_type, "PYTHON");
    assert.equal(normalized.operational_state, "MONITORING_ACTIVE");
    assert.equal(normalized.state, "WARNING");
    assert.equal(normalized.eye_metrics.perclos_pct, 18.4);
    assert.equal(normalized.eye_metrics.total_blink_count, 14);
    assert.equal(normalized.cascade.cnn_execution_mode, "SHADOW_DISABLED");
    assert.equal(normalized.cascade.onnx_inference_executed, false);
  });

  console.log(`\nPhase 4C Verification Result: ${passed} passed, ${failed} failed\n`);
  if (failed > 0) {
    process.exit(1);
  }
}

main();
