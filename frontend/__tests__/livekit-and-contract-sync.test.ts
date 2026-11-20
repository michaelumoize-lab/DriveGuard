/**
 * DriveGuard Stage 8 Integration Tests:
 * Frontend <-> FastAPI Contract, Profile Mapper, LiveKit Transport Lifecycle,
 * Authoritative Reason Codes, and Honest UI State Resolution.
 */

import assert from "node:assert/strict";
import {
  createMonitoringSession,
  DriveGuardApiError,
  requestLiveKitToken,
  stopLiveKitWorker,
  syncDriverProfileToFastApi,
} from "../lib/api/driveguard-api";
import { buildDriverProfileDtoPayload } from "../lib/api/profile-mapper";
import {
  computeBackendProcessingStatus,
  createProtocolSequenceTracker,
  establishLiveKitSession,
  LiveKitRoomLike,
  processDataChannelMessage,
  teardownLiveKitSession,
} from "../lib/livekit/livekit-session-client";
import {
  explainReasonCode,
  FrameTelemetry,
  REASON_CODE_CATALOG,
  resolveEffectiveDisplayState,
} from "../types/telemetry";

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

function makeMockTelemetry(overrides: Partial<FrameTelemetry> = {}): FrameTelemetry {
  return {
    session_id: "sess_test",
    driver_id: "driver_1",
    frame_index: 1,
    frame_seq: "1",
    event_seq: "1",
    timestamp_ms: 1000,
    fps: 20.0,
    face_detected: true,
    face_visible: true,
    state: "NORMAL",
    drowsiness_state: "NORMAL",
    alert_state: "NORMAL",
    is_recovering: false,
    authoritative_engine: "DrowsinessEngine",
    monitoring_status: "MONITORING_ACTIVE",
    risk_score: 0.05,
    pose_zone: "CENTER",
    reason_codes: ["ALERT_NORMAL"],
    head_pose: { pitch_deg: 0, yaw_deg: 0, roll_deg: 0 },
    eye_metrics: {
      ear_raw: 0.3,
      ear_left: 0.3,
      ear_right: 0.3,
      ear_normalized: 0.3,
      ear_closure_threshold: 0.2,
      is_eye_closed: false,
      fused_closure_prob: 0.05,
      perclos_pct: 2.0,
      blink_duration_ms: 180,
      blink_rcr: 0.95,
      blink_rate_per_min: 15,
      mar_yawn: 0.12,
    },
    cascade: {
      experiment_mode: "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
      cnn_triggered: false,
      geom_open_prob: 0.95,
      geom_reliability: 0.95,
      cnn_open_prob: null,
      cnn_reliability: null,
      arbitration_winner: "GEOMETRY",
      disagreement_flag: false,
      cnn_latency_ms: 0,
    },
    engine_mode: "MONITORING",
    inference_latency_ms: 18.5,
    ...overrides,
  };
}

async function run() {
  console.log("\n======================================================================");
  console.log("  [DriveGuard] LiveKit Transport & FastAPI Contract Sync Test Suite");
  console.log("======================================================================\n");

  await test("1. Profile mapper emits strict snake_case DTO & preserves neutral_rot_mat and dynamic_closure_threshold", () => {
    const identityMat = [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ];
    const dto = buildDriverProfileDtoPayload(
      {
        isCalibrated: true,
        calibrationMode: "FOUR_ZONE",
        centerEarMedian: 0.295,
        centerEarMadScaled: 0.017,
        blinkDurationMedianMs: 220,
        blinkDurationMadMs: 38,
        blinkRcrMedian: 1.04,
        blinkRcrMad: 0.16,
        neutralPitchDeg: -2.1,
        neutralYawDeg: 1.3,
        calibrationReliability: 0.94,
        zoneBaselines: {
          CENTER: {
            ear_median: 0.295,
            ear_mad_scaled: 0.017,
            dynamic_closure_threshold: 0.211,
            pitch_median_deg: -2.1,
            yaw_median_deg: 1.3,
            sample_count: 42,
          },
          _temporalBaseline: { blinkFreqMedianPerMin: 17 },
          _poseMetadata: {
            neutral_rot_mat: identityMat,
            calibration_is_mirrored: true,
          },
        },
      },
      "user_42"
    );

    assert.equal(dto.driver_id, "user_42");
    assert.equal(dto.blink_rcr_median, 1.04);
    assert.equal(dto.blink_rcr_mad, 0.16);
    assert.deepEqual(dto.neutral_rot_mat, identityMat);
    assert.equal(dto.calibration_is_mirrored, true);
    assert.equal(dto.zone_baselines?.CENTER?.dynamic_closure_threshold, 0.211);
    assert.equal("_temporalBaseline" in (dto.zone_baselines ?? {}), false);
    assert.equal("_poseMetadata" in (dto.zone_baselines ?? {}), false);
  });

  await test("2. Local-mode API client attaches Bearer token to sessions, LiveKit token, profile sync, and stop-worker", async () => {
    const calls: Array<{ url: string; method: string; auth: string | null }> = [];
    const mockFetch: typeof fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      calls.push({
        url,
        method: init?.method || "GET",
        auth: headers.get("Authorization"),
      });

      if (url === "/api/auth/backend-token") {
        return new Response(JSON.stringify({ token: "jwt_local_123", userId: "u1", expiresIn: 300 }), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      if (url.includes("/api/v1/sessions")) {
        return new Response(
          JSON.stringify({
            session_id: "sess_local_1",
            driver_id: "u1",
            owner_id: "u1",
            experiment_mode: "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
            is_calibrated: true,
            calibration_mode: "FOUR_ZONE",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.includes("/api/v1/livekit/token")) {
        return new Response(
          JSON.stringify({
            token: "lk_jwt_abc",
            url: "wss://example.livekit.cloud",
            room_name: "sess_local_1",
            participant_identity: "driver_u1_abc",
          }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.includes("/api/v1/livekit/stop-worker")) {
        return new Response(
          JSON.stringify({ status: "SUCCESS", room_name: "sess_local_1", worker_stopped: true }),
          { status: 200, headers: { "Content-Type": "application/json" } }
        );
      }
      if (url.includes("/api/v1/profiles")) {
        return new Response(init?.body as string, {
          status: 200,
          headers: { "Content-Type": "application/json" },
        });
      }
      return new Response("Not Found", { status: 404 });
    }) as typeof fetch;

    const cfg = { mode: "local" as const, localBaseUrl: "http://127.0.0.1:8000", fetchImpl: mockFetch };

    const sess = await createMonitoringSession({ session_id: "sess_local_1", driver_id: "u1" }, cfg);
    assert.equal(sess.session_id, "sess_local_1");

    const lk = await requestLiveKitToken({ roomName: "sess_local_1", driverId: "u1" }, cfg);
    assert.equal(lk.room_name, "sess_local_1");

    const dto = buildDriverProfileDtoPayload({ isCalibrated: true, centerEarMedian: 0.29 }, "u1");
    await syncDriverProfileToFastApi(dto, cfg);

    const stopRes = await stopLiveKitWorker("sess_local_1", cfg);
    assert.equal(stopRes.worker_stopped, true);

    const apiCalls = calls.filter((c) => c.url.startsWith("http://127.0.0.1:8000/api/v1/"));
    assert.equal(apiCalls.length, 4);
    for (const c of apiCalls) {
      assert.equal(c.auth, "Bearer jwt_local_123", `Missing Bearer header on ${c.method} ${c.url}`);
    }
  });

  await test("3. API client throws structured DriveGuardApiError on FastAPI 403/500 responses", async () => {
    const mockFetch: typeof fetch = (async () =>
      new Response(JSON.stringify({ detail: "Forbidden: room owned by another driver" }), {
        status: 403,
        headers: { "Content-Type": "application/json" },
      })) as typeof fetch;

    await assert.rejects(
      () =>
        requestLiveKitToken(
          { roomName: "other_room", driverId: "u2" },
          { mode: "cloud", fetchImpl: mockFetch }
        ),
      (err: unknown) => {
        assert.ok(err instanceof DriveGuardApiError);
        assert.equal(err.status, 403);
        assert.match(err.detail, /Forbidden/);
        return true;
      }
    );
  });

  await test("4. LiveKit session establish & teardown connects, publishes track once, unpublishes, disconnects, and stops worker with auth", async () => {
    const events: string[] = [];
    const mockTrack = { id: "cam-track-1", label: "Webcam HD", readyState: "live" };

    const fakeRoom: LiveKitRoomLike = {
      state: "disconnected",
      localParticipant: {
        publishTrack: async (track: any) => {
          events.push(`publishTrack:${track.id}`);
          return { track };
        },
        unpublishTrack: async (track: any, stopOnUnpublish?: boolean) => {
          events.push(`unpublishTrack:${track.id}:stop=${stopOnUnpublish}`);
        },
        publishData: async () => {},
      },
      connect: async (url: string, token: string) => {
        events.push(`connect:${url}:${token}`);
        fakeRoom.state = "connected";
      },
      disconnect: async (stopTracks?: boolean) => {
        events.push(`disconnect:stopTracks=${stopTracks}`);
        fakeRoom.state = "disconnected";
      },
      on: () => {},
    };

    let stopWorkerAuth: string | null = null;
    const mockFetch: typeof fetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      const headers = new Headers(init?.headers);
      if (url === "/api/auth/backend-token") {
        return new Response(JSON.stringify({ token: "jwt_live_999" }), { status: 200 });
      }
      if (url.includes("/api/v1/livekit/token")) {
        return new Response(
          JSON.stringify({
            token: "lk_room_jwt",
            url: "wss://sfu.example.com",
            room_name: "live_room_1",
            participant_identity: "driver_1_abc",
          }),
          { status: 200 }
        );
      }
      if (url.includes("/api/v1/livekit/stop-worker")) {
        stopWorkerAuth = headers.get("Authorization");
        events.push("stop-worker");
        return new Response(
          JSON.stringify({ status: "SUCCESS", room_name: "live_room_1", worker_stopped: true }),
          { status: 200 }
        );
      }
      return new Response("Not Found", { status: 404 });
    }) as typeof fetch;

    const active = await establishLiveKitSession({
      roomName: "live_room_1",
      driverId: "driver_1",
      experimentMode: "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
      backendMode: "local",
      localBaseUrl: "http://127.0.0.1:8000",
      fetchImpl: mockFetch,
      createRoom: () => fakeRoom,
      videoTrack: mockTrack,
    });

    assert.equal(active.bearerToken, "jwt_live_999");
    assert.deepEqual(events, [
      "connect:wss://sfu.example.com:lk_room_jwt",
      "publishTrack:cam-track-1",
    ]);

    await teardownLiveKitSession({
      room: active.room,
      publishedTrack: active.publishedTrack,
      roomName: "live_room_1",
      backendMode: "local",
      localBaseUrl: "http://127.0.0.1:8000",
      bearerToken: active.bearerToken,
      fetchImpl: mockFetch,
    });

    assert.deepEqual(events, [
      "connect:wss://sfu.example.com:lk_room_jwt",
      "publishTrack:cam-track-1",
      "unpublishTrack:cam-track-1:stop=false",
      "disconnect:stopTracks=false",
      "stop-worker",
    ]);
    assert.equal(stopWorkerAuth, "Bearer jwt_live_999");
  });

  await test("5. DataChannel protocol preserves authoritative backend reason codes verbatim without client filtering", () => {
    const tracker = createProtocolSequenceTracker();
    const t1 = makeMockTelemetry({
      frame_index: 10,
      frame_seq: "10",
      state: "NORMAL",
      reason_codes: ["BLINK_DURATION_ABOVE_PERSONAL_BASELINE", "MILD_PERCLOS"],
      eye_metrics: {
        ...makeMockTelemetry().eye_metrics,
        blink_duration_ms: 210,
        eye_closure_duration_ms: 190,
      },
    });

    const res1 = processDataChannelMessage(
      { type: "TELEMETRY", frame_seq: "10", telemetry: t1 },
      tracker
    );
    assert.equal(res1.kind, "TELEMETRY");
    assert.deepEqual(res1.telemetry?.reason_codes, [
      "BLINK_DURATION_ABOVE_PERSONAL_BASELINE",
      "MILD_PERCLOS",
    ]);

    // Stale/duplicate sequence rejected
    const resDup = processDataChannelMessage(
      { type: "TELEMETRY", frame_seq: "10", telemetry: t1 },
      tracker
    );
    assert.equal(resDup.kind, "IGNORED");
  });

  await test("6. DataChannel SAFETY_EVENT generates EVENT_ACK, deduplicates retries, and handles ERROR & CALIBRATION events", () => {
    const tracker = createProtocolSequenceTracker();
    const safetyPacket = {
      v: 2,
      type: "SAFETY_EVENT",
      event_id: "evt_001",
      session_id: "live_1",
      event_seq: "5",
      session_epoch: "ep_1",
      event_type: "STATE_TRANSITION",
      previous_state: "NORMAL",
      new_state: "CRITICAL",
      timestamp: 1700000000,
      metrics: {
        drowsiness_state: "CRITICAL_WARNING",
        alert_state: "CRITICAL",
        reasons: ["SUSTAINED_EYE_CLOSURE"],
      },
    };

    const first = processDataChannelMessage(safetyPacket, tracker);
    assert.equal(first.kind, "SAFETY_EVENT");
    assert.equal(first.ackToSend?.action, "EVENT_ACK");
    assert.equal(first.ackToSend?.event_id, "evt_001");

    // Retransmission still produces ACK so server clears retry timer, but is deduplicated
    const retry = processDataChannelMessage(safetyPacket, tracker);
    assert.equal(retry.kind, "IGNORED");
    assert.equal(retry.ackToSend?.event_id, "evt_001");

    // Calibration rejection surfaces rejection_reason cleanly
    const calRej = processDataChannelMessage(
      {
        type: "CALIBRATION_REJECTED",
        rejection_reason: "Insufficient stable CENTER samples",
        calibration: {
          current_zone: "CENTER",
          phase_index: 1,
          total_phases: 4,
          zone_progress_pct: 40,
          total_progress_pct: 10,
          stable_samples_count: 10,
          target_samples_count: 35,
          instruction: "Look straight ahead",
          is_head_in_target_zone: false,
          stability_feedback: "Face unstable",
          is_completed: true,
        },
      },
      tracker
    );
    assert.equal(calRej.kind, "CALIBRATION_EVENT");
    assert.equal(calRej.calibration?.rejection_reason, "Insufficient stable CENTER samples");

    // Worker ERROR surfaced cleanly
    const errRes = processDataChannelMessage(
      { type: "ERROR", action: "START_CALIBRATION", error: "Session closed" },
      tracker
    );
    assert.equal(errRes.kind, "ERROR");
    assert.equal(errRes.errorMessage, "Session closed");
  });

  await test("7. Honest UI & backend processing states: never NORMAL when null, stale, or MONITORING_UNAVAILABLE", () => {
    // Waiting for initial frame -> awaiting_telemetry, after 11s -> stale
    const awaiting = computeBackendProcessingStatus({
      connectionStatus: "connected",
      connectedAtMs: 1000,
      lastTelemetryAtMs: null,
      nowMs: 4000,
    });
    assert.equal(awaiting.status, "awaiting_telemetry");
    assert.equal(awaiting.isStale, false);

    const neverArrivedStale = computeBackendProcessingStatus({
      connectionStatus: "connected",
      connectedAtMs: 1000,
      lastTelemetryAtMs: null,
      nowMs: 12000,
    });
    assert.equal(neverArrivedStale.status, "stale");
    assert.equal(neverArrivedStale.isStale, true);

    // Effective display state checks
    assert.equal(resolveEffectiveDisplayState(null, { isConnected: true }), "AWAITING_TELEMETRY");
    assert.equal(
      resolveEffectiveDisplayState(makeMockTelemetry({ state: "NORMAL" }), {
        isConnected: true,
        isStale: true,
      }),
      "STALE"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "MONITORING_UNAVAILABLE", drowsiness_state: "INSUFFICIENT_DATA" }),
        { isConnected: true, isStale: false }
      ),
      "MONITORING_UNAVAILABLE"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "NORMAL", drowsiness_state: "ATTENTION", alert_state: "ATTENTION" }),
        { isConnected: true, isStale: false }
      ),
      "ATTENTION"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "CRITICAL", drowsiness_state: "CRITICAL_WARNING" }),
        { isConnected: true, isStale: false }
      ),
      "CRITICAL"
    );
  });

  await test("8. All 16 authoritative backend ReasonCode values exist in REASON_CODE_CATALOG", () => {
    const authoritativeCodes = [
      "SUSTAINED_EYE_CLOSURE",
      "NODDING_MICROSLEEP",
      "SEVERE_PERCLOS_CRISIS",
      "PERCLOS_ELEVATED",
      "PROLONGED_BLINK",
      "FREQUENT_YAWNING",
      "PERSISTENT_DISTRACTION",
      "AUTHENTIC_YAWN",
      "EXTENDED_CLOSURE",
      "SUSPECTED_HEAD_SLUMP",
      "MILD_PERCLOS",
      "MIRROR_CHECK_CONTEXT",
      "DASHBOARD_CHECK_CONTEXT",
      "ALERT_NORMAL",
      "FACE_LOST",
      "SENSOR_DEGRADED",
    ];
    for (const code of authoritativeCodes) {
      assert.ok(REASON_CODE_CATALOG[code], `Missing ReasonCode in frontend catalog: ${code}`);
      const explained = explainReasonCode(code);
      assert.equal(explained.code, code);
      assert.ok(explained.title.length > 0);
      assert.ok(explained.description.length > 0);
    }
  });

  console.log(`\n[Results] ${passed}/${passed + failed} passed, ${failed} failed.\n`);
  if (failed > 0) {
    process.exit(1);
  }
}

run();
