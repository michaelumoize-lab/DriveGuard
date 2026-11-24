/**
 * DriveGuard Phase 0 Contract & Invariant Tests:
 * - Canonical server-side profile hashing & validation
 * - OperationalState vs DrowsinessState precedence
 * - SensitivityLevel enum normalization (legacy HIGHWAY compatibility)
 * - Dual-engine profile mapper metadata propagation
 */

import assert from "node:assert/strict";
import {
  computeCanonicalProfileHash,
  validateCalibrationBiometrics,
  type CanonicalProfileInput,
} from "../lib/calibration/profile-hash";
import { buildDriverProfileDtoPayload } from "../lib/api/profile-mapper";
import { OperationalState } from "../lib/ai/types";
import {
  resolveEffectiveDisplayState,
  type FrameTelemetry,
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

function makeBaseProfile(overrides: Partial<CanonicalProfileInput> = {}): CanonicalProfileInput {
  return {
    engineType: "PYTHON",
    calibrationMode: "4_ZONE",
    isCalibrated: true,
    centerEarMedian: 0.285,
    centerEarMadScaled: 0.016,
    blinkDurationMedianMs: 240,
    blinkDurationMadMs: 38,
    blinkRcrMedian: 0.98,
    blinkRcrMad: 0.15,
    blinkFreqMedianPerMin: 16,
    blinkFreqMadPerMin: 4.5,
    baselinePerclosPct: 3.5,
    blinkAmplitudeMedian: 0.14,
    temporalBlinksSampled: 18,
    temporalWindowSec: 120,
    sensitivityLevel: "STANDARD",
    neutralPitchDeg: -1.2,
    neutralYawDeg: 0.8,
    neutralRotMat: [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1],
    ],
    calibrationIsMirrored: false,
    calibrationReliability: 0.91,
    zoneBaselines: {
      CENTER: {
        zone: "CENTER",
        ear_median: 0.285,
        ear_mad_scaled: 0.016,
        dynamic_closure_threshold: 0.21,
        pitch_median_deg: -1.2,
        yaw_median_deg: 0.8,
        sample_count: 60,
      },
      LEFT_MIRROR: {
        zone: "LEFT_MIRROR",
        ear_median: 0.252,
        ear_mad_scaled: 0.019,
        dynamic_closure_threshold: 0.19,
        pitch_median_deg: -2.0,
        yaw_median_deg: -28.0,
        sample_count: 55,
      },
    },
    ...overrides,
  };
}

function makeMockTelemetry(overrides: Partial<FrameTelemetry> = {}): FrameTelemetry {
  return {
    session_id: "sess_phase0",
    driver_id: "drv_phase0",
    frame_index: 1,
    timestamp_ms: 1700000000000,
    fps: 24,
    face_detected: true,
    state: "NORMAL",
    drowsiness_state: "NORMAL",
    alert_state: "NORMAL",
    monitoring_status: "MONITORING_ACTIVE",
    risk_score: 0.05,
    pose_zone: "CENTER",
    reason_codes: ["ALERT_NORMAL"],
    head_pose: { pitch_deg: 0, yaw_deg: 0, roll_deg: 0 },
    eye_metrics: {
      ear_raw: 0.29,
      ear_left: 0.29,
      ear_right: 0.29,
      ear_normalized: 0.29,
      ear_closure_threshold: 0.21,
      is_eye_closed: false,
      fused_closure_prob: 0.02,
      perclos_pct: 2.0,
      blink_duration_ms: 210,
      blink_rcr: 1.0,
      blink_rate_per_min: 15,
      total_blink_count: 4,
      mar_yawn: 0.25,
    },
    cascade: {
      experiment_mode: "EXPERIMENT_A_GEOMETRIC_ONLY",
      cnn_triggered: false,
      geom_open_prob: 0.98,
      geom_reliability: 0.95,
      cnn_open_prob: null,
      cnn_reliability: null,
      arbitration_winner: "GEOMETRY",
      disagreement_flag: false,
      cnn_latency_ms: 0,
    },
    engine_mode: "MONITORING",
    inference_latency_ms: 18,
    ...overrides,
  };
}

async function main() {
  console.log("Running DriveGuard Phase 0 Contract Tests...\n");

  await test("canonical profile hash is deterministic across zone key order and ignores volatile metadata", () => {
    const a = makeBaseProfile({
      zoneBaselines: {
        LEFT_MIRROR: {
          zone: "LEFT_MIRROR",
          ear_median: 0.252,
          ear_mad_scaled: 0.019,
          dynamic_closure_threshold: 0.19,
          pitch_median_deg: -2.0,
          yaw_median_deg: -28.0,
          sample_count: 55,
        },
        CENTER: {
          zone: "CENTER",
          ear_median: 0.2850000001,
          ear_mad_scaled: 0.016,
          dynamic_closure_threshold: 0.21,
          pitch_median_deg: -1.2,
          yaw_median_deg: 0.8,
          sample_count: 60,
        },
        _temporalBaseline: { calibratedAt: "2026-11-24T00:00:00.000Z" },
      },
    });
    const b = makeBaseProfile({
      zoneBaselines: {
        CENTER: {
          zone: "CENTER",
          earMedian: 0.285,
          earMadScaled: 0.016,
          dynamicClosureThreshold: 0.21,
          pitchMedianDeg: -1.2,
          yawMedianDeg: 0.8,
          sampleCount: 60,
        },
        LEFT_MIRROR: {
          zone: "LEFT_MIRROR",
          earMedian: 0.252,
          earMadScaled: 0.019,
          dynamicClosureThreshold: 0.19,
          pitchMedianDeg: -2.0,
          yawMedianDeg: -28.0,
          sampleCount: 55,
        },
        _temporalBaseline: { calibratedAt: "2099-01-01T12:00:00.000Z" },
      },
    });

    const hashA = computeCanonicalProfileHash(a);
    const hashB = computeCanonicalProfileHash(b);
    assert.equal(hashA.length, 64);
    assert.equal(hashA, hashB);
  });

  await test("canonical profile hash strictly separates BROWSER and PYTHON engines", () => {
    const pyHash = computeCanonicalProfileHash(makeBaseProfile({ engineType: "PYTHON" }));
    const brHash = computeCanonicalProfileHash(makeBaseProfile({ engineType: "BROWSER" }));
    assert.notEqual(pyHash, brHash);
  });

  await test("biometric validation rejects invalid EAR, rotation matrices, and zone EAR", () => {
    assert.equal(validateCalibrationBiometrics(makeBaseProfile()).valid, true);
    assert.equal(validateCalibrationBiometrics(makeBaseProfile({ centerEarMedian: 0 })).valid, false);
    assert.equal(validateCalibrationBiometrics(makeBaseProfile({ centerEarMedian: 1.5 })).valid, false);
    assert.equal(
      validateCalibrationBiometrics(
        makeBaseProfile({ neutralRotMat: [[1, 0], [0, 1]] as unknown as number[][] })
      ).valid,
      false
    );
    assert.equal(
      validateCalibrationBiometrics(
        makeBaseProfile({
          zoneBaselines: { CENTER: { ear_median: -0.1 } },
        })
      ).valid,
      false
    );
  });

  await test("OperationalState supersedes DrowsinessState (uncalibrated/degraded/error never show NORMAL)", () => {
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "NORMAL", operational_state: "CALIBRATION_REQUIRED" })
      ),
      "CALIBRATION_REQUIRED"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "NORMAL", operational_state: "CALIBRATING" })
      ),
      "CALIBRATING"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "NORMAL", operational_state: "DEGRADED" })
      ),
      "DEGRADED"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "NORMAL", operational_state: "ERROR" })
      ),
      "ERROR"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "NORMAL", operational_state: "MONITORING_ACTIVE" })
      ),
      "NORMAL"
    );
    assert.equal(
      resolveEffectiveDisplayState(
        makeMockTelemetry({ state: "CRITICAL", operational_state: "MONITORING_ACTIVE" })
      ),
      "CRITICAL"
    );
  });

  await test("OperationalState enum exports all 5 authoritative states", () => {
    assert.deepEqual(Object.values(OperationalState).sort(), [
      "CALIBRATING",
      "CALIBRATION_REQUIRED",
      "DEGRADED",
      "ERROR",
      "MONITORING_ACTIVE",
    ]);
  });

  await test("profile mapper propagates dual-engine metadata fields cleanly", () => {
    const dto = buildDriverProfileDtoPayload(
      {
        engineType: "PYTHON",
        profileVersion: 3,
        profileUuid: "11111111-2222-4333-8444-555555555555",
        contentHash: "abc123",
        isCalibrated: true,
        calibrationMode: "4_ZONE",
        centerEarMedian: 0.29,
        sensitivityLevel: "RELAXED",
      },
      "user_123"
    );
    assert.equal(dto.engine_type, "PYTHON");
    assert.equal(dto.profile_version, 3);
    assert.equal(dto.profile_uuid, "11111111-2222-4333-8444-555555555555");
    assert.equal(dto.content_hash, "abc123");
    assert.equal(dto.sensitivity_level, "RELAXED");
  });

  console.log(`\nPhase 0 Contract Tests Complete: ${passed} passed, ${failed} failed.`);
  if (failed > 0) {
    process.exit(1);
  }
}

main();
