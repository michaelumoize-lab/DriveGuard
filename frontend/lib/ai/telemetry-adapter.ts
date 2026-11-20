/**
 * DriveGuard — Adapter mapping Browser AI Telemetry to existing Cockpit / HUD FrameTelemetry contracts.
 */

import { FrameTelemetry, AlertState, AuthoritativeDrowsinessState, AuthoritativeAlertState } from "@/types/telemetry";
import { DetectionTelemetry, DrowsinessState } from "./types";

export function convertBrowserTelemetryToFrameTelemetry(
  t: DetectionTelemetry,
  sessionId: string = "browser_session",
  driverId: string = "driver_default"
): FrameTelemetry {
  let displayState: AlertState = "NORMAL";
  let authAlertState: AuthoritativeAlertState = "NORMAL";

  if (!t.hasFace) {
    displayState = "MONITORING_UNAVAILABLE";
  } else if (t.alertState === DrowsinessState.CRITICAL_WARNING) {
    displayState = "CRITICAL";
    authAlertState = "CRITICAL";
  } else if (t.alertState === DrowsinessState.DROWSINESS_WARNING) {
    displayState = "WARNING";
    authAlertState = "WARNING";
  } else if (t.alertState === DrowsinessState.ATTENTION) {
    displayState = "WARNING";
    authAlertState = "ATTENTION";
  } else {
    displayState = "NORMAL";
    authAlertState = "NORMAL";
  }

  const riskScore =
    displayState === "CRITICAL"
      ? 0.95
      : displayState === "WARNING"
        ? (authAlertState === "WARNING" ? 0.65 : 0.45)
        : 0.05;

  return {
    session_id: sessionId,
    driver_id: driverId,
    frame_index: t.frameId,
    timestamp_ms: t.timestampMs,
    fps: t.inferenceFps,
    face_detected: t.hasFace,
    face_visible: t.hasFace,
    state: displayState,
    drowsiness_state: t.alertState as AuthoritativeDrowsinessState,
    alert_state: authAlertState,
    monitoring_status: t.hasFace ? "MONITORING_ACTIVE" : "MONITORING_UNAVAILABLE",
    risk_score: riskScore,
    pose_zone: t.poseZone,
    reason_codes: t.reasons,
    head_pose: {
      pitch_deg: t.pitchDeg,
      yaw_deg: t.yawDeg,
      roll_deg: t.rollDeg,
    },
    eye_metrics: {
      ear_raw: t.rawEar,
      ear_left: t.leftEar,
      ear_right: t.rightEar,
      ear_normalized: t.ear,
      ear_closure_threshold: 0.20,
      is_eye_closed: t.isClosed,
      fused_closure_prob: t.isClosed ? 0.95 : 0.05,
      perclos_pct: t.perclosPct,
      blink_duration_ms: t.recentBlinkDurationMs,
      blink_rcr: t.recentBlinkRcr,
      blink_rate_per_min: t.blinkCount,
      mar_yawn: t.mar,
      eye_closure_duration_ms: t.closureDurationMs,
      perclos_coverage_pct: 100.0,
      perclos_reliable: true,
    },
    cascade: {
      experiment_mode: "BROWSER_AI_TYPESCRIPT",
      cnn_triggered: false,
      geom_open_prob: t.isClosed ? 0.05 : 0.95,
      geom_reliability: 0.95,
      cnn_open_prob: null,
      cnn_reliability: null,
      arbitration_winner: "BROWSER_GEOMETRIC_TEMPORAL",
      disagreement_flag: false,
      cnn_latency_ms: 0.0,
    },
    comparison: {
      fixed_cutoff_ear: 0.20,
      fixed_alert_state: displayState,
      fixed_false_alarm: false,
      personalised_zone: t.poseZone,
      personalised_threshold: 0.20,
      is_mirror_check_suppressed: t.poseZone === "LEFT_MIRROR" || t.poseZone === "RIGHT_MIRROR",
    },
    engine_mode: "Browser AI (MediaPipe + TypeScript Edge)",
    runtime_mode: "browser",
    runtimeMode: "browser",
    inference_latency_ms: t.inferenceLatencyMs,
    network_roundtrip_ms: 0.0,
    timing: {
      queue_latency_ms: 0.0,
      infer_duration_ms: t.inferenceLatencyMs,
      server_total_ms: t.inferenceLatencyMs,
    },
  };
}
