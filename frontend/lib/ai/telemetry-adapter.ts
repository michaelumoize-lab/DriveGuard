/**
 * DriveGuard — Adapter mapping Browser AI Telemetry to existing Cockpit / HUD FrameTelemetry contracts.
 */

import {
  AlertState,
  AuthoritativeAlertState,
  AuthoritativeDrowsinessState,
  FrameTelemetry,
  MonitoringStatus,
  OperationalState as TelemetryOperationalState,
} from "@/types/telemetry";
import { DetectionTelemetry, DrowsinessState, OperationalState } from "./types";

export function convertBrowserTelemetryToFrameTelemetry(
  t: DetectionTelemetry,
  sessionId: string = "browser_session",
  driverId: string = "driver_default"
): FrameTelemetry {
  const operationalState: TelemetryOperationalState =
    t.operationalState ??
    (t.hasFace ? OperationalState.MONITORING_ACTIVE : OperationalState.DEGRADED);

  const isUnavailable =
    !t.hasFace ||
    operationalState === OperationalState.CALIBRATION_REQUIRED ||
    operationalState === OperationalState.CALIBRATING ||
    operationalState === OperationalState.DEGRADED ||
    operationalState === OperationalState.ERROR ||
    t.alertState === DrowsinessState.INSUFFICIENT_DATA;

  let displayState: AlertState = "NORMAL";
  let authAlertState: AuthoritativeAlertState | null = "NORMAL";

  if (isUnavailable) {
    displayState = "MONITORING_UNAVAILABLE";
    authAlertState = null;
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

  const monitoringStatus: MonitoringStatus =
    operationalState === OperationalState.MONITORING_ACTIVE && t.hasFace
      ? "MONITORING_ACTIVE"
      : "MONITORING_UNAVAILABLE";

  const riskScore =
    displayState === "MONITORING_UNAVAILABLE"
      ? 0.0
      : displayState === "CRITICAL"
        ? 0.95
        : displayState === "WARNING"
          ? authAlertState === "WARNING"
            ? 0.65
            : 0.45
          : 0.05;

  const closureThreshold = t.earClosureThreshold ?? 0.20;
  const blinkRatePerMin = t.blinkCount60s ?? 0;
  const totalBlinkCount = t.totalBlinkCount ?? t.blinkCount;
  const perclosCoveragePct = t.perclosCoveragePct ?? 0.0;
  const perclosReliable = t.perclosReliable ?? false;

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
    operational_state: operationalState,
    engine_type: t.engineType ?? "BROWSER",
    authoritative_engine: "BrowserDecisionEngine",
    monitoring_status: monitoringStatus,
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
      ear_closure_threshold: closureThreshold,
      is_eye_closed: t.isClosed,
      fused_closure_prob: t.isClosed ? 0.95 : 0.05,
      perclos_pct: t.perclosPct,
      blink_duration_ms: t.recentBlinkDurationMs,
      blink_rcr: t.recentBlinkRcr,
      blink_rate_per_min: blinkRatePerMin,
      total_blink_count: totalBlinkCount,
      mar_yawn: t.mar,
      total_yawn_count: t.yawnCount,
      is_yawning: t.isYawning,
      eye_closure_duration_ms: t.closureDurationMs,
      perclos_coverage_pct: perclosCoveragePct,
      perclos_reliable: perclosReliable,
    },
    cascade: {
      experiment_mode: "EXPERIMENT_A_GEOMETRIC_ONLY",
      cnn_triggered: false,
      geom_open_prob: t.isClosed ? 0.05 : 0.95,
      geom_reliability: t.hasFace ? 0.95 : 0.0,
      cnn_open_prob: null,
      cnn_reliability: null,
      arbitration_winner: "GEOMETRIC_AUTHORITATIVE",
      disagreement_flag: false,
      cnn_latency_ms: 0.0,
      cnn_execution_mode: "NOT_APPLICABLE",
      verifier_invoked: false,
      gate_trigger_reason: "BROWSER_GEOMETRIC_ONLY",
      onnx_inference_executed: false,
      cnn_inference_duration_ms: null,
    },
    comparison: {
      fixed_cutoff_ear: 0.22,
      fixed_alert_state: displayState,
      fixed_false_alarm: false,
      personalised_zone: t.poseZone,
      personalised_threshold: closureThreshold,
      is_mirror_check_suppressed:
        t.reasons.includes("MIRROR_CHECK_CONTEXT"),
    },
    engine_mode:
      operationalState === OperationalState.CALIBRATING
        ? "CALIBRATING"
        : "MONITORING",
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
