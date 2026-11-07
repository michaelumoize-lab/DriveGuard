/**
 * TypeScript data schemas for real-time WebSocket telemetry and cascade diagnostic HUD.
 * Mirrors FastAPI backend/app/schemas/telemetry.py exactly.
 */

export type AlertState = "NORMAL" | "WARNING" | "CRITICAL" | "MONITORING_UNAVAILABLE";

export type PoseZone = "CENTER" | "LEFT_MIRROR" | "RIGHT_MIRROR" | "DASHBOARD" | "OUT_OF_BOUNDS";

export type ExperimentMode =
  | "EXPERIMENT_A_GEOMETRIC_ONLY"
  | "EXPERIMENT_B_CNN_ALWAYS"
  | "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE";

export interface HeadPose {
  pitch_deg: number;
  yaw_deg: number;
  roll_deg: number;
}

export interface EyeMetrics {
  ear_raw: number;
  ear_left: number;
  ear_right: number;
  ear_normalized: number;
  ear_closure_threshold: number;
  is_eye_closed: boolean;
  fused_closure_prob: number;
  perclos_pct: number;
  blink_duration_ms: number | null;
  blink_rcr: number | null;
  blink_rate_per_min: number;
  mar_yawn: number;
}

export interface CascadeStatus {
  experiment_mode: string;
  cnn_triggered: boolean;
  geom_open_prob: number;
  geom_reliability: number;
  cnn_open_prob: number | null;
  cnn_reliability: number | null;
  arbitration_winner: string;
  disagreement_flag: boolean;
  cnn_latency_ms: number;
}

export interface ConventionalComparison {
  fixed_cutoff_ear: number;
  fixed_alert_state: string;
  fixed_false_alarm: boolean;
  personalised_zone: string;
  personalised_threshold: number;
  is_mirror_check_suppressed: boolean;
}

export interface TimingBreakdown {
  queue_latency_ms: number;
  infer_duration_ms: number;
  server_total_ms: number;
}

export interface FrameTelemetry {
  session_id: string;
  driver_id: string;
  frame_index: number;
  timestamp_ms: number;
  fps: number;
  face_detected: boolean;
  state: AlertState;
  risk_score: number;
  pose_zone: PoseZone;
  reason_codes: string[];
  head_pose: HeadPose;
  eye_metrics: EyeMetrics;
  cascade: CascadeStatus;
  comparison?: ConventionalComparison;
  engine_mode: string;
  inference_latency_ms: number;
  network_roundtrip_ms?: number;
  timing?: TimingBreakdown;
}

export interface SafetyEvent {
  v: number;
  type: "SAFETY_EVENT";
  event_id: string;
  session_id: string;
  event_seq: number;
  event_type: string;
  previous_state: AlertState;
  new_state: AlertState;
  timestamp: number;
  metrics: {
    risk_score?: number;
    ear?: number;
    perclos?: number;
    pose_zone?: string;
    reasons?: string[];
    [key: string]: unknown;
  };
}

export interface EventAck {
  action: "EVENT_ACK";
  event_id: string;
  event_seq: number;
  client_timestamp?: number;
}

export interface TelemetryPacket {
  v: number;
  type: "TELEMETRY";
  session_id: string;
  seq: number;
  timestamp: number;
  telemetry: FrameTelemetry;
  calibration?: CalibrationPhase | null;
  echo_client_ts?: number;
}

export interface LatencyDiagnostics {
  datachannel_rtt_ms: number;
  queue_latency_ms: number;
  infer_duration_ms: number;
  estimated_e2e_latency_ms: number;
  rtt_p50: number;
  rtt_p95: number;
  last_measured_at: number;
}

export interface CalibrationPhase {
  current_zone: string;
  phase_index: number;
  total_phases: number;
  zone_progress_pct: number;
  total_progress_pct: number;
  stable_samples_count: number;
  target_samples_count: number;
  instruction: string;
  is_head_in_target_zone: boolean;
  stability_feedback: string;
  is_completed: boolean;
  rejection_reason?: string | null;
}

export interface ServerTelemetryMessage {
  type: "SESSION_READY" | "TELEMETRY" | "FATAL_ERROR";
  session_id?: string;
  driver_id?: string;
  experiment_mode?: string;
  detail?: string;
  telemetry?: FrameTelemetry;
  calibration?: CalibrationPhase | null;
}
