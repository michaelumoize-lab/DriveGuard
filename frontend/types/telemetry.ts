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
}

export interface ServerTelemetryMessage {
  type: "SESSION_READY" | "TELEMETRY" | "FATAL_ERROR";
  session_id?: string;
  driver_id?: string;
  experiment_mode?: string;
  detail?: string;
  telemetry?: FrameTelemetry;
  calibration?: any;
}
