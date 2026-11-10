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
  blink_state?: string;
  blink_classification?: string;
  perclos_coverage_pct?: number;
  perclos_reliable?: boolean;
  eye_closure_duration_ms?: number;
  sampling_gap_detected?: boolean;
  blink_avr?: number | null;
  blink_cv_ibi?: number | null;
  gaze_entropy?: number | null;
  brow_furrow_au4?: number | null;
  yawn_envelope_confidence?: number | null;
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
  server_ingest_mono_ms?: number;
  server_publish_mono_ms?: number;
  server_wall_epoch_ms?: number;
}

export interface FrameTelemetry {
  session_id: string;
  driver_id: string;
  frame_index: number;
  frame_seq?: string;
  event_seq?: string;
  worker_boot_id?: string;
  session_epoch?: string;
  timestamp_ms: number;
  fps: number;
  face_detected: boolean;
  state: AlertState;
  monitoring_status?: "MONITORING_ACTIVE" | "FACE_VISIBILITY_DEGRADED" | "MONITORING_UNAVAILABLE" | "MONITORING_RECOVERING";
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
  event_seq: string | number;
  frame_seq?: string;
  worker_boot_id?: string;
  session_epoch?: string;
  event_type: string;
  previous_state: AlertState;
  new_state: AlertState;
  timestamp: number;
  server_wall_epoch_ms?: number;
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
  event_seq: string | number;
  client_timestamp?: number;
}

export interface TelemetryPacket {
  v: number;
  type: "TELEMETRY";
  session_id: string;
  seq: number | string;
  frame_seq?: string;
  event_seq?: string;
  worker_boot_id?: string;
  session_epoch?: string;
  server_wall_epoch_ms?: number;
  timestamp: number;
  telemetry: FrameTelemetry;
  calibration?: CalibrationPhase | null;
  echo_client_ts?: number;
}

export interface LatencyDiagnostics {
  datachannel_rtt_ms: number;
  queue_latency_ms: number;
  infer_duration_ms: number;
  server_total_ms: number;
  rtt_p50: number;
  rtt_p95: number;
  telemetry_dispatch_ms?: number;
  audio_output_latency_estimate_ms?: number;
  last_measured_at: number;
}

export interface CalibratedProfilePayload {
  driver_id: string;
  calibration_mode: string;
  center_ear_median: number;
  blink_duration_median_ms: number;
  blink_duration_mad_ms: number;
  blink_rcr_median: number;
  blink_rcr_mad: number;
  blink_freq_median_per_min?: number;
  blink_freq_mad_per_min?: number;
  baseline_perclos_pct?: number;
  blink_amplitude_median?: number;
  temporal_blinks_sampled?: number;
  temporal_window_sec?: number;
  neutral_pitch_deg: number;
  neutral_yaw_deg: number;
  calibration_reliability: number;
  zone_baselines?: Record<string, ZoneBaselineEntry>;
}

export interface CalibrationPhase {
  current_zone: string;
  phase_index: number;
  total_phases: number;
  phase_type?: "SPATIAL_ZONE" | "BLINK_CADENCE" | string;
  zone_progress_pct: number;
  total_progress_pct: number;
  stable_samples_count: number;
  target_samples_count: number;
  elapsed_sec?: number;
  target_duration_sec?: number;
  collected_blinks_count?: number;
  live_blink_rate_per_min?: number;
  last_blink_duration_ms?: number | null;
  live_median_blink_duration_ms?: number | null;
  live_median_blink_rcr?: number | null;
  instruction: string;
  is_head_in_target_zone: boolean;
  stability_feedback: string;
  is_completed: boolean;
  rejection_reason?: string | null;
  quality_warnings?: string[];
  profile?: CalibratedProfilePayload | null;
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

export interface ZoneBaselineEntry {
  ear_median: number;
  ear_mad_scaled?: number;
  dynamic_closure_threshold?: number;
  sample_count?: number;
  pitch_deg_median?: number;
  yaw_deg_median?: number;
}

export interface SessionEndPayload {
  sessionCode: string;
  durationSeconds: number;
  averageFps: number;
  averageLatencyMs: number;
  averagePerclosPct: number;
  maxPerclosPct: number;
  totalBlinks: number;
  averageBlinkRate: number;
  normalFramesCount: number;
  warningAlertsCount: number;
  criticalAlertsCount: number;
  mirrorSuppressions: number;
  // Backward-compatible legacy aliases
  durationSec?: number;
  totalFrames?: number;
  maxPerclos?: number;
  criticalAlerts?: number;
}

export interface AlertLogPayload {
  sessionCode: string;
  severity: "WARNING" | "CRITICAL";
  riskScore: number;
  earValue: number;
  perclosPct: number;
  poseZone: string;
  reasons: string[];
  cascadeDecision: string;
  timestampMs?: number;
}

export interface ReasonExplanation {
  code: string;
  title: string;
  description: string;
  explanation?: string;
  subsystem: "KINEMATICS" | "PERCLOS" | "GEOMETRY" | "HEAD_POSE" | "CONTEXT" | "QUALITY";
  severity: "INFO" | "WARNING" | "CRITICAL";
}

export const REASON_CODE_CATALOG: Record<string, ReasonExplanation> = {
  PROLONGED_EYE_CLOSURE_MICROSLEEP: {
    code: "PROLONGED_EYE_CLOSURE_MICROSLEEP",
    title: "Microsleep / Prolonged Closure",
    description: "Continuous eye closure exceeded the personal microsleep safety threshold (>= 1.1s or 3.5x baseline blink duration).",
    subsystem: "KINEMATICS",
    severity: "CRITICAL",
  },
  REPEATED_ABNORMAL_BLINK_CLUSTER: {
    code: "REPEATED_ABNORMAL_BLINK_CLUSTER",
    title: "Abnormal Blink Cluster",
    description: "Multiple prolonged blinks or sluggish reopening events detected within the rolling temporal window.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  PROLONGED_BLINK_DETECTED: {
    code: "PROLONGED_BLINK_DETECTED",
    title: "Prolonged Single Blink",
    description: "Completed blink duration significantly exceeded the driver's calibrated median blink duration.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  BLINK_DURATION_ABOVE_PERSONAL_BASELINE: {
    code: "BLINK_DURATION_ABOVE_PERSONAL_BASELINE",
    title: "Elevated Blink Duration (z >= 1.8)",
    description: "Blink duration robust z-score is elevated above personal baseline or current closure exceeds 450 ms.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  EYELID_REOPENING_VELOCITY_SLOWED: {
    code: "EYELID_REOPENING_VELOCITY_SLOWED",
    title: "Sluggish Eyelid Reopening (RCR)",
    description: "Reopening-to-Closing velocity ratio (RCR) dropped significantly below the driver's alert baseline (z >= 1.8).",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  ELEVATED_BLINK_AMPLITUDE_VELOCITY_RATIO: {
    code: "ELEVATED_BLINK_AMPLITUDE_VELOCITY_RATIO",
    title: "Sluggish Eyelid Dynamics (AVR)",
    description: "Eyelid reopening velocity significantly slowed relative to blink amplitude (Optalert JDS metric).",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  ERRATIC_INTER_BLINK_INTERVAL_VARIABILITY: {
    code: "ERRATIC_INTER_BLINK_INTERVAL_VARIABILITY",
    title: "Erratic Blink Cadence (CV_IBI)",
    description: "Inter-blink intervals show high variability (CV >= 0.75), indicating staring pauses followed by burst blinks.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  INVOLUNTARY_BROW_STRAIN_AU4: {
    code: "INVOLUNTARY_BROW_STRAIN_AU4",
    title: "Involuntary Brow Strain (AU4)",
    description: "Facial Action Unit AU4 brow furrowing detected while eyelids are drooping, indicating active struggle against sleep.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  VISUAL_DISTRACTION_OFF_ROAD_DWELL: {
    code: "VISUAL_DISTRACTION_OFF_ROAD_DWELL",
    title: "Off-Road Gaze Distraction (Euro NCAP)",
    description: "Continuous gaze averted from the forward roadway exceeded 3.0 seconds during transit.",
    subsystem: "CONTEXT",
    severity: "WARNING",
  },
  COGNITIVE_TUNNELING_BLANK_STARE: {
    code: "COGNITIVE_TUNNELING_BLANK_STARE",
    title: "Blank Stare / Cognitive Tunneling",
    description: "Rigid forward gaze with near-zero saccadic eye or head movement detected for >= 5.0 seconds.",
    subsystem: "CONTEXT",
    severity: "CRITICAL",
  },
  LOW_GAZE_TRANSITION_ENTROPY: {
    code: "LOW_GAZE_TRANSITION_ENTROPY",
    title: "Low Gaze Scanpath Entropy",
    description: "Visual transition entropy (H_t) dropped below 0.35, indicating environmental situational tunneling.",
    subsystem: "CONTEXT",
    severity: "WARNING",
  },
  INCREASED_PERCLOS: {
    code: "INCREASED_PERCLOS",
    title: "Elevated 60s Rolling PERCLOS",
    description: "Time-weighted Percentage of Eye Closure over the rolling 60-second window exceeded healthy baseline (>= 12% or z >= 1.6).",
    subsystem: "PERCLOS",
    severity: "CRITICAL",
  },
  SUSTAINED_EYE_OPENNESS_REDUCTION: {
    code: "SUSTAINED_EYE_OPENNESS_REDUCTION",
    title: "Sustained Narrowed Aperture",
    description: "5-second rolling Eye Aspect Ratio (EAR) dropped >= 2.0 MAD below the active zone's personal baseline.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  SUSTAINED_DOWNWARD_HEAD_PITCH: {
    code: "SUSTAINED_DOWNWARD_HEAD_PITCH",
    title: "Sustained Head Nodding",
    description: "Downward head pitch (<= -14 deg below neutral) persisted for >= 1.6 seconds outside mirror-check context.",
    subsystem: "HEAD_POSE",
    severity: "CRITICAL",
  },
  FREQUENT_OR_PROLONGED_YAWNING: {
    code: "FREQUENT_OR_PROLONGED_YAWNING",
    title: "Frequent or Prolonged Yawning",
    description: "Mouth Aspect Ratio (MAR) indicated active yawning or >= 2 yawns within the 5-minute window.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  FACE_NOT_VISIBLE_OR_OBSCURED: {
    code: "FACE_NOT_VISIBLE_OR_OBSCURED",
    title: "Face Lost or Camera Occluded",
    description: "Driver's facial landmarks were unavailable or geometric reliability dropped below 0.22 for >= 1.2 seconds.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE: {
    code: "INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE",
    title: "Low PERCLOS Observation Coverage",
    description: "Valid face tracking coverage across the 60-second window is below 50%; PERCLOS weight is held back until coverage recovers.",
    subsystem: "QUALITY",
    severity: "INFO",
  },
  OBSERVATION_SAMPLING_GAP_DETECTED: {
    code: "OBSERVATION_SAMPLING_GAP_DETECTED",
    title: "Frame Sampling Gap Detected",
    description: "An inter-frame gap occurred; microsleep timers are protected against false triggers across dropped frames.",
    subsystem: "QUALITY",
    severity: "INFO",
  },
};

export function explainReasonCode(code: string): ReasonExplanation {
  if (REASON_CODE_CATALOG[code]) {
    const entry = REASON_CODE_CATALOG[code];
    return {
      ...entry,
      explanation: entry.description,
    };
  }
  if (code.startsWith("CONTEXT_MIRROR_CHECK_")) {
    const zone = code.replace("CONTEXT_MIRROR_CHECK_", "").replace(/_/g, " ");
    const description = `Transient gaze into ${zone} (<= 3.0s) uses the zone-conditioned baseline and suppresses false head-pose penalties.`;
    return {
      code,
      title: `Mirror / Instrument Glance (${zone})`,
      description,
      explanation: description,
      subsystem: "CONTEXT",
      severity: "INFO",
    };
  }
  const formattedTitle = code
    .split("_")
    .map((w) => (w.length > 0 ? w[0].toUpperCase() + w.slice(1).toLowerCase() : ""))
    .join(" ");
  const fallbackDesc = "Diagnostic signal emitted by the DriveGuard adaptive risk engine.";
  return {
    code,
    title: formattedTitle,
    description: fallbackDesc,
    explanation: fallbackDesc,
    subsystem: "GEOMETRY",
    severity: "WARNING",
  };
}

