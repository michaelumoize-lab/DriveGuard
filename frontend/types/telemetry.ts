/**
 * TypeScript data schemas for DriveGuard real-time LiveKit/WebSocket telemetry,
 * calibration phases, driver profiles, and explainable alert reason codes.
 *
 * Mirrors the authoritative FastAPI schemas in:
 * - backend/app/schemas/telemetry.py
 * - backend/app/schemas/calibration.py
 * - backend/app/schemas/profile.py
 * - backend/app/intelligence/reason_codes.py
 */

export type AlertState = "NORMAL" | "WARNING" | "CRITICAL" | "MONITORING_UNAVAILABLE";

export type EngineType = "BROWSER" | "PYTHON";

/**
 * Operational readiness state, strictly separate from drowsiness severity.
 * An uncalibrated driver or degraded camera must never be presented as NORMAL.
 */
export type OperationalState =
  | "CALIBRATION_REQUIRED"
  | "CALIBRATING"
  | "MONITORING_ACTIVE"
  | "DEGRADED"
  | "ERROR";

export type AuthoritativeDrowsinessState =
  | "NORMAL"
  | "ATTENTION"
  | "DROWSINESS_WARNING"
  | "CRITICAL_WARNING"
  | "INSUFFICIENT_DATA";

export type AuthoritativeAlertState = "NORMAL" | "ATTENTION" | "WARNING" | "CRITICAL";

export type MonitoringStatus =
  | "MONITORING_ACTIVE"
  | "FACE_VISIBILITY_DEGRADED"
  | "MONITORING_UNAVAILABLE"
  | "MONITORING_RECOVERING";

export type BackendProcessingStatus =
  | "idle"
  | "connecting"
  | "awaiting_telemetry"
  | "processing"
  | "stale"
  | "error";

export type PoseZone =
  | "CENTER"
  | "LEFT_MIRROR"
  | "RIGHT_MIRROR"
  | "DASHBOARD"
  | "OFF_ROAD"
  | "OUT_OF_BOUNDS"
  | "UNKNOWN";

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
  total_blink_count?: number;
  mar_yawn: number;
  total_yawn_count?: number;
  is_yawning?: boolean;
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
  cnn_execution_mode?: string;
  verifier_invoked?: boolean;
  gate_trigger_reason?: string;
  onnx_inference_executed?: boolean;
  cnn_inference_duration_ms?: number | null;
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
  frame_received_mono_ms?: number | null;
  frame_queue_wait_ms?: number | null;
  frame_convert_ms?: number | null;
  frame_preprocess_ms?: number | null;
  executor_wait_ms?: number | null;
  process_frame_ms?: number | null;
  publish_ms?: number | null;
  frame_age_ms?: number | null;
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
  face_visible?: boolean;
  grace_period_active?: boolean;
  last_seen_ms?: number | null;
  tracking_mode?: string | null;
  state: AlertState;
  drowsiness_state?: AuthoritativeDrowsinessState | null;
  alert_state?: AuthoritativeAlertState | null;
  is_recovering?: boolean;
  authoritative_engine?: string;
  operational_state?: OperationalState;
  engine_type?: EngineType;
  monitoring_status?: MonitoringStatus;
  risk_score: number;
  pose_zone: PoseZone;
  reason_codes: string[];
  head_pose: HeadPose;
  eye_metrics: EyeMetrics;
  cascade: CascadeStatus;
  comparison?: ConventionalComparison;
  engine_mode: string;
  runtime_mode?: string;
  runtimeMode?: string;
  inference_latency_ms: number;
  network_roundtrip_ms?: number;
  timing?: TimingBreakdown | null;
}

export interface SafetyEvent {
  v?: number;
  type?: "SAFETY_EVENT";
  event_id: string;
  session_id?: string;
  event_seq?: string | number;
  frame_seq?: string;
  worker_boot_id?: string;
  session_epoch?: string;
  event_type: string;
  severity?: string;
  prev_state?: AlertState;
  previous_state?: AlertState;
  new_state: AlertState;
  timestamp: number;
  server_wall_epoch_ms?: number;
  metrics: {
    risk_score?: number;
    ear?: number;
    ear_raw?: number;
    perclos?: number;
    pose_zone?: string;
    reasons?: string[];
    drowsiness_state?: AuthoritativeDrowsinessState | null;
    alert_state?: AuthoritativeAlertState | null;
    is_recovering?: boolean;
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
  network_roundtrip_ms?: number;
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

export interface ZoneBaselineEntry {
  zone?: string;
  ear_median: number;
  ear_mad_scaled?: number;
  dynamic_closure_threshold?: number;
  pitch_median_deg?: number;
  yaw_median_deg?: number;
  sample_count?: number;
  ear_left_median?: number | null;
  ear_right_median?: number | null;
  blendshape_open_median?: number | null;
  confidence_median?: number | null;
  // Legacy aliases retained for reading historical JSONB entries
  pitch_deg_median?: number;
  yaw_deg_median?: number;
}

export interface CalibratedProfilePayload {
  version?: number;
  engine_type?: EngineType;
  profile_version?: number;
  profile_uuid?: string;
  content_hash?: string | null;
  driver_id: string;
  calibrated_at?: string | null;
  is_calibrated?: boolean;
  calibration_mode: string;
  center_ear_median: number;
  center_ear_mad_scaled?: number;
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
  sensitivity_level?: string;
  neutral_pitch_deg: number;
  neutral_yaw_deg: number;
  neutral_rot_mat?: number[][] | null;
  calibration_is_mirrored?: boolean | null;
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
  zone_quality_score?: number;
  is_completed: boolean;
  rejection_reason?: string | null;
  quality_warnings?: string[];
  profile?: CalibratedProfilePayload | null;
}

export interface SessionCreateRequest {
  session_id?: string;
  driver_id?: string;
  experiment_mode?: ExperimentMode;
  profile?: CalibratedProfilePayload;
}

export interface SessionCreateResponse {
  session_id: string;
  driver_id: string;
  owner_id: string;
  experiment_mode: string;
  is_calibrated: boolean;
  calibration_mode: string;
}

export interface LiveKitTokenResponse {
  token: string;
  url: string;
  room_name: string;
  participant_identity: string;
}

export interface StopWorkerResponse {
  status: string;
  room_name: string;
  worker_stopped: boolean;
}

export interface ApiErrorPayload {
  detail?: string;
  error?: string;
  message?: string;
  status?: string;
  correlation_id?: string;
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

export interface SessionEndPayload {
  sessionCode: string;
  clientSessionId?: string;
  finalizeToken?: string;
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
  sessionCode?: string;
  clientSessionId?: string;
  eventId?: string;
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

/**
 * Authoritative ReasonCode catalog matching backend/app/intelligence/reason_codes.py
 * plus transport availability codes and legacy historical codes.
 */
export const REASON_CODE_CATALOG: Record<string, ReasonExplanation> = {
  // --- Authoritative DrowsinessEngine ReasonCode values (16) ---
  SUSTAINED_EYE_CLOSURE: {
    code: "SUSTAINED_EYE_CLOSURE",
    title: "Sustained Eye Closure (Microsleep)",
    description: "Continuous eye closure exceeded the critical safety duration threshold.",
    subsystem: "KINEMATICS",
    severity: "CRITICAL",
  },
  NODDING_MICROSLEEP: {
    code: "NODDING_MICROSLEEP",
    title: "Head Nodding Microsleep",
    description: "Combined head drop and eye closure indicative of acute microsleep onset.",
    subsystem: "HEAD_POSE",
    severity: "CRITICAL",
  },
  SEVERE_PERCLOS_CRISIS: {
    code: "SEVERE_PERCLOS_CRISIS",
    title: "Severe PERCLOS Fatigue Crisis",
    description: "Sustained 60-second time-weighted eye closure percentage exceeded the critical safety ceiling.",
    subsystem: "PERCLOS",
    severity: "CRITICAL",
  },
  PERCLOS_ELEVATED: {
    code: "PERCLOS_ELEVATED",
    title: "Elevated Rolling PERCLOS",
    description: "Percentage of eye closure over the rolling 60-second window exceeded the fatigue warning threshold.",
    subsystem: "PERCLOS",
    severity: "WARNING",
  },
  PROLONGED_BLINK: {
    code: "PROLONGED_BLINK",
    title: "Prolonged Blink Duration",
    description: "Blink duration exceeded the driver's alert baseline, indicating sluggish eyelid dynamics.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  FREQUENT_YAWNING: {
    code: "FREQUENT_YAWNING",
    title: "Frequent Yawning Cluster",
    description: "Repeated authentic yawns detected within the rolling fatigue window.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  PERSISTENT_DISTRACTION: {
    code: "PERSISTENT_DISTRACTION",
    title: "Persistent Off-Road Distraction",
    description: "Head orientation remained away from the forward roadway beyond the safe glance window.",
    subsystem: "HEAD_POSE",
    severity: "WARNING",
  },
  AUTHENTIC_YAWN: {
    code: "AUTHENTIC_YAWN",
    title: "Authentic Yawn Detected",
    description: "Mouth Aspect Ratio (MAR) sustained an authentic yawn envelope.",
    subsystem: "GEOMETRY",
    severity: "INFO",
  },
  EXTENDED_CLOSURE: {
    code: "EXTENDED_CLOSURE",
    title: "Extended Eye Closure",
    description: "Eyes remained closed longer than a normal alert blink; monitoring for escalation.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  SUSPECTED_HEAD_SLUMP: {
    code: "SUSPECTED_HEAD_SLUMP",
    title: "Suspected Head Slump",
    description: "Downward head pitch deviated below neutral posture outside dashboard glance context.",
    subsystem: "HEAD_POSE",
    severity: "WARNING",
  },
  MILD_PERCLOS: {
    code: "MILD_PERCLOS",
    title: "Early PERCLOS Elevation",
    description: "Mild increase in rolling eye closure percentage detected.",
    subsystem: "PERCLOS",
    severity: "INFO",
  },
  MIRROR_CHECK_CONTEXT: {
    code: "MIRROR_CHECK_CONTEXT",
    title: "Mirror Check Context",
    description: "Driver briefly checked a side mirror; false drowsiness alerts are suppressed.",
    subsystem: "CONTEXT",
    severity: "INFO",
  },
  DASHBOARD_CHECK_CONTEXT: {
    code: "DASHBOARD_CHECK_CONTEXT",
    title: "Dashboard Check Context",
    description: "Driver briefly checked the instrument cluster; transient downward gaze is contextualized.",
    subsystem: "CONTEXT",
    severity: "INFO",
  },
  ALERT_NORMAL: {
    code: "ALERT_NORMAL",
    title: "Attentive Baseline",
    description: "Physiological indicators are within the driver's alert baseline.",
    subsystem: "CONTEXT",
    severity: "INFO",
  },
  FACE_LOST: {
    code: "FACE_LOST",
    title: "Face Lost / Not Tracked",
    description: "Driver face could not be tracked reliably in the camera view; monitoring is paused.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  SENSOR_DEGRADED: {
    code: "SENSOR_DEGRADED",
    title: "Sensor / Lighting Quality Degraded",
    description: "Frame lighting, blur, or landmark confidence is too low for reliable drowsiness evaluation.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  CALIBRATION_REQUIRED: {
    code: "CALIBRATION_REQUIRED",
    title: "Personal Calibration Required",
    description: "Complete 4-Zone baseline calibration before drowsiness monitoring can activate.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  CALIBRATION_IN_PROGRESS: {
    code: "CALIBRATION_IN_PROGRESS",
    title: "Calibration In Progress",
    description: "Collecting personalised spatial EAR baselines across cockpit zones.",
    subsystem: "CONTEXT",
    severity: "INFO",
  },

  // --- Transport & Session Availability Codes ---
  NO_FACE_DETECTED: {
    code: "NO_FACE_DETECTED",
    title: "No Face Detected",
    description: "No driver face was detected in the incoming video frame.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  FACE_VISIBILITY_DEGRADED: {
    code: "FACE_VISIBILITY_DEGRADED",
    title: "Face Visibility Degraded",
    description: "Face tracking briefly dropped; holding grace period before marking monitoring unavailable.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  IMAGE_DECODE_FAILED: {
    code: "IMAGE_DECODE_FAILED",
    title: "Frame Decode Failed",
    description: "Backend could not decode the incoming video frame payload.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  SESSION_CLOSED: {
    code: "SESSION_CLOSED",
    title: "Monitoring Session Closed",
    description: "The backend monitoring session has been closed.",
    subsystem: "QUALITY",
    severity: "INFO",
  },

  // --- Legacy Historical Codes (Retained for Past Session Audit Logs) ---
  PROLONGED_EYE_CLOSURE_MICROSLEEP: {
    code: "PROLONGED_EYE_CLOSURE_MICROSLEEP",
    title: "Microsleep / Prolonged Closure",
    description: "Continuous eye closure exceeded the personal microsleep safety threshold.",
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
    title: "Elevated Blink Duration",
    description: "Blink duration rose above the driver's personal alert baseline.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  EYELID_REOPENING_VELOCITY_SLOWED: {
    code: "EYELID_REOPENING_VELOCITY_SLOWED",
    title: "Sluggish Eyelid Reopening (RCR)",
    description: "Reopening-to-Closing velocity ratio dropped below the driver's alert baseline.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  ELEVATED_BLINK_AMPLITUDE_VELOCITY_RATIO: {
    code: "ELEVATED_BLINK_AMPLITUDE_VELOCITY_RATIO",
    title: "Sluggish Eyelid Dynamics (AVR)",
    description: "Eyelid reopening velocity slowed relative to blink amplitude.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  ERRATIC_INTER_BLINK_INTERVAL_VARIABILITY: {
    code: "ERRATIC_INTER_BLINK_INTERVAL_VARIABILITY",
    title: "Erratic Blink Cadence (CV_IBI)",
    description: "Inter-blink intervals show high variability.",
    subsystem: "KINEMATICS",
    severity: "WARNING",
  },
  INVOLUNTARY_BROW_STRAIN_AU4: {
    code: "INVOLUNTARY_BROW_STRAIN_AU4",
    title: "Involuntary Brow Strain (AU4)",
    description: "Facial Action Unit AU4 brow furrowing detected while eyelids are drooping.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  VISUAL_DISTRACTION_OFF_ROAD_DWELL: {
    code: "VISUAL_DISTRACTION_OFF_ROAD_DWELL",
    title: "Off-Road Gaze Distraction",
    description: "Continuous gaze averted from the forward roadway exceeded the safe glance window.",
    subsystem: "CONTEXT",
    severity: "WARNING",
  },
  COGNITIVE_TUNNELING_BLANK_STARE: {
    code: "COGNITIVE_TUNNELING_BLANK_STARE",
    title: "Blank Stare / Cognitive Tunneling",
    description: "Rigid forward gaze with minimal saccadic movement detected.",
    subsystem: "CONTEXT",
    severity: "CRITICAL",
  },
  LOW_GAZE_TRANSITION_ENTROPY: {
    code: "LOW_GAZE_TRANSITION_ENTROPY",
    title: "Low Gaze Scanpath Entropy",
    description: "Visual transition entropy dropped below nominal scanpath variability.",
    subsystem: "CONTEXT",
    severity: "WARNING",
  },
  INCREASED_PERCLOS: {
    code: "INCREASED_PERCLOS",
    title: "Elevated 60s Rolling PERCLOS",
    description: "Time-weighted Percentage of Eye Closure over the rolling 60-second window exceeded healthy baseline.",
    subsystem: "PERCLOS",
    severity: "CRITICAL",
  },
  SUSTAINED_EYE_OPENNESS_REDUCTION: {
    code: "SUSTAINED_EYE_OPENNESS_REDUCTION",
    title: "Sustained Narrowed Aperture",
    description: "Rolling Eye Aspect Ratio dropped significantly below the active zone's personal baseline.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  SUSTAINED_DOWNWARD_HEAD_PITCH: {
    code: "SUSTAINED_DOWNWARD_HEAD_PITCH",
    title: "Sustained Head Nodding",
    description: "Downward head pitch persisted outside mirror-check context.",
    subsystem: "HEAD_POSE",
    severity: "CRITICAL",
  },
  FREQUENT_OR_PROLONGED_YAWNING: {
    code: "FREQUENT_OR_PROLONGED_YAWNING",
    title: "Frequent or Prolonged Yawning",
    description: "Mouth Aspect Ratio indicated active or repeated yawning.",
    subsystem: "GEOMETRY",
    severity: "WARNING",
  },
  FACE_NOT_VISIBLE_OR_OBSCURED: {
    code: "FACE_NOT_VISIBLE_OR_OBSCURED",
    title: "Face Lost or Camera Occluded",
    description: "Driver's facial landmarks were unavailable or geometric reliability dropped.",
    subsystem: "QUALITY",
    severity: "WARNING",
  },
  INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE: {
    code: "INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE",
    title: "Low PERCLOS Observation Coverage",
    description: "Valid face tracking coverage across the 60-second window is below the reliability threshold.",
    subsystem: "QUALITY",
    severity: "INFO",
  },
  OBSERVATION_SAMPLING_GAP_DETECTED: {
    code: "OBSERVATION_SAMPLING_GAP_DETECTED",
    title: "Frame Sampling Gap Detected",
    description: "An inter-frame gap occurred; microsleep timers are protected across dropped frames.",
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
    const description = `Transient gaze into ${zone} uses the zone-conditioned baseline and suppresses false head-pose penalties.`;
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
  const fallbackDesc = "Diagnostic signal emitted by the DriveGuard authoritative drowsiness engine.";
  return {
    code,
    title: formattedTitle,
    description: fallbackDesc,
    explanation: fallbackDesc,
    subsystem: "GEOMETRY",
    severity: "WARNING",
  };
}

export type EffectiveDisplayState =
  | "IDLE"
  | "AWAITING_TELEMETRY"
  | "STALE"
  | "CALIBRATION_REQUIRED"
  | "CALIBRATING"
  | "DEGRADED"
  | "ERROR"
  | "MONITORING_UNAVAILABLE"
  | "NORMAL"
  | "ATTENTION"
  | "WARNING"
  | "CRITICAL";

/**
 * Single authoritative UI state presentation resolver.
 * Never returns "NORMAL" when telemetry is absent, stale, uncalibrated,
 * calibrating, degraded, errored, or when backend monitoring is unavailable.
 */
export function resolveEffectiveDisplayState(
  telemetry: FrameTelemetry | null,
  options: {
    isConnected?: boolean;
    isStale?: boolean;
    backendStatus?: BackendProcessingStatus;
  } = {}
): EffectiveDisplayState {
  const { isConnected = false, isStale = false, backendStatus } = options;

  if (backendStatus === "stale" || (isStale && isConnected)) {
    return "STALE";
  }
  if (!telemetry) {
    if (backendStatus === "connecting" || backendStatus === "awaiting_telemetry" || isConnected) {
      return "AWAITING_TELEMETRY";
    }
    return "IDLE";
  }

  // Operational readiness states always supersede drowsiness severity
  if (telemetry.operational_state === "ERROR") {
    return "ERROR";
  }
  if (telemetry.operational_state === "CALIBRATION_REQUIRED") {
    return "CALIBRATION_REQUIRED";
  }
  if (telemetry.operational_state === "CALIBRATING") {
    return "CALIBRATING";
  }
  if (telemetry.operational_state === "DEGRADED") {
    return "DEGRADED";
  }

  if (
    telemetry.state === "MONITORING_UNAVAILABLE" ||
    telemetry.drowsiness_state === "INSUFFICIENT_DATA" ||
    telemetry.monitoring_status === "MONITORING_UNAVAILABLE"
  ) {
    return "MONITORING_UNAVAILABLE";
  }

  if (
    telemetry.state === "CRITICAL" ||
    telemetry.alert_state === "CRITICAL" ||
    telemetry.drowsiness_state === "CRITICAL_WARNING"
  ) {
    return "CRITICAL";
  }

  if (
    telemetry.state === "WARNING" ||
    telemetry.alert_state === "WARNING" ||
    telemetry.drowsiness_state === "DROWSINESS_WARNING"
  ) {
    return "WARNING";
  }

  if (
    telemetry.alert_state === "ATTENTION" ||
    telemetry.drowsiness_state === "ATTENTION"
  ) {
    return "ATTENTION";
  }

  return "NORMAL";
}
