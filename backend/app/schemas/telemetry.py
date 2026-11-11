"""
Pydantic Data Schemas for Real-Time Streaming Telemetry and Alert Events.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, List, Optional, Union
from pydantic import BaseModel, ConfigDict, Field, field_validator


class AlertStateEnum(str, Enum):
    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MONITORING_UNAVAILABLE = "MONITORING_UNAVAILABLE"


class PoseZoneEnum(str, Enum):
    CENTER = "CENTER"
    LEFT_MIRROR = "LEFT_MIRROR"
    RIGHT_MIRROR = "RIGHT_MIRROR"
    DASHBOARD = "DASHBOARD"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


class HeadPoseDTO(BaseModel):
    """Euler head pose angles in degrees estimated via solvePnP / 4x4 matrix."""

    model_config = ConfigDict(populate_by_name=True)

    pitch_deg: float = Field(description="Nodding angle (positive=downward, negative=upward)")
    yaw_deg: float = Field(description="Turning angle (negative=driver left mirror, positive=right)")
    roll_deg: float = Field(description="Head tilt angle")


class EyeMetricsDTO(BaseModel):
    """Fine-grained physiological eye openness and blink kinematics."""

    model_config = ConfigDict(populate_by_name=True)

    ear_raw: float = Field(description="3D yaw-weighted raw Eye Aspect Ratio", ge=0.0)
    ear_left: float = Field(description="Left eye EAR", ge=0.0)
    ear_right: float = Field(description="Right eye EAR", ge=0.0)
    ear_normalized: float = Field(description="Pose-conditioned normalized EAR", ge=0.0)
    ear_closure_threshold: float = Field(description="Active zone dynamic closure threshold", ge=0.0)
    is_eye_closed: bool = Field(description="Whether the eye is classified as currently closed")
    fused_closure_prob: float = Field(description="Fused closure probability from cascade in [0, 1]", ge=0.0, le=1.0)
    perclos_pct: float = Field(description="Rolling 60-second time-weighted PERCLOS percentage", ge=0.0, le=100.0)
    blink_duration_ms: Optional[float] = Field(default=None, description="Duration of most recently completed blink in ms")
    blink_rcr: Optional[float] = Field(default=None, description="Reopening velocity ratio of most recent blink")
    blink_rate_per_min: float = Field(default=0.0, description="Rolling blink frequency per minute", ge=0.0)
    mar_yawn: float = Field(default=0.0, description="Mouth Aspect Ratio for yawn tracking", ge=0.0)
    blink_state: str = Field(default="OPEN", description="Temporal blink state: OPEN, CLOSING, CLOSED, REOPENING, PROLONGED_CLOSURE, UNCERTAIN_GAP")
    blink_classification: str = Field(default="NORMAL", description="Deterministic blink classification: NORMAL, PROLONGED, REPEATED_ABNORMAL, MICROSLEEP_CANDIDATE, UNCERTAIN")
    perclos_coverage_pct: float = Field(default=100.0, description="Observation coverage percentage across PERCLOS window", ge=0.0, le=100.0)
    perclos_reliable: bool = Field(default=True, description="Whether PERCLOS coverage meets the reliable observation threshold")
    eye_closure_duration_ms: float = Field(default=0.0, description="Continuous eye closure duration in ms", ge=0.0)
    sampling_gap_detected: bool = Field(default=False, description="Whether an observation gap exceeded maximum continuous threshold")
    blink_avr: Optional[float] = Field(default=None, description="Amplitude-to-Velocity Ratio of recent blinks")
    blink_cv_ibi: Optional[float] = Field(default=None, description="Coefficient of variation of inter-blink intervals")
    gaze_entropy: Optional[float] = Field(default=None, description="Transition gaze entropy H_t across cabin zones")
    brow_furrow_au4: Optional[float] = Field(default=None, description="Facial Action Unit AU4 Brow Lowerer strain score")
    yawn_envelope_confidence: Optional[float] = Field(default=None, description="Authentic physiological yawn envelope confidence score")


class CascadeStatusDTO(BaseModel):
    """Status and arbitration decision of the Confidence-Gated Eye Cascade."""

    model_config = ConfigDict(populate_by_name=True)

    experiment_mode: str = Field(description="Active experiment mode: EXPERIMENT_A, EXPERIMENT_B, or EXPERIMENT_C")
    cnn_triggered: bool = Field(description="Whether MobileNetV3-Small was invoked on this frame")
    geom_open_prob: float = Field(description="Geometric MediaPipe open eye probability in [0, 1]", ge=0.0, le=1.0)
    geom_reliability: float = Field(description="Geometric measurement reliability in [0, 1]", ge=0.0, le=1.0)
    cnn_open_prob: Optional[float] = Field(default=None, description="MobileNetV3-Small open eye probability if triggered")
    cnn_reliability: Optional[float] = Field(default=None, description="Eye crop quality & classifier certainty reliability")
    arbitration_winner: str = Field(description="Authority source: 'GEOMETRY', 'CNN', or 'CONSENSUS'")
    disagreement_flag: bool = Field(default=False, description="True if geometry and CNN disagreed on eye closure")
    cnn_latency_ms: float = Field(default=0.0, description="MobileNetV3-Small inference latency in ms")
    cnn_execution_mode: str = Field(
        default="UNAVAILABLE",
        description="Execution mode: 'MOBILENET_ONNX', 'ANALYTICAL_FALLBACK', or 'UNAVAILABLE'",
    )
    verifier_invoked: bool = Field(
        default=False,
        description="Whether the second-stage eye verifier was invoked",
    )
    gate_trigger_reason: str = Field(
        default="NONE",
        description="Rationale for cascade gate decision (e.g. HIGH_GEOM_CONFIDENCE, LOW_GEOMETRIC_RELIABILITY, BORDERLINE_CLOSURE_THRESHOLD)",
    )
    onnx_inference_executed: bool = Field(
        default=False,
        description="True strictly if ONNX runtime model inference was physically executed",
    )
    cnn_inference_duration_ms: Optional[float] = Field(
        default=None,
        description="Isolated second-stage verifier execution time in milliseconds",
    )


class ConventionalFixedComparisonDTO(BaseModel):
    """Side-by-side benchmark metrics comparing DriveGuard with standard static threshold."""

    model_config = ConfigDict(populate_by_name=True)

    fixed_cutoff_ear: float = Field(default=0.22, description="Conventional uncalibrated static EAR cutoff")
    fixed_alert_state: str = Field(description="Alert state produced by conventional static EAR < 0.22 rule")
    fixed_false_alarm: bool = Field(description="True if conventional rule triggered during normal mirror check")
    personalised_zone: str = Field(description="Active zone driving DriveGuard's adaptive threshold")
    personalised_threshold: float = Field(description="Active zone personalised threshold")
    is_mirror_check_suppressed: bool = Field(description="True if mirror check was protected from false alarms")


class TimingBreakdownDTO(BaseModel):
    """Instrumentation of server-side frame processing stages in milliseconds."""

    model_config = ConfigDict(populate_by_name=True)

    queue_latency_ms: float = Field(default=0.0, description="Time frame waited in ingest queue before inference")
    infer_duration_ms: float = Field(default=0.0, description="MediaPipe + Cascade evaluation time")
    server_total_ms: float = Field(default=0.0, description="Total server transit and processing time")
    server_ingest_mono_ms: float = Field(default=0.0, description="Server monotonic timestamp when frame arrived in Python worker (perf_counter in ms)")
    server_publish_mono_ms: float = Field(default=0.0, description="Server monotonic timestamp when telemetry was published (perf_counter in ms)")
    server_wall_epoch_ms: float = Field(default=0.0, description="Server epoch wall-clock timestamp in milliseconds")
    frame_received_mono_ms: Optional[float] = Field(
        default=None, description="Server monotonic timestamp when frame became available in Python"
    )
    frame_queue_wait_ms: Optional[float] = Field(
        default=None, description="Duration frame spent waiting in the application queue"
    )
    frame_convert_ms: Optional[float] = Field(
        default=None, description="Duration of VideoFrame to RGB buffer conversion"
    )
    frame_preprocess_ms: Optional[float] = Field(
        default=None, description="Duration of NumPy buffer mapping, normalization, and resizing"
    )
    executor_wait_ms: Optional[float] = Field(
        default=None, description="Duration from thread submission to thread execution start"
    )
    process_frame_ms: Optional[float] = Field(
        default=None, description="Duration of SessionAIEngine.process_frame()"
    )
    publish_ms: Optional[float] = Field(
        default=None, description="Duration of telemetry publication"
    )
    frame_age_ms: Optional[float] = Field(
        default=None, description="End-to-end frame capture age, populated only when clocks are synchronized"
    )


class FrameTelemetryDTO(BaseModel):
    """Comprehensive per-frame telemetry payload emitted over WebSocket or WebRTC DataChannel."""

    model_config = ConfigDict(populate_by_name=True)

    session_id: str = Field(description="Unique monitoring session identifier")
    driver_id: str = Field(description="Driver identifier")
    frame_index: int = Field(description="Monotonically increasing processed frame sequence number", ge=0)
    frame_seq: str = Field(default="0", description="Monotonically increasing processed video frame sequence as decimal string")
    event_seq: str = Field(default="0", description="Monotonically increasing published telemetry event sequence as decimal string")
    worker_boot_id: str = Field(default="", description="Hex UUID of worker boot instance")
    session_epoch: str = Field(default="", description="Session initialization epoch timestamp or string")
    timestamp_ms: float = Field(description="Frame capture or presentation timestamp in milliseconds")
    fps: float = Field(description="Instantaneous pipeline processing frame rate", ge=0.0)
    face_detected: bool = Field(description="True if a human face was tracked in this frame")
    face_visible: bool = Field(default=False, description="Explicit driver face visibility indicator")
    grace_period_active: bool = Field(default=False, description="True if face-loss grace period is buffering dropout")
    last_seen_ms: Optional[float] = Field(default=None, description="Timestamp in ms when driver face was last detected")
    tracking_mode: Optional[str] = Field(default=None, description="Active landmarking implementation")
    
    # State & Risk
    state: AlertStateEnum = Field(description="DriveGuard FSM state: NORMAL, WARNING, CRITICAL, MONITORING_UNAVAILABLE")
    monitoring_status: str = Field(default="MONITORING_ACTIVE", description="Monitoring availability: MONITORING_ACTIVE, FACE_VISIBILITY_DEGRADED, MONITORING_UNAVAILABLE, MONITORING_RECOVERING")
    risk_score: float = Field(description="Composite personalised fatigue risk score in [0.0, 1.0]", ge=0.0, le=1.0)
    pose_zone: PoseZoneEnum = Field(description="Classified spatial zone: CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD")
    reason_codes: List[str] = Field(default_factory=list, description="Machine-explainable alert rationale codes")
    
    # Subsystem telemetry
    head_pose: HeadPoseDTO
    eye_metrics: EyeMetricsDTO
    cascade: CascadeStatusDTO
    comparison: ConventionalFixedComparisonDTO
    
    # Calibration & Engine State
    engine_mode: str = Field(default="MONITORING", description="'MONITORING' or 'CALIBRATING'")
    inference_latency_ms: float = Field(description="Total end-to-end processing latency for this frame in ms")
    timing: Optional[TimingBreakdownDTO] = Field(default=None, description="Detailed server-side stage latencies")

    @field_validator("frame_seq", "event_seq", mode="before")
    @classmethod
    def _coerce_frame_seq_to_str(cls, v: Any) -> str:
        return str(v) if v is not None else "0"


class SafetyEventDTO(BaseModel):
    """Important safety state transition or critical alert requiring reliable transport and client ACK."""

    model_config = ConfigDict(populate_by_name=True)

    v: int = Field(default=2, description="Telemetry protocol version")
    type: str = Field(default="SAFETY_EVENT", description="Message discriminator")
    event_id: str = Field(description="Unique event UUID for deduplication")
    session_id: str = Field(description="Monitoring session identifier")
    event_seq: str = Field(description="Monotonically increasing sequence number for safety events as decimal string")
    frame_seq: Optional[str] = Field(default=None, description="Frame sequence number triggering the event as decimal string")
    worker_boot_id: str = Field(default="", description="Hex UUID of worker boot instance")
    session_epoch: str = Field(default="", description="Session initialization epoch timestamp or string")
    event_type: str = Field(description="Event classification, e.g. 'STATE_TRANSITION'")
    previous_state: AlertStateEnum = Field(description="Previous FSM alert state")
    new_state: AlertStateEnum = Field(description="New FSM alert state")
    timestamp: float = Field(description="Server epoch timestamp in seconds")
    server_wall_epoch_ms: Optional[float] = Field(default=None, description="Server epoch wall-clock timestamp in ms")
    metrics: dict = Field(default_factory=dict, description="Key physiological metrics triggering the event")

    @field_validator("event_seq", mode="before")
    @classmethod
    def _coerce_event_seq_to_str(cls, v: Any) -> str:
        return str(v) if v is not None else "0"

    @field_validator("frame_seq", mode="before")
    @classmethod
    def _coerce_safety_frame_seq_to_str(cls, v: Any) -> Optional[str]:
        return str(v) if v is not None else None


class EventAckDTO(BaseModel):
    """Client acknowledgement confirming receipt of a reliable safety event."""

    model_config = ConfigDict(populate_by_name=True)

    action: str = Field(default="EVENT_ACK")
    event_id: str = Field(description="ID of the event being acknowledged")
    event_seq: Union[int, str] = Field(description="Sequence number of the acknowledged event as decimal string or int")
    client_timestamp: Optional[float] = Field(default=None, description="Client performance.now() timestamp")

    @field_validator("event_seq", mode="before")
    @classmethod
    def _coerce_ack_seq_to_str(cls, v: Any) -> str:
        return str(v) if v is not None else "0"


class TelemetryPacketDTO(BaseModel):
    """High-frequency telemetry envelope for unreliable DataChannel delivery."""

    model_config = ConfigDict(populate_by_name=True)

    v: int = Field(default=2, description="Protocol version")
    type: str = Field(default="TELEMETRY")
    session_id: str
    seq: Union[int, str] = Field(description="Monotonically increasing frame sequence number")
    frame_seq: str = Field(default="0", description="Monotonically increasing frame sequence number as decimal string")
    event_seq: str = Field(default="0", description="Monotonically increasing event sequence number as decimal string")
    worker_boot_id: str = Field(default="", description="Worker boot UUID")
    session_epoch: str = Field(default="", description="Session epoch timestamp")
    server_wall_epoch_ms: float = Field(default=0.0, description="Server epoch wall-clock timestamp in ms")
    timestamp: float = Field(description="Server epoch timestamp in seconds")
    telemetry: FrameTelemetryDTO
    calibration: Optional[dict] = None
    echo_client_ts: Optional[float] = None

    @field_validator("frame_seq", "event_seq", mode="before")
    @classmethod
    def _coerce_packet_seq_to_str(cls, v: Any) -> str:
        return str(v) if v is not None else "0"
