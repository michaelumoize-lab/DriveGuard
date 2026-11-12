"""
Unit tests for DriveGuard Intelligence & Drowsiness Decision Engine (Phase 6).

Covers:
- Default nominal alert state (`NORMAL`, severity 0, reason `ALERT_NORMAL`)
- Immediate Safety-Critical Urgency:
  - Sustained eye closure >= 1.5s immediately triggers `CRITICAL_WARNING`
  - Nodding slump during eye closure triggers `CRITICAL_WARNING` (`NODDING_MICROSLEEP`)
  - Severe PERCLOS crisis >= 30% triggers `CRITICAL_WARNING`
- Cumulative Fatigue Triggers:
  - Rolling 60s PERCLOS >= 15% triggers `DROWSINESS_WARNING`
  - Prolonged sluggish blink cluster triggers `DROWSINESS_WARNING`
  - Frequent yawns + fatigue indicators trigger `DROWSINESS_WARNING`
  - Persistent off-road gaze dwell (> 3.5s) triggers `DROWSINESS_WARNING`
- Early Warnings & Attention:
  - Extended closure (500ms - 1500ms) triggers `ATTENTION`
  - Authentic completed yawn triggers `ATTENTION`
  - Mild PERCLOS (8% - 15%) triggers `ATTENTION`
- Automotive Context Overrides:
  - Transient mirror glances (<= 3.0s) are flagged as context, suppressing distraction alerts
  - Transient dashboard glances (<= 2.0s) are suppressed without false nodding alerts
- Sensor Viability & Grace Period:
  - Face loss for >= 1.0s transitions to `INSUFFICIENT_DATA` without false drowsiness
  - Signal confidence tiers (`HIGH`, `ACCEPTABLE`, `DEGRADED`, `UNAVAILABLE`)
- Hysteresis & Recovery:
  - Asymmetric hysteresis holds alert state during recovery cooldown
  - Clean reset behavior
"""

from __future__ import annotations

import pytest

from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityResult,
    LightingCondition,
    QualityStatus,
)
from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkEvent,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult, YawnEvent
from backend.app.features.perclos import PerclosResult
from backend.app.intelligence.confidence import (
    ConfidenceAssessment,
    ConfidenceTier,
    SignalConfidenceEvaluator,
)
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.drowsiness_engine import (
    DrowsinessAssessment,
    DrowsinessEngine,
)
from backend.app.intelligence.reason_codes import (
    DrowsinessState,
    ExplainableReason,
    ReasonCode,
)
from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer, TemporalMetrics
from backend.app.vision.face_landmarker import FaceLandmarkStatus


# ==============================================================================
# Synthetic Feature Helpers
# ==============================================================================

def make_driver_features(
    timestamp_ms: float,
    has_face: bool = True,
    is_closed: bool = False,
    ear_val: float = 0.28,
    mar_val: float = 0.18,
    is_yawning: bool = False,
    yawn_duration_ms: float = 0.0,
    completed_yawn: YawnEvent | None = None,
    total_yawns: int = 0,
    zone: PoseZone = PoseZone.CENTER,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    current_closure_duration_ms: float = 0.0,
    completed_blink: BlinkEvent | None = None,
    total_blinks: int = 0,
    perclos_pct: float = 2.0,
    perclos_coverage_pct: float = 100.0,
    perclos_is_reliable: bool = True,
    ear_is_valid: bool = True,
    pose_is_valid: bool = True,
) -> DriverFeatures:
    """Helper creating deterministic immutable DriverFeatures instances."""
    face_status = FaceLandmarkStatus.TRACKING if has_face else FaceLandmarkStatus.INSUFFICIENT_DATA

    ear = EyeAspectRatioResult(
        left_ear=ear_val,
        right_ear=ear_val,
        weighted_ear=ear_val,
        compensated_ear=ear_val,
        closure_threshold=0.20,
        is_left_closed=is_closed,
        is_right_closed=is_closed,
        is_closed=is_closed,
        is_valid=ear_is_valid,
    )

    head_pose = HeadPoseResult(
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=0.0,
        zone=zone,
        is_looking_away=(zone != PoseZone.CENTER),
        is_valid=pose_is_valid,
        rel_pitch_deg=pitch_deg,
        rel_yaw_deg=yaw_deg,
    )

    mar = MouthAspectRatioResult(
        mar=mar_val,
        is_mouth_open=(mar_val >= 0.55),
        is_yawn_in_progress=is_yawning,
        current_yawn_duration_ms=yawn_duration_ms,
        is_authentic_yawn=(yawn_duration_ms >= 1500.0),
        completed_yawn=completed_yawn,
        total_yawn_count=total_yawns,
    )

    blink = BlinkSnapshot(
        state=BlinkState.CLOSED if is_closed else BlinkState.OPEN,
        is_eye_closed=is_closed,
        current_closure_duration_ms=current_closure_duration_ms,
        completed_blink=completed_blink,
        recent_blink_duration_ms=completed_blink.duration_ms if completed_blink else None,
        recent_blink_rcr=completed_blink.rcr if completed_blink else None,
        total_blink_count=total_blinks,
        prolonged_blink_count=0,
    )

    perclos = PerclosResult(
        perclos_pct=perclos_pct,
        coverage_pct=perclos_coverage_pct,
        is_reliable=perclos_is_reliable,
        closed_duration_sec=0.0,
        observed_duration_sec=60.0,
        window_sec=60.0,
    )

    return DriverFeatures(
        timestamp_ms=timestamp_ms,
        has_face=has_face,
        face_status=face_status,
        ear=ear,
        head_pose=head_pose,
        mar=mar,
        blink=blink,
        perclos=perclos,
    )


# ==============================================================================
# Unit Tests
# ==============================================================================

def test_default_alert_normal() -> None:
    """Verifies that an alert driver with nominal indicators produces NORMAL state."""
    engine = DrowsinessEngine()
    features = make_driver_features(timestamp_ms=1000.0)

    assessment = engine.evaluate(features)

    assert assessment.state == DrowsinessState.NORMAL
    assert assessment.severity == 0
    assert assessment.primary_reason is not None
    assert assessment.primary_reason.code == ReasonCode.ALERT_NORMAL
    assert assessment.status_color_bgr == (34, 197, 94)  # Green


def test_sustained_eye_closure_critical_warning() -> None:
    """Continuous eye closure >= 1500ms must immediately trigger CRITICAL_WARNING with zero delay."""
    engine = DrowsinessEngine(sustained_closure_threshold_ms=1500.0)

    # Frame 1: Eyes close at t=1000ms
    f1 = make_driver_features(timestamp_ms=1000.0, is_closed=True, ear_val=0.12)
    a1 = engine.evaluate(f1)
    assert a1.state == DrowsinessState.NORMAL  # Instant closure (0ms elapsed)

    # Frame 2: Eyes still closed at t=1600ms (600ms closed) -> ATTENTION (extended closure)
    f2 = make_driver_features(
        timestamp_ms=1600.0,
        is_closed=True,
        ear_val=0.12,
        current_closure_duration_ms=600.0,
    )
    a2 = engine.evaluate(f2)
    assert a2.state == DrowsinessState.ATTENTION
    assert a2.primary_reason.code == ReasonCode.EXTENDED_CLOSURE

    # Frame 3: Eyes still closed at t=2600ms (1600ms closed) -> CRITICAL_WARNING
    f3 = make_driver_features(
        timestamp_ms=2600.0,
        is_closed=True,
        ear_val=0.12,
        current_closure_duration_ms=1600.0,
    )
    a3 = engine.evaluate(f3)
    assert a3.state == DrowsinessState.CRITICAL_WARNING
    assert a3.severity == 3
    assert a3.primary_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE
    assert a3.status_color_bgr == (239, 68, 68)  # Red


def test_nodding_microsleep_critical() -> None:
    """Head nodding slump (pitch drop) combined with eye closure >= 800ms triggers CRITICAL_WARNING."""
    engine = DrowsinessEngine()

    # t=1000ms: Head forward, eyes closed
    f1 = make_driver_features(timestamp_ms=1000.0, is_closed=True, pitch_deg=0.0)
    engine.evaluate(f1)

    # t=1900ms: Closure reached 900ms and pitch dropped to -18 degrees (head slumped forward)
    f2 = make_driver_features(
        timestamp_ms=1900.0,
        is_closed=True,
        pitch_deg=-18.0,
        current_closure_duration_ms=900.0,
    )
    assessment = engine.evaluate(f2)

    assert assessment.state == DrowsinessState.CRITICAL_WARNING
    assert any(r.code == ReasonCode.NODDING_MICROSLEEP for r in assessment.decision.reasons)


def test_severe_perclos_crisis_critical() -> None:
    """Extreme PERCLOS accumulation (>= 30%) triggers safety-critical emergency warning."""
    engine = DrowsinessEngine()
    features = make_driver_features(timestamp_ms=1000.0, perclos_pct=34.5)

    assessment = engine.evaluate(features)

    assert assessment.state == DrowsinessState.CRITICAL_WARNING
    assert assessment.primary_reason.code == ReasonCode.SEVERE_PERCLOS_CRISIS


def test_elevated_perclos_drowsiness_warning() -> None:
    """Rolling 60s PERCLOS >= 15% triggers DROWSINESS_WARNING."""
    engine = DrowsinessEngine()
    features = make_driver_features(timestamp_ms=1000.0, perclos_pct=17.8)

    assessment = engine.evaluate(features)

    assert assessment.state == DrowsinessState.DROWSINESS_WARNING
    assert assessment.severity == 2
    assert assessment.primary_reason.code == ReasonCode.PERCLOS_ELEVATED
    assert assessment.status_color_bgr == (249, 115, 22)  # Orange


def test_mild_perclos_attention() -> None:
    """Mild PERCLOS between 8% and 15% triggers ATTENTION as early warning."""
    engine = DrowsinessEngine()
    features = make_driver_features(timestamp_ms=1000.0, perclos_pct=10.5)

    assessment = engine.evaluate(features)

    assert assessment.state == DrowsinessState.ATTENTION
    assert assessment.severity == 1
    assert assessment.primary_reason.code == ReasonCode.MILD_PERCLOS


def test_authentic_yawn_attention() -> None:
    """A completed authentic yawn triggers ATTENTION state."""
    engine = DrowsinessEngine()

    yawn_event = YawnEvent(
        start_ts_ms=1000.0,
        end_ts_ms=3200.0,
        duration_ms=2200.0,
        peak_mar=0.72,
    )
    features = make_driver_features(
        timestamp_ms=3200.0,
        completed_yawn=yawn_event,
        total_yawns=1,
    )

    assessment = engine.evaluate(features)

    assert assessment.state == DrowsinessState.ATTENTION
    assert any(r.code == ReasonCode.AUTHENTIC_YAWN for r in assessment.decision.reasons)


def test_frequent_yawns_plus_fatigue_warning() -> None:
    """Multiple yawns within 60s coupled with fatigue kinematics escalate to DROWSINESS_WARNING."""
    engine = DrowsinessEngine()

    # Yawn 1 at t=5000ms
    y1 = YawnEvent(start_ts_ms=3000.0, end_ts_ms=5000.0, duration_ms=2000.0, peak_mar=0.68)
    f1 = make_driver_features(timestamp_ms=5000.0, completed_yawn=y1, total_yawns=1)
    engine.evaluate(f1)

    # Yawn 2 at t=25000ms, with mild PERCLOS of 9.0%
    y2 = YawnEvent(start_ts_ms=23000.0, end_ts_ms=25000.0, duration_ms=2000.0, peak_mar=0.70)
    f2 = make_driver_features(
        timestamp_ms=25000.0,
        completed_yawn=y2,
        total_yawns=2,
        perclos_pct=9.5,
    )
    assessment = engine.evaluate(f2)

    assert assessment.state == DrowsinessState.DROWSINESS_WARNING
    assert any(r.code == ReasonCode.FREQUENT_YAWNING for r in assessment.decision.reasons)


def test_sluggish_prolonged_blink_cluster_warning() -> None:
    """Accumulation of sluggish prolonged blinks in 60s window triggers DROWSINESS_WARNING."""
    engine = DrowsinessEngine()

    # Feed 3 prolonged blinks (>300ms)
    ts = 1000.0
    for i in range(3):
        ts += 2000.0
        blink = BlinkEvent(
            start_ts_ms=ts - 420.0,
            min_ts_ms=ts - 200.0,
            end_ts_ms=ts,
            duration_ms=420.0,
            open_ear_before=0.28,
            min_ear=0.08,
            amplitude=0.20,
            closing_velocity_per_s=2.5,
            reopening_velocity_per_s=1.2,
            rcr=0.48,
            classification=BlinkClassification.PROLONGED,
            is_binocular=True,
        )
        f = make_driver_features(timestamp_ms=ts, completed_blink=blink, total_blinks=i + 1)
        assessment = engine.evaluate(f)

    assert assessment.state == DrowsinessState.DROWSINESS_WARNING
    assert any(r.code == ReasonCode.PROLONGED_BLINK for r in assessment.decision.reasons)


def test_mirror_check_context_suppresses_distraction() -> None:
    """Transient mirror glances (<= 3.0s) maintain mirror context and suppress distraction alarms."""
    engine = DrowsinessEngine()

    # Driver turns to check left mirror for 2.0s
    t_start = 1000.0
    f1 = make_driver_features(
        timestamp_ms=t_start,
        zone=PoseZone.LEFT_MIRROR,
        yaw_deg=25.0,
    )
    engine.evaluate(f1)

    t_check = t_start + 2000.0
    f2 = make_driver_features(
        timestamp_ms=t_check,
        zone=PoseZone.LEFT_MIRROR,
        yaw_deg=25.0,
    )
    assessment = engine.evaluate(f2)

    assert assessment.is_mirror_check_context is True
    assert assessment.state == DrowsinessState.NORMAL
    assert any(r.code == ReasonCode.MIRROR_CHECK_CONTEXT for r in assessment.decision.reasons)


def test_persistent_off_road_distraction_warning() -> None:
    """Continuous gaze off-road > 3.5s (outside mirror safety context) triggers DROWSINESS_WARNING."""
    engine = DrowsinessEngine(off_road_distraction_limit_ms=3500.0)

    # Driver stares out of bounds for 4.0s
    t_start = 1000.0
    engine.evaluate(make_driver_features(timestamp_ms=t_start, zone=PoseZone.OUT_OF_BOUNDS))

    t_4s = t_start + 4000.0
    assessment = engine.evaluate(
        make_driver_features(timestamp_ms=t_4s, zone=PoseZone.OUT_OF_BOUNDS)
    )

    assert assessment.state == DrowsinessState.DROWSINESS_WARNING
    assert any(r.code == ReasonCode.PERSISTENT_DISTRACTION for r in assessment.decision.reasons)


def test_face_loss_grace_and_insufficient_data() -> None:
    """Face loss lasting >= 1.0s transitions to INSUFFICIENT_DATA without false drowsiness."""
    engine = DrowsinessEngine()

    # Face lost at t=1000ms
    f1 = make_driver_features(timestamp_ms=1000.0, has_face=False)
    a1 = engine.evaluate(f1)
    assert a1.state == DrowsinessState.NORMAL  # Grace period active

    # Face still lost at t=2100ms (1100ms elapsed) -> INSUFFICIENT_DATA
    f2 = make_driver_features(timestamp_ms=2100.0, has_face=False)
    a2 = engine.evaluate(f2)
    assert a2.state == DrowsinessState.INSUFFICIENT_DATA
    assert a2.severity == -1
    assert a2.primary_reason.code == ReasonCode.FACE_LOST
    assert a2.status_color_bgr == (148, 163, 184)  # Slate


def test_hysteresis_and_recovery_cooldown() -> None:
    """Escalation to CRITICAL is immediate; recovery requires satisfying cooldown timer."""
    engine = DrowsinessEngine(
        sustained_closure_threshold_ms=1500.0,
        recovery_critical_ms=2000.0,
    )

    # 1. Trigger CRITICAL_WARNING via sustained closure
    f_crit_start = make_driver_features(timestamp_ms=1000.0, is_closed=True)
    engine.evaluate(f_crit_start)

    f_crit = make_driver_features(
        timestamp_ms=2600.0,
        is_closed=True,
        current_closure_duration_ms=1600.0,
    )
    a_crit = engine.evaluate(f_crit)
    assert a_crit.state == DrowsinessState.CRITICAL_WARNING

    # 2. Driver opens eyes at t=2700ms (100ms into recovery) -> Still held in CRITICAL due to hysteresis
    f_open_1 = make_driver_features(timestamp_ms=2700.0, is_closed=False)
    a_open_1 = engine.evaluate(f_open_1)
    assert a_open_1.state == DrowsinessState.CRITICAL_WARNING

    # 3. Eyes open at t=4000ms (1300ms into recovery, still < 2000ms cooldown) -> Still held in CRITICAL
    f_open_2 = make_driver_features(timestamp_ms=4000.0, is_closed=False)
    a_open_2 = engine.evaluate(f_open_2)
    assert a_open_2.state == DrowsinessState.CRITICAL_WARNING

    # 4. Eyes open at t=4800ms (2100ms continuous normal, >= 2000ms cooldown) -> Safely transitions to NORMAL
    f_open_3 = make_driver_features(timestamp_ms=4800.0, is_closed=False)
    a_open_3 = engine.evaluate(f_open_3)
    assert a_open_3.state == DrowsinessState.NORMAL


def test_signal_confidence_tier_evaluation() -> None:
    """Optical quality and landmark degradation lower confidence tier accordingly."""
    evaluator = SignalConfidenceEvaluator()

    # Nominal features
    f_nominal = make_driver_features(timestamp_ms=1000.0)
    c1 = evaluator.evaluate(f_nominal)
    assert c1.tier == ConfidenceTier.HIGH
    assert c1.is_reliable is True

    # Face lost
    f_noface = make_driver_features(timestamp_ms=1000.0, has_face=False)
    c2 = evaluator.evaluate(f_noface)
    assert c2.tier == ConfidenceTier.UNAVAILABLE
    assert c2.is_reliable is False

    # Degenerate eye contours and poor lighting
    f_degraded = make_driver_features(timestamp_ms=1000.0, ear_is_valid=False)
    q_poor = FrameQualityResult(
        mean_luminance=15.0,
        blur_variance=45.0,
        lighting=LightingCondition.UNDEREXPOSED,
        blur=BlurCondition.BLURRY,
        status=QualityStatus.POOR_LIGHTING,
        is_viable_for_inference=False,
        reasons=("Poor lighting",),
    )
    c3 = evaluator.evaluate(f_degraded, quality=q_poor)
    assert c3.tier in (ConfidenceTier.DEGRADED, ConfidenceTier.UNAVAILABLE)
    assert c3.is_reliable is False


def test_engine_reset() -> None:
    """Calling reset() restores default state and empties all history."""
    engine = DrowsinessEngine()

    # Push into warning
    f_warn = make_driver_features(timestamp_ms=1000.0, perclos_pct=20.0)
    engine.evaluate(f_warn)
    assert engine.decision_engine.current_state == DrowsinessState.DROWSINESS_WARNING

    # Reset
    engine.reset()
    assert engine.decision_engine.current_state == DrowsinessState.NORMAL

    # Fresh nominal frame
    f_fresh = make_driver_features(timestamp_ms=2000.0)
    fresh_assessment = engine.evaluate(f_fresh)
    assert fresh_assessment.state == DrowsinessState.NORMAL


def test_sensor_degraded_grace_period_prevents_cooldown_bypass() -> None:
    """
    A momentary sensor degradation (< 1000ms grace period) must hold the existing
    alert state without flapping to INSUFFICIENT_DATA or bypassing the de-escalation cooldown.
    """
    engine = DrowsinessEngine(
        perclos_warning_threshold_pct=15.0,
        sensor_unreliable_grace_ms=1000.0,
        recovery_warning_ms=1500.0,
    )

    # 1. Driver in DROWSINESS_WARNING (PERCLOS = 18%)
    f1 = make_driver_features(timestamp_ms=1000.0, perclos_pct=18.0)
    a1 = engine.evaluate(f1)
    assert a1.state == DrowsinessState.DROWSINESS_WARNING

    # 2. Frame at t=1033ms has motion blur (< 1000ms grace period)
    q_blur = FrameQualityResult(
        mean_luminance=120.0,
        blur_variance=15.0,
        lighting=LightingCondition.OPTIMAL,
        blur=BlurCondition.BLURRY,
        status=QualityStatus.BLURRY,
        is_viable_for_inference=False,
        reasons=("Motion blur",),
    )
    f2 = make_driver_features(timestamp_ms=1033.0, perclos_pct=18.0)
    a2 = engine.evaluate(f2, quality=q_blur)
    # Must hold DROWSINESS_WARNING during grace period rather than flipping to INSUFFICIENT_DATA
    assert a2.state == DrowsinessState.DROWSINESS_WARNING
    assert a2.primary_reason.code == ReasonCode.SENSOR_DEGRADED

    # 3. Frame at t=1066ms is clear again, but with nominal metrics
    f3 = make_driver_features(timestamp_ms=1066.0, perclos_pct=5.0)
    a3 = engine.evaluate(f3)
    # De-escalation cooldown (1500ms) must remain strictly enforced!
    assert a3.state == DrowsinessState.DROWSINESS_WARNING

    # 4. Prolonged sensor fault (lasting >= 1000ms) transitions to INSUFFICIENT_DATA
    f_fault_1 = make_driver_features(timestamp_ms=2000.0, perclos_pct=18.0)
    engine.evaluate(f_fault_1, quality=q_blur)

    f_fault_2 = make_driver_features(timestamp_ms=3100.0, perclos_pct=18.0)
    a_fault_2 = engine.evaluate(f_fault_2, quality=q_blur)
    assert a_fault_2.state == DrowsinessState.INSUFFICIENT_DATA
    assert a_fault_2.primary_reason.code == ReasonCode.SENSOR_DEGRADED


def test_continuous_closure_resilience_to_momentary_ear_jitter() -> None:
    """
    Continuous eye closure tracking should not reset if a single frame has degenerate EAR
    while the blink kinematics tracker confirms the eye remains closed.
    """
    engine = DrowsinessEngine(sustained_closure_threshold_ms=1500.0)

    # t=1000ms: Eyes close
    f1 = make_driver_features(
        timestamp_ms=1000.0,
        is_closed=True,
        ear_val=0.10,
        current_closure_duration_ms=0.0,
    )
    engine.evaluate(f1)

    # t=1600ms: Eyes closed for 600ms, but ear_is_valid jittered to False
    f2 = make_driver_features(
        timestamp_ms=1600.0,
        is_closed=True,
        ear_is_valid=False,
        current_closure_duration_ms=600.0,
    )
    a2 = engine.evaluate(f2)
    assert a2.temporal_metrics.active_closure_duration_ms >= 600.0

    # t=2600ms: Still closed for 1600ms -> Escalates to CRITICAL_WARNING
    f3 = make_driver_features(
        timestamp_ms=2600.0,
        is_closed=True,
        current_closure_duration_ms=1600.0,
    )
    a3 = engine.evaluate(f3)
    assert a3.state == DrowsinessState.CRITICAL_WARNING
    assert a3.primary_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE
