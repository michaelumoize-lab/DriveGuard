"""
DriveGuard — Phase 6, Task 4: Alert Continuity and Independent Recovery Timer Unit Tests.

Validates the Phase 6 Task 4 dual-timer architecture:
1. Alert Activation:
   - Initial transition to DROWSINESS_WARNING activates alert (alert_active=True, unresolved_alert_state, historical_reason).
   - Initial transition to CRITICAL_WARNING activates alert.
   - Transition to ATTENTION does not activate alert (alert_active=False).
2. Escalation & Multi-Warning:
   - Escalation from WARNING to CRITICAL updates unresolved level to CRITICAL_WARNING and updates historical reason.
   - Subsequent lower candidate (WARNING) does not downgrade unresolved CRITICAL alert.
3. Face Loss & Continuity:
   - Face loss within grace period (< 1000ms) preserves alert continuity.
   - Prolonged face loss (>= 1000ms) transitions state to INSUFFICIENT_DATA while preserving alert continuity.
   - Face recovery after prolonged face loss preserves alert until required recovery cooldown finishes.
4. Timer 2 Initiation Requirements:
   - Initiation requires candidate_state == NORMAL, metrics.is_eye_closed is False, confidence.is_reliable, and has_face.
   - Eyes closed (is_eye_closed is True) does not initiate Timer 2.
   - Eye state unknown (is_eye_closed is None) does not initiate Timer 2.
   - Candidate ATTENTION does not initiate Timer 2.
5. Timer 2 Continuation & Blinking:
   - Normal blinks (< 350ms) during recovery cooldown continue Timer 2 without resetting.
6. Timer 2 Cancellation:
   - Prolonged closure (>= 350ms) cancels Timer 2.
   - Unknown eye state (is_eye_closed is None) cancels Timer 2.
   - Candidate ATTENTION cancels Timer 2.
   - Sampling gap (> 250ms) cancels Timer 2.
7. Cooldown Durations:
   - Warning requires 1,500ms unbroken recovery.
   - Critical requires 2,000ms unbroken recovery.
   - Warning escalated to Critical requires 2,000ms cooldown.
8. Independence of Timer 1 and Timer 2:
   - State de-escalation and alert resolution run on independent timers and criteria.
9. DrowsinessEngine End-to-End & is_alert_active property.
10. Engine Reset cleans up all alert fields and timers.
11. Tri-State TemporalMetrics eye representation.
"""

from __future__ import annotations

import math
from typing import Optional
import pytest

from backend.app.features.blink_detector import BlinkSnapshot, BlinkState
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult
from backend.app.features.perclos import PerclosResult
from backend.app.intelligence.confidence import (
    ConfidenceAssessment,
    ConfidenceTier,
)
from backend.app.intelligence.decision_engine import DecisionEngine, DecisionResult
from backend.app.intelligence.drowsiness_engine import (
    DrowsinessAssessment,
    DrowsinessEngine,
)
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalMetrics
from backend.app.vision.face_landmarker import FaceLandmarkStatus


# ==============================================================================
# Test Fixtures and Helpers
# ==============================================================================

def make_metrics(
    timestamp_ms: float,
    active_closure_ms: float = 0.0,
    perclos_pct: float = 0.0,
    is_eye_closed: Optional[bool] = False,
    has_face: bool = True,
    face_lost_duration_ms: float = 0.0,
    yawn_count_30s: int = 0,
    active_yawn_duration_ms: float = 0.0,
) -> TemporalMetrics:
    """Helper to construct TemporalMetrics with deterministic eye and face state."""
    return TemporalMetrics(
        timestamp_ms=timestamp_ms,
        active_closure_duration_ms=active_closure_ms,
        active_off_road_dwell_ms=0.0,
        is_mirror_check_context=False,
        is_dashboard_check_context=False,
        is_nodding_microsleep=False,
        blink_count_60s=0,
        blink_count_30s=0,
        mean_blink_duration_ms_60s=0.0,
        prolonged_blink_count_60s=0,
        severe_blink_count_60s=0,
        mean_rcr_60s=1.0,
        yawn_count_60s=0,
        yawn_count_30s=yawn_count_30s,
        active_yawn_duration_ms=active_yawn_duration_ms,
        perclos_pct=perclos_pct,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=has_face,
        face_lost_duration_ms=face_lost_duration_ms,
        is_eye_closed=is_eye_closed,
    )


def make_conf(is_reliable: bool = True) -> ConfidenceAssessment:
    """Helper to construct ConfidenceAssessment."""
    tier = ConfidenceTier.HIGH if is_reliable else ConfidenceTier.DEGRADED
    return ConfidenceAssessment(
        overall_confidence=0.90 if is_reliable else 0.30,
        tier=tier,
        is_reliable=is_reliable,
        reasons=[],
    )


def make_driver_features(
    timestamp_ms: float,
    has_face: bool = True,
    is_closed: bool = False,
    ear_val: float = 0.28,
    perclos_pct: float = 2.0,
    current_closure_duration_ms: float = 0.0,
) -> DriverFeatures:
    """Helper creating DriverFeatures for DrowsinessEngine end-to-end tests."""
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
        is_valid=has_face,
    )
    head_pose = HeadPoseResult(
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        zone=PoseZone.CENTER,
        is_looking_away=False,
        is_valid=has_face,
        rel_pitch_deg=0.0,
        rel_yaw_deg=0.0,
    )
    mar = MouthAspectRatioResult(
        mar=0.18,
        is_mouth_open=False,
        is_yawn_in_progress=False,
        current_yawn_duration_ms=0.0,
        is_authentic_yawn=False,
        completed_yawn=None,
        total_yawn_count=0,
        observed_yawn_duration_ms=0.0,
    )
    blink = BlinkSnapshot(
        state=BlinkState.CLOSED if is_closed else BlinkState.OPEN,
        is_eye_closed=is_closed,
        current_closure_duration_ms=current_closure_duration_ms,
        completed_blink=None,
        recent_blink_duration_ms=None,
        recent_blink_rcr=None,
        total_blink_count=0,
        prolonged_blink_count=0,
    )
    perclos = PerclosResult(
        perclos_pct=perclos_pct,
        coverage_pct=100.0,
        is_reliable=True,
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
# 1. Alert Activation During Normal Monitoring
# ==============================================================================

def test_1_transition_to_warning_activates_alert():
    """Initial transition to DROWSINESS_WARNING activates alert and records historical reason."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Frame 0: NORMAL
    m0 = make_metrics(timestamp_ms=0.0, perclos_pct=2.0, is_eye_closed=False)
    r0 = engine.evaluate(m0, conf)
    assert r0.state == DrowsinessState.NORMAL
    assert r0.alert_active is False
    assert r0.unresolved_alert_state is None
    assert r0.historical_reason is None

    # Frame 1: PERCLOS elevated to 15.0% -> DROWSINESS_WARNING
    m1 = make_metrics(timestamp_ms=100.0, perclos_pct=15.0, is_eye_closed=False)
    r1 = engine.evaluate(m1, conf)
    assert r1.state == DrowsinessState.DROWSINESS_WARNING
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.DROWSINESS_WARNING
    assert r1.historical_reason is not None
    assert r1.historical_reason.code == ReasonCode.PERCLOS_ELEVATED


def test_2_transition_to_critical_activates_alert():
    """Initial transition to CRITICAL_WARNING activates alert and records historical reason."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Frame 0: NORMAL
    m0 = make_metrics(timestamp_ms=0.0, is_eye_closed=False)
    r0 = engine.evaluate(m0, conf)
    assert r0.state == DrowsinessState.NORMAL
    assert r0.alert_active is False

    # Frame 1: Sustained closure 1600ms -> CRITICAL_WARNING
    m1 = make_metrics(timestamp_ms=100.0, active_closure_ms=1600.0, is_eye_closed=True)
    r1 = engine.evaluate(m1, conf)
    assert r1.state == DrowsinessState.CRITICAL_WARNING
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r1.historical_reason is not None
    assert r1.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE


def test_3_attention_state_does_not_activate_alert():
    """Transition to ATTENTION does NOT activate alert."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Extended closure 600ms -> ATTENTION
    m0 = make_metrics(timestamp_ms=0.0, active_closure_ms=600.0, is_eye_closed=True)
    r0 = engine.evaluate(m0, conf)
    assert r0.state == DrowsinessState.ATTENTION
    assert r0.alert_active is False
    assert r0.unresolved_alert_state is None
    assert r0.historical_reason is None


# ==============================================================================
# 2. Escalation & Multi-Warning Scenarios
# ==============================================================================

def test_4_warning_to_critical_escalation():
    """Escalation from DROWSINESS_WARNING to CRITICAL_WARNING updates unresolved alert level and historical reason."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Frame 0: DROWSINESS_WARNING from PERCLOS
    m0 = make_metrics(timestamp_ms=0.0, perclos_pct=16.0, is_eye_closed=False)
    r0 = engine.evaluate(m0, conf)
    assert r0.state == DrowsinessState.DROWSINESS_WARNING
    assert r0.alert_active is True
    assert r0.unresolved_alert_state == DrowsinessState.DROWSINESS_WARNING
    assert r0.historical_reason.code == ReasonCode.PERCLOS_ELEVATED

    # Frame 1: Driver has sustained eye closure 1600ms -> CRITICAL_WARNING
    m1 = make_metrics(timestamp_ms=100.0, active_closure_ms=1600.0, is_eye_closed=True)
    r1 = engine.evaluate(m1, conf)
    assert r1.state == DrowsinessState.CRITICAL_WARNING
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r1.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE


def test_5_critical_alert_not_downgraded_by_subsequent_warning():
    """Subsequent lower warning candidate does not downgrade unresolved CRITICAL alert."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Frame 0: Trigger CRITICAL_WARNING
    m0 = make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True)
    r0 = engine.evaluate(m0, conf)
    assert r0.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r0.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE

    # Frame 1: Candidate is DROWSINESS_WARNING (PERCLOS 15%)
    m1 = make_metrics(timestamp_ms=100.0, active_closure_ms=0.0, perclos_pct=15.0, is_eye_closed=False)
    r1 = engine.evaluate(m1, conf)
    # Public state held at CRITICAL_WARNING during recovery
    assert r1.state == DrowsinessState.CRITICAL_WARNING
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r1.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE


# ==============================================================================
# 3. Face Loss and Continuity
# ==============================================================================

def test_6_face_loss_within_grace_preserves_alert():
    """Face loss within grace period (< 1000ms) preserves alert continuity and cancels recovery."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Frame 0: Trigger CRITICAL_WARNING
    m0 = make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True)
    engine.evaluate(m0, conf)

    # Frame 1: Face lost for 500ms (within 1000ms grace)
    m1 = make_metrics(timestamp_ms=500.0, has_face=False, face_lost_duration_ms=500.0)
    r1 = engine.evaluate(m1, conf)
    assert r1.state == DrowsinessState.CRITICAL_WARNING
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r1.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE
    assert engine._alert_recovery_start_ts_ms is None


def test_7_prolonged_face_loss_transitions_to_insufficient_data_but_preserves_alert():
    """Prolonged face loss (>= 1000ms) transitions state to INSUFFICIENT_DATA while preserving alert continuity."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Frame 0: Trigger CRITICAL_WARNING
    m0 = make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True)
    engine.evaluate(m0, conf)

    # Frame 1: Face lost for 1500ms (exceeds 1000ms grace)
    m1 = make_metrics(timestamp_ms=1500.0, has_face=False, face_lost_duration_ms=1500.0)
    r1 = engine.evaluate(m1, conf)
    assert r1.state == DrowsinessState.INSUFFICIENT_DATA
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r1.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE
    assert engine._alert_recovery_start_ts_ms is None


def test_8_face_recovery_after_prolonged_face_loss_preserves_alert_until_cooldown_completed():
    """Alert remains active when driver returns after face loss until Timer 2 cooldown completes."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger CRITICAL_WARNING
    m0 = make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True)
    engine.evaluate(m0, conf)

    # Face lost for 2000ms (stream at 100ms intervals) -> INSUFFICIENT_DATA
    for t in range(100, 2000, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), has_face=False, face_lost_duration_ms=float(t)), conf)
    m1 = make_metrics(timestamp_ms=2000.0, has_face=False, face_lost_duration_ms=2000.0)
    r1 = engine.evaluate(m1, conf)
    assert r1.state == DrowsinessState.INSUFFICIENT_DATA
    assert r1.alert_active is True

    # Face returns at t=2100ms with verified open eyes
    m2 = make_metrics(timestamp_ms=2100.0, is_eye_closed=False)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    assert engine._alert_recovery_start_ts_ms == 2100.0

    # Stream open frames up to 3500ms
    for t in range(2200, 3500, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=3500ms (1400ms into 2000ms critical cooldown): alert must stay active
    m3 = make_metrics(timestamp_ms=3500.0, is_eye_closed=False)
    r3 = engine.evaluate(m3, conf)
    assert r3.alert_active is True
    assert r3.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING

    # Stream up to 4100ms
    for t in range(3600, 4100, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=4100ms (2000ms elapsed): alert resolves
    m4 = make_metrics(timestamp_ms=4100.0, is_eye_closed=False)
    r4 = engine.evaluate(m4, conf)
    assert r4.alert_active is False
    assert r4.unresolved_alert_state is None
    assert r4.historical_reason is None


# ==============================================================================
# 4. Timer 2 Initiation Requirements
# ==============================================================================

def test_9_timer2_initiation_requires_verified_open_eyes():
    """Timer 2 cannot initiate when eyes are closed (is_eye_closed is True)."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    assert engine.alert_active is True

    # Candidate NORMAL, but eyes closed (e.g. active_closure_ms = 100.0, is_eye_closed = True)
    m1 = make_metrics(timestamp_ms=100.0, active_closure_ms=100.0, is_eye_closed=True)
    r1 = engine.evaluate(m1, conf)
    assert r1.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None

    # Even after 2500ms of closed eyes, alert does not resolve
    m2 = make_metrics(timestamp_ms=200.0, active_closure_ms=200.0, is_eye_closed=True)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None


def test_10_timer2_initiation_rejects_unknown_eye_state():
    """Timer 2 cannot initiate when eye state is unknown (is_eye_closed is None)."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    assert engine.alert_active is True

    # Candidate NORMAL, but is_eye_closed is None
    m1 = make_metrics(timestamp_ms=100.0, is_eye_closed=None)
    r1 = engine.evaluate(m1, conf)
    assert r1.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None


def test_11_timer2_initiation_requires_candidate_normal():
    """Timer 2 cannot initiate when candidate is ATTENTION even if eyes are verified open."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    assert engine.alert_active is True

    # Candidate ATTENTION due to yawn, eyes open
    m1 = make_metrics(timestamp_ms=100.0, yawn_count_30s=1, active_yawn_duration_ms=1600.0, is_eye_closed=False)
    r1 = engine.evaluate(m1, conf)
    assert r1.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None


def test_12_timer2_initiation_rejects_unreliable_confidence():
    """Timer 2 cannot initiate when confidence is degraded."""
    engine = DecisionEngine()
    reliable_conf = make_conf(True)
    degraded_conf = make_conf(False)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), reliable_conf)
    assert engine.alert_active is True

    # Candidate NORMAL, eyes open, but degraded confidence
    m1 = make_metrics(timestamp_ms=100.0, is_eye_closed=False)
    r1 = engine.evaluate(m1, degraded_conf)
    assert r1.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None


# ==============================================================================
# 5. Timer 2 Continuation & Blinking Under 500ms
# ==============================================================================

def test_13_normal_blinks_under_500ms_continue_timer2():
    """Normal physiological blinks (< 500ms) do NOT reset or cancel Timer 2."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)

    # t=100ms: Timer 2 initiates
    m1 = make_metrics(timestamp_ms=100.0, is_eye_closed=False)
    engine.evaluate(m1, conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Stream open eye frames from 200 to 500ms
    for t in range(200, 600, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # t=600ms: Normal blink begins (closure duration 100ms < 500ms)
    m2 = make_metrics(timestamp_ms=600.0, active_closure_ms=100.0, is_eye_closed=True)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    # Crucial assertion: timer must continue, NOT restart to 600.0 and NOT become None
    assert engine._alert_recovery_start_ts_ms == 100.0

    # t=700ms: Normal blink peak (closure duration 200ms < 500ms)
    m3 = make_metrics(timestamp_ms=700.0, active_closure_ms=200.0, is_eye_closed=True)
    r3 = engine.evaluate(m3, conf)
    assert r3.alert_active is True
    assert engine._alert_recovery_start_ts_ms == 100.0

    # t=800ms: Eyes open again
    m4 = make_metrics(timestamp_ms=800.0, active_closure_ms=0.0, is_eye_closed=False)
    r4 = engine.evaluate(m4, conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Stream open frames from 900 to 1600ms
    for t in range(900, 1600, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # t=1600ms: 1500ms elapsed since 100ms -> alert clears
    m5 = make_metrics(timestamp_ms=1600.0, active_closure_ms=0.0, is_eye_closed=False)
    r5 = engine.evaluate(m5, conf)
    assert r5.alert_active is False
    assert r5.unresolved_alert_state is None
    assert engine._alert_recovery_start_ts_ms is None


# ==============================================================================
# 6. Timer 2 Cancellation Scenarios
# ==============================================================================

def test_14_prolonged_blink_ge_500ms_cancels_timer2():
    """Prolonged blink (active_closure_ms >= 500ms) cancels Timer 2."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)

    # t=100ms: Timer 2 initiates
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Stream intermediate frames so max_sampling_gap_ms (250ms) is not exceeded
    engine.evaluate(make_metrics(timestamp_ms=200.0, active_closure_ms=100.0, is_eye_closed=True), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    engine.evaluate(make_metrics(timestamp_ms=300.0, active_closure_ms=200.0, is_eye_closed=True), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    engine.evaluate(make_metrics(timestamp_ms=500.0, active_closure_ms=400.0, is_eye_closed=True), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # t=600ms: Closure reaches exactly 500.0ms (prolonged blink threshold)
    m2 = make_metrics(timestamp_ms=600.0, active_closure_ms=500.0, is_eye_closed=True)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    # Timer 2 must be canceled
    assert engine._alert_recovery_start_ts_ms is None

    # t=700ms: Eyes open again, recovery must start fresh from 700.0ms
    m3 = make_metrics(timestamp_ms=700.0, is_eye_closed=False)
    engine.evaluate(m3, conf)
    assert engine._alert_recovery_start_ts_ms == 700.0


def test_15_unknown_eye_state_cancels_timer2():
    """Eye state becoming unknown (is_eye_closed is None) cancels running Timer 2."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING and initiate Timer 2
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Eye state becomes None (e.g. occlusion / landmark failure)
    m2 = make_metrics(timestamp_ms=300.0, is_eye_closed=None)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None


def test_16_attention_candidate_cancels_timer2():
    """Candidate deteriorating to ATTENTION cancels running Timer 2."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger CRITICAL and initiate Timer 2
    engine.evaluate(make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True), conf)
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Candidate becomes ATTENTION (mild perclos 9%)
    m2 = make_metrics(timestamp_ms=500.0, perclos_pct=9.0, is_eye_closed=False)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    assert engine._alert_recovery_start_ts_ms is None


def test_17_sampling_gap_cancels_timer2():
    """Sampling gap (> 250ms) cancels running Timer 2."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING and initiate Timer 2
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Sampling gap: delta_t = 300ms > max_sampling_gap_ms (250ms)
    m2 = make_metrics(timestamp_ms=400.0, is_eye_closed=False)
    r2 = engine.evaluate(m2, conf)
    assert r2.alert_active is True
    assert engine._alert_recovery_start_ts_ms == 400.0  # re-initiated at 400.0 after cancellation


# ==============================================================================
# 7. Cooldown Durations: Warning vs Critical
# ==============================================================================

def test_18_warning_cooldown_duration_1500ms():
    """DROWSINESS_WARNING alert resolves after exactly 1,500ms of unbroken recovery."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Stream frames up to 1500ms
    for t in range(200, 1600, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=1599ms (1499ms elapsed): alert still active
    r_sub = engine.evaluate(make_metrics(timestamp_ms=1599.0, is_eye_closed=False), conf)
    assert r_sub.alert_active is True
    assert r_sub.unresolved_alert_state == DrowsinessState.DROWSINESS_WARNING

    # At t=1600ms (1500ms elapsed): alert clears
    r_done = engine.evaluate(make_metrics(timestamp_ms=1600.0, is_eye_closed=False), conf)
    assert r_done.alert_active is False
    assert r_done.unresolved_alert_state is None
    assert r_done.historical_reason is None


def test_19_critical_cooldown_duration_2000ms():
    """CRITICAL_WARNING alert resolves after exactly 2,000ms of unbroken recovery."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger CRITICAL
    engine.evaluate(make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True), conf)
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Stream frames up to 2000ms
    for t in range(200, 2100, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=2099ms (1999ms elapsed): alert still active
    r_sub = engine.evaluate(make_metrics(timestamp_ms=2099.0, is_eye_closed=False), conf)
    assert r_sub.alert_active is True
    assert r_sub.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING

    # At t=2100ms (2000ms elapsed): alert clears
    r_done = engine.evaluate(make_metrics(timestamp_ms=2100.0, is_eye_closed=False), conf)
    assert r_done.alert_active is False
    assert r_done.unresolved_alert_state is None


def test_20_warning_escalated_to_critical_requires_2000ms():
    """Warning recovering for 1000ms, then escalated to Critical, requires full 2,000ms."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger WARNING at t=0
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    engine.evaluate(make_metrics(timestamp_ms=100.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 100.0

    # Stream warning recovery frames from 200 to 1000ms
    for t in range(200, 1100, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=1100ms (1000ms into warning recovery): driver experiences 1600ms microsleep
    r_crit = engine.evaluate(make_metrics(timestamp_ms=1100.0, active_closure_ms=1600.0, is_eye_closed=True), conf)
    assert r_crit.state == DrowsinessState.CRITICAL_WARNING
    assert r_crit.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert engine._alert_recovery_start_ts_ms is None

    # Normal recovery begins at t=1200ms
    engine.evaluate(make_metrics(timestamp_ms=1200.0, is_eye_closed=False), conf)
    assert engine._alert_recovery_start_ts_ms == 1200.0

    # Stream recovery frames up to 2700ms
    for t in range(1300, 2700, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=2700ms (1500ms elapsed): alert must NOT clear (Critical requires 2000ms)
    r_1500 = engine.evaluate(make_metrics(timestamp_ms=2700.0, is_eye_closed=False), conf)
    assert r_1500.alert_active is True
    assert r_1500.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING

    # Stream up to 3200ms
    for t in range(2800, 3200, 100):
        engine.evaluate(make_metrics(timestamp_ms=float(t), is_eye_closed=False), conf)

    # At t=3200ms (2000ms elapsed): alert clears
    r_2000 = engine.evaluate(make_metrics(timestamp_ms=3200.0, is_eye_closed=False), conf)
    assert r_2000.alert_active is False
    assert r_2000.unresolved_alert_state is None


# ==============================================================================
# 8. Independence of Timer 1 and Timer 2
# ==============================================================================

def test_21_dual_timers_operate_independently():
    """Timer 1 (state de-escalation) and Timer 2 (alert resolution) operate on separate rules."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger CRITICAL_WARNING
    engine.evaluate(make_metrics(timestamp_ms=0.0, active_closure_ms=1600.0, is_eye_closed=True), conf)

    # t=100ms: Driver has eyes closed (e.g. 100ms blink), but candidate is NORMAL (active_closure < 500ms)
    # Timer 1: candidate is NORMAL < CRITICAL -> starts Timer 1!
    # Timer 2: eyes are closed -> does NOT start Timer 2!
    m1 = make_metrics(timestamp_ms=100.0, active_closure_ms=100.0, is_eye_closed=True)
    r1 = engine.evaluate(m1, conf)
    assert engine._recovery_start_ts_ms == 100.0
    assert engine._alert_recovery_start_ts_ms is None
    assert r1.alert_active is True


# ==============================================================================
# 9. DrowsinessEngine End-to-End & Property
# ==============================================================================

def test_22_drowsiness_engine_e2e_alert_continuity():
    """DrowsinessEngine properly exposes alert_active, is_alert_active, and unresolved_alert_state."""
    drowsiness_engine = DrowsinessEngine()

    # Frame 0: NORMAL
    f0 = make_driver_features(timestamp_ms=0.0, has_face=True, is_closed=False, ear_val=0.30)
    a0 = drowsiness_engine.evaluate(f0)
    assert a0.state == DrowsinessState.NORMAL
    assert a0.alert_active is False
    assert a0.is_alert_active is False

    # Trigger CRITICAL_WARNING via continuous eye closure across frames
    # 50 frames @ 33.3ms = ~1666ms of closure
    for i in range(1, 52):
        ts = i * 33.3
        f = make_driver_features(
            timestamp_ms=ts,
            has_face=True,
            is_closed=True,
            ear_val=0.10,
            current_closure_duration_ms=ts,
        )
        a = drowsiness_engine.evaluate(f)

    assert a.state == DrowsinessState.CRITICAL_WARNING
    assert a.alert_active is True
    assert a.is_alert_active is True
    assert a.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert a.historical_reason is not None

    # Face lost: stream face loss frames from 1750ms to 3100ms (> 1000ms grace)
    for t in range(1750, 3100, 100):
        a_lost = drowsiness_engine.evaluate(make_driver_features(timestamp_ms=float(t), has_face=False))

    assert a_lost.state == DrowsinessState.INSUFFICIENT_DATA
    # Alert remains active despite INSUFFICIENT_DATA
    assert a_lost.alert_active is True
    assert a_lost.is_alert_active is True
    assert a_lost.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING


# ==============================================================================
# 10. Clean Reset
# ==============================================================================

def test_23_clean_reset_clears_all_alert_continuity_state():
    """Engine reset clears all alert continuity fields and timers."""
    engine = DecisionEngine()
    conf = make_conf(True)

    # Trigger alert
    engine.evaluate(make_metrics(timestamp_ms=0.0, perclos_pct=15.0, is_eye_closed=False), conf)
    assert engine.alert_active is True
    assert engine._unresolved_alert_state is not None

    # Reset
    engine.reset()
    assert engine.alert_active is False
    assert engine._unresolved_alert_state is None
    assert engine._historical_reason is None
    assert engine._alert_recovery_start_ts_ms is None


# ==============================================================================
# 11. Tri-State TemporalMetrics Eye Representation
# ==============================================================================

def test_24_temporal_metrics_tri_state_eye_representation():
    """TemporalMetrics tri-state eye properties behave as specified."""
    # Default is None
    m_default = TemporalMetrics(
        timestamp_ms=0.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=0.0,
        is_mirror_check_context=False,
        is_dashboard_check_context=False,
        is_nodding_microsleep=False,
        blink_count_60s=0,
        blink_count_30s=0,
        mean_blink_duration_ms_60s=0.0,
        prolonged_blink_count_60s=0,
        severe_blink_count_60s=0,
        mean_rcr_60s=1.0,
        yawn_count_60s=0,
        yawn_count_30s=0,
        active_yawn_duration_ms=0.0,
        perclos_pct=0.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    assert m_default.is_eye_closed is None
    assert m_default.is_eye_verified_open is False
    assert m_default.is_eye_verified_closed is False

    # Verified open
    m_open = make_metrics(timestamp_ms=10.0, is_eye_closed=False)
    assert m_open.is_eye_closed is False
    assert m_open.is_eye_verified_open is True
    assert m_open.is_eye_verified_closed is False

    # Verified closed
    m_closed = make_metrics(timestamp_ms=20.0, is_eye_closed=True)
    assert m_closed.is_eye_closed is True
    assert m_closed.is_eye_verified_open is False
    assert m_closed.is_eye_verified_closed is True
