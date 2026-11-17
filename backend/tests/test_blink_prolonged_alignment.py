"""
DriveGuard — Phase 6, Task 2: Prolonged Blink Threshold Alignment Unit Tests.

Comprehensive validation of the aligned 350 ms prolonged-blink boundary:
1. 349 ms: completed blink is NORMAL; temporal prolonged count is zero.
2. Exactly 350 ms: completed blink is NORMAL; temporal prolonged count is zero.
3. 350.1 ms: completed blink is PROLONGED; temporal prolonged count is one.
4. Exactly 800 ms: completed blink is MICROSLEEP_CANDIDATE; temporal prolonged count includes it.
5. Kinematic reliability separation: a valid 400 ms completed blink with is_kinematics_reliable=False
   remains classified and counted as prolonged, contributes to mean blink duration, but is excluded
   from mean RCR.
6. Average-duration rule: three completed 350 ms blinks remain individually NORMAL, with prolonged
   count zero, but the existing DecisionEngine average-duration rule still triggers at its documented boundary.
7. Rolling window: a prolonged blink expires from prolonged_blink_count_60s after it leaves the 60-second window.
8. Invalid configuration: a prolonged threshold greater than or equal to the microsleep threshold
   raises ValueError; also non-finite threshold values raise ValueError.
9. Left-censored closure: a closure interrupted by a sampling gap must not produce a fabricated
   completed prolonged blink when the eye reopens.
10. Ongoing closure boundary: verify that the ongoing closure enters PROLONGED_CLOSURE at the configured
    threshold, without incorrectly creating a completed blink.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple
import pytest

from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkEvent,
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult
from backend.app.features.perclos import PerclosResult
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer
from backend.app.vision.face_landmarker import FaceLandmarkStatus


# ==============================================================================
# Helpers
# ==============================================================================

def make_test_features(
    timestamp_ms: float,
    blink_snapshot: Optional[BlinkSnapshot] = None,
    completed_blink: Optional[BlinkEvent] = None,
    total_blinks: int = 1,
    is_closed: bool = False,
    ear_val: float = 0.28,
    zone: PoseZone = PoseZone.CENTER,
) -> DriverFeatures:
    """Creates a valid DriverFeatures instance wrapping real detector snapshots or events."""
    ear = EyeAspectRatioResult(
        left_ear=ear_val,
        right_ear=ear_val,
        weighted_ear=ear_val,
        compensated_ear=ear_val,
        closure_threshold=0.20,
        is_left_closed=is_closed,
        is_right_closed=is_closed,
        is_closed=is_closed,
        is_valid=True,
    )
    pose = HeadPoseResult(
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        zone=zone,
        is_looking_away=(zone != PoseZone.CENTER),
        is_valid=True,
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
    perclos = PerclosResult(
        perclos_pct=2.0,
        coverage_pct=100.0,
        is_reliable=True,
        closed_duration_sec=0.0,
        observed_duration_sec=60.0,
        window_sec=60.0,
    )

    if blink_snapshot is not None:
        blink = blink_snapshot
    else:
        blink = BlinkSnapshot(
            state=BlinkState.CLOSED if is_closed else BlinkState.OPEN,
            is_eye_closed=is_closed,
            current_closure_duration_ms=0.0,
            completed_blink=completed_blink,
            recent_blink_duration_ms=completed_blink.duration_ms if completed_blink else None,
            recent_blink_rcr=completed_blink.rcr if completed_blink else None,
            total_blink_count=total_blinks,
            prolonged_blink_count=1 if (completed_blink and completed_blink.classification in (
                BlinkClassification.PROLONGED, BlinkClassification.MICROSLEEP_CANDIDATE
            )) else 0,
        )

    return DriverFeatures(
        timestamp_ms=timestamp_ms,
        has_face=True,
        face_status=FaceLandmarkStatus.TRACKING,
        ear=ear,
        head_pose=pose,
        mar=mar,
        blink=blink,
        perclos=perclos,
    )


def generate_detector_blink(
    target_duration_ms: float,
    start_time_offset: float = 1000.0,
    detector: Optional[BlinkKinematicsDetector] = None,
) -> Tuple[BlinkEvent, BlinkSnapshot, BlinkKinematicsDetector]:
    """
    Drives BlinkKinematicsDetector through frame updates to generate an authentic
    completed blink event with the exact target interpolated duration_ms.
    """
    det = detector or BlinkKinematicsDetector()
    # Baseline open
    det.update(timestamp_ms=start_time_offset, left_ear=0.30, right_ear=0.30, ear=0.30)
    # Fast closure crossing: ratio = (0.240 - 0.30) / (0.15 - 0.30) = 0.4
    # t_start = start_time_offset + 50.0 * 0.4 = start_time_offset + 20.0
    det.update(timestamp_ms=start_time_offset + 50.0, left_ear=0.15, right_ear=0.15, ear=0.15)
    
    # Reopening crossing: ratio = (0.240 - 0.15) / (0.30 - 0.15) = 0.6
    # t_end = prev_t + 50.0 * 0.6 = prev_t + 30.0
    # duration = t_end - t_start = (prev_t + 30.0) - (start_time_offset + 20.0)
    # prev_t = start_time_offset + 20.0 + target_duration_ms - 30.0
    t_start = start_time_offset + 20.0
    target_prev_t = t_start + target_duration_ms - 30.0
    target_curr_t = target_prev_t + 50.0

    # Stream intermediate closed frames so max_sampling_gap_ms (250 ms) is not triggered
    t = start_time_offset + 90.0
    while t < target_prev_t - 20.0:
        det.update(timestamp_ms=t, left_ear=0.15, right_ear=0.15, ear=0.15)
        t += 40.0

    det.update(timestamp_ms=target_prev_t, left_ear=0.15, right_ear=0.15, ear=0.15)
    snap = det.update(timestamp_ms=target_curr_t, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert snap.completed_blink is not None, "Failed to produce completed blink"
    return snap.completed_blink, snap, det


# ==============================================================================
# Unit Tests (1-10)
# ==============================================================================

def test_1_blink_349ms_completed_is_normal_and_count_zero() -> None:
    """1. 349 ms: completed blink is NORMAL; temporal prolonged count is zero."""
    blink, snap, _ = generate_detector_blink(349.0, start_time_offset=1000.0)
    assert round(blink.duration_ms, 2) == 349.0
    assert blink.classification == BlinkClassification.NORMAL
    assert snap.prolonged_blink_count == 0

    analyzer = TemporalAnalyzer()
    features = make_test_features(timestamp_ms=blink.end_ts_ms, blink_snapshot=snap)
    metrics = analyzer.update(features)

    assert metrics.prolonged_blink_count_60s == 0
    assert metrics.blink_count_60s == 1
    assert round(metrics.mean_blink_duration_ms_60s, 2) == 349.0


def test_2_blink_exactly_350ms_completed_is_normal_and_count_zero() -> None:
    """2. Exactly 350 ms: completed blink is NORMAL; temporal prolonged count is zero."""
    blink, snap, _ = generate_detector_blink(350.0, start_time_offset=1000.0)
    assert round(blink.duration_ms, 2) == 350.0
    assert blink.classification == BlinkClassification.NORMAL
    assert snap.prolonged_blink_count == 0

    analyzer = TemporalAnalyzer()
    features = make_test_features(timestamp_ms=blink.end_ts_ms, blink_snapshot=snap)
    metrics = analyzer.update(features)

    assert metrics.prolonged_blink_count_60s == 0
    assert metrics.blink_count_60s == 1
    assert round(metrics.mean_blink_duration_ms_60s, 2) == 350.0


def test_3_blink_350_1ms_completed_is_prolonged_and_count_one() -> None:
    """3. 350.1 ms: completed blink is PROLONGED; temporal prolonged count is one."""
    blink, snap, _ = generate_detector_blink(350.1, start_time_offset=1000.0)
    assert abs(blink.duration_ms - 350.1) < 1e-3
    assert blink.classification == BlinkClassification.PROLONGED
    assert snap.prolonged_blink_count == 1

    analyzer = TemporalAnalyzer()
    features = make_test_features(timestamp_ms=blink.end_ts_ms, blink_snapshot=snap)
    metrics = analyzer.update(features)

    assert metrics.prolonged_blink_count_60s == 1
    assert metrics.blink_count_60s == 1
    assert abs(metrics.mean_blink_duration_ms_60s - 350.1) < 1e-3


def test_4_blink_exactly_800ms_completed_is_microsleep_candidate_and_counted_in_prolonged() -> None:
    """4. Exactly 800 ms: completed blink is MICROSLEEP_CANDIDATE; temporal prolonged count includes it."""
    blink, snap, _ = generate_detector_blink(800.0, start_time_offset=1000.0)
    assert round(blink.duration_ms, 2) == 800.0
    assert blink.classification == BlinkClassification.MICROSLEEP_CANDIDATE
    assert snap.prolonged_blink_count == 1

    analyzer = TemporalAnalyzer()
    features = make_test_features(timestamp_ms=blink.end_ts_ms, blink_snapshot=snap)
    metrics = analyzer.update(features)

    # Microsleep candidates are prolonged by definition (>=800ms > 350ms) and MUST be counted
    assert metrics.prolonged_blink_count_60s == 1
    assert metrics.blink_count_60s == 1
    assert round(metrics.mean_blink_duration_ms_60s, 2) == 800.0


def test_5_kinematic_reliability_separation() -> None:
    """
    5. Kinematic reliability separation: a valid 400 ms completed blink with
    is_kinematics_reliable=False remains classified and counted as prolonged,
    contributes to mean blink duration, but is excluded from mean RCR.
    """
    # Drive detector to produce 400ms duration with reopen_dt <= 5ms
    det = BlinkKinematicsDetector()
    det.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    det.update(timestamp_ms=1050.0, left_ear=0.15, right_ear=0.15, ear=0.15)
    for t in range(1100, 1400, 50):
        det.update(timestamp_ms=float(t), left_ear=0.15, right_ear=0.15, ear=0.15)
    det.update(timestamp_ms=1416.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    # Fast reopen step in 5.71ms: ratio=0.7 => t_end=1420.0, duration=400.0ms, reopen_dt=4.0ms <= 5ms
    curr_t = 1416.0 + 4.0 / 0.7
    snap_unreliable = det.update(timestamp_ms=curr_t, left_ear=0.30, right_ear=0.30, ear=0.30)
    b_unreliable = snap_unreliable.completed_blink

    assert b_unreliable is not None
    assert round(b_unreliable.duration_ms, 2) == 400.0
    assert b_unreliable.is_kinematics_reliable is False
    assert b_unreliable.rcr == 0.0
    assert b_unreliable.classification == BlinkClassification.PROLONGED

    analyzer = TemporalAnalyzer()
    f1 = make_test_features(timestamp_ms=curr_t, blink_snapshot=snap_unreliable, total_blinks=1)
    m1 = analyzer.update(f1)

    assert m1.prolonged_blink_count_60s == 1
    assert round(m1.mean_blink_duration_ms_60s, 2) == 400.0
    # Because kinematics was unreliable, RCR of 0.0 was excluded from rolling mean RCR
    assert m1.mean_rcr_60s == 1.0

    # Feed a second blink that has reliable kinematics (e.g. 200 ms, rcr=0.85)
    b_reliable = BlinkEvent(
        start_ts_ms=curr_t + 500.0,
        min_ts_ms=curr_t + 600.0,
        end_ts_ms=curr_t + 700.0,
        duration_ms=200.0,
        open_ear_before=0.30,
        min_ear=0.10,
        amplitude=0.20,
        closing_velocity_per_s=2.0,
        reopening_velocity_per_s=1.7,
        rcr=0.85,
        classification=BlinkClassification.NORMAL,
        is_binocular=True,
        is_kinematics_reliable=True,
    )
    f2 = make_test_features(timestamp_ms=curr_t + 700.0, completed_blink=b_reliable, total_blinks=2)
    m2 = analyzer.update(f2)

    assert m2.prolonged_blink_count_60s == 1
    assert round(m2.mean_blink_duration_ms_60s, 2) == 300.0  # (400 + 200)/2
    # Only the reliable blink's RCR (0.85) is included in mean_rcr_60s
    assert round(m2.mean_rcr_60s, 2) == 0.85


def test_6_average_duration_rule_triggers_warning_at_documented_boundary() -> None:
    """
    6. Average-duration rule: three completed 350 ms blinks remain individually NORMAL,
    with prolonged count zero, but the existing DecisionEngine average-duration rule still
    triggers at its documented boundary (mean_blink_duration_ms_60s >= 350.0 with >= 3 blinks).
    """
    engine = DrowsinessEngine()

    ts = 1000.0
    assessment = None
    for i in range(3):
        ts += 2000.0
        # Exactly 350.0 ms duration, classified as NORMAL
        b = BlinkEvent(
            start_ts_ms=ts - 350.0,
            min_ts_ms=ts - 175.0,
            end_ts_ms=ts,
            duration_ms=350.0,
            open_ear_before=0.30,
            min_ear=0.10,
            amplitude=0.20,
            closing_velocity_per_s=2.0,
            reopening_velocity_per_s=1.6,
            rcr=0.80,
            classification=BlinkClassification.NORMAL,
            is_binocular=True,
            is_kinematics_reliable=True,
        )
        f = make_test_features(timestamp_ms=ts, completed_blink=b, total_blinks=i + 1)
        assessment = engine.evaluate(f)

    assert assessment is not None
    # Discrete counter is 0 because all 3 blinks were individually NORMAL (<= 350.0ms)
    assert assessment.temporal_metrics.prolonged_blink_count_60s == 0
    assert assessment.temporal_metrics.mean_blink_duration_ms_60s == 350.0
    assert assessment.temporal_metrics.blink_count_60s == 3
    # Aggregate sluggishness rule in DecisionEngine triggers DROWSINESS_WARNING
    assert assessment.state == DrowsinessState.DROWSINESS_WARNING
    assert any(r.code == ReasonCode.PROLONGED_BLINK for r in assessment.decision.reasons)

    # Control verification: 3 blinks averaging 340.0 ms do NOT trigger the warning
    control_engine = DrowsinessEngine()
    ts_ctrl = 1000.0
    ctrl_assessment = None
    for i in range(3):
        ts_ctrl += 2000.0
        b_ctrl = BlinkEvent(
            start_ts_ms=ts_ctrl - 340.0,
            min_ts_ms=ts_ctrl - 170.0,
            end_ts_ms=ts_ctrl,
            duration_ms=340.0,
            open_ear_before=0.30,
            min_ear=0.10,
            amplitude=0.20,
            closing_velocity_per_s=2.0,
            reopening_velocity_per_s=1.6,
            rcr=0.80,
            classification=BlinkClassification.NORMAL,
            is_binocular=True,
            is_kinematics_reliable=True,
        )
        f_ctrl = make_test_features(timestamp_ms=ts_ctrl, completed_blink=b_ctrl, total_blinks=i + 1)
        ctrl_assessment = control_engine.evaluate(f_ctrl)

    assert ctrl_assessment is not None
    assert ctrl_assessment.temporal_metrics.mean_blink_duration_ms_60s == 340.0
    assert ctrl_assessment.state == DrowsinessState.NORMAL
    assert not any(r.code == ReasonCode.PROLONGED_BLINK for r in ctrl_assessment.decision.reasons)


def test_7_rolling_window_expiration() -> None:
    """7. Rolling window: a prolonged blink expires from prolonged_blink_count_60s after 60s."""
    analyzer = TemporalAnalyzer()

    # Prolonged blink ending at t=2000.0ms
    b = BlinkEvent(
        start_ts_ms=1600.0,
        min_ts_ms=1800.0,
        end_ts_ms=2000.0,
        duration_ms=400.0,
        open_ear_before=0.30,
        min_ear=0.10,
        amplitude=0.20,
        closing_velocity_per_s=2.0,
        reopening_velocity_per_s=1.5,
        rcr=0.75,
        classification=BlinkClassification.PROLONGED,
        is_binocular=True,
        is_kinematics_reliable=True,
    )
    f1 = make_test_features(timestamp_ms=2000.0, completed_blink=b, total_blinks=1)
    m1 = analyzer.update(f1)
    assert m1.prolonged_blink_count_60s == 1
    assert m1.blink_count_60s == 1

    # At t=61000.0ms (59s later), blink is still within the 60s window
    f_mid = make_test_features(timestamp_ms=61000.0, total_blinks=1)
    m_mid = analyzer.update(f_mid)
    assert m_mid.prolonged_blink_count_60s == 1

    # At t=63000.0ms (61s later), blink has expired from 60s window
    f_exp = make_test_features(timestamp_ms=63000.0, total_blinks=1)
    m_exp = analyzer.update(f_exp)
    assert m_exp.prolonged_blink_count_60s == 0
    assert m_exp.blink_count_60s == 0
    assert m_exp.mean_blink_duration_ms_60s == 0.0


def test_8_invalid_configuration_raises_value_error() -> None:
    """8. Invalid configuration: prolonged >= microsleep or non-finite values raise ValueError."""
    # prolonged >= microsleep
    with pytest.raises(ValueError, match="strictly greater"):
        BlinkKinematicsDetector(prolonged_duration_ms=800.0, microsleep_duration_ms=800.0)

    with pytest.raises(ValueError, match="strictly greater"):
        BlinkKinematicsDetector(prolonged_duration_ms=850.0, microsleep_duration_ms=800.0)

    # Non-positive prolonged
    with pytest.raises(ValueError, match="positive"):
        BlinkKinematicsDetector(prolonged_duration_ms=0.0)

    with pytest.raises(ValueError, match="positive"):
        BlinkKinematicsDetector(prolonged_duration_ms=-10.0)

    # Non-finite values
    with pytest.raises(ValueError, match="finite"):
        BlinkKinematicsDetector(prolonged_duration_ms=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        BlinkKinematicsDetector(prolonged_duration_ms=float("inf"))

    with pytest.raises(ValueError, match="finite"):
        BlinkKinematicsDetector(microsleep_duration_ms=float("nan"))

    with pytest.raises(ValueError, match="finite"):
        BlinkKinematicsDetector(microsleep_duration_ms=float("inf"))


def test_9_left_censored_closure_interrupted_by_gap() -> None:
    """
    9. Left-censored closure: a closure interrupted by a sampling gap must not
    produce a fabricated completed prolonged blink when the eye reopens.
    """
    det = BlinkKinematicsDetector()
    # 1. Baseline open frame
    det.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    # 2. Eye closes
    det.update(timestamp_ms=1050.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    # 3. 350 ms camera freeze > max_sampling_gap_ms (250 ms)
    s_gap = det.update(timestamp_ms=1400.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_gap.state == BlinkState.UNCERTAIN_GAP

    # 4. Tracking resumes with eyes still closed for 400 ms
    for t in range(1450, 1850, 50):
        det.update(timestamp_ms=float(t), left_ear=0.10, right_ear=0.10, ear=0.10)

    # 5. Eyelid reopens
    s_reopen = det.update(timestamp_ms=1900.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_reopen.state == BlinkState.OPEN
    # Crucial guarantee: no fabricated completed blink is emitted on reopening
    assert s_reopen.completed_blink is None
    assert s_reopen.prolonged_blink_count == 0


def test_10_ongoing_closure_boundary() -> None:
    """
    10. Ongoing closure boundary: verify that the ongoing closure enters PROLONGED_CLOSURE
    at the configured threshold (350 ms), without incorrectly creating a completed blink.
    """
    det = BlinkKinematicsDetector(prolonged_duration_ms=350.0)
    det.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    det.update(timestamp_ms=1050.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    closure_start = det._closure_start_ts
    assert closure_start is not None

    # Stream intermediate closed frames
    t_349 = closure_start + 349.0
    t = 1100.0
    while t < t_349 - 25.0:
        det.update(timestamp_ms=t, left_ear=0.10, right_ear=0.10, ear=0.10)
        t += 50.0

    # At 349.0 ms of ongoing closure: still CLOSED
    s_before = det.update(timestamp_ms=t_349, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_before.state == BlinkState.CLOSED
    assert s_before.current_closure_duration_ms == 349.0
    assert s_before.completed_blink is None
    assert s_before.is_eye_closed is True

    # At exactly 350.0 ms of ongoing closure: transitions to PROLONGED_CLOSURE
    t_350 = closure_start + 350.0
    s_at = det.update(timestamp_ms=t_350, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_at.state == BlinkState.PROLONGED_CLOSURE
    assert s_at.current_closure_duration_ms == 350.0
    assert s_at.completed_blink is None
    assert s_at.is_eye_closed is True
