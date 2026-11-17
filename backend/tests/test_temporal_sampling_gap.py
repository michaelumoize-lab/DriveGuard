"""
DriveGuard — Phase 6, Task 1: Temporal Sampling-Gap Protection Regression Suite.

Verifies:
- Test A: Camera freeze during a normal blink (1600ms gap does not trigger false critical warning)
- Test B: Sustained closure interrupted by a gap (new closure starts at first post-gap observation)
- Test C: Camera freeze during off-road gaze (missing interval not counted in off-road dwell)
- Test D: Recovery after a gap (temporal tracking resumes cleanly without carrying stale durations)
- Test E: Timestamp integrity (duplicate, backward, and non-finite timestamps rejected safely)
- Test F: Explicit blink sampling-gap signal (features.blink.sampling_gap_detected respected)
- Extra: PoseZone.UNKNOWN and invalid pose rejection
"""

from __future__ import annotations

import math
import pytest

from backend.app.features.blink_detector import (
    BlinkClassification,
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


def make_features(
    timestamp_ms: float,
    has_face: bool = True,
    is_closed: bool = False,
    ear_val: float = 0.28,
    zone: PoseZone = PoseZone.CENTER,
    current_closure_duration_ms: float = 0.0,
    blink_state: BlinkState = BlinkState.OPEN,
    sampling_gap_detected: bool = False,
    is_pose_valid: bool = True,
) -> DriverFeatures:
    """Deterministic synthetic DriverFeatures generator for temporal tests."""
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
        is_valid=True,
    )

    head_pose = HeadPoseResult(
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        zone=zone,
        is_looking_away=(zone != PoseZone.CENTER),
        is_valid=is_pose_valid,
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

    state = BlinkState.UNCERTAIN_GAP if sampling_gap_detected else (
        blink_state if blink_state != BlinkState.OPEN else (
            BlinkState.CLOSED if is_closed else BlinkState.OPEN
        )
    )

    blink = BlinkSnapshot(
        state=state,
        is_eye_closed=is_closed,
        current_closure_duration_ms=current_closure_duration_ms,
        completed_blink=None,
        recent_blink_duration_ms=None,
        recent_blink_rcr=None,
        total_blink_count=0,
        prolonged_blink_count=0,
    )

    perclos = PerclosResult(
        perclos_pct=2.0,
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


def test_sampling_gap_a_camera_freeze_during_normal_blink() -> None:
    """
    Test A — Camera freeze during a normal blink.
    - Eye closure begins at 1000 ms.
    - A 150 ms normal blink would ordinarily occur.
    - The next observed frame arrives after a 1600 ms gap (at t=2600ms), eyes open.
    - The analyzer must not report 1600 ms of continuous eye closure.
    - No false critical warning must be generated from the unobserved interval.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)
    engine = DrowsinessEngine(sustained_closure_threshold_ms=1500.0, max_sampling_gap_ms=250.0)

    # Frame 1: Closure begins at t=1000ms
    f1 = make_features(timestamp_ms=1000.0, is_closed=True, ear_val=0.12)
    m1 = analyzer.update(f1)
    a1 = engine.evaluate(f1)
    assert m1.active_closure_duration_ms == 0.0
    assert a1.state == DrowsinessState.NORMAL

    # Frame 2: Next frame arrives after 1600ms gap at t=2600ms, eyes open
    f2 = make_features(timestamp_ms=2600.0, is_closed=False, ear_val=0.28)
    m2 = analyzer.update(f2)
    a2 = engine.evaluate(f2)

    # The 1600ms gap must NOT be reported as continuous closure
    assert m2.active_closure_duration_ms == 0.0
    assert analyzer.current_closure_duration_ms == 0.0
    # No false critical warning generated from unobserved interval
    assert a2.state == DrowsinessState.NORMAL
    assert a2.state != DrowsinessState.CRITICAL_WARNING


def test_sampling_gap_b_sustained_closure_interrupted_by_gap() -> None:
    """
    Test B — Sustained closure interrupted by a gap.
    - Eye closure begins at 1000 ms.
    - A valid closure interval is observed (e.g., 200ms at t=1200ms).
    - A 500 ms sampling gap occurs (next frame at t=1700ms).
    - The driver is still observed with closed eyes after the gap.
    - The new continuous closure duration must start from the first post-gap observation.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # t=1000ms: Closure begins
    f1 = make_features(timestamp_ms=1000.0, is_closed=True, ear_val=0.12)
    m1 = analyzer.update(f1)
    assert m1.active_closure_duration_ms == 0.0

    # t=1100ms: Continuous closure (dt=100ms <= 250ms)
    f2 = make_features(timestamp_ms=1100.0, is_closed=True, ear_val=0.12)
    m2 = analyzer.update(f2)
    assert m2.active_closure_duration_ms == 100.0

    # t=1200ms: Valid closure interval observed
    f3 = make_features(timestamp_ms=1200.0, is_closed=True, ear_val=0.12)
    m3 = analyzer.update(f3)
    assert m3.active_closure_duration_ms == 200.0

    # t=1700ms: 500ms sampling gap occurs (dt=500ms > 250ms), driver still closed
    f4 = make_features(timestamp_ms=1700.0, is_closed=True, ear_val=0.12)
    m4 = analyzer.update(f4)

    # First post-gap observation: duration must reset to 0.0 (new start at t=1700ms)
    assert m4.active_closure_duration_ms == 0.0
    assert analyzer.current_closure_duration_ms == 0.0

    # t=1750ms: Continuous observation after gap (dt=50ms <= 250ms)
    f5 = make_features(timestamp_ms=1750.0, is_closed=True, ear_val=0.12)
    m5 = analyzer.update(f5)
    # Duration starts fresh from t=1700ms: 1750 - 1700 = 50ms, NOT 750ms!
    assert m5.active_closure_duration_ms == 50.0
    assert analyzer.current_closure_duration_ms == 50.0


def test_sampling_gap_c_camera_freeze_during_off_road_gaze() -> None:
    """
    Test C — Camera freeze during off-road gaze.
    - Driver looks away (PoseZone.OUT_OF_BOUNDS).
    - A large sampling gap occurs.
    - Off-road gaze duration must not include the missing interval.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # t=1000ms: Look away starts
    f1 = make_features(timestamp_ms=1000.0, zone=PoseZone.OUT_OF_BOUNDS)
    m1 = analyzer.update(f1)
    assert m1.active_off_road_dwell_ms == 0.0

    # t=1200ms: Valid off-road dwell of 200ms
    f2 = make_features(timestamp_ms=1200.0, zone=PoseZone.OUT_OF_BOUNDS)
    m2 = analyzer.update(f2)
    assert m2.active_off_road_dwell_ms == 200.0

    # t=4000ms: Large sampling gap (2800ms unobserved gap)
    f3 = make_features(timestamp_ms=4000.0, zone=PoseZone.OUT_OF_BOUNDS)
    m3 = analyzer.update(f3)

    # Gaze duration must not include the 2800ms missing interval!
    # On first post-gap frame, dwell starts fresh at 0.0ms
    assert m3.active_off_road_dwell_ms == 0.0
    assert analyzer.current_off_road_duration_ms == 0.0

    # t=4100ms: Next continuous observation (dt=100ms <= 250ms)
    f4 = make_features(timestamp_ms=4100.0, zone=PoseZone.OUT_OF_BOUNDS)
    m4 = analyzer.update(f4)
    # Elapsed is 100ms since resumption at t=4000ms, NOT 3100ms!
    assert m4.active_off_road_dwell_ms == 100.0


def test_sampling_gap_d_recovery_after_gap() -> None:
    """
    Test D — Recovery after a gap.
    - Verify that normal observations after the gap allow temporal tracking to resume correctly.
    - Ensure no stale closure or gaze duration is carried forward.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Active closure and gaze before gap
    f1 = make_features(timestamp_ms=1000.0, is_closed=True, zone=PoseZone.OUT_OF_BOUNDS)
    analyzer.update(f1)
    f2 = make_features(timestamp_ms=1100.0, is_closed=True, zone=PoseZone.OUT_OF_BOUNDS)
    analyzer.update(f2)

    # Gap of 2000ms arrives at t=3100ms with driver alert and looking center
    f_gap = make_features(timestamp_ms=3100.0, is_closed=False, zone=PoseZone.CENTER)
    m_gap = analyzer.update(f_gap)
    assert m_gap.active_closure_duration_ms == 0.0
    assert m_gap.active_off_road_dwell_ms == 0.0

    # Normal continuous observations resume at 30 fps (dt ≈ 33.3ms)
    f_norm1 = make_features(timestamp_ms=3133.3, is_closed=False, zone=PoseZone.CENTER)
    m_norm1 = analyzer.update(f_norm1)
    assert m_norm1.active_closure_duration_ms == 0.0
    assert m_norm1.active_off_road_dwell_ms == 0.0

    # Now driver closes eyes at t=3166.6ms
    f_close1 = make_features(timestamp_ms=3166.6, is_closed=True, zone=PoseZone.CENTER)
    m_close1 = analyzer.update(f_close1)
    assert m_close1.active_closure_duration_ms == 0.0

    f_close2 = make_features(timestamp_ms=3200.0, is_closed=True, zone=PoseZone.CENTER)
    m_close2 = analyzer.update(f_close2)
    # Correctly tracks closure: 3200.0 - 3166.6 = 33.4ms, completely unaffected by pre-gap state
    assert pytest.approx(m_close2.active_closure_duration_ms, abs=1.0) == 33.4


def test_sampling_gap_e_invalid_timestamps() -> None:
    """
    Test E — Invalid timestamps.
    - Test duplicate timestamps.
    - Test backward timestamps.
    - Test non-finite (NaN, Inf) and negative timestamps.
    - Confirm that invalid observations do not corrupt temporal analyzer state.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Valid initial frame at t=1000ms
    f_valid = make_features(timestamp_ms=1000.0, is_closed=True)
    m_valid = analyzer.update(f_valid)
    assert m_valid.timestamp_ms == 1000.0
    assert m_valid.active_closure_duration_ms == 0.0

    # 1. Duplicate timestamp (t=1000.0ms)
    f_dup = make_features(timestamp_ms=1000.0, is_closed=False)
    m_dup = analyzer.update(f_dup)
    # Must reject without corrupting state (returns last valid metrics)
    assert m_dup.timestamp_ms == 1000.0
    assert analyzer.current_closure_duration_ms == 0.0

    # 2. Backward timestamp (t=900.0ms)
    f_back = make_features(timestamp_ms=900.0, is_closed=False)
    m_back = analyzer.update(f_back)
    assert m_back.timestamp_ms == 1000.0

    # 3. Non-finite timestamps (NaN, Inf, -Inf)
    f_nan = make_features(timestamp_ms=float("nan"), is_closed=False)
    m_nan = analyzer.update(f_nan)
    assert m_nan.timestamp_ms == 1000.0

    f_inf = make_features(timestamp_ms=float("inf"), is_closed=False)
    m_inf = analyzer.update(f_inf)
    assert m_inf.timestamp_ms == 1000.0

    # 4. Negative timestamp
    f_neg = make_features(timestamp_ms=-100.0, is_closed=False)
    m_neg = analyzer.update(f_neg)
    assert m_neg.timestamp_ms == 1000.0

    # 5. Subsequent valid frame at t=1050ms must proceed normally
    f_next = make_features(timestamp_ms=1050.0, is_closed=True)
    m_next = analyzer.update(f_next)
    assert m_next.timestamp_ms == 1050.0
    # 1050 - 1000 = 50ms closure
    assert m_next.active_closure_duration_ms == 50.0


def test_sampling_gap_f_explicit_blink_sampling_gap_signal() -> None:
    """
    Test F — Explicit blink sampling-gap signal.
    Verify that features.blink.sampling_gap_detected=True is respected even if
    the timestamp-delta check alone would not identify the gap (e.g. dt = 33ms <= 250ms).
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Frame 1: Eyes closed at t=1000ms
    f1 = make_features(timestamp_ms=1000.0, is_closed=True)
    m1 = analyzer.update(f1)
    assert m1.active_closure_duration_ms == 0.0

    # Frame 2: Continuous closure at t=1100ms
    f2 = make_features(timestamp_ms=1100.0, is_closed=True)
    m2 = analyzer.update(f2)
    assert m2.active_closure_duration_ms == 100.0

    # Frame 3: Timestamp delta is only 33ms (dt <= 250ms), BUT blink detector
    # explicitly flagged sampling_gap_detected=True (or BlinkState.UNCERTAIN_GAP)
    f3 = make_features(
        timestamp_ms=1133.3,
        is_closed=True,
        sampling_gap_detected=True,
    )
    m3 = analyzer.update(f3)

    # Explicit blink gap signal must invalidate continuous closure accumulation
    assert m3.active_closure_duration_ms == 0.0
    assert analyzer.current_closure_duration_ms == 0.0

    # Frame 4: Next continuous frame at t=1166.6ms
    f4 = make_features(timestamp_ms=1166.6, is_closed=True)
    m4 = analyzer.update(f4)
    # Starts fresh from t=1133.3ms: 1166.6 - 1133.3 = 33.3ms
    assert pytest.approx(m4.active_closure_duration_ms, abs=1.0) == 33.3


def test_sampling_gap_pose_unknown_does_not_accumulate_off_road() -> None:
    """Ensure PoseZone.UNKNOWN and invalid head pose cannot accumulate off-road dwell."""
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Looking out of bounds for 200ms
    analyzer.update(make_features(timestamp_ms=1000.0, zone=PoseZone.OUT_OF_BOUNDS))
    m2 = analyzer.update(make_features(timestamp_ms=1200.0, zone=PoseZone.OUT_OF_BOUNDS))
    assert m2.active_off_road_dwell_ms == 200.0

    # PoseZone.UNKNOWN arrives at t=1300ms
    m3 = analyzer.update(make_features(timestamp_ms=1300.0, zone=PoseZone.UNKNOWN))
    assert m3.active_off_road_dwell_ms == 0.0
    assert analyzer.current_off_road_duration_ms == 0.0

    # Invalid head pose at t=1400ms
    m4 = analyzer.update(
        make_features(timestamp_ms=1400.0, zone=PoseZone.OUT_OF_BOUNDS, is_pose_valid=False)
    )
    assert m4.active_off_road_dwell_ms == 0.0
    assert analyzer.current_off_road_duration_ms == 0.0
