"""
DriveGuard Phase 7 — Task 3.1: Ocular Tracking Gap & Head Pose Independence Regression Suite.

Verifies the architectural decoupling between:
1. Actual camera sampling gaps (dt > 250 ms): unobserved intervals must reset/re-anchor temporal tracking.
2. Ocular tracking gaps (invalid EAR / BlinkState.UNCERTAIN_GAP with dt <= 250 ms): must NOT destroy
   valid head-pose pitch history or reset valid dashboard/off-road dwell.
3. Invalid head pose / UNKNOWN zone: must not accumulate dwell or corrupt pitch history.
4. Combined invalidity: camera sampling gap protection strictly takes precedence.
"""

from __future__ import annotations

import math
import pytest

from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatureExtractor, DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult
from backend.app.features.perclos import PerclosResult
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus, NormalizedLandmark


def make_features(
    timestamp_ms: float,
    has_face: bool = True,
    face_status: FaceLandmarkStatus = FaceLandmarkStatus.TRACKING,
    is_closed: bool = False,
    ear_val: float = 0.28,
    ear_is_valid: bool = True,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    zone: PoseZone = PoseZone.CENTER,
    is_pose_valid: bool = True,
    current_closure_duration_ms: float = 0.0,
    blink_state: BlinkState = BlinkState.OPEN,
    sampling_gap_detected: bool = False,
    perclos_pct: float = 2.0,
    perclos_is_reliable: bool = True,
) -> DriverFeatures:
    """Deterministic synthetic DriverFeatures generator for independence tests."""
    status = face_status if has_face else FaceLandmarkStatus.INSUFFICIENT_DATA

    ear = EyeAspectRatioResult(
        left_ear=ear_val if ear_is_valid else 0.0,
        right_ear=ear_val if ear_is_valid else 0.0,
        weighted_ear=ear_val if ear_is_valid else 0.0,
        compensated_ear=ear_val if ear_is_valid else 0.0,
        closure_threshold=0.20,
        is_left_closed=is_closed if ear_is_valid else False,
        is_right_closed=is_closed if ear_is_valid else False,
        is_closed=is_closed if ear_is_valid else False,
        is_valid=ear_is_valid,
        is_left_valid=ear_is_valid,
        is_right_valid=ear_is_valid,
    )

    head_pose = HeadPoseResult(
        pitch_deg=pitch_deg,
        yaw_deg=yaw_deg,
        roll_deg=0.0,
        zone=zone,
        is_looking_away=(zone != PoseZone.CENTER),
        is_valid=is_pose_valid,
        rel_pitch_deg=pitch_deg,
        rel_yaw_deg=yaw_deg,
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

    b_state = BlinkState.UNCERTAIN_GAP if (sampling_gap_detected or not ear_is_valid) else (
        blink_state if blink_state != BlinkState.OPEN else (
            BlinkState.CLOSED if is_closed else BlinkState.OPEN
        )
    )

    blink = BlinkSnapshot(
        state=b_state,
        is_eye_closed=is_closed if ear_is_valid and b_state != BlinkState.UNCERTAIN_GAP else False,
        current_closure_duration_ms=current_closure_duration_ms if ear_is_valid else 0.0,
        completed_blink=None,
        recent_blink_duration_ms=None,
        recent_blink_rcr=None,
        total_blink_count=0,
        prolonged_blink_count=0,
    )

    perclos = PerclosResult(
        perclos_pct=perclos_pct,
        coverage_pct=100.0,
        is_reliable=perclos_is_reliable,
        closed_duration_sec=0.0,
        observed_duration_sec=60.0,
        window_sec=60.0,
    )

    return DriverFeatures(
        timestamp_ms=timestamp_ms,
        has_face=has_face,
        face_status=status,
        ear=ear,
        head_pose=head_pose,
        mar=mar,
        blink=blink,
        perclos=perclos,
    )


def test_1_valid_head_pose_with_valid_eyes_baseline_unchanged() -> None:
    """
    Scenario 1: Valid head pose with valid eyes.
    Verifies that existing baseline behavior is completely preserved:
    - Pitch history accumulates.
    - Dashboard dwell accumulates and reflects context.
    - Closure tracking functions normally.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Stream 10 frames at nominal 30 fps (dt = 33.3ms) in PoseZone.DASHBOARD with open eyes
    for i in range(10):
        t = 1000.0 + i * 33.3
        f = make_features(
            timestamp_ms=t,
            pitch_deg=-12.0,
            zone=PoseZone.DASHBOARD,
            is_closed=False,
            ear_is_valid=True,
        )
        m = analyzer.update(f)

    # 10 frames accumulated
    assert len(analyzer.pitch_history) == 10
    assert m.active_closure_duration_ms == 0.0
    assert pytest.approx(m.active_off_road_dwell_ms, abs=1.0) == 9 * 33.3
    assert m.is_dashboard_check_context is True
    assert m.is_eye_verified_open is True


def test_2_valid_head_pose_with_invalid_ear_preserves_pitch_history() -> None:
    """
    Scenario 2: Valid head pose with invalid EAR for several consecutive frames.
    - Frames continue arriving at nominal 30 fps (dt = 33.3ms <= 250ms).
    - EAR is invalid (eyes occluded or unmeasurable), so blink detector is in UNCERTAIN_GAP.
    - Head pose remains valid.
    - Pitch history must NOT be cleared solely due to ocular tracking gaps!
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Frame 1: Valid eyes, pitch -10.0°
    f1 = make_features(
        timestamp_ms=1000.0,
        pitch_deg=-10.0,
        zone=PoseZone.DASHBOARD,
        ear_is_valid=True,
    )
    m1 = analyzer.update(f1)
    assert len(analyzer.pitch_history) == 1

    # Frames 2-6: Eyes become invalid/occluded, pitch continues down (-12°, -14°, -16°, -18°, -20°)
    # dt = 33.3ms (no camera sampling gap)
    pitches = [-12.0, -14.0, -16.0, -18.0, -20.0]
    for idx, p in enumerate(pitches):
        t = 1033.3 + idx * 33.3
        f = make_features(
            timestamp_ms=t,
            pitch_deg=p,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
            sampling_gap_detected=True,
        )
        m = analyzer.update(f)
        # Pitch history must NOT be wiped!
        assert len(analyzer.pitch_history) == idx + 2
        # Eye closure must remain 0.0ms (no false closure fabricated)
        assert m.active_closure_duration_ms == 0.0
        # Tri-state eye state is None (unknown)
        assert m.is_eye_closed is None

    # Verify all recorded pitches are intact
    recorded_pitches = [p for _, p in analyzer.pitch_history]
    assert recorded_pitches == [-10.0, -12.0, -14.0, -16.0, -18.0, -20.0]


def test_3_valid_dashboard_pose_with_invalid_ear_preserves_dwell() -> None:
    """
    Scenario 3: Valid dashboard pose with invalid EAR.
    - Driver tilts head to PoseZone.DASHBOARD.
    - Eyes become occluded / EAR invalid, but frames continue arriving normally (dt = 100ms).
    - Off-road dwell must NOT reset to 0.0ms on every frame solely due to ocular gap!
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # t=1000ms: Head enters DASHBOARD with valid eyes
    f1 = make_features(
        timestamp_ms=1000.0,
        pitch_deg=-14.0,
        zone=PoseZone.DASHBOARD,
        ear_is_valid=True,
    )
    m1 = analyzer.update(f1)
    assert m1.active_off_road_dwell_ms == 0.0

    # Stream frames from t=1100ms to t=2500ms (14 frames, dt=100ms <= 250ms)
    # with EAR invalid / blink in UNCERTAIN_GAP
    for t in range(1100, 2600, 100):
        f = make_features(
            timestamp_ms=float(t),
            pitch_deg=-16.0,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
            sampling_gap_detected=True,
        )
        m = analyzer.update(f)
        # Dwell must accumulate continuously: t - 1000.0 ms
        expected_dwell = float(t - 1000)
        assert m.active_off_road_dwell_ms == expected_dwell
        assert analyzer.current_off_road_duration_ms == expected_dwell

    # At t=2500ms, dwell is 1500ms (still within dashboard context <= 2000ms)
    assert m.is_dashboard_check_context is True

    # Continue stream to t=3200ms (2200ms elapsed > 2000ms)
    for t in range(2600, 3300, 100):
        f = make_features(
            timestamp_ms=float(t),
            pitch_deg=-16.0,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
            sampling_gap_detected=True,
        )
        m = analyzer.update(f)

    assert m.active_off_road_dwell_ms == 2200.0
    # Context expired after 2000ms
    assert m.is_dashboard_check_context is False


def test_4_actual_camera_sampling_gap_resets_pitch_history_and_reanchors_dwell() -> None:
    """
    Scenario 4: Actual camera sampling gap exceeding 250 ms.
    - True inter-frame drop: dt = 800ms > 250ms.
    - Pitch history must be cleared (cannot bridge unobserved camera gap).
    - Off-road dwell must not count missing 800ms interval.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Initial frames at t=1000, 1100, 1200ms
    for t in (1000.0, 1100.0, 1200.0):
        analyzer.update(
            make_features(
                timestamp_ms=t,
                pitch_deg=-12.0,
                zone=PoseZone.DASHBOARD,
                ear_is_valid=True,
            )
        )
    assert len(analyzer.pitch_history) == 3
    assert analyzer.current_off_road_duration_ms == 200.0

    # Camera freeze: next frame arrives at t=2000ms (dt = 800ms > 250ms)
    f_gap = make_features(
        timestamp_ms=2000.0,
        pitch_deg=-12.0,
        zone=PoseZone.DASHBOARD,
        ear_is_valid=True,
    )
    m_gap = analyzer.update(f_gap)

    # Pitch history must be cleared of pre-gap data and re-anchored to fresh post-gap observation
    assert len(analyzer.pitch_history) == 1
    assert analyzer.pitch_history[0] == (2000.0, -12.0)

    # Off-road dwell must NOT include the 800ms missing interval!
    # Re-anchors fresh to 0.0ms at t=2000ms
    assert m_gap.active_off_road_dwell_ms == 0.0
    assert analyzer.current_off_road_duration_ms == 0.0

    # Next continuous frame at t=2100ms (dt = 100ms <= 250ms)
    f_next = make_features(
        timestamp_ms=2100.0,
        pitch_deg=-12.0,
        zone=PoseZone.DASHBOARD,
        ear_is_valid=True,
    )
    m_next = analyzer.update(f_next)
    assert m_next.active_off_road_dwell_ms == 100.0
    assert len(analyzer.pitch_history) == 2


def test_5_invalid_head_pose_does_not_accumulate_dwell_or_corrupt_history() -> None:
    """
    Scenario 5: Invalid head pose or PoseZone.UNKNOWN.
    - When head pose estimation fails (is_valid=False or zone=UNKNOWN),
      off-road dwell must not accumulate.
    - Pitch history must be cleared and not receive corrupt/invalid pitch angles.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Valid frames at t=1000, 1100ms in DASHBOARD
    analyzer.update(make_features(timestamp_ms=1000.0, pitch_deg=-15.0, zone=PoseZone.DASHBOARD))
    m2 = analyzer.update(make_features(timestamp_ms=1100.0, pitch_deg=-15.0, zone=PoseZone.DASHBOARD))
    assert m2.active_off_road_dwell_ms == 100.0
    assert len(analyzer.pitch_history) == 2

    # Frame at t=1200ms has SolvePnP failure: is_pose_valid=False, zone=PoseZone.UNKNOWN
    f_invalid = make_features(
        timestamp_ms=1200.0,
        pitch_deg=0.0,
        zone=PoseZone.UNKNOWN,
        is_pose_valid=False,
    )
    m_invalid = analyzer.update(f_invalid)

    # Dwell immediately drops to 0.0ms
    assert m_invalid.active_off_road_dwell_ms == 0.0
    assert analyzer.current_off_road_duration_ms == 0.0
    # Pitch history cleared (no stale/corrupt orientation preserved)
    assert len(analyzer.pitch_history) == 0

    # Another frame with invalid pose attempting to pass extreme pitch
    f_invalid2 = make_features(
        timestamp_ms=1300.0,
        pitch_deg=-30.0,
        zone=PoseZone.UNKNOWN,
        is_pose_valid=False,
    )
    m_invalid2 = analyzer.update(f_invalid2)
    assert m_invalid2.active_off_road_dwell_ms == 0.0
    assert len(analyzer.pitch_history) == 0


def test_6_face_loss_grace_and_insufficient_data_intact() -> None:
    """
    Scenario 6: Face loss.
    Verifies that existing approved face-loss grace (1000ms) and subsequent
    transition to INSUFFICIENT_DATA in DecisionEngine / DrowsinessEngine remain intact.
    """
    engine = DrowsinessEngine(face_loss_grace_ms=1000.0)

    # t=1000ms: Face lost
    f1 = make_features(timestamp_ms=1000.0, has_face=False)
    a1 = engine.evaluate(f1)
    assert a1.state == DrowsinessState.NORMAL
    assert a1.decision.primary_reason.code == ReasonCode.FACE_LOST

    # Stream face loss within grace period (up to t=1900ms, 900ms elapsed < 1000ms)
    for t in range(1100, 2000, 100):
        a = engine.evaluate(make_features(timestamp_ms=float(t), has_face=False))
        assert a.state == DrowsinessState.NORMAL
        assert a.decision.primary_reason.code == ReasonCode.FACE_LOST

    # t=2000ms: 1000ms elapsed -> Transitions to INSUFFICIENT_DATA
    f_expired = make_features(timestamp_ms=2000.0, has_face=False)
    a_expired = engine.evaluate(f_expired)
    assert a_expired.state == DrowsinessState.INSUFFICIENT_DATA
    assert a_expired.severity == -1
    assert a_expired.decision.primary_reason.code == ReasonCode.FACE_LOST


def test_7_eye_visibility_returning_recovers_without_fabricated_blink() -> None:
    """
    Scenario 7: Eye visibility returning.
    - Pipeline transitions from occluded eyes (UNCERTAIN_GAP) to visible open eyes.
    - Blink detector left-censoring recovers cleanly to OPEN without emitting a fabricated blink.
    """
    detector = BlinkKinematicsDetector()

    # Frame 1: Tracking loss / occluded eyes (left_valid=False, right_valid=False)
    snap1 = detector.update(
        left_ear=None,
        right_ear=None,
        ear=0.0,
        timestamp_ms=1000.0,
        has_face=False,
        left_valid=False,
        right_valid=False,
    )
    assert snap1.state == BlinkState.UNCERTAIN_GAP

    # Frame 2: Eye visibility returns with open eyes (EAR = 0.28 >= reopen_threshold 0.24)
    snap2 = detector.update(
        left_ear=0.28,
        right_ear=0.28,
        ear=0.28,
        timestamp_ms=1033.3,
        has_face=True,
        left_valid=True,
        right_valid=True,
    )
    # Recovers directly to OPEN
    assert snap2.state == BlinkState.OPEN
    # Zero fabricated blink emitted
    assert snap2.completed_blink is None
    assert snap2.current_closure_duration_ms == 0.0
    assert snap2.total_blink_count == 0


def test_8_combined_camera_gap_and_ocular_gap_camera_gap_precedence() -> None:
    """
    Scenario 8: Combined camera sampling gap and ocular tracking gap.
    When both conditions occur simultaneously (e.g. dt = 600ms > 250ms AND ear_is_valid=False),
    the camera sampling-gap protection strictly takes precedence:
    - Pitch history is cleared of pre-gap data.
    - Off-road dwell resets to 0.0ms.
    - Eye closure remains 0.0ms.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Valid frames before gap
    for t in (1000.0, 1100.0):
        analyzer.update(
            make_features(
                timestamp_ms=t,
                pitch_deg=-15.0,
                zone=PoseZone.DASHBOARD,
                ear_is_valid=True,
            )
        )
    assert len(analyzer.pitch_history) == 2
    assert analyzer.current_off_road_duration_ms == 100.0

    # Frame after large gap (dt = 600ms > 250ms) with simultaneous ocular invalidity
    f_combined = make_features(
        timestamp_ms=1700.0,
        pitch_deg=-15.0,
        zone=PoseZone.DASHBOARD,
        ear_is_valid=False,
        sampling_gap_detected=True,
    )
    m_combined = analyzer.update(f_combined)

    # Camera sampling gap must reset dwell to 0.0ms (not carry forward pre-gap 100ms or 600ms gap)
    assert m_combined.active_off_road_dwell_ms == 0.0
    assert analyzer.current_off_road_duration_ms == 0.0

    # Pitch history must be cleared of pre-gap entries and retain only fresh observation
    assert len(analyzer.pitch_history) == 1
    assert analyzer.pitch_history[0] == (1700.0, -15.0)

    # Eye closure is 0.0ms
    assert m_combined.active_closure_duration_ms == 0.0


def test_a_valid_pose_degraded_overall_confidence_preserves_pitch_history() -> None:
    """
    Task 3.1.1 — Test A: Valid pose, degraded overall confidence.
    - Continuous frames with intervals below 250 ms (dt = 33.3 ms).
    - features.has_face = True.
    - Valid head pose in DASHBOARD.
    - Invalid EAR (ear_is_valid = False).
    - is_reliable = False (composite confidence degraded below 0.50).
    Expected:
    - Pitch history continues accumulating.
    - Valid head-pose observations are retained.
    - Eye state remains unknown (None).
    - No fabricated eye closure or blink (active_closure_duration_ms == 0.0).
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    pitches = [-10.0, -12.5, -15.0, -17.5, -20.0]
    for idx, p in enumerate(pitches):
        t = 1000.0 + idx * 33.3
        f = make_features(
            timestamp_ms=t,
            pitch_deg=p,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
            sampling_gap_detected=True,
        )
        # Pass is_reliable=False explicitly representing degraded overall confidence
        m = analyzer.update(f, is_reliable=False)

        # Pitch history must accumulate normally without being cleared by not is_reliable
        assert len(analyzer.pitch_history) == idx + 1
        assert analyzer.pitch_history[-1] == (t, p)
        # Eye closure must remain 0.0ms (no false closure fabricated)
        assert m.active_closure_duration_ms == 0.0
        # Tri-state eye state is None (unknown)
        assert m.is_eye_closed is None

    recorded_pitches = [p for _, p in analyzer.pitch_history]
    assert recorded_pitches == pitches


def test_b_confidence_recovers_preserves_pitch_history_continuity() -> None:
    """
    Task 3.1.1 — Test B: Confidence recovers.
    - Begin with valid head pose and is_reliable = False for several frames.
    - Later change to is_reliable = True.
    - Continuous frames (dt = 33.3 ms <= 250 ms).
    Expected:
    - No unexpected reset or pitch-history discontinuity when confidence returns.
    - Observations continue accumulating in the same rolling window.
    - Sequence ordering is preserved across the confidence transition.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # First phase: 3 frames with is_reliable = False
    for idx, p in enumerate([-10.0, -12.0, -14.0]):
        t = 1000.0 + idx * 33.3
        f = make_features(
            timestamp_ms=t,
            pitch_deg=p,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
        )
        analyzer.update(f, is_reliable=False)

    assert len(analyzer.pitch_history) == 3

    # Second phase: 3 frames with is_reliable = True (confidence recovers)
    for idx, p in enumerate([-16.0, -18.0, -20.0]):
        t = 1100.0 + idx * 33.3
        f = make_features(
            timestamp_ms=t,
            pitch_deg=p,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=True,
        )
        analyzer.update(f, is_reliable=True)

    # Sequence intact: all 6 observations retained in order
    assert len(analyzer.pitch_history) == 6
    expected_pitches = [-10.0, -12.0, -14.0, -16.0, -18.0, -20.0]
    assert [p for _, p in analyzer.pitch_history] == expected_pitches


def test_c_pose_invalidation_still_clears_pitch_history() -> None:
    """
    Task 3.1.1 — Test C: Pose invalidation still clears history.
    - Continuous frames with valid timestamps.
    - Start with valid pose entries in history.
    - Next frame: features.head_pose.is_valid = False and zone = PoseZone.UNKNOWN (even with is_reliable=True).
    - Next frame: features.has_face = False.
    Expected:
    - Pitch history is cleared on invalid pose and on face loss.
    - No invalid observations are added.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Valid frames to establish history
    analyzer.update(make_features(timestamp_ms=1000.0, pitch_deg=-10.0, zone=PoseZone.DASHBOARD), is_reliable=True)
    analyzer.update(make_features(timestamp_ms=1033.3, pitch_deg=-12.0, zone=PoseZone.DASHBOARD), is_reliable=True)
    assert len(analyzer.pitch_history) == 2

    # Pose invalidation: is_pose_valid = False, zone = UNKNOWN
    f_invalid_pose = make_features(
        timestamp_ms=1066.6,
        pitch_deg=-30.0,
        zone=PoseZone.UNKNOWN,
        is_pose_valid=False,
    )
    analyzer.update(f_invalid_pose, is_reliable=True)

    # Pitch history must be cleared, invalid observation NOT added
    assert len(analyzer.pitch_history) == 0

    # Re-establish 1 valid frame
    analyzer.update(make_features(timestamp_ms=1100.0, pitch_deg=-10.0, zone=PoseZone.DASHBOARD), is_reliable=True)
    assert len(analyzer.pitch_history) == 1

    # Face loss: has_face = False
    f_face_loss = make_features(
        timestamp_ms=1133.3,
        has_face=False,
        pitch_deg=-10.0,
        zone=PoseZone.UNKNOWN,
        is_pose_valid=False,
    )
    analyzer.update(f_face_loss, is_reliable=True)

    # Pitch history cleared on face loss
    assert len(analyzer.pitch_history) == 0


def test_d_sampling_gap_still_clears_history_even_with_degraded_confidence() -> None:
    """
    Task 3.1.1 — Test D: Sampling gap still clears history.
    - Valid head pose with is_reliable = False.
    - Frame interval exceeds 250 ms limit (dt = 750 ms > 250 ms).
    Expected:
    - Sampling gap clears pre-gap pitch history.
    - Pre-gap head pose is not carried across the gap.
    - Re-anchors to the fresh post-gap observation.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Pre-gap frames (dt = 50 ms) with degraded confidence
    analyzer.update(make_features(timestamp_ms=1000.0, pitch_deg=-10.0, zone=PoseZone.DASHBOARD), is_reliable=False)
    analyzer.update(make_features(timestamp_ms=1050.0, pitch_deg=-12.0, zone=PoseZone.DASHBOARD), is_reliable=False)
    assert len(analyzer.pitch_history) == 2

    # Camera sampling gap: dt = 750 ms > 250 ms
    f_gap = make_features(
        timestamp_ms=1800.0,
        pitch_deg=-15.0,
        zone=PoseZone.DASHBOARD,
    )
    analyzer.update(f_gap, is_reliable=False)

    # History cleared of pre-gap entries; only fresh post-gap observation retained
    assert len(analyzer.pitch_history) == 1
    assert analyzer.pitch_history[0] == (1800.0, -15.0)


def test_e_dashboard_off_road_dwell_continuation_under_degraded_confidence() -> None:
    """
    Task 3.1.1 — Test E: Dashboard/off-road dwell continuation under degraded confidence.
    - Continuous frames (dt = 100 ms).
    - Valid head pose in DASHBOARD.
    - Invalid EAR.
    - is_reliable = False.
    Expected:
    - Off-road dwell continues accumulating correctly.
    - Dashboard-check context remains valid for the first 2000 ms.
    - Dwell transitions past 2000 ms according to existing rules.
    - Degraded confidence does not prematurely zero dwell.
    """
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    # Frame 1: t=1000ms enters DASHBOARD with degraded confidence
    f1 = make_features(
        timestamp_ms=1000.0,
        pitch_deg=-14.0,
        zone=PoseZone.DASHBOARD,
        ear_is_valid=False,
    )
    m1 = analyzer.update(f1, is_reliable=False)
    assert m1.active_off_road_dwell_ms == 0.0

    # Stream continuous frames to t=2500ms (1500 ms elapsed, dt = 100ms <= 250ms)
    for t in range(1100, 2600, 100):
        f = make_features(
            timestamp_ms=float(t),
            pitch_deg=-16.0,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
        )
        m = analyzer.update(f, is_reliable=False)
        expected_dwell = float(t - 1000)
        assert m.active_off_road_dwell_ms == expected_dwell
        assert analyzer.current_off_road_duration_ms == expected_dwell

    # Within 2000 ms: is_dashboard_check_context is True
    assert m.active_off_road_dwell_ms == 1500.0
    assert m.is_dashboard_check_context is True

    # Continue stream past 2000 ms limit to t=3200ms (2200 ms elapsed)
    for t in range(2600, 3300, 100):
        f = make_features(
            timestamp_ms=float(t),
            pitch_deg=-16.0,
            zone=PoseZone.DASHBOARD,
            ear_is_valid=False,
        )
        m = analyzer.update(f, is_reliable=False)

    # Dwell continues accumulating to 2200 ms, context expired
    assert m.active_off_road_dwell_ms == 2200.0
    assert m.is_dashboard_check_context is False

