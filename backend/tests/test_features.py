"""
Unit tests for DriveGuard Physiological Feature Extraction (Phase 5).

Covers:
  - Eye Aspect Ratio (EAR) single eye calculation (open vs. closed)
  - Yaw-weighted EAR under head turns (mirror checks)
  - Pitch-compensated EAR during downward instrument glances
  - SolvePnP 6-DoF Head Pose estimation and cockpit zone classification
  - Mouth Aspect Ratio (MAR) and conversation vs. authentic yawn discrimination
  - Sub-frame blink kinematics, closing/reopening velocities, and RCR
  - Rolling 60-second PERCLOS accumulation and observation coverage tracking
  - Unified DriverFeatureExtractor coordinating all feature streams
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pytest

from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkEvent,
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import (
    DEFAULT_EAR_CLOSURE_THRESHOLD,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeAspectRatioResult,
    calculate_pitch_compensated_ear,
    calculate_single_eye_ear,
    calculate_yaw_weighted_ear,
    extract_eye_aspect_ratio,
)
from backend.app.features.feature_extractor import (
    DriverFeatureExtractor,
    DriverFeatures,
)
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    SOLVEPNP_LANDMARK_INDICES,
    HeadPoseResult,
    PoseZone,
    classify_pose_zone,
    estimate_head_pose,
)
from backend.app.features.mar import (
    DEFAULT_MAR_OPEN_THRESHOLD,
    DEFAULT_MIN_YAWN_DURATION_MS,
    MOUTH_OUTER_CORNERS,
    MOUTH_VERTICAL_PAIRS,
    MouthAspectRatioResult,
    YawnDetector,
    calculate_mar,
)
from backend.app.features.perclos import (
    PerclosCalculator,
    PerclosResult,
)
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


def _build_mock_landmarks(overrides: Dict[int, Tuple[float, float, float]] = None) -> List[NormalizedLandmark]:
    """Helper to build a 478-point list of NormalizedLandmarks with optional overrides."""
    pts = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    if overrides:
        for idx, (x, y, z) in overrides.items():
            pts[idx] = NormalizedLandmark(x=x, y=y, z=z)
    return pts


# ==============================================================================
# 1. Eye Aspect Ratio (EAR) Tests
# ==============================================================================

def test_ear_open_vs_closed_eye() -> None:
    """Verifies that calculate_single_eye_ear differentiates open and closed eyes."""
    # LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
    # p1=33 (outer), p2=160 (top1), p3=158 (top2), p4=133 (inner), p5=153 (bot2), p6=144 (bot1)
    
    # 1. Open eye: horizontal width = 0.10, vertical heights = 0.035 each -> EAR ~ 0.35
    open_overrides = {
        33: (0.30, 0.50, 0.0),
        160: (0.33, 0.465, 0.0),
        158: (0.37, 0.465, 0.0),
        133: (0.40, 0.50, 0.0),
        153: (0.37, 0.500, 0.0),  # height = 0.035
        144: (0.33, 0.500, 0.0),  # height = 0.035
    }
    open_lms = _build_mock_landmarks(open_overrides)
    ear_open = calculate_single_eye_ear(open_lms, LEFT_EYE_INDICES, aspect_ratio=1.0)
    assert 0.30 < ear_open < 0.40, f"Expected open EAR ~0.35, got {ear_open:.3f}"

    # 2. Closed eye: horizontal width = 0.10, vertical heights = 0.002 each -> EAR ~ 0.02
    closed_overrides = {
        33: (0.30, 0.50, 0.0),
        160: (0.33, 0.499, 0.0),
        158: (0.37, 0.499, 0.0),
        133: (0.40, 0.50, 0.0),
        153: (0.37, 0.501, 0.0),
        144: (0.33, 0.501, 0.0),
    }
    closed_lms = _build_mock_landmarks(closed_overrides)
    ear_closed = calculate_single_eye_ear(closed_lms, LEFT_EYE_INDICES, aspect_ratio=1.0)
    assert ear_closed < 0.08, f"Expected closed EAR < 0.08, got {ear_closed:.3f}"


def test_ear_insufficient_landmarks_returns_zero() -> None:
    """Verifies that an empty or short landmark sequence returns 0.0 safely."""
    assert calculate_single_eye_ear([], LEFT_EYE_INDICES) == 0.0
    assert calculate_single_eye_ear([NormalizedLandmark(0.5, 0.5, 0.0)] * 50, LEFT_EYE_INDICES) == 0.0


def test_calculate_yaw_weighted_ear() -> None:
    """Verifies perspective weighting shifts between eyes based on yaw angle."""
    left_val, right_val = 0.20, 0.36
    
    # Neutral forward: 50/50 balance
    neutral_ear = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=0.0)
    assert math.isclose(neutral_ear, 0.28, abs_tol=1e-4)

    # Looking left (yaw > +15°): right eye faces camera, weighted 75%
    looking_left = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=20.0)
    assert math.isclose(looking_left, 0.25 * left_val + 0.75 * right_val, abs_tol=1e-4)
    assert looking_left > neutral_ear

    # Looking right (yaw < -15°): left eye faces camera, weighted 75%
    looking_right = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=-20.0)
    assert math.isclose(looking_right, 0.75 * left_val + 0.25 * right_val, abs_tol=1e-4)
    assert looking_right < neutral_ear


def test_calculate_pitch_compensated_ear() -> None:
    """Verifies downward pitch boosts EAR to compensate for eyelid foreshortening."""
    base_ear = 0.22
    
    # Neutral pitch (0°): no compensation
    comp_neutral = calculate_pitch_compensated_ear(base_ear, pitch_deg=0.0)
    assert comp_neutral == base_ear

    # Downward glance (pitch = -12°): boosts EAR
    comp_down = calculate_pitch_compensated_ear(base_ear, pitch_deg=-12.0)
    assert comp_down > base_ear
    # Cap at 18% boost
    comp_extreme_down = calculate_pitch_compensated_ear(base_ear, pitch_deg=-45.0)
    assert comp_extreme_down <= base_ear * 1.181


def test_extract_eye_aspect_ratio_pipeline() -> None:
    """Verifies the unified extract_eye_aspect_ratio function handles degenerate and valid inputs."""
    lms = _build_mock_landmarks()
    res = extract_eye_aspect_ratio(lms, yaw_deg=0.0, pitch_deg=0.0)
    assert isinstance(res, EyeAspectRatioResult)
    # Degenerate eye landmarks are marked invalid and excluded from closure
    assert res.is_valid is False
    assert res.is_closed is False


# ==============================================================================
# 2. Head Pose & Zone Classification Tests
# ==============================================================================

def test_classify_pose_zone() -> None:
    """Verifies cockpit zone boundaries based on pitch, yaw, and neutral offsets."""
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0) == PoseZone.CENTER
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=25.0, roll_deg=0.0) == PoseZone.LEFT_MIRROR
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=-25.0, roll_deg=0.0) == PoseZone.RIGHT_MIRROR
    assert classify_pose_zone(pitch_deg=-12.0, yaw_deg=0.0, roll_deg=0.0) == PoseZone.DASHBOARD
    # Preserves natural coupled neck roll (38°) during mirror glances without tripping OOB
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=40.0, roll_deg=38.0) == PoseZone.LEFT_MIRROR
    # Extreme rotation triggers OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=45.0, yaw_deg=0.0, roll_deg=0.0) == PoseZone.OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=55.0, roll_deg=0.0) == PoseZone.OUT_OF_BOUNDS
    # Relative offset handling (e.g. camera resting yaw at -10°)
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=5.0, neutral_yaw_deg=-10.0) == PoseZone.LEFT_MIRROR


def test_estimate_head_pose_insufficient_landmarks() -> None:
    """Verifies that missing landmarks return an invalid HeadPoseResult."""
    res = estimate_head_pose([], frame_width=1280, frame_height=720)
    assert res.is_valid is False
    assert res.zone == PoseZone.OUT_OF_BOUNDS
    assert res.is_looking_away is True


def test_estimate_head_pose_synthetic_poses() -> None:
    """
    Project canonical 3D face model points with various rotations (frontal, pitch down,
    yaw left, yaw right) to produce synthetic 2D image points and verify solvePnP recovers
    the expected Euler angles and PoseZones.
    """
    frame_w, frame_h = 1280, 720
    focal = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    cam_mat = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    tvec_gt = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

    def _project_and_estimate(rvec: np.ndarray) -> HeadPoseResult:
        proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec_gt, cam_mat, dist)
        proj = proj.reshape(-1, 2)
        overrides = {
            idx: (float(proj[i, 0] / frame_w), float(proj[i, 1] / frame_h), 0.0)
            for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES)
        }
        lms = _build_mock_landmarks(overrides)
        return estimate_head_pose(lms, frame_width=frame_w, frame_height=frame_h)

    # 1. Frontal face (0°, 0°, 0°)
    res_front = _project_and_estimate(np.zeros((3, 1), dtype=np.float64))
    assert res_front.is_valid is True
    assert abs(res_front.pitch_deg) < 3.0
    assert abs(res_front.yaw_deg) < 3.0
    assert abs(res_front.roll_deg) < 3.0
    assert res_front.zone == PoseZone.CENTER

    # 2. Pitch down -15° (looking at dashboard)
    rvec_pitch_down, _ = cv2.Rodrigues(np.array([np.radians(-15.0), 0.0, 0.0]))
    res_pitch = _project_and_estimate(rvec_pitch_down)
    assert res_pitch.is_valid is True
    assert -18.0 <= res_pitch.pitch_deg <= -12.0
    assert res_pitch.zone == PoseZone.DASHBOARD

    # 3. Yaw left +20° (looking at left mirror)
    rvec_yaw_left, _ = cv2.Rodrigues(np.array([0.0, np.radians(20.0), 0.0]))
    res_yaw_left = _project_and_estimate(rvec_yaw_left)
    assert res_yaw_left.is_valid is True
    assert 17.0 <= res_yaw_left.yaw_deg <= 23.0
    assert res_yaw_left.zone == PoseZone.LEFT_MIRROR

    # 4. Yaw right -20° (looking at right mirror)
    rvec_yaw_right, _ = cv2.Rodrigues(np.array([0.0, np.radians(-20.0), 0.0]))
    res_yaw_right = _project_and_estimate(rvec_yaw_right)
    assert res_yaw_right.is_valid is True
    assert -23.0 <= res_yaw_right.yaw_deg <= -17.0
    assert res_yaw_right.zone == PoseZone.RIGHT_MIRROR


# ==============================================================================
# 3. Mouth Aspect Ratio (MAR) & Yawn Detection Tests
# ==============================================================================

def test_mar_closed_vs_open() -> None:
    """Verifies that calculate_mar distinguishes closed and wide-open mouth."""
    # Closed mouth
    closed_overrides = {
        61: (0.42, 0.70, 0.0),   # Left corner
        291: (0.58, 0.70, 0.0),  # Right corner (width = 0.16)
        13: (0.50, 0.695, 0.0),  # Top
        14: (0.50, 0.705, 0.0),  # Bottom (height = 0.01)
        81: (0.46, 0.695, 0.0),
        178: (0.46, 0.705, 0.0),
        311: (0.54, 0.695, 0.0),
        402: (0.54, 0.705, 0.0),
    }
    closed_lms = _build_mock_landmarks(closed_overrides)
    mar_closed = calculate_mar(closed_lms, aspect_ratio=1.0)
    assert mar_closed < 0.20, f"Expected closed MAR < 0.20, got {mar_closed:.3f}"

    # Wide-open mouth (yawn)
    open_overrides = {
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),  # width = 0.16
        13: (0.50, 0.64, 0.0),
        14: (0.50, 0.76, 0.0),   # height = 0.12
        81: (0.46, 0.64, 0.0),
        178: (0.46, 0.76, 0.0),
        311: (0.54, 0.64, 0.0),
        402: (0.54, 0.76, 0.0),
    }
    open_lms = _build_mock_landmarks(open_overrides)
    mar_open = calculate_mar(open_lms, aspect_ratio=1.0)
    assert mar_open > 0.65, f"Expected open MAR > 0.65, got {mar_open:.3f}"


def test_yawn_detector_speech_rejection_vs_authentic_yawn() -> None:
    """Verifies that brief speech openings (< 1.5s) are rejected while authentic yawns (>= 1.5s) trigger."""
    detector = YawnDetector(
        mar_open_threshold=0.55,
        min_yawn_duration_ms=1500.0,
    )

    open_overrides = {
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.64, 0.0),
        14: (0.50, 0.76, 0.0),
        81: (0.46, 0.64, 0.0),
        178: (0.46, 0.76, 0.0),
        311: (0.54, 0.64, 0.0),
        402: (0.54, 0.76, 0.0),
    }
    closed_overrides = {
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.70, 0.0),
        14: (0.50, 0.70, 0.0),
        81: (0.46, 0.70, 0.0),
        178: (0.46, 0.70, 0.0),
        311: (0.54, 0.70, 0.0),
        402: (0.54, 0.70, 0.0),
    }

    open_lms = _build_mock_landmarks(open_overrides)
    closed_lms = _build_mock_landmarks(closed_overrides)

    # 1. Brief speech burst: mouth open for 600 ms (t=1000 to t=1600)
    detector.update(open_lms, timestamp_ms=1000.0, aspect_ratio=1.0)
    detector.update(open_lms, timestamp_ms=1300.0, aspect_ratio=1.0)
    detector.update(open_lms, timestamp_ms=1600.0, aspect_ratio=1.0)
    res_speech_end = detector.update(closed_lms, timestamp_ms=1633.0, aspect_ratio=1.0)

    assert res_speech_end.total_yawn_count == 0
    assert res_speech_end.is_authentic_yawn is False
    assert res_speech_end.completed_yawn is None

    # 2. Authentic yawn: mouth open for 2000 ms (t=2000 to t=4000)
    for t in range(2000, 4001, 100):
        res_during = detector.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0)
        assert res_during.is_mouth_open is True
        if t >= 2750:
            assert res_during.is_yawn_in_progress is True
        if t >= 3500:
            assert res_during.is_authentic_yawn is True

    res_yawn_end = detector.update(closed_lms, timestamp_ms=4100.0, aspect_ratio=1.0)
    assert res_yawn_end.total_yawn_count == 1
    assert res_yawn_end.completed_yawn is not None
    assert res_yawn_end.completed_yawn.duration_ms >= 1500.0


# ==============================================================================
# 4. Blink Kinematics & Sub-Frame Event Detection Tests
# ==============================================================================

def test_blink_kinematics_normal_blink() -> None:
    """Verifies that a normal snappy blink (150 ms) is accurately classified with RCR."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.23,
        min_blink_duration_ms=60.0,
        prolonged_duration_ms=350.0,
    )

    # t=1000ms: eye open (EAR = 0.32)
    s1 = detector.update(timestamp_ms=1000.0, left_ear=0.32, right_ear=0.32, compensated_ear=0.32)
    assert s1.state == BlinkState.OPEN

    # t=1033ms: eye closing (EAR = 0.18 -> crossed 0.20)
    s2 = detector.update(timestamp_ms=1033.0, left_ear=0.18, right_ear=0.18, compensated_ear=0.18)
    assert s2.is_eye_closed is True

    # t=1066ms: eye closed minimum (EAR = 0.08)
    s3 = detector.update(timestamp_ms=1066.0, left_ear=0.08, right_ear=0.08, compensated_ear=0.08)
    assert s3.state == BlinkState.CLOSED

    # t=1100ms: eye reopening (EAR = 0.16)
    s4 = detector.update(timestamp_ms=1100.0, left_ear=0.16, right_ear=0.16, compensated_ear=0.16)

    # t=1133ms: eye reopened above reopen_threshold (EAR = 0.28)
    s5 = detector.update(timestamp_ms=1133.0, left_ear=0.28, right_ear=0.28, compensated_ear=0.28)
    assert s5.state == BlinkState.OPEN
    assert s5.total_blink_count == 1
    assert s5.completed_blink is not None
    assert s5.completed_blink.classification == BlinkClassification.NORMAL
    assert 80.0 <= s5.completed_blink.duration_ms <= 160.0
    assert s5.completed_blink.rcr > 0.0


def test_blink_prolonged_closure_and_microsleep() -> None:
    """Verifies that prolonged closures transition to PROLONGED_CLOSURE and MICROSLEEP."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.23,
        prolonged_duration_ms=350.0,
        microsleep_duration_ms=800.0,
    )

    detector.update(timestamp_ms=1000.0, left_ear=0.32, right_ear=0.32, compensated_ear=0.32)
    
    # Eyelids close at t=1030ms
    detector.update(timestamp_ms=1030.0, left_ear=0.10, right_ear=0.10, compensated_ear=0.10)

    # Remain closed for 900 ms
    s_prolonged = None
    for t in range(1060, 1950, 50):
        s = detector.update(timestamp_ms=float(t), left_ear=0.05, right_ear=0.05, compensated_ear=0.05)
        if t >= 1400:
            assert s.state == BlinkState.PROLONGED_CLOSURE
        s_prolonged = s

    assert s_prolonged.current_closure_duration_ms >= 800.0

    # Finally reopen at t=2000ms
    s_reopen = detector.update(timestamp_ms=2000.0, left_ear=0.30, right_ear=0.30, compensated_ear=0.30)
    assert s_reopen.completed_blink is not None
    assert s_reopen.completed_blink.classification == BlinkClassification.MICROSLEEP_CANDIDATE
    assert s_reopen.prolonged_blink_count == 1


def test_blink_gap_invalidation() -> None:
    """Verifies that a tracking gap (> 250ms) during eye closure invalidates the blink to UNCERTAIN_GAP."""
    detector = BlinkKinematicsDetector(max_sampling_gap_ms=250.0)

    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, compensated_ear=0.30)
    detector.update(timestamp_ms=1030.0, left_ear=0.10, right_ear=0.10, compensated_ear=0.10)

    # 400 ms jump while closed
    s_gap = detector.update(timestamp_ms=1430.0, left_ear=0.10, right_ear=0.10, compensated_ear=0.10)
    assert s_gap.state == BlinkState.UNCERTAIN_GAP


# ==============================================================================
# 5. PERCLOS Accumulator Tests
# ==============================================================================

def test_perclos_validation() -> None:
    """Verifies that invalid configuration parameters raise ValueError."""
    with pytest.raises(ValueError, match="window_sec must be positive"):
        PerclosCalculator(window_sec=0.0)
    with pytest.raises(ValueError, match="min_coverage_pct must be in"):
        PerclosCalculator(min_coverage_pct=0.0)
    with pytest.raises(ValueError, match="min_coverage_pct must be in"):
        PerclosCalculator(min_coverage_pct=105.0)


def test_perclos_60s_sliding_window() -> None:
    """Verifies PERCLOS accumulation over 60 seconds with reliable coverage."""
    perclos = PerclosCalculator(window_sec=60.0, min_coverage_pct=50.0)

    # Simulate 60 seconds at 30 FPS (~33.3ms per frame)
    # Eyes open for first 48 seconds, eyes closed for last 12 seconds (20% PERCLOS)
    current_time_ms = 1000.0
    for frame_idx in range(1800):  # 1800 frames * 33.33ms ~ 60s
        current_time_ms += 33.333
        is_closed = frame_idx >= 1440  # Last 360 frames (12s) closed
        res = perclos.update(timestamp_ms=current_time_ms, is_closed=is_closed, has_face=True)

    assert res.is_reliable is True
    assert 18.0 <= res.perclos_pct <= 22.0
    assert res.coverage_pct >= 90.0


def test_perclos_insufficient_coverage_unreliable() -> None:
    """Verifies that low observation coverage (< 50%) flags is_reliable as False."""
    perclos = PerclosCalculator(window_sec=60.0, min_coverage_pct=50.0)

    # Provide only 10 seconds of data in a 60-second window
    current_time_ms = 1000.0
    res = None
    for _ in range(300):  # 10 seconds
        current_time_ms += 33.333
        res = perclos.update(timestamp_ms=current_time_ms, is_closed=True, has_face=True)

    assert res is not None
    # 10s of data in 60s window is ~16.6% coverage (< 50%)
    assert res.is_reliable is False
    assert res.coverage_pct < 50.0


# ==============================================================================
# 6. Unified DriverFeatureExtractor Pipeline Tests
# ==============================================================================

def test_feature_extractor_pipeline_face_detected() -> None:
    """Verifies DriverFeatureExtractor coordinates all features for a detected face."""
    extractor = DriverFeatureExtractor()

    # Create synthetic frontal face landmarks
    overrides = {
        # solvePnP anchor points
        1: (0.50, 0.45, 0.0),
        152: (0.50, 0.70, 0.0),
        33: (0.35, 0.40, 0.0),
        263: (0.65, 0.40, 0.0),
        61: (0.42, 0.60, 0.0),
        291: (0.58, 0.60, 0.0),
    }
    lms = tuple(_build_mock_landmarks(overrides))

    mock_result = FaceLandmarkResult(
        has_face=True,
        landmarks=lms,
        presence_confidence=0.95,
        timestamp_ms=1000,
        status=FaceLandmarkStatus.TRACKING,
        inference_latency_ms=15.0,
        reasons=(),
    )

    features = extractor.extract(mock_result, frame_width=1280, frame_height=720)
    assert isinstance(features, DriverFeatures)
    assert features.has_face is True
    assert features.face_status == FaceLandmarkStatus.TRACKING
    assert isinstance(features.ear, EyeAspectRatioResult)
    assert isinstance(features.head_pose, HeadPoseResult)
    assert isinstance(features.mar, MouthAspectRatioResult)
    assert isinstance(features.blink, BlinkSnapshot)
    assert isinstance(features.perclos, PerclosResult)
    assert "EAR:" in features.summary_text


def test_feature_extractor_pipeline_no_face() -> None:
    """Verifies DriverFeatureExtractor handles lost face gracefully without crashing."""
    extractor = DriverFeatureExtractor()

    mock_result = FaceLandmarkResult(
        has_face=False,
        landmarks=(),
        presence_confidence=0.0,
        timestamp_ms=1000,
        status=FaceLandmarkStatus.INSUFFICIENT_DATA,
        inference_latency_ms=0.0,
        reasons=("NO_FACE_DETECTED",),
    )

    features = extractor.extract(mock_result, frame_width=1280, frame_height=720)
    assert features.has_face is False
    assert features.face_status == FaceLandmarkStatus.INSUFFICIENT_DATA
    assert features.ear.is_closed is False  # Does not trigger false closure on lost face
    assert features.head_pose.is_valid is False
    assert "FACE: INSUFFICIENT_DATA" in features.summary_text
