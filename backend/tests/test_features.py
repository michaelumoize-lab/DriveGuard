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
    calculate_single_eye_ear_with_validity,
    calculate_yaw_weighted_ear,
    extract_eye_aspect_ratio,
)
from backend.app.features.feature_extractor import (
    DriverFeatureExtractor,
    DriverFeatures,
)
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    MANDIBLE_SOLVEPNP_INDICES,
    RIGID_SOLVEPNP_INDICES,
    SOLVEPNP_LANDMARK_INDICES,
    HeadPoseResult,
    PoseZone,
    build_neutral_matrix,
    classify_pose_zone,
    estimate_head_pose,
    extract_euler_yxz,
    is_valid_rotation_matrix,
    validate_pnp_solution,
)
from backend.app.features.mar import (
    DEFAULT_MAR_HOLD_THRESHOLD,
    DEFAULT_MAR_OPEN_THRESHOLD,
    DEFAULT_MIN_YAWN_DURATION_MS,
    MOUTH_OUTER_CORNERS,
    MOUTH_VERTICAL_PAIRS,
    MouthAspectRatioResult,
    YawnDetector,
    YawnEvent,
    calculate_mar,
    calculate_mar_with_validity,
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
    """Verifies C1-continuous smoothstep perspective weighting shifts between eyes based on yaw angle."""
    left_val, right_val = 0.20, 0.36

    # 1. Neutral forward (yaw = 0°): exact 50/50 balance (u = 0.5, S(u) = 0.5)
    neutral_ear = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=0.0)
    assert math.isclose(neutral_ear, 0.28, abs_tol=1e-4)

    # 2. Smoothstep at yaw = +15° (u = 0.75, S(0.75) = 0.84375):
    # w_left = 0.20 + 0.60 * 0.84375 = 0.70625, w_right = 0.29375
    ramp_15 = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=15.0)
    expected_15 = 0.70625 * left_val + 0.29375 * right_val
    assert math.isclose(ramp_15, expected_15, abs_tol=1e-4)
    assert ramp_15 < neutral_ear

    # 3. Smoothstep at yaw = +20° (u = 5/6, S(5/6) = 100/108):
    # w_left = 0.20 + 0.60 * (100/108) ~ 0.75556, w_right ~ 0.24444
    w_left_20 = 0.20 + 0.60 * (100.0 / 108.0)
    ramp_20 = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=20.0)
    assert math.isclose(ramp_20, w_left_20 * left_val + (1.0 - w_left_20) * right_val, abs_tol=1e-4)

    # 4. C1-Continuous boundary clamping at +30° and extreme +45°: exactly 80% left, 20% right
    ramp_30 = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=30.0)
    ramp_45 = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=45.0)
    expected_extreme_left = 0.80 * left_val + 0.20 * right_val
    assert math.isclose(ramp_30, expected_extreme_left, abs_tol=1e-4)
    assert math.isclose(ramp_45, expected_extreme_left, abs_tol=1e-4)

    # 5. Smoothstep right at yaw = -20°: w_left ~ 0.24444, w_right ~ 0.75556
    ramp_neg20 = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=-20.0)
    assert math.isclose(ramp_neg20, (1.0 - w_left_20) * left_val + w_left_20 * right_val, abs_tol=1e-4)
    assert ramp_neg20 > neutral_ear

    # 6. Unilateral occlusion fallback: valid eye gets 100% weight, never halved!
    only_left = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=0.0, left_valid=True, right_valid=False)
    assert math.isclose(only_left, left_val, abs_tol=1e-4)

    only_right = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=25.0, left_valid=False, right_valid=True)
    assert math.isclose(only_right, right_val, abs_tol=1e-4)

    # 7. Neither valid
    assert calculate_yaw_weighted_ear(left_val, right_val, 0.0, False, False) == 0.0

    # 8. Mirrored camera mode (is_mirrored=True):
    # In mirrored mode, turning left (+yaw) favors right_ear (camera-facing eye on image right)
    ramp_mirrored_left = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=20.0, is_mirrored=True)
    assert math.isclose(ramp_mirrored_left, (1.0 - w_left_20) * left_val + w_left_20 * right_val, abs_tol=1e-4)

    ramp_mirrored_right = calculate_yaw_weighted_ear(left_val, right_val, yaw_deg=-20.0, is_mirrored=True)
    assert math.isclose(ramp_mirrored_right, w_left_20 * left_val + (1.0 - w_left_20) * right_val, abs_tol=1e-4)


def test_calculate_pitch_compensated_ear() -> None:
    """Verifies downward pitch boosts open eyes but NEVER boosts closed eyes."""
    base_ear = 0.22

    # Neutral pitch (0°): no compensation
    comp_neutral = calculate_pitch_compensated_ear(base_ear, pitch_deg=0.0)
    assert comp_neutral == base_ear

    # Downward glance (pitch = -12°): boosts open EAR
    comp_down = calculate_pitch_compensated_ear(base_ear, pitch_deg=-12.0)
    assert comp_down > base_ear
    # Cap at 18% boost
    comp_extreme_down = calculate_pitch_compensated_ear(base_ear, pitch_deg=-45.0)
    assert comp_extreme_down <= base_ear * 1.181

    # CRITICAL SAFEGUARD: Closed eye (base_ear <= closure_threshold 0.20) is NEVER boosted!
    closed_ear = 0.08
    comp_closed = calculate_pitch_compensated_ear(closed_ear, pitch_deg=-25.0, closure_threshold=0.20)
    assert comp_closed == closed_ear, "Closed eye must never be boosted during downward pitch!"

    near_threshold_ear = 0.19
    comp_near = calculate_pitch_compensated_ear(near_threshold_ear, pitch_deg=-25.0, closure_threshold=0.20)
    assert comp_near == near_threshold_ear, "Drooping eye near threshold must never be boosted above threshold!"


def test_extract_eye_aspect_ratio_pipeline() -> None:
    """Verifies the unified extract_eye_aspect_ratio function handles degenerate and valid inputs."""
    lms = _build_mock_landmarks()
    res = extract_eye_aspect_ratio(lms, yaw_deg=0.0, pitch_deg=0.0)
    assert isinstance(res, EyeAspectRatioResult)
    # Degenerate eye landmarks are marked invalid and excluded from closure
    assert res.is_valid is False
    assert res.is_closed is False
    assert res.is_left_valid is False
    assert res.is_right_valid is False


def test_ear_asymmetric_closure_and_occlusion() -> None:
    """
    Verifies independent eye validation, unilateral occlusion handling,
    and binocular vs wink discrimination.
    """
    # Open eye pattern (EAR ~ 0.35)
    open_left = {
        33: (0.30, 0.50, 0.0),
        160: (0.33, 0.465, 0.0),
        158: (0.37, 0.465, 0.0),
        133: (0.40, 0.50, 0.0),
        153: (0.37, 0.500, 0.0),
        144: (0.33, 0.500, 0.0),
    }
    open_right = {
        362: (0.60, 0.50, 0.0),
        385: (0.63, 0.465, 0.0),
        387: (0.67, 0.465, 0.0),
        263: (0.70, 0.50, 0.0),
        373: (0.67, 0.500, 0.0),
        380: (0.63, 0.500, 0.0),
    }
    # Closed eye pattern (EAR ~ 0.02)
    closed_left = {
        33: (0.30, 0.50, 0.0),
        160: (0.33, 0.499, 0.0),
        158: (0.37, 0.499, 0.0),
        133: (0.40, 0.50, 0.0),
        153: (0.37, 0.501, 0.0),
        144: (0.33, 0.501, 0.0),
    }
    closed_right = {
        362: (0.60, 0.50, 0.0),
        385: (0.63, 0.499, 0.0),
        387: (0.67, 0.499, 0.0),
        263: (0.70, 0.50, 0.0),
        373: (0.67, 0.501, 0.0),
        380: (0.63, 0.501, 0.0),
    }

    # 1. Unilateral Right Occlusion (Left eye open, Right eye occluded/collapsed)
    lms_right_occluded = _build_mock_landmarks(open_left)
    res_right_occ = extract_eye_aspect_ratio(lms_right_occluded, aspect_ratio=1.0)
    assert res_right_occ.is_valid is True
    assert res_right_occ.is_left_valid is True
    assert res_right_occ.is_right_valid is False
    assert res_right_occ.is_left_closed is False
    assert res_right_occ.is_closed is False  # Single valid eye is open -> overall open
    assert math.isclose(res_right_occ.weighted_ear, res_right_occ.left_ear, abs_tol=1e-4)  # 100% weight, NOT halved!

    # 2. Unilateral Left Occlusion (Right eye closed, Left eye occluded/collapsed)
    lms_left_occluded = _build_mock_landmarks(closed_right)
    res_left_occ = extract_eye_aspect_ratio(lms_left_occluded, aspect_ratio=1.0)
    assert res_left_occ.is_valid is True
    assert res_left_occ.is_left_valid is False
    assert res_left_occ.is_right_valid is True
    assert res_left_occ.is_right_closed is True
    assert res_left_occ.is_closed is True  # Single valid eye is closed -> closure detected!
    assert math.isclose(res_left_occ.weighted_ear, res_left_occ.right_ear, abs_tol=1e-4)

    # 3. Wink Discrimination (Left eye closed, Right eye open)
    lms_wink = _build_mock_landmarks({**closed_left, **open_right})
    res_wink = extract_eye_aspect_ratio(lms_wink, aspect_ratio=1.0)
    assert res_wink.is_valid is True
    assert res_wink.is_left_closed is True
    assert res_wink.is_right_closed is False
    assert res_wink.is_closed is False  # Binocular closure requires BOTH eyes closed

    # 4. Bilateral Genuine Closure (Both eyes closed)
    lms_closed = _build_mock_landmarks({**closed_left, **closed_right})
    res_closed = extract_eye_aspect_ratio(lms_closed, aspect_ratio=1.0)
    assert res_closed.is_valid is True
    assert res_closed.is_left_closed is True
    assert res_closed.is_right_closed is True
    assert res_closed.is_closed is True


def test_ear_downward_head_nod_genuine_closure_safety() -> None:
    """Verifies downward head pitch during genuine eye closure cannot mask closure."""
    closed_left = {
        33: (0.30, 0.50, 0.0),
        160: (0.33, 0.499, 0.0),
        158: (0.37, 0.499, 0.0),
        133: (0.40, 0.50, 0.0),
        153: (0.37, 0.501, 0.0),
        144: (0.33, 0.501, 0.0),
    }
    closed_right = {
        362: (0.60, 0.50, 0.0),
        385: (0.63, 0.499, 0.0),
        387: (0.67, 0.499, 0.0),
        263: (0.70, 0.50, 0.0),
        373: (0.67, 0.501, 0.0),
        380: (0.63, 0.501, 0.0),
    }
    lms_nod = _build_mock_landmarks({**closed_left, **closed_right})
    # Severe downward pitch: -25° (nodding off at wheel)
    res_nod = extract_eye_aspect_ratio(lms_nod, pitch_deg=-25.0, aspect_ratio=1.0)
    assert res_nod.is_closed is True
    assert res_nod.is_left_closed is True
    assert res_nod.is_right_closed is True
    assert res_nod.compensated_ear == res_nod.weighted_ear  # Safe: not boosted


def test_ear_coordinate_contracts_normalized_vs_pixel() -> None:
    """Verifies explicit normalized vs. pixel coordinate handling."""
    # Pixel coordinates: 1280x720 space
    pts_pixel = np.zeros((478, 2), dtype=np.float32)
    # Left eye: width = 100 pixels, vertical heights = 35 pixels each -> EAR = (35+35)/(2*100) = 0.35
    pts_pixel[33] = [384.0, 360.0]
    pts_pixel[160] = [410.0, 325.0]
    pts_pixel[158] = [450.0, 325.0]
    pts_pixel[133] = [484.0, 360.0]
    pts_pixel[153] = [450.0, 360.0]
    pts_pixel[144] = [410.0, 360.0]

    # Explicit pixel mode (is_normalized=False): aspect_ratio is NOT applied to pixel distances
    ear_pixel_explicit = calculate_single_eye_ear(pts_pixel, LEFT_EYE_INDICES, aspect_ratio=16/9, is_normalized=False)
    assert 0.30 < ear_pixel_explicit < 0.40

    # Auto-detected pixel mode (is_normalized=None, coords > 2.0)
    ear_pixel_auto = calculate_single_eye_ear(pts_pixel, LEFT_EYE_INDICES, aspect_ratio=16/9, is_normalized=None)
    assert math.isclose(ear_pixel_explicit, ear_pixel_auto, abs_tol=1e-4)


def test_ear_malformed_and_non_finite_landmarks() -> None:
    """Verifies that NaN, Inf, and malformed coordinates are safely rejected."""
    # 1. NaN coordinate
    nan_lms = _build_mock_landmarks({33: (float("nan"), 0.5, 0.0)})
    res_nan = extract_eye_aspect_ratio(nan_lms)
    assert res_nan.is_left_valid is False
    assert res_nan.left_ear == 0.0

    # 2. Inf coordinate
    inf_lms = _build_mock_landmarks({160: (0.33, float("inf"), 0.0)})
    res_inf = extract_eye_aspect_ratio(inf_lms)
    assert res_inf.is_left_valid is False
    assert res_inf.left_ear == 0.0

    # 3. Very low EAR (genuine closed eye): MUST BE VALID and CLOSED!
    tiny_ear_lms = _build_mock_landmarks({
        33: (0.30, 0.50, 0.0),
        160: (0.33, 0.4995, 0.0),
        158: (0.37, 0.4995, 0.0),
        133: (0.40, 0.50, 0.0),
        153: (0.37, 0.5005, 0.0),
        144: (0.33, 0.5005, 0.0),
    })
    ear_val, is_val = calculate_single_eye_ear_with_validity(tiny_ear_lms, LEFT_EYE_INDICES, aspect_ratio=1.0)
    assert is_val is True
    assert 0.0 < ear_val < 0.03


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
    """Verifies that missing landmarks return an invalid HeadPoseResult with PoseZone.UNKNOWN without flagging looking_away."""
    res = estimate_head_pose([], frame_width=1280, frame_height=720)
    assert res.is_valid is False
    assert res.zone == PoseZone.UNKNOWN
    assert res.is_looking_away is False  # Tracking loss is NOT looking away


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

    def _project_and_estimate(rvec: np.ndarray, is_mirrored: bool = False) -> HeadPoseResult:
        proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec_gt, cam_mat, dist)
        proj = proj.reshape(-1, 2)
        overrides = {
            idx: (float(proj[i, 0] / frame_w), float(proj[i, 1] / frame_h), 0.0)
            for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES)
        }
        lms = _build_mock_landmarks(overrides)
        return estimate_head_pose(lms, frame_width=frame_w, frame_height=frame_h, is_mirrored=is_mirrored)

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

    # 3. Unmirrored optical projection:
    # Turning to driver's LEFT moves face to camera right (+X in 2D image, corresponding to -Y rotation).
    rvec_unmir_left, _ = cv2.Rodrigues(np.array([0.0, np.radians(-20.0), 0.0]))
    res_unmir_left = _project_and_estimate(rvec_unmir_left, is_mirrored=False)
    assert res_unmir_left.is_valid is True
    assert 17.0 <= res_unmir_left.yaw_deg <= 23.0
    assert res_unmir_left.zone == PoseZone.LEFT_MIRROR

    rvec_unmir_right, _ = cv2.Rodrigues(np.array([0.0, np.radians(20.0), 0.0]))
    res_unmir_right = _project_and_estimate(rvec_unmir_right, is_mirrored=False)
    assert res_unmir_right.is_valid is True
    assert -23.0 <= res_unmir_right.yaw_deg <= -17.0
    assert res_unmir_right.zone == PoseZone.RIGHT_MIRROR

    # 4. Mirrored camera projection (selfie mode / cv2.flip):
    # Driver turning left is horizontally flipped to image left (+Y rotation in mirrored coordinates).
    rvec_mir_left, _ = cv2.Rodrigues(np.array([0.0, np.radians(20.0), 0.0]))
    res_mir_left = _project_and_estimate(rvec_mir_left, is_mirrored=True)
    assert res_mir_left.is_valid is True
    assert 17.0 <= res_mir_left.yaw_deg <= 23.0
    assert res_mir_left.zone == PoseZone.LEFT_MIRROR

    rvec_mir_right, _ = cv2.Rodrigues(np.array([0.0, np.radians(-20.0), 0.0]))
    res_mir_right = _project_and_estimate(rvec_mir_right, is_mirrored=True)
    assert res_mir_right.is_valid is True
    assert -23.0 <= res_mir_right.yaw_deg <= -17.0
    assert res_mir_right.zone == PoseZone.RIGHT_MIRROR


def test_head_pose_euler_continuity_full_yaw_sweep() -> None:
    """
    Verifies that intrinsic Y-X-Z Euler extraction produces continuous, monotonic yaw values
    across [-45°, +45°] without Euler branch flips, 160° glitches, or Gimbal lock discontinuities.
    """
    frame_w, frame_h = 1280, 720
    focal = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    cam_mat = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

    for is_mirrored in (False, True):
        for yaw_deg in np.linspace(-45.0, 45.0, 91):
            cam_yaw = yaw_deg if is_mirrored else -yaw_deg
            rvec, _ = cv2.Rodrigues(np.array([0.0, np.radians(cam_yaw), 0.0]))
            proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, cam_mat, dist)
            proj = proj.reshape(-1, 2)
            overrides = {
                idx: (float(proj[i, 0] / frame_w), float(proj[i, 1] / frame_h), 0.0)
                for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES)
            }
            lms = _build_mock_landmarks(overrides)
            res = estimate_head_pose(lms, frame_width=frame_w, frame_height=frame_h, is_mirrored=is_mirrored)
            assert res.is_valid is True
            assert math.isclose(res.yaw_deg, yaw_deg, abs_tol=0.25), (
                f"Yaw mismatch at GT {yaw_deg:.1f}°: got {res.yaw_deg:.2f}° (mirrored={is_mirrored})"
            )
            assert abs(res.pitch_deg) < 0.25
            assert abs(res.roll_deg) < 0.25


def test_head_pose_non_finite_and_degenerate_landmarks() -> None:
    """Verifies that non-finite coordinates, collapsed landmarks, or degenerate bounding boxes return is_valid=False and zone=UNKNOWN."""
    # 1. NaN coordinate in landmark
    nan_lms = _build_mock_landmarks({1: (float("nan"), 0.5, 0.0)})
    res_nan = estimate_head_pose(nan_lms)
    assert res_nan.is_valid is False
    assert res_nan.zone == PoseZone.UNKNOWN
    assert res_nan.is_looking_away is False

    # 2. Inf coordinate in landmark
    inf_lms = _build_mock_landmarks({6: (0.5, float("inf"), 0.0)})
    res_inf = estimate_head_pose(inf_lms)
    assert res_inf.is_valid is False
    assert res_inf.zone == PoseZone.UNKNOWN
    assert res_inf.is_looking_away is False

    # 3. Degenerate collapsed bounding box (< 5 px span)
    collapsed_lms = _build_mock_landmarks({idx: (0.50, 0.50, 0.0) for idx in SOLVEPNP_LANDMARK_INDICES})
    res_col = estimate_head_pose(collapsed_lms)
    assert res_col.is_valid is False
    assert res_col.zone == PoseZone.UNKNOWN
    assert res_col.is_looking_away is False

    # 4. None input
    res_none = estimate_head_pose(None)
    assert res_none.is_valid is False
    assert res_none.zone == PoseZone.UNKNOWN
    assert res_none.is_looking_away is False


def test_cockpit_zone_priority_mirror_vs_dashboard() -> None:
    """
    Verifies cockpit zone classification priority:
    1. OUT_OF_BOUNDS (extreme angles)
    2. LEFT_MIRROR / RIGHT_MIRROR (takes precedence over pitch tilt)
    3. DASHBOARD (confined to forward road yaw cone)
    4. CENTER (nominal forward driving)
    """
    # 1. Left mirror check with downward glance: yaw=+25°, pitch=-12° -> LEFT_MIRROR (not DASHBOARD)
    assert classify_pose_zone(pitch_deg=-12.0, yaw_deg=25.0) == PoseZone.LEFT_MIRROR

    # 2. Right mirror check with downward glance: yaw=-25°, pitch=-12° -> RIGHT_MIRROR (not DASHBOARD)
    assert classify_pose_zone(pitch_deg=-12.0, yaw_deg=-25.0) == PoseZone.RIGHT_MIRROR

    # 3. Pure dashboard glance: yaw=0°, pitch=-12° -> DASHBOARD
    assert classify_pose_zone(pitch_deg=-12.0, yaw_deg=0.0) == PoseZone.DASHBOARD

    # 4. Dashboard glance within forward road cone: yaw=+8°, pitch=-12° -> DASHBOARD
    assert classify_pose_zone(pitch_deg=-12.0, yaw_deg=8.0) == PoseZone.DASHBOARD

    # 5. Extreme yaw (> 50°) overrides mirror and dashboard: yaw=55°, pitch=-12° -> OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=-12.0, yaw_deg=55.0) == PoseZone.OUT_OF_BOUNDS

    # 6. Extreme pitch (> 38°) overrides mirror and dashboard: yaw=25°, pitch=-40° -> OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=-40.0, yaw_deg=25.0) == PoseZone.OUT_OF_BOUNDS


def test_head_pose_neutral_matrix_calibration() -> None:
    """
    Verifies that 3D relative rotation R_rel = R_neutral.T @ R_current decouples off-axis
    camera mounting (e.g. A-pillar mount with yaw=-15°, pitch=-5°) so that a pure head nod
    does not induce false yaw deflection.
    """
    frame_w, frame_h = 1280, 720
    focal = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    cam_mat = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

    # Neutral mounting: camera observes driver resting posture at yaw=-15.0°, pitch=-5.0°
    neutral_rot = build_neutral_matrix(neutral_pitch_deg=-5.0, neutral_yaw_deg=-15.0, is_mirrored=False)
    rvec_neutral, _ = cv2.Rodrigues(neutral_rot)

    # 1. Driver in neutral resting posture
    proj_neutral, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec_neutral, tvec, cam_mat, dist)
    proj_neutral = proj_neutral.reshape(-1, 2)
    overrides_neutral = {
        idx: (float(proj_neutral[i, 0] / frame_w), float(proj_neutral[i, 1] / frame_h), 0.0)
        for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES)
    }
    lms_neutral = _build_mock_landmarks(overrides_neutral)
    res_rest = estimate_head_pose(
        lms_neutral,
        neutral_pitch_deg=-5.0,
        neutral_yaw_deg=-15.0,
        is_mirrored=False,
    )
    assert res_rest.is_valid is True
    assert math.isclose(res_rest.rel_yaw_deg, 0.0, abs_tol=0.2)
    assert math.isclose(res_rest.rel_pitch_deg, 0.0, abs_tol=0.2)
    assert res_rest.zone == PoseZone.CENTER

    # 2. Driver executes pure head nod (-15.0° pitch) from neutral posture:
    theta_nod = float(np.radians(-15.0))
    Rx_nod = np.array(
        [[1.0, 0.0, 0.0], [0.0, np.cos(theta_nod), -np.sin(theta_nod)], [0.0, np.sin(theta_nod), np.cos(theta_nod)]],
        dtype=np.float64,
    )
    R_curr = neutral_rot @ Rx_nod
    rvec_nod, _ = cv2.Rodrigues(R_curr)
    proj_nod, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec_nod, tvec, cam_mat, dist)
    proj_nod = proj_nod.reshape(-1, 2)
    overrides_nod = {
        idx: (float(proj_nod[i, 0] / frame_w), float(proj_nod[i, 1] / frame_h), 0.0)
        for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES)
    }
    lms_nod = _build_mock_landmarks(overrides_nod)
    res_nod = estimate_head_pose(
        lms_nod,
        neutral_pitch_deg=-5.0,
        neutral_yaw_deg=-15.0,
        is_mirrored=False,
    )
    assert res_nod.is_valid is True
    assert math.isclose(res_nod.rel_pitch_deg, -15.0, abs_tol=0.5)
    # Yaw MUST remain 0.0° without cross-axis bleed
    assert math.isclose(res_nod.rel_yaw_deg, 0.0, abs_tol=0.5)
    assert res_nod.zone == PoseZone.DASHBOARD


def test_head_pose_jaw_motion_pitch_resilience() -> None:
    """
    Verifies that cranial rigid landmarks (Sellion, Glabella, Eye Sockets, Nose Tip) render
    head pose estimation invariant to mandible/jaw displacement during yawning or speech.
    """
    frame_w, frame_h = 1280, 720
    focal = float(frame_w)
    center = (frame_w / 2.0, frame_h / 2.0)
    cam_mat = np.array(
        [[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist = np.zeros((4, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

    # 1. Neutral face projection
    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, np.zeros((3, 1)), tvec, cam_mat, dist)
    proj = proj.reshape(-1, 2)

    # 2. Simulate 30mm vertical jaw opening (chin 152 moves down by ~64 px, mouth corners 61 & 291 move down by ~20 px)
    proj_yawn = proj.copy()
    idx_chin = SOLVEPNP_LANDMARK_INDICES.index(152)
    idx_mouth_r = SOLVEPNP_LANDMARK_INDICES.index(61)
    idx_mouth_l = SOLVEPNP_LANDMARK_INDICES.index(291)
    proj_yawn[idx_chin, 1] += 64.0
    proj_yawn[idx_mouth_r, 1] += 20.0
    proj_yawn[idx_mouth_l, 1] += 20.0

    overrides = {
        idx: (float(proj_yawn[i, 0] / frame_w), float(proj_yawn[i, 1] / frame_h), 0.0)
        for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES)
    }
    lms_yawn = _build_mock_landmarks(overrides)
    res_yawn = estimate_head_pose(lms_yawn, frame_width=frame_w, frame_height=frame_h)

    assert res_yawn.is_valid is True
    # Invariant to jaw opening: pitch deviation must remain < 1.0° (NOT -20° or -26°)
    assert abs(res_yawn.pitch_deg) < 1.0, f"Spurious pitch on yawn: {res_yawn.pitch_deg:.2f}°"
    assert abs(res_yawn.yaw_deg) < 1.0
    assert abs(res_yawn.roll_deg) < 1.0
    assert res_yawn.zone == PoseZone.CENTER


def test_zone_transition_hysteresis() -> None:
    """
    Verifies that a 1.5° hysteresis deadband prevents zone boundary chattering
    when the driver's head hovers near boundary angles (e.g. 13.9° <-> 14.1°).
    """
    # 1. Approaching LEFT_MIRROR from CENTER
    z1 = classify_pose_zone(pitch_deg=0.0, yaw_deg=13.8, current_zone=PoseZone.CENTER)
    assert z1 == PoseZone.CENTER

    # 2. Crosses nominal threshold 14.0° into LEFT_MIRROR
    z2 = classify_pose_zone(pitch_deg=0.0, yaw_deg=14.0, current_zone=PoseZone.CENTER)
    assert z2 == PoseZone.LEFT_MIRROR

    # 3. Dips slightly to 13.9°: HYSTERESIS MUST PRESERVE LEFT_MIRROR (exit is 14.0 - 1.5 = 12.5°)
    z3 = classify_pose_zone(pitch_deg=0.0, yaw_deg=13.9, current_zone=z2)
    assert z3 == PoseZone.LEFT_MIRROR

    z4 = classify_pose_zone(pitch_deg=0.0, yaw_deg=13.0, current_zone=z3)
    assert z4 == PoseZone.LEFT_MIRROR

    z5 = classify_pose_zone(pitch_deg=0.0, yaw_deg=12.6, current_zone=z4)
    assert z5 == PoseZone.LEFT_MIRROR

    # 4. Falls below 12.5°: exits back to CENTER
    z6 = classify_pose_zone(pitch_deg=0.0, yaw_deg=12.4, current_zone=z5)
    assert z6 == PoseZone.CENTER


def test_head_pose_matrix_validation_orthogonality_and_determinant() -> None:
    """Verifies that rotation matrix validation enforces 3x3 dimension, finiteness, orthogonality, and det ~ 1."""
    # 1. Valid identity
    assert is_valid_rotation_matrix(np.eye(3)) is True

    # 2. Valid Rodrigues rotation
    rvec = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    R_valid, _ = cv2.Rodrigues(rvec)
    assert is_valid_rotation_matrix(R_valid) is True

    # 3. Non-orthogonal matrix (sheared)
    R_sheared = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    assert is_valid_rotation_matrix(R_sheared) is False

    # 4. Reflection matrix (det = -1.0)
    R_refl = np.diag([1.0, 1.0, -1.0])
    assert is_valid_rotation_matrix(R_refl) is False

    # 5. Non-finite values
    R_nan = np.eye(3)
    R_nan[0, 0] = float("nan")
    assert is_valid_rotation_matrix(R_nan) is False

    # 6. Wrong shape
    assert is_valid_rotation_matrix(np.eye(4)) is False


def test_head_pose_gimbal_lock_singularity_branch() -> None:
    """Verifies that pitch singularities near +/-90° resolve smoothly without NaN or ZeroDivisionError."""
    # Pitch = +90°: R_x(+90°)
    R_pitch_90 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ], dtype=np.float64)
    yaw, pitch, roll = extract_euler_yxz(R_pitch_90)
    assert math.isclose(pitch, 90.0, abs_tol=1e-4)
    assert math.isfinite(yaw)
    assert math.isfinite(roll)

    # Pitch = -90°: R_x(-90°)
    R_pitch_neg90 = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ], dtype=np.float64)
    yaw_neg, pitch_neg, roll_neg = extract_euler_yxz(R_pitch_neg90)
    assert math.isclose(pitch_neg, -90.0, abs_tol=1e-4)
    assert math.isfinite(yaw_neg)
    assert math.isfinite(roll_neg)


def test_face_loss_does_not_accumulate_off_road_distraction() -> None:
    """
    Verifies that tracking loss does NOT treat the driver as looking away,
    does NOT accumulate off-road dwell time, and does NOT trigger distraction warnings.
    """
    from backend.app.intelligence.decision_engine import DecisionEngine
    from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer
    from backend.app.intelligence.confidence import SignalConfidenceEvaluator
    from backend.app.intelligence.reason_codes import DrowsinessState

    extractor = DriverFeatureExtractor()
    temporal = TemporalAnalyzer()
    decision = DecisionEngine()
    conf_eval = SignalConfidenceEvaluator()

    # 1. Normal frontal face for 10 frames
    t = 1000.0
    lms_present = tuple(_build_mock_landmarks())
    for _ in range(10):
        t += 33.333
        mock_present = FaceLandmarkResult(
            has_face=True,
            landmarks=lms_present,
            presence_confidence=0.95,
            timestamp_ms=int(t),
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=12.0,
            reasons=(),
        )
        feats = extractor.extract(mock_present)
        metrics = temporal.update(feats)
        dec = decision.evaluate(metrics, conf_eval.evaluate(feats))
        assert metrics.active_off_road_dwell_ms == 0.0

    # 2. Face is lost for 2.0 seconds (e.g. driver obstructed or leaves frame)
    for _ in range(60):  # 2.0 seconds
        t += 33.333
        mock_lost = FaceLandmarkResult(
            has_face=False,
            landmarks=(),
            presence_confidence=0.0,
            timestamp_ms=int(t),
            status=FaceLandmarkStatus.INSUFFICIENT_DATA,
            inference_latency_ms=0.0,
            reasons=("NO_FACE_DETECTED",),
        )
        feats_lost = extractor.extract(mock_lost)
        # Pose must be invalid and looking_away MUST be False
        assert feats_lost.head_pose.is_valid is False
        assert feats_lost.head_pose.is_looking_away is False

        metrics_lost = temporal.update(feats_lost)
        # Off-road dwell must NOT accumulate!
        assert metrics_lost.active_off_road_dwell_ms == 0.0

        dec_lost = decision.evaluate(metrics_lost, conf_eval.evaluate(feats_lost))
        # Must not be in DROWSINESS_WARNING for persistent distraction
        assert dec_lost.state != DrowsinessState.DROWSINESS_WARNING


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


def test_mar_coordinate_contracts_normalized_vs_pixel() -> None:
    """Verifies that MAR handles normalized vs. pixel coordinates correctly without aspect-ratio distortion."""
    # 1. Normalized space: width = 0.16, vertical height = 0.08
    # With aspect_ratio = 16/9, dx is scaled by 16/9:
    # horiz = 0.16 * (16/9) ~ 0.28444
    # vert = 0.08
    # mar = 0.08 / 0.28444 ~ 0.28125
    norm_overrides = {
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.66, 0.0),
        14: (0.50, 0.74, 0.0),
        81: (0.46, 0.66, 0.0),
        178: (0.46, 0.74, 0.0),
        311: (0.54, 0.66, 0.0),
        402: (0.54, 0.74, 0.0),
    }
    norm_lms = _build_mock_landmarks(norm_overrides)
    mar_norm = calculate_mar(norm_lms, aspect_ratio=16.0 / 9.0, is_normalized=True)
    expected_norm = 0.08 / (0.16 * (16.0 / 9.0))
    assert math.isclose(mar_norm, expected_norm, abs_tol=1e-4)

    # 2. Pixel space (1280x720): width = 200 pixels, vertical height = 100 pixels
    # In pixel space, aspect_ratio must NOT scale pixel coordinates (isotropic camera pixels)
    # mar = 100 / 200 = 0.50
    pixel_pts = np.zeros((478, 2), dtype=np.float32)
    pixel_pts[61] = [540.0, 500.0]
    pixel_pts[291] = [740.0, 500.0]
    pixel_pts[13] = [640.0, 450.0]
    pixel_pts[14] = [640.0, 550.0]
    pixel_pts[81] = [590.0, 450.0]
    pixel_pts[178] = [590.0, 550.0]
    pixel_pts[311] = [690.0, 450.0]
    pixel_pts[402] = [690.0, 550.0]

    mar_pixel_explicit = calculate_mar(pixel_pts, aspect_ratio=16.0 / 9.0, is_normalized=False)
    assert math.isclose(mar_pixel_explicit, 0.50, abs_tol=1e-4)

    # 3. Auto-detection fallback (coords > 2.0 -> pixel mode)
    mar_pixel_auto = calculate_mar(pixel_pts, aspect_ratio=16.0 / 9.0, is_normalized=None)
    assert math.isclose(mar_pixel_auto, 0.50, abs_tol=1e-4)


def test_mar_invalid_and_collapsed_landmarks() -> None:
    """Verifies that non-finite, out-of-bounds, or collapsed mouth landmarks return is_valid=False."""
    # 1. Collapsed outer corners (width = 0)
    collapsed_lms = _build_mock_landmarks({
        61: (0.50, 0.70, 0.0),
        291: (0.50, 0.70, 0.0),
        13: (0.50, 0.68, 0.0),
        14: (0.50, 0.72, 0.0),
    })
    mar_val, is_val = calculate_mar_with_validity(collapsed_lms)
    assert is_val is False
    assert mar_val == 0.0

    # 2. NaN coordinate
    nan_lms = _build_mock_landmarks({61: (float("nan"), 0.70, 0.0)})
    mar_nan, is_nan = calculate_mar_with_validity(nan_lms)
    assert is_nan is False
    assert mar_nan == 0.0

    # 3. Inf coordinate
    inf_lms = _build_mock_landmarks({14: (0.50, float("inf"), 0.0)})
    mar_inf, is_inf = calculate_mar_with_validity(inf_lms)
    assert is_inf is False
    assert mar_inf == 0.0

    # 4. Insufficient landmarks (< 402 points)
    mar_short, is_short = calculate_mar_with_validity([])
    assert is_short is False
    assert mar_short == 0.0


def test_yawn_detector_hysteresis_and_fluctuations() -> None:
    """
    Verifies that dual-threshold hysteresis (0.55 open, 0.45 hold) prevents boundary jitter
    from prematurely killing an in-flight opening when MAR dips to 0.48.
    """
    detector = YawnDetector(
        mar_open_threshold=0.55,
        mar_hold_threshold=0.45,
        min_yawn_duration_ms=1500.0,
    )

    def _make_mouth_lms(vert_height: float) -> List[NormalizedLandmark]:
        # width = 0.16, vert_height determines MAR = vert_height / 0.16 (aspect_ratio = 1.0)
        return _build_mock_landmarks({
            61: (0.42, 0.70, 0.0),
            291: (0.58, 0.70, 0.0),
            13: (0.50, 0.70 - vert_height / 2.0, 0.0),
            14: (0.50, 0.70 + vert_height / 2.0, 0.0),
            81: (0.46, 0.70 - vert_height / 2.0, 0.0),
            178: (0.46, 0.70 + vert_height / 2.0, 0.0),
            311: (0.54, 0.70 - vert_height / 2.0, 0.0),
            402: (0.54, 0.70 + vert_height / 2.0, 0.0),
        })

    lms_wide = _make_mouth_lms(0.10)      # MAR ~ 0.625 (> 0.55)
    lms_flutter = _make_mouth_lms(0.077)  # MAR ~ 0.481 (between 0.45 and 0.55)
    lms_closed = _make_mouth_lms(0.01)    # MAR ~ 0.0625 (< 0.45)

    # 1. Mouth opening triggers at t=1000 with MAR=0.625 >= 0.55
    res1 = detector.update(lms_wide, timestamp_ms=1000.0, aspect_ratio=1.0)
    assert res1.is_mouth_open is True

    # 2. Advance opening in 100ms steps up to t=1400
    for t in (1100, 1200, 1300, 1400):
        detector.update(lms_wide, timestamp_ms=float(t), aspect_ratio=1.0)

    # 3. Mouth flutters to 0.48 from t=1500 to t=1800: HYSTERESIS MUST HOLD!
    for t in (1500, 1600, 1700, 1800):
        res_flut = detector.update(lms_flutter, timestamp_ms=float(t), aspect_ratio=1.0)
        assert res_flut.is_mouth_open is True, "MAR 0.48 must remain open due to 0.45 hold threshold!"

    # 4. Mouth returns to wide open from t=1900 to t=2600
    for t in range(1900, 2601, 100):
        res_wide = detector.update(lms_wide, timestamp_ms=float(t), aspect_ratio=1.0)
        assert res_wide.is_mouth_open is True

    # 5. Mouth closes below 0.45 at t=2700 (duration = 1600ms >= 1500ms)
    res_end = detector.update(lms_closed, timestamp_ms=2700.0, aspect_ratio=1.0)
    assert res_end.is_mouth_open is False
    assert res_end.total_yawn_count == 1
    assert res_end.completed_yawn is not None
    assert res_end.completed_yawn.duration_ms >= 1500.0


def test_yawn_detector_tracking_dropout_grace_vs_prolonged_gap() -> None:
    """
    Verifies that brief tracking loss (<= 200ms) preserves candidate without adding unobserved time,
    while prolonged loss (> 200ms) or sampling gaps (> 500ms) cancel in-flight opening.
    """
    detector = YawnDetector(
        mar_open_threshold=0.55,
        mar_hold_threshold=0.45,
        min_yawn_duration_ms=1500.0,
        dropout_grace_ms=200.0,
    )

    open_lms = _build_mock_landmarks({
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.64, 0.0),
        14: (0.50, 0.76, 0.0),
        81: (0.46, 0.64, 0.0),
        178: (0.46, 0.76, 0.0),
        311: (0.54, 0.64, 0.0),
        402: (0.54, 0.76, 0.0),
    })
    closed_lms = _build_mock_landmarks({
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.70, 0.0),
        14: (0.50, 0.70, 0.0),
        81: (0.46, 0.70, 0.0),
        178: (0.46, 0.70, 0.0),
        311: (0.54, 0.70, 0.0),
        402: (0.54, 0.70, 0.0),
    })

    # Case A: Brief tracking loss (100 ms <= 200 ms grace)
    # Open for 1000 ms in 100ms steps (t=1000 to t=2000)
    for t in range(1000, 2001, 100):
        detector.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0)

    # Face momentarily lost at t=2050 and t=2100 (100 ms loss)
    res_drop1 = detector.update([], timestamp_ms=2050.0, aspect_ratio=1.0, has_face=False)
    assert res_drop1.is_valid is False
    res_drop2 = detector.update([], timestamp_ms=2100.0, aspect_ratio=1.0, has_face=False)
    assert res_drop2.is_valid is False

    # Face recovers at t=2200 and stays open until t=2800 in 100ms steps
    for t in range(2200, 2801, 100):
        detector.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0)

    # Mouth closes at t=2900
    res_end = detector.update(closed_lms, timestamp_ms=2900.0, aspect_ratio=1.0)
    assert res_end.total_yawn_count == 1
    assert res_end.completed_yawn is not None
    # Unobserved time was NOT credited to duration_ms
    assert res_end.completed_yawn.observed_duration_ms < res_end.completed_yawn.wall_clock_duration_ms
    assert res_end.completed_yawn.observed_duration_ms >= 1500.0

    # Case B: Prolonged tracking loss (300 ms > 200 ms grace) cancels in-flight candidate
    detector.reset()
    for t in range(5000, 6001, 100):
        detector.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0)

    # Face lost for 300 ms (t=6050 to t=6350)
    detector.update([], timestamp_ms=6050.0, has_face=False)
    res_expired = detector.update([], timestamp_ms=6350.0, has_face=False)
    assert res_expired.is_yawn_in_progress is False

    # Closing mouth after expired grace must NOT emit a completed yawn
    res_after = detector.update(closed_lms, timestamp_ms=6400.0, aspect_ratio=1.0)
    assert res_after.total_yawn_count == 0
    assert res_after.completed_yawn is None


def test_yawn_detector_explicit_sequences_a_b_c_d() -> None:
    """
    Explicit verification of edge-case sequences required by Phase 3:
    Sequence A: Opening -> MAR 0.48 -> 100 ms tracking loss -> MAR 0.48 (survives, unobserved time uncredited).
    Sequence B: Opening -> 100 ms tracking loss -> MAR 0.40 (terminates correctly).
    Sequence C: Opening -> tracking loss exceeding 200 ms (aborts without completed yawn).
    Sequence D: Opening -> large sampling gap exceeding 500 ms (resets safely).
    """
    def _make_lms(mar_target: float) -> List[NormalizedLandmark]:
        # width = 0.16, vert_height = mar_target * 0.16 (aspect_ratio = 1.0)
        h = mar_target * 0.16
        return _build_mock_landmarks({
            61: (0.42, 0.70, 0.0),
            291: (0.58, 0.70, 0.0),
            13: (0.50, 0.70 - h / 2.0, 0.0),
            14: (0.50, 0.70 + h / 2.0, 0.0),
            81: (0.46, 0.70 - h / 2.0, 0.0),
            178: (0.46, 0.70 + h / 2.0, 0.0),
            311: (0.54, 0.70 - h / 2.0, 0.0),
            402: (0.54, 0.70 + h / 2.0, 0.0),
        })

    lms_open = _make_lms(0.60)
    lms_048 = _make_lms(0.48)
    lms_040 = _make_lms(0.40)
    lms_closed = _make_lms(0.08)

    # --------------------------------------------------------------------------
    # Sequence A: Opening -> MAR 0.48 -> 100 ms tracking loss -> MAR 0.48
    # --------------------------------------------------------------------------
    det_a = YawnDetector(mar_open_threshold=0.55, mar_hold_threshold=0.45, dropout_grace_ms=200.0)
    # 1. Triggers open at t=1000
    res_a1 = det_a.update(lms_open, timestamp_ms=1000.0, aspect_ratio=1.0)
    assert res_a1.is_mouth_open is True
    # 2. Advance to t=1100 with open (observed = 100 ms)
    res_a2 = det_a.update(lms_open, timestamp_ms=1100.0, aspect_ratio=1.0)
    assert res_a2.observed_yawn_duration_ms == 100.0
    # 3. Flutters to MAR 0.48 at t=1200 (observed = 200 ms)
    res_a3 = det_a.update(lms_048, timestamp_ms=1200.0, aspect_ratio=1.0)
    assert res_a3.is_mouth_open is True
    assert res_a3.observed_yawn_duration_ms == 200.0
    # 4. 100 ms tracking loss at t=1250 and t=1300
    det_a.update([], timestamp_ms=1250.0, has_face=False)
    res_a_loss = det_a.update([], timestamp_ms=1300.0, has_face=False)
    assert res_a_loss.is_valid is False
    # 5. Recovery at t=1350 with MAR 0.48
    res_a_rec = det_a.update(lms_048, timestamp_ms=1350.0, aspect_ratio=1.0)
    assert res_a_rec.is_mouth_open is True
    # Missing time (150ms from 1200 to 1350) is NOT credited!
    assert res_a_rec.observed_yawn_duration_ms == 200.0

    # --------------------------------------------------------------------------
    # Sequence B: Opening -> 100 ms tracking loss -> MAR 0.40 (terminates)
    # --------------------------------------------------------------------------
    det_b = YawnDetector(mar_open_threshold=0.55, mar_hold_threshold=0.45, dropout_grace_ms=200.0)
    det_b.update(lms_open, timestamp_ms=1000.0, aspect_ratio=1.0)
    det_b.update(lms_open, timestamp_ms=1100.0, aspect_ratio=1.0)
    # 100 ms tracking loss
    det_b.update([], timestamp_ms=1150.0, has_face=False)
    det_b.update([], timestamp_ms=1200.0, has_face=False)
    # Recovery with MAR 0.40 (< 0.45 hold threshold)
    res_b_term = det_b.update(lms_040, timestamp_ms=1250.0, aspect_ratio=1.0)
    assert res_b_term.is_mouth_open is False
    assert res_b_term.total_yawn_count == 0
    assert res_b_term.completed_yawn is None

    # --------------------------------------------------------------------------
    # Sequence C: Opening -> tracking loss exceeding 200 ms (aborts cleanly)
    # --------------------------------------------------------------------------
    det_c = YawnDetector(mar_open_threshold=0.55, min_yawn_duration_ms=1500.0, dropout_grace_ms=200.0)
    for t in range(1000, 2601, 100):
        det_c.update(lms_open, timestamp_ms=float(t), aspect_ratio=1.0)
    # Tracking loss for 250 ms (> 200 ms grace)
    det_c.update([], timestamp_ms=2650.0, has_face=False)
    det_c.update([], timestamp_ms=2750.0, has_face=False)
    res_c_abort = det_c.update([], timestamp_ms=2900.0, has_face=False)
    assert res_c_abort.is_yawn_in_progress is False
    # Next closed frame must NOT register a completed yawn
    res_c_close = det_c.update(lms_closed, timestamp_ms=2950.0, aspect_ratio=1.0)
    assert res_c_close.total_yawn_count == 0
    assert res_c_close.completed_yawn is None

    # --------------------------------------------------------------------------
    # Sequence D: Opening -> large sampling gap exceeding 500 ms (resets safely)
    # --------------------------------------------------------------------------
    det_d = YawnDetector(mar_open_threshold=0.55, min_yawn_duration_ms=1500.0, max_sampling_gap_ms=500.0)
    for t in range(1000, 2601, 100):
        det_d.update(lms_open, timestamp_ms=float(t), aspect_ratio=1.0)
    # Large sampling gap: next frame arrives 600 ms later at t=3200 (> 500 ms)
    res_d_gap = det_d.update(lms_closed, timestamp_ms=3200.0, aspect_ratio=1.0)
    assert res_d_gap.total_yawn_count == 0
    assert res_d_gap.completed_yawn is None
    assert res_d_gap.is_mouth_open is False


def test_yawn_active_duration_observed_vs_wall_clock_downstream_integration() -> None:
    """
    Regression test:
    Demonstrates that an active mouth opening with 1,400 ms of observed duration and 100 ms
    of tracking dropout (1,600 ms wall clock) does NOT trigger the 1,500 ms active-yawn
    decision threshold in DecisionEngine, while genuinely reaching 1,500 ms of observed duration does.
    """
    from backend.app.intelligence.decision_engine import DecisionEngine
    from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer
    from backend.app.intelligence.confidence import SignalConfidenceEvaluator
    from backend.app.intelligence.reason_codes import DrowsinessState
    from backend.tests.test_drowsiness_engine import make_driver_features

    open_lms = _build_mock_landmarks({
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.64, 0.0),
        14: (0.50, 0.76, 0.0),
        81: (0.46, 0.64, 0.0),
        178: (0.46, 0.76, 0.0),
        311: (0.54, 0.64, 0.0),
        402: (0.54, 0.76, 0.0),
    })

    detector = YawnDetector(mar_open_threshold=0.55, min_yawn_duration_ms=1500.0, dropout_grace_ms=200.0)
    temporal = TemporalAnalyzer()
    decision = DecisionEngine()
    conf_eval = SignalConfidenceEvaluator()

    # 1. 1000 ms of valid open mouth (t=1000 to t=2000 in 100ms steps)
    for t in range(1000, 2001, 100):
        mar_res = detector.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0)
        feats = make_driver_features(timestamp_ms=float(t), mar_val=mar_res.mar)
        feats = feats.__class__(**{**feats.__dict__, "mar": mar_res})
        m = temporal.update(feats)
        d = decision.evaluate(metrics=m, confidence=conf_eval.evaluate(feats))

    # 2. 100 ms of tracking dropout (t=2050, 2100)
    for t in (2050, 2100):
        mar_res = detector.update([], timestamp_ms=float(t), has_face=False)
        feats = make_driver_features(timestamp_ms=float(t), mar_val=mar_res.mar)
        feats = feats.__class__(**{**feats.__dict__, "mar": mar_res})
        m = temporal.update(feats)
        d = decision.evaluate(metrics=m, confidence=conf_eval.evaluate(feats))

    # 3. Recover at t=2200 and open until t=2600 (400ms observed after recovery)
    # Total observed: 1000 + 400 = 1400 ms. Wall clock: 2600 - 1000 = 1600 ms.
    for t in (2200, 2300, 2400, 2500, 2600):
        mar_res = detector.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0)
        feats = make_driver_features(timestamp_ms=float(t), mar_val=mar_res.mar)
        feats = feats.__class__(**{**feats.__dict__, "mar": mar_res})
        m = temporal.update(feats)
        d = decision.evaluate(metrics=m, confidence=conf_eval.evaluate(feats))

    # CRITICAL CHECK: Observed duration is 1400 ms, NOT triggering the 1500ms threshold
    assert mar_res.observed_yawn_duration_ms == 1400.0
    assert m.active_yawn_duration_ms == 1400.0
    assert mar_res.current_yawn_duration_ms == 1600.0  # Wall-clock preserved for diagnostics
    assert d.state != DrowsinessState.ATTENTION, "1400 ms observed duration must NOT trigger 1500 ms attention alert!"

    # 4. Advance one more 100 ms frame to t=2700 (1500 ms observed duration reached!)
    mar_res = detector.update(open_lms, timestamp_ms=2700.0, aspect_ratio=1.0)
    feats = make_driver_features(timestamp_ms=2700.0, mar_val=mar_res.mar)
    feats = feats.__class__(**{**feats.__dict__, "mar": mar_res})
    m = temporal.update(feats)
    d = decision.evaluate(metrics=m, confidence=conf_eval.evaluate(feats))

    # Genuinely reaching 1500 ms observed duration triggers ATTENTION
    assert mar_res.observed_yawn_duration_ms == 1500.0
    assert m.active_yawn_duration_ms == 1500.0
    assert d.state == DrowsinessState.ATTENTION, "Genuinely observed 1500 ms MUST trigger ATTENTION state!"


def test_yawn_detector_baseline_tracking_and_opening_ratio() -> None:
    """Verifies resting closed-mouth baseline tracking and opening ratio without baseline contamination."""
    detector = YawnDetector(
        mar_open_threshold=0.55,
        closed_baseline_threshold=0.28,
        baseline_alpha=0.10,
    )
    assert detector.baseline_mar == 0.15

    # 1. Stable closed mouth at MAR=0.10 updates baseline smoothly downward
    closed_10 = _build_mock_landmarks({
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.692, 0.0),
        14: (0.50, 0.708, 0.0),  # height = 0.016 / 0.16 = 0.10
        81: (0.46, 0.692, 0.0),
        178: (0.46, 0.708, 0.0),
        311: (0.54, 0.692, 0.0),
        402: (0.54, 0.708, 0.0),
    })
    for t in range(1000, 1500, 100):
        res = detector.update(closed_10, timestamp_ms=float(t), aspect_ratio=1.0)
        assert res.opening_ratio < 1.0

    assert detector.baseline_mar < 0.15, "Closed mouth observations must adaptively lower baseline"

    # 2. Wide-open mouth (MAR=0.625) must NOT contaminate the baseline
    baseline_before_open = detector.baseline_mar
    open_lms = _build_mock_landmarks({
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.64, 0.0),
        14: (0.50, 0.76, 0.0),  # height = 0.12 / 0.16 = 0.75
        81: (0.46, 0.64, 0.0),
        178: (0.46, 0.76, 0.0),
        311: (0.54, 0.64, 0.0),
        402: (0.54, 0.76, 0.0),
    })
    res_open = detector.update(open_lms, timestamp_ms=2000.0, aspect_ratio=1.0)
    assert res_open.opening_ratio > 4.0
    assert detector.baseline_mar == baseline_before_open, "Open mouth must NEVER contaminate resting baseline!"

    # 3. Head turn (is_looking_away=True) must NOT update the baseline
    baseline_before_head_turn = detector.baseline_mar
    detector.update(closed_10, timestamp_ms=2100.0, aspect_ratio=1.0, is_looking_away=True)
    assert detector.baseline_mar == baseline_before_head_turn, "Head turn must NOT update resting baseline!"


def test_yawn_detector_timestamp_anomalies_and_recovery() -> None:
    """Verifies that non-finite, negative, duplicate, or backwards timestamps are rejected without state corruption."""
    detector = YawnDetector(mar_open_threshold=0.55)
    closed_lms = _build_mock_landmarks({
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.70, 0.0),
        14: (0.50, 0.70, 0.0),
        81: (0.46, 0.70, 0.0),
        178: (0.46, 0.70, 0.0),
        311: (0.54, 0.70, 0.0),
        402: (0.54, 0.70, 0.0),
    })

    # Valid step
    res_valid = detector.update(closed_lms, timestamp_ms=1000.0, aspect_ratio=1.0)
    assert res_valid.is_valid is True

    # 1. NaN timestamp
    res_nan = detector.update(closed_lms, timestamp_ms=float("nan"), aspect_ratio=1.0)
    assert res_nan.is_valid is False

    # 2. Negative timestamp
    res_neg = detector.update(closed_lms, timestamp_ms=-50.0, aspect_ratio=1.0)
    assert res_neg.is_valid is False

    # 3. Duplicate timestamp (dt = 0)
    res_dup = detector.update(closed_lms, timestamp_ms=1000.0, aspect_ratio=1.0)
    assert res_dup.is_valid is False

    # 4. Backwards timestamp (dt < 0)
    res_back = detector.update(closed_lms, timestamp_ms=800.0, aspect_ratio=1.0)
    assert res_back.is_valid is False

    # 5. Recovery on valid monotonic timestamp
    res_recovered = detector.update(closed_lms, timestamp_ms=1100.0, aspect_ratio=1.0)
    assert res_recovered.is_valid is True


def test_mar_result_and_event_backward_compatibility() -> None:
    """Verifies that MouthAspectRatioResult and YawnEvent maintain full backward compatibility."""
    event = YawnEvent(
        start_ts_ms=1000.0,
        end_ts_ms=3000.0,
        duration_ms=2000.0,
        peak_mar=0.68,
    )
    # Existing fields accessible
    assert event.start_ts_ms == 1000.0
    assert event.end_ts_ms == 3000.0
    assert event.duration_ms == 2000.0
    assert event.peak_mar == 0.68
    # New fields have defaults
    assert event.observed_duration_ms == 0.0
    assert event.wall_clock_duration_ms == 0.0
    assert event.mean_mar == 0.0
    assert event.opening_ratio == 1.0

    res = MouthAspectRatioResult(
        mar=0.62,
        is_mouth_open=True,
        is_yawn_in_progress=True,
        current_yawn_duration_ms=1600.0,
        is_authentic_yawn=True,
        completed_yawn=event,
        total_yawn_count=1,
    )
    assert res.mar == 0.62
    assert res.is_mouth_open is True
    assert res.is_yawn_in_progress is True
    assert res.current_yawn_duration_ms == 1600.0
    assert res.is_authentic_yawn is True
    assert res.completed_yawn == event
    assert res.total_yawn_count == 1
    assert res.is_valid is True
    assert res.baseline_mar == 0.15
    assert res.opening_ratio == 1.0


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


def test_blink_sub40ms_and_low_amplitude_rejection() -> None:
    """Verifies that sub-40 ms noise twitches and low amplitude flutter are classified as UNCERTAIN and suppressed."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.23,
        min_blink_duration_ms=40.0,
        min_amplitude=0.04,
    )
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    # Rapid 20 ms dip
    detector.update(timestamp_ms=1010.0, left_ear=0.18, right_ear=0.18, ear=0.18)
    s_reopen = detector.update(timestamp_ms=1025.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    # Must be suppressed from completed_blink emission
    assert s_reopen.completed_blink is None
    assert s_reopen.total_blink_count == 0


def test_blink_unilateral_wink_exclusion() -> None:
    """Verifies that a unilateral wink is marked is_binocular=False and does NOT increment binocular totals."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.23,
    )
    # Open eyes
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    # Left eye closes (0.10), right eye stays open (0.30), weighted ear drops to 0.18
    detector.update(timestamp_ms=1033.0, left_ear=0.10, right_ear=0.30, ear=0.18)
    detector.update(timestamp_ms=1066.0, left_ear=0.08, right_ear=0.30, ear=0.16)
    detector.update(timestamp_ms=1100.0, left_ear=0.14, right_ear=0.30, ear=0.19)

    # Left eye reopens
    s_end = detector.update(timestamp_ms=1133.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    assert s_end.completed_blink is not None
    assert s_end.completed_blink.is_binocular is False
    # Unilateral closures must NOT increment binocular blink totals
    assert s_end.total_blink_count == 0
    assert s_end.prolonged_blink_count == 0


def test_blink_closure_underway_before_tracking_lost() -> None:
    """Verifies that closure underway before tracking loss is left-censored on recovery and does not fabricate a blink."""
    detector = BlinkKinematicsDetector(max_sampling_gap_ms=250.0)

    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    # Closure starts at t=1030ms
    detector.update(timestamp_ms=1030.0, left_ear=0.10, right_ear=0.10, ear=0.10)

    # Tracking lost for 400 ms (t=1030 to t=1430)
    s_lost = detector.update(timestamp_ms=1430.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_lost.state == BlinkState.UNCERTAIN_GAP
    assert s_lost.is_eye_closed is False

    # Frame t=1460: Tracking recovered, eyes still closed
    s_rec = detector.update(timestamp_ms=1460.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_rec.state == BlinkState.CLOSED
    assert s_rec.is_eye_closed is True
    # Closure duration measured from recovery timestamp (1460), not backdated to 1030
    assert s_rec.current_closure_duration_ms == 0.0

    # Frame t=1500: Eyes still closed (40 ms observed closure since recovery)
    s_mid = detector.update(timestamp_ms=1500.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_mid.current_closure_duration_ms == 40.0

    # Frame t=1550: Eyes reopen
    s_reopen = detector.update(timestamp_ms=1550.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_reopen.state == BlinkState.OPEN
    # True onset was unobserved in gap; must NOT emit a completed blink with fabricated duration
    assert s_reopen.completed_blink is None
    assert s_reopen.total_blink_count == 0


def test_blink_closure_beginning_during_tracking_gap() -> None:
    """Verifies that a closure beginning during a tracking gap is discovered on recovery and left-censored."""
    detector = BlinkKinematicsDetector(max_sampling_gap_ms=250.0)

    # Normal open eyes before gap
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    # 400 ms gap where eyes closed during the gap
    s_gap = detector.update(timestamp_ms=1400.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_gap.state == BlinkState.UNCERTAIN_GAP

    # Recovery frame at t=1430: Eye observed closed
    s_rec = detector.update(timestamp_ms=1430.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_rec.state == BlinkState.CLOSED
    assert s_rec.is_eye_closed is True
    assert s_rec.current_closure_duration_ms == 0.0

    # Eyes reopen at t=1500
    s_open = detector.update(timestamp_ms=1500.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_open.state == BlinkState.OPEN
    assert s_open.completed_blink is None
    assert s_open.total_blink_count == 0


def test_blink_prolonged_closure_after_tracking_recovery() -> None:
    """Verifies that an eye closed on recovery continues unbroken into PROLONGED_CLOSURE."""
    detector = BlinkKinematicsDetector(
        prolonged_duration_ms=300.0,
        microsleep_duration_ms=800.0,
        max_sampling_gap_ms=250.0,
    )
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    # Gap to t=1400
    detector.update(timestamp_ms=1400.0, left_ear=0.10, right_ear=0.10, ear=0.10)

    # Recovery frame at t=1430 with closed eyes
    detector.update(timestamp_ms=1430.0, left_ear=0.10, right_ear=0.10, ear=0.10)

    # Remains closed for 900 ms unbroken from t=1430 to t=2330
    s_pro = None
    for t in range(1480, 2350, 50):
        s_pro = detector.update(timestamp_ms=float(t), left_ear=0.08, right_ear=0.08, ear=0.08)

    assert s_pro.state == BlinkState.PROLONGED_CLOSURE
    assert s_pro.is_eye_closed is True
    assert s_pro.current_closure_duration_ms >= 800.0

    # Reopening after prolonged closure does NOT fabricate a microsleep blink event
    s_reopen = detector.update(timestamp_ms=2380.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_reopen.state == BlinkState.OPEN
    assert s_reopen.completed_blink is None
    assert s_reopen.prolonged_blink_count == 0


def test_blink_sampling_gap_stale_interpolation_prevention() -> None:
    """Verifies that large sampling gaps clear interpolation references to prevent cross-gap interpolation."""
    detector = BlinkKinematicsDetector(max_sampling_gap_ms=250.0)

    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    # 500 ms gap to t=1500
    s_gap = detector.update(timestamp_ms=1500.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_gap.state == BlinkState.UNCERTAIN_GAP

    # Next frame at t=1533: Eyes open
    s_open = detector.update(timestamp_ms=1533.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_open.state == BlinkState.OPEN


def test_blink_input_anomalies_and_timestamp_resilience() -> None:
    """Verifies detector resilience against NaN, Inf, negative EAR, duplicate, and backwards timestamps."""
    detector = BlinkKinematicsDetector()
    s0 = detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s0.state == BlinkState.OPEN

    # NaN and Inf EAR
    s_nan = detector.update(timestamp_ms=1033.0, left_ear=float("nan"), right_ear=0.30, ear=float("nan"))
    assert s_nan.state == BlinkState.OPEN

    s_inf = detector.update(timestamp_ms=1066.0, left_ear=float("inf"), right_ear=0.30, ear=float("inf"))
    assert s_inf.state == BlinkState.OPEN

    # Negative EAR
    s_neg = detector.update(timestamp_ms=1100.0, left_ear=-0.10, right_ear=-0.10, ear=-0.10)
    assert s_neg.state == BlinkState.OPEN

    # Valid frame to establish state
    s_valid = detector.update(timestamp_ms=1133.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_valid.state == BlinkState.OPEN

    # Duplicate timestamp (gap = 0)
    s_dup = detector.update(timestamp_ms=1133.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_dup.state == BlinkState.OPEN  # Ignored safely

    # Backwards timestamp (gap < 0)
    s_back = detector.update(timestamp_ms=1100.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    assert s_back.state == BlinkState.OPEN  # Ignored safely

    # Next valid frame proceeds normally
    s_next = detector.update(timestamp_ms=1166.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s_next.state == BlinkState.OPEN


def test_blink_kinematic_reliability_vs_validity() -> None:
    """Verifies that a sub-5ms interval marks kinematics unreliable and zeroes velocity/RCR while preserving the event."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.24,
        min_blink_duration_ms=40.0,
    )
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    # Extremely rapid descent at t=1004 (closing phase is only ~3 ms <= 5 ms)
    detector.update(timestamp_ms=1004.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    # Reopened at t=1060 (total duration ~ 58ms >= 40ms, but close phase was degenerate <= 5ms)
    s = detector.update(timestamp_ms=1060.0, left_ear=0.30, right_ear=0.30, ear=0.30)

    assert s.completed_blink is not None
    assert s.completed_blink.classification == BlinkClassification.NORMAL
    assert s.total_blink_count == 1
    # Check that kinematics unreliability zeroes velocity and RCR without fabricating values
    assert s.completed_blink.is_kinematics_reliable is False
    assert s.completed_blink.closing_velocity_per_s == 0.0
    assert s.completed_blink.reopening_velocity_per_s == 0.0
    assert s.completed_blink.rcr == 0.0
    # Measured duration is preserved
    assert s.completed_blink.duration_ms > 40.0
    # recent_blink_rcr is NOT contaminated by unreliable kinematics
    assert s.recent_blink_rcr is None
    # recent_blink_duration_ms IS updated because event is valid
    assert s.recent_blink_duration_ms == s.completed_blink.duration_ms


def test_blink_measured_sub1ms_duration_not_clamped() -> None:
    """Verifies that a measured sub-1ms duration is not silently clamped to 1.0ms."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.24,
        min_blink_duration_ms=0.0,  # Disable provisional noise filter for direct measurement check
        min_amplitude=0.0,
    )
    # Start at t=1000.0
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    # Extremely fast micro-drop crossing at t=1000.2
    detector.update(timestamp_ms=1000.2, left_ear=0.10, right_ear=0.10, ear=0.10)
    # Reopened at t=1000.6 (duration is exactly 0.4ms)
    s = detector.update(timestamp_ms=1000.6, left_ear=0.30, right_ear=0.30, ear=0.30)

    assert s.completed_blink is not None
    # Must NOT be clamped to 1.0 ms! Exact measured interval is preserved.
    assert s.completed_blink.duration_ms < 1.0
    assert s.completed_blink.duration_ms == pytest.approx(0.4, abs=0.1)


def test_blink_zero_or_negative_duration_rejected() -> None:
    """Verifies that non-positive durations cannot create a valid completed blink."""
    detector = BlinkKinematicsDetector(
        min_blink_duration_ms=0.0,
        min_amplitude=0.0,
    )
    detector.update(timestamp_ms=1000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    detector.update(timestamp_ms=1033.0, left_ear=0.10, right_ear=0.10, ear=0.10)
    # Manually simulate degenerate identical start/end timestamp
    detector._blink_start_ts = 1050.0
    detector._closure_start_ts = 1050.0
    detector._prev_ts = 1050.0
    detector._prev_ear = 0.10
    # Reopen exactly at t=1050.0 (t_end - start_ts = 0.0)
    s = detector.update(timestamp_ms=1050.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    assert s.completed_blink is None


def test_blink_amplitude_only_rejection_separate_from_duration() -> None:
    """Verifies that an amplitude-only rejection is tested separately from a duration-only rejection."""
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.24,
        min_blink_duration_ms=40.0,
        min_amplitude=0.04,
    )
    # Case 1: Normal duration (150ms >= 40ms), but small amplitude (0.02 < 0.04)
    detector.update(timestamp_ms=1000.0, left_ear=0.23, right_ear=0.23, ear=0.23)
    detector.update(timestamp_ms=1050.0, left_ear=0.21, right_ear=0.21, ear=0.21)
    detector.update(timestamp_ms=1100.0, left_ear=0.21, right_ear=0.21, ear=0.21)
    s_amp = detector.update(timestamp_ms=1150.0, left_ear=0.25, right_ear=0.25, ear=0.25)
    # Rejected strictly due to amplitude failure
    assert s_amp.completed_blink is None
    assert s_amp.total_blink_count == 0

    detector.reset()

    # Case 2: Normal amplitude (0.15 >= 0.04), but short duration (25ms < 40ms)
    detector.update(timestamp_ms=2000.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    detector.update(timestamp_ms=2010.0, left_ear=0.15, right_ear=0.15, ear=0.15)
    s_dur = detector.update(timestamp_ms=2025.0, left_ear=0.30, right_ear=0.30, ear=0.30)
    # Rejected strictly due to duration failure
    assert s_dur.completed_blink is None
    assert s_dur.total_blink_count == 0


def test_blink_downstream_temporal_analyzer_and_decision_engine_integration() -> None:
    """Verifies that unilateral winks and left-censored closures cannot trigger prolonged blink warnings."""
    from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer
    from backend.app.intelligence.decision_engine import DecisionEngine
    from backend.app.intelligence.confidence import ConfidenceAssessment, ConfidenceTier
    from backend.app.intelligence.reason_codes import DrowsinessState
    from backend.tests.test_drowsiness_engine import make_driver_features

    temporal = TemporalAnalyzer()
    decision = DecisionEngine()

    # 1. Feed 3 unilateral prolonged winks (duration = 400ms each, is_binocular=False)
    for i in range(3):
        ts = 1000.0 + i * 2000.0
        wink = BlinkEvent(
            start_ts_ms=ts - 400.0,
            min_ts_ms=ts - 200.0,
            end_ts_ms=ts,
            duration_ms=400.0,
            open_ear_before=0.30,
            min_ear=0.10,
            amplitude=0.20,
            closing_velocity_per_s=1.0,
            reopening_velocity_per_s=1.0,
            rcr=1.0,
            classification=BlinkClassification.PROLONGED,
            is_binocular=False,  # Unilateral wink!
            is_kinematics_reliable=True,
        )
        features = make_driver_features(timestamp_ms=ts, completed_blink=wink)
        metrics = temporal.update(features)
        conf = ConfidenceAssessment(overall_confidence=1.0, tier=ConfidenceTier.HIGH, is_reliable=True)
        res = decision.evaluate(metrics, conf)

        # Must NOT count winks into binocular metrics
        assert metrics.blink_count_60s == 0
        assert metrics.prolonged_blink_count_60s == 0
        assert res.state == DrowsinessState.NORMAL

    # 2. Feed 3 genuine binocular prolonged blinks (duration = 400ms, is_binocular=True)
    for i in range(3):
        ts = 8000.0 + i * 2000.0
        blink = BlinkEvent(
            start_ts_ms=ts - 400.0,
            min_ts_ms=ts - 200.0,
            end_ts_ms=ts,
            duration_ms=400.0,
            open_ear_before=0.30,
            min_ear=0.10,
            amplitude=0.20,
            closing_velocity_per_s=1.0,
            reopening_velocity_per_s=1.0,
            rcr=1.0,
            classification=BlinkClassification.PROLONGED,
            is_binocular=True,  # Genuine binocular prolonged blink
            is_kinematics_reliable=True,
        )
        features = make_driver_features(timestamp_ms=ts, completed_blink=blink)
        metrics = temporal.update(features)
        conf = ConfidenceAssessment(overall_confidence=1.0, tier=ConfidenceTier.HIGH, is_reliable=True)
        res = decision.evaluate(metrics, conf)

    assert metrics.prolonged_blink_count_60s == 3
    # 3 prolonged binocular blinks trigger DROWSINESS_WARNING
    assert res.state == DrowsinessState.DROWSINESS_WARNING


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
    assert res.effective_perclos_pct is None  # Gated when unreliable


def test_perclos_window_boundary_clipping() -> None:
    """
    Verifies that observation intervals crossing the rolling-window boundary
    are accurately clipped to the exact boundary rather than being counted in full.
    """
    # 10-second window, max_valid_dt = 1.0s
    calc = PerclosCalculator(window_sec=10.0, min_coverage_pct=50.0, max_valid_dt_sec=1.0)

    # Interval 1: [0.0s, 2.0s] - eyes open (dt = 0.5s steps)
    t = 0.0
    calc.update(timestamp_ms=t, is_closed=False)
    for _ in range(4):
        t += 500.0
        calc.update(timestamp_ms=t, is_closed=False)

    # Interval 2: [2.0s, 12.0s] - eyes closed (dt = 500ms steps)
    # At t=12.0s, window cutoff is 12.0 - 10.0 = 2.0s.
    # The early open interval [0s, 2s] has completely rolled out.
    # The remaining window [2s, 12s] (10s) is 100% closed.
    for _ in range(20):
        t += 500.0
        res = calc.update(timestamp_ms=t, is_closed=True)

    assert t == 12000.0
    assert res.is_reliable is True
    assert math.isclose(res.observed_duration_sec, 10.0, abs_tol=0.05)
    assert math.isclose(res.closed_duration_sec, 10.0, abs_tol=0.05)
    assert math.isclose(res.perclos_pct, 100.0, abs_tol=0.1)

    # Advance by 400ms: t=12400. Cutoff is 2400ms.
    # The interval [2000ms, 2500ms] straddles cutoff (2400ms).
    # Effective duration of that first sample must be clipped to (2500 - 2400) = 100ms.
    res_straddle = calc.update(timestamp_ms=12400.0, is_closed=True)
    # Total observed within 10s window must not exceed 10.0s
    assert res_straddle.observed_duration_sec <= 10.0
    assert res_straddle.closed_duration_sec <= 10.0
    assert math.isclose(res_straddle.unobserved_duration_sec + res_straddle.observed_duration_sec, 10.0, abs_tol=1e-4)


def test_perclos_timestamp_anomalies_and_recovery() -> None:
    """
    Verifies that non-monotonic, duplicate, negative, and non-finite timestamps
    are safely rejected without corrupting the previous valid timestamp state.
    """
    calc = PerclosCalculator(window_sec=10.0, min_coverage_pct=50.0, max_valid_dt_sec=0.25)

    # Valid initial frames at 1000ms and 1033ms
    calc.update(timestamp_ms=1000.0, is_closed=False)
    res_valid = calc.update(timestamp_ms=1033.0, is_closed=False)
    assert math.isclose(res_valid.observed_duration_sec, 0.033, abs_tol=1e-3)

    # 1. Negative timestamp: rejected, state unchanged
    res_neg = calc.update(timestamp_ms=-500.0, is_closed=False)
    assert math.isclose(res_neg.observed_duration_sec, res_valid.observed_duration_sec, abs_tol=1e-4)

    # 2. Non-finite timestamps (NaN, Inf): rejected
    res_nan = calc.update(timestamp_ms=float("nan"), is_closed=False)
    assert math.isclose(res_nan.observed_duration_sec, res_valid.observed_duration_sec, abs_tol=1e-4)

    res_inf = calc.update(timestamp_ms=float("inf"), is_closed=False)
    assert math.isclose(res_inf.observed_duration_sec, res_valid.observed_duration_sec, abs_tol=1e-4)

    # 3. Duplicate timestamp: rejected from interval accumulation
    res_dup = calc.update(timestamp_ms=1033.0, is_closed=False)
    assert math.isclose(res_dup.observed_duration_sec, res_valid.observed_duration_sec, abs_tol=1e-4)

    # 4. Out-of-order timestamp: time went backwards to 900ms
    res_back = calc.update(timestamp_ms=900.0, is_closed=False)
    assert math.isclose(res_back.observed_duration_sec, res_valid.observed_duration_sec, abs_tol=1e-4)

    # 5. Normal subsequent frame at 1066ms: computes dt from 1033ms, NOT from 900ms!
    res_next = calc.update(timestamp_ms=1066.0, is_closed=False)
    assert math.isclose(res_next.observed_duration_sec, 0.066, abs_tol=1e-3)


def test_perclos_unavailable_measurements_vs_open_eyes() -> None:
    """
    Verifies that unavailable measurements (is_closed=None, is_valid=False, has_face=False)
    are treated as unobserved gaps rather than being misclassified as open eyes.
    """
    calc_open = PerclosCalculator(window_sec=10.0, min_coverage_pct=50.0, max_valid_dt_sec=0.25)
    calc_unavail = PerclosCalculator(window_sec=10.0, min_coverage_pct=50.0, max_valid_dt_sec=0.25)

    # Both establish initial frame
    calc_open.update(timestamp_ms=1000.0, is_closed=True)
    calc_unavail.update(timestamp_ms=1000.0, is_closed=True)

    # Both record 2 seconds of closed eyes
    t = 1000.0
    for _ in range(60):  # 60 * 33.3ms ~ 2.0s
        t += 33.333
        calc_open.update(timestamp_ms=t, is_closed=True)
        calc_unavail.update(timestamp_ms=t, is_closed=True)

    # Now for the next 2 seconds:
    # calc_open sees genuinely open eyes (is_closed=False)
    # calc_unavail has missing/occluded eyes (is_closed=None or is_valid=False)
    for _ in range(60):
        t += 33.333
        res_open = calc_open.update(timestamp_ms=t, is_closed=False, has_face=True, is_valid=True)
        res_unavail = calc_unavail.update(timestamp_ms=t, is_closed=None, has_face=True, is_valid=False)

    # calc_open observed 4 seconds total (2s closed, 2s open -> 50% PERCLOS)
    assert math.isclose(res_open.observed_duration_sec, 4.0, abs_tol=0.1)
    assert math.isclose(res_open.perclos_pct, 50.0, abs_tol=1.0)

    # calc_unavail observed ONLY 2 seconds total (2s closed, 2s unobserved -> 100% PERCLOS on valid time!)
    # Missing data DID NOT dilute PERCLOS to 50%!
    assert math.isclose(res_unavail.observed_duration_sec, 2.0, abs_tol=0.1)
    assert math.isclose(res_unavail.perclos_pct, 100.0, abs_tol=1.0)
    # But observation coverage for unavail dropped to 20% (< 50% threshold), so is_reliable is False!
    assert res_unavail.is_reliable is False
    assert res_unavail.effective_perclos_pct is None


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
    assert features.head_pose.zone == PoseZone.UNKNOWN
    assert features.head_pose.is_looking_away is False
    assert "FACE: INSUFFICIENT_DATA" in features.summary_text


# -----------------------------------------------------------------------------
# 7. Phase 5 Head Pose Validation & Rigid-First Refinement Tests
# -----------------------------------------------------------------------------

def test_head_pose_pnp_reprojection_validation_rejects_implausible_depth() -> None:
    """Verifies that validate_pnp_solution rejects negative, sub-minimum (<100mm), or excessive (>3000mm) depth."""
    w, h = 1280, 720
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)

    # 1. Valid depth at 600mm
    tvec_valid = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
    proj_valid, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec_valid, cam_mat, dist)
    d_ref = float(np.linalg.norm(proj_valid[2] - proj_valid[3]))
    ok, mean_e, max_e = validate_pnp_solution(CANONICAL_3D_FACE_POINTS, proj_valid.reshape(-1, 2), rvec, tvec_valid, cam_mat, dist, d_ref)
    assert ok is True
    assert mean_e < 1.0

    # 2. Negative depth (-200mm, behind camera)
    tvec_neg = np.array([[0.0], [0.0], [-200.0]], dtype=np.float64)
    ok_neg, _, _ = validate_pnp_solution(CANONICAL_3D_FACE_POINTS, proj_valid.reshape(-1, 2), rvec, tvec_neg, cam_mat, dist, d_ref)
    assert ok_neg is False

    # 3. Sub-minimum depth (50mm < 100mm)
    tvec_sub = np.array([[0.0], [0.0], [50.0]], dtype=np.float64)
    ok_sub, _, _ = validate_pnp_solution(CANONICAL_3D_FACE_POINTS, proj_valid.reshape(-1, 2), rvec, tvec_sub, cam_mat, dist, d_ref)
    assert ok_sub is False

    # 4. Excessive depth (4500mm > 3000mm)
    tvec_far = np.array([[0.0], [0.0], [4500.0]], dtype=np.float64)
    ok_far, _, _ = validate_pnp_solution(CANONICAL_3D_FACE_POINTS, proj_valid.reshape(-1, 2), rvec, tvec_far, cam_mat, dist, d_ref)
    assert ok_far is False


def test_head_pose_pnp_reprojection_validation_rejects_gross_reprojection_error() -> None:
    """Verifies that corrupted landmark coordinates exceeding 0.25 * d_ref are rejected as invalid with PoseZone.UNKNOWN."""
    w, h = 1280, 720
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, cam_mat, dist)
    pts = proj.reshape(-1, 2).copy()

    # Corrupt nose tip by 150px so that mean and max reprojection residuals fail validation
    pts[0, 1] += 150.0

    mock_lms = {
        SOLVEPNP_LANDMARK_INDICES[i]: (float(pts[i, 0] / w), float(pts[i, 1] / h), 0.0)
        for i in range(len(SOLVEPNP_LANDMARK_INDICES))
    }
    landmarks = _build_mock_landmarks(mock_lms)

    res = estimate_head_pose(landmarks, frame_width=w, frame_height=h)
    assert res.is_valid is False
    assert res.zone == PoseZone.UNKNOWN
    assert res.is_looking_away is False


def test_head_pose_rigid_first_mandible_consistency_and_refinement() -> None:
    """
    Verifies that:
    1. A neutral resting jaw has max mandible residual <= 0.06 * d_ref and permits refinement.
    2. Talking (10mm chin drop) produces residual > 0.06 * d_ref, rejecting refinement and preserving pure rigid solve.
    3. Yawning (35mm chin drop) produces residual >> 0.06 * d_ref, rejecting refinement and keeping pitch invariant.
    """
    w, h = 1280, 720
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    # Ground truth: neutral pitch 0°, yaw 0°, at 600mm
    rvec_gt = np.zeros((3, 1), dtype=np.float64)
    tvec_gt = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec_gt, tvec_gt, cam_mat, dist)
    pts_clean = proj.reshape(-1, 2).copy()

    # 1. Neutral face: refinement enabled
    mock_neutral = {
        SOLVEPNP_LANDMARK_INDICES[i]: (float(pts_clean[i, 0] / w), float(pts_clean[i, 1] / h), 0.0)
        for i in range(len(SOLVEPNP_LANDMARK_INDICES))
    }
    lms_neutral = _build_mock_landmarks(mock_neutral)
    res_neutral = estimate_head_pose(lms_neutral, frame_width=w, frame_height=h, enable_mandible_refinement=True)
    assert res_neutral.is_valid is True
    assert abs(res_neutral.pitch_deg) < 0.1
    assert abs(res_neutral.yaw_deg) < 0.1
    assert res_neutral.zone == PoseZone.CENTER

    # 2. Talking face: chin drops by 10mm (~21.3 px on camera)
    pts_talking = pts_clean.copy()
    pts_talking[1, 1] += 21.3  # MediaPipe 152 (chin) index 1 in SOLVEPNP_LANDMARK_INDICES

    mock_talking = {
        SOLVEPNP_LANDMARK_INDICES[i]: (float(pts_talking[i, 0] / w), float(pts_talking[i, 1] / h), 0.0)
        for i in range(len(SOLVEPNP_LANDMARK_INDICES))
    }
    lms_talking = _build_mock_landmarks(mock_talking)
    res_talking = estimate_head_pose(lms_talking, frame_width=w, frame_height=h, enable_mandible_refinement=True)
    assert res_talking.is_valid is True
    # Pure rigid solve was retained: pitch remains near 0°, completely immune to talking!
    assert abs(res_talking.pitch_deg) < 0.1
    assert res_talking.zone == PoseZone.CENTER

    # 3. Yawning face: chin drops by 35mm (~74.6 px on camera)
    pts_yawn = pts_clean.copy()
    pts_yawn[1, 1] += 74.6  # Large chin displacement

    mock_yawn = {
        SOLVEPNP_LANDMARK_INDICES[i]: (float(pts_yawn[i, 0] / w), float(pts_yawn[i, 1] / h), 0.0)
        for i in range(len(SOLVEPNP_LANDMARK_INDICES))
    }
    lms_yawn = _build_mock_landmarks(mock_yawn)
    res_yawn = estimate_head_pose(lms_yawn, frame_width=w, frame_height=h, enable_mandible_refinement=True)
    assert res_yawn.is_valid is True
    # Pure rigid solve was retained: pitch remains near 0°, zero false nodding!
    assert abs(res_yawn.pitch_deg) < 0.1
    assert res_yawn.zone == PoseZone.CENTER


def test_face_loss_returns_pose_zone_unknown_and_preserves_oob() -> None:
    """
    Verifies that PoseZone.UNKNOWN is strictly returned for lost face or solver failures,
    while PoseZone.OUT_OF_BOUNDS is reserved exclusively for authentic extreme head angles.
    """
    # 1. Missing landmarks -> UNKNOWN
    res_empty = estimate_head_pose([])
    assert res_empty.zone == PoseZone.UNKNOWN
    assert res_empty.is_valid is False
    assert res_empty.is_looking_away is False

    # 2. None input -> UNKNOWN
    res_none = estimate_head_pose(None)
    assert res_none.zone == PoseZone.UNKNOWN
    assert res_none.is_valid is False
    assert res_none.is_looking_away is False

    # 3. DriverFeatureExtractor on lost face -> UNKNOWN
    extractor = DriverFeatureExtractor()
    lost_face = FaceLandmarkResult(
        has_face=False,
        landmarks=(),
        presence_confidence=0.0,
        timestamp_ms=5000,
        status=FaceLandmarkStatus.INSUFFICIENT_DATA,
        inference_latency_ms=0.0,
        reasons=("NO_FACE_DETECTED",),
    )
    features = extractor.extract(lost_face, frame_width=1280, frame_height=720)
    assert features.head_pose.zone == PoseZone.UNKNOWN
    assert features.head_pose.is_valid is False
    assert features.head_pose.is_looking_away is False

    # 4. Authentic extreme head poses -> strictly OUT_OF_BOUNDS
    # Extreme yaw (> 50°)
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=55.0) == PoseZone.OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=-55.0) == PoseZone.OUT_OF_BOUNDS
    # Extreme pitch (> 38°)
    assert classify_pose_zone(pitch_deg=45.0, yaw_deg=0.0) == PoseZone.OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=-45.0, yaw_deg=0.0) == PoseZone.OUT_OF_BOUNDS
    # Non-extreme angles are NOT OUT_OF_BOUNDS
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=20.0) == PoseZone.LEFT_MIRROR
    assert classify_pose_zone(pitch_deg=-15.0, yaw_deg=0.0) == PoseZone.DASHBOARD
    assert classify_pose_zone(pitch_deg=0.0, yaw_deg=0.0) == PoseZone.CENTER


def test_head_pose_pointwise_camera_depth_validation() -> None:
    """
    Verifies that validate_pnp_solution() enforces explicit point-wise camera-space depth:
        P_camera = R @ P_model + t
    requiring Z_camera > 0 for every 3D model point evaluated before projection.

    Validates:
    a) Valid solution where all model points have positive camera-space depth (Z_camera > 0).
    b) Positive tvec[2] (120mm >= min_depth_mm) but with a model point behind the lens (Z_camera <= 0).
    c) Model point with exactly zero camera-space depth (Z_camera == 0.0).
    d) Non-finite transformed coordinates (NaN / Inf).
    e) Both SQPnP and fallback ITERATIVE solvers pass through this validation.
    """
    w, h = 1280, 720
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)  # Identity rotation R = I

    # (a) Valid solution where all model points have positive camera depth
    tvec_valid = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
    pts_3d_valid = CANONICAL_3D_FACE_POINTS.copy()
    proj_valid, _ = cv2.projectPoints(pts_3d_valid, rvec, tvec_valid, cam_mat, dist)
    d_ref = float(np.linalg.norm(proj_valid[2] - proj_valid[3]))
    ok, mean_e, max_e = validate_pnp_solution(
        pts_3d_valid, proj_valid.reshape(-1, 2), rvec, tvec_valid, cam_mat, dist, d_ref
    )
    assert ok is True
    assert mean_e < 1.0
    assert max_e < 1.0

    # (b) Positive origin depth tvec[2] = 120mm (within [100, 3000]),
    # but a model point is behind the camera lens (Z_camera <= 0)
    # Model point with Z_model = -150mm -> Z_cam = -150mm + 120mm = -30mm < 0
    tvec_shallow = np.array([[0.0], [0.0], [120.0]], dtype=np.float64)
    pts_3d_behind = pts_3d_valid.copy()
    pts_3d_behind[0, 2] = -150.0  # Nose tip placed behind lens
    proj_behind, _ = cv2.projectPoints(pts_3d_behind, rvec, tvec_shallow, cam_mat, dist)
    ok_behind, mean_behind, max_behind = validate_pnp_solution(
        pts_3d_behind, proj_behind.reshape(-1, 2), rvec, tvec_shallow, cam_mat, dist, d_ref
    )
    assert ok_behind is False
    assert math.isinf(mean_behind)
    assert math.isinf(max_behind)

    # (c) Point with exactly zero camera-space depth (Z_camera == 0.0)
    # Model point with Z_model = -120mm -> Z_cam = -120mm + 120mm = 0.0
    pts_3d_zero = pts_3d_valid.copy()
    pts_3d_zero[0, 2] = -120.0
    proj_zero, _ = cv2.projectPoints(pts_3d_zero, rvec, tvec_shallow, cam_mat, dist)
    ok_zero, mean_zero, max_zero = validate_pnp_solution(
        pts_3d_zero, proj_zero.reshape(-1, 2), rvec, tvec_shallow, cam_mat, dist, d_ref
    )
    assert ok_zero is False
    assert math.isinf(mean_zero)
    assert math.isinf(max_zero)

    # (d) Non-finite transformed coordinates (NaN / Inf)
    pts_3d_nan = pts_3d_valid.copy()
    pts_3d_nan[0, 0] = np.nan
    ok_nan, mean_nan, max_nan = validate_pnp_solution(
        pts_3d_nan, proj_valid.reshape(-1, 2), rvec, tvec_valid, cam_mat, dist, d_ref
    )
    assert ok_nan is False
    assert math.isinf(mean_nan)
    assert math.isinf(max_nan)

    pts_3d_inf = pts_3d_valid.copy()
    pts_3d_inf[0, 2] = np.inf
    ok_inf, mean_inf, max_inf = validate_pnp_solution(
        pts_3d_inf, proj_valid.reshape(-1, 2), rvec, tvec_valid, cam_mat, dist, d_ref
    )
    assert ok_inf is False
    assert math.isinf(mean_inf)
    assert math.isinf(max_inf)


def test_pnp_solvers_enforce_pointwise_depth_rejection() -> None:
    """
    Verifies that both SQPnP and ITERATIVE fallback paths reject solutions
    where camera-space depth is non-positive or invalid, returning is_valid=False and PoseZone.UNKNOWN.
    """
    w, h = 1280, 720
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    # 1. Primary SQPnP validation on cranial points
    # When all points are strictly in front of lens, SQPnP succeeds
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, cam_mat, dist)
    mock_lms = {
        SOLVEPNP_LANDMARK_INDICES[i]: (float(proj[i, 0, 0] / w), float(proj[i, 0, 1] / h), 0.0)
        for i in range(len(SOLVEPNP_LANDMARK_INDICES))
    }
    lms = _build_mock_landmarks(mock_lms)
    res = estimate_head_pose(lms, frame_width=w, frame_height=h)
    assert res.is_valid is True
    assert res.zone == PoseZone.CENTER

    # 2. Corrupted cranial geometry that leads to non-positive depth or failed PnP
    mock_corrupted = mock_lms.copy()
    # Place nose tip at infinity/large negative coordinate
    mock_corrupted[1] = (0.01, 0.99, 0.0)
    mock_corrupted[6] = (0.99, 0.01, 0.0)
    lms_corrupted = _build_mock_landmarks(mock_corrupted)
    res_corrupted = estimate_head_pose(lms_corrupted, frame_width=w, frame_height=h)
    assert res_corrupted.is_valid is False
    assert res_corrupted.zone == PoseZone.UNKNOWN

