"""
DriveGuard — 2D Head-Pose Profile Fallback Regression Tests.

Directly targets the real-webcam LEFT_MIRROR -> CENTER defect observed in
`session_20261120_005232` (oblique mount neutral_yaw_deg = +27.32 deg,
neutral_pitch_deg = +6.56 deg, mirrored=True).

Covers:
  1. Normal frontal camera mounting (mirrored & unmirrored)
  2. Oblique mounting with +27.3 deg neutral yaw (real session parameters)
  3. LEFT_MIRROR glances at progressively larger angles (monotonic progression)
  4. RIGHT_MIRROR glances (mirrored & unmirrored, frontal & oblique)
  5. Seamless transition from valid 3D solvePnP to 2D profile fallback
  6. Extreme yaw and far-eye foreshortening bounds
  7. Invalid / missing / corrupted / inverted landmarks
  8. Incorrect / non-profile / contradictory nose ratios & anti-CENTER fabrication guard
  9. Continuous leftward head movement with zero LEFT_MIRROR -> CENTER flip
"""

from __future__ import annotations

import math
from typing import List
from unittest.mock import patch

import cv2
import numpy as np
import pytest

from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    PoseZone,
    SOLVEPNP_LANDMARK_INDICES,
    _estimate_2d_profile_fallback_yaw,
    build_neutral_matrix,
    estimate_head_pose,
)
from backend.app.vision.face_landmarker import NormalizedLandmark


# Real calibrated parameters from session_20261120_005232 (driver_default_profile.json)
REAL_SESSION_NEUTRAL_YAW_DEG = 27.32
REAL_SESSION_NEUTRAL_PITCH_DEG = 6.56
REAL_SESSION_NEUTRAL_ROT_MAT = np.array(
    [
        [0.886099, 0.082759, 0.456048],
        [-0.149305, 0.982449, 0.111813],
        [-0.438790, -0.167168, 0.882903],
    ],
    dtype=np.float64,
)


def _rot_mat_for_driver_pose(
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    is_mirrored: bool = True,
) -> np.ndarray:
    cam_yaw = yaw_deg if is_mirrored else -yaw_deg
    cam_pitch = -pitch_deg
    cam_roll = -roll_deg if is_mirrored else roll_deg
    psi, theta, phi = np.radians(cam_yaw), np.radians(cam_pitch), np.radians(cam_roll)
    Ry = np.array([[np.cos(psi), 0.0, np.sin(psi)], [0.0, 1.0, 0.0], [-np.sin(psi), 0.0, np.cos(psi)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def _project_canonical_landmarks(
    yaw_deg: float,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    is_mirrored: bool = True,
    frame_width: int = 1280,
    frame_height: int = 720,
    depth_mm: float = 600.0,
) -> List[NormalizedLandmark]:
    rot_mat = _rot_mat_for_driver_pose(yaw_deg, pitch_deg, roll_deg, is_mirrored=is_mirrored)
    focal = float(frame_width)
    K = np.array(
        [[focal, 0.0, frame_width / 2.0], [0.0, focal, frame_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rot_mat)
    tvec = np.array([[0.0], [0.0], [depth_mm]], dtype=np.float64)
    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, K, np.zeros((4, 1)))
    pts = proj.reshape(-1, 2)
    lms = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES):
        lms[idx] = NormalizedLandmark(float(pts[i, 0] / frame_width), float(pts[i, 1] / frame_height), 0.0)
    return lms


def _make_2d_profile_landmarks(
    nose_ratio: float,
    sellion_ratio: float,
    eye_span_px: float = 115.0,
    face_height_px: float = 200.0,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> List[NormalizedLandmark]:
    """Construct normalized landmarks with exact pixel eye_span_x, nose_ratio, and sellion_ratio."""
    left_eye_x = (frame_width - eye_span_px) / 2.0
    right_eye_x = left_eye_x + eye_span_px
    nose_x = left_eye_x + nose_ratio * eye_span_px
    sellion_x = left_eye_x + sellion_ratio * eye_span_px
    eye_y = frame_height * 0.42
    sellion_y = eye_y - 12.0
    nose_y = sellion_y + 0.48 * face_height_px
    chin_y = sellion_y + face_height_px
    glabella_y = sellion_y - 0.30 * face_height_px

    lms = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    px_map = {
        1: (nose_x, nose_y),
        152: (sellion_x, chin_y),
        33: (left_eye_x, eye_y),
        263: (right_eye_x, eye_y),
        61: (left_eye_x + 0.20 * eye_span_px, nose_y + 40.0),
        291: (left_eye_x + 0.80 * eye_span_px, nose_y + 40.0),
        6: (sellion_x, sellion_y),
        10: (sellion_x, glabella_y),
    }
    for idx, (px, py) in px_map.items():
        lms[idx] = NormalizedLandmark(px / frame_width, py / frame_height, 0.0)
    return lms


# ---------------------------------------------------------------------------
# 1. Frontal Camera Mounting (Mirrored & Unmirrored)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("is_mirrored", [True, False])
def test_frontal_camera_fallback_left_and_right_profile(is_mirrored: bool) -> None:
    """
    On a frontal camera (neutral_yaw=0), extreme profile landmarks (55 deg left / 55 deg right)
    falling back from 3D PnP must preserve exact sign and classify as off-road (never CENTER).
    """
    lms_left = _project_canonical_landmarks(yaw_deg=55.0, is_mirrored=is_mirrored)
    lms_right = _project_canonical_landmarks(yaw_deg=-55.0, is_mirrored=is_mirrored)

    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        res_l = estimate_head_pose(
            lms_left,
            neutral_yaw_deg=0.0,
            neutral_pitch_deg=0.0,
            is_mirrored=is_mirrored,
        )
        assert res_l.is_valid is True
        assert res_l.rot_mat is None
        assert res_l.yaw_deg == pytest.approx(55.0, abs=3.0)
        assert res_l.rel_yaw_deg == pytest.approx(55.0, abs=3.0)
        assert res_l.zone in (PoseZone.LEFT_MIRROR, PoseZone.OUT_OF_BOUNDS)
        assert res_l.zone != PoseZone.CENTER
        assert res_l.is_looking_away is True

        res_r = estimate_head_pose(
            lms_right,
            neutral_yaw_deg=0.0,
            neutral_pitch_deg=0.0,
            is_mirrored=is_mirrored,
        )
        assert res_r.is_valid is True
        assert res_r.rot_mat is None
        assert res_r.yaw_deg == pytest.approx(-55.0, abs=3.0)
        assert res_r.rel_yaw_deg == pytest.approx(-55.0, abs=3.0)
        assert res_r.zone in (PoseZone.RIGHT_MIRROR, PoseZone.OUT_OF_BOUNDS)
        assert res_r.zone != PoseZone.CENTER
        assert res_r.is_looking_away is True


# ---------------------------------------------------------------------------
# 2. Oblique Mounting (+27.3 deg Neutral Yaw — Real Session Reference)
# ---------------------------------------------------------------------------

def test_oblique_mount_27_3_real_session_fallback_stays_left_mirror() -> None:
    """
    Reproduces the exact real-session landmark ratios from `session_20261120_005232`:
      - Frame 88:  nose_ratio=+0.002, sellion_ratio=+0.231, eye_span_x=119.7px
      - Frame 95:  nose_ratio=-0.092, sellion_ratio=+0.185, eye_span_x=109.5px
      - Frame 506: nose_ratio=+0.019, sellion_ratio=+0.224, eye_span_x=113.4px
    Previously all three snapped to raw_yaw=+28.0 -> rel_yaw=+0.7 -> CENTER!
    With geometric 2D fallback, all three must yield raw_yaw ~ +48..56 deg -> rel_yaw ~ +21..28 deg -> LEFT_MIRROR.
    """
    real_session_samples = [
        # (frame_label, nose_ratio, sellion_ratio, eye_span_px, expected_min_raw, expected_max_raw)
        ("frame_88", 0.002, 0.231, 119.7, 47.0, 54.0),
        ("frame_95", -0.092, 0.185, 109.5, 52.0, 59.0),
        ("frame_506", 0.019, 0.224, 113.4, 46.0, 53.0),
    ]

    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        for label, nr, sr, span_px, min_raw, max_raw in real_session_samples:
            lms = _make_2d_profile_landmarks(nose_ratio=nr, sellion_ratio=sr, eye_span_px=span_px)
            res = estimate_head_pose(
                lms,
                frame_width=1280,
                frame_height=720,
                neutral_pitch_deg=REAL_SESSION_NEUTRAL_PITCH_DEG,
                neutral_yaw_deg=REAL_SESSION_NEUTRAL_YAW_DEG,
                neutral_rot_mat=REAL_SESSION_NEUTRAL_ROT_MAT,
                is_mirrored=True,
                current_zone=PoseZone.LEFT_MIRROR,
                neutral_is_mirrored=True,
            )
            assert res.is_valid is True, f"{label} should produce valid fallback pose"
            assert res.rot_mat is None
            assert min_raw <= res.yaw_deg <= max_raw, f"{label}: unexpected raw yaw {res.yaw_deg:.1f}"
            assert 18.0 <= res.rel_yaw_deg <= 32.0, f"{label}: unexpected rel yaw {res.rel_yaw_deg:.1f}"
            assert res.zone == PoseZone.LEFT_MIRROR, f"{label} regressed to {res.zone} instead of LEFT_MIRROR"
            assert res.is_looking_away is True


# ---------------------------------------------------------------------------
# 3 & 4. Progressive LEFT_MIRROR & RIGHT_MIRROR Angles
# ---------------------------------------------------------------------------

def test_progressive_left_mirror_glances_are_monotonic_and_never_flip() -> None:
    """
    As the driver turns progressively further left on an oblique +27.3 deg mount
    (camera yaw 52 -> 56 -> 60 -> 64 deg), fallback yaw must increase monotonically
    and remain in LEFT_MIRROR (never regressing to CENTER).
    """
    prev_yaw = None
    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        for true_cam_yaw in (52.0, 56.0, 60.0, 64.0):
            lms = _project_canonical_landmarks(yaw_deg=true_cam_yaw, is_mirrored=True)
            res = estimate_head_pose(
                lms,
                neutral_pitch_deg=REAL_SESSION_NEUTRAL_PITCH_DEG,
                neutral_yaw_deg=REAL_SESSION_NEUTRAL_YAW_DEG,
                neutral_rot_mat=REAL_SESSION_NEUTRAL_ROT_MAT,
                is_mirrored=True,
                current_zone=PoseZone.LEFT_MIRROR,
                neutral_is_mirrored=True,
            )
            assert res.is_valid is True
            assert res.zone == PoseZone.LEFT_MIRROR
            assert res.yaw_deg == pytest.approx(true_cam_yaw, abs=3.0)
            if prev_yaw is not None:
                assert res.yaw_deg > prev_yaw, "Fallback yaw must increase monotonically with leftward turn"
            prev_yaw = res.yaw_deg


@pytest.mark.parametrize("is_mirrored", [True, False])
def test_right_mirror_glances_oblique_and_frontal(is_mirrored: bool) -> None:
    """
    Extreme rightward glances (-54 deg camera yaw) on an oblique camera (-25 deg neutral yaw)
    must yield ~-29 deg relative yaw -> RIGHT_MIRROR in both mirrored and unmirrored modes.
    """
    r_neut = build_neutral_matrix(neutral_pitch_deg=0.0, neutral_yaw_deg=-25.0, is_mirrored=is_mirrored)
    lms = _project_canonical_landmarks(yaw_deg=-54.0, is_mirrored=is_mirrored)

    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        res = estimate_head_pose(
            lms,
            neutral_pitch_deg=0.0,
            neutral_yaw_deg=-25.0,
            neutral_rot_mat=r_neut,
            is_mirrored=is_mirrored,
            current_zone=PoseZone.RIGHT_MIRROR,
            neutral_is_mirrored=is_mirrored,
        )
        assert res.is_valid is True
        assert res.rot_mat is None
        assert res.yaw_deg == pytest.approx(-54.0, abs=3.0)
        assert res.rel_yaw_deg == pytest.approx(-29.0, abs=3.0)
        assert res.zone == PoseZone.RIGHT_MIRROR
        assert res.is_looking_away is True


# ---------------------------------------------------------------------------
# 5 & 9. Continuous Leftward Sweep Across 3D PnP -> 2D Fallback Transition
# ---------------------------------------------------------------------------

def test_continuous_leftward_sweep_pnp_to_fallback_has_no_jump_or_center_flip() -> None:
    """
    Simulate a continuous head turn from forward driving (rel_yaw = 0 deg, raw = +27.3 deg)
    to deep LEFT_MIRROR glance (rel_yaw = +36 deg, raw = +63.3 deg), switching from real
    3D solvePnP (<= 51 deg raw) to 2D fallback (> 51 deg raw).

    Verifies:
      - Smooth yaw continuity across the 3D PnP -> 2D fallback boundary (step jump < 6 deg).
      - Once LEFT_MIRROR is entered, zone NEVER flips back to CENTER.
    """
    current_zone = PoseZone.CENTER
    entered_left_mirror = False
    prev_rel_yaw = None

    for rel_step in range(0, 38, 2):
        cam_yaw = REAL_SESSION_NEUTRAL_YAW_DEG + float(rel_step)
        lms = _project_canonical_landmarks(
            yaw_deg=cam_yaw,
            pitch_deg=REAL_SESSION_NEUTRAL_PITCH_DEG,
            is_mirrored=True,
        )

        # Real 3D solvePnP succeeds up to ~51 deg camera yaw, then fails past 51 deg (matching session_20261120_005232)
        if cam_yaw > 51.0:
            with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
                res = estimate_head_pose(
                    lms,
                    neutral_pitch_deg=REAL_SESSION_NEUTRAL_PITCH_DEG,
                    neutral_yaw_deg=REAL_SESSION_NEUTRAL_YAW_DEG,
                    neutral_rot_mat=REAL_SESSION_NEUTRAL_ROT_MAT,
                    is_mirrored=True,
                    current_zone=current_zone,
                    neutral_is_mirrored=True,
                )
            assert res.rot_mat is None, "Should be on 2D fallback past 51 deg camera yaw"
        else:
            res = estimate_head_pose(
                lms,
                neutral_pitch_deg=REAL_SESSION_NEUTRAL_PITCH_DEG,
                neutral_yaw_deg=REAL_SESSION_NEUTRAL_YAW_DEG,
                neutral_rot_mat=REAL_SESSION_NEUTRAL_ROT_MAT,
                is_mirrored=True,
                current_zone=current_zone,
                neutral_is_mirrored=True,
            )
            assert res.rot_mat is not None, "Should be on 3D PnP at <= 51 deg camera yaw"

        assert res.is_valid is True
        if prev_rel_yaw is not None:
            step_delta = res.rel_yaw_deg - prev_rel_yaw
            assert -1.5 <= step_delta <= 6.0, (
                f"Discontinuous yaw jump at cam_yaw={cam_yaw:.1f}: "
                f"prev_rel={prev_rel_yaw:.1f} -> cur_rel={res.rel_yaw_deg:.1f}"
            )
        if res.zone == PoseZone.LEFT_MIRROR:
            entered_left_mirror = True
        if entered_left_mirror:
            assert res.zone != PoseZone.CENTER, (
                f"Contradictory LEFT_MIRROR -> CENTER flip at cam_yaw={cam_yaw:.1f} (rel={res.rel_yaw_deg:.1f})"
            )

        current_zone = res.zone
        prev_rel_yaw = res.rel_yaw_deg

    assert entered_left_mirror is True


# ---------------------------------------------------------------------------
# 6. Extreme Yaw & Far-Eye Foreshortening Handling
# ---------------------------------------------------------------------------

def test_extreme_yaw_and_far_eye_foreshortening_bounded_and_safe() -> None:
    """
    At extreme profile angles (~68-75 deg) with severe far-eye foreshortening,
    fallback yaw remains finite, bounded (<= 75 deg), and classifies as OUT_OF_BOUNDS on a frontal cam
    (or LEFT_MIRROR on a +27.3 deg oblique cam). When eyes collapse below minimum aspect ratio,
    it cleanly rejects as UNKNOWN rather than dividing by near-zero.
    """
    # Extreme valid foreshortening (eye_span_px=55px, face_height=200px -> ratio=0.275 >= 0.12)
    lms_extreme = _make_2d_profile_landmarks(
        nose_ratio=-0.35, sellion_ratio=0.10, eye_span_px=55.0, face_height_px=200.0
    )
    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        res_frontal = estimate_head_pose(lms_extreme, neutral_yaw_deg=0.0, is_mirrored=True)
        assert res_frontal.is_valid is True
        assert math.isfinite(res_frontal.yaw_deg)
        assert 55.0 <= res_frontal.yaw_deg <= 75.0
        assert res_frontal.zone == PoseZone.OUT_OF_BOUNDS

    # Collapsed far-eye foreshortening (eye_span_px=15px vs face_height=220px -> ratio=0.068 < 0.12) -> reject!
    lms_collapsed = _make_2d_profile_landmarks(
        nose_ratio=-0.20, sellion_ratio=0.15, eye_span_px=15.0, face_height_px=220.0
    )
    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        res_col = estimate_head_pose(lms_collapsed, neutral_yaw_deg=0.0, is_mirrored=True)
        assert res_col.is_valid is False
        assert res_col.zone == PoseZone.UNKNOWN


# ---------------------------------------------------------------------------
# 7 & 8. Invalid Landmarks, Bad Nose Ratios & Anti-Fabricated-CENTER Guard
# ---------------------------------------------------------------------------

def test_invalid_landmarks_and_corrupted_nose_ratios_rejected_cleanly() -> None:
    """
    Malformed landmarks, inverted eyes, non-profile nose ratios, out-of-range nose ratios,
    and contradictory nose-vs-sellion cues must ALL return invalid_result (is_valid=False, UNKNOWN)
    and NEVER fabricate CENTER.
    """
    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        # 1. Frontal face with PnP failure (nose_ratio = 0.50, not profile) -> must NOT fabricate CENTER
        lms_frontal_fail = _make_2d_profile_landmarks(nose_ratio=0.50, sellion_ratio=0.50)
        r1 = estimate_head_pose(lms_frontal_fail, is_mirrored=True)
        assert r1.is_valid is False and r1.zone == PoseZone.UNKNOWN

        # 2. Wildly out-of-range nose_ratio (-2.5) -> rejected
        lms_wild_nr = _make_2d_profile_landmarks(nose_ratio=-2.5, sellion_ratio=0.20)
        r2 = estimate_head_pose(lms_wild_nr, is_mirrored=True)
        assert r2.is_valid is False and r2.zone == PoseZone.UNKNOWN

        # 3. Contradictory nose vs. sellion direction (nose_ratio=-0.05 says LEFT, sellion_x < nose_x says RIGHT)
        lms_contradict = _make_2d_profile_landmarks(nose_ratio=-0.05, sellion_ratio=-0.25)
        r3 = estimate_head_pose(lms_contradict, is_mirrored=True)
        assert r3.is_valid is False and r3.zone == PoseZone.UNKNOWN

        # 4. Inverted eyes (left_eye_x > right_eye_x -> negative eye_span_x)
        lms_inv = _make_2d_profile_landmarks(nose_ratio=-0.05, sellion_ratio=0.20)
        lms_inv[33], lms_inv[263] = lms_inv[263], lms_inv[33]
        r4 = estimate_head_pose(lms_inv, is_mirrored=True)
        assert r4.is_valid is False and r4.zone == PoseZone.UNKNOWN

        # 5. Broken vertical ordering (chin above nose)
        lms_bad_vert = _make_2d_profile_landmarks(nose_ratio=-0.05, sellion_ratio=0.20)
        lms_bad_vert[152] = NormalizedLandmark(lms_bad_vert[152].x, lms_bad_vert[1].y - 0.10, 0.0)
        r5 = estimate_head_pose(lms_bad_vert, is_mirrored=True)
        assert r5.is_valid is False and r5.zone == PoseZone.UNKNOWN

        # 6. NaN landmark coordinate
        lms_nan = _make_2d_profile_landmarks(nose_ratio=-0.05, sellion_ratio=0.20)
        lms_nan[1] = NormalizedLandmark(float("nan"), 0.5, 0.0)
        r6 = estimate_head_pose(lms_nan, is_mirrored=True)
        assert r6.is_valid is False and r6.zone == PoseZone.UNKNOWN

        # 7. Missing / insufficient landmarks
        r7 = estimate_head_pose([], is_mirrored=True)
        assert r7.is_valid is False and r7.zone == PoseZone.UNKNOWN

        # 8. Anti-fabricated-CENTER guard: even if fallback produces valid ~+52 deg raw yaw,
        #    if neutral_yaw is absurdly extreme (+52 deg) so rel_yaw lands in CENTER, reject as UNKNOWN!
        lms_valid_prof = _make_2d_profile_landmarks(nose_ratio=-0.04, sellion_ratio=0.21)
        r8 = estimate_head_pose(lms_valid_prof, neutral_yaw_deg=52.0, is_mirrored=True)
        assert r8.is_valid is False and r8.zone == PoseZone.UNKNOWN
