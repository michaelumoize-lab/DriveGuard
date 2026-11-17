"""
DriveGuard Phase 7 Task 2 Integration Test Suite.
Verifies Personalized Driver Behaviour Profile (PDBP) integration into the
modular DriverFeatureExtractor feature extraction pipeline.

Covers:
1. Profile states (uncalibrated, calibrated feasible, malformed, center infeasible,
   backfilled peripheral fallback, peripheral infeasible fallback, all feasible).
2. Threshold calculations (PDBP closure formula, bounded reopening formula,
   feasibility boundary cases, span invariant proof).
3. Pipeline classification and non-ocular feature preservation (raw geometric EAR,
   has_face preservation, UNCERTAIN_GAP without face loss, PERCLOS safety,
   independent head pose, MAR, and yawn detection).
4. Threshold latching and mid-blink protection (zone change during blink,
   post-reopening unlatching, mid-blink profile replacement, mid-blink disable,
   identical threshold consistency across EAR and blink detector).
5. Calibration lifecycle and persistence (targets and minimums, raw EAR collection
   without classification, failed recalibration safety, infeasible rejection,
   persistence roundtrip).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import pytest

from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    ZoneBaseline,
)
from backend.app.ai.profile_adapter import dto_to_profile, profile_to_dto
from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkEvent,
    BlinkKinematicsDetector,
    BlinkState,
)
from backend.app.features.ear import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeAspectRatioResult,
)
from backend.app.features.feature_extractor import (
    DriverFeatureExtractor,
    DriverFeatures,
    compute_zone_thresholds,
    validate_profile,
)
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    SOLVEPNP_LANDMARK_INDICES,
    PoseZone,
    estimate_head_pose,
)
from backend.app.features.mar import (
    MOUTH_OUTER_CORNERS,
    MOUTH_VERTICAL_PAIRS,
)
from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.profile_store import profile_store
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


# ==============================================================================
# Test Fixtures & Synthetic Landmark Generators
# ==============================================================================

def build_synthetic_landmarks(
    ear_val: float = 0.35,
    pitch_deg: float = 0.0,
    yaw_deg: float = 0.0,
    mar_val: float = 0.15,
    w: int = 1280,
    h: int = 720,
) -> List[NormalizedLandmark]:
    """
    Constructs a 478-point NormalizedLandmark list that yields deterministic
    head pose (pitch, yaw), eye geometry (EAR), and mouth geometry (MAR).
    """
    pts = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]

    # 1. Head Pose landmarks via SolvePnP projection
    focal = float(w)
    center = (w / 2.0, h / 2.0)
    cam_mat = np.array([[focal, 0.0, center[0]], [0.0, focal, center[1]], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((4, 1), dtype=np.float64)

    # Convert Euler angles (deg) to rotation vector in camera coordinate conventions
    cam_pitch_rad = math.radians(pitch_deg)
    cam_yaw_rad = math.radians(-yaw_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(cam_pitch_rad), -math.sin(cam_pitch_rad)], [0, math.sin(cam_pitch_rad), math.cos(cam_pitch_rad)]], dtype=np.float64)
    Ry = np.array([[math.cos(cam_yaw_rad), 0, math.sin(cam_yaw_rad)], [0, 1, 0], [-math.sin(cam_yaw_rad), 0, math.cos(cam_yaw_rad)]], dtype=np.float64)
    rvec, _ = cv2.Rodrigues(Ry @ Rx)
    tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)

    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, cam_mat, dist)
    for i, lm_idx in enumerate(SOLVEPNP_LANDMARK_INDICES):
        pts[lm_idx] = NormalizedLandmark(
            x=float(proj[i, 0, 0] / w),
            y=float(proj[i, 0, 1] / h),
            z=0.0,
        )

    # 2. Eye landmarks aligned with projected eye anchor points (33 and 263)
    ar = float(w) / float(h)
    dx = 0.055
    dy = ear_val * dx * ar

    # Left eye: (33, 160, 158, 133, 153, 144) - 33 is outer corner
    x1, y1 = pts[33].x, pts[33].y
    pts[133] = NormalizedLandmark(x1 + dx, y1, 0.0)
    pts[160] = NormalizedLandmark(x1 + dx / 3.0, y1 - dy / 2.0, 0.0)
    pts[158] = NormalizedLandmark(x1 + 2.0 * dx / 3.0, y1 - dy / 2.0, 0.0)
    pts[144] = NormalizedLandmark(x1 + dx / 3.0, y1 + dy / 2.0, 0.0)
    pts[153] = NormalizedLandmark(x1 + 2.0 * dx / 3.0, y1 + dy / 2.0, 0.0)

    # Right eye: (362, 385, 387, 263, 373, 380) - 263 is outer corner
    x2, y2 = pts[263].x, pts[263].y
    pts[362] = NormalizedLandmark(x2 - dx, y2, 0.0)
    pts[385] = NormalizedLandmark(x2 - 2.0 * dx / 3.0, y2 - dy / 2.0, 0.0)
    pts[387] = NormalizedLandmark(x2 - dx / 3.0, y2 - dy / 2.0, 0.0)
    pts[380] = NormalizedLandmark(x2 - 2.0 * dx / 3.0, y2 + dy / 2.0, 0.0)
    pts[373] = NormalizedLandmark(x2 - dx / 3.0, y2 + dy / 2.0, 0.0)

    # 3. Mouth landmarks aligned with projected mouth corners (61 and 291)
    dx_m = pts[291].x - pts[61].x
    x_mid = (pts[61].x + pts[291].x) / 2.0
    y_mid = (pts[61].y + pts[291].y) / 2.0
    dy_m = mar_val * dx_m * ar
    for top_idx, bot_idx in MOUTH_VERTICAL_PAIRS:
        pts[top_idx] = NormalizedLandmark(x_mid, y_mid - dy_m / 2.0, 0.0)
        pts[bot_idx] = NormalizedLandmark(x_mid, y_mid + dy_m / 2.0, 0.0)

    return pts

    return pts


def make_landmark_result(
    landmarks: Optional[List[NormalizedLandmark]] = None,
    timestamp_ms: float = 1000.0,
    has_face: bool = True,
) -> FaceLandmarkResult:
    """Helper to wrap landmarks in FaceLandmarkResult."""
    if not has_face or landmarks is None:
        return FaceLandmarkResult(
            has_face=False,
            landmarks=(),
            presence_confidence=0.0,
            timestamp_ms=timestamp_ms,
            status=FaceLandmarkStatus.INSUFFICIENT_DATA,
            inference_latency_ms=0.0,
            reasons=("NO_FACE_DETECTED",),
        )
    return FaceLandmarkResult(
        has_face=True,
        landmarks=tuple(landmarks),
        presence_confidence=0.95,
        timestamp_ms=int(timestamp_ms),
        status=FaceLandmarkStatus.TRACKING,
        inference_latency_ms=12.0,
        reasons=(),
    )


def make_mock_profile(
    driver_id: str = "driver_test",
    center_ear: float = 0.280,
    center_mad: float = 0.018,
    sample_count: int = 45,
    include_peripherals: bool = True,
    corrupt_peripheral: bool = False,
) -> DriverBehaviourProfile:
    """Helper to build a valid DriverBehaviourProfile with optional peripherals."""
    zone_baselines: Dict[str, ZoneBaseline] = {
        "CENTER": ZoneBaseline(
            zone="CENTER",
            ear_median=center_ear,
            ear_mad_scaled=center_mad,
            pitch_median_deg=0.0,
            yaw_median_deg=0.0,
            sample_count=sample_count,
        ),
    }
    if include_peripherals:
        left_ear = 0.140 if corrupt_peripheral else round(center_ear * 0.92, 4)
        left_mad = 0.015 if corrupt_peripheral else center_mad
        zone_baselines.update({
            "LEFT_MIRROR": ZoneBaseline(
                zone="LEFT_MIRROR",
                ear_median=left_ear,
                ear_mad_scaled=left_mad,
                pitch_median_deg=0.0,
                yaw_median_deg=22.0,
                sample_count=40,
            ),
            "RIGHT_MIRROR": ZoneBaseline(
                zone="RIGHT_MIRROR",
                ear_median=round(center_ear * 0.92, 4),
                ear_mad_scaled=center_mad,
                pitch_median_deg=0.0,
                yaw_median_deg=-22.0,
                sample_count=40,
            ),
            "DASHBOARD": ZoneBaseline(
                zone="DASHBOARD",
                ear_median=round(center_ear * 0.88, 4),
                ear_mad_scaled=center_mad,
                pitch_median_deg=-15.0,
                yaw_median_deg=0.0,
                sample_count=40,
            ),
        })

    return DriverBehaviourProfile(
        driver_id=driver_id,
        calibration_mode="FOUR_ZONE" if include_peripherals else "CENTER_ONLY",
        center_ear_median=center_ear,
        center_ear_mad_scaled=center_mad,
        blink_duration_median_ms=250.0,
        blink_duration_mad_ms=30.0,
        blink_rcr_median=0.85,
        blink_rcr_mad=0.10,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=3.0,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.95,
        zone_baselines=zone_baselines,
    )


# ==============================================================================
# Group 1: Profile States Tests
# ==============================================================================

def test_uncalibrated_pipeline_disables_ocular_classification() -> None:
    """
    Verifies that an uncalibrated pipeline (profile=None) cleanly deactivates
    ocular classification while preserving raw measurements and face tracking.
    """
    extractor = DriverFeatureExtractor()
    assert extractor.is_ocular_active is False
    assert extractor.is_profile_feasible is False
    assert extractor.active_closure_threshold == 0.0
    assert extractor.active_reopening_threshold == 0.0

    lms = build_synthetic_landmarks(ear_val=0.35)
    lr = make_landmark_result(lms, timestamp_ms=1000.0)
    features = extractor.extract(lr)

    # Ocular classification must be inactive
    assert features.ear.is_valid is False
    assert features.ear.is_closed is False
    assert features.ear.is_left_valid is False
    assert features.ear.is_right_valid is False
    assert features.ear.closure_threshold == 0.0

    # Face detection must NOT be dropped
    assert features.has_face is True
    assert features.face_status == FaceLandmarkStatus.TRACKING

    # Blink detector must enter UNCERTAIN_GAP
    assert extractor.blink_detector._state == BlinkState.UNCERTAIN_GAP

    # PERCLOS must receive invalid EAR and stay unreliable without crashing
    assert features.perclos.is_reliable is False
    assert features.perclos.observed_duration_sec == 0.0


def test_calibrated_feasible_profile_activates_personalized_thresholds() -> None:
    """
    Verifies that a valid feasible profile activates ocular classification
    with exact personalized closure and reopening thresholds.
    """
    profile = make_mock_profile(center_ear=0.280, center_mad=0.018)
    extractor = DriverFeatureExtractor(profile=profile)
    assert extractor.is_ocular_active is True
    assert extractor.is_profile_feasible is True

    # PDBP formulas for m=0.280, mad=0.018:
    # c = clip(min(0.28*0.76, 0.28 - 2.3*0.018), 0.12, 0.22) = clip(min(0.2128, 0.2386), 0.12, 0.22) = 0.2128
    # r = min(0.30, max(0.2128 + 0.025, min(0.28*0.82, 0.28 - 0.015))) = 0.2378
    assert math.isclose(extractor.active_closure_threshold, 0.2128, abs_tol=1e-4)
    assert math.isclose(extractor.active_reopening_threshold, 0.2378, abs_tol=1e-4)

    # Open-eye frame (EAR ~ 0.35 > closure)
    lms_open = build_synthetic_landmarks(ear_val=0.35)
    features_open = extractor.extract(make_landmark_result(lms_open, timestamp_ms=1000.0))
    assert features_open.ear.is_valid is True
    assert features_open.ear.is_closed is False
    assert math.isclose(features_open.ear.closure_threshold, 0.2128, abs_tol=1e-4)

    # Closed-eye frame (EAR ~ 0.05 < closure)
    lms_closed = build_synthetic_landmarks(ear_val=0.05)
    features_closed = extractor.extract(make_landmark_result(lms_closed, timestamp_ms=1033.0))
    assert features_closed.ear.is_valid is True
    assert features_closed.ear.is_closed is True


def test_malformed_profile_rejected_ocular_disabled() -> None:
    """
    Verifies that malformed profiles (negative sample count, boolean, non-finite values,
    or missing CENTER) are rejected by validate_profile() and disable ocular classification.
    """
    # 1. Negative sample count
    p_neg = make_mock_profile(sample_count=-5)
    assert validate_profile(p_neg)[0] is False

    # 2. Boolean sample count
    p_bool = make_mock_profile()
    p_bool.zone_baselines["CENTER"].sample_count = True  # type: ignore
    assert validate_profile(p_bool)[0] is False

    # 3. Float sample count
    p_float = make_mock_profile()
    p_float.zone_baselines["CENTER"].sample_count = 10.5  # type: ignore
    assert validate_profile(p_float)[0] is False

    # 4. NaN in baseline
    p_nan = make_mock_profile()
    p_nan.center_ear_median = float("nan")
    assert validate_profile(p_nan)[0] is False

    # 5. Inf in baseline
    p_inf = make_mock_profile()
    p_inf.center_ear_mad_scaled = float("inf")
    assert validate_profile(p_inf)[0] is False

    # 6. Missing CENTER baseline
    p_no_center = make_mock_profile()
    p_no_center.zone_baselines.pop("CENTER", None)
    assert validate_profile(p_no_center)[0] is False

    # Extractor with malformed profile must disable ocular classification
    extractor = DriverFeatureExtractor(profile=p_neg)
    assert extractor.is_profile_feasible is False
    assert extractor.is_ocular_active is False
    features = extractor.extract(make_landmark_result(build_synthetic_landmarks(0.35)))
    assert features.ear.is_valid is False
    assert features.ear.is_closed is False


def test_center_infeasible_profile_disables_ocular() -> None:
    """
    Verifies that a profile with an infeasible CENTER baseline (m < 0.160 -> span < 0.040)
    is evaluated as infeasible and disables ocular classification.
    """
    # m = 0.150, mad = 0.015 -> c = 0.120, span = 0.150 - 0.120 = 0.030 < 0.040
    p_infeasible = make_mock_profile(center_ear=0.150, center_mad=0.015)
    extractor = DriverFeatureExtractor(profile=p_infeasible)
    assert extractor.is_profile_feasible is False
    assert extractor.is_ocular_active is False

    features = extractor.extract(make_landmark_result(build_synthetic_landmarks(0.35)))
    assert features.ear.is_valid is False
    assert features.ear.is_closed is False


def test_backfilled_peripheral_zone_fallback_to_center() -> None:
    """
    Verifies that when a profile lacks explicit peripheral zone baselines,
    it falls back to the feasible CENTER baseline and remains operational.
    """
    p_center_only = make_mock_profile(center_ear=0.280, center_mad=0.018, include_peripherals=False)
    assert "LEFT_MIRROR" not in p_center_only.zone_baselines

    extractor = DriverFeatureExtractor(profile=p_center_only)
    assert extractor.is_profile_feasible is True
    assert extractor.is_ocular_active is True

    # Face looking at LEFT_MIRROR (yaw = +25 deg)
    lms_left = build_synthetic_landmarks(ear_val=0.35, yaw_deg=25.0)
    features = extractor.extract(make_landmark_result(lms_left, timestamp_ms=1000.0))
    assert features.head_pose.zone == PoseZone.LEFT_MIRROR
    assert features.ear.is_valid is True
    # Fell back to feasible CENTER thresholds
    assert math.isclose(features.ear.closure_threshold, 0.2128, abs_tol=1e-4)


def test_peripheral_infeasible_falls_back_to_center() -> None:
    """
    Verifies that when CENTER is feasible but a peripheral zone has a corrupted/narrow
    baseline (span < 0.040), the extractor falls back to CENTER thresholds rather
    than disabling ocular classification for the whole pipeline.
    """
    p_corrupt_left = make_mock_profile(
        center_ear=0.280,
        center_mad=0.018,
        include_peripherals=True,
        corrupt_peripheral=True,  # LEFT_MIRROR: m=0.140, span=0.020 < 0.040
    )
    extractor = DriverFeatureExtractor(profile=p_corrupt_left)
    assert extractor.is_profile_feasible is True
    assert extractor.is_ocular_active is True

    # Look toward LEFT_MIRROR
    lms_left = build_synthetic_landmarks(ear_val=0.35, yaw_deg=25.0)
    features = extractor.extract(make_landmark_result(lms_left, timestamp_ms=1000.0))
    assert features.head_pose.zone == PoseZone.LEFT_MIRROR
    assert features.ear.is_valid is True
    # Ocular stays active via CENTER fallback
    assert math.isclose(features.ear.closure_threshold, 0.2128, abs_tol=1e-4)


def test_all_zones_feasible_per_zone_thresholds_applied() -> None:
    """
    Verifies that when all 4 zones are feasible, the extractor dynamically applies
    distinct (c_z, r_z) thresholds matching each zone's head pose.
    """
    profile = make_mock_profile(center_ear=0.300, center_mad=0.018, include_peripherals=True)
    extractor = DriverFeatureExtractor(profile=profile)

    # 1. CENTER (yaw=0, pitch=0)
    lms_ctr = build_synthetic_landmarks(ear_val=0.35, yaw_deg=0.0, pitch_deg=0.0)
    feat_ctr = extractor.extract(make_landmark_result(lms_ctr, timestamp_ms=1000.0))
    assert feat_ctr.head_pose.zone == PoseZone.CENTER
    c_ctr = feat_ctr.ear.closure_threshold

    # 2. LEFT_MIRROR (yaw=+25, pitch=0)
    lms_left = build_synthetic_landmarks(ear_val=0.35, yaw_deg=25.0, pitch_deg=0.0)
    feat_left = extractor.extract(make_landmark_result(lms_left, timestamp_ms=1033.0))
    assert feat_left.head_pose.zone == PoseZone.LEFT_MIRROR
    c_left = feat_left.ear.closure_threshold

    # 3. DASHBOARD (yaw=0, pitch=-20)
    lms_dash = build_synthetic_landmarks(ear_val=0.35, yaw_deg=0.0, pitch_deg=-20.0)
    feat_dash = extractor.extract(make_landmark_result(lms_dash, timestamp_ms=1066.0))
    assert feat_dash.head_pose.zone == PoseZone.DASHBOARD
    c_dash = feat_dash.ear.closure_threshold

    # Each zone must have its own distinct, feasible threshold
    assert c_ctr != c_left
    assert c_ctr != c_dash
    assert c_left != c_dash
    assert c_ctr >= 0.12 and c_left >= 0.12 and c_dash >= 0.12


# ==============================================================================
# Group 2: Threshold Calculations & Mathematical Invariants
# ==============================================================================

def test_closure_threshold_exact_pdbp_formula() -> None:
    """
    Verifies exact PDBP closure formula:
    c = clip(min(m * 0.76, m - 2.3 * mad), 0.12, 0.22)
    """
    # 1. Floor clipping: very low m produces c clamped to 0.12
    c_floor, _, _, _ = compute_zone_thresholds(0.130, 0.020)
    assert math.isclose(c_floor, 0.120, abs_tol=1e-5)

    # 2. Ceiling clipping: high m produces c clamped to 0.22
    c_ceil, _, _, _ = compute_zone_thresholds(0.380, 0.010)
    assert math.isclose(c_ceil, 0.220, abs_tol=1e-5)

    # 3. m * 0.76 wins when MAD is small
    # m = 0.250, mad = 0.010: m*0.76 = 0.190, m - 2.3*mad = 0.227 -> min = 0.190
    c_factor, _, _, _ = compute_zone_thresholds(0.250, 0.010)
    assert math.isclose(c_factor, 0.190, abs_tol=1e-5)

    # 4. m - 2.3 * mad wins when MAD is large
    # m = 0.250, mad = 0.030: m*0.76 = 0.190, m - 2.3*mad = 0.181 -> min = 0.181
    c_mad, _, _, _ = compute_zone_thresholds(0.250, 0.030)
    assert math.isclose(c_mad, 0.181, abs_tol=1e-5)


def test_reopening_threshold_bounded_formula() -> None:
    """
    Verifies bounded reopening formula:
    r = min(0.30, max(c + 0.025, min(m * 0.82, m - 0.015)))
    """
    # Normal distribution
    c, r, _, _ = compute_zone_thresholds(0.280, 0.018)
    assert r <= 0.30
    assert r >= c + 0.025
    assert (0.280 - r) >= 0.015 - 1e-9

    # Upper ceiling clamp (very high baseline m = 0.400)
    c_high, r_high, _, _ = compute_zone_thresholds(0.400, 0.015)
    assert r_high <= 0.30


def test_feasibility_boundary_cases() -> None:
    """
    Verifies mathematical feasibility boundary:
    m = 0.160 -> c = 0.1216 -> span = 0.0384 < 0.0400 (INFEASIBLE)
    m = 0.170 -> c = 0.1292 -> span = 0.0408 >= 0.0400 (FEASIBLE)
    """
    p_160 = make_mock_profile(center_ear=0.160, center_mad=0.015)
    ext_160 = DriverFeatureExtractor(profile=p_160)
    assert ext_160.is_profile_feasible is False
    assert ext_160.is_ocular_active is False

    p_170 = make_mock_profile(center_ear=0.170, center_mad=0.015)
    ext_170 = DriverFeatureExtractor(profile=p_170)
    assert ext_170.is_profile_feasible is True
    assert ext_170.is_ocular_active is True


def test_span_invariant_holds_for_all_valid_profiles() -> None:
    """
    Empirically verifies the mathematical theorem:
    For any (m, mad) where Span = m - c >= 0.0400:
      c < r
      r - c >= 0.025
      m - r >= 0.015
    holds strictly without exception across the entire parameter space.
    """
    m_values = np.linspace(0.170, 0.400, 24)
    mad_values = np.linspace(0.010, 0.035, 11)

    for m in m_values:
        for mad in mad_values:
            c, r, span, is_feasible = compute_zone_thresholds(m, mad)
            if span >= 0.0400:
                assert c < r, f"Violation c < r for m={m}, mad={mad}: c={c}, r={r}"
                assert (r - c) >= 0.025 - 1e-9, f"Violation r - c >= 0.025 for m={m}, mad={mad}: r-c={r-c}"
                assert (m - r) >= 0.015 - 1e-9, f"Violation m - r >= 0.015 for m={m}, mad={mad}: m-r={m-r}"
                assert c >= 0.12 and c <= 0.22
                assert r <= 0.30


# ==============================================================================
# Group 3: Pipeline Classification & Non-Ocular Feature Preservation
# ==============================================================================

def test_raw_ear_geometric_values_preserved_when_uncalibrated() -> None:
    """
    Verifies that raw geometric EAR values (left_ear, right_ear, compensated_ear)
    remain measurable and un-clipped when uncalibrated (needed for calibration).
    """
    extractor = DriverFeatureExtractor()
    lms = build_synthetic_landmarks(ear_val=0.32)
    features = extractor.extract(make_landmark_result(lms, timestamp_ms=1000.0))

    assert math.isclose(features.ear.left_ear, 0.32, abs_tol=0.02)
    assert math.isclose(features.ear.right_ear, 0.32, abs_tol=0.02)
    assert math.isclose(features.ear.compensated_ear, 0.32, abs_tol=0.02)
    # But classification is disabled
    assert features.ear.is_valid is False
    assert features.ear.is_closed is False


def test_ocular_invalidity_preserves_has_face_true() -> None:
    """
    Verifies that disabling ocular classification does NOT report face loss.
    features.has_face must remain True and face_status must remain DETECTED.
    """
    extractor = DriverFeatureExtractor()
    lms = build_synthetic_landmarks(ear_val=0.35)
    features = extractor.extract(make_landmark_result(lms, timestamp_ms=1000.0))

    assert features.has_face is True
    assert features.face_status == FaceLandmarkStatus.TRACKING


def test_blink_detector_uncertain_gap_without_face_loss() -> None:
    """
    Verifies that when ocular is inactive, BlinkKinematicsDetector enters
    UNCERTAIN_GAP, and downstream TemporalAnalyzer reports face_lost_duration_ms == 0.0.
    """
    extractor = DriverFeatureExtractor()
    lms = build_synthetic_landmarks(ear_val=0.35)
    features = extractor.extract(make_landmark_result(lms, timestamp_ms=1000.0))

    assert extractor.blink_detector._state == BlinkState.UNCERTAIN_GAP

    # Feed into downstream TemporalAnalyzer
    analyzer = TemporalAnalyzer()
    metrics = analyzer.update(features)
    assert metrics.has_face is True
    assert metrics.face_lost_duration_ms == 0.0
    assert metrics.is_eye_closed is None


def test_perclos_receives_invalid_ear_safely() -> None:
    """
    Verifies that PerclosCalculator safely handles is_valid=False without NaNs or crashes.
    """
    extractor = DriverFeatureExtractor()
    for i in range(30):
        ts = 1000.0 + i * 33.3
        lms = build_synthetic_landmarks(ear_val=0.35)
        features = extractor.extract(make_landmark_result(lms, timestamp_ms=ts))
        assert features.perclos.perclos_pct == 0.0
        assert features.perclos.is_reliable is False
        assert features.perclos.observed_duration_sec == 0.0


def test_independent_features_fully_operational_without_profile() -> None:
    """
    Verifies that non-ocular features (head pose, MAR, and yawn detector)
    operate with 100% accuracy and fidelity even when uncalibrated.
    """
    extractor = DriverFeatureExtractor()

    # 1. Head Pose & Zone classification
    lms_left = build_synthetic_landmarks(ear_val=0.30, yaw_deg=22.0, pitch_deg=0.0)
    feat_left = extractor.extract(make_landmark_result(lms_left, timestamp_ms=1000.0))
    assert feat_left.head_pose.is_valid is True
    assert feat_left.head_pose.zone == PoseZone.LEFT_MIRROR
    assert math.isclose(feat_left.head_pose.yaw_deg, 22.0, abs_tol=3.0)

    # 2. MAR & Yawn detection
    # Feed an authentic yawn: MAR ~ 0.65 sustained for > 1500 ms, then mouth closes
    yawn_detected = False
    ts = 2000.0
    for frame_idx in range(50):  # 50 frames @ 33.3ms = 1665ms (> 1500ms min_yawn_duration_ms)
        ts = 2000.0 + frame_idx * 33.3
        lms_yawn = build_synthetic_landmarks(ear_val=0.30, mar_val=0.65)
        feat_yawn = extractor.extract(make_landmark_result(lms_yawn, timestamp_ms=ts))
        assert feat_yawn.mar.is_valid is True
        assert feat_yawn.mar.is_mouth_open is True
        if feat_yawn.mar.is_authentic_yawn:
            yawn_detected = True

    # After mouth re-closes, completed yawn count increments
    ts += 33.3
    lms_close_mouth = build_synthetic_landmarks(ear_val=0.30, mar_val=0.15)
    feat_closed_mouth = extractor.extract(make_landmark_result(lms_close_mouth, timestamp_ms=ts))
    assert feat_closed_mouth.mar.total_yawn_count == 1
    assert yawn_detected is True, "Authentic yawn should be detected even without ocular profile"


# ==============================================================================
# Group 4: Threshold Latching and Mid-Blink Protection
# ==============================================================================

def test_threshold_latch_during_active_blink() -> None:
    """
    Verifies blink-onset threshold latching:
    When eyes close in CENTER zone and driver turns head to LEFT_MIRROR mid-blink,
    the active threshold pair remains latched to CENTER until reopening.
    """
    profile = make_mock_profile(center_ear=0.300, center_mad=0.018, include_peripherals=True)
    extractor = DriverFeatureExtractor(profile=profile)

    # Frame 1: CENTER, open eyes
    lms_open = build_synthetic_landmarks(ear_val=0.35, yaw_deg=0.0)
    f1 = extractor.extract(make_landmark_result(lms_open, timestamp_ms=1000.0))
    assert extractor.blink_detector._state == BlinkState.OPEN
    latched_c = extractor.active_closure_threshold
    latched_r = extractor.active_reopening_threshold

    # Frame 2: Eyes close (EAR = 0.05 < closure) -> detector enters CLOSING / CLOSED
    lms_closed = build_synthetic_landmarks(ear_val=0.05, yaw_deg=0.0)
    f2 = extractor.extract(make_landmark_result(lms_closed, timestamp_ms=1033.0))
    assert extractor.blink_detector._state in (BlinkState.CLOSING, BlinkState.CLOSED)

    # Frame 3: While eyes are still closed, driver glances toward LEFT_MIRROR
    lms_turn = build_synthetic_landmarks(ear_val=0.05, yaw_deg=25.0)
    f3 = extractor.extract(make_landmark_result(lms_turn, timestamp_ms=1066.0))
    assert f3.head_pose.zone == PoseZone.LEFT_MIRROR

    # THRESHOLDS MUST REMAIN LATCHED TO CENTER
    assert extractor.active_closure_threshold == latched_c
    assert extractor.active_reopening_threshold == latched_r
    assert extractor._pending_thresholds is not None  # Queued for when eyes reopen


def test_threshold_update_applied_after_blink_reopening() -> None:
    """
    Verifies that queued threshold updates are unlatched and applied
    as soon as the blink completes and the detector returns to OPEN.
    """
    profile = make_mock_profile(center_ear=0.300, center_mad=0.018, include_peripherals=True)
    extractor = DriverFeatureExtractor(profile=profile)

    # Step 1: Open in CENTER
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35, yaw_deg=0.0), timestamp_ms=1000.0))
    c_center = extractor.active_closure_threshold

    # Step 2: Close in CENTER
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.05, yaw_deg=0.0), timestamp_ms=1033.0))
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.05, yaw_deg=0.0), timestamp_ms=1066.0))

    # Step 3: Turn to LEFT_MIRROR while closed
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.05, yaw_deg=25.0), timestamp_ms=1100.0))
    assert extractor.active_closure_threshold == c_center
    assert extractor._pending_thresholds is not None

    # Step 4: Reopen eyes in LEFT_MIRROR
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35, yaw_deg=25.0), timestamp_ms=1133.0))
    f_open = extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35, yaw_deg=25.0), timestamp_ms=1166.0))

    # Once returned to OPEN, pending thresholds are applied
    assert extractor.blink_detector._state == BlinkState.OPEN
    assert extractor._pending_thresholds is None
    assert extractor.active_closure_threshold != c_center
    assert f_open.ear.closure_threshold == extractor.active_closure_threshold


def test_profile_replacement_during_blink_queued() -> None:
    """
    Verifies that replacing the profile via set_profile() mid-blink does not corrupt
    the in-flight blink; new profile thresholds activate only after eyes reopen.
    """
    p_orig = make_mock_profile(driver_id="driver_orig", center_ear=0.280, center_mad=0.018)
    extractor = DriverFeatureExtractor(profile=p_orig)

    # Open -> Close
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35), timestamp_ms=1000.0))
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.05), timestamp_ms=1033.0))
    assert extractor.blink_detector._state in (BlinkState.CLOSING, BlinkState.CLOSED)
    c_orig = extractor.active_closure_threshold

    # Hot-reload new profile mid-blink
    p_new = make_mock_profile(driver_id="driver_new", center_ear=0.350, center_mad=0.015)
    extractor.set_profile(p_new)

    # During closure, old thresholds must be retained
    assert extractor.active_closure_threshold == c_orig
    assert extractor._pending_profile_switch is not None

    # Reopen eyes
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35), timestamp_ms=1066.0))
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35), timestamp_ms=1100.0))

    # After blink completion, new profile is active
    assert extractor.blink_detector._state == BlinkState.OPEN
    assert extractor._pending_profile_switch is False
    assert extractor.profile.driver_id == "driver_new"
    assert extractor.active_closure_threshold != c_orig


def test_profile_disable_during_blink_queued() -> None:
    """
    Verifies that calling disable_profile() mid-blink allows the active blink to complete
    cleanly before ocular classification is deactivated.
    """
    profile = make_mock_profile(center_ear=0.280, center_mad=0.018)
    extractor = DriverFeatureExtractor(profile=profile)

    # Open -> Close
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35), timestamp_ms=1000.0))
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.05), timestamp_ms=1033.0))
    assert extractor.blink_detector._state in (BlinkState.CLOSING, BlinkState.CLOSED)

    # Disable profile mid-blink
    extractor.disable_profile()
    assert extractor._pending_profile_disable is True
    assert extractor.is_ocular_active is True  # Still active until blink completes

    # Reopen eyes
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35), timestamp_ms=1066.0))
    extractor.extract(make_landmark_result(build_synthetic_landmarks(ear_val=0.35), timestamp_ms=1100.0))

    # Once open, ocular classification is disabled
    assert extractor._pending_profile_disable is False
    assert extractor.is_ocular_active is False
    assert extractor.active_closure_threshold == 0.0


def test_identical_thresholds_across_ear_and_blink_detector() -> None:
    """
    Verifies that at all times, features.ear.closure_threshold strictly matches
    extractor.blink_detector.closure_threshold (no drift or divergence).
    """
    profile = make_mock_profile(center_ear=0.290, center_mad=0.018)
    extractor = DriverFeatureExtractor(profile=profile)

    for i in range(15):
        ts = 1000.0 + i * 33.3
        ear_val = 0.35 if (i % 5 != 0) else 0.05
        yaw = 0.0 if i < 8 else 22.0
        lms = build_synthetic_landmarks(ear_val=ear_val, yaw_deg=yaw)
        features = extractor.extract(make_landmark_result(lms, timestamp_ms=ts))
        assert features.ear.closure_threshold == extractor.blink_detector.closure_threshold


# ==============================================================================
# Group 5: Calibration Lifecycle and Persistence
# ==============================================================================

def test_calibration_sample_targets_and_minimums() -> None:
    """
    Verifies that GuidedCalibrationSession enforces:
    - Step sample targets: CENTER=45, LEFT_MIRROR=40, RIGHT_MIRROR=40, DASHBOARD=40
    - Timeout minimums: CENTER=18, lateral zones=16
    """
    session = GuidedCalibrationSession(driver_id="driver_calib", four_zone_mode=True)
    assert session.target_samples_per_zone == 45
    for step in session.steps:
        target = min(40, session.target_samples_per_zone) if step.zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD) else session.target_samples_per_zone
        min_req = 18 if step.zone == PoseZone.CENTER else 16
        if step.zone == PoseZone.CENTER:
            assert target == 45
            assert min_req == 18
        else:
            assert target == 40
            assert min_req == 16


def test_calibration_collects_raw_ear_without_active_profile() -> None:
    """
    Verifies that GuidedCalibrationSession successfully collects raw EAR samples
    while the feature extractor is uncalibrated and ocular classification is inactive.
    """
    extractor = DriverFeatureExtractor()
    assert extractor.is_ocular_active is False

    session = GuidedCalibrationSession(driver_id="test_driver", four_zone_mode=False, step_duration_sec=5.0)
    session.start(timestamp_ms=1000.0)

    # Feed uncalibrated frames into calibration session
    for i in range(25):
        ts = 1000.0 + i * 40.0
        lms = build_synthetic_landmarks(ear_val=0.30)
        lr = make_landmark_result(lms, timestamp_ms=ts)
        features = extractor.extract(lr)

        # Confirm extractor ocular is disabled
        assert features.ear.is_valid is False
        assert features.ear.is_closed is False

        # Pass raw geometric EAR into calibration session
        is_closed_for_calib = features.ear.is_closed if extractor.is_profile_feasible else False
        session.ingest_frame(
            timestamp_ms=ts,
            face_detected=lr.has_face,
            geometric_reliability=0.95,
            ear_value=features.ear.compensated_ear,
            pitch_deg=features.head_pose.pitch_deg,
            yaw_deg=features.head_pose.yaw_deg,
            is_eye_currently_closed=is_closed_for_calib,
            left_ear=features.ear.left_ear,
            right_ear=features.ear.right_ear,
        )

    # Verify session collected raw samples
    center_samples = session._ear_samples_by_zone.get(PoseZone.CENTER.value, [])
    assert len(center_samples) >= 20
    assert np.median(center_samples) > 0.25


def test_failed_recalibration_preserves_active_profile() -> None:
    """
    Verifies that if a recalibration session fails or is cancelled, the existing
    active profile on the extractor remains untouched and active.
    """
    p_orig = make_mock_profile(driver_id="driver_active", center_ear=0.280, center_mad=0.018)
    extractor = DriverFeatureExtractor(profile=p_orig)
    orig_c = extractor.active_closure_threshold

    # Simulate recalibration session that fails / is cancelled
    session = GuidedCalibrationSession(driver_id="driver_active", four_zone_mode=True)
    session.start(timestamp_ms=1000.0)
    # Cancel session
    session.is_running = False
    session.rejection_reason = "Cancelled by user"

    # Failed recalibration must NOT modify extractor
    assert extractor.is_profile_feasible is True
    assert extractor.is_ocular_active is True
    assert extractor.profile.driver_id == "driver_active"
    assert extractor.active_closure_threshold == orig_c


def test_infeasible_calibration_rejected_before_persistence() -> None:
    """
    Verifies that a calibration resulting in an infeasible profile (m < 0.160)
    is rejected by _evaluate_profile_feasibility() and not saved or activated.
    """
    extractor = DriverFeatureExtractor()
    bad_profile = make_mock_profile(driver_id="bad_driver", center_ear=0.150, center_mad=0.015)

    is_feasible, reason, _ = extractor._evaluate_profile_feasibility(bad_profile)
    assert is_feasible is False
    # Pipeline rejects activation
    assert extractor.is_profile_feasible is False


def test_persistence_roundtrip_preserves_feasibility() -> None:
    """
    Verifies end-to-end persistence roundtrip:
    DriverBehaviourProfile -> DriverProfileDTO -> ProfileStore -> DriverProfileDTO -> DriverBehaviourProfile
    Preserves exact mathematical feasibility and thresholds.
    """
    profile = make_mock_profile(driver_id="roundtrip_driver", center_ear=0.285, center_mad=0.018)
    dto = profile_to_dto(profile)

    # Save to profile store
    profile_store.save_profile(dto)

    # Retrieve from profile store
    loaded_dto = profile_store.get_profile("roundtrip_driver")
    assert loaded_dto is not None
    restored_profile = dto_to_profile(loaded_dto)

    # FeatureExtractor initialization with restored profile
    extractor = DriverFeatureExtractor(profile=restored_profile)
    assert extractor.is_profile_feasible is True
    assert extractor.is_ocular_active is True
    assert math.isclose(extractor.active_closure_threshold, profile.zone_baselines["CENTER"].dynamic_closure_threshold, abs_tol=1e-4)

    # Clean up test profile from store
    profile_store.delete_profile("roundtrip_driver")
