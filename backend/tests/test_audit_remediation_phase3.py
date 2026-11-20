"""
DriveGuard — Phase 3 Core System Audit Remediation Regression Tests.

Covers:
  - HIGH-1: Calibration uses raw uncompensated weighted_ear (never pitch-compensated EAR)
  - HIGH-4: DriverFeatureExtractor rejects UNCALIBRATED_DEFAULT and CENTER.sample_count == 0
  - HIGH-5: Real geometric reliability gating, non-hardcoded landmark confidence, SO(3) rotation rejection fails calibration
  - MEDIUM-4: load_profile_from_file uses model_validate (not broken from_dict), recalibration passes existing_profile,
              and deferred activation is distinguished from immediate activation
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    ZoneBaseline,
)
from backend.app.ai.profile_adapter import profile_to_dto
from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityResult,
    LightingCondition,
    QualityStatus,
)
from backend.app.features.blink_detector import BlinkSnapshot, BlinkState
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatureExtractor, DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult
from backend.app.features.perclos import PerclosResult
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)
from backend.app.services.local_monitoring_service import LocalMonitoringService
from scripts.run_camera_test import (
    compute_calibration_geometric_reliability,
    load_profile_from_file,
    run_benchmark,
)


def _rot_y(deg: float) -> np.ndarray:
    r = np.radians(deg)
    return np.array(
        [[np.cos(r), 0.0, np.sin(r)], [0.0, 1.0, 0.0], [-np.sin(r), 0.0, np.cos(r)]],
        dtype=np.float64,
    )


def _make_dummy_features(
    pose_valid: bool = True,
    rot_mat: np.ndarray | None = None,
    ear_geom_valid: bool = True,
) -> DriverFeatures:
    return DriverFeatures(
        timestamp_ms=1000.0,
        has_face=True,
        face_status=FaceLandmarkStatus.TRACKING,
        ear=EyeAspectRatioResult(
            left_ear=0.30,
            right_ear=0.30,
            weighted_ear=0.30,
            compensated_ear=0.33,
            closure_threshold=0.20,
            is_left_closed=False,
            is_right_closed=False,
            is_closed=False,
            is_valid=ear_geom_valid,
            is_left_valid=ear_geom_valid,
            is_right_valid=ear_geom_valid,
            is_geometric_valid=ear_geom_valid,
        ),
        head_pose=HeadPoseResult(
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            zone=PoseZone.CENTER if pose_valid else PoseZone.UNKNOWN,
            is_looking_away=False,
            is_valid=pose_valid,
            rot_mat=rot_mat,
        ),
        mar=MouthAspectRatioResult(
            mar=0.15,
            is_mouth_open=False,
            is_yawn_in_progress=False,
            current_yawn_duration_ms=0.0,
            is_authentic_yawn=False,
            completed_yawn=None,
            total_yawn_count=0,
            is_valid=True,
        ),
        blink=BlinkSnapshot(
            state=BlinkState.OPEN,
            is_eye_closed=False,
            current_closure_duration_ms=0.0,
            completed_blink=None,
            recent_blink_duration_ms=None,
            recent_blink_rcr=None,
            total_blink_count=0,
            prolonged_blink_count=0,
        ),
        perclos=PerclosResult(
            perclos_pct=0.0,
            closed_duration_sec=0.0,
            observed_duration_sec=1.0,
            window_sec=60.0,
            coverage_pct=100.0,
            is_reliable=True,
        ),
    )


# ---------------------------------------------------------------------------
# HIGH-1: Calibration Ingests Raw Uncompensated weighted_ear
# ---------------------------------------------------------------------------

def test_high1_run_camera_test_ingests_weighted_ear_not_compensated_ear() -> None:
    """
    Verify LocalMonitoringService (used by run_benchmark) passes raw uncompensated
    `features.ear.weighted_ear` to `calibration_session.ingest_frame`, never `compensated_ear`.
    """
    src = inspect.getsource(LocalMonitoringService.process_frame)
    assert "ear_value=features.ear.weighted_ear" in src
    assert "ear_value=features.ear.compensated_ear" not in src
    runner_src = inspect.getsource(run_benchmark)
    assert "LocalMonitoringService" in runner_src


# ---------------------------------------------------------------------------
# HIGH-4: Reject UNCALIBRATED_DEFAULT and CENTER.sample_count == 0
# ---------------------------------------------------------------------------

def test_high4_extractor_rejects_uncalibrated_default_and_zero_center_samples() -> None:
    ext = DriverFeatureExtractor()

    # 1. Default uncalibrated profile must be strictly rejected
    default_prof = DriverBehaviourProfile.create_default_uncalibrated()
    valid, reason = ext.validate_profile(default_prof)
    assert valid is False
    assert "UNCALIBRATED" in reason.upper() or "uncalibrated" in reason.lower()
    assert ext.set_profile(default_prof) is False
    assert ext.is_profile_feasible is False

    # 2. Profile claiming CENTER_ONLY but with CENTER sample_count == 0 must be rejected
    zero_sample_prof = DriverBehaviourProfile(
        driver_id="zero_samples",
        calibration_mode="CENTER_ONLY",
        center_ear_median=0.30,
        center_ear_mad_scaled=0.015,
        blink_duration_median_ms=220.0,
        blink_duration_mad_ms=35.0,
        blink_rcr_median=1.0,
        blink_rcr_mad=0.15,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.90,
        zone_baselines={
            "CENTER": ZoneBaseline(
                zone="CENTER",
                ear_median=0.30,
                ear_mad_scaled=0.015,
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=0,
            )
        },
    )
    feasible, f_reason = ext.validate_profile_feasibility(zero_sample_prof)
    assert feasible is False
    assert "sample_count" in f_reason
    assert ext.set_profile(zero_sample_prof) is False
    assert ext.is_profile_feasible is False


# ---------------------------------------------------------------------------
# HIGH-5: Real Quality Gates, Landmark Confidence & SO(3) Rejection
# ---------------------------------------------------------------------------

def test_high5_calibration_geometric_reliability_respects_optical_and_geometry() -> None:
    lm_good = FaceLandmarkResult(
        has_face=True,
        landmarks=(),
        presence_confidence=0.95,
        timestamp_ms=1000,
        status=FaceLandmarkStatus.TRACKING,
        inference_latency_ms=5.0,
        reasons=("OK",),
    )
    q_opt = FrameQualityResult(
        mean_luminance=120.0,
        blur_variance=180.0,
        lighting=LightingCondition.OPTIMAL,
        blur=BlurCondition.SHARP,
        status=QualityStatus.EXCELLENT,
        is_viable_for_inference=True,
        reasons=(),
    )
    q_unviable = FrameQualityResult(
        mean_luminance=8.0,
        blur_variance=5.0,
        lighting=LightingCondition.UNDEREXPOSED,
        blur=BlurCondition.BLURRY,
        status=QualityStatus.SEVERELY_DEGRADED,
        is_viable_for_inference=False,
        reasons=("Too dark and blurred",),
    )

    feats_ok = _make_dummy_features(pose_valid=True, rot_mat=np.eye(3), ear_geom_valid=True)
    feats_bad_pose = _make_dummy_features(pose_valid=False, rot_mat=None, ear_geom_valid=True)
    feats_bad_ear = _make_dummy_features(pose_valid=True, rot_mat=np.eye(3), ear_geom_valid=False)

    rel_good = compute_calibration_geometric_reliability(lm_good, feats_ok, q_opt)
    rel_unviable = compute_calibration_geometric_reliability(lm_good, feats_ok, q_unviable)
    rel_bad_pose = compute_calibration_geometric_reliability(lm_good, feats_bad_pose, q_opt)
    rel_bad_ear = compute_calibration_geometric_reliability(lm_good, feats_bad_ear, q_opt)

    assert rel_good >= 0.90
    assert rel_unviable == 0.0
    assert rel_bad_pose < 0.35
    assert rel_bad_ear < 0.35


def test_high5_face_landmarker_confidence_is_derived_not_hardcoded() -> None:
    # Well-framed face
    good_lms = [MagicMock(x=0.5, y=0.5, z=0.0, presence=None) for _ in range(478)]
    good_lms[33] = MagicMock(x=0.40, y=0.42, z=0.0, presence=None)
    good_lms[263] = MagicMock(x=0.60, y=0.42, z=0.0, presence=None)
    good_lms[10] = MagicMock(x=0.50, y=0.25, z=0.0, presence=None)
    good_lms[152] = MagicMock(x=0.50, y=0.72, z=0.0, presence=None)
    assert DriveGuardFaceLandmarker._estimate_landmark_confidence(good_lms) >= 0.95

    # Collapsed tiny/degenerate face -> low confidence (< 0.5)
    collapsed_lms = [MagicMock(x=0.5, y=0.5, z=0.0, presence=None) for _ in range(478)]
    assert DriveGuardFaceLandmarker._estimate_landmark_confidence(collapsed_lms) < 0.2

    # Half off-screen clipped face -> degraded confidence (< 0.5)
    clipped_lms = [MagicMock(x=1.2, y=0.5, z=0.0, presence=None) for _ in range(478)]
    clipped_lms[33] = MagicMock(x=0.95, y=0.42, z=0.0, presence=None)
    clipped_lms[263] = MagicMock(x=1.15, y=0.42, z=0.0, presence=None)
    clipped_lms[10] = MagicMock(x=1.05, y=0.25, z=0.0, presence=None)
    clipped_lms[152] = MagicMock(x=1.05, y=0.72, z=0.0, presence=None)
    assert DriveGuardFaceLandmarker._estimate_landmark_confidence(clipped_lms) < 0.5


def test_high5_so3_rotation_rejection_fails_calibration_instead_of_silent_none() -> None:
    """
    If candidate SO(3) rotation matrices are fed during CENTER calibration but have excessive
    angular dispersion (unstable head), GuidedCalibrationSession._finalize() must reject the
    calibration (`return None` + `REJECTED_HEAD_POSE_INSTABILITY`) instead of silently
    completing with `neutral_rot_mat=None`.
    """
    session = GuidedCalibrationSession(
        driver_id="unstable_driver",
        four_zone_mode=False,
        step_duration_sec=15.0,
    )
    session.target_samples_per_zone = 24
    session.start(timestamp_ms=0.0)

    # Feed two widely separated clusters (12 frames at -15 deg, 12 frames at +15 deg)
    # -> 50% inlier ratio (< 60% required) while staying inside initial_center_max_yaw_deg=32 deg
    ts = 0.0
    result = None
    for i in range(25):
        ts += 450.0  # smooth transition < 80 deg/s
        yaw = -15.0 if i < 12 else 15.0
        result = session.ingest_frame(
            timestamp_ms=ts,
            face_detected=True,
            geometric_reliability=0.92,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=yaw,
            is_eye_currently_closed=False,
            rot_mat=_rot_y(yaw),
            ear_geometric_valid=True,
        )
        if not session.is_running:
            break

    assert result is None, "Calibration must reject unstable SO(3) rotation candidates"
    assert session.rejection_reason is not None
    assert "REJECTED_HEAD_POSE_INSTABILITY" in session.rejection_reason


# ---------------------------------------------------------------------------
# MEDIUM-4: DTO File Loading, Recalibration existing_profile & Deferred Banner
# ---------------------------------------------------------------------------

def test_medium4_load_profile_from_file_supports_dto_and_domain_json(tmp_path: Path) -> None:
    prof = DriverBehaviourProfile(
        driver_id="dto_file_driver",
        calibration_mode="CENTER_ONLY",
        center_ear_median=0.30,
        center_ear_mad_scaled=0.015,
        blink_duration_median_ms=220.0,
        blink_duration_mad_ms=35.0,
        blink_rcr_median=1.0,
        blink_rcr_mad=0.15,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        neutral_pitch_deg=-3.0,
        neutral_yaw_deg=5.0,
        calibration_reliability=0.95,
        zone_baselines={
            "CENTER": ZoneBaseline(
                zone="CENTER",
                ear_median=0.30,
                ear_mad_scaled=0.015,
                pitch_median_deg=-3.0,
                yaw_median_deg=5.0,
                sample_count=45,
            )
        },
        calibration_is_mirrored=True,
    )

    # 1. DTO JSON file (with optional schema_version key)
    dto_path = tmp_path / "dto_profile.json"
    dto_payload = profile_to_dto(prof).model_dump(mode="json")
    dto_payload["schema_version"] = 1
    dto_path.write_text(json.dumps(dto_payload), encoding="utf-8")

    loaded_dto = load_profile_from_file(dto_path)
    assert loaded_dto.driver_id == "dto_file_driver"
    assert loaded_dto.calibration_is_mirrored is True

    # 2. Legacy domain JSON file
    domain_path = tmp_path / "domain_profile.json"
    prof.save_json(domain_path)
    loaded_domain = load_profile_from_file(domain_path)
    assert loaded_domain.driver_id == "dto_file_driver"
    assert loaded_domain.calibration_is_mirrored is True

    # 3. Verify LocalMonitoringService passes existing_profile on recalibration and checks has_pending_profile
    svc_start_src = inspect.getsource(LocalMonitoringService.start_calibration)
    svc_proc_src = inspect.getsource(LocalMonitoringService.process_frame)
    assert "existing_profile=" in svc_start_src
    assert "self.feature_extractor.has_pending_profile" in svc_proc_src
    assert "DriverProfileDTO.from_dict" not in inspect.getsource(run_benchmark)
