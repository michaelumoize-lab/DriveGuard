"""
DriveGuard — Phase 2 Core System Audit Remediation Regression Tests.

Covers:
  - HIGH-2: BlinkState.REOPENING timeout prevents deadlock & fabricated MICROSLEEP blinks
  - HIGH-6: head_pose 2D profile fallback returns rot_mat=None and preserves camera-vs-neutral yaw
  - HIGH-7: Persisted neutral_rot_mat mirror-mode conjugation across toggle + DTO/JSON roundtrip
  - MEDIUM-2: YawnDetector 250ms gap, max_yawn_duration cap, dropout-grace flags, DecisionEngine yawn_viable gate
  - MEDIUM-3: BlinkKinematicsDetector rejects NaN/backward timestamps without NaN/negative closure duration
  - MEDIUM-8: BlinkSnapshot invariant (is_eye_closed=False => current_closure_duration_ms == 0.0)
  - MEDIUM-9: CLOSING re-anchor uses re-anchored onset EAR for v_close
  - LOW-2: min_ear_ts == 0.0 handled via `is not None`
  - LOW-3: Face-loss branch reports active_closure_threshold
  - LOW-5: DriveGuardFaceLandmarker grace timeout uses monotonic frame timestamps
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    ZoneBaseline,
    create_immutable_profile_snapshot,
)
from backend.app.ai.profile_adapter import dto_to_profile, profile_to_dto
from backend.app.features.blink_detector import BlinkKinematicsDetector, BlinkState
from backend.app.features.feature_extractor import DriverFeatureExtractor
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    PoseZone,
    SOLVEPNP_LANDMARK_INDICES,
    build_neutral_matrix,
    estimate_head_pose,
    is_valid_rotation_matrix,
)
from backend.app.features.mar import DEFAULT_MAX_SAMPLING_GAP_MS, YawnDetector
from backend.app.intelligence.confidence import ConfidenceAssessment, ConfidenceTier
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalMetrics
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkerConfig,
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


def _make_rot_mat(yaw_deg: float, pitch_deg: float, roll_deg: float, is_mirrored: bool) -> np.ndarray:
    """Build camera-frame rotation matrix matching head_pose.py conventions."""
    cam_yaw = yaw_deg if is_mirrored else -yaw_deg
    cam_pitch = -pitch_deg
    cam_roll = -roll_deg if is_mirrored else roll_deg
    psi = np.radians(cam_yaw)
    theta = np.radians(cam_pitch)
    phi = np.radians(cam_roll)
    Ry = np.array([[np.cos(psi), 0.0, np.sin(psi)], [0.0, 1.0, 0.0], [-np.sin(psi), 0.0, np.cos(psi)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def _project_pose_landmarks(
    rot_mat: np.ndarray,
    frame_width: int = 1280,
    frame_height: int = 720,
    depth_mm: float = 600.0,
) -> list[NormalizedLandmark]:
    focal = float(frame_width)
    cam_mat = np.array(
        [[focal, 0.0, frame_width / 2.0], [0.0, focal, frame_height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    rvec, _ = cv2.Rodrigues(rot_mat)
    tvec = np.array([[0.0], [0.0], [depth_mm]], dtype=np.float64)
    proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, cam_mat, np.zeros((4, 1)))
    pts = proj.reshape(-1, 2)
    lms = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES):
        lms[idx] = NormalizedLandmark(float(pts[i, 0] / frame_width), float(pts[i, 1] / frame_height), 0.0)
    return lms


def _make_mouth_landmarks(mar_target: float) -> list[NormalizedLandmark]:
    h = mar_target * 0.16
    lms = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    overrides = {
        61: (0.42, 0.70, 0.0),
        291: (0.58, 0.70, 0.0),
        13: (0.50, 0.70 - h / 2.0, 0.0),
        14: (0.50, 0.70 + h / 2.0, 0.0),
        81: (0.46, 0.70 - h / 2.0, 0.0),
        178: (0.46, 0.70 + h / 2.0, 0.0),
        311: (0.54, 0.70 - h / 2.0, 0.0),
        402: (0.54, 0.70 + h / 2.0, 0.0),
    }
    for idx, (x, y, z) in overrides.items():
        lms[idx] = NormalizedLandmark(x, y, z)
    return lms


# ---------------------------------------------------------------------------
# HIGH-2: REOPENING Timeout Prevents Deadlock & Fabricated Multi-Second Blinks
# ---------------------------------------------------------------------------

def test_high2_reopening_stalled_times_out_cleanly_without_fabricated_blink() -> None:
    """
    If EAR enters REOPENING (c < EAR < r) and stays stalled there beyond REOPENING_TIMEOUT_MS
    (e.g., partial squint), the detector must cleanly reset to OPEN without emitting a fabricated
    multi-second MICROSLEEP_CANDIDATE blink when EAR later rises above r.
    """
    det = BlinkKinematicsDetector(closure_threshold=0.20, reopen_threshold=0.25)

    # Open -> Closed -> Reopening
    det.update(0.30, 0.30, 0.30, timestamp_ms=1000.0)
    det.update(0.16, 0.16, 0.16, timestamp_ms=1033.0)
    det.update(0.15, 0.15, 0.15, timestamp_ms=1066.0)
    snap_reopen = det.update(0.22, 0.22, 0.22, timestamp_ms=1100.0)
    assert snap_reopen.state == BlinkState.REOPENING

    # Stall at EAR=0.22 (between 0.20 and 0.25) for 1200ms (> 1000ms timeout)
    last_snap = snap_reopen
    for step in range(1, 40):
        ts = 1100.0 + step * 33.0
        last_snap = det.update(0.22, 0.22, 0.22, timestamp_ms=ts)
        assert last_snap.completed_blink is None

    assert last_snap.state == BlinkState.OPEN, (
        f"Expected REOPENING stall to exit cleanly to OPEN, got {last_snap.state}"
    )
    assert last_snap.total_blink_count == 0

    # Later EAR rises to 0.30 -> must NOT fabricate a multi-second blink
    after = det.update(0.30, 0.30, 0.30, timestamp_ms=2500.0)
    assert after.state == BlinkState.OPEN
    assert after.completed_blink is None
    assert after.total_blink_count == 0


# ---------------------------------------------------------------------------
# HIGH-6: 2D Profile Fallback Never Fabricates rot_mat=I or Distorts Yaw/Pitch
# ---------------------------------------------------------------------------

def test_high6_2d_profile_fallback_returns_no_rot_mat_and_consistent_yaw() -> None:
    """
    When 3D solvePnP fails and the 2D profile fallback triggers on geometrically plausible
    profile landmarks, HeadPoseResult must:
    - return rot_mat=None (never np.eye(3)),
    - estimate camera-relative yaw geometrically (not hardcoded +-28 deg) and subtract
      `neutral_yaw_deg` for `rel_yaw_deg`,
    - keep an oblique mount (+27.3 deg neutral yaw) in LEFT_MIRROR rather than snapping to CENTER,
    - return invalid_result (is_valid=False, UNKNOWN) rather than ever fabricating CENTER.
    """
    lms = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]
    # Realistic profile geometry matching real oblique session_20261120_005232 (frame 89):
    # eye_span_x = 0.090 * 1280 = 115.2px, nose_ratio = -0.04, sellion_ratio = +0.21
    lms[33] = NormalizedLandmark(0.450, 0.40, 0.0)    # left outer eye corner
    lms[263] = NormalizedLandmark(0.540, 0.40, 0.0)   # right outer eye corner
    lms[6] = NormalizedLandmark(0.469, 0.36, 0.0)     # sellion (between eyes, ratio ~0.21)
    lms[1] = NormalizedLandmark(0.446, 0.50, 0.0)     # nose tip (ratio ~-0.044)
    lms[152] = NormalizedLandmark(0.475, 0.68, 0.0)   # chin
    lms[61] = NormalizedLandmark(0.455, 0.60, 0.0)
    lms[291] = NormalizedLandmark(0.515, 0.60, 0.0)
    lms[10] = NormalizedLandmark(0.475, 0.25, 0.0)

    # Force cv2.solvePnP to fail so 2D fallback runs deterministically
    with patch("backend.app.features.head_pose.cv2.solvePnP", return_value=(False, None, None)):
        # Case 1: Oblique camera calibrated at neutral_yaw_deg = +27.3 -> stays LEFT_MIRROR (~+25 deg rel)
        res_oblique_cam = estimate_head_pose(
            lms,
            frame_width=1280,
            frame_height=720,
            neutral_pitch_deg=6.6,
            neutral_yaw_deg=27.3,
            is_mirrored=True,
        )
        assert res_oblique_cam.is_valid is True
        assert res_oblique_cam.rot_mat is None, "2D fallback must NEVER fabricate np.eye(3)"
        assert 48.0 <= res_oblique_cam.yaw_deg <= 58.0
        assert res_oblique_cam.rel_yaw_deg == pytest.approx(res_oblique_cam.yaw_deg - 27.3, abs=1e-6)
        assert res_oblique_cam.rel_pitch_deg == pytest.approx(0.0)
        assert res_oblique_cam.zone == PoseZone.LEFT_MIRROR
        assert res_oblique_cam.is_looking_away is True

        # Case 2: Extreme/ambiguous neutral_yaw_deg = +50.0 where fallback would land in CENTER
        # -> must reject as UNKNOWN/invalid rather than fabricating CENTER!
        res_ambiguous = estimate_head_pose(
            lms,
            frame_width=1280,
            frame_height=720,
            neutral_pitch_deg=0.0,
            neutral_yaw_deg=50.0,
            is_mirrored=True,
        )
        assert res_ambiguous.is_valid is False
        assert res_ambiguous.zone == PoseZone.UNKNOWN
        assert res_ambiguous.rot_mat is None


# ---------------------------------------------------------------------------
# HIGH-7: Persisted neutral_rot_mat Conjugation Across Mirror Mode Toggle
# ---------------------------------------------------------------------------

def test_high7_neutral_rot_mat_mirror_conjugation_and_persistence(tmp_path: Path) -> None:
    """
    A profile calibrated with is_mirrored=True on an oblique camera (+22 deg yaw, -7 deg pitch, +4 deg roll)
    must remain accurate when evaluated at runtime with is_mirrored=False (mirror toggled off),
    and `calibration_is_mirrored` must survive DTO, JSON, and ImmutableProfileSnapshot roundtrips.
    """
    r_neutral_mirrored = _make_rot_mat(yaw_deg=22.0, pitch_deg=-7.0, roll_deg=4.0, is_mirrored=True)

    # Verify build_neutral_matrix conjugates cleanly into unmirrored SO(3)
    r_conjugated = build_neutral_matrix(
        neutral_rot_mat=r_neutral_mirrored,
        is_mirrored=False,
        neutral_is_mirrored=True,
    )
    assert is_valid_rotation_matrix(r_conjugated)
    r_expected_unmirrored = _make_rot_mat(yaw_deg=22.0, pitch_deg=-7.0, roll_deg=4.0, is_mirrored=False)
    assert np.allclose(r_conjugated, r_expected_unmirrored, atol=1e-6)

    # Build profile with calibration_is_mirrored=True and test roundtrips
    profile = DriverBehaviourProfile(
        driver_id="mirror_test_driver",
        calibration_mode="CENTER_ONLY",
        center_ear_median=0.30,
        center_ear_mad_scaled=0.018,
        blink_duration_median_ms=220.0,
        blink_duration_mad_ms=35.0,
        blink_rcr_median=1.0,
        blink_rcr_mad=0.15,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        neutral_pitch_deg=-7.0,
        neutral_yaw_deg=22.0,
        calibration_reliability=0.92,
        zone_baselines={
            "CENTER": ZoneBaseline(
                zone="CENTER",
                ear_median=0.30,
                ear_mad_scaled=0.018,
                pitch_median_deg=-7.0,
                yaw_median_deg=22.0,
                sample_count=45,
            )
        },
        neutral_rot_mat=r_neutral_mirrored,
        calibration_is_mirrored=True,
    )

    # DTO roundtrip
    dto = profile_to_dto(profile)
    assert dto.calibration_is_mirrored is True
    roundtripped = dto_to_profile(dto)
    assert roundtripped.calibration_is_mirrored is True

    # JSON roundtrip
    json_path = tmp_path / "profile.json"
    profile.save_json(json_path)
    loaded = DriverBehaviourProfile.load_json(json_path)
    assert loaded.calibration_is_mirrored is True

    # Snapshot roundtrip
    snap = create_immutable_profile_snapshot(loaded)
    assert snap.calibration_is_mirrored is True
    assert snap.to_domain().calibration_is_mirrored is True

    # End-to-end head pose check: driver looking straight ahead at their calibrated neutral posture
    # while runtime camera is UNMIRRORED (is_mirrored=False)
    lms_unmirrored_neutral = _project_pose_landmarks(r_expected_unmirrored)
    extractor = DriverFeatureExtractor(is_mirrored=False, profile=loaded)
    lm_res = FaceLandmarkResult(
        has_face=True,
        landmarks=tuple(lms_unmirrored_neutral),
        presence_confidence=1.0,
        timestamp_ms=1000,
        status=FaceLandmarkStatus.TRACKING,
        inference_latency_ms=5.0,
        reasons=("OK",),
    )
    feats = extractor.extract(lm_res, frame_width=1280, frame_height=720, is_mirrored=False)
    assert feats.head_pose.is_valid is True
    assert feats.head_pose.zone == PoseZone.CENTER
    assert abs(feats.head_pose.rel_yaw_deg) < 2.0
    assert abs(feats.head_pose.rel_pitch_deg) < 2.0
    assert abs(feats.head_pose.rel_roll_deg) < 2.0


# ---------------------------------------------------------------------------
# MEDIUM-2: YawnDetector Alignment, Max Cap, Dropout Grace & DecisionEngine Gate
# ---------------------------------------------------------------------------

def test_medium2_yawn_detector_gap_cap_dropout_and_decision_gate() -> None:
    assert DEFAULT_MAX_SAMPLING_GAP_MS == 400.0

    open_lms = _make_mouth_landmarks(0.65)
    closed_lms = _make_mouth_landmarks(0.08)

    # 0. Brief ~10 FPS jitter (284ms <= 400ms) preserves in-progress yawn; >400ms (450ms) aborts it
    det_gap = YawnDetector(min_yawn_duration_ms=1500.0, max_yawn_duration_ms=8000.0)
    for t in range(1000, 2701, 100):
        det_gap.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0, is_normalized=True)
    # 284ms jitter frame while still yawning -> must NOT abort
    det_gap.update(open_lms, timestamp_ms=2984.0, aspect_ratio=1.0, is_normalized=True)
    res_completed = det_gap.update(closed_lms, timestamp_ms=3084.0, aspect_ratio=1.0, is_normalized=True)
    assert res_completed.total_yawn_count == 1
    assert res_completed.completed_yawn is not None

    det_gap.reset()
    for t in range(1000, 2701, 100):
        det_gap.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0, is_normalized=True)
    # 450ms camera freeze > 400ms -> must abort cleanly
    det_gap.update(open_lms, timestamp_ms=3150.0, aspect_ratio=1.0, is_normalized=True)
    res_aborted = det_gap.update(closed_lms, timestamp_ms=3250.0, aspect_ratio=1.0, is_normalized=True)
    assert res_aborted.total_yawn_count == 0
    assert res_aborted.completed_yawn is None

    # 1. Dropout-grace frame must never emit is_yawn_in_progress=True or is_authentic_yawn=True
    det = YawnDetector(min_yawn_duration_ms=1500.0, max_yawn_duration_ms=8000.0, dropout_grace_ms=200.0)
    for t in range(1000, 2601, 100):
        res_open = det.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0, is_normalized=True)
    assert res_open.is_valid is True
    assert res_open.is_authentic_yawn is True

    res_drop = det.update([], timestamp_ms=2650.0, aspect_ratio=1.0, has_face=False)
    assert res_drop.is_valid is False
    assert res_drop.is_yawn_in_progress is False
    assert res_drop.is_authentic_yawn is False
    assert res_drop.observed_yawn_duration_ms == 0.0

    # 2. Opening sustained past max_yawn_duration_ms (8000ms) must clear active yawn flags
    #    and MUST NOT re-trigger new yawn episodes while mouth remains open
    det.reset()
    for t in range(1000, 11501, 100):
        res = det.update(open_lms, timestamp_ms=float(t), aspect_ratio=1.0, is_normalized=True)
        if t - 1000 > 8000:
            assert res.is_yawn_in_progress is False
            assert res.is_authentic_yawn is False
    res_close = det.update(closed_lms, timestamp_ms=11600.0, aspect_ratio=1.0, is_normalized=True)
    assert res_close.completed_yawn is None
    assert res_close.total_yawn_count == 0

    # 3. DecisionEngine gates active yawn on confidence.yawn_viable
    engine = DecisionEngine()
    metrics = TemporalMetrics(
        timestamp_ms=3000.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=0.0,
        is_mirror_check_context=False,
        is_dashboard_check_context=False,
        is_nodding_microsleep=False,
        blink_count_60s=5,
        blink_count_30s=3,
        mean_blink_duration_ms_60s=200.0,
        prolonged_blink_count_60s=0,
        severe_blink_count_60s=0,
        mean_rcr_60s=1.0,
        yawn_count_60s=0,
        yawn_count_30s=0,
        active_yawn_duration_ms=1800.0,
        perclos_pct=3.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
        is_eye_closed=False,
    )
    conf_no_yawn = ConfidenceAssessment(
        overall_confidence=0.80,
        tier=ConfidenceTier.HIGH,
        is_reliable=True,
        is_ocular_available=True,
        is_yawn_available=False,
        is_head_pose_available=True,
    )
    dec_blocked = engine.evaluate(metrics=metrics, confidence=conf_no_yawn)
    assert dec_blocked.state == DrowsinessState.NORMAL
    assert not any(r.code == ReasonCode.AUTHENTIC_YAWN for r in dec_blocked.reasons)


# ---------------------------------------------------------------------------
# MEDIUM-3, MEDIUM-8, MEDIUM-9, LOW-2: BlinkKinematicsDetector Edge Invariants
# ---------------------------------------------------------------------------

def test_medium3_medium8_medium9_low2_blink_detector_invariants() -> None:
    # MEDIUM-3: NaN / backward timestamps never yield NaN or negative current_closure_duration_ms
    det = BlinkKinematicsDetector(closure_threshold=0.20, reopen_threshold=0.25)
    det.update(0.30, 0.30, 0.30, timestamp_ms=1000.0)
    det.update(0.15, 0.15, 0.15, timestamp_ms=1050.0)
    det.update(0.15, 0.15, 0.15, timestamp_ms=1150.0)

    snap_nan = det.update(0.15, 0.15, 0.15, timestamp_ms=float("nan"))
    assert math.isfinite(snap_nan.current_closure_duration_ms)
    assert snap_nan.current_closure_duration_ms >= 0.0

    snap_back = det.update(0.15, 0.15, 0.15, timestamp_ms=900.0)
    assert math.isfinite(snap_back.current_closure_duration_ms)
    assert snap_back.current_closure_duration_ms >= 0.0

    # MEDIUM-8: Invariant `not is_eye_closed => current_closure_duration_ms == 0.0`
    det2 = BlinkKinematicsDetector(closure_threshold=0.20, reopen_threshold=0.25)
    det2.update(0.30, 0.30, 0.30, timestamp_ms=1000.0)
    # Asymmetric blink: left valid & closed (0.14), right valid & open (0.30), weighted = 0.22 (in (c, r))
    s1 = det2.update(0.14, 0.30, 0.22, timestamp_ms=1033.0)
    if not s1.is_eye_closed:
        assert s1.current_closure_duration_ms == 0.0

    # MEDIUM-9: Re-anchored CLOSING computes v_close from re-anchored onset EAR while keeping open_ear_before
    det3 = BlinkKinematicsDetector(closure_threshold=0.20, reopen_threshold=0.25)
    det3.update(0.32, 0.32, 0.32, timestamp_ms=1000.0)
    # Hover in CLOSING near 0.24 for > CLOSING_STALL_MS (300ms) -> triggers re-anchor at 0.24
    for t in (1050.0, 1150.0, 1250.0, 1360.0):
        det3.update(0.24, 0.24, 0.24, timestamp_ms=t)
    # Now drop briskly from 0.24 -> 0.14 over 50ms, then reopen to 0.30
    det3.update(0.14, 0.14, 0.14, timestamp_ms=1410.0)
    done = det3.update(0.30, 0.30, 0.30, timestamp_ms=1480.0)
    assert done.completed_blink is not None
    assert done.completed_blink.open_ear_before == pytest.approx(0.32, abs=1e-4)
    # Displacement used for closing_velocity_per_s should be (0.24 - 0.14) / 0.05s = 2.0 EAR/s (NOT (0.32 - 0.14)/0.05 = 3.6)
    assert done.completed_blink.closing_velocity_per_s == pytest.approx(2.0, abs=0.25)

    # LOW-2: min_ear_ts == 0.0 must be respected (not treated as falsy midpoint fallback)
    det4 = BlinkKinematicsDetector(closure_threshold=0.20, reopen_threshold=0.25)
    det4._state = BlinkState.CLOSED
    det4._prev_ts = 0.0
    det4._prev_ear = 0.14
    det4._blink_start_ts = 0.0
    det4._closure_start_ts = 0.0
    det4._open_ear_before = 0.30
    det4._min_ear = 0.14
    det4._min_ear_ts = 0.0
    det4._was_left_closed = True
    det4._was_right_closed = True
    done0 = det4.update(0.30, 0.30, 0.30, timestamp_ms=100.0)
    assert done0.completed_blink is not None
    assert done0.completed_blink.min_ts_ms == pytest.approx(0.0, abs=1e-6)


# ---------------------------------------------------------------------------
# LOW-3 & LOW-5: Face-Loss Threshold Consistency & Monotonic Frame Grace
# ---------------------------------------------------------------------------

def test_low3_face_loss_reports_active_closure_threshold() -> None:
    # Uncalibrated extractor: active_closure_threshold == 0.0
    ext_uncal = DriverFeatureExtractor(ear_closure_threshold=0.20)
    no_face = FaceLandmarkResult(
        has_face=False,
        landmarks=(),
        presence_confidence=0.0,
        timestamp_ms=1000,
        status=FaceLandmarkStatus.INSUFFICIENT_DATA,
        inference_latency_ms=2.0,
        reasons=("No face",),
    )
    res_uncal = ext_uncal.extract(no_face)
    assert res_uncal.ear.closure_threshold == pytest.approx(0.0)

    # Calibrated extractor: active_closure_threshold matches CENTER threshold
    profile = DriverBehaviourProfile(
        driver_id="low3_driver",
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
        calibration_reliability=0.95,
        zone_baselines={
            "CENTER": ZoneBaseline(
                zone="CENTER",
                ear_median=0.30,
                ear_mad_scaled=0.015,
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=45,
            )
        },
    )
    ext_cal = DriverFeatureExtractor(profile=profile)
    res_cal = ext_cal.extract(no_face)
    assert res_cal.ear.closure_threshold == pytest.approx(ext_cal.active_closure_threshold)
    assert res_cal.ear.closure_threshold > 0.0


def test_low5_face_landmarker_grace_uses_frame_timestamps() -> None:
    """
    Face-loss grace timeout in DriveGuardFaceLandmarker must use monotonic frame timestamps
    rather than wall-clock `time.monotonic()`, ensuring deterministic offline replay.
    """
    cfg = FaceLandmarkerConfig(face_loss_grace_timeout_sec=0.3)
    landmarker = DriveGuardFaceLandmarker(cfg)
    try:
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        # Anatomically plausible well-framed face landmarks so confidence >= 0.5 (TRACKING)
        mock_lms = [MagicMock(x=0.5, y=0.5, z=0.0, presence=None) for _ in range(478)]
        mock_lms[33] = MagicMock(x=0.40, y=0.42, z=0.0, presence=None)
        mock_lms[263] = MagicMock(x=0.60, y=0.42, z=0.0, presence=None)
        mock_lms[10] = MagicMock(x=0.50, y=0.25, z=0.0, presence=None)
        mock_lms[152] = MagicMock(x=0.50, y=0.72, z=0.0, presence=None)
        mock_hit = MagicMock(face_landmarks=[mock_lms])
        mock_miss = MagicMock(face_landmarks=[])

        # Freeze wall-clock time.monotonic() to a constant value; advance ONLY frame timestamps!
        with patch("backend.app.vision.face_landmarker.time.monotonic", return_value=500.0):
            with patch.object(landmarker._landmarker, "detect_for_video", return_value=mock_hit):
                res_hit = landmarker.detect(dummy_img, timestamp_ms=1000)
                assert res_hit.has_face is True
                assert res_hit.status == FaceLandmarkStatus.TRACKING

            with patch.object(landmarker._landmarker, "detect_for_video", return_value=mock_miss):
                # 150ms later (< 300ms grace) -> FACE_LOST_GRACE
                res_grace = landmarker.detect(dummy_img, timestamp_ms=1150)
                assert res_grace.status == FaceLandmarkStatus.FACE_LOST_GRACE

                # 350ms later (>= 300ms grace) -> INSUFFICIENT_DATA even though wall clock did not move!
                res_expired = landmarker.detect(dummy_img, timestamp_ms=1350)
                assert res_expired.status == FaceLandmarkStatus.INSUFFICIENT_DATA
    finally:
        landmarker.close()
