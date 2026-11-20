"""
DriveGuard — End-to-End Authoritative Local Pipeline Integration Tests (Phase 5 / TEST-1).

Wires the real authoritative local pipeline end-to-end:
  FrameQualityAnalyzer
    -> DriverFeatureExtractor (EAR, Head Pose, MAR/Yawn, Blink Kinematics, PERCLOS)
    -> DrowsinessEngine (SignalConfidenceEvaluator -> TemporalAnalyzer -> DecisionEngine)
    -> AlertStateManager
    -> AlertDispatcher + AlertEventLogger
  plus GuidedCalibrationSession -> commit_and_activate_profile -> ProfileStore.

Exercises 15 realistic multi-frame driving and failure scenarios without mocking internal contracts.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
)
from backend.app.alerts.alert_dispatcher import AlertDispatcher
from backend.app.alerts.alert_state import AlertState
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.alerts.audio_settings import AudioSettingsStore
from backend.app.alerts.event_logger import AlertEventLogger
from backend.app.camera.frame_quality import FrameQualityAnalyzer
from backend.app.features.blink_detector import BlinkState
from backend.app.features.feature_extractor import DriverFeatureExtractor
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    PoseZone,
    SOLVEPNP_LANDMARK_INDICES,
)
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.services.profile_store import ProfileStore
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)
from scripts.run_camera_test import (
    commit_and_activate_profile,
    compute_calibration_geometric_reliability,
)


# ---------------------------------------------------------------------------
# Synthetic Frame & 3D/2D Landmark Generators (Physics-Consistent)
# ---------------------------------------------------------------------------

def _make_rot_mat(yaw_deg: float, pitch_deg: float, roll_deg: float, is_mirrored: bool = True) -> np.ndarray:
    cam_yaw = yaw_deg if is_mirrored else -yaw_deg
    cam_pitch = -pitch_deg
    cam_roll = -roll_deg if is_mirrored else roll_deg
    psi, theta, phi = np.radians(cam_yaw), np.radians(cam_pitch), np.radians(cam_roll)
    Ry = np.array([[np.cos(psi), 0.0, np.sin(psi)], [0.0, 1.0, 0.0], [-np.sin(psi), 0.0, np.cos(psi)]])
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]])
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]])
    return Ry @ Rx @ Rz


def _make_sharp_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """High-contrast textured frame -> QualityStatus.EXCELLENT."""
    rng = np.random.default_rng(42)
    base = np.full((height, width, 3), 120, dtype=np.uint8)
    # Checkerboard edges produce high Laplacian variance (> 200)
    for y in range(0, height, 8):
        for x in range(0, width, 8):
            if ((x // 8) + (y // 8)) % 2 == 0:
                base[y : y + 8, x : x + 8] = 165
            else:
                base[y : y + 8, x : x + 8] = 85
    noise = rng.integers(-5, 6, size=(height, width, 3), dtype=np.int16)
    return np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _make_mild_blur_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """Textured mid-luminance frame with mild defocus -> QualityStatus.BLURRY (20 < var < 80, still viable)."""
    sharp = _make_sharp_frame(width=width, height=height)
    # Apply moderate blur so Laplacian variance lands cleanly in (25, 70) -> BLURRY & viable
    return cv2.GaussianBlur(sharp, (7, 7), 1.8)


def _make_pitch_black_frame(width: int = 320, height: int = 180) -> np.ndarray:
    """Pitch-black uniform frame -> QualityStatus.SEVERELY_DEGRADED (not viable)."""
    return np.full((height, width, 3), 5, dtype=np.uint8)


def _make_synthetic_landmarks(
    ear_target: float = 0.30,
    mar_target: float = 0.12,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    is_mirrored: bool = True,
    corrupt_pose: bool = False,
    frame_width: int = 1280,
    frame_height: int = 720,
) -> Tuple[NormalizedLandmark, ...]:
    """
    Construct a 478-point landmark set with physically projected SolvePnP cranial anchors
    plus exact EAR eye contours and MAR mouth contours.
    """
    ar = frame_width / float(frame_height)
    lms: List[NormalizedLandmark] = [NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478)]

    # 1. Project 8 SolvePnP anchors from exact 3D rotation (unless corrupt_pose requested)
    if not corrupt_pose:
        rot_mat = _make_rot_mat(yaw_deg, pitch_deg, roll_deg, is_mirrored=is_mirrored)
        focal = float(frame_width)
        cam_mat = np.array(
            [[focal, 0.0, frame_width / 2.0], [0.0, focal, frame_height / 2.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        rvec, _ = cv2.Rodrigues(rot_mat)
        tvec = np.array([[0.0], [0.0], [600.0]], dtype=np.float64)
        proj, _ = cv2.projectPoints(CANONICAL_3D_FACE_POINTS, rvec, tvec, cam_mat, np.zeros((4, 1)))
        pts = proj.reshape(-1, 2)
        for i, idx in enumerate(SOLVEPNP_LANDMARK_INDICES):
            lms[idx] = NormalizedLandmark(float(pts[i, 0] / frame_width), float(pts[i, 1] / frame_height), 0.0)
    else:
        # Collapse non-eye pose anchors (nose/chin/sellion/glabella) so SolvePnP fails
        # while eye landmarks stay completely intact and geometrically valid!
        for idx in (1, 152, 6, 10):
            lms[idx] = NormalizedLandmark(float("nan"), float("nan"), 0.0)

    # 2. Place Left Eye (33, 160, 158, 133, 153, 144) & Right Eye (362, 385, 387, 263, 373, 380)
    # Note: 33 and 263 are shared with SolvePnP; place inner/vertical points relative to actual 33 & 263
    # so EAR equals ear_target accurately while keeping SolvePnP corners untouched!
    def _set_eye(outer_idx: int, inner_idx: int, top1: int, top2: int, bot1: int, bot2: int, sign: float) -> None:
        ox, oy = lms[outer_idx].x, lms[outer_idx].y
        if not (np.isfinite(ox) and np.isfinite(oy)):
            ox, oy = (0.40 if sign > 0 else 0.60), 0.42
            lms[outer_idx] = NormalizedLandmark(ox, oy, 0.0)
        dx_norm = 0.045 * sign  # horizontal width in normalized x
        ix, iy = ox + dx_norm, oy
        lms[inner_idx] = NormalizedLandmark(ix, iy, 0.0)
        horiz_iso = abs(dx_norm) * ar
        vert_iso = ear_target * horiz_iso
        half_v = vert_iso / 2.0
        x_a = ox + 0.33 * dx_norm
        x_b = ox + 0.67 * dx_norm
        lms[top1] = NormalizedLandmark(x_a, oy - half_v, 0.0)
        lms[bot1] = NormalizedLandmark(x_a, oy + half_v, 0.0)
        lms[top2] = NormalizedLandmark(x_b, oy - half_v, 0.0)
        lms[bot2] = NormalizedLandmark(x_b, oy + half_v, 0.0)

    _set_eye(33, 133, 160, 158, 144, 153, sign=+1.0)
    _set_eye(263, 362, 387, 385, 373, 380, sign=-1.0)

    # 3. Place inner mouth landmarks relative to actual corners (61, 291) so MAR equals mar_target
    lx, ly = lms[61].x, lms[61].y
    rx, ry = lms[291].x, lms[291].y
    if not (np.isfinite(lx) and np.isfinite(rx)):
        lx, ly, rx, ry = 0.44, 0.66, 0.56, 0.66
        lms[61] = NormalizedLandmark(lx, ly, 0.0)
        lms[291] = NormalizedLandmark(rx, ry, 0.0)
    mx, my = (lx + rx) / 2.0, (ly + ry) / 2.0
    m_horiz = float(np.hypot((rx - lx) * ar, ry - ly))
    m_half_v = (mar_target * m_horiz) / 2.0
    for frac, top_idx, bot_idx in ((0.50, 13, 14), (0.35, 81, 178), (0.65, 311, 402)):
        px = lx + frac * (rx - lx)
        py = ly + frac * (ry - ly)
        lms[top_idx] = NormalizedLandmark(px, py - m_half_v, 0.0)
        lms[bot_idx] = NormalizedLandmark(px, py + m_half_v, 0.0)

    return tuple(lms)


class _PipelineHarness:
    """Thin real-component harness matching the authoritative local pipeline."""

    def __init__(self, tmp_path: Path, is_mirrored: bool = True) -> None:
        self.is_mirrored = is_mirrored
        self.quality_analyzer = FrameQualityAnalyzer()
        self.extractor = DriverFeatureExtractor(is_mirrored=is_mirrored)
        self.engine = DrowsinessEngine()
        self.alert_manager = AlertStateManager()
        self.mock_audio = MagicMock()
        self.mock_audio.enqueue_alert.return_value = True
        self.mock_audio.enqueue_recovery.return_value = True
        self.settings_store = AudioSettingsStore(settings_file=tmp_path / "audio.json")
        self.settings_store.set_audio_enabled(True)
        self.event_logger = AlertEventLogger(log_file=tmp_path / "events.jsonl", enable_file_logging=True)
        self.dispatcher = AlertDispatcher(
            audio_engine=self.mock_audio,
            settings_store=self.settings_store,
            event_logger=self.event_logger,
            critical_reminder_cooldown_ms=1000.0,
        )
        self.store = ProfileStore(storage_dir=tmp_path / "profiles")
        self.sharp_frame = _make_sharp_frame()
        self.blur_frame = _make_mild_blur_frame()
        self.black_frame = _make_pitch_black_frame()

    def step(
        self,
        ts_ms: float,
        ear: float = 0.30,
        mar: float = 0.12,
        yaw_deg: float = 0.0,
        pitch_deg: float = 0.0,
        roll_deg: float = 0.0,
        has_face: bool = True,
        frame: Optional[np.ndarray] = None,
        corrupt_pose: bool = False,
        face_status: Optional[FaceLandmarkStatus] = None,
        is_mirrored: Optional[bool] = None,
    ):
        mir = self.is_mirrored if is_mirrored is None else is_mirrored
        img = self.sharp_frame if frame is None else frame
        quality = self.quality_analyzer.analyze(img)

        if has_face:
            lms = _make_synthetic_landmarks(
                ear_target=ear,
                mar_target=mar,
                yaw_deg=yaw_deg,
                pitch_deg=pitch_deg,
                roll_deg=roll_deg,
                is_mirrored=mir,
                corrupt_pose=corrupt_pose,
            )
            lm_res = FaceLandmarkResult(
                has_face=True,
                landmarks=lms,
                presence_confidence=0.96,
                timestamp_ms=int(ts_ms),
                status=face_status or FaceLandmarkStatus.TRACKING,
                inference_latency_ms=6.0,
                reasons=("OK",),
            )
        else:
            lm_res = FaceLandmarkResult(
                has_face=False,
                landmarks=(),
                presence_confidence=0.0,
                timestamp_ms=int(ts_ms),
                status=face_status or FaceLandmarkStatus.INSUFFICIENT_DATA,
                inference_latency_ms=4.0,
                reasons=("Face lost",),
            )

        feats = self.extractor.extract(lm_res, frame_width=1280, frame_height=720, is_mirrored=mir)
        assessment = self.engine.evaluate(features=feats, quality=quality)
        transition = self.alert_manager.update(assessment)
        self.dispatcher.dispatch(
            transition=transition,
            current_state=self.alert_manager.current_state,
            timestamp_ms=ts_ms,
            is_recovering=self.alert_manager.is_recovering,
        )
        return quality, lm_res, feats, assessment, transition

    def calibrate_driver(
        self,
        driver_id: str = "e2e_driver",
        neutral_yaw: float = 0.0,
        neutral_pitch: float = 0.0,
        ear_open: float = 0.30,
        start_ts_ms: float = 0.0,
    ) -> Tuple[DriverBehaviourProfile, float]:
        session = GuidedCalibrationSession(
            driver_id=driver_id,
            four_zone_mode=True,
            step_duration_sec=5.0,
            is_mirrored=self.is_mirrored,
        )
        session.target_samples_per_zone = 22
        session.start(timestamp_ms=start_ts_ms)

        zones = [
            (neutral_yaw, neutral_pitch, ear_open),
            (neutral_yaw + 20.0, neutral_pitch, ear_open * 0.94),
            (neutral_yaw - 20.0, neutral_pitch, ear_open * 0.94),
            (neutral_yaw, neutral_pitch - 11.0, ear_open * 0.90),
        ]
        ts = start_ts_ms
        completed: Optional[DriverBehaviourProfile] = None
        for idx, (y_deg, p_deg, e_val) in enumerate(zones):
            if idx > 0:
                ts += 2100.0  # Clear 2.0s inter-zone grace
            for _ in range(26):
                ts += 40.0
                q, lm, feats, _, _ = self.step(ts, ear=e_val, yaw_deg=y_deg, pitch_deg=p_deg)
                geom_rel = compute_calibration_geometric_reliability(lm, feats, q)
                prof = session.ingest_frame(
                    timestamp_ms=ts,
                    face_detected=bool(lm.has_face and q.is_viable_for_inference),
                    geometric_reliability=geom_rel,
                    ear_value=feats.ear.weighted_ear,
                    pitch_deg=feats.head_pose.pitch_deg,
                    yaw_deg=feats.head_pose.yaw_deg,
                    is_eye_currently_closed=(feats.ear.is_closed if self.extractor.is_profile_feasible else None),
                    left_ear=feats.ear.left_ear,
                    right_ear=feats.ear.right_ear,
                    landmark_confidence=lm.presence_confidence,
                    rot_mat=feats.head_pose.rot_mat,
                    ear_geometric_valid=feats.ear.is_geometric_valid,
                )
                if prof is not None:
                    completed = prof
                    break
            if completed is not None:
                break

        assert completed is not None, f"Calibration failed: {session.rejection_reason}"
        ok, msg = commit_and_activate_profile(completed, self.extractor, store=self.store)
        assert ok, f"commit_and_activate_profile failed: {msg}"
        return completed, ts

    def close(self) -> None:
        self.dispatcher.shutdown()


# ---------------------------------------------------------------------------
# End-to-End Integration Scenarios (1 through 15)
# ---------------------------------------------------------------------------

def test_e2e_01_uncalibrated_startup_disables_ocular_keeps_pose_and_yawn(tmp_path: Path) -> None:
    h = _PipelineHarness(tmp_path)
    try:
        assert h.extractor.is_profile_feasible is False
        # Even with very low EAR, uncalibrated extractor must NOT trigger false eye-closure alerts
        for i in range(60):
            _, _, feats, assessment, _ = h.step(1000.0 + i * 33.0, ear=0.10)
            assert feats.ear.is_valid is False
            assert feats.head_pose.is_valid is True
            assert assessment.state != DrowsinessState.CRITICAL_WARNING
    finally:
        h.close()


def test_e2e_02_and_03_four_zone_calibration_and_alert_driving_normal_blinks(tmp_path: Path) -> None:
    h = _PipelineHarness(tmp_path, is_mirrored=True)
    try:
        prof, ts = h.calibrate_driver(driver_id="alice", neutral_yaw=0.0, neutral_pitch=0.0, ear_open=0.30)
        assert h.extractor.is_profile_feasible is True
        assert prof.calibration_is_mirrored is True
        assert prof.neutral_rot_mat is not None
        assert h.store.get_profile("alice") is not None

        # Simulate 3 normal alert blinks (~150ms each) separated by 2s
        h.engine.reset()
        h.alert_manager.reset()
        t = ts + 500.0
        for blink_idx in range(3):
            for _ in range(15):
                t += 33.0
                _, _, _, a, _ = h.step(t, ear=0.30)
                assert h.alert_manager.current_state == AlertState.NORMAL
            for ear_v in (0.24, 0.14, 0.12, 0.16, 0.28):
                t += 33.0
                _, _, feats, a, _ = h.step(t, ear=ear_v)
            assert h.alert_manager.current_state == AlertState.NORMAL
        assert feats.blink.total_blink_count >= 3
    finally:
        h.close()


def test_e2e_04_sustained_closure_with_noisy_intermediate_ear_triggers_critical(tmp_path: Path) -> None:
    """CRITICAL-2 E2E: noisy intermediate EAR inside (c, r) must not break a 1600ms microsleep."""
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        c_thr = h.extractor.active_closure_threshold
        r_thr = h.extractor.active_reopening_threshold
        mid_ear = (c_thr + r_thr) / 2.0

        t = ts + 500.0
        for _ in range(10):
            t += 33.0
            h.step(t, ear=0.30)

        # 1700ms continuous closure with noisy intermediate EAR frames at steps 20 and 35
        for step in range(55):
            t += 33.0
            ear_val = mid_ear if step in (20, 35) else 0.11
            _, _, feats, assessment, _ = h.step(t, ear=ear_val)

        assert assessment.state == DrowsinessState.CRITICAL_WARNING
        assert h.alert_manager.current_state == AlertState.CRITICAL
    finally:
        h.close()


def test_e2e_05_sustained_closure_under_head_pose_failure_and_mild_blur_triggers_critical(tmp_path: Path) -> None:
    """CRITICAL-1 E2E: head-pose failure + mild frame blur must NOT disable ocular microsleep detection."""
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        t = ts + 500.0
        for _ in range(5):
            t += 33.0
            h.step(t, ear=0.30)

        # Close eyes for ~1750ms while head pose fails (corrupt_pose=True) AND frame has mild blur
        q_last = None
        for _ in range(55):
            t += 33.0
            q_last, _, feats, assessment, _ = h.step(
                t, ear=0.11, frame=h.blur_frame, corrupt_pose=True
            )

        assert q_last is not None and q_last.is_viable_for_inference is True
        assert feats.head_pose.is_valid is False
        assert assessment.confidence.is_reliable is False
        assert assessment.confidence.ocular_viable is True
        assert assessment.state == DrowsinessState.CRITICAL_WARNING
        assert h.alert_manager.current_state == AlertState.CRITICAL
    finally:
        h.close()


def test_e2e_06_nodding_microsleep_triggers_critical(tmp_path: Path) -> None:
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        t = ts + 500.0
        for _ in range(5):
            t += 33.0
            h.step(t, ear=0.30, pitch_deg=0.0)

        # Eyes closed for ~900ms while head nods down to -18 deg
        for _ in range(30):
            t += 33.0
            _, _, _, assessment, _ = h.step(t, ear=0.11, pitch_deg=-18.0)

        assert assessment.state == DrowsinessState.CRITICAL_WARNING
        assert any(r.code == ReasonCode.NODDING_MICROSLEEP for r in assessment.decision.reasons)
    finally:
        h.close()


def test_e2e_07_authentic_yawn_triggers_attention(tmp_path: Path) -> None:
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        t = ts + 500.0
        # Open mouth wide (MAR=0.68) for 1800ms, then close
        for _ in range(56):
            t += 33.0
            _, _, feats, assessment, _ = h.step(t, ear=0.30, mar=0.68)
        t += 33.0
        _, _, feats_end, assessment_end, _ = h.step(t, ear=0.30, mar=0.12)

        assert feats_end.mar.total_yawn_count >= 1
        assert assessment_end.state == DrowsinessState.ATTENTION
        assert h.alert_manager.current_state == AlertState.ATTENTION
        assert any(r.code == ReasonCode.AUTHENTIC_YAWN for r in assessment_end.decision.reasons)
    finally:
        h.close()


def test_e2e_08_and_09_mirror_check_vs_persistent_off_road_distraction(tmp_path: Path) -> None:
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        t = ts + 500.0
        # Scenario 8: Brief 1.5s LEFT_MIRROR glance (+24 deg yaw) -> NORMAL with MIRROR_CHECK_CONTEXT
        for _ in range(45):
            t += 33.0
            _, _, feats, assessment, _ = h.step(t, ear=0.28, yaw_deg=24.0)
        assert feats.head_pose.zone == PoseZone.LEFT_MIRROR
        assert assessment.is_mirror_check_context is True
        assert h.alert_manager.current_state == AlertState.NORMAL

        # Scenario 9: Continue looking away for > 3.5s -> WARNING (PERSISTENT_DISTRACTION)
        for _ in range(80):
            t += 33.0
            _, _, _, assessment, _ = h.step(t, ear=0.28, yaw_deg=24.0)
        assert assessment.state == DrowsinessState.DROWSINESS_WARNING
        assert h.alert_manager.current_state == AlertState.WARNING
        assert any(r.code == ReasonCode.PERSISTENT_DISTRACTION for r in assessment.decision.reasons)
    finally:
        h.close()


def test_e2e_10_and_11_face_loss_during_critical_closed_reacquire_and_genuine_recovery(tmp_path: Path) -> None:
    """HIGH-3 & MEDIUM-1 E2E: face loss preserves CRITICAL, closed-eye reacquire != recovering, open-eye 2s recovers."""
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        t = ts + 500.0
        for _ in range(5):
            t += 33.0
            h.step(t, ear=0.30)

        # Escalate to CRITICAL via 1650ms eye closure
        for _ in range(52):
            t += 33.0
            h.step(t, ear=0.11)
        assert h.alert_manager.current_state == AlertState.CRITICAL

        # Face lost for 1500ms -> INSUFFICIENT_DATA in DecisionEngine, AlertStateManager stays CRITICAL
        for _ in range(46):
            t += 33.0
            h.step(t, has_face=False)
        assert h.alert_manager.current_state == AlertState.CRITICAL
        assert h.alert_manager.is_recovering is False

        # Reacquire with eyes STILL CLOSED -> must NOT mark is_recovering=True
        for _ in range(10):
            t += 33.0
            h.step(t, ear=0.11)
        assert h.alert_manager.current_state == AlertState.CRITICAL
        assert h.alert_manager.is_recovering is False

        # Now open eyes wide and hold steady for 2200ms -> clean recovery to NORMAL
        for _ in range(70):
            t += 33.0
            h.step(t, ear=0.30)
        assert h.alert_manager.current_state == AlertState.NORMAL
        assert h.alert_manager.is_recovering is False

        # Verify wall-clock UTC ISO timestamp in AlertEventLogger (not 1970)
        entries = h.event_logger.get_recent_entries(limit=20)
        assert len(entries) >= 2
        for entry in entries:
            assert not entry.iso_time.startswith("1970-")
            assert datetime.fromisoformat(entry.iso_time).year == datetime.now(timezone.utc).year
    finally:
        h.close()


def test_e2e_12_oblique_camera_mirror_toggle_preserves_centered_pose(tmp_path: Path) -> None:
    """HIGH-7 E2E: calibrate oblique (+20 deg yaw, -6 deg pitch) mirrored -> toggle unmirrored -> stays CENTER."""
    h = _PipelineHarness(tmp_path, is_mirrored=True)
    try:
        prof, ts = h.calibrate_driver(
            driver_id="oblique_bob",
            neutral_yaw=20.0,
            neutral_pitch=-6.0,
            ear_open=0.30,
        )
        assert prof.calibration_is_mirrored is True

        # Toggle runtime camera to unmirrored (is_mirrored=False) while driver sits in exact same neutral pose
        t = ts + 500.0
        _, _, feats, _, _ = h.step(t, ear=0.30, yaw_deg=20.0, pitch_deg=-6.0, is_mirrored=False)
        assert feats.head_pose.is_valid is True
        assert feats.head_pose.zone == PoseZone.CENTER
        assert abs(feats.head_pose.rel_yaw_deg) < 2.5
        assert abs(feats.head_pose.rel_pitch_deg) < 2.5
    finally:
        h.close()


def test_e2e_13_recalibration_during_active_blink_defers_and_promotes_safely(tmp_path: Path) -> None:
    """Deferred safe-latch promotion when committing a new profile mid-blink."""
    h = _PipelineHarness(tmp_path)
    try:
        prof1, ts = h.calibrate_driver(driver_id="driver_v1", ear_open=0.30)
        t = ts + 500.0
        # Open -> enter CLOSED mid-blink
        h.step(t, ear=0.30)
        t += 33.0
        _, _, feats_closed, _, _ = h.step(t, ear=0.12)
        assert feats_closed.blink.state == BlinkState.CLOSED

        # Build a second distinct valid profile and commit mid-blink -> must queue as PENDING
        prof2 = DriverBehaviourProfile.load_json if False else prof1
        import copy
        prof2 = copy.deepcopy(prof1)
        prof2.driver_id = "driver_v2"
        prof2.center_ear_median = 0.32
        prof2.zone_baselines["CENTER"].ear_median = 0.32

        ok, msg = commit_and_activate_profile(prof2, h.extractor, store=h.store)
        assert ok is True
        assert "deferred" in msg.lower()
        assert h.extractor.has_pending_profile is True
        assert h.extractor.profile.driver_id == "driver_v1"

        # Eyes reopen above reopen_threshold -> safe-latch promotes driver_v2 on next open frame
        t += 33.0
        h.step(t, ear=0.32)
        t += 33.0
        h.step(t, ear=0.32)
        assert h.extractor.has_pending_profile is False
        assert h.extractor.profile.driver_id == "driver_v2"
    finally:
        h.close()


def test_e2e_14_severely_degraded_optical_frame_enters_insufficient_data(tmp_path: Path) -> None:
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        t = ts + 500.0
        # Pitch-black non-viable frames for 1300ms -> INSUFFICIENT_DATA, no false warning
        for _ in range(42):
            t += 33.0
            q, _, _, assessment, _ = h.step(t, ear=0.12, frame=h.black_frame)
        assert q.is_viable_for_inference is False
        assert assessment.state == DrowsinessState.INSUFFICIENT_DATA
        assert h.alert_manager.current_state == AlertState.NORMAL
    finally:
        h.close()


def test_e2e_15_stalled_reopening_squint_times_out_cleanly(tmp_path: Path) -> None:
    """HIGH-2 E2E: stalled REOPENING > 1000ms resets cleanly to OPEN without fabricated MICROSLEEP blink."""
    h = _PipelineHarness(tmp_path)
    try:
        _, ts = h.calibrate_driver(ear_open=0.30)
        h.engine.reset()
        h.alert_manager.reset()

        c_thr = h.extractor.active_closure_threshold
        r_thr = h.extractor.active_reopening_threshold
        mid_ear = (c_thr + r_thr) / 2.0

        t = ts + 500.0
        h.step(t, ear=0.30)
        t += 33.0
        h.step(t, ear=0.12)
        t += 33.0
        _, _, f_reopen, _, _ = h.step(t, ear=mid_ear)
        assert f_reopen.blink.state == BlinkState.REOPENING

        # Stall in (c, r) for 1200ms
        for _ in range(40):
            t += 33.0
            _, _, f_last, _, _ = h.step(t, ear=mid_ear)

        assert f_last.blink.state == BlinkState.OPEN
        # Reopen to 0.30 -> must not fabricate a multi-second blink
        t += 33.0
        _, _, f_after, _, _ = h.step(t, ear=0.30)
        assert f_after.blink.completed_blink is None
    finally:
        h.close()
