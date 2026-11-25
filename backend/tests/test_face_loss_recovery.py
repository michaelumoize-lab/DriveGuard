"""
Unit Tests for Workstream C — Face-loss and Recovery State Transitions.

Verifies on the authoritative SessionAIEngine -> LocalMonitoringService pipeline:
1. Transient single-frame face drops do not produce alert state bouncing.
2. Honest observation reporting: face_detected is False and fresh eye metrics are not fabricated.
3. Configurable grace period: sustained face loss transitions to MONITORING_UNAVAILABLE.
4. Temporal kinematics and gaze trackers do not accumulate invalid closure/dwell time across gaps.
5. Prior risk state and score are preserved during grace period rather than falsely claiming driver is safe.
6. Recovery restores MONITORING_ACTIVE cleanly.
"""

import numpy as np
import pytest

from backend.app.ai.geometry import PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.telemetry import AlertStateEnum, PoseZoneEnum
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)


def _make_calibrated_profile(driver_id: str = "driver_test") -> DriverBehaviourProfile:
    """Create a valid, feasible calibrated profile so ocular monitoring is active (HIGH-4)."""
    center_zb = ZoneBaseline(
        zone=PoseZone.CENTER,
        ear_median=0.30,
        ear_mad_scaled=0.018,
        pitch_median_deg=0.0,
        yaw_median_deg=0.0,
        sample_count=60,
    )
    return DriverBehaviourProfile(
        driver_id=driver_id,
        calibration_mode="4_ZONE",
        center_ear_median=0.30,
        center_ear_mad_scaled=0.018,
        blink_duration_median_ms=190.0,
        blink_duration_mad_ms=25.0,
        blink_rcr_median=1.05,
        blink_rcr_mad=0.15,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        zone_baselines={PoseZone.CENTER: center_zb},
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.95,
        calibration_is_mirrored=True,
    )


class ControllableMockLandmarker:
    """Authoritative mock landmarker allowing dynamic manipulation of face visibility and pose."""

    def __init__(
        self,
        face_detected: bool = True,
        ear: float = 0.28,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        self.face_detected = face_detected
        self.ear = ear
        self.pitch = pitch
        self.yaw = yaw
        self.active_implementation = "MOCK_TASKS"

    def detect(self, input_data, timestamp_ms: int = 0) -> FaceLandmarkResult:
        if not self.face_detected:
            return FaceLandmarkResult(
                has_face=False,
                landmarks=(),
                presence_confidence=0.0,
                timestamp_ms=int(timestamp_ms or 0),
                status=FaceLandmarkStatus.INSUFFICIENT_DATA,
                inference_latency_ms=3.0,
                reasons=("NO_FACE",),
            )

        return FaceLandmarkResult(
            has_face=True,
            landmarks=_make_synthetic_landmarks(
                ear_target=self.ear,
                yaw_deg=self.yaw,
                pitch_deg=self.pitch,
                is_mirrored=True,
            ),
            presence_confidence=0.96,
            timestamp_ms=int(timestamp_ms or 0),
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=3.0,
            reasons=("OK",),
        )

    def close(self) -> None:
        pass


@pytest.fixture
def dummy_frame() -> np.ndarray:
    return _make_sharp_frame(width=640, height=480)


def _make_calibrated_face_loss_profile(driver_id: str = "driver_fl"):
    from backend.app.ai.geometry import PoseZone
    from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline

    return DriverBehaviourProfile(
        driver_id=driver_id,
        calibration_mode="4_ZONE",
        center_ear_median=0.28,
        center_ear_mad_scaled=0.016,
        blink_duration_median_ms=200.0,
        blink_duration_mad_ms=25.0,
        blink_rcr_median=1.0,
        blink_rcr_mad=0.15,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.95,
        zone_baselines={
            PoseZone.CENTER: ZoneBaseline(
                zone=PoseZone.CENTER,
                ear_median=0.28,
                ear_mad_scaled=0.016,
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=60,
            )
        },
    )


def test_single_frame_face_drop_does_not_bounce_state(dummy_frame):
    """A transient single-frame landmark drop must not bounce state to MONITORING_UNAVAILABLE."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(
        landmarker=mock_lm,
        profile=_make_calibrated_face_loss_profile(),
        face_loss_grace_ms=450.0,
        is_mirrored=True,
    )
    try:
        ts = 1000.0
        for _ in range(3):
            dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
            assert dto.state == AlertStateEnum.NORMAL
            assert dto.face_detected is True
            assert dto.monitoring_status == "MONITORING_ACTIVE"
            ts += 33.3

        # Drop 1 frame
        mock_lm.face_detected = False
        dto_drop = engine.process_frame(dummy_frame, timestamp_ms=ts)

        assert dto_drop.face_detected is False
        assert dto_drop.eye_metrics.ear_raw == 0.0
        assert dto_drop.eye_metrics.is_eye_closed is False
        assert dto_drop.eye_metrics.blink_state == "UNCERTAIN_GAP"
        assert dto_drop.monitoring_status == "FACE_VISIBILITY_DEGRADED"
        assert dto_drop.state == AlertStateEnum.NORMAL
        ts += 33.3

        # Next frame face returns
        mock_lm.face_detected = True
        dto_recover = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto_recover.face_detected is True
        assert dto_recover.monitoring_status == "MONITORING_ACTIVE"
        assert dto_recover.state == AlertStateEnum.NORMAL
    finally:
        engine.close()


def test_sustained_face_loss_exceeding_grace_period_transitions_to_unavailable(dummy_frame):
    """Face loss lasting >= face_loss_grace_ms must transition to MONITORING_UNAVAILABLE."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(
        landmarker=mock_lm,
        profile=_make_calibrated_face_loss_profile(),
        face_loss_grace_ms=450.0,
        is_mirrored=True,
    )
    try:
        ts = 2000.0
        dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto.state == AlertStateEnum.NORMAL

        mock_lm.face_detected = False
        ts += 33.3

        while ts <= 2400.0:
            dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
            assert dto.face_detected is False
            assert dto.monitoring_status == "FACE_VISIBILITY_DEGRADED"
            assert dto.state == AlertStateEnum.NORMAL
            ts += 33.3

        ts = 2500.0
        dto_unavail = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto_unavail.face_detected is False
        assert dto_unavail.monitoring_status == "MONITORING_UNAVAILABLE"
        assert dto_unavail.state == AlertStateEnum.MONITORING_UNAVAILABLE
    finally:
        engine.close()


def test_recovery_from_sustained_face_loss_restores_monitoring(dummy_frame):
    """Recovery after sustained UNAVAILABLE transitions through RECOVERING to ACTIVE."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(
        landmarker=mock_lm,
        profile=_make_calibrated_face_loss_profile(),
        face_loss_grace_ms=300.0,
        is_mirrored=True,
    )
    try:
        ts = 3000.0
        engine.process_frame(dummy_frame, timestamp_ms=ts)

        mock_lm.face_detected = False
        ts += 33.3
        loss_start = ts
        while (ts - loss_start) < 600.0:
            dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
            ts += 33.3

        assert dto.monitoring_status == "MONITORING_UNAVAILABLE"
        assert dto.state == AlertStateEnum.MONITORING_UNAVAILABLE

        mock_lm.face_detected = True
        dto_rec1 = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto_rec1.face_detected is True
        assert dto_rec1.monitoring_status == "MONITORING_RECOVERING"
        ts += 33.3

        dto_rec2 = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto_rec2.face_detected is True
        assert dto_rec2.monitoring_status == "MONITORING_RECOVERING"
        ts += 33.3

        dto_rec3 = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto_rec3.face_detected is True
        assert dto_rec3.monitoring_status == "MONITORING_ACTIVE"
        assert dto_rec3.state == AlertStateEnum.NORMAL
    finally:
        engine.close()


def test_gaze_dwell_timer_does_not_accumulate_across_face_loss_gap(dummy_frame):
    """Head-orientation dwell timers must not accumulate unobserved time across a face-loss gap."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28, pitch=-15.0, yaw=0.0)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=400.0)
    try:
        ts = 4000.0
        for _ in range(12):
            dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
            assert dto.pose_zone == PoseZoneEnum.DASHBOARD
            ts += 33.3

        # Camera/face drops for 3000ms
        mock_lm.face_detected = False
        gap_start = ts
        while (ts - gap_start) < 3000.0:
            engine.process_frame(dummy_frame, timestamp_ms=ts)
            ts += 100.0

        # Face returns looking at DASHBOARD
        mock_lm.face_detected = True
        dto_after = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto_after.face_detected is True
        assert dto_after.pose_zone == PoseZoneEnum.DASHBOARD
        assert "PERSISTENT_DISTRACTION" not in dto_after.reason_codes
        assert "OFF_ROAD_DWELL_EXCEEDED" not in dto_after.reason_codes
    finally:
        engine.close()


def test_preserves_warning_risk_state_during_grace_period(dummy_frame):
    """Engine must preserve prior WARNING/CRITICAL state during grace period without false safe claim."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.11)
    engine = SessionAIEngine(
        landmarker=mock_lm,
        profile=_make_calibrated_profile(),
        face_loss_grace_ms=450.0,
    )
    try:
        ts = 5000.0
        warn_triggered = False
        dto = None
        for _ in range(70):
            dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
            if dto.state in (AlertStateEnum.WARNING, AlertStateEnum.CRITICAL):
                warn_triggered = True
                break
            ts += 33.3

        assert warn_triggered, "Expected WARNING or CRITICAL state from sustained low EAR."
        last_state = dto.state
        last_risk = dto.risk_score
        assert last_risk > 0.0

        mock_lm.face_detected = False
        ts += 33.3
        dto_drop = engine.process_frame(dummy_frame, timestamp_ms=ts)

        assert dto_drop.state == last_state
        assert dto_drop.risk_score == pytest.approx(last_risk, abs=1e-3)
        assert dto_drop.monitoring_status == "FACE_VISIBILITY_DEGRADED"
    finally:
        engine.close()


def test_configurable_grace_period(dummy_frame):
    """Engine must honor custom face_loss_grace_ms."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=150.0)
    try:
        ts = 6000.0
        engine.process_frame(dummy_frame, timestamp_ms=ts)

        mock_lm.face_detected = False
        ts += 50.0
        dto1 = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto1.monitoring_status == "FACE_VISIBILITY_DEGRADED"

        ts += 200.0
        dto2 = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto2.monitoring_status == "MONITORING_UNAVAILABLE"
        assert dto2.state == AlertStateEnum.MONITORING_UNAVAILABLE
    finally:
        engine.close()
