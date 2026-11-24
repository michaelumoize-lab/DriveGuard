"""
DriveGuard Phase 2 Tests — Uncalibrated Driver Operational State Gating.

Verifies Decision 2 invariants on the Python AI engine:
1. Uncalibrated SessionAIEngine emits operational_state=CALIBRATION_REQUIRED and
   drowsiness_state=INSUFFICIENT_DATA across all frames (0 NORMAL frames), even with
   clean face detection and valid head pose.
2. Uncalibrated LocalMonitoringService emits operational_state=CALIBRATION_REQUIRED
   and drowsiness_state=INSUFFICIENT_DATA (0 NORMAL frames), and tags passive shadow
   observations with shadow_is_authoritative=False.
3. During active calibration, operational_state=CALIBRATING and drowsiness_state=INSUFFICIENT_DATA.
4. Once a calibrated feasible profile is loaded, operational_state transitions to
   MONITORING_ACTIVE and drowsiness_state becomes NORMAL.
"""

from __future__ import annotations

import numpy as np

from backend.app.ai.geometry import PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.intelligence.reason_codes import DrowsinessState, OperationalState
from backend.app.schemas.telemetry import AlertStateEnum, EngineTypeEnum, OperationalStateEnum
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringService,
)
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)


class _StubLandmarker:
    active_implementation = "SYNTHETIC_STUB"

    def __init__(self, ear: float = 0.30, yaw_deg: float = 0.0, pitch_deg: float = 0.0) -> None:
        self.ear = ear
        self.yaw_deg = yaw_deg
        self.pitch_deg = pitch_deg

    def detect(self, frame, timestamp_ms: int = 0) -> FaceLandmarkResult:
        lms = _make_synthetic_landmarks(
            ear_target=self.ear,
            yaw_deg=self.yaw_deg,
            pitch_deg=self.pitch_deg,
            is_mirrored=False,
            frame_width=320,
            frame_height=180,
        )
        return FaceLandmarkResult(
            has_face=True,
            status=FaceLandmarkStatus.TRACKING,
            landmarks=lms,
            timestamp_ms=float(timestamp_ms),
            inference_latency_ms=4.0,
            presence_confidence=0.98,
            reasons=(),
        )

    def close(self) -> None:
        pass


def _make_calibrated_profile(driver_id: str) -> DriverBehaviourProfile:
    zb = ZoneBaseline(
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
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.95,
        zone_baselines={PoseZone.CENTER: zb},
    )


def test_uncalibrated_session_ai_engine_never_emits_normal():
    stub = _StubLandmarker(ear=0.30)
    engine = SessionAIEngine(
        session_id="sess_uncal_p2",
        driver_id="drv_uncal_p2",
        profile=None,
        landmarker=stub,
        is_mirrored=False,
    )
    frame = _make_sharp_frame()
    try:
        for i in range(1, 25):
            dto = engine.process_frame(frame, timestamp_ms=float(i * 50), is_rgb=False)
            assert dto.operational_state == OperationalStateEnum.CALIBRATION_REQUIRED
            assert dto.engine_type == EngineTypeEnum.PYTHON
            assert dto.drowsiness_state == DrowsinessState.INSUFFICIENT_DATA.value
            assert dto.state == AlertStateEnum.MONITORING_UNAVAILABLE
            assert dto.alert_state is None
            assert "CALIBRATION_REQUIRED" in dto.reason_codes
            assert dto.cascade.arbitration_winner == "GEOMETRIC_AUTHORITATIVE"
            assert dto.cascade.cnn_execution_mode == "SHADOW_DISABLED"
            assert dto.cascade.onnx_inference_executed is False
    finally:
        engine.close()


def test_uncalibrated_local_monitoring_service_gates_until_calibrated():
    stub = _StubLandmarker(ear=0.30)
    service = LocalMonitoringService(
        config=LocalMonitoringConfig(driver_id="drv_local_uncal", is_mirrored=False, cnn_shadow_enabled=False),
        landmarker=stub,
        enable_landmarker=False,
        enable_shadow_worker=False,
        enable_audio=False,
        enable_event_logging=False,
        auto_load_profile=False,
        persist_calibration_to_store=False,
        initial_profile=None,
    )
    frame = _make_sharp_frame()
    try:
        # 1. Uncalibrated frames -> CALIBRATION_REQUIRED + INSUFFICIENT_DATA
        for i in range(1, 10):
            res = service.process_frame(frame, timestamp_ms=float(i * 50), frame_index=i)
            assert res.operational_state == OperationalState.CALIBRATION_REQUIRED
            assert res.assessment.operational_state == OperationalState.CALIBRATION_REQUIRED
            assert res.assessment.state == DrowsinessState.INSUFFICIENT_DATA
            assert res.telemetry.operational_state == OperationalState.CALIBRATION_REQUIRED.value
            assert res.telemetry.shadow_is_authoritative is False

        # 2. During calibration -> CALIBRATING + INSUFFICIENT_DATA
        service.start_calibration(timestamp_ms=500.0, is_mirrored=False, four_zone_mode=True, step_duration_sec=5.0)
        res_cal = service.process_frame(frame, timestamp_ms=550.0, frame_index=10)
        assert res_cal.operational_state == OperationalState.CALIBRATING
        assert res_cal.assessment.state == DrowsinessState.INSUFFICIENT_DATA
        service.cancel_calibration()

        # 3. After activating calibrated profile -> MONITORING_ACTIVE + NORMAL
        assert service.set_profile(_make_calibrated_profile("drv_local_uncal")) is True
        res_active = service.process_frame(frame, timestamp_ms=650.0, frame_index=11)
        assert res_active.operational_state == OperationalState.MONITORING_ACTIVE
        assert res_active.assessment.state == DrowsinessState.NORMAL
        assert res_active.telemetry.operational_state == OperationalState.MONITORING_ACTIVE.value
    finally:
        service.close()
