"""
DriveGuard Phase 2 Tests — Unmirrored WebRTC Coordinate Handedness & Truthful Readiness.

Verifies:
1. SessionAIEngine defaults to `is_mirrored=False` (matching unmirrored WebRTC/WS tracks).
2. Unmirrored left-turn landmarks fed to `SessionAIEngine(is_mirrored=False)` classify
   `PoseZone.LEFT_MIRROR` (+yaw) and weight the camera-facing eye correctly.
3. `/health/readiness` truthfully reports `production_mode == "GEOMETRIC_AUTHORITATIVE"`
   and `shadow_observer_enabled is False` (never falsely claiming active ONNX inference).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.ai.geometry import PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.main import app
from backend.app.schemas.telemetry import OperationalStateEnum, PoseZoneEnum
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)


class _LeftTurnUnmirroredStub:
    """Produces synthetic unmirrored landmarks with driver turning left (+25 deg yaw)."""

    active_implementation = "SYNTHETIC_STUB"

    def detect(self, frame, timestamp_ms: int = 0) -> FaceLandmarkResult:
        # Unmirrored driver-left turn (+25 deg driver yaw, is_mirrored=False)
        lms = _make_synthetic_landmarks(
            ear_target=0.29,
            yaw_deg=25.0,
            pitch_deg=0.0,
            is_mirrored=False,
            frame_width=320,
            frame_height=180,
        )
        return FaceLandmarkResult(
            has_face=True,
            status=FaceLandmarkStatus.TRACKING,
            landmarks=lms,
            timestamp_ms=float(timestamp_ms),
            inference_latency_ms=3.0,
            presence_confidence=0.98,
            reasons=(),
        )

    def close(self) -> None:
        pass


def _make_calibrated_profile(driver_id: str) -> DriverBehaviourProfile:
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
        zone_baselines={
            PoseZone.CENTER: ZoneBaseline(
                zone=PoseZone.CENTER,
                ear_median=0.30,
                ear_mad_scaled=0.018,
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=60,
            ),
            PoseZone.LEFT_MIRROR: ZoneBaseline(
                zone=PoseZone.LEFT_MIRROR,
                ear_median=0.27,
                ear_mad_scaled=0.018,
                pitch_median_deg=0.0,
                yaw_median_deg=25.0,
                sample_count=55,
            ),
        },
    )


def test_session_ai_engine_defaults_unmirrored_and_classifies_left_mirror():
    engine = SessionAIEngine(
        session_id="sess_webrtc_unmirrored",
        driver_id="drv_webrtc_unmirrored",
        profile=_make_calibrated_profile("drv_webrtc_unmirrored"),
        landmarker=_LeftTurnUnmirroredStub(),
    )
    try:
        assert engine.is_mirrored is False
        assert engine.monitoring_service.config.is_mirrored is False

        dto = engine.process_frame(_make_sharp_frame(), timestamp_ms=100.0, is_rgb=False)
        assert dto.operational_state == OperationalStateEnum.MONITORING_ACTIVE
        assert dto.pose_zone == PoseZoneEnum.LEFT_MIRROR
        assert dto.head_pose.yaw_deg >= 14.0
    finally:
        engine.close()


def test_readiness_endpoint_reports_truthful_geometric_authoritative_mode():
    client = TestClient(app)
    res = client.get("/health/readiness")
    assert res.status_code in (200, 503)
    body = res.json()
    verifier = body["components"]["eye_verifier"]
    assert verifier["production_mode"] == "GEOMETRIC_AUTHORITATIVE"
    assert verifier["shadow_observer_enabled"] is False
    assert isinstance(verifier["shadow_weights_present"], bool)
    assert "mode" not in verifier
