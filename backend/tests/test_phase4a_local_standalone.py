"""
DriveGuard — Phase 4A Mandatory Test B: Standalone Python AI Local Monitoring E2E.

Proves Python AI operates 100% autonomously with:
- Zero Next.js server / Better Auth tokens
- Zero Browser AI
- Zero LiveKit configuration
- Zero Neon PostgreSQL connectivity
- Direct local camera acquisition (mocked BaseCamera tracking open/close lifecycle)
- Local JSON ProfileStore hydration (`engine_type="PYTHON"`)
- Loopback `/ws/local/telemetry` streaming normalized telemetry + CRITICAL SAFETY_EVENT
- On-demand `/api/v1/local/preview.mjpg` MJPEG frame streaming & viewer cleanup
- SQLite `local_sessions` + `local_alerts` persistence
- Synchronous camera hardware release (`camera.close()` called before HTTP 200 on `/api/v1/local/stop`)
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.api.routes.local_monitor import get_local_camera_controller
from backend.app.camera.base import BaseCamera, FramePacket
from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO
from backend.app.services.local_camera_controller import LocalCameraController
from backend.app.services.local_db import LocalDB
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringService,
)
from backend.app.services.profile_store import ProfileStore
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


def _build_synthetic_landmarks(ear_scale: float = 1.0) -> Tuple[NormalizedLandmark, ...]:
    pts = [NormalizedLandmark(x=0.5, y=0.5, z=0.0) for _ in range(478)]
    # Cranial anchors for 3D HeadPoseEstimator
    pts[1] = NormalizedLandmark(x=0.50, y=0.50, z=-0.05)
    pts[152] = NormalizedLandmark(x=0.50, y=0.68, z=-0.02)
    pts[199] = NormalizedLandmark(x=0.50, y=0.66, z=-0.02)
    pts[61] = NormalizedLandmark(x=0.44, y=0.59, z=-0.02)
    pts[291] = NormalizedLandmark(x=0.56, y=0.59, z=-0.02)

    h = 0.016 * ear_scale
    # Left eye [33, 160, 158, 133, 153, 144]
    pts[33] = NormalizedLandmark(x=0.38, y=0.44, z=0.0)
    pts[133] = NormalizedLandmark(x=0.46, y=0.44, z=0.0)
    pts[160] = NormalizedLandmark(x=0.405, y=0.44 - h, z=0.0)
    pts[158] = NormalizedLandmark(x=0.435, y=0.44 - h, z=0.0)
    pts[144] = NormalizedLandmark(x=0.405, y=0.44 + h, z=0.0)
    pts[153] = NormalizedLandmark(x=0.435, y=0.44 + h, z=0.0)

    # Right eye [362, 385, 387, 263, 373, 380]
    pts[362] = NormalizedLandmark(x=0.54, y=0.44, z=0.0)
    pts[263] = NormalizedLandmark(x=0.62, y=0.44, z=0.0)
    pts[385] = NormalizedLandmark(x=0.565, y=0.44 - h, z=0.0)
    pts[387] = NormalizedLandmark(x=0.595, y=0.44 - h, z=0.0)
    pts[380] = NormalizedLandmark(x=0.565, y=0.44 + h, z=0.0)
    pts[373] = NormalizedLandmark(x=0.595, y=0.44 + h, z=0.0)

    # Mouth for MAR
    pts[13] = NormalizedLandmark(x=0.50, y=0.585, z=-0.02)
    pts[14] = NormalizedLandmark(x=0.50, y=0.595, z=-0.02)
    pts[81] = NormalizedLandmark(x=0.48, y=0.586, z=-0.02)
    pts[178] = NormalizedLandmark(x=0.48, y=0.594, z=-0.02)
    pts[311] = NormalizedLandmark(x=0.52, y=0.586, z=-0.02)
    pts[402] = NormalizedLandmark(x=0.52, y=0.594, z=-0.02)
    return tuple(pts)


class FakeLocalCamera(BaseCamera):
    """Deterministic in-memory BaseCamera producing crisp well-lit BGR frames."""

    def __init__(self, flip_horizontal: bool = True) -> None:
        self._opened = False
        self.open_count = 0
        self.close_count = 0
        self._frame_idx = 0
        self._flip = flip_horizontal
        self._ts_ns = 10_000_000_000  # 10,000 ms monotonic base
        # Crisp patterned BGR frame (optimal luminance ~128 & high Laplacian sharpness)
        rng = np.random.default_rng(42)
        self._frame = rng.integers(80, 180, size=(240, 320, 3), dtype=np.uint8)

    def open(self) -> bool:
        self._opened = True
        self.open_count += 1
        return True

    def read(self) -> Tuple[bool, Optional[FramePacket]]:
        if not self._opened:
            return False, None
        self._frame_idx += 1
        self._ts_ns += 100_000_000  # +100ms per synthetic frame
        return True, FramePacket(
            frame=self._frame,
            timestamp_ns=self._ts_ns,
            frame_index=self._frame_idx,
            source_id="mock_local_cam_0",
        )

    def close(self) -> None:
        if self._opened:
            self._opened = False
            self.close_count += 1

    def is_opened(self) -> bool:
        return self._opened

    def get_actual_fps(self) -> float:
        return 20.0

    def get_actual_resolution(self) -> Tuple[int, int]:
        return (320, 240)

    @property
    def is_flip_horizontal(self) -> bool:
        return self._flip


class ScriptedFaceLandmarker:
    """Delivers open eyes first, then sustained eye closure to force CRITICAL transition."""

    def __init__(self, close_after_frames: int = 8) -> None:
        self._count = 0
        self._close_after = close_after_frames
        self._open_lms = _build_synthetic_landmarks(ear_scale=1.0)
        self._closed_lms = _build_synthetic_landmarks(ear_scale=0.10)

    def detect(self, frame_or_packet, timestamp_ms: int = 0) -> FaceLandmarkResult:
        self._count += 1
        lms = self._open_lms if self._count <= self._close_after else self._closed_lms
        return FaceLandmarkResult(
            has_face=True,
            landmarks=lms,
            presence_confidence=0.98,
            timestamp_ms=int(timestamp_ms),
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=4.5,
            reasons=(),
        )

    def close(self) -> None:
        pass


def test_mandatory_test_b_standalone_python_local_monitoring_e2e(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulate standalone offline deployment: LiveKit unconfigured, local API enabled
    monkeypatch.setattr(settings, "LOCAL_CAMERA_API_ENABLED", True)
    monkeypatch.setattr(settings, "LIVEKIT_URL", "")
    monkeypatch.setattr(settings, "LIVEKIT_API_KEY", "")
    monkeypatch.setattr(settings, "LIVEKIT_API_SECRET", "")

    db = LocalDB(db_path=tmp_path / "driveguard_local.db")
    store = ProfileStore(storage_dir=tmp_path / "profiles")

    # Seed a valid calibrated PYTHON profile in local JSON ProfileStore
    seeded_dto = DriverProfileDTO(
        driver_id="driver_standalone_b",
        engine_type="PYTHON",
        is_calibrated=True,
        calibration_mode="FOUR_ZONE",
        calibrated_at="2026-11-24T12:00:00Z",
        center_ear_median=0.33,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        neutral_rot_mat=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        calibration_is_mirrored=True,
        zone_baselines={
            "CENTER": ZoneBaselineDTO(
                zone="CENTER",
                ear_median=0.33,
                ear_mad_scaled=0.015,
                dynamic_closure_threshold=0.22,
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=40,
            ),
        },
    )
    store.save_profile(seeded_dto)

    fake_cam = FakeLocalCamera(flip_horizontal=True)

    def _cam_factory(**kwargs):
        return fake_cam

    def _svc_factory(cfg: LocalMonitoringConfig, prof_store: ProfileStore) -> LocalMonitoringService:
        return LocalMonitoringService(
            config=cfg,
            store=prof_store,
            landmarker=ScriptedFaceLandmarker(close_after_frames=6),
            enable_landmarker=False,
            enable_shadow_worker=False,
            enable_audio=False,
            enable_event_logging=False,
            auto_load_profile=True,
            persist_calibration_to_store=True,
        )

    controller = LocalCameraController(
        db=db,
        store=store,
        camera_factory=_cam_factory,
        service_factory=_svc_factory,
    )

    app.dependency_overrides[get_local_camera_controller] = lambda: controller
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 51000))

    try:
        # 1. Start local monitoring via loopback REST (zero Bearer token required!)
        start_res = client.post(
            "/api/v1/local/start",
            json={
                "driver_id": "driver_standalone_b",
                "source": 0,
                "fps": 30,
                "enable_audio": False,
            },
        )
        assert start_res.status_code == 200, start_res.text
        start_body = start_res.json()
        assert start_body["running"] is True
        assert start_body["camera_opened"] is True
        assert start_body["is_calibrated"] is True
        assert start_body["engine_type"] == "PYTHON"
        session_id = start_body["active_session"]["session_id"]
        assert fake_cam.open_count == 1

        # 2. Stream normalized telemetry & SAFETY_EVENT over /ws/local/telemetry
        received_telemetry = []
        received_safety_events = []

        with client.websocket_connect(
            "/ws/local/telemetry",
            headers={"Host": "127.0.0.1:8000", "Origin": "http://localhost:3000"},
        ) as ws:
            deadline = time.monotonic() + 8.0
            while time.monotonic() < deadline and (
                len(received_telemetry) < 10 or len(received_safety_events) == 0
            ):
                msg = ws.receive_json()
                if msg.get("type") == "TELEMETRY":
                    received_telemetry.append(msg)
                elif msg.get("type") == "SAFETY_EVENT":
                    received_safety_events.append(msg)

        assert len(received_telemetry) >= 5, "Expected live telemetry over /ws/local/telemetry"
        sample_tel = received_telemetry[0]
        assert sample_tel["engine_type"] == "PYTHON"
        assert sample_tel["operational_state"] == "MONITORING_ACTIVE"
        assert sample_tel["is_calibrated"] is True
        assert sample_tel["cascade"]["cnn_execution_mode"] == "SHADOW_DISABLED"
        assert sample_tel["cascade"]["onnx_inference_executed"] is False

        assert len(received_safety_events) >= 1, "Expected CRITICAL/WARNING SAFETY_EVENT from sustained closure"
        assert any(ev["new_state"] in ("WARNING", "CRITICAL") for ev in received_safety_events)

        # 3. Verify on-demand MJPEG preview stream produces valid JPEG frame bytes
        with client.stream("GET", "/api/v1/local/preview.mjpg") as stream_res:
            assert stream_res.status_code == 200
            assert "multipart/x-mixed-replace" in stream_res.headers["content-type"]
            buf = b""
            for chunk in stream_res.iter_bytes():
                buf += chunk
                if b"--frame\r\n" in buf and b"\xff\xd8" in buf and b"\xff\xd9" in buf:
                    break
            assert b"--frame\r\n" in buf
            assert b"\xff\xd8" in buf  # JPEG SOI

        # 4. Stop local monitoring -> camera MUST be released synchronously BEFORE HTTP 200 returns
        stop_res = client.post("/api/v1/local/stop")
        assert stop_res.status_code == 200
        stop_body = stop_res.json()
        assert stop_body["stopped"] is True
        assert stop_body["camera_released"] is True
        assert fake_cam.close_count == 1
        assert fake_cam.is_opened() is False

        # 5. Verify SQLite sessions & alerts endpoints reflect persisted records
        sessions_res = client.get("/api/v1/local/sessions", params={"driver_id": "driver_standalone_b"})
        assert sessions_res.status_code == 200
        sessions = sessions_res.json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == session_id
        assert sessions[0]["status"] == "COMPLETED"
        assert sessions[0]["total_frames"] >= 5

        alerts_res = client.get(f"/api/v1/local/sessions/{session_id}/alerts")
        assert alerts_res.status_code == 200
        alerts = alerts_res.json()["alerts"]
        assert len(alerts) >= 1
        assert all(a["engine_type"] == "PYTHON" for a in alerts)

        # 6. Verify /api/v1/local/profile returns hydrated PYTHON profile
        prof_res = client.get("/api/v1/local/profile", params={"driver_id": "driver_standalone_b"})
        assert prof_res.status_code == 200
        assert prof_res.json()["found"] is True
        assert prof_res.json()["profile"]["engine_type"] == "PYTHON"
    finally:
        controller.stop()
        db.close()
        app.dependency_overrides.pop(get_local_camera_controller, None)
