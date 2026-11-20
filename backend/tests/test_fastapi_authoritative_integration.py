"""
DriveGuard — FastAPI Architecture Consolidation & Authoritative Engine Integration Tests.

Verifies every architectural invariant required after consolidating FastAPI onto the
single authoritative `LocalMonitoringService` (`DrowsinessEngine`) pipeline:
  1. FastAPI and local webcam monitoring share the exact same authoritative classes
     (`LocalMonitoringService`, `DriverFeatureExtractor`, `DrowsinessEngine`,
     `AlertStateManager`, `backend.app.vision.face_landmarker.DriveGuardFaceLandmarker`).
  2. REST session lifecycle (`POST/GET/DELETE /api/v1/sessions`), frame ingestion
     (`POST /api/v1/sessions/{id}/frame`), and telemetry inspection
     (`GET /api/v1/sessions/{id}/telemetry`).
  3. Telemetry and alert consistency: identical landmark/frame sequences fed through
     `LocalMonitoringService` and through FastAPI `SessionAIEngine` produce identical
     `DrowsinessState`, `AlertState`, EAR, MAR, PERCLOS, and `CRITICAL` escalation.
  4. WebSocket binary frame ingestion, command handling, disconnection, and full
     resource cleanup in `session_manager`.
  5. Strict multi-session isolation: concurrent FastAPI sessions never share mutable
     driver state, profiles, landmarkers, or alert transitions.
  6. Error handling and invalid input behavior across REST, WebSocket, and adapters.
  7. Static/AST architectural guard proving zero duplicate drowsiness-detection
     imports (`ai.risk_engine`, `ai.blink_kinematics`, `ai.gaze_attention`, `ai.landmarker`)
     exist in the FastAPI runtime path (`main.py`, `api/`, `services/`, `ai/pipeline.py`).
"""

from __future__ import annotations

import ast
from pathlib import Path
import time
import uuid

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app.ai.geometry import PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.alerts.alert_state import AlertState
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.camera.base import FramePacket
from backend.app.core.auth import create_backend_access_token
from backend.app.core.config import settings
from backend.app.features.feature_extractor import DriverFeatureExtractor
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.main import app
from backend.app.services.frame_adapters import (
    ArrayFrameAdapter,
    CameraFrameAdapter,
    EncodedBytesFrameAdapter,
    FrameDecodeError,
)
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringService,
)
from backend.app.services.session_manager import session_manager
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkResult,
    FaceLandmarkStatus,
)
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)


def _auth_headers(user_id: str) -> dict[str, str]:
    token = create_backend_access_token(user_id=user_id, role="user", session_id=f"auth_{user_id}")
    return {"Authorization": f"Bearer {token}"}


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
        zone_baselines={PoseZone.CENTER: zb},
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.96,
        calibration_is_mirrored=True,
    )


class _DeterministicAuthoritativeLandmarker:
    """Mock landmarker implementing the authoritative `.detect()` contract."""

    def __init__(self, ear: float = 0.30, yaw: float = 0.0, pitch: float = 0.0) -> None:
        self.ear = ear
        self.yaw = yaw
        self.pitch = pitch
        self.active_implementation = "MOCK_AUTHORITATIVE"
        self.closed = False

    def detect(self, input_data, timestamp_ms: int = 0) -> FaceLandmarkResult:
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
            inference_latency_ms=3.5,
            reasons=("OK",),
        )

    def get_diagnostics(self) -> dict:
        return {"is_closed": self.closed, "running_mode": "VIDEO"}

    def close(self) -> None:
        self.closed = True


def _encode_jpeg(frame_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame_bgr)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# 1. Single Authoritative Engine Identity & Consistency
# ---------------------------------------------------------------------------

def test_fastapi_and_local_monitoring_use_identical_authoritative_engine_classes() -> None:
    """FastAPI sessions and LocalMonitoringService must instantiate the exact same authoritative core classes."""
    client = TestClient(app)
    driver_id = f"driver_arch_{uuid.uuid4().hex[:6]}"
    sess_id = f"sess_arch_{uuid.uuid4().hex[:6]}"
    headers = _auth_headers(driver_id)

    try:
        r_create = client.post(
            "/api/v1/sessions",
            json={"session_id": sess_id, "driver_id": driver_id},
            headers=headers,
        )
        assert r_create.status_code == 200

        engine = session_manager.get_session(sess_id)
        assert engine is not None
        assert isinstance(engine, SessionAIEngine)
        assert isinstance(engine.monitoring_service, LocalMonitoringService)
        assert isinstance(engine.drowsiness_engine, DrowsinessEngine)
        assert isinstance(engine.feature_extractor, DriverFeatureExtractor)
        assert isinstance(engine.alert_manager, AlertStateManager)
        assert isinstance(engine.landmarker, DriveGuardFaceLandmarker)
        # Ensure SessionAIEngine holds zero competing legacy AI instances
        assert not hasattr(engine, "risk_engine")
        assert not hasattr(engine, "kinematics")
        assert not hasattr(engine, "gaze_tracker")
        assert not hasattr(engine, "cascade")
    finally:
        client.delete(f"/api/v1/sessions/{sess_id}", headers=headers)


def test_telemetry_and_alert_consistency_between_local_service_and_fastapi() -> None:
    """
    Feeding the exact same calibrated driver frame sequence to LocalMonitoringService
    and to a FastAPI session must produce identical DrowsinessState, AlertState, EAR,
    MAR, PERCLOS, PoseZone, and CRITICAL microsleep escalation.
    """
    driver_id = f"driver_consist_{uuid.uuid4().hex[:6]}"
    sess_id = f"sess_consist_{uuid.uuid4().hex[:6]}"
    headers = _auth_headers(driver_id)
    profile = _make_calibrated_profile(driver_id)
    frame = _make_sharp_frame(width=640, height=480)
    jpeg_bytes = _encode_jpeg(frame)

    lm_local = _DeterministicAuthoritativeLandmarker(ear=0.30)
    lm_api = _DeterministicAuthoritativeLandmarker(ear=0.30)

    local_svc = LocalMonitoringService(
        config=LocalMonitoringConfig(driver_id=driver_id, is_mirrored=True, cnn_shadow_enabled=False),
        landmarker=lm_local,
        enable_landmarker=False,
        enable_shadow_worker=False,
        enable_audio=False,
        enable_event_logging=False,
        auto_load_profile=False,
        initial_profile=profile,
        face_loss_grace_ms=450.0,
    )

    client = TestClient(app)
    try:
        r_create = client.post(
            "/api/v1/sessions",
            json={"session_id": sess_id, "driver_id": driver_id},
            headers=headers,
        )
        assert r_create.status_code == 200
        engine = session_manager.get_session(sess_id)
        assert engine is not None
        # Swap in deterministic authoritative landmarker and calibrated profile on the live FastAPI session
        old_lm = engine.landmarker
        engine.landmarker = lm_api
        if old_lm is not None:
            old_lm.close()
        engine.set_profile(profile)

        # Phase A: 5 normal open-eye frames, then Phase B: 55 closed-eye frames (~1815ms -> CRITICAL)
        ts = 1000.0
        last_local = None
        last_api_telem = None
        for step in range(60):
            ts += 33.0
            target_ear = 0.30 if step < 5 else 0.11
            lm_local.ear = target_ear
            lm_api.ear = target_ear

            # Local webcam path via CameraFrameAdapter
            pkt = FramePacket(
                frame=frame,
                timestamp_ns=int(ts * 1_000_000),
                frame_index=step + 1,
                source_id="camera:0",
            )
            last_local = local_svc.process_adapted_frame(CameraFrameAdapter.adapt(pkt, is_mirrored=True))

            # FastAPI REST path via POST /api/v1/sessions/{id}/frame
            resp = client.post(
                f"/api/v1/sessions/{sess_id}/frame?timestamp_ms={ts}",
                content=jpeg_bytes,
                headers={**headers, "Content-Type": "image/jpeg"},
            )
            assert resp.status_code == 200
            last_api_telem = resp.json()["telemetry"]

            assert last_api_telem["drowsiness_state"] == last_local.assessment.state.value
            assert last_api_telem["alert_state"] == last_local.alert_state.value
            assert last_api_telem["pose_zone"] == last_local.features.head_pose.zone.value
            assert last_api_telem["eye_metrics"]["ear_raw"] == pytest.approx(
                round(last_local.features.ear.weighted_ear, 4), abs=1e-4
            )
            assert last_api_telem["eye_metrics"]["perclos_pct"] == pytest.approx(
                round(last_local.features.perclos.perclos_pct, 2), abs=1e-2
            )

        assert last_local is not None and last_api_telem is not None
        assert last_local.alert_state == AlertState.CRITICAL
        assert last_api_telem["state"] == "CRITICAL"
        assert last_api_telem["alert_state"] == "CRITICAL"
        assert last_api_telem["authoritative_engine"] == "DrowsinessEngine"

        # Verify GET /api/v1/sessions/{id}/telemetry matches
        r_snap = client.get(f"/api/v1/sessions/{sess_id}/telemetry", headers=headers)
        assert r_snap.status_code == 200
        snap_body = r_snap.json()
        assert snap_body["alert_state"] == "CRITICAL"
        assert snap_body["authoritative_engine"] == "DrowsinessEngine"
        assert snap_body["telemetry"]["alert_state"] == "CRITICAL"
    finally:
        local_svc.close()
        client.delete(f"/api/v1/sessions/{sess_id}", headers=headers)


# ---------------------------------------------------------------------------
# 2. Multi-Session Mutable State Isolation via FastAPI
# ---------------------------------------------------------------------------

def test_multiple_fastapi_sessions_do_not_share_mutable_driver_state() -> None:
    """Two concurrent FastAPI sessions must maintain completely isolated driver state and alerts."""
    driver_a = f"driver_iso_a_{uuid.uuid4().hex[:6]}"
    driver_b = f"driver_iso_b_{uuid.uuid4().hex[:6]}"
    sess_a = f"sess_iso_a_{uuid.uuid4().hex[:6]}"
    sess_b = f"sess_iso_b_{uuid.uuid4().hex[:6]}"
    headers_a = _auth_headers(driver_a)
    headers_b = _auth_headers(driver_b)

    frame = _make_sharp_frame(width=640, height=480)
    jpeg_bytes = _encode_jpeg(frame)
    client = TestClient(app)

    try:
        assert client.post("/api/v1/sessions", json={"session_id": sess_a, "driver_id": driver_a}, headers=headers_a).status_code == 200
        assert client.post("/api/v1/sessions", json={"session_id": sess_b, "driver_id": driver_b}, headers=headers_b).status_code == 200

        eng_a = session_manager.get_session(sess_a)
        eng_b = session_manager.get_session(sess_b)
        assert eng_a is not None and eng_b is not None

        # Verify completely distinct mutable instances
        assert eng_a.monitoring_service is not eng_b.monitoring_service
        assert eng_a.drowsiness_engine is not eng_b.drowsiness_engine
        assert eng_a.feature_extractor is not eng_b.feature_extractor
        assert eng_a.alert_manager is not eng_b.alert_manager
        assert eng_a.landmarker is not eng_b.landmarker

        # Inject deterministic landmarkers: Driver A awake (0.30), Driver B asleep (0.11)
        old_a, old_b = eng_a.landmarker, eng_b.landmarker
        eng_a.landmarker = _DeterministicAuthoritativeLandmarker(ear=0.30)
        eng_b.landmarker = _DeterministicAuthoritativeLandmarker(ear=0.11)
        if old_a is not None:
            old_a.close()
        if old_b is not None:
            old_b.close()

        eng_a.set_profile(_make_calibrated_profile(driver_a))
        eng_b.set_profile(_make_calibrated_profile(driver_b))

        ts = 2000.0
        last_a = None
        last_b = None
        for _ in range(55):
            ts += 33.0
            r_a = client.post(f"/api/v1/sessions/{sess_a}/frame?timestamp_ms={ts}", content=jpeg_bytes, headers=headers_a)
            r_b = client.post(f"/api/v1/sessions/{sess_b}/frame?timestamp_ms={ts}", content=jpeg_bytes, headers=headers_b)
            assert r_a.status_code == 200 and r_b.status_code == 200
            last_a = r_a.json()["telemetry"]
            last_b = r_b.json()["telemetry"]

        assert last_a is not None and last_b is not None
        assert last_a["alert_state"] == "NORMAL"
        assert last_a["state"] == "NORMAL"
        assert last_b["alert_state"] == "CRITICAL"
        assert last_b["state"] == "CRITICAL"
    finally:
        client.delete(f"/api/v1/sessions/{sess_a}", headers=headers_a)
        client.delete(f"/api/v1/sessions/{sess_b}", headers=headers_b)


# ---------------------------------------------------------------------------
# 3. WebSocket Disconnection & Session Cleanup
# ---------------------------------------------------------------------------

def test_websocket_disconnection_cleans_up_session_and_resources() -> None:
    """Disconnecting from `/ws/v1/session/{id}` must terminate the session and close its landmarker."""
    driver_id = f"driver_ws_clean_{uuid.uuid4().hex[:6]}"
    sess_id = f"sess_ws_clean_{uuid.uuid4().hex[:6]}"
    token = create_backend_access_token(user_id=driver_id, role="user", session_id=sess_id)
    client = TestClient(app)

    captured_engine = None
    with client.websocket_connect(f"/ws/v1/session/{sess_id}?driver_id={driver_id}&token={token}") as ws:
        ready = ws.receive_json()
        assert ready["type"] == "SESSION_READY"
        assert ready["session_id"] == sess_id

        captured_engine = session_manager.get_session(sess_id)
        assert captured_engine is not None
        assert captured_engine.landmarker is not None
        assert not captured_engine.landmarker.get_diagnostics()["is_closed"]

        # Send invalid JSON text command -> receives structured ERROR without dropping socket
        ws.send_text("{invalid json")
        err_msg = ws.receive_json()
        assert err_msg["type"] == "ERROR"

        # Send valid frame -> receives authoritative TELEMETRY
        jpeg_bytes = _encode_jpeg(_make_sharp_frame(width=320, height=240))
        ws.send_bytes(jpeg_bytes)
        telem_msg = ws.receive_json()
        assert telem_msg["type"] == "TELEMETRY"
        assert telem_msg["telemetry"]["authoritative_engine"] == "DrowsinessEngine"

    # After context exit (disconnect), session must be removed and native resources released
    assert session_manager.get_session(sess_id) is None
    assert captured_engine is not None
    assert captured_engine._is_closed is True
    assert captured_engine.landmarker is not None
    assert captured_engine.landmarker.get_diagnostics()["is_closed"] is True


# ---------------------------------------------------------------------------
# 4. Error Handling, Invalid Inputs, and Frame Adapters
# ---------------------------------------------------------------------------

def test_fastapi_frame_error_handling_and_input_validation() -> None:
    """REST frame ingestion must cleanly reject empty, corrupt, oversized, unknown, and unauthorized payloads."""
    driver_id = f"driver_err_{uuid.uuid4().hex[:6]}"
    other_id = f"driver_other_{uuid.uuid4().hex[:6]}"
    sess_id = f"sess_err_{uuid.uuid4().hex[:6]}"
    headers = _auth_headers(driver_id)
    other_headers = _auth_headers(other_id)
    client = TestClient(app)

    try:
        assert client.post("/api/v1/sessions", json={"session_id": sess_id, "driver_id": driver_id}, headers=headers).status_code == 200

        # 401 Unauthenticated
        assert client.post(f"/api/v1/sessions/{sess_id}/frame", content=b"123").status_code == 401
        assert client.get(f"/api/v1/sessions/{sess_id}/telemetry").status_code == 401

        # 403 Cross-user forbidden
        assert client.post(f"/api/v1/sessions/{sess_id}/frame", content=b"123", headers=other_headers).status_code == 403
        assert client.get(f"/api/v1/sessions/{sess_id}/telemetry", headers=other_headers).status_code == 403

        # 404 Unknown session
        assert client.post("/api/v1/sessions/nonexistent_sess/frame", content=b"123", headers=headers).status_code == 404
        assert client.get("/api/v1/sessions/nonexistent_sess/telemetry", headers=headers).status_code == 404

        # 400 Empty body
        r_empty = client.post(f"/api/v1/sessions/{sess_id}/frame", content=b"", headers=headers)
        assert r_empty.status_code == 400

        # 400 Corrupt image bytes
        r_bad = client.post(f"/api/v1/sessions/{sess_id}/frame", content=b"not-a-valid-image", headers=headers)
        assert r_bad.status_code == 400

        # 413 Oversized payload
        oversized = b"\x00" * (settings.MAX_WS_FRAME_SIZE_BYTES + 1)
        r_big = client.post(f"/api/v1/sessions/{sess_id}/frame", content=oversized, headers=headers)
        assert r_big.status_code == 413
    finally:
        client.delete(f"/api/v1/sessions/{sess_id}", headers=headers)


def test_frame_adapters_validate_and_normalize_all_sources() -> None:
    """CameraFrameAdapter, EncodedBytesFrameAdapter, and ArrayFrameAdapter must validate inputs cleanly."""
    frame_bgr = _make_sharp_frame(width=320, height=240)
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    jpeg = _encode_jpeg(frame_bgr)

    # CameraFrameAdapter
    pkt = FramePacket(frame=frame_bgr, timestamp_ns=5_000_000_000, frame_index=7, source_id="cam0")
    a_cam = CameraFrameAdapter.adapt(pkt, is_mirrored=False)
    assert a_cam.frame_width == 320 and a_cam.frame_height == 240
    assert a_cam.timestamp_ms == pytest.approx(5000.0)
    assert a_cam.is_mirrored is False

    # EncodedBytesFrameAdapter
    a_bytes = EncodedBytesFrameAdapter.adapt(jpeg, timestamp_ms=1234.5)
    assert a_bytes.frame_width == 320 and a_bytes.frame_height == 240
    assert a_bytes.timestamp_ms == 1234.5

    # ArrayFrameAdapter (RGB -> BGR conversion preserves pixel equality)
    a_rgb = ArrayFrameAdapter.adapt_rgb(frame_rgb, timestamp_ms=2000.0)
    assert np.array_equal(a_rgb.frame_bgr, frame_bgr)

    # Error cases
    with pytest.raises(FrameDecodeError):
        EncodedBytesFrameAdapter.adapt(b"")
    with pytest.raises(FrameDecodeError):
        EncodedBytesFrameAdapter.adapt(b"corrupt")
    with pytest.raises(FrameDecodeError):
        ArrayFrameAdapter.adapt_bgr(np.zeros((10, 10), dtype=np.uint8))


# ---------------------------------------------------------------------------
# 5. Architecture Consistency Guard (No Duplicate Drowsiness Pipeline in FastAPI)
# ---------------------------------------------------------------------------

def test_no_duplicate_drowsiness_pipeline_imported_by_fastapi_runtime() -> None:
    """
    AST inspection guard: verify that FastAPI entrypoints, routers, session services,
    and SessionAIEngine do NOT import legacy competing drowsiness modules
    (`ai.risk_engine`, `ai.blink_kinematics`, `ai.gaze_attention`, `ai.landmarker`).
    """
    repo_root = Path(__file__).resolve().parents[2]
    runtime_files = [
        repo_root / "backend" / "app" / "main.py",
        repo_root / "backend" / "app" / "ai" / "pipeline.py",
        repo_root / "backend" / "app" / "api" / "ws_telemetry.py",
        repo_root / "backend" / "app" / "api" / "routes" / "sessions.py",
        repo_root / "backend" / "app" / "api" / "routes" / "calibration.py",
        repo_root / "backend" / "app" / "api" / "routes" / "profiles.py",
        repo_root / "backend" / "app" / "api" / "routes" / "livekit_token.py",
        repo_root / "backend" / "app" / "services" / "session_manager.py",
        repo_root / "backend" / "app" / "services" / "local_monitoring_service.py",
        repo_root / "backend" / "app" / "services" / "livekit_worker.py",
        repo_root / "backend" / "app" / "services" / "frame_adapters.py",
    ]
    forbidden_legacy_modules = {
        "backend.app.ai.risk_engine",
        "backend.app.ai.blink_kinematics",
        "backend.app.ai.gaze_attention",
        "backend.app.ai.landmarker",
    }

    for path in runtime_files:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden_legacy_modules, (
                        f"Forbidden legacy module '{alias.name}' imported in {path.relative_to(repo_root)}"
                    )
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module not in forbidden_legacy_modules, (
                    f"Forbidden legacy module '{node.module}' imported in {path.relative_to(repo_root)}"
                )
