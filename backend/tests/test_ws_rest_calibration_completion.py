"""
DriveGuard Phase 2 Tests — WebSocket & REST Calibration Completion Emission.

Verifies that the final frame completing calibration over BOTH:
  - REST `POST /api/v1/sessions/{session_id}/frame`
  - WebSocket `/ws/v1/session/{session_id}`
returns a non-null `calibration` object with `is_completed == True` and a valid `profile`
(fixing the previous bug where `engine.is_calibrating` turned False on the final frame
and caused `"calibration": None` to drop the completion payload).
"""

from __future__ import annotations

import asyncio
import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.auth import create_backend_access_token
from backend.app.main import app
from backend.app.services.session_manager import session_manager
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)


class _CenterCalibStubLandmarker:
    active_implementation = "SYNTHETIC_STUB"

    def detect(self, frame, timestamp_ms: int = 0) -> FaceLandmarkResult:
        lms = _make_synthetic_landmarks(
            ear_target=0.30,
            yaw_deg=0.0,
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


def _encode_sharp_jpeg() -> bytes:
    ok, buf = cv2.imencode(".jpg", _make_sharp_frame())
    assert ok
    return buf.tobytes()


def test_rest_frame_endpoint_emits_completed_calibration_payload():
    client = TestClient(app)
    driver_id = "drv_rest_cal_done"
    session_id = "sess_rest_cal_done"
    token = create_backend_access_token(user_id=driver_id, role="USER")
    headers = {"Authorization": f"Bearer {token}"}
    jpg_bytes = _encode_sharp_jpeg()

    try:
        r_create = client.post(
            "/api/v1/sessions",
            json={"session_id": session_id, "driver_id": driver_id},
            headers=headers,
        )
        assert r_create.status_code == 200

        engine = session_manager.get_session(session_id)
        assert engine is not None
        engine.landmarker = _CenterCalibStubLandmarker()

        # Start fast 1-zone CENTER calibration (0.2s duration)
        engine.start_calibration(
            four_zone_mode=False,
            three_zone_mode=False,
            target_zone="CENTER",
            zone_duration_sec=0.2,
            timestamp_ms=1000.0,
        )

        completed_payload = None
        for i in range(1, 35):
            ts = 1000.0 + i * 25.0
            res = client.post(
                f"/api/v1/sessions/{session_id}/frame?timestamp_ms={ts}",
                content=jpg_bytes,
                headers={**headers, "Content-Type": "application/octet-stream"},
            )
            assert res.status_code == 200
            body = res.json()
            cal = body.get("calibration")
            if cal is not None and cal.get("is_completed") is True:
                completed_payload = cal
                break

        assert completed_payload is not None, "Final REST /frame response dropped completed calibration payload!"
        assert completed_payload["is_completed"] is True
        assert completed_payload["profile"] is not None
        assert completed_payload["profile"]["is_calibrated"] is True
    finally:
        client.delete(f"/api/v1/sessions/{session_id}", headers=headers)


def test_ws_frame_stream_emits_completed_calibration_payload():
    client = TestClient(app)
    driver_id = "drv_ws_cal_done"
    session_id = "sess_ws_cal_done"
    token = create_backend_access_token(user_id=driver_id, role="USER")
    jpg_bytes = _encode_sharp_jpeg()

    try:
        with client.websocket_connect(
            f"/ws/v1/session/{session_id}?driver_id={driver_id}&token={token}"
        ) as ws:
            ready = ws.receive_json()
            assert ready["type"] == "SESSION_READY"

            engine = session_manager.get_session(session_id)
            assert engine is not None
            engine.landmarker = _CenterCalibStubLandmarker()

            # Start fast 1-zone CENTER calibration via engine so wall-clock progression completes quickly
            engine.start_calibration(
                four_zone_mode=False,
                three_zone_mode=False,
                target_zone="CENTER",
                zone_duration_sec=0.05,
            )

            completed_payload = None
            for _ in range(35):
                ws.send_bytes(jpg_bytes)
                msg = ws.receive_json()
                assert msg["type"] == "TELEMETRY"
                cal = msg.get("calibration")
                if cal is not None and cal.get("is_completed") is True:
                    completed_payload = cal
                    break

            assert completed_payload is not None, "Final WS frame dropped completed calibration payload!"
            assert completed_payload["is_completed"] is True
            assert completed_payload["profile"] is not None
            assert completed_payload["profile"]["is_calibrated"] is True
    finally:
        if session_manager.get_session(session_id) is not None:
            asyncio.run(session_manager.terminate_session(session_id))
