"""
Integration Tests for DriveGuard FastAPI REST APIs and Binary WebSocket Endpoint.
Updated with cryptographic backend authentication tokens.
"""

import time
import unittest
import cv2
import jwt
import numpy as np
from fastapi.testclient import TestClient

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO


def make_test_token(driver_id: str) -> str:
    now = int(time.time())
    payload = {
        "sub": driver_id,
        "role": "USER",
        "iss": settings.AUTH_TOKEN_ISSUER,
        "aud": settings.AUTH_TOKEN_AUDIENCE,
        "iat": now,
        "exp": now + 300,
        "scope": ["*"],
    }
    return jwt.encode(payload, settings.AUTH_SECRET, algorithm="HS256")


class TestDriveGuardAPI(unittest.TestCase):
    """Test suite for REST endpoints and WebSocket telemetry streaming."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_root_and_health_check(self) -> None:
        """Verify root information and system health status."""
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)
        self.assertEqual(r_root.json()["status"], "RUNNING")

        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)
        data = r_health.json()
        self.assertEqual(data["status"], "HEALTHY")
        self.assertIn("weights", data)
        self.assertTrue(data["weights"]["mobilenetv3_eye_onnx"])
        self.assertEqual(data["liveness"], "/health/liveness")
        self.assertEqual(data["readiness"], "/health/readiness")

    def test_liveness_probe(self) -> None:
        """Verify liveness probe returns HTTP 200 with status ALIVE."""
        resp = self.client.get("/health/liveness")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data["status"], "ALIVE")
        self.assertIn("uptime_sec", data)
        self.assertIn("service", data)

    def test_readiness_probe(self) -> None:
        """Verify readiness probe returns HTTP 200/503 with component status."""
        resp = self.client.get("/health/readiness")
        self.assertIn(resp.status_code, [200, 503])
        data = resp.json()
        self.assertIn("status", data)
        self.assertIn("ready", data)
        self.assertIn("components", data)
        self.assertIn("face_landmarker_task", data["components"])
        self.assertIn("eye_verifier", data["components"])
        self.assertIn("livekit", data["components"])

    def test_session_lifecycle_rest_api(self) -> None:
        """Create, query, and terminate a driver monitoring session."""
        driver_id = "driver_carol"
        token = make_test_token(driver_id)
        headers = {"Authorization": f"Bearer {token}"}

        sess_id = "test_rest_sess_01"
        payload = {
            "session_id": sess_id,
            "driver_id": driver_id,
            "experiment_mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        }

        # 1. Create Session
        r_create = self.client.post("/api/v1/sessions", json=payload, headers=headers)
        self.assertEqual(r_create.status_code, 200)
        self.assertEqual(r_create.json()["session_id"], sess_id)

        # 2. List Sessions
        r_list = self.client.get("/api/v1/sessions", headers=headers)
        self.assertEqual(r_list.status_code, 200)
        ids = [s["session_id"] for s in r_list.json()]
        self.assertIn(sess_id, ids)

        # 3. Get Session Details
        r_get = self.client.get(f"/api/v1/sessions/{sess_id}", headers=headers)
        self.assertEqual(r_get.status_code, 200)
        self.assertEqual(r_get.json()["driver_id"], driver_id)

        # 4. Terminate Session
        r_del = self.client.delete(f"/api/v1/sessions/{sess_id}", headers=headers)
        self.assertEqual(r_del.status_code, 200)

        # 5. Verify 404 after termination
        r_after = self.client.get(f"/api/v1/sessions/{sess_id}", headers=headers)
        self.assertEqual(r_after.status_code, 404)

    def test_driver_profile_persistence_rest_api(self) -> None:
        """Save, retrieve, list, and delete driver baseline profiles."""
        driver_id = "test_driver_dave"
        token = make_test_token(driver_id)
        headers = {"Authorization": f"Bearer {token}"}

        profile_dto = DriverProfileDTO(
            driver_id=driver_id,
            calibrated_at="2026-11-04T12:00:00Z",
            is_calibrated=True,
            calibration_mode="4ZONE",
            center_ear_median=0.252,
            blink_duration_median_ms=205.0,
            blink_duration_mad_ms=22.0,
            blink_rcr_median=2.15,
            blink_rcr_mad=0.20,
            neutral_pitch_deg=-1.5,
            neutral_yaw_deg=0.5,
            calibration_reliability=0.96,
            zone_baselines={
                "CENTER": ZoneBaselineDTO(
                    zone="CENTER",
                    ear_median=0.252,
                    ear_mad_scaled=0.016,
                    dynamic_closure_threshold=0.191,
                    sample_count=80,
                )
            },
        )

        # 1. Save Profile
        r_save = self.client.post("/api/v1/profiles", json=profile_dto.model_dump(), headers=headers)
        self.assertEqual(r_save.status_code, 200)
        self.assertEqual(r_save.json()["driver_id"], driver_id)

        # 2. Get Profile
        r_get = self.client.get(f"/api/v1/profiles/{driver_id}", headers=headers)
        self.assertEqual(r_get.status_code, 200)
        self.assertAlmostEqual(r_get.json()["center_ear_median"], 0.252, places=3)

        # 3. List Profiles
        r_list = self.client.get("/api/v1/profiles", headers=headers)
        self.assertEqual(r_list.status_code, 200)
        self.assertIn(driver_id, r_list.json())

        # 4. Delete Profile
        r_del = self.client.delete(f"/api/v1/profiles/{driver_id}", headers=headers)
        self.assertEqual(r_del.status_code, 200)

        # 5. Verify 404 after deletion
        r_after = self.client.get(f"/api/v1/profiles/{driver_id}", headers=headers)
        self.assertEqual(r_after.status_code, 404)

    def test_websocket_telemetry_streaming(self) -> None:
        """Stream binary image frames over WebSocket and receive real-time FrameTelemetryDTO."""
        driver_id = "driver_ws"
        token = make_test_token(driver_id)
        ws_sess_id = "test_ws_streaming_sess"

        with self.client.websocket_connect(
            f"/ws/v1/session/{ws_sess_id}?driver_id={driver_id}&token={token}"
        ) as ws:
            # 0. Initial Session Ready Handshake
            ready_resp = ws.receive_json()
            self.assertEqual(ready_resp["type"], "SESSION_READY")
            self.assertEqual(ready_resp["session_id"], ws_sess_id)

            # 1. Test Control Command (PING)
            ws.send_text('{"action": "PING"}')
            pong_resp = ws.receive_json()
            self.assertEqual(pong_resp["type"], "PONG")
            self.assertEqual(pong_resp["session_id"], ws_sess_id)

            # 2. Test Binary JPEG Frame Ingestion
            synthetic_frame = np.full((240, 320, 3), 125, dtype=np.uint8)
            ok, jpg_bytes = cv2.imencode(".jpg", synthetic_frame)
            self.assertTrue(ok)

            ws.send_bytes(jpg_bytes.tobytes())
            telemetry_resp = ws.receive_json()

            self.assertEqual(telemetry_resp["type"], "TELEMETRY")
            self.assertIn("telemetry", telemetry_resp)
            telem = telemetry_resp["telemetry"]
            self.assertEqual(telem["session_id"], ws_sess_id)
            self.assertIn("state", telem)
            self.assertIn("risk_score", telem)
            self.assertIn("eye_metrics", telem)
            self.assertIn("cascade", telem)
            self.assertIn("comparison", telem)


if __name__ == "__main__":
    unittest.main()
