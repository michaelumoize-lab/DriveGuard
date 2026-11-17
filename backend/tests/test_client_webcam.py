"""
Unit Tests for DriveGuard Live WebSocket Webcam Client Test & Diagnostic HUD.
=============================================================================
Tests:
  1. Resolution string parsing (e.g. 640x480, 1280x720, fallbacks).
  2. Non-blocking audio alarm synthesizer (queueing, rate-limiting, and mute).
  3. Dynamic synthetic frame generator at multiple resolutions (640x480 and 1280x720).
  4. HUD rendering with full advanced DMS metrics (Optalert AVR, CV_IBI, Gaze Entropy, AU4, Timing Waterfall).
  5. Calibration overlay rendering during 4-zone guided wizard.
  6. Server message dispatcher (TELEMETRY, CALIBRATION_STARTED, PROFILE_SAVED, EXPERIMENT_SWITCHED).
  7. End-to-end WebSocket client telemetry roundtrip via FastAPI TestClient.
"""

from __future__ import annotations

import json
import time
import unittest
from typing import Any, Dict

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.research.client_webcam_test import (
    LiveWebcamHUDClient,
    NonBlockingAudioAlarm,
    parse_resolution,
)


class TestClientWebcamHUD(unittest.TestCase):
    """Test suite for client webcam test script and diagnostic HUD."""

    def test_parse_resolution(self) -> None:
        """Verify resolution strings are parsed correctly into (width, height) tuples."""
        self.assertEqual(parse_resolution("640x480"), (640, 480))
        self.assertEqual(parse_resolution("1280x720"), (1280, 720))
        self.assertEqual(parse_resolution("1920X1080"), (1920, 1080))
        # Fallback for malformed strings
        self.assertEqual(parse_resolution("invalid"), (640, 480))
        self.assertEqual(parse_resolution(""), (640, 480))

    def test_audio_alarm_synthesizer_cooldown(self) -> None:
        """Verify audio alarms can be triggered without blocking and adhere to cooldowns."""
        alarm = NonBlockingAudioAlarm(enabled=True)
        # First trigger should register
        alarm.play_critical()
        first_time = alarm._last_critical_time
        self.assertGreater(first_time, 0.0)

        # Immediate re-trigger should be suppressed by cooldown
        alarm.play_critical()
        self.assertEqual(alarm._last_critical_time, first_time)

        # Warning trigger should work independently
        alarm.play_warning()
        self.assertGreater(alarm._last_warning_time, 0.0)

        # Chime trigger
        alarm.play_chime(1200, 50)

        # Disabling audio silences triggers
        alarm.enabled = False
        old_time = alarm._last_critical_time
        alarm.play_critical()
        self.assertEqual(alarm._last_critical_time, old_time)

    def test_synthetic_frame_generation_dimensions(self) -> None:
        """Verify synthetic driver frames scale to requested resolution cleanly."""
        client_vga = LiveWebcamHUDClient(
            server_ws_url="ws://127.0.0.1:8000/ws/v1/session/test",
            width=640,
            height=480,
            synthetic=True,
            audio_enabled=False,
            headless=True,
        )
        frame_vga = client_vga.generate_synthetic_frame(1)
        self.assertEqual(frame_vga.shape, (480, 640, 3))
        self.assertEqual(frame_vga.dtype, np.uint8)

        client_hd = LiveWebcamHUDClient(
            server_ws_url="ws://127.0.0.1:8000/ws/v1/session/test",
            width=1280,
            height=720,
            synthetic=True,
            audio_enabled=False,
            headless=True,
        )
        frame_hd = client_hd.generate_synthetic_frame(1)
        self.assertEqual(frame_hd.shape, (720, 1280, 3))
        self.assertEqual(frame_hd.dtype, np.uint8)

    def test_hud_rendering_with_advanced_metrics(self) -> None:
        """Verify HUD renders cleanly with all Phase 6 / EXP-1A advanced metrics."""
        client = LiveWebcamHUDClient(
            server_ws_url="ws://127.0.0.1:8000/ws/v1/session/test",
            width=640,
            height=480,
            synthetic=True,
            audio_enabled=False,
            show_diagnostics=True,
            headless=True,
        )

        mock_telemetry: Dict[str, Any] = {
            "session_id": "test_session",
            "driver_id": "driver_01",
            "frame_index": 42,
            "timestamp_ms": 1700000000.0,
            "fps": 24.8,
            "face_detected": True,
            "state": "WARNING",
            "risk_score": 0.58,
            "pose_zone": "CENTER",
            "reason_codes": ["PROLONGED_CLOSURE", "AVR_HIGH_DROWSINESS"],
            "head_pose": {
                "pitch_deg": -4.2,
                "yaw_deg": 3.1,
                "roll_deg": 1.0,
            },
            "eye_metrics": {
                "ear_raw": 0.175,
                "ear_left": 0.170,
                "ear_right": 0.180,
                "ear_normalized": 0.175,
                "ear_closure_threshold": 0.220,
                "is_eye_closed": True,
                "fused_closure_prob": 0.82,
                "perclos_pct": 18.5,
                "blink_duration_ms": 280.0,
                "blink_rcr": 0.85,
                "blink_rate_per_min": 14.0,
                "blink_state": "CLOSED",
                "blink_classification": "PROLONGED",
                "eye_closure_duration_ms": 320.0,
                "blink_avr": 39.5,  # >35.0 Optalert high fatigue threshold
                "blink_cv_ibi": 0.38,
                "gaze_entropy": 1.25,
                "brow_furrow_au4": 0.62,
                "yawn_envelope_confidence": 0.88,
            },
            "cascade": {
                "experiment_mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
                "cnn_triggered": True,
                "geom_open_prob": 0.18,
                "geom_reliability": 0.90,
                "cnn_open_prob": 0.12,
                "cnn_reliability": 0.94,
                "arbitration_winner": "CONSENSUS",
                "disagreement_flag": False,
                "cnn_latency_ms": 8.2,
            },
            "comparison": {
                "fixed_cutoff_ear": 0.22,
                "fixed_alert_state": "WARNING",
                "fixed_false_alarm": False,
                "personalised_zone": "CENTER",
                "personalised_threshold": 0.220,
                "is_mirror_check_suppressed": False,
            },
            "inference_latency_ms": 38.5,
            "timing": {
                "queue_latency_ms": 2.1,
                "infer_duration_ms": 36.4,
                "server_total_ms": 38.5,
                "server_ingest_mono_ms": 1000.0,
                "server_publish_mono_ms": 1038.5,
                "server_wall_epoch_ms": 1700000000.0,
            },
        }

        client.latest_telemetry = mock_telemetry
        client.connected = True
        client.last_roundtrip_ms = 44.0

        canvas = client.generate_synthetic_frame(1)
        hud_output = client.draw_hud(canvas)

        # Output canvas must be same shape, uint8, and non-empty
        self.assertEqual(hud_output.shape, (480, 640, 3))
        self.assertEqual(hud_output.dtype, np.uint8)
        self.assertGreater(int(np.mean(hud_output)), 0)

    def test_hud_rendering_during_calibration(self) -> None:
        """Verify calibration wizard overlay is rendered when calibration is active."""
        client = LiveWebcamHUDClient(
            server_ws_url="ws://127.0.0.1:8000/ws/v1/session/test",
            width=640,
            height=480,
            synthetic=True,
            audio_enabled=False,
            headless=True,
        )

        client.latest_calibration = {
            "phase_index": 2,
            "total_phases": 4,
            "current_zone": "LEFT_MIRROR",
            "instruction": "LOOK AT LEFT MIRROR",
            "zone_progress_pct": 65.0,
            "total_progress_pct": 41.2,
            "stability_feedback": "Holding Steady",
            "is_head_in_target_zone": True,
            "is_completed": False,
        }

        canvas = client.generate_synthetic_frame(1)
        hud_output = client.draw_hud(canvas)

        self.assertEqual(hud_output.shape, (480, 640, 3))
        self.assertEqual(hud_output.dtype, np.uint8)

    def test_handle_server_message_types(self) -> None:
        """Verify dispatcher processes all server event payloads."""
        client = LiveWebcamHUDClient(
            server_ws_url="ws://127.0.0.1:8000/ws/v1/session/test",
            synthetic=True,
            audio_enabled=False,
            headless=True,
        )

        # 1. Telemetry message
        client.handle_server_message(
            {
                "type": "TELEMETRY",
                "telemetry": {"state": "NORMAL", "risk_score": 0.12},
                "calibration": None,
            }
        )
        self.assertEqual(client.total_telemetry_received, 1)
        self.assertEqual(client.latest_telemetry["state"], "NORMAL")

        # 2. Calibration started
        client.handle_server_message(
            {
                "type": "CALIBRATION_STARTED",
                "calibration": {"phase_index": 1, "is_completed": False},
            }
        )
        self.assertIsNotNone(client.latest_calibration)
        self.assertIn("Calibration started", client.notification_msg)

        # 3. Calibration cancelled
        client.handle_server_message({"type": "CALIBRATION_CANCELLED"})
        self.assertIsNone(client.latest_calibration)
        self.assertIn("cancelled", client.notification_msg)

        # 4. Profile saved
        client.handle_server_message(
            {
                "type": "PROFILE_SAVED",
                "profile": {"is_calibrated": True, "calibration_mode": "4_ZONE", "center_ear_median": 0.285},
            }
        )
        self.assertIn("saved to disk", client.notification_msg)

        # 5. Experiment switched
        client.handle_server_message(
            {
                "type": "EXPERIMENT_SWITCHED",
                "active_mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
            }
        )
        self.assertIn("switched", client.notification_msg)

    def test_websocket_telemetry_roundtrip_with_testclient(self) -> None:
        """Verify binary JPEG frames sent over WebSocket return complete telemetry DTO."""
        test_client = TestClient(app)
        session_id = f"test_webcam_{int(time.time())}"
        from backend.app.core.auth import create_backend_access_token
        token = create_backend_access_token("test_user_webcam", role="user", session_id=session_id)

        with test_client.websocket_connect(f"/ws/v1/session/{session_id}?token={token}") as ws:
            # 1. Receive SESSION_READY handshake
            handshake = ws.receive_json()
            self.assertEqual(handshake.get("type"), "SESSION_READY")
            self.assertEqual(handshake.get("session_id"), session_id)

            # 2. Generate and encode 3 synthetic driver frames
            hud_client = LiveWebcamHUDClient(
                server_ws_url="mock",
                width=640,
                height=480,
                synthetic=True,
                audio_enabled=False,
                headless=True,
            )

            for frame_idx in range(1, 4):
                frame = hud_client.generate_synthetic_frame(frame_idx)
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                jpeg_bytes = buffer.tobytes()

                # Send binary bytes
                ws.send_bytes(jpeg_bytes)

                # Receive telemetry JSON
                resp = ws.receive_json()
                self.assertEqual(resp.get("type"), "TELEMETRY")
                telemetry = resp.get("telemetry")
                self.assertIsNotNone(telemetry)
                self.assertIn("state", telemetry)
                self.assertIn("risk_score", telemetry)
                self.assertIn("eye_metrics", telemetry)
                self.assertIn("head_pose", telemetry)
                self.assertIn("cascade", telemetry)
                self.assertIn("timing", telemetry)

                # Update HUD client state and verify HUD renders
                hud_client.handle_server_message(resp)
                rendered = hud_client.draw_hud(frame)
                self.assertEqual(rendered.shape, (480, 640, 3))

            self.assertEqual(hud_client.total_telemetry_received, 3)


if __name__ == "__main__":
    unittest.main()
