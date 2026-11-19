"""
Unit tests for DriveGuard Session Recorder Subsystem.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from backend.app.camera.session_recorder import SessionRecorder, CSV_COLUMNS


class TestSessionRecorder(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.base_path = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_recorder_lifecycle_creates_expected_files_and_schema(self) -> None:
        recorder = SessionRecorder(
            base_dir=self.base_path,
            session_id="test_session_01",
            driver_id="driver_test",
            frame_width=320,
            frame_height=240,
            fps=30.0,
            profile_metadata={"active_profile": "driver_alpha"},
        )
        self.assertFalse(recorder.is_recording)
        self.assertTrue(recorder.start())
        self.assertTrue(recorder.is_recording)

        # Record 10 dummy frames
        for i in range(10):
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
            # Draw a simple white square
            frame[50:100, 50:100] = 255
            telemetry = {
                "face_detected": True,
                "face_status": "TRACKING",
                "presence_confidence": 0.98,
                "pose_zone": "CENTER" if i < 5 else "LEFT_MIRROR",
                "pitch_deg": -1.5,
                "yaw_deg": 18.0 if i >= 5 else 2.0,
                "roll_deg": 0.5,
                "rel_pitch_deg": 0.0,
                "rel_yaw_deg": 16.0 if i >= 5 else 0.0,
                "rel_roll_deg": 0.5,
                "ear_left": 0.28,
                "ear_right": 0.29,
                "ear_compensated": 0.285,
                "ear_closure_threshold": 0.21,
                "is_eye_closed": False,
                "is_geometric_valid": True,
                "mar": 0.08,
                "is_yawning": False,
                "total_blinks": 2,
                "total_yawns": 0,
                "perclos_pct": 5.2,
                "drowsiness_state": "NORMAL",
                "drowsiness_severity": 0,
                "primary_reason": "ALL_METRICS_NOMINAL",
                "luminance": 120.5,
                "blur_variance": 450.0,
                "optical_status": "EXCELLENT",
                "inference_latency_ms": 12.4,
            }
            recorder.record_frame(frame, timestamp_ms=1000.0 + i * 33.3, telemetry=telemetry)

        self.assertEqual(recorder.frame_count, 10)
        summary = recorder.stop()
        self.assertFalse(recorder.is_recording)
        self.assertEqual(summary["total_frames"], 10)
        self.assertEqual(summary["status"], "COMPLETED")

        # Verify files exist on disk
        session_dir = self.base_path / "test_session_01"
        self.assertTrue(session_dir.is_dir())
        self.assertTrue((session_dir / "stream.mp4").is_file() or (session_dir / "stream.avi").is_file())
        self.assertTrue((session_dir / "telemetry.csv").is_file())
        self.assertTrue((session_dir / "telemetry.jsonl").is_file())
        self.assertTrue((session_dir / "session_metadata.json").is_file())

        # Verify CSV content
        with open(session_dir / "telemetry.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, CSV_COLUMNS)
            rows = list(reader)
            self.assertEqual(len(rows), 10)
            self.assertEqual(rows[0]["frame_id"], "1")
            self.assertEqual(rows[0]["pose_zone"], "CENTER")
            self.assertEqual(rows[5]["frame_id"], "6")
            self.assertEqual(rows[5]["pose_zone"], "LEFT_MIRROR")

        # Verify JSONL lines count
        with open(session_dir / "telemetry.jsonl", "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()]
            self.assertEqual(len(lines), 10)
            self.assertEqual(lines[9]["frame_id"], 10)


if __name__ == "__main__":
    unittest.main()
