"""
Unit tests for DriveGuard Recording Analyzer and Anomaly Detection Rules.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import tempfile
import unittest

from scripts.analyze_recording import RecordingAnalyzer


class TestRecordingAnalyzer(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.session_path = Path(self.temp_dir.name) / "test_session"
        self.session_path.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.session_path / "telemetry.csv"

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_csv(self, rows: list) -> None:
        if not rows:
            return
        fieldnames = list(rows[0].keys())
        with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_clean_session_yields_zero_anomalies(self) -> None:
        rows = []
        for i in range(20):
            rows.append({
                "frame_id": i + 1,
                "timestamp_ms": 1000.0 + i * 33.3,
                "face_detected": True,
                "face_status": "TRACKING",
                "presence_confidence": 0.98,
                "pose_zone": "CENTER",
                "pitch_deg": -1.0,
                "yaw_deg": 18.0,
                "roll_deg": 0.0,
                "rel_pitch_deg": 0.0,
                "rel_yaw_deg": 0.0,
                "rel_roll_deg": 0.0,
                "ear_left": 0.28,
                "ear_right": 0.28,
                "ear_compensated": 0.28,
                "ear_closure_threshold": 0.20,
                "is_eye_closed": False,
                "mar": 0.08,
                "is_yawning": False,
                "drowsiness_state": "NORMAL",
                "luminance": 100.0,
                "blur_variance": 400.0,
            })
        self._write_csv(rows)

        analyzer = RecordingAnalyzer(self.session_path)
        anomalies = analyzer.analyze()
        self.assertEqual(len(anomalies), 0)
        report = analyzer.generate_report()
        self.assertIn("Zero anomalies detected", report)

    def test_zone_chattering_detected(self) -> None:
        # Create a 1-frame flicker to LEFT_MIRROR within CENTER
        rows = []
        for i in range(10):
            zone = "LEFT_MIRROR" if i == 5 else "CENTER"
            rows.append({
                "frame_id": i + 1,
                "timestamp_ms": 1000.0 + i * 33.3,
                "face_detected": True,
                "face_status": "TRACKING",
                "presence_confidence": 0.98,
                "pose_zone": zone,
                "pitch_deg": -1.0,
                "yaw_deg": 18.0,
                "roll_deg": 0.0,
                "rel_pitch_deg": 0.0,
                "rel_yaw_deg": 14.1 if i == 5 else 0.0,
                "rel_roll_deg": 0.0,
                "ear_left": 0.28,
                "ear_right": 0.28,
                "ear_compensated": 0.28,
                "ear_closure_threshold": 0.20,
                "is_eye_closed": False,
                "mar": 0.08,
                "is_yawning": False,
                "drowsiness_state": "NORMAL",
                "luminance": 100.0,
                "blur_variance": 400.0,
            })
        self._write_csv(rows)

        analyzer = RecordingAnalyzer(self.session_path)
        anomalies = analyzer.analyze()
        chattering = [a for a in anomalies if a.category == "ZONE_CHATTERING"]
        self.assertEqual(len(chattering), 1)
        self.assertEqual(chattering[0].frame_id, 6)

    def test_optical_foreshortening_false_closure_detected(self) -> None:
        rows = [
            {
                "frame_id": 1,
                "timestamp_ms": 1000.0,
                "face_detected": True,
                "face_status": "TRACKING",
                "presence_confidence": 0.95,
                "pose_zone": "LEFT_MIRROR",
                "pitch_deg": -1.0,
                "yaw_deg": 38.0,
                "roll_deg": 0.0,
                "rel_pitch_deg": 0.0,
                "rel_yaw_deg": 20.0,
                "rel_roll_deg": 0.0,
                "ear_left": 0.16,  # Far eye collapsed
                "ear_right": 0.27,  # Near eye open > threshold (0.20)
                "ear_compensated": 0.18,
                "ear_closure_threshold": 0.20,
                "is_eye_closed": True,  # Flagged closed falsely
                "mar": 0.08,
                "is_yawning": False,
                "drowsiness_state": "NORMAL",
                "luminance": 100.0,
                "blur_variance": 400.0,
            }
        ]
        self._write_csv(rows)

        analyzer = RecordingAnalyzer(self.session_path)
        anomalies = analyzer.analyze()
        foreshortening = [a for a in anomalies if a.category == "OPTICAL_FORESHORTENING_CLOSURE"]
        self.assertEqual(len(foreshortening), 1)
        self.assertEqual(foreshortening[0].severity, "HIGH")


if __name__ == "__main__":
    unittest.main()
