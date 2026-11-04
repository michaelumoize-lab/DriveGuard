"""
Unit Tests for DriveGuard SessionAIEngine Orchestrator and Pydantic Schemas.
"""

import unittest
from datetime import datetime
import cv2
import numpy as np

from backend.app.ai.geometry import (
    FrameGeometryFeatures,
    PoseZone,
)
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.calibration import CalibrationPhaseDTO
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.schemas.telemetry import (
    AlertStateEnum,
    FrameTelemetryDTO,
    PoseZoneEnum,
)


class DummyMockLandmarker:
    """Deterministic mock landmarker to test engine orchestration without webcam hardware."""

    def __init__(self, ear: float = 0.28, pitch: float = 0.0, yaw: float = 0.0) -> None:
        self.ear = ear
        self.pitch = pitch
        self.yaw = yaw

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float,
        neutral_pitch_deg: float = 0.0,
        neutral_yaw_deg: float = 0.0,
        reference_open_ear: float = 0.28,
        extract_crops: bool = True,
    ):
        geom = FrameGeometryFeatures(
            timestamp_ms=timestamp_ms,
            face_detected=True,
            landmark_confidence=0.95,
            left_ear=self.ear,
            right_ear=self.ear,
            yaw_weighted_ear=self.ear,
            blendshape_blink_left=0.05,
            blendshape_blink_right=0.05,
            blendshape_blink_mean=0.05,
            mar=0.12,
            pitch_deg=self.pitch,
            yaw_deg=self.yaw,
            roll_deg=0.0,
            pose_zone=PoseZone.CENTER if abs(self.yaw) < 15.0 else PoseZone.LEFT_MIRROR,
            geometric_reliability=0.92,
        )
        return geom, None


class TestSessionAIEngine(unittest.TestCase):
    """Verify state management, byte decoding, calibration, and Pydantic contract compliance."""

    def setUp(self) -> None:
        self.engine = SessionAIEngine(
            session_id="test_sess_001",
            driver_id="driver_alice",
            landmarker=DummyMockLandmarker(ear=0.28, pitch=0.0, yaw=0.0),
        )

    def test_engine_initialization_defaults(self) -> None:
        """Engine must initialize with correct identifiers and uncalibrated fallback defaults."""
        self.assertEqual(self.engine.session_id, "test_sess_001")
        self.assertEqual(self.engine.driver_id, "driver_alice")
        self.assertFalse(self.engine.is_calibrating)
        self.assertAlmostEqual(self.engine.profile.center_ear_median, 0.28, places=2)

    def test_process_frame_with_numpy_and_raw_bytes(self) -> None:
        """Engine must process both NumPy BGR frames and raw encoded image bytes into valid DTOs."""
        # 1. NumPy frame
        synthetic_frame = np.full((240, 320, 3), 120, dtype=np.uint8)
        dto_numpy = self.engine.process_frame(synthetic_frame, timestamp_ms=1000.0)
        self.assertIsInstance(dto_numpy, FrameTelemetryDTO)
        self.assertEqual(dto_numpy.session_id, "test_sess_001")
        self.assertTrue(dto_numpy.face_detected)
        self.assertEqual(dto_numpy.state, AlertStateEnum.NORMAL)
        self.assertEqual(dto_numpy.pose_zone, PoseZoneEnum.CENTER)

        # 2. Raw JPEG bytes (as received from WebSockets)
        ok, encoded_bytes = cv2.imencode(".jpg", synthetic_frame)
        self.assertTrue(ok)
        raw_bytes = encoded_bytes.tobytes()

        dto_bytes = self.engine.process_frame(raw_bytes, timestamp_ms=1033.0)
        self.assertIsInstance(dto_bytes, FrameTelemetryDTO)
        self.assertEqual(dto_bytes.frame_index, 2)
        self.assertTrue(dto_bytes.face_detected)

        # 3. Serialization to JSON without errors
        json_str = dto_bytes.model_dump_json()
        self.assertIn('"session_id":"test_sess_001"', json_str)
        self.assertIn('"state":"NORMAL"', json_str)

    def test_profile_export_and_import_cycle(self) -> None:
        """Exporting a profile to DTO and loading it into another engine must preserve baselines."""
        # Customize profile
        self.engine.profile.center_ear_median = 0.245
        self.engine.profile.blink_duration_median_ms = 195.0
        self.engine.profile.calibration_mode = "4ZONE"

        exported_dto = self.engine.export_profile_dto()
        self.assertIsInstance(exported_dto, DriverProfileDTO)
        self.assertEqual(exported_dto.driver_id, "driver_alice")
        self.assertAlmostEqual(exported_dto.center_ear_median, 0.245, places=3)
        self.assertAlmostEqual(exported_dto.blink_duration_median_ms, 195.0, places=1)

        # Load into brand new engine
        new_engine = SessionAIEngine(session_id="new_sess", driver_id="driver_bob")
        new_engine.load_profile_dto(exported_dto)

        self.assertAlmostEqual(new_engine.profile.center_ear_median, 0.245, places=3)
        self.assertAlmostEqual(new_engine.profile.blink_duration_median_ms, 195.0, places=1)
        self.assertEqual(new_engine.profile.calibration_mode, "4ZONE")

    def test_calibration_session_workflow(self) -> None:
        """Engine must handle calibration initiation, frame ingestion, and profile activation."""
        cal_dto = self.engine.start_calibration(four_zone_mode=False, zone_duration_sec=0.2)
        self.assertIsInstance(cal_dto, CalibrationPhaseDTO)
        self.assertTrue(self.engine.is_calibrating)
        self.assertEqual(cal_dto.current_zone, "CENTER")

        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        ts = 2000.0
        # Ingest frames to complete calibration
        for _ in range(12):
            dto = self.engine.process_frame(frame, timestamp_ms=ts)
            ts += 33.3

        # Calibration should now be completed
        status = self.engine.get_calibration_status()
        self.assertIsNotNone(status)
        self.assertTrue(status.is_completed)
        self.assertFalse(self.engine.is_calibrating)

    def test_side_by_side_comparison_mirror_check(self) -> None:
        """Lateral mirror turn should trigger mirror-check suppression without false alert."""
        # Driver looks at left mirror (yaw = -24 deg) with apparent lower EAR (0.20)
        mirror_landmarker = DummyMockLandmarker(ear=0.20, pitch=0.0, yaw=-24.0)
        engine = SessionAIEngine(landmarker=mirror_landmarker)

        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        # Mirror check lasting 500ms (> 350ms threshold of conventional detector)
        ts = 5000.0
        dto = None
        for _ in range(16):
            dto = engine.process_frame(frame, timestamp_ms=ts)
            ts += 33.3

        self.assertIsNotNone(dto)
        self.assertEqual(dto.pose_zone, PoseZoneEnum.LEFT_MIRROR)
        self.assertTrue(dto.comparison.is_mirror_check_suppressed)
        # Conventional fixed rule (EAR < 0.22) trips false alarm after 350ms, DriveGuard stays NORMAL
        self.assertEqual(dto.state, AlertStateEnum.NORMAL)
        self.assertTrue(dto.comparison.fixed_false_alarm)


if __name__ == "__main__":
    unittest.main()
