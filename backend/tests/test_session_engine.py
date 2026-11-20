"""
Unit Tests for DriveGuard SessionAIEngine Thin Adapter and Pydantic Schemas.
Verifies that SessionAIEngine delegates end-to-end to the authoritative
LocalMonitoringService (DriverFeatureExtractor -> DrowsinessEngine -> AlertStateManager).
"""

import unittest
import cv2
import numpy as np

from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.calibration import CalibrationPhaseDTO
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.schemas.telemetry import (
    AlertStateEnum,
    FrameTelemetryDTO,
    PoseZoneEnum,
)
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)


class DummyMockLandmarker:
    """Deterministic authoritative mock landmarker returning synthetic 478-point FaceLandmarkResult."""

    def __init__(self, ear: float = 0.28, pitch: float = 0.0, yaw: float = 0.0) -> None:
        self.ear = ear
        self.pitch = pitch
        self.yaw = yaw
        self.active_implementation = "MOCK_TASKS"

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
            inference_latency_ms=4.0,
            reasons=("OK",),
        )

    def close(self) -> None:
        pass


class TestSessionAIEngine(unittest.TestCase):
    """Verify state management, byte decoding, calibration, and Pydantic contract compliance."""

    def setUp(self) -> None:
        self.engine = SessionAIEngine(
            session_id="test_sess_001",
            driver_id="driver_alice",
            landmarker=DummyMockLandmarker(ear=0.28, pitch=0.0, yaw=0.0),
        )

    def tearDown(self) -> None:
        self.engine.close()

    def test_engine_initialization_defaults(self) -> None:
        """Engine must initialize with correct identifiers and uncalibrated fallback defaults."""
        self.assertEqual(self.engine.session_id, "test_sess_001")
        self.assertEqual(self.engine.driver_id, "driver_alice")
        self.assertFalse(self.engine.is_calibrating)
        self.assertAlmostEqual(self.engine.profile.center_ear_median, 0.28, places=2)

    def test_process_frame_with_numpy_and_raw_bytes(self) -> None:
        """Engine must process both NumPy frames and raw encoded image bytes into valid DTOs."""
        # 1. NumPy frame (16:9 matching canonical synthetic landmarks)
        synthetic_frame = _make_sharp_frame(width=1280, height=720)
        dto_numpy = self.engine.process_frame(synthetic_frame, timestamp_ms=1000.0)
        self.assertIsInstance(dto_numpy, FrameTelemetryDTO)
        self.assertEqual(dto_numpy.session_id, "test_sess_001")
        self.assertTrue(dto_numpy.face_detected)
        self.assertEqual(dto_numpy.state, AlertStateEnum.NORMAL)
        self.assertEqual(dto_numpy.pose_zone, PoseZoneEnum.CENTER)
        self.assertEqual(dto_numpy.authoritative_engine, "DrowsinessEngine")

        # 2. Raw JPEG bytes (as received from WebSockets / HTTP uploads)
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
        self.engine.profile.center_ear_median = 0.245
        self.engine.profile.blink_duration_median_ms = 195.0
        self.engine.profile.calibration_mode = "4ZONE"

        exported_dto = self.engine.export_profile_dto()
        self.assertIsInstance(exported_dto, DriverProfileDTO)
        self.assertEqual(exported_dto.driver_id, "driver_alice")
        self.assertAlmostEqual(exported_dto.center_ear_median, 0.245, places=3)
        self.assertAlmostEqual(exported_dto.blink_duration_median_ms, 195.0, places=1)

        new_engine = SessionAIEngine(
            session_id="new_sess",
            driver_id="driver_bob",
            landmarker=DummyMockLandmarker(),
        )
        try:
            new_engine.load_profile_dto(exported_dto)
            self.assertAlmostEqual(new_engine.profile.center_ear_median, 0.245, places=3)
            self.assertAlmostEqual(new_engine.profile.blink_duration_median_ms, 195.0, places=1)
            self.assertEqual(new_engine.profile.calibration_mode, "4ZONE")
        finally:
            new_engine.close()

    def test_calibration_session_workflow(self) -> None:
        """Engine must handle calibration initiation, frame ingestion, and profile activation."""
        cal_dto = self.engine.start_calibration(
            four_zone_mode=False, zone_duration_sec=5.0, timestamp_ms=2000.0
        )
        self.assertIsInstance(cal_dto, CalibrationPhaseDTO)
        self.assertTrue(self.engine.is_calibrating)
        self.assertEqual(cal_dto.current_zone, "CENTER")
        self.engine.calibration_session.target_samples_per_zone = 22

        frame = _make_sharp_frame(width=1280, height=720)
        ts = 2000.0
        for _ in range(30):
            ts += 40.0
            self.engine.process_frame(frame, timestamp_ms=ts)
            if not self.engine.is_calibrating:
                break

        status = self.engine.get_calibration_status()
        self.assertIsNotNone(status)
        self.assertTrue(status.is_completed)
        self.assertIsNone(status.rejection_reason)
        self.assertFalse(self.engine.is_calibrating)
        self.assertTrue(self.engine.feature_extractor.is_profile_feasible)

    def test_side_by_side_comparison_mirror_check(self) -> None:
        """Lateral mirror turn should trigger mirror-check suppression without false alert."""
        mirror_landmarker = DummyMockLandmarker(ear=0.20, pitch=0.0, yaw=20.0)
        engine = SessionAIEngine(landmarker=mirror_landmarker)
        try:
            frame = _make_sharp_frame(width=1280, height=720)
            ts = 5000.0
            dto = None
            for _ in range(16):
                dto = engine.process_frame(frame, timestamp_ms=ts)
                ts += 33.3

            self.assertIsNotNone(dto)
            self.assertEqual(dto.pose_zone, PoseZoneEnum.LEFT_MIRROR)
            self.assertTrue(dto.comparison.is_mirror_check_suppressed)
            self.assertEqual(dto.state, AlertStateEnum.NORMAL)
            self.assertTrue(dto.comparison.fixed_false_alarm)
        finally:
            engine.close()

    def test_target_acquisition_spatial_gating_calibration(self) -> None:
        """Calibration progress must ONLY fill when driver is actively looking in target zone."""
        mock_lm = DummyMockLandmarker(ear=0.28, pitch=0.0, yaw=0.0)
        engine = SessionAIEngine(landmarker=mock_lm)
        try:
            cal_dto = engine.start_calibration(
                four_zone_mode=True, zone_duration_sec=5.0, timestamp_ms=3000.0
            )
            self.assertTrue(engine.is_calibrating)
            self.assertEqual(cal_dto.current_zone, "CENTER")
            engine.calibration_session.target_samples_per_zone = 20

            frame = _make_sharp_frame(width=1280, height=720)
            ts = 3000.0

            # Step 1: Center (yaw=0.0). Ingest frames until CENTER completes.
            for _ in range(25):
                ts += 40.0
                engine.process_frame(frame, timestamp_ms=ts)
                if engine.get_calibration_status().current_zone == "LEFT_MIRROR":
                    break

            status = engine.get_calibration_status()
            self.assertIsNotNone(status)
            self.assertEqual(status.current_zone, "LEFT_MIRROR")
            self.assertEqual(status.phase_index, 2)
            self.assertEqual(status.zone_progress_pct, 0.0)

            # Advance past the 2.0-second inter-step grace period
            ts += 2100.0

            # Driver still looking CENTER (yaw=0.0). Progress MUST NOT advance!
            for _ in range(5):
                ts += 40.0
                engine.process_frame(frame, timestamp_ms=ts)

            status_after_unaligned = engine.get_calibration_status()
            self.assertEqual(status_after_unaligned.zone_progress_pct, 0.0)
            self.assertFalse(status_after_unaligned.is_head_in_target_zone)
            self.assertIn("Driver Mirror", status_after_unaligned.stability_feedback)

            # Now driver turns to LEFT_MIRROR (yaw = +20.0)
            mock_lm.yaw = 20.0
            for _ in range(25):
                ts += 40.0
                engine.process_frame(frame, timestamp_ms=ts)
                if engine.get_calibration_status().current_zone == "RIGHT_MIRROR":
                    break

            status_after_aligned = engine.get_calibration_status()
            self.assertEqual(status_after_aligned.current_zone, "RIGHT_MIRROR")
            self.assertEqual(status_after_aligned.phase_index, 3)
        finally:
            engine.close()


if __name__ == "__main__":
    unittest.main()
