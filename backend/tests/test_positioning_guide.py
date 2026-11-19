"""
Unit tests for DriveGuard Camera Positioning and Framing Guide.
"""

from __future__ import annotations

import unittest
import numpy as np

from backend.app.camera.frame_quality import FrameQualityResult, QualityStatus, BlurCondition, LightingCondition
from backend.app.camera.positioning_guide import (
    evaluate_camera_position,
    draw_positioning_guide,
    CameraPositioningResult,
)
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus


class TestCameraPositioningGuide(unittest.TestCase):
    def _create_mock_landmarks(self, top_y: float = 0.30, chin_y: float = 0.60, nose_x: float = 0.50) -> np.ndarray:
        lms = np.zeros((478, 3), dtype=np.float32)
        lms[1] = [nose_x, (top_y + chin_y) / 2.0, 0.0]  # nose tip
        lms[10] = [nose_x, top_y, 0.0]                  # forehead
        lms[152] = [nose_x, chin_y, 0.0]                # chin
        lms[33] = [nose_x - 0.12, (top_y + chin_y) / 2.0, 0.0]  # left eye outer
        lms[263] = [nose_x + 0.12, (top_y + chin_y) / 2.0, 0.0] # right eye outer
        return lms

    def _make_lm_result(self, landmarks: np.ndarray, has_face: bool = True) -> FaceLandmarkResult:
        return FaceLandmarkResult(
            has_face=has_face,
            landmarks=landmarks,
            presence_confidence=0.98,
            timestamp_ms=1000,
            status=FaceLandmarkStatus.TRACKING if has_face else FaceLandmarkStatus.NO_FACE_DETECTED,
            inference_latency_ms=25.0,
            reasons=(),
        )

    def test_missing_face_returns_not_optimal(self) -> None:
        res = evaluate_camera_position(landmark_result=None, head_pose=None, quality=None)
        self.assertFalse(res.is_optimal)
        self.assertFalse(res.has_face)
        self.assertEqual(res.distance_status, "UNKNOWN")

    def test_optimal_placement_identified(self) -> None:
        lms = self._create_mock_landmarks(top_y=0.32, chin_y=0.62, nose_x=0.50)  # height = 0.30 (30%)
        lm_res = self._make_lm_result(lms)
        pose_res = HeadPoseResult(
            pitch_deg=-2.0,
            yaw_deg=16.0,
            roll_deg=1.0,
            zone=PoseZone.CENTER,
            is_looking_away=False,
            is_valid=True,
            rel_pitch_deg=0.0,
            rel_yaw_deg=0.0,
            rel_roll_deg=0.0,
        )
        qual_res = FrameQualityResult(
            status=QualityStatus.EXCELLENT,
            mean_luminance=120.0,
            blur_variance=250.0,
            blur=BlurCondition.SHARP,
            lighting=LightingCondition.OPTIMAL,
            is_viable_for_inference=True,
            reasons=(),
        )

        guide = evaluate_camera_position(lm_res, pose_res, qual_res)
        self.assertTrue(guide.is_optimal)
        self.assertEqual(guide.distance_status, "PERFECT")
        self.assertEqual(guide.centering_status, "CENTERED")
        self.assertEqual(guide.angle_status, "OPTIMAL")
        self.assertIn("optimal", guide.primary_instruction.lower())

    def test_face_too_far_flagged(self) -> None:
        lms = self._create_mock_landmarks(top_y=0.45, chin_y=0.58)  # height = 0.13 (13% < 20%)
        lm_res = self._make_lm_result(lms)
        guide = evaluate_camera_position(lm_res, None, None)
        self.assertFalse(guide.is_optimal)
        self.assertEqual(guide.distance_status, "TOO_FAR")

    def test_face_too_close_flagged(self) -> None:
        lms = self._create_mock_landmarks(top_y=0.15, chin_y=0.75)  # height = 0.60 (60% > 48%)
        lm_res = self._make_lm_result(lms)
        guide = evaluate_camera_position(lm_res, None, None)
        self.assertFalse(guide.is_optimal)
        self.assertEqual(guide.distance_status, "TOO_CLOSE")

    def test_extreme_yaw_angle_flagged(self) -> None:
        lms = self._create_mock_landmarks()
        lm_res = self._make_lm_result(lms)
        pose_res = HeadPoseResult(
            pitch_deg=0.0,
            yaw_deg=36.0,  # > 32°
            roll_deg=0.0,
            zone=PoseZone.CENTER,
            is_looking_away=False,
            is_valid=True,
            rel_pitch_deg=0.0,
            rel_yaw_deg=0.0,
            rel_roll_deg=0.0,
        )
        guide = evaluate_camera_position(lm_res, pose_res, None)
        self.assertFalse(guide.is_optimal)
        self.assertEqual(guide.angle_status, "TOO_OBLIQUE")

    def test_oblique_road_gaze_up_to_31_deg_accepted(self) -> None:
        lms = self._create_mock_landmarks(top_y=0.32, chin_y=0.62)
        lm_res = self._make_lm_result(lms)
        pose_res = HeadPoseResult(
            pitch_deg=-4.0,
            yaw_deg=31.0,  # <= 32° (realistic oblique phone mount)
            roll_deg=2.0,
            zone=PoseZone.CENTER,
            is_looking_away=False,
            is_valid=True,
            rel_pitch_deg=0.0,
            rel_yaw_deg=0.0,
            rel_roll_deg=0.0,
        )
        qual_res = FrameQualityResult(
            status=QualityStatus.EXCELLENT,
            mean_luminance=120.0,
            blur_variance=250.0,
            blur=BlurCondition.SHARP,
            lighting=LightingCondition.OPTIMAL,
            is_viable_for_inference=True,
            reasons=(),
        )
        guide = evaluate_camera_position(lm_res, pose_res, qual_res)
        self.assertTrue(guide.is_optimal)
        self.assertEqual(guide.angle_status, "OPTIMAL")

    def test_tilted_phone_mount_flagged(self) -> None:
        lms = self._create_mock_landmarks()
        lm_res = self._make_lm_result(lms)
        pose_res = HeadPoseResult(
            pitch_deg=0.0,
            yaw_deg=10.0,
            roll_deg=18.0,  # > 12°
            zone=PoseZone.CENTER,
            is_looking_away=False,
            is_valid=True,
            rel_pitch_deg=0.0,
            rel_yaw_deg=0.0,
            rel_roll_deg=0.0,
        )
        guide = evaluate_camera_position(lm_res, pose_res, None)
        self.assertFalse(guide.is_optimal)
        self.assertEqual(guide.angle_status, "TILTED")

    def test_draw_positioning_guide_renders_on_canvas(self) -> None:
        canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
        guide_res = CameraPositioningResult(
            is_optimal=True,
            has_face=True,
            distance_status="PERFECT",
            distance_feedback="Ideal",
            face_height_ratio=0.32,
            angle_status="OPTIMAL",
            angle_feedback="Yaw: +16 deg",
            yaw_deg=16.0,
            pitch_deg=-2.0,
            roll_deg=1.0,
            centering_status="CENTERED",
            centering_feedback="Centered",
            face_center_norm=(0.50, 0.48),
            face_bbox_norm=(0.35, 0.25, 0.65, 0.75),
            optical_status="EXCELLENT",
            optical_feedback="Sharp",
            primary_instruction="Position is optimal! Press 'c' to calibrate.",
        )
        rendered = draw_positioning_guide(canvas, guide_res, is_calibrating=False)
        self.assertEqual(rendered.shape, (720, 1280, 3))
        # Verify pixels were altered
        self.assertGreater(np.sum(rendered), 0)


if __name__ == "__main__":
    unittest.main()
