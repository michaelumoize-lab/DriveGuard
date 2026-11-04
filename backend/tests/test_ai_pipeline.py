"""
Automated Verification Tests for DriveGuard Phase 1 Core AI Engine.

Run with:
    python -m unittest discover -s backend/tests -v
"""

from __future__ import annotations

import unittest
import numpy as np

from backend.app.ai.blink_kinematics import TemporalKinematicsTracker
from backend.app.ai.cascade_classifier import ConfidenceGatedEyeCascade, ExperimentMode
from backend.app.ai.geometry import (
    EyeCropQuality,
    FrameGeometryFeatures,
    PoseZone,
    classify_pose_zone,
)
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
)
from backend.app.ai.risk_engine import AlertState, PersonalisedAdaptiveRiskEngine


class TestDriveGuardPhase1Pipeline(unittest.TestCase):
    def test_four_zone_pose_classification(self) -> None:
        """Verify head pitch/yaw correctly map to CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD."""
        self.assertEqual(classify_pose_zone(0.0, 0.0), PoseZone.CENTER)
        self.assertEqual(classify_pose_zone(0.0, -20.0), PoseZone.LEFT_MIRROR)
        self.assertEqual(classify_pose_zone(0.0, +20.0), PoseZone.RIGHT_MIRROR)
        self.assertEqual(classify_pose_zone(-15.0, 0.0), PoseZone.DASHBOARD)
        self.assertEqual(classify_pose_zone(0.0, 55.0), PoseZone.OUT_OF_BOUNDS)

    def test_calibration_rejects_poor_visibility(self) -> None:
        """Calibration must reject sessions where face visibility is below the quality threshold."""
        cal = GuidedCalibrationSession(driver_id="bad_vis_driver", four_zone_mode=False)
        cal.start(0.0)
        profile = None
        for idx in range(410):
            ts = idx * 50.0
            profile = cal.ingest_frame(
                timestamp_ms=ts,
                face_detected=(idx % 4 == 0),  # Only 25% valid frames
                geometric_reliability=0.30,
                ear_value=0.26,
                pitch_deg=0.0,
                yaw_deg=0.0,
                is_eye_currently_closed=False,
            )
        self.assertIsNone(profile)
        self.assertIsNotNone(cal.rejection_reason)
        self.assertIn("REJECTED_POOR_VISIBILITY", cal.rejection_reason or "")

    def test_cascade_does_not_blindly_trust_unreliable_cnn_on_disagreement(self) -> None:
        """
        Verify user requirement: when disagreement triggers the cascade, if the eye crop is
        blurry/low-reliability while geometric landmarks are reliable, arbitration must favor
        GEOMETRIC rather than blindly assuming the CNN is right.
        """
        cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.C_CONFIDENCE_GATED_CASCADE)
        blurry_crop = EyeCropQuality(
            crop_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
            sharpness_score=0.05,
            exposure_score=0.10,
            glare_fraction=0.0,
            overall_reliability=0.10,  # Very poor crop quality
        )
        geom = FrameGeometryFeatures(
            timestamp_ms=1000.0,
            face_detected=True,
            landmark_confidence=0.95,
            left_ear=0.28,
            right_ear=0.28,
            yaw_weighted_ear=0.28,
            blendshape_blink_left=0.45,  # Causes EAR vs blendshape disagreement to trigger gate
            blendshape_blink_right=0.45,
            blendshape_blink_mean=0.45,
            mar=0.15,
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            pose_zone=PoseZone.CENTER,
            geometric_reliability=0.78,
            left_eye_crop=blurry_crop,
            right_eye_crop=blurry_crop,
        )
        res = cascade.evaluate(geom=geom, reference_open_ear=0.28, close_threshold_ear=0.20)
        self.assertTrue(res.cnn_invoked)
        self.assertIn("GEOMETRIC", res.winning_authority)
        self.assertLess(res.fused_closure_score, 0.30)

    def test_mirror_check_context_prevents_false_warning(self) -> None:
        """
        A driver checking their left mirror for 2 seconds has a lower apparent 2D EAR (0.195),
        which matches their calibrated LEFT_MIRROR baseline and must NOT trigger a warning.
        """
        profile = DriverBehaviourProfile.create_default_uncalibrated("mirror_tester")
        engine = PersonalisedAdaptiveRiskEngine(profile=profile)
        kin = TemporalKinematicsTracker()
        cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.A_GEOMETRIC_ONLY)

        for i in range(40):  # 2.0 seconds at 20 FPS
            ts = i * 50.0
            geom = FrameGeometryFeatures(
                timestamp_ms=ts,
                face_detected=True,
                landmark_confidence=0.92,
                left_ear=0.255,
                right_ear=0.255,
                yaw_weighted_ear=0.255,
                blendshape_blink_left=0.08,
                blendshape_blink_right=0.08,
                blendshape_blink_mean=0.08,
                mar=0.14,
                pitch_deg=0.0,
                yaw_deg=-22.0,  # Left mirror check
                roll_deg=0.0,
                pose_zone=PoseZone.LEFT_MIRROR,
                geometric_reliability=0.85,
            )
            zb = profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(geom, zb.ear_median, zb.dynamic_closure_threshold)
            k_res = kin.update(ts, geom.yaw_weighted_ear, c_res.fused_closure_score, geom.mar, zb.dynamic_closure_threshold)
            r_res = engine.evaluate(geom, c_res, k_res)
            self.assertEqual(r_res.state, AlertState.NORMAL)
            self.assertTrue(r_res.deviations.is_mirror_check_context)


if __name__ == "__main__":
    unittest.main()
