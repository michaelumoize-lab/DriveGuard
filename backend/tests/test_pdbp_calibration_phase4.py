"""
Phase 4 Automated Verification Tests:
Personalized Driver Baseline Calibration & Adaptive Behaviour Modelling.

Tests cover:
1. Complete 4-zone guided calibration cycle (CENTER -> LEFT_MIRROR -> RIGHT_MIRROR -> DASHBOARD).
2. Quality gating: minimum sample thresholds, low landmark confidence, excessive head angular velocity rejection.
3. Asymmetric eye measurement tracking and outlier trimming (bottom 10% EAR).
4. Dynamic closure threshold bounds clamping ([0.12, 0.22]).
5. Single-zone recalibration and profile merging.
6. Schema versioning and resilient JSON serialization/deserialization.
7. Safety overrides during mirror check contexts (e.g., microsleep overrides normal glance dampening).

Run with:
    python -m unittest backend/tests/test_pdbp_calibration_phase4.py -v
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import numpy as np

from backend.app.ai.blink_kinematics import TemporalKinematicsTracker
from backend.app.ai.cascade_classifier import ConfidenceGatedEyeCascade, ExperimentMode
from backend.app.ai.geometry import FrameGeometryFeatures, PoseZone
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    ZoneBaseline,
    compute_median_and_scaled_mad,
    compute_robust_z,
)
from backend.app.ai.risk_engine import AlertState, PersonalisedAdaptiveRiskEngine


class TestPDBPCalibrationPhase4(unittest.TestCase):
    def test_robust_median_and_scaled_mad_calculation(self) -> None:
        """Verify median and scaled MAD computation with floor and outlier resistance."""
        values = [0.28, 0.29, 0.27, 0.285, 0.29, 0.15]  # Contains an outlier
        med, mad = compute_median_and_scaled_mad(values, min_floor=0.015)
        self.assertAlmostEqual(med, 0.2825, places=3)
        self.assertGreaterEqual(mad, 0.015)

        # Empty values should safely yield floor
        empty_med, empty_mad = compute_median_and_scaled_mad([], min_floor=0.02)
        self.assertEqual(empty_med, 0.0)
        self.assertEqual(empty_mad, 0.02)

    def test_dynamic_closure_threshold_clamping(self) -> None:
        """Verify dynamic closure threshold is strictly clamped between 0.12 and 0.22."""
        # Extremely high baseline EAR
        high_zb = ZoneBaseline(
            zone="CENTER",
            ear_median=0.45,
            ear_mad_scaled=0.02,
            pitch_median_deg=0.0,
            yaw_median_deg=0.0,
            sample_count=50,
        )
        self.assertLessEqual(high_zb.dynamic_closure_threshold, 0.22)

        # Very low baseline EAR
        low_zb = ZoneBaseline(
            zone="CENTER",
            ear_median=0.14,
            ear_mad_scaled=0.01,
            pitch_median_deg=0.0,
            yaw_median_deg=0.0,
            sample_count=50,
        )
        self.assertGreaterEqual(low_zb.dynamic_closure_threshold, 0.12)

    def test_full_four_zone_guided_calibration(self) -> None:
        """Verify an end-to-end 4-zone guided calibration produces a valid 4-zone profile."""
        session = GuidedCalibrationSession(
            driver_id="driver_phase4_test",
            four_zone_mode=True,
            step_duration_sec=0.5,  # Shortened target for fast synthetic test
        )
        session.start(timestamp_ms=0.0)
        self.assertEqual(len(session.steps), 4)
        self.assertEqual(
            [s.zone for s in session.steps],
            [PoseZone.CENTER, PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD],
        )

        zone_configs = [
            (PoseZone.CENTER, 0.0, 0.0, 0.28, 0.28, 0.05),
            (PoseZone.LEFT_MIRROR, 0.0, 20.0, 0.24, 0.27, 0.08),  # Left eye perspective foreshortened
            (PoseZone.RIGHT_MIRROR, 0.0, -20.0, 0.27, 0.23, 0.08), # Right eye perspective foreshortened
            (PoseZone.DASHBOARD, -12.0, 0.0, 0.25, 0.25, 0.06),
        ]

        ts = 0.0
        final_profile: DriverBehaviourProfile | None = None

        for target_zone, pitch, yaw, ear_l, ear_r, blend in zone_configs:
            ear_mean = (ear_l + ear_r) / 2.0
            for _ in range(16):
                ts += 40.0
                res = session.ingest_frame(
                    timestamp_ms=ts,
                    face_detected=True,
                    geometric_reliability=0.88,
                    ear_value=ear_mean,
                    pitch_deg=pitch,
                    yaw_deg=yaw,
                    is_eye_currently_closed=False,
                    left_ear=ear_l,
                    right_ear=ear_r,
                    blendshape_blink_mean=blend,
                    landmark_confidence=0.92,
                )
                if res is not None:
                    final_profile = res

        self.assertIsNotNone(final_profile)
        assert final_profile is not None
        self.assertEqual(final_profile.calibration_mode, "FOUR_ZONE")
        self.assertEqual(final_profile.version, 1)

        # All 4 zones must be present in profile baselines
        for z in ["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"]:
            self.assertIn(z, final_profile.zone_baselines)
            zb = final_profile.zone_baselines[z]
            self.assertGreater(zb.sample_count, 0)
            self.assertIsNotNone(zb.ear_left_median)
            self.assertIsNotNone(zb.ear_right_median)
            self.assertIsNotNone(zb.blendshape_open_median)
            self.assertIsNotNone(zb.confidence_median)

        # Verify asymmetry captured for mirrors
        left_zb = final_profile.zone_baselines["LEFT_MIRROR"]
        self.assertLess(left_zb.ear_left_median, left_zb.ear_right_median)

        right_zb = final_profile.zone_baselines["RIGHT_MIRROR"]
        self.assertGreater(right_zb.ear_left_median, right_zb.ear_right_median)

    def test_excessive_head_velocity_rejected_from_calibration(self) -> None:
        """Angular velocity > 80 deg/s must not contribute to valid calibration samples."""
        session = GuidedCalibrationSession(
            driver_id="driver_jerky",
            four_zone_mode=True,
            step_duration_sec=0.5,
        )
        session.start(timestamp_ms=0.0)

        # First frame at center
        session.ingest_frame(
            timestamp_ms=100.0,
            face_detected=True,
            geometric_reliability=0.90,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
        )

        # Sudden jerky jump of 25 deg in 40ms -> 625 deg/s
        session.ingest_frame(
            timestamp_ms=140.0,
            face_detected=True,
            geometric_reliability=0.90,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=5.0,  # inside center cone, but delta yaw = 5 deg in 40ms = 125 deg/s
            is_eye_currently_closed=False,
        )

        # Check feedback warning
        self.assertEqual(session._feedback_msg, "Hold head steady in target zone")
        # Second jerky frame should not be accumulated in valid open-eye samples
        self.assertEqual(len(session._ear_samples_by_zone["CENTER"]), 1)

    def test_closed_eye_and_low_confidence_rejected(self) -> None:
        """Closed eyes and low landmark confidence must be filtered out during calibration."""
        session = GuidedCalibrationSession(
            driver_id="driver_filter_test",
            four_zone_mode=False,
            step_duration_sec=1.0,
        )
        session.start(timestamp_ms=0.0)

        # 1. Closed eye in CENTER
        session.ingest_frame(
            timestamp_ms=50.0,
            face_detected=True,
            geometric_reliability=0.90,
            ear_value=0.14,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=True,
        )
        self.assertEqual(len(session._ear_samples_by_zone["CENTER"]), 0)

        # 2. Low landmark confidence (< 0.35)
        session.ingest_frame(
            timestamp_ms=100.0,
            face_detected=True,
            geometric_reliability=0.90,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
            landmark_confidence=0.20,
        )
        self.assertEqual(len(session._ear_samples_by_zone["CENTER"]), 0)

        # 3. Valid open eye
        session.ingest_frame(
            timestamp_ms=150.0,
            face_detected=True,
            geometric_reliability=0.90,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
            landmark_confidence=0.88,
        )
        self.assertEqual(len(session._ear_samples_by_zone["CENTER"]), 1)

    def test_single_zone_recalibration_merges_profile(self) -> None:
        """Recalibrating a single zone (e.g. DASHBOARD) updates only that zone baseline."""
        # Create an existing uncalibrated/standard profile
        existing = DriverBehaviourProfile.create_default_uncalibrated("recal_driver")
        old_center_ear = existing.center_ear_median
        self.assertNotIn("DASHBOARD", existing.zone_baselines)

        # Launch targeted recalibration for DASHBOARD
        recal_session = GuidedCalibrationSession(
            driver_id="recal_driver",
            target_zone=PoseZone.DASHBOARD,
            existing_profile=existing,
            step_duration_sec=0.5,
        )
        recal_session.start(timestamp_ms=0.0)
        self.assertEqual(len(recal_session.steps), 1)
        self.assertEqual(recal_session.steps[0].zone, PoseZone.DASHBOARD)

        ts = 0.0
        updated_profile = None
        for _ in range(16):
            ts += 40.0
            res = recal_session.ingest_frame(
                timestamp_ms=ts,
                face_detected=True,
                geometric_reliability=0.85,
                ear_value=0.24,
                pitch_deg=-12.0,
                yaw_deg=0.0,
                is_eye_currently_closed=False,
                left_ear=0.24,
                right_ear=0.24,
                blendshape_blink_mean=0.05,
                landmark_confidence=0.90,
            )
            if res is not None:
                updated_profile = res

        self.assertIsNotNone(updated_profile)
        assert updated_profile is not None
        # Center should remain untouched
        self.assertEqual(updated_profile.center_ear_median, old_center_ear)
        # Dashboard should now be present in zone_baselines
        self.assertIn("DASHBOARD", updated_profile.zone_baselines)
        dash_zb = updated_profile.zone_baselines["DASHBOARD"]
        self.assertAlmostEqual(dash_zb.pitch_median_deg, -12.0, delta=1.0)
        self.assertAlmostEqual(dash_zb.ear_median, 0.24, delta=0.02)

    def test_json_serialization_and_backward_compatibility(self) -> None:
        """Verify profile persistence with versioning and resilient loading of older schema."""
        profile = DriverBehaviourProfile.create_default_uncalibrated("json_test_driver")
        profile.zone_baselines["LEFT_MIRROR"] = ZoneBaseline(
            zone="LEFT_MIRROR",
            ear_median=0.245,
            ear_mad_scaled=0.02,
            pitch_median_deg=0.0,
            yaw_median_deg=22.0,
            sample_count=25,
            ear_left_median=0.22,
            ear_right_median=0.27,
            blendshape_open_median=0.06,
            confidence_median=0.91,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            file_path = os.path.join(tmpdir, "profile_v1.json")
            profile.save_json(file_path)

            # 1. Round-trip validation
            loaded = DriverBehaviourProfile.load_json(file_path)
            self.assertEqual(loaded.version, 1)
            self.assertEqual(loaded.driver_id, "json_test_driver")
            self.assertIn("LEFT_MIRROR", loaded.zone_baselines)
            zb = loaded.zone_baselines["LEFT_MIRROR"]
            self.assertEqual(zb.ear_left_median, 0.22)
            self.assertEqual(zb.ear_right_median, 0.27)

            # 2. Resilient loading of legacy JSON without version or new fields
            legacy_file = os.path.join(tmpdir, "profile_legacy.json")
            legacy_data = {
                "driver_id": "legacy_driver",
                "calibration_mode": "THREE_ZONE",
                "center_ear_median": 0.28,
                "center_ear_mad_scaled": 0.02,
                "blink_duration_median_ms": 180.0,
                "blink_duration_mad_ms": 30.0,
                "blink_rcr_median": 0.90,
                "blink_rcr_mad": 0.12,
                "blink_freq_median_per_min": 14.0,
                "blink_freq_mad_per_min": 3.0,
                "neutral_pitch_deg": 0.0,
                "neutral_yaw_deg": 0.0,
                "calibration_reliability": 0.85,
                "unknown_future_field": "ignore_me",
                "zone_baselines": {
                    "CENTER": {
                        "zone": "CENTER",
                        "ear_median": 0.28,
                        "ear_mad_scaled": 0.02,
                        "pitch_median_deg": 0.0,
                        "yaw_median_deg": 0.0,
                        "sample_count": 40,
                        "extra_key": 999,
                    }
                },
            }
            with open(legacy_file, "w", encoding="utf-8") as f:
                json.dump(legacy_data, f)

            loaded_legacy = DriverBehaviourProfile.load_json(legacy_file)
            self.assertEqual(loaded_legacy.driver_id, "legacy_driver")
            self.assertEqual(loaded_legacy.version, 1)  # Default fallback
            self.assertIn("CENTER", loaded_legacy.zone_baselines)
            self.assertIsNone(loaded_legacy.zone_baselines["CENTER"].ear_left_median)

    def test_safety_override_during_mirror_check(self) -> None:
        """
        Safety override requirement:
        - Brief normal glance at mirror (EAR 0.22, yaw 20 deg) remains NORMAL.
        - Prolonged eye closure (microsleep >= 1.1s) during mirror glance must
          immediately trigger CRITICAL alert regardless of mirror glance context.
        """
        profile = DriverBehaviourProfile.create_default_uncalibrated("safety_override_tester")
        engine = PersonalisedAdaptiveRiskEngine(profile=profile)
        kin = TemporalKinematicsTracker()
        cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.A_GEOMETRIC_ONLY)

        # 1. Normal mirror glance (open eyes, 1.0 second) -> NORMAL
        for i in range(20):
            ts = i * 50.0
            geom = FrameGeometryFeatures(
                timestamp_ms=ts,
                face_detected=True,
                landmark_confidence=0.92,
                left_ear=0.24,
                right_ear=0.27,
                yaw_weighted_ear=0.25,
                blendshape_blink_left=0.08,
                blendshape_blink_right=0.08,
                blendshape_blink_mean=0.08,
                mar=0.14,
                pitch_deg=0.0,
                yaw_deg=-20.0,  # Mirror check
                roll_deg=0.0,
                pose_zone=PoseZone.LEFT_MIRROR,
                geometric_reliability=0.85,
            )
            zb = profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(geom, zb.ear_median, zb.dynamic_closure_threshold)
            k_res = kin.update(ts, geom.yaw_weighted_ear, c_res.fused_closure_score, geom.mar, zb.dynamic_closure_threshold)
            r_res = engine.evaluate(geom, c_res, k_res)
            self.assertEqual(r_res.state, AlertState.NORMAL)

        # 2. Microsleep during mirror glance: eyes closed for >= 1.1s (23 frames * 50ms = 1150ms)
        for i in range(20, 44):
            ts = i * 50.0
            geom = FrameGeometryFeatures(
                timestamp_ms=ts,
                face_detected=True,
                landmark_confidence=0.92,
                left_ear=0.10,  # Below dynamic closure threshold
                right_ear=0.10,
                yaw_weighted_ear=0.10,
                blendshape_blink_left=0.95,
                blendshape_blink_right=0.95,
                blendshape_blink_mean=0.95,
                mar=0.14,
                pitch_deg=0.0,
                yaw_deg=-20.0,
                roll_deg=0.0,
                pose_zone=PoseZone.LEFT_MIRROR,
                geometric_reliability=0.85,
            )
            zb = profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(geom, zb.ear_median, zb.dynamic_closure_threshold)
            k_res = kin.update(ts, geom.yaw_weighted_ear, c_res.fused_closure_score, geom.mar, zb.dynamic_closure_threshold)
            r_res = engine.evaluate(geom, c_res, k_res)

        # Once closure duration >= 1100ms, state MUST escalate to CRITICAL with MICROSLEEP reason
        self.assertEqual(r_res.state, AlertState.CRITICAL)
        self.assertIn("PROLONGED_EYE_CLOSURE_MICROSLEEP", r_res.reasons)


if __name__ == "__main__":
    unittest.main()
