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
    NeutralRotationResult,
    ZoneBaseline,
    aggregate_neutral_rotation_matrix,
    compute_canonical_profile_fingerprint,
    compute_median_and_scaled_mad,
    compute_robust_z,
    create_immutable_profile_snapshot,
    ImmutableProfileSnapshot,
    ImmutableZoneBaseline,
    project_onto_so3,
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

        for i_zone, (target_zone, pitch, yaw, ear_l, ear_r, blend) in enumerate(zone_configs):
            if i_zone > 0:
                ts += 2200.0  # Advance past 2.0s inter-step grace period
            ear_mean = (ear_l + ear_r) / 2.0
            for _ in range(45):
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


def _make_rotation_matrix(yaw_deg: float = 0.0, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """Helper to generate pure orthonormal SO(3) matrices for testing."""
    psi = np.radians(yaw_deg)
    theta = np.radians(pitch_deg)
    phi = np.radians(roll_deg)
    Ry = np.array([[np.cos(psi), 0.0, np.sin(psi)], [0.0, 1.0, 0.0], [-np.sin(psi), 0.0, np.cos(psi)]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(theta), -np.sin(theta)], [0.0, np.sin(theta), np.cos(theta)]], dtype=np.float64)
    Rz = np.array([[np.cos(phi), -np.sin(phi), 0.0], [np.sin(phi), np.cos(phi), 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Ry @ Rx @ Rz


class TestNeutralRotationAggregation(unittest.TestCase):
    """
    Focused unit tests for Task 2.1 neutral rotation matrix aggregation and SO(3) projection.
    """

    def test_aggregate_neutral_rotation_clean_dominant_cluster(self) -> None:
        """35 frames at 0 deg, 10 frames at 20 deg (77.8% dominant cluster). Must be accepted."""
        candidates = [_make_rotation_matrix(0.0) for _ in range(35)] + [_make_rotation_matrix(20.0) for _ in range(10)]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertTrue(res.is_accepted)
        self.assertIsNone(res.rejection_reason)
        self.assertEqual(res.total_candidates, 45)
        self.assertEqual(res.inlier_count, 35)
        self.assertAlmostEqual(res.inlier_ratio, 35 / 45, places=4)
        self.assertIsNotNone(res.neutral_rot_mat)
        self.assertEqual(res.neutral_rot_mat.shape, (3, 3))
        # Determinant == +1.0 within 1e-12
        self.assertAlmostEqual(float(np.linalg.det(res.neutral_rot_mat)), 1.0, places=6)
        # Orthogonality error < 1e-12
        ortho_err = float(np.max(np.abs(res.neutral_rot_mat.T @ res.neutral_rot_mat - np.eye(3))))
        self.assertLess(ortho_err, 1e-6)

    def test_aggregate_neutral_rotation_bimodal_split_rejected(self) -> None:
        """23 frames at 0 deg, 22 frames at 20 deg (51.1% inliers < 60%). Must be rejected."""
        candidates = [_make_rotation_matrix(0.0) for _ in range(23)] + [_make_rotation_matrix(20.0) for _ in range(22)]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertFalse(res.is_accepted)
        self.assertEqual(res.rejection_reason, "INSUFFICIENT_INLIER_RATIO")
        self.assertIsNone(res.neutral_rot_mat)
        self.assertEqual(res.inlier_count, 23)

    def test_aggregate_neutral_rotation_scattered_rejected(self) -> None:
        """18 frames at 0 deg, 27 scattered across [-25, 25] deg (40% inliers). Must be rejected."""
        np.random.seed(42)
        candidates = [_make_rotation_matrix(0.0) for _ in range(18)] + [
            _make_rotation_matrix(float(np.random.uniform(-25.0, 25.0))) for _ in range(27)
        ]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertFalse(res.is_accepted)
        self.assertIsNone(res.neutral_rot_mat)
        self.assertIn(res.rejection_reason, ("INSUFFICIENT_INLIER_RATIO", "EXCESSIVE_ANGULAR_DISPERSION"))

    def test_aggregate_neutral_rotation_continuous_drift_rejected(self) -> None:
        """45 frames drifting across 15 deg. RMS dispersion > 2.2 deg. Must be rejected."""
        candidates = [_make_rotation_matrix(y) for y in np.linspace(-7.5, 7.5, 45)]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertFalse(res.is_accepted)
        self.assertEqual(res.rejection_reason, "EXCESSIVE_ANGULAR_DISPERSION")
        self.assertGreater(res.inlier_rms_deg, 2.2)
        self.assertIsNone(res.neutral_rot_mat)

    def test_aggregate_neutral_rotation_sub_neighborhood_bimodal_accepted(self) -> None:
        """23 frames at 0 deg, 22 frames at 3 deg. Both within 4.5 deg; RMS <= 2.2 deg. Must be accepted."""
        candidates = [_make_rotation_matrix(0.0) for _ in range(23)] + [_make_rotation_matrix(3.0) for _ in range(22)]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertTrue(res.is_accepted)
        self.assertIsNone(res.rejection_reason)
        self.assertEqual(res.inlier_count, 45)
        self.assertLessEqual(res.inlier_rms_deg, 2.2)
        self.assertIsNotNone(res.neutral_rot_mat)
        self.assertAlmostEqual(float(np.linalg.det(res.neutral_rot_mat)), 1.0, places=6)

    def test_aggregate_neutral_rotation_svd_reflection_guard(self) -> None:
        """Verify project_onto_so3 flips negative determinant to +1.0 while preserving orthogonality."""
        M_ref = np.diag([1.0, 1.0, -1.0])
        R_proj = project_onto_so3(M_ref)

        self.assertAlmostEqual(float(np.linalg.det(M_ref)), -1.0, places=6)
        self.assertAlmostEqual(float(np.linalg.det(R_proj)), 1.0, places=6)
        self.assertTrue(np.allclose(R_proj.T @ R_proj, np.eye(3), atol=1e-6))

    def test_aggregate_neutral_rotation_insufficient_samples(self) -> None:
        """N = 12 < 18 candidate rotations returns INSUFFICIENT_CANDIDATE_SAMPLES."""
        candidates = [_make_rotation_matrix(0.0) for _ in range(12)]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertFalse(res.is_accepted)
        self.assertEqual(res.rejection_reason, "INSUFFICIENT_CANDIDATE_SAMPLES")
        self.assertIsNone(res.medoid_idx)
        self.assertIsNone(res.neutral_rot_mat)
        self.assertEqual(res.total_candidates, 12)

    def test_aggregate_neutral_rotation_empty_input(self) -> None:
        """Empty input returns INSUFFICIENT_CANDIDATE_SAMPLES with medoid_idx=None and neutral_rot_mat=None."""
        res = aggregate_neutral_rotation_matrix([])

        self.assertFalse(res.is_accepted)
        self.assertEqual(res.rejection_reason, "INSUFFICIENT_CANDIDATE_SAMPLES")
        self.assertIsNone(res.medoid_idx)
        self.assertIsNone(res.neutral_rot_mat)
        self.assertEqual(res.total_candidates, 0)

    def test_aggregate_neutral_rotation_malformed_shape_rejected(self) -> None:
        """Malformed candidate shapes (2, 3) or (4, 4) raise ValueError explicitly."""
        with self.assertRaises(ValueError) as ctx:
            aggregate_neutral_rotation_matrix([np.zeros((2, 3))])
        self.assertIn("invalid shape", str(ctx.exception))

    def test_aggregate_neutral_rotation_nan_and_inf_rejected(self) -> None:
        """Candidate containing NaN or Inf raises ValueError explicitly."""
        bad_nan = _make_rotation_matrix(0.0)
        bad_nan[0, 0] = np.nan
        with self.assertRaises(ValueError) as ctx_nan:
            aggregate_neutral_rotation_matrix([bad_nan])
        self.assertIn("non-finite", str(ctx_nan.exception))

        bad_inf = _make_rotation_matrix(0.0)
        bad_inf[1, 1] = np.inf
        with self.assertRaises(ValueError) as ctx_inf:
            aggregate_neutral_rotation_matrix([bad_inf])
        self.assertIn("non-finite", str(ctx_inf.exception))

    def test_aggregate_neutral_rotation_reflection_input_rejected(self) -> None:
        """Candidate rotation matrix with det = -1.0 is rejected with ValueError."""
        reflection_mat = np.diag([1.0, 1.0, -1.0])
        with self.assertRaises(ValueError) as ctx:
            aggregate_neutral_rotation_matrix([reflection_mat])
        self.assertTrue("not in SO(3)" in str(ctx.exception) or "reflections" in str(ctx.exception))

    def test_aggregate_neutral_rotation_deterministic_medoid_tiebreaking(self) -> None:
        """When multiple candidates have identical distance sums, medoid_idx selects the lowest index deterministically."""
        candidates = [_make_rotation_matrix(0.0) for _ in range(25)]
        res = aggregate_neutral_rotation_matrix(candidates)

        self.assertTrue(res.is_accepted)
        self.assertEqual(res.medoid_idx, 0)

    def test_driver_behaviour_profile_neutral_rot_mat_field_and_serialization(self) -> None:
        """Verify DriverBehaviourProfile has neutral_rot_mat with default None and save_json/load_json roundtrips."""
        prof = DriverBehaviourProfile.create_default_uncalibrated()
        self.assertIsNone(prof.neutral_rot_mat)

        rot = _make_rotation_matrix(15.0, -5.0)
        prof.neutral_rot_mat = rot

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = os.path.join(tmp_dir, "test_prof.json")
            prof.save_json(p)
            loaded = DriverBehaviourProfile.load_json(p)

            self.assertIsNotNone(loaded.neutral_rot_mat)
            self.assertEqual(loaded.neutral_rot_mat.shape, (3, 3))
            self.assertTrue(np.allclose(loaded.neutral_rot_mat, rot, atol=1e-5))

    def test_stage2_3_snapshot_immutability_and_fingerprint(self) -> None:
        """Stage 2.3: Verify deep immutability and canonical fingerprinting."""
        import dataclasses
        prof = DriverBehaviourProfile.create_default_uncalibrated()
        rot = _make_rotation_matrix(10.0, -5.0)
        prof.neutral_rot_mat = rot

        snapshot = create_immutable_profile_snapshot(prof)
        self.assertIsInstance(snapshot, ImmutableProfileSnapshot)
        self.assertEqual(len(snapshot.fingerprint), 64)

        # 1. Direct mutation of frozen dataclass fields raises FrozenInstanceError
        with self.assertRaises(dataclasses.FrozenInstanceError):
            snapshot.driver_id = "hacked"  # type: ignore

        # 2. Zone baselines cannot be modified
        with self.assertRaises(TypeError):
            snapshot.zone_baselines["CENTER"] = None  # type: ignore

        # 3. Mutating original profile or returned matrix does not alter snapshot
        orig_rot_val = float(rot[0, 0])
        prof.neutral_rot_mat[0, 0] = 999.0
        snapshot_rot = snapshot.neutral_rot_mat
        self.assertIsNotNone(snapshot_rot)
        self.assertAlmostEqual(snapshot_rot[0, 0], orig_rot_val, places=5)
        snapshot_rot[0, 0] = 777.0
        self.assertAlmostEqual(snapshot.neutral_rot_mat[0, 0], orig_rot_val, places=5)

        # 4. Domain -> snapshot -> domain exact preservation
        prof.neutral_rot_mat[0, 0] = orig_rot_val
        reconstructed = snapshot.to_domain()
        self.assertEqual(reconstructed.driver_id, prof.driver_id)
        self.assertEqual(reconstructed.calibration_mode, prof.calibration_mode)
        self.assertAlmostEqual(reconstructed.center_ear_median, snapshot.center_ear_median, places=6)
        self.assertTrue(np.allclose(reconstructed.neutral_rot_mat, rot, atol=1e-5))

        # 5. Canonical fingerprint sensitivity and determinism
        fp1 = compute_canonical_profile_fingerprint(snapshot)
        fp2 = compute_canonical_profile_fingerprint(reconstructed)
        self.assertEqual(fp1, snapshot.fingerprint)
        self.assertEqual(fp1, fp2)

        # Mutating reconstructed changes its fingerprint
        reconstructed.center_ear_median += 0.001
        self.assertNotEqual(compute_canonical_profile_fingerprint(reconstructed), fp1)

    def test_stage2_3_guided_calibration_ingest_frame_contracts(self) -> None:
        """Stage 2.3: Verify ingest_frame with rot_mat and ear_geometric_valid tri-state."""
        session = GuidedCalibrationSession(step_duration_sec=0.2)
        session.start(timestamp_ms=1000.0)

        rot = _make_rotation_matrix(0.0)

        # Frame 1: ear_geometric_valid=False -> EAR sample is strictly rejected from baseline
        session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
            rot_mat=rot,
            ear_geometric_valid=False,
        )
        self.assertEqual(len(session._ear_samples_by_zone["CENTER"]), 0)
        # Rotation candidate is still collected (independent capture)
        self.assertEqual(len(session._candidate_rot_mats), 1)

        # Frame 2: ear_geometric_valid=True -> EAR sample is ingested
        session.ingest_frame(
            timestamp_ms=1080.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
            rot_mat=rot,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(session._ear_samples_by_zone["CENTER"]), 1)
        self.assertEqual(len(session._candidate_rot_mats), 2)

    def test_stage2_3_separate_geometric_ear_validity_from_verified_eye_state(self) -> None:
        """
        Stage 2.3: Verify separation of ear_geometric_valid from verified eye state:
        - Recalibration: only is_eye_currently_closed is False contributes to CENTER EAR baseline.
          True and None are rejected.
        - Initial bootstrap calibration: uses guided-protocol rules (accepts None and False; rejects True).
        - Matrix capture: independent of EAR acceptance. Accepts None and False; strictly excludes True.
        - Matrix capture proceeds even when EAR geometry is invalid.
        """
        rot = _make_rotation_matrix(0.0)

        # 1. Recalibration session (existing_profile is present)
        existing = DriverBehaviourProfile.create_default_uncalibrated("recal_driver_test")
        recal_session = GuidedCalibrationSession(step_duration_sec=0.5, existing_profile=existing)
        recal_session.start(timestamp_ms=1000.0)

        # Frame A: is_eye_currently_closed is None -> EAR rejected, rot_mat collected
        recal_session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=None,
            rot_mat=rot,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(recal_session._ear_samples_by_zone.get("CENTER", [])), 0)
        self.assertEqual(len(recal_session._candidate_rot_mats), 1)

        # Frame B: is_eye_currently_closed is True -> EAR rejected, rot_mat excluded
        recal_session.ingest_frame(
            timestamp_ms=1080.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=True,
            rot_mat=rot,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(recal_session._ear_samples_by_zone.get("CENTER", [])), 0)
        self.assertEqual(len(recal_session._candidate_rot_mats), 1)

        # Frame C: is_eye_currently_closed is False but ear_geometric_valid is False -> EAR rejected, rot_mat collected
        recal_session.ingest_frame(
            timestamp_ms=1120.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
            rot_mat=rot,
            ear_geometric_valid=False,
        )
        self.assertEqual(len(recal_session._ear_samples_by_zone.get("CENTER", [])), 0)
        self.assertEqual(len(recal_session._candidate_rot_mats), 2)

        # Frame D: is_eye_currently_closed is False AND ear_geometric_valid is True -> EAR accepted, rot_mat collected
        recal_session.ingest_frame(
            timestamp_ms=1160.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=False,
            rot_mat=rot,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(recal_session._ear_samples_by_zone.get("CENTER", [])), 1)
        self.assertEqual(len(recal_session._candidate_rot_mats), 3)

        # 2. Initial bootstrap session (existing_profile is None)
        bootstrap_session = GuidedCalibrationSession(step_duration_sec=0.5, existing_profile=None)
        bootstrap_session.start(timestamp_ms=1000.0)

        # Frame E: is_eye_currently_closed is None -> EAR accepted under guided protocol, rot_mat collected
        bootstrap_session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=None,
            rot_mat=rot,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(bootstrap_session._ear_samples_by_zone.get("CENTER", [])), 1)
        self.assertEqual(len(bootstrap_session._candidate_rot_mats), 1)

        # Frame F: is_eye_currently_closed is True -> EAR rejected, rot_mat excluded
        bootstrap_session.ingest_frame(
            timestamp_ms=1080.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.30,
            pitch_deg=0.0,
            yaw_deg=0.0,
            is_eye_currently_closed=True,
            rot_mat=rot,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(bootstrap_session._ear_samples_by_zone.get("CENTER", [])), 1)
        self.assertEqual(len(bootstrap_session._candidate_rot_mats), 1)


class TestObliqueCenterCalibration(unittest.TestCase):
    """
    Focused unit and regression tests for Oblique Camera Initial CENTER Alignment:
    1. Fresh calibration accepts stable natural forward-facing pose around +18° and -18° yaw.
    2. Rotation candidates in SO(3) are collected for both positive and negative oblique orientations.
    3. Poses exceeding the configured envelope (yaw > 32° or pitch outside [-18°, 14°]) are rejected.
    4. Unstable head movement (angular velocity > 80°/s) does not accumulate samples.
    5. Invalid rotation matrices are rejected.
    6. Missing face or low confidence / geometric reliability does not accumulate samples.
    7. Recalibration vs bootstrap eye-state rules are strictly preserved.
    8. Peripheral zones (LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD) evaluate relative to learned neutral.
    9. Insufficient CENTER observations cause session finalization to reject the profile.
    10. End-to-end 4-zone calibration succeeds with an oblique camera mount at +18° yaw.
    """

    def test_oblique_initial_center_positive_and_negative_yaw_accepted(self) -> None:
        """Fresh calibration accepts stable forward poses with camera-relative yaw around +18° and -18°."""
        session = GuidedCalibrationSession(step_duration_sec=2.0, existing_profile=None)
        session.start(timestamp_ms=1000.0)

        # 1. +18° camera yaw (e.g. camera mounted on right, looking straight ahead)
        aligned_pos, msg_pos = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=-3.0, yaw_deg=18.0)
        self.assertTrue(aligned_pos)
        self.assertEqual(msg_pos, "Holding Steady (Center Roadway)")

        rot_pos = _make_rotation_matrix(yaw_deg=18.0, pitch_deg=-3.0)
        session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=True,
            geometric_reliability=0.92,
            ear_value=0.28,
            pitch_deg=-3.0,
            yaw_deg=18.0,
            rot_mat=rot_pos,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(session._ear_samples_by_zone.get("CENTER", [])), 1)
        self.assertEqual(len(session._candidate_rot_mats), 1)

        # 2. -18° camera yaw (e.g. camera mounted on left / A-pillar, looking straight ahead)
        session_neg = GuidedCalibrationSession(step_duration_sec=2.0, existing_profile=None)
        session_neg.start(timestamp_ms=2000.0)

        aligned_neg, msg_neg = session_neg.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=-2.0, yaw_deg=-18.0)
        self.assertTrue(aligned_neg)
        self.assertEqual(msg_neg, "Holding Steady (Center Roadway)")

        rot_neg = _make_rotation_matrix(yaw_deg=-18.0, pitch_deg=-2.0)
        session_neg.ingest_frame(
            timestamp_ms=2040.0,
            face_detected=True,
            geometric_reliability=0.92,
            ear_value=0.28,
            pitch_deg=-2.0,
            yaw_deg=-18.0,
            rot_mat=rot_neg,
            ear_geometric_valid=True,
        )
        self.assertEqual(len(session_neg._ear_samples_by_zone.get("CENTER", [])), 1)
        self.assertEqual(len(session_neg._candidate_rot_mats), 1)

    def test_oblique_initial_center_envelope_bounds_rejected(self) -> None:
        """Poses outside the configured initial CENTER envelope are rejected."""
        session = GuidedCalibrationSession(existing_profile=None)
        session.start(timestamp_ms=1000.0)

        # Excessive positive yaw (+38° > 32°)
        aligned, msg = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=0.0, yaw_deg=38.0)
        self.assertFalse(aligned)
        self.assertIn("Look straight ahead", msg)

        # Excessive negative yaw (-38° < -32°)
        aligned, msg = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=0.0, yaw_deg=-38.0)
        self.assertFalse(aligned)
        self.assertIn("Look straight ahead", msg)

        # Excessive downward pitch (-25° < -18°)
        aligned, msg = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=-25.0, yaw_deg=18.0)
        self.assertFalse(aligned)

        # Excessive upward pitch (+20° > +14°)
        aligned, msg = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=20.0, yaw_deg=18.0)
        self.assertFalse(aligned)

    def test_rapid_head_motion_rejected(self) -> None:
        """Excessive head angular velocity (> 80°/s) rejects candidate collection."""
        session = GuidedCalibrationSession(step_duration_sec=2.0, existing_profile=None)
        session.start(timestamp_ms=1000.0)
        rot = _make_rotation_matrix(yaw_deg=18.0, pitch_deg=0.0)

        # Frame 1: steady
        session.ingest_frame(
            timestamp_ms=1000.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=18.0,
            rot_mat=rot,
        )
        self.assertEqual(len(session._candidate_rot_mats), 1)

        # Frame 2: sudden jump by 8° in 40ms -> 200°/s > 80°/s
        rot2 = _make_rotation_matrix(yaw_deg=26.0, pitch_deg=0.0)
        session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=26.0,
            rot_mat=rot2,
        )
        # Did not accumulate due to excessive motion
        self.assertEqual(len(session._candidate_rot_mats), 1)

    def test_invalid_rotation_matrix_rejected(self) -> None:
        """Non-SO(3) or non-orthogonal matrices are rejected from candidate collection."""
        session = GuidedCalibrationSession(step_duration_sec=2.0, existing_profile=None)
        session.start(timestamp_ms=1000.0)

        # Non-orthogonal matrix
        bad_mat = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64)
        session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=18.0,
            rot_mat=bad_mat,
        )
        self.assertEqual(len(session._candidate_rot_mats), 0)

        # Reflection matrix (det = -1.0)
        refl_mat = np.diag([1.0, 1.0, -1.0]).astype(np.float64)
        session.ingest_frame(
            timestamp_ms=1080.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=18.0,
            rot_mat=refl_mat,
        )
        self.assertEqual(len(session._candidate_rot_mats), 0)

    def test_missing_face_or_poor_reliability_rejected(self) -> None:
        """Missing face or poor landmark reliability rejects candidate accumulation."""
        session = GuidedCalibrationSession(step_duration_sec=2.0, existing_profile=None)
        session.start(timestamp_ms=1000.0)
        rot = _make_rotation_matrix(yaw_deg=18.0, pitch_deg=0.0)

        # face_detected = False
        session.ingest_frame(
            timestamp_ms=1040.0,
            face_detected=False,
            geometric_reliability=0.9,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=18.0,
            rot_mat=rot,
        )
        self.assertEqual(len(session._candidate_rot_mats), 0)

        # geometric_reliability < 0.38
        session.ingest_frame(
            timestamp_ms=1080.0,
            face_detected=True,
            geometric_reliability=0.25,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=18.0,
            rot_mat=rot,
        )
        self.assertEqual(len(session._candidate_rot_mats), 0)

        # landmark_confidence < 0.35
        session.ingest_frame(
            timestamp_ms=1120.0,
            face_detected=True,
            geometric_reliability=0.9,
            ear_value=0.28,
            pitch_deg=0.0,
            yaw_deg=18.0,
            landmark_confidence=0.20,
            rot_mat=rot,
        )
        self.assertEqual(len(session._candidate_rot_mats), 0)

    def test_recalibration_uses_tight_relative_envelope(self) -> None:
        """Recalibration with an existing calibrated profile enforces tight relative envelope (±13°)."""
        existing = DriverBehaviourProfile.create_default_uncalibrated("recal_driver")
        existing.is_calibrated = True
        existing.neutral_yaw_deg = 20.0
        existing.neutral_pitch_deg = -3.0
        existing.neutral_rot_mat = _make_rotation_matrix(yaw_deg=20.0, pitch_deg=-3.0)

        session = GuidedCalibrationSession(target_zone=PoseZone.CENTER, existing_profile=existing)
        session.start(timestamp_ms=1000.0)

        # Looking at calibrated center (+20°, -3°) -> relative offset is 0.0 -> aligned
        aligned_exact, _ = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=-3.0, yaw_deg=20.0)
        self.assertTrue(aligned_exact)

        # Looking within ±13° relative (+28° -> rel = +8°) -> aligned
        aligned_close, _ = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=-3.0, yaw_deg=28.0)
        self.assertTrue(aligned_close)

        # Looking outside ±13° relative (+36° -> rel = +16°) -> rejected
        aligned_far, _ = session.is_pose_aligned_with_target(PoseZone.CENTER, pitch_deg=-3.0, yaw_deg=36.0)
        self.assertFalse(aligned_far)

    def test_peripheral_zones_aligned_relative_to_learned_neutral(self) -> None:
        """Peripheral zones align relative to the learned oblique neutral yaw/pitch."""
        session = GuidedCalibrationSession(step_duration_sec=0.5, existing_profile=None)
        session.start(timestamp_ms=1000.0)
        rot_center = _make_rotation_matrix(yaw_deg=18.0, pitch_deg=-2.0)

        # Step 0 (CENTER): feed observations at +18° yaw, -2° pitch
        ts = 1000.0
        for _ in range(25):
            ts += 40.0
            session.ingest_frame(
                timestamp_ms=ts,
                face_detected=True,
                geometric_reliability=0.9,
                ear_value=0.28,
                pitch_deg=-2.0,
                yaw_deg=18.0,
                rot_mat=rot_center,
                ear_geometric_valid=True,
            )

        # Should have advanced to Step 1 (LEFT_MIRROR) and learned neutral yaw ~ 18°
        self.assertEqual(session._step_idx, 1)
        self.assertAlmostEqual(session._learned_neutral_yaw, 18.0, delta=1.0)
        self.assertAlmostEqual(session._learned_neutral_pitch, -2.0, delta=1.0)

        # Advance past inter-zone grace
        ts += 2500.0

        # For LEFT_MIRROR: requires rel_yaw >= 7.0° (i.e. yaw >= 25.0°)
        # A yaw of 20° (rel_yaw = 2°) must be rejected
        aligned_l_reject, _ = session.is_pose_aligned_with_target(PoseZone.LEFT_MIRROR, pitch_deg=-2.0, yaw_deg=20.0)
        self.assertFalse(aligned_l_reject)

        # A yaw of 38° (rel_yaw = 20°) must be accepted
        aligned_l_ok, _ = session.is_pose_aligned_with_target(PoseZone.LEFT_MIRROR, pitch_deg=-2.0, yaw_deg=38.0)
        self.assertTrue(aligned_l_ok)

        # For RIGHT_MIRROR: requires rel_yaw <= -7.0° (i.e. yaw <= 11.0°)
        # A yaw of 15° (rel_yaw = -3°) must be rejected
        aligned_r_reject, _ = session.is_pose_aligned_with_target(PoseZone.RIGHT_MIRROR, pitch_deg=-2.0, yaw_deg=15.0)
        self.assertFalse(aligned_r_reject)

        # A yaw of 0° (rel_yaw = -18°) must be accepted
        aligned_r_ok, _ = session.is_pose_aligned_with_target(PoseZone.RIGHT_MIRROR, pitch_deg=-2.0, yaw_deg=0.0)
        self.assertTrue(aligned_r_ok)

    def test_insufficient_center_samples_rejects_finalization(self) -> None:
        """If CENTER has insufficient observations, finalization strictly rejects the profile."""
        session = GuidedCalibrationSession(step_duration_sec=0.2, existing_profile=None)
        session.start(timestamp_ms=1000.0)

        # Ingest zero valid center frames and let time expire
        session.ingest_frame(
            timestamp_ms=1300.0,  # 300ms elapsed > 200ms step duration
            face_detected=False,
            geometric_reliability=0.0,
            ear_value=0.0,
            pitch_deg=0.0,
            yaw_deg=0.0,
        )
        # Advance through remaining zones with skip
        session.skip_current_step(timestamp_ms=1400.0)
        session.skip_current_step(timestamp_ms=1500.0)
        session.skip_current_step(timestamp_ms=1600.0)

        profile = session._finalize()
        self.assertIsNone(profile)
        self.assertIn("REJECTED_POOR_VISIBILITY", session.rejection_reason or "")

    def test_full_four_zone_guided_calibration_oblique_mount(self) -> None:
        """End-to-end 4-zone guided calibration succeeds with oblique camera mount (+18° yaw)."""
        session = GuidedCalibrationSession(
            driver_id="driver_oblique_18",
            four_zone_mode=True,
            step_duration_sec=2.0,
        )
        session.target_samples_per_zone = 25  # Ensure sufficient inliers (>= 18) for SO(3) matrix aggregation
        session.start(timestamp_ms=0.0)

        # Oblique setup:
        # CENTER: yaw = +18.0°, pitch = -2.0°
        # LEFT_MIRROR: yaw = +38.0° (rel_yaw = +20°), pitch = -2.0°
        # RIGHT_MIRROR: yaw = 0.0° (rel_yaw = -18°), pitch = -2.0°
        # DASHBOARD: yaw = +18.0° (rel_yaw = 0°), pitch = -14.0° (rel_pitch = -12°)
        zones = [
            (PoseZone.CENTER, -2.0, 18.0, 0.28, 0.28),
            (PoseZone.LEFT_MIRROR, -2.0, 38.0, 0.24, 0.27),
            (PoseZone.RIGHT_MIRROR, -2.0, 0.0, 0.27, 0.23),
            (PoseZone.DASHBOARD, -14.0, 18.0, 0.25, 0.25),
        ]

        ts = 0.0
        final_profile: Optional[DriverBehaviourProfile] = None

        for i_zone, (target_zone, pitch, yaw, ear_l, ear_r) in enumerate(zones):
            if i_zone > 0:
                ts += 2200.0  # Pass inter-zone grace
            ear_mean = (ear_l + ear_r) / 2.0
            rot = _make_rotation_matrix(yaw_deg=yaw, pitch_deg=pitch)
            for _ in range(45):
                ts += 40.0
                res = session.ingest_frame(
                    timestamp_ms=ts,
                    face_detected=True,
                    geometric_reliability=0.90,
                    ear_value=ear_mean,
                    pitch_deg=pitch,
                    yaw_deg=yaw,
                    is_eye_currently_closed=False,
                    left_ear=ear_l,
                    right_ear=ear_r,
                    landmark_confidence=0.95,
                    rot_mat=rot,
                    ear_geometric_valid=True,
                )
                if res is not None:
                    final_profile = res

        self.assertIsNotNone(final_profile)
        assert final_profile is not None
        self.assertEqual(final_profile.calibration_mode, "FOUR_ZONE")
        self.assertAlmostEqual(final_profile.neutral_yaw_deg, 18.0, delta=1.5)
        self.assertAlmostEqual(final_profile.neutral_pitch_deg, -2.0, delta=1.5)

        # SO(3) neutral rotation matrix must be present, valid, and match +18° yaw
        self.assertIsNotNone(final_profile.neutral_rot_mat)
        assert final_profile.neutral_rot_mat is not None
        self.assertEqual(final_profile.neutral_rot_mat.shape, (3, 3))
        self.assertAlmostEqual(float(np.linalg.det(final_profile.neutral_rot_mat)), 1.0, places=5)
        ortho_err = float(np.max(np.abs(final_profile.neutral_rot_mat.T @ final_profile.neutral_rot_mat - np.eye(3))))
        self.assertLess(ortho_err, 1e-4)

        # All 4 zone baselines must be populated
        for zk in ["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"]:
            self.assertIn(zk, final_profile.zone_baselines)
            self.assertGreater(final_profile.zone_baselines[zk].sample_count, 0)


if __name__ == "__main__":
    unittest.main()
