"""
Phase 5 Automated Verification Tests:
Temporal Blink Kinematics, Sampling-Gap Awareness, & Reliable PERCLOS.

Tests cover:
1. Normal blink detection with sub-frame crossing interpolation, velocity, and RCR.
2. Duplicate blink prevention (each blink counted exactly once).
3. Prolonged blink detection (400-800ms) and classification.
4. Continuous eye closure and safety-critical microsleep candidate transition.
5. Asymmetric eye tracking (independent left/right EAR and closure states).
6. Sampling gaps exceeding 250ms (invalidation of in-flight blinks, UNCERTAIN_GAP state).
7. Face loss handling and clean recovery when face reappears without fabricated closure.
8. Reliable rolling PERCLOS with observation coverage % and low-coverage invalidation.
9. Rolling-window expiration and sliding interval management.
10. Repeated abnormal blink cluster detection.
11. Mirror-glance context safety overrides (microsleep overrides glance dampening).
12. Personalized driver baseline calibration integration for blink duration.
13. Session isolation (independent per-session trackers).

Run with:
    python -m unittest backend/tests/test_temporal_kinematics_phase5.py -v
"""

from __future__ import annotations

import unittest
import numpy as np

from backend.app.ai.blink_kinematics import (
    BlinkClassification,
    BlinkState,
    TemporalKinematicsTracker,
)
from backend.app.ai.cascade_classifier import (
    CascadeStageResult,
    ConfidenceGatedEyeCascade,
    ExperimentMode,
)
from backend.app.ai.geometry import (
    EyeCropQuality,
    FrameGeometryFeatures,
    PoseZone,
)
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile
from backend.app.ai.risk_engine import AlertState, PersonalisedAdaptiveRiskEngine


class TestTemporalKinematicsPhase5(unittest.TestCase):
    def test_normal_blink_detection_and_kinematics(self) -> None:
        """Verify normal blink detection, sub-frame crossing interpolation, and RCR."""
        tracker = TemporalKinematicsTracker(min_blink_duration_ms=50.0, prolonged_blink_duration_ms=400.0)

        # Baseline open eyes at 30 FPS (~33.3ms intervals)
        for i in range(5):
            snap = tracker.update(
                timestamp_ms=i * 33.3,
                ear_value=0.28,
                fused_closure_score=0.10,
                mar_value=0.12,
                ear_close_threshold=0.20,
            )
            self.assertEqual(snap.blink_state, BlinkState.OPEN)
            self.assertFalse(snap.is_eye_closed)
            self.assertIsNone(snap.completed_blink)

        # Eyes closing (t=166.5ms to t=266.4ms -> ~100ms blink)
        tracker.update(200.0, 0.16, 0.70, 0.12, 0.20)  # closed
        tracker.update(233.3, 0.12, 0.85, 0.12, 0.20)  # peak closure
        tracker.update(266.6, 0.15, 0.75, 0.12, 0.20)  # reopening

        # Eyes reopened at t=300ms
        snap = tracker.update(300.0, 0.28, 0.10, 0.12, 0.20)
        self.assertEqual(snap.blink_state, BlinkState.OPEN)
        self.assertIsNotNone(snap.completed_blink)

        blink = snap.completed_blink
        assert blink is not None
        self.assertGreater(blink.duration_ms, 50.0)
        self.assertLess(blink.duration_ms, 300.0)
        self.assertEqual(blink.classification, BlinkClassification.NORMAL)
        self.assertGreater(blink.closing_velocity_per_s, 0.0)
        self.assertGreater(blink.reopening_velocity_per_s, 0.0)
        self.assertGreater(blink.rcr, 0.0)

    def test_duplicate_blink_prevention(self) -> None:
        """A completed blink must only be emitted once and recorded in history once."""
        tracker = TemporalKinematicsTracker()

        # Open eyes
        tracker.update(0.0, 0.28, 0.10, 0.12, 0.20)
        tracker.update(33.3, 0.28, 0.10, 0.12, 0.20)

        # Closed (3 frames = ~100ms realistic blink)
        tracker.update(66.6, 0.14, 0.80, 0.12, 0.20)
        tracker.update(100.0, 0.12, 0.85, 0.12, 0.20)
        tracker.update(133.3, 0.14, 0.80, 0.12, 0.20)

        # Reopen (emits blink)
        snap1 = tracker.update(166.6, 0.28, 0.10, 0.12, 0.20)
        self.assertIsNotNone(snap1.completed_blink)
        self.assertEqual(len(tracker.blink_history), 1)

        # Subsequent open frame must NOT re-emit the completed blink
        snap2 = tracker.update(200.0, 0.28, 0.10, 0.12, 0.20)
        self.assertIsNone(snap2.completed_blink)
        self.assertEqual(len(tracker.blink_history), 1)

    def test_prolonged_blink_classification(self) -> None:
        """Blinks lasting between 400ms and 800ms must be classified as PROLONGED."""
        tracker = TemporalKinematicsTracker(prolonged_blink_duration_ms=400.0, microsleep_duration_ms=800.0)

        # Open
        tracker.update(0.0, 0.28, 0.10, 0.12, 0.20)

        # Closed for 500ms (t=50ms to t=550ms)
        ts = 50.0
        while ts <= 550.0:
            snap = tracker.update(ts, 0.14, 0.85, 0.12, 0.20)
            if ts >= 500.0:
                self.assertEqual(snap.blink_state, BlinkState.PROLONGED_CLOSURE)
            ts += 33.3

        # Reopen at t=600ms
        final_snap = tracker.update(600.0, 0.28, 0.10, 0.12, 0.20)
        self.assertIsNotNone(final_snap.completed_blink)
        blink = final_snap.completed_blink
        assert blink is not None
        self.assertGreaterEqual(blink.duration_ms, 400.0)
        self.assertLess(blink.duration_ms, 800.0)
        self.assertEqual(blink.classification, BlinkClassification.PROLONGED)

    def test_continuous_closure_and_microsleep_candidate(self) -> None:
        """Continuous closure >= 800ms transitions active classification to MICROSLEEP_CANDIDATE."""
        tracker = TemporalKinematicsTracker(microsleep_duration_ms=800.0)
        tracker.update(0.0, 0.28, 0.10, 0.12, 0.20)

        snap = None
        for i in range(1, 35):  # 35 frames * 33.3ms = ~1165ms of closure
            ts = i * 33.3
            snap = tracker.update(ts, 0.12, 0.90, 0.12, 0.20)

        assert snap is not None
        self.assertEqual(snap.blink_state, BlinkState.PROLONGED_CLOSURE)
        self.assertEqual(snap.blink_classification, BlinkClassification.MICROSLEEP_CANDIDATE)
        self.assertGreaterEqual(snap.current_closure_duration_ms, 800.0)

    def test_sampling_gap_invalidates_in_flight_blink(self) -> None:
        """
        If a sampling gap > 250ms occurs during closure, the in-flight blink must be
        invalidated to UNCERTAIN_GAP rather than fabricating a multi-second closure duration.
        """
        tracker = TemporalKinematicsTracker(max_sampling_gap_ms=250.0)

        # Open frame
        tracker.update(100.0, 0.28, 0.10, 0.12, 0.20)

        # Closed frame (enters closure at t=133.3ms)
        tracker.update(133.3, 0.14, 0.85, 0.12, 0.20)
        self.assertEqual(tracker._state, BlinkState.CLOSED)

        # Sudden gap of 800ms (next frame arrives at t=933.3ms)
        snap_after_gap = tracker.update(933.3, 0.28, 0.10, 0.12, 0.20)

        # Gap must be detected, in-flight blink must NOT be emitted as a 800ms blink!
        self.assertTrue(snap_after_gap.sampling_gap_detected)
        self.assertIsNone(snap_after_gap.completed_blink)
        self.assertEqual(len(tracker.blink_history), 0)

    def test_face_loss_handling_and_clean_recovery(self) -> None:
        """
        Face loss notification must cancel in-flight blinks and transition cleanly
        when face detection is restored.
        """
        tracker = TemporalKinematicsTracker()

        # Eyes closing at t=500ms
        tracker.update(500.0, 0.13, 0.80, 0.12, 0.20)
        self.assertEqual(tracker._state, BlinkState.CLOSED)

        # Face lost at t=550ms
        tracker.handle_face_lost(550.0)
        self.assertEqual(tracker._state, BlinkState.UNCERTAIN_GAP)

        # Face lost persists for 2 seconds...
        tracker.handle_face_lost(2550.0)
        self.assertEqual(tracker._state, BlinkState.UNCERTAIN_GAP)

        # Face reappears at t=2600ms with open eyes
        snap = tracker.update(2600.0, 0.28, 0.10, 0.12, 0.20)
        self.assertEqual(snap.blink_state, BlinkState.OPEN)
        self.assertEqual(snap.current_closure_duration_ms, 0.0)
        self.assertEqual(len(tracker.blink_history), 0)

    def test_independent_left_right_eye_tracking(self) -> None:
        """Verify left and right eyes are tracked independently without false binocular alerts."""
        tracker = TemporalKinematicsTracker()

        # Coordinated binocular open
        snap1 = tracker.update(
            timestamp_ms=100.0,
            ear_value=0.28,
            fused_closure_score=0.10,
            mar_value=0.12,
            ear_close_threshold=0.20,
            left_ear=0.28,
            right_ear=0.27,
            head_yaw_deg=0.0,
        )
        self.assertFalse(snap1.left_eye_closed)
        self.assertFalse(snap1.right_eye_closed)

        # Unilateral closure (e.g. left eye wink at 0.12, right eye open at 0.28)
        snap2 = tracker.update(
            timestamp_ms=133.3,
            ear_value=0.20,
            fused_closure_score=0.40,
            mar_value=0.12,
            ear_close_threshold=0.20,
            left_ear=0.12,
            right_ear=0.28,
            head_yaw_deg=0.0,
        )
        self.assertTrue(snap2.left_eye_closed)
        self.assertFalse(snap2.right_eye_closed)

    def test_reliable_rolling_perclos_and_coverage(self) -> None:
        """
        Verify PERCLOS is calculated as:
            (Observed Closed Duration / Valid Observed Duration) * 100
        and flags perclos_reliable=False when observation coverage < 50%.
        """
        tracker = TemporalKinematicsTracker(perclos_window_sec=60.0, min_perclos_coverage_pct=50.0)

        # Feed 15 seconds of open eyes (15,000ms / 60,000ms = 25% coverage)
        for i in range(150):
            ts = i * 100.0
            snap = tracker.update(ts, 0.28, 0.10, 0.12, 0.20)

        # Coverage is ~25%, which is < 50% minimum threshold
        self.assertLess(snap.perclos_coverage_pct, 50.0)
        self.assertFalse(snap.perclos_reliable)

        # Feed another 20 seconds, including 7 seconds of closed eyes
        # Total observed = 35s (> 50% of 60s)
        for i in range(150, 280):
            ts = i * 100.0
            tracker.update(ts, 0.28, 0.10, 0.12, 0.20)

        # 70 frames of closed eyes at 100ms = 7000ms closed
        for i in range(280, 350):
            ts = i * 100.0
            snap = tracker.update(ts, 0.12, 0.85, 0.12, 0.20)

        # Coverage is now >= 50%, so PERCLOS is reliable
        self.assertGreaterEqual(snap.perclos_coverage_pct, 50.0)
        self.assertTrue(snap.perclos_reliable)
        self.assertGreater(snap.perclos_60s_pct, 15.0)

    def test_repeated_abnormal_blink_cluster_detection(self) -> None:
        """Multiple consecutive prolonged blinks trigger abnormal cluster flag."""
        tracker = TemporalKinematicsTracker(prolonged_blink_duration_ms=400.0)

        # Baseline open
        tracker.update(0.0, 0.28, 0.10, 0.12, 0.20)

        # First prolonged blink (~466ms, 14 frames at 33.3ms)
        ts = 33.3
        for _ in range(14):
            tracker.update(ts, 0.14, 0.85, 0.12, 0.20)
            ts += 33.3
        tracker.update(ts, 0.28, 0.10, 0.12, 0.20)  # Reopened
        ts += 33.3

        # Brief open pause (5 frames)
        for _ in range(5):
            tracker.update(ts, 0.28, 0.10, 0.12, 0.20)
            ts += 33.3

        # Second prolonged blink (~466ms, 14 frames at 33.3ms)
        for _ in range(14):
            tracker.update(ts, 0.14, 0.85, 0.12, 0.20)
            ts += 33.3
        snap = tracker.update(ts, 0.28, 0.10, 0.12, 0.20)  # Reopened

        self.assertTrue(snap.abnormal_blink_cluster_detected)
        self.assertEqual(snap.blink_classification, BlinkClassification.REPEATED_ABNORMAL)

    def test_mirror_glance_safety_override(self) -> None:
        """
        Verify that mirror glance dampening protects against false positives
        during normal glance, but prolonged closure (microsleep >= 1.1s)
        immediately overrides mirror context and triggers CRITICAL alert.
        """
        profile = DriverBehaviourProfile.create_default_uncalibrated("mirror_safety_driver")
        engine = PersonalisedAdaptiveRiskEngine(profile=profile)
        cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.A_GEOMETRIC_ONLY)
        tracker = TemporalKinematicsTracker()

        dummy_crop = EyeCropQuality(
            crop_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            sharpness_score=0.9,
            exposure_score=0.8,
            glare_fraction=0.0,
            overall_reliability=0.9,
        )

        # 1. Normal mirror glance (2.0s with lower apparent EAR) -> NORMAL
        for i in range(40):
            ts = i * 50.0
            geom = FrameGeometryFeatures(
                timestamp_ms=ts,
                face_detected=True,
                landmark_confidence=0.92,
                left_ear=0.22,
                right_ear=0.26,
                yaw_weighted_ear=0.24,
                blendshape_blink_left=0.08,
                blendshape_blink_right=0.08,
                blendshape_blink_mean=0.08,
                mar=0.14,
                pitch_deg=0.0,
                yaw_deg=-20.0,  # Left mirror
                roll_deg=0.0,
                pose_zone=PoseZone.LEFT_MIRROR,
                geometric_reliability=0.85,
                left_eye_crop=dummy_crop,
                right_eye_crop=dummy_crop,
            )
            zb = profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(geom, zb.ear_median, zb.dynamic_closure_threshold)
            k_res = tracker.update(ts, geom.yaw_weighted_ear, c_res.fused_closure_score, geom.mar, zb.dynamic_closure_threshold)
            r_res = engine.evaluate(geom, c_res, k_res)
            self.assertEqual(r_res.state, AlertState.NORMAL)

        # 2. Prolonged eye closure (microsleep >= 1.1s) during mirror glance
        for i in range(40, 68):  # 28 frames * 50ms = 1400ms closure
            ts = i * 50.0
            geom = FrameGeometryFeatures(
                timestamp_ms=ts,
                face_detected=True,
                landmark_confidence=0.92,
                left_ear=0.10,
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
                left_eye_crop=dummy_crop,
                right_eye_crop=dummy_crop,
            )
            zb = profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(geom, zb.ear_median, zb.dynamic_closure_threshold)
            k_res = tracker.update(ts, geom.yaw_weighted_ear, c_res.fused_closure_score, geom.mar, zb.dynamic_closure_threshold)
            r_res = engine.evaluate(geom, c_res, k_res)

        # Must trigger CRITICAL with MICROSLEEP reason despite mirror zone
        self.assertEqual(r_res.state, AlertState.CRITICAL)
        self.assertIn("PROLONGED_EYE_CLOSURE_MICROSLEEP", r_res.reasons)

    def test_session_isolation(self) -> None:
        """Two separate tracker instances must maintain completely independent state."""
        tracker1 = TemporalKinematicsTracker()
        tracker2 = TemporalKinematicsTracker()

        # Session 1: closed eyes
        tracker1.update(100.0, 0.12, 0.85, 0.12, 0.20)
        self.assertEqual(tracker1._state, BlinkState.CLOSED)

        # Session 2: open eyes
        tracker2.update(100.0, 0.28, 0.10, 0.12, 0.20)
        self.assertEqual(tracker2._state, BlinkState.OPEN)

        self.assertNotEqual(tracker1._state, tracker2._state)

    def test_rolling_window_expiration(self) -> None:
        """Verify observation samples older than 60 seconds roll off the sliding window."""
        tracker = TemporalKinematicsTracker(perclos_window_sec=60.0)

        # Feed 70 seconds of data (from t=0s to t=70s at 100ms intervals = 701 frames)
        for i in range(701):
            ts = i * 100.0
            tracker.update(ts, 0.28, 0.10, 0.12, 0.20)

        # Oldest sample in _observation_samples must be >= 70,000 - 60,000 = 10,000ms
        self.assertGreaterEqual(tracker._observation_samples[0][0], 10000.0)
        valid_ms = sum(s[1] for s in tracker._observation_samples if s[2])
        self.assertAlmostEqual(valid_ms, 60000.0, delta=200.0)

    def test_calibrated_blink_baseline_integration(self) -> None:
        """Verify personalized blink duration median and MAD scale the prolonged classification cutoff."""
        tracker = TemporalKinematicsTracker(min_blink_duration_ms=50.0, prolonged_blink_duration_ms=400.0)

        # Personalized baseline with naturally longer blinks (median 300ms, MAD 60ms -> cutoff 450ms)
        tracker.set_personalized_baseline(median_duration_ms=300.0, mad_duration_ms=60.0)

        # Execute a 400ms blink (12 frames at 33.3ms)
        tracker.update(0.0, 0.28, 0.10, 0.12, 0.20)
        ts = 33.3
        for _ in range(12):
            tracker.update(ts, 0.14, 0.85, 0.12, 0.20)
            ts += 33.3
        tracker.update(ts, 0.28, 0.10, 0.12, 0.20)

        self.assertEqual(len(tracker.blink_history), 1)
        # With cutoff at 450ms, a 400ms blink is classified as NORMAL rather than PROLONGED
        self.assertEqual(tracker.blink_history[0].classification, BlinkClassification.NORMAL)

    def test_risk_engine_perclos_unreliable_and_sampling_gap_reasons(self) -> None:
        """
        Verify that:
        1. When perclos_reliable is False, PERCLOS does not contribute to risk score and
           INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE reason is emitted.
        2. When sampling gap is detected, OBSERVATION_SAMPLING_GAP_DETECTED is emitted.
        3. UNCERTAIN_GAP state is excluded from triggering microsleep.
        """
        profile = DriverBehaviourProfile.create_default_uncalibrated("test_driver")
        engine = PersonalisedAdaptiveRiskEngine(profile=profile)
        cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.A_GEOMETRIC_ONLY)
        tracker = TemporalKinematicsTracker()

        dummy_crop = EyeCropQuality(
            crop_rgb=np.zeros((32, 32, 3), dtype=np.uint8),
            sharpness_score=0.9,
            exposure_score=0.8,
            glare_fraction=0.0,
            overall_reliability=0.9,
        )

        geom = FrameGeometryFeatures(
            timestamp_ms=100.0,
            face_detected=True,
            landmark_confidence=0.92,
            left_ear=0.28,
            right_ear=0.28,
            yaw_weighted_ear=0.28,
            blendshape_blink_left=0.08,
            blendshape_blink_right=0.08,
            blendshape_blink_mean=0.08,
            mar=0.14,
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            pose_zone=PoseZone.CENTER,
            geometric_reliability=0.90,
            left_eye_crop=dummy_crop,
            right_eye_crop=dummy_crop,
        )
        c_res = cascade.evaluate(geom, 0.28, 0.20)

        # 1. Low observation coverage (<50%): perclos_reliable is False
        k_snap = tracker.update(100.0, 0.28, 0.10, 0.12, 0.20)
        self.assertFalse(k_snap.perclos_reliable)
        r_res = engine.evaluate(geom, c_res, k_snap)
        self.assertIn("INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE", r_res.reasons)

        # 2. Trigger a sampling gap > 250ms
        k_snap_gap = tracker.update(500.0, 0.28, 0.10, 0.12, 0.20)
        self.assertTrue(k_snap_gap.sampling_gap_detected)
        r_res_gap = engine.evaluate(geom, c_res, k_snap_gap)
        self.assertIn("OBSERVATION_SAMPLING_GAP_DETECTED", r_res_gap.reasons)

        # 3. UNCERTAIN_GAP state must not trigger microsleep override
        tracker._state = BlinkState.UNCERTAIN_GAP
        k_snap_uncertain = tracker.update(533.3, 0.12, 0.85, 0.12, 0.20)
        r_res_uncertain = engine.evaluate(geom, c_res, k_snap_uncertain)
        self.assertNotIn("PROLONGED_EYE_CLOSURE_MICROSLEEP", r_res_uncertain.reasons)


if __name__ == "__main__":
    unittest.main()
