"""
Unit tests for Advanced DMS Fatigue Metrics (Phase 6):
1. Delivery Post 1: Optalert JDS Amplitude-to-Velocity Ratio (AVR) & Inter-Blink Interval (CV_IBI).
2. Delivery Post 2: Euro NCAP 2026 Gaze Scanpath Entropy (H_t) & Dwell Time Limits.
3. Delivery Post 3: Facial Action Unit AU4 Brow Strain & Authentic Yawn Trapezoidal Filtering.
"""

from __future__ import annotations

import unittest
import numpy as np

from backend.app.ai.blink_kinematics import (
    BlinkClassification,
    BlinkEvent,
    BlinkState,
    TemporalKinematicsTracker,
)
from backend.app.ai.cascade_classifier import CascadeStageResult, ExperimentMode
from backend.app.ai.gaze_attention import GazeAttentionSnapshot, GazeAttentionTracker
from backend.app.ai.geometry import FrameGeometryFeatures, PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.ai.risk_engine import AlertState, PersonalisedAdaptiveRiskEngine


class MockLandmarker:
    """Configurable mock landmarker for deterministic multi-frame simulation."""

    def __init__(
        self,
        ear: float = 0.28,
        pitch: float = 0.0,
        yaw: float = 0.0,
        mar: float = 0.15,
        brow_down: float = 0.0,
        eye_squint: float = 0.0,
    ) -> None:
        self.ear = ear
        self.pitch = pitch
        self.yaw = yaw
        self.mar = mar
        self.brow_down = brow_down
        self.eye_squint = eye_squint

    def process_frame(self, frame_bgr, timestamp_ms=0.0, **kwargs):
        zone = PoseZone.CENTER
        if self.yaw > 15.0:
            zone = PoseZone.LEFT_MIRROR
        elif self.yaw < -15.0:
            zone = PoseZone.RIGHT_MIRROR
        elif self.pitch <= -8.0 and abs(self.yaw) < 10.0:
            zone = PoseZone.DASHBOARD

        features = FrameGeometryFeatures(
            timestamp_ms=timestamp_ms,
            face_detected=True,
            landmark_confidence=0.95,
            left_ear=self.ear,
            right_ear=self.ear,
            yaw_weighted_ear=self.ear,
            blendshape_blink_left=None,
            blendshape_blink_right=None,
            blendshape_blink_mean=None,
            mar=self.mar,
            pitch_deg=self.pitch,
            yaw_deg=self.yaw,
            roll_deg=0.0,
            pose_zone=zone,
            geometric_reliability=0.92,
            blendshape_brow_down=self.brow_down,
            blendshape_eye_squint=self.eye_squint,
        )
        return features, None


class TestAdvancedMetricsPhase6(unittest.TestCase):
    """Rigorous verification of the newly introduced advanced metric modules."""

    def setUp(self) -> None:
        self.tracker = TemporalKinematicsTracker(
            personalized_blink_median_ms=180.0,
            personalized_blink_mad_ms=30.0,
        )
        self.gaze_tracker = GazeAttentionTracker()
        self.risk_engine = PersonalisedAdaptiveRiskEngine(
            profile=DriverBehaviourProfile.create_default_uncalibrated(),
        )

    # --------------------------------------------------------------------------
    # 1. Delivery Post 1: Ocular Dynamics Upgrade (AVR & CV_IBI)
    # --------------------------------------------------------------------------

    def test_blink_amplitude_to_velocity_ratio_alert_vs_drowsy(self) -> None:
        """AVR must remain low during alert snappy blinks and surge during sluggish reopening."""
        # Simulate snappy alert blink (Duration 140ms: closed for 100ms, fast reopening)
        ts = 1000.0
        self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)
        ts += 33.3
        # Eye closes
        self.tracker.update(ts, ear_value=0.10, fused_closure_score=0.9, mar_value=0.15, ear_close_threshold=0.20)
        ts += 66.6
        self.tracker.update(ts, ear_value=0.10, fused_closure_score=0.9, mar_value=0.15, ear_close_threshold=0.20)
        ts += 40.0
        # Eye reopens quickly (t_open ~ 40ms, high v_open)
        snap = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)

        self.assertIsNotNone(snap.completed_blink)
        alert_blink = snap.completed_blink
        self.assertLess(alert_blink.avr, 0.15, f"Alert blink AVR should be swift (<0.15), got {alert_blink.avr}")

        # Simulate drowsy sluggish blink with impaired levator palpebrae reopening velocity
        ts = 5000.0
        self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)
        ts += 33.3
        # Closure to min EAR 0.08
        self.tracker.update(ts, ear_value=0.08, fused_closure_score=0.95, mar_value=0.15, ear_close_threshold=0.20)
        ts += 100.0  # Held closed
        self.tracker.update(ts, ear_value=0.08, fused_closure_score=0.95, mar_value=0.15, ear_close_threshold=0.20)
        ts += 100.0  # Sluggish reopening intermediate frame (below 250ms gap threshold)
        self.tracker.update(ts, ear_value=0.14, fused_closure_score=0.80, mar_value=0.15, ear_close_threshold=0.20)
        ts += 120.0  # Reopened fully
        snap_drowsy = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)

        self.assertIsNotNone(snap_drowsy.completed_blink)
        drowsy_blink = snap_drowsy.completed_blink
        self.assertGreater(drowsy_blink.avr, alert_blink.avr, "Drowsy blink AVR must exceed alert blink AVR")
        self.assertGreaterEqual(drowsy_blink.avr, 0.18, f"Drowsy blink AVR should surge >= 0.18, got {drowsy_blink.avr}")

    def test_inter_blink_interval_variability_cv_ibi(self) -> None:
        """Erratic staring spells followed by compensatory blink bursts must elevate CV_IBI."""
        ts = 1000.0
        # Three evenly spaced rhythmic blinks (IBI = 2000ms each, duration = 120ms)
        for _ in range(3):
            self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)
            ts += 40.0
            self.tracker.update(ts, ear_value=0.10, fused_closure_score=0.9, mar_value=0.15, ear_close_threshold=0.20)
            ts += 80.0
            self.tracker.update(ts, ear_value=0.10, fused_closure_score=0.9, mar_value=0.15, ear_close_threshold=0.20)
            ts += 40.0
            snap = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)
            ts += 1840.0

        self.assertIsNotNone(snap.blink_cv_ibi)
        self.assertLess(snap.blink_cv_ibi, 0.35, f"Rhythmic blinking should have low CV_IBI (<0.35), got {snap.blink_cv_ibi}")

        # Now introduce an erratic 15-second staring spell followed by rapid bursts
        ts += 15000.0
        for _ in range(2):
            self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)
            ts += 40.0
            self.tracker.update(ts, ear_value=0.10, fused_closure_score=0.9, mar_value=0.15, ear_close_threshold=0.20)
            ts += 80.0
            self.tracker.update(ts, ear_value=0.10, fused_closure_score=0.9, mar_value=0.15, ear_close_threshold=0.20)
            ts += 40.0
            snap_erratic = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.15, ear_close_threshold=0.20)
            ts += 300.0

        self.assertIsNotNone(snap_erratic.blink_cv_ibi)
        self.assertGreater(snap_erratic.blink_cv_ibi, 0.70, f"Erratic blinking should have elevated CV_IBI (>=0.70), got {snap_erratic.blink_cv_ibi}")

    # --------------------------------------------------------------------------
    # 2. Delivery Post 3: Neuromuscular Tone & Authentic Yawn Filtering
    # --------------------------------------------------------------------------

    def test_authentic_yawn_vs_speech_discrimination(self) -> None:
        """Brief mouth opening during normal conversation must NOT trigger a yawn alert."""
        # 1. Normal speech/talking: mouth opens to MAR=0.58 for 800ms, NO eye squint
        ts = 1000.0
        self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.20, ear_close_threshold=0.20, eye_squint=0.05)
        ts += 100.0
        for _ in range(6):
            snap = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.58, ear_close_threshold=0.20, eye_squint=0.05)
            ts += 100.0
        # Speech ends
        snap_after_speech = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.20, ear_close_threshold=0.20, eye_squint=0.05)

        self.assertEqual(len(self.tracker.yawn_history), 0, "Talking/Speech must NOT be recorded as a yawn event")
        self.assertEqual(snap_after_speech.yawn_count_5min, 0)
        self.assertFalse(snap_after_speech.is_yawning)

        # 2. Authentic physiological yawn: sustained for 3.2s with involuntary eye squint (AU6/AU43 coupling)
        ts = 5000.0
        for i in range(32):
            snap_yawn = self.tracker.update(
                ts,
                ear_value=0.22,  # Eye narrowing
                fused_closure_score=0.1,
                mar_value=0.62,  # Wide mouth opening
                ear_close_threshold=0.20,
                eye_squint=0.45,  # Involuntary squinting
            )
            ts += 100.0

        # Yawn completes
        snap_completed_yawn = self.tracker.update(ts, ear_value=0.28, fused_closure_score=0.0, mar_value=0.20, ear_close_threshold=0.20, eye_squint=0.05)

        self.assertEqual(len(self.tracker.yawn_history), 1, "Authentic yawn must be recorded in yawn history")
        self.assertEqual(snap_completed_yawn.yawn_count_5min, 1)
        self.assertTrue(self.tracker.yawn_history[0].is_authentic)
        self.assertGreaterEqual(self.tracker.yawn_history[0].peak_squint, 0.40)

    def test_facs_au4_brow_furrow_strain_reason_code(self) -> None:
        """Involuntary brow lowerer (AU4) activation during drooping eyelids flags brow strain."""
        geom = FrameGeometryFeatures(
            timestamp_ms=1000.0,
            face_detected=True,
            landmark_confidence=0.95,
            left_ear=0.21,
            right_ear=0.21,
            yaw_weighted_ear=0.21,
            blendshape_blink_left=None,
            blendshape_blink_right=None,
            blendshape_blink_mean=None,
            mar=0.15,
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            pose_zone=PoseZone.CENTER,
            geometric_reliability=0.90,
            blendshape_brow_down=0.65,  # Strong AU4 brow furrowing strain
            blendshape_eye_squint=0.10,
        )
        cascade = CascadeStageResult(
            experiment_mode=ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
            geom_closure_score=0.0,
            geom_reliability=0.90,
            disagreement_ear_vs_blendshape=0.0,
            cnn_invoked=False,
            gate_trigger_reason="CONFIDENT_GEOMETRY",
            winning_authority="GEOMETRY",
            stage_latency_ms=0.1,
            cnn_closed_prob=None,
            cnn_open_prob=None,
            cnn_reliability=None,
            fused_closure_score=0.0,
        )
        kinematics = self.tracker.update(
            timestamp_ms=1000.0,
            ear_value=0.21,
            fused_closure_score=0.0,
            mar_value=0.15,
            ear_close_threshold=0.20,
        )

        res = self.risk_engine.evaluate(geom=geom, cascade=cascade, kinematics=kinematics)
        self.assertIn("INVOLUNTARY_BROW_STRAIN_AU4", res.reasons)
        self.assertGreaterEqual(res.deviations.au4_brow_strain, 0.60)

    # --------------------------------------------------------------------------
    # 3. Delivery Post 2: Euro NCAP 2026 Visual Scanpath & Dwell Limits
    # --------------------------------------------------------------------------

    def test_gaze_entropy_active_vs_tunneling(self) -> None:
        """Active scanning across cockpit zones yields high entropy; rigid forward staring yields low entropy."""
        ts = 1000.0
        # Active driver scanning between Center and Mirrors
        zones_sequence = [
            PoseZone.CENTER, PoseZone.LEFT_MIRROR, PoseZone.CENTER,
            PoseZone.RIGHT_MIRROR, PoseZone.CENTER, PoseZone.DASHBOARD,
            PoseZone.CENTER, PoseZone.LEFT_MIRROR, PoseZone.CENTER,
        ]
        active_snap = None
        for z in zones_sequence:
            for _ in range(5):
                active_snap = self.gaze_tracker.update(ts, zone=z, pitch_deg=0.0, yaw_deg=0.0)
                ts += 200.0

        self.assertIsNotNone(active_snap)
        self.assertGreaterEqual(active_snap.transition_entropy_ht, 0.70, f"Active scanning should have H_t >= 0.70, got {active_snap.transition_entropy_ht}")

        # Now test cognitive tunneling / rigid frozen staring (only CENTER for prolonged duration)
        rigid_tracker = GazeAttentionTracker()
        ts = 50000.0
        rigid_snap = None
        for _ in range(100):
            rigid_snap = rigid_tracker.update(ts, zone=PoseZone.CENTER, pitch_deg=0.0, yaw_deg=0.0)
            ts += 100.0

        self.assertIsNotNone(rigid_snap)
        self.assertEqual(rigid_snap.prcf_pct, 100.0)
        self.assertEqual(rigid_snap.recent_zone_transitions_count, 0)

    def test_euro_ncap_off_road_dwell_and_blank_stare_limits(self) -> None:
        """Off-road dwell >= 3.0s flags distraction; rigid forward stare >= 5.0s flags blank stare."""
        ts = 1000.0
        # 1. Driver glances at Left Mirror for 2.0s (Normal mirror check <= 3.0s)
        for _ in range(10):
            snap_brief = self.gaze_tracker.update(ts, zone=PoseZone.LEFT_MIRROR, pitch_deg=0.0, yaw_deg=22.0)
            ts += 200.0
        self.assertFalse(snap_brief.is_off_road_dwell_exceeded)

        # Continues looking for another 1.2s (Total 3.2s > 3.0s Euro NCAP limit)
        for _ in range(6):
            snap_prolonged = self.gaze_tracker.update(ts, zone=PoseZone.LEFT_MIRROR, pitch_deg=0.0, yaw_deg=22.0)
            ts += 200.0
        self.assertTrue(snap_prolonged.is_off_road_dwell_exceeded, "Off-road gaze >= 3.0s must exceed Euro NCAP limit")

        # 2. Cognitive Tunneling / Blank Stare: rigid center stare with zero saccades for >= 5.0s
        self.gaze_tracker.reset()
        ts = 20000.0
        for _ in range(30):
            snap_stare = self.gaze_tracker.update(
                ts,
                zone=PoseZone.CENTER,
                pitch_deg=0.05,  # Virtually motionless head and eye posture
                yaw_deg=-0.05,
            )
            ts += 200.0  # 30 * 200ms = 6.0s

        self.assertTrue(snap_stare.is_blank_stare_detected, "Rigid forward stare >= 5.0s must flag cognitive tunneling / blank stare")

    # --------------------------------------------------------------------------
    # 4. End-to-End SessionAIEngine Pipeline Integration
    # --------------------------------------------------------------------------

    def test_session_ai_engine_emits_advanced_telemetry(self) -> None:
        """FrameTelemetryDTO emitted by SessionAIEngine must contain all new Phase 6 metrics."""
        mock_lm = MockLandmarker(ear=0.26, pitch=0.0, yaw=0.0, mar=0.15, brow_down=0.45, eye_squint=0.30)
        engine = SessionAIEngine(landmarker=mock_lm)

        frame = np.full((120, 160, 3), 100, dtype=np.uint8)
        dto = engine.process_frame(frame, timestamp_ms=1000.0)

        self.assertIsNotNone(dto)
        self.assertIsNotNone(dto.eye_metrics.gaze_entropy)
        self.assertIsNotNone(dto.eye_metrics.brow_furrow_au4)
        self.assertEqual(dto.eye_metrics.brow_furrow_au4, 0.45)
        self.assertIn(dto.state, (AlertState.NORMAL, AlertState.WARNING, AlertState.CRITICAL))


if __name__ == "__main__":
    unittest.main()
