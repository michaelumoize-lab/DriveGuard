"""
Tests for AI Integrity, Monitoring Availability, and Provenance (Phase 2: DG-02, DG-06).

Verifies:
1. DG-02: Elimination of fabricated face detections on blank frames, random noise,
   uninitialized models, and headless degraded fallbacks.
2. DG-06: Robust face-loss state handling, timestamp-based grace period buffering,
   preservation of risk scores without accumulating false eye-closure durations,
   dwell timer freezing, and 3-frame recovery hysteresis.
"""

import time
import numpy as np
import pytest

from backend.app.ai.geometry import PoseZone
from backend.app.ai.landmarker import DriveGuardFaceLandmarker
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.ai.gaze_attention import GazeAttentionTracker
from backend.app.schemas.telemetry import AlertStateEnum


class TestLandmarkerIntegrityDG02:
    """DG-02: Verification that landmarker never fabricates driver presence."""

    def test_blank_frame_reports_no_face(self) -> None:
        """Blank (all black) frames must report face_detected=False, 0.0 confidence, 0.0 reliability."""
        landmarker = DriveGuardFaceLandmarker(running_mode="IMAGE")
        blank_frame = np.zeros((480, 640, 3), dtype=np.uint8)

        geom, landmarks = landmarker.process_frame(frame_rgb=blank_frame, timestamp_ms=100.0)

        assert not geom.face_detected, "Blank frame falsely reported face_detected=True"
        assert geom.landmark_confidence == 0.0
        assert geom.geometric_reliability == 0.0
        assert geom.left_eye_crop is None
        assert geom.right_eye_crop is None
        assert geom.pose_zone == PoseZone.OUT_OF_BOUNDS
        assert landmarks is None
        landmarker.close()

    def test_random_noise_frame_reports_no_face(self) -> None:
        """Random uniform noise frames must not trigger false positive face detection."""
        landmarker = DriveGuardFaceLandmarker(running_mode="IMAGE")
        noise_frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

        geom, landmarks = landmarker.process_frame(frame_rgb=noise_frame, timestamp_ms=200.0)

        assert not geom.face_detected, "Noise frame falsely reported face_detected=True"
        assert geom.landmark_confidence == 0.0
        assert geom.geometric_reliability == 0.0
        assert landmarks is None
        landmarker.close()

    def test_degraded_fallback_mode_never_fabricates_face(self) -> None:
        """
        When landmarker is forced into DEGRADED_UNAVAILABLE fallback mode,
        it must explicitly report face_detected=False and tracking_mode=DEGRADED_UNAVAILABLE.
        """
        landmarker = DriveGuardFaceLandmarker(running_mode="IMAGE")
        # Explicitly simulate missing/failed MediaPipe initialization
        landmarker.active_implementation = "DEGRADED_UNAVAILABLE"
        landmarker._use_fallback_tracker = True

        diag = landmarker.get_diagnostics()
        assert diag["active_implementation"] == "DEGRADED_UNAVAILABLE"

        dummy_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        geom, landmarks = landmarker.process_frame(frame_rgb=dummy_frame, timestamp_ms=300.0)

        assert not geom.face_detected, "Degraded fallback mode falsely reported face_detected=True"
        assert geom.landmark_confidence == 0.0
        assert geom.geometric_reliability == 0.0
        assert geom.tracking_mode == "DEGRADED_UNAVAILABLE"
        assert geom.left_eye_crop is None
        assert geom.right_eye_crop is None
        landmarker.close()


class TestFaceLossGracePeriodAndRecoveryDG06:
    """DG-06: Verification of face-loss state machine, grace period, and recovery."""

    def test_grace_period_buffers_brief_dropout(self) -> None:
        """
        When face tracking drops for < face_loss_grace_ms:
        - monitoring_status is FACE_VISIBILITY_DEGRADED
        - grace_period_active is True
        - face_visible is False
        - last valid risk score and alert state are preserved
        - eye_closure_duration does NOT accumulate false closure
        """
        engine = SessionAIEngine(
            session_id="test_grace_sess",
            driver_id="driver_test",
            face_loss_grace_ms=500.0,
        )

        # 1. Establish normal monitoring with a simulated detected face
        class MockGeomLandmarker:
            def __init__(self, detected: bool):
                self.detected = detected
                self.active_implementation = "MOCK_TASKS"

            def process_frame(self, **kwargs):
                from backend.app.ai.geometry import FrameGeometryFeatures, PoseZone
                if self.detected:
                    return (
                        FrameGeometryFeatures(
                            timestamp_ms=kwargs.get("timestamp_ms", 0.0),
                            face_detected=True,
                            landmark_confidence=0.95,
                            left_ear=0.28,
                            right_ear=0.28,
                            yaw_weighted_ear=0.28,
                            blendshape_blink_left=0.05,
                            blendshape_blink_right=0.05,
                            blendshape_blink_mean=0.05,
                            mar=0.12,
                            pitch_deg=0.0,
                            yaw_deg=0.0,
                            roll_deg=0.0,
                            pose_zone=PoseZone.CENTER,
                            geometric_reliability=0.90,
                            tracking_mode="MOCK_TASKS",
                        ),
                        None,
                    )
                else:
                    return (
                        FrameGeometryFeatures(
                            timestamp_ms=kwargs.get("timestamp_ms", 0.0),
                            face_detected=False,
                            landmark_confidence=0.0,
                            left_ear=0.0,
                            right_ear=0.0,
                            yaw_weighted_ear=0.0,
                            blendshape_blink_left=None,
                            blendshape_blink_right=None,
                            blendshape_blink_mean=None,
                            mar=0.0,
                            pitch_deg=0.0,
                            yaw_deg=0.0,
                            roll_deg=0.0,
                            pose_zone=PoseZone.OUT_OF_BOUNDS,
                            geometric_reliability=0.0,
                            tracking_mode="MOCK_TASKS",
                        ),
                        None,
                    )

        mock_landmarker = MockGeomLandmarker(detected=True)
        engine.landmarker = mock_landmarker

        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        # Send valid frames at 1000ms and 1033ms
        t1 = engine.process_frame(frame_rgb, timestamp_ms=1000.0)
        assert t1.face_detected is True
        assert t1.face_visible is True
        assert t1.monitoring_status == "MONITORING_ACTIVE"
        assert t1.grace_period_active is False
        assert t1.last_seen_ms == 1000.0
        initial_risk = t1.risk_score
        initial_state = t1.state

        t2 = engine.process_frame(frame_rgb, timestamp_ms=1033.0)
        assert t2.monitoring_status == "MONITORING_ACTIVE"
        assert t2.last_seen_ms == 1033.0

        # 2. Simulate camera dropout (face loss) at 1200ms (167ms after last seen, < 500ms grace)
        mock_landmarker.detected = False
        t3 = engine.process_frame(frame_rgb, timestamp_ms=1200.0)

        assert t3.face_detected is False
        assert t3.face_visible is False
        assert t3.grace_period_active is True
        assert t3.monitoring_status == "FACE_VISIBILITY_DEGRADED"
        assert t3.risk_score == initial_risk
        assert t3.state == initial_state
        assert t3.last_seen_ms == 1033.0
        # Crucial: eye closure duration must NOT accumulate during face loss
        assert t3.eye_metrics.eye_closure_duration_ms == 0.0
        assert t3.eye_metrics.is_eye_closed is False

        # Another frame at 1350ms (317ms after last seen, still < 500ms grace)
        t4 = engine.process_frame(frame_rgb, timestamp_ms=1350.0)
        assert t4.grace_period_active is True
        assert t4.monitoring_status == "FACE_VISIBILITY_DEGRADED"
        assert t4.eye_metrics.eye_closure_duration_ms == 0.0

        # 3. Exceed grace period at 1600ms (567ms after last seen >= 500ms grace)
        t5 = engine.process_frame(frame_rgb, timestamp_ms=1600.0)
        assert t5.grace_period_active is False
        assert t5.monitoring_status == "MONITORING_UNAVAILABLE"
        assert t5.state == AlertStateEnum.MONITORING_UNAVAILABLE
        assert "NO_FACE_DETECTED" in t5.reason_codes

        engine.close()

    def test_sustained_outage_requires_three_consecutive_frames_to_recover(self) -> None:
        """
        After sustained face loss (MONITORING_UNAVAILABLE), full monitoring is not restored
        on a single frame. It must transition through MONITORING_RECOVERING for 2 frames
        and achieve MONITORING_ACTIVE on the 3rd consecutive frame.
        """
        engine = SessionAIEngine(
            session_id="test_recovery_sess",
            driver_id="driver_test",
            face_loss_grace_ms=450.0,
        )

        class MockLandmarker:
            def __init__(self):
                self.detected = True
                self.active_implementation = "MOCK_TASKS"

            def process_frame(self, **kwargs):
                from backend.app.ai.geometry import FrameGeometryFeatures, PoseZone
                if self.detected:
                    return (
                        FrameGeometryFeatures(
                            timestamp_ms=kwargs.get("timestamp_ms", 0.0),
                            face_detected=True,
                            landmark_confidence=0.95,
                            left_ear=0.28,
                            right_ear=0.28,
                            yaw_weighted_ear=0.28,
                            blendshape_blink_left=0.05,
                            blendshape_blink_right=0.05,
                            blendshape_blink_mean=0.05,
                            mar=0.12,
                            pitch_deg=0.0,
                            yaw_deg=0.0,
                            roll_deg=0.0,
                            pose_zone=PoseZone.CENTER,
                            geometric_reliability=0.90,
                            tracking_mode="MOCK_TASKS",
                        ),
                        None,
                    )
                else:
                    return (
                        FrameGeometryFeatures(
                            timestamp_ms=kwargs.get("timestamp_ms", 0.0),
                            face_detected=False,
                            landmark_confidence=0.0,
                            left_ear=0.0,
                            right_ear=0.0,
                            yaw_weighted_ear=0.0,
                            blendshape_blink_left=None,
                            blendshape_blink_right=None,
                            blendshape_blink_mean=None,
                            mar=0.0,
                            pitch_deg=0.0,
                            yaw_deg=0.0,
                            roll_deg=0.0,
                            pose_zone=PoseZone.OUT_OF_BOUNDS,
                            geometric_reliability=0.0,
                            tracking_mode="MOCK_TASKS",
                        ),
                        None,
                    )

        mock = MockLandmarker()
        engine.landmarker = mock
        frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)

        # Step 1: establish active monitoring
        engine.process_frame(frame_rgb, timestamp_ms=1000.0)
        engine.process_frame(frame_rgb, timestamp_ms=1033.0)

        # Step 2: sustained outage (> 500ms)
        mock.detected = False
        t_drop1 = engine.process_frame(frame_rgb, timestamp_ms=1100.0)
        assert t_drop1.monitoring_status == "FACE_VISIBILITY_DEGRADED"

        t_drop2 = engine.process_frame(frame_rgb, timestamp_ms=1700.0)
        assert t_drop2.monitoring_status == "MONITORING_UNAVAILABLE"

        # Step 3: Face reappears at 3000ms
        mock.detected = True

        # Recovery Frame 1: must be MONITORING_RECOVERING
        rec1 = engine.process_frame(frame_rgb, timestamp_ms=3000.0)
        assert rec1.face_detected is True
        assert rec1.monitoring_status == "MONITORING_RECOVERING"

        # Recovery Frame 2: must still be MONITORING_RECOVERING
        rec2 = engine.process_frame(frame_rgb, timestamp_ms=3033.0)
        assert rec2.face_detected is True
        assert rec2.monitoring_status == "MONITORING_RECOVERING"

        # Recovery Frame 3: threshold achieved -> MONITORING_ACTIVE
        rec3 = engine.process_frame(frame_rgb, timestamp_ms=3066.0)
        assert rec3.face_detected is True
        assert rec3.monitoring_status == "MONITORING_ACTIVE"

        engine.close()

    def test_gaze_dwell_freezes_during_grace_and_does_not_fire_false_distraction(self) -> None:
        """
        Verify that GazeAttentionTracker freezes dwell accumulation during grace period
        and does not trigger spurious off-road distraction alarms.
        """
        tracker = GazeAttentionTracker(off_road_dwell_threshold_ms=2000.0)

        # 1. Driver glances at LEFT_MIRROR for 1000ms (1000 to 2000ms)
        tracker.update(timestamp_ms=1000.0, zone=PoseZone.LEFT_MIRROR)
        s1 = tracker.update(timestamp_ms=2000.0, zone=PoseZone.LEFT_MIRROR)
        assert s1.current_zone_dwell_ms == 1000.0
        assert not s1.is_off_road_dwell_exceeded

        # 2. Camera tracking dropout occurs between 2000ms and 2400ms (< 500ms grace)
        tracker.freeze(timestamp_ms=2000.0)

        # 3. Camera tracking recovers at 2400ms: dwell accumulation resumes from 1000ms, NOT 1400ms
        s2 = tracker.update(timestamp_ms=2400.0, zone=PoseZone.LEFT_MIRROR)
        assert s2.current_zone_dwell_ms == 1000.0
        assert not s2.is_off_road_dwell_exceeded

        # 4. Driver returns to center at 2600ms
        s3 = tracker.update(timestamp_ms=2600.0, zone=PoseZone.CENTER)
        assert s3.current_zone == PoseZone.CENTER
        assert not s3.is_off_road_dwell_exceeded
