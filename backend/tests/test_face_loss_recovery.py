"""
Unit Tests for Workstream C — Face-loss and Recovery State Transitions.

Verifies:
1. Transient single-frame face drops do not produce alert state bouncing.
2. Honest observation reporting: face_detected is False and fresh eye metrics are not fabricated.
3. Configurable grace period: sustained face loss transitions to MONITORING_UNAVAILABLE.
4. Temporal kinematics and gaze trackers do not accumulate invalid closure/dwell time across gaps.
5. Prior risk state and score are preserved during grace period rather than falsely claiming driver is safe.
6. Recovery restores MONITORING_ACTIVE cleanly.
"""

import numpy as np
import pytest

from backend.app.ai.geometry import (
    FrameGeometryFeatures,
    PoseZone,
    classify_pose_zone,
)
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.telemetry import AlertStateEnum, PoseZoneEnum


class ControllableMockLandmarker:
    """Mock landmarker allowing dynamic manipulation of face visibility and pose."""

    def __init__(
        self,
        face_detected: bool = True,
        ear: float = 0.28,
        pitch: float = 0.0,
        yaw: float = 0.0,
    ) -> None:
        self.face_detected = face_detected
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
        is_rgb: bool = True,
    ):
        if not self.face_detected:
            geom = FrameGeometryFeatures(
                timestamp_ms=timestamp_ms,
                face_detected=False,
                landmark_confidence=0.0,
                left_ear=0.0,
                right_ear=0.0,
                yaw_weighted_ear=0.0,
                blendshape_blink_left=0.0,
                blendshape_blink_right=0.0,
                blendshape_blink_mean=0.0,
                mar=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
                pose_zone=PoseZone.OUT_OF_BOUNDS,
                geometric_reliability=0.0,
            )
            return geom, None

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
            pose_zone=classify_pose_zone(self.pitch, self.yaw),
            geometric_reliability=0.92,
        )
        return geom, None


@pytest.fixture
def dummy_frame() -> np.ndarray:
    return np.zeros((120, 160, 3), dtype=np.uint8)


def test_single_frame_face_drop_does_not_bounce_state(dummy_frame):
    """A transient single-frame landmark drop must not bounce state to MONITORING_UNAVAILABLE."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=450.0)

    # Ingest 3 normal frames
    ts = 1000.0
    for _ in range(3):
        dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto.state == AlertStateEnum.NORMAL
        assert dto.face_detected is True
        assert dto.monitoring_status == "MONITORING_ACTIVE"
        ts += 33.3

    # Drop 1 frame
    mock_lm.face_detected = False
    dto_drop = engine.process_frame(dummy_frame, timestamp_ms=ts)

    # Honest observation reporting
    assert dto_drop.face_detected is False
    assert dto_drop.eye_metrics.ear_raw == 0.0
    assert dto_drop.eye_metrics.is_eye_closed is False
    assert dto_drop.eye_metrics.blink_state == "UNCERTAIN_GAP"

    # Availability vs Risk separation
    assert dto_drop.monitoring_status == "FACE_VISIBILITY_DEGRADED"
    assert dto_drop.state == AlertStateEnum.NORMAL  # Preserved! No bounce to UNAVAILABLE
    ts += 33.3

    # Next frame face returns
    mock_lm.face_detected = True
    dto_recover = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto_recover.face_detected is True
    assert dto_recover.monitoring_status == "MONITORING_ACTIVE"
    assert dto_recover.state == AlertStateEnum.NORMAL


def test_sustained_face_loss_exceeding_grace_period_transitions_to_unavailable(dummy_frame):
    """Face loss lasting >= face_loss_grace_ms must transition to MONITORING_UNAVAILABLE."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=450.0)

    # Establish baseline
    ts = 2000.0
    dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto.state == AlertStateEnum.NORMAL

    # Begin face loss at ts=2033
    mock_lm.face_detected = False
    ts += 33.3  # 2033.3 (loss start)

    # Ingest frames during grace period (dt = 300ms < 450ms)
    while ts <= 2400.0:
        dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto.face_detected is False
        assert dto.monitoring_status == "FACE_VISIBILITY_DEGRADED"
        assert dto.state == AlertStateEnum.NORMAL
        ts += 33.3

    # Now exceed grace period (ts = 2500.0 -> loss duration >= 466.7ms >= 450ms)
    ts = 2500.0
    dto_unavail = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto_unavail.face_detected is False
    assert dto_unavail.monitoring_status == "MONITORING_UNAVAILABLE"
    assert dto_unavail.state == AlertStateEnum.MONITORING_UNAVAILABLE


def test_recovery_from_sustained_face_loss_restores_monitoring(dummy_frame):
    """Recovery after sustained UNAVAILABLE transitions through RECOVERING to ACTIVE."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=300.0)

    # Establish baseline
    ts = 3000.0
    engine.process_frame(dummy_frame, timestamp_ms=ts)

    # Sustained face loss for 600ms
    mock_lm.face_detected = False
    ts += 33.3
    loss_start = ts
    while (ts - loss_start) < 600.0:
        dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
        ts += 33.3

    assert dto.monitoring_status == "MONITORING_UNAVAILABLE"
    assert dto.state == AlertStateEnum.MONITORING_UNAVAILABLE

    # Frame 1 of recovery: face reappears
    mock_lm.face_detected = True
    dto_rec1 = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto_rec1.face_detected is True
    assert dto_rec1.monitoring_status == "MONITORING_RECOVERING"
    ts += 33.3

    # Frame 2 of recovery: still confirming (consecutive count = 2)
    dto_rec2 = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto_rec2.face_detected is True
    assert dto_rec2.monitoring_status == "MONITORING_RECOVERING"
    ts += 33.3

    # Frame 3 of recovery: confirmed active (consecutive count = 3 >= REQUIRED_RECOVERY_FRAMES)
    dto_rec3 = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto_rec3.face_detected is True
    assert dto_rec3.monitoring_status == "MONITORING_ACTIVE"
    assert dto_rec3.state == AlertStateEnum.NORMAL


def test_gaze_dwell_timer_does_not_accumulate_across_face_loss_gap(dummy_frame):
    """Gaze dwell timers must not accumulate unobserved time across a face-loss gap."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28, pitch=-15.0, yaw=0.0)
    # pitch = -15 is PoseZone.DASHBOARD (off-road)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=400.0)

    # Driver looks at dashboard for 400ms
    ts = 4000.0
    for _ in range(12):
        dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
        assert dto.pose_zone == PoseZoneEnum.DASHBOARD
        ts += 33.3

    # Camera/face drops for 3000ms (3 seconds)
    mock_lm.face_detected = False
    gap_start = ts
    while (ts - gap_start) < 3000.0:
        engine.process_frame(dummy_frame, timestamp_ms=ts)
        ts += 100.0

    # Face returns at ts=7500 looking at DASHBOARD
    mock_lm.face_detected = True
    dto_after = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto_after.face_detected is True
    assert dto_after.pose_zone == PoseZoneEnum.DASHBOARD

    # Crucial check: dwell timer must NOT be 3400ms+ (which would trigger distraction alarm)!
    # It must have been reset/synchronized to the recovery timestamp.
    # Check that reasons do NOT report OFF_ROAD_DWELL_EXCEEDED
    assert "OFF_ROAD_DWELL_EXCEEDED" not in dto_after.reason_codes


def test_preserves_warning_risk_state_during_grace_period(dummy_frame):
    """Engine must preserve prior WARNING risk state during grace period without false safe claim."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.12)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=450.0)

    # Simulate drowsy closure for > 2.0s to satisfy 1800ms hysteresis requirement for WARNING
    ts = 5000.0
    warn_triggered = False
    for _ in range(70):  # 70 * 33.3ms ~ 2331ms > 1800ms
        dto = engine.process_frame(dummy_frame, timestamp_ms=ts)
        if dto.state in (AlertStateEnum.WARNING, AlertStateEnum.CRITICAL):
            warn_triggered = True
            break
        ts += 33.3

    assert warn_triggered, "Expected WARNING or CRITICAL state from sustained low EAR."
    last_state = dto.state
    last_risk = dto.risk_score
    assert last_risk > 0.0

    # Now drop face for 100ms (< 450ms grace period)
    mock_lm.face_detected = False
    ts += 33.3
    dto_drop = engine.process_frame(dummy_frame, timestamp_ms=ts)

    # Prior risk and state preserved!
    assert dto_drop.state == last_state
    assert dto_drop.risk_score == pytest.approx(last_risk, abs=1e-3)
    assert dto_drop.monitoring_status == "FACE_VISIBILITY_DEGRADED"


def test_configurable_grace_period(dummy_frame):
    """Engine must honor custom face_loss_grace_ms."""
    mock_lm = ControllableMockLandmarker(face_detected=True, ear=0.28)
    engine = SessionAIEngine(landmarker=mock_lm, face_loss_grace_ms=150.0)

    ts = 6000.0
    engine.process_frame(dummy_frame, timestamp_ms=ts)

    # Face lost at ts=6050
    mock_lm.face_detected = False
    ts += 50.0  # 6050ms (loss starts here)
    dto1 = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto1.monitoring_status == "FACE_VISIBILITY_DEGRADED"

    # Advance past 150ms grace period: 6050 + 200 = 6250ms (loss duration = 200ms > 150ms)
    ts += 200.0
    dto2 = engine.process_frame(dummy_frame, timestamp_ms=ts)
    assert dto2.monitoring_status == "MONITORING_UNAVAILABLE"
    assert dto2.state == AlertStateEnum.MONITORING_UNAVAILABLE
