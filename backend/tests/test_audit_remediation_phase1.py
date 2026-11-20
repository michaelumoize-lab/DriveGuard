"""
DriveGuard — Core System Audit Remediation Phase 1 Regression Tests.

Covers:
  - CRITICAL-1: Separate ocular-channel viability from scalar global confidence reliability.
    Verifies sustained eye closure with head-pose failure + mild blur triggers CRITICAL_WARNING
    (never false-safe NORMAL), while severely degraded optical frames enter INSUFFICIENT_DATA.
  - CRITICAL-2: Schmitt-trigger blink & prolonged-closure hysteresis.
    Reproduces the 1,400 ms closure + noisy intermediate EAR (c + 0.001 < reopen_threshold)
    scenario across BlinkKinematicsDetector, DriverFeatureExtractor, TemporalAnalyzer, and
    DrowsinessEngine, proving closure start is preserved and CRITICAL_WARNING triggers.
  - HIGH-3: DecisionEngine Timer 1 & AlertStateManager recovery synchronization.
    Verifies closed-eye reacquisition after prolonged face loss preserves CRITICAL without
    falsely marking is_recovering=True or suppressing periodic CRITICAL reminders, and
    verifies eye occlusion cannot prematurely clear ocular ATTENTION / CRITICAL alerts.
"""

from __future__ import annotations

import pytest

from backend.app.alerts.alert_dispatcher import AlertDispatcher
from backend.app.alerts.alert_state import AlertState
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.alerts.audio_settings import AudioSettingsStore
from backend.app.alerts.audio_warning_engine import AudioWarningEngine, MockAudioBackend
from backend.app.alerts.event_logger import AlertEventLogger
from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityResult,
    LightingCondition,
    QualityStatus,
)
from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult
from backend.app.features.perclos import PerclosResult
from backend.app.intelligence.confidence import ConfidenceAssessment, ConfidenceTier
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalMetrics
from backend.app.vision.face_landmarker import FaceLandmarkStatus


def _make_features(
    timestamp_ms: float,
    *,
    has_face: bool = True,
    face_status: FaceLandmarkStatus = FaceLandmarkStatus.TRACKING,
    ear_val: float = 0.28,
    ear_is_valid: bool = True,
    is_closed: bool = False,
    pose_is_valid: bool = True,
    zone: PoseZone = PoseZone.CENTER,
    blink_state: BlinkState = BlinkState.OPEN,
    blink_closed: bool = False,
    closure_dur_ms: float = 0.0,
    perclos_pct: float = 2.0,
    perclos_is_reliable: bool = False,
) -> DriverFeatures:
    return DriverFeatures(
        timestamp_ms=timestamp_ms,
        has_face=has_face,
        face_status=face_status if has_face else FaceLandmarkStatus.INSUFFICIENT_DATA,
        ear=EyeAspectRatioResult(
            left_ear=ear_val,
            right_ear=ear_val,
            weighted_ear=ear_val,
            compensated_ear=ear_val,
            closure_threshold=0.20,
            is_left_closed=is_closed,
            is_right_closed=is_closed,
            is_closed=is_closed,
            is_valid=ear_is_valid,
            raw_ear=ear_val,
            is_left_valid=ear_is_valid,
            is_right_valid=ear_is_valid,
            is_geometric_valid=ear_is_valid,
        ),
        head_pose=HeadPoseResult(
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            zone=zone if pose_is_valid else PoseZone.UNKNOWN,
            is_looking_away=False,
            is_valid=pose_is_valid,
        ),
        mar=MouthAspectRatioResult(
            mar=0.15,
            is_mouth_open=False,
            is_yawn_in_progress=False,
            current_yawn_duration_ms=0.0,
            is_authentic_yawn=False,
            completed_yawn=None,
            total_yawn_count=0,
            is_valid=has_face,
        ),
        blink=BlinkSnapshot(
            state=blink_state,
            is_eye_closed=blink_closed,
            current_closure_duration_ms=closure_dur_ms,
            completed_blink=None,
            recent_blink_duration_ms=None,
            recent_blink_rcr=None,
            total_blink_count=0,
            prolonged_blink_count=0,
        ),
        perclos=PerclosResult(
            perclos_pct=perclos_pct,
            coverage_pct=100.0 if perclos_is_reliable else 10.0,
            is_reliable=perclos_is_reliable,
            closed_duration_sec=0.0,
            observed_duration_sec=60.0 if perclos_is_reliable else 6.0,
            window_sec=60.0,
        ),
    )


# ==============================================================================
# CRITICAL-1: Ocular Viability Decoupled from Global Confidence Reliability
# ==============================================================================


def test_critical1_head_pose_failure_and_mild_blur_still_detects_microsleep() -> None:
    """
    Reproduces audit CRITICAL-1:
    head_pose.is_valid=False (0.70) + QualityStatus.BLURRY (0.70, viable=True) ->
    overall_confidence=0.49 < 0.50 (is_reliable=False), while ocular_viable=True.
    Sustained 2,000 ms eye closure MUST trigger CRITICAL_WARNING (never NORMAL).
    """
    engine = DrowsinessEngine()
    mild_blur = FrameQualityResult(
        mean_luminance=120.0,
        blur_variance=50.0,
        lighting=LightingCondition.OPTIMAL,
        blur=BlurCondition.BLURRY,
        status=QualityStatus.BLURRY,
        is_viable_for_inference=True,
        reasons=("Mild motion blur",),
    )

    assessment = None
    for t in range(0, 2100, 100):
        features = _make_features(
            timestamp_ms=float(t),
            ear_val=0.10,
            ear_is_valid=True,
            is_closed=True,
            pose_is_valid=False,
            blink_state=BlinkState.PROLONGED_CLOSURE if t >= 500 else BlinkState.CLOSED,
            blink_closed=True,
            closure_dur_ms=float(t),
        )
        assessment = engine.evaluate(features, quality=mild_blur)

    assert assessment is not None
    assert assessment.confidence.overall_confidence == pytest.approx(0.49, abs=1e-4)
    assert assessment.confidence.is_reliable is False
    assert assessment.confidence.ocular_viable is True
    assert assessment.temporal_metrics.active_closure_duration_ms >= 1500.0
    assert assessment.state == DrowsinessState.CRITICAL_WARNING
    assert assessment.primary_reason is not None
    assert assessment.primary_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE


def test_critical1_marginal_face_and_poor_lighting_still_tracks_eye_closure() -> None:
    """
    OCCLUDED_OR_MARGINAL (0.50) + POOR_LIGHTING (0.75) -> overall_confidence=0.375 (< 0.50),
    yet ocular_viable=True; sustained eye closure must escalate to CRITICAL_WARNING.
    """
    engine = DrowsinessEngine()
    poor_light = FrameQualityResult(
        mean_luminance=28.0,
        blur_variance=120.0,
        lighting=LightingCondition.UNDEREXPOSED,
        blur=BlurCondition.SHARP,
        status=QualityStatus.POOR_LIGHTING,
        is_viable_for_inference=True,
        reasons=("Dim cabin lighting",),
    )

    assessment = None
    for t in range(0, 1800, 100):
        features = _make_features(
            timestamp_ms=float(t),
            face_status=FaceLandmarkStatus.OCCLUDED_OR_MARGINAL,
            ear_val=0.11,
            ear_is_valid=True,
            is_closed=True,
            pose_is_valid=True,
            blink_state=BlinkState.PROLONGED_CLOSURE if t >= 500 else BlinkState.CLOSED,
            blink_closed=True,
            closure_dur_ms=float(t),
        )
        assessment = engine.evaluate(features, quality=poor_light)

    assert assessment is not None
    assert assessment.confidence.is_reliable is False
    assert assessment.confidence.ocular_viable is True
    assert assessment.state == DrowsinessState.CRITICAL_WARNING


def test_critical1_severely_degraded_optical_still_enters_insufficient_data() -> None:
    """
    Severely degraded optical quality (is_viable_for_inference=False) disables both
    ocular_viable and head_pose_viable, transitioning to INSUFFICIENT_DATA after grace.
    """
    engine = DrowsinessEngine()
    pitch_black = FrameQualityResult(
        mean_luminance=5.0,
        blur_variance=5.0,
        lighting=LightingCondition.UNDEREXPOSED,
        blur=BlurCondition.BLURRY,
        status=QualityStatus.SEVERELY_DEGRADED,
        is_viable_for_inference=False,
        reasons=("Pitch black frame",),
    )

    assessment = None
    for t in range(0, 1300, 100):
        features = _make_features(
            timestamp_ms=float(t),
            ear_val=0.10,
            ear_is_valid=True,
            is_closed=True,
            pose_is_valid=True,
            blink_state=BlinkState.CLOSED,
            blink_closed=True,
            closure_dur_ms=float(t),
        )
        assessment = engine.evaluate(features, quality=pitch_black)

    assert assessment is not None
    assert assessment.confidence.ocular_viable is False
    assert assessment.confidence.head_pose_viable is False
    assert assessment.state == DrowsinessState.INSUFFICIENT_DATA


# ==============================================================================
# CRITICAL-2: Blink & Prolonged-Closure Hysteresis Under Noisy EAR
# ==============================================================================


def test_critical2_noisy_intermediate_ear_does_not_reset_1400ms_prolonged_closure() -> None:
    """
    Reproduces audit CRITICAL-2:
    1,400 ms eye closure interrupted by a single noisy frame at ear=0.201
    (barely above closure_threshold=0.20, still well below reopen_threshold=0.24)
    followed by ear=0.14 must preserve original closure onset and reach CRITICAL_WARNING.
    """
    detector = BlinkKinematicsDetector(
        closure_threshold=0.20,
        reopen_threshold=0.24,
        prolonged_duration_ms=500.0,
        microsleep_duration_ms=800.0,
    )
    engine = DrowsinessEngine(sustained_closure_threshold_ms=1500.0)

    # Frame 0: eyes open
    s0 = detector.update(0.30, 0.30, ear=0.30, timestamp_ms=0.0)
    engine.evaluate(
        _make_features(
            0.0,
            ear_val=0.30,
            is_closed=False,
            blink_state=s0.state,
            blink_closed=s0.is_eye_closed,
            closure_dur_ms=s0.current_closure_duration_ms,
        )
    )

    # Frames 100..1400 ms: eyes closed (EAR = 0.14)
    for t in range(100, 1500, 100):
        s = detector.update(0.14, 0.14, ear=0.14, timestamp_ms=float(t))
        a = engine.evaluate(
            _make_features(
                float(t),
                ear_val=0.14,
                is_closed=True,
                blink_state=s.state,
                blink_closed=s.is_eye_closed,
                closure_dur_ms=s.current_closure_duration_ms,
            )
        )
    assert s.state == BlinkState.PROLONGED_CLOSURE
    assert s.current_closure_duration_ms > 1300.0

    # Noisy frame at t=1500 ms: EAR bounces to 0.201 (above c=0.20, below r=0.24)
    s_jit = detector.update(0.201, 0.201, ear=0.201, timestamp_ms=1500.0)
    assert s_jit.state == BlinkState.REOPENING
    assert s_jit.is_eye_closed is True
    assert s_jit.current_closure_duration_ms > 1400.0

    a_jit = engine.evaluate(
        _make_features(
            1500.0,
            ear_val=0.201,
            is_closed=False,  # raw ear.py has no hysteresis
            blink_state=s_jit.state,
            blink_closed=s_jit.is_eye_closed,
            closure_dur_ms=s_jit.current_closure_duration_ms,
        )
    )
    assert a_jit.temporal_metrics.active_closure_duration_ms > 1400.0

    # t=1600 ms: EAR drops back to 0.14 (< c=0.20) -> MUST NOT reset to ~33 ms!
    s_back = detector.update(0.14, 0.14, ear=0.14, timestamp_ms=1600.0)
    assert s_back.state == BlinkState.PROLONGED_CLOSURE
    assert s_back.is_eye_closed is True
    assert s_back.current_closure_duration_ms >= 1500.0

    a_back = engine.evaluate(
        _make_features(
            1600.0,
            ear_val=0.14,
            is_closed=True,
            blink_state=s_back.state,
            blink_closed=s_back.is_eye_closed,
            closure_dur_ms=s_back.current_closure_duration_ms,
        )
    )
    assert a_back.temporal_metrics.active_closure_duration_ms >= 1500.0
    assert a_back.state == DrowsinessState.CRITICAL_WARNING
    assert a_back.primary_reason is not None
    assert a_back.primary_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE


def test_critical2_genuine_reopen_above_reopen_threshold_completes_blink() -> None:
    """Crossing reopen_threshold cleanly ends closure and emits completed blink."""
    detector = BlinkKinematicsDetector(closure_threshold=0.20, reopen_threshold=0.24)
    detector.update(0.30, 0.30, ear=0.30, timestamp_ms=0.0)
    detector.update(0.14, 0.14, ear=0.14, timestamp_ms=100.0)
    s_mid = detector.update(0.22, 0.22, ear=0.22, timestamp_ms=150.0)
    assert s_mid.state == BlinkState.REOPENING
    assert s_mid.is_eye_closed is True

    s_open = detector.update(0.28, 0.28, ear=0.28, timestamp_ms=200.0)
    assert s_open.state == BlinkState.OPEN
    assert s_open.is_eye_closed is False
    assert s_open.current_closure_duration_ms == 0.0
    assert s_open.completed_blink is not None
    assert s_open.completed_blink.classification == BlinkClassification.NORMAL


# ==============================================================================
# HIGH-3: DecisionEngine & AlertStateManager Recovery Synchronization
# ==============================================================================


def test_high3_closed_eye_reacquisition_after_face_loss_does_not_mark_recovering(tmp_path) -> None:
    """
    Reproduces Scenario L / audit HIGH-3:
    CRITICAL_WARNING -> prolonged face loss (INSUFFICIENT_DATA) -> face reappears with EYES CLOSED.
    - DecisionEngine.state must remain CRITICAL_WARNING (not drop to NORMAL).
    - DecisionEngine.is_alert_recovering must be False.
    - AlertStateManager.current_state must remain CRITICAL with is_recovering=False.
    - Periodic CRITICAL audio reminders must NOT be suppressed.
    """
    engine = DrowsinessEngine()
    alert_mgr = AlertStateManager()
    mock_audio = MockAudioBackend()
    audio_engine = AudioWarningEngine(backend=mock_audio, assets_dir=tmp_path)
    settings_store = AudioSettingsStore(settings_file=tmp_path / "audio.json")
    dispatcher = AlertDispatcher(
        audio_engine=audio_engine,
        settings_store=settings_store,
        event_logger=AlertEventLogger(enable_file_logging=False),
        critical_reminder_cooldown_ms=1000.0,
    )

    try:
        # 1. Trigger CRITICAL_WARNING via sustained eye closure (0..1600ms)
        for t in range(0, 1700, 100):
            f = _make_features(
                float(t),
                ear_val=0.10,
                is_closed=True,
                blink_state=BlinkState.PROLONGED_CLOSURE if t >= 500 else BlinkState.CLOSED,
                blink_closed=True,
                closure_dur_ms=float(t),
            )
            a = engine.evaluate(f)
            tr = alert_mgr.update(a)
            dispatcher.dispatch(tr, alert_mgr.current_state, float(t), alert_mgr.is_recovering)

        assert a.state == DrowsinessState.CRITICAL_WARNING
        assert alert_mgr.current_state == AlertState.CRITICAL
        assert alert_mgr.is_recovering is False

        # 2. Prolonged face loss (1700..2900ms -> INSUFFICIENT_DATA)
        for t in range(1700, 3000, 100):
            f = _make_features(
                float(t),
                has_face=False,
                ear_val=0.0,
                ear_is_valid=False,
                pose_is_valid=False,
                blink_state=BlinkState.UNCERTAIN_GAP,
            )
            a = engine.evaluate(f)
            tr = alert_mgr.update(a)
            dispatcher.dispatch(tr, alert_mgr.current_state, float(t), alert_mgr.is_recovering)

        assert a.state == DrowsinessState.INSUFFICIENT_DATA
        assert a.alert_active is True
        assert alert_mgr.current_state == AlertState.CRITICAL
        assert alert_mgr.is_recovering is False

        # 3. Face reappears at t=3000ms with EYES STILL CLOSED (left-censored closure=0ms)
        f_reacq_closed = _make_features(
            3000.0,
            ear_val=0.10,
            is_closed=True,
            blink_state=BlinkState.CLOSED,
            blink_closed=True,
            closure_dur_ms=0.0,
        )
        a_reacq = engine.evaluate(f_reacq_closed)
        tr_reacq = alert_mgr.update(a_reacq)

        assert a_reacq.state == DrowsinessState.CRITICAL_WARNING
        assert a_reacq.alert_active is True
        assert a_reacq.is_alert_recovering is False
        assert alert_mgr.current_state == AlertState.CRITICAL
        assert alert_mgr.is_recovering is False

        # Periodic CRITICAL reminder continues firing while eyes remain closed
        # (last reminder fired at 2500ms during face loss; next fires at 3500ms with 1000ms cooldown)
        last_crit_before = dispatcher._last_announced_ts_ms.get(AlertState.CRITICAL, 0.0)
        for t in range(3100, 3600, 100):
            f_still_closed = _make_features(
                float(t),
                ear_val=0.10,
                is_closed=True,
                blink_state=BlinkState.CLOSED,
                blink_closed=True,
                closure_dur_ms=float(t - 3000),
            )
            a_sc = engine.evaluate(f_still_closed)
            tr_sc = alert_mgr.update(a_sc)
            assert alert_mgr.is_recovering is False
            dispatcher.dispatch(tr_sc, alert_mgr.current_state, float(t), alert_mgr.is_recovering)

        assert dispatcher._last_announced_ts_ms.get(AlertState.CRITICAL, 0.0) >= 3500.0
        assert dispatcher._last_announced_ts_ms[AlertState.CRITICAL] > last_crit_before
    finally:
        dispatcher.shutdown()


def test_high3_eye_occlusion_cannot_clear_ocular_attention_or_critical() -> None:
    """
    Reproduces Scenario I / audit HIGH-3:
    After an ocular-triggered ATTENTION (EXTENDED_CLOSURE), occluding eyes (is_eye_closed=None,
    ocular_viable=False while head_pose_viable=True) must NOT clear ATTENTION to NORMAL or emit
    a false recovery transition.
    """
    engine = DrowsinessEngine()
    alert_mgr = AlertStateManager()

    # Trigger ATTENTION via 600 ms eye closure
    for t in range(0, 700, 100):
        f = _make_features(
            float(t),
            ear_val=0.10,
            is_closed=True,
            blink_state=BlinkState.PROLONGED_CLOSURE if t >= 500 else BlinkState.CLOSED,
            blink_closed=True,
            closure_dur_ms=float(t),
        )
        a = engine.evaluate(f)
        alert_mgr.update(a)

    assert a.state == DrowsinessState.ATTENTION
    assert alert_mgr.current_state == AlertState.ATTENTION

    # Occlude eyes for 2,000 ms while head pose remains valid (> 1,000 ms ATTENTION cooldown)
    for t in range(700, 2800, 100):
        f_occ = _make_features(
            float(t),
            ear_val=0.0,
            ear_is_valid=False,
            is_closed=False,
            pose_is_valid=True,
            zone=PoseZone.CENTER,
            blink_state=BlinkState.UNCERTAIN_GAP,
            blink_closed=False,
            closure_dur_ms=0.0,
        )
        a_occ = engine.evaluate(f_occ)
        tr_occ = alert_mgr.update(a_occ)
        assert a_occ.state == DrowsinessState.ATTENTION
        assert alert_mgr.current_state == AlertState.ATTENTION
        assert tr_occ is None  # No false recovery!

    # Now eyes become visible and VERIFIED OPEN for >= 1,000 ms -> genuine recovery
    recovered = False
    for t in range(2800, 4000, 100):
        f_open = _make_features(
            float(t),
            ear_val=0.30,
            ear_is_valid=True,
            is_closed=False,
            pose_is_valid=True,
            zone=PoseZone.CENTER,
            blink_state=BlinkState.OPEN,
            blink_closed=False,
            closure_dur_ms=0.0,
        )
        a_open = engine.evaluate(f_open)
        tr_open = alert_mgr.update(a_open)
        if tr_open is not None and tr_open.is_recovery:
            recovered = True
            assert tr_open.new_state == AlertState.NORMAL

    assert recovered is True
    assert alert_mgr.current_state == AlertState.NORMAL
