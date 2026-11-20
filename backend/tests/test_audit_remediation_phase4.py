"""
DriveGuard — Phase 4 Core System Audit Remediation Regression Tests.

Covers:
  - MEDIUM-1: AlertEventLogger emits wall-clock UTC ISO timestamps (never 1970-01-01 from monotonic ms)
  - MEDIUM-5: DecisionEngine preserves active elevated reason during hysteresis hold (no contradictory ALERT_NORMAL)
  - MEDIUM-6: AlertStateManager.force_recovery() sets ReasonCode.ALERT_NORMAL (not DrowsinessState.NORMAL)
  - MEDIUM-7: Head-pose proxy distraction wording clarifies head orientation proxy
  - LOW-4: AlertDispatcher unseen timestamp default None (first alert at t=2000ms not suppressed) + periodic CRITICAL log
  - LOW-6: DrowsinessEngine exposes severe_perclos_threshold_pct; NODDING_MICROSLEEP emits accurate threshold
  - LOW-7: DriveGuardFaceLandmarker.draw_landmarks() never mutates caller's FramePacket.frame in place
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.app.alerts.alert_dispatcher import AlertDispatcher
from backend.app.alerts.alert_state import AlertState, AlertTransition
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.alerts.audio_settings import AudioSettingsStore
from backend.app.alerts.event_logger import AlertEventLogger
from backend.app.intelligence.confidence import ConfidenceAssessment, ConfidenceTier
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.drowsiness_engine import DrowsinessAssessment, DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ExplainableReason, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalMetrics
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


def _make_metrics(
    ts_ms: float,
    active_closure_ms: float = 0.0,
    is_eye_closed: bool = False,
    is_nodding: bool = False,
    off_road_ms: float = 0.0,
    perclos_pct: float = 3.0,
) -> TemporalMetrics:
    return TemporalMetrics(
        timestamp_ms=ts_ms,
        active_closure_duration_ms=active_closure_ms,
        active_off_road_dwell_ms=off_road_ms,
        is_mirror_check_context=False,
        is_dashboard_check_context=False,
        is_nodding_microsleep=is_nodding,
        blink_count_60s=5,
        blink_count_30s=3,
        mean_blink_duration_ms_60s=200.0,
        prolonged_blink_count_60s=0,
        severe_blink_count_60s=0,
        mean_rcr_60s=1.0,
        yawn_count_60s=0,
        yawn_count_30s=0,
        active_yawn_duration_ms=0.0,
        perclos_pct=perclos_pct,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
        is_eye_closed=is_eye_closed,
    )


def _reliable_conf() -> ConfidenceAssessment:
    return ConfidenceAssessment(
        overall_confidence=0.95,
        tier=ConfidenceTier.HIGH,
        is_reliable=True,
        is_head_pose_available=True,
        is_ocular_available=True,
        is_yawn_available=True,
    )


# ---------------------------------------------------------------------------
# MEDIUM-1: Wall-Clock UTC ISO Timestamps in AlertEventLogger
# ---------------------------------------------------------------------------

def test_medium1_alert_event_logger_emits_wall_clock_utc_iso_time(tmp_path: Path) -> None:
    log_path = tmp_path / "events.jsonl"
    event_logger = AlertEventLogger(log_file=log_path, enable_file_logging=True)

    # Small monotonic timestamp (1.5 seconds since process boot)
    entry = event_logger.log_event(
        timestamp_ms=1500.0,
        previous_state="NORMAL",
        new_state="CRITICAL",
        severity=3,
        reason_code="SUSTAINED_EYE_CLOSURE",
        reason_message="Sustained eye closure",
    )
    assert entry.timestamp_ms == 1500.0
    assert not entry.iso_time.startswith("1970-"), f"Got epoch 1970 timestamp: {entry.iso_time}"
    parsed = datetime.fromisoformat(entry.iso_time)
    assert parsed.tzinfo is not None
    assert parsed.year == datetime.now(timezone.utc).year


# ---------------------------------------------------------------------------
# MEDIUM-5 & MEDIUM-7: Hysteresis Hold Reason Preservation & Head-Pose Proxy Wording
# ---------------------------------------------------------------------------

def test_medium5_and_medium7_hysteresis_hold_preserves_reason_and_proxy_wording() -> None:
    engine = DecisionEngine(
        sustained_closure_threshold_ms=1500.0,
        recovery_critical_ms=2000.0,
        off_road_distraction_limit_ms=3500.0,
    )
    conf = _reliable_conf()

    # Escalate to CRITICAL via sustained eye closure
    d1 = engine.evaluate(_make_metrics(1000.0, active_closure_ms=1600.0, is_eye_closed=True), conf)
    assert d1.state == DrowsinessState.CRITICAL_WARNING
    assert d1.primary_reason is not None
    assert d1.primary_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE

    # Eyes reopen -> candidate is NORMAL, but CRITICAL is held during 2000ms recovery cooldown
    d_hold = engine.evaluate(_make_metrics(1500.0, active_closure_ms=0.0, is_eye_closed=False), conf)
    assert d_hold.state == DrowsinessState.CRITICAL_WARNING
    assert d_hold.primary_reason is not None
    assert d_hold.primary_reason.code != ReasonCode.ALERT_NORMAL, (
        "Must never emit contradictory ALERT_NORMAL primary_reason while holding CRITICAL_WARNING"
    )
    assert d_hold.primary_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE

    # MEDIUM-7: Persistent off-road distraction message clarifies head orientation proxy
    engine.reset()
    d_dist = engine.evaluate(_make_metrics(2000.0, off_road_ms=4000.0), conf)
    assert d_dist.state == DrowsinessState.DROWSINESS_WARNING
    assert d_dist.primary_reason is not None
    assert d_dist.primary_reason.code == ReasonCode.PERSISTENT_DISTRACTION
    assert "head orientation" in d_dist.primary_reason.message.lower()


# ---------------------------------------------------------------------------
# MEDIUM-6: AlertStateManager.force_recovery() Uses ReasonCode.ALERT_NORMAL
# ---------------------------------------------------------------------------

def test_medium6_force_recovery_uses_reason_code_enum() -> None:
    mgr = AlertStateManager()
    # Push into CRITICAL state first
    dec_crit = DecisionEngine().evaluate(
        _make_metrics(1000.0, active_closure_ms=1600.0, is_eye_closed=True),
        _reliable_conf(),
    )
    assessment = DrowsinessAssessment(
        timestamp_ms=1000.0,
        state=dec_crit.state,
        severity=dec_crit.severity,
        decision=dec_crit,
        temporal_metrics=_make_metrics(1000.0, active_closure_ms=1600.0, is_eye_closed=True),
        confidence=_reliable_conf(),
        is_mirror_check_context=False,
        alert_active=True,
        unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
        historical_reason=dec_crit.primary_reason,
        is_alert_recovering=False,
    )
    mgr.update(assessment)
    assert mgr.current_state == AlertState.CRITICAL

    tr = mgr.force_recovery(timestamp_ms=1500.0)
    assert tr is not None
    assert tr.reason is not None
    assert isinstance(tr.reason.code, ReasonCode)
    assert tr.reason.code == ReasonCode.ALERT_NORMAL


# ---------------------------------------------------------------------------
# LOW-4: AlertDispatcher First-Alert Unseen Timestamp & Periodic CRITICAL Logging
# ---------------------------------------------------------------------------

def test_low4_dispatcher_early_session_alert_and_periodic_critical_log(tmp_path: Path) -> None:
    mock_audio = MagicMock()
    mock_audio.enqueue_alert.return_value = True
    mock_audio.enqueue_recovery.return_value = True

    settings_store = AudioSettingsStore(settings_file=tmp_path / "audio_settings.json")
    settings_store.set_audio_enabled(True)
    event_logger = AlertEventLogger(log_file=tmp_path / "events.jsonl", enable_file_logging=False)

    dispatcher = AlertDispatcher(
        audio_engine=mock_audio,
        settings_store=settings_store,
        event_logger=event_logger,
        attention_cooldown_ms=12000.0,
        critical_reminder_cooldown_ms=1000.0,
    )
    try:
        # 1. Early-session non-forced alert at timestamp_ms=2000 (< 12000ms cooldown) MUST NOT be suppressed!
        queued = dispatcher._dispatch_audio_alert(
            state=AlertState.ATTENTION,
            timestamp_ms=2000.0,
            force=False,
        )
        assert queued is True, "First ATTENTION alert at t=2000ms must not be suppressed against fake 0.0"

        # 2. Periodic CRITICAL reminder must be logged to event_logger
        crit_tr = AlertTransition(
            timestamp_ms=3000.0,
            previous_state=AlertState.NORMAL,
            new_state=AlertState.CRITICAL,
            severity=3,
            reason=ExplainableReason(code=ReasonCode.SUSTAINED_EYE_CLOSURE, message="Closed"),
            reason_code=ReasonCode.SUSTAINED_EYE_CLOSURE.value,
            is_recovery=False,
            alert_active=True,
            unresolved_alert_state=AlertState.CRITICAL.value,
        )
        dispatcher.dispatch(crit_tr, current_state=AlertState.CRITICAL, timestamp_ms=3000.0, is_recovering=False)
        # 1100ms later (>= 1000ms reminder cooldown), no transition -> periodic reminder fires & logs
        dispatcher.dispatch(None, current_state=AlertState.CRITICAL, timestamp_ms=4100.0, is_recovering=False)

        entries = event_logger.get_recent_entries(limit=10)
        reminder_entries = [e for e in entries if e.reason_code == "CRITICAL_PERIODIC_REMINDER"]
        assert len(reminder_entries) == 1
        assert reminder_entries[0].details.get("periodic_reminder") is True

        # Next frame 100ms later must NOT re-log reminder
        dispatcher.dispatch(None, current_state=AlertState.CRITICAL, timestamp_ms=4200.0, is_recovering=False)
        entries_after = event_logger.get_recent_entries(limit=10)
        assert len([e for e in entries_after if e.reason_code == "CRITICAL_PERIODIC_REMINDER"]) == 1
    finally:
        dispatcher.shutdown()


# ---------------------------------------------------------------------------
# LOW-6: DrowsinessEngine severe_perclos_threshold_pct & NODDING_MICROSLEEP Threshold
# ---------------------------------------------------------------------------

def test_low6_drowsiness_engine_severe_perclos_and_nodding_threshold() -> None:
    eng = DrowsinessEngine(severe_perclos_threshold_pct=28.0)
    assert eng.decision_engine.severe_perclos_threshold_pct == 28.0

    # Condition 2 nodding microsleep (sudden pitch drop while eyes closed >= 500ms, < 800ms)
    dec = eng.decision_engine.evaluate(
        _make_metrics(1000.0, active_closure_ms=600.0, is_eye_closed=True, is_nodding=True),
        _reliable_conf(),
    )
    nod_reasons = [r for r in dec.reasons if r.code == ReasonCode.NODDING_MICROSLEEP]
    assert len(nod_reasons) == 1
    assert nod_reasons[0].threshold == 500.0


# ---------------------------------------------------------------------------
# LOW-7: draw_landmarks Does Not Mutate Caller Frame In-Place
# ---------------------------------------------------------------------------

def test_low7_draw_landmarks_does_not_mutate_input_frame() -> None:
    landmarker = DriveGuardFaceLandmarker()
    try:
        raw_frame = np.zeros((240, 320, 3), dtype=np.uint8)
        original_copy = raw_frame.copy()

        lms = tuple(NormalizedLandmark(0.5, 0.5, 0.0) for _ in range(478))
        res = FaceLandmarkResult(
            has_face=True,
            landmarks=lms,
            presence_confidence=0.95,
            timestamp_ms=100,
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=4.0,
            reasons=("OK",),
        )
        annotated = landmarker.draw_landmarks(raw_frame, res, draw_mesh=False, draw_eyes=True, draw_irises=True)
        assert annotated is not raw_frame
        assert np.array_equal(raw_frame, original_copy), "draw_landmarks must not mutate input frame in-place"
        assert not np.array_equal(annotated, original_copy), "annotated frame should contain drawn overlays"
    finally:
        landmarker.close()
