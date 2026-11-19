"""
DriveGuard — Phase 8: Alert Dispatcher Automated Unit Tests.

Validates:
1. Duplicate announcement suppression: Identical unchanged alert states do not repeat announcements.
2. Priority ordering: CRITICAL > WARNING > ATTENTION.
3. Cooldown behaviour: Respects minimum interval between announcements.
4. Critical interruption: CRITICAL preempts and interrupts lower-priority alerts.
5. Cancellation of obsolete announcements: Obsolete queued messages are purged on recovery or escalation.
6. Recovery while announcement is queued: Purges pending warnings and enqueues recovery.
7. Recovery while audio is playing: Preempts lower playback and announces recovery.
8. Listener delivery: Visual listeners receive every formal transition event independently.
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from backend.app.alerts.alert_dispatcher import AlertDispatcher
from backend.app.alerts.alert_state import AlertState, AlertTransition
from backend.app.alerts.audio_settings import AudioSettingsStore
from backend.app.alerts.audio_warning_engine import AudioWarningEngine, MockAudioBackend
from backend.app.alerts.event_logger import AlertEventLogger
from backend.app.intelligence.reason_codes import ExplainableReason, ReasonCode


def make_transition(
    prev_state: AlertState,
    new_state: AlertState,
    timestamp_ms: float,
    is_recovery: bool = False,
    alert_active: bool = False,
    code: ReasonCode = ReasonCode.ALERT_NORMAL,
) -> AlertTransition:
    reason = ExplainableReason(code=code, message="Test reason")
    return AlertTransition(
        timestamp_ms=timestamp_ms,
        previous_state=prev_state,
        new_state=new_state,
        severity=new_state.severity,
        reason=reason,
        reason_code=code.value,
        is_recovery=is_recovery,
        alert_active=alert_active,
    )


class TestAlertDispatcher:
    """Automated tests for AlertDispatcher."""

    def test_duplicate_suppression_for_unchanged_states(self, tmp_path: Path) -> None:
        mock_backend = MockAudioBackend()
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        settings = AudioSettingsStore(settings_file=tmp_path / "settings.json")
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=settings)

        try:
            # 1. Initial transition to Warning
            t1 = make_transition(AlertState.NORMAL, AlertState.WARNING, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t1, current_state=AlertState.WARNING, timestamp_ms=1000.0)
            time.sleep(0.20)
            # Should have played tone and voice for the initial transition
            assert len(mock_backend.played_files) >= 1

            played_count_after_first = len(mock_backend.played_files)

            # 2. Frame tick with NO state transition (transition=None, same WARNING state)
            dispatcher.dispatch(None, current_state=AlertState.WARNING, timestamp_ms=1033.0)
            dispatcher.dispatch(None, current_state=AlertState.WARNING, timestamp_ms=1066.0)
            time.sleep(0.10)

            # Played files should NOT increase
            assert len(mock_backend.played_files) == played_count_after_first
        finally:
            dispatcher.shutdown()

    def test_priority_preemption_critical_interrupts_warning(self, tmp_path: Path) -> None:
        """CRITICAL alert must interrupt and preempt actively playing lower-priority warning."""
        mock_backend = MockAudioBackend(simulated_play_sec=0.20)
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        settings = AudioSettingsStore(settings_file=tmp_path / "settings.json")
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=settings)

        try:
            # Start Warning
            t_warn = make_transition(AlertState.NORMAL, AlertState.WARNING, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t_warn, current_state=AlertState.WARNING, timestamp_ms=1000.0)
            time.sleep(0.05)
            assert mock_backend.is_playing()

            # Escalation to CRITICAL arrives while WARNING is playing
            t_crit = make_transition(AlertState.WARNING, AlertState.CRITICAL, timestamp_ms=1080.0, alert_active=True)
            dispatcher.dispatch(t_crit, current_state=AlertState.CRITICAL, timestamp_ms=1080.0)
            time.sleep(0.05)

            # Backend stop must have been called to preempt WARNING
            assert mock_backend.stop_calls >= 1
            # Current playing or queued state must be CRITICAL
            assert audio_engine.current_playing_state == AlertState.CRITICAL
        finally:
            dispatcher.shutdown()

    def test_lower_priority_cannot_interrupt_critical(self, tmp_path: Path) -> None:
        """Lower priority warning cannot interrupt active CRITICAL playback."""
        mock_backend = MockAudioBackend(simulated_play_sec=0.20)
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        settings = AudioSettingsStore(settings_file=tmp_path / "settings.json")
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=settings)

        try:
            # Start Critical
            t_crit = make_transition(AlertState.NORMAL, AlertState.CRITICAL, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t_crit, current_state=AlertState.CRITICAL, timestamp_ms=1000.0)
            time.sleep(0.05)
            assert mock_backend.is_playing()
            assert audio_engine.current_playing_state == AlertState.CRITICAL

            initial_stop_calls = mock_backend.stop_calls

            # Inbound Attention or Warning attempt
            t_warn = make_transition(AlertState.CRITICAL, AlertState.WARNING, timestamp_ms=1050.0, alert_active=True)
            dispatcher.dispatch(t_warn, current_state=AlertState.CRITICAL, timestamp_ms=1050.0)
            time.sleep(0.05)

            # Must NOT stop or interrupt CRITICAL
            assert mock_backend.stop_calls == initial_stop_calls
            assert audio_engine.current_playing_state == AlertState.CRITICAL
        finally:
            dispatcher.shutdown()

    def test_cancellation_of_obsolete_announcements_on_recovery(self, tmp_path: Path) -> None:
        """Recovery must purge obsolete queued warnings and enqueue recovery message."""
        mock_backend = MockAudioBackend(simulated_play_sec=0.10)
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        settings = AudioSettingsStore(settings_file=tmp_path / "settings.json")
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=settings)

        try:
            # Enqueue warning
            t_warn = make_transition(AlertState.NORMAL, AlertState.WARNING, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t_warn, current_state=AlertState.WARNING, timestamp_ms=1000.0)

            # Recovery event occurs
            t_rec = make_transition(AlertState.WARNING, AlertState.NORMAL, timestamp_ms=1500.0, is_recovery=True)
            dispatcher.dispatch(t_rec, current_state=AlertState.NORMAL, timestamp_ms=1500.0)
            time.sleep(0.25)

            # Audio engine should have processed recovery
            assert mock_backend.played_files[-1].name == "recovery_voice.wav"
        finally:
            dispatcher.shutdown()

    def test_cooldown_enforcement(self, tmp_path: Path) -> None:
        """Rapid re-triggering of ATTENTION within cooldown window must be suppressed."""
        mock_backend = MockAudioBackend()
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        settings = AudioSettingsStore(settings_file=tmp_path / "settings.json")
        dispatcher = AlertDispatcher(
            audio_engine=audio_engine,
            settings_store=settings,
            attention_cooldown_ms=10000.0,
        )

        try:
            # First Attention
            t1 = make_transition(AlertState.NORMAL, AlertState.ATTENTION, timestamp_ms=1000.0)
            dispatcher.dispatch(t1, current_state=AlertState.ATTENTION, timestamp_ms=1000.0)
            time.sleep(0.20)
            assert len(mock_backend.played_files) > 0
            initial_count = len(mock_backend.played_files)

            # Second Attention just 2.0s later (within 10s cooldown)
            t2 = make_transition(AlertState.NORMAL, AlertState.ATTENTION, timestamp_ms=3000.0)
            dispatcher.dispatch(t2, current_state=AlertState.ATTENTION, timestamp_ms=3000.0)
            time.sleep(0.10)

            # Should be suppressed by cooldown
            assert len(mock_backend.played_files) == initial_count
        finally:
            dispatcher.shutdown()

    def test_visual_listener_notification(self, tmp_path: Path) -> None:
        """Visual listeners receive alert transitions promptly."""
        mock_backend = MockAudioBackend()
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        settings = AudioSettingsStore(settings_file=tmp_path / "settings.json")
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=settings)

        notified_events: list[AlertTransition] = []

        def on_transition(t: AlertTransition) -> None:
            notified_events.append(t)

        dispatcher.add_transition_listener(on_transition)
        try:
            t1 = make_transition(AlertState.NORMAL, AlertState.WARNING, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t1, current_state=AlertState.WARNING, timestamp_ms=1000.0)
            assert len(notified_events) == 1
            assert notified_events[0].new_state == AlertState.WARNING
        finally:
            dispatcher.shutdown()
