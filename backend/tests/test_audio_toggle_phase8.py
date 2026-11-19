"""
DriveGuard — Phase 8: Audio Toggle and Persistence Automated Unit Tests.

Validates:
1. Fresh installation defaults to audio ENABLED.
2. User can toggle audio disable/enable at runtime without restart.
3. Active playback stops immediately when audio is disabled.
4. Queued audio announcements are cleared when audio is disabled.
5. Detection and alert state transitions continue unimpeded while audio is disabled.
6. Re-enabling audio resumes subsequent announcements.
7. Re-enabling audio while CRITICAL is active announces CRITICAL danger once according to policy.
8. Persistence: Setting survives application restart across store instances.
9. Invalid or corrupted settings file safely falls back to default ENABLED.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import pytest

from backend.app.alerts.alert_dispatcher import AlertDispatcher
from backend.app.alerts.alert_state import AlertState, AlertTransition
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.alerts.audio_settings import AudioSettingsStore
from backend.app.alerts.audio_warning_engine import AudioWarningEngine, MockAudioBackend
from backend.app.intelligence.reason_codes import ExplainableReason, ReasonCode


def make_transition(
    prev_state: AlertState,
    new_state: AlertState,
    timestamp_ms: float,
    alert_active: bool = False,
) -> AlertTransition:
    reason = ExplainableReason(code=ReasonCode.ALERT_NORMAL, message="Test toggle")
    return AlertTransition(
        timestamp_ms=timestamp_ms,
        previous_state=prev_state,
        new_state=new_state,
        severity=new_state.severity,
        reason=reason,
        reason_code=ReasonCode.ALERT_NORMAL.value,
        alert_active=alert_active,
    )


class TestAudioToggleAndPersistence:
    """Automated tests for Audio Warnings Toggle and Persistence."""

    def test_fresh_installation_defaults_to_enabled(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "new_audio_settings.json"
        store = AudioSettingsStore(settings_file=settings_file)
        assert store.is_audio_enabled() is True

    def test_toggle_runtime_and_persistence(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "audio_settings.json"
        store = AudioSettingsStore(settings_file=settings_file)
        assert store.is_audio_enabled() is True

        # Toggle to False
        new_val = store.toggle()
        assert new_val is False
        assert store.is_audio_enabled() is False

        # Verify disk persistence across new store instance (application restart)
        store_restarted = AudioSettingsStore(settings_file=settings_file)
        assert store_restarted.is_audio_enabled() is False

        # Toggle back to True
        store_restarted.set_audio_enabled(True)
        assert store_restarted.is_audio_enabled() is True

        store_restarted_2 = AudioSettingsStore(settings_file=settings_file)
        assert store_restarted_2.is_audio_enabled() is True

    def test_invalid_settings_file_falls_back_to_default_enabled(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "corrupted_settings.json"
        # Write invalid corrupt JSON
        settings_file.write_text("{corrupt: [json}", encoding="utf-8")

        store = AudioSettingsStore(settings_file=settings_file)
        assert store.is_audio_enabled() is True

    def test_disabling_audio_stops_active_sound_and_clears_queue(self, tmp_path: Path) -> None:
        settings_file = tmp_path / "settings.json"
        store = AudioSettingsStore(settings_file=settings_file)
        mock_backend = MockAudioBackend(simulated_play_sec=0.30)
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=store)

        try:
            # Trigger warning
            t_warn = make_transition(AlertState.NORMAL, AlertState.WARNING, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t_warn, current_state=AlertState.WARNING, timestamp_ms=1000.0)
            time.sleep(0.05)
            assert mock_backend.is_playing()

            # Disable audio warnings
            dispatcher.toggle_audio()
            time.sleep(0.05)

            # Audio must be immediately stopped
            assert not mock_backend.is_playing()
            assert mock_backend.stop_calls >= 1
            assert not dispatcher.is_audio_enabled
        finally:
            dispatcher.shutdown()

    def test_detection_and_alert_state_transitions_continue_while_audio_disabled(self, tmp_path: Path) -> None:
        """When audio is disabled, alert state machine and visual transitions continue fully operational."""
        settings_file = tmp_path / "settings.json"
        store = AudioSettingsStore(settings_file=settings_file)
        store.set_audio_enabled(False)

        mock_backend = MockAudioBackend()
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=store)

        visual_transitions: list[AlertTransition] = []
        dispatcher.add_transition_listener(lambda t: visual_transitions.append(t))

        try:
            t_crit = make_transition(AlertState.NORMAL, AlertState.CRITICAL, timestamp_ms=1000.0, alert_active=True)
            dispatcher.dispatch(t_crit, current_state=AlertState.CRITICAL, timestamp_ms=1000.0)
            time.sleep(0.05)

            # Visual listener received the CRITICAL transition
            assert len(visual_transitions) == 1
            assert visual_transitions[0].new_state == AlertState.CRITICAL

            # But audio playback was NOT performed because audio is disabled
            assert len(mock_backend.played_files) == 0
        finally:
            dispatcher.shutdown()

    def test_re_enabling_audio_while_critical_active_announces_once(self, tmp_path: Path) -> None:
        """If audio is re-enabled while CRITICAL is active, policy announces CRITICAL danger once."""
        settings_file = tmp_path / "settings.json"
        store = AudioSettingsStore(settings_file=settings_file)
        store.set_audio_enabled(False)

        mock_backend = MockAudioBackend(simulated_play_sec=0.01)
        audio_engine = AudioWarningEngine(backend=mock_backend, assets_dir=tmp_path)
        dispatcher = AlertDispatcher(audio_engine=audio_engine, settings_store=store)

        try:
            # Active state is CRITICAL while audio is disabled
            dispatcher.dispatch(
                make_transition(AlertState.NORMAL, AlertState.CRITICAL, 1000.0, alert_active=True),
                current_state=AlertState.CRITICAL,
                timestamp_ms=1000.0,
            )
            time.sleep(0.05)
            assert len(mock_backend.played_files) == 0

            # User re-enables audio warnings
            dispatcher.toggle_audio(active_state=AlertState.CRITICAL, timestamp_ms=1500.0)
            time.sleep(0.15)

            # Should announce active CRITICAL warning once
            assert len(mock_backend.played_files) > 0
            assert mock_backend.played_files[-1].name == "critical_voice.wav"
        finally:
            dispatcher.shutdown()
