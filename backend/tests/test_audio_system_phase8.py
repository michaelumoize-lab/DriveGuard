"""
DriveGuard — Phase 8: Audio Warning System Automated Unit Tests.

Validates:
1. Voice and warning-tone playback requests.
2. Playback failure handling (isolated, non-crashing).
3. Missing audio file handling (isolated, logged, non-crashing).
4. Audio device unavailable (PortAudio / backend failure does not crash pipeline).
5. Preemption / interruption of active playback by CRITICAL alert.
6. Queue clearing and cancellation of obsolete items.
7. Clean shutdown and thread termination.
8. Disabling audio: stops active sound, clears queue, rejects new requests.
9. Re-enabling audio: resumes new announcements, does not replay purged backlog.
"""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from backend.app.alerts.alert_state import AlertState
from backend.app.alerts.audio_warning_engine import (
    AudioWarningEngine,
    MockAudioBackend,
    SoundDeviceAudioBackend,
)


class TestAudioWarningEngine:
    """Automated tests for AudioWarningEngine."""

    def test_voice_and_tone_playback_requests(self, tmp_path: Path) -> None:
        mock = MockAudioBackend(simulated_play_sec=0.01)
        engine = AudioWarningEngine(backend=mock, assets_dir=tmp_path)
        try:
            ok = engine.enqueue_alert(
                state=AlertState.WARNING,
                timestamp_ms=1000.0,
                play_tone=True,
                play_voice=True,
            )
            assert ok
            time.sleep(0.15)
            # Both tone and voice were requested for playback
            assert len(mock.played_files) == 2
            assert mock.played_files[0].name == "warning_tone.wav"
            assert mock.played_files[1].name == "warning_voice.wav"
        finally:
            engine.shutdown()

    def test_playback_failure_handled_gracefully(self, tmp_path: Path) -> None:
        """Simulate backend hardware playback failure; engine must not crash or deadlock."""
        mock = MockAudioBackend(simulate_failure=True)
        engine = AudioWarningEngine(backend=mock, assets_dir=tmp_path)
        try:
            ok = engine.enqueue_alert(
                state=AlertState.CRITICAL,
                timestamp_ms=1000.0,
                play_tone=True,
                play_voice=True,
            )
            assert ok
            time.sleep(0.10)
            # Engine worker processed the item without raising unhandled exception
            assert not engine.is_playing
        finally:
            engine.shutdown()

    def test_sounddevice_backend_missing_file_safety(self, tmp_path: Path) -> None:
        """Production SoundDeviceAudioBackend handles non-existent file safely without crashing."""
        backend = SoundDeviceAudioBackend()
        missing_file = tmp_path / "non_existent_voice.wav"
        success = backend.play_wav(missing_file)
        assert success is False
        assert not backend.is_playing()

    def test_preemption_of_active_lower_priority_audio(self, tmp_path: Path) -> None:
        """A CRITICAL alert immediately interrupts active WARNING playback."""
        mock = MockAudioBackend(simulated_play_sec=0.30)
        engine = AudioWarningEngine(backend=mock, assets_dir=tmp_path)
        try:
            engine.enqueue_alert(state=AlertState.WARNING, timestamp_ms=1000.0)
            time.sleep(0.05)
            assert mock.is_playing()
            assert engine.current_playing_state == AlertState.WARNING

            # Enqueue CRITICAL while WARNING is playing
            engine.enqueue_alert(state=AlertState.CRITICAL, timestamp_ms=1050.0)
            time.sleep(0.05)

            # Backend stop must have been invoked
            assert mock.stop_calls >= 1
            # Current playing state must have switched to CRITICAL
            assert engine.current_playing_state == AlertState.CRITICAL
        finally:
            engine.shutdown()

    def test_audio_disabled_stops_playback_and_clears_queue(self, tmp_path: Path) -> None:
        """Disabling audio immediately stops current sound, drains queue, and rejects new audio."""
        mock = MockAudioBackend(simulated_play_sec=0.30)
        engine = AudioWarningEngine(backend=mock, assets_dir=tmp_path)
        try:
            engine.enqueue_alert(state=AlertState.WARNING, timestamp_ms=1000.0)
            time.sleep(0.05)
            assert mock.is_playing()

            # User disables audio
            engine.set_enabled(False)
            time.sleep(0.05)

            # Audio stopped immediately
            assert not mock.is_playing()
            assert mock.stop_calls >= 1

            # New requests while disabled are rejected
            rejected = engine.enqueue_alert(state=AlertState.CRITICAL, timestamp_ms=1100.0)
            assert rejected is False
            rejected_rec = engine.enqueue_recovery(timestamp_ms=1200.0)
            assert rejected_rec is False
        finally:
            engine.shutdown()

    def test_re_enabling_audio_does_not_replay_cleared_backlog(self, tmp_path: Path) -> None:
        """When re-enabled, purged announcements are not replayed."""
        mock = MockAudioBackend(simulated_play_sec=0.01)
        engine = AudioWarningEngine(backend=mock, assets_dir=tmp_path)
        try:
            engine.set_enabled(False)
            engine.enqueue_alert(state=AlertState.WARNING, timestamp_ms=1000.0)
            engine.enqueue_alert(state=AlertState.CRITICAL, timestamp_ms=1050.0)

            # Re-enable audio
            engine.set_enabled(True)
            time.sleep(0.05)

            # Backlog was cleared; no old files were played
            assert len(mock.played_files) == 0

            # New subsequent alert plays normally
            ok = engine.enqueue_alert(state=AlertState.ATTENTION, timestamp_ms=2000.0)
            assert ok
            time.sleep(0.15)
            assert len(mock.played_files) > 0
        finally:
            engine.shutdown()

    def test_clean_shutdown(self, tmp_path: Path) -> None:
        mock = MockAudioBackend(simulated_play_sec=0.10)
        engine = AudioWarningEngine(backend=mock, assets_dir=tmp_path)
        engine.enqueue_alert(state=AlertState.CRITICAL, timestamp_ms=1000.0)
        engine.shutdown(timeout_sec=0.5)
        assert not engine._running
        assert not engine._worker_thread.is_alive()
