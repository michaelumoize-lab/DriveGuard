"""
DriveGuard — Audio Warning Engine (Phase 8).

Manages offline, asynchronous, and non-blocking playback of:
1. Urgent human-sounding voice announcements.
2. Differentiated warning tones.

Provides:
- Thread-safe priority playback queue (CRITICAL preempts lower levels).
- Immediate preemption of playing lower-priority audio when CRITICAL triggers.
- Protection of active CRITICAL warnings against lower-priority interruption.
- Obsolete queued message cancellation.
- Immediate stopping and queue clearing when audio is disabled.
- Graceful error isolation (missing files or device errors never crash the vision pipeline).
- Full pluggability for unit tests via MockAudioBackend.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import os
from pathlib import Path
import queue
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple
import wave

import numpy as np

from backend.app.alerts.alert_state import AlertState

logger = logging.getLogger("driveguard.audio_engine")

ASSETS_DIR = Path(__file__).resolve().parent / "assets"


class AudioBackend(Protocol):
    """Protocol for audio device backends (hardware sounddevice or test mock)."""

    def play_wav(self, file_path: Path, interrupt_check: Optional[Callable[[], bool]] = None) -> bool:
        """Play a WAV file to completion. Return True on success, False on error or interrupt."""
        ...

    def stop(self) -> None:
        """Immediately stop all active audio playback."""
        ...

    def is_playing(self) -> bool:
        """Return True if audio is currently playing."""
        ...


class SoundDeviceAudioBackend:
    """Production audio backend using sounddevice and standard library wave."""

    def __init__(self) -> None:
        self._is_playing = False
        self._lock = threading.Lock()

    def play_wav(self, file_path: Path, interrupt_check: Optional[Callable[[], bool]] = None) -> bool:
        if not file_path.is_file():
            logger.warning("Audio asset not found: %s", file_path)
            return False

        try:
            import sounddevice as sd

            with wave.open(str(file_path), "rb") as wf:
                n_channels = wf.getnchannels()
                sampwidth = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                raw_bytes = wf.readframes(n_frames)

            if sampwidth == 2:
                data = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sampwidth == 1:
                data = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            else:
                logger.error("Unsupported audio bit depth (%d bytes) in %s", sampwidth, file_path)
                return False

            if n_channels > 1:
                data = data.reshape(-1, n_channels)

            with self._lock:
                self._is_playing = True

            sd.play(data, framerate)

            # Wait in small intervals so we can react promptly to interrupt_check
            duration = len(data) / framerate
            start = time.monotonic()
            while (time.monotonic() - start) < duration:
                if interrupt_check and interrupt_check():
                    sd.stop()
                    with self._lock:
                        self._is_playing = False
                    return False
                time.sleep(0.02)

            sd.wait()
            with self._lock:
                self._is_playing = False
            return True

        except Exception as exc:
            logger.warning("Audio playback failure for %s: %s", file_path.name, exc)
            with self._lock:
                self._is_playing = False
            return False

    def stop(self) -> None:
        try:
            import sounddevice as sd
            sd.stop()
        except Exception:
            pass
        finally:
            with self._lock:
                self._is_playing = False

    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing


class MockAudioBackend:
    """Test-controlled fake audio backend for fast, deterministic, offline unit tests."""

    def __init__(self, simulate_failure: bool = False, simulated_play_sec: float = 0.0) -> None:
        self.simulate_failure = simulate_failure
        self.simulated_play_sec = simulated_play_sec
        self.played_files: List[Path] = []
        self.stop_calls: int = 0
        self._is_playing = False
        self._lock = threading.Lock()

    def play_wav(self, file_path: Path, interrupt_check: Optional[Callable[[], bool]] = None) -> bool:
        if self.simulate_failure:
            return False

        with self._lock:
            self._is_playing = True
            self.played_files.append(file_path)

        if self.simulated_play_sec > 0:
            elapsed = 0.0
            while elapsed < self.simulated_play_sec:
                if interrupt_check and interrupt_check():
                    with self._lock:
                        self._is_playing = False
                    return False
                step = min(0.02, self.simulated_play_sec - elapsed)
                time.sleep(step)
                elapsed += step

        with self._lock:
            self._is_playing = False
        return True

    def stop(self) -> None:
        with self._lock:
            self.stop_calls += 1
            self._is_playing = False

    def is_playing(self) -> bool:
        with self._lock:
            return self._is_playing


@dataclass(order=True)
class AudioQueueItem:
    """
    Priority queue item for auditory alerts.
    Lower sort_key = higher priority (CRITICAL=0, WARNING=1, ATTENTION=2, RECOVERY=3).
    """
    sort_key: int
    timestamp_ms: float = field(compare=False)
    state: AlertState = field(compare=False)
    priority: int = field(compare=False)  # 3: CRITICAL, 2: WARNING, 1: ATTENTION/RECOVERY
    tone_file: Optional[Path] = field(default=None, compare=False)
    voice_file: Optional[Path] = field(default=None, compare=False)
    message_text: str = field(default="", compare=False)
    item_id: str = field(default="", compare=False)


class AudioWarningEngine:
    """
    Asynchronous, non-blocking audio engine orchestrating prioritized voice announcements
    and warning tones.
    """

    def __init__(
        self,
        backend: Optional[AudioBackend] = None,
        assets_dir: Optional[Path] = None,
        enabled_by_default: bool = True,
    ) -> None:
        self.assets_dir = assets_dir or ASSETS_DIR
        self.backend = backend or SoundDeviceAudioBackend()
        self._enabled = enabled_by_default

        self._queue: queue.PriorityQueue[AudioQueueItem] = queue.PriorityQueue()
        self._lock = threading.RLock()
        self._running = True
        self._interrupt_active_playback = threading.Event()
        self._current_item: Optional[AudioQueueItem] = None
        self._last_played_state: Optional[AlertState] = None
        self._last_played_ts_ms: float = 0.0
        self._item_counter = 0

        # Dedicated worker thread
        self._worker_thread = threading.Thread(
            target=self._playback_worker,
            name="DriveGuardAudioWorker",
            daemon=True,
        )
        self._worker_thread.start()

    @property
    def is_enabled(self) -> bool:
        with self._lock:
            return self._enabled

    @property
    def is_playing(self) -> bool:
        return self.backend.is_playing()

    @property
    def current_playing_state(self) -> Optional[AlertState]:
        with self._lock:
            if self.backend.is_playing() and self._current_item is not None:
                return self._current_item.state
            return None

    def set_enabled(self, enabled: bool) -> None:
        """
        Enable or disable audio warnings.
        Disabling immediately silences active playback and flushes pending announcements.
        """
        with self._lock:
            self._enabled = bool(enabled)
            if not self._enabled:
                self._stop_active_playback_locked()
                self._clear_queue_locked()

        logger.info("AudioWarningEngine set_enabled: %s", "ENABLED" if enabled else "DISABLED")

    def enqueue_alert(
        self,
        state: AlertState,
        timestamp_ms: float,
        play_tone: bool = True,
        play_voice: bool = True,
        custom_message: Optional[str] = None,
    ) -> bool:
        """
        Enqueue an alert announcement.
        Handles priority preemption: CRITICAL immediately preempts active lower-priority playback.
        Returns True if item was queued or preemption initiated; False if disabled or rejected.
        """
        with self._lock:
            if not self._enabled:
                return False

            priority = state.severity
            # Priority sort_key: CRITICAL (sev 3) -> 0; WARNING (sev 2) -> 1; ATTENTION (sev 1) -> 2; NORMAL/RECOVERY -> 3
            sort_key = max(0, 3 - priority)

            # Resolve asset paths
            tone_path = self._resolve_tone_path(state) if play_tone else None
            voice_path = self._resolve_voice_path(state) if play_voice else None

            self._item_counter += 1
            item = AudioQueueItem(
                sort_key=sort_key,
                timestamp_ms=timestamp_ms,
                state=state,
                priority=priority,
                tone_file=tone_path,
                voice_file=voice_path,
                message_text=custom_message or state.value,
                item_id=f"{state.value}_{self._item_counter}_{int(timestamp_ms)}",
            )

            # Preemption check:
            # If a CRITICAL alert arrives while lower-priority audio is playing or queued,
            # preempt active audio immediately and clear obsolete lower-priority items.
            if priority >= 3:
                if self._current_item is not None and self._current_item.priority < 3:
                    logger.info("CRITICAL alert preempting active lower-priority playback (%s).", self._current_item.state.value)
                    self._interrupt_active_playback.set()
                    self.backend.stop()

                # Purge obsolete lower-priority items from queue
                self._purge_lower_priority_locked(min_priority=3)

            elif self._current_item is not None and self._current_item.priority >= 3:
                # Lower-priority announcement cannot interrupt or preempt active CRITICAL
                logger.debug("Suppressing lower alert (%s) while CRITICAL is playing.", state.value)
                return False

            self._queue.put(item)
            return True

    def enqueue_recovery(self, timestamp_ms: float) -> bool:
        """
        Enqueue a recovery announcement ("Your alert status has returned to normal").
        Purges obsolete pending warnings from the queue and stops lower active warning playback.
        """
        with self._lock:
            if not self._enabled:
                return False

            # Purge pending warnings from the queue because state has recovered
            self._clear_queue_locked()

            # If actively playing a warning or attention tone/voice, stop it so recovery can sound
            if self._current_item is not None and self._current_item.priority < 3:
                self._interrupt_active_playback.set()
                self.backend.stop()

            self._item_counter += 1
            item = AudioQueueItem(
                sort_key=3,  # Lowest priority
                timestamp_ms=timestamp_ms,
                state=AlertState.NORMAL,
                priority=1,
                tone_file=None,
                voice_file=self.assets_dir / "recovery_voice.wav",
                message_text="Alert status returned to normal",
                item_id=f"RECOVERY_{self._item_counter}_{int(timestamp_ms)}",
            )
            self._queue.put(item)
            return True

    def cancel_obsolete(self, target_state: AlertState) -> None:
        """Purge queued messages that are no longer relevant to target_state."""
        with self._lock:
            if target_state == AlertState.NORMAL:
                self._clear_queue_locked()
            elif target_state == AlertState.CRITICAL:
                self._purge_lower_priority_locked(min_priority=3)

    def shutdown(self, timeout_sec: float = 1.0) -> None:
        """Cleanly shut down the audio engine and terminate the worker thread."""
        with self._lock:
            self._running = False
            self._interrupt_active_playback.set()
            self.backend.stop()
            self._clear_queue_locked()

        # Unblock queue.get with dummy sentinel
        try:
            sentinel = AudioQueueItem(
                sort_key=999,
                timestamp_ms=0.0,
                state=AlertState.NORMAL,
                priority=-1,
                item_id="__SHUTDOWN__",
            )
            self._queue.put(sentinel)
        except Exception:
            pass

        self._worker_thread.join(timeout=timeout_sec)
        logger.info("AudioWarningEngine shutdown complete.")

    # -------------------------------------------------------------------------
    # Internal Worker Loop & Helper Methods
    # -------------------------------------------------------------------------

    def _playback_worker(self) -> None:
        """Background thread executing sequential, priority-ordered audio playback."""
        while self._running:
            try:
                item: AudioQueueItem = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if not self._running or item.item_id == "__SHUTDOWN__":
                break

            with self._lock:
                if not self._enabled:
                    self._queue.task_done()
                    continue
                self._current_item = item
                self._interrupt_active_playback.clear()

            def should_interrupt() -> bool:
                return not self._running or not self._enabled or self._interrupt_active_playback.is_set()

            try:
                # 1. Play Tone (if present)
                if item.tone_file is not None and not should_interrupt():
                    tone_ok = self.backend.play_wav(item.tone_file, interrupt_check=should_interrupt)
                    if tone_ok and item.voice_file is not None and not should_interrupt():
                        # Short 100ms pause between tone and voice
                        time.sleep(0.10)

                # 2. Play Voice Announcement (if present and not interrupted)
                if item.voice_file is not None and not should_interrupt():
                    self.backend.play_wav(item.voice_file, interrupt_check=should_interrupt)

                with self._lock:
                    if not should_interrupt():
                        self._last_played_state = item.state
                        self._last_played_ts_ms = item.timestamp_ms

            except Exception as e:
                logger.error("Error during audio playback in worker thread: %s", e)

            finally:
                with self._lock:
                    self._current_item = None
                self._queue.task_done()

    def _stop_active_playback_locked(self) -> None:
        """Stop backend playback and trigger interrupt flag."""
        self._interrupt_active_playback.set()
        self.backend.stop()

    def _clear_queue_locked(self) -> None:
        """Drain all pending items from the queue."""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break

    def _purge_lower_priority_locked(self, min_priority: int) -> None:
        """Purge all queue items with priority strictly less than min_priority."""
        retained: List[AudioQueueItem] = []
        while not self._queue.empty():
            try:
                it = self._queue.get_nowait()
                if it.priority >= min_priority:
                    retained.append(it)
                self._queue.task_done()
            except (queue.Empty, ValueError):
                break
        for it in retained:
            self._queue.put(it)

    def _resolve_tone_path(self, state: AlertState) -> Optional[Path]:
        if state == AlertState.ATTENTION:
            return self.assets_dir / "attention_tone.wav"
        elif state == AlertState.WARNING:
            return self.assets_dir / "warning_tone.wav"
        elif state == AlertState.CRITICAL:
            return self.assets_dir / "critical_tone.wav"
        return None

    def _resolve_voice_path(self, state: AlertState) -> Optional[Path]:
        if state == AlertState.ATTENTION:
            return self.assets_dir / "attention_voice.wav"
        elif state == AlertState.WARNING:
            return self.assets_dir / "warning_voice.wav"
        elif state == AlertState.CRITICAL:
            return self.assets_dir / "critical_voice.wav"
        elif state == AlertState.NORMAL:
            return self.assets_dir / "recovery_voice.wav"
        return None
