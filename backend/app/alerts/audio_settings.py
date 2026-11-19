"""
DriveGuard — Audio Settings Persistence (Phase 8).

Thread-safe, atomic file persistence for user-facing audio warning preferences
(Enabled / Disabled). Survives application restarts with fail-safe defaults.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Callable, List, Optional

from backend.app.core.config import settings

logger = logging.getLogger("driveguard.audio_settings")

DEFAULT_SETTINGS_PATH = settings.BASE_DIR / "data" / "audio_settings.json"


class AudioSettingsStore:
    """
    Manages persistent user preferences for local voice warnings and audio tones.
    Thread-safe with atomic file writes.
    """

    def __init__(self, settings_file: Optional[Path] = None) -> None:
        self.settings_file = settings_file or DEFAULT_SETTINGS_PATH
        self.settings_file.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._listeners: List[Callable[[bool], None]] = []
        self._audio_enabled: bool = self._load()

    def is_audio_enabled(self) -> bool:
        """Returns True if local audio warnings and voice announcements are enabled."""
        with self._lock:
            return self._audio_enabled

    def set_audio_enabled(self, enabled: bool) -> None:
        """
        Updates and persists the audio preference immediately.
        Notifies all registered listeners.
        """
        with self._lock:
            if self._audio_enabled == enabled:
                return
            self._audio_enabled = bool(enabled)
            self._save(self._audio_enabled)

        # Notify listeners outside the lock to prevent re-entrancy deadlocks
        self._notify_listeners(self._audio_enabled)

    def toggle(self) -> bool:
        """Toggles the current audio setting and returns the new state."""
        with self._lock:
            new_state = not self._audio_enabled
        self.set_audio_enabled(new_state)
        return new_state

    def add_listener(self, callback: Callable[[bool], None]) -> None:
        """Register a callback for audio toggle events."""
        with self._lock:
            if callback not in self._listeners:
                self._listeners.append(callback)

    def remove_listener(self, callback: Callable[[bool], None]) -> None:
        """Unregister a callback."""
        with self._lock:
            if callback in self._listeners:
                self._listeners.remove(callback)

    def _notify_listeners(self, enabled: bool) -> None:
        """Notify all callbacks of setting change."""
        with self._lock:
            listeners = list(self._listeners)
        for cb in listeners:
            try:
                cb(enabled)
            except Exception as e:
                logger.error("Error in audio settings listener: %s", e)

    def _load(self) -> bool:
        """
        Load persisted setting from disk.
        Returns True (default enabled) if missing, unreadable, or corrupted.
        """
        if not self.settings_file.is_file():
            # Fresh installation defaults to enabled=True
            logger.info("No audio settings found; defaulting to audio warnings ENABLED.")
            return True

        try:
            raw = self.settings_file.read_text(encoding="utf-8")
            data = json.loads(raw)
            if isinstance(data, dict) and "audio_warnings_enabled" in data:
                val = bool(data["audio_warnings_enabled"])
                logger.info("Loaded audio warnings setting: %s", "ENABLED" if val else "DISABLED")
                return val
            logger.warning("Invalid audio settings structure; defaulting to ENABLED.")
            return True
        except Exception as err:
            logger.warning("Could not read audio settings (%s); falling back to ENABLED.", err)
            return True

    def _save(self, enabled: bool) -> None:
        """Atomically persist setting to disk."""
        data = {
            "version": "1.0",
            "audio_warnings_enabled": enabled,
        }
        try:
            target_dir = self.settings_file.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Atomic write via tempfile and replace
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=str(target_dir),
                delete=False,
            ) as tf:
                json.dump(data, tf, indent=2)
                temp_path = Path(tf.name)

            temp_path.replace(self.settings_file)
            logger.info("Persisted audio warnings setting: %s", "ENABLED" if enabled else "DISABLED")
        except Exception as err:
            logger.error("Failed to persist audio settings to %s: %s", self.settings_file, err)


# Global default instance
audio_settings_store = AudioSettingsStore()
