"""
DriveGuard — Structured Alert Event Logger (Phase 8).

Maintains a thread-safe structured log of significant alert transitions, recoveries,
and dispatch actions. Keeps a bounded ring-buffer in memory and appends to an optional
local JSONL audit file without introducing cloud or external dependencies.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional

from backend.app.core.config import settings

logger = logging.getLogger("driveguard.alert_logger")

DEFAULT_LOG_PATH = settings.BASE_DIR / "data" / "logs" / "alert_events.jsonl"


@dataclass(frozen=True)
class AlertLogEntry:
    """Immutable structured record of an alert lifecycle event."""
    timestamp_ms: float
    iso_time: str
    previous_state: str
    new_state: str
    severity: int
    reason_code: Optional[str] = None
    reason_message: Optional[str] = None
    is_recovery: bool = False
    announcement_requested: bool = False
    audio_enabled: bool = True
    playback_status: str = "NOT_REQUESTED"  # SUCCESS, FAILED, MUTED, SKIPPED_COOLDOWN, SKIPPED_DUPLICATE
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class AlertEventLogger:
    """
    Thread-safe event logger recording alert transitions and audio outcomes.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        max_memory_entries: int = 500,
        enable_file_logging: bool = True,
    ) -> None:
        self.log_file = log_file or DEFAULT_LOG_PATH
        self.max_memory_entries = max_memory_entries
        self.enable_file_logging = enable_file_logging
        self._entries: List[AlertLogEntry] = []
        self._lock = threading.RLock()

        if self.enable_file_logging:
            try:
                self.log_file.parent.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning("Could not create alert log directory: %s", e)

    def log_event(
        self,
        timestamp_ms: float,
        previous_state: str,
        new_state: str,
        severity: int,
        reason_code: Optional[str] = None,
        reason_message: Optional[str] = None,
        is_recovery: bool = False,
        announcement_requested: bool = False,
        audio_enabled: bool = True,
        playback_status: str = "NOT_REQUESTED",
        details: Optional[Dict[str, Any]] = None,
    ) -> AlertLogEntry:
        """
        Record a structured alert event to memory and append to disk.
        """
        entry = AlertLogEntry(
            timestamp_ms=timestamp_ms,
            iso_time=datetime.now(timezone.utc).isoformat(timespec="seconds"),
            previous_state=str(previous_state),
            new_state=str(new_state),
            severity=int(severity),
            reason_code=str(reason_code) if reason_code else None,
            reason_message=str(reason_message) if reason_message else None,
            is_recovery=bool(is_recovery),
            announcement_requested=bool(announcement_requested),
            audio_enabled=bool(audio_enabled),
            playback_status=str(playback_status),
            details=details or {},
        )

        with self._lock:
            self._entries.append(entry)
            if len(self._entries) > self.max_memory_entries:
                self._entries.pop(0)

            if self.enable_file_logging:
                try:
                    with open(self.log_file, "a", encoding="utf-8") as f:
                        f.write(json.dumps(entry.to_dict()) + "\n")
                except Exception as err:
                    logger.warning("Failed to append to alert log file (%s): %s", self.log_file, err)

        return entry

    def get_recent_entries(self, limit: int = 50) -> List[AlertLogEntry]:
        """Return the most recent entries up to limit."""
        with self._lock:
            return list(self._entries[-limit:])

    def clear(self) -> None:
        """Clear in-memory log entries."""
        with self._lock:
            self._entries.clear()
