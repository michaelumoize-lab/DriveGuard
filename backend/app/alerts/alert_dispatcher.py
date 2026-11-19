"""
DriveGuard — Alert Dispatcher (Phase 8).

Decouples AlertStateManager from output channels (AudioWarningEngine, Visual HUD,
and EventLogger).

Enforces:
1. Event-driven priority dispatch (CRITICAL > WARNING > ATTENTION > RECOVERY).
2. Duplicate announcement suppression (no repeated voice warnings for unchanged states).
3. Cooldown management to eliminate alarm fatigue.
4. Cancellation of obsolete queued announcements upon state transition or recovery.
5. Preemption coordination (CRITICAL preempts lower-priority playback; lower alerts cannot interrupt CRITICAL).
6. Runtime audio Enable/Disable toggle integration (immediate silencing without affecting safety detection).
7. Independent delivery to visual, auditory, and logging sinks.
8. Asynchronous non-blocking operation (never blocks the webcam capture loop).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

from backend.app.alerts.alert_state import AlertState, AlertTransition
from backend.app.alerts.audio_settings import AudioSettingsStore, audio_settings_store
from backend.app.alerts.audio_warning_engine import AudioWarningEngine
from backend.app.alerts.event_logger import AlertEventLogger

logger = logging.getLogger("driveguard.alert_dispatcher")


class AlertDispatcher:
    """
    Central alert distribution coordinator directing validated alert transitions
    to visual interfaces, audio warning engines, and audit loggers.
    """

    def __init__(
        self,
        audio_engine: AudioWarningEngine,
        settings_store: Optional[AudioSettingsStore] = None,
        event_logger: Optional[AlertEventLogger] = None,
        attention_cooldown_ms: float = 12000.0,
        warning_cooldown_ms: float = 15000.0,
        critical_reminder_cooldown_ms: float = 10000.0,
    ) -> None:
        self.audio_engine = audio_engine
        self.settings_store = settings_store or audio_settings_store
        self.event_logger = event_logger or AlertEventLogger()

        self.attention_cooldown_ms = attention_cooldown_ms
        self.warning_cooldown_ms = warning_cooldown_ms
        self.critical_reminder_cooldown_ms = critical_reminder_cooldown_ms

        # State tracking for deduplication & cooldowns
        self._last_announced_state: Optional[AlertState] = None
        self._last_announced_ts_ms: Dict[AlertState, float] = {}
        self._listeners: List[Callable[[AlertTransition], None]] = []

        # Connect settings store listener
        self.settings_store.add_listener(self._on_audio_setting_changed)
        # Sync initial state
        self.audio_engine.set_enabled(self.settings_store.is_audio_enabled())

    @property
    def is_audio_enabled(self) -> bool:
        return self.settings_store.is_audio_enabled()

    def add_transition_listener(self, callback: Callable[[AlertTransition], None]) -> None:
        """Register a visual/UI listener for alert state transitions."""
        if callback not in self._listeners:
            self._listeners.append(callback)

    def remove_transition_listener(self, callback: Callable[[AlertTransition], None]) -> None:
        """Unregister a transition listener."""
        if callback in self._listeners:
            self._listeners.remove(callback)

    def toggle_audio(self, active_state: Optional[AlertState] = None, timestamp_ms: float = 0.0) -> bool:
        """
        User-facing toggle for audio warnings.
        If re-enabled while CRITICAL is active, triggers one CRITICAL announcement.
        Returns the new boolean audio enabled state.
        """
        new_enabled = self.settings_store.toggle()
        logger.info("AlertDispatcher toggled audio warnings: %s", "ENABLED" if new_enabled else "MUTED")

        # If re-enabled and an active CRITICAL alert is present, announce once according to policy
        if new_enabled and active_state == AlertState.CRITICAL:
            logger.info("Audio re-enabled while CRITICAL is active; announcing danger warning.")
            self._dispatch_audio_alert(
                state=AlertState.CRITICAL,
                timestamp_ms=timestamp_ms,
                force=True,
            )

        return new_enabled

    def dispatch(
        self,
        transition: Optional[AlertTransition],
        current_state: AlertState,
        timestamp_ms: float,
        is_recovering: bool = False,
    ) -> None:
        """
        Dispatch alert evaluation results to audio, visual, and logging subsystems.

        Handles:
        - Significant transitions (new state or recovery event).
        - Periodic reminders for active critical alerts if configured cooldown passes.
        """
        audio_enabled = self.is_audio_enabled

        # Case 1: A formal state transition or recovery occurred
        if transition is not None:
            # 1. Notify visual listeners immediately
            self._notify_visual_listeners(transition)

            # 2. Cancel obsolete queued announcements
            self.audio_engine.cancel_obsolete(target_state=transition.new_state)

            # 3. Determine auditory announcement requirement
            playback_status = "MUTED" if not audio_enabled else "SKIPPED"
            announcement_requested = False

            if transition.is_recovery:
                # Recovery back to normal
                self._last_announced_state = AlertState.NORMAL
                announcement_requested = True
                if audio_enabled:
                    queued = self.audio_engine.enqueue_recovery(timestamp_ms=transition.timestamp_ms)
                    playback_status = "SUCCESS" if queued else "FAILED"

            elif transition.new_state in (AlertState.ATTENTION, AlertState.WARNING, AlertState.CRITICAL):
                # Escalation or new warning level
                announcement_requested = True
                if audio_enabled:
                    last_sev = self._last_announced_state.severity if self._last_announced_state else 0
                    is_severity_escalation = transition.new_state.severity > last_sev
                    queued = self._dispatch_audio_alert(
                        state=transition.new_state,
                        timestamp_ms=transition.timestamp_ms,
                        force=is_severity_escalation,
                    )
                    playback_status = "SUCCESS" if queued else "SKIPPED_COOLDOWN"

            # 4. Log structured alert event
            self.event_logger.log_event(
                timestamp_ms=transition.timestamp_ms,
                previous_state=transition.previous_state.value,
                new_state=transition.new_state.value,
                severity=transition.severity,
                reason_code=transition.reason_code,
                reason_message=str(transition.reason) if transition.reason else None,
                is_recovery=transition.is_recovery,
                announcement_requested=announcement_requested,
                audio_enabled=audio_enabled,
                playback_status=playback_status,
                details=transition.details,
            )
            return

        # Case 2: Ongoing state without transition
        # Check if periodic reminder is warranted for sustained CRITICAL alarm
        if current_state == AlertState.CRITICAL and not is_recovering:
            last_ts = self._last_announced_ts_ms.get(AlertState.CRITICAL, 0.0)
            if (timestamp_ms - last_ts) >= self.critical_reminder_cooldown_ms:
                if audio_enabled:
                    queued = self._dispatch_audio_alert(
                        state=AlertState.CRITICAL,
                        timestamp_ms=timestamp_ms,
                        force=True,
                    )
                    if queued:
                        logger.info("Dispatched periodic CRITICAL reminder audio announcement.")

    def shutdown(self) -> None:
        """Shutdown dispatcher and underlying audio engine."""
        self.settings_store.remove_listener(self._on_audio_setting_changed)
        self.audio_engine.shutdown()

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    def _dispatch_audio_alert(
        self,
        state: AlertState,
        timestamp_ms: float,
        force: bool = False,
    ) -> bool:
        """
        Evaluate cooldown and enqueue audio alert if eligible.
        """
        last_announced_ts = self._last_announced_ts_ms.get(state, 0.0)
        cooldown = self._get_cooldown_for_state(state)

        # Allow immediate dispatch on state escalation or if cooldown elapsed
        if not force and (timestamp_ms - last_announced_ts) < cooldown:
            logger.debug("Suppressing audio alert (%s) due to active cooldown.", state.value)
            return False

        # Enqueue with both tone and voice
        queued = self.audio_engine.enqueue_alert(
            state=state,
            timestamp_ms=timestamp_ms,
            play_tone=True,
            play_voice=True,
        )

        if queued:
            self._last_announced_state = state
            self._last_announced_ts_ms[state] = timestamp_ms

        return queued

    def _get_cooldown_for_state(self, state: AlertState) -> float:
        if state == AlertState.ATTENTION:
            return self.attention_cooldown_ms
        elif state == AlertState.WARNING:
            return self.warning_cooldown_ms
        elif state == AlertState.CRITICAL:
            return self.critical_reminder_cooldown_ms
        return 5000.0

    def _notify_visual_listeners(self, transition: AlertTransition) -> None:
        for cb in list(self._listeners):
            try:
                cb(transition)
            except Exception as e:
                logger.error("Error in visual transition listener: %s", e)

    def _on_audio_setting_changed(self, enabled: bool) -> None:
        """Callback when user toggles audio settings."""
        self.audio_engine.set_enabled(enabled)
        # Clear announced state if disabled so re-enabling behaves cleanly
        if not enabled:
            logger.info("Audio disabled: active audio terminated and pending alerts cleared.")
