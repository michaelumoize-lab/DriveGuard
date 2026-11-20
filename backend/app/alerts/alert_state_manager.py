"""
DriveGuard — Alert State Manager (Phase 8).

Translates authoritative DrowsinessAssessment streams into stable, explainable
user-facing alert states (NORMAL, ATTENTION, WARNING, CRITICAL).

Enforces:
1. Priority-preserving transitions (CRITICAL > WARNING > ATTENTION > NORMAL).
2. Phase 6 alert continuity and cooldown compliance.
3. No premature downgrading of active unresolved alerts.
4. Safe handling of face-loss grace, tracking interruptions, and invalid telemetry.
5. Deduplication of identical assessments to eliminate alert chattering.
6. Explicit recovery tracking and structured transition logging.
"""

from __future__ import annotations

import logging
import math
from typing import List, Optional

from backend.app.alerts.alert_state import (
    AlertState,
    AlertTransition,
    map_drowsiness_state_to_alert_state,
)
from backend.app.intelligence.drowsiness_engine import DrowsinessAssessment
from backend.app.intelligence.reason_codes import DrowsinessState, ExplainableReason, ReasonCode

logger = logging.getLogger("driveguard.alert_manager")


class AlertStateManager:
    """
    Dedicated state manager translating intelligence assessments into reliable,
    stable, and explainable user-facing alerts.
    """

    def __init__(self, max_history_size: int = 1000) -> None:
        self.max_history_size = max_history_size
        self._current_state: AlertState = AlertState.NORMAL
        self._active_alert_level: Optional[AlertState] = None
        self._is_recovering: bool = False
        self._last_reason: Optional[ExplainableReason] = None
        self._last_assessment_ts_ms: Optional[float] = None
        self._last_transition_ts_ms: Optional[float] = None
        self._tracking_available: bool = True
        self._history: List[AlertTransition] = []

    @property
    def current_state(self) -> AlertState:
        """Current active user-facing alert state."""
        return self._current_state

    @property
    def severity(self) -> int:
        """Current numeric severity (0 to 3)."""
        return self._current_state.severity

    @property
    def is_recovering(self) -> bool:
        """True if an active alert is currently undergoing recovery cooldown."""
        return self._is_recovering

    @property
    def is_alert_active(self) -> bool:
        """True if an alert (WARNING or CRITICAL) is actively uncompleted."""
        return self._active_alert_level is not None or self._current_state in (
            AlertState.WARNING,
            AlertState.CRITICAL,
        )

    @property
    def tracking_available(self) -> bool:
        """True if vision/tracking sensors are viable and delivering observations."""
        return self._tracking_available

    @property
    def active_reason(self) -> Optional[ExplainableReason]:
        """Most recent or historical explainable reason for the active alert."""
        return self._last_reason

    @property
    def history(self) -> List[AlertTransition]:
        """Read-only copy of transition history."""
        return list(self._history)

    def update(self, assessment: Optional[DrowsinessAssessment]) -> Optional[AlertTransition]:
        """
        Ingest a new DrowsinessAssessment from the intelligence pipeline.

        Returns an AlertTransition if a state transition or significant alert lifecycle
        event occurred; returns None if the state remains unchanged or assessment is invalid.
        """
        # 1. Safe handling of missing or invalid assessments
        if assessment is None:
            return None

        ts_ms = assessment.timestamp_ms
        if not math.isfinite(ts_ms) or ts_ms < 0.0:
            return None

        if self._last_assessment_ts_ms is not None and ts_ms < self._last_assessment_ts_ms:
            # Monotonicity check: reject backward timestamps
            return None

        self._last_assessment_ts_ms = ts_ms

        # 2. Sensor and face loss handling (Phase 6 compliance)
        if assessment.state == DrowsinessState.INSUFFICIENT_DATA:
            self._tracking_available = False
            # If an alert was active, preserve active alert continuity.
            # Never downgrade to NORMAL on face loss.
            if self._active_alert_level is not None:
                self._is_recovering = False
            return None

        self._tracking_available = True

        # 3. Determine candidate alert level from assessment
        candidate_alert = map_drowsiness_state_to_alert_state(assessment.state)
        if candidate_alert is None:
            candidate_alert = AlertState.NORMAL

        # Map unresolved alert state if present
        unresolved_mapped: Optional[AlertState] = None
        if assessment.unresolved_alert_state is not None:
            unresolved_mapped = map_drowsiness_state_to_alert_state(assessment.unresolved_alert_state)

        # 4. Alert Lifecycle & Priority Arbitration
        prev_state = self._current_state
        target_state = self._current_state
        is_recovery_event = False

        if assessment.alert_active:
            # An alert (WARNING or CRITICAL) is actively uncompleted
            # Base active level is the higher of unresolved level or candidate
            active_candidate = candidate_alert
            if unresolved_mapped is not None and unresolved_mapped.severity > active_candidate.severity:
                active_candidate = unresolved_mapped

            # Guard: Do not downgrade a higher-priority active alert
            # (e.g. CRITICAL cannot be downgraded to WARNING simply because candidate is WARNING)
            if self._active_alert_level is not None:
                if active_candidate.severity > self._active_alert_level.severity:
                    self._active_alert_level = active_candidate
            else:
                self._active_alert_level = active_candidate

            target_state = self._active_alert_level

            # Recovery status comes authoritatively from DecisionEngine's Timer 2.
            # Never mark recovering while eyes are closed or unverified.
            if assessment.is_alert_recovering is not None:
                self._is_recovering = bool(assessment.is_alert_recovering)
            else:
                eye_closed = (
                    assessment.temporal_metrics.is_eye_closed
                    if assessment.temporal_metrics is not None
                    else None
                )
                self._is_recovering = (
                    candidate_alert == AlertState.NORMAL
                    and eye_closed is not True
                )

            # Update historical/active reason
            if assessment.primary_reason is not None and candidate_alert.severity >= target_state.severity:
                self._last_reason = assessment.primary_reason
            elif assessment.historical_reason is not None:
                self._last_reason = assessment.historical_reason
            elif assessment.primary_reason is not None and self._last_reason is None:
                self._last_reason = assessment.primary_reason

        else:
            # Alert is NOT active (alert_active == False)
            if self._active_alert_level is not None:
                # Active alert was completed through verified recovery cooldown
                is_recovery_event = True
                self._active_alert_level = None
                self._is_recovering = False
                target_state = candidate_alert
                self._last_reason = assessment.primary_reason
            else:
                # Normal or Attention tracking
                if prev_state == AlertState.ATTENTION and candidate_alert == AlertState.NORMAL:
                    is_recovery_event = True
                    self._is_recovering = False
                elif candidate_alert.severity > prev_state.severity:
                    # Escalation to Attention
                    self._is_recovering = False
                
                target_state = candidate_alert
                self._last_reason = assessment.primary_reason

        # 5. Check for State Transition or Recovery Event
        if target_state != prev_state or is_recovery_event:
            transition = AlertTransition(
                timestamp_ms=ts_ms,
                previous_state=prev_state,
                new_state=target_state,
                severity=target_state.severity,
                reason=self._last_reason,
                reason_code=self._last_reason.code.value if self._last_reason else None,
                is_recovery=is_recovery_event,
                alert_active=assessment.alert_active,
                unresolved_alert_state=self._active_alert_level.value if self._active_alert_level else None,
                details={
                    "confidence": assessment.confidence.overall_confidence if assessment.confidence else 1.0,
                    "perclos_pct": assessment.temporal_metrics.perclos_pct if assessment.temporal_metrics else 0.0,
                    "is_mirror_check": assessment.is_mirror_check_context,
                },
            )

            self._current_state = target_state
            self._last_transition_ts_ms = ts_ms
            self._record_transition(transition)

            logger.info(
                "AlertTransition: %s -> %s (Sev %d, Recovery=%s, Reason=%s)",
                prev_state.value,
                target_state.value,
                target_state.severity,
                is_recovery_event,
                transition.reason_code,
            )
            return transition

        # State remained identical and no recovery event: suppress duplicate
        return None

    def force_recovery(self, timestamp_ms: float) -> Optional[AlertTransition]:
        """
        Explicitly reset active alerts to NORMAL (e.g. driver acknowledgment or test reset).
        """
        if self._current_state == AlertState.NORMAL and self._active_alert_level is None:
            return None

        prev_state = self._current_state
        self._current_state = AlertState.NORMAL
        self._active_alert_level = None
        self._is_recovering = False
        self._last_reason = ExplainableReason(
            code=ReasonCode.ALERT_NORMAL,
            message="Alert status returned to normal by explicit recovery.",
        )

        transition = AlertTransition(
            timestamp_ms=timestamp_ms,
            previous_state=prev_state,
            new_state=AlertState.NORMAL,
            severity=0,
            reason=self._last_reason,
            reason_code="MANUAL_RECOVERY",
            is_recovery=True,
            alert_active=False,
            unresolved_alert_state=None,
        )
        self._record_transition(transition)
        return transition

    def reset(self) -> None:
        """Reset internal state machine for a new monitoring session."""
        self._current_state = AlertState.NORMAL
        self._active_alert_level = None
        self._is_recovering = False
        self._last_reason = None
        self._last_assessment_ts_ms = None
        self._last_transition_ts_ms = None
        self._tracking_available = True
        self._history.clear()

    def _record_transition(self, transition: AlertTransition) -> None:
        """Append transition to history, maintaining maximum bounded size."""
        self._history.append(transition)
        if len(self._history) > self.max_history_size:
            self._history.pop(0)
