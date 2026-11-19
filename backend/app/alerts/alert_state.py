"""
DriveGuard — Alert State Definitions and Transition Data Structures (Phase 8).

Defines the 4 user-facing alert states (NORMAL, ATTENTION, WARNING, CRITICAL)
and immutable transition event records driven by DrowsinessAssessment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple

from backend.app.intelligence.reason_codes import DrowsinessState, ExplainableReason


class AlertState(str, Enum):
    """
    Four-level discrete user-facing alert state hierarchy for DriveGuard warnings.
    
    Graduated severity:
    - NORMAL (0): Driver attentive, normal baselines, nominal operating state.
    - ATTENTION (1): Early warning, minor fatigue indicator or distraction glance.
    - WARNING (2): Moderate fatigue detected, break recommended.
    - CRITICAL (3): Imminent danger, prolonged closure or microsleep, pull over immediately.
    """
    NORMAL = "NORMAL"
    ATTENTION = "ATTENTION"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"

    @property
    def severity(self) -> int:
        """Numeric severity level: 0 (NORMAL) to 3 (CRITICAL)."""
        if self == AlertState.NORMAL:
            return 0
        elif self == AlertState.ATTENTION:
            return 1
        elif self == AlertState.WARNING:
            return 2
        elif self == AlertState.CRITICAL:
            return 3
        return 0

    @property
    def hud_color_bgr(self) -> Tuple[int, int, int]:
        """OpenCV BGR color tuple for HUD rendering."""
        if self == AlertState.NORMAL:
            return (34, 197, 94)    # Emerald-500
        elif self == AlertState.ATTENTION:
            return (8, 179, 234)    # Amber-500 (in BGR)
        elif self == AlertState.WARNING:
            return (22, 115, 249)   # Orange-500 (in BGR)
        elif self == AlertState.CRITICAL:
            return (68, 68, 239)    # Red-500 (in BGR)
        return (148, 163, 184)


def map_drowsiness_state_to_alert_state(state: DrowsinessState) -> Optional[AlertState]:
    """
    Maps Phase 6 DrowsinessState enum to Phase 8 AlertState.
    Returns None for INSUFFICIENT_DATA as it requires context-aware handling.
    """
    if state == DrowsinessState.NORMAL:
        return AlertState.NORMAL
    elif state == DrowsinessState.ATTENTION:
        return AlertState.ATTENTION
    elif state == DrowsinessState.DROWSINESS_WARNING:
        return AlertState.WARNING
    elif state == DrowsinessState.CRITICAL_WARNING:
        return AlertState.CRITICAL
    return None


@dataclass(frozen=True)
class AlertTransition:
    """
    Immutable representation of an alert state transition or significant alert event.
    """
    timestamp_ms: float
    previous_state: AlertState
    new_state: AlertState
    severity: int
    reason: Optional[ExplainableReason] = None
    reason_code: Optional[str] = None
    is_recovery: bool = False
    alert_active: bool = False
    unresolved_alert_state: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_escalation(self) -> bool:
        """True if the new state has higher severity than the previous state."""
        return self.new_state.severity > self.previous_state.severity

    @property
    def is_deescalation(self) -> bool:
        """True if the new state has lower severity than the previous state."""
        return self.new_state.severity < self.previous_state.severity

    @property
    def reason_text(self) -> str:
        """Human-readable explanation of the alert reason."""
        if self.reason is not None:
            return str(self.reason)
        if self.reason_code is not None:
            return f"[{self.reason_code}]"
        return "Normal"
