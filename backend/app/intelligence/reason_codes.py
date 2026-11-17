"""
DriveGuard — Explainable Alert Reason Codes and State Enums (Phase 6).

Defines standardized, machine-readable, and viva-defendable explainability
constructs for the driver drowsiness decision engine.
Every decision made by DriveGuard carries explicit reason codes with underlying
metric values and comparative safety thresholds.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class DrowsinessState(str, Enum):
    """
    Five-level discrete alert state hierarchy for DriveGuard.
    
    Graduated severity:
    - NORMAL (0): Driver alert and attentive; nominal physiological baselines.
    - ATTENTION (1): Early warning; single fatigue indicator or early glance deviation.
    - DROWSINESS_WARNING (2): Moderate/accumulated fatigue; elevated PERCLOS or sluggish blinks.
    - CRITICAL_WARNING (3): Imminent safety hazard; sustained closure (>=1.5s) or nodding microsleep.
    - INSUFFICIENT_DATA (-1): Vision or sensor stream unviable, face lost, or occluded.
    """
    NORMAL = "NORMAL"
    ATTENTION = "ATTENTION"
    DROWSINESS_WARNING = "DROWSINESS_WARNING"
    CRITICAL_WARNING = "CRITICAL_WARNING"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


class ReasonCode(str, Enum):
    """
    Standardized, explainable diagnostic reason codes.
    
    Aligned with Euro NCAP 2026 driver monitoring protocols, SAE J941 automotive
    ergonomics, and Dinges & Grace PERCLOS validation literature.
    """
    # Safety-Critical Urgency (Immediate Action Required)
    SUSTAINED_EYE_CLOSURE = "SUSTAINED_EYE_CLOSURE"
    NODDING_MICROSLEEP = "NODDING_MICROSLEEP"
    SEVERE_PERCLOS_CRISIS = "SEVERE_PERCLOS_CRISIS"

    # Cumulative Fatigue (High Risk)
    PERCLOS_ELEVATED = "PERCLOS_ELEVATED"
    PROLONGED_BLINK = "PROLONGED_BLINK"
    FREQUENT_YAWNING = "FREQUENT_YAWNING"
    PERSISTENT_DISTRACTION = "PERSISTENT_DISTRACTION"

    # Early Warning & Contextual Triggers (Low/Moderate Severity)
    AUTHENTIC_YAWN = "AUTHENTIC_YAWN"
    EXTENDED_CLOSURE = "EXTENDED_CLOSURE"
    SUSPECTED_HEAD_SLUMP = "SUSPECTED_HEAD_SLUMP"
    MILD_PERCLOS = "MILD_PERCLOS"
    MIRROR_CHECK_CONTEXT = "MIRROR_CHECK_CONTEXT"
    DASHBOARD_CHECK_CONTEXT = "DASHBOARD_CHECK_CONTEXT"

    # Nominal Alert State
    ALERT_NORMAL = "ALERT_NORMAL"

    # Vision & Sensor Stream Viability
    FACE_LOST = "FACE_LOST"
    SENSOR_DEGRADED = "SENSOR_DEGRADED"


@dataclass(frozen=True)
class ExplainableReason:
    """
    Fully structured, human-readable and machine-verifiable explanation
    for an alert decision.
    """
    code: ReasonCode
    message: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None

    def __str__(self) -> str:
        if self.metric_value is not None and self.threshold is not None:
            return f"[{self.code.value}] {self.message} (value: {self.metric_value:.2f}, limit: {self.threshold:.2f})"
        if self.metric_value is not None:
            return f"[{self.code.value}] {self.message} (value: {self.metric_value:.2f})"
        return f"[{self.code.value}] {self.message}"
