"""
DriveGuard — Unified Baseline Drowsiness Intelligence Engine (Phase 6).

Coordinates the multi-signal confidence evaluation, multi-window temporal analysis,
and deterministic decision engine to produce immutable, explainable drowsiness assessments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from backend.app.camera.frame_quality import FrameQualityResult
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.intelligence.confidence import (
    ConfidenceAssessment,
    SignalConfidenceEvaluator,
)
from backend.app.intelligence.decision_engine import DecisionEngine, DecisionResult
from backend.app.intelligence.reason_codes import DrowsinessState, ExplainableReason
from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer, TemporalMetrics


@dataclass(frozen=True)
class DrowsinessAssessment:
    """
    Immutable comprehensive result emitted for each frame by the DrowsinessEngine.
    """
    timestamp_ms: float
    state: DrowsinessState
    severity: int
    decision: DecisionResult
    temporal_metrics: TemporalMetrics
    confidence: ConfidenceAssessment
    is_mirror_check_context: bool

    @property
    def primary_reason(self) -> Optional[ExplainableReason]:
        return self.decision.primary_reason

    @property
    def primary_reason_text(self) -> str:
        if self.decision.primary_reason is not None:
            return str(self.decision.primary_reason)
        return "Normal"

    @property
    def status_color_bgr(self) -> Tuple[int, int, int]:
        """OpenCV BGR color mapping for HUD display."""
        if self.state == DrowsinessState.NORMAL:
            return (34, 197, 94)    # Emerald-500 (Green)
        elif self.state == DrowsinessState.ATTENTION:
            return (234, 179, 8)    # Amber-500 (Yellow)
        elif self.state == DrowsinessState.DROWSINESS_WARNING:
            return (249, 115, 22)   # Orange-500
        elif self.state == DrowsinessState.CRITICAL_WARNING:
            return (239, 68, 68)    # Red-500 (Alert)
        else:  # INSUFFICIENT_DATA
            return (148, 163, 184)  # Slate-400


class DrowsinessEngine:
    """
    High-level orchestrator for the DriveGuard Phase 6 intelligence subsystem.
    """

    def __init__(
        self,
        sustained_closure_threshold_ms: float = 1500.0,
        extended_closure_threshold_ms: float = 500.0,
        perclos_warning_threshold_pct: float = 15.0,
        perclos_attention_threshold_pct: float = 8.0,
        prolonged_blink_threshold_ms: float = 350.0,
        off_road_distraction_limit_ms: float = 3500.0,
        recovery_critical_ms: float = 2000.0,
        recovery_warning_ms: float = 1500.0,
        recovery_attention_ms: float = 1000.0,
        face_loss_grace_ms: float = 1000.0,
        sensor_unreliable_grace_ms: float = 1000.0,
    ) -> None:
        self.temporal_analyzer = TemporalAnalyzer()
        self.confidence_evaluator = SignalConfidenceEvaluator()
        self.decision_engine = DecisionEngine(
            sustained_closure_threshold_ms=sustained_closure_threshold_ms,
            extended_closure_threshold_ms=extended_closure_threshold_ms,
            perclos_warning_threshold_pct=perclos_warning_threshold_pct,
            perclos_attention_threshold_pct=perclos_attention_threshold_pct,
            prolonged_blink_threshold_ms=prolonged_blink_threshold_ms,
            off_road_distraction_limit_ms=off_road_distraction_limit_ms,
            recovery_critical_ms=recovery_critical_ms,
            recovery_warning_ms=recovery_warning_ms,
            recovery_attention_ms=recovery_attention_ms,
            face_loss_grace_ms=face_loss_grace_ms,
            sensor_unreliable_grace_ms=sensor_unreliable_grace_ms,
        )

    def evaluate(
        self,
        features: DriverFeatures,
        quality: Optional[FrameQualityResult] = None,
    ) -> DrowsinessAssessment:
        """
        Execute full intelligence arbitration for a single frame.
        """
        # 1. Evaluate upstream sensing confidence
        confidence_res = self.confidence_evaluator.evaluate(features=features, quality=quality)

        # 2. Accumulate temporal observations across sliding windows
        temporal_res = self.temporal_analyzer.update(features=features)

        # 3. Arbitrate alert state through deterministic decision engine
        decision_res = self.decision_engine.evaluate(
            metrics=temporal_res,
            confidence=confidence_res,
        )

        return DrowsinessAssessment(
            timestamp_ms=features.timestamp_ms,
            state=decision_res.state,
            severity=decision_res.severity,
            decision=decision_res,
            temporal_metrics=temporal_res,
            confidence=confidence_res,
            is_mirror_check_context=decision_res.is_mirror_check_context,
        )

    def reset(self) -> None:
        """Reset all internal state across temporal analyzer, confidence, and decision engine."""
        self.temporal_analyzer.reset()
        self.confidence_evaluator.reset()
        self.decision_engine.reset()
