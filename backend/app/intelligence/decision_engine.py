"""
DriveGuard — Deterministic Multi-Indicator Decision Engine with Hysteresis (Phase 6).

Implements a 5-level deterministic state machine:
  `NORMAL` -> `ATTENTION` -> `DROWSINESS_WARNING` -> `CRITICAL_WARNING` / `INSUFFICIENT_DATA`
with strict safety-critical urgency, explainable reason codes, and asymmetric hysteresis
to eliminate alert chattering and alarm fatigue.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from backend.app.intelligence.confidence import ConfidenceAssessment
from backend.app.intelligence.reason_codes import (
    DrowsinessState,
    ExplainableReason,
    ReasonCode,
)
from backend.app.intelligence.temporal_analyzer import TemporalMetrics


@dataclass(frozen=True)
class DecisionResult:
    """
    Final evaluated state, severity, and explainable reasons for a single frame.
    """
    timestamp_ms: float
    state: DrowsinessState
    severity: int  # 0: NORMAL, 1: ATTENTION, 2: WARNING, 3: CRITICAL, -1: INSUFFICIENT_DATA
    reasons: List[ExplainableReason] = field(default_factory=list)
    primary_reason: Optional[ExplainableReason] = None
    is_mirror_check_context: bool = False

    @property
    def summary_text(self) -> str:
        code_str = self.primary_reason.code.value if self.primary_reason else "NONE"
        return f"[{self.state.value} (Sev {self.severity}) | {code_str}]"


class DecisionEngine:
    """
    Stateful decision arbiter enforcing priority rules, hysteresis, and explainability.
    """

    SEVERITY_MAP = {
        DrowsinessState.NORMAL: 0,
        DrowsinessState.ATTENTION: 1,
        DrowsinessState.DROWSINESS_WARNING: 2,
        DrowsinessState.CRITICAL_WARNING: 3,
        DrowsinessState.INSUFFICIENT_DATA: -1,
    }

    def __init__(
        self,
        sustained_closure_threshold_ms: float = 1500.0,
        extended_closure_threshold_ms: float = 500.0,
        perclos_warning_threshold_pct: float = 15.0,
        perclos_attention_threshold_pct: float = 8.0,
        severe_perclos_threshold_pct: float = 30.0,
        prolonged_blink_threshold_ms: float = 350.0,
        off_road_distraction_limit_ms: float = 3500.0,
        recovery_critical_ms: float = 2000.0,
        recovery_warning_ms: float = 1500.0,
        recovery_attention_ms: float = 1000.0,
        face_loss_grace_ms: float = 1000.0,
    ) -> None:
        self.sustained_closure_threshold_ms = sustained_closure_threshold_ms
        self.extended_closure_threshold_ms = extended_closure_threshold_ms
        self.perclos_warning_threshold_pct = perclos_warning_threshold_pct
        self.perclos_attention_threshold_pct = perclos_attention_threshold_pct
        self.severe_perclos_threshold_pct = severe_perclos_threshold_pct
        self.prolonged_blink_threshold_ms = prolonged_blink_threshold_ms
        self.off_road_distraction_limit_ms = off_road_distraction_limit_ms

        self.recovery_critical_ms = recovery_critical_ms
        self.recovery_warning_ms = recovery_warning_ms
        self.recovery_attention_ms = recovery_attention_ms
        self.face_loss_grace_ms = face_loss_grace_ms

        # Active FSM State
        self.current_state: DrowsinessState = DrowsinessState.NORMAL
        self._recovery_start_ts_ms: Optional[float] = None
        self._stable_face_start_ts_ms: Optional[float] = None

    def evaluate(
        self,
        metrics: TemporalMetrics,
        confidence: ConfidenceAssessment,
    ) -> DecisionResult:
        """
        Evaluate candidate state against priority rules and apply hysteresis guards.
        """
        ts_ms = metrics.timestamp_ms
        reasons: List[ExplainableReason] = []

        # ------------------------------------------------------------------
        # STEP 1: Sensor & Face Viability Evaluation
        # ------------------------------------------------------------------
        is_face_lost = not metrics.has_face
        is_prolonged_face_loss = is_face_lost and (metrics.face_lost_duration_ms >= self.face_loss_grace_ms)
        is_sensor_unreliable = not confidence.is_reliable and not is_face_lost

        if is_prolonged_face_loss or is_sensor_unreliable:
            self._stable_face_start_ts_ms = None
            candidate_state = DrowsinessState.INSUFFICIENT_DATA
            primary = ExplainableReason(
                code=ReasonCode.FACE_LOST if is_prolonged_face_loss else ReasonCode.SENSOR_DEGRADED,
                message=(
                    f"Driver face not detected for {metrics.face_lost_duration_ms:.0f}ms"
                    if is_prolonged_face_loss
                    else "Sensor reliability compromised"
                ),
                metric_value=metrics.face_lost_duration_ms if is_prolonged_face_loss else confidence.overall_confidence,
                threshold=self.face_loss_grace_ms if is_prolonged_face_loss else 0.50,
            )
            reasons.append(primary)
            self.current_state = candidate_state
            return DecisionResult(
                timestamp_ms=ts_ms,
                state=candidate_state,
                severity=self.SEVERITY_MAP[candidate_state],
                reasons=reasons,
                primary_reason=primary,
                is_mirror_check_context=False,
            )

        if is_face_lost:
            # Face loss within grace period: hold current state without triggering false alarms
            primary = ExplainableReason(
                code=ReasonCode.FACE_LOST,
                message=f"Face loss grace period active ({metrics.face_lost_duration_ms:.0f}ms)",
                metric_value=metrics.face_lost_duration_ms,
                threshold=self.face_loss_grace_ms,
            )
            return DecisionResult(
                timestamp_ms=ts_ms,
                state=self.current_state,
                severity=self.SEVERITY_MAP[self.current_state],
                reasons=[primary],
                primary_reason=primary,
                is_mirror_check_context=False,
            )

        # Track stable face presence recovery
        if self._stable_face_start_ts_ms is None:
            self._stable_face_start_ts_ms = ts_ms

        # ------------------------------------------------------------------
        # STEP 2: Candidate State Arbitration (Highest Severity First)
        # ------------------------------------------------------------------
        candidate_state = DrowsinessState.NORMAL

        # Tier 3: CRITICAL_WARNING (Safety Critical Urgency)
        if metrics.active_closure_duration_ms >= self.sustained_closure_threshold_ms:
            candidate_state = DrowsinessState.CRITICAL_WARNING
            reasons.append(
                ExplainableReason(
                    code=ReasonCode.SUSTAINED_EYE_CLOSURE,
                    message=f"Continuous eye closure of {metrics.active_closure_duration_ms:.0f}ms exceeds safety limit",
                    metric_value=metrics.active_closure_duration_ms,
                    threshold=self.sustained_closure_threshold_ms,
                )
            )

        if metrics.is_nodding_microsleep:
            candidate_state = DrowsinessState.CRITICAL_WARNING
            reasons.append(
                ExplainableReason(
                    code=ReasonCode.NODDING_MICROSLEEP,
                    message="Head nodding slump detected during active eye closure",
                    metric_value=metrics.active_closure_duration_ms,
                    threshold=800.0,
                )
            )

        if (
            metrics.perclos_is_reliable
            and metrics.perclos_pct >= self.severe_perclos_threshold_pct
        ):
            candidate_state = DrowsinessState.CRITICAL_WARNING
            reasons.append(
                ExplainableReason(
                    code=ReasonCode.SEVERE_PERCLOS_CRISIS,
                    message=f"Critical PERCLOS accumulation of {metrics.perclos_pct:.1f}% exceeds emergency limit",
                    metric_value=metrics.perclos_pct,
                    threshold=self.severe_perclos_threshold_pct,
                )
            )

        # Tier 2: DROWSINESS_WARNING (Cumulative Fatigue)
        if candidate_state != DrowsinessState.CRITICAL_WARNING:
            if (
                metrics.perclos_is_reliable
                and metrics.perclos_pct >= self.perclos_warning_threshold_pct
            ):
                candidate_state = DrowsinessState.DROWSINESS_WARNING
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.PERCLOS_ELEVATED,
                        message=f"Rolling 60s PERCLOS of {metrics.perclos_pct:.1f}% indicates driver drowsiness",
                        metric_value=metrics.perclos_pct,
                        threshold=self.perclos_warning_threshold_pct,
                    )
                )

            if (
                metrics.prolonged_blink_count_60s >= 3
                or (metrics.mean_blink_duration_ms_60s >= self.prolonged_blink_threshold_ms and metrics.blink_count_60s >= 3)
            ):
                candidate_state = DrowsinessState.DROWSINESS_WARNING
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.PROLONGED_BLINK,
                        message=f"Sluggish blink kinematics ({metrics.prolonged_blink_count_60s} prolonged blinks, mean {metrics.mean_blink_duration_ms_60s:.0f}ms)",
                        metric_value=metrics.mean_blink_duration_ms_60s,
                        threshold=self.prolonged_blink_threshold_ms,
                    )
                )

            if metrics.yawn_count_60s >= 2 and (metrics.perclos_pct >= 8.0 or metrics.prolonged_blink_count_60s >= 1):
                candidate_state = DrowsinessState.DROWSINESS_WARNING
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.FREQUENT_YAWNING,
                        message=f"Repeated yawns ({metrics.yawn_count_60s} in 60s) combined with fatigue kinematics",
                        metric_value=float(metrics.yawn_count_60s),
                        threshold=2.0,
                    )
                )

            if (
                metrics.active_off_road_dwell_ms >= self.off_road_distraction_limit_ms
                and not metrics.is_mirror_check_context
            ):
                candidate_state = DrowsinessState.DROWSINESS_WARNING
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.PERSISTENT_DISTRACTION,
                        message=f"Off-road gaze dwell of {metrics.active_off_road_dwell_ms / 1000.0:.1f}s exceeds safety threshold",
                        metric_value=metrics.active_off_road_dwell_ms,
                        threshold=self.off_road_distraction_limit_ms,
                    )
                )

        # Tier 1: ATTENTION (Early Warning / Context)
        if candidate_state not in (DrowsinessState.CRITICAL_WARNING, DrowsinessState.DROWSINESS_WARNING):
            if metrics.active_closure_duration_ms >= self.extended_closure_threshold_ms:
                candidate_state = DrowsinessState.ATTENTION
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.EXTENDED_CLOSURE,
                        message=f"Extended eye closure of {metrics.active_closure_duration_ms:.0f}ms",
                        metric_value=metrics.active_closure_duration_ms,
                        threshold=self.extended_closure_threshold_ms,
                    )
                )

            if metrics.yawn_count_30s >= 1 or metrics.active_yawn_duration_ms >= 1500.0:
                candidate_state = DrowsinessState.ATTENTION
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.AUTHENTIC_YAWN,
                        message="Authentic yawn detected",
                        metric_value=max(metrics.active_yawn_duration_ms, 1500.0),
                        threshold=1500.0,
                    )
                )

            if (
                metrics.perclos_is_reliable
                and metrics.perclos_pct >= self.perclos_attention_threshold_pct
            ):
                candidate_state = DrowsinessState.ATTENTION
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.MILD_PERCLOS,
                        message=f"Mildly elevated PERCLOS of {metrics.perclos_pct:.1f}% indicates emerging fatigue",
                        metric_value=metrics.perclos_pct,
                        threshold=self.perclos_attention_threshold_pct,
                    )
                )

            # Contextual Glances
            if metrics.is_mirror_check_context:
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.MIRROR_CHECK_CONTEXT,
                        message="Normal driving mirror check",
                        metric_value=metrics.active_off_road_dwell_ms,
                        threshold=3000.0,
                    )
                )
            elif metrics.is_dashboard_check_context:
                reasons.append(
                    ExplainableReason(
                        code=ReasonCode.DASHBOARD_CHECK_CONTEXT,
                        message="Instrument cluster glance",
                        metric_value=metrics.active_off_road_dwell_ms,
                        threshold=2000.0,
                    )
                )

        # Tier 0: NORMAL (Nominal Alert Driving)
        if candidate_state == DrowsinessState.NORMAL and not reasons:
            reasons.append(
                ExplainableReason(
                    code=ReasonCode.ALERT_NORMAL,
                    message="Driver alert; gaze and eye metrics within nominal ranges",
                )
            )

        # ------------------------------------------------------------------
        # STEP 3: Hysteresis and Persistence State Machine
        # ------------------------------------------------------------------
        current_sev = self.SEVERITY_MAP[self.current_state]
        candidate_sev = self.SEVERITY_MAP[candidate_state]

        # Case A: Escalation (Immediate or Zero-Delay for Critical)
        if candidate_sev > current_sev:
            self.current_state = candidate_state
            self._recovery_start_ts_ms = None

        # Case B: De-escalation (Requires Continuous Recovery Cooldown)
        elif candidate_sev < current_sev:
            required_cooldown = self.recovery_attention_ms
            if current_sev == 3:  # CRITICAL -> lower
                required_cooldown = self.recovery_critical_ms
            elif current_sev == 2:  # WARNING -> lower
                required_cooldown = self.recovery_warning_ms

            if self._recovery_start_ts_ms is None:
                self._recovery_start_ts_ms = ts_ms

            elapsed_recovery = ts_ms - self._recovery_start_ts_ms
            if elapsed_recovery >= required_cooldown:
                # Cooldown satisfied, safely downgrade state
                self.current_state = candidate_state
                self._recovery_start_ts_ms = None
            else:
                # Still within cooldown; hold elevated state to prevent alert flapping
                pass

        # Case C: State matches candidate
        else:
            self._recovery_start_ts_ms = None

        primary_reason = reasons[0] if reasons else None

        return DecisionResult(
            timestamp_ms=ts_ms,
            state=self.current_state,
            severity=self.SEVERITY_MAP[self.current_state],
            reasons=reasons,
            primary_reason=primary_reason,
            is_mirror_check_context=metrics.is_mirror_check_context,
        )

    def reset(self) -> None:
        """Reset decision state for a new session."""
        self.current_state = DrowsinessState.NORMAL
        self._recovery_start_ts_ms = None
        self._stable_face_start_ts_ms = None
