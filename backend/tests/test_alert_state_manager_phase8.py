"""
DriveGuard — Phase 8: Alert State Manager Automated Unit Tests.

Validates:
1. NORMAL to ATTENTION transition.
2. ATTENTION to WARNING transition.
3. WARNING to CRITICAL transition.
4. CRITICAL continuity: CRITICAL alert is maintained.
5. Severity downgrade handling: Active higher alert is not downgraded by lower candidate.
6. Recovery: Orderly recovery to NORMAL when alert_active resolves.
7. Repeated identical assessments: Deduplication prevents repeated transition events.
8. Missing or invalid assessments: Gracefully handled without state corruption or crash.
9. Face-loss continuity: Face loss holds active alert without reporting driver safe.
10. Existing Phase 6 recovery and cooldown compatibility: Follows DrowsinessEngine decisions.
"""

from __future__ import annotations

import math
from typing import Optional
import pytest

from backend.app.alerts.alert_state import AlertState, AlertTransition
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.intelligence.confidence import ConfidenceAssessment, ConfidenceTier
from backend.app.intelligence.decision_engine import DecisionResult
from backend.app.intelligence.drowsiness_engine import DrowsinessAssessment
from backend.app.intelligence.reason_codes import (
    DrowsinessState,
    ExplainableReason,
    ReasonCode,
)
from backend.app.intelligence.temporal_analyzer import TemporalMetrics


def make_assessment(
    timestamp_ms: float,
    state: DrowsinessState,
    severity: int,
    alert_active: bool = False,
    unresolved_alert_state: Optional[DrowsinessState] = None,
    reason_code: ReasonCode = ReasonCode.ALERT_NORMAL,
    reason_message: str = "Test reason",
    has_face: bool = True,
) -> DrowsinessAssessment:
    """Helper to synthesize deterministic DrowsinessAssessment fixtures."""
    reason = ExplainableReason(code=reason_code, message=reason_message)
    dec = DecisionResult(
        timestamp_ms=timestamp_ms,
        state=state,
        severity=severity,
        reasons=[reason],
        primary_reason=reason,
        alert_active=alert_active,
        unresolved_alert_state=unresolved_alert_state,
        historical_reason=reason if alert_active else None,
    )
    metrics = TemporalMetrics(
        timestamp_ms=timestamp_ms,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=0.0,
        is_mirror_check_context=False,
        is_dashboard_check_context=False,
        is_nodding_microsleep=False,
        blink_count_60s=0,
        blink_count_30s=0,
        mean_blink_duration_ms_60s=0.0,
        prolonged_blink_count_60s=0,
        severe_blink_count_60s=0,
        mean_rcr_60s=1.0,
        yawn_count_60s=0,
        yawn_count_30s=0,
        active_yawn_duration_ms=0.0,
        perclos_pct=0.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=has_face,
        face_lost_duration_ms=0.0 if has_face else 1200.0,
    )
    conf = ConfidenceAssessment(
        overall_confidence=0.95,
        tier=ConfidenceTier.HIGH,
        is_reliable=True,
    )
    return DrowsinessAssessment(
        timestamp_ms=timestamp_ms,
        state=state,
        severity=severity,
        decision=dec,
        temporal_metrics=metrics,
        confidence=conf,
        is_mirror_check_context=False,
        alert_active=alert_active,
        unresolved_alert_state=unresolved_alert_state,
        historical_reason=reason if alert_active else None,
    )


class TestAlertStateManager:
    """Comprehensive test suite for AlertStateManager."""

    def test_initial_state_is_normal(self) -> None:
        mgr = AlertStateManager()
        assert mgr.current_state == AlertState.NORMAL
        assert mgr.severity == 0
        assert not mgr.is_alert_active
        assert not mgr.is_recovering

    def test_normal_to_attention_transition(self) -> None:
        mgr = AlertStateManager()
        asm = make_assessment(
            timestamp_ms=1000.0,
            state=DrowsinessState.ATTENTION,
            severity=1,
            reason_code=ReasonCode.AUTHENTIC_YAWN,
        )
        trans = mgr.update(asm)
        assert trans is not None
        assert trans.previous_state == AlertState.NORMAL
        assert trans.new_state == AlertState.ATTENTION
        assert trans.severity == 1
        assert trans.is_escalation
        assert mgr.current_state == AlertState.ATTENTION

    def test_attention_to_warning_transition(self) -> None:
        mgr = AlertStateManager()
        # Step 1: Attention
        mgr.update(make_assessment(1000.0, DrowsinessState.ATTENTION, 1))
        # Step 2: Warning
        asm = make_assessment(
            timestamp_ms=2000.0,
            state=DrowsinessState.DROWSINESS_WARNING,
            severity=2,
            alert_active=True,
            unresolved_alert_state=DrowsinessState.DROWSINESS_WARNING,
            reason_code=ReasonCode.PERCLOS_ELEVATED,
        )
        trans = mgr.update(asm)
        assert trans is not None
        assert trans.previous_state == AlertState.ATTENTION
        assert trans.new_state == AlertState.WARNING
        assert trans.severity == 2
        assert trans.is_escalation
        assert mgr.current_state == AlertState.WARNING
        assert mgr.is_alert_active

    def test_warning_to_critical_transition(self) -> None:
        mgr = AlertStateManager()
        # Step 1: Warning
        mgr.update(
            make_assessment(
                1000.0,
                DrowsinessState.DROWSINESS_WARNING,
                2,
                alert_active=True,
                unresolved_alert_state=DrowsinessState.DROWSINESS_WARNING,
            )
        )
        # Step 2: Critical
        asm = make_assessment(
            timestamp_ms=2500.0,
            state=DrowsinessState.CRITICAL_WARNING,
            severity=3,
            alert_active=True,
            unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
            reason_code=ReasonCode.SUSTAINED_EYE_CLOSURE,
        )
        trans = mgr.update(asm)
        assert trans is not None
        assert trans.previous_state == AlertState.WARNING
        assert trans.new_state == AlertState.CRITICAL
        assert trans.severity == 3
        assert trans.is_escalation
        assert mgr.current_state == AlertState.CRITICAL

    def test_critical_continuity_and_no_downgrade(self) -> None:
        """A lower-priority assessment must not downgrade an active unresolved CRITICAL alert."""
        mgr = AlertStateManager()
        mgr.update(
            make_assessment(
                1000.0,
                DrowsinessState.CRITICAL_WARNING,
                3,
                alert_active=True,
                unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
                reason_code=ReasonCode.SUSTAINED_EYE_CLOSURE,
            )
        )
        assert mgr.current_state == AlertState.CRITICAL

        # Incoming assessment drops to WARNING, but unresolved is still CRITICAL
        asm2 = make_assessment(
            timestamp_ms=1500.0,
            state=DrowsinessState.DROWSINESS_WARNING,
            severity=2,
            alert_active=True,
            unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
            reason_code=ReasonCode.PERCLOS_ELEVATED,
        )
        trans2 = mgr.update(asm2)
        # State must remain CRITICAL without flipping or generating a spurious downgrade
        assert trans2 is None
        assert mgr.current_state == AlertState.CRITICAL

    def test_recovery_flow_when_alert_resolves(self) -> None:
        """When recovery cooldown finishes and alert_active becomes False, state returns to NORMAL."""
        mgr = AlertStateManager()
        # Trigger Warning
        mgr.update(
            make_assessment(
                1000.0,
                DrowsinessState.DROWSINESS_WARNING,
                2,
                alert_active=True,
                unresolved_alert_state=DrowsinessState.DROWSINESS_WARNING,
            )
        )
        assert mgr.current_state == AlertState.WARNING

        # Eyes open, recovery underway (candidate NORMAL, alert_active True)
        asm_rec = make_assessment(
            timestamp_ms=2000.0,
            state=DrowsinessState.NORMAL,
            severity=0,
            alert_active=True,
            unresolved_alert_state=DrowsinessState.DROWSINESS_WARNING,
        )
        trans_rec = mgr.update(asm_rec)
        assert trans_rec is None  # Still within recovery cooldown; hold WARNING
        assert mgr.is_recovering

        # Recovery cooldown finishes (alert_active becomes False)
        asm_done = make_assessment(
            timestamp_ms=3600.0,
            state=DrowsinessState.NORMAL,
            severity=0,
            alert_active=False,
            unresolved_alert_state=None,
        )
        trans_done = mgr.update(asm_done)
        assert trans_done is not None
        assert trans_done.previous_state == AlertState.WARNING
        assert trans_done.new_state == AlertState.NORMAL
        assert trans_done.is_recovery
        assert mgr.current_state == AlertState.NORMAL
        assert not mgr.is_recovering

    def test_repeated_identical_assessments_are_deduplicated(self) -> None:
        """Repeated identical assessments must not generate repeated transition events."""
        mgr = AlertStateManager()
        asm1 = make_assessment(1000.0, DrowsinessState.ATTENTION, 1)
        t1 = mgr.update(asm1)
        assert t1 is not None

        # Repeat identical Attention assessment
        asm2 = make_assessment(1033.0, DrowsinessState.ATTENTION, 1)
        t2 = mgr.update(asm2)
        assert t2 is None  # Suppressed

        asm3 = make_assessment(1066.0, DrowsinessState.ATTENTION, 1)
        t3 = mgr.update(asm3)
        assert t3 is None  # Suppressed

    def test_missing_or_invalid_assessments_handled_safely(self) -> None:
        mgr = AlertStateManager()
        assert mgr.update(None) is None

        # Invalid timestamp
        asm_nan = make_assessment(float("nan"), DrowsinessState.ATTENTION, 1)
        assert mgr.update(asm_nan) is None

        asm_neg = make_assessment(-100.0, DrowsinessState.ATTENTION, 1)
        assert mgr.update(asm_neg) is None

        # Monotonicity: backward timestamp
        asm_valid = make_assessment(2000.0, DrowsinessState.ATTENTION, 1)
        assert mgr.update(asm_valid) is not None

        asm_backward = make_assessment(1500.0, DrowsinessState.ATTENTION, 1)
        assert mgr.update(asm_backward) is None
        assert mgr.current_state == AlertState.ATTENTION

    def test_face_loss_preserves_active_alert_continuity(self) -> None:
        """Face loss (INSUFFICIENT_DATA) must preserve active alert continuity and not report safe."""
        mgr = AlertStateManager()
        # Trigger Critical Alert
        mgr.update(
            make_assessment(
                1000.0,
                DrowsinessState.CRITICAL_WARNING,
                3,
                alert_active=True,
                unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
            )
        )
        assert mgr.current_state == AlertState.CRITICAL

        # Face Lost prolonged (>= 1000ms grace)
        asm_lost = make_assessment(
            2200.0,
            DrowsinessState.INSUFFICIENT_DATA,
            -1,
            alert_active=True,
            unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
            has_face=False,
        )
        trans_lost = mgr.update(asm_lost)
        assert trans_lost is None
        # Must maintain CRITICAL alert state
        assert mgr.current_state == AlertState.CRITICAL
        assert not mgr.tracking_available

    def test_force_recovery(self) -> None:
        mgr = AlertStateManager()
        mgr.update(
            make_assessment(
                1000.0,
                DrowsinessState.CRITICAL_WARNING,
                3,
                alert_active=True,
                unresolved_alert_state=DrowsinessState.CRITICAL_WARNING,
            )
        )
        assert mgr.current_state == AlertState.CRITICAL

        trans = mgr.force_recovery(timestamp_ms=2000.0)
        assert trans is not None
        assert trans.new_state == AlertState.NORMAL
        assert trans.is_recovery
        assert mgr.current_state == AlertState.NORMAL
        assert not mgr.is_alert_active
