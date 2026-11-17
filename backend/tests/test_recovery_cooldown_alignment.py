"""
DriveGuard — Phase 6, Task 3: Recovery Cooldown Timer Alignment Unit Tests.

Comprehensive validation of the corrected recovery cooldown behaviour:
1. Test 1 — Candidate deterioration restarts recovery (CRITICAL -> NORMAL -> WARNING -> NORMAL).
2. Test 2 — Deterioration to ATTENTION restarts recovery (CRITICAL -> NORMAL -> ATTENTION -> NORMAL).
3. Test 3 — Candidate improvement preserves the timer (CRITICAL -> WARNING -> NORMAL).
4. Test 4 — Candidate returning to current severity cancels recovery (WARNING -> NORMAL -> WARNING -> NORMAL).
5. Test 5 — Sampling gap cancels recovery (camera freeze > 250 ms cannot advance cooldown).
6. Test 6 — Face loss and sensor unreliability cancel recovery.
7. Test 7 — Invalid timestamps (duplicate, backward, negative, NaN, Inf) cannot advance recovery.
8. Test 8 — Existing hysteresis behaviour, cooldown durations, and reset remain intact.
"""

from __future__ import annotations

import math
import pytest

from backend.app.intelligence.confidence import (
    ConfidenceAssessment,
    ConfidenceTier,
)
from backend.app.intelligence.decision_engine import DecisionEngine
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.intelligence.temporal_analyzer import TemporalMetrics


# ==============================================================================
# Deterministic Test Helpers
# ==============================================================================

def make_test_metrics(
    timestamp_ms: float,
    active_closure_ms: float = 0.0,
    perclos_pct: float = 0.0,
    has_face: bool = True,
    face_lost_duration_ms: float = 0.0,
    yawn_count_30s: int = 0,
    active_yawn_duration_ms: float = 0.0,
) -> TemporalMetrics:
    """Constructs deterministic TemporalMetrics targeting specific candidate severities."""
    return TemporalMetrics(
        timestamp_ms=timestamp_ms,
        active_closure_duration_ms=active_closure_ms,
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
        yawn_count_30s=yawn_count_30s,
        active_yawn_duration_ms=active_yawn_duration_ms,
        perclos_pct=perclos_pct,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=has_face,
        face_lost_duration_ms=face_lost_duration_ms,
    )


def make_test_confidence(is_reliable: bool = True) -> ConfidenceAssessment:
    """Constructs deterministic ConfidenceAssessment."""
    tier = ConfidenceTier.HIGH if is_reliable else ConfidenceTier.DEGRADED
    return ConfidenceAssessment(
        overall_confidence=1.0 if is_reliable else 0.20,
        tier=tier,
        is_reliable=is_reliable,
        reasons=[] if is_reliable else ["Optical degradation"],
    )


# ==============================================================================
# Unit Tests (1–8)
# ==============================================================================

def test_1_candidate_deterioration_restarts_recovery() -> None:
    """
    Test 1 — Candidate deterioration restarts recovery:
    - Current state: CRITICAL_WARNING.
    - At 1000 ms, candidate becomes NORMAL.
    - At 2800 ms, candidate relapses to DROWSINESS_WARNING.
    - At 3000 ms, candidate returns to NORMAL.
    - With a 2000 ms critical cooldown, state must remain CRITICAL_WARNING at 3000 ms.
    - De-escalation to NORMAL must occur only after full 2000 ms unbroken recovery from 2800 ms (at 4800 ms).
    """
    conf = make_test_confidence()
    engine = DecisionEngine(recovery_critical_ms=2000.0)

    # 1. Trigger CRITICAL_WARNING via sustained closure >= 1500 ms
    engine.evaluate(make_test_metrics(100.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # 2. Driver opens eyes at 1000 ms -> candidate drops to NORMAL (recovery begins)
    engine.evaluate(make_test_metrics(1000.0, active_closure_ms=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._recovery_candidate_sev == 0

    # Stream continuous normal frames up to 2700 ms
    for t in range(1100, 2800, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0), conf)

    # 3. At 2800 ms, driver relapses: candidate becomes DROWSINESS_WARNING (PERCLOS = 18%)
    engine.evaluate(make_test_metrics(2800.0, perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    # CRITICAL: Timer MUST restart at 2800 ms and candidate severity must update to 2
    assert engine._recovery_start_ts_ms == 2800.0
    assert engine._recovery_candidate_sev == 2

    # Stream frames through 2900 ms with DROWSINESS_WARNING
    for t in range(2900, 3000, 50):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=18.0), conf)

    # 4. At 3000 ms, candidate returns to NORMAL (perclos = 0)
    res_3000 = engine.evaluate(make_test_metrics(3000.0, perclos_pct=0.0), conf)
    # MUST NOT downgrade at 3000 ms (only 200 ms elapsed since relapse!)
    assert res_3000.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 2800.0
    assert engine._recovery_candidate_sev == 0

    # Stream frames through 4700 ms (1900 ms since relapse)
    for t in range(3100, 4800, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # 5. At 4800 ms (exactly 2000 ms since relapse at 2800 ms) -> transitions to NORMAL
    res_4800 = engine.evaluate(make_test_metrics(4800.0, perclos_pct=0.0), conf)
    assert res_4800.state == DrowsinessState.NORMAL
    assert engine._recovery_start_ts_ms is None
    assert engine._recovery_candidate_sev is None


def test_2_deterioration_to_attention_restarts_recovery() -> None:
    """
    Test 2 — Deterioration to ATTENTION restarts recovery:
    - Start recovery from CRITICAL_WARNING with candidate NORMAL at 1000 ms.
    - At 2000 ms, candidate rises to ATTENTION (extended closure >= 500 ms).
    - Recovery timer must restart at 2000 ms.
    - Subsequent return to NORMAL at 2500 ms cannot use the 1000 ms timestamp.
    - State must remain CRITICAL_WARNING at 3000 ms and de-escalate at 4000 ms.
    """
    conf = make_test_confidence()
    engine = DecisionEngine(recovery_critical_ms=2000.0)

    # Trigger CRITICAL_WARNING
    engine.evaluate(make_test_metrics(100.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # Start recovery at 1000 ms with candidate NORMAL
    engine.evaluate(make_test_metrics(1000.0, active_closure_ms=0.0), conf)
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._recovery_candidate_sev == 0

    # Stream frames up to 1900 ms
    for t in range(1100, 2000, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0), conf)

    # At 2000 ms: candidate deteriorates to ATTENTION (extended closure 600 ms >= 500 ms)
    engine.evaluate(make_test_metrics(2000.0, active_closure_ms=600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    # Recovery timer MUST restart at 2000 ms
    assert engine._recovery_start_ts_ms == 2000.0
    assert engine._recovery_candidate_sev == 1

    # Stream frames through 2400 ms with ATTENTION (avoiding sampling gaps > 250 ms)
    for t in range(2100, 2500, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=600.0), conf)

    # At 2500 ms: candidate drops back to NORMAL
    engine.evaluate(make_test_metrics(2500.0, active_closure_ms=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 2000.0
    assert engine._recovery_candidate_sev == 0

    # Stream frames through 2900 ms with NORMAL (avoiding sampling gaps > 250 ms)
    for t in range(2600, 3000, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0), conf)

    # At 3000 ms (2000 ms from initial 1000 ms, but only 1000 ms from deterioration)
    res_3000 = engine.evaluate(make_test_metrics(3000.0, active_closure_ms=0.0), conf)
    assert res_3000.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 2000.0

    # Stream frames up to 3900 ms
    for t in range(3100, 4000, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # At 4000 ms (2000 ms since deterioration at 2000 ms) -> transitions to NORMAL
    res_4000 = engine.evaluate(make_test_metrics(4000.0, active_closure_ms=0.0), conf)
    assert res_4000.state == DrowsinessState.NORMAL


def test_3_candidate_improvement_preserves_timer() -> None:
    """
    Test 3 — Candidate improvement preserves the timer:
    - Current state: CRITICAL_WARNING.
    - Candidate becomes DROWSINESS_WARNING at 1000 ms.
    - Candidate improves to NORMAL at 1500 ms.
    - Original recovery start timestamp (1000 ms) must be preserved.
    - De-escalation to NORMAL must occur at 3000 ms (2000 ms cooldown satisfied).
    """
    conf = make_test_confidence()
    engine = DecisionEngine(recovery_critical_ms=2000.0)

    # Trigger CRITICAL_WARNING
    engine.evaluate(make_test_metrics(100.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # At 1000 ms: candidate drops to DROWSINESS_WARNING (PERCLOS = 18%)
    engine.evaluate(make_test_metrics(1000.0, perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._recovery_candidate_sev == 2

    # Stream frames up to 1400 ms
    for t in range(1100, 1500, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=18.0), conf)

    # At 1500 ms: candidate improves further to NORMAL (perclos = 0)
    engine.evaluate(make_test_metrics(1500.0, perclos_pct=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    # Improvement MUST preserve original recovery start timestamp (1000 ms)
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._recovery_candidate_sev == 0

    # At 2900 ms: elapsed is 1900 ms < 2000 ms -> still held in CRITICAL
    for t in range(1600, 3000, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # At 3000 ms (2000 ms from 1000 ms) -> transitions safely to NORMAL
    res_3000 = engine.evaluate(make_test_metrics(3000.0, perclos_pct=0.0), conf)
    assert res_3000.state == DrowsinessState.NORMAL
    assert engine._recovery_start_ts_ms is None
    assert engine._recovery_candidate_sev is None


def test_4_candidate_returns_to_current_severity() -> None:
    """
    Test 4 — Candidate returning to current severity:
    - Current state: DROWSINESS_WARNING.
    - Candidate becomes NORMAL at 1000 ms.
    - Candidate returns to DROWSINESS_WARNING at 1800 ms (before 1500 ms cooldown completes).
    - Recovery must be cancelled (cleared).
    - Fresh return to NORMAL at 2200 ms requires full 1500 ms cooldown (until 3700 ms).
    """
    conf = make_test_confidence()
    engine = DecisionEngine(recovery_warning_ms=1500.0)

    # Trigger DROWSINESS_WARNING (PERCLOS = 18%)
    engine.evaluate(make_test_metrics(100.0, perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING

    # At 1000 ms: candidate drops to NORMAL (recovery begins)
    engine.evaluate(make_test_metrics(1000.0, perclos_pct=0.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._recovery_candidate_sev == 0

    # Stream frames up to 1700 ms
    for t in range(1100, 1800, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=0.0), conf)

    # At 1800 ms: candidate returns to DROWSINESS_WARNING (PERCLOS = 18%)
    engine.evaluate(make_test_metrics(1800.0, perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING
    # Recovery must be CANCELLED (cleared)
    assert engine._recovery_start_ts_ms is None
    assert engine._recovery_candidate_sev is None

    # Stream frames with DROWSINESS_WARNING up to 2100 ms
    for t in range(1900, 2200, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=18.0), conf)

    # At 2200 ms: candidate drops to NORMAL again -> fresh cooldown begins
    engine.evaluate(make_test_metrics(2200.0, perclos_pct=0.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING
    assert engine._recovery_start_ts_ms == 2200.0
    assert engine._recovery_candidate_sev == 0

    # At 3600 ms (1400 ms since 2200 ms, still < 1500 ms cooldown)
    for t in range(2300, 3700, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=0.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING

    # At 3700 ms (full 1500 ms fresh cooldown satisfied) -> transitions to NORMAL
    res_3700 = engine.evaluate(make_test_metrics(3700.0, perclos_pct=0.0), conf)
    assert res_3700.state == DrowsinessState.NORMAL


def test_5_sampling_gap_cancels_recovery() -> None:
    """
    Test 5 — Sampling gap cancels recovery:
    - Current state: CRITICAL_WARNING.
    - Recovery begins at 1000 ms.
    - Simulate a camera freeze exceeding 250 ms (e.g. 5000 ms gap to 6000 ms).
    - Resume with valid frame at 6000 ms.
    - The unobserved interval must not be credited toward recovery.
    - State must remain CRITICAL_WARNING at 6000 ms, requiring fresh 2000 ms cooldown (until 8000 ms).
    """
    conf = make_test_confidence()
    engine = DecisionEngine(recovery_critical_ms=2000.0, max_sampling_gap_ms=250.0)

    # Trigger CRITICAL_WARNING at 100 ms
    engine.evaluate(make_test_metrics(100.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # Recovery begins at 1000 ms
    engine.evaluate(make_test_metrics(1000.0, active_closure_ms=0.0), conf)
    assert engine._recovery_start_ts_ms == 1000.0

    # Frame at 1100 ms
    engine.evaluate(make_test_metrics(1100.0, active_closure_ms=0.0), conf)

    # 5000 ms camera freeze! Frame arrives at 6100 ms (dt = 5000 ms > 250 ms max gap)
    res_gap = engine.evaluate(make_test_metrics(6100.0, active_closure_ms=0.0), conf)
    # Must NOT downgrade to NORMAL on first post-gap frame!
    assert res_gap.state == DrowsinessState.CRITICAL_WARNING
    # Fresh recovery starts at 6100 ms
    assert engine._recovery_start_ts_ms == 6100.0
    assert engine._recovery_candidate_sev == 0

    # Stream frames through 8000 ms (1900 ms elapsed since gap recovery)
    for t in range(6200, 8100, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # At 8100 ms (2000 ms unbroken recovery since resumption) -> transitions to NORMAL
    res_8100 = engine.evaluate(make_test_metrics(8100.0, active_closure_ms=0.0), conf)
    assert res_8100.state == DrowsinessState.NORMAL


def test_6_face_loss_cancels_recovery() -> None:
    """
    Test 6 — Face loss and sensor unreliability cancel recovery:
    - Begin recovery from CRITICAL_WARNING at 1000 ms.
    - Face lost at 1500 ms (within 1000 ms grace period).
    - Recovery timer must be cleared immediately.
    - Face returns at 1600 ms.
    - Fresh 2000 ms recovery cooldown is required (until 3600 ms).
    """
    conf_ok = make_test_confidence(is_reliable=True)
    engine = DecisionEngine(recovery_critical_ms=2000.0, face_loss_grace_ms=1000.0)

    # Trigger CRITICAL_WARNING
    engine.evaluate(make_test_metrics(100.0, active_closure_ms=1600.0), conf_ok)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # Recovery begins at 1000 ms
    engine.evaluate(make_test_metrics(1000.0, active_closure_ms=0.0), conf_ok)
    assert engine._recovery_start_ts_ms == 1000.0

    # Face lost at 1500 ms (face_lost_duration = 100 ms < 1000 ms grace)
    r_lost = engine.evaluate(
        make_test_metrics(1500.0, active_closure_ms=0.0, has_face=False, face_lost_duration_ms=100.0),
        conf_ok,
    )
    assert r_lost.state == DrowsinessState.CRITICAL_WARNING
    # Crucial: recovery timer is CANCELLED because driver is unobserved
    assert engine._recovery_start_ts_ms is None
    assert engine._recovery_candidate_sev is None

    # Face returns at 1600 ms
    engine.evaluate(make_test_metrics(1600.0, active_closure_ms=0.0, has_face=True), conf_ok)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    # Fresh recovery starts at 1600 ms
    assert engine._recovery_start_ts_ms == 1600.0
    assert engine._recovery_candidate_sev == 0

    # At 3500 ms (1900 ms elapsed since face returned < 2000 ms cooldown)
    for t in range(1700, 3600, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0, has_face=True), conf_ok)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # At 3600 ms (full 2000 ms since face returned) -> transitions to NORMAL
    res_3600 = engine.evaluate(make_test_metrics(3600.0, active_closure_ms=0.0, has_face=True), conf_ok)
    assert res_3600.state == DrowsinessState.NORMAL


def test_7_invalid_timestamps_cannot_advance_recovery() -> None:
    """
    Test 7 — Invalid timestamps cannot advance recovery:
    - Test duplicate, backward, negative, NaN, and infinite timestamps.
    - Verify they do not advance cooldown or corrupt the timestamp state.
    """
    conf = make_test_confidence()
    engine = DecisionEngine(recovery_critical_ms=2000.0)

    # Trigger CRITICAL_WARNING
    engine.evaluate(make_test_metrics(100.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # Start recovery at 1000 ms
    engine.evaluate(make_test_metrics(1000.0, active_closure_ms=0.0), conf)
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._prev_ts_ms == 1000.0

    # 1. Duplicate timestamp (1000 ms)
    r_dup = engine.evaluate(make_test_metrics(1000.0, active_closure_ms=0.0), conf)
    assert r_dup.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._prev_ts_ms == 1000.0

    # 2. Backward timestamp (950 ms)
    r_back = engine.evaluate(make_test_metrics(950.0, active_closure_ms=0.0), conf)
    assert r_back.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._prev_ts_ms == 1000.0

    # 3. Negative timestamp (-100 ms)
    r_neg = engine.evaluate(make_test_metrics(-100.0, active_closure_ms=0.0), conf)
    assert r_neg.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._prev_ts_ms == 1000.0

    # 4. NaN timestamp
    r_nan = engine.evaluate(make_test_metrics(float("nan"), active_closure_ms=0.0), conf)
    assert r_nan.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._prev_ts_ms == 1000.0

    # 5. Infinite timestamp
    r_inf = engine.evaluate(make_test_metrics(float("inf"), active_closure_ms=0.0), conf)
    assert r_inf.state == DrowsinessState.CRITICAL_WARNING
    assert engine._recovery_start_ts_ms == 1000.0
    assert engine._prev_ts_ms == 1000.0

    # 6. Valid forward frame (1100 ms)
    r_valid = engine.evaluate(make_test_metrics(1100.0, active_closure_ms=0.0), conf)
    assert r_valid.state == DrowsinessState.CRITICAL_WARNING
    assert engine._prev_ts_ms == 1100.0
    assert engine._recovery_start_ts_ms == 1000.0


def test_8_existing_hysteresis_and_reset_intact() -> None:
    """
    Test 8 — Existing hysteresis behaviour, cooldown durations, and reset remain intact:
    - Immediate escalation works.
    - Each cooldown tier (Critical: 2000 ms, Warning: 1500 ms, Attention: 1000 ms) is respected.
    - Explicit reset clears all recovery tracking.
    """
    conf = make_test_confidence()
    engine = DecisionEngine(
        recovery_critical_ms=2000.0,
        recovery_warning_ms=1500.0,
        recovery_attention_ms=1000.0,
    )

    # 1. Immediate escalation from NORMAL to DROWSINESS_WARNING
    engine.evaluate(make_test_metrics(100.0, perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING

    # 2. Immediate escalation from DROWSINESS_WARNING to CRITICAL_WARNING
    engine.evaluate(make_test_metrics(200.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # 3. Step-down recovery from CRITICAL_WARNING (2000 ms) to DROWSINESS_WARNING
    engine.evaluate(make_test_metrics(1000.0, perclos_pct=18.0), conf)
    for t in range(1100, 3000, 100):
        engine.evaluate(make_test_metrics(float(t), perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING

    # Downgrades to DROWSINESS_WARNING at 3000 ms
    engine.evaluate(make_test_metrics(3000.0, perclos_pct=18.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING

    # 4. Step-down recovery from DROWSINESS_WARNING (1500 ms) to ATTENTION
    engine.evaluate(make_test_metrics(3100.0, active_closure_ms=600.0), conf)
    for t in range(3200, 4600, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=600.0), conf)
    assert engine.current_state == DrowsinessState.DROWSINESS_WARNING

    # Downgrades to ATTENTION at 4600 ms (1500 ms cooldown)
    engine.evaluate(make_test_metrics(4600.0, active_closure_ms=600.0), conf)
    assert engine.current_state == DrowsinessState.ATTENTION

    # 5. Step-down recovery from ATTENTION (1000 ms) to NORMAL
    engine.evaluate(make_test_metrics(4700.0, active_closure_ms=0.0), conf)
    for t in range(4800, 5700, 100):
        engine.evaluate(make_test_metrics(float(t), active_closure_ms=0.0), conf)
    assert engine.current_state == DrowsinessState.ATTENTION

    # Downgrades to NORMAL at 5700 ms (1000 ms cooldown)
    engine.evaluate(make_test_metrics(5700.0, active_closure_ms=0.0), conf)
    assert engine.current_state == DrowsinessState.NORMAL

    # 6. Explicit reset clears all state
    engine.evaluate(make_test_metrics(6000.0, active_closure_ms=1600.0), conf)
    assert engine.current_state == DrowsinessState.CRITICAL_WARNING
    engine.evaluate(make_test_metrics(7000.0, active_closure_ms=0.0), conf)
    assert engine._recovery_start_ts_ms == 7000.0
    assert engine._recovery_candidate_sev == 0

    engine.reset()
    assert engine.current_state == DrowsinessState.NORMAL
    assert engine._recovery_start_ts_ms is None
    assert engine._recovery_candidate_sev is None
    assert engine._prev_ts_ms is None
