"""
DriveGuard Phase 7 — Task 3.2: Dedicated Regression Test Suite.
Independent Downward-Head Monitoring, Channel Availability, and Alert Continuity.

Tests 29 required integration scenarios across Groups A, B, C, D:
- Group A: Channel availability decoupling (Tests 1–9)
- Group B: Downward posture dwell tracking (Tests 10–15)
- Group C: Decision engine arbitration and boundary verification (Tests 16–23)
- Group D: Yawn recording during faults, recovery arbitration, and alert continuity (Tests 24–29)
"""

import math
from typing import Optional
import pytest

from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityResult,
    LightingCondition,
    QualityStatus,
)
from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkEvent,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import EyeAspectRatioResult
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import HeadPoseResult, PoseZone
from backend.app.features.mar import MouthAspectRatioResult, YawnEvent
from backend.app.features.perclos import PerclosResult
from backend.app.intelligence.confidence import (
    ConfidenceAssessment,
    ConfidenceTier,
    SignalConfidenceEvaluator,
)
from backend.app.intelligence.decision_engine import DecisionEngine, DecisionResult
from backend.app.intelligence.reason_codes import (
    DrowsinessState,
    ExplainableReason,
    ReasonCode,
)
from backend.app.intelligence.temporal_analyzer import TemporalAnalyzer, TemporalMetrics
from backend.app.vision.face_landmarker import FaceLandmarkStatus


# ==============================================================================
# Synthetic Feature Helpers
# ==============================================================================

def make_features(
    timestamp_ms: float,
    has_face: bool = True,
    face_status: FaceLandmarkStatus = FaceLandmarkStatus.TRACKING,
    ear_valid: bool = True,
    ear_val: float = 0.28,
    is_closed: bool = False,
    pose_valid: bool = True,
    zone: PoseZone = PoseZone.CENTER,
    rel_pitch: float = 0.0,
    rel_yaw: float = 0.0,
    mar_valid: bool = True,
    mar_val: float = 0.18,
    is_yawning: bool = False,
    completed_yawn: Optional[YawnEvent] = None,
    blink_state: BlinkState = BlinkState.OPEN,
    sampling_gap_detected: bool = False,
    current_closure_duration_ms: float = 0.0,
    perclos_pct: float = 2.0,
    perclos_reliable: bool = True,
) -> DriverFeatures:
    """Helper creating DriverFeatures for pipeline integration testing."""
    ear = EyeAspectRatioResult(
        left_ear=ear_val,
        right_ear=ear_val,
        weighted_ear=ear_val,
        compensated_ear=ear_val,
        closure_threshold=0.20,
        is_left_closed=is_closed,
        is_right_closed=is_closed,
        is_closed=is_closed,
        is_valid=ear_valid,
    )
    head_pose = HeadPoseResult(
        pitch_deg=rel_pitch,
        yaw_deg=rel_yaw,
        roll_deg=0.0,
        zone=zone,
        is_looking_away=(zone != PoseZone.CENTER),
        is_valid=pose_valid,
        rel_pitch_deg=rel_pitch,
        rel_yaw_deg=rel_yaw,
    )
    mar = MouthAspectRatioResult(
        mar=mar_val,
        is_mouth_open=is_yawning,
        is_yawn_in_progress=is_yawning,
        current_yawn_duration_ms=2000.0 if is_yawning else 0.0,
        is_authentic_yawn=is_yawning,
        completed_yawn=completed_yawn,
        total_yawn_count=1 if completed_yawn is not None else 0,
        is_valid=mar_valid,
        observed_yawn_duration_ms=2000.0 if is_yawning else 0.0,
    )
    blink = BlinkSnapshot(
        state=blink_state,
        is_eye_closed=is_closed,
        current_closure_duration_ms=current_closure_duration_ms,
        completed_blink=None,
        recent_blink_duration_ms=None,
        recent_blink_rcr=None,
        total_blink_count=0,
        prolonged_blink_count=0,
    )
    perclos = PerclosResult(
        perclos_pct=perclos_pct,
        coverage_pct=100.0,
        is_reliable=perclos_reliable,
        closed_duration_sec=0.0,
        observed_duration_sec=60.0,
        window_sec=60.0,
    )
    return DriverFeatures(
        timestamp_ms=timestamp_ms,
        has_face=has_face,
        face_status=face_status,
        ear=ear,
        head_pose=head_pose,
        mar=mar,
        blink=blink,
        perclos=perclos,
    )


def make_quality(
    is_viable: bool = True,
    status: QualityStatus = QualityStatus.EXCELLENT,
    lighting: LightingCondition = LightingCondition.OPTIMAL,
    blur: BlurCondition = BlurCondition.SHARP,
) -> FrameQualityResult:
    """Helper constructing FrameQualityResult."""
    return FrameQualityResult(
        mean_luminance=120.0 if is_viable else 5.0,
        blur_variance=250.0 if is_viable else 10.0,
        lighting=lighting,
        blur=blur,
        status=status,
        is_viable_for_inference=is_viable,
        reasons=(),
    )


# ==============================================================================
# Group A: Channel Availability Decoupling (Tests 1–9)
# ==============================================================================

def test_1_valid_head_pose_with_invalid_ear():
    """Scenario 1: Valid head pose with invalid EAR leaves head pose available."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(0.0, ear_valid=False, pose_valid=True, zone=PoseZone.CENTER)
    conf = evaluator.evaluate(features)

    assert conf.is_head_pose_available is True
    assert conf.head_pose_viable is True
    assert conf.is_ocular_available is False
    assert conf.ocular_viable is False


def test_2_valid_ear_with_invalid_head_pose():
    """Scenario 2: Valid EAR with invalid head pose leaves ocular channel available."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(0.0, ear_valid=True, pose_valid=False, zone=PoseZone.UNKNOWN)
    conf = evaluator.evaluate(features)

    assert conf.is_ocular_available is True
    assert conf.ocular_viable is True
    assert conf.is_head_pose_available is False
    assert conf.head_pose_viable is False


def test_3_invalid_ear_and_invalid_pose():
    """Scenario 3: Both primary channels invalid results in neither primary channel available."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(0.0, ear_valid=False, pose_valid=False, zone=PoseZone.UNKNOWN)
    conf = evaluator.evaluate(features)

    assert conf.is_head_pose_available is False
    assert conf.is_ocular_available is False
    assert (conf.head_pose_viable or conf.ocular_viable) is False


def test_4_valid_mar_alone_does_not_make_primary_viable():
    """Scenario 4: Valid MAR alone provides yawn availability but primary viability remains False."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(0.0, ear_valid=False, pose_valid=False, mar_valid=True)
    conf = evaluator.evaluate(features)

    assert conf.is_yawn_available is True
    assert conf.yawn_viable is True
    assert (conf.head_pose_viable or conf.ocular_viable) is False


def test_5_dashboard_pose_does_not_invalidate_mar_availability():
    """Scenario 5: Dashboard pose (is_looking_away=True) preserves MAR availability."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(
        0.0,
        pose_valid=True,
        zone=PoseZone.DASHBOARD,
        rel_pitch=-14.0,
        mar_valid=True,
    )
    conf = evaluator.evaluate(features)

    assert conf.is_yawn_available is True
    assert conf.yawn_viable is True


def test_6_uncertain_gap_makes_ocular_availability_false():
    """Scenario 6: BlinkState.UNCERTAIN_GAP invalidates ocular channel even if EAR is valid."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(
        0.0,
        ear_valid=True,
        blink_state=BlinkState.UNCERTAIN_GAP,
    )
    conf = evaluator.evaluate(features)

    assert conf.is_ocular_available is False
    assert conf.ocular_viable is False


def test_7_sampling_gap_detected_makes_ocular_availability_false():
    """Scenario 7: Sampling gap detected on blink detector invalidates ocular availability."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(
        0.0,
        ear_valid=True,
        blink_state=BlinkState.UNCERTAIN_GAP,
    )
    assert features.blink.sampling_gap_detected is True
    conf = evaluator.evaluate(features)

    assert conf.is_ocular_available is False
    assert conf.ocular_viable is False


def test_8_face_loss_and_unusable_frame():
    """Scenario 8: Complete face loss or unviable frame quality disables all channels."""
    evaluator = SignalConfidenceEvaluator()

    # Case 8a: Face loss
    features_no_face = make_features(0.0, has_face=False, face_status=FaceLandmarkStatus.INSUFFICIENT_DATA)
    conf_no_face = evaluator.evaluate(features_no_face)
    assert conf_no_face.is_head_pose_available is False
    assert conf_no_face.is_ocular_available is False
    assert conf_no_face.is_yawn_available is False

    # Case 8b: Optical unviable frame
    features_face = make_features(0.0, has_face=True)
    quality_bad = make_quality(is_viable=False, status=QualityStatus.SEVERELY_DEGRADED)
    conf_bad_qual = evaluator.evaluate(features_face, quality_bad)
    assert conf_bad_qual.is_head_pose_available is False
    assert conf_bad_qual.is_ocular_available is False
    assert conf_bad_qual.is_yawn_available is False


def test_9_poor_but_viable_lighting_preserves_head_pose():
    """Scenario 9: POOR_LIGHTING (viable for inference) does not disable valid head pose."""
    evaluator = SignalConfidenceEvaluator()
    features = make_features(0.0, pose_valid=True, zone=PoseZone.CENTER)
    quality_poor = make_quality(is_viable=True, status=QualityStatus.POOR_LIGHTING, lighting=LightingCondition.UNDEREXPOSED)
    conf = evaluator.evaluate(features, quality_poor)

    assert conf.is_head_pose_available is True
    assert conf.head_pose_viable is True


# ==============================================================================
# Group B: Downward Dwell Tracking (Tests 10–15)
# ==============================================================================

def test_10_valid_dashboard_downward_accumulates_dwell():
    """Scenario 10: Continuous downward posture in DASHBOARD accumulates downward_dwell_ms."""
    analyzer = TemporalAnalyzer()

    # t=0ms: initial downward observation
    m0 = analyzer.update(make_features(0.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0, rel_yaw=0.0))
    assert m0.downward_dwell_ms == 0.0

    # Stream frames every 100ms (no camera sampling gap)
    for t in range(100, 501, 100):
        m1 = analyzer.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-13.0, rel_yaw=2.0))
    assert m1.downward_dwell_ms == 500.0

    for t in range(600, 2001, 100):
        m2 = analyzer.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-15.0, rel_yaw=-3.0))
    assert m2.downward_dwell_ms == 2000.0


def test_11_leaving_downward_posture_resets_dwell():
    """Scenario 11: Raising head or turning laterally resets downward_dwell_ms to 0.0."""
    analyzer = TemporalAnalyzer()

    for t in range(0, 1001, 100):
        m1 = analyzer.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m1.downward_dwell_ms == 1000.0

    # Return to CENTER (forward road) at 1100ms
    m2 = analyzer.update(make_features(1100.0, zone=PoseZone.CENTER, rel_pitch=0.0))
    assert m2.downward_dwell_ms == 0.0
    assert analyzer.current_downward_dwell_ms == 0.0


def test_12_invalid_pose_face_loss_unknown_resets_dwell():
    """Scenario 12: Invalid head pose, face loss, or UNKNOWN zone resets dwell."""
    # 12a: Invalid head pose
    a1 = TemporalAnalyzer()
    for t in range(0, 1001, 100):
        a1.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    m_inv = a1.update(make_features(1100.0, pose_valid=False, zone=PoseZone.UNKNOWN))
    assert m_inv.downward_dwell_ms == 0.0

    # 12b: Face loss
    a2 = TemporalAnalyzer()
    for t in range(0, 1001, 100):
        a2.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    m_fl = a2.update(make_features(1100.0, has_face=False))
    assert m_fl.downward_dwell_ms == 0.0

    # 12c: Pitch insufficiently downward (e.g. rel_pitch = -8.0 > -12.0)
    a3 = TemporalAnalyzer()
    for t in range(0, 1001, 100):
        a3.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    m_shallow = a3.update(make_features(1100.0, zone=PoseZone.DASHBOARD, rel_pitch=-8.0))
    assert m_shallow.downward_dwell_ms == 0.0


def test_13_ocular_tracking_gap_preserves_head_pose_dwell():
    """Scenario 13: Ocular tracking gaps (e.g. UNCERTAIN_GAP, invalid EAR) do NOT reset downward dwell."""
    analyzer = TemporalAnalyzer()

    # t=0: nominal downward
    analyzer.update(make_features(0.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))

    # Stream frames every 100ms where eyes are occluded (ear invalid, blink UNCERTAIN_GAP)
    for t in range(100, 1501, 100):
        f = make_features(
            float(t),
            zone=PoseZone.DASHBOARD,
            rel_pitch=-14.5,
            ear_valid=False,
            blink_state=BlinkState.UNCERTAIN_GAP,
        )
        m = analyzer.update(f)

    assert m.downward_dwell_ms == 1500.0


def test_14_camera_sampling_gap_reanchors_dwell():
    """Scenario 14: Camera sampling gap (> 250ms) re-anchors dwell without crediting missing interval."""
    analyzer = TemporalAnalyzer(max_sampling_gap_ms=250.0)

    analyzer.update(make_features(0.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    m1 = analyzer.update(make_features(200.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m1.downward_dwell_ms == 200.0

    # Frame arrives at 1000ms (gap = 800ms > 250ms)
    # Re-anchors at t=1000ms with zero elapsed time credited
    f_gap = make_features(1000.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0)
    m_gap = analyzer.update(f_gap)
    assert m_gap.downward_dwell_ms == 0.0

    # Next frame at 1100ms (delta = 100ms <= 250ms) accumulates 100ms from the new anchor
    m_next = analyzer.update(make_features(1100.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m_next.downward_dwell_ms == 100.0


def test_15_duplicate_backward_invalid_timestamps_do_not_corrupt_dwell():
    """Scenario 15: Non-finite, negative, duplicate, or backward timestamps do not corrupt dwell state."""
    analyzer = TemporalAnalyzer()

    for t in range(100, 501, 100):
        m1 = analyzer.update(make_features(float(t), zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m1.downward_dwell_ms == 400.0

    # Duplicate timestamp
    m_dup = analyzer.update(make_features(500.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m_dup.downward_dwell_ms == 400.0

    # Backward timestamp
    m_back = analyzer.update(make_features(400.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m_back.downward_dwell_ms == 400.0

    # NaN timestamp
    m_nan = analyzer.update(make_features(float("nan"), zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m_nan.downward_dwell_ms == 400.0

    # Normal monotonic frame resumes cleanly
    m_next = analyzer.update(make_features(600.0, zone=PoseZone.DASHBOARD, rel_pitch=-14.0))
    assert m_next.downward_dwell_ms == 500.0


# ==============================================================================
# Group C: Decision Arbitration & Boundary Verification (Tests 16–23)
# ==============================================================================

def test_16_downward_dwell_1999ms_no_slump_candidate():
    """Scenario 16: Dwell of 1,999 ms is benign dashboard context (<= 2000.0 ms); no slump candidate."""
    engine = DecisionEngine()
    conf = ConfidenceAssessment(
        overall_confidence=0.45,
        tier=ConfidenceTier.DEGRADED,
        is_reliable=False,
        is_head_pose_available=True,
        is_ocular_available=False,
        is_yawn_available=False,
    )
    # active_off_road_dwell_ms = 1999.0 <= 2000.0 is dashboard check context
    metrics = TemporalMetrics(
        timestamp_ms=1999.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=1999.0,
        is_mirror_check_context=False,
        is_dashboard_check_context=True,
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
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
        downward_dwell_ms=1999.0,
    )
    res = engine.evaluate(metrics, conf)

    assert res.state == DrowsinessState.NORMAL
    assert res.primary_reason.code == ReasonCode.DASHBOARD_CHECK_CONTEXT


def test_17_downward_dwell_exact_2000ms_no_slump_candidate():
    """Scenario 17: Dwell of exactly 2,000 ms is benign dashboard boundary (<= 2000.0 ms); no slump."""
    engine = DecisionEngine()
    conf = ConfidenceAssessment(
        overall_confidence=0.45,
        tier=ConfidenceTier.DEGRADED,
        is_reliable=False,
        is_head_pose_available=True,
        is_ocular_available=False,
        is_yawn_available=False,
    )
    metrics = TemporalMetrics(
        timestamp_ms=2000.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=2000.0,
        is_mirror_check_context=False,
        is_dashboard_check_context=True,
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
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
        downward_dwell_ms=2000.0,
    )
    res = engine.evaluate(metrics, conf)

    assert res.state == DrowsinessState.NORMAL
    assert res.primary_reason.code == ReasonCode.DASHBOARD_CHECK_CONTEXT


def test_18_downward_dwell_2001ms_triggers_slump_attention():
    """Scenario 18: Dwell of 2,001 ms (> 2000.0 ms) triggers Tier 1 ATTENTION SUSPECTED_HEAD_SLUMP."""
    engine = DecisionEngine()
    conf = ConfidenceAssessment(
        overall_confidence=0.45,
        tier=ConfidenceTier.DEGRADED,
        is_reliable=False,
        is_head_pose_available=True,
        is_ocular_available=False,
        is_yawn_available=False,
    )
    metrics = TemporalMetrics(
        timestamp_ms=2001.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=2001.0,
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
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
        downward_dwell_ms=2001.0,
    )
    res = engine.evaluate(metrics, conf)

    assert res.state == DrowsinessState.ATTENTION
    assert res.primary_reason.code == ReasonCode.SUSPECTED_HEAD_SLUMP
    assert res.primary_reason.metric_value == 2001.0
    assert res.primary_reason.threshold == 2000.0


def test_19_available_ocular_monitoring_suppresses_slump_rule():
    """Scenario 19: When ocular monitoring is available, downward dwell > 2000ms does NOT trigger slump."""
    engine = DecisionEngine()
    conf = ConfidenceAssessment(
        overall_confidence=0.90,
        tier=ConfidenceTier.HIGH,
        is_reliable=True,
        is_head_pose_available=True,
        is_ocular_available=True,
        is_yawn_available=True,
    )
    metrics = TemporalMetrics(
        timestamp_ms=2500.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=2500.0,
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
        perclos_pct=2.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
        is_eye_closed=False,
        downward_dwell_ms=2500.0,
    )
    res = engine.evaluate(metrics, conf)

    # Slump is suppressed because driver's eyes are visible and open
    assert res.primary_reason.code != ReasonCode.SUSPECTED_HEAD_SLUMP


def test_20_unavailable_head_pose_suppresses_slump_even_with_stale_dwell():
    """Scenario 20: If head pose is unavailable, slump rule is suppressed even if dwell > 2000ms."""
    engine = DecisionEngine()
    # Head pose unavailable, ocular available (e.g. driver looking forward through camera)
    conf = ConfidenceAssessment(
        overall_confidence=0.60,
        tier=ConfidenceTier.ACCEPTABLE,
        is_reliable=True,
        is_head_pose_available=False,
        is_ocular_available=True,
        is_yawn_available=False,
    )
    metrics = TemporalMetrics(
        timestamp_ms=2500.0,
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
        perclos_pct=2.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
        is_eye_closed=False,
        downward_dwell_ms=2500.0,  # Stale dwell metric
    )
    res = engine.evaluate(metrics, conf)

    assert res.primary_reason.code != ReasonCode.SUSPECTED_HEAD_SLUMP


def test_21_neither_primary_channel_available_enters_sensor_fault():
    """Scenario 21: When both primary channels are unavailable, sensor-fault grace then INSUFFICIENT_DATA."""
    engine = DecisionEngine(sensor_unreliable_grace_ms=1000.0)
    conf = ConfidenceAssessment(
        overall_confidence=0.10,
        tier=ConfidenceTier.UNAVAILABLE,
        is_reliable=False,
        is_head_pose_available=False,
        is_ocular_available=False,
        is_yawn_available=False,
    )
    metrics_0 = TemporalMetrics(
        timestamp_ms=0.0,
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    # Frame 0: sensor fault grace starts (elapsed = 0ms < 1000ms)
    r0 = engine.evaluate(metrics_0, conf)
    assert r0.state == DrowsinessState.NORMAL
    assert r0.primary_reason.code == ReasonCode.SENSOR_DEGRADED

    # Frame at 1500ms (elapsed = 1500ms >= 1000ms grace)
    metrics_1500 = TemporalMetrics(
        timestamp_ms=1500.0,
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    r1 = engine.evaluate(metrics_1500, conf)
    assert r1.state == DrowsinessState.INSUFFICIENT_DATA
    assert r1.primary_reason.code == ReasonCode.SENSOR_DEGRADED


def test_22_mar_alone_cannot_prevent_insufficient_data():
    """Scenario 22: Valid MAR / active yawn cannot prevent INSUFFICIENT_DATA when primary channels are down."""
    engine = DecisionEngine(sensor_unreliable_grace_ms=1000.0)
    conf = ConfidenceAssessment(
        overall_confidence=0.30,
        tier=ConfidenceTier.DEGRADED,
        is_reliable=False,
        is_head_pose_available=False,
        is_ocular_available=False,
        is_yawn_available=True,  # Yawn channel is viable!
    )
    metrics_start = TemporalMetrics(
        timestamp_ms=0.0,
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
        yawn_count_60s=1,
        yawn_count_30s=1,
        active_yawn_duration_ms=2500.0,
        perclos_pct=0.0,
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    engine.evaluate(metrics_start, conf)

    metrics_prolonged = TemporalMetrics(
        timestamp_ms=1200.0,
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
        yawn_count_60s=1,
        yawn_count_30s=1,
        active_yawn_duration_ms=3700.0,
        perclos_pct=0.0,
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    res = engine.evaluate(metrics_prolonged, conf)
    assert res.state == DrowsinessState.INSUFFICIENT_DATA


def test_23_higher_severity_candidate_never_downgraded_by_slump():
    """Scenario 23: Existing DROWSINESS_WARNING or CRITICAL candidate is never overwritten by slump rule."""
    engine = DecisionEngine()
    conf = ConfidenceAssessment(
        overall_confidence=0.45,
        tier=ConfidenceTier.DEGRADED,
        is_reliable=False,
        is_head_pose_available=True,
        is_ocular_available=False,
        is_yawn_available=False,
    )
    # Metrics qualify for DROWSINESS_WARNING (e.g. repeated yawns with fatigue kinematics or prolonged distraction)
    metrics = TemporalMetrics(
        timestamp_ms=2500.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=3600.0,  # Exceeds off_road_distraction_limit_ms (3000ms)
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
        downward_dwell_ms=2500.0,  # Qualifies for slump rule
    )
    res = engine.evaluate(metrics, conf)

    # Must retain DROWSINESS_WARNING, never downgraded to ATTENTION
    assert res.state == DrowsinessState.DROWSINESS_WARNING
    assert res.primary_reason.code == ReasonCode.PERSISTENT_DISTRACTION


# ==============================================================================
# Group D: Yawn & Alert Regression (Tests 24–29)
# ==============================================================================

def test_24_completed_yawn_recorded_during_primary_channel_failure():
    """Scenario 24: TemporalAnalyzer records completed yawn even during primary channel failure."""
    analyzer = TemporalAnalyzer()
    yawn = YawnEvent(
        start_ts_ms=500.0,
        end_ts_ms=2500.0,
        duration_ms=2000.0,
        peak_mar=0.65,
    )
    # Primary channels down (ear invalid, pose invalid)
    features = make_features(
        timestamp_ms=2500.0,
        ear_valid=False,
        pose_valid=False,
        completed_yawn=yawn,
    )
    metrics = analyzer.update(features)

    assert metrics.yawn_count_30s == 1
    assert metrics.yawn_count_60s == 1


def test_25_candidate_arbitration_bypassed_during_sensor_fault_early_return():
    """Scenario 25: Sensor-fault early return branch bypasses candidate arbitration for yawn."""
    engine = DecisionEngine(sensor_unreliable_grace_ms=1000.0)
    conf = ConfidenceAssessment(
        overall_confidence=0.10,
        tier=ConfidenceTier.UNAVAILABLE,
        is_reliable=False,
        is_head_pose_available=False,
        is_ocular_available=False,
        is_yawn_available=True,
    )
    metrics = TemporalMetrics(
        timestamp_ms=500.0,
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
        yawn_count_60s=1,
        yawn_count_30s=1,
        active_yawn_duration_ms=0.0,
        perclos_pct=0.0,
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    res = engine.evaluate(metrics, conf)

    # Returns early in Step 1 holding state with SENSOR_DEGRADED; does NOT emit ATTENTION
    assert res.state == DrowsinessState.NORMAL
    assert res.primary_reason.code == ReasonCode.SENSOR_DEGRADED


def test_26_recovery_within_30s_evaluates_recorded_yawn_once():
    """Scenario 26: Primary channel recovery within 30s evaluates recorded yawn as AUTHENTIC_YAWN."""
    engine = DecisionEngine(sensor_unreliable_grace_ms=1000.0)

    # 1. Fault frame at t=0ms
    conf_fault = ConfidenceAssessment(
        overall_confidence=0.10,
        tier=ConfidenceTier.UNAVAILABLE,
        is_reliable=False,
        is_head_pose_available=False,
        is_ocular_available=False,
    )
    m_fault = TemporalMetrics(
        timestamp_ms=0.0,
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
        yawn_count_60s=1,
        yawn_count_30s=1,
        active_yawn_duration_ms=0.0,
        perclos_pct=0.0,
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    engine.evaluate(m_fault, conf_fault)

    # 2. Recovery frame at t=2000ms: primary channel (head pose) recovers
    conf_recovered = ConfidenceAssessment(
        overall_confidence=0.85,
        tier=ConfidenceTier.HIGH,
        is_reliable=True,
        is_head_pose_available=True,
        is_ocular_available=True,
    )
    m_recovered = TemporalMetrics(
        timestamp_ms=2000.0,
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
        yawn_count_60s=1,
        yawn_count_30s=1,
        active_yawn_duration_ms=0.0,
        perclos_pct=2.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
        is_eye_closed=False,
    )
    res = engine.evaluate(m_recovered, conf_recovered)

    assert res.state == DrowsinessState.ATTENTION
    assert res.primary_reason.code == ReasonCode.AUTHENTIC_YAWN


def test_27_yawn_event_deduplication_intact():
    """Scenario 27: Repeated frames with the same completed yawn are not duplicated."""
    analyzer = TemporalAnalyzer()
    yawn = YawnEvent(
        start_ts_ms=1000.0,
        end_ts_ms=3000.0,
        duration_ms=2000.0,
        peak_mar=0.65,
    )
    f1 = make_features(3000.0, completed_yawn=yawn)
    m1 = analyzer.update(f1)
    assert m1.yawn_count_30s == 1

    # Next frame still carries same completed_yawn reference
    f2 = make_features(3100.0, completed_yawn=yawn)
    m2 = analyzer.update(f2)
    assert m2.yawn_count_30s == 1
    assert m2.yawn_count_60s == 1


def test_28_active_alert_continuity_intact_across_dropouts():
    """Scenario 28: CRITICAL_WARNING alert remains latched across prolonged sensor dropout."""
    engine = DecisionEngine(sensor_unreliable_grace_ms=1000.0)
    conf_ok = ConfidenceAssessment(1.0, ConfidenceTier.HIGH, True, is_head_pose_available=True, is_ocular_available=True)

    # 1. Trigger CRITICAL_WARNING via severe eye closure
    m_crit = TemporalMetrics(
        timestamp_ms=0.0,
        active_closure_duration_ms=2100.0,  # Exceeds severe_closure_threshold_ms (2000ms)
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
        perclos_pct=50.0,
        perclos_coverage_pct=100.0,
        perclos_is_reliable=True,
        has_face=True,
        face_lost_duration_ms=0.0,
        is_eye_closed=True,
    )
    r0 = engine.evaluate(m_crit, conf_ok)
    assert r0.state == DrowsinessState.CRITICAL_WARNING
    assert r0.alert_active is True
    assert r0.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING

    # 2. Both sensors drop out: initial fault at 500ms starts the sensor-fault timer
    conf_down = ConfidenceAssessment(0.0, ConfidenceTier.UNAVAILABLE, False, is_head_pose_available=False, is_ocular_available=False)
    m_drop_init = TemporalMetrics(
        timestamp_ms=500.0,
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    r_grace = engine.evaluate(m_drop_init, conf_down)
    # Holds state during grace period
    assert r_grace.state == DrowsinessState.CRITICAL_WARNING
    assert r_grace.alert_active is True

    # 3. Fault continues at 1600ms (elapsed = 1100ms >= 1000ms grace) -> INSUFFICIENT_DATA
    m_drop_prolonged = TemporalMetrics(
        timestamp_ms=1600.0,
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    r1 = engine.evaluate(m_drop_prolonged, conf_down)
    assert r1.state == DrowsinessState.INSUFFICIENT_DATA
    # Alert remains latched!
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.CRITICAL_WARNING
    assert r1.historical_reason.code == ReasonCode.SUSTAINED_EYE_CLOSURE


def test_29_attention_from_slump_or_yawn_does_not_resolve_active_alert():
    """Scenario 29: ATTENTION candidate (from slump or yawn) cannot resolve active warning/critical alert."""
    engine = DecisionEngine()
    conf_ok = ConfidenceAssessment(1.0, ConfidenceTier.HIGH, True, is_head_pose_available=True, is_ocular_available=True)

    # 1. Trigger DROWSINESS_WARNING
    m_warn = TemporalMetrics(
        timestamp_ms=0.0,
        active_closure_duration_ms=0.0,
        active_off_road_dwell_ms=3500.0,
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
    )
    r0 = engine.evaluate(m_warn, conf_ok)
    assert r0.state == DrowsinessState.DROWSINESS_WARNING
    assert r0.alert_active is True

    # 2. Next frame: candidate is ATTENTION from SUSPECTED_HEAD_SLUMP
    conf_slump = ConfidenceAssessment(0.50, ConfidenceTier.ACCEPTABLE, True, is_head_pose_available=True, is_ocular_available=False)
    m_slump = TemporalMetrics(
        timestamp_ms=100.0,
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
        perclos_coverage_pct=0.0,
        perclos_is_reliable=False,
        has_face=True,
        face_lost_duration_ms=0.0,
        downward_dwell_ms=2500.0,
    )
    r1 = engine.evaluate(m_slump, conf_slump)

    # Alert remains active; Timer 2 resolution cannot initiate because candidate != NORMAL
    assert r1.alert_active is True
    assert r1.unresolved_alert_state == DrowsinessState.DROWSINESS_WARNING
