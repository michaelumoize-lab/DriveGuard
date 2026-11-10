"""
Personalised Adaptive Risk Engine & Explainable Alert State Machine.

Implements:
- Zone-aware baseline lookup (CENTER primary reference + LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD)
- Context-aware mirror-check handling:
  - Normal transient mirror glances (<= 3.0s) use that mirror zone's baseline and suppress
    false head-pose penalties.
  - Sustained non-center gaze (> 3.5s) flags distracted gaze without confusing it with sleepiness.
- Directional robust z-scores (z_EAR, z_blink_duration, z_RCR, z_PERCLOS)
- Explainable progressive alerts (NORMAL, WARNING, CRITICAL, MONITORING_UNAVAILABLE)
  with persistence, hysteresis, and recovery cooldowns.
- Parallel Fixed-Threshold Evaluator for side-by-side academic comparison against PDBP.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, List, Optional

import numpy as np

from backend.app.ai.blink_kinematics import (
    BlinkClassification,
    BlinkState,
    TemporalKinematicsSnapshot,
)
from backend.app.ai.cascade_classifier import CascadeStageResult
from backend.app.ai.gaze_attention import GazeAttentionSnapshot
from backend.app.ai.geometry import FrameGeometryFeatures, PoseZone
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, compute_robust_z


class AlertState(str, Enum):
    """Primary DriveGuard alert states."""

    NORMAL = "NORMAL"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    MONITORING_UNAVAILABLE = "MONITORING_UNAVAILABLE"


@dataclass
class DeviationScores:
    """Directional robust z-scores relative to the driver's personal baseline."""

    z_ear: float
    z_blink_duration: float
    z_blink_rcr: float
    z_perclos: float
    active_zone: str
    zone_ear_baseline: float
    zone_closure_threshold: float
    is_mirror_check_context: bool
    z_blink_avr: float = 0.0
    cv_ibi: float = 0.0
    gaze_entropy_ht: float = 1.20
    au4_brow_strain: float = 0.0
    is_visual_distraction: bool = False
    is_cognitive_tunneling: bool = False


@dataclass
class RiskAssessmentResult:
    """Complete explainable output of the DriveGuard Risk Engine for a single frame."""

    timestamp_ms: float
    state: AlertState
    risk_score: float
    camera_quality: str  # "GOOD", "DEGRADED", "UNAVAILABLE"
    deviations: DeviationScores
    reasons: List[str] = field(default_factory=list)
    # Parallel conventional fixed-threshold baseline output for Experiment 1 comparison
    conventional_fixed_state: AlertState = AlertState.NORMAL
    conventional_fixed_ear_threshold: float = 0.22


class PersonalisedAdaptiveRiskEngine:
    """
    Stateful per-session risk evaluator with rolling persistence and hysteresis.
    """

    def __init__(
        self,
        profile: Optional[DriverBehaviourProfile] = None,
        fixed_ear_threshold: float = 0.22,
    ) -> None:
        self.profile: DriverBehaviourProfile = (
            profile or DriverBehaviourProfile.create_default_uncalibrated()
        )
        self.fixed_ear_threshold = fixed_ear_threshold

        self.current_state: AlertState = AlertState.NORMAL
        self._face_lost_since_ms: Optional[float] = None
        self._non_center_zone_since_ms: Optional[float] = None
        self._downward_pitch_since_ms: Optional[float] = None

        # Hysteresis timers
        self._high_warning_since_ms: Optional[float] = None
        self._high_critical_since_ms: Optional[float] = None
        self._recovery_below_crit_since_ms: Optional[float] = None
        self._recovery_below_warn_since_ms: Optional[float] = None

        # Conventional fixed-threshold state tracker for live comparison
        self._fixed_closed_since_ms: Optional[float] = None

        # Rolling 5-second open-EAR buffer in CENTER zone for sustained EAR reduction check
        self._recent_ear_window: Deque[tuple[float, float]] = deque()

    def set_profile(self, profile: DriverBehaviourProfile) -> None:
        """Update the active driver profile (e.g., after guided calibration completes)."""
        self.profile = profile
        self.reset_state()

    def reset_state(self) -> None:
        """Reset alert state machine timers."""
        self.current_state = AlertState.NORMAL
        self._face_lost_since_ms = None
        self._non_center_zone_since_ms = None
        self._downward_pitch_since_ms = None
        self._high_warning_since_ms = None
        self._high_critical_since_ms = None
        self._recovery_below_crit_since_ms = None
        self._recovery_below_warn_since_ms = None
        self._fixed_closed_since_ms = None
        self._recent_ear_window.clear()

    def evaluate_conventional_fixed(
        self,
        timestamp_ms: float,
        face_detected: bool,
        ear_value: float,
        perclos_60s: float,
    ) -> AlertState:
        """
        Evaluate a standard non-personalised detector (fixed EAR < 0.22) on the exact same frame
        so the user and examiners can compare False Alarms and Recall side-by-side.
        """
        if not face_detected:
            self._fixed_closed_since_ms = None
            return AlertState.MONITORING_UNAVAILABLE

        if ear_value < self.fixed_ear_threshold:
            if self._fixed_closed_since_ms is None:
                self._fixed_closed_since_ms = timestamp_ms
            closed_dur_ms = timestamp_ms - self._fixed_closed_since_ms
            if closed_dur_ms >= 1000.0 or perclos_60s >= 20.0:
                return AlertState.CRITICAL
            if closed_dur_ms >= 350.0 or perclos_60s >= 12.0:
                return AlertState.WARNING
        else:
            self._fixed_closed_since_ms = None
            if perclos_60s >= 15.0:
                return AlertState.WARNING

        return AlertState.NORMAL

    def evaluate(
        self,
        geom: FrameGeometryFeatures,
        cascade: CascadeStageResult,
        kinematics: TemporalKinematicsSnapshot,
        gaze: Optional[GazeAttentionSnapshot] = None,
    ) -> RiskAssessmentResult:
        """
        Compute personalised deviation scores, context-aware fatigue risk R_t,
        and hysteresis state transitions with explainable reason codes.
        """
        ts = geom.timestamp_ms
        conv_state = self.evaluate_conventional_fixed(
            ts, geom.face_detected, geom.yaw_weighted_ear, kinematics.perclos_60s_pct
        )

        # --- 1. Camera / Monitoring Availability Check ---
        if not geom.face_detected or geom.geometric_reliability < 0.22:
            if self._face_lost_since_ms is None:
                self._face_lost_since_ms = ts
            lost_dur_ms = ts - self._face_lost_since_ms
            if lost_dur_ms >= 1200.0:
                self.current_state = AlertState.MONITORING_UNAVAILABLE
                return RiskAssessmentResult(
                    timestamp_ms=ts,
                    state=AlertState.MONITORING_UNAVAILABLE,
                    risk_score=0.0,
                    camera_quality="UNAVAILABLE",
                    deviations=DeviationScores(
                        z_ear=0.0,
                        z_blink_duration=0.0,
                        z_blink_rcr=0.0,
                        z_perclos=0.0,
                        active_zone=PoseZone.OUT_OF_BOUNDS.value,
                        zone_ear_baseline=self.profile.center_ear_median,
                        zone_closure_threshold=0.20,
                        is_mirror_check_context=False,
                        z_blink_avr=0.0,
                        cv_ibi=0.0,
                        gaze_entropy_ht=1.20,
                        au4_brow_strain=0.0,
                        is_visual_distraction=False,
                        is_cognitive_tunneling=False,
                    ),
                    reasons=["FACE_NOT_VISIBLE_OR_OBSCURED"],
                    conventional_fixed_state=conv_state,
                    conventional_fixed_ear_threshold=self.fixed_ear_threshold,
                )
        else:
            self._face_lost_since_ms = None

        camera_quality = "GOOD" if geom.geometric_reliability >= 0.65 else "DEGRADED"

        # --- 2. Pose Context & Zone-Specific Baseline Selection ---
        zone = geom.pose_zone
        zone_base = self.profile.get_zone_baseline(zone)

        is_mirror_check = False
        if zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD):
            if self._non_center_zone_since_ms is None:
                self._non_center_zone_since_ms = ts
            glance_dur_ms = ts - self._non_center_zone_since_ms
            # A brief glance (<= 3.0s) is normal driving mirror/instrument checking context
            is_mirror_check = glance_dur_ms <= 3000.0
        else:
            self._non_center_zone_since_ms = None

        # Track rolling 5-second normalized EAR relative to the active zone.
        # If geometric reliability is degraded and the cascade invoked Stage 2, use the arbitrated closure score
        # so landmark glare jitter does not corrupt the 5-second openness average.
        if cascade.cnn_invoked and geom.geometric_reliability < 0.65:
            effective_ear = zone_base.ear_median * (1.0 - 0.50 * cascade.fused_closure_score)
        else:
            effective_ear = geom.yaw_weighted_ear

        ear_ratio_to_zone = effective_ear / max(zone_base.ear_median, 0.12)
        normalized_center_equiv_ear = ear_ratio_to_zone * (
            self.profile.active_center_ear_median or self.profile.center_ear_median
        )
        self._recent_ear_window.append((ts, normalized_center_equiv_ear))
        while self._recent_ear_window and self._recent_ear_window[0][0] < (ts - 5000.0):
            self._recent_ear_window.popleft()

        mean_5s_ear = float(np.mean([x[1] for x in self._recent_ear_window]))

        # --- 3. Directional Robust Deviation Scores (z_r >= 0 for fatigue direction) ---
        ref_ear = self.profile.active_center_ear_median or self.profile.center_ear_median
        raw_z_ear = compute_robust_z(
            ref_ear - mean_5s_ear, 0.0, self.profile.center_ear_mad_scaled
        )
        # During a normal brief mirror check, dampen EAR deviation sensitivity by 40%
        if is_mirror_check:
            raw_z_ear *= 0.60
        z_ear = max(0.0, raw_z_ear)

        effective_dur_mad = max(self.profile.blink_duration_mad_ms, 35.0)
        if kinematics.recent_blink_duration_ms is not None:
            raw_z_dur = compute_robust_z(
                kinematics.recent_blink_duration_ms,
                self.profile.blink_duration_median_ms,
                effective_dur_mad,
            )
            z_blink_dur = max(0.0, raw_z_dur)
        else:
            z_blink_dur = 0.0

        effective_rcr_mad = max(self.profile.blink_rcr_mad, 0.14)
        if kinematics.recent_blink_rcr is not None:
            # Slower reopening velocity -> lower RCR than baseline -> positive fatigue z_rcr
            raw_z_rcr = compute_robust_z(
                self.profile.blink_rcr_median - kinematics.recent_blink_rcr,
                0.0,
                effective_rcr_mad,
            )
            z_blink_rcr = max(0.0, raw_z_rcr)
        else:
            z_blink_rcr = 0.0

        # Amplitude-to-Velocity Ratio (AVR) deviation (Optalert JDS metric)
        if kinematics.recent_blink_avr is not None:
            raw_z_avr = compute_robust_z(kinematics.recent_blink_avr, 0.08, 0.035)
            z_blink_avr = max(0.0, raw_z_avr)
        else:
            z_blink_avr = 0.0

        # Inter-Blink Interval Variability (CV_IBI)
        cv_ibi = kinematics.blink_cv_ibi or 0.0

        # Facial Action Unit AU4 Brow Lowerer (brow furrowing strain against sleepiness)
        au4_brow_strain = geom.blendshape_brow_down or 0.0
        is_brow_strained = (au4_brow_strain >= 0.40) and (z_ear >= 1.0)

        # Euro NCAP 2026 Gaze Attention & Distraction
        ht_val = gaze.transition_entropy_ht if gaze is not None else 1.20
        is_visual_distraction = gaze.is_off_road_dwell_exceeded if gaze is not None else False
        is_blank_stare = gaze.is_blank_stare_detected if gaze is not None else False

        # Personal blink rate deviation (elevated blink bursts relative to personal cadence baseline)
        freq_med = getattr(self.profile, "blink_freq_median_per_min", 16.0) or 16.0
        freq_mad = max(getattr(self.profile, "blink_freq_mad_per_min", 4.5) or 4.5, 3.5)
        if kinematics.perclos_coverage_pct >= 35.0 and kinematics.blink_rate_per_min > 0:
            z_blink_freq = max(0.0, compute_robust_z(kinematics.blink_rate_per_min, freq_med, freq_mad))
        else:
            z_blink_freq = 0.0

        # PERCLOS baseline anchored to the driver's personal resting PERCLOS (default ~3.5-4.5%)
        # Only contribute to linear risk when observation coverage meets reliability threshold
        base_perclos = float(np.clip(getattr(self.profile, "baseline_perclos_pct", 4.0) or 4.0, 1.5, 8.5))
        if kinematics.perclos_reliable:
            z_perclos = max(0.0, (kinematics.perclos_60s_pct - max(4.2, base_perclos + 1.0)) / 3.8)
        else:
            z_perclos = 0.0

        # Sensitivity Presets: HIGH, STANDARD (Default), or RELAXED/HIGHWAY
        sensitivity = str(getattr(self.profile, "sensitivity_level", "STANDARD") or "STANDARD").upper()
        if sensitivity == "HIGH":
            dynamic_perclos_cutoff = max(12.0, base_perclos * 2.0)
            microsleep_cutoff_ms = max(950.0, 3.0 * self.profile.blink_duration_median_ms)
            linear_center = 1.30
            warn_threshold = 0.50
        elif sensitivity in ("RELAXED", "HIGHWAY"):
            dynamic_perclos_cutoff = max(16.0, base_perclos * 3.0)
            microsleep_cutoff_ms = max(1350.0, 4.0 * self.profile.blink_duration_median_ms)
            linear_center = 1.65
            warn_threshold = 0.65
        else:
            # STANDARD (Recommended)
            dynamic_perclos_cutoff = max(14.0, base_perclos * 2.5)
            microsleep_cutoff_ms = max(1100.0, 3.5 * self.profile.blink_duration_median_ms)
            linear_center = 1.45
            warn_threshold = 0.56

        # Repeated abnormal blink clusters (multiple prolonged blinks or sluggish RCR)
        is_abnormal_cluster = kinematics.abnormal_blink_cluster_detected

        # --- 4. Head Nodding & Yawning Supporting Indicators ---
        rel_pitch = geom.pitch_deg - self.profile.neutral_pitch_deg
        is_head_nodding = False
        if rel_pitch <= -14.0 and not is_mirror_check:
            if self._downward_pitch_since_ms is None:
                self._downward_pitch_since_ms = ts
            elif (ts - self._downward_pitch_since_ms) >= 1600.0:
                is_head_nodding = True
        else:
            self._downward_pitch_since_ms = None

        # --- 5. Combined Fatigue-Risk Score R_t in [0, 1] ---
        # Weighted combination of robust deviations
        linear_risk = (
            0.28 * min(z_perclos, 4.5)
            + 0.22 * min(z_blink_dur, 4.5)
            + 0.16 * min(z_blink_rcr, 4.0)
            + 0.14 * min(z_ear, 4.0)
            + 0.12 * min(z_blink_avr, 4.0)
            + 0.05 * min(z_blink_freq, 3.0)
            + 0.10 * min(max(0.0, (cv_ibi - 0.45) / 0.35), 3.0)
            + (0.35 if is_head_nodding else 0.0)
            + (0.24 if is_abnormal_cluster else 0.0)
            + (0.20 if is_brow_strained else 0.0)
            + (0.35 if is_visual_distraction else 0.0)
            + (0.28 if is_blank_stare else 0.0)
            + (0.20 if (kinematics.is_yawning or kinematics.yawn_count_5min >= 2) else 0.0)
        )
        # Logistic mapping centered at linear_center (governed by sensitivity preset)
        risk_score = float(1.0 / (1.0 + np.exp(-1.85 * (linear_risk - linear_center))))

        # Safety-critical microsleep override: continuous eye closure >= microsleep_cutoff_ms
        # Observation gap periods must never fabricate or trigger a microsleep event
        is_microsleep = (
            kinematics.is_eye_closed
            and kinematics.blink_state != BlinkState.UNCERTAIN_GAP
            and kinematics.current_closure_duration_ms >= microsleep_cutoff_ms
            and abs(geom.yaw_deg - self.profile.neutral_yaw_deg) < 28.0
        )
        if is_microsleep:
            risk_score = max(risk_score, 0.95)
        elif kinematics.is_eye_closed and kinematics.current_closure_duration_ms >= 550.0:
            risk_score = max(risk_score, 0.64)

        # --- 6. Explainable Reason Generation ---
        reasons: List[str] = []
        if is_microsleep:
            reasons.append("PROLONGED_EYE_CLOSURE_MICROSLEEP")
        if is_abnormal_cluster:
            reasons.append("REPEATED_ABNORMAL_BLINK_CLUSTER")
        if kinematics.completed_blink and kinematics.completed_blink.classification in (
            BlinkClassification.PROLONGED,
            BlinkClassification.MICROSLEEP_CANDIDATE,
        ):
            reasons.append("PROLONGED_BLINK_DETECTED")
        personal_dur_cutoff_ms = max(
            310.0, self.profile.blink_duration_median_ms + 2.2 * effective_dur_mad
        )
        if (
            z_blink_dur >= 2.2
            and kinematics.recent_blink_duration_ms is not None
            and kinematics.recent_blink_duration_ms >= personal_dur_cutoff_ms
        ) or (
            kinematics.is_eye_closed
            and kinematics.current_closure_duration_ms >= max(480.0, self.profile.blink_duration_median_ms * 2.1)
        ):
            reasons.append("BLINK_DURATION_ABOVE_PERSONAL_BASELINE")
        if z_blink_rcr >= 2.0:
            reasons.append("EYELID_REOPENING_VELOCITY_SLOWED")
        if z_blink_avr >= 2.0:
            reasons.append("ELEVATED_BLINK_AMPLITUDE_VELOCITY_RATIO")
        if cv_ibi >= 0.80:
            reasons.append("ERRATIC_INTER_BLINK_INTERVAL_VARIABILITY")
        if is_brow_strained:
            reasons.append("INVOLUNTARY_BROW_STRAIN_AU4")
        if is_visual_distraction:
            reasons.append("VISUAL_DISTRACTION_OFF_ROAD_DWELL")
        if is_blank_stare:
            reasons.append("COGNITIVE_TUNNELING_BLANK_STARE")
        if gaze is not None and gaze.transition_entropy_ht < 0.35 and gaze.recent_zone_transitions_count >= 1:
            reasons.append("LOW_GAZE_TRANSITION_ENTROPY")
        if kinematics.perclos_reliable and (z_perclos >= 1.8 or kinematics.perclos_60s_pct >= dynamic_perclos_cutoff):
            reasons.append("INCREASED_PERCLOS")
        if not kinematics.perclos_reliable and kinematics.perclos_coverage_pct < 50.0:
            reasons.append("INSUFFICIENT_PERCLOS_OBSERVATION_COVERAGE")
        if kinematics.sampling_gap_detected:
            reasons.append("OBSERVATION_SAMPLING_GAP_DETECTED")
        if z_ear >= 2.0:
            reasons.append("SUSTAINED_EYE_OPENNESS_REDUCTION")
        if is_head_nodding:
            reasons.append("SUSTAINED_DOWNWARD_HEAD_PITCH")
        if kinematics.is_yawning or kinematics.yawn_count_5min >= 2:
            reasons.append("FREQUENT_OR_PROLONGED_YAWNING")
        if is_mirror_check:
            reasons.append(f"CONTEXT_MIRROR_CHECK_{zone.value}")

        # --- 7. Hysteresis State Machine Transitions ---
        if self.current_state == AlertState.MONITORING_UNAVAILABLE:
            self.current_state = AlertState.NORMAL

        if is_microsleep:
            self.current_state = AlertState.CRITICAL
            self._recovery_below_crit_since_ms = None
            self._recovery_below_warn_since_ms = None
        else:
            # Escalation checks with persistence
            if risk_score >= 0.78:
                if self._high_critical_since_ms is None:
                    self._high_critical_since_ms = ts
                if (ts - self._high_critical_since_ms) >= 1200.0:
                    self.current_state = AlertState.CRITICAL
            else:
                self._high_critical_since_ms = None

            if self.current_state != AlertState.CRITICAL:
                if risk_score >= warn_threshold:
                    if self._high_warning_since_ms is None:
                        self._high_warning_since_ms = ts
                    if (ts - self._high_warning_since_ms) >= 1800.0:
                        self.current_state = AlertState.WARNING
                else:
                    self._high_warning_since_ms = None

            # De-escalation checks with hysteresis cooldown
            if self.current_state == AlertState.CRITICAL:
                if risk_score < 0.60:
                    if self._recovery_below_crit_since_ms is None:
                        self._recovery_below_crit_since_ms = ts
                    elif (ts - self._recovery_below_crit_since_ms) >= 4000.0:
                        self.current_state = AlertState.WARNING
                        self._recovery_below_crit_since_ms = None
                else:
                    self._recovery_below_crit_since_ms = None

            if self.current_state == AlertState.WARNING:
                if risk_score < 0.35:
                    if self._recovery_below_warn_since_ms is None:
                        self._recovery_below_warn_since_ms = ts
                    elif (ts - self._recovery_below_warn_since_ms) >= 5000.0:
                        self.current_state = AlertState.NORMAL
                        self._recovery_below_warn_since_ms = None
                else:
                    self._recovery_below_warn_since_ms = None

        # --- 8. Controlled Drift-Safe Baseline Update (only when clearly NORMAL & CENTER) ---
        if (
            self.current_state == AlertState.NORMAL
            and risk_score < 0.22
            and zone == PoseZone.CENTER
            and not kinematics.is_eye_closed
        ):
            self.profile.update_drift_safe_baseline(
                recent_open_ear=geom.yaw_weighted_ear,
                is_driver_alert=True,
                camera_reliability=geom.geometric_reliability,
            )

        deviations = DeviationScores(
            z_ear=round(z_ear, 2),
            z_blink_duration=round(z_blink_dur, 2),
            z_blink_rcr=round(z_blink_rcr, 2),
            z_perclos=round(z_perclos, 2),
            active_zone=zone.value,
            zone_ear_baseline=round(zone_base.ear_median, 3),
            zone_closure_threshold=round(zone_base.dynamic_closure_threshold, 3),
            is_mirror_check_context=is_mirror_check,
            z_blink_avr=round(z_blink_avr, 2),
            cv_ibi=round(cv_ibi, 3),
            gaze_entropy_ht=round(ht_val, 3),
            au4_brow_strain=round(au4_brow_strain, 3),
            is_visual_distraction=is_visual_distraction,
            is_cognitive_tunneling=is_blank_stare,
        )

        return RiskAssessmentResult(
            timestamp_ms=ts,
            state=self.current_state,
            risk_score=round(risk_score, 3),
            camera_quality=camera_quality,
            deviations=deviations,
            reasons=reasons,
            conventional_fixed_state=conv_state,
            conventional_fixed_ear_threshold=self.fixed_ear_threshold,
        )
