"""
DriveGuard — Unified Driver Feature Extractor with PDBP Personalization (Phase 7).

1. What it does:
   Fuses the individual feature modules (`ear`, `head_pose`, `mar`, `blink_detector`, `perclos`)
   into a single, coordinated, per-frame pipeline pass.
   In Phase 7 Task 2, integrates Personalized Driver Behaviour Profiles (PDBP):
   - Computes dynamic per-zone closure and bounded reopening thresholds using authoritative PDBP formulas.
   - Enforces mathematical feasibility (span >= 0.040, gap >= 0.025, margin >= 0.015).
   - Enforces blink-onset threshold latching across zone transitions and mid-blink profile swaps.
   - Disables ocular classification when uncalibrated, missing, or infeasible, while cleanly
     preserving raw geometric EAR measurements and non-ocular telemetry (head pose, MAR, yawns).

2. Why it exists:
   Provides a clean, unified interface between raw facial landmark localization (Phase 4)
   and the multi-indicator decision engine (Phase 6). Ensures all physiological
   features share consistent timestamps and synchronized temporal history.

3. Failure behavior:
   - Gracefully handles face loss by marking features invalid or uncertain without
     triggering false closure alerts or crashing.
   - When no usable profile exists, deactivates ocular channels without falsely fabricating
     face loss.
"""

from __future__ import annotations

import dataclasses
import logging
import math
from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline

from backend.app.features.blink_detector import (
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import (
    EyeAspectRatioResult,
    extract_eye_aspect_ratio,
)
from backend.app.features.head_pose import (
    HeadPoseResult,
    PoseZone,
    estimate_head_pose,
)
from backend.app.features.mar import (
    MouthAspectRatioResult,
    YawnDetector,
)
from backend.app.features.perclos import (
    PerclosCalculator,
    PerclosResult,
)
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
)

logger = logging.getLogger("driveguard.features.extractor")


def compute_zone_thresholds(
    ear_median: float,
    ear_mad_scaled: float,
) -> Tuple[float, float, float, bool]:
    """
    Compute dynamic closure threshold, reopening threshold, and feasibility for an EAR baseline.

    Closure formula (authoritative PDBP):
      raw = min(ear_median * 0.76, ear_median - 2.3 * ear_mad_scaled)
      c = clip(raw, 0.12, 0.22)

    Reopening formula (bounded):
      r = min(0.30, max(c + 0.025, min(ear_median * 0.82, ear_median - 0.015)))

    Feasibility requirements:
      span = ear_median - c >= 0.040
      c < r
      r - c >= 0.025
      ear_median - r >= 0.015
    """
    raw_c = min(ear_median * 0.76, ear_median - 2.3 * ear_mad_scaled)
    c = float(np.clip(raw_c, 0.12, 0.22))
    span = ear_median - c
    r = min(0.30, max(c + 0.025, min(ear_median * 0.82, ear_median - 0.015)))

    is_feasible = bool(
        (span >= 0.040 - 1e-9)
        and (c < r)
        and (r - c >= 0.025 - 1e-9)
        and (ear_median - r >= 0.015 - 1e-9)
    )
    return c, r, span, is_feasible


@dataclass(frozen=True)
class DriverFeatures:
    """
    Unified immutable record of all physiological driver measurements for one frame.
    """
    timestamp_ms: float
    has_face: bool
    face_status: FaceLandmarkStatus
    ear: EyeAspectRatioResult
    head_pose: HeadPoseResult
    mar: MouthAspectRatioResult
    blink: BlinkSnapshot
    perclos: PerclosResult

    @property
    def summary_text(self) -> str:
        """Brief human-readable diagnostic tag for HUD overlay."""
        if not self.has_face:
            return f"FACE: {self.face_status.value}"
        return (
            f"EAR: {self.ear.compensated_ear:.2f} | "
            f"PERCLOS: {self.perclos.perclos_pct:.1f}% | "
            f"Blinks: {self.blink.total_blink_count} | "
            f"Pose: {self.head_pose.zone.value} | "
            f"MAR: {self.mar.mar:.2f}"
        )


class DriverFeatureExtractor:
    """
    Coordinated multi-feature pipeline transforming 478 3D landmarks into DriverFeatures.
    Enhanced in Phase 7 to support Personalized Driver Behaviour Profiles (PDBP).
    """

    def __init__(
        self,
        ear_closure_threshold: float = 0.20,
        perclos_window_sec: float = 60.0,
        min_yawn_duration_ms: float = 1500.0,
        neutral_pitch_deg: float = 0.0,
        neutral_yaw_deg: float = 0.0,
        is_mirrored: bool = False,
        open_ear_baseline: Optional[float] = None,
        auto_adaptive_thresholds: bool = True,
        profile: Optional[DriverBehaviourProfile] = None,
    ) -> None:
        """
        Initialize the feature extraction pipeline.
        """
        self.ear_closure_threshold = ear_closure_threshold
        self.neutral_pitch_deg = neutral_pitch_deg
        self.neutral_yaw_deg = neutral_yaw_deg
        self.is_mirrored = is_mirrored
        self.open_ear_baseline = open_ear_baseline
        self.auto_adaptive_thresholds = auto_adaptive_thresholds
        self._open_ear_history: Deque[float] = deque(maxlen=150)

        # Phase 7 PDBP Integration state
        self._profile: Optional[DriverBehaviourProfile] = None
        self._is_profile_feasible: bool = False
        self._zone_thresholds: Dict[PoseZone, Tuple[float, float]] = {}
        self._latched_thresholds: Optional[Tuple[float, float]] = None
        self._pending_thresholds: Optional[Tuple[float, float]] = None
        self._current_active_thresholds: Optional[Tuple[float, float]] = None

        # Profile swap during blink protection
        self._pending_profile_switch: bool = False
        self._pending_profile_disable: bool = False
        self._next_profile: Optional[DriverBehaviourProfile] = None
        self._next_zone_thresholds: Dict[PoseZone, Tuple[float, float]] = {}

        initial_closure = ear_closure_threshold
        initial_reopen = max(ear_closure_threshold + 0.03, ear_closure_threshold * 1.15)
        if open_ear_baseline is not None and open_ear_baseline > 0.10:
            initial_closure = min(0.26, max(0.16, open_ear_baseline * 0.70))
            initial_reopen = min(0.30, max(0.20, open_ear_baseline * 0.82))

        self._yawn_detector = YawnDetector(min_yawn_duration_ms=min_yawn_duration_ms)
        self._blink_detector = BlinkKinematicsDetector(
            closure_threshold=initial_closure,
            reopen_threshold=initial_reopen,
        )
        self._perclos_calculator = PerclosCalculator(window_sec=perclos_window_sec)
        self._last_zone: Optional[PoseZone] = None

        if profile is not None:
            self.set_profile(profile)

    @property
    def profile(self) -> Optional[DriverBehaviourProfile]:
        """Currently active DriverBehaviourProfile, or None if uncalibrated."""
        return self._profile

    @property
    def is_profile_feasible(self) -> bool:
        """True only when a valid profile with a feasible CENTER baseline is active."""
        return self._is_profile_feasible

    @property
    def is_ocular_active(self) -> bool:
        """True if ocular classification is currently operational and active."""
        return self._is_profile_feasible

    def disable_profile(self) -> None:
        """Deactivate ocular classification with blink-latch protection."""
        self.set_profile(None)

    @property
    def has_calibrated_profile(self) -> bool:
        """True if any profile object is loaded, regardless of feasibility."""
        return self._profile is not None

    @property
    def active_thresholds(self) -> Optional[Tuple[float, float]]:
        """Current active (closure, reopen) thresholds applied to EAR and blink detector."""
        return self._current_active_thresholds

    @property
    def active_closure_threshold(self) -> float:
        """Currently active closure threshold (or 0.0 if ocular is inactive)."""
        if not self._is_profile_feasible or self._current_active_thresholds is None:
            return 0.0
        return self._current_active_thresholds[0]

    @property
    def active_reopening_threshold(self) -> float:
        """Currently active reopening threshold (or 0.0 if ocular is inactive)."""
        if not self._is_profile_feasible or self._current_active_thresholds is None:
            return 0.0
        return self._current_active_thresholds[1]

    @property
    def latched_thresholds(self) -> Optional[Tuple[float, float]]:
        """Threshold pair latched for an in-flight blink, or None if detector is OPEN."""
        return self._latched_thresholds

    @property
    def pending_thresholds(self) -> Optional[Tuple[float, float]]:
        """Queued target thresholds waiting for in-flight blink completion."""
        return self._pending_thresholds

    @property
    def blink_detector(self) -> BlinkKinematicsDetector:
        """The underlying BlinkKinematicsDetector instance."""
        return self._blink_detector

    @staticmethod
    def validate_profile(profile: Any) -> Tuple[bool, str]:
        """
        Validate structure, types, and values of a DriverBehaviourProfile.
        Rejects booleans/floats for sample counts, non-finite values, and invalid zones.
        """
        from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline

        if not isinstance(profile, DriverBehaviourProfile):
            return False, f"Expected DriverBehaviourProfile instance, got {type(profile).__name__}"

        if not (isinstance(profile.calibration_mode, str) and profile.calibration_mode.strip()):
            return False, "Profile has invalid or empty calibration_mode"

        if not math.isfinite(profile.center_ear_median) or profile.center_ear_median <= 0.0:
            return False, f"Invalid center_ear_median: {profile.center_ear_median}"

        if not math.isfinite(profile.center_ear_mad_scaled) or profile.center_ear_mad_scaled <= 0.0:
            return False, f"Invalid center_ear_mad_scaled: {profile.center_ear_mad_scaled}"

        if not (math.isfinite(profile.neutral_pitch_deg) and math.isfinite(profile.neutral_yaw_deg)):
            return False, "Neutral head pose angles must be finite numbers"

        if not isinstance(profile.zone_baselines, dict) or not profile.zone_baselines:
            return False, "Profile zone_baselines must be a non-empty dictionary"

        # Check for CENTER zone baseline
        center_key = None
        for zk in profile.zone_baselines:
            name = zk.value if isinstance(zk, PoseZone) else str(zk)
            if name == "CENTER":
                center_key = zk
                break

        if center_key is None:
            return False, "Profile lacks a mandatory CENTER zone baseline"

        # Validate each zone baseline
        for zk, zb in profile.zone_baselines.items():
            zk_name = zk.value if isinstance(zk, PoseZone) else str(zk)
            if not isinstance(zb, ZoneBaseline):
                return False, f"Zone '{zk_name}' baseline is not a ZoneBaseline instance"

            # Strict integer check for sample_count (reject bool, float, str)
            if isinstance(zb.sample_count, bool) or not isinstance(zb.sample_count, int):
                return False, f"Zone '{zk_name}' sample_count must be an integer, got {type(zb.sample_count).__name__}"

            if zb.sample_count < 0:
                return False, f"Zone '{zk_name}' sample_count cannot be negative ({zb.sample_count})"

            if not math.isfinite(zb.ear_median) or zb.ear_median <= 0.0:
                return False, f"Zone '{zk_name}' has non-finite or non-positive ear_median: {zb.ear_median}"

            if not math.isfinite(zb.ear_mad_scaled) or zb.ear_mad_scaled <= 0.0:
                return False, f"Zone '{zk_name}' has non-finite or non-positive ear_mad_scaled: {zb.ear_mad_scaled}"

            if not (math.isfinite(zb.pitch_median_deg) and math.isfinite(zb.yaw_median_deg)):
                return False, f"Zone '{zk_name}' has non-finite neutral angles"

            # Verify dynamic_closure_threshold property consistency
            raw_c = min(zb.ear_median * 0.76, zb.ear_median - 2.3 * zb.ear_mad_scaled)
            expected_c = float(np.clip(raw_c, 0.12, 0.22))
            if abs(zb.dynamic_closure_threshold - expected_c) > 1e-4:
                return False, (
                    f"Zone '{zk_name}' dynamic_closure_threshold ({zb.dynamic_closure_threshold}) "
                    f"inconsistent with formula ({expected_c:.4f})"
                )

        return True, "Profile structurally valid"

    def _evaluate_profile_feasibility(
        self, profile: DriverBehaviourProfile
    ) -> Tuple[bool, str, Dict[PoseZone, Tuple[float, float]]]:
        """
        Evaluate mathematical feasibility of CENTER and peripheral zones.
        Implements zone selection and fallback hierarchy.
        """
        center_zb = None
        for zk, zb in profile.zone_baselines.items():
            name = zk.value if isinstance(zk, PoseZone) else str(zk)
            if name == "CENTER":
                center_zb = zb
                break

        if center_zb is None:
            return False, "Missing CENTER baseline", {}

        c_center, r_center, span_center, center_feasible = compute_zone_thresholds(
            center_zb.ear_median, center_zb.ear_mad_scaled
        )

        if not center_feasible:
            return (
                False,
                f"Infeasible CENTER zone: median={center_zb.ear_median:.4f}, cutoff={c_center:.4f}, span={span_center:.4f} < 0.040",
                {},
            )

        thresholds: Dict[PoseZone, Tuple[float, float]] = {
            PoseZone.CENTER: (c_center, r_center),
            PoseZone.UNKNOWN: (c_center, r_center),
        }

        # Evaluate peripheral zones
        peripheral_zones = (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD)
        for pz in peripheral_zones:
            pzb = None
            for zk, zb in profile.zone_baselines.items():
                name = zk.value if isinstance(zk, PoseZone) else str(zk)
                if name == pz.value:
                    pzb = zb
                    break

            if pzb is None or pzb.sample_count == 0:
                # Backfilled zone -> use empirical CENTER thresholds
                thresholds[pz] = (c_center, r_center)
            else:
                c_p, r_p, span_p, p_feasible = compute_zone_thresholds(pzb.ear_median, pzb.ear_mad_scaled)
                if p_feasible:
                    thresholds[pz] = (c_p, r_p)
                else:
                    # Infeasible empirical peripheral zone -> fallback to empirical CENTER
                    thresholds[pz] = (c_center, r_center)

        return True, "Feasible", thresholds

    def _is_in_active_blink(self) -> bool:
        """Check if blink detector is currently in an in-flight blink waveform."""
        return self._blink_detector._state in (
            BlinkState.CLOSING,
            BlinkState.CLOSED,
            BlinkState.REOPENING,
            BlinkState.PROLONGED_CLOSURE,
        )

    def set_profile(self, profile: Optional[DriverBehaviourProfile]) -> bool:
        """
        Activate or replace driver profile with validation and blink latch protection.
        Returns True if a valid and feasible profile was activated, False otherwise.
        """
        if profile is None:
            if self._is_in_active_blink():
                self._pending_profile_disable = True
                self._pending_profile_switch = False
            else:
                self._profile = None
                self._is_profile_feasible = False
                self._zone_thresholds.clear()
                self._current_active_thresholds = None
                self._latched_thresholds = None
                self._pending_thresholds = None
            return False

        valid, err_msg = self.validate_profile(profile)
        if not valid:
            logger.warning("Rejected profile validation: %s", err_msg)
            if self._is_in_active_blink():
                self._pending_profile_disable = True
                self._pending_profile_switch = False
            else:
                self._profile = None
                self._is_profile_feasible = False
                self._zone_thresholds.clear()
            return False

        feasible, reason, zone_thresh = self._evaluate_profile_feasibility(profile)
        if not feasible:
            logger.warning("Profile rejected due to feasibility failure: %s", reason)
            if self._is_in_active_blink():
                self._pending_profile_disable = True
                self._pending_profile_switch = False
            else:
                self._profile = None
                self._is_profile_feasible = False
                self._zone_thresholds.clear()
            return False

        # Profile is valid and feasible
        if self._is_in_active_blink():
            self._next_profile = profile
            self._next_zone_thresholds = zone_thresh
            self._pending_profile_switch = True
            self._pending_profile_disable = False
        else:
            self._profile = profile
            self._zone_thresholds = zone_thresh
            self._is_profile_feasible = True
            self._pending_profile_switch = False
            self._pending_profile_disable = False
            self.neutral_pitch_deg = profile.neutral_pitch_deg
            self.neutral_yaw_deg = profile.neutral_yaw_deg
            c_center, r_center = zone_thresh[PoseZone.CENTER]
            self._current_active_thresholds = (c_center, r_center)
            self._blink_detector.update_thresholds(c_center, r_center)

        return True

    def get_effective_thresholds(self) -> Tuple[float, float]:
        """
        Return the currently active threshold pair (closure, reopen).
        """
        if self._is_profile_feasible:
            if self._latched_thresholds is not None:
                return self._latched_thresholds
            if self._current_active_thresholds is not None:
                return self._current_active_thresholds
            return self._zone_thresholds.get(
                PoseZone.CENTER,
                (self.ear_closure_threshold, max(self.ear_closure_threshold + 0.03, self.ear_closure_threshold * 1.15)),
            )

        return (
            self.ear_closure_threshold,
            max(self.ear_closure_threshold + 0.03, self.ear_closure_threshold * 1.15),
        )

    def set_calibrated_baseline(self, baseline: Optional[float]) -> None:
        """Update or clear driver-specific calibrated open-eye EAR baseline."""
        self.open_ear_baseline = baseline

    def extract(
        self,
        landmark_result: FaceLandmarkResult,
        frame_width: int = 1280,
        frame_height: int = 720,
        is_mirrored: Optional[bool] = None,
    ) -> DriverFeatures:
        """
        Extract all geometric, kinematic, and temporal features from a FaceLandmarkResult.
        """
        ts_ms = float(landmark_result.timestamp_ms)
        has_face = landmark_result.has_face and len(landmark_result.landmarks) >= 468
        aspect_ratio = frame_width / max(1.0, float(frame_height))
        mirrored = self.is_mirrored if is_mirrored is None else is_mirrored

        if has_face:
            # 1. 6-DoF Head Pose Estimation & Cockpit Zone
            pose_res = estimate_head_pose(
                landmarks=landmark_result.landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
                neutral_pitch_deg=self.neutral_pitch_deg,
                neutral_yaw_deg=self.neutral_yaw_deg,
                is_mirrored=mirrored,
                current_zone=self._last_zone,
            )
            self._last_zone = pose_res.zone if pose_res.is_valid else None
            current_zone = pose_res.zone if pose_res.is_valid else PoseZone.CENTER

            # 2. Mouth Aspect Ratio & Yawn Tracking (always operational with face present)
            mar_res = self._yawn_detector.update(
                landmarks=landmark_result.landmarks,
                timestamp_ms=ts_ms,
                aspect_ratio=aspect_ratio,
                has_face=has_face,
                is_looking_away=pose_res.is_looking_away,
            )

            # 3. Check Ocular Monitoring Status & Threshold Selection
            if self._is_profile_feasible:
                prev_state = self._blink_detector._state
                is_in_blink = prev_state in (
                    BlinkState.CLOSING,
                    BlinkState.CLOSED,
                    BlinkState.REOPENING,
                    BlinkState.PROLONGED_CLOSURE,
                )

                target_c, target_r = self._zone_thresholds.get(
                    current_zone, self._zone_thresholds[PoseZone.CENTER]
                )

                if is_in_blink:
                    # Latch thresholds during active blink
                    if self._latched_thresholds is None:
                        self._latched_thresholds = self._current_active_thresholds or (target_c, target_r)
                    active_c, active_r = self._latched_thresholds
                    if (target_c, target_r) != (active_c, active_r):
                        self._pending_thresholds = (target_c, target_r)
                else:
                    # OPEN or UNCERTAIN_GAP: safely apply pending profile updates
                    if self._pending_profile_disable:
                        self._profile = None
                        self._is_profile_feasible = False
                        self._zone_thresholds.clear()
                        self._pending_profile_disable = False
                        self._latched_thresholds = None
                        self._pending_thresholds = None
                    elif self._pending_profile_switch:
                        self._profile = self._next_profile
                        self._zone_thresholds = self._next_zone_thresholds
                        self._pending_profile_switch = False
                        self._next_profile = None
                        self._next_zone_thresholds = {}
                        target_c, target_r = self._zone_thresholds.get(
                            current_zone, self._zone_thresholds[PoseZone.CENTER]
                        )
                        active_c, active_r = target_c, target_r
                        self._latched_thresholds = None
                        self._pending_thresholds = None
                    elif self._pending_thresholds is not None:
                        active_c, active_r = self._pending_thresholds
                        self._pending_thresholds = None
                        self._latched_thresholds = None
                    else:
                        active_c, active_r = target_c, target_r
                        self._latched_thresholds = None

            if not self._is_profile_feasible:
                # Ocular Monitoring is DISABLED (Uncalibrated, Missing, or Infeasible Profile)
                # Compute raw EAR measurements from landmarks
                raw_c = self.ear_closure_threshold
                ear_res_raw = extract_eye_aspect_ratio(
                    landmarks=landmark_result.landmarks,
                    yaw_deg=pose_res.rel_yaw_deg,
                    pitch_deg=pose_res.rel_pitch_deg,
                    aspect_ratio=aspect_ratio,
                    closure_threshold=raw_c,
                    is_mirrored=mirrored,
                )

                # Explicitly invalidate classification while preserving raw measurements
                ear_res = dataclasses.replace(
                    ear_res_raw,
                    closure_threshold=0.0,
                    is_closed=False,
                    is_left_closed=False,
                    is_right_closed=False,
                    is_valid=False,
                    is_left_valid=False,
                    is_right_valid=False,
                )

                # Pass actual face presence with both eye validity flags as False
                # to trigger UNCERTAIN_GAP without falsely fabricating face loss
                blink_res = self._blink_detector.update(
                    left_ear=ear_res_raw.left_ear,
                    right_ear=ear_res_raw.right_ear,
                    compensated_ear=ear_res_raw.weighted_ear,
                    timestamp_ms=ts_ms,
                    has_face=has_face,
                    left_valid=False,
                    right_valid=False,
                )

                # PERCLOS receives invalid observation rather than fabricated open sample
                perclos_res = self._perclos_calculator.update(
                    timestamp_ms=ts_ms,
                    is_closed=None,
                    has_face=has_face,
                    is_valid=False,
                )

            else:
                # Ocular Monitoring is ACTIVE with verified personalized thresholds
                self._current_active_thresholds = (active_c, active_r)

                # 1. EAR extraction using identical active closure threshold
                ear_res = extract_eye_aspect_ratio(
                    landmarks=landmark_result.landmarks,
                    yaw_deg=pose_res.rel_yaw_deg,
                    pitch_deg=pose_res.rel_pitch_deg,
                    aspect_ratio=aspect_ratio,
                    closure_threshold=active_c,
                    is_mirrored=mirrored,
                )

                # 2. Update blink detector using identical threshold pair
                self._blink_detector.update_thresholds(active_c, active_r)

                valid_eye_tracking = has_face and ear_res.is_valid
                blink_res = self._blink_detector.update(
                    left_ear=ear_res.left_ear,
                    right_ear=ear_res.right_ear,
                    compensated_ear=ear_res.weighted_ear,
                    timestamp_ms=ts_ms,
                    has_face=valid_eye_tracking,
                    left_valid=ear_res.is_left_valid,
                    right_valid=ear_res.is_right_valid,
                )

                # 3. Post-blink latch management
                new_state = blink_res.state
                if new_state in (BlinkState.OPEN, BlinkState.UNCERTAIN_GAP):
                    self._latched_thresholds = None
                elif new_state in (
                    BlinkState.CLOSING,
                    BlinkState.CLOSED,
                    BlinkState.REOPENING,
                    BlinkState.PROLONGED_CLOSURE,
                ):
                    self._latched_thresholds = (active_c, active_r)

                # 4. PERCLOS uses exact same ear_res.is_closed
                perclos_res = self._perclos_calculator.update(
                    timestamp_ms=ts_ms,
                    is_closed=ear_res.is_closed if ear_res.is_valid else None,
                    has_face=has_face,
                    is_valid=ear_res.is_valid,
                )

        else:
            # Face Loss Branch — gracefully freeze/invalidate without false alarms
            self._last_zone = None
            self._latched_thresholds = None
            self._pending_thresholds = None
            pose_res = HeadPoseResult(
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
                zone=PoseZone.UNKNOWN,
                is_looking_away=False,
                is_valid=False,
                rel_pitch_deg=0.0,
                rel_yaw_deg=0.0,
                rel_roll_deg=0.0,
                rot_mat=None,
            )
            ear_res = EyeAspectRatioResult(
                left_ear=0.0,
                right_ear=0.0,
                weighted_ear=0.0,
                compensated_ear=0.0,
                closure_threshold=self.ear_closure_threshold,
                is_left_closed=False,
                is_right_closed=False,
                is_closed=False,
                is_valid=False,
                is_left_valid=False,
                is_right_valid=False,
            )
            mar_res = self._yawn_detector.update(
                landmarks=(),
                timestamp_ms=ts_ms,
                aspect_ratio=aspect_ratio,
                has_face=False,
            )
            blink_res = self._blink_detector.update(
                left_ear=0.0,
                right_ear=0.0,
                compensated_ear=0.0,
                timestamp_ms=ts_ms,
                has_face=False,
                left_valid=False,
                right_valid=False,
            )
            perclos_res = self._perclos_calculator.update(
                timestamp_ms=ts_ms,
                is_closed=None,
                has_face=False,
                is_valid=False,
            )

        return DriverFeatures(
            timestamp_ms=ts_ms,
            has_face=has_face,
            face_status=landmark_result.status,
            ear=ear_res,
            head_pose=pose_res,
            mar=mar_res,
            blink=blink_res,
            perclos=perclos_res,
        )

    def reset(self) -> None:
        """Reset internal temporal states for a new monitoring session."""
        self._yawn_detector.reset()
        self._blink_detector.reset()
        self._perclos_calculator.reset()
        self._open_ear_history.clear()
        self._latched_thresholds = None
        self._pending_thresholds = None
        self._last_zone = None
        self._pending_profile_switch = False
        self._pending_profile_disable = False
        self._next_profile = None
        self._next_zone_thresholds.clear()


validate_profile = DriverFeatureExtractor.validate_profile


