"""
Personalised Driver Behaviour Profile (PDBP) and Staged 4-Zone Guided Calibration Engine.

Implements:
- Staged Calibration:
  - Stage 1 (Quick Center-Only): Calibrates CENTER zone primary baseline and derives heuristic zone offsets
  - Stage 2 (Full 4-Zone Guided): Calibrates separate behavioural baselines for:
      1. CENTER (Looking straight ahead - Primary Reference)
      2. LEFT_MIRROR (Looking toward left mirror)
      3. RIGHT_MIRROR (Looking toward right mirror)
      4. DASHBOARD (Looking down toward instrument cluster)
- Robust Median & Scaled MAD (1.4826 * MAD + eps) statistics with minimum floors
- Dual-Anchor Drift-Safe Baseline Adaptation (frozen during suspected drowsiness or low reliability)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.app.ai.blink_kinematics import BlinkEvent
from backend.app.ai.geometry import PoseZone


def compute_median_and_scaled_mad(
    values: List[float], min_floor: float = 0.01, eps: float = 1e-6
) -> Tuple[float, float]:
    """
    Compute (median, max(1.4826 * MAD + eps, min_floor)) for robust z-score normalization.
    """
    if not values:
        return 0.0, min_floor
    arr = np.asarray(values, dtype=np.float64)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    scaled_mad = max(1.4826 * mad + eps, min_floor)
    return med, scaled_mad


def compute_robust_z(
    x: float, median_val: float, scaled_mad: float, eps: float = 1e-6
) -> float:
    """
    Compute robust deviation score:
        z_r = (x - median(B)) / (1.4826 * MAD(B) + eps)
    """
    return float((x - median_val) / (scaled_mad + eps))


@dataclass
class ZoneBaseline:
    """Zone-specific EAR and head-pose baseline."""

    zone: str
    ear_median: float
    ear_mad_scaled: float
    pitch_median_deg: float
    yaw_median_deg: float
    sample_count: int

    @property
    def dynamic_closure_threshold(self) -> float:
        """Driver-specific and zone-specific eye closure cutoff."""
        return max(0.11, min(self.ear_median * 0.76, self.ear_median - 2.3 * self.ear_mad_scaled))


@dataclass
class DriverBehaviourProfile:
    """
    Complete Personalised Driver Behaviour Profile (PDBP).
    Stores the immutable Calibration Anchor (B0) and the Drift-Clamped Rolling Baseline (Bt).
    """

    driver_id: str
    calibration_mode: str  # "FOUR_ZONE" or "CENTER_ONLY"
    # Primary Center Reference Baseline (B0)
    center_ear_median: float
    center_ear_mad_scaled: float
    blink_duration_median_ms: float
    blink_duration_mad_ms: float
    blink_rcr_median: float
    blink_rcr_mad: float
    blink_freq_median_per_min: float
    blink_freq_mad_per_min: float
    neutral_pitch_deg: float
    neutral_yaw_deg: float
    calibration_reliability: float
    # Per-zone baselines keyed by PoseZone value ("CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD")
    zone_baselines: Dict[str, ZoneBaseline] = field(default_factory=dict)
    # Drift-clamped active center EAR baseline (Bt)
    active_center_ear_median: Optional[float] = None

    def __post_init__(self) -> None:
        if self.active_center_ear_median is None:
            self.active_center_ear_median = self.center_ear_median

    def get_zone_baseline(self, zone: PoseZone) -> ZoneBaseline:
        """
        Return the baseline for `zone`, falling back to a scaled CENTER baseline if not present.
        """
        if zone.value in self.zone_baselines:
            return self.zone_baselines[zone.value]

        # Fallback scaling relative to primary CENTER baseline
        scale = 1.0
        if zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR):
            scale = 0.92
        elif zone == PoseZone.DASHBOARD:
            scale = 0.88

        return ZoneBaseline(
            zone=zone.value,
            ear_median=self.center_ear_median * scale,
            ear_mad_scaled=max(self.center_ear_mad_scaled, 0.015),
            pitch_median_deg=self.neutral_pitch_deg + (-12.0 if zone == PoseZone.DASHBOARD else 0.0),
            yaw_median_deg=self.neutral_yaw_deg
            + (-22.0 if zone == PoseZone.LEFT_MIRROR else (22.0 if zone == PoseZone.RIGHT_MIRROR else 0.0)),
            sample_count=0,
        )

    def update_drift_safe_baseline(
        self,
        recent_open_ear: float,
        is_driver_alert: bool,
        camera_reliability: float,
        learning_rate: float = 0.005,
        max_drift_ratio: float = 0.08,
    ) -> bool:
        """
        Hybrid controlled baseline update:
        - Frozen during warnings, suspected drowsiness, or poor camera visibility.
        - Clamped within +/-8% of initial calibration anchor B0 so progressive fatigue
          is never learned as the driver's new normal.
        """
        if not is_driver_alert or camera_reliability < 0.78:
            return False

        current = self.active_center_ear_median or self.center_ear_median
        candidate = (1.0 - learning_rate) * current + learning_rate * recent_open_ear
        min_allowed = self.center_ear_median * (1.0 - max_drift_ratio)
        max_allowed = self.center_ear_median * (1.0 + max_drift_ratio)
        self.active_center_ear_median = float(np.clip(candidate, min_allowed, max_allowed))
        return True

    def save_json(self, path: str | Path) -> None:
        """Persist the driver's profile to JSON."""
        out_path = Path(path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "DriverBehaviourProfile":
        """Load a saved driver profile from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        zones_raw = data.pop("zone_baselines", {})
        zones = {k: ZoneBaseline(**v) for k, v in zones_raw.items()}
        return cls(**data, zone_baselines=zones)

    @classmethod
    def create_default_uncalibrated(cls, driver_id: str = "default_driver") -> "DriverBehaviourProfile":
        """Create a reasonable population-average profile before personal calibration is run."""
        center_zone = ZoneBaseline(
            zone=PoseZone.CENTER.value,
            ear_median=0.285,
            ear_mad_scaled=0.020,
            pitch_median_deg=0.0,
            yaw_median_deg=0.0,
            sample_count=0,
        )
        return cls(
            driver_id=driver_id,
            calibration_mode="UNCALIBRATED_DEFAULT",
            center_ear_median=0.285,
            center_ear_mad_scaled=0.020,
            blink_duration_median_ms=185.0,
            blink_duration_mad_ms=35.0,
            blink_rcr_median=0.92,
            blink_rcr_mad=0.14,
            blink_freq_median_per_min=15.0,
            blink_freq_mad_per_min=4.0,
            neutral_pitch_deg=0.0,
            neutral_yaw_deg=0.0,
            calibration_reliability=0.50,
            zone_baselines={PoseZone.CENTER.value: center_zone},
        )


@dataclass
class CalibrationStepConfig:
    """Configuration for a single step in the guided calibration sequence."""

    zone: PoseZone
    duration_sec: float
    instruction: str


FOUR_ZONE_CALIBRATION_PLAN: Tuple[CalibrationStepConfig, ...] = (
    CalibrationStepConfig(
        zone=PoseZone.CENTER,
        duration_sec=18.0,
        instruction="STEP 1/4 [CENTER]: Sit naturally, look straight ahead at the road, and blink normally.",
    ),
    CalibrationStepConfig(
        zone=PoseZone.LEFT_MIRROR,
        duration_sec=6.0,
        instruction="STEP 2/4 [LEFT MIRROR]: Glance toward your left side mirror naturally.",
    ),
    CalibrationStepConfig(
        zone=PoseZone.RIGHT_MIRROR,
        duration_sec=6.0,
        instruction="STEP 3/4 [RIGHT MIRROR]: Glance toward your right side mirror naturally.",
    ),
    CalibrationStepConfig(
        zone=PoseZone.DASHBOARD,
        duration_sec=6.0,
        instruction="STEP 4/4 [DASHBOARD]: Glance slightly down toward the instrument cluster.",
    ),
)

CENTER_ONLY_CALIBRATION_PLAN: Tuple[CalibrationStepConfig, ...] = (
    CalibrationStepConfig(
        zone=PoseZone.CENTER,
        duration_sec=20.0,
        instruction="STEP 1/1 [CENTER]: Sit naturally, look straight ahead, and blink normally.",
    ),
)


class GuidedCalibrationSession:
    """
    Manages interactive 1-zone or 4-zone guided calibration, quality gating,
    and construction of a DriverBehaviourProfile.
    """

    def __init__(
        self,
        driver_id: str = "driver_001",
        four_zone_mode: bool = True,
        step_duration_sec: Optional[float] = None,
    ) -> None:
        self.driver_id = driver_id
        self.four_zone_mode = four_zone_mode
        self.step_duration_sec = step_duration_sec
        base_steps = FOUR_ZONE_CALIBRATION_PLAN if four_zone_mode else CENTER_ONLY_CALIBRATION_PLAN
        if step_duration_sec is not None:
            self.steps = tuple(
                CalibrationStepConfig(
                    zone=s.zone,
                    duration_sec=float(step_duration_sec),
                    instruction=s.instruction,
                )
                for s in base_steps
            )
        else:
            self.steps = base_steps

        self.is_running: bool = False
        self.is_completed: bool = False
        self.rejection_reason: Optional[str] = None

        self._step_idx: int = 0
        self._step_start_ts_ms: Optional[float] = None
        self._total_frames_by_zone: Dict[str, int] = {s.zone.value: 0 for s in self.steps}
        self._valid_frames_by_zone: Dict[str, int] = {s.zone.value: 0 for s in self.steps}
        self._ear_samples_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._pitch_samples_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._yaw_samples_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._reliabilities: List[float] = []
        self._collected_blinks: List[BlinkEvent] = []

        # Target-Acquisition Spatial Gating
        if step_duration_sec is not None and step_duration_sec <= 1.0:
            self.target_samples_per_zone: int = max(4, int(step_duration_sec * 25))
        else:
            self.target_samples_per_zone = 35  # ~1.4 - 1.7 seconds of stable gaze per zone
        self._learned_neutral_pitch: float = 0.0
        self._learned_neutral_yaw: float = 0.0
        self._is_aligned: bool = False
        self._feedback_msg: str = "Aligning..."

    def start(self, timestamp_ms: Optional[float] = None) -> None:
        """Start or restart the guided calibration workflow."""
        self.is_running = True
        self.is_completed = False
        self.rejection_reason = None
        self._step_idx = 0
        self._step_start_ts_ms = timestamp_ms
        self._learned_neutral_pitch = 0.0
        self._learned_neutral_yaw = 0.0
        self._is_aligned = False
        self._feedback_msg = "Look straight ahead at center roadway"
        for s in self.steps:
            z = s.zone.value
            self._total_frames_by_zone[z] = 0
            self._valid_frames_by_zone[z] = 0
            self._ear_samples_by_zone[z] = []
            self._pitch_samples_by_zone[z] = []
            self._yaw_samples_by_zone[z] = []
        self._reliabilities.clear()
        self._collected_blinks.clear()

    @property
    def current_step(self) -> CalibrationStepConfig:
        return self.steps[min(self._step_idx, len(self.steps) - 1)]

    def is_pose_aligned_with_target(
        self,
        target_zone: PoseZone,
        pitch_deg: float,
        yaw_deg: float,
    ) -> Tuple[bool, str]:
        """
        Verify whether the driver's head is oriented toward target_zone.
        Returns (is_aligned, feedback_instruction).
        """
        ref_yaw = self._learned_neutral_yaw if self._step_idx > 0 else 0.0
        ref_pitch = self._learned_neutral_pitch if self._step_idx > 0 else 0.0
        rel_yaw = yaw_deg - ref_yaw
        rel_pitch = pitch_deg - ref_pitch

        if target_zone == PoseZone.CENTER:
            if abs(rel_yaw) > 13.0 or abs(rel_pitch) > 11.0:
                return False, "Look straight ahead at center roadway"
            return True, "Holding Steady (Center Roadway)"

        elif target_zone == PoseZone.LEFT_MIRROR:
            if rel_yaw > -10.0:
                return False, "Turn head toward Driver Mirror  <-"
            return True, "Holding Steady (Driver Mirror)"

        elif target_zone == PoseZone.RIGHT_MIRROR:
            if rel_yaw < 10.0:
                return False, "Turn head toward Passenger Mirror  ->"
            return True, "Holding Steady (Passenger Mirror)"

        elif target_zone == PoseZone.DASHBOARD:
            if rel_pitch > -7.5:
                return False, "Glance downward at Dashboard  v"
            return True, "Holding Steady (Dashboard)"

        return True, "Holding Steady"

    def get_progress(self, timestamp_ms: float) -> Tuple[int, int, float, str, bool, str]:
        """
        Return (current_step_num, total_steps, step_progress_ratio, instruction_text, is_aligned, feedback_text).
        """
        if not self.is_running:
            return 0, len(self.steps), 0.0, "Calibration idle", False, "Idle"
        step = self.current_step
        z_key = step.zone.value
        samples = len(self._ear_samples_by_zone.get(z_key, []))
        ratio = float(np.clip(samples / max(1, self.target_samples_per_zone), 0.0, 1.0))
        return (
            self._step_idx + 1,
            len(self.steps),
            ratio,
            step.instruction,
            self._is_aligned,
            self._feedback_msg,
        )

    def ingest_frame(
        self,
        timestamp_ms: float,
        face_detected: bool,
        geometric_reliability: float,
        ear_value: float,
        pitch_deg: float,
        yaw_deg: float,
        is_eye_currently_closed: bool,
        completed_blink: Optional[BlinkEvent] = None,
    ) -> Optional[DriverBehaviourProfile]:
        """
        Feed one frame of measurements into the active calibration step.
        Progress ONLY fills when the head is actively oriented into the target zone!
        Returns the finalized `DriverBehaviourProfile` when all steps complete.
        """
        if not self.is_running:
            return None

        if self._step_start_ts_ms is None:
            self._step_start_ts_ms = timestamp_ms

        step = self.current_step
        z_key = step.zone.value
        self._total_frames_by_zone[z_key] += 1

        is_aligned, feedback = self.is_pose_aligned_with_target(step.zone, pitch_deg, yaw_deg)
        self._is_aligned = is_aligned
        self._feedback_msg = feedback

        # ONLY accumulate valid open-eye samples when head is oriented toward target!
        if face_detected and geometric_reliability >= 0.45 and is_aligned:
            self._valid_frames_by_zone[z_key] += 1
            self._reliabilities.append(geometric_reliability)
            # Only record open-eye EAR samples (exclude mid-blink frames)
            if not is_eye_currently_closed and ear_value > 0.12:
                self._ear_samples_by_zone[z_key].append(ear_value)
                self._pitch_samples_by_zone[z_key].append(pitch_deg)
                self._yaw_samples_by_zone[z_key].append(yaw_deg)

        if completed_blink is not None:
            self._collected_blinks.append(completed_blink)

        # Step advances ONLY when required target samples are gathered in this zone!
        current_samples = len(self._ear_samples_by_zone.get(z_key, []))
        if current_samples >= self.target_samples_per_zone:
            # If Step 1 (CENTER) finishes, lock in driver's neutral posture!
            if self._step_idx == 0:
                p_list = self._pitch_samples_by_zone.get(z_key, [0.0])
                y_list = self._yaw_samples_by_zone.get(z_key, [0.0])
                self._learned_neutral_pitch = float(np.median(p_list)) if p_list else 0.0
                self._learned_neutral_yaw = float(np.median(y_list)) if y_list else 0.0

            self._step_idx += 1
            self._step_start_ts_ms = timestamp_ms
            self._is_aligned = False
            if self._step_idx >= len(self.steps):
                return self._finalize()

        # Timeout safety guard: if excessive frames pass in a zone without converging
        max_zone_frames = max(60, self.target_samples_per_zone * 6)
        if self._total_frames_by_zone[z_key] >= max_zone_frames:
            self._step_idx += 1
            self._step_start_ts_ms = timestamp_ms
            self._is_aligned = False
            if self._step_idx >= len(self.steps):
                return self._finalize()

        return None

    def _finalize(self) -> Optional[DriverBehaviourProfile]:
        """Validate calibration quality and build the DriverBehaviourProfile."""
        self.is_running = False
        center_key = PoseZone.CENTER.value
        total_center = max(1, self._total_frames_by_zone.get(center_key, 0))
        valid_center = self._valid_frames_by_zone.get(center_key, 0)
        valid_ratio = valid_center / total_center

        center_ears = self._ear_samples_by_zone.get(center_key, [])
        min_samples = 4 if (self.step_duration_sec is not None and self.step_duration_sec < 2.0) else 15
        if valid_ratio < 0.70 or len(center_ears) < min_samples:
            self.rejection_reason = (
                f"REJECTED_POOR_VISIBILITY: Valid face tracking ratio was {valid_ratio:.0%} "
                f"({len(center_ears)} open-eye samples). Please ensure good lighting and camera framing."
            )
            return None

        # Trim bottom 10% of EAR samples in each zone in case partial blink frames slipped in
        zone_baselines: Dict[str, ZoneBaseline] = {}
        for step in self.steps:
            zk = step.zone.value
            ears = sorted(self._ear_samples_by_zone.get(zk, []))
            if len(ears) >= 5:
                trim_idx = max(1, int(len(ears) * 0.10))
                trimmed_ears = ears[trim_idx:]
                ear_med, ear_mad = compute_median_and_scaled_mad(trimmed_ears, min_floor=0.015)
                pitch_med, _ = compute_median_and_scaled_mad(
                    self._pitch_samples_by_zone.get(zk, [0.0]), min_floor=1.5
                )
                yaw_med, _ = compute_median_and_scaled_mad(
                    self._yaw_samples_by_zone.get(zk, [0.0]), min_floor=1.5
                )
                zone_baselines[zk] = ZoneBaseline(
                    zone=zk,
                    ear_median=ear_med,
                    ear_mad_scaled=ear_mad,
                    pitch_median_deg=pitch_med,
                    yaw_median_deg=yaw_med,
                    sample_count=len(trimmed_ears),
                )

        center_zb = zone_baselines[center_key]

        # Blink kinematics baseline (fallback to physiological priors mingled with observed blinks if few blinks occurred)
        if len(self._collected_blinks) >= 2:
            durs = [b.duration_ms for b in self._collected_blinks]
            rcrs = [b.rcr for b in self._collected_blinks]
            dur_med, dur_mad = compute_median_and_scaled_mad(durs, min_floor=25.0)
            rcr_med, rcr_mad = compute_median_and_scaled_mad(rcrs, min_floor=0.08)
            total_cal_min = max(0.25, sum(s.duration_sec for s in self.steps) / 60.0)
            freq_med = float(np.clip(len(self._collected_blinks) / total_cal_min, 6.0, 32.0))
            freq_mad = 4.0
        else:
            dur_med, dur_mad = 185.0, 35.0
            rcr_med, rcr_mad = 0.90, 0.14
            freq_med, freq_mad = 15.0, 4.0

        mean_rel = float(np.mean(self._reliabilities)) if self._reliabilities else 0.75

        profile = DriverBehaviourProfile(
            driver_id=self.driver_id,
            calibration_mode="FOUR_ZONE" if self.four_zone_mode else "CENTER_ONLY",
            center_ear_median=center_zb.ear_median,
            center_ear_mad_scaled=center_zb.ear_mad_scaled,
            blink_duration_median_ms=dur_med,
            blink_duration_mad_ms=dur_mad,
            blink_rcr_median=rcr_med,
            blink_rcr_mad=rcr_mad,
            blink_freq_median_per_min=freq_med,
            blink_freq_mad_per_min=freq_mad,
            neutral_pitch_deg=center_zb.pitch_median_deg,
            neutral_yaw_deg=center_zb.yaw_median_deg,
            calibration_reliability=mean_rel,
            zone_baselines=zone_baselines,
        )
        self.is_completed = True
        return profile
