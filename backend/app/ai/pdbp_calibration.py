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

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import struct
from types import MappingProxyType
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union
from dataclasses import asdict, dataclass, field

import numpy as np

from backend.app.ai.blink_kinematics import BlinkEvent
from backend.app.ai.geometry import PoseZone
from backend.app.core.time_utils import validate_iso_timestamp
from backend.app.features.head_pose import is_valid_rotation_matrix


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
    """Zone-specific EAR, asymmetry, and head-pose baseline."""

    zone: str
    ear_median: float
    ear_mad_scaled: float
    pitch_median_deg: float
    yaw_median_deg: float
    sample_count: int
    ear_left_median: Optional[float] = None
    ear_right_median: Optional[float] = None
    blendshape_open_median: Optional[float] = None
    confidence_median: Optional[float] = None

    @property
    def dynamic_closure_threshold(self) -> float:
        """Driver-specific and zone-specific eye closure cutoff."""
        raw = min(self.ear_median * 0.76, self.ear_median - 2.3 * self.ear_mad_scaled)
        return float(np.clip(raw, 0.12, 0.22))


@dataclass
class DriverBehaviourProfile:
    """
    Complete Personalised Driver Behaviour Profile (PDBP).
    Stores the immutable Calibration Anchor (B0) and the Drift-Clamped Rolling Baseline (Bt).
    """

    driver_id: str
    calibration_mode: str  # "FOUR_ZONE", "THREE_ZONE", or "CENTER_ONLY"
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
    # Extended Temporal Cadence Baselines (2-Min Natural Blink Stream)
    baseline_perclos_pct: float = 3.5
    blink_amplitude_median: float = 0.14
    temporal_blinks_sampled: int = 0
    temporal_window_sec: float = 120.0
    sensitivity_level: str = "STANDARD"
    version: int = 1
    calibrated_at: Optional[str] = None
    neutral_rot_mat: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        if self.active_center_ear_median is None:
            self.active_center_ear_median = self.center_ear_median

    def get_zone_baseline(self, zone: Union[PoseZone, str]) -> ZoneBaseline:
        """
        Return the baseline for `zone`, falling back to a scaled CENTER baseline if not present.
        Strictly validates that zone is either a canonical PoseZone enum member or a canonical
        PoseZone string value.
        """
        if isinstance(zone, PoseZone):
            zone_enum = zone
        elif isinstance(zone, str):
            try:
                zone_enum = PoseZone(zone)
            except ValueError:
                raise ValueError(
                    f"Invalid zone string: '{zone}'. Must be one of {[z.value for z in PoseZone]}."
                )
        else:
            raise TypeError(
                f"Zone must be a PoseZone enum or string, got {type(zone).__name__}: {zone!r}"
            )

        zone_key = zone_enum.value
        if zone_key in self.zone_baselines:
            return self.zone_baselines[zone_key]

        # Fallback scaling relative to primary CENTER baseline
        scale = 1.0
        if zone_enum in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR):
            scale = 0.92
        elif zone_enum == PoseZone.DASHBOARD:
            scale = 0.88

        return ZoneBaseline(
            zone=zone_key,
            ear_median=self.center_ear_median * scale,
            ear_mad_scaled=max(self.center_ear_mad_scaled, 0.015),
            pitch_median_deg=self.neutral_pitch_deg + (-12.0 if zone_enum == PoseZone.DASHBOARD else 0.0),
            yaw_median_deg=self.neutral_yaw_deg
            + (20.0 if zone_enum == PoseZone.LEFT_MIRROR else (-20.0 if zone_enum == PoseZone.RIGHT_MIRROR else 0.0)),
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
        data = asdict(self)
        if data.get("neutral_rot_mat") is not None:
            data["neutral_rot_mat"] = [
                [round(float(v), 6) for v in row] for row in data["neutral_rot_mat"]
            ]
        out_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "DriverBehaviourProfile":
        """Load a saved driver profile from JSON."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        zones_raw = data.pop("zone_baselines", {})
        zones = {}
        for k, v in zones_raw.items():
            valid_keys = {
                "zone",
                "ear_median",
                "ear_mad_scaled",
                "pitch_median_deg",
                "yaw_median_deg",
                "sample_count",
                "ear_left_median",
                "ear_right_median",
                "blendshape_open_median",
                "confidence_median",
            }
            clean_v = {vk: vv for vk, vv in v.items() if vk in valid_keys}
            zones[k] = ZoneBaseline(**clean_v)

        valid_profile_keys = {
            "driver_id",
            "calibration_mode",
            "center_ear_median",
            "center_ear_mad_scaled",
            "blink_duration_median_ms",
            "blink_duration_mad_ms",
            "blink_rcr_median",
            "blink_rcr_mad",
            "blink_freq_median_per_min",
            "blink_freq_mad_per_min",
            "neutral_pitch_deg",
            "neutral_yaw_deg",
            "calibration_reliability",
            "active_center_ear_median",
            "baseline_perclos_pct",
            "blink_amplitude_median",
            "temporal_blinks_sampled",
            "temporal_window_sec",
            "sensitivity_level",
            "version",
            "neutral_rot_mat",
        }
        clean_data = {k: v for k, v in data.items() if k in valid_profile_keys}
        if clean_data.get("neutral_rot_mat") is not None:
            clean_data["neutral_rot_mat"] = np.asarray(clean_data["neutral_rot_mat"], dtype=np.float64)
        clean_data["calibrated_at"] = validate_iso_timestamp(data.get("calibrated_at"))
        return cls(**clean_data, zone_baselines=zones)

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
            blink_duration_median_ms=235.0,
            blink_duration_mad_ms=40.0,
            blink_rcr_median=0.95,
            blink_rcr_mad=0.18,
            blink_freq_median_per_min=16.0,
            blink_freq_mad_per_min=4.5,
            neutral_pitch_deg=0.0,
            neutral_yaw_deg=0.0,
            calibration_reliability=0.50,
            zone_baselines={PoseZone.CENTER.value: center_zone},
            baseline_perclos_pct=3.5,
            blink_amplitude_median=0.14,
            temporal_blinks_sampled=0,
            temporal_window_sec=120.0,
        )


def _pack_float(v: Optional[float], field_name: str) -> bytes:
    if v is None:
        return b"\x00"
    if type(v) is bool:
        raise TypeError(f"Field '{field_name}' must be float, got bool {v!r}")
    try:
        f_val = float(v)
    except (TypeError, ValueError) as e:
        raise TypeError(f"Field '{field_name}' must be numeric: {e}")
    if not math.isfinite(f_val):
        raise ValueError(f"Field '{field_name}' contains non-finite float: {f_val}")
    if f_val == 0.0:
        f_val = 0.0
    return b"\x01" + struct.pack(">d", f_val)


def _pack_int(v: int, field_name: str) -> bytes:
    if type(v) is bool:
        raise TypeError(f"Field '{field_name}' must be int, got bool {v!r}")
    if not isinstance(v, (int, np.integer)):
        raise TypeError(f"Field '{field_name}' must be int, got {type(v).__name__}")
    return struct.pack(">q", int(v))


def _pack_str(v: Optional[str], field_name: str) -> bytes:
    if v is None:
        return b"\x00"
    if not isinstance(v, str):
        raise TypeError(f"Field '{field_name}' must be str, got {type(v).__name__}")
    encoded = v.encode("utf-8")
    return struct.pack(">I", len(encoded)) + encoded


def _pack_bool(v: bool, field_name: str) -> bytes:
    if type(v) is not bool and not isinstance(v, (bool, np.bool_)):
        raise TypeError(f"Field '{field_name}' must be bool, got {type(v).__name__}")
    return b"\x01" if bool(v) else b"\x00"


def _pack_matrix(mat: Any, field_name: str) -> bytes:
    if mat is None:
        return b"\x00"
    try:
        arr = np.asarray(mat, dtype=np.float64)
    except Exception as e:
        raise ValueError(f"Field '{field_name}' matrix conversion error: {e}")
    if arr.shape != (3, 3):
        raise ValueError(f"Field '{field_name}' must be 3x3, got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"Field '{field_name}' contains non-finite numbers")
    out = [b"\x01"]
    for row in arr:
        for val in row:
            f = float(val)
            if f == 0.0:
                f = 0.0
            out.append(struct.pack(">d", f))
    return b"".join(out)


def compute_canonical_profile_fingerprint(
    profile: Union[DriverBehaviourProfile, "ImmutableProfileSnapshot", Any]
) -> str:
    """
    Compute strict, deterministic 64-character SHA-256 fingerprint over all 24 profile fields
    and 11 zone fields, using big-endian IEEE 754 encoding and canonical key ordering.
    """
    h = hashlib.sha256()

    # 1. version: int
    h.update(_pack_int(profile.version, "version"))

    # 2. driver_id: str
    h.update(_pack_str(profile.driver_id, "driver_id"))

    # 3. calibrated_at: Optional[str]
    h.update(_pack_str(profile.calibrated_at, "calibrated_at"))

    # 4. is_calibrated: bool
    is_cal = getattr(profile, "is_calibrated", None)
    cal_mode = getattr(profile, "calibration_mode", None)
    if is_cal is None:
        is_cal = (cal_mode is not None and cal_mode != "UNCALIBRATED_DEFAULT")
    h.update(_pack_bool(is_cal, "is_calibrated"))

    # 5. calibration_mode: str
    cal_mode_str = cal_mode or ("4ZONE" if is_cal else "UNCALIBRATED_DEFAULT")
    h.update(_pack_str(cal_mode_str, "calibration_mode"))

    # 6-15. float statistics
    h.update(_pack_float(profile.center_ear_median, "center_ear_median"))
    h.update(_pack_float(profile.center_ear_mad_scaled, "center_ear_mad_scaled"))
    h.update(_pack_float(profile.blink_duration_median_ms, "blink_duration_median_ms"))
    h.update(_pack_float(profile.blink_duration_mad_ms, "blink_duration_mad_ms"))
    h.update(_pack_float(profile.blink_rcr_median, "blink_rcr_median"))
    h.update(_pack_float(profile.blink_rcr_mad, "blink_rcr_mad"))
    h.update(_pack_float(profile.blink_freq_median_per_min, "blink_freq_median_per_min"))
    h.update(_pack_float(profile.blink_freq_mad_per_min, "blink_freq_mad_per_min"))
    h.update(_pack_float(profile.baseline_perclos_pct, "baseline_perclos_pct"))
    h.update(_pack_float(profile.blink_amplitude_median, "blink_amplitude_median"))

    # 16. temporal_blinks_sampled: int
    h.update(_pack_int(profile.temporal_blinks_sampled, "temporal_blinks_sampled"))

    # 17. temporal_window_sec: float
    h.update(_pack_float(profile.temporal_window_sec, "temporal_window_sec"))

    # 18. sensitivity_level: str
    h.update(_pack_str(profile.sensitivity_level, "sensitivity_level"))

    # 19-21. pose and reliability
    h.update(_pack_float(profile.neutral_pitch_deg, "neutral_pitch_deg"))
    h.update(_pack_float(profile.neutral_yaw_deg, "neutral_yaw_deg"))
    h.update(_pack_float(profile.calibration_reliability, "calibration_reliability"))

    # 22. active_center_ear_median: Optional[float]
    act_ear = getattr(profile, "active_center_ear_median", None)
    h.update(_pack_float(act_ear, "active_center_ear_median"))

    # 23. neutral_rot_mat
    rot_mat = getattr(profile, "neutral_rot_mat", None)
    if rot_mat is None and hasattr(profile, "_neutral_rot_mat_tuple"):
        rot_mat = profile._neutral_rot_mat_tuple
    h.update(_pack_matrix(rot_mat, "neutral_rot_mat"))

    # 24. zone_baselines
    zb_dict = profile.zone_baselines
    if not isinstance(zb_dict, (dict, Mapping)):
        raise TypeError(f"zone_baselines must be a mapping, got {type(zb_dict).__name__}")

    zone_keys = sorted(str(k.value if isinstance(k, PoseZone) else k) for k in zb_dict.keys())
    h.update(_pack_int(len(zone_keys), "zone_baselines_count"))
    for zk in zone_keys:
        zb = zb_dict[zk] if zk in zb_dict else zb_dict[PoseZone(zk)]
        h.update(_pack_str(zk, f"zone_key_{zk}"))
        h.update(_pack_str(zb.zone, f"{zk}.zone"))
        h.update(_pack_float(zb.ear_median, f"{zk}.ear_median"))
        h.update(_pack_float(zb.ear_mad_scaled, f"{zk}.ear_mad_scaled"))
        h.update(_pack_float(zb.pitch_median_deg, f"{zk}.pitch_median_deg"))
        h.update(_pack_float(zb.yaw_median_deg, f"{zk}.yaw_median_deg"))
        h.update(_pack_int(zb.sample_count, f"{zk}.sample_count"))
        h.update(_pack_float(zb.ear_left_median, f"{zk}.ear_left_median"))
        h.update(_pack_float(zb.ear_right_median, f"{zk}.ear_right_median"))
        h.update(_pack_float(zb.blendshape_open_median, f"{zk}.blendshape_open_median"))
        h.update(_pack_float(zb.confidence_median, f"{zk}.confidence_median"))
        h.update(_pack_float(zb.dynamic_closure_threshold, f"{zk}.dynamic_closure_threshold"))

    return h.hexdigest()


@dataclass(frozen=True)
class ImmutableZoneBaseline:
    """Immutable snapshot of a single spatial zone baseline."""

    zone: str
    ear_median: float
    ear_mad_scaled: float
    pitch_median_deg: float
    yaw_median_deg: float
    sample_count: int
    ear_left_median: Optional[float] = None
    ear_right_median: Optional[float] = None
    blendshape_open_median: Optional[float] = None
    confidence_median: Optional[float] = None

    @property
    def dynamic_closure_threshold(self) -> float:
        raw = min(self.ear_median * 0.76, self.ear_median - 2.3 * self.ear_mad_scaled)
        return float(np.clip(raw, 0.12, 0.22))


@dataclass(frozen=True)
class ImmutableProfileSnapshot:
    """
    Immutable, deeply frozen snapshot of a DriverBehaviourProfile.
    Guarantees that caller-owned profile mutations cannot modify queued or active runtime states.
    """

    driver_id: str
    calibration_mode: str
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
    zone_baselines: Mapping[str, ImmutableZoneBaseline]
    active_center_ear_median: Optional[float] = None
    baseline_perclos_pct: float = 3.5
    blink_amplitude_median: float = 0.14
    temporal_blinks_sampled: int = 0
    temporal_window_sec: float = 120.0
    sensitivity_level: str = "STANDARD"
    version: int = 1
    calibrated_at: Optional[str] = None
    is_calibrated: bool = True
    _neutral_rot_mat_tuple: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = None
    fingerprint: str = ""

    @property
    def neutral_rot_mat(self) -> Optional[np.ndarray]:
        if self._neutral_rot_mat_tuple is None:
            return None
        return np.asarray(self._neutral_rot_mat_tuple, dtype=np.float64)

    def get_zone_baseline(self, zone: Union[PoseZone, str]) -> ImmutableZoneBaseline:
        """Return the baseline for `zone`, falling back to scaled CENTER if missing."""
        if isinstance(zone, PoseZone):
            zone_enum = zone
        elif isinstance(zone, str):
            try:
                zone_enum = PoseZone(zone)
            except ValueError:
                raise ValueError(
                    f"Invalid zone string: '{zone}'. Must be one of {[z.value for z in PoseZone]}."
                )
        else:
            raise TypeError(
                f"Zone must be a PoseZone enum or string, got {type(zone).__name__}: {zone!r}"
            )

        zone_key = zone_enum.value
        if zone_key in self.zone_baselines:
            return self.zone_baselines[zone_key]

        scale = 1.0
        if zone_enum in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR):
            scale = 0.92
        elif zone_enum == PoseZone.DASHBOARD:
            scale = 0.88

        return ImmutableZoneBaseline(
            zone=zone_key,
            ear_median=self.center_ear_median * scale,
            ear_mad_scaled=max(self.center_ear_mad_scaled, 0.015),
            pitch_median_deg=self.neutral_pitch_deg + (-12.0 if zone_enum == PoseZone.DASHBOARD else 0.0),
            yaw_median_deg=self.neutral_yaw_deg
            + (20.0 if zone_enum == PoseZone.LEFT_MIRROR else (-20.0 if zone_enum == PoseZone.RIGHT_MIRROR else 0.0)),
            sample_count=0,
        )

    def to_domain(self) -> DriverBehaviourProfile:
        """Convert this immutable snapshot into a mutable DriverBehaviourProfile domain object."""
        return DriverBehaviourProfile(
            driver_id=self.driver_id,
            calibration_mode=self.calibration_mode,
            center_ear_median=self.center_ear_median,
            center_ear_mad_scaled=self.center_ear_mad_scaled,
            blink_duration_median_ms=self.blink_duration_median_ms,
            blink_duration_mad_ms=self.blink_duration_mad_ms,
            blink_rcr_median=self.blink_rcr_median,
            blink_rcr_mad=self.blink_rcr_mad,
            blink_freq_median_per_min=self.blink_freq_median_per_min,
            blink_freq_mad_per_min=self.blink_freq_mad_per_min,
            neutral_pitch_deg=self.neutral_pitch_deg,
            neutral_yaw_deg=self.neutral_yaw_deg,
            calibration_reliability=self.calibration_reliability,
            zone_baselines={
                k: ZoneBaseline(
                    zone=v.zone,
                    ear_median=v.ear_median,
                    ear_mad_scaled=v.ear_mad_scaled,
                    pitch_median_deg=v.pitch_median_deg,
                    yaw_median_deg=v.yaw_median_deg,
                    sample_count=v.sample_count,
                    ear_left_median=v.ear_left_median,
                    ear_right_median=v.ear_right_median,
                    blendshape_open_median=v.blendshape_open_median,
                    confidence_median=v.confidence_median,
                )
                for k, v in self.zone_baselines.items()
            },
            active_center_ear_median=self.active_center_ear_median,
            baseline_perclos_pct=self.baseline_perclos_pct,
            blink_amplitude_median=self.blink_amplitude_median,
            temporal_blinks_sampled=self.temporal_blinks_sampled,
            temporal_window_sec=self.temporal_window_sec,
            sensitivity_level=self.sensitivity_level,
            version=self.version,
            calibrated_at=self.calibrated_at,
            neutral_rot_mat=(
                np.asarray(self._neutral_rot_mat_tuple, dtype=np.float64)
                if self._neutral_rot_mat_tuple is not None
                else None
            ),
        )


def create_immutable_profile_snapshot(profile: DriverBehaviourProfile) -> ImmutableProfileSnapshot:
    """
    Construct a verified, deeply immutable snapshot from a DriverBehaviourProfile.
    """
    if not isinstance(profile, DriverBehaviourProfile):
        raise TypeError(f"Expected DriverBehaviourProfile instance, got {type(profile).__name__}")

    imm_zones: Dict[str, ImmutableZoneBaseline] = {}
    for zk, zb in profile.zone_baselines.items():
        zk_name = zk.value if isinstance(zk, PoseZone) else str(zk)
        imm_zones[zk_name] = ImmutableZoneBaseline(
            zone=zb.zone,
            ear_median=float(zb.ear_median),
            ear_mad_scaled=float(zb.ear_mad_scaled),
            pitch_median_deg=float(zb.pitch_median_deg),
            yaw_median_deg=float(zb.yaw_median_deg),
            sample_count=int(zb.sample_count),
            ear_left_median=float(zb.ear_left_median) if zb.ear_left_median is not None else None,
            ear_right_median=float(zb.ear_right_median) if zb.ear_right_median is not None else None,
            blendshape_open_median=float(zb.blendshape_open_median) if zb.blendshape_open_median is not None else None,
            confidence_median=float(zb.confidence_median) if zb.confidence_median is not None else None,
        )

    rot_tuple: Optional[Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]] = None
    if profile.neutral_rot_mat is not None:
        arr = np.asarray(profile.neutral_rot_mat, dtype=np.float64)
        if arr.shape != (3, 3):
            raise ValueError(f"neutral_rot_mat must have shape (3, 3), got {arr.shape}")
        if not np.all(np.isfinite(arr)):
            raise ValueError("neutral_rot_mat contains non-finite numbers")
        rot_tuple = (
            (float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
            (float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
            (float(arr[2, 0]), float(arr[2, 1]), float(arr[2, 2])),
        )

    is_cal = (profile.calibration_mode != "UNCALIBRATED_DEFAULT")
    fp = compute_canonical_profile_fingerprint(profile)

    return ImmutableProfileSnapshot(
        driver_id=profile.driver_id,
        calibration_mode=profile.calibration_mode,
        center_ear_median=float(profile.center_ear_median),
        center_ear_mad_scaled=float(profile.center_ear_mad_scaled),
        blink_duration_median_ms=float(profile.blink_duration_median_ms),
        blink_duration_mad_ms=float(profile.blink_duration_mad_ms),
        blink_rcr_median=float(profile.blink_rcr_median),
        blink_rcr_mad=float(profile.blink_rcr_mad),
        blink_freq_median_per_min=float(profile.blink_freq_median_per_min),
        blink_freq_mad_per_min=float(profile.blink_freq_mad_per_min),
        neutral_pitch_deg=float(profile.neutral_pitch_deg),
        neutral_yaw_deg=float(profile.neutral_yaw_deg),
        calibration_reliability=float(profile.calibration_reliability),
        zone_baselines=MappingProxyType(imm_zones),
        active_center_ear_median=float(profile.active_center_ear_median) if profile.active_center_ear_median is not None else None,
        baseline_perclos_pct=float(profile.baseline_perclos_pct),
        blink_amplitude_median=float(profile.blink_amplitude_median),
        temporal_blinks_sampled=int(profile.temporal_blinks_sampled),
        temporal_window_sec=float(profile.temporal_window_sec),
        sensitivity_level=str(profile.sensitivity_level),
        version=int(profile.version),
        calibrated_at=profile.calibrated_at,
        is_calibrated=is_cal,
        _neutral_rot_mat_tuple=rot_tuple,
        fingerprint=fp,
    )


@dataclass(frozen=True)
class NeutralRotationResult:
    """
    Structured outcome of standalone neutral rotation aggregation.

    Metric units:
        - inlier_rms_deg: Root-mean-square geodesic residual of inliers [degrees]
        - inlier_mad_deg: Scaled Median Absolute Deviation of inlier residuals [degrees] (diagnostic)
        - pre_filter_mad_deg: Scaled Median Absolute Deviation of all candidate residuals from medoid [degrees] (diagnostic)
        - inlier_ratio: Fraction of candidates accepted as inliers in [0.0, 1.0] [dimensionless]
        - total_candidates, inlier_count: Sample counts [dimensionless]
    """

    is_accepted: bool
    """True if both dominant-cluster count and inlier dispersion bounds are satisfied."""

    neutral_rot_mat: Optional[np.ndarray]
    """3x3 orthonormal SO(3) rotation matrix (det == +1.0). None if is_accepted is False."""

    inlier_count: int
    """Number of candidate rotations within geodesic neighborhood (residual <= max_residual_deg)."""

    total_candidates: int
    """Total valid candidate rotations supplied to aggregation (N >= 18)."""

    inlier_ratio: float
    """Ratio of inlier_count to total_candidates in [0.0, 1.0]."""

    medoid_idx: Optional[int]
    """0-based index of chosen geodesic medoid in input sequence (None if medoid cannot be calculated)."""

    inlier_rms_deg: float
    """Root-mean-square geodesic residual of all inliers relative to medoid [degrees]."""

    inlier_mad_deg: float
    """Scaled Median Absolute Deviation (1.4826 * MAD) of inlier residuals [degrees] (diagnostic)."""

    pre_filter_mad_deg: float
    """Scaled Median Absolute Deviation (1.4826 * MAD) of all candidate residuals from medoid [degrees] (diagnostic)."""

    rejection_reason: Optional[str]
    """None if accepted; structured rejection code string if rejected."""


def project_onto_so3(M: np.ndarray) -> np.ndarray:
    """
    Project a 3x3 real matrix M onto SO(3) using SVD with Kabsch reflection guard.

    Solves:
        argmin_{R in SO(3)} || R - M ||_F^2

    Guarantees det(R) == +1.0 even if det(U @ Vt) == -1.0.
    """
    U, s, Vt = np.linalg.svd(M)
    d = float(np.linalg.det(U @ Vt))
    sign = 1.0 if d >= 0.0 else -1.0
    S = np.diag([1.0, 1.0, sign])
    return U @ S @ Vt


def aggregate_neutral_rotation_matrix(
    rotations: Sequence[np.ndarray],
    max_residual_deg: float = 4.5,
    max_inlier_rms_deg: float = 2.2,
    min_inlier_ratio: float = 0.60,
    min_absolute_inliers: int = 18,
) -> NeutralRotationResult:
    """
    Pure mathematical aggregation of candidate SolvePnP rotation matrices into a single R_neutral in SO(3).

    Acceptance criteria:
        1. At least max(min_absolute_inliers, ceil(min_inlier_ratio * N)) inliers within max_residual_deg (4.5°)
           of the geodesic medoid.
        2. Inlier RMS geodesic dispersion <= max_inlier_rms_deg (2.2°).

    All metric units:
        - max_residual_deg: degrees [°]
        - max_inlier_rms_deg: degrees [°]
        - inlier_rms_deg: degrees [°]
        - inlier_mad_deg: degrees [°]
        - pre_filter_mad_deg: degrees [°]

    Parameters:
        rotations: Sequence of valid 3x3 orthonormal SO(3) rotation matrices.
        max_residual_deg: Geodesic angular radius for the inlier neighborhood (default: 4.5°).
        max_inlier_rms_deg: Maximum allowable inlier RMS geodesic dispersion (default: 2.2°).
        min_inlier_ratio: Minimum ratio of inliers to total candidates (default: 0.60).
        min_absolute_inliers: Minimum absolute inlier count (default: 18).

    Returns:
        NeutralRotationResult with acceptance status, SO(3) matrix, and diagnostic dispersion metrics.
    """
    # 1. Strict validation of all candidate matrices in the input sequence
    validated_rotations: List[np.ndarray] = []
    ident = np.eye(3, dtype=np.float64)

    for idx, r in enumerate(rotations):
        try:
            arr = np.asarray(r, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"Candidate rotation matrix at index {idx} is malformed: {e}")

        if arr.shape != (3, 3):
            raise ValueError(
                f"Candidate rotation matrix at index {idx} has invalid shape {arr.shape}; expected (3, 3)"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(
                f"Candidate rotation matrix at index {idx} contains non-finite values (NaN or Inf)"
            )

        if not np.allclose(arr.T @ arr, ident, atol=1e-3):
            raise ValueError(f"Candidate rotation matrix at index {idx} is not orthogonal")

        det = float(np.linalg.det(arr))
        if abs(det - 1.0) > 1e-3:
            raise ValueError(
                f"Candidate rotation matrix at index {idx} is not in SO(3) "
                f"(det={det:.6f} != 1.0; reflections and invalid rotations are rejected)"
            )

        validated_rotations.append(arr)

    n_candidates = len(validated_rotations)

    # 2. Check candidate sample count budget
    if n_candidates < min_absolute_inliers:
        return NeutralRotationResult(
            is_accepted=False,
            neutral_rot_mat=None,
            inlier_count=0,
            total_candidates=n_candidates,
            inlier_ratio=0.0,
            medoid_idx=None,
            inlier_rms_deg=0.0,
            inlier_mad_deg=0.0,
            pre_filter_mad_deg=0.0,
            rejection_reason="INSUFFICIENT_CANDIDATE_SAMPLES",
        )

    # 3. Compute pairwise geodesic distance matrix D in degrees
    # D_ij = arccos(clip((Tr(R_i^T @ R_j) - 1) / 2, -1.0, 1.0)) * (180.0 / pi)
    dists = np.zeros((n_candidates, n_candidates), dtype=np.float64)
    for i in range(n_candidates):
        r_i = validated_rotations[i]
        for j in range(i + 1, n_candidates):
            r_j = validated_rotations[j]
            trace_val = float(np.trace(r_i.T @ r_j))
            cos_val = np.clip((trace_val - 1.0) / 2.0, -1.0, 1.0)
            deg = float(np.degrees(np.arccos(cos_val)))
            dists[i, j] = deg
            dists[j, i] = deg

    # 4. Deterministic medoid selection: argmin of distance sums
    # np.argmin deterministically selects the lowest index on ties
    distance_sums = np.sum(dists, axis=1)
    medoid_idx = int(np.argmin(distance_sums))
    residuals = dists[medoid_idx, :]

    # Diagnostic pre-filter MAD (scaled MAD of all candidate residuals from medoid)
    pre_filter_med = float(np.median(residuals))
    pre_filter_mad_deg = float(np.median(np.abs(residuals - pre_filter_med))) * 1.4826

    # 5. Filter inliers within max_residual_deg neighborhood
    inlier_mask = residuals <= max_residual_deg
    inlier_indices = np.where(inlier_mask)[0]
    inlier_count = int(len(inlier_indices))
    inlier_ratio = float(inlier_count / n_candidates)
    inlier_residuals = residuals[inlier_mask]

    # Diagnostic inlier dispersion metrics
    if inlier_count > 0:
        inlier_rms_deg = float(np.sqrt(np.mean(inlier_residuals ** 2)))
        inlier_res_med = float(np.median(inlier_residuals))
        inlier_mad_deg = float(np.median(np.abs(inlier_residuals - inlier_res_med))) * 1.4826
    else:
        inlier_rms_deg = 0.0
        inlier_mad_deg = 0.0

    # 6. Evaluate acceptance conditions
    min_required_inliers = max(min_absolute_inliers, int(np.ceil(min_inlier_ratio * n_candidates)))

    if inlier_count < min_required_inliers:
        return NeutralRotationResult(
            is_accepted=False,
            neutral_rot_mat=None,
            inlier_count=inlier_count,
            total_candidates=n_candidates,
            inlier_ratio=inlier_ratio,
            medoid_idx=medoid_idx,
            inlier_rms_deg=inlier_rms_deg,
            inlier_mad_deg=inlier_mad_deg,
            pre_filter_mad_deg=pre_filter_mad_deg,
            rejection_reason="INSUFFICIENT_INLIER_RATIO",
        )

    if inlier_rms_deg > max_inlier_rms_deg:
        return NeutralRotationResult(
            is_accepted=False,
            neutral_rot_mat=None,
            inlier_count=inlier_count,
            total_candidates=n_candidates,
            inlier_ratio=inlier_ratio,
            medoid_idx=medoid_idx,
            inlier_rms_deg=inlier_rms_deg,
            inlier_mad_deg=inlier_mad_deg,
            pre_filter_mad_deg=pre_filter_mad_deg,
            rejection_reason="EXCESSIVE_ANGULAR_DISPERSION",
        )

    # 7. SVD projection of inlier arithmetic mean onto SO(3)
    inlier_rotations = [validated_rotations[idx] for idx in inlier_indices]
    m_mean = np.mean(inlier_rotations, axis=0)
    r_neutral = project_onto_so3(m_mean)

    # Sanity verification of projected matrix
    if not (np.all(np.isfinite(r_neutral)) and np.allclose(r_neutral.T @ r_neutral, ident, atol=1e-5)):
        raise RuntimeError("SO(3) projection produced non-orthonormal matrix")

    det_proj = float(np.linalg.det(r_neutral))
    if abs(det_proj - 1.0) > 1e-5:
        raise RuntimeError(f"SO(3) projection produced matrix with invalid determinant: {det_proj:.6f}")

    return NeutralRotationResult(
        is_accepted=True,
        neutral_rot_mat=r_neutral,
        inlier_count=inlier_count,
        total_candidates=n_candidates,
        inlier_ratio=inlier_ratio,
        medoid_idx=medoid_idx,
        inlier_rms_deg=inlier_rms_deg,
        inlier_mad_deg=inlier_mad_deg,
        pre_filter_mad_deg=pre_filter_mad_deg,
        rejection_reason=None,
    )


@dataclass
class CalibrationStepConfig:
    """Configuration for a single step in the guided calibration sequence."""

    zone: PoseZone
    duration_sec: float
    instruction: str
    step_type: str = "SPATIAL_ZONE"  # "SPATIAL_ZONE" or "BLINK_CADENCE"


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

THREE_ZONE_CALIBRATION_PLAN: Tuple[CalibrationStepConfig, ...] = (
    CalibrationStepConfig(
        zone=PoseZone.CENTER,
        duration_sec=18.0,
        instruction="STEP 1/3 [CENTER]: Sit naturally, look straight ahead at the road, and blink normally.",
    ),
    CalibrationStepConfig(
        zone=PoseZone.LEFT_MIRROR,
        duration_sec=6.0,
        instruction="STEP 2/3 [LEFT MIRROR]: Glance toward your left side mirror naturally.",
    ),
    CalibrationStepConfig(
        zone=PoseZone.RIGHT_MIRROR,
        duration_sec=6.0,
        instruction="STEP 3/3 [RIGHT MIRROR]: Glance toward your right side mirror naturally.",
    ),
)

CENTER_ONLY_CALIBRATION_PLAN: Tuple[CalibrationStepConfig, ...] = (
    CalibrationStepConfig(
        zone=PoseZone.CENTER,
        duration_sec=20.0,
        instruction="STEP 1/1 [CENTER]: Sit naturally, look straight ahead, and blink normally.",
    ),
)


# Initial CENTER Alignment Policy for Oblique Camera Mounting:
# Allows camera placement off-center (e.g., A-pillar, center console, dashboard)
# within +/- 32.0 degrees yaw and -18.0 to +14.0 degrees pitch relative to camera frame.
# Beyond 35 degrees yaw, perspective foreshortening and facial landmark degradation
# degrade SO(3) PnP stability.
DEFAULT_INITIAL_CENTER_MAX_YAW_DEG: float = 32.0
DEFAULT_INITIAL_CENTER_MIN_PITCH_DEG: float = -18.0
DEFAULT_INITIAL_CENTER_MAX_PITCH_DEG: float = 14.0

DEFAULT_RECALIBRATION_CENTER_MAX_YAW_DEG: float = 13.0
DEFAULT_RECALIBRATION_CENTER_MAX_PITCH_DEG: float = 11.0


class GuidedCalibrationSession:
    """
    Manages interactive 4-zone (CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD) guided calibration,
    individual zone recalibration, quality gating, and construction of a DriverBehaviourProfile.
    """

    def __init__(
        self,
        driver_id: str = "driver_001",
        four_zone_mode: bool = True,
        three_zone_mode: bool = False,
        step_duration_sec: Optional[float] = None,
        target_zone: Optional[PoseZone | str] = None,
        existing_profile: Optional[DriverBehaviourProfile] = None,
        include_blink_baseline: bool = False,
        blink_baseline_duration_sec: float = 120.0,
        initial_center_max_yaw_deg: float = DEFAULT_INITIAL_CENTER_MAX_YAW_DEG,
        initial_center_min_pitch_deg: float = DEFAULT_INITIAL_CENTER_MIN_PITCH_DEG,
        initial_center_max_pitch_deg: float = DEFAULT_INITIAL_CENTER_MAX_PITCH_DEG,
        recalibration_center_max_yaw_deg: float = DEFAULT_RECALIBRATION_CENTER_MAX_YAW_DEG,
        recalibration_center_max_pitch_deg: float = DEFAULT_RECALIBRATION_CENTER_MAX_PITCH_DEG,
    ) -> None:
        self.driver_id = driver_id
        self.four_zone_mode = four_zone_mode
        self.three_zone_mode = three_zone_mode
        self.step_duration_sec = step_duration_sec
        self.include_blink_baseline = include_blink_baseline
        self.blink_baseline_duration_sec = float(blink_baseline_duration_sec)
        self.blink_only_mode = str(target_zone) == "BLINK_CADENCE"
        self.target_zone = None if self.blink_only_mode else target_zone
        self.existing_profile = existing_profile
        self.initial_center_max_yaw_deg = float(initial_center_max_yaw_deg)
        self.initial_center_min_pitch_deg = float(initial_center_min_pitch_deg)
        self.initial_center_max_pitch_deg = float(initial_center_max_pitch_deg)
        self.recalibration_center_max_yaw_deg = float(recalibration_center_max_yaw_deg)
        self.recalibration_center_max_pitch_deg = float(recalibration_center_max_pitch_deg)

        if self.blink_only_mode:
            base_steps: Tuple[CalibrationStepConfig, ...] = (
                CalibrationStepConfig(
                    zone=PoseZone.CENTER,
                    duration_sec=self.blink_baseline_duration_sec,
                    instruction="STEP 1/1 [2-MIN BLINK STREAM]: Sit comfortably, look straight ahead naturally, and blink normally for 2 minutes.",
                    step_type="BLINK_CADENCE",
                ),
            )
        elif self.target_zone is not None:
            zone_enum = PoseZone(self.target_zone) if isinstance(self.target_zone, str) else self.target_zone
            matching = [s for s in FOUR_ZONE_CALIBRATION_PLAN if s.zone == zone_enum]
            base_steps = (
                (matching[0],)
                if matching
                else (
                    CalibrationStepConfig(
                        zone=zone_enum,
                        duration_sec=6.0,
                        instruction=f"Calibrate [{zone_enum.value}]",
                    ),
                )
            )
        elif four_zone_mode:
            base_steps = FOUR_ZONE_CALIBRATION_PLAN
        elif three_zone_mode:
            base_steps = THREE_ZONE_CALIBRATION_PLAN
        else:
            base_steps = CENTER_ONLY_CALIBRATION_PLAN

        if step_duration_sec is not None and not self.blink_only_mode:
            spatial_steps = tuple(
                CalibrationStepConfig(
                    zone=s.zone,
                    duration_sec=float(step_duration_sec),
                    instruction=s.instruction,
                    step_type=s.step_type,
                )
                for s in base_steps
            )
        else:
            spatial_steps = base_steps

        if self.include_blink_baseline and not self.blink_only_mode and self.target_zone is None:
            total_n = len(spatial_steps) + 1
            blink_step = CalibrationStepConfig(
                zone=PoseZone.CENTER,
                duration_sec=self.blink_baseline_duration_sec,
                instruction=(
                    f"STEP {total_n}/{total_n} [2-MIN BLINK STREAM]: Sit comfortably, look ahead naturally, "
                    "and blink normally for 2 minutes so DriveGuard learns your personal blink rate & speed."
                ),
                step_type="BLINK_CADENCE",
            )
            self.steps = spatial_steps + (blink_step,)
        else:
            self.steps = spatial_steps

        self.is_running: bool = False
        self.is_completed: bool = False
        self.rejection_reason: Optional[str] = None
        self.quality_warnings: List[str] = []

        self._step_idx: int = 0
        self._step_start_ts_ms: Optional[float] = None
        self._total_frames_by_zone: Dict[str, int] = {s.zone.value: 0 for s in self.steps}
        self._valid_frames_by_zone: Dict[str, int] = {s.zone.value: 0 for s in self.steps}
        self._ear_samples_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._ear_left_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._ear_right_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._blendshape_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._pitch_samples_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._yaw_samples_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._confidence_by_zone: Dict[str, List[float]] = {s.zone.value: [] for s in self.steps}
        self._reliabilities: List[float] = []
        self._collected_blinks: List[BlinkEvent] = []
        self._candidate_rot_mats: List[np.ndarray] = []

        # 2-Minute Natural Blink Cadence Phase Accumulators
        self._blink_phase_elapsed_sec: float = 0.0
        self._blink_phase_valid_ms: float = 0.0
        self._blink_phase_closed_ms: float = 0.0
        self._last_blink_duration_ms: Optional[float] = None

        # Target-Acquisition Spatial Gating
        if step_duration_sec is not None and step_duration_sec <= 1.0:
            self.target_samples_per_zone: int = max(4, int(step_duration_sec * 25))
        else:
            self.target_samples_per_zone = 45  # ~1.5 seconds of stable gaze per zone
        self._learned_neutral_pitch: float = 0.0
        self._learned_neutral_yaw: float = 0.0
        self._is_aligned: bool = False
        self._feedback_msg: str = "Aligning..."
        self._last_pose_ts: Optional[float] = None
        self._last_pitch: Optional[float] = None
        self._last_yaw: Optional[float] = None

        # Inter-Zone Transition Grace Period (2.0s buffer to comfortably reposition between zones)
        self._inter_zone_grace_sec: float = 2.0
        self._grace_until_ts_ms: Optional[float] = None
        self._grace_zone_name: Optional[str] = None

    def start(self, timestamp_ms: Optional[float] = None) -> None:
        """Start or restart the guided calibration workflow."""
        self.is_running = True
        self.is_completed = False
        self.rejection_reason = None
        self.quality_warnings = []
        self._candidate_rot_mats = []
        self._step_idx = 0
        self._step_start_ts_ms = timestamp_ms
        self._learned_neutral_pitch = 0.0
        self._learned_neutral_yaw = 0.0
        self._is_aligned = False
        self._feedback_msg = "Look straight ahead at center roadway"
        self._grace_until_ts_ms = None
        self._grace_zone_name = None
        for s in self.steps:
            z = s.zone.value
            self._total_frames_by_zone[z] = 0
            self._valid_frames_by_zone[z] = 0
            self._ear_samples_by_zone[z] = []
            self._ear_left_by_zone[z] = []
            self._ear_right_by_zone[z] = []
            self._blendshape_by_zone[z] = []
            self._pitch_samples_by_zone[z] = []
            self._yaw_samples_by_zone[z] = []
            self._confidence_by_zone[z] = []
        self._reliabilities.clear()
        self._collected_blinks.clear()
        self._blink_phase_elapsed_sec = 0.0
        self._blink_phase_valid_ms = 0.0
        self._blink_phase_closed_ms = 0.0
        self._last_blink_duration_ms = None
        self._last_pose_ts = None
        self._last_pitch = None
        self._last_yaw = None

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
        has_calibrated_reference = (
            self.existing_profile is not None
            and (
                getattr(self.existing_profile, "is_calibrated", False)
                or getattr(self.existing_profile, "neutral_rot_mat", None) is not None
                or getattr(self.existing_profile, "neutral_yaw_deg", 0.0) != 0.0
                or getattr(self.existing_profile, "neutral_pitch_deg", 0.0) != 0.0
            )
        )

        if target_zone == PoseZone.CENTER:
            if self._step_idx == 0 and not has_calibrated_reference:
                # Dedicated initial CENTER alignment policy for fresh calibration:
                # Allows camera placement off-center (e.g., A-pillar, center console)
                # up to self.initial_center_max_yaw_deg and within [min_pitch, max_pitch].
                if abs(yaw_deg) > self.initial_center_max_yaw_deg:
                    return False, "Look straight ahead at center roadway"
                if (
                    pitch_deg < self.initial_center_min_pitch_deg
                    or pitch_deg > self.initial_center_max_pitch_deg
                ):
                    return False, "Look straight ahead at center roadway"
                return True, "Holding Steady (Center Roadway)"
            else:
                # Established reference (recalibration or subsequent step with established reference)
                if has_calibrated_reference:
                    ref_yaw = (
                        self.existing_profile.neutral_yaw_deg
                        if getattr(self.existing_profile, "neutral_yaw_deg", None) is not None
                        else 0.0
                    )
                    ref_pitch = (
                        self.existing_profile.neutral_pitch_deg
                        if getattr(self.existing_profile, "neutral_pitch_deg", None) is not None
                        else 0.0
                    )
                elif self._step_idx > 0:
                    ref_yaw = self._learned_neutral_yaw
                    ref_pitch = self._learned_neutral_pitch
                else:
                    ref_yaw = 0.0
                    ref_pitch = 0.0
                rel_yaw = yaw_deg - ref_yaw
                rel_pitch = pitch_deg - ref_pitch
                if (
                    abs(rel_yaw) > self.recalibration_center_max_yaw_deg
                    or abs(rel_pitch) > self.recalibration_center_max_pitch_deg
                ):
                    return False, "Look straight ahead at center roadway"
                return True, "Holding Steady (Center Roadway)"

        # Relative reference for peripheral zones
        if self.target_zone is not None and has_calibrated_reference:
            ref_yaw = (
                self.existing_profile.neutral_yaw_deg
                if getattr(self.existing_profile, "neutral_yaw_deg", None) is not None
                else 0.0
            )
            ref_pitch = (
                self.existing_profile.neutral_pitch_deg
                if getattr(self.existing_profile, "neutral_pitch_deg", None) is not None
                else 0.0
            )
        elif self._step_idx > 0:
            ref_yaw = self._learned_neutral_yaw
            ref_pitch = self._learned_neutral_pitch
        else:
            ref_yaw = 0.0
            ref_pitch = 0.0
        rel_yaw = yaw_deg - ref_yaw
        rel_pitch = pitch_deg - ref_pitch

        if target_zone == PoseZone.LEFT_MIRROR:
            # Driver turning left yields positive relative yaw in front-facing camera frame
            if rel_yaw < 7.0:
                return False, "Turn head toward Driver Mirror (Left)  <-"
            return True, "Holding Steady (Driver Mirror)"

        elif target_zone == PoseZone.RIGHT_MIRROR:
            # Driver turning right yields negative relative yaw in front-facing camera frame
            if rel_yaw > -7.0:
                return False, "Turn head toward Passenger Mirror (Right)  ->"
            return True, "Holding Steady (Passenger Mirror)"

        elif target_zone == PoseZone.DASHBOARD:
            # Require head yaw to be centered (|yaw| < 16.0 deg) and pitch to be intentionally down (<= -6.0 deg)
            # so returning from a side mirror cannot accidentally trigger it.
            if abs(rel_yaw) >= 16.0:
                return False, "Center your head, then glance down at Dashboard"
            if rel_pitch > -6.0:
                return False, "Glance downward at Dashboard  v"
            return True, "Holding Steady (Dashboard)"

        return True, "Holding Steady"

    def get_progress(self, timestamp_ms: float) -> Tuple[int, int, float, str, bool, str]:
        """
        Return (current_step_num, total_steps, step_progress_ratio, instruction_text, is_aligned, feedback_text).
        """
        if not self.is_running:
            return 0, len(self.steps), 0.0, "Calibration idle", False, "Idle"

        # Display transition grace countdown if currently waiting between zones
        if self._grace_until_ts_ms is not None and timestamp_ms < self._grace_until_ts_ms:
            remaining_sec = max(0.1, (self._grace_until_ts_ms - timestamp_ms) / 1000.0)
            return (
                self._step_idx + 1,
                len(self.steps),
                0.0,
                f"Get ready to glance at {self._grace_zone_name}...",
                False,
                f"Get ready to glance at {self._grace_zone_name} ({remaining_sec:.1f}s)...",
            )

        step = self.current_step
        if step.step_type == "BLINK_CADENCE":
            ratio = float(np.clip(self._blink_phase_elapsed_sec / max(1.0, step.duration_sec), 0.0, 1.0))
            return (
                self._step_idx + 1,
                len(self.steps),
                ratio,
                step.instruction,
                self._is_aligned,
                self._feedback_msg,
            )

        z_key = step.zone.value
        samples = len(self._ear_samples_by_zone.get(z_key, []))
        zone_target = (
            min(40, self.target_samples_per_zone)
            if step.zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD)
            else self.target_samples_per_zone
        )
        ratio = float(np.clip(samples / max(1, zone_target), 0.0, 1.0))
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
        is_eye_currently_closed: Optional[bool] = None,
        completed_blink: Optional[BlinkEvent] = None,
        left_ear: Optional[float] = None,
        right_ear: Optional[float] = None,
        blendshape_blink_mean: Optional[float] = None,
        landmark_confidence: Optional[float] = None,
        rot_mat: Optional[np.ndarray] = None,
        ear_geometric_valid: Optional[bool] = None,
    ) -> Optional[DriverBehaviourProfile]:
        """
        Feed one frame of measurements into the active calibration step.
        Progress ONLY fills when the head is actively oriented into the target zone and head is steady!
        Returns the finalized `DriverBehaviourProfile` when all steps complete.
        """
        if not self.is_running:
            return None

        # Inter-Zone Grace Period: Hold sampling for 2.0 seconds between zones to allow comfortable repositioning
        if self._grace_until_ts_ms is not None:
            if timestamp_ms < self._grace_until_ts_ms:
                remaining_sec = max(0.1, (self._grace_until_ts_ms - timestamp_ms) / 1000.0)
                self._is_aligned = False
                self._feedback_msg = f"Get ready to glance at {self._grace_zone_name} ({remaining_sec:.1f}s)..."
                self._step_start_ts_ms = timestamp_ms
                return None
            else:
                self._grace_until_ts_ms = None
                self._grace_zone_name = None
                self._step_start_ts_ms = timestamp_ms

        if self._step_start_ts_ms is None:
            self._step_start_ts_ms = timestamp_ms

        step = self.current_step
        z_key = step.zone.value
        self._total_frames_by_zone[z_key] = self._total_frames_by_zone.get(z_key, 0) + 1

        # Rapid head motion check (> 80 deg/s) to reject jerky frames
        is_moving_excessively = False
        dt_s = 0.04
        if self._last_pose_ts is not None and self._last_pitch is not None and self._last_yaw is not None:
            dt_s = max(1e-3, (timestamp_ms - self._last_pose_ts) / 1000.0)
            if dt_s < 0.5:
                ang_vel = float(np.hypot(pitch_deg - self._last_pitch, yaw_deg - self._last_yaw) / dt_s)
                if ang_vel > 80.0:
                    is_moving_excessively = True
        self._last_pose_ts = timestamp_ms
        self._last_pitch = pitch_deg
        self._last_yaw = yaw_deg

        # --- Special handling for the 2-Minute Natural Blink & Cadence Phase ---
        if step.step_type == "BLINK_CADENCE":
            ref_yaw = self._learned_neutral_yaw
            ref_pitch = self._learned_neutral_pitch
            rel_yaw = yaw_deg - ref_yaw
            rel_pitch = pitch_deg - ref_pitch
            # Relaxed natural posture bounds during 2-minute blink stream
            is_relaxed_forward = face_detected and abs(rel_yaw) <= 20.0 and abs(rel_pitch) <= 16.0
            self._is_aligned = is_relaxed_forward

            if not face_detected:
                self._feedback_msg = "Keep face visible in camera frame"
            elif not is_relaxed_forward:
                self._feedback_msg = "Look naturally toward the road ahead"
            else:
                step_dt_s = min(dt_s, 0.25)
                self._blink_phase_elapsed_sec += step_dt_s
                self._blink_phase_valid_ms += step_dt_s * 1000.0
                if is_eye_currently_closed:
                    self._blink_phase_closed_ms += step_dt_s * 1000.0
                self._valid_frames_by_zone[z_key] = self._valid_frames_by_zone.get(z_key, 0) + 1
                self._reliabilities.append(geometric_reliability)
                if not is_eye_currently_closed and ear_value > 0.12:
                    self._ear_samples_by_zone.setdefault(z_key, []).append(ear_value)
                rem_sec = max(0, int(round(step.duration_sec - self._blink_phase_elapsed_sec)))
                self._feedback_msg = (
                    f"Natural Blink Stream ({len(self._collected_blinks)} blinks · {rem_sec}s left)"
                )

            if completed_blink is not None and not is_moving_excessively and abs(rel_yaw) <= 18.0:
                self._collected_blinks.append(completed_blink)
                self._last_blink_duration_ms = completed_blink.duration_ms

            wall_elapsed_sec = (
                (timestamp_ms - self._step_start_ts_ms) / 1000.0
                if self._step_start_ts_ms is not None
                else 0.0
            )
            if self._blink_phase_elapsed_sec >= step.duration_sec or wall_elapsed_sec >= (step.duration_sec * 1.15):
                self._step_idx += 1
                self._step_start_ts_ms = timestamp_ms
                self._is_aligned = False
                if self._step_idx >= len(self.steps):
                    return self._finalize()
            return None

        # --- Standard Spatial Zone Alignment & Sampling ---
        is_aligned, feedback = self.is_pose_aligned_with_target(step.zone, pitch_deg, yaw_deg)
        self._is_aligned = is_aligned
        self._feedback_msg = "Hold head steady in target zone" if is_moving_excessively else feedback

        # ONLY accumulate valid open-eye samples when head is oriented toward target!
        # For lateral mirror zones and dashboard, perspective yaw foreshortens one eye
        # and reduces geometric landmark reliability naturally.
        if step.zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR):
            min_geom_rel = 0.28
            min_ear_val = 0.11
            zone_target = min(40, self.target_samples_per_zone)
        elif step.zone == PoseZone.DASHBOARD:
            min_geom_rel = 0.28
            min_ear_val = 0.10
            zone_target = min(40, self.target_samples_per_zone)
        else:
            min_geom_rel = 0.38
            min_ear_val = 0.12
            zone_target = self.target_samples_per_zone

        if (
            face_detected
            and geometric_reliability >= min_geom_rel
            and (landmark_confidence is None or landmark_confidence >= 0.35)
            and is_aligned
            and not is_moving_excessively
        ):
            # Stage 2.1 rotation candidate collection for CENTER zone:
            # Independent of EAR acceptance, but verified-closed observations (is_eye_currently_closed is True)
            # must be excluded. A None eye state remains acceptable if head-pose quality gates pass.
            if step.zone == PoseZone.CENTER and rot_mat is not None and is_eye_currently_closed is not True:
                mat = np.asarray(rot_mat, dtype=np.float64)
                if is_valid_rotation_matrix(mat):
                    self._candidate_rot_mats.append(mat)

            self._valid_frames_by_zone[z_key] += 1
            self._reliabilities.append(geometric_reliability)

            # Scalar compatibility tracking associated with valid pose observations
            if is_eye_currently_closed is not True:
                self._pitch_samples_by_zone[z_key].append(pitch_deg)
                self._yaw_samples_by_zone[z_key].append(yaw_deg)

            # EAR Baseline Accumulation:
            # - For active-profile recalibration: ONLY is_eye_currently_closed is False may contribute.
            #   True and None must be rejected.
            # - Initial bootstrap calibration may use separately agreed guided-protocol rules (not verified closed).
            # - In lateral zones, avoid letting frontal blink classifier falsely classify foreshortened eyes as closed.
            is_recalibration = (self.existing_profile is not None)
            if step.zone == PoseZone.CENTER:
                eye_open_for_baseline = (is_eye_currently_closed is False) if is_recalibration else (is_eye_currently_closed is not True)
            else:
                eye_open_for_baseline = (ear_value >= min_ear_val)

            # Tri-state geometric validity for open EAR accumulation:
            # - False: strictly rejected (never treated as open)
            # - True: explicitly verified
            # - None: backward compatible fallback
            is_geometric_ok = True if ear_geometric_valid is None else ear_geometric_valid

            if eye_open_for_baseline and is_geometric_ok and ear_value > min_ear_val:
                self._ear_samples_by_zone[z_key].append(ear_value)
                self._ear_left_by_zone[z_key].append(left_ear if left_ear is not None else ear_value)
                self._ear_right_by_zone[z_key].append(right_ear if right_ear is not None else ear_value)
                self._blendshape_by_zone[z_key].append(blendshape_blink_mean if blendshape_blink_mean is not None else 0.05)
                self._confidence_by_zone[z_key].append(landmark_confidence if landmark_confidence is not None else geometric_reliability)

        if completed_blink is not None and not is_moving_excessively:
            self._collected_blinks.append(completed_blink)
            self._last_blink_duration_ms = completed_blink.duration_ms

        # Minimum required valid observations per zone
        min_required = 18 if step.zone == PoseZone.CENTER else 16
        if self.step_duration_sec is not None and self.step_duration_sec <= 1.0:
            min_required = max(3, int(self.step_duration_sec * 20))

        current_samples = len(self._ear_samples_by_zone.get(z_key, []))
        should_advance = current_samples >= zone_target

        # Timeout safety guard: elapsed-time budget based on step.duration_sec and timestamp_ms
        step_budget_sec = step.duration_sec
        elapsed_sec = (
            (timestamp_ms - self._step_start_ts_ms) / 1000.0
            if self._step_start_ts_ms is not None
            else 0.0
        )
        time_expired = elapsed_sec >= step_budget_sec
        if time_expired and not should_advance:
            if current_samples >= min_required:
                should_advance = True
            else:
                self.quality_warnings.append(
                    f"POOR_VISIBILITY: Insufficient valid observations ({current_samples}/{min_required}) in "
                    f"zone '{step.zone.value}'. Zone baseline will use default/fallback thresholds."
                )
                should_advance = True

        if should_advance:
            # If Step 1 (CENTER) finishes, lock in driver's neutral posture!
            if self._step_idx == 0:
                p_list = self._pitch_samples_by_zone.get(z_key, [])
                y_list = self._yaw_samples_by_zone.get(z_key, [])
                if len(y_list) >= min_required:
                    self._learned_neutral_pitch = float(np.median(p_list))
                    self._learned_neutral_yaw = float(np.median(y_list))
                elif p_list and y_list:
                    self._learned_neutral_pitch = float(np.median(p_list))
                    self._learned_neutral_yaw = float(np.median(y_list))
                else:
                    self._learned_neutral_pitch = 0.0
                    self._learned_neutral_yaw = 0.0

            self._step_idx += 1
            self._step_start_ts_ms = timestamp_ms
            self._is_aligned = False
            if self._step_idx >= len(self.steps):
                return self._finalize()
            else:
                next_step = self.steps[self._step_idx]
                next_zone_name = (
                    "Dashboard"
                    if next_step.zone == PoseZone.DASHBOARD
                    else ("2-Min Blink Stream" if next_step.step_type == "BLINK_CADENCE" else next_step.zone.value.replace("_", " ").title())
                )
                self._grace_until_ts_ms = timestamp_ms + (self._inter_zone_grace_sec * 1000.0)
                self._grace_zone_name = next_zone_name

        return None

    def skip_current_step(self, timestamp_ms: Optional[float] = None) -> Optional[DriverBehaviourProfile]:
        """Manually skip current zone (advances to next or finalizes with synthetic fill)."""
        if not self.is_running:
            return None
        self._step_idx += 1
        self._step_start_ts_ms = timestamp_ms
        self._is_aligned = False
        self._grace_until_ts_ms = None
        self._grace_zone_name = None
        if self._step_idx >= len(self.steps):
            return self._finalize()
        return None

    def get_live_blink_telemetry(self) -> Dict[str, Optional[float | int | str]]:
        """Return real-time temporal blink cadence metrics during calibration."""
        step = self.current_step if self.steps else None
        phase_type = step.step_type if step else "SPATIAL_ZONE"
        target_dur = step.duration_sec if step else 5.0
        elapsed = (
            self._blink_phase_elapsed_sec
            if phase_type == "BLINK_CADENCE"
            else 0.0
        )
        n_blinks = len(self._collected_blinks)
        active_min = max(0.15, elapsed / 60.0) if elapsed > 0 else 0.25
        live_rate = float(np.clip(n_blinks / active_min, 0.0, 60.0)) if n_blinks > 0 else 0.0
        med_dur = (
            float(np.median([b.duration_ms for b in self._collected_blinks]))
            if n_blinks > 0
            else None
        )
        med_rcr = (
            float(np.median([b.rcr for b in self._collected_blinks]))
            if n_blinks > 0
            else None
        )
        return {
            "phase_type": phase_type,
            "elapsed_sec": round(elapsed, 1),
            "target_duration_sec": round(target_dur, 1),
            "collected_blinks_count": n_blinks,
            "live_blink_rate_per_min": round(live_rate, 1),
            "last_blink_duration_ms": round(self._last_blink_duration_ms, 1) if self._last_blink_duration_ms is not None else None,
            "live_median_blink_duration_ms": round(med_dur, 1) if med_dur is not None else None,
            "live_median_blink_rcr": round(med_rcr, 2) if med_rcr is not None else None,
        }

    def _compute_temporal_blink_baselines(self) -> Tuple[float, float, float, float, float, float, float, float]:
        """
        Compute (dur_med, dur_mad, rcr_med, rcr_mad, freq_med, freq_mad, amp_med, base_perclos_pct)
        from collected blink events and active observation window.
        """
        if len(self._collected_blinks) >= 2:
            durs = [float(np.clip(b.duration_ms, 70.0, 520.0)) for b in self._collected_blinks]
            rcrs = [float(np.clip(b.rcr, 0.25, 3.20)) for b in self._collected_blinks]
            amps = [float(np.clip(b.amplitude, 0.04, 0.38)) for b in self._collected_blinks]
            dur_med, dur_mad = compute_median_and_scaled_mad(durs, min_floor=35.0)
            rcr_med, rcr_mad = compute_median_and_scaled_mad(rcrs, min_floor=0.14)
            amp_med = float(np.median(amps))

            # Derive blink rate from both total window count and inter-blink intervals (IBIs)
            observed_sec = (
                self._blink_phase_elapsed_sec
                if self._blink_phase_elapsed_sec >= 10.0
                else max(15.0, sum(s.duration_sec for s in self.steps))
            )
            overall_rate = (len(self._collected_blinks) / max(0.20, observed_sec / 60.0))

            ibi_rates: List[float] = []
            for idx in range(1, len(self._collected_blinks)):
                ibi_ms = self._collected_blinks[idx].start_ts_ms - self._collected_blinks[idx - 1].start_ts_ms
                if 1200.0 <= ibi_ms <= 25000.0:
                    ibi_rates.append(60000.0 / ibi_ms)

            if len(ibi_rates) >= 2:
                ibi_med, ibi_mad = compute_median_and_scaled_mad(ibi_rates, min_floor=3.5)
                freq_med = float(np.clip(0.65 * overall_rate + 0.35 * ibi_med, 5.0, 38.0))
                freq_mad = float(np.clip(ibi_mad, 3.5, 10.0))
            else:
                freq_med = float(np.clip(overall_rate, 6.0, 34.0))
                freq_mad = 4.5
        elif len(self._collected_blinks) == 1:
            b0 = self._collected_blinks[0]
            dur_med = float(np.clip(0.6 * b0.duration_ms + 0.4 * 235.0, 160.0, 310.0))
            dur_mad = 40.0
            rcr_med = float(np.clip(0.6 * b0.rcr + 0.4 * 0.95, 0.55, 1.80))
            rcr_mad = 0.18
            amp_med = float(np.clip(b0.amplitude, 0.08, 0.28))
            freq_med, freq_mad = 16.0, 4.5
        else:
            dur_med, dur_mad = 235.0, 40.0
            rcr_med, rcr_mad = 0.95, 0.18
            amp_med = 0.14
            freq_med, freq_mad = 16.0, 4.5

        if self._blink_phase_valid_ms >= 5000.0:
            raw_perclos = (self._blink_phase_closed_ms / self._blink_phase_valid_ms) * 100.0
            base_perclos_pct = float(np.clip(raw_perclos, 1.0, 9.5))
        else:
            # Estimate resting closed-eye percentage from personal blink rate * personal blink duration
            est_closed_ms_per_min = freq_med * dur_med
            base_perclos_pct = float(np.clip((est_closed_ms_per_min / 60000.0) * 100.0, 1.5, 8.5))

        return dur_med, dur_mad, rcr_med, rcr_mad, freq_med, freq_mad, amp_med, base_perclos_pct

    def _finalize(self) -> Optional[DriverBehaviourProfile]:
        """Validate calibration quality, calculate robust baselines, and return DriverBehaviourProfile."""
        self.is_running = False
        center_key = PoseZone.CENTER.value

        dur_med, dur_mad, rcr_med, rcr_mad, freq_med, freq_mad, amp_med, base_perclos_pct = (
            self._compute_temporal_blink_baselines()
        )

        # Blink-only recalibration merge into existing profile
        if self.blink_only_mode and self.existing_profile is not None:
            self.existing_profile.blink_duration_median_ms = dur_med
            self.existing_profile.blink_duration_mad_ms = dur_mad
            self.existing_profile.blink_rcr_median = rcr_med
            self.existing_profile.blink_rcr_mad = rcr_mad
            self.existing_profile.blink_freq_median_per_min = freq_med
            self.existing_profile.blink_freq_mad_per_min = freq_mad
            self.existing_profile.blink_amplitude_median = amp_med
            self.existing_profile.baseline_perclos_pct = base_perclos_pct
            self.existing_profile.temporal_blinks_sampled = len(self._collected_blinks)
            self.existing_profile.temporal_window_sec = round(
                max(self._blink_phase_elapsed_sec, self.blink_baseline_duration_sec), 1
            )
            self.is_completed = True
            return self.existing_profile

        # If full calibration (including CENTER), verify center quality
        if any(s.zone == PoseZone.CENTER and s.step_type == "SPATIAL_ZONE" for s in self.steps):
            total_center = max(1, self._total_frames_by_zone.get(center_key, 0))
            valid_center = self._valid_frames_by_zone.get(center_key, 0)
            valid_ratio = valid_center / total_center

            center_ears = self._ear_samples_by_zone.get(center_key, [])
            min_samples = 3 if (self.step_duration_sec is not None and self.step_duration_sec < 2.0) else 10
            if valid_ratio < 0.60 or len(center_ears) < min_samples:
                self.rejection_reason = (
                    f"REJECTED_POOR_VISIBILITY: Center tracking ratio was {valid_ratio:.0%} "
                    f"({len(center_ears)} open-eye samples). Please ensure good lighting and camera framing."
                )
                return None

        # Natural Blink & Gaze Filter: Bilateral trimming
        # Discard bottom 10% (transient blinks) AND top 10% (wide-eyed staring bias)
        zone_baselines: Dict[str, ZoneBaseline] = {}
        for step in self.steps:
            if step.step_type != "SPATIAL_ZONE":
                continue
            zk = step.zone.value
            ears = sorted(self._ear_samples_by_zone.get(zk, []))
            if len(ears) >= 3:
                low_idx = int(len(ears) * 0.10) if len(ears) >= 10 else 0
                high_idx = len(ears) - (int(len(ears) * 0.10) if len(ears) >= 10 else 0)
                trimmed_ears = ears[low_idx:high_idx] if high_idx > low_idx else ears
                ear_med, ear_mad = compute_median_and_scaled_mad(trimmed_ears, min_floor=0.015)

                left_ears = self._ear_left_by_zone.get(zk, [])
                right_ears = self._ear_right_by_zone.get(zk, [])
                ear_l_med = float(np.median(left_ears)) if left_ears else ear_med
                ear_r_med = float(np.median(right_ears)) if right_ears else ear_med

                blends = self._blendshape_by_zone.get(zk, [])
                blend_med = float(np.median(blends)) if blends else 0.05

                confs = self._confidence_by_zone.get(zk, [])
                conf_med = float(np.median(confs)) if confs else 0.85

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
                    ear_left_median=ear_l_med,
                    ear_right_median=ear_r_med,
                    blendshape_open_median=blend_med,
                    confidence_median=conf_med,
                )

        # Single zone recalibration merge
        if self.target_zone is not None and self.existing_profile is not None:
            tz_str = self.target_zone.value if isinstance(self.target_zone, PoseZone) else str(self.target_zone)
            if tz_str in zone_baselines:
                new_zb = zone_baselines[tz_str]
                self.existing_profile.zone_baselines[tz_str] = new_zb
                if tz_str == PoseZone.CENTER.value:
                    self.existing_profile.center_ear_median = new_zb.ear_median
                    self.existing_profile.center_ear_mad_scaled = new_zb.ear_mad_scaled
                    self.existing_profile.active_center_ear_median = new_zb.ear_median
                    self.existing_profile.neutral_pitch_deg = new_zb.pitch_median_deg
                    self.existing_profile.neutral_yaw_deg = new_zb.yaw_median_deg
                    if self._candidate_rot_mats:
                        try:
                            rot_res = aggregate_neutral_rotation_matrix(self._candidate_rot_mats)
                            if rot_res.is_accepted and rot_res.neutral_rot_mat is not None:
                                self.existing_profile.neutral_rot_mat = rot_res.neutral_rot_mat
                        except Exception:
                            pass
                self.is_completed = True
                return self.existing_profile
            else:
                self.rejection_reason = (
                    f"REJECTED_CALIBRATION_FAILED: No valid baseline generated for zone '{tz_str}'."
                )
                return None

        if center_key not in zone_baselines:
            center_ears = self._ear_samples_by_zone.get(center_key, [])
            ear_med = float(np.median(center_ears)) if center_ears else 0.28
            ear_mad = (
                float(np.median(np.abs(np.array(center_ears) - ear_med))) * 1.4826
                if len(center_ears) > 1
                else 0.015
            )
            zone_baselines[center_key] = ZoneBaseline(
                zone=center_key,
                ear_median=ear_med,
                ear_mad_scaled=max(ear_mad, 0.015),
                pitch_median_deg=0.0,
                yaw_median_deg=0.0,
                sample_count=len(center_ears),
                ear_left_median=ear_med,
                ear_right_median=ear_med,
                blendshape_open_median=0.05,
                confidence_median=0.85,
            )

        center_zb = zone_baselines[center_key]

        # Backfill any missing zones relative to the primary CENTER anchor
        if PoseZone.LEFT_MIRROR.value not in zone_baselines:
            zone_baselines[PoseZone.LEFT_MIRROR.value] = ZoneBaseline(
                zone=PoseZone.LEFT_MIRROR.value,
                ear_median=round(center_zb.ear_median * 0.92, 4),
                ear_mad_scaled=max(center_zb.ear_mad_scaled, 0.015),
                pitch_median_deg=center_zb.pitch_median_deg,
                yaw_median_deg=center_zb.yaw_median_deg + 20.0,
                sample_count=0,
                ear_left_median=round(center_zb.ear_median * 0.95, 4),
                ear_right_median=round(center_zb.ear_median * 0.89, 4),
                blendshape_open_median=0.06,
                confidence_median=0.75,
            )
        if PoseZone.RIGHT_MIRROR.value not in zone_baselines:
            zone_baselines[PoseZone.RIGHT_MIRROR.value] = ZoneBaseline(
                zone=PoseZone.RIGHT_MIRROR.value,
                ear_median=round(center_zb.ear_median * 0.92, 4),
                ear_mad_scaled=max(center_zb.ear_mad_scaled, 0.015),
                pitch_median_deg=center_zb.pitch_median_deg,
                yaw_median_deg=center_zb.yaw_median_deg - 20.0,
                sample_count=0,
                ear_left_median=round(center_zb.ear_median * 0.89, 4),
                ear_right_median=round(center_zb.ear_median * 0.95, 4),
                blendshape_open_median=0.06,
                confidence_median=0.75,
            )
        if PoseZone.DASHBOARD.value not in zone_baselines:
            zone_baselines[PoseZone.DASHBOARD.value] = ZoneBaseline(
                zone=PoseZone.DASHBOARD.value,
                ear_median=round(center_zb.ear_median * 0.88, 4),
                ear_mad_scaled=max(center_zb.ear_mad_scaled, 0.015),
                pitch_median_deg=center_zb.pitch_median_deg - 10.0,
                yaw_median_deg=center_zb.yaw_median_deg,
                sample_count=0,
                ear_left_median=round(center_zb.ear_median * 0.88, 4),
                ear_right_median=round(center_zb.ear_median * 0.88, 4),
                blendshape_open_median=0.08,
                confidence_median=0.78,
            )

        mean_rel = float(np.mean(self._reliabilities)) if self._reliabilities else 0.75
        cal_mode = (
            "FOUR_ZONE_PLUS_BLINK"
            if (self.four_zone_mode and self.include_blink_baseline)
            else ("FOUR_ZONE" if self.four_zone_mode else ("THREE_ZONE" if self.three_zone_mode else "CENTER_ONLY"))
        )

        # Stage 2.1 SO(3) Neutral Head Pose Rotation Matrix Aggregation
        neutral_rot_mat: Optional[np.ndarray] = None
        if self._candidate_rot_mats:
            try:
                rot_res = aggregate_neutral_rotation_matrix(self._candidate_rot_mats)
                if rot_res.is_accepted and rot_res.neutral_rot_mat is not None:
                    neutral_rot_mat = rot_res.neutral_rot_mat
                else:
                    self.quality_warnings.append(
                        f"ROTATION_REJECTED: Neutral head pose rotation aggregation rejected: {rot_res.rejection_reason}"
                    )
            except Exception as e:
                self.quality_warnings.append(
                    f"ROTATION_ERROR: Neutral head pose rotation aggregation error: {e}"
                )

        profile = DriverBehaviourProfile(
            driver_id=self.driver_id,
            calibration_mode=cal_mode,
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
            baseline_perclos_pct=base_perclos_pct,
            blink_amplitude_median=amp_med,
            temporal_blinks_sampled=len(self._collected_blinks),
            temporal_window_sec=round(
                self._blink_phase_elapsed_sec if self.include_blink_baseline else sum(s.duration_sec for s in self.steps),
                1,
            ),
            version=1,
            calibrated_at=datetime.now(timezone.utc).isoformat(),
            neutral_rot_mat=neutral_rot_mat,
        )
        self.is_completed = True
        return profile
