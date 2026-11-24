"""
Pydantic Data Schemas for Driver Behaviour Profiles (PDBP).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from backend.app.core.time_utils import validate_iso_timestamp


class ZoneBaselineDTO(BaseModel):
    """Calibrated physiological baseline parameters for a single spatial zone."""

    model_config = ConfigDict(populate_by_name=True)

    zone: str = Field(description="Spatial zone name (e.g. CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD)")
    ear_median: float = Field(description="Median open-eye EAR observed in this zone", ge=0.0, le=1.0)
    ear_mad_scaled: float = Field(description="Robust scaled Median Absolute Deviation of EAR", ge=0.0)
    dynamic_closure_threshold: float = Field(
        description="Learned closure threshold (median - 2.3 * MAD, clipped to [0.12, 0.22])",
        ge=0.0,
        le=1.0,
    )
    pitch_median_deg: float = Field(default=0.0, description="Median head pitch in degrees")
    yaw_median_deg: float = Field(default=0.0, description="Median head yaw in degrees")
    sample_count: int = Field(description="Number of valid calibration frames collected in this zone", ge=0)
    ear_left_median: Optional[float] = Field(default=None, description="Median left eye EAR in this zone")
    ear_right_median: Optional[float] = Field(default=None, description="Median right eye EAR in this zone")
    blendshape_open_median: Optional[float] = Field(default=None, description="Median blink blendshape score")
    confidence_median: Optional[float] = Field(default=None, description="Median landmark confidence")


class DriverProfileDTO(BaseModel):
    """Personalised Driver Behaviour Profile (PDBP) transferred between AI engine and API/Database."""

    model_config = ConfigDict(populate_by_name=True)

    version: int = Field(default=1, description="Calibration format algorithm version")
    engine_type: str = Field(default="PYTHON", description="Engine ownership tag: PYTHON or BROWSER")
    profile_version: int = Field(default=1, description="Server-authoritative monotonic profile version", ge=1)
    profile_uuid: Optional[str] = Field(default=None, description="Unique profile revision UUID")
    content_hash: Optional[str] = Field(default=None, description="Canonical server-computed SHA-256 profile hash")
    driver_id: str = Field(default="driver_default", description="Unique driver identifier")
    calibrated_at: Optional[str] = Field(
        default=None, description="ISO timestamp of calibration completion"
    )
    is_calibrated: bool = Field(default=False, description="Whether this profile contains learned driver baselines")
    calibration_mode: Optional[str] = Field(
        default=None,
        description="Calibration scope: 4ZONE, FOUR_ZONE, THREE_ZONE, CENTER_ONLY, or UNCALIBRATED_DEFAULT",
    )

    # Zone Baselines
    zone_baselines: Dict[str, ZoneBaselineDTO] = Field(default_factory=dict, description="Baselines keyed by zone name")
    center_ear_median: float = Field(default=0.28, description="Center reference open EAR", ge=0.0, le=1.0)

    # Temporal Kinematics Baselines
    blink_duration_median_ms: float = Field(default=235.0, description="Median alert blink duration in ms", ge=0.0)
    blink_duration_mad_ms: float = Field(default=40.0, description="Robust MAD of blink duration in ms", ge=0.0)
    blink_rcr_median: float = Field(default=0.95, description="Median Reopening-to-Closing velocity ratio", ge=0.0)
    blink_rcr_mad: float = Field(default=0.18, description="Robust MAD of RCR velocity ratio", ge=0.0)
    blink_freq_median_per_min: float = Field(default=16.0, description="Personal natural blink rate in blinks/min", ge=0.0)
    blink_freq_mad_per_min: float = Field(default=4.5, description="Robust MAD of personal blink rate in blinks/min", ge=0.0)
    baseline_perclos_pct: float = Field(default=3.5, description="Personal alert resting PERCLOS percentage", ge=0.0, le=100.0)
    blink_amplitude_median: float = Field(default=0.14, description="Personal median blink EAR excursion amplitude", ge=0.0, le=1.0)
    temporal_blinks_sampled: int = Field(default=0, description="Number of natural blinks sampled during 2-min cadence stream", ge=0)
    temporal_window_sec: float = Field(default=120.0, description="Duration of temporal blink baseline window in seconds", ge=0.0)
    sensitivity_level: str = Field(default="STANDARD", description="Fatigue cascade sensitivity: STANDARD, HIGH, or RELAXED")

    # Neutral Head Pose
    neutral_pitch_deg: float = Field(default=0.0, description="Learned neutral head pitch in degrees")
    neutral_yaw_deg: float = Field(default=0.0, description="Learned neutral head yaw in degrees")
    neutral_rot_mat: Optional[List[List[float]]] = Field(
        default=None,
        description="Learned 3x3 orthonormal neutral rotation matrix in camera coordinate frame",
    )
    calibration_is_mirrored: Optional[bool] = Field(
        default=None,
        description="Camera horizontal mirror state (is_mirrored) active when neutral_rot_mat was calibrated",
    )

    # Quality & Reliability
    calibration_reliability: float = Field(default=1.0, description="Overall calibration quality reliability score in [0, 1]")

    @field_validator("neutral_rot_mat", mode="before")
    @classmethod
    def validate_neutral_rot_mat(cls, v: Any) -> Optional[List[List[float]]]:
        if v is None:
            return None

        # 1. Structural check: exactly 3 rows
        if not isinstance(v, list) or len(v) != 3:
            raise ValueError("neutral_rot_mat must be a 3x3 matrix (list of 3 rows)")

        for row_idx, row in enumerate(v):
            if not isinstance(row, list) or len(row) != 3:
                raise ValueError(
                    f"Row {row_idx} of neutral_rot_mat must be a list containing exactly 3 numeric values"
                )
            # 2. Strict boolean rejection: reject booleans explicitly (isinstance(True, int) is True in Python)
            for col_idx, item in enumerate(row):
                if type(item) is bool:
                    raise ValueError(
                        f"neutral_rot_mat[{row_idx}][{col_idx}] is boolean; boolean values are strictly rejected"
                    )

        # 3. Numeric & Finite check
        try:
            arr = np.asarray(v, dtype=np.float64)
        except (ValueError, TypeError) as e:
            raise ValueError(f"neutral_rot_mat contains non-numeric data: {e}")

        if not np.all(np.isfinite(arr)):
            raise ValueError("neutral_rot_mat elements must be finite numbers (no NaN or Inf)")

        # 4. Orthogonality check with tolerance 1e-5
        ident = np.eye(3, dtype=np.float64)
        if not np.allclose(arr.T @ arr, ident, atol=1e-5):
            raise ValueError("neutral_rot_mat is not orthogonal (R^T @ R != I within tolerance 1e-5)")

        # 5. Determinant & reflection check with tolerance 1e-5
        det = float(np.linalg.det(arr))
        if abs(det - 1.0) > 1e-5:
            raise ValueError(
                f"neutral_rot_mat is not a valid SO(3) rotation (det(R)={det:.6f} != 1.0 within tolerance 1e-5; reflections rejected)"
            )

        # Do not round values inside validator
        return [[float(x) for x in row] for row in v]

    @field_validator("calibrated_at")
    @classmethod
    def validate_timestamp(cls, v: Any) -> Optional[str]:
        return validate_iso_timestamp(v)

    @model_validator(mode="after")
    def resolve_calibration_mode(self) -> "DriverProfileDTO":
        fields_set = self.model_fields_set
        has_mode = "calibration_mode" in fields_set
        has_cal = "is_calibrated" in fields_set

        # 1. Reject explicit null
        if has_mode and self.calibration_mode is None:
            raise ValueError(
                "calibration_mode cannot be null; must be omitted or a valid string mode"
            )

        # 2. Both fields omitted -> clean uncalibrated default
        if not has_mode and not has_cal:
            self.is_calibrated = False
            self.calibration_mode = "UNCALIBRATED_DEFAULT"
            return self

        # 3. calibration_mode omitted from payload -> legacy migration
        if not has_mode:
            if self.is_calibrated:
                # Preserves exact historical default "4ZONE"
                self.calibration_mode = "4ZONE"
            else:
                self.calibration_mode = "UNCALIBRATED_DEFAULT"
            return self

        # 4. is_calibrated omitted from payload -> infer from explicit mode
        if not has_cal:
            self.is_calibrated = (self.calibration_mode != "UNCALIBRATED_DEFAULT")
            return self

        # 5. Both explicitly provided -> enforce strict consistency (no silent rewrites)
        if self.is_calibrated and self.calibration_mode == "UNCALIBRATED_DEFAULT":
            raise ValueError(
                "Explicit contradiction in profile: is_calibrated=True cannot be paired with calibration_mode='UNCALIBRATED_DEFAULT'"
            )
        if not self.is_calibrated and self.calibration_mode != "UNCALIBRATED_DEFAULT":
            raise ValueError(
                f"Explicit contradiction in profile: is_calibrated=False cannot be paired with active calibration_mode='{self.calibration_mode}'"
            )

        return self
