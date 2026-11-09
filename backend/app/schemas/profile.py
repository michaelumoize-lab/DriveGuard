"""
Pydantic Data Schemas for Driver Behaviour Profiles (PDBP).
"""

from __future__ import annotations

from typing import Dict, Optional
from pydantic import BaseModel, ConfigDict, Field


class ZoneBaselineDTO(BaseModel):
    """Calibrated physiological baseline parameters for a single spatial zone."""

    model_config = ConfigDict(populate_by_name=True)

    zone: str = Field(description="Spatial zone name (e.g. CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD)")
    ear_median: float = Field(description="Median open-eye EAR observed in this zone", ge=0.0, le=1.0)
    ear_mad_scaled: float = Field(description="Robust scaled Median Absolute Deviation of EAR", ge=0.0)
    dynamic_closure_threshold: float = Field(description="Learned closure threshold (median - 2.5 * MAD)", ge=0.0, le=1.0)
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
    driver_id: str = Field(default="driver_default", description="Unique driver identifier")
    calibrated_at: str = Field(description="ISO timestamp of calibration completion")
    is_calibrated: bool = Field(default=False, description="Whether this profile contains learned driver baselines")
    calibration_mode: str = Field(default="4ZONE", description="Calibration scope: 4ZONE, 1ZONE, or UNCALIBRATED_DEFAULT")
    
    # Zone Baselines
    zone_baselines: Dict[str, ZoneBaselineDTO] = Field(default_factory=dict, description="Baselines keyed by zone name")
    center_ear_median: float = Field(default=0.28, description="Center reference open EAR", ge=0.0, le=1.0)
    
    # Temporal Kinematics Baselines
    blink_duration_median_ms: float = Field(default=210.0, description="Median alert blink duration in ms", ge=0.0)
    blink_duration_mad_ms: float = Field(default=28.0, description="Robust MAD of blink duration in ms", ge=0.0)
    blink_rcr_median: float = Field(default=2.10, description="Median Reopening-to-Closing velocity ratio", ge=0.0)
    blink_rcr_mad: float = Field(default=0.25, description="Robust MAD of RCR velocity ratio", ge=0.0)
    
    # Neutral Head Pose
    neutral_pitch_deg: float = Field(default=0.0, description="Learned neutral head pitch in degrees")
    neutral_yaw_deg: float = Field(default=0.0, description="Learned neutral head yaw in degrees")
    
    # Quality & Reliability
    calibration_reliability: float = Field(default=1.0, description="Overall calibration quality reliability score in [0, 1]")
