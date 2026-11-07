"""
Pydantic Data Schemas for Guided Calibration Sessions.
"""

from __future__ import annotations

from typing import List, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field

from backend.app.schemas.profile import DriverProfileDTO


class CalibrationPhaseDTO(BaseModel):
    """Real-time progress state for a single calibration phase."""

    model_config = ConfigDict(populate_by_name=True)

    current_zone: str = Field(description="Target zone: CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD")
    phase_index: int = Field(description="1-based index of the current phase (e.g. 1 of 4)", ge=1)
    total_phases: int = Field(default=4, description="Total number of guided calibration phases", ge=1)
    
    # Progress & Stability
    zone_progress_pct: float = Field(description="Percentage progress for current target zone (0 to 100%)", ge=0.0, le=100.0)
    total_progress_pct: float = Field(description="Overall calibration session progress (0 to 100%)", ge=0.0, le=100.0)
    stable_samples_count: int = Field(description="Number of stable, high-quality samples acquired in this zone", ge=0)
    target_samples_count: int = Field(description="Target number of samples required for convergence", ge=1)
    
    # Guidance & Quality
    instruction: str = Field(description="Human-readable driver instruction prompt")
    is_head_in_target_zone: bool = Field(description="True if driver's head is correctly oriented toward target zone")
    stability_feedback: str = Field(description="Real-time feedback: 'Holding Steady', 'Turn Further Left', etc.")
    zone_quality_score: float = Field(default=1.0, description="Real-time tracking quality in this zone")
    is_completed: bool = Field(default=False, description="Whether calibration has completed successfully")
    rejection_reason: Optional[str] = Field(default=None, description="Reason if calibration was rejected")
    quality_warnings: List[str] = Field(default_factory=list, description="Non-fatal warnings recorded during calibration")
    profile: Optional[DriverProfileDTO] = Field(default=None, description="Calibrated driver profile if completed")


class CalibrationStartRequest(BaseModel):
    """Request payload to initiate a new calibration session."""

    model_config = ConfigDict(populate_by_name=True)

    driver_id: str = Field(default="driver_default", description="Driver identifier")
    four_zone_mode: bool = Field(default=True, description="True for 4-zone calibration, False for quick 1-zone")
    zone_duration_sec: float = Field(default=5.0, description="Duration per zone in seconds", ge=1.0, le=30.0)
    recalibrate_zone: Optional[Literal["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"]] = Field(
        default=None,
        description="Specific zone to recalibrate individually (CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD)",
    )
