"""
REST Endpoints for Driving Calibration Control.
Secured with cryptographic backend token authentication and resource ownership checks.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.auth import AuthenticatedUser, get_current_user, verify_resource_ownership
from backend.app.schemas.calibration import CalibrationPhaseDTO
from backend.app.services.session_manager import session_manager

router = APIRouter(prefix="/calibration", tags=["Calibration"])


class StartCalibrationPayload(BaseModel):
    session_id: str = Field(description="Active session ID to calibrate")
    four_zone_mode: bool = Field(default=True, description="True for 4-zone, False for 1-zone")
    zone_duration_sec: float = Field(default=5.0, description="Duration per zone in seconds", ge=1.0, le=30.0)


@router.post("/start", response_model=CalibrationPhaseDTO)
def start_calibration(
    payload: StartCalibrationPayload,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CalibrationPhaseDTO:
    """Initiate a guided calibration workflow for an authorized monitoring session."""
    engine = session_manager.get_session(payload.session_id)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{payload.session_id}' not found.",
        )
    owner = getattr(engine, "owner_id", engine.driver_id)
    verify_resource_ownership(current_user, owner, "monitoring session")

    return engine.start_calibration(
        four_zone_mode=payload.four_zone_mode,
        zone_duration_sec=payload.zone_duration_sec,
    )


@router.post("/cancel/{session_id}", response_model=dict)
def cancel_calibration(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Cancel an ongoing calibration session owned by the authenticated user."""
    engine = session_manager.get_session(session_id)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    owner = getattr(engine, "owner_id", engine.driver_id)
    verify_resource_ownership(current_user, owner, "monitoring session")

    engine.cancel_calibration()
    return {"status": "CALIBRATION_CANCELLED", "session_id": session_id}


@router.get("/{session_id}/status", response_model=CalibrationPhaseDTO)
def get_calibration_status(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> CalibrationPhaseDTO:
    """Fetch real-time progress and visual cues for an authorized active calibration session."""
    engine = session_manager.get_session(session_id)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    owner = getattr(engine, "owner_id", engine.driver_id)
    verify_resource_ownership(current_user, owner, "monitoring session")

    status_phase = engine.get_calibration_status()
    if status_phase is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Session '{session_id}' is not currently calibrating.",
        )
    return status_phase
