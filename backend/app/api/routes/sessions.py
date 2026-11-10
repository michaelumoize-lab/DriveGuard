"""
REST Endpoints for Monitoring Session Lifecycle.
"""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.session_manager import session_manager

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Unique session identifier (generated if omitted)")
    driver_id: str = Field(default="driver_default", description="Driver identifier")
    experiment_mode: str = Field(
        default="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        description="A_GEOMETRIC_ONLY, B_CNN_ALWAYS, or C_CONFIDENCE_GATED_CASCADE",
    )
    profile: Optional[DriverProfileDTO] = Field(
        default=None,
        description="Optional pre-calibrated baseline profile transmitted directly from PostgreSQL",
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_sessions() -> List[Dict[str, Any]]:
    """List all currently active in-memory monitoring sessions."""
    return session_manager.list_active_sessions()


@router.post("", response_model=Dict[str, Any])
async def create_session(request: CreateSessionRequest) -> Dict[str, Any]:
    """Initialize a new monitoring session for a driver."""
    sess_id = request.session_id or f"sess_{uuid.uuid4().hex[:10]}"
    engine = await session_manager.get_or_create_session(
        session_id=sess_id,
        driver_id=request.driver_id,
        experiment_mode=request.experiment_mode,
        profile_dto=request.profile,
    )
    return {
        "session_id": engine.session_id,
        "driver_id": engine.driver_id,
        "experiment_mode": engine.experiment_mode.value,
        "is_calibrated": engine.profile.calibration_mode != "UNCALIBRATED_DEFAULT",
        "calibration_mode": engine.profile.calibration_mode,
    }


@router.get("/{session_id}", response_model=Dict[str, Any])
def get_session_details(session_id: str) -> Dict[str, Any]:
    """Fetch diagnostic metrics and summary for an active session."""
    engine = session_manager.get_session(session_id)
    if engine is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return engine.get_session_summary()


@router.delete("/{session_id}", response_model=Dict[str, Any])
async def terminate_session(session_id: str) -> Dict[str, Any]:
    """Terminate an active session and return final summary metrics."""
    summary = await session_manager.terminate_session(session_id)
    if summary is None:
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found.")
    return summary
