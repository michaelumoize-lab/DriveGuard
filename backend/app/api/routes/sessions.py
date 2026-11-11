"""
REST Endpoints for Monitoring Session Lifecycle.
Secured with cryptographic backend token authentication and resource ownership checks.
"""

import uuid
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.core.auth import AuthenticatedUser, get_current_user, verify_resource_ownership
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.session_manager import session_manager

router = APIRouter(prefix="/sessions", tags=["Sessions"])


class CreateSessionRequest(BaseModel):
    session_id: Optional[str] = Field(default=None, description="Unique session identifier (generated if omitted)")
    driver_id: Optional[str] = Field(default=None, description="Driver identifier (defaults to authenticated user)")
    experiment_mode: str = Field(
        default="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        description="A_GEOMETRIC_ONLY, B_CNN_ALWAYS, or C_CONFIDENCE_GATED_CASCADE",
    )
    profile: Optional[DriverProfileDTO] = Field(
        default=None,
        description="Optional pre-calibrated baseline profile transmitted directly from PostgreSQL",
    )


class SwitchExperimentRequest(BaseModel):
    experiment_mode: str = Field(
        ...,
        description="Target experiment mode: A_GEOMETRIC_ONLY, B_CNN_ALWAYS, or C_CONFIDENCE_GATED_CASCADE",
    )


@router.get("", response_model=List[Dict[str, Any]])
def list_sessions(
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> List[Dict[str, Any]]:
    """List all currently active in-memory monitoring sessions owned by the authenticated user."""
    all_sessions = session_manager.list_active_sessions()
    if current_user.is_admin:
        return all_sessions
    return [
        s for s in all_sessions
        if s.get("owner_id") == current_user.user_id or s.get("driver_id") == current_user.user_id
    ]


@router.post("", response_model=Dict[str, Any])
async def create_session(
    request: CreateSessionRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Initialize a new monitoring session for the authenticated driver."""
    # Enforce driver identity: non-admins cannot create sessions for other users
    target_driver_id = request.driver_id or current_user.user_id
    if not current_user.is_admin and target_driver_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Cannot create monitoring session for driver '{target_driver_id}'.",
        )

    sess_id = request.session_id or f"sess_{uuid.uuid4().hex[:10]}"
    try:
        engine = await session_manager.get_or_create_session(
            session_id=sess_id,
            driver_id=target_driver_id,
            experiment_mode=request.experiment_mode,
            profile_dto=request.profile,
            owner_id=current_user.user_id,
        )
    except PermissionError as exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))

    return {
        "session_id": engine.session_id,
        "driver_id": engine.driver_id,
        "owner_id": getattr(engine, "owner_id", engine.driver_id),
        "experiment_mode": engine.experiment_mode.value,
        "is_calibrated": engine.profile.calibration_mode != "UNCALIBRATED_DEFAULT",
        "calibration_mode": engine.profile.calibration_mode,
    }


@router.get("/{session_id}", response_model=Dict[str, Any])
def get_session_details(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Fetch diagnostic metrics and summary for an authorized active session."""
    engine = session_manager.get_session(session_id)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    owner = getattr(engine, "owner_id", engine.driver_id)
    verify_resource_ownership(current_user, owner, "monitoring session")
    return engine.get_session_summary()


@router.post("/{session_id}/experiment", response_model=Dict[str, Any])
async def switch_session_experiment_mode(
    session_id: str,
    request: SwitchExperimentRequest,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """
    Synchronously switch the active AI experiment mode for a running session at the next frame boundary.
    Restricted to the owner of the monitoring session.
    """
    engine = session_manager.get_session(session_id)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    owner = getattr(engine, "owner_id", engine.driver_id)
    verify_resource_ownership(current_user, owner, "monitoring session")

    try:
        new_mode = await engine.submit_command("SWITCH_EXPERIMENT", mode=request.experiment_mode)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        )

    return {
        "session_id": session_id,
        "experiment_mode": new_mode.value if hasattr(new_mode, "value") else str(new_mode),
        "status": "SUCCESS",
    }


@router.delete("/{session_id}", response_model=Dict[str, Any])
async def terminate_session(
    session_id: str,
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> Dict[str, Any]:
    """Terminate an active session owned by the authenticated user and return final summary metrics."""
    engine = session_manager.get_session(session_id)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    owner = getattr(engine, "owner_id", engine.driver_id)
    verify_resource_ownership(current_user, owner, "monitoring session")

    summary = await session_manager.terminate_session(session_id)
    if summary is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )
    return summary
