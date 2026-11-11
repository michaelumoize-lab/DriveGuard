"""
LiveKit Cloud WebRTC Token Minting and Session Worker Initiation Endpoint.
Secured with cryptographic backend token authentication and resource ownership checks.
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel

from backend.app.core.auth import AuthenticatedUser, get_current_user, verify_resource_ownership
from backend.app.core.config import settings
from backend.app.services.livekit_worker import livekit_worker_manager

logger = logging.getLogger("driveguard.livekit")
router = APIRouter(prefix="/livekit", tags=["LiveKit WebRTC"])


class LivekitTokenResponse(BaseModel):
    token: str
    url: str
    room_name: str
    participant_identity: str


@router.get("/token", response_model=LivekitTokenResponse)
async def get_livekit_token(
    room_name: str = Query(..., description="Unique room / session identifier"),
    driver_id: Optional[str] = Query(default=None, description="Driver identity (defaults to authenticated user)"),
    participant_name: Optional[str] = Query(default=None),
    experiment_mode: Optional[str] = Query(default=None),
    auto_start_worker: bool = Query(default=True, description="Automatically start Python AI worker"),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> LivekitTokenResponse:
    """
    Mint a cryptographically signed LiveKit AccessToken JWT for an authorized client to join a room,
    and automatically ensure the DriveGuard Python AI Worker is active in that room.
    """
    if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET or not settings.LIVEKIT_URL:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="LiveKit credentials (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) are not properly configured.",
        )

    # Enforce driver identity: non-admins cannot impersonate another driver ID
    target_driver_id = driver_id or current_user.user_id
    if not current_user.is_admin and target_driver_id != current_user.user_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: Cannot request token for driver '{target_driver_id}'.",
        )

    try:
        import uuid
        from livekit import api

        unique_suffix = uuid.uuid4().hex[:6]
        participant_identity = (
            f"driver_{target_driver_id}_{unique_suffix}"
            if not participant_name
            else f"{participant_name}_{unique_suffix}"
        )

        # Mint driver participant token (Can publish camera track, can send/receive data)
        grant = api.VideoGrants(
            room_join=True,
            room=room_name,
            can_publish=True,
            can_subscribe=True,
            can_publish_data=True,
        )
        token = (
            api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
            .with_identity(participant_identity)
            .with_name(participant_name or target_driver_id)
            .with_grants(grant)
            .to_jwt()
        )

        # Ensure Python AI worker is connected to this room to process video
        if auto_start_worker:
            await livekit_worker_manager.ensure_worker_running(
                room_name=room_name,
                driver_id=target_driver_id,
                experiment_mode=experiment_mode,
                owner_id=current_user.user_id,
            )

        return LivekitTokenResponse(
            token=token,
            url=settings.LIVEKIT_URL,
            room_name=room_name,
            participant_identity=participant_identity,
        )
    except PermissionError as p_exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(p_exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Failed to generate LiveKit token: {exc}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Token minting failed: {exc}",
        )


@router.post("/stop-worker")
async def stop_worker(
    room_name: str = Query(...),
    current_user: AuthenticatedUser = Depends(get_current_user),
) -> dict:
    """Manually terminate the Python AI worker for an authorized room."""
    try:
        stopped = await livekit_worker_manager.stop_worker(
            room_name=room_name,
            requested_by=current_user.user_id,
            is_admin=current_user.is_admin,
        )
    except PermissionError as p_exc:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(p_exc))

    return {"status": "SUCCESS", "room_name": room_name, "worker_stopped": stopped}
