"""
LiveKit Cloud WebRTC Token Minting and Session Worker Initiation Endpoint.
"""

from __future__ import annotations

import logging
from typing import Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

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
    driver_id: str = Query(default="driver_default", description="Driver identity"),
    participant_name: Optional[str] = Query(default=None),
    experiment_mode: Optional[str] = Query(default=None),
    auto_start_worker: bool = Query(default=True, description="Automatically start Python AI worker"),
) -> LivekitTokenResponse:
    """
    Mint a cryptographically signed LiveKit AccessToken JWT for a client to join a room,
    and automatically ensure the DriveGuard Python AI Worker is active in that room.
    """
    if not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET or not settings.LIVEKIT_URL:
        raise HTTPException(
            status_code=500,
            detail="LiveKit credentials (LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET) are not properly configured.",
        )

    try:
        import uuid
        from livekit import api

        unique_suffix = uuid.uuid4().hex[:6]
        participant_identity = (
            f"driver_{driver_id}_{unique_suffix}"
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
            .with_name(participant_name or driver_id)
            .with_grants(grant)
            .to_jwt()
        )

        # Ensure Python AI worker is connected to this room to process video
        if auto_start_worker:
            await livekit_worker_manager.ensure_worker_running(
                room_name=room_name,
                driver_id=driver_id,
                experiment_mode=experiment_mode,
            )

        return LivekitTokenResponse(
            token=token,
            url=settings.LIVEKIT_URL,
            room_name=room_name,
            participant_identity=participant_identity,
        )
    except Exception as exc:
        logger.error(f"Failed to generate LiveKit token: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Token minting failed: {exc}")


@router.post("/stop-worker")
async def stop_worker(room_name: str = Query(...)) -> dict:
    """Manually terminate the Python AI worker for a given room."""
    stopped = await livekit_worker_manager.stop_worker(room_name)
    return {"status": "SUCCESS", "room_name": room_name, "worker_stopped": stopped}
