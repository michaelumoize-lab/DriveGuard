"""
DriveGuard — Loopback-Only Local Python Camera & Telemetry API Routes (Phase 4A).

All HTTP, MJPEG, and WebSocket routes in this module are strictly guarded by
`require_loopback_http` / `require_loopback_ws` (returning 403 Forbidden for any
non-loopback peer, reverse-proxy header, external Origin/Host, or when
`LOCAL_CAMERA_API_ENABLED=False`).

Requires ZERO Next.js authentication, LiveKit credentials, or Neon connectivity.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncGenerator, Dict, Optional, Union

from fastapi import APIRouter, Depends, HTTPException, Query, WebSocket, WebSocketDisconnect, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.app.core.local_guard import require_loopback_http, require_loopback_ws
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.local_camera_controller import (
    LocalCameraController,
    local_camera_controller,
)

logger = logging.getLogger("driveguard.api.local_monitor")


def get_local_camera_controller() -> LocalCameraController:
    return local_camera_controller


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------

class LocalStartRequest(BaseModel):
    driver_id: str = Field(default="driver_default", min_length=1, max_length=128)
    source: Union[int, str] = Field(default=0, description="Local OpenCV camera index or phone stream URL")
    width: int = Field(default=640, ge=160, le=1920)
    height: int = Field(default=480, ge=120, le=1080)
    fps: int = Field(default=20, ge=5, le=30)
    backend: str = Field(default="DSHOW")
    is_mirrored: bool = Field(default=True)
    enable_audio: bool = Field(default=False)
    start_calibrating: bool = Field(default=False)


class LocalCalibrationStartRequest(BaseModel):
    four_zone_mode: bool = Field(default=True)
    step_duration_sec: float = Field(default=4.0, ge=1.0, le=20.0)


class LocalCalibrationRecalibrateRequest(BaseModel):
    zone: str = Field(description="CENTER, LEFT_MIRROR, RIGHT_MIRROR, or DASHBOARD")
    step_duration_sec: float = Field(default=4.0, ge=1.0, le=20.0)


class LocalProfileReconcileRequest(BaseModel):
    driver_id: str = Field(default="driver_default", min_length=1, max_length=128)
    remote_profile: DriverProfileDTO


# ---------------------------------------------------------------------------
# Loopback-Guarded REST + MJPEG Router (mounted under /api/v1/local)
# ---------------------------------------------------------------------------

router = APIRouter(
    prefix="/local",
    tags=["Local Python AI (Loopback Only)"],
    dependencies=[Depends(require_loopback_http)],
)


@router.get("/status")
async def get_local_status(
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    """Return current local Python camera state, operational state, calibration status, and active session."""
    return controller.get_status()


@router.get("/devices")
async def get_local_devices(
    max_probe: int = Query(default=3, ge=1, le=6),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    """Enumerate available local OpenCV capture indices and confirm phone URL support."""
    return await asyncio.to_thread(controller.list_devices, max_probe)


@router.post("/start")
async def start_local_monitoring(
    payload: LocalStartRequest = LocalStartRequest(),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    """Acquire local camera hardware and begin standalone Python AI monitoring + SQLite recording."""
    try:
        return await asyncio.to_thread(
            controller.start,
            driver_id=payload.driver_id,
            source=payload.source,
            width=payload.width,
            height=payload.height,
            fps=payload.fps,
            backend=payload.backend,
            is_mirrored=payload.is_mirrored,
            enable_audio=payload.enable_audio,
            start_calibrating=payload.start_calibrating,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc


@router.post("/stop")
async def stop_local_monitoring(
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    """
    Stop local monitoring and guarantee synchronous hardware camera release (`cap.release()`)
    and SQLite session finalization BEFORE returning HTTP 200.
    """
    return await asyncio.to_thread(controller.stop)


@router.post("/calibration/start")
async def start_local_calibration(
    payload: LocalCalibrationStartRequest = LocalCalibrationStartRequest(),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(
            controller.start_calibration,
            four_zone_mode=payload.four_zone_mode,
            step_duration_sec=payload.step_duration_sec,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/calibration/skip")
async def skip_local_calibration(
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(controller.skip_calibration_step)
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc


@router.post("/calibration/cancel")
async def cancel_local_calibration(
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    return await asyncio.to_thread(controller.cancel_calibration)


@router.post("/calibration/recalibrate")
async def recalibrate_local_zone(
    payload: LocalCalibrationRecalibrateRequest,
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(
            controller.recalibrate_zone,
            payload.zone,
            payload.step_duration_sec,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/profile")
async def get_local_profile(
    driver_id: str = Query(default="driver_default"),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    profile = await asyncio.to_thread(controller.get_profile, driver_id)
    return {
        "driver_id": driver_id,
        "engine_type": "PYTHON",
        "found": profile is not None,
        "profile": profile,
    }


@router.post("/profile/reconcile")
async def reconcile_local_profile(
    payload: LocalProfileReconcileRequest,
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    try:
        return await asyncio.to_thread(
            controller.reconcile_profile,
            payload.driver_id,
            payload.remote_profile,
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc


@router.get("/sessions")
async def list_local_sessions(
    driver_id: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=500),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    sessions = await asyncio.to_thread(controller.list_sessions, driver_id, limit)
    return {"sessions": sessions, "count": len(sessions)}


@router.get("/sessions/{session_id}/alerts")
async def list_local_session_alerts(
    session_id: str,
    limit: int = Query(default=200, ge=1, le=1000),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> Dict[str, Any]:
    alerts = await asyncio.to_thread(controller.list_session_alerts, session_id, limit)
    return {"session_id": session_id, "alerts": alerts, "count": len(alerts)}


@router.get("/preview.mjpg")
async def stream_local_mjpeg_preview(
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> StreamingResponse:
    """
    Loopback-only MJPEG live preview stream (`multipart/x-mixed-replace; boundary=frame`).
    JPEG encoding is activated on-demand only while at least one preview client is attached.
    """
    if not controller.get_status().get("running"):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Local Python camera is not currently running.",
        )

    async def mjpeg_generator() -> AsyncGenerator[bytes, None]:
        controller.register_mjpeg_viewer()
        last_seq = -1
        try:
            while True:
                seq, jpeg_bytes, is_running = await asyncio.to_thread(
                    controller.wait_for_next_jpeg, last_seq, 0.5
                )
                if not is_running:
                    break
                if jpeg_bytes is not None and seq != last_seq:
                    last_seq = seq
                    header = (
                        b"--frame\r\n"
                        b"Content-Type: image/jpeg\r\n"
                        + f"Content-Length: {len(jpeg_bytes)}\r\n\r\n".encode("ascii")
                    )
                    yield header + jpeg_bytes + b"\r\n"
        except asyncio.CancelledError:
            pass
        finally:
            controller.unregister_mjpeg_viewer()

    return StreamingResponse(
        mjpeg_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            "Pragma": "no-cache",
        },
    )


# ---------------------------------------------------------------------------
# Loopback-Guarded WebSocket Telemetry Router (/ws/local/telemetry)
# ---------------------------------------------------------------------------

local_ws_router = APIRouter(tags=["Local Python AI WebSocket (Loopback Only)"])


@local_ws_router.websocket("/ws/local/telemetry")
async def local_telemetry_websocket(
    websocket: WebSocket,
    _: None = Depends(require_loopback_ws),
    controller: LocalCameraController = Depends(get_local_camera_controller),
) -> None:
    """
    Loopback-only WebSocket streaming normalized Python AI telemetry, calibration updates,
    and authoritative SAFETY_EVENT alerts.
    """
    await websocket.accept()
    loop = asyncio.get_running_loop()
    queue = controller.subscribe_ws(loop)

    try:
        # Send immediate initial status snapshot
        await websocket.send_json({"type": "STATUS", **controller.get_status()})

        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=15.0)
                await websocket.send_json(msg)
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "HEARTBEAT", "running": controller.get_status()["running"]})
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("Local telemetry WebSocket closed: %s", exc)
    finally:
        controller.unsubscribe_ws(queue)
