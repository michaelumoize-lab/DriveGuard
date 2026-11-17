"""
Binary WebSocket Telemetry Endpoint for Real-Time Client Video Streaming.
Secured with cryptographic backend token authentication, driver ownership checks,
frame size limits, and idle timeouts.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.core.auth import verify_ws_token
from backend.app.core.config import settings
from backend.app.services.profile_store import profile_store
from backend.app.services.session_manager import session_manager

logger = logging.getLogger("driveguard.websocket")
ws_router = APIRouter(tags=["WebSocket Telemetry"])


@ws_router.websocket("/ws/v1/session/{session_id}")
async def websocket_session_stream(
    websocket: WebSocket,
    session_id: str,
    token: Optional[str] = Query(default=None, description="Cryptographically signed backend access token"),
    driver_id: Optional[str] = Query(default=None, description="Driver identifier"),
    experiment_mode: Optional[str] = Query(default=None),
) -> None:
    """
    Bi-directional streaming WebSocket endpoint for DriveGuard edge monitoring.

    Requires a valid, signed backend access token. Enforces driver ownership,
    maximum binary frame size, idle timeouts, and safe disconnect cleanup.
    """
    # 1. Authenticate before accepting frame ingestion
    try:
        user = verify_ws_token(token)
    except ValueError as auth_exc:
        await websocket.accept()
        await websocket.send_json({"type": "FATAL_ERROR", "detail": f"Unauthorized: {auth_exc}"})
        await websocket.close(code=1008, reason="Unauthorized")
        return

    # 2. Ownership enforcement: Non-admins cannot stream on behalf of other drivers
    effective_driver = driver_id or user.user_id
    if not user.is_admin and effective_driver != user.user_id:
        await websocket.accept()
        await websocket.send_json(
            {"type": "FATAL_ERROR", "detail": f"Forbidden: Cannot stream for driver '{effective_driver}'."}
        )
        await websocket.close(code=1008, reason="Forbidden")
        return

    await websocket.accept()

    try:
        engine = await session_manager.get_or_create_session(
            session_id=session_id,
            driver_id=effective_driver,
            experiment_mode=experiment_mode,
            owner_id=user.user_id,
        )
        # Send immediate ready handshake so client knows session is active
        await websocket.send_json(
            {
                "type": "SESSION_READY",
                "session_id": session_id,
                "driver_id": effective_driver,
                "experiment_mode": engine.experiment_mode.value,
            }
        )
    except PermissionError as perm_err:
        await websocket.send_json({"type": "FATAL_ERROR", "detail": str(perm_err)})
        await websocket.close(code=1008, reason="Forbidden")
        return
    except Exception as exc:
        logger.error(f"[WS] Failed to instantiate AI session '{session_id}': {exc}", exc_info=True)
        try:
            await websocket.send_json(
                {"type": "FATAL_ERROR", "detail": f"Failed to initialize AI session: {exc}"}
            )
            await websocket.close(code=1011)
        except Exception:
            pass
        return

    try:
        while True:
            # 3. Idle timeout enforcement
            try:
                message = await asyncio.wait_for(
                    websocket.receive(),
                    timeout=settings.WS_IDLE_TIMEOUT_SEC,
                )
            except asyncio.TimeoutError:
                logger.info(f"[WS] Session '{session_id}' reached idle timeout ({settings.WS_IDLE_TIMEOUT_SEC}s).")
                await websocket.send_json(
                    {"type": "DISCONNECT", "detail": f"Session idle timeout ({settings.WS_IDLE_TIMEOUT_SEC}s) reached."}
                )
                await websocket.close(code=1000, reason="Idle timeout")
                break

            # Handle Binary Frame Stream (JPEG / PNG from webcam canvas)
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]

                # 4. Maximum frame size validation
                if len(frame_bytes) > settings.MAX_WS_FRAME_SIZE_BYTES:
                    logger.warning(
                        f"[WS] Oversized frame ({len(frame_bytes)} bytes) rejected on session '{session_id}'."
                    )
                    await websocket.send_json(
                        {
                            "type": "ERROR",
                            "message": f"Frame size exceeds allowed limit ({settings.MAX_WS_FRAME_SIZE_BYTES} bytes).",
                        }
                    )
                    continue

                # Run synchronous CPU-bound CV pipeline in thread pool to prevent blocking asyncio loop
                telemetry_dto = await asyncio.to_thread(engine.process_frame, frame_bytes)

                # Periodic terminal log every 15 frames (~1 sec)
                if telemetry_dto.frame_index % 15 == 0 or telemetry_dto.frame_index == 1:
                    logger.debug(
                        f"[AI Telemetry] Frame #{telemetry_dto.frame_index}: "
                        f"State={telemetry_dto.state.value} | "
                        f"EAR={telemetry_dto.eye_metrics.ear_raw:.3f} | "
                        f"Zone={telemetry_dto.pose_zone.value} | "
                        f"Latency={telemetry_dto.inference_latency_ms:.1f}ms"
                    )

                cal_status = engine.get_calibration_status()
                response_payload = {
                    "type": "TELEMETRY",
                    "telemetry": telemetry_dto.model_dump(),
                    "calibration": (
                        cal_status.model_dump()
                        if (engine.is_calibrating or (cal_status is not None and not cal_status.is_completed))
                        else None
                    ),
                }
                await websocket.send_json(response_payload)

            # Handle JSON Command Stream
            elif "text" in message and message["text"]:
                try:
                    cmd_data = json.loads(message["text"])
                    if not isinstance(cmd_data, dict):
                        await websocket.send_json({"type": "ERROR", "message": "Expected JSON object."})
                        continue

                    # Accept either "action" or "command"
                    action = cmd_data.get("action", cmd_data.get("command", "")).upper()

                    if action in ("START_CALIBRATION", "START_4ZONE_CALIBRATION"):
                        four_zone = bool(cmd_data.get("four_zone", True))
                        duration_sec = float(cmd_data.get("zone_duration_sec", 5.0))
                        cal_dto = engine.start_calibration(
                            four_zone_mode=four_zone, zone_duration_sec=duration_sec
                        )
                        await websocket.send_json(
                            {
                                "type": "CALIBRATION_STARTED",
                                "calibration": cal_dto.model_dump(),
                                "phase": cal_dto.model_dump(),
                            }
                        )

                    elif action == "CANCEL_CALIBRATION":
                        engine.cancel_calibration()
                        await websocket.send_json({"type": "CALIBRATION_CANCELLED"})

                    elif action in ("SKIP_CALIBRATION_STEP", "SKIP_ZONE"):
                        cal_dto = engine.skip_calibration_step()
                        await websocket.send_json(
                            {
                                "type": "CALIBRATION_PHASE",
                                "calibration": cal_dto.model_dump() if cal_dto else None,
                                "phase": cal_dto.model_dump() if cal_dto else None,
                            }
                        )

                    elif action == "SWITCH_EXPERIMENT":
                        target_mode = cmd_data.get("mode", "")
                        new_mode = await engine.submit_command("SWITCH_EXPERIMENT", mode=target_mode)
                        await websocket.send_json(
                            {
                                "type": "EXPERIMENT_SWITCHED",
                                "active_mode": new_mode.value,
                            }
                        )

                    elif action == "SAVE_PROFILE":
                        dto = engine.export_profile_dto()
                        profile_store.save_profile(dto)
                        await websocket.send_json(
                            {"type": "PROFILE_SAVED", "profile": dto.model_dump()}
                        )

                    elif action == "PING":
                        await websocket.send_json(
                            {"type": "PONG", "session_id": session_id}
                        )
                    else:
                        await websocket.send_json(
                            {"type": "ERROR", "message": f"Unrecognized action: '{action}'."}
                        )

                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "ERROR", "message": "Invalid JSON text message"}
                    )
                except Exception as cmd_err:
                    await websocket.send_json(
                        {"type": "ERROR", "action": action if 'action' in locals() else "UNKNOWN", "message": str(cmd_err)}
                    )

    except WebSocketDisconnect:
        # Client closed connection — clean up session and persist profile if calibrated
        await session_manager.terminate_session(session_id)
    except Exception as exc:
        try:
            await websocket.send_json({"type": "FATAL_ERROR", "detail": str(exc)})
            await websocket.close()
        except Exception:
            pass
        await session_manager.terminate_session(session_id)
