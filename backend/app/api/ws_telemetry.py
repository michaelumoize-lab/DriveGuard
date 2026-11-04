"""
Binary WebSocket Telemetry Endpoint for Real-Time Client Video Streaming.
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.services.profile_store import profile_store
from backend.app.services.session_manager import session_manager

ws_router = APIRouter(tags=["WebSocket Telemetry"])


@ws_router.websocket("/ws/v1/session/{session_id}")
async def websocket_session_stream(
    websocket: WebSocket,
    session_id: str,
    driver_id: str = Query(default="driver_default"),
    experiment_mode: Optional[str] = Query(default=None),
) -> None:
    """
    Bi-directional streaming WebSocket endpoint for DriveGuard edge monitoring.

    Protocol:
      - Client sends raw binary JPEG frames (e.g. from canvas.toBlob) -> Server processes
        and streams back JSON `FrameTelemetryDTO`.
      - Client can send JSON control messages:
          * `{"action": "START_CALIBRATION", "four_zone": true, "zone_duration_sec": 5.0}`
          * `{"action": "CANCEL_CALIBRATION"}`
          * `{"action": "SWITCH_EXPERIMENT", "mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"}`
          * `{"action": "SAVE_PROFILE"}`
          * `{"action": "PING"}`
    """
    await websocket.accept()

    try:
        engine = await session_manager.get_or_create_session(
            session_id=session_id,
            driver_id=driver_id,
            experiment_mode=experiment_mode,
        )
        # Send immediate ready handshake so client knows session is active
        await websocket.send_json(
            {
                "type": "SESSION_READY",
                "session_id": session_id,
                "driver_id": driver_id,
                "experiment_mode": engine.experiment_mode.value,
            }
        )
    except Exception as exc:
        import traceback
        tb = traceback.format_exc()
        print(f"[ERROR] Failed to instantiate AI session '{session_id}': {exc}\n{tb}")
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
            message = await websocket.receive()

            # Handle Binary Frame Stream (JPEG / PNG from webcam canvas)
            if "bytes" in message and message["bytes"]:
                frame_bytes = message["bytes"]
                # Run synchronous CPU-bound CV pipeline in thread pool to prevent blocking asyncio loop
                telemetry_dto = await asyncio.to_thread(engine.process_frame, frame_bytes)

                # Periodic terminal log every 15 frames (~1 sec)
                if telemetry_dto.frame_index % 15 == 0 or telemetry_dto.frame_index == 1:
                    print(
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
                        for em in ExperimentMode:
                            if em.value == target_mode or em.name == target_mode:
                                engine.experiment_mode = em
                                engine.cascade.mode = em
                                break
                        await websocket.send_json(
                            {
                                "type": "EXPERIMENT_SWITCHED",
                                "active_mode": engine.experiment_mode.value,
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

                except json.JSONDecodeError:
                    await websocket.send_json(
                        {"type": "ERROR", "message": "Invalid JSON text message"}
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

