"""
DriveGuard — Singleton Local Camera & Standalone Python Monitoring Controller (Phase 4A).

Wraps `LocalMonitoringService`, `BaseCamera` (`OpenCVCamera` / `PhoneCamera`),
`ProfileStore`, and `LocalDB` (SQLite WAL) so the Python AI engine runs 100%
autonomously with zero Next.js, Browser AI, LiveKit, or Neon dependency:
- Owns exclusive OpenCV/Phone camera start & synchronous release-before-return stop
- Runs authoritative `LocalMonitoringService.process_frame()` on a dedicated capture thread
- Persists every session start, idempotent `AlertStateManager` transition, and session end to SQLite
- Encodes downsampled MJPEG preview frames ONLY when >= 1 preview viewer is attached
- Broadcasts normalized telemetry, calibration status, and safety events to loopback WebSocket subscribers
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.ai.profile_adapter import dto_to_profile, profile_to_dto
from backend.app.alerts.alert_state import AlertState
from backend.app.camera.base import BaseCamera
from backend.app.camera.camera_manager import create_camera, discover_local_devices
from backend.app.intelligence.reason_codes import OperationalState
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.local_db import LocalDB, local_db
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringFrameResult,
    LocalMonitoringService,
)
from backend.app.services.profile_store import ProfileStore, profile_store

logger = logging.getLogger("driveguard.services.local_camera_controller")


CameraFactory = Callable[..., BaseCamera]
ServiceFactory = Callable[[LocalMonitoringConfig, ProfileStore], LocalMonitoringService]


class LocalCameraController:
    """
    Thread-safe hardware coordinator for standalone Python local monitoring.
    Guarantees exclusive camera access and synchronous `camera.close()` before `stop()` returns.
    """

    def __init__(
        self,
        db: Optional[LocalDB] = None,
        store: Optional[ProfileStore] = None,
        camera_factory: Optional[CameraFactory] = None,
        service_factory: Optional[ServiceFactory] = None,
    ) -> None:
        self._db: LocalDB = db or local_db
        self._store: ProfileStore = store or profile_store
        self._camera_factory: CameraFactory = camera_factory or create_camera
        self._service_factory: Optional[ServiceFactory] = service_factory

        self._lock = threading.RLock()
        self._camera: Optional[BaseCamera] = None
        self._service: Optional[LocalMonitoringService] = None
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

        # Runtime state
        self._is_running: bool = False
        self._driver_id: str = "driver_default"
        self._camera_source: Union[int, str] = 0
        self._target_fps: int = 20
        self._is_mirrored: bool = True
        self._active_session: Optional[Dict[str, Any]] = None
        self._session_start_mono: float = 0.0
        self._last_error: Optional[str] = None
        self._alert_seq: int = 0

        # Session accumulators
        self._stats = self._new_stats()

        # Latest snapshots
        self._latest_telemetry_payload: Optional[Dict[str, Any]] = None
        self._latest_calibration_payload: Optional[Dict[str, Any]] = None
        self._latest_operational_state: str = OperationalState.CALIBRATION_REQUIRED.value

        # On-demand MJPEG preview buffer
        self._mjpeg_cond = threading.Condition(self._lock)
        self._mjpeg_viewer_count: int = 0
        self._latest_jpeg: Optional[bytes] = None
        self._latest_jpeg_seq: int = 0

        # Loopback WebSocket subscribers: list of (asyncio.AbstractEventLoop, asyncio.Queue)
        self._ws_lock = threading.Lock()
        self._ws_subscribers: List[Tuple[asyncio.AbstractEventLoop, asyncio.Queue]] = []

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {
            "total_frames": 0,
            "normal_frames_count": 0,
            "fps_sum": 0.0,
            "latency_sum": 0.0,
            "perclos_sum": 0.0,
            "max_perclos": 0.0,
            "total_blinks": 0,
            "mirror_suppressions": 0,
            "last_frame_mono": 0.0,
        }

    # ------------------------------------------------------------------
    # Lifecycle: Start & Synchronous Stop (Camera Release Guarantee)
    # ------------------------------------------------------------------

    def start(
        self,
        *,
        driver_id: str = "driver_default",
        source: Union[int, str] = 0,
        width: int = 640,
        height: int = 480,
        fps: int = 20,
        backend: str = "DSHOW",
        is_mirrored: bool = True,
        enable_audio: bool = False,
        start_calibrating: bool = False,
    ) -> Dict[str, Any]:
        """
        Acquire the local camera exclusively, initialize LocalMonitoringService,
        create a SQLite session, and start the background inference loop.
        """
        with self._lock:
            if self._is_running:
                self.stop()

            self._last_error = None
            self._stop_event.clear()
            self._driver_id = driver_id or "driver_default"
            self._camera_source = source
            self._target_fps = max(5, min(30, int(fps)))
            self._is_mirrored = bool(is_mirrored)
            self._stats = self._new_stats()
            self._alert_seq = 0
            self._latest_telemetry_payload = None
            self._latest_calibration_payload = None
            self._latest_jpeg = None

            camera = self._camera_factory(
                source=source,
                target_width=int(width),
                target_height=int(height),
                target_fps=self._target_fps,
                backend=backend,
                flip_horizontal=self._is_mirrored,
            )

            opened = False
            try:
                opened = bool(camera.open())
            except Exception as exc:
                try:
                    camera.close()
                except Exception:
                    pass
                self._last_error = f"Camera open error: {exc}"
                raise RuntimeError(self._last_error) from exc

            if not opened:
                try:
                    camera.close()
                except Exception:
                    pass
                self._last_error = f"Failed to open local camera source '{source}'."
                raise RuntimeError(self._last_error)

            actual_mirrored = bool(getattr(camera, "is_flip_horizontal", self._is_mirrored))
            svc_cfg = LocalMonitoringConfig(
                driver_id=self._driver_id,
                is_mirrored=actual_mirrored,
                start_calibrating=bool(start_calibrating),
                calibration_step_duration_sec=4.0,
                cnn_shadow_enabled=False,
            )

            try:
                if self._service_factory is not None:
                    service = self._service_factory(svc_cfg, self._store)
                else:
                    service = LocalMonitoringService(
                        config=svc_cfg,
                        store=self._store,
                        enable_landmarker=True,
                        enable_shadow_worker=False,
                        enable_audio=bool(enable_audio),
                        enable_event_logging=False,
                        auto_load_profile=True,
                        persist_calibration_to_store=True,
                    )
            except Exception as exc:
                try:
                    camera.close()
                except Exception:
                    pass
                self._last_error = f"Monitoring service init error: {exc}"
                raise RuntimeError(self._last_error) from exc

            stored_dto = self._store.get_profile(self._driver_id)
            session_record = self._db.create_session(
                driver_id=self._driver_id,
                camera_source=str(source),
                engine_type="PYTHON",
                profile_uuid=stored_dto.profile_uuid if stored_dto else None,
                profile_version=stored_dto.profile_version if stored_dto else None,
            )

            self._camera = camera
            self._service = service
            self._active_session = session_record
            self._session_start_mono = time.monotonic()
            self._is_running = True
            self._latest_operational_state = (
                OperationalState.CALIBRATING.value
                if service.is_calibrating
                else (
                    OperationalState.MONITORING_ACTIVE.value
                    if service.calibrated_profile_id is not None
                    else OperationalState.CALIBRATION_REQUIRED.value
                )
            )

            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                name=f"driveguard-local-cam-{self._driver_id}",
                daemon=True,
            )
            self._capture_thread.start()

            logger.info(
                "Local Python monitoring started: session=%s driver=%s source=%s calibrated=%s",
                session_record["session_code"],
                self._driver_id,
                source,
                service.calibrated_profile_id is not None,
            )

            return self.get_status()

    def stop(self) -> Dict[str, Any]:
        """
        Stop the capture loop, join the worker thread, synchronously release hardware camera
        (`camera.close()`), and finalize the active SQLite session BEFORE returning.
        Safe to call repeatedly (idempotent).
        """
        thread_to_join: Optional[threading.Thread] = None
        with self._lock:
            self._stop_event.set()
            thread_to_join = self._capture_thread

        # Join outside lock so capture loop can exit cleanly if waiting on lock
        if thread_to_join is not None and thread_to_join.is_alive():
            if thread_to_join is not threading.current_thread():
                thread_to_join.join(timeout=5.0)
                if thread_to_join.is_alive():
                    logger.warning("Local camera capture thread did not terminate within 5.0s.")

        with self._lock:
            self._capture_thread = None

            if self._camera is not None:
                try:
                    self._camera.close()
                except Exception as exc:
                    logger.warning("Error closing local camera: %s", exc)
                self._camera = None

            if self._service is not None:
                try:
                    self._service.close()
                except Exception as exc:
                    logger.warning("Error closing LocalMonitoringService: %s", exc)
                self._service = None

            finalized_session: Optional[Dict[str, Any]] = None
            if self._active_session is not None:
                sid = self._active_session["session_id"]
                elapsed_sec = max(
                    1 if self._stats["total_frames"] > 0 else 0,
                    int(round(time.monotonic() - self._session_start_mono))
                    if self._session_start_mono > 0
                    else 0,
                )
                frames = max(1, self._stats["total_frames"])
                duration_min = max(elapsed_sec / 60.0, 1.0 / 60.0)
                avg_blink_rate = (
                    float(self._stats["total_blinks"]) / duration_min
                    if self._stats["total_frames"] > 0
                    else 0.0
                )
                finalized_session = self._db.end_session(
                    sid,
                    duration_seconds=elapsed_sec,
                    total_frames=self._stats["total_frames"],
                    normal_frames_count=self._stats["normal_frames_count"],
                    average_fps=self._stats["fps_sum"] / frames,
                    average_latency_ms=self._stats["latency_sum"] / frames,
                    average_perclos_pct=self._stats["perclos_sum"] / frames,
                    max_perclos_pct=self._stats["max_perclos"],
                    total_blinks=self._stats["total_blinks"],
                    average_blink_rate=avg_blink_rate,
                    mirror_suppressions=self._stats["mirror_suppressions"],
                )
                self._active_session = None

            self._is_running = False
            self._latest_jpeg = None
            self._mjpeg_cond.notify_all()

        self._broadcast_ws(
            {
                "type": "STATUS",
                "running": False,
                "camera_released": True,
                "session": finalized_session,
            },
            critical=True,
        )

        return {
            "stopped": True,
            "camera_released": True,
            "session": finalized_session,
        }

    # ------------------------------------------------------------------
    # Capture & Inference Loop
    # ------------------------------------------------------------------

    def _capture_loop(self) -> None:
        frame_period_sec = 1.0 / float(max(5, self._target_fps))
        consecutive_read_failures = 0

        while not self._stop_event.is_set():
            loop_start = time.monotonic()

            with self._lock:
                camera = self._camera
                service = self._service
                session = self._active_session

            if camera is None or service is None or session is None:
                break

            try:
                ok, packet = camera.read()
            except Exception as exc:
                logger.warning("Local camera read exception: %s", exc)
                ok, packet = False, None

            if not ok or packet is None or packet.frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 50:
                    with self._lock:
                        self._last_error = "Camera read timeout / stream lost."
                time.sleep(0.02)
                continue

            consecutive_read_failures = 0
            frame_bgr = packet.frame
            ts_ms = float(packet.timestamp_ms)

            try:
                with self._lock:
                    if self._service is None or self._stop_event.is_set():
                        break
                    result = self._service.process_frame(
                        frame_bgr=frame_bgr,
                        timestamp_ms=ts_ms,
                        frame_index=packet.frame_index,
                        packet=packet,
                    )
                    self._handle_frame_result_locked(result, frame_bgr, session)
            except Exception as exc:
                logger.error("Local monitoring frame processing error: %s", exc, exc_info=True)
                with self._lock:
                    self._last_error = str(exc)

            elapsed = time.monotonic() - loop_start
            sleep_sec = frame_period_sec - elapsed
            if sleep_sec > 0.001:
                self._stop_event.wait(timeout=sleep_sec)

    def _handle_frame_result_locked(
        self,
        result: LocalMonitoringFrameResult,
        frame_bgr: np.ndarray,
        session: Dict[str, Any],
    ) -> None:
        now_mono = time.monotonic()
        dt = now_mono - self._stats["last_frame_mono"] if self._stats["last_frame_mono"] > 0 else 0.0
        inst_fps = (1.0 / dt) if dt > 0.001 else float(self._target_fps)
        self._stats["last_frame_mono"] = now_mono

        snap = result.telemetry
        self._stats["total_frames"] += 1
        if snap.alert_state == AlertState.NORMAL.value and snap.operational_state == OperationalState.MONITORING_ACTIVE.value:
            self._stats["normal_frames_count"] += 1
        self._stats["fps_sum"] += min(60.0, inst_fps)
        self._stats["latency_sum"] += float(snap.inference_latency_ms)
        self._stats["perclos_sum"] += float(snap.perclos_pct)
        if snap.perclos_pct > self._stats["max_perclos"]:
            self._stats["max_perclos"] = float(snap.perclos_pct)
        self._stats["total_blinks"] = max(self._stats["total_blinks"], int(snap.total_blinks))
        if result.assessment.decision.is_mirror_check_context:
            self._stats["mirror_suppressions"] += 1

        self._latest_operational_state = snap.operational_state

        # Build calibration payload if active or just completed/rejected
        cal_payload: Optional[Dict[str, Any]] = None
        if result.calibration_progress is not None:
            idx, total, pct, zone_name, aligned, feedback = result.calibration_progress
            cal_payload = {
                "current_zone": zone_name,
                "phase_index": idx,
                "total_phases": total,
                "zone_progress_pct": round(float(pct), 1),
                "total_progress_pct": round(((idx - 1) + (float(pct) / 100.0)) / max(1, total) * 100.0, 1),
                "instruction": feedback,
                "is_completed": False,
                "rejection_reason": None,
                "is_head_in_target_zone": bool(aligned),
                "stability_feedback": feedback,
            }
        elif result.completed_calibration_profile is not None or result.calibration_rejection_reason is not None:
            dto_dict = (
                profile_to_dto(result.completed_calibration_profile).model_dump()
                if result.completed_calibration_profile is not None
                else None
            )
            cal_payload = {
                "current_zone": "CENTER",
                "phase_index": 4,
                "total_phases": 4,
                "zone_progress_pct": 100.0 if dto_dict else 0.0,
                "total_progress_pct": 100.0 if dto_dict else 0.0,
                "instruction": result.calibration_status_message or "Calibration completed",
                "is_completed": True,
                "rejection_reason": result.calibration_rejection_reason,
                "profile": dto_dict,
            }

        if cal_payload is not None:
            self._latest_calibration_payload = cal_payload

        # Build normalized dashboard-compatible telemetry payload
        reasons = [r.code.value for r in result.assessment.decision.reasons]
        telemetry_payload: Dict[str, Any] = {
            "type": "TELEMETRY",
            "engine_type": "PYTHON",
            "runtime_mode": "python_local",
            "session_id": session["session_id"],
            "session_code": session["session_code"],
            "driver_id": self._driver_id,
            "frame_id": self._stats["total_frames"],
            "timestamp_ms": int(round(snap.timestamp_ms)),
            "fps": round(min(60.0, inst_fps), 1),
            "inference_latency_ms": round(float(snap.inference_latency_ms), 2),
            "face_detected": bool(snap.face_detected),
            "operational_state": snap.operational_state,
            "state": (
                snap.alert_state
                if snap.operational_state == OperationalState.MONITORING_ACTIVE.value
                else "MONITORING_UNAVAILABLE"
            ),
            "drowsiness_state": snap.drowsiness_state,
            "monitoring_status": snap.operational_state,
            "is_calibrated": snap.calibrated_profile_id is not None,
            "calibrated_profile_id": snap.calibrated_profile_id,
            "risk_score": round(min(1.0, max(0.0, float(snap.drowsiness_severity) / 3.0)), 3),
            "pose_zone": snap.pose_zone,
            "reason_codes": reasons,
            "primary_reason": snap.primary_reason,
            "eye_metrics": {
                "ear_left": round(float(snap.ear_left), 4),
                "ear_right": round(float(snap.ear_right), 4),
                "ear_raw": round(float(snap.ear_weighted), 4),
                "ear_normalized": round(float(snap.ear_compensated), 4),
                "ear_closure_threshold": round(float(snap.ear_closure_threshold), 4),
                "is_eye_closed": bool(snap.is_eye_closed),
                "perclos_pct": round(float(snap.perclos_pct), 2),
                "perclos_reliable": bool(snap.perclos_is_reliable),
                "perclos_coverage_pct": round(float(snap.perclos_coverage_pct), 1),
                "total_blink_count": int(snap.total_blinks),
            },
            "head_pose": {
                "pitch_deg": round(float(snap.pitch_deg), 2),
                "yaw_deg": round(float(snap.yaw_deg), 2),
                "roll_deg": round(float(snap.roll_deg), 2),
                "rel_pitch_deg": round(float(snap.rel_pitch_deg), 2),
                "rel_yaw_deg": round(float(snap.rel_yaw_deg), 2),
                "rel_roll_deg": round(float(snap.rel_roll_deg), 2),
                "zone": snap.pose_zone,
            },
            "mouth_metrics": {
                "mar": round(float(snap.mar), 4),
                "is_yawning": bool(snap.is_yawning),
                "total_yawns": int(snap.total_yawns),
            },
            "optical_quality": {
                "luminance": round(float(snap.luminance), 1),
                "blur_variance": round(float(snap.blur_variance), 1),
                "status": snap.optical_status,
            },
            "cascade": {
                "arbitration_winner": "GEOMETRIC_AUTHORITATIVE",
                "cnn_execution_mode": "SHADOW_DISABLED",
                "onnx_inference_executed": False,
            },
            "calibration": cal_payload,
        }
        self._latest_telemetry_payload = telemetry_payload

        # Encode downsampled MJPEG preview ONLY when at least one viewer is attached
        if self._mjpeg_viewer_count > 0:
            self._encode_preview_jpeg_locked(frame_bgr)

        # Persist WARNING / CRITICAL transitions idempotently to SQLite & emit SAFETY_EVENT
        safety_event_payload: Optional[Dict[str, Any]] = None
        tr = result.alert_transition
        if tr is not None and tr.new_state in (AlertState.WARNING, AlertState.CRITICAL):
            self._alert_seq += 1
            event_id = f"evt_py_{session['session_id'][:8]}_{int(round(snap.timestamp_ms))}_{self._alert_seq}"
            risk_score = 0.88 if tr.new_state == AlertState.CRITICAL else 0.58
            self._db.record_alert(
                event_id=event_id,
                session_id=session["session_id"],
                session_code=session["session_code"],
                driver_id=self._driver_id,
                severity=tr.new_state.value,
                previous_state=tr.previous_state.value,
                new_state=tr.new_state.value,
                risk_score=risk_score,
                ear_value=float(snap.ear_compensated),
                perclos_pct=float(snap.perclos_pct),
                pose_zone=snap.pose_zone,
                reasons=reasons or [f"{tr.new_state.value}_TRANSITION"],
                timestamp_ms=int(round(snap.timestamp_ms)),
                engine_type="PYTHON",
            )
            safety_event_payload = {
                "type": "SAFETY_EVENT",
                "event_id": event_id,
                "session_id": session["session_id"],
                "session_code": session["session_code"],
                "previous_state": tr.previous_state.value,
                "new_state": tr.new_state.value,
                "timestamp": float(snap.timestamp_ms) / 1000.0,
                "metrics": {
                    "ear": round(float(snap.ear_compensated), 4),
                    "perclos": round(float(snap.perclos_pct), 2),
                    "pose_zone": snap.pose_zone,
                    "risk_score": risk_score,
                    "reasons": reasons,
                },
            }

        # Broadcast outside critical computation
        self._broadcast_ws(telemetry_payload, critical=False)
        if cal_payload is not None and cal_payload.get("is_completed"):
            self._broadcast_ws({"type": "CALIBRATION", "calibration": cal_payload}, critical=True)
        if safety_event_payload is not None:
            self._broadcast_ws(safety_event_payload, critical=True)

    def _encode_preview_jpeg_locked(self, frame_bgr: np.ndarray) -> None:
        try:
            h, w = frame_bgr.shape[:2]
            preview = frame_bgr
            if w > 640:
                scale = 640.0 / float(w)
                preview = cv2.resize(frame_bgr, (640, max(1, int(round(h * scale)))))
            ok, encoded = cv2.imencode(".jpg", preview, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            if ok:
                self._latest_jpeg = encoded.tobytes()
                self._latest_jpeg_seq += 1
                self._mjpeg_cond.notify_all()
        except Exception as exc:
            logger.debug("MJPEG preview encode error: %s", exc)

    # ------------------------------------------------------------------
    # Calibration & Profile Operations
    # ------------------------------------------------------------------

    def start_calibration(
        self,
        *,
        four_zone_mode: bool = True,
        step_duration_sec: float = 4.0,
    ) -> Dict[str, Any]:
        with self._lock:
            if not self._is_running or self._service is None:
                raise RuntimeError("Local camera monitoring must be running before starting calibration.")
            self._service.start_calibration(
                four_zone_mode=four_zone_mode,
                step_duration_sec=step_duration_sec,
            )
            self._latest_operational_state = OperationalState.CALIBRATING.value
            return {
                "started": True,
                "driver_id": self._driver_id,
                "four_zone_mode": four_zone_mode,
                "step_duration_sec": step_duration_sec,
            }

    def skip_calibration_step(self) -> Dict[str, Any]:
        with self._lock:
            if not self._is_running or self._service is None:
                raise RuntimeError("Local camera monitoring is not running.")
            completed = self._service.skip_calibration_step()
            return {
                "skipped": True,
                "completed": completed is not None,
                "profile": profile_to_dto(completed).model_dump() if completed is not None else None,
            }

    def cancel_calibration(self) -> Dict[str, Any]:
        with self._lock:
            if self._service is not None:
                self._service.cancel_calibration()
            self._latest_calibration_payload = None
            return {"cancelled": True}

    def recalibrate_zone(
        self,
        zone: str,
        step_duration_sec: float = 4.0,
    ) -> Dict[str, Any]:
        with self._lock:
            if not self._is_running or self._service is None:
                raise RuntimeError("Local camera monitoring must be running before recalibrating a zone.")
            self._service.start_calibration(
                four_zone_mode=False,
                target_zone=zone.upper(),
                step_duration_sec=step_duration_sec,
            )
            self._latest_operational_state = OperationalState.CALIBRATING.value
            return {
                "started": True,
                "driver_id": self._driver_id,
                "target_zone": zone.upper(),
                "step_duration_sec": step_duration_sec,
            }

    def get_profile(self, driver_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        target_driver = driver_id or self._driver_id
        dto = self._store.get_profile(target_driver)
        return dto.model_dump() if dto is not None else None

    def reconcile_profile(
        self,
        driver_id: str,
        remote_profile: DriverProfileDTO,
    ) -> Dict[str, Any]:
        result = self._store.reconcile_profile(driver_id, remote_profile)
        active_dto: DriverProfileDTO = result["profile"]
        with self._lock:
            if self._service is not None and self._driver_id == driver_id and active_dto.is_calibrated:
                try:
                    self._service.set_profile(dto_to_profile(active_dto))
                except Exception as exc:
                    logger.warning("Failed to hot-activate reconciled profile: %s", exc)
        return {
            "action": result["action"],
            "archived_previous": result["archived_previous"],
            "profile": active_dto.model_dump(),
        }

    # ------------------------------------------------------------------
    # Status, Devices & SQLite Query Helpers
    # ------------------------------------------------------------------

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            is_calibrated = bool(
                self._service is not None and self._service.calibrated_profile_id is not None
            )
            is_calibrating = bool(self._service is not None and self._service.is_calibrating)
            camera_opened = bool(self._camera is not None and self._camera.is_opened())
            return {
                "running": self._is_running,
                "camera_opened": camera_opened,
                "camera_source": self._camera_source if self._is_running else None,
                "driver_id": self._driver_id,
                "engine_type": "PYTHON",
                "operational_state": self._latest_operational_state if self._is_running else "IDLE",
                "is_calibrated": is_calibrated,
                "is_calibrating": is_calibrating,
                "active_session": self._active_session,
                "frames_processed": self._stats["total_frames"],
                "last_error": self._last_error,
                "latest_telemetry": self._latest_telemetry_payload,
            }

    def list_devices(self, max_probe: int = 3) -> Dict[str, Any]:
        with self._lock:
            running = self._is_running
            active_src = self._camera_source

        if running and isinstance(active_src, int):
            indices = [active_src]
        else:
            indices = discover_local_devices(max_devices_to_probe=max(1, min(6, int(max_probe))))

        return {
            "devices": [
                {
                    "id": idx,
                    "label": f"Local Camera #{idx}",
                    "type": "OPENCV_LOCAL",
                    "in_use": bool(running and active_src == idx),
                }
                for idx in indices
            ],
            "supports_phone_url": True,
            "active_source": active_src if running else None,
        }

    def list_sessions(self, driver_id: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        return self._db.list_sessions(driver_id=driver_id, limit=limit)

    def list_session_alerts(self, session_id: str, limit: int = 200) -> List[Dict[str, Any]]:
        return self._db.list_session_alerts(session_id=session_id, limit=limit)

    # ------------------------------------------------------------------
    # MJPEG Preview & WebSocket Subscriber Plumbing
    # ------------------------------------------------------------------

    def register_mjpeg_viewer(self) -> None:
        with self._lock:
            self._mjpeg_viewer_count += 1

    def unregister_mjpeg_viewer(self) -> None:
        with self._lock:
            self._mjpeg_viewer_count = max(0, self._mjpeg_viewer_count - 1)

    def wait_for_next_jpeg(self, last_seq: int, timeout_sec: float = 0.5) -> Tuple[int, Optional[bytes], bool]:
        with self._mjpeg_cond:
            if self._is_running and self._latest_jpeg_seq == last_seq:
                self._mjpeg_cond.wait(timeout=timeout_sec)
            return self._latest_jpeg_seq, self._latest_jpeg, self._is_running

    def subscribe_ws(self, loop: asyncio.AbstractEventLoop) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=16)
        with self._ws_lock:
            self._ws_subscribers.append((loop, q))
        return q

    def unsubscribe_ws(self, queue: asyncio.Queue) -> None:
        with self._ws_lock:
            self._ws_subscribers = [(lp, q) for (lp, q) in self._ws_subscribers if q is not queue]

    def _broadcast_ws(self, payload: Dict[str, Any], *, critical: bool = False) -> None:
        with self._ws_lock:
            subscribers = list(self._ws_subscribers)
        if not subscribers:
            return

        def _enqueue(q: asyncio.Queue, item: Dict[str, Any], is_crit: bool) -> None:
            if q.full():
                if not is_crit:
                    return
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            try:
                q.put_nowait(item)
            except asyncio.QueueFull:
                pass

        for loop, queue in subscribers:
            if loop.is_closed():
                continue
            try:
                loop.call_soon_threadsafe(_enqueue, queue, payload, critical)
            except RuntimeError:
                pass


local_camera_controller = LocalCameraController()
