"""
DriveGuard Session AI Engine — Thin Session & Transport Adapter.

Wraps the single authoritative `LocalMonitoringService` pipeline:
  FrameQualityAnalyzer
    -> DriveGuardFaceLandmarker (backend.app.vision.face_landmarker)
    -> DriverFeatureExtractor (EAR, Head Pose, MAR/Yawn, Blink Kinematics, PERCLOS, PDBP)
    -> DrowsinessEngine (SignalConfidenceEvaluator -> TemporalAnalyzer -> DecisionEngine)
    -> AlertStateManager

Responsibilities retained strictly for session & transport orchestration:
  1. Per-session identity and ownership metadata (`session_id`, `driver_id`, `owner_id`).
  2. Thread-safe execution serialization (`_process_lock`) and FIFO frame-boundary
     command queue (`submit_command`, `_drain_pending_commands_locked`).
  3. Frame input adaptation via `frame_adapters` (JPEG/PNG bytes, RGB24 arrays, BGR24 arrays).
  4. Projection of authoritative `LocalMonitoringFrameResult` outputs into FastAPI /
     WebSocket / LiveKit Pydantic wire contracts (`FrameTelemetryDTO`, `CalibrationPhaseDTO`).
  5. Bounded latency / FPS diagnostics and safe native resource cleanup (`close()`).

Zero duplicated drowsiness detection, EAR, MAR, blink, PERCLOS, head-pose, or
risk-scoring logic lives in this module.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Union

import numpy as np

from backend.app.core.experiment_mode import ExperimentMode
from backend.app.ai.pdbp_calibration import DriverBehaviourProfile
from backend.app.ai.profile_adapter import dto_to_profile, profile_to_dto
from backend.app.alerts.alert_state import AlertState
from backend.app.alerts.alert_state_manager import AlertStateManager
from backend.app.features.feature_extractor import DriverFeatureExtractor
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState
from backend.app.schemas.calibration import CalibrationPhaseDTO
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.schemas.telemetry import (
    AlertStateEnum,
    CascadeStatusDTO,
    ConventionalFixedComparisonDTO,
    EngineTypeEnum,
    EyeMetricsDTO,
    FrameTelemetryDTO,
    HeadPoseDTO,
    OperationalStateEnum,
    PoseZoneEnum,
)
from backend.app.services.frame_adapters import (
    ArrayFrameAdapter,
    EncodedBytesFrameAdapter,
    FrameDecodeError,
)
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringFrameResult,
    LocalMonitoringService,
    ProcessedTelemetrySnapshot,
)
from backend.app.services.profile_store import ProfileStore
from backend.app.vision.face_landmarker import DriveGuardFaceLandmarker

logger = logging.getLogger("driveguard.pipeline")


def _map_authoritative_state_to_wire(
    alert_state: AlertState,
    drowsiness_state: DrowsinessState,
    alert_active: bool,
) -> tuple[AlertStateEnum, float]:
    """
    Map authoritative Phase 6/8 alert state to the backward-compatible 4-level
    wire `AlertStateEnum` and normalized diagnostic risk score in [0.0, 1.0].
    """
    if drowsiness_state == DrowsinessState.INSUFFICIENT_DATA and not alert_active:
        return AlertStateEnum.MONITORING_UNAVAILABLE, 0.0
    if alert_state == AlertState.CRITICAL:
        return AlertStateEnum.CRITICAL, 0.95
    if alert_state == AlertState.WARNING:
        return AlertStateEnum.WARNING, 0.65
    if alert_state == AlertState.ATTENTION:
        return AlertStateEnum.WARNING, 0.45
    return AlertStateEnum.NORMAL, 0.05


@dataclass
class EngineCommand:
    """Encapsulates a state mutation command queued for execution at a safe frame boundary."""

    action: str
    params: Dict[str, Any]
    future: Optional[asyncio.Future] = None
    event_loop: Optional[asyncio.AbstractEventLoop] = None


class SessionAIEngine:
    """
    Thin stateful session adapter delegating all computer vision, feature extraction,
    drowsiness inference, calibration, and alerting to `LocalMonitoringService`.
    """

    def __init__(
        self,
        session_id: str = "session_default",
        driver_id: str = "driver_default",
        profile: Optional[DriverBehaviourProfile] = None,
        experiment_mode: ExperimentMode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
        onnx_model_path: Optional[str | Path] = None,
        landmarker: Optional[DriveGuardFaceLandmarker] = None,
        face_loss_grace_ms: float = 450.0,
        owner_id: Optional[str] = None,
        is_mirrored: bool = False,
        store: Optional[ProfileStore] = None,
    ) -> None:
        self.session_id = session_id
        self.driver_id = driver_id
        self.owner_id: str = owner_id or driver_id
        self.is_mirrored: bool = bool(is_mirrored)
        self._onnx_model_path = onnx_model_path

        if isinstance(experiment_mode, str):
            try:
                self.experiment_mode = ExperimentMode(experiment_mode)
            except ValueError:
                self.experiment_mode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE
        else:
            self.experiment_mode = experiment_mode

        # Authoritative profile record held by session adapter
        if profile is not None:
            self._profile = profile
        else:
            self._profile = DriverBehaviourProfile.create_default_uncalibrated(driver_id=driver_id)

        # Single authoritative monitoring pipeline (headless: no server speaker I/O)
        self.monitoring_service = LocalMonitoringService(
            config=LocalMonitoringConfig(
                driver_id=driver_id,
                is_mirrored=self.is_mirrored,
                cnn_shadow_enabled=False,
            ),
            store=store,
            landmarker=landmarker,
            enable_landmarker=(landmarker is None),
            enable_shadow_worker=False,
            enable_audio=False,
            enable_event_logging=False,
            auto_load_profile=False,
            persist_calibration_to_store=False,
            initial_profile=profile,
            face_loss_grace_ms=float(face_loss_grace_ms),
            landmarker_grace_timeout_sec=max(0.05, float(face_loss_grace_ms) / 1000.0),
        )

        # Calibration DTO cache for WebSocket / REST status polling
        self._last_completed_calibration_dto: Optional[CalibrationPhaseDTO] = None
        self._last_frame_result: Optional[LocalMonitoringFrameResult] = None

        # Telemetry & performance counters
        self.frame_index: int = 0
        self.prev_timestamp_ms: Optional[float] = None
        self.fps_ema: float = 30.0
        self.total_cnn_invocations: int = 0
        self.total_disagreements: int = 0
        self.total_mirror_check_suppressions: int = 0
        self.total_fixed_false_alarms: int = 0

        # Hybrid latency history management:
        # 1. Exact O(1) lifetime accumulators for session summary mean
        # 2. Bounded circular buffer (1500 samples ~= 60s at 25 FPS)
        self._lifetime_latency_count: int = 0
        self._lifetime_latency_sum_ms: float = 0.0
        self._recent_latencies_ms: Deque[float] = deque(maxlen=1500)

        self._process_lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._pending_commands: Deque[EngineCommand] = deque()
        self._is_closed = False

        # Face-loss availability envelope (transport observability without competing AI)
        self.face_loss_grace_ms: float = float(face_loss_grace_ms)
        self._face_lost_start_ms: Optional[float] = None
        self._last_seen_ms: Optional[float] = None
        self._has_seen_face: bool = False
        self._last_valid_risk_score: float = 0.0
        self._last_valid_alert_state: AlertStateEnum = AlertStateEnum.MONITORING_UNAVAILABLE
        self._monitoring_status: str = "MONITORING_UNAVAILABLE"
        self._recovery_frame_count: int = 0
        self.REQUIRED_RECOVERY_FRAMES: int = 3

    # --------------------------------------------------------------------------
    # Authoritative Core Component Accessors
    # --------------------------------------------------------------------------

    @property
    def drowsiness_engine(self) -> DrowsinessEngine:
        """Authoritative DrowsinessEngine shared with local monitoring."""
        return self.monitoring_service.drowsiness_engine

    @property
    def feature_extractor(self) -> DriverFeatureExtractor:
        """Authoritative DriverFeatureExtractor shared with local monitoring."""
        return self.monitoring_service.feature_extractor

    @property
    def alert_manager(self) -> AlertStateManager:
        """Authoritative AlertStateManager shared with local monitoring."""
        return self.monitoring_service.alert_manager

    @property
    def landmarker(self) -> Optional[DriveGuardFaceLandmarker]:
        """Authoritative vision landmarker owned by LocalMonitoringService."""
        return self.monitoring_service.landmarker

    @landmarker.setter
    def landmarker(self, value: Optional[DriveGuardFaceLandmarker]) -> None:
        self.monitoring_service.landmarker = value

    @property
    def profile(self) -> DriverBehaviourProfile:
        """Active session driver profile."""
        return self._profile

    @profile.setter
    def profile(self, value: DriverBehaviourProfile) -> None:
        self._profile = value

    @property
    def calibration_session(self):
        return self.monitoring_service.calibration_session

    @calibration_session.setter
    def calibration_session(self, value) -> None:
        self.monitoring_service.calibration_session = value

    @property
    def latencies_ms(self) -> List[float]:
        """Shallow list copy of recent rolling window of up to 1500 samples."""
        return list(self._recent_latencies_ms)

    def get_recent_latencies(self) -> List[float]:
        """Return shallow list copy of recent rolling window of up to 1500 latency samples."""
        return list(self._recent_latencies_ms)

    def get_latest_telemetry_snapshot(self) -> Optional[ProcessedTelemetrySnapshot]:
        """Return the latest authoritative `ProcessedTelemetrySnapshot` (zero raw video)."""
        return self.monitoring_service.get_latest_telemetry()

    # --------------------------------------------------------------------------
    # Calibration Lifecycle Management (Delegated to LocalMonitoringService)
    # --------------------------------------------------------------------------

    def start_calibration(
        self,
        four_zone_mode: Optional[bool] = None,
        three_zone_mode: Optional[bool] = None,
        zone_duration_sec: float = 5.0,
        target_zone: Optional[str] = None,
        timestamp_ms: Optional[float] = None,
        include_blink_baseline: bool = False,
        blink_baseline_duration_sec: float = 120.0,
    ) -> CalibrationPhaseDTO:
        """Begin a new guided calibration session (synchronized under process lock)."""
        with self._process_lock:
            return self._start_calibration_locked(
                four_zone_mode=four_zone_mode,
                three_zone_mode=three_zone_mode,
                zone_duration_sec=zone_duration_sec,
                target_zone=target_zone,
                timestamp_ms=timestamp_ms,
                include_blink_baseline=include_blink_baseline,
                blink_baseline_duration_sec=blink_baseline_duration_sec,
            )

    def _start_calibration_locked(
        self,
        four_zone_mode: Optional[bool] = None,
        three_zone_mode: Optional[bool] = None,
        zone_duration_sec: float = 5.0,
        target_zone: Optional[str] = None,
        timestamp_ms: Optional[float] = None,
        include_blink_baseline: bool = False,
        blink_baseline_duration_sec: float = 120.0,
    ) -> CalibrationPhaseDTO:
        if four_zone_mode is None and three_zone_mode is None and target_zone is None:
            is_4z, is_3z = True, False
        else:
            is_4z = bool(four_zone_mode)
            is_3z = not is_4z and bool(three_zone_mode)

        self.monitoring_service.start_calibration(
            timestamp_ms=timestamp_ms,
            is_mirrored=self.is_mirrored,
            four_zone_mode=is_4z,
            three_zone_mode=is_3z,
            step_duration_sec=zone_duration_sec,
            target_zone=target_zone,
            include_blink_baseline=include_blink_baseline,
            blink_baseline_duration_sec=blink_baseline_duration_sec,
        )
        self._last_completed_calibration_dto = None
        return self._build_calibration_dto(timestamp_ms=timestamp_ms)

    def cancel_calibration(self) -> None:
        """Cancel an ongoing calibration session."""
        with self._process_lock:
            self._cancel_calibration_locked()

    def _cancel_calibration_locked(self) -> None:
        self.monitoring_service.cancel_calibration()

    def skip_calibration_step(self, timestamp_ms: Optional[float] = None) -> Optional[CalibrationPhaseDTO]:
        """Manually advance past the current calibration zone."""
        with self._process_lock:
            return self._skip_calibration_step_locked(timestamp_ms=timestamp_ms)

    def _skip_calibration_step_locked(self, timestamp_ms: Optional[float] = None) -> Optional[CalibrationPhaseDTO]:
        if not self.monitoring_service.is_calibrating:
            return None
        completed_profile = self.monitoring_service.skip_calibration_step(timestamp_ms=timestamp_ms)
        if completed_profile is not None:
            self._profile = completed_profile
            self._last_completed_calibration_dto = CalibrationPhaseDTO(
                current_zone="CENTER",
                phase_index=1,
                total_phases=1,
                zone_progress_pct=100.0,
                total_progress_pct=100.0,
                stable_samples_count=len(completed_profile.zone_baselines),
                target_samples_count=15,
                instruction="Calibration complete.",
                is_head_in_target_zone=True,
                stability_feedback="Profile Ready",
                is_completed=True,
                profile=self.export_profile_dto(),
            )
            return self._last_completed_calibration_dto
        return self._build_calibration_dto(timestamp_ms=timestamp_ms)

    @property
    def is_calibrating(self) -> bool:
        """Check whether the session is currently in active calibration mode."""
        return self.monitoring_service.is_calibrating

    def get_calibration_status(self) -> Optional[CalibrationPhaseDTO]:
        """Return the current calibration progress or the last completed calibration DTO."""
        if self.monitoring_service.calibration_session is not None:
            return self._build_calibration_dto()
        return self._last_completed_calibration_dto

    def _build_calibration_dto(self, timestamp_ms: Optional[float] = None) -> CalibrationPhaseDTO:
        """Construct the Pydantic CalibrationPhaseDTO from the active GuidedCalibrationSession."""
        cal_sess = self.monitoring_service.calibration_session
        if cal_sess is None or not cal_sess.is_running:
            if self._last_completed_calibration_dto is not None:
                return self._last_completed_calibration_dto
            return CalibrationPhaseDTO(
                current_zone="CENTER",
                phase_index=1,
                total_phases=4,
                zone_progress_pct=100.0,
                total_progress_pct=100.0,
                stable_samples_count=0,
                target_samples_count=15,
                instruction="Calibration complete.",
                is_head_in_target_zone=True,
                stability_feedback="Profile Ready",
                is_completed=True,
                rejection_reason=cal_sess.rejection_reason if cal_sess else None,
                quality_warnings=list(cal_sess.quality_warnings) if cal_sess else [],
                profile=self.export_profile_dto() if (cal_sess and cal_sess.is_completed) else None,
            )

        ts = timestamp_ms if timestamp_ms is not None else (time.perf_counter() * 1000.0)
        curr_step_num, total_steps, ratio, instruction, is_aligned, feedback = cal_sess.get_progress(ts)
        phase_idx = max(1, curr_step_num)
        step = cal_sess.current_step
        zone_samples = len(cal_sess._ear_samples_by_zone.get(step.zone.value, []))
        total_pct = max(0.0, min(100.0, (((phase_idx - 1) + ratio) / max(1, total_steps)) * 100.0))
        zone_pct = max(0.0, min(100.0, ratio * 100.0))
        live_blink = cal_sess.get_live_blink_telemetry()
        curr_zone_name = "BLINK_CADENCE" if step.step_type == "BLINK_CADENCE" else step.zone.value

        return CalibrationPhaseDTO(
            current_zone=curr_zone_name,
            phase_index=phase_idx,
            total_phases=total_steps,
            phase_type=str(live_blink.get("phase_type", "SPATIAL_ZONE")),
            zone_progress_pct=round(zone_pct, 1),
            total_progress_pct=round(total_pct, 1),
            stable_samples_count=zone_samples,
            target_samples_count=cal_sess.target_samples_per_zone,
            elapsed_sec=float(live_blink.get("elapsed_sec") or 0.0),
            target_duration_sec=float(live_blink.get("target_duration_sec") or step.duration_sec),
            collected_blinks_count=int(live_blink.get("collected_blinks_count") or 0),
            live_blink_rate_per_min=float(live_blink.get("live_blink_rate_per_min") or 0.0),
            last_blink_duration_ms=live_blink.get("last_blink_duration_ms"),  # type: ignore[arg-type]
            live_median_blink_duration_ms=live_blink.get("live_median_blink_duration_ms"),  # type: ignore[arg-type]
            live_median_blink_rcr=live_blink.get("live_median_blink_rcr"),  # type: ignore[arg-type]
            instruction=instruction,
            is_head_in_target_zone=is_aligned,
            stability_feedback=feedback,
            is_completed=cal_sess.is_completed,
            rejection_reason=cal_sess.rejection_reason,
            quality_warnings=list(cal_sess.quality_warnings),
        )

    # --------------------------------------------------------------------------
    # Profile Import / Export (Synchronized & Delegated to Authoritative Extractor)
    # --------------------------------------------------------------------------

    def set_profile(self, profile: DriverBehaviourProfile) -> None:
        """Update active personal baseline profile on the authoritative extractor."""
        with self._process_lock:
            self._set_profile_locked(profile)

    def _set_profile_locked(self, profile: DriverBehaviourProfile) -> None:
        self._profile = profile
        self.monitoring_service.set_profile(profile)

    def export_profile_dto(self) -> DriverProfileDTO:
        """Export the active driver profile as a validated Pydantic DTO."""
        with self._process_lock:
            return profile_to_dto(self._profile)

    def load_profile_dto(self, dto: DriverProfileDTO) -> None:
        """Load and activate a driver profile from a Pydantic DTO."""
        with self._process_lock:
            self._load_profile_dto_locked(dto)

    def _load_profile_dto_locked(self, dto: DriverProfileDTO) -> None:
        new_profile = dto_to_profile(dto)
        self._set_profile_locked(new_profile)

    def set_experiment_mode(self, mode: Union[ExperimentMode, str]) -> ExperimentMode:
        """Validate and update session experiment metadata mode."""
        validated = self._validate_experiment_mode(mode)
        with self._process_lock:
            return self._set_experiment_mode_locked(validated)

    def _validate_experiment_mode(self, mode: Union[ExperimentMode, str, None]) -> ExperimentMode:
        if mode is None:
            raise ValueError("Experiment mode cannot be None")
        if isinstance(mode, ExperimentMode):
            return mode
        if isinstance(mode, str):
            for em in ExperimentMode:
                if em.value == mode or em.name == mode:
                    return em
        valid_modes = [m.value for m in ExperimentMode]
        raise ValueError(f"Invalid experiment mode: '{mode}'. Must be one of {valid_modes}")

    def _set_experiment_mode_locked(self, mode: ExperimentMode) -> ExperimentMode:
        self.experiment_mode = mode
        return mode

    # --------------------------------------------------------------------------
    # Thread-Safe Command Boundary Mechanism
    # --------------------------------------------------------------------------

    async def submit_command(self, action: str, **params: Any) -> Any:
        """
        Thread-safe asynchronous entry point for engine state mutations.
        Enqueues command for FIFO execution at the next frame boundary under `_process_lock`.
        """
        if self._is_closed:
            raise RuntimeError(
                f"SessionAIEngine '{self.session_id}' is closed; command '{action}' rejected."
            )

        if action == "SWITCH_EXPERIMENT":
            self._validate_experiment_mode(params.get("mode"))

        loop = asyncio.get_running_loop()
        fut: asyncio.Future = loop.create_future()
        cmd = EngineCommand(action=action, params=params, future=fut, event_loop=loop)

        with self._command_lock:
            if self._is_closed:
                raise RuntimeError(
                    f"SessionAIEngine '{self.session_id}' is closed; command '{action}' rejected."
                )
            self._pending_commands.append(cmd)

        drain_task = asyncio.create_task(asyncio.to_thread(self._drain_pending_commands))
        try:
            return await fut
        finally:
            if not drain_task.done():
                try:
                    await drain_task
                except Exception:
                    pass

    def _drain_pending_commands(self) -> None:
        with self._process_lock:
            self._drain_pending_commands_locked()

    def _drain_pending_commands_locked(self) -> None:
        with self._command_lock:
            if not self._pending_commands:
                return
            cmds_to_run = list(self._pending_commands)
            self._pending_commands.clear()

        for cmd in cmds_to_run:
            if cmd.future is not None and cmd.future.done():
                continue
            try:
                res = self._execute_command_internal_locked(cmd.action, cmd.params)
                if cmd.future and cmd.event_loop and not cmd.future.done():
                    cmd.event_loop.call_soon_threadsafe(cmd.future.set_result, res)
            except Exception as exc:
                if cmd.future and cmd.event_loop and not cmd.future.done():
                    cmd.event_loop.call_soon_threadsafe(cmd.future.set_exception, exc)
                elif not cmd.future:
                    logger.error(
                        "[SessionAIEngine] Error executing command '%s': %s",
                        cmd.action,
                        exc,
                        exc_info=True,
                    )

    def _execute_command_internal_locked(self, action: str, params: Dict[str, Any]) -> Any:
        if self._is_closed:
            raise RuntimeError(f"Cannot execute '{action}' on closed engine.")

        if action == "START_CALIBRATION":
            return self._start_calibration_locked(
                four_zone_mode=params.get("four_zone_mode"),
                three_zone_mode=params.get("three_zone_mode"),
                zone_duration_sec=params.get("zone_duration_sec", 5.0),
                target_zone=params.get("target_zone"),
                timestamp_ms=params.get("timestamp_ms"),
                include_blink_baseline=params.get("include_blink_baseline", False),
                blink_baseline_duration_sec=params.get("blink_baseline_duration_sec", 120.0),
            )
        elif action == "CANCEL_CALIBRATION":
            self._cancel_calibration_locked()
            return None
        elif action == "SKIP_CALIBRATION_STEP":
            return self._skip_calibration_step_locked(timestamp_ms=params.get("timestamp_ms"))
        elif action == "SWITCH_EXPERIMENT":
            mode = self._validate_experiment_mode(params["mode"])
            return self._set_experiment_mode_locked(mode)
        elif action == "SET_PROFILE":
            self._set_profile_locked(params["profile"])
            return None
        elif action == "LOAD_PROFILE_DTO":
            self._load_profile_dto_locked(params["dto"])
            return None
        else:
            raise ValueError(f"Unknown engine command action: '{action}'")

    # --------------------------------------------------------------------------
    # Real-Time Frame Processing (Delegated to Authoritative LocalMonitoringService)
    # --------------------------------------------------------------------------

    def process_frame(
        self,
        frame_input: Union[np.ndarray, bytes],
        timestamp_ms: Optional[float] = None,
        *,
        is_rgb: bool = True,
    ) -> FrameTelemetryDTO:
        """
        Adapt incoming frame bytes or NumPy array and execute the single authoritative
        `LocalMonitoringService` pipeline under `_process_lock`.
        """
        with self._process_lock:
            if self._is_closed:
                return self._build_unavailable_frame(
                    timestamp_ms if timestamp_ms is not None else (time.perf_counter() * 1000.0),
                    latency_ms=0.0,
                    reason="SESSION_CLOSED",
                )
            return self._process_frame_locked(frame_input, timestamp_ms, is_rgb=is_rgb)

    def _process_frame_locked(
        self,
        frame_input: Union[np.ndarray, bytes],
        timestamp_ms: Optional[float] = None,
        *,
        is_rgb: bool = True,
    ) -> FrameTelemetryDTO:
        t_start = time.perf_counter()
        self._drain_pending_commands_locked()
        self.frame_index += 1

        ts_ms = (t_start * 1000.0) if timestamp_ms is None else float(timestamp_ms)

        if self.prev_timestamp_ms is not None:
            dt_sec = max(1e-4, (ts_ms - self.prev_timestamp_ms) / 1000.0)
            self.fps_ema = 0.90 * self.fps_ema + 0.10 * (1.0 / dt_sec)
        self.prev_timestamp_ms = ts_ms

        # 1. Adapt heterogeneous input source via canonical frame adapters
        try:
            if isinstance(frame_input, (bytes, bytearray)):
                adapted = EncodedBytesFrameAdapter.adapt(
                    bytes(frame_input),
                    timestamp_ms=ts_ms,
                    frame_index=self.frame_index,
                    source_id=f"session:{self.session_id}:bytes",
                    is_mirrored=self.is_mirrored,
                )
            elif isinstance(frame_input, np.ndarray):
                if is_rgb:
                    adapted = ArrayFrameAdapter.adapt_rgb(
                        frame_input,
                        timestamp_ms=ts_ms,
                        frame_index=self.frame_index,
                        source_id=f"session:{self.session_id}:rgb",
                        is_mirrored=self.is_mirrored,
                    )
                else:
                    adapted = ArrayFrameAdapter.adapt_bgr(
                        frame_input,
                        timestamp_ms=ts_ms,
                        frame_index=self.frame_index,
                        source_id=f"session:{self.session_id}:bgr",
                        is_mirrored=self.is_mirrored,
                    )
            else:
                raise FrameDecodeError(f"Unsupported frame input type: {type(frame_input).__name__}")
        except FrameDecodeError as exc:
            logger.debug("[SessionAIEngine] Frame decode failed on '%s': %s", self.session_id, exc)
            return self._build_unavailable_frame(ts_ms, latency_ms=0.5, reason="IMAGE_DECODE_FAILED")

        # 2. Run the SINGLE authoritative LocalMonitoringService pipeline
        try:
            cal_session_before = self.monitoring_service.calibration_session
            cal_steps_before = len(cal_session_before.steps) if cal_session_before is not None else 4
            cal_warnings_before = (
                list(cal_session_before.quality_warnings) if cal_session_before is not None else []
            )
            frame_res = self.monitoring_service.process_adapted_frame(adapted)
        except Exception as exc:
            logger.error(
                "[SessionAIEngine] Authoritative pipeline exception at %.1fms: %s",
                ts_ms,
                exc,
                exc_info=True,
            )
            return self._handle_face_loss_or_fault(
                ts_ms=ts_ms,
                latency_ms=(time.perf_counter() - t_start) * 1000.0,
                fault_reason=f"LANDMARKER_EXCEPTION: {type(exc).__name__}",
                grace_reason=f"FACE_VISIBILITY_DEGRADED: {type(exc).__name__}",
            )

        self._last_frame_result = frame_res

        # 3. Capture calibration completion/rejection DTO if calibration finished on this frame
        if frame_res.completed_calibration_profile is not None:
            self._profile = frame_res.completed_calibration_profile
            profile_dto = self.export_profile_dto()
            self._last_completed_calibration_dto = CalibrationPhaseDTO(
                current_zone="CENTER",
                phase_index=cal_steps_before,
                total_phases=cal_steps_before,
                zone_progress_pct=100.0,
                total_progress_pct=100.0,
                stable_samples_count=len(self._profile.zone_baselines),
                target_samples_count=15,
                instruction="Calibration successful! Personal baselines activated.",
                is_head_in_target_zone=True,
                stability_feedback="Profile Ready",
                is_completed=True,
                quality_warnings=cal_warnings_before,
                profile=profile_dto,
            )
        elif frame_res.calibration_rejection_reason is not None:
            self._last_completed_calibration_dto = CalibrationPhaseDTO(
                current_zone="CENTER",
                phase_index=1,
                total_phases=cal_steps_before,
                zone_progress_pct=0.0,
                total_progress_pct=0.0,
                stable_samples_count=0,
                target_samples_count=15,
                instruction="Calibration rejected. Please ensure good lighting and face visibility.",
                is_head_in_target_zone=False,
                stability_feedback="Rejected",
                is_completed=True,
                rejection_reason=frame_res.calibration_rejection_reason,
                quality_warnings=cal_warnings_before,
            )

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self._lifetime_latency_count += 1
        self._lifetime_latency_sum_ms += latency_ms
        self._recent_latencies_ms.append(latency_ms)

        landmark = frame_res.landmark
        features = frame_res.features
        assessment = frame_res.assessment

        # 4. Handle face-not-detected availability envelope cleanly
        if not landmark.has_face:
            return self._handle_face_loss_or_fault(
                ts_ms=ts_ms,
                latency_ms=latency_ms,
                fault_reason="NO_FACE_DETECTED",
                grace_reason="FACE_VISIBILITY_DEGRADED",
                frame_res=frame_res,
            )

        # 5. Face detected: update availability & 3-frame recovery status
        was_lost = self._face_lost_start_ms is not None
        was_sustained_outage = was_lost and (
            (ts_ms - (self._face_lost_start_ms or ts_ms)) >= self.face_loss_grace_ms
            or self._monitoring_status in ("MONITORING_UNAVAILABLE", "FACE_LOST")
        )
        if was_lost:
            self._face_lost_start_ms = None
        self._last_seen_ms = ts_ms

        if was_sustained_outage or (
            0 < self._recovery_frame_count < self.REQUIRED_RECOVERY_FRAMES
        ):
            self._recovery_frame_count += 1
            if self._recovery_frame_count < self.REQUIRED_RECOVERY_FRAMES:
                self._monitoring_status = "MONITORING_RECOVERING"
            else:
                self._monitoring_status = "MONITORING_ACTIVE"
                self._recovery_frame_count = 0
        else:
            self._recovery_frame_count = 0
            self._monitoring_status = "MONITORING_ACTIVE"
        self._has_seen_face = True

        # 6. Project authoritative DrowsinessEngine & AlertStateManager outputs to FrameTelemetryDTO
        op_state = OperationalStateEnum(assessment.operational_state.value)
        is_operational_gated = op_state in (
            OperationalStateEnum.CALIBRATION_REQUIRED,
            OperationalStateEnum.CALIBRATING,
        )
        wire_state, risk_score = _map_authoritative_state_to_wire(
            alert_state=frame_res.alert_state,
            drowsiness_state=assessment.state,
            alert_active=assessment.alert_active,
        )
        if is_operational_gated:
            wire_state = AlertStateEnum.MONITORING_UNAVAILABLE
            risk_score = 0.0
            self._monitoring_status = "MONITORING_UNAVAILABLE"

        self._last_valid_risk_score = risk_score
        self._last_valid_alert_state = wire_state

        is_mirror_suppressed = bool(assessment.is_mirror_check_context)
        if is_mirror_suppressed:
            self.total_mirror_check_suppressions += 1

        pose_zone_dto = features.head_pose.zone
        fixed_closed = bool(features.ear.is_geometric_valid and features.ear.weighted_ear < 0.22)
        fixed_state_str = "WARNING" if fixed_closed else "NORMAL"
        fixed_fa = bool(
            fixed_closed
            and frame_res.alert_state == AlertState.NORMAL
            and pose_zone_dto
            in (PoseZoneEnum.LEFT_MIRROR, PoseZoneEnum.RIGHT_MIRROR, PoseZoneEnum.DASHBOARD)
        )
        if fixed_fa:
            self.total_fixed_false_alarms += 1

        reason_codes = [r.code.value for r in assessment.decision.reasons] or ["ALERT_NORMAL"]
        completed_blink = features.blink.completed_blink
        active_thresh = max(0.0, float(features.ear.closure_threshold))

        return FrameTelemetryDTO(
            session_id=self.session_id,
            driver_id=self.driver_id,
            frame_index=self.frame_index,
            timestamp_ms=round(ts_ms, 1),
            fps=round(self.fps_ema, 1),
            face_detected=True,
            face_visible=True,
            grace_period_active=False,
            last_seen_ms=self._last_seen_ms,
            tracking_mode=getattr(self.landmarker, "active_implementation", "MEDIAPIPE_TASKS"),
            state=wire_state,
            drowsiness_state=assessment.state.value,
            alert_state=None if is_operational_gated else frame_res.alert_state.value,
            is_recovering=frame_res.is_recovering,
            authoritative_engine="DrowsinessEngine",
            operational_state=op_state,
            engine_type=EngineTypeEnum.PYTHON,
            monitoring_status=self._monitoring_status,
            risk_score=round(risk_score, 3),
            pose_zone=pose_zone_dto,
            reason_codes=reason_codes,
            head_pose=HeadPoseDTO(
                pitch_deg=round(float(features.head_pose.pitch_deg), 1),
                yaw_deg=round(float(features.head_pose.yaw_deg), 1),
                roll_deg=round(float(features.head_pose.roll_deg), 1),
            ),
            eye_metrics=EyeMetricsDTO(
                ear_raw=round(max(0.0, float(features.ear.weighted_ear)), 4),
                ear_left=round(max(0.0, float(features.ear.left_ear)), 4),
                ear_right=round(max(0.0, float(features.ear.right_ear)), 4),
                ear_normalized=round(max(0.0, float(features.ear.compensated_ear)), 4),
                ear_closure_threshold=round(active_thresh, 4),
                is_eye_closed=bool(features.ear.is_closed),
                fused_closure_prob=1.0 if features.ear.is_closed else 0.0,
                perclos_pct=round(max(0.0, min(100.0, float(features.perclos.perclos_pct))), 2),
                blink_duration_ms=round(float(completed_blink.duration_ms), 1) if completed_blink else None,
                blink_rcr=round(float(completed_blink.rcr), 2) if completed_blink else None,
                blink_rate_per_min=round(float(assessment.temporal_metrics.blink_count_60s), 1),
                total_blink_count=int(features.blink.total_blink_count),
                mar_yawn=round(max(0.0, float(features.mar.mar)), 3),
                blink_state=features.blink.state.value,
                blink_classification=(
                    completed_blink.classification.value if completed_blink else "NORMAL"
                ),
                perclos_coverage_pct=round(
                    max(0.0, min(100.0, float(features.perclos.coverage_pct))), 1
                ),
                perclos_reliable=bool(features.perclos.is_reliable),
                eye_closure_duration_ms=round(
                    max(0.0, float(features.blink.current_closure_duration_ms)), 1
                ),
                sampling_gap_detected=bool(features.blink.sampling_gap_detected),
            ),
            cascade=CascadeStatusDTO(
                experiment_mode=self.experiment_mode.value,
                cnn_triggered=False,
                geom_open_prob=0.0 if features.ear.is_closed else 1.0,
                geom_reliability=round(max(0.0, min(1.0, float(assessment.confidence.overall_confidence))), 3),
                cnn_open_prob=None,
                cnn_reliability=None,
                arbitration_winner="GEOMETRIC_AUTHORITATIVE",
                disagreement_flag=False,
                cnn_latency_ms=0.0,
                cnn_execution_mode="SHADOW_DISABLED",
                verifier_invoked=False,
                gate_trigger_reason="GEOMETRIC_AUTHORITATIVE_PIPELINE",
                onnx_inference_executed=False,
                cnn_inference_duration_ms=None,
            ),
            comparison=ConventionalFixedComparisonDTO(
                fixed_cutoff_ear=0.22,
                fixed_alert_state=fixed_state_str,
                fixed_false_alarm=fixed_fa,
                personalised_zone=pose_zone_dto.value,
                personalised_threshold=round(active_thresh, 4),
                is_mirror_check_suppressed=is_mirror_suppressed,
            ),
            engine_mode="CALIBRATING" if self.is_calibrating else "MONITORING",
            inference_latency_ms=round(latency_ms, 2),
        )

    def _handle_face_loss_or_fault(
        self,
        *,
        ts_ms: float,
        latency_ms: float,
        fault_reason: str,
        grace_reason: str,
        frame_res: Optional[LocalMonitoringFrameResult] = None,
    ) -> FrameTelemetryDTO:
        """Apply honest face-loss / fault availability reporting without fabricating metrics."""
        self._recovery_frame_count = 0

        if not self._has_seen_face:
            self._monitoring_status = "MONITORING_UNAVAILABLE"
            return self._build_unavailable_frame(
                ts_ms,
                latency_ms=latency_ms,
                reason=fault_reason,
                state=AlertStateEnum.MONITORING_UNAVAILABLE,
                risk_score=0.0,
                monitoring_status="MONITORING_UNAVAILABLE",
                face_visible=False,
                grace_period_active=False,
                last_seen_ms=self._last_seen_ms,
                drowsiness_state=(
                    frame_res.assessment.state.value if frame_res is not None else DrowsinessState.INSUFFICIENT_DATA.value
                ),
                alert_state=(
                    frame_res.alert_state.value if frame_res is not None else AlertState.NORMAL.value
                ),
            )

        if self._face_lost_start_ms is None:
            self._face_lost_start_ms = self._last_seen_ms if self._last_seen_ms is not None else ts_ms
        loss_duration_ms = ts_ms - self._face_lost_start_ms

        if loss_duration_ms < self.face_loss_grace_ms:
            self._monitoring_status = "FACE_VISIBILITY_DEGRADED"
            return self._build_unavailable_frame(
                ts_ms,
                latency_ms=latency_ms,
                reason=grace_reason,
                state=self._last_valid_alert_state,
                risk_score=self._last_valid_risk_score,
                monitoring_status="FACE_VISIBILITY_DEGRADED",
                face_visible=False,
                grace_period_active=True,
                last_seen_ms=self._last_seen_ms,
                drowsiness_state=(
                    frame_res.assessment.state.value if frame_res is not None else None
                ),
                alert_state=(
                    frame_res.alert_state.value if frame_res is not None else None
                ),
            )

        self._monitoring_status = "MONITORING_UNAVAILABLE"
        # Preserve unresolved safety-critical alert if authoritative engine holds alert_active=True
        if frame_res is not None and frame_res.assessment.alert_active:
            held_state, held_risk = _map_authoritative_state_to_wire(
                alert_state=frame_res.alert_state,
                drowsiness_state=frame_res.assessment.state,
                alert_active=True,
            )
            return self._build_unavailable_frame(
                ts_ms,
                latency_ms=latency_ms,
                reason=fault_reason,
                state=held_state,
                risk_score=held_risk,
                monitoring_status="MONITORING_UNAVAILABLE",
                face_visible=False,
                grace_period_active=False,
                last_seen_ms=self._last_seen_ms,
                drowsiness_state=frame_res.assessment.state.value,
                alert_state=frame_res.alert_state.value,
            )

        return self._build_unavailable_frame(
            ts_ms,
            latency_ms=latency_ms,
            reason=fault_reason,
            state=AlertStateEnum.MONITORING_UNAVAILABLE,
            risk_score=0.0,
            monitoring_status="MONITORING_UNAVAILABLE",
            face_visible=False,
            grace_period_active=False,
            last_seen_ms=self._last_seen_ms,
            drowsiness_state=(
                frame_res.assessment.state.value if frame_res is not None else DrowsinessState.INSUFFICIENT_DATA.value
            ),
            alert_state=(
                frame_res.alert_state.value if frame_res is not None else AlertState.NORMAL.value
            ),
        )

    def _build_unavailable_frame(
        self,
        timestamp_ms: float,
        latency_ms: float,
        reason: str,
        state: AlertStateEnum = AlertStateEnum.MONITORING_UNAVAILABLE,
        risk_score: float = 0.0,
        monitoring_status: str = "MONITORING_UNAVAILABLE",
        face_visible: bool = False,
        grace_period_active: bool = False,
        last_seen_ms: Optional[float] = None,
        tracking_mode: Optional[str] = None,
        drowsiness_state: Optional[str] = None,
        alert_state: Optional[str] = None,
        operational_state: Optional[OperationalStateEnum] = None,
    ) -> FrameTelemetryDTO:
        """Construct fallback telemetry when face tracking or video decoding drops."""
        resolved_op_state = operational_state
        if resolved_op_state is None:
            if reason.startswith("LANDMARKER_EXCEPTION"):
                resolved_op_state = OperationalStateEnum.ERROR
            elif self.is_calibrating:
                resolved_op_state = OperationalStateEnum.CALIBRATING
            elif not self.feature_extractor.is_profile_feasible and reason not in (
                "IMAGE_DECODE_FAILED",
                "SESSION_CLOSED",
            ):
                resolved_op_state = OperationalStateEnum.CALIBRATION_REQUIRED
            else:
                resolved_op_state = OperationalStateEnum.DEGRADED

        return FrameTelemetryDTO(
            session_id=self.session_id,
            driver_id=self.driver_id,
            frame_index=self.frame_index,
            timestamp_ms=round(timestamp_ms, 1),
            fps=round(self.fps_ema, 1),
            face_detected=face_visible,
            face_visible=face_visible,
            grace_period_active=grace_period_active,
            last_seen_ms=last_seen_ms if last_seen_ms is not None else self._last_seen_ms,
            tracking_mode=tracking_mode or getattr(self.landmarker, "active_implementation", "DEGRADED_UNAVAILABLE"),
            state=state,
            drowsiness_state=drowsiness_state or DrowsinessState.INSUFFICIENT_DATA.value,
            alert_state=alert_state or self.alert_manager.current_state.value,
            is_recovering=self.alert_manager.is_recovering,
            authoritative_engine="DrowsinessEngine",
            operational_state=resolved_op_state,
            engine_type=EngineTypeEnum.PYTHON,
            monitoring_status=monitoring_status,
            risk_score=risk_score,
            pose_zone=PoseZoneEnum.UNKNOWN,
            reason_codes=[reason],
            head_pose=HeadPoseDTO(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0),
            eye_metrics=EyeMetricsDTO(
                ear_raw=0.0,
                ear_left=0.0,
                ear_right=0.0,
                ear_normalized=0.0,
                ear_closure_threshold=round(max(0.0, float(self.feature_extractor.active_closure_threshold)), 4),
                is_eye_closed=False,
                fused_closure_prob=0.0,
                perclos_pct=0.0,
                blink_duration_ms=None,
                blink_rcr=None,
                blink_rate_per_min=0.0,
                total_blink_count=int(
                    getattr(
                        self.feature_extractor.blink_detector,
                        "total_blink_count",
                        getattr(self.feature_extractor.blink_detector, "_total_blink_count", 0),
                    )
                ),
                mar_yawn=0.0,
                blink_state="UNCERTAIN_GAP",
                blink_classification="UNCERTAIN",
                perclos_coverage_pct=0.0,
                perclos_reliable=False,
                eye_closure_duration_ms=0.0,
                sampling_gap_detected=True,
            ),
            cascade=CascadeStatusDTO(
                experiment_mode=self.experiment_mode.value if hasattr(self.experiment_mode, "value") else str(self.experiment_mode),
                cnn_triggered=False,
                geom_open_prob=0.5,
                geom_reliability=0.0,
                cnn_open_prob=None,
                cnn_reliability=None,
                arbitration_winner="GEOMETRIC_AUTHORITATIVE",
                disagreement_flag=False,
                cnn_latency_ms=0.0,
                cnn_execution_mode="SHADOW_DISABLED",
                verifier_invoked=False,
                gate_trigger_reason=reason,
                onnx_inference_executed=False,
                cnn_inference_duration_ms=None,
            ),
            comparison=ConventionalFixedComparisonDTO(
                fixed_cutoff_ear=0.22,
                fixed_alert_state="UNAVAILABLE",
                fixed_false_alarm=False,
                personalised_zone="UNKNOWN",
                personalised_threshold=round(max(0.0, float(self.feature_extractor.active_closure_threshold)), 4),
                is_mirror_check_suppressed=False,
            ),
            engine_mode="CALIBRATING" if self.is_calibrating else "MONITORING",
            inference_latency_ms=round(latency_ms, 2),
        )

    def close(self) -> None:
        """Cleanly abort pending commands and release authoritative monitoring resources."""
        with self._process_lock:
            self._is_closed = True
            with self._command_lock:
                while self._pending_commands:
                    cmd = self._pending_commands.popleft()
                    if cmd.future and cmd.event_loop and not cmd.future.done():
                        cmd.event_loop.call_soon_threadsafe(
                            cmd.future.set_exception,
                            RuntimeError(f"Engine is closing; command '{cmd.action}' aborted."),
                        )
            self.monitoring_service.close()

    # --------------------------------------------------------------------------
    # Session Statistics & Diagnostics
    # --------------------------------------------------------------------------

    def get_session_summary(self) -> Dict[str, object]:
        """Return diagnostic metrics and summary for this monitoring session."""
        mean_lat = (
            float(self._lifetime_latency_sum_ms / self._lifetime_latency_count)
            if self._lifetime_latency_count > 0
            else 0.0
        )
        cnn_rate = (self.total_cnn_invocations / max(1, self.frame_index)) * 100.0
        summary: Dict[str, object] = {
            "session_id": self.session_id,
            "driver_id": self.driver_id,
            "owner_id": self.owner_id,
            "authoritative_engine": "DrowsinessEngine",
            "orchestrator": "LocalMonitoringService",
            "experiment_mode": self.experiment_mode.value,
            "total_frames_processed": self.frame_index,
            "mean_inference_latency_ms": round(mean_lat, 2),
            "cnn_trigger_rate_pct": round(cnn_rate, 2),
            "total_disagreements_arbitrated": self.total_disagreements,
            "mirror_check_false_alarms_prevented": self.total_mirror_check_suppressions,
            "conventional_fixed_false_alarms": self.total_fixed_false_alarms,
            "is_calibrated": self._profile.calibration_mode != "UNCALIBRATED_DEFAULT",
            "is_ocular_active": self.feature_extractor.is_profile_feasible,
            "active_calibration_mode": self._profile.calibration_mode,
            "current_alert_state": self.alert_manager.current_state.value,
            "cascade_diagnostics": {
                "mode": self.experiment_mode.value,
                "authority": "AUTHORITATIVE_DROWSINESS_ENGINE",
                "verifier_enabled": False,
            },
        }
        if self.landmarker is not None and hasattr(self.landmarker, "get_diagnostics"):
            summary["landmarker_diagnostics"] = self.landmarker.get_diagnostics()
        return summary
