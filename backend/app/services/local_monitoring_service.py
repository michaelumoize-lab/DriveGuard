"""
DriveGuard — Authoritative Local Monitoring Service (ARCH-1).

Encapsulates the complete local real-time driver drowsiness intelligence pipeline:
  FrameQualityAnalyzer
    -> DriveGuardFaceLandmarker (optional injection for testing/headless use)
    -> DriverFeatureExtractor (EAR, Head Pose, MAR/Yawn, Blink Kinematics, PERCLOS, PDBP)
    -> DrowsinessEngine (SignalConfidenceEvaluator -> TemporalAnalyzer -> DecisionEngine)
    -> AlertStateManager
    -> AlertDispatcher (AudioWarningEngine + AlertEventLogger)
    -> GuidedCalibrationSession + atomic profile commit/activation
    -> Optional non-authoritative passive EyeCNNShadowWorker

Design Contract:
  - All AI inference runs locally; this service exposes structured, processed telemetry
    (`ProcessedTelemetrySnapshot`) rather than transmitting raw video frames.
  - `scripts/run_camera_test.py` (diagnostic desktop HUD) and future local API/UI consumers
    delegate core monitoring, calibration, and alerting lifecycle here rather than duplicating
    pipeline wiring.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import numpy as np

from backend.app.features.head_pose import PoseZone
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    create_immutable_profile_snapshot,
)
from backend.app.ai.profile_adapter import dto_to_profile, profile_to_dto

if TYPE_CHECKING:
    from backend.app.ai.eye_shadow_observer import (
        BinocularShadowTelemetry,
        EyeCNNInvocationPolicy,
        EyeCNNShadowWorker,
    )
from backend.app.alerts import (
    AlertDispatcher,
    AlertState,
    AlertStateManager,
    AlertTransition,
    AudioSettingsStore,
    AudioWarningEngine,
    audio_settings_store,
)
from backend.app.alerts.event_logger import AlertEventLogger
from backend.app.camera.base import FramePacket
from backend.app.camera.frame_quality import (
    FrameQualityAnalyzer,
    FrameQualityResult,
    QualityStatus,
)
from backend.app.features import DriverFeatureExtractor, DriverFeatures
from backend.app.intelligence import DrowsinessAssessment, DrowsinessEngine
from backend.app.intelligence.reason_codes import OperationalState
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.frame_adapters import AdaptedFrame
from backend.app.services.profile_store import ProfileStore, profile_store
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkerConfig,
    FaceLandmarkResult,
)

logger = logging.getLogger("driveguard.services.local_monitoring")


# ---------------------------------------------------------------------------
# Authoritative Profile & Calibration Helpers
# ---------------------------------------------------------------------------

def commit_and_activate_profile(
    profile: DriverBehaviourProfile,
    feature_extractor: DriverFeatureExtractor,
    store: Optional[ProfileStore] = None,
) -> Tuple[bool, str]:
    """
    Coordinated transaction: validates feasibility, persists atomically to disk,
    and activates on the runtime feature extractor with automatic disk rollback on failure.
    If a valid profile already existed, rolls back to exact previous content; otherwise unlinks new file.
    Rejects any new commit while another profile transaction is pending safe-latch activation.
    """
    target_store = store or profile_store

    if getattr(feature_extractor, "has_pending_profile", False):
        return False, (
            f"Cannot commit profile for driver '{profile.driver_id}': "
            "another profile transaction is currently pending safe-latch activation."
        )

    is_feasible, reason = feature_extractor.validate_profile_feasibility(profile)
    if not is_feasible:
        return False, f"Infeasible profile: {reason}"

    try:
        snapshot = create_immutable_profile_snapshot(profile)
        dto = profile_to_dto(profile)
    except Exception as prep_err:
        return False, f"Profile preparation error: {prep_err}"

    previous_content: Optional[str] = None
    try:
        target_path = target_store._file_path(profile.driver_id)
        if target_path.is_file():
            previous_content = target_path.read_text(encoding="utf-8")
    except Exception as read_err:
        logger.warning("Could not read previous profile for '%s': %s", profile.driver_id, read_err)

    try:
        target_store.save_profile(dto)
    except Exception as save_err:
        return False, f"Persistence error: {save_err}"

    rollback_executed = False

    def do_rollback() -> None:
        nonlocal rollback_executed
        if not rollback_executed:
            rollback_executed = True
            target_store.rollback_new_profile(profile.driver_id, previous_content=previous_content)

    try:
        feature_extractor.set_pending_rollback_callback(
            do_rollback,
            fingerprint=snapshot.fingerprint,
        )
        activated = feature_extractor.set_profile(snapshot)
        if not activated:
            do_rollback()
            feature_extractor.set_pending_rollback_callback(None)
            return False, "Runtime activation rejected profile; disk persistence rolled back."
    except Exception as act_err:
        try:
            do_rollback()
        except Exception as rb_err:
            logger.critical("Critical failure during persistence rollback: %s", rb_err)
            feature_extractor.set_pending_rollback_callback(None)
            return False, f"Activation error: {act_err}; Rollback failed: {rb_err}"
        feature_extractor.set_pending_rollback_callback(None)
        return False, f"Runtime activation error: {act_err}; disk persistence rolled back."

    if getattr(feature_extractor, "has_pending_profile", False):
        return True, "Profile persisted and queued for deferred activation at next safe latch."

    return True, "Profile successfully persisted and activated."


def compute_calibration_geometric_reliability(
    landmark_result: Optional[FaceLandmarkResult],
    features: Optional[DriverFeatures],
    quality: Optional[FrameQualityResult],
) -> float:
    """
    Compute genuine per-frame geometric reliability in [0.0, 1.0] for GuidedCalibrationSession
    from optical frame quality, landmark confidence, 3D head-pose validity, and eye geometry.
    """
    if landmark_result is None or not landmark_result.has_face or features is None:
        return 0.0
    if quality is not None and not quality.is_viable_for_inference:
        return 0.0

    score = float(np.clip(landmark_result.presence_confidence, 0.0, 1.0))
    if quality is not None and quality.status not in (QualityStatus.EXCELLENT, QualityStatus.ACCEPTABLE):
        score *= 0.75
    if not features.head_pose.is_valid:
        score *= 0.25
    elif features.head_pose.rot_mat is None:
        score *= 0.60
    if not features.ear.is_geometric_valid:
        score *= 0.30
    return float(np.clip(score, 0.0, 1.0))


def load_profile_from_file(p_path: Path) -> DriverBehaviourProfile:
    """
    Load a DriverBehaviourProfile from either a DriverProfileDTO JSON file or a legacy
    DriverBehaviourProfile JSON file using Pydantic v2 `model_validate`.
    """
    with open(p_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "zone_baselines" in data and (
        "is_calibrated" in data or "schema_version" in data
    ):
        dto = DriverProfileDTO.model_validate(data)
        return dto_to_profile(dto)
    return DriverBehaviourProfile.load_json(p_path)


# ---------------------------------------------------------------------------
# Processed Telemetry & Service Output Contracts (No Raw Video Transmitted)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LocalMonitoringConfig:
    """Configuration for the local authoritative drowsiness monitoring service."""
    driver_id: str = "driver_default"
    profile_arg: Optional[str] = None
    is_mirrored: bool = True
    start_calibrating: bool = False
    calibration_step_duration_sec: float = 6.0
    cnn_shadow_enabled: bool = False
    cnn_shadow_hz: float = 15.0


@dataclass(frozen=True)
class ProcessedTelemetrySnapshot:
    """
    Processed, privacy-preserving per-frame telemetry snapshot containing zero raw image data.
    Suitable for local UI display, JSONL recording, or future local telemetry endpoints.
    """
    timestamp_ms: float
    face_detected: bool
    face_status: str
    presence_confidence: float
    pose_zone: str
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    rel_pitch_deg: float
    rel_yaw_deg: float
    rel_roll_deg: float
    ear_left: float
    ear_right: float
    ear_weighted: float
    ear_compensated: float
    ear_closure_threshold: float
    is_eye_closed: bool
    is_geometric_valid: bool
    mar: float
    is_yawning: bool
    total_blinks: int
    total_yawns: int
    perclos_pct: float
    perclos_is_reliable: bool
    perclos_coverage_pct: float
    drowsiness_state: str
    drowsiness_severity: int
    primary_reason: str
    alert_state: str
    alert_severity: int
    is_recovering: bool
    audio_warnings_enabled: bool
    luminance: float
    blur_variance: float
    optical_status: str
    inference_latency_ms: float
    calibrated_profile_id: Optional[str] = None
    operational_state: str = OperationalState.MONITORING_ACTIVE.value
    shadow_status: str = "DISABLED"
    shadow_is_authoritative: bool = False
    shadow_left_cnn: Optional[str] = None
    shadow_right_cnn: Optional[str] = None
    shadow_binoc_agreement: Optional[bool] = None
    shadow_latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalMonitoringFrameResult:
    """Complete result produced by processing one local camera frame."""
    quality: FrameQualityResult
    landmark: FaceLandmarkResult
    features: DriverFeatures
    assessment: DrowsinessAssessment
    alert_transition: Optional[AlertTransition]
    alert_state: AlertState
    is_recovering: bool
    active_reason_text: Optional[str]
    calibrated_profile_id: Optional[str]
    calibration_progress: Optional[Tuple[int, int, float, str, bool, str]]
    calibration_status_message: Optional[str]
    shadow_observation: Optional[BinocularShadowTelemetry]
    telemetry: ProcessedTelemetrySnapshot
    completed_calibration_profile: Optional[DriverBehaviourProfile] = None
    calibration_rejection_reason: Optional[str] = None
    operational_state: OperationalState = OperationalState.MONITORING_ACTIVE


class LocalMonitoringService:
    """
    Reusable authoritative orchestrator for the DriveGuard drowsiness monitoring pipeline.

    Coordinates frame quality analysis, facial landmark localization, feature extraction,
    personalized calibration, drowsiness intelligence, alerting, and processed telemetry
    for both local desktop/webcam operation (`scripts/run_camera_test.py`) and FastAPI
    API/WebSocket/LiveKit session adapters (`SessionAIEngine`).
    """

    def __init__(
        self,
        config: Optional[LocalMonitoringConfig] = None,
        store: Optional[ProfileStore] = None,
        landmarker: Optional[DriveGuardFaceLandmarker] = None,
        audio_engine: Optional[AudioWarningEngine] = None,
        settings_store: Optional[AudioSettingsStore] = None,
        event_logger: Optional[AlertEventLogger] = None,
        enable_landmarker: bool = True,
        enable_shadow_worker: bool = True,
        enable_audio: bool = True,
        enable_event_logging: bool = True,
        auto_load_profile: bool = True,
        persist_calibration_to_store: bool = True,
        initial_profile: Optional[DriverBehaviourProfile] = None,
        face_loss_grace_ms: float = 1000.0,
        landmarker_grace_timeout_sec: Optional[float] = None,
    ) -> None:
        self.config = config or LocalMonitoringConfig()
        self.store = store or profile_store
        self._persist_calibration_to_store = bool(persist_calibration_to_store)

        # 1. Resolve driver profile (explicit profile takes priority, else optional disk auto-load)
        if initial_profile is not None:
            profile_loaded: Optional[DriverBehaviourProfile] = initial_profile
        elif auto_load_profile:
            profile_loaded = self._load_initial_profile(self.config.profile_arg, self.config.driver_id)
        else:
            profile_loaded = None

        # 2. Core authoritative pipeline stages
        self.quality_analyzer = FrameQualityAnalyzer()
        if landmarker is not None:
            self.landmarker: Optional[DriveGuardFaceLandmarker] = landmarker
        elif enable_landmarker:
            lm_cfg = (
                FaceLandmarkerConfig(face_loss_grace_timeout_sec=float(landmarker_grace_timeout_sec))
                if landmarker_grace_timeout_sec is not None
                else None
            )
            self.landmarker = DriveGuardFaceLandmarker(config=lm_cfg)
        else:
            self.landmarker = None
        self._owns_landmarker = (landmarker is None and self.landmarker is not None)

        self.feature_extractor = DriverFeatureExtractor(
            is_mirrored=self.config.is_mirrored,
            profile=profile_loaded,
        )
        self.drowsiness_engine = DrowsinessEngine(
            face_loss_grace_ms=float(face_loss_grace_ms),
        )
        self.alert_manager = AlertStateManager()

        if enable_audio or enable_event_logging or audio_engine is not None or event_logger is not None:
            self._audio_engine: Optional[AudioWarningEngine] = (
                audio_engine
                if audio_engine is not None
                else (AudioWarningEngine() if enable_audio else None)
            )
            self.alert_dispatcher: Optional[AlertDispatcher] = AlertDispatcher(
                audio_engine=self._audio_engine,
                settings_store=settings_store or audio_settings_store,
                event_logger=event_logger if enable_event_logging else None,
            )
        else:
            self._audio_engine = None
            self.alert_dispatcher = None

        # 3. Optional non-authoritative passive Eye CNN shadow observer (lazy-imported only if enabled)
        self.shadow_policy: Optional[EyeCNNInvocationPolicy] = None
        self.shadow_worker: Optional[EyeCNNShadowWorker] = None
        if enable_shadow_worker:
            from backend.app.ai.eye_shadow_observer import (
                EyeCNNInvocationPolicy,
                EyeCNNShadowWorker,
            )

            self.shadow_policy = EyeCNNInvocationPolicy(
                enabled=self.config.cnn_shadow_enabled,
                max_frequency_hz=self.config.cnn_shadow_hz,
                max_staleness_ms=200.0,
            )
            self.shadow_worker = EyeCNNShadowWorker(policy=self.shadow_policy)
            self.shadow_worker.start()

        # 4. Guided calibration state
        self.calibration_session: Optional[GuidedCalibrationSession] = None
        if self.config.start_calibrating:
            self.start_calibration(is_mirrored=self.config.is_mirrored)

        self._latest_telemetry: Optional[ProcessedTelemetrySnapshot] = None
        self._frames_processed: int = 0

    def _load_initial_profile(
        self,
        profile_arg: Optional[str],
        driver_id: str,
    ) -> Optional[DriverBehaviourProfile]:
        profile_loaded: Optional[DriverBehaviourProfile] = None
        if profile_arg:
            p_path = Path(profile_arg)
            if p_path.is_file():
                try:
                    profile_loaded = load_profile_from_file(p_path)
                    logger.info(
                        "Loaded driver profile from file: %s (driver_id: %s)",
                        p_path,
                        profile_loaded.driver_id,
                    )
                except Exception as exc:
                    logger.warning("Could not parse profile file %s: %s", p_path, exc)
            if profile_loaded is None:
                profile_dto = self.store.get_profile(profile_arg)
                if profile_dto is not None:
                    try:
                        profile_loaded = dto_to_profile(profile_dto)
                        logger.info("Loaded driver profile '%s' from profile store.", profile_arg)
                    except Exception as exc:
                        logger.warning("Failed to convert profile DTO '%s': %s", profile_arg, exc)

        if profile_loaded is None and driver_id:
            profile_dto = self.store.get_profile(driver_id)
            if profile_dto is not None:
                try:
                    profile_loaded = dto_to_profile(profile_dto)
                    logger.info("Auto-loaded saved driver profile for '%s'.", driver_id)
                except Exception as exc:
                    logger.warning("Failed to auto-convert profile DTO for '%s': %s", driver_id, exc)

        return profile_loaded

    @property
    def calibrated_profile_id(self) -> Optional[str]:
        if self.feature_extractor.is_profile_feasible and self.feature_extractor.profile is not None:
            return self.feature_extractor.profile.driver_id
        return None

    @property
    def is_calibrating(self) -> bool:
        return self.calibration_session is not None and self.calibration_session.is_running

    def set_profile(self, profile: Optional[DriverBehaviourProfile]) -> bool:
        """
        Activate a DriverBehaviourProfile on the authoritative feature extractor.
        Uncalibrated default profiles safely deactivate ocular classification (HIGH-4).
        """
        if profile is None:
            return self.feature_extractor.set_profile(None)
        is_feasible, _ = self.feature_extractor.validate_profile_feasibility(profile)
        if not is_feasible:
            self.feature_extractor.set_profile(None)
            return False
        snapshot = create_immutable_profile_snapshot(profile)
        return self.feature_extractor.set_profile(snapshot)

    def start_calibration(
        self,
        timestamp_ms: Optional[float] = None,
        is_mirrored: Optional[bool] = None,
        four_zone_mode: bool = True,
        three_zone_mode: bool = False,
        step_duration_sec: Optional[float] = None,
        target_zone: Optional[str] = None,
        include_blink_baseline: bool = False,
        blink_baseline_duration_sec: float = 120.0,
    ) -> GuidedCalibrationSession:
        """Start or restart interactive guided calibration using current profile as reference if feasible."""
        mir = self.config.is_mirrored if is_mirrored is None else is_mirrored
        t_zone: Optional[PoseZone | str] = None
        is_4z = bool(four_zone_mode)
        is_3z = not is_4z and bool(three_zone_mode)
        if target_zone:
            is_4z, is_3z = False, False
            if str(target_zone) == "BLINK_CADENCE":
                t_zone = "BLINK_CADENCE"
            else:
                t_zone = PoseZone(target_zone) if isinstance(target_zone, str) else target_zone

        self.calibration_session = GuidedCalibrationSession(
            driver_id=self.config.driver_id,
            four_zone_mode=is_4z,
            three_zone_mode=is_3z,
            step_duration_sec=step_duration_sec or self.config.calibration_step_duration_sec,
            target_zone=t_zone,
            existing_profile=(
                self.feature_extractor.profile
                if self.feature_extractor.is_profile_feasible
                else None
            ),
            is_mirrored=mir,
            include_blink_baseline=include_blink_baseline,
            blink_baseline_duration_sec=blink_baseline_duration_sec,
        )
        self.calibration_session.start(timestamp_ms=timestamp_ms)
        logger.info("Started guided calibration for driver '%s' (mirrored=%s).", self.config.driver_id, mir)
        return self.calibration_session

    def skip_calibration_step(self, timestamp_ms: Optional[float] = None) -> Optional[DriverBehaviourProfile]:
        """Advance past the current calibration zone; returns completed profile if final step."""
        if self.calibration_session is None or not self.calibration_session.is_running:
            return None
        completed = self.calibration_session.skip_current_step(timestamp_ms=timestamp_ms)
        if completed is not None:
            if self._persist_calibration_to_store:
                commit_and_activate_profile(
                    profile=completed,
                    feature_extractor=self.feature_extractor,
                    store=self.store,
                )
            else:
                self.set_profile(completed)
            self.calibration_session = None
        return completed

    def cancel_calibration(self) -> None:
        """Cancel any active guided calibration session."""
        self.calibration_session = None

    def toggle_audio(self, timestamp_ms: float = 0.0) -> bool:
        """Toggle runtime audio warnings on/off."""
        if self.alert_dispatcher is None:
            return False
        return self.alert_dispatcher.toggle_audio(
            active_state=self.alert_manager.current_state,
            timestamp_ms=timestamp_ms,
        )

    def toggle_shadow(self) -> bool:
        """Toggle passive non-authoritative Eye CNN shadow observer."""
        if self.shadow_policy is None:
            return False
        self.shadow_policy.enabled = not self.shadow_policy.enabled
        return self.shadow_policy.enabled

    def get_latest_telemetry(self) -> Optional[ProcessedTelemetrySnapshot]:
        """Return the most recent processed telemetry snapshot (never contains raw video)."""
        return self._latest_telemetry

    def process_adapted_frame(
        self,
        adapted: AdaptedFrame,
        landmark_result_override: Optional[FaceLandmarkResult] = None,
    ) -> LocalMonitoringFrameResult:
        """Process a canonical `AdaptedFrame` produced by any input adapter."""
        return self.process_frame(
            frame_bgr=adapted.frame_bgr,
            timestamp_ms=adapted.timestamp_ms,
            frame_index=adapted.frame_index,
            is_mirrored=adapted.is_mirrored,
            frame_width=adapted.frame_width,
            frame_height=adapted.frame_height,
            packet=adapted.packet,
            landmark_result_override=landmark_result_override,
        )

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float,
        frame_index: int = 0,
        is_mirrored: Optional[bool] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        packet: Optional[FramePacket] = None,
        landmark_result_override: Optional[FaceLandmarkResult] = None,
    ) -> LocalMonitoringFrameResult:
        """
        Execute one full authoritative local pipeline step on a single camera frame.
        """
        self._frames_processed += 1
        mir = self.config.is_mirrored if is_mirrored is None else is_mirrored
        f_h = frame_height if (frame_height and frame_height > 0) else int(frame_bgr.shape[0])
        f_w = frame_width if (frame_width and frame_width > 0) else int(frame_bgr.shape[1])

        # 1. Optical frame quality analysis
        quality = self.quality_analyzer.analyze(frame_bgr)

        # 2. Facial landmark localization
        if landmark_result_override is not None:
            landmark = landmark_result_override
        elif self.landmarker is not None:
            landmark = self.landmarker.detect(packet if packet is not None else frame_bgr, timestamp_ms=int(timestamp_ms))
        else:
            raise RuntimeError("No FaceLandmarker configured and no landmark_result_override supplied.")

        # 3. Physiological feature extraction (EAR, Pose, MAR, Blink, PERCLOS)
        features = self.feature_extractor.extract(
            landmark,
            frame_width=f_w,
            frame_height=f_h,
            is_mirrored=mir,
        )

        # 4. Passive non-authoritative Eye CNN shadow observer
        shadow_obs: Optional[BinocularShadowTelemetry] = None
        if self.shadow_worker is not None:
            try:
                should_shadow, _ = self.shadow_policy.should_invoke(
                    timestamp_ms=timestamp_ms,
                    has_face=landmark.has_face,
                    ear_valid=features.ear.is_geometric_valid,
                )
                if should_shadow and landmark.has_face:
                    lms_xyz = np.array(
                        [[lm.x * f_w, lm.y * f_h, lm.z * f_w] for lm in landmark.landmarks],
                        dtype=np.float64,
                    )
                    self.shadow_worker.submit_frame(
                        frame_bgr=frame_bgr,
                        landmarks_pixel_xyz=lms_xyz,
                        timestamp_ms=timestamp_ms,
                        frame_id=frame_index or self._frames_processed,
                        geom_left_closed=features.ear.is_left_closed,
                        geom_right_closed=features.ear.is_right_closed,
                        raw_left_ear=features.ear.left_ear,
                        raw_right_ear=features.ear.right_ear,
                        ear_threshold=features.ear.closure_threshold,
                    )
                shadow_obs = self.shadow_worker.get_latest_observation(timestamp_ms)
            except Exception as shadow_exc:
                logger.warning("Error during eye shadow observer ingestion: %s", shadow_exc)

        # 5. Active guided calibration ingestion (uses raw uncompensated weighted_ear)
        calib_progress: Optional[Tuple[int, int, float, str, bool, str]] = None
        calib_status_msg: Optional[str] = None
        completed_calib_profile: Optional[DriverBehaviourProfile] = None
        calib_rejection_reason: Optional[str] = None
        if self.calibration_session is not None and self.calibration_session.is_running:
            calib_progress = self.calibration_session.get_progress(timestamp_ms)
            is_closed_for_calib = (
                features.ear.is_closed if self.feature_extractor.is_profile_feasible else None
            )
            geom_rel = compute_calibration_geometric_reliability(landmark, features, quality)
            completed_profile = self.calibration_session.ingest_frame(
                timestamp_ms=timestamp_ms,
                face_detected=bool(landmark.has_face and quality.is_viable_for_inference),
                geometric_reliability=geom_rel,
                ear_value=features.ear.weighted_ear,
                pitch_deg=features.head_pose.pitch_deg,
                yaw_deg=features.head_pose.yaw_deg,
                is_eye_currently_closed=is_closed_for_calib,
                completed_blink=features.blink.completed_blink,
                left_ear=features.ear.left_ear,
                right_ear=features.ear.right_ear,
                landmark_confidence=landmark.presence_confidence,
                rot_mat=features.head_pose.rot_mat,
                ear_geometric_valid=features.ear.is_geometric_valid,
            )
            if completed_profile is not None:
                completed_calib_profile = completed_profile
                if self._persist_calibration_to_store:
                    ok, status_msg = commit_and_activate_profile(
                        profile=completed_profile,
                        feature_extractor=self.feature_extractor,
                        store=self.store,
                    )
                else:
                    ok = self.set_profile(completed_profile)
                    status_msg = (
                        "Profile activated in memory."
                        if ok
                        else "Runtime activation rejected profile."
                    )
                if ok:
                    if self.feature_extractor.has_pending_profile:
                        calib_status_msg = (
                            f"CALIBRATION SAVED: '{completed_profile.driver_id}' pending safe activation "
                            "(after current blink)"
                        )
                    else:
                        c_ctr = self.feature_extractor.active_closure_threshold
                        r_ctr = self.feature_extractor.active_reopening_threshold
                        calib_status_msg = (
                            f"CALIBRATION COMPLETE: '{completed_profile.driver_id}' active! "
                            f"Thresholds: Center c={c_ctr:.3f}/r={r_ctr:.3f}"
                        )
                else:
                    calib_status_msg = f"CALIBRATION FAILED: {status_msg}"
                    calib_rejection_reason = status_msg
                self.calibration_session = None
            elif not self.calibration_session.is_running:
                reason = getattr(self.calibration_session, "rejection_reason", "Session terminated")
                calib_rejection_reason = reason
                calib_status_msg = f"CALIBRATION FAILED: {reason}"
                self.calibration_session = None

        # 6. Drowsiness intelligence evaluation with strict OperationalState calibration gate
        assessment = self.drowsiness_engine.evaluate(
            features=features,
            quality=quality,
            is_profile_calibrated=self.feature_extractor.is_profile_feasible,
            is_calibrating=self.is_calibrating,
        )

        # 7. Alert state manager & optional local dispatcher
        transition = self.alert_manager.update(assessment)
        if self.alert_dispatcher is not None:
            self.alert_dispatcher.dispatch(
                transition=transition,
                current_state=self.alert_manager.current_state,
                timestamp_ms=timestamp_ms,
                is_recovering=self.alert_manager.is_recovering,
            )

        active_profile_id = self.calibrated_profile_id
        active_reason_text = (
            str(self.alert_manager.active_reason)
            if self.alert_manager.active_reason is not None
            else None
        )

        # 8. Build processed telemetry snapshot (no raw video)
        telemetry = ProcessedTelemetrySnapshot(
            timestamp_ms=timestamp_ms,
            face_detected=landmark.has_face,
            face_status=landmark.status.value,
            presence_confidence=landmark.presence_confidence,
            pose_zone=features.head_pose.zone.value,
            pitch_deg=features.head_pose.pitch_deg,
            yaw_deg=features.head_pose.yaw_deg,
            roll_deg=features.head_pose.roll_deg,
            rel_pitch_deg=features.head_pose.rel_pitch_deg,
            rel_yaw_deg=features.head_pose.rel_yaw_deg,
            rel_roll_deg=features.head_pose.rel_roll_deg,
            ear_left=features.ear.left_ear,
            ear_right=features.ear.right_ear,
            ear_weighted=features.ear.weighted_ear,
            ear_compensated=features.ear.compensated_ear,
            ear_closure_threshold=features.ear.closure_threshold,
            is_eye_closed=features.ear.is_closed,
            is_geometric_valid=features.ear.is_geometric_valid,
            mar=features.mar.mar,
            is_yawning=bool(getattr(features.mar, "is_yawn_in_progress", False)),
            total_blinks=features.blink.total_blink_count,
            total_yawns=features.mar.total_yawn_count,
            perclos_pct=features.perclos.perclos_pct,
            perclos_is_reliable=features.perclos.is_reliable,
            perclos_coverage_pct=features.perclos.coverage_pct,
            drowsiness_state=assessment.state.value,
            drowsiness_severity=assessment.severity,
            primary_reason=active_reason_text or assessment.primary_reason_text,
            alert_state=self.alert_manager.current_state.value,
            alert_severity=self.alert_manager.severity,
            is_recovering=self.alert_manager.is_recovering,
            audio_warnings_enabled=(
                self.alert_dispatcher.is_audio_enabled if self.alert_dispatcher is not None else False
            ),
            luminance=quality.mean_luminance,
            blur_variance=quality.blur_variance,
            optical_status=quality.status.value,
            inference_latency_ms=landmark.inference_latency_ms,
            calibrated_profile_id=active_profile_id,
            operational_state=assessment.operational_state.value,
            shadow_status=shadow_obs.status_label if shadow_obs else "DISABLED",
            shadow_is_authoritative=False,
            shadow_left_cnn=shadow_obs.left_eye.cnn_predicted_class if shadow_obs else None,
            shadow_right_cnn=shadow_obs.right_eye.cnn_predicted_class if shadow_obs else None,
            shadow_binoc_agreement=shadow_obs.binocular_agreement if shadow_obs else None,
            shadow_latency_ms=shadow_obs.inference_latency_ms if shadow_obs else 0.0,
        )
        self._latest_telemetry = telemetry

        return LocalMonitoringFrameResult(
            quality=quality,
            landmark=landmark,
            features=features,
            assessment=assessment,
            alert_transition=transition,
            alert_state=self.alert_manager.current_state,
            is_recovering=self.alert_manager.is_recovering,
            active_reason_text=active_reason_text,
            calibrated_profile_id=active_profile_id,
            calibration_progress=calib_progress,
            calibration_status_message=calib_status_msg,
            shadow_observation=shadow_obs,
            telemetry=telemetry,
            completed_calibration_profile=completed_calib_profile,
            calibration_rejection_reason=calib_rejection_reason,
            operational_state=assessment.operational_state,
        )

    def close(self) -> None:
        """Cleanly stop background workers and release native resources."""
        if self.shadow_worker is not None:
            self.shadow_worker.stop()
        if self.alert_dispatcher is not None:
            self.alert_dispatcher.shutdown()
        if self._owns_landmarker and self.landmarker is not None:
            self.landmarker.close()
