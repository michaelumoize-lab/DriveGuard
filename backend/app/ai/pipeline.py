"""
DriveGuard Session AI Engine — Unified Pipeline Orchestrator.

Encapsulates the complete real-time computer vision and adaptive inference pipeline:
  1. Image Ingestion (Decodes raw JPEG/PNG bytes or receives pre-decoded NumPy BGR frames).
  2. MediaPipe 478 3D Face Landmarking & Iris Tracking.
  3. 6DoF Head Pose Estimation & 4-Zone Pose Classification.
  4. 4-Zone Pose-Conditioned Guided Calibration Session Management.
  5. Sub-Frame Blink Kinematics (Duration, Reopening RCR, Rolling 60s PERCLOS).
  6. Confidence-Gated Eye Cascade with Reliability-Weighted Arbitration.
  7. Personalised Adaptive Risk Engine (4-state FSM, Explainable Reason Codes).
  8. Side-by-side Conventional Fixed Threshold Benchmark Comparison.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.app.ai.blink_kinematics import TemporalKinematicsTracker
from backend.app.ai.cascade_classifier import (
    ConfidenceGatedEyeCascade,
    ExperimentMode,
)
from backend.app.ai.geometry import (
    FrameGeometryFeatures,
    PoseZone,
)
from backend.app.ai.landmarker import DriveGuardFaceLandmarker
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    ZoneBaseline,
)
from backend.app.ai.risk_engine import (
    AlertState,
    PersonalisedAdaptiveRiskEngine,
)
from backend.app.schemas.calibration import CalibrationPhaseDTO
from backend.app.schemas.profile import DriverProfileDTO, ZoneBaselineDTO
from backend.app.schemas.telemetry import (
    AlertStateEnum,
    CascadeStatusDTO,
    ConventionalFixedComparisonDTO,
    EyeMetricsDTO,
    FrameTelemetryDTO,
    HeadPoseDTO,
    PoseZoneEnum,
)


class SessionAIEngine:
    """
    Stateful AI pipeline orchestrator managing one driver's real-time monitoring session.
    """

    def __init__(
        self,
        session_id: str = "session_default",
        driver_id: str = "driver_default",
        profile: Optional[DriverBehaviourProfile] = None,
        experiment_mode: ExperimentMode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
        onnx_model_path: Optional[str | Path] = None,
        landmarker: Optional[DriveGuardFaceLandmarker] = None,
    ) -> None:
        self.session_id = session_id
        self.driver_id = driver_id
        self.experiment_mode = experiment_mode

        # AI Subsystem Components
        self.landmarker = landmarker if landmarker is not None else DriveGuardFaceLandmarker()
        self.cascade = ConfidenceGatedEyeCascade(
            mode=self.experiment_mode, onnx_model_path=onnx_model_path
        )
        self.kinematics = TemporalKinematicsTracker()

        # Driver Profile & Risk Engine
        if profile is not None:
            self.profile = profile
        else:
            self.profile = DriverBehaviourProfile.create_default_uncalibrated(driver_id=driver_id)
        self.risk_engine = PersonalisedAdaptiveRiskEngine(profile=self.profile, fixed_ear_threshold=0.22)

        # Calibration State
        self.calibration_session: Optional[GuidedCalibrationSession] = None
        self._last_completed_calibration_dto: Optional[CalibrationPhaseDTO] = None

        # Telemetry & Performance Counters
        self.frame_index: int = 0
        self.prev_timestamp_ms: Optional[float] = None
        self.fps_ema: float = 30.0
        self.total_cnn_invocations: int = 0
        self.total_disagreements: int = 0
        self.total_mirror_check_suppressions: int = 0
        self.total_fixed_false_alarms: int = 0
        self.latencies_ms: List[float] = []
        self._process_lock = threading.Lock()
        self._is_closed = False

    # --------------------------------------------------------------------------
    # Calibration Lifecycle Management
    # --------------------------------------------------------------------------

    def start_calibration(
        self,
        four_zone_mode: Optional[bool] = None,
        three_zone_mode: Optional[bool] = None,
        zone_duration_sec: float = 5.0,
        target_zone: Optional[str] = None,
        timestamp_ms: Optional[float] = None,
    ) -> CalibrationPhaseDTO:
        """Begin a new guided calibration session (4-Zone default, 3-Zone, 1-Zone, or single-zone recalibration)."""
        t_zone = None
        if target_zone:
            is_4z, is_3z = False, False
            t_zone = PoseZone(target_zone) if isinstance(target_zone, str) else target_zone
        elif four_zone_mode is not None:
            is_4z = bool(four_zone_mode)
            is_3z = not is_4z and bool(three_zone_mode)
        else:
            # Default to full 4-Zone calibration
            is_4z, is_3z = True, False

        self.calibration_session = GuidedCalibrationSession(
            driver_id=self.driver_id,
            four_zone_mode=is_4z,
            three_zone_mode=is_3z,
            step_duration_sec=zone_duration_sec,
            target_zone=t_zone,
            existing_profile=self.profile if (t_zone is not None and self.profile.calibration_mode != "UNCALIBRATED_DEFAULT") else None,
        )
        self.calibration_session.start(timestamp_ms=timestamp_ms)
        self._last_completed_calibration_dto = None
        return self._build_calibration_dto(timestamp_ms=timestamp_ms)

    def cancel_calibration(self) -> None:
        """Cancel an ongoing calibration session."""
        self.calibration_session = None

    def skip_calibration_step(self, timestamp_ms: Optional[float] = None) -> Optional[CalibrationPhaseDTO]:
        """Manually advance past the current zone if tracking is difficult or user wishes to proceed."""
        if self.calibration_session is not None and self.calibration_session.is_running:
            new_profile = self.calibration_session.skip_current_step(timestamp_ms=timestamp_ms)
            if new_profile is not None:
                self.set_profile(new_profile)
                self._last_completed_calibration_dto = self._build_calibration_dto(timestamp_ms=timestamp_ms)
            return self._build_calibration_dto(timestamp_ms=timestamp_ms)
        return None

    @property
    def is_calibrating(self) -> bool:
        """Check whether the session is currently in active calibration mode."""
        return self.calibration_session is not None and self.calibration_session.is_running

    def get_calibration_status(self) -> Optional[CalibrationPhaseDTO]:
        """Return the current calibration progress or the last completed calibration DTO."""
        if self.calibration_session is not None:
            return self._build_calibration_dto()
        return self._last_completed_calibration_dto

    def _build_calibration_dto(self, timestamp_ms: Optional[float] = None) -> CalibrationPhaseDTO:
        """Construct the Pydantic CalibrationPhaseDTO from the internal session state."""
        if self.calibration_session is None or not self.calibration_session.is_running:
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
                rejection_reason=self.calibration_session.rejection_reason if self.calibration_session else None,
            )

        ts = timestamp_ms or (time.perf_counter() * 1000.0)
        curr_step_num, total_steps, ratio, instruction, is_aligned, feedback = self.calibration_session.get_progress(ts)
        phase_idx = max(1, curr_step_num)
        step = self.calibration_session.current_step
        zone_samples = len(self.calibration_session._ear_samples_by_zone.get(step.zone.value, []))
        total_pct = max(0.0, min(100.0, (((phase_idx - 1) + ratio) / max(1, total_steps)) * 100.0))
        zone_pct = max(0.0, min(100.0, ratio * 100.0))

        return CalibrationPhaseDTO(
            current_zone=step.zone.value,
            phase_index=phase_idx,
            total_phases=total_steps,
            zone_progress_pct=round(zone_pct, 1),
            total_progress_pct=round(total_pct, 1),
            stable_samples_count=zone_samples,
            target_samples_count=self.calibration_session.target_samples_per_zone,
            instruction=instruction,
            is_head_in_target_zone=is_aligned,
            stability_feedback=feedback,
            is_completed=self.calibration_session.is_completed,
            rejection_reason=self.calibration_session.rejection_reason,
        )

    # --------------------------------------------------------------------------
    # Profile Import / Export
    # --------------------------------------------------------------------------

    def set_profile(self, profile: DriverBehaviourProfile) -> None:
        """Update active personal baseline profile."""
        self.profile = profile
        self.risk_engine.set_profile(profile)

    def export_profile_dto(self) -> DriverProfileDTO:
        """Export the active driver profile as a validated Pydantic DTO."""
        zone_dtos: Dict[str, ZoneBaselineDTO] = {}
        for zk, zb in self.profile.zone_baselines.items():
            zk_name = zk.value if isinstance(zk, PoseZone) else str(zk)
            zone_dtos[zk_name] = ZoneBaselineDTO(
                zone=zk_name,
                ear_median=round(zb.ear_median, 4),
                ear_mad_scaled=round(zb.ear_mad_scaled, 4),
                dynamic_closure_threshold=round(zb.dynamic_closure_threshold, 4),
                pitch_median_deg=round(zb.pitch_median_deg, 2),
                yaw_median_deg=round(zb.yaw_median_deg, 2),
                sample_count=zb.sample_count,
                ear_left_median=round(zb.ear_left_median, 4) if zb.ear_left_median is not None else None,
                ear_right_median=round(zb.ear_right_median, 4) if zb.ear_right_median is not None else None,
                blendshape_open_median=round(zb.blendshape_open_median, 3) if zb.blendshape_open_median is not None else None,
                confidence_median=round(zb.confidence_median, 3) if zb.confidence_median is not None else None,
            )

        cal_time = getattr(self.profile, "calibrated_at", None)
        iso_time = cal_time.isoformat() if isinstance(cal_time, datetime) else datetime.now(timezone.utc).isoformat()

        return DriverProfileDTO(
            version=getattr(self.profile, "version", 1),
            driver_id=self.profile.driver_id,
            calibrated_at=iso_time,
            is_calibrated=self.profile.calibration_mode != "UNCALIBRATED_DEFAULT",
            calibration_mode=self.profile.calibration_mode,
            zone_baselines=zone_dtos,
            center_ear_median=round(self.profile.center_ear_median, 4),
            blink_duration_median_ms=round(self.profile.blink_duration_median_ms, 1),
            blink_duration_mad_ms=round(self.profile.blink_duration_mad_ms, 1),
            blink_rcr_median=round(self.profile.blink_rcr_median, 3),
            blink_rcr_mad=round(self.profile.blink_rcr_mad, 3),
            neutral_pitch_deg=round(self.profile.neutral_pitch_deg, 2),
            neutral_yaw_deg=round(self.profile.neutral_yaw_deg, 2),
            calibration_reliability=round(self.profile.calibration_reliability, 3),
        )

    def load_profile_dto(self, dto: DriverProfileDTO) -> None:
        """Load and activate a driver profile from a Pydantic DTO."""
        baselines: Dict[str, ZoneBaseline] = {}
        for zk_str, zb_dto in dto.zone_baselines.items():
            baselines[zk_str] = ZoneBaseline(
                zone=zk_str,
                ear_median=zb_dto.ear_median,
                ear_mad_scaled=zb_dto.ear_mad_scaled,
                pitch_median_deg=zb_dto.pitch_median_deg,
                yaw_median_deg=zb_dto.yaw_median_deg,
                sample_count=zb_dto.sample_count,
                ear_left_median=zb_dto.ear_left_median,
                ear_right_median=zb_dto.ear_right_median,
                blendshape_open_median=zb_dto.blendshape_open_median,
                confidence_median=zb_dto.confidence_median,
            )

        center_mad = (
            dto.zone_baselines["CENTER"].ear_mad_scaled
            if "CENTER" in dto.zone_baselines
            else 0.015
        )

        new_profile = DriverBehaviourProfile(
            driver_id=dto.driver_id,
            calibration_mode=dto.calibration_mode,
            center_ear_median=dto.center_ear_median,
            center_ear_mad_scaled=center_mad,
            blink_duration_median_ms=dto.blink_duration_median_ms,
            blink_duration_mad_ms=dto.blink_duration_mad_ms,
            blink_rcr_median=dto.blink_rcr_median,
            blink_rcr_mad=dto.blink_rcr_mad,
            blink_freq_median_per_min=18.0,
            blink_freq_mad_per_min=4.0,
            neutral_pitch_deg=dto.neutral_pitch_deg,
            neutral_yaw_deg=dto.neutral_yaw_deg,
            calibration_reliability=dto.calibration_reliability,
            zone_baselines=baselines,
            version=dto.version,
        )
        self.set_profile(new_profile)

    # --------------------------------------------------------------------------
    # Real-Time Frame Processing Orchestration
    # --------------------------------------------------------------------------

    def process_frame(
        self,
        frame_input: Union[np.ndarray, bytes],
        timestamp_ms: Optional[float] = None,
    ) -> FrameTelemetryDTO:
        """
        Execute the complete DriveGuard pipeline on a single video frame.

        Args:
            frame_input: Either a pre-decoded BGR image (NumPy array) or raw JPEG/PNG image bytes.
            timestamp_ms: Optional presentation timestamp in ms; if omitted, current clock is used.

        Returns:
            FrameTelemetryDTO containing full explainable metrics, risk score, and cascade decisions.
        """
        with self._process_lock:
            if self._is_closed:
                return self._build_unavailable_frame(
                    timestamp_ms if timestamp_ms is not None else (time.perf_counter() * 1000.0),
                    latency_ms=0.0,
                    reason="SESSION_CLOSED",
                )
            return self._process_frame_locked(frame_input, timestamp_ms)

    def _process_frame_locked(
        self,
        frame_input: Union[np.ndarray, bytes],
        timestamp_ms: Optional[float] = None,
    ) -> FrameTelemetryDTO:
        t_start = time.perf_counter()
        self.frame_index += 1

        # 1. Resolve timestamp & instantaneous FPS
        if timestamp_ms is None:
            ts_ms = t_start * 1000.0
        else:
            ts_ms = float(timestamp_ms)

        if self.prev_timestamp_ms is not None:
            dt_sec = max(1e-4, (ts_ms - self.prev_timestamp_ms) / 1000.0)
            self.fps_ema = 0.90 * self.fps_ema + 0.10 * (1.0 / dt_sec)
        self.prev_timestamp_ms = ts_ms

        # 2. Decode raw image bytes if necessary
        frame_bgr: Optional[np.ndarray] = None
        if isinstance(frame_input, bytes):
            nparr = np.frombuffer(frame_input, np.uint8)
            frame_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        elif isinstance(frame_input, np.ndarray):
            frame_bgr = frame_input

        if frame_bgr is None or frame_bgr.size == 0:
            return self._build_unavailable_frame(ts_ms, latency_ms=0.5, reason="IMAGE_DECODE_FAILED")

        # 3. MediaPipe Face Landmark & 3D Pose Extraction
        ref_ear = self.profile.center_ear_median
        geom, _ = self.landmarker.process_frame(
            frame_bgr=frame_bgr,
            timestamp_ms=ts_ms,
            neutral_pitch_deg=self.profile.neutral_pitch_deg,
            neutral_yaw_deg=self.profile.neutral_yaw_deg,
            reference_open_ear=ref_ear,
            extract_crops=True,
        )

        if not geom.face_detected:
            latency_ms = (time.perf_counter() - t_start) * 1000.0
            return self._build_unavailable_frame(ts_ms, latency_ms=latency_ms, reason="NO_FACE_DETECTED")

        # 4. Zone-Adaptive Baseline Selection
        zone_baseline = self.profile.get_zone_baseline(geom.pose_zone)

        # 5. Confidence-Gated Eye Cascade with Reliability-Weighted Arbitration
        cascade_res = self.cascade.evaluate(
            geom=geom,
            reference_open_ear=zone_baseline.ear_median,
            close_threshold_ear=zone_baseline.dynamic_closure_threshold,
        )
        if cascade_res.cnn_invoked:
            self.total_cnn_invocations += 1
        if cascade_res.disagreement_ear_vs_blendshape > 0.40:
            self.total_disagreements += 1

        # 6. Sub-Frame Blink Kinematics (Duration, RCR Velocity, Rolling 60s PERCLOS)
        kinematics_res = self.kinematics.update(
            timestamp_ms=ts_ms,
            ear_value=geom.yaw_weighted_ear,
            fused_closure_score=cascade_res.fused_closure_score,
            mar_value=geom.mar,
            ear_close_threshold=zone_baseline.dynamic_closure_threshold,
        )

        # 7. Calibration Step Ingestion (if session is actively calibrating)
        if self.calibration_session is not None and self.calibration_session.is_running:
            new_profile = self.calibration_session.ingest_frame(
                timestamp_ms=ts_ms,
                face_detected=geom.face_detected,
                geometric_reliability=geom.geometric_reliability,
                ear_value=geom.yaw_weighted_ear,
                pitch_deg=geom.pitch_deg,
                yaw_deg=geom.yaw_deg,
                is_eye_currently_closed=kinematics_res.is_eye_closed,
                completed_blink=kinematics_res.completed_blink,
                left_ear=geom.left_ear,
                right_ear=geom.right_ear,
                blendshape_blink_mean=geom.blendshape_blink_mean,
                landmark_confidence=geom.landmark_confidence,
            )
            if new_profile is not None:
                total_steps = len(self.calibration_session.steps)
                self.set_profile(new_profile)
                profile_dto = self.export_profile_dto()
                self._last_completed_calibration_dto = CalibrationPhaseDTO(
                    current_zone="CENTER",
                    phase_index=total_steps,
                    total_phases=total_steps,
                    zone_progress_pct=100.0,
                    total_progress_pct=100.0,
                    stable_samples_count=len(new_profile.zone_baselines),
                    target_samples_count=15,
                    instruction="Calibration successful! Personal baselines activated.",
                    is_head_in_target_zone=True,
                    stability_feedback="Profile Ready",
                    is_completed=True,
                    profile=profile_dto,
                )
                self.calibration_session = None
            elif not self.calibration_session.is_running and self.calibration_session.rejection_reason is not None:
                self._last_completed_calibration_dto = CalibrationPhaseDTO(
                    current_zone="CENTER",
                    phase_index=1,
                    total_phases=4,
                    zone_progress_pct=0.0,
                    total_progress_pct=0.0,
                    stable_samples_count=0,
                    target_samples_count=15,
                    instruction="Calibration rejected. Please ensure good lighting and face visibility.",
                    is_head_in_target_zone=False,
                    stability_feedback="Rejected",
                    is_completed=True,
                    rejection_reason=self.calibration_session.rejection_reason,
                )
                self.calibration_session = None

        # 8. Personalised Adaptive Risk Engine (FSM State, Reason Codes, Dual-Anchor Update)
        risk_res = self.risk_engine.evaluate(
            geom=geom,
            cascade=cascade_res,
            kinematics=kinematics_res,
        )
        is_mirror_suppressed = risk_res.deviations.is_mirror_check_context
        if is_mirror_suppressed:
            self.total_mirror_check_suppressions += 1

        # 9. Parallel Fixed-Threshold Comparison (Conventional EAR < 0.22)
        fixed_state_str = risk_res.conventional_fixed_state.value
        fixed_fa = (
            fixed_state_str != "NORMAL"
            and risk_res.state == AlertState.NORMAL
            and geom.pose_zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD)
        )
        if fixed_fa:
            self.total_fixed_false_alarms += 1

        latency_ms = (time.perf_counter() - t_start) * 1000.0
        self.latencies_ms.append(latency_ms)

        # 10. Assemble validated Pydantic DTO
        return FrameTelemetryDTO(
            session_id=self.session_id,
            driver_id=self.driver_id,
            frame_index=self.frame_index,
            timestamp_ms=round(ts_ms, 1),
            fps=round(self.fps_ema, 1),
            face_detected=True,
            state=AlertStateEnum(risk_res.state.value),
            risk_score=round(risk_res.risk_score, 3),
            pose_zone=PoseZoneEnum(geom.pose_zone.value),
            reason_codes=risk_res.reasons,
            head_pose=HeadPoseDTO(
                pitch_deg=round(geom.pitch_deg, 1),
                yaw_deg=round(geom.yaw_deg, 1),
                roll_deg=round(geom.roll_deg, 1),
            ),
            eye_metrics=EyeMetricsDTO(
                ear_raw=round(geom.yaw_weighted_ear, 4),
                ear_left=round(geom.left_ear, 4),
                ear_right=round(geom.right_ear, 4),
                ear_normalized=round(kinematics_res.smoothed_ear, 4),
                ear_closure_threshold=round(zone_baseline.dynamic_closure_threshold, 4),
                is_eye_closed=kinematics_res.is_eye_closed,
                fused_closure_prob=round(cascade_res.fused_closure_score, 3),
                perclos_pct=round(kinematics_res.perclos_60s_pct, 2),
                blink_duration_ms=round(kinematics_res.completed_blink.duration_ms, 1) if kinematics_res.completed_blink else None,
                blink_rcr=round(kinematics_res.completed_blink.rcr, 2) if kinematics_res.completed_blink else None,
                blink_rate_per_min=round(kinematics_res.blink_rate_per_min, 1),
                mar_yawn=round(geom.mar, 3),
            ),
            cascade=CascadeStatusDTO(
                experiment_mode=self.experiment_mode.value,
                cnn_triggered=cascade_res.cnn_invoked,
                geom_open_prob=round(1.0 - cascade_res.geom_closure_score, 3),
                geom_reliability=round(geom.geometric_reliability, 3),
                cnn_open_prob=round(cascade_res.cnn_open_prob, 3) if cascade_res.cnn_open_prob is not None else None,
                cnn_reliability=round(cascade_res.cnn_reliability, 3) if cascade_res.cnn_reliability is not None else None,
                arbitration_winner=cascade_res.winning_authority,
                disagreement_flag=cascade_res.cnn_invoked or cascade_res.disagreement_ear_vs_blendshape > 0.40,
                cnn_latency_ms=round(cascade_res.stage_latency_ms, 2),
            ),
            comparison=ConventionalFixedComparisonDTO(
                fixed_cutoff_ear=0.22,
                fixed_alert_state=fixed_state_str,
                fixed_false_alarm=fixed_fa,
                personalised_zone=geom.pose_zone.value,
                personalised_threshold=round(zone_baseline.dynamic_closure_threshold, 4),
                is_mirror_check_suppressed=is_mirror_suppressed,
            ),
            engine_mode="CALIBRATING" if self.is_calibrating else "MONITORING",
            inference_latency_ms=round(latency_ms, 2),
        )

    def _build_unavailable_frame(
        self, timestamp_ms: float, latency_ms: float, reason: str
    ) -> FrameTelemetryDTO:
        """Construct fallback telemetry when face tracking or video decoding drops."""
        return FrameTelemetryDTO(
            session_id=self.session_id,
            driver_id=self.driver_id,
            frame_index=self.frame_index,
            timestamp_ms=round(timestamp_ms, 1),
            fps=round(self.fps_ema, 1),
            face_detected=False,
            state=AlertStateEnum.MONITORING_UNAVAILABLE,
            risk_score=0.0,
            pose_zone=PoseZoneEnum.OUT_OF_BOUNDS,
            reason_codes=[reason],
            head_pose=HeadPoseDTO(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0),
            eye_metrics=EyeMetricsDTO(
                ear_raw=0.0,
                ear_left=0.0,
                ear_right=0.0,
                ear_normalized=0.0,
                ear_closure_threshold=0.20,
                is_eye_closed=False,
                fused_closure_prob=0.0,
                perclos_pct=0.0,
                blink_duration_ms=None,
                blink_rcr=None,
                blink_rate_per_min=0.0,
                mar_yawn=0.0,
            ),
            cascade=CascadeStatusDTO(
                experiment_mode=self.experiment_mode.value,
                cnn_triggered=False,
                geom_open_prob=0.5,
                geom_reliability=0.0,
                cnn_open_prob=None,
                cnn_reliability=None,
                arbitration_winner="NONE",
                disagreement_flag=False,
                cnn_latency_ms=0.0,
            ),
            comparison=ConventionalFixedComparisonDTO(
                fixed_cutoff_ear=0.22,
                fixed_alert_state="UNAVAILABLE",
                fixed_false_alarm=False,
                personalised_zone="OUT_OF_BOUNDS",
                personalised_threshold=0.20,
                is_mirror_check_suppressed=False,
            ),
            engine_mode="CALIBRATING" if self.is_calibrating else "MONITORING",
            inference_latency_ms=round(latency_ms, 2),
        )

    def close(self) -> None:
        """Cleanly release AI resources (MediaPipe FaceLandmarker, Cascade ONNX sessions)."""
        with self._process_lock:
            self._is_closed = True
            if self.landmarker is not None and hasattr(self.landmarker, "close"):
                try:
                    self.landmarker.close()
                except Exception:
                    pass

    # --------------------------------------------------------------------------
    # Session Statistics & Diagnostics
    # --------------------------------------------------------------------------

    def get_session_summary(self) -> Dict[str, object]:
        """Return diagnostic metrics and summary for this monitoring session."""
        mean_lat = float(np.mean(self.latencies_ms)) if self.latencies_ms else 0.0
        cnn_rate = (self.total_cnn_invocations / max(1, self.frame_index)) * 100.0
        summary: Dict[str, object] = {
            "session_id": self.session_id,
            "driver_id": self.driver_id,
            "total_frames_processed": self.frame_index,
            "mean_inference_latency_ms": round(mean_lat, 2),
            "cnn_trigger_rate_pct": round(cnn_rate, 2),
            "total_disagreements_arbitrated": self.total_disagreements,
            "mirror_check_false_alarms_prevented": self.total_mirror_check_suppressions,
            "conventional_fixed_false_alarms": self.total_fixed_false_alarms,
            "is_calibrated": self.profile.calibration_mode != "UNCALIBRATED_DEFAULT",
            "active_calibration_mode": self.profile.calibration_mode,
        }
        if hasattr(self.landmarker, "get_diagnostics"):
            summary["landmarker_diagnostics"] = self.landmarker.get_diagnostics()
        return summary
