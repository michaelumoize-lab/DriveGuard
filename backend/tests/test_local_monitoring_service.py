"""
DriveGuard — LocalMonitoringService & Phase 6 Architecture/Performance Tests (ARCH-1 & MEDIUM-10).

Verifies:
  - Reusable LocalMonitoringService orchestrates the authoritative local pipeline cleanly
  - ProcessedTelemetrySnapshot is JSON-serializable and exposes zero raw video frames
  - Calibration lifecycle via LocalMonitoringService persists, activates, and detects drowsiness
  - MEDIUM-10: EyeCNNShadowWorker uses bounded deque; HUD ROI blending renders cleanly without mutating input
"""

from __future__ import annotations

from collections import deque
import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from backend.app.ai.eye_shadow_observer import EyeCNNInvocationPolicy, EyeCNNShadowWorker
from backend.app.alerts.alert_state import AlertState
from backend.app.alerts.audio_settings import AudioSettingsStore
from backend.app.alerts.event_logger import AlertEventLogger
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringService,
    ProcessedTelemetrySnapshot,
)
from backend.app.services.profile_store import ProfileStore
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus
from backend.tests.test_end_to_end_authoritative_pipeline import (
    _make_sharp_frame,
    _make_synthetic_landmarks,
)
from scripts.run_camera_test import draw_diagnostic_hud


def _build_service(tmp_path: Path, driver_id: str = "svc_driver") -> LocalMonitoringService:
    mock_audio = MagicMock()
    mock_audio.enqueue_alert.return_value = True
    mock_audio.enqueue_recovery.return_value = True
    return LocalMonitoringService(
        config=LocalMonitoringConfig(
            driver_id=driver_id,
            is_mirrored=True,
            cnn_shadow_enabled=False,
        ),
        store=ProfileStore(storage_dir=tmp_path / "profiles"),
        audio_engine=mock_audio,
        settings_store=AudioSettingsStore(settings_file=tmp_path / "audio.json"),
        event_logger=AlertEventLogger(log_file=tmp_path / "events.jsonl", enable_file_logging=False),
        enable_landmarker=False,
        enable_shadow_worker=False,
    )


def _lm_result(
    ts_ms: float,
    ear: float = 0.30,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    is_mirrored: bool = True,
) -> FaceLandmarkResult:
    return FaceLandmarkResult(
        has_face=True,
        landmarks=_make_synthetic_landmarks(
            ear_target=ear,
            yaw_deg=yaw_deg,
            pitch_deg=pitch_deg,
            is_mirrored=is_mirrored,
        ),
        presence_confidence=0.96,
        timestamp_ms=int(ts_ms),
        status=FaceLandmarkStatus.TRACKING,
        inference_latency_ms=5.0,
        reasons=("OK",),
    )


def test_local_monitoring_service_calibration_detection_and_telemetry_contract(tmp_path: Path) -> None:
    svc = _build_service(tmp_path, driver_id="local_driver")
    frame = _make_sharp_frame()
    try:
        # 1. Start uncalibrated -> ocular disabled, processed telemetry is strictly JSON-serializable (no raw video)
        res0 = svc.process_frame(
            frame_bgr=frame,
            timestamp_ms=1000.0,
            frame_width=1280,
            frame_height=720,
            landmark_result_override=_lm_result(1000.0, ear=0.30),
        )
        assert res0.calibrated_profile_id is None
        assert isinstance(res0.telemetry, ProcessedTelemetrySnapshot)
        telemetry_dict = res0.telemetry.to_dict()
        # Verify strict JSON serializability and absence of raw image fields
        encoded = json.dumps(telemetry_dict)
        assert "frame" not in telemetry_dict
        assert "image" not in telemetry_dict
        assert len(encoded) < 2500

        # 2. Run 4-zone guided calibration directly through LocalMonitoringService
        svc.start_calibration(timestamp_ms=2000.0, is_mirrored=True, four_zone_mode=True, step_duration_sec=5.0)
        assert svc.is_calibrating is True
        assert svc.calibration_session is not None
        svc.calibration_session.target_samples_per_zone = 22

        ts = 2000.0
        completed_msg = None
        for idx, (y_deg, p_deg, e_val) in enumerate(
            [(0.0, 0.0, 0.30), (20.0, 0.0, 0.28), (-20.0, 0.0, 0.28), (0.0, -11.0, 0.27)]
        ):
            if idx > 0:
                ts += 2100.0
            for _ in range(26):
                ts += 40.0
                r = svc.process_frame(
                    frame_bgr=frame,
                    timestamp_ms=ts,
                    frame_width=1280,
                    frame_height=720,
                    landmark_result_override=_lm_result(ts, ear=e_val, yaw_deg=y_deg, pitch_deg=p_deg),
                )
                if r.calibration_status_message is not None:
                    completed_msg = r.calibration_status_message
                    break
            if completed_msg is not None:
                break

        assert completed_msg is not None and "CALIBRATION COMPLETE" in completed_msg
        assert svc.calibrated_profile_id == "local_driver"
        assert svc.is_calibrating is False

        # 3. Sustained eye closure (>= 1600ms) escalates cleanly to CRITICAL
        svc.drowsiness_engine.reset()
        svc.alert_manager.reset()
        ts += 500.0
        for _ in range(5):
            ts += 33.0
            svc.process_frame(frame, ts, frame_width=1280, frame_height=720, landmark_result_override=_lm_result(ts, ear=0.30))

        last_res = None
        for _ in range(55):
            ts += 33.0
            last_res = svc.process_frame(
                frame, ts, frame_width=1280, frame_height=720, landmark_result_override=_lm_result(ts, ear=0.11)
            )

        assert last_res is not None
        assert last_res.alert_state == AlertState.CRITICAL
        assert last_res.telemetry.alert_state == "CRITICAL"
        assert last_res.telemetry.calibrated_profile_id == "local_driver"
        assert svc.get_latest_telemetry() == last_res.telemetry
    finally:
        svc.close()


def test_medium10_bounded_shadow_timestamps_and_roi_hud_blending(tmp_path: Path) -> None:
    # 1. EyeCNNShadowWorker uses bounded deque for _completed_inference_timestamps_ms
    worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=False))
    assert isinstance(worker._completed_inference_timestamps_ms, deque)
    assert worker._completed_inference_timestamps_ms.maxlen == 600

    # 2. draw_diagnostic_hud blends all banners via ROI slices without mutating caller frame
    svc = _build_service(tmp_path)
    try:
        frame = _make_sharp_frame(width=960, height=540)
        original = frame.copy()
        res = svc.process_frame(
            frame_bgr=frame,
            timestamp_ms=1000.0,
            frame_width=960,
            frame_height=540,
            landmark_result_override=_lm_result(1000.0, ear=0.30),
        )
        hud = draw_diagnostic_hud(
            frame=frame,
            source_id="camera:0",
            actual_res=(960, 540),
            ingestion_fps=30.0,
            display_fps=30.0,
            frame_age_ms=8.0,
            buffer_stats={"total_dropped": 0},
            quality=res.quality,
            landmark_result=res.landmark,
            driver_features=res.features,
            assessment=res.assessment,
            calibration_banner="CALIBRATION COMPLETE: 'test' active!",
            calibrated_profile_name="test",
            alert_state=res.alert_state,
            is_recovering=res.is_recovering,
            active_reason_text=res.active_reason_text,
        )
        assert hud.shape == frame.shape
        assert np.array_equal(frame, original), "draw_diagnostic_hud must not mutate input frame"
        assert not np.array_equal(hud, original), "HUD overlay must render onto output canvas"
    finally:
        svc.close()
