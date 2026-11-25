"""
Phase 6 Verification Tests — Python RGB Passthrough, Sensitivity Presets,
Mirror Suppression Toggle, Production JWT Startup Guard, and CLI Fix.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest

from backend.app.ai.pdbp_calibration import DriverBehaviourProfile, ZoneBaseline
from backend.app.core.config import DEFAULT_DEV_AUTH_SECRET, Settings
from backend.app.features.feature_extractor import DriverFeatureExtractor
from backend.app.features.head_pose import PoseZone
from backend.app.intelligence.decision_engine import (
    SENSITIVITY_PRESETS,
    DecisionEngine,
    normalize_sensitivity_level,
)
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.intelligence.reason_codes import DrowsinessState, ReasonCode
from backend.app.services.frame_adapters import ArrayFrameAdapter
from backend.app.services.local_monitoring_service import (
    LocalMonitoringConfig,
    LocalMonitoringService,
)
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


def _make_calibrated_profile() -> DriverBehaviourProfile:
    identity = np.eye(3, dtype=np.float64)
    zones = {
        PoseZone.CENTER.value: ZoneBaseline(
            zone=PoseZone.CENTER.value,
            ear_median=0.285,
            ear_mad_scaled=0.015,
            pitch_median_deg=0.0,
            yaw_median_deg=0.0,
            sample_count=45,
        ),
        PoseZone.LEFT_MIRROR.value: ZoneBaseline(
            zone=PoseZone.LEFT_MIRROR.value,
            ear_median=0.275,
            ear_mad_scaled=0.015,
            pitch_median_deg=0.0,
            yaw_median_deg=24.0,
            sample_count=40,
        ),
        PoseZone.RIGHT_MIRROR.value: ZoneBaseline(
            zone=PoseZone.RIGHT_MIRROR.value,
            ear_median=0.275,
            ear_mad_scaled=0.015,
            pitch_median_deg=0.0,
            yaw_median_deg=-24.0,
            sample_count=40,
        ),
        PoseZone.DASHBOARD.value: ZoneBaseline(
            zone=PoseZone.DASHBOARD.value,
            ear_median=0.275,
            ear_mad_scaled=0.015,
            pitch_median_deg=-15.0,
            yaw_median_deg=0.0,
            sample_count=40,
        ),
    }
    return DriverBehaviourProfile(
        driver_id="phase6_test_driver",
        calibration_mode="FOUR_ZONE",
        center_ear_median=0.285,
        center_ear_mad_scaled=0.015,
        blink_duration_median_ms=235.0,
        blink_duration_mad_ms=30.0,
        blink_rcr_median=0.95,
        blink_rcr_mad=0.12,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.95,
        zone_baselines=zones,
        active_center_ear_median=0.285,
        neutral_rot_mat=identity,
        calibration_is_mirrored=False,
    )


def test_livekit_rgb_passthrough_zero_rgb2bgr_roundtrips() -> None:
    """
    Prove that processing an RGB frame via ArrayFrameAdapter.adapt_rgb ->
    LocalMonitoringService.process_adapted_frame -> DriveGuardFaceLandmarker.detect
    performs ZERO cv2.cvtColor(..., COLOR_RGB2BGR) and ZERO cv2.cvtColor(..., COLOR_BGR2RGB) calls.
    """
    rgb_frame = np.full((240, 320, 3), (120, 140, 160), dtype=np.uint8)
    real_cvtColor = cv2.cvtColor
    cvt_codes: list[int] = []

    def spy_cvtColor(src, code, *args, **kwargs):
        cvt_codes.append(code)
        return real_cvtColor(src, code, *args, **kwargs)

    # Create a headless DriveGuardFaceLandmarker instance with mocked MediaPipe backend
    landmarker = DriveGuardFaceLandmarker.__new__(DriveGuardFaceLandmarker)
    landmarker._config = MagicMock(
        min_face_presence_confidence=0.5,
        face_loss_grace_timeout_sec=0.4,
    )
    landmarker._lock = __import__("threading").Lock()
    landmarker._is_closed = False
    landmarker._last_timestamp_ms = None
    landmarker._duplicate_or_unordered_count = 0
    landmarker._last_face_seen_ms = None
    landmarker._last_face_seen_time = None
    landmarker._total_inferences = 0
    landmarker._total_face_hits = 0

    fake_mp = MagicMock()
    fake_mp.ImageFormat.SRGB = "SRGB"
    fake_mp.Image.side_effect = lambda image_format, data: (image_format, data)
    landmarker._mp = fake_mp

    fake_detector = MagicMock()
    fake_detector.detect_for_video.return_value = MagicMock(face_landmarks=[])
    landmarker._landmarker = fake_detector

    service = LocalMonitoringService(
        config=LocalMonitoringConfig(driver_id="rgb_test", is_mirrored=False, cnn_shadow_enabled=False),
        landmarker=landmarker,
        enable_landmarker=False,
        enable_shadow_worker=False,
        enable_audio=False,
        enable_event_logging=False,
        auto_load_profile=False,
        initial_profile=_make_calibrated_profile(),
    )

    with patch("cv2.cvtColor", side_effect=spy_cvtColor):
        adapted = ArrayFrameAdapter.adapt_rgb(rgb_frame, timestamp_ms=1000.0, is_mirrored=False)
        assert adapted.color_format == "RGB"
        service.process_adapted_frame(adapted)

    assert cv2.COLOR_RGB2BGR not in cvt_codes, f"Unexpected RGB->BGR conversion: {cvt_codes}"
    assert cv2.COLOR_BGR2RGB not in cvt_codes, f"Unexpected BGR->RGB conversion: {cvt_codes}"
    assert cv2.COLOR_RGB2GRAY in cvt_codes, "Expected native RGB->GRAY in FrameQualityAnalyzer"


def test_sensitivity_presets_monotonicity_and_safety_bounds() -> None:
    """
    Verify Section 2.4 monotonicity (HIGH < STANDARD < RELAXED), constant recovery
    cooldowns, safety bounds, and legacy HIGHWAY normalization.
    """
    high = SENSITIVITY_PRESETS["HIGH"]
    std = SENSITIVITY_PRESETS["STANDARD"]
    rel = SENSITIVITY_PRESETS["RELAXED"]

    for key in (
        "sustained_closure_threshold_ms",
        "extended_closure_threshold_ms",
        "perclos_attention_threshold_pct",
        "perclos_warning_threshold_pct",
        "severe_perclos_threshold_pct",
        "prolonged_blink_threshold_ms",
        "off_road_distraction_limit_ms",
    ):
        assert high[key] < std[key] < rel[key], f"Monotonicity violated for {key}"

    # Safety bounds
    for level_name, p in SENSITIVITY_PRESETS.items():
        assert 1000.0 <= p["sustained_closure_threshold_ms"] <= 2000.0, level_name
        assert 10.0 <= p["perclos_warning_threshold_pct"] <= 22.0, level_name
        assert p["recovery_critical_ms"] == 2000.0
        assert p["recovery_warning_ms"] == 1500.0
        assert p["recovery_attention_ms"] == 1000.0

    # Legacy HIGHWAY normalizes to STANDARD
    assert normalize_sensitivity_level("HIGHWAY") == "STANDARD"
    assert normalize_sensitivity_level("relaxed") == "RELAXED"
    assert normalize_sensitivity_level("HIGH") == "HIGH"
    assert normalize_sensitivity_level(None) == "STANDARD"


def test_sensitivity_feature_invariance_and_mirror_suppression_toggle() -> None:
    """
    Verify raw extracted features are identical (diff == 0.0) across HIGH/STANDARD/RELAXED,
    while DecisionEngine escalates monotonically, and mirror_suppression_enabled=False
    allows persistent distraction alerts during sustained mirror glances.
    """
    from scripts.validation.cross_language_parity_harness import create_face_landmarks

    profile = _make_calibrated_profile()
    closed_lms, _ = create_face_landmarks(ear_target=0.11, mar_target=0.15)
    open_lms, _ = create_face_landmarks(ear_target=0.285, mar_target=0.15)

    feature_traces: dict[str, list[tuple[float, float, float, float, float, float]]] = {}
    critical_onset_ms: dict[str, float | None] = {}

    for level in ("HIGH", "STANDARD", "RELAXED"):
        extractor = DriverFeatureExtractor(profile=profile)
        engine = DrowsinessEngine(sensitivity_level=level, mirror_suppression_enabled=True)
        trace = []
        first_crit = None
        for idx in range(23):
            ts = idx * 100
            raw = open_lms if idx < 2 else closed_lms
            lm_res = FaceLandmarkResult(
                has_face=True,
                landmarks=tuple(NormalizedLandmark(x=p["x"], y=p["y"], z=p["z"]) for p in raw),
                presence_confidence=0.95,
                timestamp_ms=ts,
                status=FaceLandmarkStatus.TRACKING,
                inference_latency_ms=1.5,
                reasons=("Tracking",),
            )
            feat = extractor.extract(lm_res, frame_width=640, frame_height=480, is_mirrored=False)
            res = engine.evaluate(feat, quality=None, is_profile_calibrated=True, is_calibrating=False)
            trace.append(
                (
                    feat.ear.compensated_ear,
                    feat.mar.mar,
                    feat.perclos.perclos_pct,
                    feat.head_pose.pitch_deg,
                    feat.head_pose.yaw_deg,
                    feat.head_pose.roll_deg,
                )
            )
            if first_crit is None and res.state == DrowsinessState.CRITICAL_WARNING:
                first_crit = float(ts)
        feature_traces[level] = trace
        critical_onset_ms[level] = first_crit

    assert feature_traces["HIGH"] == feature_traces["STANDARD"] == feature_traces["RELAXED"]
    assert critical_onset_ms["HIGH"] is not None
    assert critical_onset_ms["STANDARD"] is not None
    assert critical_onset_ms["RELAXED"] is not None
    assert critical_onset_ms["HIGH"] < critical_onset_ms["STANDARD"] < critical_onset_ms["RELAXED"]

    # Verify mirror_suppression_enabled=False fires PERSISTENT_DISTRACTION at 2600ms under HIGH
    mirror_lms, _ = create_face_landmarks(ear_target=0.275, yaw_deg=24.0)
    ext_on = DriverFeatureExtractor(profile=profile)
    eng_on = DrowsinessEngine(sensitivity_level="HIGH", mirror_suppression_enabled=True)
    ext_off = DriverFeatureExtractor(profile=profile)
    eng_off = DrowsinessEngine(sensitivity_level="HIGH", mirror_suppression_enabled=False)

    res_on = None
    res_off = None
    for idx in range(29):  # 200ms center + 2600ms left mirror
        ts = idx * 100
        raw = open_lms if idx < 2 else mirror_lms
        lm_res = FaceLandmarkResult(
            has_face=True,
            landmarks=tuple(NormalizedLandmark(x=p["x"], y=p["y"], z=p["z"]) for p in raw),
            presence_confidence=0.95,
            timestamp_ms=ts,
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=1.5,
            reasons=("Tracking",),
        )
        res_on = eng_on.evaluate(ext_on.extract(lm_res, 640, 480, is_mirrored=False), is_profile_calibrated=True)
        res_off = eng_off.evaluate(ext_off.extract(lm_res, 640, 480, is_mirrored=False), is_profile_calibrated=True)

    assert res_on is not None and res_on.state == DrowsinessState.NORMAL
    assert res_on.is_mirror_check_context is True
    assert res_off is not None and res_off.state == DrowsinessState.DROWSINESS_WARNING
    assert res_off.primary_reason is not None and res_off.primary_reason.code == ReasonCode.PERSISTENT_DISTRACTION
    assert res_off.is_mirror_check_context is False


def test_production_settings_rejects_default_auth_secret() -> None:
    """
    Production startup (ENV=production or RENDER=True) must refuse to boot with the
    default dev AUTH_SECRET or a short (<32-byte) secret.
    """
    with pytest.raises(ValueError, match="Production startup blocked"):
        Settings(ENV="production", AUTH_SECRET=DEFAULT_DEV_AUTH_SECRET, _env_file=None)

    with pytest.raises(ValueError, match="Production startup blocked"):
        Settings(RENDER=True, AUTH_SECRET="too-short", _env_file=None)

    # Strong non-default >=32-byte secret succeeds in production
    valid_prod = Settings(
        ENV="production",
        RENDER=True,
        AUTH_SECRET="prod-super-strong-unique-secret-key-at-least-32-chars",
        _env_file=None,
    )
    assert valid_prod.ENV == "production"

    # Dev mode continues to accept default dev secret
    valid_dev = Settings(ENV="development", RENDER=False, AUTH_SECRET=DEFAULT_DEV_AUTH_SECRET, _env_file=None)
    assert valid_dev.ENV == "development"


def test_run_camera_test_cli_shadow_worker_fix() -> None:
    """Verify scripts/run_camera_test.py uses monitoring_service.shadow_worker instead of dead locals() lookup."""
    cli_path = Path(__file__).resolve().parents[2] / "scripts" / "run_camera_test.py"
    src = cli_path.read_text(encoding="utf-8")
    assert '"shadow_worker" in locals()' not in src
    assert 'getattr(monitoring_service, "shadow_worker", None)' in src
