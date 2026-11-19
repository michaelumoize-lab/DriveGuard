"""
Unit tests for Phase B1: Passive Eye CNN Shadow Integration.

Validates:
1. Crop-to-eye landmark association and labeling.
2. Preprocessing contract (64x64, luminance 3-channel replication, ImageNet norm, NCHW).
3. Class mapping (Class 0 = OPEN, Class 1 = CLOSED).
4. Missing model file / bad path graceful degradation.
5. Forward-pass exception handling (cv2.error / Exception safety).
6. Stale result rejection (>200 ms).
7. Bounded queue drop-oldest behavior (maxsize=1).
8. Worker startup and clean shutdown.
9. Conservative invocation policy (enable/disable, face presence, rate limiting).
10. Strict architectural isolation: DrowsinessAssessment, DecisionResult, and AlertState
    are completely unaffected by shadow worker state or predictions.
"""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest

from backend.app.ai.eye_shadow_observer import (
    DEFAULT_MOBILENETV3_WEIGHTS,
    BinocularShadowTelemetry,
    EyeCNNInvocationPolicy,
    EyeCNNShadowWorker,
    SingleEyeShadowTelemetry,
    preprocess_crop_for_mobilenetv3,
    IMAGENET_MEAN,
    IMAGENET_STD,
)
from backend.app.ai.geometry import (
    EyeCropQuality,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
)
from backend.app.alerts.alert_state_manager import AlertState, AlertStateManager
from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityResult,
    LightingCondition,
    QualityStatus,
)
from backend.tests.test_drowsiness_engine import make_driver_features
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.vision.face_landmarker import FaceLandmarkResult, FaceLandmarkStatus


def _create_synthetic_crop(val: int = 120) -> EyeCropQuality:
    """Create a synthetic 64x64 eye crop for testing."""
    crop = np.full((64, 64, 3), val, dtype=np.uint8)
    return EyeCropQuality(
        crop_rgb=crop,
        sharpness_score=0.85,
        exposure_score=0.90,
        glare_fraction=0.01,
        overall_reliability=0.85,
    )


class TestEyeCNNShadowPhaseB1:
    """Targeted unit test suite for Eye CNN Shadow Integration."""

    def test_crop_to_eye_association_indices(self):
        """Verify left and right eye indices match authoritative MediaPipe definitions."""
        assert LEFT_EYE_INDICES == (33, 160, 158, 133, 153, 144)
        assert RIGHT_EYE_INDICES == (362, 385, 387, 263, 373, 380)

    def test_preprocessing_contract(self):
        """Verify 64x64, luminance replication, ImageNet norm, and NCHW layout."""
        # Create a non-64x64 RGB input to test auto-resize and normalization
        raw_rgb = np.full((48, 72, 3), 150, dtype=np.uint8)
        tensor = preprocess_crop_for_mobilenetv3(raw_rgb)

        assert tensor.shape == (3, 64, 64)
        assert tensor.dtype == np.float32

        # Verify 3-channel replication: all 3 channels should be identical
        # except for channel-specific ImageNet mean/std standardization
        gray_val = float(cv2.cvtColor(np.full((1, 1, 3), 150, dtype=np.uint8), cv2.COLOR_RGB2GRAY)[0, 0])
        lum = gray_val / 255.0
        for ch in range(3):
            expected_val = (lum - IMAGENET_MEAN[ch]) / IMAGENET_STD[ch]
            np.testing.assert_allclose(tensor[ch, 0, 0], expected_val, rtol=1e-4)

    def test_class_mapping_open_closed(self):
        """Verify Class 0 maps to OPEN and Class 1 maps to CLOSED."""
        worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        if not worker.is_available:
            pytest.skip("mobilenetv3_eye.onnx not available in test environment")

        worker.start()
        try:
            # Mock the net forward return: batch of 2 where sample 0 has [10.0, -10.0] (OPEN)
            # and sample 1 has [-10.0, 10.0] (CLOSED)
            worker.cv2_net = MagicMock()
            worker.cv2_net.forward.return_value = np.array(
                [[10.0, -10.0], [-10.0, 10.0]], dtype=np.float32
            )

            worker.submit_crops(
                left_crop=_create_synthetic_crop(120),
                right_crop=_create_synthetic_crop(120),
                timestamp_ms=1000.0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=True,
                raw_left_ear=0.30,
                raw_right_ear=0.15,
                ear_threshold=0.20,
            )

            # Wait for worker to process
            time.sleep(0.1)
            telemetry = worker.get_latest_observation(1050.0)
            assert telemetry is not None
            assert telemetry.left_eye.cnn_predicted_class == "OPEN"
            assert telemetry.left_eye.cnn_open_prob > 0.99
            assert telemetry.left_eye.geometric_state == "OPEN"
            assert telemetry.left_eye.is_agreement is True

            assert telemetry.right_eye.cnn_predicted_class == "CLOSED"
            assert telemetry.right_eye.cnn_closed_prob > 0.99
            assert telemetry.right_eye.geometric_state == "CLOSED"
            assert telemetry.right_eye.is_agreement is True

            assert telemetry.binocular_agreement is True
        finally:
            worker.stop()

    def test_missing_model_graceful_degradation(self):
        """Verify worker handles nonexistent model path without crashing."""
        nonexistent = Path("nonexistent_path/weights/missing.onnx")
        worker = EyeCNNShadowWorker(model_path=nonexistent)

        assert worker.is_available is False
        assert worker.cv2_net is None
        assert "Model file not found" in str(worker.init_error)

        worker.start()
        try:
            submitted = worker.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=1000.0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.30,
                raw_right_ear=0.30,
                ear_threshold=0.20,
            )
            assert submitted is False
            assert worker.get_latest_observation(1000.0) is None
        finally:
            worker.stop()

    def test_forward_pass_exception_handling(self):
        """Verify model forward-pass exceptions (e.g. cv2.error) are safely caught."""
        worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        worker.is_available = True
        worker.cv2_net = MagicMock()
        worker.cv2_net.forward.side_effect = cv2.error("Simulated OpenCV DNN forward error")

        worker.start()
        try:
            worker.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=1000.0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.30,
                raw_right_ear=0.30,
                ear_threshold=0.20,
            )
            time.sleep(0.1)
            telemetry = worker.get_latest_observation(1050.0)
            assert telemetry is not None
            assert telemetry.status_label == "ERROR"
            assert telemetry.left_eye.cnn_availability == "ERROR"
            assert telemetry.left_eye.cnn_predicted_class is None
            assert telemetry.binocular_agreement is None
        finally:
            worker.stop()

    def test_stale_result_rejection(self):
        """Verify observations older than max_staleness_ms (200ms) are marked STALE."""
        worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True, max_staleness_ms=200.0))
        worker.is_available = True
        worker.cv2_net = MagicMock()
        worker.cv2_net.forward.return_value = np.array([[2.0, -2.0], [2.0, -2.0]], dtype=np.float32)

        worker.start()
        try:
            worker.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=1000.0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.30,
                raw_right_ear=0.30,
                ear_threshold=0.20,
            )
            time.sleep(0.08)

            # Fresh observation within 150 ms
            fresh = worker.get_latest_observation(1150.0)
            assert fresh is not None
            assert fresh.is_stale is False
            assert fresh.status_label == "ACTIVE"

            # Stale observation at 1205 ms (205 ms elapsed > 200 ms)
            stale = worker.get_latest_observation(1205.0)
            assert stale is not None
            assert stale.is_stale is True
            assert stale.status_label == "STALE"
        finally:
            worker.stop()

    def test_bounded_queue_drop_oldest(self):
        """Verify worker queue never exceeds maxsize=1 and drops oldest jobs."""
        worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        worker.is_available = True
        worker.cv2_net = MagicMock()
        # Make forward pass sleep to simulate high workload
        def _slow_forward(*args, **kwargs):
            time.sleep(0.05)
            return np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)

        worker.cv2_net.forward.side_effect = _slow_forward

        worker.start()
        try:
            # Submit several jobs in rapid succession
            for f_id in range(5):
                worker.submit_crops(
                    left_crop=_create_synthetic_crop(),
                    right_crop=_create_synthetic_crop(),
                    timestamp_ms=1000.0 + f_id * 10,
                    frame_id=f_id,
                    geom_left_closed=False,
                    geom_right_closed=False,
                    raw_left_ear=0.30,
                    raw_right_ear=0.30,
                    ear_threshold=0.20,
                )
            # Queue size should never exceed 1
            assert worker._queue.qsize() <= 1
            assert worker._dropped_jobs_count > 0
        finally:
            worker.stop()

    def test_clean_worker_start_stop(self):
        """Verify worker starts and stops cleanly without hanging."""
        worker = EyeCNNShadowWorker()
        assert not worker.is_running
        worker.start()
        assert worker.is_running
        worker.stop()
        assert not worker.is_running

    def test_invocation_policy_rate_limiting_and_gating(self):
        """Verify invocation policy respects enabled flag, face detection, and rate limits."""
        policy = EyeCNNInvocationPolicy(
            enabled=False,
            max_frequency_hz=15.0,  # Min interval = 66.67 ms
        )

        # 1. Disabled
        should_run, reason = policy.should_invoke(timestamp_ms=1000.0, has_face=True, ear_valid=True)
        assert should_run is False
        assert reason == "DISABLED"

        policy.enabled = True

        # 2. No face
        should_run, reason = policy.should_invoke(timestamp_ms=1000.0, has_face=False, ear_valid=True)
        assert should_run is False
        assert reason == "NO_FACE"

        # 3. Invalid EAR
        should_run, reason = policy.should_invoke(timestamp_ms=1000.0, has_face=True, ear_valid=False)
        assert should_run is False
        assert reason == "GEOMETRIC_EAR_INVALID"

        # 4. Valid first run
        should_run, reason = policy.should_invoke(timestamp_ms=1000.0, has_face=True, ear_valid=True)
        assert should_run is True
        assert reason is None

        # 5. Rate limited within 30 ms (< 66.67 ms interval)
        should_run, reason = policy.should_invoke(timestamp_ms=1030.0, has_face=True, ear_valid=True)
        assert should_run is False
        assert reason == "RATE_LIMITED"

        # 6. Valid after interval elapsed (70 ms > 66.67 ms)
        should_run, reason = policy.should_invoke(timestamp_ms=1070.0, has_face=True, ear_valid=True)
        assert should_run is True
        assert reason is None

    def test_strict_isolation_from_drowsiness_assessment(self):
        """
        Verify that DrowsinessEngine evaluation and AlertStateManager state transitions
        are completely unaffected by the shadow worker's presence, predictions, or errors.
        """
        drowsiness_engine = DrowsinessEngine()
        alert_manager = AlertStateManager()

        # Build dummy driver features representing a nominal alert driver
        features = make_driver_features(
            timestamp_ms=1000.0,
            has_face=True,
            is_closed=False,
            ear_val=0.28,
        )

        quality = FrameQualityResult(
            mean_luminance=120.0,
            blur_variance=250.0,
            lighting=LightingCondition.OPTIMAL,
            blur=BlurCondition.SHARP,
            status=QualityStatus.EXCELLENT,
            is_viable_for_inference=True,
            reasons=(),
        )

        # Baseline evaluation with no shadow worker
        baseline_assessment = drowsiness_engine.evaluate(features=features, quality=quality)
        baseline_transition = alert_manager.update(baseline_assessment)

        # Scenario A: Shadow worker active with all CLOSED predictions (contradicting geometry)
        worker_closed = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        worker_closed.is_available = True
        worker_closed.cv2_net = MagicMock()
        worker_closed.cv2_net.forward.return_value = np.array([[-10.0, 10.0], [-10.0, 10.0]], dtype=np.float32)
        worker_closed.start()

        try:
            worker_closed.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=1000.0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.28,
                raw_right_ear=0.28,
                ear_threshold=0.20,
            )
            time.sleep(0.08)

            # Re-evaluate DrowsinessEngine & AlertStateManager
            test_engine_a = DrowsinessEngine()
            test_alert_a = AlertStateManager()
            assessment_a = test_engine_a.evaluate(features=features, quality=quality)
            transition_a = test_alert_a.update(assessment_a)

            # Assert strict bit-for-bit equivalence with baseline
            assert assessment_a.state == baseline_assessment.state
            assert assessment_a.confidence.overall_confidence == baseline_assessment.confidence.overall_confidence
            assert assessment_a.primary_reason_text == baseline_assessment.primary_reason_text
            assert test_alert_a.current_state == alert_manager.current_state
            assert transition_a == baseline_transition
        finally:
            worker_closed.stop()

        # Scenario B: Shadow worker throws OpenCV runtime error
        worker_err = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        worker_err.is_available = True
        worker_err.cv2_net = MagicMock()
        worker_err.cv2_net.forward.side_effect = cv2.error("Fatal DNN crash")
        worker_err.start()

        try:
            worker_err.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=1000.0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.28,
                raw_right_ear=0.28,
                ear_threshold=0.20,
            )
            time.sleep(0.08)

            test_engine_b = DrowsinessEngine()
            test_alert_b = AlertStateManager()
            assessment_b = test_engine_b.evaluate(features=features, quality=quality)
            transition_b = test_alert_b.update(assessment_b)

            assert assessment_b.state == baseline_assessment.state
            assert assessment_b.confidence.overall_confidence == baseline_assessment.confidence.overall_confidence
            assert assessment_b.primary_reason_text == baseline_assessment.primary_reason_text
            assert test_alert_b.current_state == alert_manager.current_state
            assert transition_b == baseline_transition
        finally:
            worker_err.stop()

    def test_submit_frame_with_frame_packet_metadata(self):
        """Verify submitting full frames using FramePacket attribute contract."""
        from backend.app.camera.base import FramePacket

        worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        worker.is_available = True
        worker.cv2_net = MagicMock()
        worker.cv2_net.forward.return_value = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        worker.start()

        try:
            raw_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            packet = FramePacket(
                frame=raw_frame,
                timestamp_ns=1_000_000_000,
                frame_index=42,
                source_id="test_cam",
            )
            # Create synthetic landmarks covering 478 points
            lms_xyz = np.ones((478, 3), dtype=np.float64) * 200.0

            submitted = worker.submit_frame(
                frame_bgr=packet.frame,
                landmarks_pixel_xyz=lms_xyz,
                timestamp_ms=packet.timestamp_ms,
                frame_id=getattr(packet, "frame_index", 0),
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.28,
                raw_right_ear=0.28,
                ear_threshold=0.20,
            )
            assert submitted is True
            time.sleep(0.08)
            obs = worker.get_latest_observation(packet.timestamp_ms + 10)
            assert obs is not None
            assert obs.frame_id == 42
        finally:
            worker.stop()

    def test_clock_domain_consistency_and_monotonic_latency(self):
        """
        Phase B1.1.1 Regression Test:
        Verify capture_timestamp_ms and processed_timestamp_ms share the same monotonic
        timebase, avoiding catastrophic ~1.795e12 ms latency from mixing time.time() with monotonic.
        """
        worker = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
        worker.is_available = True
        worker.cv2_net = MagicMock()
        worker.cv2_net.forward.return_value = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        worker.start()

        try:
            mono_now_ms = time.monotonic() * 1000.0
            worker.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=mono_now_ms,
                frame_id=101,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.28,
                raw_right_ear=0.28,
                ear_threshold=0.20,
            )
            time.sleep(0.06)

            query_mono_ms = time.monotonic() * 1000.0
            obs = worker.get_latest_observation(query_mono_ms)
            assert obs is not None

            # 1. capture_timestamp_ms must equal submitted monotonic timestamp
            assert obs.capture_timestamp_ms == pytest.approx(mono_now_ms, abs=1e-3)

            # 2. processed_timestamp_ms must be >= capture_timestamp_ms
            assert obs.processed_timestamp_ms >= obs.capture_timestamp_ms

            # 3. total_latency_ms must represent true elapsed time (< 500 ms, NOT ~1.795e12 ms)
            assert 0.0 <= obs.total_latency_ms < 500.0

            # 4. Monotonic consistency between processed timestamp and query time
            assert obs.processed_timestamp_ms <= query_mono_ms + 50.0
        finally:
            worker.stop()

    def test_pipeline_diagnostics_and_staleness_counters(self):
        """
        Phase B1.1.1 Test:
        Verify full pipeline diagnostic funnel and distinct consumer query staleness counts.
        """
        policy = EyeCNNInvocationPolicy(enabled=True, max_frequency_hz=10.0, max_staleness_ms=100.0)
        worker = EyeCNNShadowWorker(policy=policy)
        worker.is_available = True
        worker.cv2_net = MagicMock()
        worker.cv2_net.forward.return_value = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        worker.start()

        try:
            # Query before any submission -> queries_none
            none_obs = worker.get_latest_observation(time.monotonic() * 1000.0)
            assert none_obs is None

            # Policy rejections:
            # 1. No face
            policy.should_invoke(timestamp_ms=100.0, has_face=False)
            # 2. Accepted
            policy.should_invoke(timestamp_ms=100.0, has_face=True)
            # 3. Rate limited (too close in time)
            policy.should_invoke(timestamp_ms=120.0, has_face=True)

            t0 = time.monotonic() * 1000.0
            worker.submit_crops(
                left_crop=_create_synthetic_crop(),
                right_crop=_create_synthetic_crop(),
                timestamp_ms=t0,
                frame_id=1,
                geom_left_closed=False,
                geom_right_closed=False,
                raw_left_ear=0.28,
                raw_right_ear=0.28,
                ear_threshold=0.20,
            )
            time.sleep(0.06)

            # Query fresh (within 100 ms)
            fresh_obs = worker.get_latest_observation(t0 + 50.0)
            assert fresh_obs is not None
            assert fresh_obs.is_stale is False

            # Query stale (after 100 ms)
            stale_obs = worker.get_latest_observation(t0 + 250.0)
            assert stale_obs is not None
            assert stale_obs.is_stale is True

            diag = worker.get_pipeline_diagnostics()
            assert diag["evaluations_total"] == 3
            assert diag["eligible_opportunities"] == 2  # accepted (1) + rate_limited (1)
            assert diag["evaluations_accepted"] == 1
            assert diag["rejections_no_face"] == 1
            assert diag["rejections_rate_limited"] == 1
            assert diag["jobs_submitted"] == 1
            assert diag["jobs_processed"] == 1
            assert diag["inferences_successful"] == 1
            assert diag["queries_total"] == 3
            assert diag["queries_none"] == 1
            assert diag["queries_fresh"] == 1
            assert diag["queries_stale"] == 1
        finally:
            worker.stop()


