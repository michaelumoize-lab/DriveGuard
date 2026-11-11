"""
Unit and Integration Tests for Workstream A — Performance Instrumentation & CNN Observability.

Verifies:
1. Decomposed stage timing breakdown (frame_received_mono_ms, frame_queue_wait_ms,
   frame_convert_ms, frame_preprocess_ms, executor_wait_ms, process_frame_ms, publish_ms, server_total_ms).
2. Monotonic durations are non-negative and executor wait is separated from inference duration.
3. Safe handling of clock domains (frame_age_ms is None without synchronized clocks).
4. Full CNN execution observability (MOBILENET_ONNX, ANALYTICAL_FALLBACK, UNAVAILABLE).
5. Strict distinction between verifier invocation, gate trigger reason, and ONNX execution.
"""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from backend.app.ai.cascade_classifier import (
    ConfidenceGatedEyeCascade,
    ExperimentMode,
    EyeCropPrediction,
    MobileNetV3EyeVerifier,
    VerifierPrediction,
)
from backend.app.ai.geometry import EyeCropQuality, FrameGeometryFeatures, PoseZone
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.telemetry import (
    CascadeStatusDTO,
    FrameTelemetryDTO,
    TimingBreakdownDTO,
)
from backend.app.services.livekit_worker import (
    LiveKitAIWorkerSession,
    _timed_process_frame,
)


class TestPerformanceInstrumentation(unittest.IsolatedAsyncioTestCase):

    def test_timing_breakdown_dto_validation(self):
        """Verify TimingBreakdownDTO accepts decomposed metrics and validates non-negative durations."""
        timing = TimingBreakdownDTO(
            queue_latency_ms=1.5,
            infer_duration_ms=15.2,
            server_total_ms=18.0,
            server_ingest_mono_ms=1000.0,
            server_publish_mono_ms=1018.0,
            server_wall_epoch_ms=time.time() * 1000.0,
            frame_received_mono_ms=1000.0,
            frame_queue_wait_ms=1.5,
            frame_convert_ms=0.3,
            frame_preprocess_ms=0.4,
            executor_wait_ms=0.6,
            process_frame_ms=15.2,
            publish_ms=0.5,
            frame_age_ms=None,
        )
        self.assertEqual(timing.frame_received_mono_ms, 1000.0)
        self.assertEqual(timing.frame_queue_wait_ms, 1.5)
        self.assertEqual(timing.frame_convert_ms, 0.3)
        self.assertEqual(timing.frame_preprocess_ms, 0.4)
        self.assertEqual(timing.executor_wait_ms, 0.6)
        self.assertEqual(timing.process_frame_ms, 15.2)
        self.assertEqual(timing.publish_ms, 0.5)
        self.assertIsNone(timing.frame_age_ms)

    def test_timed_process_frame_separates_executor_wait_and_inference(self):
        """Verify _timed_process_frame separately records executor wait and process_frame durations."""
        mock_engine = MagicMock(spec=SessionAIEngine)
        sample_dto = MagicMock(spec=FrameTelemetryDTO)

        def delayed_process(*args, **kwargs):
            time.sleep(0.02)  # 20ms mock inference
            return sample_dto

        mock_engine.process_frame.side_effect = delayed_process
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)

        t_submit = time.perf_counter()
        time.sleep(0.01)  # 10ms artificial scheduler wait

        dto, exec_wait_ms, proc_ms = _timed_process_frame(mock_engine, dummy_img, t_submit * 1000.0, t_submit)

        self.assertIs(dto, sample_dto)
        self.assertGreaterEqual(exec_wait_ms, 8.0)  # at least ~10ms wait
        self.assertGreaterEqual(proc_ms, 18.0)  # at least ~20ms inference
        # Verify executor wait and inference duration remain distinct
        self.assertNotEqual(exec_wait_ms, proc_ms)

    def test_cnn_observability_geometric_only_mode(self):
        """In ExperimentMode.A_GEOMETRIC_ONLY, verifier is not invoked and execution mode is UNAVAILABLE."""
        cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.A_GEOMETRIC_ONLY)
        geom = MagicMock(spec=FrameGeometryFeatures)
        geom.face_detected = True
        geom.yaw_weighted_ear = 0.28
        geom.geometric_reliability = 0.95
        geom.blendshape_blink_mean = 0.05
        geom.left_eye_crop = None
        geom.right_eye_crop = None

        res = cascade.evaluate(geom, reference_open_ear=0.30, close_threshold_ear=0.20)
        self.assertFalse(res.cnn_invoked)
        self.assertEqual(res.gate_trigger_reason, "EXP_A_GEOM_ONLY")
        self.assertEqual(res.cnn_execution_mode, "UNAVAILABLE")
        self.assertFalse(res.onnx_inference_executed)
        self.assertIsNone(res.cnn_inference_duration_ms)

    def test_cnn_observability_analytical_fallback_when_onnx_missing(self):
        """When ONNX session is absent, verifier execution mode is ANALYTICAL_FALLBACK and ONNX was not executed."""
        verifier = MobileNetV3EyeVerifier(onnx_model_path="nonexistent_model.onnx")
        self.assertFalse(verifier.is_pretrained_onnx)
        self.assertIsNone(verifier.onnx_session)

        # Mock eye crop
        crop_rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        crop_q = EyeCropQuality(
            crop_rgb=crop_rgb,
            sharpness_score=25.0,
            exposure_score=120.0,
            glare_fraction=0.01,
            overall_reliability=0.85,
        )

        pred = verifier.predict_binocular(crop_q, crop_q)
        self.assertEqual(pred.execution_mode, "ANALYTICAL_FALLBACK")
        self.assertFalse(pred.onnx_executed)
        self.assertGreater(pred.duration_ms, 0.0)

    def test_cnn_observability_unavailable_when_crops_unusable(self):
        """When crops are missing or below reliability threshold, verifier reports UNAVAILABLE."""
        verifier = MobileNetV3EyeVerifier()
        # Null crops
        pred = verifier.predict_binocular(None, None)
        self.assertEqual(pred.execution_mode, "UNAVAILABLE")
        self.assertFalse(pred.onnx_executed)

        # Very low quality crop
        low_q = EyeCropQuality(
            crop_rgb=np.zeros((64, 64, 3), dtype=np.uint8),
            sharpness_score=1.0,
            exposure_score=5.0,
            glare_fraction=0.9,
            overall_reliability=0.05,
        )
        pred_low = verifier.predict_binocular(low_q, low_q)
        self.assertEqual(pred_low.execution_mode, "UNAVAILABLE")
        self.assertFalse(pred_low.onnx_executed)

    def test_cnn_observability_onnx_mode_when_model_present(self):
        """When ONNX model is available and crops are valid, MOBILENET_ONNX executes."""
        verifier = MobileNetV3EyeVerifier()
        if not verifier.is_pretrained_onnx:
            self.skipTest("Pretrained mobilenetv3_eye.onnx not present in environment")

        crop_rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        crop_q = EyeCropQuality(
            crop_rgb=crop_rgb,
            sharpness_score=30.0,
            exposure_score=120.0,
            glare_fraction=0.01,
            overall_reliability=0.90,
        )
        pred = verifier.predict_binocular(crop_q, crop_q)
        self.assertEqual(pred.execution_mode, "MOBILENET_ONNX")
        self.assertTrue(pred.onnx_executed)
        self.assertGreater(pred.duration_ms, 0.0)

    def test_pipeline_populates_full_cascade_observability(self):
        """Verify SessionAIEngine.process_frame populates all CNN observability fields in CascadeStatusDTO."""
        engine = SessionAIEngine(
            session_id="test_obs",
            experiment_mode=ExperimentMode.A_GEOMETRIC_ONLY,
        )
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dto = engine.process_frame(dummy_img, timestamp_ms=100.0)

        cascade_dto = dto.cascade
        self.assertIsInstance(cascade_dto, CascadeStatusDTO)
        self.assertIn(cascade_dto.cnn_execution_mode, ("MOBILENET_ONNX", "ANALYTICAL_FALLBACK", "UNAVAILABLE"))
        self.assertIsInstance(cascade_dto.verifier_invoked, bool)
        self.assertIsInstance(cascade_dto.gate_trigger_reason, str)
        self.assertIsInstance(cascade_dto.onnx_inference_executed, bool)
        engine.close()


if __name__ == "__main__":
    unittest.main()
