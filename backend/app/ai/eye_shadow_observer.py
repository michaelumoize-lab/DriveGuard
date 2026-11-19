"""
Phase B1: Passive Eye CNN Shadow Integration Observer and Asynchronous Worker.

Architecture & Invariants:
1. Passive Diagnostic Observer:
   The authoritative geometric detection pipeline (EAR, BlinkDetector, PERCLOS,
   TemporalAnalyzer, DecisionEngine, DrowsinessEngine) remains the SOLE authority
   for drowsiness assessment and alerts. CNN predictions must NOT modify or feed into
   any safety-critical alert paths.
2. Thread Safety & Decoupled Latency:
   The CNN worker runs asynchronously on a dedicated daemon thread with a bounded
   queue (maxsize=1) using drop-oldest / newest-frame preference to guarantee that
   the 30 FPS camera loop is never blocked.
3. Strict Preprocessing Contract:
   - 64x64 resolution
   - Grayscale luminance extraction
   - 3-channel replication
   - Float32 scaling: [0.0, 1.0]
   - ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
   - NCHW tensor layout: (B, 3, 64, 64)
   - Class mapping: 0 = OPEN, 1 = CLOSED
4. Fault Tolerance:
   Catches all (cv2.error, Exception) during model execution, recording diagnostic
   error states without crashing the webcam or camera pipeline.
5. Temporal Staleness:
   Rejects observations older than 200 ms relative to the current camera timestamp.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
import logging
from pathlib import Path
import queue
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.ai.geometry import (
    EyeCropQuality,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    extract_eye_crop_with_quality,
)

logger = logging.getLogger(__name__)

DEFAULT_MOBILENETV3_WEIGHTS = (
    Path(__file__).resolve().parents[2] / "weights" / "mobilenetv3_eye.onnx"
)

# Verified ImageNet normalization parameters from training
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class SingleEyeShadowTelemetry:
    """Independent shadow diagnostic telemetry for a single eye."""

    eye_label: str  # "left" or "right"
    raw_ear: float
    ear_threshold: float
    geometric_state: str  # "OPEN", "CLOSED", "UNKNOWN"
    cnn_predicted_class: Optional[str] = None  # "OPEN", "CLOSED", or None
    cnn_open_prob: Optional[float] = None
    cnn_closed_prob: Optional[float] = None
    crop_reliability: float = 0.0
    crop_quality: Optional[EyeCropQuality] = None
    inference_latency_ms: float = 0.0
    cnn_availability: str = "UNAVAILABLE"  # "AVAILABLE", "QUALITY_REJECTED", "ERROR", "UNAVAILABLE", "STALE_REJECTED"
    is_agreement: Optional[bool] = None  # True if geom == cnn, False if mismatch, None if unavailable
    skip_reason: Optional[str] = None


@dataclass
class BinocularShadowTelemetry:
    """Binocular shadow diagnostic telemetry capturing both eyes and pipeline health."""

    frame_id: int
    capture_timestamp_ms: float
    processed_timestamp_ms: float
    left_eye: SingleEyeShadowTelemetry
    right_eye: SingleEyeShadowTelemetry
    binocular_geometric_state: str  # "OPEN", "CLOSED", "ASYMMETRIC", "UNKNOWN"
    binocular_cnn_state: Optional[str]  # "OPEN", "CLOSED", "ASYMMETRIC", None
    binocular_agreement: Optional[bool]  # True if both agree, False if either or both disagree, None if unavailable
    total_latency_ms: float  # Capture to completion latency
    inference_latency_ms: float  # Pure neural forward-pass duration
    worker_queue_depth: int
    is_stale: bool = False
    status_label: str = "ACTIVE"  # "ACTIVE", "DISABLED", "STALE", "UNAVAILABLE", "ERROR", "QUALITY_REJECTED"


@dataclass
class EyeCNNInvocationPolicy:
    """
    Conservative configurable invocation policy separating CNN execution
    decisions from the classification model itself.
    """

    enabled: bool = False
    max_frequency_hz: float = 15.0
    min_crop_quality: float = 0.15
    max_staleness_ms: float = 200.0
    _last_invocation_timestamp_ms: float = 0.0

    # Diagnostic Telemetry Counters
    evaluations_count: int = 0
    accepted_count: int = 0
    rejections_rate_limited: int = 0
    rejections_no_face: int = 0
    rejections_ear_invalid: int = 0
    rejections_disabled: int = 0

    def should_invoke(
        self,
        timestamp_ms: float,
        has_face: bool,
        ear_valid: bool = True,
    ) -> Tuple[bool, Optional[str]]:
        """
        Evaluate whether the shadow CNN worker should ingest the current frame.
        """
        self.evaluations_count += 1
        if not self.enabled:
            self.rejections_disabled += 1
            return False, "DISABLED"
        if not has_face:
            self.rejections_no_face += 1
            return False, "NO_FACE"
        if not ear_valid:
            self.rejections_ear_invalid += 1
            return False, "GEOMETRIC_EAR_INVALID"

        min_interval_ms = 1000.0 / max(1.0, self.max_frequency_hz)
        if (timestamp_ms - self._last_invocation_timestamp_ms) < min_interval_ms:
            self.rejections_rate_limited += 1
            return False, "RATE_LIMITED"

        self.accepted_count += 1
        self._last_invocation_timestamp_ms = timestamp_ms
        return True, None


def preprocess_crop_for_mobilenetv3(crop_rgb: np.ndarray) -> np.ndarray:
    """
    Apply verified MobileNetV3 preprocessing to a single RGB eye crop.
    
    Steps:
      1. Ensure 64x64 resolution.
      2. Convert RGB to grayscale (luminance).
      3. Replicate single luminance channel to 3 channels (bridges IR/webcam gap).
      4. Convert to float32 and scale to [0.0, 1.0].
      5. Standardize with ImageNet mean and std.
      6. Transpose from HWC to CHW: (3, 64, 64).
    """
    if crop_rgb.shape[0] != 64 or crop_rgb.shape[1] != 64:
        crop_rgb = cv2.resize(crop_rgb, (64, 64), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY)
    lum3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    normalized = (lum3 - IMAGENET_MEAN) / IMAGENET_STD
    tensor_chw = np.transpose(normalized, (2, 0, 1))
    return tensor_chw


class EyeCNNShadowWorker:
    """
    Bounded, asynchronous background worker executing MobileNetV3 eye classification
    in passive shadow mode.
    """

    def __init__(
        self,
        model_path: Optional[Path | str] = None,
        policy: Optional[EyeCNNInvocationPolicy] = None,
    ) -> None:
        self.model_path = Path(model_path) if model_path is not None else DEFAULT_MOBILENETV3_WEIGHTS
        self.policy = policy or EyeCNNInvocationPolicy()

        self.cv2_net: Optional[cv2.dnn.Net] = None
        self.is_available: bool = False
        self.init_error: Optional[str] = None

        self._queue: queue.Queue = queue.Queue(maxsize=1)
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        self._latest_telemetry: Optional[BinocularShadowTelemetry] = None
        self._jobs_submitted_count: int = 0
        self._jobs_processed_count: int = 0
        self._crop_quality_rejections_count: int = 0
        self._dropped_jobs_count: int = 0
        self._total_inferences_count: int = 0
        self._error_count: int = 0

        # Query staleness diagnostics
        self._queries_total_count: int = 0
        self._queries_fresh_count: int = 0
        self._queries_stale_count: int = 0
        self._queries_none_count: int = 0

        # Completed inference timestamps for inter-inference interval tracking
        self._completed_inference_timestamps_ms: List[float] = []

        self._initialize_model()

    def _initialize_model(self) -> None:
        """Load MobileNetV3 ONNX model using OpenCV DNN."""
        if not self.model_path.is_file():
            self.is_available = False
            self.init_error = f"Model file not found: {self.model_path}"
            logger.warning("EyeCNNShadowWorker unavailable: %s", self.init_error)
            return

        try:
            self.cv2_net = cv2.dnn.readNetFromONNX(str(self.model_path))
            self.is_available = True
            self.init_error = None
            logger.info("EyeCNNShadowWorker loaded ONNX model from %s", self.model_path)
        except Exception as exc:
            self.cv2_net = None
            self.is_available = False
            self.init_error = f"Failed to load ONNX model via cv2.dnn: {exc}"
            logger.error("EyeCNNShadowWorker initialization failed: %s", self.init_error)

    def start(self) -> None:
        """Start the background worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._worker_loop,
            name="DriveGuardEyeCNNShadowWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("EyeCNNShadowWorker thread started.")

    def stop(self, timeout_sec: float = 2.0) -> None:
        """Stop worker cleanly and join thread."""
        self._stop_event.set()
        # Wake worker loop if waiting on queue
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=timeout_sec)
        self._thread = None
        logger.info("EyeCNNShadowWorker stopped.")

    @property
    def is_running(self) -> bool:
        """Check if worker thread is active."""
        return self._thread is not None and self._thread.is_alive()

    def submit_frame(
        self,
        frame_bgr: np.ndarray,
        landmarks_pixel_xyz: np.ndarray,
        timestamp_ms: float,
        frame_id: int,
        geom_left_closed: Optional[bool],
        geom_right_closed: Optional[bool],
        raw_left_ear: float,
        raw_right_ear: float,
        ear_threshold: float,
    ) -> bool:
        """
        Submit a new frame job to the bounded worker queue.
        Implements drop-oldest preference to guarantee zero camera loop blocking.
        """
        if not self.is_available or not self.is_running:
            return False

        self._jobs_submitted_count += 1

        # Make a shallow frame copy to isolate worker from any downstream in-place landmark rendering
        job = {
            "type": "FRAME",
            "frame_bgr": frame_bgr.copy(),
            "landmarks_pixel_xyz": landmarks_pixel_xyz,
            "timestamp_ms": timestamp_ms,
            "frame_id": frame_id,
            "geom_left_closed": geom_left_closed,
            "geom_right_closed": geom_right_closed,
            "raw_left_ear": raw_left_ear,
            "raw_right_ear": raw_right_ear,
            "ear_threshold": ear_threshold,
        }

        # Drop oldest pending job if queue is occupied
        try:
            self._queue.get_nowait()
            self._dropped_jobs_count += 1
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self._dropped_jobs_count += 1
            return False

    def submit_crops(
        self,
        left_crop: Optional[EyeCropQuality],
        right_crop: Optional[EyeCropQuality],
        timestamp_ms: float,
        frame_id: int,
        geom_left_closed: Optional[bool],
        geom_right_closed: Optional[bool],
        raw_left_ear: float,
        raw_right_ear: float,
        ear_threshold: float,
    ) -> bool:
        """
        Submit pre-extracted eye crops directly (used for targeted testing or custom pipelines).
        """
        if not self.is_available or not self.is_running:
            return False

        self._jobs_submitted_count += 1

        job = {
            "type": "CROPS",
            "left_crop": left_crop,
            "right_crop": right_crop,
            "timestamp_ms": timestamp_ms,
            "frame_id": frame_id,
            "geom_left_closed": geom_left_closed,
            "geom_right_closed": geom_right_closed,
            "raw_left_ear": raw_left_ear,
            "raw_right_ear": raw_right_ear,
            "ear_threshold": ear_threshold,
        }

        try:
            self._queue.get_nowait()
            self._dropped_jobs_count += 1
        except queue.Empty:
            pass

        try:
            self._queue.put_nowait(job)
            return True
        except queue.Full:
            self._dropped_jobs_count += 1
            return False

    def get_latest_observation(
        self,
        current_timestamp_ms: float,
    ) -> Optional[BinocularShadowTelemetry]:
        """
        Retrieve the latest telemetry observation.
        Evaluates staleness relative to the current camera timestamp:
        if older than policy.max_staleness_ms (200 ms), marks is_stale=True.
        Guarantees consistent monotonic time calculations.
        """
        self._queries_total_count += 1
        with self._lock:
            telemetry = self._latest_telemetry

        if telemetry is None:
            self._queries_none_count += 1
            return None

        age_ms = max(0.0, current_timestamp_ms - telemetry.capture_timestamp_ms)
        if age_ms > self.policy.max_staleness_ms:
            self._queries_stale_count += 1
            return dataclasses.replace(
                telemetry,
                is_stale=True,
                status_label="STALE",
            )

        self._queries_fresh_count += 1
        return telemetry

    def _worker_loop(self) -> None:
        """Main loop executed by the background worker thread."""
        while True:
            try:
                job = self._queue.get(timeout=0.05)
            except queue.Empty:
                if self._stop_event.is_set():
                    break
                continue

            if job is None:
                break

            try:
                self._process_job(job)
            except Exception as exc:
                self._error_count += 1
                logger.error("Unhandled exception in EyeCNNShadowWorker loop: %s", exc, exc_info=True)

            if self._stop_event.is_set() and self._queue.empty():
                break

    def _process_job(self, job: dict) -> None:
        """Process a single frame or crop job through MobileNetV3 inference."""
        self._jobs_processed_count += 1
        t_capture_ms = job["timestamp_ms"]
        frame_id = job["frame_id"]
        geom_left_closed = job["geom_left_closed"]
        geom_right_closed = job["geom_right_closed"]
        raw_left_ear = job["raw_left_ear"]
        raw_right_ear = job["raw_right_ear"]
        ear_threshold = job["ear_threshold"]

        # 1. Obtain Left and Right Eye Crops
        if job["type"] == "FRAME":
            frame_bgr = job["frame_bgr"]
            landmarks_pixel_xyz = job["landmarks_pixel_xyz"]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            left_crop = extract_eye_crop_with_quality(
                frame_rgb=frame_rgb,
                landmarks_pixel_xyz=landmarks_pixel_xyz,
                eye_indices=LEFT_EYE_INDICES,
                output_size=(64, 64),
            )
            right_crop = extract_eye_crop_with_quality(
                frame_rgb=frame_rgb,
                landmarks_pixel_xyz=landmarks_pixel_xyz,
                eye_indices=RIGHT_EYE_INDICES,
                output_size=(64, 64),
            )
        else:
            left_crop = job["left_crop"]
            right_crop = job["right_crop"]

        # Map geometric states to string enum: "OPEN", "CLOSED", "UNKNOWN"
        def _to_geom_str(closed_flag: Optional[bool]) -> str:
            if closed_flag is True:
                return "CLOSED"
            if closed_flag is False:
                return "OPEN"
            return "UNKNOWN"

        geom_left_str = _to_geom_str(geom_left_closed)
        geom_right_str = _to_geom_str(geom_right_closed)

        # 2. Check crop validity and reliability thresholds
        min_q = self.policy.min_crop_quality
        l_valid = (
            left_crop is not None
            and left_crop.crop_rgb is not None
            and left_crop.overall_reliability >= min_q
        )
        r_valid = (
            right_crop is not None
            and right_crop.crop_rgb is not None
            and right_crop.overall_reliability >= min_q
        )

        left_telemetry = SingleEyeShadowTelemetry(
            eye_label="left",
            raw_ear=raw_left_ear,
            ear_threshold=ear_threshold,
            geometric_state=geom_left_str,
            crop_quality=left_crop,
            crop_reliability=left_crop.overall_reliability if left_crop else 0.0,
            cnn_availability="AVAILABLE" if l_valid else "QUALITY_REJECTED",
        )
        right_telemetry = SingleEyeShadowTelemetry(
            eye_label="right",
            raw_ear=raw_right_ear,
            ear_threshold=ear_threshold,
            geometric_state=geom_right_str,
            crop_quality=right_crop,
            crop_reliability=right_crop.overall_reliability if right_crop else 0.0,
            cnn_availability="AVAILABLE" if r_valid else "QUALITY_REJECTED",
        )

        # If neither crop meets minimum quality, skip inference safely
        if not l_valid and not r_valid:
            self._crop_quality_rejections_count += 1
            now_mono_ms = time.monotonic() * 1000.0
            telemetry = BinocularShadowTelemetry(
                frame_id=frame_id,
                capture_timestamp_ms=t_capture_ms,
                processed_timestamp_ms=now_mono_ms,
                left_eye=left_telemetry,
                right_eye=right_telemetry,
                binocular_geometric_state=self._derive_binocular_state(geom_left_str, geom_right_str),
                binocular_cnn_state=None,
                binocular_agreement=None,
                total_latency_ms=max(0.0, now_mono_ms - t_capture_ms),
                inference_latency_ms=0.0,
                worker_queue_depth=self._queue.qsize(),
                is_stale=False,
                status_label="QUALITY_REJECTED",
            )
            with self._lock:
                self._latest_telemetry = telemetry
            return

        # 3. Prepare Batch Tensor for MobileNetV3 Inference
        crops_to_run = []
        indices_map = []
        if l_valid:
            crops_to_run.append(preprocess_crop_for_mobilenetv3(left_crop.crop_rgb))
            indices_map.append("left")
        if r_valid:
            crops_to_run.append(preprocess_crop_for_mobilenetv3(right_crop.crop_rgb))
            indices_map.append("right")

        batch_blob = np.stack(crops_to_run, axis=0)  # Shape: (B, 3, 64, 64)

        t_infer_start = time.perf_counter()
        cnn_error = False
        error_msg = None

        try:
            if self.cv2_net is None:
                raise RuntimeError("OpenCV DNN Net is not initialized")
            self.cv2_net.setInput(batch_blob)
            logits = self.cv2_net.forward()  # Shape: (B, 2)
            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            self._total_inferences_count += 1
        except (cv2.error, Exception) as exc:
            infer_ms = (time.perf_counter() - t_infer_start) * 1000.0
            cnn_error = True
            error_msg = str(exc)
            self._error_count += 1
            logger.warning("MobileNetV3 eye inference failed: %s", exc)

        # 4. Interpret Predictions or Handle Safe Failure
        if cnn_error:
            if l_valid:
                left_telemetry.cnn_availability = "ERROR"
                left_telemetry.skip_reason = error_msg
            if r_valid:
                right_telemetry.cnn_availability = "ERROR"
                right_telemetry.skip_reason = error_msg

            now_mono_ms = time.monotonic() * 1000.0
            telemetry = BinocularShadowTelemetry(
                frame_id=frame_id,
                capture_timestamp_ms=t_capture_ms,
                processed_timestamp_ms=now_mono_ms,
                left_eye=left_telemetry,
                right_eye=right_telemetry,
                binocular_geometric_state=self._derive_binocular_state(geom_left_str, geom_right_str),
                binocular_cnn_state=None,
                binocular_agreement=None,
                total_latency_ms=max(0.0, now_mono_ms - t_capture_ms),
                inference_latency_ms=infer_ms,
                worker_queue_depth=self._queue.qsize(),
                is_stale=False,
                status_label="ERROR",
            )
            with self._lock:
                self._latest_telemetry = telemetry
            return

        # Compute Softmax: Class 0 = OPEN, Class 1 = CLOSED
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)

        for i, eye_side in enumerate(indices_map):
            open_p = float(probs[i, 0])
            closed_p = float(probs[i, 1])
            pred_class = "OPEN" if open_p >= closed_p else "CLOSED"

            if eye_side == "left":
                left_telemetry.cnn_open_prob = open_p
                left_telemetry.cnn_closed_prob = closed_p
                left_telemetry.cnn_predicted_class = pred_class
                left_telemetry.inference_latency_ms = infer_ms
                left_telemetry.cnn_availability = "AVAILABLE"
                if geom_left_str in ("OPEN", "CLOSED"):
                    left_telemetry.is_agreement = (geom_left_str == pred_class)
            else:
                right_telemetry.cnn_open_prob = open_p
                right_telemetry.cnn_closed_prob = closed_p
                right_telemetry.cnn_predicted_class = pred_class
                right_telemetry.inference_latency_ms = infer_ms
                right_telemetry.cnn_availability = "AVAILABLE"
                if geom_right_str in ("OPEN", "CLOSED"):
                    right_telemetry.is_agreement = (geom_right_str == pred_class)

        # Derive binocular agreement and states
        binoc_geom = self._derive_binocular_state(geom_left_str, geom_right_str)
        binoc_cnn = self._derive_binocular_state(
            left_telemetry.cnn_predicted_class,
            right_telemetry.cnn_predicted_class,
        )

        binoc_agreement = None
        if left_telemetry.is_agreement is not None and right_telemetry.is_agreement is not None:
            binoc_agreement = bool(left_telemetry.is_agreement and right_telemetry.is_agreement)
        elif left_telemetry.is_agreement is not None:
            binoc_agreement = left_telemetry.is_agreement
        elif right_telemetry.is_agreement is not None:
            binoc_agreement = right_telemetry.is_agreement

        now_mono_ms = time.monotonic() * 1000.0
        telemetry = BinocularShadowTelemetry(
            frame_id=frame_id,
            capture_timestamp_ms=t_capture_ms,
            processed_timestamp_ms=now_mono_ms,
            left_eye=left_telemetry,
            right_eye=right_telemetry,
            binocular_geometric_state=binoc_geom,
            binocular_cnn_state=binoc_cnn,
            binocular_agreement=binoc_agreement,
            total_latency_ms=max(0.0, now_mono_ms - t_capture_ms),
            inference_latency_ms=infer_ms,
            worker_queue_depth=self._queue.qsize(),
            is_stale=False,
            status_label="ACTIVE",
        )

        with self._lock:
            self._latest_telemetry = telemetry
            self._completed_inference_timestamps_ms.append(now_mono_ms)

    def get_pipeline_diagnostics(self) -> Dict[str, Any]:
        """Return comprehensive pipeline execution, funnel, and staleness diagnostics."""
        with self._lock:
            intervals = [
                self._completed_inference_timestamps_ms[i] - self._completed_inference_timestamps_ms[i - 1]
                for i in range(1, len(self._completed_inference_timestamps_ms))
            ]
        return {
            # Invocation Funnel
            "evaluations_total": self.policy.evaluations_count,
            "eligible_opportunities": self.policy.accepted_count + self.policy.rejections_rate_limited,
            "evaluations_accepted": self.policy.accepted_count,
            "rejections_disabled": self.policy.rejections_disabled,
            "rejections_no_face": self.policy.rejections_no_face,
            "rejections_ear_invalid": self.policy.rejections_ear_invalid,
            "rejections_rate_limited": self.policy.rejections_rate_limited,
            # Worker Funnel
            "jobs_submitted": self._jobs_submitted_count,
            "jobs_dropped_queue": self._dropped_jobs_count,
            "jobs_processed": self._jobs_processed_count,
            "crop_quality_rejections": self._crop_quality_rejections_count,
            "inferences_successful": self._total_inferences_count,
            "inference_errors": self._error_count,
            # Staleness Query Metrics
            "queries_total": self._queries_total_count,
            "queries_fresh": self._queries_fresh_count,
            "queries_stale": self._queries_stale_count,
            "queries_none": self._queries_none_count,
            # Inter-inference intervals
            "inter_inference_intervals_ms": intervals,
        }

    @staticmethod
    def _derive_binocular_state(
        left_state: Optional[str],
        right_state: Optional[str],
    ) -> Optional[str]:
        """Synthesize binocular state from individual eye states."""
        if left_state is None and right_state is None:
            return None
        if left_state == "CLOSED" and right_state == "CLOSED":
            return "CLOSED"
        if left_state == "OPEN" and right_state == "OPEN":
            return "OPEN"
        if (left_state == "CLOSED" and right_state == "OPEN") or (left_state == "OPEN" and right_state == "CLOSED"):
            return "ASYMMETRIC"
        if left_state in ("OPEN", "CLOSED"):
            return left_state
        if right_state in ("OPEN", "CLOSED"):
            return right_state
        return "UNKNOWN"
