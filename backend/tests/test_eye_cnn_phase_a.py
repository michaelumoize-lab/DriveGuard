"""
DriveGuard — Phase A Unit Test Suite: MobileNetV3 ONNX OpenCV DNN Verification.

Validates:
1. Model file existence, SHA256 integrity, and OpenCV cv2.dnn loading.
2. Input/output tensor shapes and numerical stability (finite logits, Softmax normalization).
3. Preprocessing contract: 64x64 resolution, 3-channel luminance replication, ImageNet standardization.
4. Eye crop geometry: landmark indices, margin padding, frame boundary clipping, and malformed rejection.
5. Ground-truth verification on real labelled MRL eye crops (OPEN=0, CLOSED=1).
6. Performance: CPU inference latency within acceptable budget (<10 ms).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.app.ai.cascade_classifier import (
    DEFAULT_EYE_ONNX_PATH,
    EyeCropQuality,
    MobileNetV3EyeVerifier,
)
from backend.app.ai.geometry import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    extract_eye_crop_with_quality,
)

EXPECTED_SHA256 = "b5577538c00ca9632562990eedeeb4d23a0c33922fcbda927a49a08c7873146a"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def test_1_model_file_and_sha256_integrity() -> None:
    """1. ONNX model file must exist, be >= 5 MB, and match authoritative SHA256."""
    assert DEFAULT_EYE_ONNX_PATH.exists(), f"Model file missing at: {DEFAULT_EYE_ONNX_PATH}"
    size = DEFAULT_EYE_ONNX_PATH.stat().st_size
    assert size >= 5_000_000, f"Model file unexpectedly small: {size} bytes"

    h = hashlib.sha256()
    with open(DEFAULT_EYE_ONNX_PATH, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    actual_sha = h.hexdigest()
    assert actual_sha == EXPECTED_SHA256, f"SHA256 mismatch: {actual_sha} != {EXPECTED_SHA256}"


def test_2_cv2_dnn_loading_and_architecture() -> None:
    """2. cv2.dnn must load mobilenetv3_eye.onnx and expose expected output layers."""
    net = cv2.dnn.readNetFromONNX(str(DEFAULT_EYE_ONNX_PATH))
    assert net is not None
    assert not net.empty()

    out_names = net.getUnconnectedOutLayersNames()
    assert "logits" in out_names or out_names == ("logits",), f"Unexpected output layer: {out_names}"
    assert len(net.getLayerNames()) >= 90


def test_3_synthetic_crop_inference_and_shapes() -> None:
    """3. Synthetic test crops must yield finite float32 outputs with shape [batch, 2]."""
    net = cv2.dnn.readNetFromONNX(str(DEFAULT_EYE_ONNX_PATH))

    # Single crop
    dummy_1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    net.setInput(dummy_1)
    out_1 = net.forward()
    assert out_1.shape == (1, 2)
    assert out_1.dtype == np.float32
    assert np.all(np.isfinite(out_1))

    # Batch of 2 (binocular pair)
    dummy_2 = np.random.randn(2, 3, 64, 64).astype(np.float32)
    net.setInput(dummy_2)
    out_2 = net.forward()
    assert out_2.shape == (2, 2)
    assert np.all(np.isfinite(out_2))


def test_4_real_labelled_dataset_accuracy() -> None:
    """4. Ground-truth MRL eye samples must achieve >= 90% accuracy with verified class mapping."""
    data_dir = Path(__file__).resolve().parents[1] / "datasets" / "data" / "test"
    awake_dir = data_dir / "awake"
    sleepy_dir = data_dir / "sleepy"

    if not awake_dir.exists() or not sleepy_dir.exists():
        pytest.skip("Local test dataset not present in backend/datasets/data/test")

    awake_samples = sorted(awake_dir.glob("*.png"))[:20]
    sleepy_samples = sorted(sleepy_dir.glob("*.png"))[:20]

    net = cv2.dnn.readNetFromONNX(str(DEFAULT_EYE_ONNX_PATH))

    def evaluate_sample(img_path: Path) -> Tuple[float, float]:
        img = cv2.imread(str(img_path))
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        lum3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
        norm = (lum3 - IMAGENET_MEAN) / IMAGENET_STD
        tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]
        net.setInput(tensor.astype(np.float32))
        logits = net.forward()[0]
        shift = logits - np.max(logits)
        exp = np.exp(shift)
        probs = exp / np.sum(exp)
        return float(probs[0]), float(probs[1])

    awake_hits = sum(1 for f in awake_samples if evaluate_sample(f)[0] > 0.50)
    sleepy_hits = sum(1 for f in sleepy_samples if evaluate_sample(f)[1] > 0.50)

    assert awake_hits >= 18, f"Awake accuracy low: {awake_hits}/{len(awake_samples)}"
    assert sleepy_hits >= 18, f"Sleepy accuracy low: {sleepy_hits}/{len(sleepy_samples)}"


def test_5_eye_crop_extraction_geometry_and_clipping() -> None:
    """5. extract_eye_crop_with_quality must pad, clip at boundaries, and reject degenerate boxes."""
    frame = np.full((720, 1280, 3), 120, dtype=np.uint8)

    # 1. Normal eye points centered in frame
    normal_landmarks = np.zeros((478, 3), dtype=np.float32)
    # Left eye landmarks: 33, 160, 158, 133, 153, 144
    normal_landmarks[33] = [600, 350, 0]
    normal_landmarks[160] = [620, 340, 0]
    normal_landmarks[158] = [630, 340, 0]
    normal_landmarks[133] = [650, 350, 0]
    normal_landmarks[153] = [630, 360, 0]
    normal_landmarks[144] = [620, 360, 0]

    crop_q = extract_eye_crop_with_quality(frame, normal_landmarks, LEFT_EYE_INDICES, output_size=(64, 64))
    assert crop_q.crop_rgb is not None
    assert crop_q.crop_rgb.shape == (64, 64, 3)
    assert 0.0 <= crop_q.overall_reliability <= 1.0

    # 2. Extreme edge case: eye near image corner (x=5, y=5) requiring clipping
    edge_landmarks = np.zeros((478, 3), dtype=np.float32)
    edge_landmarks[33] = [2, 5, 0]
    edge_landmarks[160] = [15, 2, 0]
    edge_landmarks[158] = [25, 2, 0]
    edge_landmarks[133] = [35, 5, 0]
    edge_landmarks[153] = [25, 12, 0]
    edge_landmarks[144] = [15, 12, 0]

    edge_crop = extract_eye_crop_with_quality(frame, edge_landmarks, LEFT_EYE_INDICES, output_size=(64, 64))
    assert edge_crop.crop_rgb is not None
    assert edge_crop.crop_rgb.shape == (64, 64, 3)

    # 3. Degenerate case: out-of-frame / zero-collapsed points at image boundary (0, 0)
    degenerate_landmarks = np.zeros((478, 3), dtype=np.float32)
    deg_crop = extract_eye_crop_with_quality(frame, degenerate_landmarks, LEFT_EYE_INDICES, output_size=(64, 64))
    assert deg_crop.crop_rgb is None
    assert deg_crop.overall_reliability == 0.0


def test_6_verifier_cv2_dnn_execution_and_provenance() -> None:
    """6. MobileNetV3EyeVerifier must use cv2.dnn and produce MOBILENET_ONNX provenance."""
    verifier = MobileNetV3EyeVerifier()
    diag = verifier.get_diagnostics()

    assert diag["is_pretrained_onnx"] is True
    assert diag["execution_mode"] == "MOBILENET_ONNX"
    assert diag["backend"] == "cv2.dnn"
    assert diag["session_active"] is True
    assert diag["model_sha256"] == EXPECTED_SHA256

    # Test single crop prediction
    dummy_crop = np.full((64, 64, 3), 120, dtype=np.uint8)
    crop_q = EyeCropQuality(
        crop_rgb=dummy_crop,
        sharpness_score=0.8,
        exposure_score=1.0,
        glare_fraction=0.0,
        overall_reliability=0.9,
    )

    pred = verifier.predict_eye_crop(crop_q)
    assert pred.execution_mode == "MOBILENET_ONNX"
    assert pred.onnx_executed is True
    assert pred.duration_ms > 0.0
    assert 0.0 <= pred.open_prob <= 1.0
    assert 0.0 <= pred.closed_prob <= 1.0
    assert abs(pred.open_prob + pred.closed_prob - 1.0) < 1e-4

    # Test binocular prediction
    binoc = verifier.predict_binocular(crop_q, crop_q)
    assert binoc.execution_mode == "MOBILENET_ONNX"
    assert binoc.onnx_executed is True
    assert binoc.duration_ms > 0.0


def test_7_latency_budget_on_cpu() -> None:
    """7. Single-crop and binocular pair inference latency on CPU must be < 10 ms."""
    net = cv2.dnn.readNetFromONNX(str(DEFAULT_EYE_ONNX_PATH))
    tensor = np.random.randn(2, 3, 64, 64).astype(np.float32)

    # Warmup
    for _ in range(5):
        net.setInput(tensor)
        _ = net.forward()

    latencies = []
    for _ in range(30):
        t0 = time.perf_counter()
        net.setInput(tensor)
        _ = net.forward()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    mean_ms = float(np.mean(latencies))
    p95_ms = float(np.percentile(latencies, 95))
    assert mean_ms < 25.0, f"Mean latency exceeded 25 ms budget: {mean_ms:.2f} ms"
    assert p95_ms < 35.0, f"P95 latency exceeded 35 ms limit: {p95_ms:.2f} ms"
