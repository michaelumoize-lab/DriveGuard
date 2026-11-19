"""
DriveGuard — Phase A: MobileNetV3-Small ONNX Runtime Verification & Inference Prototype.

Verifies:
1. Model integrity, file size, and SHA256 fingerprint.
2. cv2.dnn backend initialization and layer structure.
3. Preprocessing contract: 64x64, grayscale 3-channel replication, ImageNet normalization, NCHW layout.
4. Numerical validity and latency over synthetic and real labelled eye crops.
5. Output interpretation: 2-class logits with Softmax activation (Index 0 = OPEN, Index 1 = CLOSED).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


# Model path
WEIGHTS_DIR = Path(__file__).resolve().parent.parent / "backend" / "weights"
MODEL_PATH = WEIGHTS_DIR / "mobilenetv3_eye.onnx"

# ImageNet normalization parameters used during MRL fine-tuning
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def compute_sha256(path: Path) -> str:
    """Compute SHA256 hexadecimal digest of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def preprocess_eye_crop(crop_bgr_or_rgb: np.ndarray, is_bgr: bool = True) -> np.ndarray:
    """
    Apply verified preprocessing pipeline matching train_mobilenetv3_mrl.py:
    1. Resize to (64, 64) via INTER_AREA.
    2. Convert to single-channel luminance (Grayscale) to bridge RGB/IR domain gap.
    3. Replicate across 3 channels (num_output_channels=3).
    4. Rescale to [0.0, 1.0].
    5. Standardize with ImageNet mean/std.
    6. Transpose to NCHW layout: (1, 3, 64, 64) float32.
    """
    if crop_bgr_or_rgb.shape[:2] != (64, 64):
        resized = cv2.resize(crop_bgr_or_rgb, (64, 64), interpolation=cv2.INTER_AREA)
    else:
        resized = crop_bgr_or_rgb

    if len(resized.shape) == 2:
        gray = resized
    else:
        code = cv2.COLOR_BGR2GRAY if is_bgr else cv2.COLOR_RGB2GRAY
        gray = cv2.cvtColor(resized, code)

    # 3-channel grayscale replication
    lum3 = np.stack([gray, gray, gray], axis=-1).astype(np.float32) / 255.0
    norm = (lum3 - IMAGENET_MEAN) / IMAGENET_STD
    tensor = np.transpose(norm, (2, 0, 1))[np.newaxis, ...]  # (1, 3, 64, 64)
    return tensor.astype(np.float32)


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    shift = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shift)
    return exp / np.sum(exp, axis=-1, keepdims=True)


def run_verification() -> Dict[str, any]:
    print("=" * 70)
    print("DriveGuard Phase A: MobileNetV3 ONNX OpenCV DNN Verification")
    print("=" * 70)

    # 1. Model Inspection
    print("\n--- 1. Model Path & Fingerprint ---")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}")

    file_size = MODEL_PATH.stat().st_size
    sha256 = compute_sha256(MODEL_PATH)
    print(f"Path:    {MODEL_PATH.resolve()}")
    print(f"Size:    {file_size:,} bytes ({file_size / (1024 * 1024):.2f} MB)")
    print(f"SHA256:  {sha256}")

    # 2. OpenCV DNN Loading
    print("\n--- 2. OpenCV cv2.dnn Backend Initialization ---")
    t_load_start = time.perf_counter()
    net = cv2.dnn.readNetFromONNX(str(MODEL_PATH))
    load_time_ms = (time.perf_counter() - t_load_start) * 1000.0
    print(f"cv2.dnn loaded successfully in {load_time_ms:.2f} ms")
    
    layer_names = net.getLayerNames()
    out_layers = net.getUnconnectedOutLayersNames()
    print(f"Total network layers: {len(layer_names)}")
    print(f"Output layer names:   {out_layers}")

    # 3. Synthetic Crop Inference
    print("\n--- 3. Synthetic Test Crops (Numerical Stability) ---")
    synthetic_tests = {
        "Uniform Mid-Gray (128)": np.full((64, 64, 3), 128, dtype=np.uint8),
        "Uniform Black (0)": np.zeros((64, 64, 3), dtype=np.uint8),
        "Uniform White (255)": np.full((64, 64, 3), 255, dtype=np.uint8),
        "Random Noise": np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8),
    }

    successful_inferences = 0
    failed_inferences = 0

    for name, crop in synthetic_tests.items():
        try:
            tensor = preprocess_eye_crop(crop, is_bgr=True)
            net.setInput(tensor)
            raw_logits = net.forward()
            probs = softmax(raw_logits[0])

            is_finite = np.all(np.isfinite(raw_logits)) and np.all(np.isfinite(probs))
            if is_finite and raw_logits.shape == (1, 2):
                successful_inferences += 1
                status = "PASS (finite)"
            else:
                failed_inferences += 1
                status = f"FAIL (shape={raw_logits.shape}, finite={is_finite})"

            pred_label = "OPEN" if probs[0] > probs[1] else "CLOSED"
            print(f"  [{status}] {name:25s} -> logits: [{raw_logits[0, 0]:.3f}, {raw_logits[0, 1]:.3f}] | probs: [open={probs[0]:.4f}, closed={probs[1]:.4f}] -> {pred_label}")
        except Exception as e:
            failed_inferences += 1
            print(f"  [ERROR] {name}: {e}")

    # 4. Real Labelled Eye Crops Verification (MRL Dataset Subsets)
    print("\n--- 4. Real Labelled Test Eye Crops (Ground Truth Verification) ---")
    data_dir = Path(__file__).resolve().parent.parent / "backend" / "datasets" / "data" / "test"
    
    awake_samples: List[Path] = []
    sleepy_samples: List[Path] = []
    
    if data_dir.exists():
        awake_dir = data_dir / "awake"
        sleepy_dir = data_dir / "sleepy"
        if awake_dir.exists():
            awake_samples = sorted(awake_dir.glob("*.png"))[:50]
        if sleepy_dir.exists():
            sleepy_samples = sorted(sleepy_dir.glob("*.png"))[:50]

    print(f"Discovered {len(awake_samples)} labelled AWAKE (OPEN) crops and {len(sleepy_samples)} labelled SLEEPY (CLOSED) crops.")

    awake_correct = 0
    awake_logits_list = []
    for f in awake_samples:
        try:
            img = cv2.imread(str(f))
            tensor = preprocess_eye_crop(img, is_bgr=True)
            net.setInput(tensor)
            logits = net.forward()[0]
            probs = softmax(logits)
            successful_inferences += 1
            awake_logits_list.append(logits)
            if probs[0] > probs[1]:  # Class 0 = OPEN
                awake_correct += 1
        except Exception as e:
            failed_inferences += 1

    sleepy_correct = 0
    sleepy_logits_list = []
    for f in sleepy_samples:
        try:
            img = cv2.imread(str(f))
            tensor = preprocess_eye_crop(img, is_bgr=True)
            net.setInput(tensor)
            logits = net.forward()[0]
            probs = softmax(logits)
            successful_inferences += 1
            sleepy_logits_list.append(logits)
            if probs[1] >= probs[0]:  # Class 1 = CLOSED
                sleepy_correct += 1
        except Exception as e:
            failed_inferences += 1

    if awake_samples:
        awake_acc = (awake_correct / len(awake_samples)) * 100.0
        mean_awake_logits = np.mean(awake_logits_list, axis=0)
        print(f"AWAKE (OPEN) Ground Truth:  {awake_correct}/{len(awake_samples)} correct ({awake_acc:.1f}%) | Mean logits [open, closed]: [{mean_awake_logits[0]:.2f}, {mean_awake_logits[1]:.2f}]")
    if sleepy_samples:
        sleepy_acc = (sleepy_correct / len(sleepy_samples)) * 100.0
        mean_sleepy_logits = np.mean(sleepy_logits_list, axis=0)
        print(f"SLEEPY (CLOSED) Ground Truth: {sleepy_correct}/{len(sleepy_samples)} correct ({sleepy_acc:.1f}%) | Mean logits [open, closed]: [{mean_sleepy_logits[0]:.2f}, {mean_sleepy_logits[1]:.2f}]")

    # Sample detailed outputs
    print("\nSample individual predictions on real labelled crops:")
    for f in awake_samples[:3]:
        img = cv2.imread(str(f))
        tensor = preprocess_eye_crop(img, is_bgr=True)
        net.setInput(tensor)
        l = net.forward()[0]
        p = softmax(l)
        print(f"  [Real AWAKE]  {f.name} -> logits: [{l[0]:.2f}, {l[1]:.2f}] | P(open)={p[0]:.4f}, P(closed)={p[1]:.4f}")

    for f in sleepy_samples[:3]:
        img = cv2.imread(str(f))
        tensor = preprocess_eye_crop(img, is_bgr=True)
        net.setInput(tensor)
        l = net.forward()[0]
        p = softmax(l)
        print(f"  [Real SLEEPY] {f.name} -> logits: [{l[0]:.2f}, {l[1]:.2f}] | P(open)={p[0]:.4f}, P(closed)={p[1]:.4f}")

    # 5. Latency Benchmarking
    print("\n--- 5. Inference Latency Benchmark (100 Iterations) ---")
    # Single crop benchmark
    bench_tensor_1 = np.random.randn(1, 3, 64, 64).astype(np.float32)
    latencies_1 = []
    # Warmup
    for _ in range(10):
        net.setInput(bench_tensor_1)
        _ = net.forward()

    for _ in range(100):
        t0 = time.perf_counter()
        net.setInput(bench_tensor_1)
        _ = net.forward()
        latencies_1.append((time.perf_counter() - t0) * 1000.0)

    # Binocular pair benchmark (batch size 2: left + right eye)
    bench_tensor_2 = np.random.randn(2, 3, 64, 64).astype(np.float32)
    latencies_2 = []
    for _ in range(10):
        net.setInput(bench_tensor_2)
        _ = net.forward()

    for _ in range(100):
        t0 = time.perf_counter()
        net.setInput(bench_tensor_2)
        _ = net.forward()
        latencies_2.append((time.perf_counter() - t0) * 1000.0)

    print(f"Single Eye Crop (Batch=1):")
    print(f"  Mean Latency: {np.mean(latencies_1):.2f} ms")
    print(f"  Median (P50): {np.percentile(latencies_1, 50):.2f} ms")
    print(f"  P95 Latency:  {np.percentile(latencies_1, 95):.2f} ms")
    print(f"  Min / Max:    {np.min(latencies_1):.2f} ms / {np.max(latencies_1):.2f} ms")

    print(f"\nBinocular Eye Pair (Batch=2):")
    print(f"  Mean Latency: {np.mean(latencies_2):.2f} ms")
    print(f"  Median (P50): {np.percentile(latencies_2, 50):.2f} ms")
    print(f"  P95 Latency:  {np.percentile(latencies_2, 95):.2f} ms")
    print(f"  Min / Max:    {np.min(latencies_2):.2f} ms / {np.max(latencies_2):.2f} ms")

    # 6. Summary
    print("\n--- 6. Phase A Execution Summary ---")
    print(f"Execution Backend:     OpenCV cv2.dnn (CPUExecution)")
    print(f"Total Inferences Run:  {successful_inferences + failed_inferences}")
    print(f"Successful Inferences: {successful_inferences}")
    print(f"Failed Inferences:     {failed_inferences}")
    print(f"Model Input Spec:      eye_crop_rgb: [batch_size, 3, 64, 64], float32")
    print(f"Model Output Spec:     logits: [batch_size, 2], float32 (Index 0 = OPEN, Index 1 = CLOSED)")
    print("=" * 70)

    return {
        "model_sha256": sha256,
        "backend": "cv2.dnn",
        "successful_inferences": successful_inferences,
        "failed_inferences": failed_inferences,
        "latency_batch1_mean_ms": float(np.mean(latencies_1)),
        "latency_batch1_p95_ms": float(np.percentile(latencies_1, 95)),
        "latency_batch2_mean_ms": float(np.mean(latencies_2)),
        "latency_batch2_p95_ms": float(np.percentile(latencies_2, 95)),
        "awake_acc_pct": float(awake_acc) if awake_samples else None,
        "sleepy_acc_pct": float(sleepy_acc) if sleepy_samples else None,
    }


if __name__ == "__main__":
    run_verification()
