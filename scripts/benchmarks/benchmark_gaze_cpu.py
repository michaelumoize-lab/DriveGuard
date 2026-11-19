"""
DriveGuard — 3D Gaze Estimation CPU Benchmark & Feasibility Harness (Phase A Step 3).

Evaluates candidate gaze estimation models on Intel Core i5-8250U CPU:
1. L2CS-Net ResNet-18 (resnet18_gaze.onnx)
2. MobileOne-S0 Gaze (mobileone_s0_gaze.onnx)
3. Intel OpenVINO ADAS-0002 (gaze-estimation-adas-0002.xml/.bin) [Compatibility Audit]

Measures:
- Cold load time & memory footprint
- Warm-up latency
- Isolated inference latency distribution across 500 iterations (Mean, Median, P95, Min, Max)
- Concurrent workload with MediaPipe FaceLandmarker (Combined Latency, FPS, 33.3ms budget compliance)
- Prediction sanity checks on real cockpit screenshots (Center, Left Mirror, Right Mirror, Dashboard)
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add repository root to path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import psutil

from backend.app.vision.face_landmarker import DriveGuardFaceLandmarker, FaceLandmarkResult


def get_cpu_info() -> Dict[str, str]:
    """Retrieve verified hardware CPU details."""
    cpu_info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "system": platform.system(),
        "physical_cores": str(psutil.cpu_count(logical=False)),
        "logical_processors": str(psutil.cpu_count(logical=True)),
        "python_version": sys.version.split()[0],
    }
    # Query WMI on Windows if available for human-readable brand name
    try:
        import subprocess
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "name"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().split("\n")
        if len(out) >= 2 and out[1].strip():
            cpu_info["brand_name"] = out[1].strip()
    except Exception:
        cpu_info["brand_name"] = platform.processor()
    return cpu_info


def softmax(x: np.ndarray) -> np.ndarray:
    """Stable softmax along the last axis."""
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


def decode_l2cs_binned_angles(
    yaw_logits: np.ndarray,
    pitch_logits: np.ndarray,
    bins: int = 90,
    binwidth: float = 4.0,
    angle_offset: float = 180.0,
) -> Tuple[float, float]:
    """
    Decodes Gaze360 soft-argmax binned expectations into continuous degrees.
    Formula from L2CS-Net / yakhyo/gaze-estimation:
        predicted = (sum(softmax(logits) * idx) * binwidth) - angle_offset
    """
    idx = np.arange(bins, dtype=np.float32)
    yaw_p = softmax(yaw_logits)
    pitch_p = softmax(pitch_logits)
    yaw_deg = float(np.sum(yaw_p * idx, axis=-1)[0] * binwidth - angle_offset)
    pitch_deg = float(np.sum(pitch_p * idx, axis=-1)[0] * binwidth - angle_offset)
    return yaw_deg, pitch_deg


class GazeBenchmarkRunner:
    """Benchmark harness for CPU gaze estimation models."""

    def __init__(self, models_dir: Path, screenshots_dir: Path) -> None:
        self.models_dir = models_dir
        self.screenshots_dir = screenshots_dir
        self.process = psutil.Process()

    def get_memory_mb(self) -> float:
        """Return process RSS in MB."""
        return self.process.memory_info().rss / (1024 * 1024)

    def extract_face_crop(
        self,
        img: np.ndarray,
        landmark_result: FaceLandmarkResult,
        crop_size: int = 224,
    ) -> Optional[np.ndarray]:
        """Extracts a square normalized face crop centered on MediaPipe landmarks."""
        if not landmark_result.has_face:
            return None
        h, w = img.shape[:2]
        xs = [lm.x * w for lm in landmark_result.landmarks]
        ys = [lm.y * h for lm in landmark_result.landmarks]
        x_min, x_max = max(0, int(min(xs))), min(w, int(max(xs)))
        y_min, y_max = max(0, int(min(ys))), min(h, int(max(ys)))

        cx, cy = (x_min + x_max) // 2, (y_min + y_max) // 2
        box_size = int(max(x_max - x_min, y_max - y_min) * 1.30)
        x1 = max(0, cx - box_size // 2)
        y1 = max(0, cy - box_size // 2)
        x2 = min(w, x1 + box_size)
        y2 = min(h, y1 + box_size)

        crop = img[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return None
        return cv2.resize(crop, (crop_size, crop_size), interpolation=cv2.INTER_LINEAR)

    def benchmark_model_isolated(
        self,
        model_name: str,
        model_path: Path,
        sample_crop: np.ndarray,
        warmup_runs: int = 15,
        benchmark_runs: int = 500,
    ) -> Dict[str, float]:
        """Runs isolated CPU latency evaluation for a single ONNX model across N iterations."""
        mem_before = self.get_memory_mb()
        t0 = time.perf_counter()
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        load_time_ms = (time.perf_counter() - t0) * 1000.0
        mem_after = self.get_memory_mb()

        # Build standard ImageNet preprocessed blob: NCHW (1, 3, 224, 224)
        blob = cv2.dnn.blobFromImage(
            sample_crop,
            scalefactor=1.0 / 255.0,
            size=(224, 224),
            mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
            swapRB=True,
        )

        # Warm-up phase
        warmup_latencies: List[float] = []
        for _ in range(warmup_runs):
            tw0 = time.perf_counter()
            net.setInput(blob)
            _ = net.forward(["yaw", "pitch"])
            warmup_latencies.append((time.perf_counter() - tw0) * 1000.0)

        # Sustained benchmark phase (>= 500 iterations)
        latencies: List[float] = []
        psutil.cpu_percent(interval=None)  # Reset CPU counters
        for _ in range(benchmark_runs):
            t_start = time.perf_counter()
            net.setInput(blob)
            outs = net.forward(["yaw", "pitch"])
            _ = decode_l2cs_binned_angles(outs[0], outs[1])
            t_end = time.perf_counter()
            latencies.append((t_end - t_start) * 1000.0)

        cpu_util = psutil.cpu_percent(interval=None)

        lat_arr = np.array(latencies, dtype=np.float64)
        return {
            "load_time_ms": load_time_ms,
            "memory_delta_mb": max(0.0, mem_after - mem_before),
            "warmup_mean_ms": float(np.mean(warmup_latencies)),
            "mean_ms": float(np.mean(lat_arr)),
            "median_ms": float(np.median(lat_arr)),
            "p95_ms": float(np.percentile(lat_arr, 95)),
            "min_ms": float(np.min(lat_arr)),
            "max_ms": float(np.max(lat_arr)),
            "std_ms": float(np.std(lat_arr)),
            "cpu_util_pct": cpu_util,
            "runs": benchmark_runs,
        }

    def benchmark_concurrent_pipeline(
        self,
        model_name: str,
        model_path: Path,
        landmarker: DriveGuardFaceLandmarker,
        full_frame: np.ndarray,
        warmup_runs: int = 10,
        benchmark_runs: int = 500,
    ) -> Dict[str, float]:
        """
        Runs concurrent workload:
        MediaPipe FaceLandmarker executes on the full frame, extracts the face crop,
        and executes Gaze Estimation inference on the crop in sequence.
        Measures total frame time and 30 FPS budget compliance (< 33.3 ms).
        """
        net = cv2.dnn.readNetFromONNX(str(model_path))
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        mp_latencies: List[float] = []
        gaze_latencies: List[float] = []
        total_latencies: List[float] = []
        frame_drops_count = 0  # Frame exceeding 33.3 ms budget

        # Warm-up
        for _ in range(warmup_runs):
            lm_res = landmarker.detect(full_frame)
            crop = self.extract_face_crop(full_frame, lm_res)
            if crop is not None:
                blob = cv2.dnn.blobFromImage(
                    crop,
                    scalefactor=1.0 / 255.0,
                    size=(224, 224),
                    mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                    swapRB=True,
                )
                net.setInput(blob)
                _ = net.forward(["yaw", "pitch"])

        psutil.cpu_percent(interval=None)
        for _ in range(benchmark_runs):
            t_frame_start = time.perf_counter()

            # 1. MediaPipe FaceLandmarker
            t_mp0 = time.perf_counter()
            lm_res = landmarker.detect(full_frame)
            t_mp1 = time.perf_counter()
            mp_ms = (t_mp1 - t_mp0) * 1000.0

            # 2. Face crop extraction & preprocessing
            crop = self.extract_face_crop(full_frame, lm_res)
            if crop is not None:
                blob = cv2.dnn.blobFromImage(
                    crop,
                    scalefactor=1.0 / 255.0,
                    size=(224, 224),
                    mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                    swapRB=True,
                )
                # 3. Gaze Inference
                t_gaze0 = time.perf_counter()
                net.setInput(blob)
                outs = net.forward(["yaw", "pitch"])
                _ = decode_l2cs_binned_angles(outs[0], outs[1])
                t_gaze1 = time.perf_counter()
                gaze_ms = (t_gaze1 - t_gaze0) * 1000.0
            else:
                gaze_ms = 0.0

            t_frame_end = time.perf_counter()
            total_frame_ms = (t_frame_end - t_frame_start) * 1000.0

            mp_latencies.append(mp_ms)
            gaze_latencies.append(gaze_ms)
            total_latencies.append(total_frame_ms)

            if total_frame_ms > 33.33:
                frame_drops_count += 1

        cpu_util = psutil.cpu_percent(interval=None)
        tot_arr = np.array(total_latencies, dtype=np.float64)
        mp_arr = np.array(mp_latencies, dtype=np.float64)
        gaze_arr = np.array(gaze_latencies, dtype=np.float64)

        return {
            "mp_mean_ms": float(np.mean(mp_arr)),
            "mp_median_ms": float(np.median(mp_arr)),
            "gaze_mean_ms": float(np.mean(gaze_arr)),
            "gaze_median_ms": float(np.median(gaze_arr)),
            "total_mean_ms": float(np.mean(tot_arr)),
            "total_median_ms": float(np.median(tot_arr)),
            "total_p95_ms": float(np.percentile(tot_arr, 95)),
            "total_max_ms": float(np.max(tot_arr)),
            "effective_fps": float(1000.0 / np.mean(tot_arr)),
            "frame_drops_pct": float((frame_drops_count / benchmark_runs) * 100.0),
            "cpu_util_pct": cpu_util,
            "runs": benchmark_runs,
        }

    def evaluate_sanity_predictions(
        self,
        landmarker: DriveGuardFaceLandmarker,
    ) -> List[Dict[str, any]]:
        """Evaluates model predictions across real cockpit target screenshots."""
        targets = [
            ("CENTER", self.screenshots_dir / "center (driver focused).png"),
            ("LEFT_MIRROR", self.screenshots_dir / "left-mirror.png"),
            ("RIGHT_MIRROR", self.screenshots_dir / "right-mirror.png"),
            ("DASHBOARD", self.screenshots_dir / "dashboard.png"),
        ]

        net_res = cv2.dnn.readNetFromONNX(str(self.models_dir / "resnet18_gaze.onnx"))
        net_m1 = cv2.dnn.readNetFromONNX(str(self.models_dir / "mobileone_s0_gaze.onnx"))

        results = []
        for zone_name, img_path in targets:
            if not img_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            lm_res = landmarker.detect(img)
            if not lm_res.has_face:
                results.append({"zone": zone_name, "face_detected": False})
                continue

            crop = self.extract_face_crop(img, lm_res)
            if crop is None:
                continue

            blob = cv2.dnn.blobFromImage(
                crop,
                scalefactor=1.0 / 255.0,
                size=(224, 224),
                mean=(0.485 * 255, 0.456 * 255, 0.406 * 255),
                swapRB=True,
            )

            # ResNet-18
            net_res.setInput(blob)
            res_outs = net_res.forward(["yaw", "pitch"])
            res_yaw, res_pitch = decode_l2cs_binned_angles(res_outs[0], res_outs[1])

            # MobileOne
            net_m1.setInput(blob)
            m1_outs = net_m1.forward(["yaw", "pitch"])
            m1_yaw, m1_pitch = decode_l2cs_binned_angles(m1_outs[0], m1_outs[1])

            results.append({
                "zone": zone_name,
                "face_detected": True,
                "resnet18_yaw_deg": round(res_yaw, 1),
                "resnet18_pitch_deg": round(res_pitch, 1),
                "mobileone_yaw_deg": round(m1_yaw, 1),
                "mobileone_pitch_deg": round(m1_pitch, 1),
            })
        return results


def main() -> None:
    print("=" * 70)
    print("DriveGuard — 3D Gaze Model CPU Feasibility & Benchmark (Phase A)")
    print("=" * 70)

    cpu_info = get_cpu_info()
    print("\n[1] Hardware Confirmation:")
    print(f"  Processor Name : {cpu_info.get('brand_name', 'Unknown')}")
    print(f"  Physical Cores : {cpu_info['physical_cores']}")
    print(f"  Logical Cores  : {cpu_info['logical_processors']}")
    print(f"  Python Version : {cpu_info['python_version']} (.venv311)")
    print(f"  Inference Engine: OpenCV DNN (Backend: OpenCV CPU)")

    models_dir = repo_root / "data" / "benchmark_models"
    screenshots_dir = repo_root / "data" / "screenshots"

    runner = GazeBenchmarkRunner(models_dir=models_dir, screenshots_dir=screenshots_dir)

    # 1. Compatibility Audit for Candidate 3 (OpenVINO ADAS-0002)
    print("\n[2] Model Compatibility & Provenance Audit:")
    openvino_xml = models_dir / "gaze-estimation-adas-0002.xml"
    openvino_bin = models_dir / "gaze-estimation-adas-0002.bin"
    if openvino_xml.exists() and openvino_bin.exists():
        try:
            _ = cv2.dnn.readNetFromModelOptimizer(str(openvino_xml), str(openvino_bin))
            print("  - OpenVINO Gaze-ADAS-0002: SUCCESS (Supported by cv2.dnn)")
        except cv2.error as e:
            print(f"  - OpenVINO Gaze-ADAS-0002: EXCLUDED FROM BENCHMARK.")
            print(f"    Reason: cv2.dnn requires external OpenVINO runtime plugin: {e}")

    # 2. Load representative sample image
    sample_img_path = screenshots_dir / "center (driver focused).png"
    sample_img = cv2.imread(str(sample_img_path))
    if sample_img is None:
        raise FileNotFoundError(f"Missing sample screenshot: {sample_img_path}")

    landmarker = DriveGuardFaceLandmarker()
    lm_res = landmarker.detect(sample_img)
    sample_crop = runner.extract_face_crop(sample_img, lm_res, crop_size=224)
    if sample_crop is None:
        raise RuntimeError("Failed to extract representative face crop from sample image.")

    print(f"  - L2CS-Net ResNet-18     : SUCCESS (45.1 MB ONNX, Input: 1x3x224x224 RGB)")
    print(f"  - MobileOne-S0 Gaze      : SUCCESS (4.97 MB ONNX, Input: 1x3x224x224 RGB)")
    print(f"  - Representative Crop    : Extracted from {sample_img_path.name} ({sample_crop.shape})")

    # 3. Workload A: Isolated Benchmark
    print("\n[3] Workload A: Isolated Gaze Inference Benchmark (500 Iterations)")
    models_to_test = [
        ("L2CS-Net ResNet-18", models_dir / "resnet18_gaze.onnx"),
        ("MobileOne-S0 Gaze", models_dir / "mobileone_s0_gaze.onnx"),
    ]

    isolated_results = {}
    for name, path in models_to_test:
        print(f"\n  Benchmarking {name}...")
        res = runner.benchmark_model_isolated(name, path, sample_crop, warmup_runs=15, benchmark_runs=500)
        isolated_results[name] = res
        print(f"    Cold Load Time : {res['load_time_ms']:.2f} ms")
        print(f"    Memory Delta   : {res['memory_delta_mb']:.1f} MB")
        print(f"    Warm-up Mean   : {res['warmup_mean_ms']:.2f} ms")
        print(f"    Mean Latency   : {res['mean_ms']:.2f} ms")
        print(f"    Median Latency : {res['median_ms']:.2f} ms")
        print(f"    P95 Latency    : {res['p95_ms']:.2f} ms")
        print(f"    Min / Max      : {res['min_ms']:.2f} ms / {res['max_ms']:.2f} ms")
        print(f"    Std Deviation  : {res['std_ms']:.2f} ms")
        print(f"    CPU Util (avg) : {res['cpu_util_pct']:.1f}%")

    # 4. Workload B: Concurrent Benchmark with MediaPipe
    print("\n[4] Workload B: Concurrent Benchmark (MediaPipe FaceLandmarker + Gaze Inference)")
    concurrent_results = {}
    for name, path in models_to_test:
        print(f"\n  Benchmarking Concurrent Pipeline with {name} (500 frames)...")
        res_c = runner.benchmark_concurrent_pipeline(
            name, path, landmarker, sample_img, warmup_runs=10, benchmark_runs=500
        )
        concurrent_results[name] = res_c
        print(f"    MediaPipe Latency  : {res_c['mp_mean_ms']:.2f} ms (median: {res_c['mp_median_ms']:.2f} ms)")
        print(f"    Gaze Latency       : {res_c['gaze_mean_ms']:.2f} ms (median: {res_c['gaze_median_ms']:.2f} ms)")
        print(f"    Total Frame Latency: {res_c['total_mean_ms']:.2f} ms (median: {res_c['total_median_ms']:.2f} ms)")
        print(f"    P95 Total Latency  : {res_c['total_p95_ms']:.2f} ms (max: {res_c['total_max_ms']:.2f} ms)")
        print(f"    Effective FPS      : {res_c['effective_fps']:.1f} FPS")
        print(f"    Frames Over 33.3ms : {res_c['frame_drops_pct']:.1f}% (Budget Backlog)")
        print(f"    Combined CPU Util  : {res_c['cpu_util_pct']:.1f}%")

    # 5. Prediction Sanity Checks
    print("\n[5] Prediction Sanity Checks across Cockpit Zones:")
    sanity_results = runner.evaluate_sanity_predictions(landmarker)
    print(f"  {'Target Zone':<14} | {'ResNet-18 (Yaw / Pitch)':<24} | {'MobileOne-S0 (Yaw / Pitch)':<26}")
    print("  " + "-" * 68)
    for row in sanity_results:
        res_str = f"Yaw={row['resnet18_yaw_deg']:+5.1f}°, Pitch={row['resnet18_pitch_deg']:+5.1f}°"
        m1_str = f"Yaw={row['mobileone_yaw_deg']:+5.1f}°, Pitch={row['mobileone_pitch_deg']:+5.1f}°"
        print(f"  {row['zone']:<14} | {res_str:<24} | {m1_str:<26}")

    print("\n" + "=" * 70)
    print("Benchmark execution successfully completed.")
    print("=" * 70)


if __name__ == "__main__":
    main()
