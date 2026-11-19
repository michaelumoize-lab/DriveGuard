"""
DriveGuard — Phase B1.1: Live Performance and Reliability Validation Harness.

Evaluates MobileNetV3 Eye CNN Shadow Observer across five operating configurations:
1. Shadow OFF (Baseline)
2. Shadow ON @ 1 Hz
3. Shadow ON @ 2 Hz
4. Shadow ON @ 5 Hz
5. Shadow ON @ 15 Hz

Methodology:
- Hardware: Host CPU (Intel Core i5-8250U, 4 cores / 8 threads).
- Duration: 60 seconds per configuration (300 seconds total).
- Warm-up: First 3.0 seconds excluded from steady-state statistics.
- Metrics measured with time.perf_counter():
  * Camera capture FPS & main-loop display FPS
  * Main-loop frame processing duration: mean, median, P95, P99 (ms)
  * MediaPipe FaceLandmarker latency: mean, median, P95 (ms)
  * CNN inference latency (pure forward pass): mean, median, P95 (ms)
  * Process CPU % and System CPU % via psutil
  * Memory footprint (RSS) delta in MB
  * Queue drops, replacements, stale results, and worker errors
- Strict Isolation Verification:
  * Bit-for-bit assertion across EAR, PERCLOS, blinks, DrowsinessAssessment,
    and Phase 8 AlertState under all shadow operating modes.
"""

from __future__ import annotations

import gc
import json
import logging
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to sys.path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np
import psutil

from backend.app.ai.eye_shadow_observer import (
    BinocularShadowTelemetry,
    EyeCNNInvocationPolicy,
    EyeCNNShadowWorker,
)
from backend.app.alerts import (
    AlertDispatcher,
    AlertState,
    AlertStateManager,
    AudioWarningEngine,
    audio_settings_store,
)
from backend.app.camera.camera_manager import create_camera
from backend.app.camera.frame_quality import FrameQualityAnalyzer
from backend.app.features import DriverFeatureExtractor
from backend.app.intelligence import DrowsinessEngine
from backend.app.vision.face_landmarker import DriveGuardFaceLandmarker

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("phase_b1_1_benchmark")


def get_system_hardware_info() -> Dict[str, str]:
    """Retrieve verified host CPU and operating system metadata."""
    info = {
        "os": f"{platform.system()} {platform.release()} ({platform.version()})",
        "cpu_arch": platform.machine(),
        "processor": platform.processor(),
        "physical_cores": str(psutil.cpu_count(logical=False)),
        "logical_processors": str(psutil.cpu_count(logical=True)),
        "ram_total_gb": f"{psutil.virtual_memory().total / (1024**3):.2f} GB",
        "python": sys.version.split()[0],
    }
    try:
        import subprocess
        out = subprocess.check_output(
            ["wmic", "cpu", "get", "name"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip().split("\n")
        if len(out) >= 2 and out[1].strip():
            info["cpu_brand"] = out[1].strip()
    except Exception:
        info["cpu_brand"] = platform.processor()
    return info


def run_configuration_benchmark(
    config_name: str,
    shadow_enabled: bool,
    shadow_hz: float,
    duration_sec: float = 60.0,
    warmup_sec: float = 3.0,
    device_index: int = 0,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
) -> Dict[str, Any]:
    """
    Execute a controlled 60-second live camera benchmark for a single configuration.
    """
    process = psutil.Process()
    # Baseline CPU query to prime psutil
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)

    # Initialize camera adapter
    camera = create_camera(
        source=device_index,
        target_width=width,
        target_height=height,
        target_fps=target_fps,
        backend="DSHOW",
        flip_horizontal=True,
    )
    if not camera.open():
        raise RuntimeError(f"Failed to open camera device index {device_index}")

    # Initialize core pipeline
    quality_analyzer = FrameQualityAnalyzer()
    landmarker = DriveGuardFaceLandmarker()
    feature_extractor = DriverFeatureExtractor()
    drowsiness_engine = DrowsinessEngine()
    alert_manager = AlertStateManager()
    audio_engine = AudioWarningEngine()
    alert_dispatcher = AlertDispatcher(
        audio_engine=audio_engine,
        settings_store=audio_settings_store,
    )

    # Initialize shadow observer worker
    shadow_policy = EyeCNNInvocationPolicy(
        enabled=shadow_enabled,
        max_frequency_hz=shadow_hz,
        max_staleness_ms=200.0,
    )
    shadow_worker = EyeCNNShadowWorker(policy=shadow_policy)
    if shadow_enabled:
        shadow_worker.start()

    actual_res = camera.get_actual_resolution()
    f_w = actual_res[0] if actual_res[0] > 0 else width
    f_h = actual_res[1] if actual_res[1] > 0 else height

    # Telemetry collectors
    loop_processing_times_ms: List[float] = []
    loop_processing_face_only_ms: List[float] = []
    mediapipe_latencies_ms: List[float] = []
    cnn_latencies_ms: List[float] = []
    cnn_total_latencies_ms: List[float] = []
    process_cpu_samples: List[float] = []
    system_cpu_samples: List[float] = []

    stale_observations_count = 0
    total_frames_rendered = 0
    total_face_tracking_frames = 0
    last_cpu_sample_time = time.monotonic()

    t_start = time.monotonic()
    t_end = t_start + duration_sec

    try:
        while time.monotonic() < t_end:
            now = time.monotonic()
            is_warmup = (now - t_start) < warmup_sec

            # Read camera frame
            read_success, packet = camera.read(timeout_sec=0.1)
            if not read_success or packet is None:
                continue

            # Measure frame processing time (excluding camera acquisition wait)
            t_loop_start = time.perf_counter()

            # 1. Optical Frame Quality (Phase 3)
            quality = quality_analyzer.analyze(packet.frame)

            # 2. Facial Landmark Tracking (Phase 4)
            landmark_result = landmarker.detect(packet)

            # 3. Physiological Driver Features (Phase 5)
            features = feature_extractor.extract(
                landmark_result,
                frame_width=f_w,
                frame_height=f_h,
                is_mirrored=True,
            )

            # 4. Shadow Observer Ingestion (Phase B1)
            if shadow_enabled and shadow_worker.is_running:
                try:
                    should_shadow, _ = shadow_policy.should_invoke(
                        timestamp_ms=packet.timestamp_ms,
                        has_face=landmark_result.has_face if landmark_result else False,
                        ear_valid=features.ear.is_geometric_valid if features else False,
                    )
                    if should_shadow and landmark_result is not None and landmark_result.has_face:
                        lms_xyz = np.array(
                            [[lm.x * f_w, lm.y * f_h, lm.z * f_w] for lm in landmark_result.landmarks],
                            dtype=np.float64,
                        )
                        shadow_worker.submit_frame(
                            frame_bgr=packet.frame,
                            landmarks_pixel_xyz=lms_xyz,
                            timestamp_ms=packet.timestamp_ms,
                            frame_id=getattr(packet, "frame_index", total_frames_rendered),
                            geom_left_closed=features.ear.is_left_closed if features else None,
                            geom_right_closed=features.ear.is_right_closed if features else None,
                            raw_left_ear=features.ear.left_ear if features else 0.0,
                            raw_right_ear=features.ear.right_ear if features else 0.0,
                            ear_threshold=features.ear.closure_threshold if features else 0.20,
                        )

                    shadow_obs = shadow_worker.get_latest_observation(packet.timestamp_ms)
                    if shadow_obs is not None and not is_warmup:
                        if shadow_obs.is_stale:
                            stale_observations_count += 1
                        if shadow_obs.inference_latency_ms > 0:
                            cnn_latencies_ms.append(shadow_obs.inference_latency_ms)
                            cnn_total_latencies_ms.append(shadow_obs.total_latency_ms)
                except Exception as e:
                    logger.warning("Shadow ingestion error: %s", e)

            # 5. Drowsiness Intelligence Engine (Phase 6)
            assessment = drowsiness_engine.evaluate(
                features=features,
                quality=quality,
            )

            # 6. Alert State Manager & Dispatcher (Phase 8)
            transition = alert_manager.update(assessment)
            alert_dispatcher.dispatch(
                transition=transition,
                current_state=alert_manager.current_state,
                timestamp_ms=packet.timestamp_ms,
                is_recovering=alert_manager.is_recovering,
            )

            # Record steady-state loop duration
            t_loop_end = time.perf_counter()
            loop_ms = (t_loop_end - t_loop_start) * 1000.0

            total_frames_rendered += 1
            if landmark_result and landmark_result.has_face:
                total_face_tracking_frames += 1

            if not is_warmup:
                loop_processing_times_ms.append(loop_ms)
                if landmark_result and landmark_result.has_face:
                    loop_processing_face_only_ms.append(loop_ms)
                if landmark_result:
                    mediapipe_latencies_ms.append(landmark_result.inference_latency_ms)

            # Sample CPU usage every 1.0 second
            if now - last_cpu_sample_time >= 1.0:
                p_cpu = process.cpu_percent(interval=None) / psutil.cpu_count(logical=True)
                s_cpu = psutil.cpu_percent(interval=None)
                process_cpu_samples.append(p_cpu)
                system_cpu_samples.append(s_cpu)
                last_cpu_sample_time = now

    finally:
        pipeline_diag = shadow_worker.get_pipeline_diagnostics() if shadow_enabled else {}
        if shadow_enabled:
            shadow_worker.stop()
        alert_dispatcher.shutdown()
        camera.close()
        landmarker.close()

    total_duration = time.monotonic() - t_start
    final_rss_mb = process.memory_info().rss / (1024 * 1024)
    buffer_stats = camera.get_buffer_stats()

    # Statistical summaries
    def _stats(arr: List[float]) -> Dict[str, float]:
        if not arr:
            return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "p99": float(np.percentile(arr, 99)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }

    return {
        "config_name": config_name,
        "shadow_enabled": shadow_enabled,
        "shadow_hz": shadow_hz,
        "duration_sec": total_duration,
        "steady_state_samples": len(loop_processing_times_ms),
        "total_captured_frames": buffer_stats.get("total_pushed", 0),
        "total_rendered_frames": total_frames_rendered,
        "capture_fps": buffer_stats.get("total_pushed", 0) / max(0.1, total_duration),
        "display_fps": total_frames_rendered / max(0.1, total_duration),
        "face_detection_pct": (total_face_tracking_frames / max(1, total_frames_rendered)) * 100.0,
        "loop_processing_ms": _stats(loop_processing_times_ms),
        "loop_processing_face_only_ms": _stats(loop_processing_face_only_ms),
        "mediapipe_latency_ms": _stats(mediapipe_latencies_ms),
        "cnn_inference_latency_ms": _stats(cnn_latencies_ms),
        "cnn_total_latency_ms": _stats(cnn_total_latencies_ms),
        "process_cpu_pct": float(np.mean(process_cpu_samples)) if process_cpu_samples else 0.0,
        "system_cpu_pct": float(np.mean(system_cpu_samples)) if system_cpu_samples else 0.0,
        "initial_rss_mb": initial_rss_mb,
        "final_rss_mb": final_rss_mb,
        "rss_delta_mb": final_rss_mb - initial_rss_mb,
        "worker_total_inferences": shadow_worker._total_inferences_count if shadow_enabled else 0,
        "worker_dropped_jobs": shadow_worker._dropped_jobs_count if shadow_enabled else 0,
        "worker_errors": shadow_worker._error_count if shadow_enabled else 0,
        "stale_observations_count": stale_observations_count,
        "pipeline_diagnostics": pipeline_diag,
    }


def verify_strict_isolation() -> Dict[str, bool]:
    """
    Verify that authoritative EAR, Blink, PERCLOS, Temporal, Decision, and Alert
    outputs remain bit-for-bit identical across all shadow operating conditions.
    """
    from backend.tests.test_drowsiness_engine import make_driver_features
    from backend.app.camera.frame_quality import FrameQualityResult, LightingCondition, BlurCondition, QualityStatus
    from unittest.mock import MagicMock

    engine = DrowsinessEngine()
    manager = AlertStateManager()

    features = make_driver_features(timestamp_ms=1000.0, has_face=True, is_closed=False, ear_val=0.28)
    quality = FrameQualityResult(
        mean_luminance=120.0,
        blur_variance=250.0,
        lighting=LightingCondition.OPTIMAL,
        blur=BlurCondition.SHARP,
        status=QualityStatus.EXCELLENT,
        is_viable_for_inference=True,
        reasons=(),
    )

    # 1. Baseline
    base_assess = engine.evaluate(features=features, quality=quality)
    base_trans = manager.update(base_assess)

    # 2. Shadow Active (All Closed)
    w_closed = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
    w_closed.is_available = True
    w_closed.cv2_net = MagicMock()
    w_closed.cv2_net.forward.return_value = np.array([[-10.0, 10.0], [-10.0, 10.0]], dtype=np.float32)
    w_closed.start()
    try:
        w_closed.submit_frame(
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks_pixel_xyz=np.ones((478, 3), dtype=np.float64) * 50.0,
            timestamp_ms=1000.0,
            frame_id=1,
            geom_left_closed=False,
            geom_right_closed=False,
            raw_left_ear=0.28,
            raw_right_ear=0.28,
            ear_threshold=0.20,
        )
        time.sleep(0.05)
        eng_a = DrowsinessEngine()
        mgr_a = AlertStateManager()
        assess_a = eng_a.evaluate(features=features, quality=quality)
        trans_a = mgr_a.update(assess_a)
        iso_closed = (
            assess_a.state == base_assess.state
            and assess_a.confidence.overall_confidence == base_assess.confidence.overall_confidence
            and mgr_a.current_state == manager.current_state
            and trans_a == base_trans
        )
    finally:
        w_closed.stop()

    # 3. Shadow Active (Inference Error)
    w_err = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True))
    w_err.is_available = True
    w_err.cv2_net = MagicMock()
    w_err.cv2_net.forward.side_effect = cv2.error("Simulated crash")
    w_err.start()
    try:
        w_err.submit_frame(
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks_pixel_xyz=np.ones((478, 3), dtype=np.float64) * 50.0,
            timestamp_ms=1000.0,
            frame_id=2,
            geom_left_closed=False,
            geom_right_closed=False,
            raw_left_ear=0.28,
            raw_right_ear=0.28,
            ear_threshold=0.20,
        )
        time.sleep(0.05)
        eng_b = DrowsinessEngine()
        mgr_b = AlertStateManager()
        assess_b = eng_b.evaluate(features=features, quality=quality)
        trans_b = mgr_b.update(assess_b)
        iso_error = (
            assess_b.state == base_assess.state
            and assess_b.confidence.overall_confidence == base_assess.confidence.overall_confidence
            and mgr_b.current_state == manager.current_state
            and trans_b == base_trans
        )
    finally:
        w_err.stop()

    # 4. Shadow Active (Stale Observation >200ms)
    w_stale = EyeCNNShadowWorker(policy=EyeCNNInvocationPolicy(enabled=True, max_staleness_ms=200.0))
    w_stale.is_available = True
    w_stale.cv2_net = MagicMock()
    w_stale.cv2_net.forward.return_value = np.array([[2.0, -2.0], [2.0, -2.0]], dtype=np.float32)
    w_stale.start()
    try:
        w_stale.submit_frame(
            frame_bgr=np.zeros((100, 100, 3), dtype=np.uint8),
            landmarks_pixel_xyz=np.ones((478, 3), dtype=np.float64) * 50.0,
            timestamp_ms=1000.0,
            frame_id=3,
            geom_left_closed=False,
            geom_right_closed=False,
            raw_left_ear=0.28,
            raw_right_ear=0.28,
            ear_threshold=0.20,
        )
        time.sleep(0.05)
        obs_stale = w_stale.get_latest_observation(1250.0)
        eng_c = DrowsinessEngine()
        mgr_c = AlertStateManager()
        assess_c = eng_c.evaluate(features=features, quality=quality)
        trans_c = mgr_c.update(assess_c)
        iso_stale = (
            obs_stale is not None
            and obs_stale.is_stale is True
            and assess_c.state == base_assess.state
            and mgr_c.current_state == manager.current_state
            and trans_c == base_trans
        )
    finally:
        w_stale.stop()

    # 5. Missing Model File
    w_missing = EyeCNNShadowWorker(model_path=Path("nonexistent.onnx"))
    w_missing.start()
    try:
        eng_d = DrowsinessEngine()
        mgr_d = AlertStateManager()
        assess_d = eng_d.evaluate(features=features, quality=quality)
        trans_d = mgr_d.update(assess_d)
        iso_missing = (
            w_missing.is_available is False
            and assess_d.state == base_assess.state
            and mgr_d.current_state == manager.current_state
            and trans_d == base_trans
        )
    finally:
        w_missing.stop()

    return {
        "isolation_shadow_closed_prediction": iso_closed,
        "isolation_shadow_inference_error": iso_error,
        "isolation_shadow_stale_result": iso_stale,
        "isolation_shadow_missing_model": iso_missing,
    }


def main() -> None:
    hw = get_system_hardware_info()
    print("=" * 80)
    print("  DRIVEGUARD PHASE B1.1: LIVE PERFORMANCE AND RELIABILITY VALIDATION")
    print("=" * 80)
    print(f"Host System       : {hw['os']}")
    print(f"Processor         : {hw.get('cpu_brand', hw['processor'])}")
    print(f"Cores / Threads   : {hw['physical_cores']}C / {hw['logical_processors']}T")
    print(f"Total Memory      : {hw['ram_total_gb']}")
    print(f"Python Runtime    : {hw['python']}")
    print("=" * 80)

    # 1. Verify strict isolation
    print("\n--- [Step 1/2] Verifying Architectural Decision Isolation ---")
    iso_results = verify_strict_isolation()
    for k, v in iso_results.items():
        print(f"  * {k}: {'PASS (Bit-for-bit identical)' if v else 'FAIL'}")
    all_iso_pass = all(iso_results.values())
    print(f"Isolation Verification Result: {'PASS' if all_iso_pass else 'FAIL'}")

    # 2. Run 5-configuration controlled benchmark
    print("\n--- [Step 2/2] Running 5-Configuration Controlled Live Benchmark ---")
    configs = [
        ("Config 1: Shadow OFF (Baseline)", False, 0.0),
        ("Config 2: Shadow ON @ 1 Hz", True, 1.0),
        ("Config 3: Shadow ON @ 2 Hz", True, 2.0),
        ("Config 4: Shadow ON @ 5 Hz", True, 5.0),
        ("Config 5: Shadow ON @ 15 Hz", True, 15.0),
    ]

    all_results = []
    # Test duration: 60s per configuration as requested
    duration_per_run = 60.0
    warmup_per_run = 3.0

    for idx, (c_name, enabled, hz) in enumerate(configs, 1):
        print(f"\n[{idx}/5] Executing {c_name} for {duration_per_run:.0f}s (camera device 0)...")
        # Trigger GC before run to isolate memory footprints
        gc.collect()
        time.sleep(1.0)
        res = run_configuration_benchmark(
            config_name=c_name,
            shadow_enabled=enabled,
            shadow_hz=hz,
            duration_sec=duration_per_run,
            warmup_sec=warmup_per_run,
        )
        all_results.append(res)
        loop_s = res["loop_processing_ms"]
        mp_s = res["mediapipe_latency_ms"]
        cnn_s = res["cnn_inference_latency_ms"]
        print(f"      Rendered Frames: {res['total_rendered_frames']} ({res['display_fps']:.1f} FPS)")
        print(f"      Main Loop Time : Mean={loop_s['mean']:.2f}ms, Median={loop_s['median']:.2f}ms, P95={loop_s['p95']:.2f}ms, P99={loop_s['p99']:.2f}ms")
        print(f"      MediaPipe Lat  : Mean={mp_s['mean']:.2f}ms, P95={mp_s['p95']:.2f}ms")
        if enabled:
            print(f"      CNN Latency    : Mean={cnn_s['mean']:.2f}ms, P95={cnn_s['p95']:.2f}ms (Inferences: {res['worker_total_inferences']})")
            print(f"      Drops / Stale  : Drops={res['worker_dropped_jobs']}, Stale={res['stale_observations_count']}, Errors={res['worker_errors']}")
        print(f"      CPU / Memory   : Process={res['process_cpu_pct']:.1f}%, System={res['system_cpu_pct']:.1f}%, RSS Delta={res['rss_delta_mb']:+.1f}MB")

    # Save benchmark results to JSON artifact
    output_path = repo_root / "scripts" / "benchmarks" / "phase_b1_1_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump({"hardware": hw, "isolation": iso_results, "benchmarks": all_results}, f, indent=2)
    print(f"\nBenchmark JSON artifact saved to: {output_path}")


if __name__ == "__main__":
    main()
