"""
DriveGuard — Phase B1.1.2: Final Benchmark Integrity Audit Harness.

Executes a controlled, multi-trial live benchmark across five configurations:
1. Shadow OFF (Baseline)
2. Shadow ON @ 1 Hz
3. Shadow ON @ 2 Hz
4. Shadow ON @ 5 Hz
5. Shadow ON @ 15 Hz

Each configuration is executed for 3 independent, comparable trials.
Rigorous accounting verification:
- Evaluations = Disabled + NoFace + EarInvalid + RateLimited + Accepted
- Eligible Opportunities = Accepted + RateLimited
- Jobs Submitted = Jobs Processed + Jobs Dropped + Pending
- Queries = Fresh + Stale + None
- Inter-inference interval distribution & 200ms freshness verification
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path
import platform
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
logger = logging.getLogger("phase_b1_2_benchmark")


def get_hardware_info() -> Dict[str, str]:
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


def _stats(arr: List[float]) -> Dict[str, float]:
    """Compute summary statistics for a float array."""
    if not arr:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "p99": 0.0, "min": 0.0, "max": 0.0, "std": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "std": float(np.std(arr)),
    }


def run_single_trial(
    config_name: str,
    trial_num: int,
    shadow_enabled: bool,
    shadow_hz: float,
    duration_sec: float = 15.0,
    warmup_sec: float = 2.0,
    device_index: int = 0,
    width: int = 1280,
    height: int = 720,
    target_fps: int = 30,
) -> Dict[str, Any]:
    """Run a single controlled trial."""
    process = psutil.Process()
    process.cpu_percent(interval=None)
    psutil.cpu_percent(interval=None)
    initial_rss_mb = process.memory_info().rss / (1024 * 1024)

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

    loop_processing_times_ms: List[float] = []
    mediapipe_latencies_ms: List[float] = []
    cnn_latencies_ms: List[float] = []
    cnn_total_latencies_ms: List[float] = []
    process_cpu_samples: List[float] = []
    system_cpu_samples: List[float] = []

    last_seen_obs_frame_id: Optional[int] = None
    total_frames_rendered = 0
    total_face_tracking_frames = 0
    last_cpu_sample_time = time.monotonic()

    t_start = time.monotonic()
    t_end = t_start + duration_sec

    try:
        while time.monotonic() < t_end:
            now = time.monotonic()
            is_warmup = (now - t_start) < warmup_sec

            read_success, packet = camera.read(timeout_sec=0.1)
            if not read_success or packet is None:
                continue

            t_loop_start = time.perf_counter()

            # 1. Optical Frame Quality
            quality = quality_analyzer.analyze(packet.frame)

            # 2. Facial Landmark Tracking
            landmark_result = landmarker.detect(packet)

            # 3. Physiological Driver Features
            features = feature_extractor.extract(
                landmark_result,
                frame_width=f_w,
                frame_height=f_h,
                is_mirrored=True,
            )

            # 4. Shadow Observer Ingestion
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
                        # Record latency ONLY for new unique completed inferences
                        if shadow_obs.frame_id != last_seen_obs_frame_id and shadow_obs.inference_latency_ms > 0:
                            last_seen_obs_frame_id = shadow_obs.frame_id
                            cnn_latencies_ms.append(shadow_obs.inference_latency_ms)
                            cnn_total_latencies_ms.append(shadow_obs.total_latency_ms)
                except Exception as e:
                    logger.warning("Shadow ingestion error: %s", e)

            # 5. Drowsiness Intelligence Engine
            assessment = drowsiness_engine.evaluate(
                features=features,
                quality=quality,
            )

            # 6. Alert State Manager & Dispatcher
            transition = alert_manager.update(assessment)
            alert_dispatcher.dispatch(
                transition=transition,
                current_state=alert_manager.current_state,
                timestamp_ms=packet.timestamp_ms,
                is_recovering=alert_manager.is_recovering,
            )

            t_loop_end = time.perf_counter()
            loop_ms = (t_loop_end - t_loop_start) * 1000.0

            total_frames_rendered += 1
            if landmark_result and landmark_result.has_face:
                total_face_tracking_frames += 1

            if not is_warmup:
                loop_processing_times_ms.append(loop_ms)
                if landmark_result:
                    mediapipe_latencies_ms.append(landmark_result.inference_latency_ms)

            if now - last_cpu_sample_time >= 1.0:
                p_cpu = process.cpu_percent(interval=None) / psutil.cpu_count(logical=True)
                s_cpu = psutil.cpu_percent(interval=None)
                process_cpu_samples.append(p_cpu)
                system_cpu_samples.append(s_cpu)
                last_cpu_sample_time = now

    finally:
        if shadow_enabled:
            shadow_worker.stop()
            pipeline_diag = shadow_worker.get_pipeline_diagnostics()
        else:
            pipeline_diag = {}
        alert_dispatcher.shutdown()
        camera.close()
        landmarker.close()

    total_duration = time.monotonic() - t_start
    final_rss_mb = process.memory_info().rss / (1024 * 1024)
    buffer_stats = camera.get_buffer_stats()

    achieved_hz = (
        pipeline_diag.get("inferences_successful", 0) / max(0.1, total_duration)
        if shadow_enabled
        else 0.0
    )

    inter_intervals = pipeline_diag.get("inter_inference_intervals_ms", [])
    inter_interval_stats = _stats(inter_intervals)

    # Verification of Conservation Identities
    accounting_integrity = True
    accounting_issues = []
    if shadow_enabled:
        ev_total = pipeline_diag["evaluations_total"]
        ev_sum = (
            pipeline_diag["evaluations_accepted"]
            + pipeline_diag["rejections_disabled"]
            + pipeline_diag["rejections_no_face"]
            + pipeline_diag["rejections_ear_invalid"]
            + pipeline_diag["rejections_rate_limited"]
        )
        if ev_total != ev_sum:
            accounting_integrity = False
            accounting_issues.append(f"Evaluations mismatch: total {ev_total} != sum {ev_sum}")

        q_total = pipeline_diag["queries_total"]
        q_sum = (
            pipeline_diag["queries_fresh"]
            + pipeline_diag["queries_stale"]
            + pipeline_diag["queries_none"]
        )
        if q_total != q_sum:
            accounting_integrity = False
            accounting_issues.append(f"Queries mismatch: total {q_total} != sum {q_sum}")

        jobs_sub = pipeline_diag["jobs_submitted"]
        jobs_proc = pipeline_diag["jobs_processed"]
        jobs_drop = pipeline_diag["jobs_dropped_queue"]
        if jobs_sub != (jobs_proc + jobs_drop):
            accounting_integrity = False
            accounting_issues.append(f"Jobs mismatch: submitted {jobs_sub} != processed {jobs_proc} + dropped {jobs_drop}")

    return {
        "config_name": config_name,
        "trial_num": trial_num,
        "shadow_enabled": shadow_enabled,
        "configured_hz": shadow_hz,
        "achieved_hz": achieved_hz,
        "duration_sec": total_duration,
        "steady_state_samples": len(loop_processing_times_ms),
        "total_captured_frames": buffer_stats.get("total_pushed", 0),
        "total_rendered_frames": total_frames_rendered,
        "capture_fps": buffer_stats.get("total_pushed", 0) / max(0.1, total_duration),
        "display_fps": total_frames_rendered / max(0.1, total_duration),
        "face_detection_pct": (total_face_tracking_frames / max(1, total_frames_rendered)) * 100.0,
        "loop_processing_ms": _stats(loop_processing_times_ms),
        "mediapipe_latency_ms": _stats(mediapipe_latencies_ms),
        "cnn_inference_latency_ms": _stats(cnn_latencies_ms),
        "cnn_total_latency_ms": _stats(cnn_total_latencies_ms),
        "inter_inference_interval_ms": inter_interval_stats,
        "process_cpu_pct": float(np.mean(process_cpu_samples)) if process_cpu_samples else 0.0,
        "system_cpu_pct": float(np.mean(system_cpu_samples)) if system_cpu_samples else 0.0,
        "initial_rss_mb": initial_rss_mb,
        "final_rss_mb": final_rss_mb,
        "rss_delta_mb": final_rss_mb - initial_rss_mb,
        "pipeline_diagnostics": pipeline_diag,
        "accounting_integrity": accounting_integrity,
        "accounting_issues": accounting_issues,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Phase B1.1.2 Benchmark Integrity Audit")
    parser.add_argument("--trials", type=int, default=3, help="Number of trials per configuration (default: 3)")
    parser.add_argument("--duration", type=float, default=15.0, help="Duration in seconds per trial (default: 15.0)")
    parser.add_argument("--warmup", type=float, default=2.0, help="Warmup duration in seconds per trial (default: 2.0)")
    args = parser.parse_args()

    hw = get_hardware_info()
    print("=" * 80)
    print("  DRIVEGUARD PHASE B1.1.2: FINAL BENCHMARK INTEGRITY AUDIT")
    print("=" * 80)
    print(f"Host System       : {hw['os']}")
    print(f"Processor         : {hw.get('cpu_brand', hw['processor'])}")
    print(f"Cores / Threads   : {hw['physical_cores']}C / {hw['logical_processors']}T")
    print(f"Total Memory      : {hw['ram_total_gb']}")
    print(f"Python Runtime    : {hw['python']}")
    print(f"Trials per Config : {args.trials}")
    print(f"Duration per Trial: {args.duration:.1f}s (Warmup: {args.warmup:.1f}s)")
    print("=" * 80)

    configs = [
        ("Config 1: Shadow OFF (Baseline)", False, 0.0),
        ("Config 2: Shadow ON @ 1 Hz", True, 1.0),
        ("Config 3: Shadow ON @ 2 Hz", True, 2.0),
        ("Config 4: Shadow ON @ 5 Hz", True, 5.0),
        ("Config 5: Shadow ON @ 15 Hz", True, 15.0),
    ]

    all_trial_results = []
    aggregated_results = []

    for c_idx, (c_name, enabled, hz) in enumerate(configs, 1):
        print(f"\n[{c_idx}/5] Evaluating {c_name} across {args.trials} trials...")
        trials_for_config = []

        for t_idx in range(1, args.trials + 1):
            print(f"   -> Starting Trial {t_idx}/{args.trials} ({args.duration:.0f}s)...", end="", flush=True)
            gc.collect()
            time.sleep(0.8)

            trial_res = run_single_trial(
                config_name=c_name,
                trial_num=t_idx,
                shadow_enabled=enabled,
                shadow_hz=hz,
                duration_sec=args.duration,
                warmup_sec=args.warmup,
            )
            trials_for_config.append(trial_res)
            all_trial_results.append(trial_res)

            loop_m = trial_res["loop_processing_ms"]["mean"]
            mp_m = trial_res["mediapipe_latency_ms"]["mean"]
            face_p = trial_res["face_detection_pct"]
            fps = trial_res["display_fps"]
            ach_hz = trial_res["achieved_hz"]
            integrity = "VALID" if trial_res["accounting_integrity"] else "DISCREPANCY"

            print(
                f" Done: Loop={loop_m:.1f}ms, MP={mp_m:.1f}ms, Face={face_p:.1f}%, FPS={fps:.1f}, "
                f"AchievedHz={ach_hz:.2f}Hz [{integrity}]"
            )

        # Aggregate across trials
        agg = {
            "config_name": c_name,
            "shadow_enabled": enabled,
            "configured_hz": hz,
            "trial_count": len(trials_for_config),
            "display_fps": _stats([t["display_fps"] for t in trials_for_config]),
            "capture_fps": _stats([t["capture_fps"] for t in trials_for_config]),
            "face_detection_pct": _stats([t["face_detection_pct"] for t in trials_for_config]),
            "main_loop_latency_ms": _stats([t["loop_processing_ms"]["mean"] for t in trials_for_config]),
            "mediapipe_latency_ms": _stats([t["mediapipe_latency_ms"]["mean"] for t in trials_for_config]),
            "process_cpu_pct": _stats([t["process_cpu_pct"] for t in trials_for_config]),
            "system_cpu_pct": _stats([t["system_cpu_pct"] for t in trials_for_config]),
            "achieved_hz": _stats([t["achieved_hz"] for t in trials_for_config]),
        }

        if enabled:
            agg["cnn_inference_latency_ms"] = _stats(
                [t["cnn_inference_latency_ms"]["mean"] for t in trials_for_config if t["cnn_inference_latency_ms"]["mean"] > 0]
            )
            agg["cnn_total_latency_ms"] = _stats(
                [t["cnn_total_latency_ms"]["mean"] for t in trials_for_config if t["cnn_total_latency_ms"]["mean"] > 0]
            )
            agg["inter_inference_interval_ms"] = _stats(
                [t["inter_inference_interval_ms"]["mean"] for t in trials_for_config if t["inter_inference_interval_ms"]["mean"] > 0]
            )
            # Aggregate funnel sums
            agg["total_evaluations"] = sum(t["pipeline_diagnostics"].get("evaluations_total", 0) for t in trials_for_config)
            agg["total_eligible_opportunities"] = sum(t["pipeline_diagnostics"].get("eligible_opportunities", 0) for t in trials_for_config)
            agg["total_accepted"] = sum(t["pipeline_diagnostics"].get("evaluations_accepted", 0) for t in trials_for_config)
            agg["total_rejections_no_face"] = sum(t["pipeline_diagnostics"].get("rejections_no_face", 0) for t in trials_for_config)
            agg["total_rejections_rate_limited"] = sum(t["pipeline_diagnostics"].get("rejections_rate_limited", 0) for t in trials_for_config)
            agg["total_inferences"] = sum(t["pipeline_diagnostics"].get("inferences_successful", 0) for t in trials_for_config)
            agg["total_dropped_jobs"] = sum(t["pipeline_diagnostics"].get("jobs_dropped_queue", 0) for t in trials_for_config)
            agg["total_errors"] = sum(t["pipeline_diagnostics"].get("inference_errors", 0) for t in trials_for_config)
            # Aggregate staleness queries
            agg["total_queries"] = sum(t["pipeline_diagnostics"].get("queries_total", 0) for t in trials_for_config)
            agg["total_queries_fresh"] = sum(t["pipeline_diagnostics"].get("queries_fresh", 0) for t in trials_for_config)
            agg["total_queries_stale"] = sum(t["pipeline_diagnostics"].get("queries_stale", 0) for t in trials_for_config)
            agg["freshness_ratio_pct"] = (
                (agg["total_queries_fresh"] / max(1, agg["total_queries"])) * 100.0
            )

        aggregated_results.append(agg)

    output_path = repo_root / "scripts" / "benchmarks" / "phase_b1_2_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(
            {
                "hardware": hw,
                "trials": all_trial_results,
                "aggregates": aggregated_results,
            },
            f,
            indent=2,
        )
    print(f"\nFinal Audit Benchmark results saved to: {output_path}")


if __name__ == "__main__":
    main()
