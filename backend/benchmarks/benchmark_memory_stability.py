"""
DriveGuard Phase 7 Stage 4: Memory Stability & Resource Profiling Benchmark.

Evaluates long-duration memory stability, resource reclamation, and lifecycle hygiene:
  Experiment A: Continuous 10-minute (15,000 frames at 25 FPS) paced real-time session.
  Experiment B: 10 repeated session lifecycles (150 frames each + teardown + GC settling).
  Experiment C: 60-second post-load idle observation.
  Experiment D: Mid-stream cancellation, unsubscription, and failure cleanup.

Metrics & Profiling:
  - Process RSS, VMS, CPU utilization (psutil)
  - Process thread count and OS handles/file descriptors
  - Python heap allocation and peak tracing (tracemalloc)
  - Container cgroup memory limits & usage (cgroup v2 / v1 discovery)
  - OLS regression slope & 95% Confidence Interval for steady-state memory growth
  - LiveKit frame accounting conservation invariant verification
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import cv2
import numpy as np
import psutil
from scipy import stats

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.services.livekit_worker import (
    FrameAccountingSnapshot,
    LiveKitAIWorkerSession,
    normalize_frame_width,
)
from backend.app.services.session_manager import session_manager
from backend.benchmarks.benchmark_inference_offline import generate_benchmark_frame


logger = logging.getLogger("benchmark_memory_stability")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


class MockVideoFrame:
    """Mock WebRTC VideoFrame wrapping numpy RGB buffer for memory simulation."""

    def __init__(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.width = img_bgr.shape[1]
        self.height = img_bgr.shape[0]
        self._data = img_rgb.tobytes()

    def convert(self, buffer_type: Any) -> MockVideoFrame:
        return self

    @property
    def data(self) -> bytes:
        return self._data


class NullDataPublisher:
    """
    Mock WebRTC local participant data channel that extracts timing metrics for throughput
    and latency distribution analysis and discards payload bytes without unbounded mock call history accumulation.
    """

    def __init__(self) -> None:
        self.published_count = 0
        self.infer_durations_ms: List[float] = []

    async def publish_data(self, payload: bytes, reliable: bool = False) -> None:
        self.published_count += 1
        try:
            msg = json.loads(payload)
            infer_ms = msg.get("telemetry", {}).get("timing", {}).get("infer_duration_ms")
            if infer_ms is not None:
                self.infer_durations_ms.append(float(infer_ms))
        except Exception:
            pass


def get_git_commit_hash() -> str:
    """Retrieve current git HEAD commit hash or fallback string."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        return res.stdout.strip()
    except Exception:
        return "UNKNOWN_COMMIT"


def get_cgroup_memory_info() -> Dict[str, Any]:
    """
    Discover container cgroup memory usage and limit by inspecting process cgroup membership.
    Supports both cgroup v2 and cgroup v1.
    Provides graceful, explicit detection for non-Linux or non-containerized environments (e.g. Windows).
    """
    info: Dict[str, Any] = {
        "available": False,
        "cgroup_version": None,
        "usage_bytes": None,
        "usage_mb": None,
        "limit_bytes": None,
        "limit_mb": None,
        "details": "",
    }
    if not sys.platform.startswith("linux"):
        info["details"] = f"Non-Linux platform ({sys.platform}); cgroups unavailable."
        return info

    try:
        cgroup_path = Path("/proc/self/cgroup")
        if not cgroup_path.exists():
            info["details"] = "/proc/self/cgroup does not exist."
            return info

        # Check cgroup v2
        cg2_usage_file = Path("/sys/fs/cgroup/memory.current")
        cg2_limit_file = Path("/sys/fs/cgroup/memory.max")
        if cg2_usage_file.exists():
            usage_bytes = int(cg2_usage_file.read_text().strip())
            limit_str = cg2_limit_file.read_text().strip() if cg2_limit_file.exists() else "max"
            limit_bytes = int(limit_str) if limit_str.isdigit() else None
            info.update({
                "available": True,
                "cgroup_version": "v2",
                "usage_bytes": usage_bytes,
                "usage_mb": round(usage_bytes / (1024 * 1024), 2),
                "limit_bytes": limit_bytes,
                "limit_mb": round(limit_bytes / (1024 * 1024), 2) if limit_bytes else None,
                "details": "cgroup v2 active",
            })
            return info

        # Check cgroup v1
        cg1_usage_file = Path("/sys/fs/cgroup/memory/memory.usage_in_bytes")
        cg1_limit_file = Path("/sys/fs/cgroup/memory/memory.limit_in_bytes")
        if cg1_usage_file.exists():
            usage_bytes = int(cg1_usage_file.read_text().strip())
            limit_bytes = int(cg1_limit_file.read_text().strip()) if cg1_limit_file.exists() else None
            if limit_bytes and limit_bytes > (1 << 60):
                limit_bytes = None
            info.update({
                "available": True,
                "cgroup_version": "v1",
                "usage_bytes": usage_bytes,
                "usage_mb": round(usage_bytes / (1024 * 1024), 2),
                "limit_bytes": limit_bytes,
                "limit_mb": round(limit_bytes / (1024 * 1024), 2) if limit_bytes else None,
                "details": "cgroup v1 active",
            })
            return info

        info["details"] = "Linux host without mounted cgroup memory controller."
        return info
    except Exception as exc:
        info["details"] = f"Error inspecting cgroup: {exc}"
        return info


def capture_resource_sample(
    proc: psutil.Process,
    t_start: float,
    sample_index: int,
    stage_label: str = "",
    run_gc: bool = True,
) -> Dict[str, Any]:
    """Capture unified system and runtime resource metrics at a discrete point in time."""
    t_now = time.perf_counter()
    elapsed_sec = t_now - t_start

    # Force generation cycle collection so tracemalloc measures true retained live memory
    # rather than transient uncollected generational cycles between frame allocations
    if run_gc:
        gc.collect()

    # OS memory & CPU
    mem_info = proc.memory_info()
    rss_mb = mem_info.rss / (1024 * 1024)
    vms_mb = mem_info.vms / (1024 * 1024)
    cpu_pct = proc.cpu_percent(interval=None)
    thread_count = proc.num_threads()
    handles_or_fds = proc.num_handles() if hasattr(proc, "num_handles") else proc.num_fds()

    # Python runtime heap
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    heap_current_mb = traced_current / (1024 * 1024)
    heap_peak_mb = traced_peak / (1024 * 1024)

    return {
        "sample_index": sample_index,
        "stage": stage_label,
        "elapsed_sec": round(elapsed_sec, 2),
        "rss_mb": round(rss_mb, 2),
        "vms_mb": round(vms_mb, 2),
        "cpu_pct": round(cpu_pct, 1),
        "thread_count": thread_count,
        "handles_or_fds": handles_or_fds,
        "heap_current_mb": round(heap_current_mb, 2),
        "heap_peak_mb": round(heap_peak_mb, 2),
    }


def compute_ols_regression(
    x_vals: List[float],
    y_vals: List[float],
    x_unit_scale: float = 60.0,
    hac_lag: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Perform Ordinary Least Squares (OLS) linear regression and compute both:
    1. Conventional OLS standard error and 95% Confidence Interval (assuming i.i.d. errors).
    2. Newey-West Heteroskedasticity and Autocorrelation Consistent (HAC) standard error
       and autocorrelation-aware 95% Confidence Interval using Bartlett kernel weights.

    x_unit_scale: Divisor for x units (default 60.0 converts elapsed seconds to minutes).
    hac_lag: Maximum lag truncation for Bartlett kernel. Defaults to floor(4 * (N / 100)^(2/9)).
    """
    n = len(x_vals)
    if n < 3:
        return {
            "n_samples": n,
            "slope_per_min": 0.0,
            "intercept": 0.0,
            "r_squared": 0.0,
            "p_value": 1.0,
            "std_err": 0.0,
            "slope_ci_95": [0.0, 0.0],
            "hac": {
                "method": "Newey-West HAC (Bartlett kernel)",
                "lag_truncation": 0,
                "autocorrelation_lag1": 0.0,
                "std_err_hac": 0.0,
                "slope_ci_95_hac": [0.0, 0.0],
            },
        }

    x_arr = np.array(x_vals, dtype=np.float64) / x_unit_scale
    y_arr = np.array(y_vals, dtype=np.float64)

    # Conventional OLS estimation via linregress
    res = stats.linregress(x_arr, y_arr)
    slope = float(res.slope)
    intercept = float(res.intercept)
    r_squared = float(res.rvalue ** 2)
    p_val = float(res.pvalue)
    std_err_ols = float(res.stderr) if res.stderr is not None else 0.0

    df = n - 2
    t_crit = float(stats.t.ppf(0.975, df=df)) if df > 0 else 1.96
    ci_lower_ols = slope - (t_crit * std_err_ols)
    ci_upper_ols = slope + (t_crit * std_err_ols)

    # Design matrix Z: [1, x_i]
    Z = np.column_stack([np.ones(n, dtype=np.float64), x_arr])
    # Fitted values and residuals
    y_hat = intercept + slope * x_arr
    e = y_arr - y_hat

    # Residual first-order autocorrelation
    var_e = float(np.sum(e ** 2))
    rho_1 = float(np.sum(e[1:] * e[:-1]) / var_e) if var_e > 1e-12 else 0.0

    # Newey-West HAC Covariance Estimation
    # Automatic lag selection rule of thumb: L = floor(4 * (N / 100)^(2/9))
    if hac_lag is None:
        L = max(1, int(np.floor(4.0 * ((n / 100.0) ** (2.0 / 9.0)))))
    else:
        L = max(1, hac_lag)

    try:
        ZtZ_inv = np.linalg.inv(Z.T @ Z)
        # Gamma_0 = sum(e_t^2 * z_t * z_t^T)
        Omega = np.zeros((2, 2), dtype=np.float64)
        for t in range(n):
            z_t = Z[t : t + 1, :]
            Omega += (e[t] ** 2) * (z_t.T @ z_t)

        # Autocovariance terms with Bartlett kernel w_j = 1 - j / (L + 1)
        for j in range(1, min(L + 1, n)):
            w_j = 1.0 - (float(j) / float(L + 1))
            Gamma_j = np.zeros((2, 2), dtype=np.float64)
            for t in range(j, n):
                z_t = Z[t : t + 1, :]
                z_tj = Z[t - j : t - j + 1, :]
                Gamma_j += (e[t] * e[t - j]) * (z_t.T @ z_tj)
            Omega += w_j * (Gamma_j + Gamma_j.T)

        V_hac = ZtZ_inv @ Omega @ ZtZ_inv
        std_err_hac = float(np.sqrt(max(0.0, V_hac[1, 1])))
    except Exception as exc:
        logger.warning(f"Error computing HAC covariance: {exc}; falling back to OLS std error")
        std_err_hac = std_err_ols
        L = 0

    ci_lower_hac = slope - (t_crit * std_err_hac)
    ci_upper_hac = slope + (t_crit * std_err_hac)

    return {
        "n_samples": n,
        "slope_per_min": round(slope, 5),
        "intercept": round(intercept, 4),
        "r_squared": round(r_squared, 4),
        "p_value": round(p_val, 6),
        "std_err": round(std_err_ols, 6),
        "slope_ci_95": [round(ci_lower_ols, 5), round(ci_upper_ols, 5)],
        "hac": {
            "method": "Newey-West HAC (Bartlett kernel)",
            "lag_truncation": L,
            "autocorrelation_lag1": round(rho_1, 4),
            "std_err_hac": round(std_err_hac, 6),
            "slope_ci_95_hac": [round(ci_lower_hac, 5), round(ci_upper_hac, 5)],
        },
    }


def classify_memory_stability(
    ols_res: Dict[str, Any],
    resource_type: str = "heap",
) -> Dict[str, Any]:
    """
    Classify memory stability using autocorrelation-aware Newey-West HAC confidence intervals:
    - STABLE_PLATEAU: Evidence supports no practically meaningful positive growth under defined limits.
      (slope <= 0.0, or slope > 0 with HAC CI spanning zero and upper bound <= stability tolerance)
    - ACTIVE_LEAK: Statistically supported positive growth (HAC CI lower bound strictly > 0, R^2 >= 0.50).
    - INCONCLUSIVE: Insufficient evidence, wide HAC CI, positive slope with high noise. Requires extended run.
    """
    slope = ols_res.get("slope_per_min", 0.0)
    hac_info = ols_res.get("hac", {})
    ci_hac = hac_info.get("slope_ci_95_hac", ols_res.get("slope_ci_95", [0.0, 0.0]))
    ci_lower, ci_upper = ci_hac[0], ci_hac[1]
    std_err_hac = hac_info.get("std_err_hac", ols_res.get("std_err", 0.0))
    r2 = ols_res.get("r_squared", 0.0)
    rho_1 = hac_info.get("autocorrelation_lag1", 0.0)

    # Tolerance threshold for upper CI bound under noise: 0.05 MB/min for heap, 0.50 MB/min for RSS
    tolerance_upper = 0.05 if resource_type == "heap" else 0.50

    if slope <= 0.0:
        classification = "STABLE_PLATEAU"
        is_pass = True
        reason = (
            f"Estimated slope is non-positive ({slope:.4f} MB/min); memory is contracting or on a flat plateau "
            f"(HAC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], lag-1 autocorrelation={rho_1:.3f})."
        )
    elif ci_lower <= 0.0 and ci_upper <= tolerance_upper:
        classification = "STABLE_PLATEAU"
        is_pass = True
        reason = (
            f"Estimated slope ({slope:.4f} MB/min) is statistically indistinguishable from zero at alpha=0.05 "
            f"(HAC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}] spans zero; upper bound <= {tolerance_upper} MB/min tolerance). "
            f"Note: This does not prove zero growth, but confirms that any true drift is bounded within the allocator compaction envelope."
        )
    elif ci_lower > 0.0 and r2 >= 0.50 and slope > 0.05:
        classification = "ACTIVE_LEAK"
        is_pass = False
        reason = (
            f"Statistically supported positive growth confirmed under autocorrelation-aware HAC estimator "
            f"(HAC 95% CI: [{ci_lower:.4f}, {ci_upper:.4f}], R^2={r2:.3f}, slope={slope:.4f} MB/min)."
        )
    else:
        classification = "INCONCLUSIVE"
        is_pass = False
        reason = (
            f"Inconclusive measurement: slope={slope:.4f} MB/min with uncertain HAC 95% CI [{ci_lower:.4f}, {ci_upper:.4f}] "
            f"(R^2={r2:.3f}, std_err_hac={std_err_hac:.4f}). Extended measurement run required."
        )

    return {
        "classification": classification,
        "is_acceptable": is_pass,
        "reason": reason,
        "slope": slope,
        "ci_95_hac": [ci_lower, ci_upper],
        "std_err_hac": std_err_hac,
        "r_squared": r2,
        "autocorrelation_lag1": rho_1,
    }


async def run_experiment_a_continuous(
    proc: psutil.Process,
    duration_seconds: float = 600.0,
    target_fps: float = 25.0,
    sample_interval_sec: float = 5.0,
    warmup_discard_sec: float = 60.0,
) -> Dict[str, Any]:
    """
    Experiment A: Long-duration continuous paced session (15,000 frames at 25 FPS).
    Records periodic memory and thread snapshots. Computes OLS growth slope across steady-state window.
    """
    logger.info(
        f"[Exp A] Starting continuous session: duration={duration_seconds}s, "
        f"target_fps={target_fps} (total target frames={int(duration_seconds * target_fps)})"
    )

    room_name = f"bench_mem_exp_a_{int(time.time())}"
    worker = LiveKitAIWorkerSession(
        room_name=room_name,
        driver_id="driver_bench_mem_a",
        experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
    )
    worker.room = MagicMock()
    worker.room.disconnect = AsyncMock()
    publisher = NullDataPublisher()
    worker.room.local_participant = publisher
    worker.is_running = True

    worker.engine = await session_manager.get_or_create_session(
        session_id=room_name,
        driver_id="driver_bench_mem_a",
        experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
    )
    worker._inference_task = asyncio.create_task(worker._inference_loop())

    frame_interval = 1.0 / target_fps
    total_frames = int(duration_seconds * target_fps)

    samples: List[Dict[str, Any]] = []
    t_start = time.perf_counter()
    next_sample_time = t_start

    # Initial sample
    samples.append(capture_resource_sample(proc, t_start, len(samples), "EXP_A_START"))

    for frame_idx in range(total_frames):
        t_frame_start = time.perf_counter()

        img_bgr = generate_benchmark_frame(frame_idx)
        mock_frame = MockVideoFrame(img_bgr)
        ingest_mono_ms = time.perf_counter() * 1000.0

        worker._frames_received += 1
        worker._frame_seq += 1

        if worker.frame_queue.full():
            try:
                worker.frame_queue.get_nowait()
                worker._frames_replaced += 1
            except asyncio.QueueEmpty:
                pass

        await worker.frame_queue.put((mock_frame, ingest_mono_ms, worker._frame_seq))

        now = time.perf_counter()
        if now >= next_sample_time:
            sample = capture_resource_sample(proc, t_start, len(samples), "EXP_A_PACED")
            snapshot = worker.get_frame_accounting_snapshot()
            sample.update({
                "frames_received": snapshot.frames_received,
                "frames_processed": snapshot.frames_processed,
                "frames_replaced": snapshot.frames_replaced,
                "frames_in_flight": snapshot.frames_in_flight,
                "invariant_valid": snapshot.is_invariant_satisfied,
            })
            samples.append(sample)
            next_sample_time = now + sample_interval_sec

        # Precise pacing
        elapsed_frame = time.perf_counter() - t_frame_start
        sleep_dur = frame_interval - elapsed_frame
        if sleep_dur > 0:
            await asyncio.sleep(sleep_dur)

    # 1. Capture running-state snapshot before shutdown
    running_snapshot = worker.get_frame_accounting_snapshot()

    # 2. Perform orderly shutdown
    await worker.stop()

    # 3. Allow background thread reconciliation and run garbage collection
    gc.collect()
    await asyncio.sleep(1.0)

    # 4. Capture post-shutdown reconciled snapshot
    post_shutdown_snapshot = worker.get_frame_accounting_snapshot()

    samples.append(capture_resource_sample(proc, t_start, len(samples), "EXP_A_STOPPED"))

    # Steady state analysis: discard initial warmup seconds
    steady_samples = [s for s in samples if s["elapsed_sec"] >= warmup_discard_sec]
    if len(steady_samples) < 3:
        steady_samples = samples  # fallback for short test runs

    x_times = [s["elapsed_sec"] for s in steady_samples]
    rss_vals = [s["rss_mb"] for s in steady_samples]
    heap_vals = [s["heap_current_mb"] for s in steady_samples]

    rss_ols = compute_ols_regression(x_times, rss_vals, x_unit_scale=60.0)
    heap_ols = compute_ols_regression(x_times, heap_vals, x_unit_scale=60.0)

    # Statistical stability evaluation via Section 7.2 decision matrix
    rss_eval = classify_memory_stability(rss_ols, "rss")
    heap_eval = classify_memory_stability(heap_ols, "heap")
    rss_slope_acceptable = rss_eval["is_acceptable"]
    heap_slope_acceptable = heap_eval["is_acceptable"]

    # Inference duration and throughput analysis
    durations = publisher.infer_durations_ms
    if durations:
        durs_arr = np.array(durations, dtype=np.float64)
        latency_stats = {
            "sample_count": len(durs_arr),
            "mean_inference_duration_ms": round(float(np.mean(durs_arr)), 2),
            "median_inference_duration_ms": round(float(np.median(durs_arr)), 2),
            "p95_inference_duration_ms": round(float(np.percentile(durs_arr, 95)), 2),
            "p99_inference_duration_ms": round(float(np.percentile(durs_arr, 99)), 2),
            "min_inference_duration_ms": round(float(np.min(durs_arr)), 2),
            "max_inference_duration_ms": round(float(np.max(durs_arr)), 2),
        }
    else:
        latency_stats = {"sample_count": 0}

    processed_throughput_fps = round(post_shutdown_snapshot.frames_processed / duration_seconds, 3)
    measured_replacement_rate_pct = round(
        (post_shutdown_snapshot.frames_replaced / max(1, post_shutdown_snapshot.frames_received)) * 100.0,
        2,
    )
    shutdown_reconciled_pass = (
        post_shutdown_snapshot.frames_queued == 0
        and post_shutdown_snapshot.frames_in_flight == 0
        and post_shutdown_snapshot.is_invariant_satisfied
    )

    logger.info(
        f"[Exp A] Finished: Ingested={post_shutdown_snapshot.frames_received}, "
        f"Processed={post_shutdown_snapshot.frames_processed}, Replaced={post_shutdown_snapshot.frames_replaced}, "
        f"Aborted={post_shutdown_snapshot.frames_aborted}. "
        f"Throughput={processed_throughput_fps} FPS, ReplacementRate={measured_replacement_rate_pct}%. "
        f"RSS Slope={rss_ols['slope_per_min']:.4f} MB/min ({rss_eval['classification']}), "
        f"Heap Slope={heap_ols['slope_per_min']:.4f} MB/min ({heap_eval['classification']})"
    )

    return {
        "duration_seconds": duration_seconds,
        "target_fps": target_fps,
        "total_frames_target": total_frames,
        "throughput_and_latency": {
            "duration_seconds": duration_seconds,
            "target_fps": target_fps,
            "processed_throughput_fps": processed_throughput_fps,
            "measured_replacement_rate_pct": measured_replacement_rate_pct,
            "inference_duration_distribution_ms": latency_stats,
            "interpretation_note": (
                "Processed throughput (FPS) and frame replacement rate reflect the combined "
                "service-time distribution, scheduling, and 40ms camera arrival intervals under a "
                "latest-frame (maxsize=1) policy, rather than a theoretical reciprocal of average latency."
            ),
        },
        "running_frame_accounting": {
            "frames_received": running_snapshot.frames_received,
            "frames_processed": running_snapshot.frames_processed,
            "frames_replaced": running_snapshot.frames_replaced,
            "frames_aborted": running_snapshot.frames_aborted,
            "frames_errored": running_snapshot.frames_errored,
            "frames_stale_discarded": running_snapshot.frames_stale_discarded,
            "frames_queued": running_snapshot.frames_queued,
            "frames_in_flight": running_snapshot.frames_in_flight,
            "invariant_satisfied": running_snapshot.is_invariant_satisfied,
        },
        "post_shutdown_frame_accounting": {
            "frames_received": post_shutdown_snapshot.frames_received,
            "frames_processed": post_shutdown_snapshot.frames_processed,
            "frames_replaced": post_shutdown_snapshot.frames_replaced,
            "frames_aborted": post_shutdown_snapshot.frames_aborted,
            "frames_errored": post_shutdown_snapshot.frames_errored,
            "frames_stale_discarded": post_shutdown_snapshot.frames_stale_discarded,
            "frames_queued": post_shutdown_snapshot.frames_queued,
            "frames_in_flight": post_shutdown_snapshot.frames_in_flight,
            "invariant_satisfied": post_shutdown_snapshot.is_invariant_satisfied,
        },
        "final_frame_accounting": {
            "frames_received": post_shutdown_snapshot.frames_received,
            "frames_processed": post_shutdown_snapshot.frames_processed,
            "frames_replaced": post_shutdown_snapshot.frames_replaced,
            "frames_aborted": post_shutdown_snapshot.frames_aborted,
            "frames_errored": post_shutdown_snapshot.frames_errored,
            "frames_stale_discarded": post_shutdown_snapshot.frames_stale_discarded,
            "frames_queued": post_shutdown_snapshot.frames_queued,
            "frames_in_flight": post_shutdown_snapshot.frames_in_flight,
            "invariant_satisfied": post_shutdown_snapshot.is_invariant_satisfied,
        },
        "rss_ols": rss_ols,
        "heap_ols": heap_ols,
        "rss_evaluation": rss_eval,
        "heap_evaluation": heap_eval,
        "criteria": {
            "rss_slope_pass": rss_slope_acceptable,
            "heap_slope_pass": heap_slope_acceptable,
            "invariant_pass": post_shutdown_snapshot.is_invariant_satisfied,
            "shutdown_reconciled_pass": shutdown_reconciled_pass,
        },
        "samples_count": len(samples),
        "steady_state_samples_count": len(steady_samples),
        "samples": samples,
    }


async def run_experiment_b_lifecycle(
    proc: psutil.Process,
    cycles_count: int = 10,
    frames_per_cycle: int = 150,
    target_fps: float = 25.0,
    rest_settle_sec: float = 1.5,
) -> Dict[str, Any]:
    """
    Experiment B: Repeated session lifecycle (10 sequential cycles).
    Starts session, processes 150 frames, shuts down worker, runs gc.collect(),
    and measures baseline retention to detect per-session memory or thread leaks.
    """
    logger.info(f"[Exp B] Starting lifecycle testing across {cycles_count} cycles ({frames_per_cycle} frames/cycle)")

    cycle_records: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    initial_baseline = capture_resource_sample(proc, t_start, 0, "EXP_B_PRE_CYCLE_BASELINE")
    base_thread_count = initial_baseline["thread_count"]

    frame_interval = 1.0 / target_fps

    for cycle_idx in range(1, cycles_count + 1):
        room_name = f"bench_mem_cycle_{cycle_idx}_{int(time.time())}"
        worker = LiveKitAIWorkerSession(
            room_name=room_name,
            driver_id=f"driver_cycle_{cycle_idx}",
            experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        )
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = NullDataPublisher()
        worker.is_running = True

        worker.engine = await session_manager.get_or_create_session(
            session_id=room_name,
            driver_id=f"driver_cycle_{cycle_idx}",
            experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
        )
        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Feed frames
        for f_idx in range(frames_per_cycle):
            t_f_start = time.perf_counter()
            img_bgr = generate_benchmark_frame(f_idx)
            mock_frame = MockVideoFrame(img_bgr)
            ingest_mono_ms = time.perf_counter() * 1000.0

            worker._frames_received += 1
            worker._frame_seq += 1
            if worker.frame_queue.full():
                try:
                    worker.frame_queue.get_nowait()
                    worker._frames_replaced += 1
                except asyncio.QueueEmpty:
                    pass
            await worker.frame_queue.put((mock_frame, ingest_mono_ms, worker._frame_seq))

            elapsed = time.perf_counter() - t_f_start
            slp = frame_interval - elapsed
            if slp > 0:
                await asyncio.sleep(slp)

        snap = worker.get_frame_accounting_snapshot()
        await worker.stop()

        # Settling & garbage collection
        gc.collect()
        await asyncio.sleep(rest_settle_sec)

        post_sample = capture_resource_sample(proc, t_start, cycle_idx, f"CYCLE_{cycle_idx}_END")
        post_sample.update({
            "cycle": cycle_idx,
            "frames_processed": snap.frames_processed,
            "invariant_satisfied": snap.is_invariant_satisfied,
            "thread_delta": post_sample["thread_count"] - base_thread_count,
        })
        cycle_records.append(post_sample)
        logger.info(
            f"[Exp B] Cycle {cycle_idx}/{cycles_count} complete: "
            f"RSS={post_sample['rss_mb']}MB, Heap={post_sample['heap_current_mb']}MB, "
            f"Threads={post_sample['thread_count']} (Delta={post_sample['thread_delta']})"
        )

    # Linear regression across post-warmup cycles (cycles 3 to cycles_count)
    warmup_cutoff = 2
    eval_records = [r for r in cycle_records if r["cycle"] > warmup_cutoff]
    if not eval_records:
        eval_records = cycle_records

    c_nums = [r["cycle"] for r in eval_records]
    rss_vals = [r["rss_mb"] for r in eval_records]
    heap_vals = [r["heap_current_mb"] for r in eval_records]

    # scale=1.0 for rate per cycle
    cycle_rss_ols = compute_ols_regression(c_nums, rss_vals, x_unit_scale=1.0)
    cycle_heap_ols = compute_ols_regression(c_nums, heap_vals, x_unit_scale=1.0)

    cycle_rss_eval = classify_memory_stability(cycle_rss_ols, "rss")
    cycle_heap_eval = classify_memory_stability(cycle_heap_ols, "heap")

    final_record = cycle_records[-1]
    final_thread_delta = abs(final_record["thread_count"] - base_thread_count)
    cycle_1_threads = cycle_records[0]["thread_count"]
    inter_cycle_thread_growth = max(0, final_record["thread_count"] - cycle_1_threads)
    thread_leak_pass = (final_thread_delta <= 2) and (inter_cycle_thread_growth <= 1)
    rss_cycle_slope_pass = cycle_rss_eval["is_acceptable"]

    return {
        "cycles_count": cycles_count,
        "frames_per_cycle": frames_per_cycle,
        "initial_baseline": initial_baseline,
        "cycle_records": cycle_records,
        "cycle_rss_ols": cycle_rss_ols,
        "cycle_heap_ols": cycle_heap_ols,
        "cycle_rss_evaluation": cycle_rss_eval,
        "cycle_heap_evaluation": cycle_heap_eval,
        "criteria": {
            "rss_cycle_slope_mb_per_cycle": cycle_rss_ols["slope_per_min"],
            "rss_cycle_slope_pass": rss_cycle_slope_pass,
            "heap_cycle_slope_pass": cycle_heap_eval["is_acceptable"],
            "final_thread_delta": final_thread_delta,
            "thread_leak_pass": thread_leak_pass,
        },
    }


async def run_experiment_c_idle(
    proc: psutil.Process,
    idle_seconds: float = 60.0,
    sample_interval_sec: float = 5.0,
) -> Dict[str, Any]:
    """
    Experiment C: Post-session idle observation (60 seconds).
    Verifies that zero active sessions results in stable idle resource consumption
    with no spontaneous growth or zombie tasks.
    """
    logger.info(f"[Exp C] Starting idle observation for {idle_seconds}s")
    samples: List[Dict[str, Any]] = []
    t_start = time.perf_counter()

    t_end = t_start + idle_seconds
    sample_idx = 0
    while time.perf_counter() < t_end:
        sample = capture_resource_sample(proc, t_start, sample_idx, "EXP_C_IDLE")
        samples.append(sample)
        sample_idx += 1
        await asyncio.sleep(sample_interval_sec)

    samples.append(capture_resource_sample(proc, t_start, sample_idx, "EXP_C_IDLE_END"))

    x_times = [s["elapsed_sec"] for s in samples]
    rss_vals = [s["rss_mb"] for s in samples]
    heap_vals = [s["heap_current_mb"] for s in samples]

    rss_ols = compute_ols_regression(x_times, rss_vals, x_unit_scale=60.0)
    heap_ols = compute_ols_regression(x_times, heap_vals, x_unit_scale=60.0)

    rss_eval = classify_memory_stability(rss_ols, "rss")
    heap_eval = classify_memory_stability(heap_ols, "heap")
    rss_drift_pass = rss_eval["is_acceptable"]
    heap_drift_pass = heap_eval["is_acceptable"]

    logger.info(
        f"[Exp C] Idle observation finished: RSS Slope={rss_ols['slope_per_min']:.4f} MB/min ({rss_eval['classification']}), "
        f"Heap Slope={heap_ols['slope_per_min']:.4f} MB/min ({heap_eval['classification']})"
    )

    return {
        "idle_seconds": idle_seconds,
        "samples_count": len(samples),
        "rss_ols": rss_ols,
        "heap_ols": heap_ols,
        "rss_evaluation": rss_eval,
        "heap_evaluation": heap_eval,
        "criteria": {
            "rss_idle_drift_pass": rss_drift_pass,
            "heap_idle_drift_pass": heap_drift_pass,
        },
        "samples": samples,
    }


async def run_experiment_d_cancellation_cleanup(
    proc: psutil.Process,
) -> Dict[str, Any]:
    """
    Experiment D: Stress test cancellation mid-stream, track unsubscription,
    and worker shutdown during active inference to verify safe resource cleanup.
    """
    logger.info("[Exp D] Starting cancellation & abnormal teardown stress test")
    t_start = time.perf_counter()
    pre_sample = capture_resource_sample(proc, t_start, 0, "EXP_D_PRE")

    room_name = f"bench_mem_exp_d_{int(time.time())}"
    worker = LiveKitAIWorkerSession(
        room_name=room_name,
        driver_id="driver_bench_d",
        experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
    )
    worker.room = MagicMock()
    worker.room.disconnect = AsyncMock()
    publisher = NullDataPublisher()
    worker.room.local_participant = publisher
    worker.is_running = True

    worker.engine = await session_manager.get_or_create_session(
        session_id=room_name,
        driver_id="driver_bench_d",
        experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
    )
    worker._inference_task = asyncio.create_task(worker._inference_loop())

    # Feed 20 frames
    for i in range(20):
        img_bgr = generate_benchmark_frame(i)
        mock_frame = MockVideoFrame(img_bgr)
        ingest_mono_ms = time.perf_counter() * 1000.0
        worker._frames_received += 1
        worker._frame_seq += 1
        await worker.frame_queue.put((mock_frame, ingest_mono_ms, worker._frame_seq))
        await asyncio.sleep(0.01)

    # Abruptly cancel inference task mid-operation
    if worker._inference_task and not worker._inference_task.done():
        worker._inference_task.cancel()

    # Call stop() which triggers shielded stream teardown, thread reconciliation, and session manager release
    await worker.stop()

    gc.collect()
    await asyncio.sleep(1.0)

    post_sample = capture_resource_sample(proc, t_start, 1, "EXP_D_POST")
    post_snapshot = worker.get_frame_accounting_snapshot()

    thread_delta = post_sample["thread_count"] - pre_sample["thread_count"]
    clean_cleanup_pass = (thread_delta <= 1) and (post_snapshot.frames_in_flight == 0)

    logger.info(
        f"[Exp D] Cancellation teardown complete: "
        f"InFlight={post_snapshot.frames_in_flight}, Aborted={post_snapshot.frames_aborted}, "
        f"ThreadDelta={thread_delta} (Pass={clean_cleanup_pass})"
    )

    return {
        "pre_sample": pre_sample,
        "post_sample": post_sample,
        "post_frame_accounting": {
            "frames_received": post_snapshot.frames_received,
            "frames_processed": post_snapshot.frames_processed,
            "frames_aborted": post_snapshot.frames_aborted,
            "frames_in_flight": post_snapshot.frames_in_flight,
            "invariant_satisfied": post_snapshot.is_invariant_satisfied,
        },
        "criteria": {
            "frames_in_flight_zero": post_snapshot.frames_in_flight == 0,
            "thread_delta_pass": thread_delta <= 1,
            "overall_pass": clean_cleanup_pass,
        },
    }


async def main_async(args: argparse.Namespace) -> None:
    print("=" * 80)
    print("  DriveGuard Phase 7 Stage 4: Memory Stability & Resource Profiling Benchmark")
    print("=" * 80)

    proc = psutil.Process(os.getpid())
    tracemalloc.start()

    cgroup_info = get_cgroup_memory_info()
    git_commit = get_git_commit_hash()

    metadata = {
        "benchmark_timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_commit": git_commit,
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            "ram_total_gb": round(psutil.virtual_memory().total / (1024 ** 3), 2),
        },
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "compiler": platform.python_compiler(),
        },
        "cgroup_memory": cgroup_info,
        "configuration": {
            "duration_seconds": args.duration,
            "target_fps": args.target_fps,
            "cycles_count": args.cycles,
            "idle_seconds": args.idle_seconds,
            "quick_mode": args.quick,
        },
    }

    print(f"Platform: {metadata['platform']['system']} {metadata['platform']['release']} ({metadata['platform']['machine']})")
    print(f"Commit: {git_commit}")
    print(f"Cgroup Discovery: {cgroup_info['details']}")
    print(f"Mode: {'QUICK VERIFICATION' if args.quick else 'FULL PRODUCTION BENCHMARK'}")
    print("-" * 80)

    # Run Experiments
    exp_a_res = await run_experiment_a_continuous(
        proc=proc,
        duration_seconds=args.duration,
        target_fps=args.target_fps,
        sample_interval_sec=2.0 if args.quick else 5.0,
        warmup_discard_sec=args.warmup_discard,
    )

    exp_b_res = await run_experiment_b_lifecycle(
        proc=proc,
        cycles_count=args.cycles,
        frames_per_cycle=30 if args.quick else 150,
        target_fps=args.target_fps,
        rest_settle_sec=0.5 if args.quick else 1.5,
    )

    exp_c_res = await run_experiment_c_idle(
        proc=proc,
        idle_seconds=args.idle_seconds,
        sample_interval_sec=2.0 if args.quick else 5.0,
    )

    exp_d_res = await run_experiment_d_cancellation_cleanup(proc=proc)

    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    overall_verdict = (
        exp_a_res["criteria"]["rss_slope_pass"]
        and exp_a_res["criteria"]["heap_slope_pass"]
        and exp_a_res["criteria"]["invariant_pass"]
        and exp_a_res["criteria"]["shutdown_reconciled_pass"]
        and exp_b_res["criteria"]["rss_cycle_slope_pass"]
        and exp_b_res["criteria"]["heap_cycle_slope_pass"]
        and exp_b_res["criteria"]["thread_leak_pass"]
        and exp_c_res["criteria"]["rss_idle_drift_pass"]
        and exp_c_res["criteria"]["heap_idle_drift_pass"]
        and exp_d_res["criteria"]["overall_pass"]
    )

    results = {
        "metadata": metadata,
        "overall_verdict": "PASS" if overall_verdict else "FAIL",
        "peak_runtime_heap_mb": round(traced_peak / (1024 * 1024), 2),
        "experiment_a_continuous": exp_a_res,
        "experiment_b_lifecycle": exp_b_res,
        "experiment_c_idle": exp_c_res,
        "experiment_d_cancellation_cleanup": exp_d_res,
    }

    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print(f"  BENCHMARK COMPLETE — OVERALL VERDICT: {'PASS' if overall_verdict else 'FAIL'}")
    print(f"  Results saved to: {out_path.resolve()}")
    print("=" * 80)
    print(f"  Exp A RSS Slope:  {exp_a_res['rss_ols']['slope_per_min']:.4f} MB/min")
    print(f"    - Conventional 95% CI: {exp_a_res['rss_ols']['slope_ci_95']}")
    print(f"    - Newey-West HAC 95% CI: {exp_a_res['rss_ols']['hac']['slope_ci_95_hac']} (lag={exp_a_res['rss_ols']['hac']['lag_truncation']}, rho1={exp_a_res['rss_ols']['hac']['autocorrelation_lag1']})")
    print(f"  Exp A Heap Slope: {exp_a_res['heap_ols']['slope_per_min']:.4f} MB/min")
    print(f"    - Conventional 95% CI: {exp_a_res['heap_ols']['slope_ci_95']}")
    print(f"    - Newey-West HAC 95% CI: {exp_a_res['heap_ols']['hac']['slope_ci_95_hac']} (lag={exp_a_res['heap_ols']['hac']['lag_truncation']}, rho1={exp_a_res['heap_ols']['hac']['autocorrelation_lag1']})")
    if "throughput_and_latency" in exp_a_res:
        tl = exp_a_res["throughput_and_latency"]
        print(f"  Exp A Throughput: {tl['processed_throughput_fps']} FPS (Replacement Rate: {tl['measured_replacement_rate_pct']}%)")
        lat = tl.get("inference_duration_distribution_ms", {})
        if lat.get("sample_count", 0) > 0:
            print(f"  Exp A Inference Durations: mean={lat['mean_inference_duration_ms']}ms, median={lat['median_inference_duration_ms']}ms, P95={lat['p95_inference_duration_ms']}ms, P99={lat['p99_inference_duration_ms']}ms")
    print(f"  Exp B Cycle Slope: {exp_b_res['cycle_rss_ols']['slope_per_min']:.4f} MB/cycle")
    print(f"  Exp B Thread Delta: {exp_b_res['criteria']['final_thread_delta']}")
    print(f"  Exp C Idle Drift: {exp_c_res['rss_ols']['slope_per_min']:.4f} MB/min")
    print(f"  Exp D Cleanup:    {'PASS' if exp_d_res['criteria']['overall_pass'] else 'FAIL'}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Phase 7 Stage 4 Memory Stability Benchmark")
    parser.add_argument("--duration", type=float, default=600.0, help="Exp A duration in seconds (default: 600.0s = 10 min)")
    parser.add_argument("--target-fps", type=float, default=25.0, help="Target camera frame rate (default: 25.0 FPS)")
    parser.add_argument("--cycles", type=int, default=10, help="Exp B session lifecycle cycles (default: 10)")
    parser.add_argument("--idle-seconds", type=float, default=60.0, help="Exp C idle observation seconds (default: 60.0s)")
    parser.add_argument("--warmup-discard", type=float, default=60.0, help="Warmup seconds discarded from steady-state OLS (default: 60.0s)")
    parser.add_argument("--quick", action="store_true", help="Run short verification mode for fast CI checks")
    parser.add_argument(
        "--output-json",
        type=str,
        default="backend/benchmarks/results/memory_stability_results.json",
        help="Path for JSON benchmark output",
    )

    args = parser.parse_args()
    if args.quick:
        args.duration = 15.0
        args.cycles = 3
        args.idle_seconds = 10.0
        args.warmup_discard = 5.0

    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
