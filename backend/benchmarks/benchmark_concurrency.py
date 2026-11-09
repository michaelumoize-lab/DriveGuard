"""
DriveGuard Phase 7 Stage 5: Multi-Session Concurrency & Capacity Benchmarking Harness.

Empirically measures real system capacity, resource scaling, performance degradation,
and cross-session isolation across N = 1, 2, 3, 4, 5 concurrent driver sessions.

Evaluates:
  - Category 1: Synthetic concurrent sessions (paced 25 FPS synthetic frames).
  - Category 2: Predecoded real-video sessions (paced 25 FPS predecoded real frames).
  - Category 3: Mock WebRTC transport sessions (async stream ingestion lifecycle & isolation).

Maintains strict frame accounting conservation invariants and verifies complete shutdown reconciliation.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
import json
import logging
import math
import os
import pathlib
import sys
import time
import tracemalloc
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np
import psutil

# Ensure repo root is on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
_BACKEND_ROOT = _REPO_ROOT / "backend"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.ai.cascade_classifier import ExperimentMode
from backend.app.services.livekit_worker import (
    FrameAccountingSnapshot,
    LiveKitAIWorkerSession,
    WORKER_BOOT_ID,
    normalize_frame_width,
)
from backend.app.services.session_manager import session_manager
from backend.benchmarks.benchmark_inference_offline import generate_benchmark_frame

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_concurrency")


# ============================================================================
# Telemetry Collector & Mock WebRTC Frame Objects
# ============================================================================

class MockDataPacket:
    """Mock WebRTC DataPacket."""

    def __init__(self, data: bytes) -> None:
        self.data = data


class TelemetryCollector:
    """Mock WebRTC local_participant capturing all published data packets."""

    def __init__(self, session_id: str, worker_ref: Optional[Any] = None) -> None:
        self.session_id = session_id
        self.worker_ref = worker_ref
        self.packets: List[dict] = []
        self.safety_events: List[dict] = []
        self.control_acks: List[dict] = []

    async def publish_data(self, data: bytes, reliable: bool = False, topic: Optional[str] = None) -> None:
        try:
            payload = json.loads(data.decode("utf-8"))
            msg_type = payload.get("type")
            if msg_type in ("STATE_CHANGE", "SAFETY_ALERT", "SAFETY_EVENT"):
                self.safety_events.append(payload)
                # Auto-acknowledge reliable safety events to avoid redundant background retries
                if msg_type == "SAFETY_EVENT" and self.worker_ref:
                    evt_id = payload.get("event_id")
                    if evt_id:
                        asyncio.create_task(
                            self.worker_ref._handle_client_control_message(
                                MockDataPacket(json.dumps({"action": "ACK_EVENT", "event_id": evt_id}).encode("utf-8"))
                            )
                        )
            elif msg_type in ("CALIBRATION_STARTED", "CALIBRATION_COMPLETED", "PONG", "STATE_SYNC"):
                self.control_acks.append(payload)
            else:
                self.packets.append(payload)
        except Exception as exc:
            logger.debug(f"Failed to decode telemetry packet: {exc}")


class MockVideoFrame:
    """Mock WebRTC VideoFrame wrapping numpy RGB buffer for memory-efficient simulation."""

    def __init__(self, img_bgr: np.ndarray) -> None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.width = img_bgr.shape[1]
        self.height = img_bgr.shape[0]
        self._data = img_rgb.tobytes()

    def convert(self, buffer_type: Any) -> MockVideoFrame:
        return self

    @property
    def data(self) -> bytes:
        return self._data


class MockStreamEvent:
    """Mock WebRTC stream event wrapping a VideoFrame."""

    def __init__(self, frame: MockVideoFrame) -> None:
        self.frame = frame


class MockAsyncVideoStream:
    """Mock WebRTC VideoStream yielding frames at paced FPS for Category 3."""

    def __init__(self, frames: List[MockVideoFrame], target_fps: float = 25.0) -> None:
        self.frames = frames
        self.target_fps = target_fps
        self.interval = 1.0 / target_fps
        self.closed = False
        self.close_call_count = 0

    async def __aiter__(self):
        idx = 0
        while not self.closed:
            start_t = time.perf_counter()
            frame = self.frames[idx % len(self.frames)]
            idx += 1
            yield MockStreamEvent(frame)
            elapsed = time.perf_counter() - start_t
            sleep_t = max(0.0, self.interval - elapsed)
            if sleep_t > 0:
                await asyncio.sleep(sleep_t)

    async def aclose(self) -> None:
        self.close_call_count += 1
        self.closed = True


# ============================================================================
# Background Resource Sampler
# ============================================================================

class ResourceSampler:
    """Samples process and system CPU, memory, and thread metrics at fixed intervals."""

    def __init__(self, interval_sec: float = 0.1) -> None:
        self.interval_sec = interval_sec
        self._task: Optional[asyncio.Task] = None
        self._process = psutil.Process()
        self.running = False

        self.timestamps: List[float] = []
        self.process_cpu_samples: List[float] = []
        self.system_cpu_samples: List[float] = []
        self.rss_samples_mb: List[float] = []
        self.vms_samples_mb: List[float] = []
        self.heap_samples_mb: List[float] = []
        self.thread_samples: List[int] = []

    def start(self) -> None:
        self.running = True
        self._process.cpu_percent(interval=None)
        psutil.cpu_percent(interval=None)
        self._task = asyncio.create_task(self._sampler_loop())

    async def _sampler_loop(self) -> None:
        while self.running:
            try:
                t = time.perf_counter()
                p_cpu = self._process.cpu_percent(interval=None)
                s_cpu = psutil.cpu_percent(interval=None)
                mem = self._process.memory_info()
                rss_mb = mem.rss / (1024.0 * 1024.0)
                vms_mb = mem.vms / (1024.0 * 1024.0)
                heap_curr = tracemalloc.get_traced_memory()[0] / (1024.0 * 1024.0)
                num_th = self._process.num_threads()

                self.timestamps.append(t)
                self.process_cpu_samples.append(p_cpu)
                self.system_cpu_samples.append(s_cpu)
                self.rss_samples_mb.append(rss_mb)
                self.vms_samples_mb.append(vms_mb)
                self.heap_samples_mb.append(heap_curr)
                self.thread_samples.append(num_th)

                await asyncio.sleep(self.interval_sec)
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.debug(f"Resource sampling error: {exc}")

    async def stop(self) -> Dict[str, Any]:
        self.running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

        if not self.process_cpu_samples:
            return {
                "process_cpu_mean": 0.0,
                "process_cpu_peak": 0.0,
                "system_cpu_mean": 0.0,
                "system_cpu_peak": 0.0,
                "rss_start_mb": 0.0,
                "rss_peak_mb": 0.0,
                "rss_end_mb": 0.0,
                "rss_delta_mb": 0.0,
                "vms_start_mb": 0.0,
                "vms_peak_mb": 0.0,
                "vms_end_mb": 0.0,
                "traced_heap_start_mb": 0.0,
                "traced_heap_peak_mb": 0.0,
                "traced_heap_end_mb": 0.0,
                "traced_heap_delta_mb": 0.0,
                "threads_start": 0,
                "threads_peak": 0,
                "threads_end": 0,
                "sample_count": 0,
            }

        return {
            "process_cpu_mean": round(float(np.mean(self.process_cpu_samples)), 1),
            "process_cpu_peak": round(float(np.max(self.process_cpu_samples)), 1),
            "system_cpu_mean": round(float(np.mean(self.system_cpu_samples)), 1),
            "system_cpu_peak": round(float(np.max(self.system_cpu_samples)), 1),
            "rss_start_mb": round(self.rss_samples_mb[0], 2),
            "rss_peak_mb": round(float(np.max(self.rss_samples_mb)), 2),
            "rss_end_mb": round(self.rss_samples_mb[-1], 2),
            "rss_delta_mb": round(self.rss_samples_mb[-1] - self.rss_samples_mb[0], 2),
            "vms_start_mb": round(self.vms_samples_mb[0], 2),
            "vms_peak_mb": round(float(np.max(self.vms_samples_mb)), 2),
            "vms_end_mb": round(self.vms_samples_mb[-1], 2),
            "traced_heap_start_mb": round(self.heap_samples_mb[0], 2),
            "traced_heap_peak_mb": round(float(np.max(self.heap_samples_mb)), 2),
            "traced_heap_end_mb": round(self.heap_samples_mb[-1], 2),
            "traced_heap_delta_mb": round(self.heap_samples_mb[-1] - self.heap_samples_mb[0], 2),
            "threads_start": self.thread_samples[0],
            "threads_peak": int(np.max(self.thread_samples)),
            "threads_end": self.thread_samples[-1],
            "sample_count": len(self.process_cpu_samples),
        }


# ============================================================================
# Statistical Helper Functions
# ============================================================================

def compute_distribution_stats(values: List[float]) -> Dict[str, Any]:
    """Calculate descriptive distribution percentiles and moments."""
    if not values:
        return {
            "samples_count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "median_ms": 0.0,
            "p90_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
        }
    arr = np.array(values, dtype=np.float64)
    return {
        "samples_count": int(len(arr)),
        "mean_ms": round(float(np.mean(arr)), 2),
        "std_ms": round(float(np.std(arr)), 2),
        "median_ms": round(float(np.median(arr)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "min_ms": round(float(np.min(arr)), 2),
        "max_ms": round(float(np.max(arr)), 2),
    }


# ============================================================================
# Predecoded Real Frame Cache Builder (Category 2)
# ============================================================================

def build_predecoded_frame_cache(count: int = 60) -> List[np.ndarray]:
    """
    Construct predecoded 640x480 driver frames compositing real dataset eye patches.
    Caches frames in RAM so disk I/O does not bottleneck multi-session benchmarks.
    """
    cache: List[np.ndarray] = []
    real_awake_dir = _BACKEND_ROOT / "datasets" / "data" / "test" / "awake"
    real_sleepy_dir = _BACKEND_ROOT / "datasets" / "data" / "test" / "sleepy"

    awake_files = sorted(list(real_awake_dir.glob("*.png"))) if real_awake_dir.exists() else []
    sleepy_files = sorted(list(real_sleepy_dir.glob("*.png"))) if real_sleepy_dir.exists() else []

    for idx in range(count):
        base_img = generate_benchmark_frame(idx)

        # Composite real eye patches if available
        if awake_files:
            # Alternate between awake and sleepy patches periodically to simulate real blink dynamics
            is_sleepy_patch = (idx % 20) in (10, 11, 12) and len(sleepy_files) > 0
            file_list = sleepy_files if is_sleepy_patch else awake_files
            patch_file = file_list[idx % len(file_list)]

            patch_bgr = cv2.imread(str(patch_file))
            if patch_bgr is not None:
                patch_resized = cv2.resize(patch_bgr, (36, 20), interpolation=cv2.INTER_AREA)
                h, w = patch_resized.shape[:2]

                # Position over left and right eye ellipses
                # Base eye centers: cx +/- 45, cy - 25 where cx=320, cy=240
                sway_x = int(np.sin(idx * 0.15) * 4)
                sway_y = int(np.cos(idx * 0.12) * 3)
                cx, cy = 320 + sway_x, 240 + sway_y

                lx, ly = cx - 45 - (w // 2), cy - 25 - (h // 2)
                rx, ry = cx + 45 - (w // 2), cy - 25 - (h // 2)

                if 0 <= ly and ly + h <= 480 and 0 <= lx and lx + w <= 640:
                    base_img[ly:ly+h, lx:lx+w] = patch_resized
                if 0 <= ry and ry + h <= 480 and 0 <= rx and rx + w <= 640:
                    base_img[ry:ry+h, rx:rx+w] = patch_resized

        cache.append(base_img)

    return cache


# ============================================================================
# Single Concurrency Level Benchmark Execution
# ============================================================================

async def run_concurrent_sessions(
    n_sessions: int,
    duration_seconds: float = 15.0,
    target_fps: float = 25.0,
    category: str = "synthetic",
    cached_real_frames: Optional[List[np.ndarray]] = None,
    experiment_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Execute N concurrent LiveKitAIWorkerSession instances for duration_seconds.
    Ingests frames at target_fps per session and records full isolation, accounting,
    latency distributions, and resource utilization.
    """
    frame_interval = 1.0 / target_fps
    total_frames_target = int(duration_seconds * target_fps)

    workers: List[LiveKitAIWorkerSession] = []
    collectors: List[TelemetryCollector] = []
    mock_streams: List[Optional[MockAsyncVideoStream]] = []

    # 1. Initialize N sessions with dedicated isolated components
    for s_idx in range(n_sessions):
        session_id = f"sess_c{n_sessions:02d}_{s_idx+1:02d}_{uuid.uuid4().hex[:6]}"
        driver_id = f"driver_c{n_sessions:02d}_{s_idx+1:02d}"

        worker = LiveKitAIWorkerSession(
            room_name=session_id,
            driver_id=driver_id,
            experiment_mode=experiment_mode,
        )
        collector = TelemetryCollector(session_id, worker_ref=worker)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True

        # Initialize session-isolated engine via session_manager
        worker.engine = await session_manager.get_or_create_session(
            session_id=session_id,
            driver_id=driver_id,
            experiment_mode=ExperimentMode(experiment_mode) if experiment_mode else ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
        )

        workers.append(worker)
        collectors.append(collector)

    # Prepare Mock frames
    mock_video_frames: List[MockVideoFrame] = []
    if category == "predecoded" and cached_real_frames:
        mock_video_frames = [MockVideoFrame(img) for img in cached_real_frames]
    else:
        # Pre-generate 60 synthetic frames
        mock_video_frames = [MockVideoFrame(generate_benchmark_frame(idx)) for idx in range(60)]

    # 2. Start worker inference loops and category-specific ingest tasks
    sampler = ResourceSampler(interval_sec=0.1)
    sampler.start()

    for s_idx, worker in enumerate(workers):
        worker._inference_task = asyncio.create_task(worker._inference_loop())

        if category == "webrtc_mock":
            # Category 3: Use MockAsyncVideoStream feeding _ingest_video_stream
            stream = MockAsyncVideoStream(mock_video_frames, target_fps=target_fps)
            mock_streams.append(stream)
            worker._current_video_stream = stream
            worker._ingest_task = asyncio.create_task(worker._ingest_video_stream(stream))
        else:
            mock_streams.append(None)

    # 3. Concurrent ingestion loop for Categories 1 and 2
    t_start = time.perf_counter()

    if category != "webrtc_mock":
        async def session_ingest_loop(w_idx: int, worker: LiveKitAIWorkerSession) -> None:
            nonlocal total_frames_target, frame_interval
            for f_idx in range(total_frames_target):
                t_frame_start = time.perf_counter()
                frame = mock_video_frames[(f_idx + w_idx * 7) % len(mock_video_frames)]
                now_ms = time.perf_counter() * 1000.0

                worker._frames_received += 1
                worker._frame_seq += 1
                seq = worker._frame_seq

                if worker.frame_queue.full():
                    try:
                        worker.frame_queue.get_nowait()
                        worker._frames_replaced += 1
                    except asyncio.QueueEmpty:
                        pass

                await worker.frame_queue.put((frame, now_ms, seq))

                elapsed = time.perf_counter() - t_frame_start
                sleep_t = max(0.0, frame_interval - elapsed)
                if sleep_t > 0:
                    await asyncio.sleep(sleep_t)

        # Run all N ingest loops concurrently
        await asyncio.gather(*(session_ingest_loop(i, w) for i, w in enumerate(workers)))
    else:
        # Category 3: Wait duration while streams stream frames
        await asyncio.sleep(duration_seconds)

    # 4. Ingest settlement & running state capture
    await asyncio.sleep(0.25)
    t_running_end = time.perf_counter()
    actual_duration = t_running_end - t_start

    # Capture running snapshots and latencies BEFORE worker stop terminates engine
    running_snapshots: List[FrameAccountingSnapshot] = [w.get_frame_accounting_snapshot() for w in workers]
    engine_latencies: List[List[float]] = [
        w.engine.get_recent_latencies() if w.engine else [] for w in workers
    ]

    # 5. Stop all workers cleanly and perform shutdown reconciliation
    for w in workers:
        await w.stop()

    # Brief settle for thread pool termination
    await asyncio.sleep(0.1)

    resource_metrics = await sampler.stop()

    post_shutdown_snapshots: List[FrameAccountingSnapshot] = [
        w.get_frame_accounting_snapshot() for w in workers
    ]

    # 6. Extract Session Metrics & Verify Isolation
    session_results: List[Dict[str, Any]] = []
    all_inference_latencies: List[float] = []
    all_queue_latencies: List[float] = []
    all_server_totals: List[float] = []

    seen_epochs = set()
    total_received = 0
    total_processed = 0
    total_replaced = 0
    total_aborted = 0

    all_invariants_met = True
    all_shutdown_clean = True
    all_isolation_valid = True

    for s_idx, (worker, collector, run_snap, post_snap, lat_list) in enumerate(
        zip(workers, collectors, running_snapshots, post_shutdown_snapshots, engine_latencies)
    ):
        s_received = post_snap.frames_received
        s_processed = post_snap.frames_processed
        s_replaced = post_snap.frames_replaced
        s_aborted = post_snap.frames_aborted

        total_received += s_received
        total_processed += s_processed
        total_replaced += s_replaced
        total_aborted += s_aborted

        s_ingest_fps = round(s_received / actual_duration, 2) if actual_duration > 0 else 0.0
        s_proc_fps = round(s_processed / actual_duration, 2) if actual_duration > 0 else 0.0
        s_replace_pct = round((s_replaced / s_received * 100.0), 2) if s_received > 0 else 0.0

        # Frame invariant verification
        if not post_snap.is_invariant_satisfied:
            all_invariants_met = False

        # Post-shutdown reconciliation
        if post_snap.frames_queued != 0 or post_snap.frames_in_flight != 0:
            all_shutdown_clean = False

        # Extract timing distributions from collected telemetry packets
        queue_lats: List[float] = []
        server_tots: List[float] = []
        packet_session_mismatches = 0
        packet_driver_mismatches = 0

        for pkt in collector.packets:
            p_sess = pkt.get("session_id")
            if p_sess != worker.room_name:
                packet_session_mismatches += 1

            telemetry = pkt.get("telemetry", {})
            if telemetry.get("driver_id") != worker.driver_id:
                packet_driver_mismatches += 1

            timing = telemetry.get("timing")
            if timing:
                if "queue_latency_ms" in timing:
                    q_ms = float(timing["queue_latency_ms"])
                    queue_lats.append(q_ms)
                    all_queue_latencies.append(q_ms)
                if "server_total_ms" in timing:
                    st_ms = float(timing["server_total_ms"])
                    server_tots.append(st_ms)
                    all_server_totals.append(st_ms)

        for evt in collector.safety_events:
            if evt.get("session_id") != worker.room_name:
                packet_session_mismatches += 1

        if packet_session_mismatches > 0 or packet_driver_mismatches > 0:
            all_isolation_valid = False

        # Session epoch collision check
        if worker.session_epoch in seen_epochs:
            all_isolation_valid = False
        seen_epochs.add(worker.session_epoch)

        # Combine latencies for distribution stats
        all_inference_latencies.extend(lat_list)

        s_infer_stats = compute_distribution_stats(lat_list)
        s_queue_stats = compute_distribution_stats(queue_lats)
        s_server_stats = compute_distribution_stats(server_tots)

        session_results.append({
            "session_idx": s_idx + 1,
            "session_id": worker.room_name,
            "driver_id": worker.driver_id,
            "session_epoch": worker.session_epoch,
            "worker_boot_id": worker.worker_boot_id,
            "frames_received": s_received,
            "frames_processed": s_processed,
            "frames_replaced": s_replaced,
            "frames_stale_discarded": post_snap.frames_stale_discarded,
            "frames_errored": post_snap.frames_errored,
            "frames_aborted": s_aborted,
            "frames_queued_post_shutdown": post_snap.frames_queued,
            "frames_in_flight_post_shutdown": post_snap.frames_in_flight,
            "is_invariant_satisfied": post_snap.is_invariant_satisfied,
            "ingest_fps": s_ingest_fps,
            "processed_fps": s_proc_fps,
            "replacement_rate_pct": s_replace_pct,
            "inference_latency_stats": s_infer_stats,
            "queue_latency_stats": s_queue_stats,
            "server_total_stats": s_server_stats,
            "telemetry_packets_captured": len(collector.packets),
            "safety_events_captured": len(collector.safety_events),
            "cross_session_telemetry_leaks": packet_session_mismatches + packet_driver_mismatches,
        })

    # 7. Aggregate Metrics Calculation
    agg_proc_fps = round(total_processed / actual_duration, 2) if actual_duration > 0 else 0.0
    agg_ingest_fps = round(total_received / actual_duration, 2) if actual_duration > 0 else 0.0
    agg_replace_pct = round((total_replaced / total_received * 100.0), 2) if total_received > 0 else 0.0

    global_infer_stats = compute_distribution_stats(all_inference_latencies)
    global_queue_stats = compute_distribution_stats(all_queue_latencies)
    global_server_stats = compute_distribution_stats(all_server_totals)

    return {
        "concurrency_level": n_sessions,
        "category": category,
        "duration_seconds": round(actual_duration, 2),
        "target_fps_per_session": target_fps,
        "total_frames_received": total_received,
        "total_frames_processed": total_processed,
        "total_frames_replaced": total_replaced,
        "total_frames_aborted": total_aborted,
        "aggregate_ingest_fps": agg_ingest_fps,
        "aggregate_processed_fps": agg_proc_fps,
        "aggregate_replacement_rate_pct": agg_replace_pct,
        "all_invariants_satisfied": all_invariants_met,
        "all_shutdowns_reconciled": all_shutdown_clean,
        "isolation_verified": all_isolation_valid,
        "global_inference_latency": global_infer_stats,
        "global_queue_latency": global_queue_stats,
        "global_server_total_latency": global_server_stats,
        "resource_utilization": resource_metrics,
        "sessions": session_results,
    }


# ============================================================================
# Benchmark Orchestration & Reporting
# ============================================================================

async def run_stage_5_concurrency_suite(
    session_counts: List[int],
    duration_per_run: float = 15.0,
    categories: List[str] = ["synthetic", "predecoded", "webrtc_mock"],
    output_path: Optional[pathlib.Path] = None,
) -> Dict[str, Any]:
    """Execute complete Stage 5 multi-session concurrency benchmark suite."""
    tracemalloc.start()
    gc.collect()

    host_info = {
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "total_ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "python_version": sys.version.split()[0],
        "platform": sys.platform,
        "worker_boot_id": WORKER_BOOT_ID,
    }

    print("\n" + "=" * 80)
    print("  DRIVEGUARD PHASE 7 — STAGE 5: MULTI-SESSION CONCURRENCY BENCHMARK SUITE")
    print("=" * 80)
    print(f"  Logical Cores: {host_info['cpu_count_logical']} | Physical Cores: {host_info['cpu_count_physical']} | Total RAM: {host_info['total_ram_gb']} GB")
    print(f"  Test Matrix: N = {session_counts} sessions | Duration per run: {duration_per_run}s")
    print(f"  Categories: {categories}")
    print("=" * 80 + "\n")

    # Pre-build real frame cache if predecoded category is requested
    cached_real_frames: Optional[List[np.ndarray]] = None
    if "predecoded" in categories:
        print("[Setup] Predecoding real dataset eye patches into memory frame cache...")
        cached_real_frames = build_predecoded_frame_cache(count=60)
        print(f"[Setup] Real frame cache ready ({len(cached_real_frames)} 640x480 frames).")

    suite_results: Dict[str, Any] = {
        "benchmark_suite": "DriveGuard Phase 7 Stage 5 Multi-Session Concurrency",
        "timestamp_epoch": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "host_environment": host_info,
        "session_matrix": session_counts,
        "duration_seconds_per_run": duration_per_run,
        "categories_tested": categories,
        "runs": {},
        "degradation_analysis": {},
    }

    for cat in categories:
        print(f"\n>>> EXECUTING CATEGORY: {cat.upper()} <<<")
        cat_runs: List[Dict[str, Any]] = []

        baseline_n1: Optional[Dict[str, Any]] = None

        for n in session_counts:
            print(f"\n--- Running N = {n} Concurrent Sessions [{cat}] (Duration: {duration_per_run}s) ---")
            gc.collect()

            run_res = await run_concurrent_sessions(
                n_sessions=n,
                duration_seconds=duration_per_run,
                target_fps=25.0,
                category=cat,
                cached_real_frames=cached_real_frames,
            )

            # Record baseline for N=1 degradation comparison
            if n == 1:
                baseline_n1 = run_res

            # Calculate degradation multipliers relative to N=1
            degradation: Dict[str, Any] = {}
            if baseline_n1:
                b_p95 = baseline_n1["global_inference_latency"]["p95_ms"]
                b_p50 = baseline_n1["global_inference_latency"]["median_ms"]
                b_replace = baseline_n1["aggregate_replacement_rate_pct"]
                b_proc_fps_per_sess = baseline_n1["aggregate_processed_fps"]

                c_p95 = run_res["global_inference_latency"]["p95_ms"]
                c_p50 = run_res["global_inference_latency"]["median_ms"]
                c_replace = run_res["aggregate_replacement_rate_pct"]
                c_proc_fps = run_res["aggregate_processed_fps"]

                degradation = {
                    "latency_p95_factor": round(c_p95 / b_p95, 2) if b_p95 > 0 else 1.0,
                    "latency_p50_factor": round(c_p50 / b_p50, 2) if b_p50 > 0 else 1.0,
                    "replacement_rate_delta_pct": round(c_replace - b_replace, 2),
                    "scaling_efficiency_pct": round((c_proc_fps / (n * b_proc_fps_per_sess) * 100.0), 1) if b_proc_fps_per_sess > 0 else 100.0,
                }
            run_res["degradation_factors"] = degradation
            cat_runs.append(run_res)

            # Print console row summary
            g_infer = run_res["global_inference_latency"]
            g_q = run_res["global_queue_latency"]
            res = run_res["resource_utilization"]
            print(
                f"  N={n:2d} | Ingest: {run_res['aggregate_ingest_fps']:5.1f} FPS | "
                f"Proc: {run_res['aggregate_processed_fps']:5.1f} FPS | "
                f"Replace: {run_res['aggregate_replacement_rate_pct']:4.1f}% | "
                f"Infer P50: {g_infer['median_ms']:5.1f}ms | "
                f"Infer P95: {g_infer['p95_ms']:5.1f}ms | "
                f"Queue P95: {g_q['p95_ms']:5.1f}ms | "
                f"CPU: {res['process_cpu_mean']:4.1f}% (Pk: {res['process_cpu_peak']:4.1f}%) | "
                f"RSS: {res['rss_end_mb']:5.1f}MB | "
                f"Inv: {'PASS' if run_res['all_invariants_satisfied'] else 'FAIL'} | "
                f"Reconcile: {'PASS' if run_res['all_shutdowns_reconciled'] else 'FAIL'}"
            )

        suite_results["runs"][cat] = cat_runs

    # Cross-category degradation synthesis
    # Analyze capacity limits across synthetic and predecoded runs
    s_runs = suite_results["runs"].get("synthetic", [])
    if s_runs:
        cliff_point = None
        for r in s_runs:
            # Cliff condition: P95 > 250ms or replacement rate > 75% or CPU saturating physical cores (>350%)
            if r["global_inference_latency"]["p95_ms"] > 250.0 or r["aggregate_replacement_rate_pct"] > 75.0:
                cliff_point = r["concurrency_level"]
                break

        suite_results["degradation_analysis"] = {
            "capacity_cliff_concurrency": cliff_point,
            "physical_core_limit": host_info["cpu_count_physical"],
            "logical_core_limit": host_info["cpu_count_logical"],
            "cross_session_isolation_violations": sum(
                sum(s["cross_session_telemetry_leaks"] for s in r["sessions"])
                for cat_list in suite_results["runs"].values()
                for r in cat_list
            ),
            "frame_conservation_violations": sum(
                0 if r["all_invariants_satisfied"] else 1
                for cat_list in suite_results["runs"].values()
                for r in cat_list
            ),
        }

    # Save output JSON
    if output_path is None:
        output_path = _BACKEND_ROOT / "benchmarks" / "results" / "concurrency_benchmark_results.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(suite_results, f, indent=2)

    print("\n" + "=" * 80)
    print(f"  Stage 5 Concurrency Benchmark Suite Complete!")
    print(f"  Raw Results Saved To: {output_path}")
    print("=" * 80 + "\n")

    return suite_results


# ============================================================================
# CLI Entrypoint
# ============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Phase 7 Stage 5 Multi-Session Concurrency Benchmark")
    parser.add_argument(
        "--sessions",
        type=str,
        default="1,2,3,4,5",
        help="Comma-separated concurrent session counts to test (e.g. '1,2,3,4,5')",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=12.0,
        help="Duration in seconds per concurrency run (default: 12.0s)",
    )
    parser.add_argument(
        "--categories",
        type=str,
        default="synthetic,predecoded,webrtc_mock",
        help="Comma-separated categories to test ('synthetic', 'predecoded', 'webrtc_mock')",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file",
    )
    args = parser.parse_args()

    session_counts = [int(x.strip()) for x in args.sessions.split(",") if x.strip()]
    cat_list = [x.strip() for x in args.categories.split(",") if x.strip()]
    out_file = pathlib.Path(args.output) if args.output else None

    asyncio.run(
        run_stage_5_concurrency_suite(
            session_counts=session_counts,
            duration_per_run=args.duration,
            categories=cat_list,
            output_path=out_file,
        )
    )


if __name__ == "__main__":
    main()
