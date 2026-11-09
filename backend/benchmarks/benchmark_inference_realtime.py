"""
DriveGuard Phase 7 Stage 2: Real-Time Ingest Queue & Replacement Benchmark.

Protocol:
- Simulates WebRTC frame arrival at target 25 FPS (40ms interval) into LiveKit worker frame_queue.
- Runs concurrent _inference_loop processing frames via thread pool.
- Records:
    - Ingested frames vs. Processed frames
    - Dropped/Replaced frames due to queue maxsize=1 zero-lag policy
    - Effective processed FPS vs. camera FPS
    - Verification of atomic FrameAccountingSnapshot invariant:
        F_received = F_replaced + F_stale + F_errored + F_aborted + F_processed + F_queued + F_in_flight
    - Queue dwell time distribution
"""

import asyncio
import json
import os
import platform
import sys
import time
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock

import cv2
import numpy as np

from backend.app.services.livekit_worker import (
    LiveKitAIWorkerSession,
    FrameAccountingSnapshot,
)
from backend.benchmarks.benchmark_inference_offline import generate_benchmark_frame


class MockVideoFrame:
    """Mock WebRTC VideoFrame wrapping numpy RGB buffer for memory-efficient simulation."""

    def __init__(self, img_bgr: np.ndarray):
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.width = img_bgr.shape[1]
        self.height = img_bgr.shape[0]
        self._data = img_rgb.tobytes()

    def convert(self, buffer_type):
        return self

    @property
    def data(self):
        return self._data


async def run_realtime_ingest_benchmark(
    duration_seconds: float = 20.0,
    target_fps: float = 25.0,
    experiment_mode: str = "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
) -> Dict[str, any]:
    """Run real-time ingestion simulation against LiveKitAIWorkerSession."""
    print("=" * 70)
    print(f"  DriveGuard: Real-Time WebRTC Ingest Benchmark ({experiment_mode})")
    print(f"  Duration: {duration_seconds}s | Target Feed: {target_fps} FPS (Interval: {1000/target_fps:.1f}ms)")
    print("=" * 70)

    room_name = f"bench_rt_{int(time.time())}"
    worker = LiveKitAIWorkerSession(
        room_name=room_name,
        driver_id="bench_driver_rt",
        experiment_mode=experiment_mode,
    )

    # Mock Room and LocalParticipant to capture published data without network SFU
    worker.room = MagicMock()
    worker.room.local_participant = MagicMock()
    worker.room.local_participant.publish_data = AsyncMock()
    worker.is_running = True

    # Start worker engine
    from backend.app.services.session_manager import session_manager
    worker.engine = await session_manager.get_or_create_session(
        session_id=room_name,
        driver_id="bench_driver_rt",
        experiment_mode=experiment_mode,
    )

    # Start inference loop task
    worker._inference_task = asyncio.create_task(worker._inference_loop())

    frame_interval = 1.0 / target_fps
    total_frames_target = int(duration_seconds * target_fps)

    print(f"[RealTime] Ingesting {total_frames_target} synthetic camera frames at {target_fps} FPS...")

    t_start = time.perf_counter()

    for i in range(total_frames_target):
        t_frame_start = time.perf_counter()

        img_bgr = generate_benchmark_frame(i)
        mock_frame = MockVideoFrame(img_bgr)
        ingest_mono_ms = time.perf_counter() * 1000.0

        # Simulate WebRTC ingestion logic on event loop
        worker._frames_received += 1
        worker._frame_seq += 1
        current_frame_seq = worker._frame_seq

        if worker.frame_queue.full():
            try:
                worker.frame_queue.get_nowait()
                worker._frames_replaced += 1
            except asyncio.QueueEmpty:
                pass

        await worker.frame_queue.put((mock_frame, ingest_mono_ms, current_frame_seq))

        # Pace to target camera FPS
        elapsed = time.perf_counter() - t_frame_start
        sleep_time = max(0.0, frame_interval - elapsed)
        if sleep_time > 0:
            await asyncio.sleep(sleep_time)

    # Allow final in-flight frame to complete
    await asyncio.sleep(0.2)

    total_duration = time.perf_counter() - t_start
    snapshot = worker.get_frame_accounting_snapshot()

    # Clean shutdown
    await worker.stop()

    processed_fps = snapshot.frames_processed / total_duration if total_duration > 0 else 0.0
    ingest_fps = snapshot.frames_received / total_duration if total_duration > 0 else 0.0
    replacement_rate_pct = (
        (snapshot.frames_replaced / snapshot.frames_received * 100.0)
        if snapshot.frames_received > 0
        else 0.0
    )

    result = {
        "benchmark": "realtime_ingest_benchmark",
        "duration_seconds": round(total_duration, 2),
        "target_fps": target_fps,
        "experiment_mode": experiment_mode,
        "frames_received": snapshot.frames_received,
        "frames_processed": snapshot.frames_processed,
        "frames_replaced": snapshot.frames_replaced,
        "frames_stale_discarded": snapshot.frames_stale_discarded,
        "frames_errored": snapshot.frames_errored,
        "frames_aborted": snapshot.frames_aborted,
        "frames_queued_final": snapshot.frames_queued,
        "frames_in_flight_final": snapshot.frames_in_flight,
        "accounting_discrepancy": snapshot.discrepancy,
        "is_invariant_satisfied": snapshot.is_invariant_satisfied,
        "effective_ingest_fps": round(ingest_fps, 2),
        "effective_processed_fps": round(processed_fps, 2),
        "replacement_rate_pct": round(replacement_rate_pct, 2),
    }

    print("\n--- Real-Time Ingestion Results ---")
    print(f"  Frames Received: {snapshot.frames_received} ({ingest_fps:.1f} FPS)")
    print(f"  Frames Processed: {snapshot.frames_processed} ({processed_fps:.1f} FPS)")
    print(f"  Frames Replaced (Zero-Lag Drop): {snapshot.frames_replaced} ({replacement_rate_pct:.1f}%)")
    print(f"  Frames Errored: {snapshot.frames_errored} | Aborted: {snapshot.frames_aborted}")
    print(f"  Accounting Invariant Satisfied: {snapshot.is_invariant_satisfied} (Discrepancy: {snapshot.discrepancy})")

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "realtime_ingest_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"[Benchmark Complete] Real-time metrics saved to: {out_path}\n")
    return result


def main():
    asyncio.run(run_realtime_ingest_benchmark(duration_seconds=15.0, target_fps=25.0))


if __name__ == "__main__":
    main()
