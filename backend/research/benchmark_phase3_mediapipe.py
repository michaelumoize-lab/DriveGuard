"""
Phase 3 Benchmark: MediaPipe IMAGE vs VIDEO Mode & Multi-Session Scalability Benchmark.
Measures:
1. Average processing time per frame
2. Median (p50) processing time
3. 95th-percentile (p95) processing time
4. Effective processed FPS
5. CPU utilization & Memory RSS growth
6. Landmark temporal stability across sequential frames
"""

import os
import time
import psutil
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor

from backend.app.ai.landmarker import DriveGuardFaceLandmarker


def generate_synthetic_driver_frames(n_frames: int = 60, width: int = 640, height: int = 480):
    """Generate realistic synthetic driver facial frames with subtle natural head sway and eye blinks."""
    frames = []
    for i in range(n_frames):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        # Gradient background
        for y in range(height):
            img[y, :] = (int(30 + 20 * (y / height)), int(30 + 15 * (y / height)), 35)

        # Subtle realistic sway
        sway_x = int(np.sin(i * 0.15) * 4)
        sway_y = int(np.cos(i * 0.12) * 3)

        cx, cy = 320 + sway_x, 240 + sway_y
        # Face oval
        cv2.ellipse(img, (cx, cy), (105, 145), 0, 0, 360, (215, 205, 195), -1)

        # Nose bridge
        cv2.line(img, (cx, cy - 20), (cx, cy + 30), (180, 170, 160), 4)

        # Mouth
        cv2.ellipse(img, (cx, cy + 70), (35, 12), 0, 0, 360, (140, 90, 90), -1)

        # Eyes (simulate a blink around frame 25-28)
        eye_h = 3 if 25 <= i <= 28 else 10
        cv2.ellipse(img, (cx - 45, cy - 25), (18, eye_h), 0, 0, 360, (245, 245, 245), -1)
        cv2.ellipse(img, (cx + 45, cy - 25), (18, eye_h), 0, 0, 360, (245, 245, 245), -1)
        if eye_h > 3:
            cv2.circle(img, (cx - 45, cy - 25), 6, (60, 45, 30), -1)
            cv2.circle(img, (cx + 45, cy - 25), 6, (60, 45, 30), -1)

        frames.append(img)
    return frames


def run_single_session_benchmark(mode: str, frames: list) -> dict:
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    landmarker = DriveGuardFaceLandmarker(running_mode=mode)

    # Warmup
    for w in range(5):
        landmarker.process_frame(frames[0], timestamp_ms=float(w * 33.3))

    latencies = []
    prev_landmarks = None
    jitter_distances = []

    cpu_start = process.cpu_percent(interval=None)
    t_start = time.perf_counter()

    for idx, frame in enumerate(frames):
        ts_ms = (idx + 5) * 33.33
        t0 = time.perf_counter()
        features, lms = landmarker.process_frame(frame, timestamp_ms=ts_ms)
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)

        if lms is not None and prev_landmarks is not None and len(lms) == len(prev_landmarks):
            diff = np.linalg.norm(lms - prev_landmarks, axis=1)
            jitter_distances.append(float(np.mean(diff)))
        if lms is not None:
            prev_landmarks = lms

    total_time = time.perf_counter() - t_start
    cpu_end = process.cpu_percent(interval=None)
    mem_after_mb = process.memory_info().rss / (1024 * 1024)

    landmarker.close()

    latencies = np.array(latencies)
    return {
        "mode": mode,
        "n_frames": len(frames),
        "total_time_s": total_time,
        "mean_latency_ms": float(np.mean(latencies)),
        "p50_latency_ms": float(np.percentile(latencies, 50)),
        "p95_latency_ms": float(np.percentile(latencies, 95)),
        "min_latency_ms": float(np.min(latencies)),
        "max_latency_ms": float(np.max(latencies)),
        "effective_fps": len(frames) / total_time,
        "cpu_percent": cpu_end,
        "mem_growth_mb": mem_after_mb - mem_before_mb,
        "mean_jitter_px": float(np.mean(jitter_distances)) if jitter_distances else 0.0,
    }


def run_dual_session_concurrent_benchmark(frames: list) -> dict:
    process = psutil.Process(os.getpid())
    mem_before_mb = process.memory_info().rss / (1024 * 1024)

    lm1 = DriveGuardFaceLandmarker(running_mode="VIDEO")
    lm2 = DriveGuardFaceLandmarker(running_mode="VIDEO")

    def session_task(lm, offset_ts):
        session_lats = []
        for i, f in enumerate(frames):
            ts = offset_ts + i * 33.33
            t0 = time.perf_counter()
            lm.process_frame(f, timestamp_ms=ts)
            t1 = time.perf_counter()
            session_lats.append((t1 - t0) * 1000.0)
        return session_lats

    t_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as executor:
        f1 = executor.submit(session_task, lm1, 1000.0)
        f2 = executor.submit(session_task, lm2, 20000.0)
        lats1 = f1.result()
        lats2 = f2.result()
    total_time = time.perf_counter() - t_start

    mem_after_mb = process.memory_info().rss / (1024 * 1024)
    lm1.close()
    lm2.close()

    all_lats = np.array(lats1 + lats2)
    return {
        "mode": "DUAL_SESSION_VIDEO_CONCURRENT",
        "total_frames": len(frames) * 2,
        "total_time_s": total_time,
        "aggregate_fps": (len(frames) * 2) / total_time,
        "per_session_fps": len(frames) / total_time,
        "mean_latency_ms": float(np.mean(all_lats)),
        "p50_latency_ms": float(np.percentile(all_lats, 50)),
        "p95_latency_ms": float(np.percentile(all_lats, 95)),
        "mem_growth_mb": mem_after_mb - mem_before_mb,
    }


if __name__ == "__main__":
    print("=" * 70)
    print("   [DriveGuard Phase 3] MediaPipe IMAGE vs VIDEO Mode Benchmark")
    print("=" * 70)

    frames = generate_synthetic_driver_frames(n_frames=60)
    print(f"Generated {len(frames)} synthetic 640x480 test frames.")

    print("\n--- 1. Evaluating IMAGE Mode (Single Session) ---")
    res_img = run_single_session_benchmark("IMAGE", frames)
    print(f"  IMAGE Mean Latency:  {res_img['mean_latency_ms']:.2f} ms")
    print(f"  IMAGE p50 Latency:   {res_img['p50_latency_ms']:.2f} ms")
    print(f"  IMAGE p95 Latency:   {res_img['p95_latency_ms']:.2f} ms")
    print(f"  IMAGE Effective FPS: {res_img['effective_fps']:.1f} FPS")
    print(f"  IMAGE Memory Growth: {res_img['mem_growth_mb']:.1f} MB")
    print(f"  IMAGE Landmark Jitter: {res_img['mean_jitter_px']:.3f} px")

    print("\n--- 2. Evaluating VIDEO Mode (Single Session) ---")
    res_vid = run_single_session_benchmark("VIDEO", frames)
    print(f"  VIDEO Mean Latency:  {res_vid['mean_latency_ms']:.2f} ms")
    print(f"  VIDEO p50 Latency:   {res_vid['p50_latency_ms']:.2f} ms")
    print(f"  VIDEO p95 Latency:   {res_vid['p95_latency_ms']:.2f} ms")
    print(f"  VIDEO Effective FPS: {res_vid['effective_fps']:.1f} FPS")
    print(f"  VIDEO Memory Growth: {res_vid['mem_growth_mb']:.1f} MB")
    print(f"  VIDEO Landmark Jitter: {res_vid['mean_jitter_px']:.3f} px")

    print("\n--- 3. Evaluating Multi-Session Scalability (Dual Concurrent Sessions) ---")
    res_dual = run_dual_session_concurrent_benchmark(frames)
    print(f"  DUAL Aggregate Throughput: {res_dual['aggregate_fps']:.1f} FPS ({res_dual['per_session_fps']:.1f} FPS/session)")
    print(f"  DUAL Mean Latency:         {res_dual['mean_latency_ms']:.2f} ms")
    print(f"  DUAL p50 Latency:          {res_dual['p50_latency_ms']:.2f} ms")
    print(f"  DUAL p95 Latency:          {res_dual['p95_latency_ms']:.2f} ms")
    print(f"  DUAL Memory Footprint:     {res_dual['mem_growth_mb']:.1f} MB")
    print("=" * 70)
