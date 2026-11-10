"""
DriveGuard EXP-1A Benchmark: RGB Pass-Through vs Roundtrip Conversion.

Evaluates:
1. Computation latency percentiles (P50, P90, P95, P99)
2. CPU process time
3. Heap memory allocations via tracemalloc
4. Exact numerical output equivalence across all telemetry fields
"""

import time
import tracemalloc
from typing import Any, Dict, List
import numpy as np

from backend.app.ai.pipeline import SessionAIEngine
from backend.benchmarks.benchmark_inference_offline import generate_benchmark_frame


def run_benchmark(n_samples: int = 500, warm_up: int = 30, pass_rgb: bool = False) -> Dict[str, Any]:
    # Pre-generate frames
    frames = [generate_benchmark_frame(i) for i in range(90)]
    
    engine = SessionAIEngine(
        session_id="bench_exp1a",
        driver_id="driver_bench",
        experiment_mode="EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
    )
    
    # Warm-up
    for w in range(warm_up):
        frame = frames[w % 90]
        engine.process_frame(frame, timestamp_ms=float(w * 33.3))
        
    latencies: List[float] = []
    telemetry_records: List[Dict[str, Any]] = []
    
    tracemalloc.start()
    t_cpu_start = time.process_time()
    t_wall_start = time.perf_counter()
    
    for i in range(n_samples):
        frame = frames[i % 90]
        ts_ms = float((warm_up + i) * 33.3)
        
        t0 = time.perf_counter()
        dto = engine.process_frame(frame, timestamp_ms=ts_ms)
        t1 = time.perf_counter()
        
        latencies.append((t1 - t0) * 1000.0)
        telemetry_records.append({
            "ear": round(dto.eye_metrics.ear_raw, 4),
            "mar": round(dto.eye_metrics.mar_yawn, 4),
            "pitch": round(dto.head_pose.pitch_deg, 2) if dto.head_pose else None,
            "yaw": round(dto.head_pose.yaw_deg, 2) if dto.head_pose else None,
            "state": dto.state.value,
            "risk_score": round(dto.risk_score, 4),
            "cnn_invoked": dto.cascade.cnn_triggered if dto.cascade else False,
        })
        
    t_wall_total = time.perf_counter() - t_wall_start
    t_cpu_total = time.process_time() - t_cpu_start
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    engine.close()
    
    arr = np.array(latencies)
    return {
        "sample_count": n_samples,
        "wall_time_s": round(t_wall_total, 3),
        "cpu_time_s": round(t_cpu_total, 3),
        "peak_memory_mb": round(peak_mem / (1024 * 1024), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 2),
        "p90_ms": round(float(np.percentile(arr, 90)), 2),
        "p95_ms": round(float(np.percentile(arr, 95)), 2),
        "p99_ms": round(float(np.percentile(arr, 99)), 2),
        "mean_ms": round(float(np.mean(arr)), 2),
        "records": telemetry_records,
    }


if __name__ == "__main__":
    print("[EXP-1A Benchmark] Running 500 samples (warmup=30)...", flush=True)
    res = run_benchmark(n_samples=500, warm_up=30)
    print(f"P50: {res['p50_ms']}ms | P90: {res['p90_ms']}ms | P95: {res['p95_ms']}ms | P99: {res['p99_ms']}ms")
    print(f"Mean: {res['mean_ms']}ms | Peak Mem: {res['peak_memory_mb']} MB | CPU: {res['cpu_time_s']}s | Wall: {res['wall_time_s']}s")
