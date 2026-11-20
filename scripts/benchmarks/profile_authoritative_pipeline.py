"""
DriveGuard Phase 9.4 — Authoritative Pipeline Stage-by-Stage Performance Profiler.

Measures per-stage latency (mean, P50, P95, P99) and effective FPS across:
  1. Encoded JPEG decode (`EncodedBytesFrameAdapter`)
  2. Color conversion (BGR -> RGB / `ArrayFrameAdapter`)
  3. Frame quality analysis (`FrameQualityAnalyzer`)
  4. MediaPipe Face Landmarker inference (`DriveGuardFaceLandmarker`)
  5. Feature extraction (`DriverFeatureExtractor`)
  6. Drowsiness decision engine (`DrowsinessEngine`)
  7. Alert state manager (`AlertStateManager`)
  8. Full `SessionAIEngine.process_frame_rgb()` + `FrameTelemetryDTO.model_dump_json()` serialization

Outputs both a human-readable summary table and JSON metrics for reproducible tracking.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ai.pipeline import SessionAIEngine
from backend.app.services.frame_adapters import ArrayFrameAdapter, EncodedBytesFrameAdapter


def _percentile(values: List[float], pct: float) -> float:
    if not values:
        return 0.0
    arr = np.asarray(values, dtype=np.float64)
    return float(np.percentile(arr, pct))


def _summarize(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": round(float(np.mean(arr)), 3),
        "p50_ms": round(float(np.percentile(arr, 50)), 3),
        "p95_ms": round(float(np.percentile(arr, 95)), 3),
        "p99_ms": round(float(np.percentile(arr, 99)), 3),
    }


def _make_realistic_frame(width: int, height: int) -> np.ndarray:
    """Create a realistic textured BGR test frame (avoiding uniform zero-byte shortcuts)."""
    rng = np.random.default_rng(42)
    base = rng.integers(70, 180, size=(height, width, 3), dtype=np.uint8)
    # Draw an illuminated oval in center to resemble a driver head region
    cv2.ellipse(
        base,
        (width // 2, height // 2),
        (width // 6, height // 3),
        0,
        0,
        360,
        (150, 175, 205),
        -1,
    )
    return base


def profile_resolution(width: int, height: int, frames: int, warmup: int) -> Dict[str, object]:
    bgr_frame = _make_realistic_frame(width, height)
    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    ok, jpeg_buf = cv2.imencode(".jpg", bgr_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    assert ok
    jpeg_bytes = jpeg_buf.tobytes()

    engine = SessionAIEngine(
        session_id=f"prof_{width}x{height}",
        driver_id="profile_driver",
    )
    svc = engine.monitoring_service
    jpeg_adapter = EncodedBytesFrameAdapter()
    array_adapter = ArrayFrameAdapter()

    stages: Dict[str, List[float]] = {
        "jpeg_decode": [],
        "rgb_adapt": [],
        "frame_quality": [],
        "mediapipe_landmarker": [],
        "feature_extraction": [],
        "drowsiness_engine": [],
        "alert_manager": [],
        "dto_json_serialize": [],
        "full_session_rgb_pipeline": [],
    }

    try:
        # Warmup frames (initialize MediaPipe XNNPACK delegates / caches)
        for i in range(warmup):
            ts = float((i + 1) * 40.0)
            engine.process_frame(rgb_frame, timestamp_ms=ts, is_rgb=True)

        base_ts = float((warmup + 1) * 40.0)

        # 1. Stage-isolated breakdown using real authoritative components
        for i in range(frames):
            ts = base_ts + float(i * 40.0)

            t0 = time.perf_counter()
            jpeg_adapter.adapt(jpeg_bytes, timestamp_ms=ts)
            stages["jpeg_decode"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            adapted = array_adapter.adapt_rgb(rgb_frame, timestamp_ms=ts)
            stages["rgb_adapt"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            fq = svc.quality_analyzer.analyze(adapted.frame_bgr)
            stages["frame_quality"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            lm = svc.landmarker.detect(adapted.frame_bgr, int(ts))
            stages["mediapipe_landmarker"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            feats = svc.feature_extractor.extract(
                lm,
                frame_width=adapted.frame_width,
                frame_height=adapted.frame_height,
                is_mirrored=True,
            )
            stages["feature_extraction"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            assessment = svc.drowsiness_engine.evaluate(feats)
            stages["drowsiness_engine"].append((time.perf_counter() - t0) * 1000.0)

            t0 = time.perf_counter()
            svc.alert_manager.update(assessment)
            stages["alert_manager"].append((time.perf_counter() - t0) * 1000.0)

        # 2. End-to-end SessionAIEngine.process_frame(..., is_rgb=True) + JSON serialization
        e2e_base_ts = base_ts + float((frames + 5) * 40.0)
        for i in range(frames):
            ts = e2e_base_ts + float(i * 40.0)
            t0 = time.perf_counter()
            dto = engine.process_frame(rgb_frame, timestamp_ms=ts, is_rgb=True)
            t_proc = (time.perf_counter() - t0) * 1000.0
            stages["full_session_rgb_pipeline"].append(t_proc)

            t_ser0 = time.perf_counter()
            _ = dto.model_dump_json()
            stages["dto_json_serialize"].append((time.perf_counter() - t_ser0) * 1000.0)
    finally:
        engine.close()

    summaries = {k: _summarize(v) for k, v in stages.items()}
    mean_total_ms = summaries["full_session_rgb_pipeline"]["mean_ms"]
    effective_fps = round(1000.0 / mean_total_ms, 2) if mean_total_ms > 0 else 0.0

    stage_sum_ms = sum(
        summaries[k]["mean_ms"]
        for k in (
            "rgb_adapt",
            "frame_quality",
            "mediapipe_landmarker",
            "feature_extraction",
            "drowsiness_engine",
            "alert_manager",
            "dto_json_serialize",
        )
    )
    shares = {
        k: round((summaries[k]["mean_ms"] / stage_sum_ms) * 100.0, 1) if stage_sum_ms > 0 else 0.0
        for k in (
            "rgb_adapt",
            "frame_quality",
            "mediapipe_landmarker",
            "feature_extraction",
            "drowsiness_engine",
            "alert_manager",
            "dto_json_serialize",
        )
    }

    return {
        "resolution": f"{width}x{height}",
        "frames_profiled": frames,
        "warmup_frames": warmup,
        "effective_fps": effective_fps,
        "stages": summaries,
        "stage_share_pct": shares,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile DriveGuard authoritative pipeline stages.")
    parser.add_argument("--frames", type=int, default=40, help="Frames per resolution")
    parser.add_argument("--warmup", type=int, default=8, help="Warmup frames")
    parser.add_argument("--json-out", type=str, default="", help="Optional path to save JSON report")
    args = parser.parse_args()

    results = [
        profile_resolution(640, 480, frames=args.frames, warmup=args.warmup),
        profile_resolution(1280, 720, frames=args.frames, warmup=args.warmup),
    ]

    print("\n=== DriveGuard Authoritative Pipeline Performance Profile ===")
    for res in results:
        print(f"\nResolution: {res['resolution']} | Effective Throughput: {res['effective_fps']} FPS")
        print(f"{'Stage':<28} {'Mean(ms)':>9} {'P50(ms)':>9} {'P95(ms)':>9} {'P99(ms)':>9} {'Share%':>8}")
        print("-" * 76)
        for name, stats in res["stages"].items():
            share = res["stage_share_pct"].get(name, 0.0)
            share_str = f"{share:5.1f}%" if name in res["stage_share_pct"] else "   -  "
            print(
                f"{name:<28} {stats['mean_ms']:9.3f} {stats['p50_ms']:9.3f} "
                f"{stats['p95_ms']:9.3f} {stats['p99_ms']:9.3f} {share_str:>8}"
            )

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved JSON profile to: {out_path}")


if __name__ == "__main__":
    main()
