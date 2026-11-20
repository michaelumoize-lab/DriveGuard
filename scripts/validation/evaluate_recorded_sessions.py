"""
DriveGuard Phase 9.5 — Repeatable Real-World Validation Session Evaluator.

Computes standardized empirical metrics from recorded `FrameTelemetryDTO` JSONL
logs paired with ground-truth episode/zone annotations:
  - Critical Event Recall & False Negative Rate (FNR)
  - Attentive False Positive Count & False Alerts per Hour
  - Critical Detection Latency (P50 / P95 ms from episode onset)
  - Pose-Zone Classification Accuracy
  - Face Visibility & Frame Processing Latency (P50 / P95 ms)

Includes `--self-test` for deterministic verification without external datasets.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def evaluate_session(
    telemetry_frames: List[Dict[str, Any]],
    ground_truth: Dict[str, Any],
    detection_grace_after_end_ms: float = 1000.0,
) -> Dict[str, Any]:
    """Compute quantitative validation metrics for one recorded session."""
    frames = sorted(telemetry_frames, key=lambda f: float(f.get("timestamp_ms", 0.0)))
    if not frames:
        raise ValueError("Telemetry log contains zero frames.")

    # 1. Critical episode recall, FNR, and detection latency
    episodes = ground_truth.get("critical_episodes", [])
    detected_count = 0
    latencies_ms: List[float] = []
    for ep in episodes:
        onset = float(ep["onset_ms"])
        end = float(ep["end_ms"]) + detection_grace_after_end_ms
        matched_ts: Optional[float] = None
        for fr in frames:
            ts = float(fr.get("timestamp_ms", 0.0))
            if onset <= ts <= end:
                state = str(fr.get("state", ""))
                d_state = str(fr.get("drowsiness_state", ""))
                if state == "CRITICAL" or d_state == "CRITICAL_WARNING":
                    matched_ts = ts
                    break
        if matched_ts is not None:
            detected_count += 1
            latencies_ms.append(max(0.0, matched_ts - onset))

    total_episodes = len(episodes)
    recall = (detected_count / total_episodes) if total_episodes > 0 else 1.0
    fnr = 1.0 - recall

    # 2. Attentive intervals false positive rate (warnings/criticals during attentive driving)
    attentive_intervals = ground_truth.get("attentive_intervals", [])
    attentive_duration_ms = 0.0
    false_positive_events = 0
    for iv in attentive_intervals:
        start = float(iv["start_ms"])
        end = float(iv["end_ms"])
        attentive_duration_ms += max(0.0, end - start)
        in_fp_episode = False
        for fr in frames:
            ts = float(fr.get("timestamp_ms", 0.0))
            if start <= ts <= end:
                is_alert = str(fr.get("state", "")) in ("WARNING", "CRITICAL")
                if is_alert and not in_fp_episode:
                    false_positive_events += 1
                    in_fp_episode = True
                elif not is_alert:
                    in_fp_episode = False

    attentive_hours = attentive_duration_ms / 3_600_000.0
    false_alerts_per_hour = (
        round(false_positive_events / attentive_hours, 2) if attentive_hours > 0 else 0.0
    )

    # 3. Pose-zone accuracy
    zone_intervals = ground_truth.get("zone_intervals", [])
    zone_total = 0
    zone_correct = 0
    for ziv in zone_intervals:
        start = float(ziv["start_ms"])
        end = float(ziv["end_ms"])
        expected_zone = str(ziv["zone"])
        for fr in frames:
            ts = float(fr.get("timestamp_ms", 0.0))
            if start <= ts <= end and fr.get("face_detected", True):
                zone_total += 1
                if str(fr.get("pose_zone", "")) == expected_zone:
                    zone_correct += 1
    zone_accuracy = (zone_correct / zone_total) if zone_total > 0 else None

    # 4. Runtime inference latency & face tracking coverage
    infer_latencies = [
        float(fr["inference_latency_ms"])
        for fr in frames
        if fr.get("inference_latency_ms") is not None
    ]
    face_detected_frames = sum(1 for fr in frames if fr.get("face_detected"))

    return {
        "session_id": ground_truth.get("session_id", "unknown"),
        "frames_evaluated": len(frames),
        "face_tracking_coverage_pct": round((face_detected_frames / len(frames)) * 100.0, 2),
        "critical_episodes_total": total_episodes,
        "critical_episodes_detected": detected_count,
        "critical_recall": round(recall, 4),
        "critical_fnr": round(fnr, 4),
        "detection_latency_p50_ms": round(float(np.percentile(latencies_ms, 50)), 2) if latencies_ms else None,
        "detection_latency_p95_ms": round(float(np.percentile(latencies_ms, 95)), 2) if latencies_ms else None,
        "attentive_false_positive_events": false_positive_events,
        "false_alerts_per_hour": false_alerts_per_hour,
        "pose_zone_accuracy": round(zone_accuracy, 4) if zone_accuracy is not None else None,
        "pose_zone_frames_evaluated": zone_total,
        "inference_latency_p50_ms": round(float(np.percentile(infer_latencies, 50)), 2) if infer_latencies else None,
        "inference_latency_p95_ms": round(float(np.percentile(infer_latencies, 95)), 2) if infer_latencies else None,
        "certification_disclaimer": "Empirical evaluation report only — not road-safety or automotive certified.",
    }


def _run_self_test() -> None:
    synthetic_frames = [
        {"timestamp_ms": 1000, "state": "NORMAL", "drowsiness_state": "NORMAL", "pose_zone": "CENTER", "face_detected": True, "inference_latency_ms": 22.0},
        {"timestamp_ms": 2000, "state": "NORMAL", "drowsiness_state": "NORMAL", "pose_zone": "LEFT_MIRROR", "face_detected": True, "inference_latency_ms": 24.0},
        {"timestamp_ms": 15000, "state": "WARNING", "drowsiness_state": "DROWSINESS_WARNING", "pose_zone": "CENTER", "face_detected": True, "inference_latency_ms": 25.0},
        {"timestamp_ms": 15180, "state": "CRITICAL", "drowsiness_state": "CRITICAL_WARNING", "pose_zone": "CENTER", "face_detected": True, "inference_latency_ms": 26.0},
        {"timestamp_ms": 18000, "state": "NORMAL", "drowsiness_state": "NORMAL", "pose_zone": "CENTER", "face_detected": True, "inference_latency_ms": 21.0},
    ]
    gt = {
        "session_id": "self_test_trial",
        "critical_episodes": [{"onset_ms": 15000, "end_ms": 17000, "label": "SUSTAINED_EYE_CLOSURE"}],
        "attentive_intervals": [{"start_ms": 0, "end_ms": 10000}],
        "zone_intervals": [
            {"start_ms": 0, "end_ms": 1500, "zone": "CENTER"},
            {"start_ms": 1501, "end_ms": 2500, "zone": "LEFT_MIRROR"},
        ],
    }
    report = evaluate_session(synthetic_frames, gt)
    assert report["critical_recall"] == 1.0
    assert report["critical_fnr"] == 0.0
    assert report["detection_latency_p50_ms"] == 180.0
    assert report["attentive_false_positive_events"] == 0
    assert report["pose_zone_accuracy"] == 1.0
    print("[PASS] Validation evaluator self-test passed:", json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate recorded DriveGuard validation sessions.")
    parser.add_argument("--telemetry", type=str, help="Path to recorded JSONL telemetry file")
    parser.add_argument("--ground-truth", type=str, help="Path to ground-truth JSON annotations")
    parser.add_argument("--report-out", type=str, default="", help="Optional path to write JSON report")
    parser.add_argument("--self-test", action="store_true", help="Run built-in deterministic self-test")
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    if not args.telemetry or not args.ground_truth:
        parser.error("--telemetry and --ground-truth are required unless --self-test is passed.")

    frames = [
        json.loads(line)
        for line in Path(args.telemetry).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gt = json.loads(Path(args.ground_truth).read_text(encoding="utf-8"))
    report = evaluate_session(frames, gt)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.report_out:
        out_path = Path(args.report_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
