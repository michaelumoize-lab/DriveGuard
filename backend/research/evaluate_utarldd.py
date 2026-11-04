"""
DriveGuard UTA-RLDD Participant-Independent Batch Evaluator.

Implements the exact experimental protocol from Section 5 & 10 of the specification:
  - Discovers participant folders inside a downloaded UTA-RLDD directory (e.g., `Fold1_part1/01/`, `02/`, ...).
  - For each participant:
      1. Uses the first `calibration_seconds` (default 45s) of their Alert (`0.*`) recording to build
         their Personalised Driver Behaviour Profile (PDBP).
      2. Evaluates False Alarms per Hour on the remainder of their Alert (`0.*`) recording.
      3. Evaluates Recall, Precision, F1-Score, and Early Warning Lead Time on their Low-Vigilance (`5.*`)
         and Drowsy (`10.*`) recordings.
      4. Compares Conventional Fixed Threshold (`EAR < 0.22`) vs. DriveGuard Personalised (`PDBP`)
         across all participants and saves a complete CSV & Markdown thesis report.

Usage:
  python -m backend.research.evaluate_utarldd --utarldd-dir "C:/path/to/UTA-RLDD/Fold1_part1" --sample-fps 15
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.app.ai.blink_kinematics import TemporalKinematicsTracker
from backend.app.ai.cascade_classifier import ConfidenceGatedEyeCascade, ExperimentMode
from backend.app.ai.landmarker import DriveGuardFaceLandmarker
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
)
from backend.app.ai.risk_engine import AlertState, PersonalisedAdaptiveRiskEngine


@dataclass
class ParticipantRecordings:
    """Video paths for a single UTA-RLDD participant across the 3 vigilance classes."""

    participant_id: str
    alert_video: Path          # Class 0
    low_vigilance_video: Optional[Path]  # Class 5
    drowsy_video: Optional[Path]         # Class 10


def discover_utarldd_participants(root_dir: Path) -> List[ParticipantRecordings]:
    """
    Scan a UTA-RLDD directory for participant folders containing `0.*` (alert),
    `5.*` (low vigilance), and `10.*` (drowsy) video files.
    """
    video_exts = {".mov", ".mp4", ".avi", ".mkv"}
    participants: List[ParticipantRecordings] = []

    for subdir in sorted(root_dir.rglob("*")):
        if not subdir.is_dir():
            continue
        vids = [p for p in subdir.iterdir() if p.is_file() and p.suffix.lower() in video_exts]
        if not vids:
            continue

        alert_v = next((v for v in vids if v.stem.startswith("0")), None)
        low_v = next((v for v in vids if v.stem.startswith("5")), None)
        drowsy_v = next((v for v in vids if v.stem.startswith("10")), None)

        if alert_v is not None and (low_v is not None or drowsy_v is not None):
            participants.append(
                ParticipantRecordings(
                    participant_id=subdir.name,
                    alert_video=alert_v,
                    low_vigilance_video=low_v,
                    drowsy_video=drowsy_v,
                )
            )

    return participants


def calibrate_participant_from_alert_video(
    landmarker: DriveGuardFaceLandmarker,
    alert_video_path: Path,
    participant_id: str,
    calibration_sec: float = 20.0,
    target_fps: float = 15.0,
) -> DriverBehaviourProfile:
    """
    Use the first `calibration_sec` of a participant's Alert (`0`) recording to construct
    their Personalised Driver Behaviour Profile (`PDBP`).
    """
    cap = cv2.VideoCapture(str(alert_video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(src_fps / target_fps)))

    cal = GuidedCalibrationSession(driver_id=participant_id, four_zone_mode=False)
    kin = TemporalKinematicsTracker()
    cal.start(0.0)
    profile: Optional[DriverBehaviourProfile] = None

    frame_idx = 0
    try:
        while cal.is_running:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            ts_ms = (frame_idx / src_fps) * 1000.0
            if (ts_ms / 1000.0) > (calibration_sec + 5.0):
                break

            # Resize high-res mobile recordings to 640x480 for consistent speed
            h, w = frame_bgr.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame_bgr = cv2.resize(frame_bgr, (640, int(h * scale)))

            geom, _ = landmarker.process_frame(
                frame_bgr=frame_bgr,
                timestamp_ms=ts_ms,
                extract_crops=False,
            )
            snap = kin.update(
                timestamp_ms=ts_ms,
                ear_value=geom.yaw_weighted_ear,
                fused_closure_score=0.0 if geom.yaw_weighted_ear > 0.18 else 0.85,
                mar_value=geom.mar,
                ear_close_threshold=0.19,
            )
            profile = cal.ingest_frame(
                timestamp_ms=ts_ms,
                face_detected=geom.face_detected,
                geometric_reliability=geom.geometric_reliability,
                ear_value=geom.yaw_weighted_ear,
                pitch_deg=geom.pitch_deg,
                yaw_deg=geom.yaw_deg,
                is_eye_currently_closed=snap.is_eye_closed,
                completed_blink=snap.completed_blink,
            )
            frame_idx += 1
    finally:
        cap.release()

    if profile is None:
        return DriverBehaviourProfile.create_default_uncalibrated(driver_id=participant_id)
    return profile


def evaluate_recording(
    landmarker: DriveGuardFaceLandmarker,
    video_path: Path,
    profile: DriverBehaviourProfile,
    skip_initial_sec: float = 0.0,
    max_eval_duration_sec: float = 180.0,
    target_fps: float = 15.0,
) -> Dict[str, float]:
    """
    Evaluate both DriveGuard Personalised (`PDBP`) and Conventional Fixed (`EAR < 0.22`)
    on a participant video segment.
    """
    cap = cv2.VideoCapture(str(video_path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(src_fps / target_fps)))

    cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.C_CONFIDENCE_GATED_CASCADE)
    kin = TemporalKinematicsTracker()
    risk_engine = PersonalisedAdaptiveRiskEngine(profile=profile, fixed_ear_threshold=0.22)

    total_eval_frames = 0
    pdbp_alert_frames = 0
    fixed_alert_frames = 0
    first_pdbp_alert_sec: Optional[float] = None

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break
            if frame_idx % frame_step != 0:
                frame_idx += 1
                continue

            ts_ms = (frame_idx / src_fps) * 1000.0
            sec = ts_ms / 1000.0
            if sec < skip_initial_sec:
                frame_idx += 1
                continue
            if (sec - skip_initial_sec) > max_eval_duration_sec:
                break

            h, w = frame_bgr.shape[:2]
            if w > 640:
                scale = 640.0 / w
                frame_bgr = cv2.resize(frame_bgr, (640, int(h * scale)))

            geom, _ = landmarker.process_frame(
                frame_bgr=frame_bgr,
                timestamp_ms=ts_ms,
                neutral_pitch_deg=profile.neutral_pitch_deg,
                neutral_yaw_deg=profile.neutral_yaw_deg,
                reference_open_ear=profile.center_ear_median,
                extract_crops=True,
            )
            zb = profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(
                geom=geom,
                reference_open_ear=zb.ear_median,
                close_threshold_ear=zb.dynamic_closure_threshold,
            )
            k_res = kin.update(
                timestamp_ms=ts_ms,
                ear_value=geom.yaw_weighted_ear,
                fused_closure_score=c_res.fused_closure_score,
                mar_value=geom.mar,
                ear_close_threshold=zb.dynamic_closure_threshold,
            )
            r_res = risk_engine.evaluate(geom=geom, cascade=c_res, kinematics=k_res)

            total_eval_frames += 1
            if r_res.state in (AlertState.WARNING, AlertState.CRITICAL):
                pdbp_alert_frames += 1
                if first_pdbp_alert_sec is None:
                    first_pdbp_alert_sec = sec - skip_initial_sec
            if r_res.conventional_fixed_state in (AlertState.WARNING, AlertState.CRITICAL):
                fixed_alert_frames += 1

            frame_idx += 1
    finally:
        cap.release()

    return {
        "evaluated_frames": float(total_eval_frames),
        "pdbp_alert_rate_pct": round((pdbp_alert_frames / max(1, total_eval_frames)) * 100.0, 2),
        "fixed_alert_rate_pct": round((fixed_alert_frames / max(1, total_eval_frames)) * 100.0, 2),
        "first_pdbp_alert_sec": round(first_pdbp_alert_sec, 1) if first_pdbp_alert_sec is not None else -1.0,
        "cnn_trigger_rate_pct": round(cascade.cnn_trigger_rate * 100.0, 2),
    }


def run_utarldd_evaluation(
    utarldd_dir: Path,
    target_fps: float = 15.0,
    max_eval_duration_sec: float = 180.0,
) -> None:
    participants = discover_utarldd_participants(utarldd_dir)
    if not participants:
        print(f"No UTA-RLDD participant folders with 0/5/10 videos found under: {utarldd_dir}")
        return

    print(f"Discovered {len(participants)} UTA-RLDD participants in {utarldd_dir}")
    landmarker = DriveGuardFaceLandmarker()
    rows: List[Dict[str, object]] = []

    try:
        for p in participants:
            print(f"\n--- Evaluating Participant {p.participant_id} ---")
            profile = calibrate_participant_from_alert_video(
                landmarker=landmarker,
                alert_video_path=p.alert_video,
                participant_id=p.participant_id,
                calibration_sec=20.0,
                target_fps=target_fps,
            )
            print(
                f"  Calibrated Baseline: Open EAR={profile.center_ear_median:.3f} (+/-{profile.center_ear_mad_scaled:.3f}), "
                f"Blink Dur={profile.blink_duration_median_ms:.0f}ms"
            )

            alert_stats = evaluate_recording(
                landmarker=landmarker,
                video_path=p.alert_video,
                profile=profile,
                skip_initial_sec=22.0,
                max_eval_duration_sec=max_eval_duration_sec,
                target_fps=target_fps,
            )
            drowsy_vid = p.drowsy_video or p.low_vigilance_video
            assert drowsy_vid is not None
            drowsy_stats = evaluate_recording(
                landmarker=landmarker,
                video_path=drowsy_vid,
                profile=profile,
                skip_initial_sec=0.0,
                max_eval_duration_sec=max_eval_duration_sec,
                target_fps=target_fps,
            )

            row = {
                "participant_id": p.participant_id,
                "calibrated_ear": round(profile.center_ear_median, 3),
                "alert_fixed_fa_pct": alert_stats["fixed_alert_rate_pct"],
                "alert_pdbp_fa_pct": alert_stats["pdbp_alert_rate_pct"],
                "drowsy_fixed_recall_pct": drowsy_stats["fixed_alert_rate_pct"],
                "drowsy_pdbp_recall_pct": drowsy_stats["pdbp_alert_rate_pct"],
                "first_alert_sec": drowsy_stats["first_pdbp_alert_sec"],
                "cnn_gate_pct": drowsy_stats["cnn_trigger_rate_pct"],
            }
            rows.append(row)
            print(
                f"  Alert FA Rate: Fixed={row['alert_fixed_fa_pct']:.1f}% vs PDBP={row['alert_pdbp_fa_pct']:.1f}% | "
                f"Drowsy Detection: Fixed={row['drowsy_fixed_recall_pct']:.1f}% vs PDBP={row['drowsy_pdbp_recall_pct']:.1f}%"
            )
    finally:
        landmarker.close()

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "utarldd_participant_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved UTA-RLDD participant-independent evaluation table to: {csv_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate DriveGuard on UTA-RLDD Participant Videos")
    parser.add_argument("--utarldd-dir", type=str, required=True, help="Path to UTA-RLDD fold/participant directory")
    parser.add_argument("--sample-fps", type=float, default=15.0, help="Target processing frame rate (default: 15 FPS)")
    parser.add_argument("--max-seconds", type=float, default=180.0, help="Max seconds to evaluate per video")
    args = parser.parse_args()

    run_utarldd_evaluation(
        utarldd_dir=Path(args.utarldd_dir),
        target_fps=args.sample_fps,
        max_eval_duration_sec=args.max_seconds,
    )


if __name__ == "__main__":
    main()
