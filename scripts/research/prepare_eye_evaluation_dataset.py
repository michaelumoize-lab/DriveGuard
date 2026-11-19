"""
DriveGuard — Phase B1.2A: Independent Eye-State Dataset Preparation.

Extracts representative eye observations from existing synchronized DriveGuard recordings
(stream.mp4 + telemetry.csv) and generates a blind annotation template for independent
human labelling without exposing EAR or CNN predictions.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Optional, Set, Tuple

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import cv2
import numpy as np

from backend.app.ai.geometry import (
    EyeCropQuality,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    extract_eye_crop_with_quality,
)
from backend.app.camera.base import FramePacket
from backend.app.vision.face_landmarker import DriveGuardFaceLandmarker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("prepare_eye_dataset")


def extract_context_crop(
    frame_bgr: np.ndarray,
    landmarks_pixel_xyz: np.ndarray,
    padding_scale: float = 1.8,
    target_size: Tuple[int, int] = (256, 128),
) -> Optional[np.ndarray]:
    """
    Extract a broad ocular-region context crop showing both eyes and brows.
    Provides human annotator with anatomical context without exposing model overlays.
    """
    h, w = frame_bgr.shape[:2]
    all_eye_indices = list(LEFT_EYE_INDICES) + list(RIGHT_EYE_INDICES)
    eye_pts = landmarks_pixel_xyz[all_eye_indices]

    x_min, y_min = np.min(eye_pts[:, :2], axis=0)
    x_max, y_max = np.max(eye_pts[:, :2], axis=0)

    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    box_w = max(48.0, (x_max - x_min) * padding_scale)
    box_h = max(24.0, (y_max - y_min) * padding_scale)

    x1 = max(0, int(center_x - box_w / 2.0))
    y1 = max(0, int(center_y - box_h / 2.0))
    x2 = min(w, int(center_x + box_w / 2.0))
    y2 = min(h, int(center_y + box_h / 2.0))

    if (x2 - x1) < 10 or (y2 - y1) < 10:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    resized = cv2.resize(crop, target_size, interpolation=cv2.INTER_AREA)
    return resized


def sample_session_frames(
    telemetry_rows: List[Dict[str, Any]],
    target_sample_count: int = 150,
    seed: int = 42,
) -> List[Tuple[int, str]]:
    """
    Sample representative frames across five distinct categories:
    1. RANDOM_TIME: Uniform time-based samples independent of EAR.
    2. EAR_EVENT: Peak, onset, and recovery frames of candidate blink & closure events.
    3. OPTICAL_DEGRADED: Blurry frames or extreme exposure transitions.
    4. HEAD_POSE: Non-center mirror glances or elevated pitch/yaw angles.
    5. BORDERLINE_EAR: Frames near the geometric closure threshold.
    """
    rng = random.Random(seed)
    total_rows = len(telemetry_rows)
    if total_rows == 0:
        return []

    sampled_map: Dict[int, str] = {}  # frame_idx (0-based) -> category

    # 1. RANDOM_TIME (~30% of target)
    random_count = max(10, int(target_sample_count * 0.30))
    step = max(1, total_rows // random_count)
    for i in range(0, total_rows, step):
        idx = min(total_rows - 1, i + rng.randint(0, max(0, step - 1)))
        sampled_map[idx] = "RANDOM_TIME"

    # 2. EAR_EVENT (~30% of target)
    # Detect contiguous segments of is_eye_closed == True
    closure_events: List[List[int]] = []
    current_event: List[int] = []
    for idx, row in enumerate(telemetry_rows):
        is_closed = str(row.get("is_eye_closed", "")).lower() == "true"
        if is_closed:
            current_event.append(idx)
        else:
            if current_event:
                closure_events.append(current_event)
                current_event = []
    if current_event:
        closure_events.append(current_event)

    for ev in closure_events:
        # Sample onset, peak (min EAR), and offset
        if len(ev) == 1:
            sampled_map[ev[0]] = "EAR_EVENT"
        elif len(ev) == 2:
            sampled_map[ev[0]] = "EAR_EVENT"
            sampled_map[ev[1]] = "EAR_EVENT"
        else:
            onset = ev[0]
            offset = ev[-1]
            # Find peak closure (lowest ear_compensated)
            peak_idx = ev[len(ev) // 2]
            lowest_ear = float("inf")
            for f_idx in ev:
                try:
                    ear_val = float(telemetry_rows[f_idx].get("ear_compensated", 1.0))
                    if ear_val < lowest_ear:
                        lowest_ear = ear_val
                        peak_idx = f_idx
                except (ValueError, TypeError):
                    pass
            sampled_map[onset] = "EAR_EVENT"
            sampled_map[peak_idx] = "EAR_EVENT"
            sampled_map[offset] = "EAR_EVENT"

    # 3. OPTICAL_DEGRADED (~15% of target)
    blurry_indices = [
        idx for idx, row in enumerate(telemetry_rows)
        if row.get("optical_status") == "BLURRY"
    ]
    if blurry_indices:
        blurry_sample = rng.sample(blurry_indices, min(len(blurry_indices), int(target_sample_count * 0.15)))
        for idx in blurry_sample:
            if idx not in sampled_map:
                sampled_map[idx] = "OPTICAL_DEGRADED"

    # 4. HEAD_POSE (~15% of target)
    head_pose_indices = [
        idx for idx, row in enumerate(telemetry_rows)
        if row.get("pose_zone") in ("RIGHT_MIRROR", "LEFT_MIRROR", "DASHBOARD", "OUT_OF_BOUNDS")
        or abs(float(row.get("yaw_deg") or 0.0)) > 20.0
        or abs(float(row.get("pitch_deg") or 0.0)) > 15.0
    ]
    if head_pose_indices:
        pose_sample = rng.sample(head_pose_indices, min(len(head_pose_indices), int(target_sample_count * 0.15)))
        for idx in pose_sample:
            if idx not in sampled_map:
                sampled_map[idx] = "HEAD_POSE"

    # 5. BORDERLINE_EAR (~10% of target)
    borderline_indices = []
    for idx, row in enumerate(telemetry_rows):
        try:
            ear_c = float(row.get("ear_compensated") or 0.0)
            thresh = float(row.get("ear_closure_threshold") or 0.175)
            if abs(ear_c - thresh) <= 0.035 and str(row.get("is_geometric_valid", "")).lower() == "true":
                borderline_indices.append(idx)
        except (ValueError, TypeError):
            pass

    if borderline_indices:
        border_sample = rng.sample(borderline_indices, min(len(borderline_indices), int(target_sample_count * 0.10)))
        for idx in border_sample:
            if idx not in sampled_map:
                sampled_map[idx] = "BORDERLINE_EAR"

    # Sort deterministically by frame index
    sorted_samples = sorted(sampled_map.items(), key=lambda x: x[0])
    return sorted_samples


def process_dataset(
    recordings_dir: Path,
    output_dir: Path,
    target_frames_per_session: int = 140,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Extracts crops and generates the independent annotation template.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    crops_base_dir = output_dir / "crops"
    crops_base_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in recordings_dir.iterdir() if d.is_dir() and d.name.startswith("session_")])
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found in: {recordings_dir}")

    landmarker = DriveGuardFaceLandmarker()

    manifest_samples: Dict[str, Any] = {}
    csv_rows: List[Dict[str, str]] = []

    total_frames_extracted = 0
    total_left_crops = 0
    total_right_crops = 0
    rejections_count = 0
    rejection_reasons: Dict[str, int] = {}
    samples_by_category: Dict[str, int] = {}
    samples_by_session: Dict[str, int] = {}

    try:
        for s_dir in session_dirs:
            session_id = s_dir.name
            video_path = s_dir / "stream.mp4"
            csv_path = s_dir / "telemetry.csv"

            if not video_path.is_file() or not csv_path.is_file():
                logger.warning("Skipping session %s: missing video or telemetry", session_id)
                continue

            logger.info("Processing session: %s", session_id)
            session_crops_dir = crops_base_dir / session_id
            session_crops_dir.mkdir(parents=True, exist_ok=True)

            with open(csv_path, "r", encoding="utf-8") as f:
                telemetry_rows = list(csv.DictReader(f))

            cap = cv2.VideoCapture(str(video_path))
            total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info("  Total frames: Video=%d, CSV=%d", total_v_frames, len(telemetry_rows))

            sampled_frames = sample_session_frames(
                telemetry_rows=telemetry_rows,
                target_sample_count=target_frames_per_session,
                seed=seed,
            )
            logger.info("  Sampled candidate frames: %d", len(sampled_frames))

            for frame_idx, category in sampled_frames:
                t_row = telemetry_rows[frame_idx]
                frame_id = int(t_row.get("frame_id", frame_idx + 1))
                timestamp_ms = float(t_row.get("timestamp_ms", 0.0))

                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame_bgr = cap.read()
                if not ret or frame_bgr is None:
                    rejections_count += 1
                    rejection_reasons["FRAME_READ_ERROR"] = rejection_reasons.get("FRAME_READ_ERROR", 0) + 1
                    continue

                h, w = frame_bgr.shape[:2]
                packet = FramePacket(
                    frame=frame_bgr,
                    timestamp_ns=int(timestamp_ms * 1_000_000),
                    frame_index=frame_id,
                    source_id=f"eval:{session_id}",
                )

                lm_res = landmarker.detect(packet)
                if not lm_res.has_face or not lm_res.landmarks:
                    rejections_count += 1
                    rejection_reasons["NO_FACE_DETECTED"] = rejection_reasons.get("NO_FACE_DETECTED", 0) + 1
                    continue

                lms_xyz = np.array([[p.x * w, p.y * h, p.z * w] for p in lm_res.landmarks], dtype=np.float64)
                frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

                left_crop_q = extract_eye_crop_with_quality(
                    frame_rgb=frame_rgb,
                    landmarks_pixel_xyz=lms_xyz,
                    eye_indices=LEFT_EYE_INDICES,
                    output_size=(64, 64),
                )
                right_crop_q = extract_eye_crop_with_quality(
                    frame_rgb=frame_rgb,
                    landmarks_pixel_xyz=lms_xyz,
                    eye_indices=RIGHT_EYE_INDICES,
                    output_size=(64, 64),
                )

                context_crop = extract_context_crop(frame_bgr, lms_xyz)

                # Base sample naming
                base_sample_id = f"{session_id}_F{frame_id:05d}"
                context_rel_path = f"crops/{session_id}/sample_{base_sample_id}_context.png"
                if context_crop is not None:
                    cv2.imwrite(str(output_dir / context_rel_path), context_crop)

                # Process LEFT EYE
                left_sample_id = f"{base_sample_id}_L"
                left_rel_path = f"crops/{session_id}/sample_{left_sample_id}.png"
                if left_crop_q is not None and left_crop_q.crop_rgb is not None:
                    cv2.imwrite(str(output_dir / left_rel_path), cv2.cvtColor(left_crop_q.crop_rgb, cv2.COLOR_RGB2BGR))
                    total_left_crops += 1

                    csv_rows.append({
                        "sample_id": left_sample_id,
                        "session_id": session_id,
                        "frame_id": str(frame_id),
                        "timestamp_ms": f"{timestamp_ms:.2f}",
                        "eye_side": "LEFT",
                        "crop_path": left_rel_path,
                        "context_path": context_rel_path if context_crop is not None else "",
                        "sampling_category": category,
                        "human_label": "",  # Blind field to be populated by human annotator
                        "annotation_notes": "",
                    })

                    manifest_samples[left_sample_id] = {
                        "sample_id": left_sample_id,
                        "session_id": session_id,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "eye_side": "LEFT",
                        "sampling_category": category,
                        "crop_path": left_rel_path,
                        "context_path": context_rel_path if context_crop is not None else "",
                        "crop_quality": {
                            "sharpness_score": round(left_crop_q.sharpness_score, 4),
                            "exposure_score": round(left_crop_q.exposure_score, 4),
                            "glare_fraction": round(left_crop_q.glare_fraction, 4),
                            "overall_reliability": round(left_crop_q.overall_reliability, 4),
                        },
                        "telemetry_ground_state": {
                            "raw_ear": float(t_row.get("ear_left") or 0.0),
                            "ear_compensated": float(t_row.get("ear_compensated") or 0.0),
                            "ear_closure_threshold": float(t_row.get("ear_closure_threshold") or 0.175),
                            "is_eye_closed_ear": str(t_row.get("is_eye_closed", "")).lower() == "true",
                            "is_geometric_valid": str(t_row.get("is_geometric_valid", "")).lower() == "true",
                            "pose_zone": t_row.get("pose_zone", "UNKNOWN"),
                            "pitch_deg": float(t_row.get("pitch_deg") or 0.0),
                            "yaw_deg": float(t_row.get("yaw_deg") or 0.0),
                            "roll_deg": float(t_row.get("roll_deg") or 0.0),
                            "luminance": float(t_row.get("luminance") or 0.0),
                            "blur_variance": float(t_row.get("blur_variance") or 0.0),
                            "optical_status": t_row.get("optical_status", "UNKNOWN"),
                        },
                    }
                else:
                    rejections_count += 1
                    rejection_reasons["LEFT_CROP_EXTRACTION_FAILED"] = rejection_reasons.get("LEFT_CROP_EXTRACTION_FAILED", 0) + 1

                # Process RIGHT EYE
                right_sample_id = f"{base_sample_id}_R"
                right_rel_path = f"crops/{session_id}/sample_{right_sample_id}.png"
                if right_crop_q is not None and right_crop_q.crop_rgb is not None:
                    cv2.imwrite(str(output_dir / right_rel_path), cv2.cvtColor(right_crop_q.crop_rgb, cv2.COLOR_RGB2BGR))
                    total_right_crops += 1

                    csv_rows.append({
                        "sample_id": right_sample_id,
                        "session_id": session_id,
                        "frame_id": str(frame_id),
                        "timestamp_ms": f"{timestamp_ms:.2f}",
                        "eye_side": "RIGHT",
                        "crop_path": right_rel_path,
                        "context_path": context_rel_path if context_crop is not None else "",
                        "sampling_category": category,
                        "human_label": "",
                        "annotation_notes": "",
                    })

                    manifest_samples[right_sample_id] = {
                        "sample_id": right_sample_id,
                        "session_id": session_id,
                        "frame_id": frame_id,
                        "timestamp_ms": timestamp_ms,
                        "eye_side": "RIGHT",
                        "sampling_category": category,
                        "crop_path": right_rel_path,
                        "context_path": context_rel_path if context_crop is not None else "",
                        "crop_quality": {
                            "sharpness_score": round(right_crop_q.sharpness_score, 4),
                            "exposure_score": round(right_crop_q.exposure_score, 4),
                            "glare_fraction": round(right_crop_q.glare_fraction, 4),
                            "overall_reliability": round(right_crop_q.overall_reliability, 4),
                        },
                        "telemetry_ground_state": {
                            "raw_ear": float(t_row.get("ear_right") or 0.0),
                            "ear_compensated": float(t_row.get("ear_compensated") or 0.0),
                            "ear_closure_threshold": float(t_row.get("ear_closure_threshold") or 0.175),
                            "is_eye_closed_ear": str(t_row.get("is_eye_closed", "")).lower() == "true",
                            "is_geometric_valid": str(t_row.get("is_geometric_valid", "")).lower() == "true",
                            "pose_zone": t_row.get("pose_zone", "UNKNOWN"),
                            "pitch_deg": float(t_row.get("pitch_deg") or 0.0),
                            "yaw_deg": float(t_row.get("yaw_deg") or 0.0),
                            "roll_deg": float(t_row.get("roll_deg") or 0.0),
                            "luminance": float(t_row.get("luminance") or 0.0),
                            "blur_variance": float(t_row.get("blur_variance") or 0.0),
                            "optical_status": t_row.get("optical_status", "UNKNOWN"),
                        },
                    }
                else:
                    rejections_count += 1
                    rejection_reasons["RIGHT_CROP_EXTRACTION_FAILED"] = rejection_reasons.get("RIGHT_CROP_EXTRACTION_FAILED", 0) + 1

                total_frames_extracted += 1
                samples_by_category[category] = samples_by_category.get(category, 0) + 1
                samples_by_session[session_id] = samples_by_session.get(session_id, 0) + 1

            cap.release()

    finally:
        landmarker.close()

    # 1. Write Annotation CSV Template (BLIND: Zero Model Predictions or Confidence Scores)
    annotation_csv_path = output_dir / "annotation_template.csv"
    with open(annotation_csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id",
            "session_id",
            "frame_id",
            "timestamp_ms",
            "eye_side",
            "crop_path",
            "context_path",
            "sampling_category",
            "human_label",
            "annotation_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in csv_rows:
            writer.writerow(row)
    logger.info("Saved blind annotation template (%d rows) to: %s", len(csv_rows), annotation_csv_path)

    # 2. Write Dataset Manifest JSON
    manifest = {
        "metadata": {
            "title": "DriveGuard Phase B1.2A Independent Eye-State Evaluation Dataset",
            "random_seed": seed,
            "target_frames_per_session": target_frames_per_session,
            "total_frames_extracted": total_frames_extracted,
            "total_eye_crops": len(manifest_samples),
            "left_crops_count": total_left_crops,
            "right_crops_count": total_right_crops,
            "total_rejections": rejections_count,
            "rejections_by_reason": rejection_reasons,
            "samples_by_category": samples_by_category,
            "samples_by_session": samples_by_session,
            "allowed_labels": ["OPEN", "CLOSED", "UNCERTAIN"],
        },
        "samples": manifest_samples,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved dataset manifest to: %s", manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Phase B1.2A Dataset Preparation")
    parser.add_argument("--recordings-dir", type=Path, default=repo_root / "data" / "recordings")
    parser.add_argument("--output-dir", type=Path, default=repo_root / "data" / "evaluation")
    parser.add_argument("--target-frames", type=int, default=140)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    manifest = process_dataset(
        recordings_dir=args.recordings_dir,
        output_dir=args.output_dir,
        target_frames_per_session=args.target_frames,
        seed=args.seed,
    )
    meta = manifest["metadata"]
    print("\n" + "=" * 60)
    print("  PHASE B1.2A DATASET EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total Video Frames Sampled: {meta['total_frames_extracted']}")
    print(f"Total Eye Crops Extracted : {meta['total_eye_crops']} (Left: {meta['left_crops_count']}, Right: {meta['right_crops_count']})")
    print(f"Crop Rejections           : {meta['total_rejections']} ({meta['rejections_by_reason']})")
    print(f"Samples by Category       : {meta['samples_by_category']}")
    print(f"Samples by Session        : {meta['samples_by_session']}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
