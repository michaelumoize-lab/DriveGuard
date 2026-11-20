"""
DriveGuard Phase B1.2A.1: Independent Eye-State Dataset Preparation Pipeline.

Extracts representative eye observations from synchronized DriveGuard recording sessions,
generates opaque, double-blind annotation templates, preserves private ground-truth pairings,
and captures exact monocular EAR reconstruction metrics.

STRICT CONSTRAINTS:
- No changes to backend/app/ production code.
- No changes to EAR thresholds, blink detection, PERCLOS, or alert systems.
- Zero provenance (session, frame, timestamp, eye side, sampling category, event ID/phase)
  exposed in the annotator-facing CSV or image file paths.
- All provenance, left/right bilateral pairings, and event groupings preserved privately in manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

repo_root = Path(__file__).resolve().parent.parent.parent
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
from backend.app.ai.landmarker import DriveGuardFaceLandmarker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("DriveGuardDatasetPrep")


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


def identify_closure_events(
    telemetry_rows: List[Dict[str, Any]],
    session_id: str,
) -> Tuple[List[List[int]], Dict[int, Tuple[str, str]]]:
    """
    Identifies contiguous segments of is_eye_closed == True and maps
    sampled frame indices to their (event_id, event_phase).

    Event phases:
      - 'peak': single-frame closure or nadir frame with lowest compensated EAR
      - 'onset': opening frame of multi-frame closure
      - 'offset': terminal frame of multi-frame closure
    """
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

    event_map: Dict[int, Tuple[str, str]] = {}
    for ev_idx, ev in enumerate(closure_events):
        ev_id = f"{session_id}_EV{ev_idx + 1:03d}"
        if len(ev) == 1:
            event_map[ev[0]] = (ev_id, "peak")
        elif len(ev) == 2:
            event_map[ev[0]] = (ev_id, "onset")
            event_map[ev[1]] = (ev_id, "offset")
        else:
            onset_idx = ev[0]
            offset_idx = ev[-1]
            lowest_ear = float("inf")
            peak_idx = ev[len(ev) // 2]
            for f_idx in ev:
                try:
                    ear_val = float(telemetry_rows[f_idx].get("ear_compensated", 1.0))
                    if ear_val < lowest_ear:
                        lowest_ear = ear_val
                        peak_idx = f_idx
                except (ValueError, TypeError):
                    pass
            event_map[onset_idx] = (ev_id, "onset")
            event_map[offset_idx] = (ev_id, "offset")
            if peak_idx != onset_idx and peak_idx != offset_idx:
                event_map[peak_idx] = (ev_id, "peak")
            else:
                mid = ev[len(ev) // 2]
                if mid != onset_idx and mid != offset_idx:
                    event_map[mid] = (ev_id, "peak")

    return closure_events, event_map


def sample_session_frames(
    telemetry_rows: List[Dict[str, Any]],
    target_sample_count: int = 140,
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

    sampled_map: Dict[int, str] = {}

    # 1. RANDOM_TIME (~30% of target)
    random_count = max(10, int(target_sample_count * 0.30))
    step = max(1, total_rows // random_count)
    for i in range(0, total_rows, step):
        idx = min(total_rows - 1, i + rng.randint(0, max(0, step - 1)))
        sampled_map[idx] = "RANDOM_TIME"

    # 2. EAR_EVENT (~30% of target)
    closure_events, _ = identify_closure_events(telemetry_rows, "tmp")
    for ev in closure_events:
        if len(ev) == 1:
            sampled_map[ev[0]] = "EAR_EVENT"
        elif len(ev) == 2:
            sampled_map[ev[0]] = "EAR_EVENT"
            sampled_map[ev[1]] = "EAR_EVENT"
        else:
            onset = ev[0]
            offset = ev[-1]
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

    sorted_samples = sorted(sampled_map.items(), key=lambda x: x[0])
    return sorted_samples


def process_dataset(
    recordings_dir: Path,
    output_dir: Path,
    target_frames_per_session: int = 140,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Extracts crops, assigns opaque double-blind IDs, and generates:
    1. Blind annotation template CSV (zero metadata, shuffled rows).
    2. Private dataset manifest JSON (full provenance, pairings, event IDs, EAR reconstructions).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    eyes_dir = output_dir / "crops" / "eyes"
    contexts_dir = output_dir / "crops" / "contexts"

    # Clean legacy directories if present
    legacy_crops = output_dir / "crops"
    if legacy_crops.exists():
        for item in legacy_crops.iterdir():
            if item.is_dir() and item.name.startswith("session_"):
                shutil.rmtree(item)

    eyes_dir.mkdir(parents=True, exist_ok=True)
    contexts_dir.mkdir(parents=True, exist_ok=True)

    session_dirs = sorted([d for d in recordings_dir.iterdir() if d.is_dir() and d.name.startswith("session_")])
    if not session_dirs:
        raise FileNotFoundError(f"No session directories found in: {recordings_dir}")

    landmarker = DriveGuardFaceLandmarker()

    raw_extracted_frames: List[Dict[str, Any]] = []
    total_frames_extracted = 0
    rejections_count = 0
    rejection_reasons: Dict[str, int] = {}
    samples_by_category: Dict[str, int] = {}
    samples_by_session: Dict[str, int] = {}
    event_phases_count: Dict[str, int] = {"onset": 0, "peak": 0, "offset": 0}
    total_closure_events_count = 0

    try:
        for s_dir in session_dirs:
            session_id = s_dir.name
            video_path = s_dir / "stream.mp4"
            csv_path = s_dir / "telemetry.csv"

            if not video_path.is_file() or not csv_path.is_file():
                logger.warning("Skipping session %s: missing video or telemetry", session_id)
                continue

            logger.info("Processing session: %s", session_id)
            with open(csv_path, "r", encoding="utf-8") as f:
                telemetry_rows = list(csv.DictReader(f))

            closure_events, event_map = identify_closure_events(telemetry_rows, session_id)
            total_closure_events_count += len(closure_events)

            cap = cv2.VideoCapture(str(video_path))
            total_v_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info("  Total frames: Video=%d, CSV=%d, Closure Events=%d", total_v_frames, len(telemetry_rows), len(closure_events))

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

                features, lms_xyz = landmarker.process_frame(
                    frame_bgr=frame_bgr,
                    timestamp_ms=timestamp_ms,
                    extract_crops=False,
                )

                if not features.face_detected or lms_xyz is None:
                    rejections_count += 1
                    rejection_reasons["NO_FACE_DETECTED"] = rejection_reasons.get("NO_FACE_DETECTED", 0) + 1
                    continue

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
                if left_crop_q is None or left_crop_q.crop_rgb is None or right_crop_q is None or right_crop_q.crop_rgb is None or context_crop is None:
                    rejections_count += 1
                    rejection_reasons["CROP_EXTRACTION_FAILED"] = rejection_reasons.get("CROP_EXTRACTION_FAILED", 0) + 1
                    continue

                # Event-level grouping
                ev_id: Optional[str] = None
                ev_phase: Optional[str] = None
                if category == "EAR_EVENT" and frame_idx in event_map:
                    ev_id, ev_phase = event_map[frame_idx]
                    event_phases_count[ev_phase] = event_phases_count.get(ev_phase, 0) + 1

                raw_extracted_frames.append({
                    "session_id": session_id,
                    "frame_id": frame_id,
                    "timestamp_ms": timestamp_ms,
                    "category": category,
                    "event_id": ev_id,
                    "event_phase": ev_phase,
                    "left_crop_rgb": left_crop_q.crop_rgb,
                    "left_crop_q": left_crop_q,
                    "right_crop_rgb": right_crop_q.crop_rgb,
                    "right_crop_q": right_crop_q,
                    "context_crop_bgr": context_crop,
                    "t_row": t_row,
                })

                total_frames_extracted += 1
                samples_by_category[category] = samples_by_category.get(category, 0) + 1
                samples_by_session[session_id] = samples_by_session.get(session_id, 0) + 1

            cap.release()

    finally:
        landmarker.close()

    # Build 808 individual eye observations
    observations: List[Dict[str, Any]] = []
    for f_data in raw_extracted_frames:
        frame_key = f"{f_data['session_id']}_F{f_data['frame_id']:05d}"
        t_row = f_data["t_row"]
        l_ear = float(t_row.get("ear_left") or 0.0)
        r_ear = float(t_row.get("ear_right") or 0.0)
        thresh = float(t_row.get("ear_closure_threshold") or 0.20)
        is_geo_valid = str(t_row.get("is_geometric_valid", "")).lower() == "true"
        is_closed_runtime = str(t_row.get("is_eye_closed", "")).lower() == "true"

        # Left Eye Observation
        observations.append({
            "temp_key": f"{frame_key}_L",
            "frame_key": frame_key,
            "session_id": f_data["session_id"],
            "frame_id": f_data["frame_id"],
            "timestamp_ms": f_data["timestamp_ms"],
            "eye_side": "LEFT",
            "category": f_data["category"],
            "event_id": f_data["event_id"],
            "event_phase": f_data["event_phase"],
            "crop_rgb": f_data["left_crop_rgb"],
            "crop_q": f_data["left_crop_q"],
            "context_crop_bgr": f_data["context_crop_bgr"],
            "raw_ear": l_ear,
            "other_ear": r_ear,
            "ear_compensated": float(t_row.get("ear_compensated") or 0.0),
            "ear_closure_threshold": thresh,
            "is_geometric_valid": is_geo_valid,
            "is_eye_closed_recorded_runtime": is_closed_runtime,
            "reconstructed_monocular_closed": bool(is_geo_valid and (l_ear < thresh)),
            "reconstructed_binocular_closed": bool(is_geo_valid and (l_ear < thresh and r_ear < thresh)),
            "pose_zone": t_row.get("pose_zone", "UNKNOWN"),
            "pitch_deg": float(t_row.get("pitch_deg") or 0.0),
            "yaw_deg": float(t_row.get("yaw_deg") or 0.0),
            "roll_deg": float(t_row.get("roll_deg") or 0.0),
            "luminance": float(t_row.get("luminance") or 0.0),
            "blur_variance": float(t_row.get("blur_variance") or 0.0),
            "optical_status": t_row.get("optical_status", "UNKNOWN"),
        })

        # Right Eye Observation
        observations.append({
            "temp_key": f"{frame_key}_R",
            "frame_key": frame_key,
            "session_id": f_data["session_id"],
            "frame_id": f_data["frame_id"],
            "timestamp_ms": f_data["timestamp_ms"],
            "eye_side": "RIGHT",
            "category": f_data["category"],
            "event_id": f_data["event_id"],
            "event_phase": f_data["event_phase"],
            "crop_rgb": f_data["right_crop_rgb"],
            "crop_q": f_data["right_crop_q"],
            "context_crop_bgr": f_data["context_crop_bgr"],
            "raw_ear": r_ear,
            "other_ear": l_ear,
            "ear_compensated": float(t_row.get("ear_compensated") or 0.0),
            "ear_closure_threshold": thresh,
            "is_geometric_valid": is_geo_valid,
            "is_eye_closed_recorded_runtime": is_closed_runtime,
            "reconstructed_monocular_closed": bool(is_geo_valid and (r_ear < thresh)),
            "reconstructed_binocular_closed": bool(is_geo_valid and (l_ear < thresh and r_ear < thresh)),
            "pose_zone": t_row.get("pose_zone", "UNKNOWN"),
            "pitch_deg": float(t_row.get("pitch_deg") or 0.0),
            "yaw_deg": float(t_row.get("yaw_deg") or 0.0),
            "roll_deg": float(t_row.get("roll_deg") or 0.0),
            "luminance": float(t_row.get("luminance") or 0.0),
            "blur_variance": float(t_row.get("blur_variance") or 0.0),
            "optical_status": t_row.get("optical_status", "UNKNOWN"),
        })

    # Deterministic opaque shuffle with seed
    rng_shuffle = random.Random(seed)
    rng_shuffle.shuffle(observations)

    # Assign opaque annotation IDs and build pairing index
    frame_to_ann: Dict[str, Dict[str, str]] = {}
    for i, obs in enumerate(observations):
        ann_id = f"ANN_{i + 1:04d}"
        obs["annotation_id"] = ann_id
        obs["crop_rel_path"] = f"crops/eyes/{ann_id}.png"
        obs["context_rel_path"] = f"crops/contexts/{ann_id}_context.png"
        frame_to_ann.setdefault(obs["frame_key"], {})[obs["eye_side"]] = ann_id

    # Populate pair_annotation_id
    for obs in observations:
        side = obs["eye_side"]
        other_side = "RIGHT" if side == "LEFT" else "LEFT"
        obs["pair_annotation_id"] = frame_to_ann[obs["frame_key"]][other_side]

    # Write images to disk
    logger.info("Writing %d eye crops and context previews to disk...", len(observations))
    for obs in observations:
        crop_disk_path = output_dir / obs["crop_rel_path"]
        context_disk_path = output_dir / obs["context_rel_path"]

        cv2.imwrite(str(crop_disk_path), cv2.cvtColor(obs["crop_rgb"], cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(context_disk_path), obs["context_crop_bgr"])

    # 1. Write Annotator-Facing CSV (STRICTLY BLIND: Opaque IDs, zero provenance)
    csv_path = output_dir / "annotation_template.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "annotation_id",
            "crop_path",
            "context_path",
            "human_label",
            "annotation_notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for obs in observations:
            writer.writerow({
                "annotation_id": obs["annotation_id"],
                "crop_path": obs["crop_rel_path"],
                "context_path": obs["context_rel_path"],
                "human_label": "",
                "annotation_notes": "",
            })
    logger.info("Saved blind annotation template (%d rows) to: %s", len(observations), csv_path)

    # 2. Write Private Dataset Manifest JSON
    manifest_annotations: Dict[str, Any] = {}
    for obs in observations:
        manifest_annotations[obs["annotation_id"]] = {
            "annotation_id": obs["annotation_id"],
            "crop_path": obs["crop_rel_path"],
            "context_path": obs["context_rel_path"],
            "provenance": {
                "session_id": obs["session_id"],
                "frame_id": obs["frame_id"],
                "timestamp_ms": obs["timestamp_ms"],
                "eye_side": obs["eye_side"],
                "pair_annotation_id": obs["pair_annotation_id"],
                "sampling_category": obs["category"],
                "event_id": obs["event_id"],
                "event_phase": obs["event_phase"],
            },
            "crop_quality": {
                "sharpness_score": round(obs["crop_q"].sharpness_score, 4),
                "exposure_score": round(obs["crop_q"].exposure_score, 4),
                "glare_fraction": round(obs["crop_q"].glare_fraction, 4),
                "overall_reliability": round(obs["crop_q"].overall_reliability, 4),
            },
            "telemetry_ground_state": {
                "raw_ear": round(obs["raw_ear"], 4),
                "ear_compensated": round(obs["ear_compensated"], 4),
                "ear_closure_threshold": round(obs["ear_closure_threshold"], 4),
                "is_eye_closed_recorded_runtime": obs["is_eye_closed_recorded_runtime"],
                "is_geometric_valid": obs["is_geometric_valid"],
                "reconstructed_monocular_closed": obs["reconstructed_monocular_closed"],
                "reconstructed_binocular_closed": obs["reconstructed_binocular_closed"],
                "pose_zone": obs["pose_zone"],
                "pitch_deg": round(obs["pitch_deg"], 2),
                "yaw_deg": round(obs["yaw_deg"], 2),
                "roll_deg": round(obs["roll_deg"], 2),
                "luminance": round(obs["luminance"], 2),
                "blur_variance": round(obs["blur_variance"], 2),
                "optical_status": obs["optical_status"],
            },
        }

    manifest = {
        "metadata": {
            "title": "DriveGuard Phase B1.2A Independent Eye-State Evaluation Dataset",
            "random_seed": seed,
            "target_frames_per_session": target_frames_per_session,
            "total_frames_extracted": total_frames_extracted,
            "total_annotations": len(manifest_annotations),
            "left_crops_count": total_frames_extracted,
            "right_crops_count": total_frames_extracted,
            "total_rejections": rejections_count,
            "rejections_by_reason": rejection_reasons,
            "samples_by_category": samples_by_category,
            "samples_by_session": samples_by_session,
            "event_grouping_stats": {
                "total_closure_events": total_closure_events_count,
                "total_ear_event_frames": samples_by_category.get("EAR_EVENT", 0),
                "phases": event_phases_count,
            },
            "allowed_labels": ["OPEN", "CLOSED", "UNCERTAIN"],
            "operational_definitions": {
                "OPEN": "less than 20% eyelid closure (<20% closure, palpebral fissure >= 80% open)",
                "CLOSED": "at least 80% eyelid closure (>=80% closure, palpebral fissure <= 20% open)",
                "UNCERTAIN": "20% to less than 80% closure, or insufficient visual evidence (blur, glare, occlusion, ambiguity)",
            },
        },
        "annotations": manifest_annotations,
    }

    manifest_path = output_dir / "dataset_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Saved private dataset manifest to: %s", manifest_path)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Phase B1.2A.1 Dataset Preparation")
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
    print("\n" + "=" * 65)
    print("  PHASE B1.2A.1 DATASET PREPARATION SUMMARY")
    print("=" * 65)
    print(f"Total Video Frames Sampled: {meta['total_frames_extracted']}")
    print(f"Total Eye Annotations     : {meta['total_annotations']} (Left: {meta['left_crops_count']}, Right: {meta['right_crops_count']})")
    print(f"Crop Rejections           : {meta['total_rejections']} ({meta['rejections_by_reason']})")
    print(f"Samples by Category       : {meta['samples_by_category']}")
    print(f"Samples by Session        : {meta['samples_by_session']}")
    print(f"Event Grouping Stats      : {meta['event_grouping_stats']}")
    print(f"Operational Definitions   : {meta['operational_definitions']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
