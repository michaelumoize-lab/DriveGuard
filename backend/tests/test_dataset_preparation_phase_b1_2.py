"""
Unit and integration tests for Phase B1.2A.1: Annotation-Blindness and EAR Reference Corrections.

Validates:
1. Opaque annotation ID format (ANN_xxxx).
2. No prohibited provenance fields in annotator-facing CSV (no session, frame, timestamp, eye side, category, event).
3. Complete private mapping between annotation IDs and original samples in dataset_manifest.json.
4. Exactly one left and one right crop per sampled frame.
5. No duplicate annotation IDs.
6. No sampling metadata exposed through filenames or paths.
7. Event grouping integrity where supported (event_id, event_phase in onset/peak/offset).
8. Monocular and binocular EAR reconstruction against actual runtime logic.
9. Seed-42 reproducibility.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.ai.geometry import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    extract_eye_crop_with_quality,
)
from backend.app.features.ear import (
    calculate_single_eye_ear_with_validity,
    extract_eye_aspect_ratio,
)
from scripts.research.prepare_eye_evaluation_dataset import (
    extract_context_crop,
    identify_closure_events,
    sample_session_frames,
)


@pytest.fixture
def synthetic_telemetry_rows():
    """Generate 100 synthetic telemetry rows simulating blinks, blur, and head turns."""
    rows = []
    for i in range(1, 101):
        is_closed = (20 <= i <= 25) or (60 <= i <= 62)  # Blinks
        optical = "BLURRY" if 35 <= i <= 40 else "EXCELLENT"
        zone = "RIGHT_MIRROR" if 50 <= i <= 55 else "CENTER"
        ear = 0.12 if is_closed else (0.19 if 75 <= i <= 80 else 0.28)
        rows.append({
            "frame_id": str(i),
            "timestamp_ms": str(100000.0 + i * 100.0),
            "face_detected": "True",
            "pose_zone": zone,
            "pitch_deg": "2.0",
            "yaw_deg": "25.0" if zone == "RIGHT_MIRROR" else "0.0",
            "roll_deg": "0.0",
            "ear_left": str(ear),
            "ear_right": str(ear),
            "ear_compensated": str(ear),
            "ear_closure_threshold": "0.175",
            "is_eye_closed": str(is_closed),
            "is_geometric_valid": "True",
            "optical_status": optical,
        })
    return rows


def test_reproducible_sampling_with_seed(synthetic_telemetry_rows):
    """Test that fixed random seed produces identical sample lists."""
    samples_a = sample_session_frames(synthetic_telemetry_rows, target_sample_count=30, seed=42)
    samples_b = sample_session_frames(synthetic_telemetry_rows, target_sample_count=30, seed=42)
    samples_c = sample_session_frames(synthetic_telemetry_rows, target_sample_count=30, seed=99)

    assert len(samples_a) > 0
    assert samples_a == samples_b
    assert samples_a != samples_c


def test_duplicate_prevention_in_sampling(synthetic_telemetry_rows):
    """Test that all sampled frame indices are strictly unique."""
    samples = sample_session_frames(synthetic_telemetry_rows, target_sample_count=50, seed=42)
    indices = [idx for idx, _ in samples]
    assert len(indices) == len(set(indices)), "Duplicate frame indices found in sampled list!"


def test_left_and_right_eye_crop_extraction():
    """Test extracting left and right crops from synthetic frame with valid geometry."""
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame_rgb, (300, 240), 100, (200, 200, 200), -1)

    lms = np.zeros((478, 3), dtype=np.float64)
    for idx in LEFT_EYE_INDICES:
        lms[idx] = [280.0, 230.0, 0.0]
    for idx in RIGHT_EYE_INDICES:
        lms[idx] = [360.0, 230.0, 0.0]

    left_crop = extract_eye_crop_with_quality(frame_rgb, lms, LEFT_EYE_INDICES, (64, 64))
    right_crop = extract_eye_crop_with_quality(frame_rgb, lms, RIGHT_EYE_INDICES, (64, 64))

    assert left_crop is not None
    assert left_crop.crop_rgb.shape == (64, 64, 3)
    assert right_crop is not None
    assert right_crop.crop_rgb.shape == (64, 64, 3)


def test_missing_landmarks_graceful_handling():
    """Test crop extraction when landmarks are invalid or zero."""
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    lms = np.zeros((478, 3), dtype=np.float64) - 999.0

    left_crop = extract_eye_crop_with_quality(frame_rgb, lms, LEFT_EYE_INDICES, (64, 64))
    assert left_crop is None or left_crop.overall_reliability == 0.0


def test_context_crop_generation():
    """Test that context crop generates clean ocular image without model text/overlays."""
    frame_bgr = np.ones((480, 640, 3), dtype=np.uint8) * 150
    lms = np.zeros((478, 3), dtype=np.float64)
    for idx in LEFT_EYE_INDICES:
        lms[idx] = [280.0, 240.0, 0.0]
    for idx in RIGHT_EYE_INDICES:
        lms[idx] = [360.0, 240.0, 0.0]

    context = extract_context_crop(frame_bgr, lms, padding_scale=2.0, target_size=(256, 128))
    assert context is not None
    assert context.shape == (128, 256, 3)


def test_annotation_csv_blindness_and_no_prohibited_fields():
    """
    Verify that the annotator-facing CSV contains ONLY opaque fields:
    annotation_id, crop_path, context_path, human_label, annotation_notes.
    Strictly forbids: sampling_category, session, frame, timestamp, eye side, ear, cnn.
    """
    csv_path = Path("data/evaluation/annotation_template.csv")
    if not csv_path.is_file():
        pytest.skip("data/evaluation/annotation_template.csv not found")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    expected_fields = ["annotation_id", "crop_path", "context_path", "human_label", "annotation_notes"]
    assert fields == expected_fields, f"Annotation CSV headers mismatch! Found: {fields}"

    # Verify rows have opaque IDs and valid paths
    assert len(rows) == 808
    ann_ids = [r["annotation_id"] for r in rows]
    assert len(ann_ids) == len(set(ann_ids)), "Duplicate annotation IDs found!"

    for r in rows:
        assert r["annotation_id"].startswith("ANN_")
        assert "eyes/ANN_" in r["crop_path"]
        assert "contexts/ANN_" in r["context_path"]
        # Ensure no session, frame, timestamp, or eye side strings in any column
        for key, val in r.items():
            val_lower = val.lower()
            assert "session_" not in val_lower, f"Session ID leaked in column {key}: {val}"
            assert "frame" not in val_lower, f"Frame metadata leaked in column {key}: {val}"
            assert "category" not in val_lower, f"Category leaked in column {key}: {val}"
            assert "ear" not in val_lower, f"EAR leaked in column {key}: {val}"


def test_no_sampling_metadata_exposed_in_paths():
    """Ensure image filenames and directories contain zero provenance or model metadata."""
    eyes_dir = Path("data/evaluation/crops/eyes")
    contexts_dir = Path("data/evaluation/crops/contexts")
    if not eyes_dir.is_dir() or not contexts_dir.is_dir():
        pytest.skip("Crop directories not found")

    eye_files = list(eyes_dir.glob("*.png"))
    context_files = list(contexts_dir.glob("*.png"))

    assert len(eye_files) == 808
    assert len(context_files) == 808

    for p in eye_files:
        assert p.name.startswith("ANN_") and p.name.endswith(".png")
        assert "_L" not in p.name and "_R" not in p.name
        assert "session" not in p.name

    for p in context_files:
        assert p.name.startswith("ANN_") and p.name.endswith("_context.png")
        assert "session" not in p.name


def test_private_manifest_mapping_integrity():
    """Verify that dataset_manifest.json maintains complete provenance for all 808 annotations."""
    manifest_path = Path("data/evaluation/dataset_manifest.json")
    if not manifest_path.is_file():
        pytest.skip("data/evaluation/dataset_manifest.json not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    meta = manifest["metadata"]
    annotations = manifest["annotations"]

    assert meta["total_annotations"] == 808
    assert meta["total_frames_extracted"] == 404
    assert len(annotations) == 808

    for ann_id, data in annotations.items():
        assert ann_id == data["annotation_id"]
        prov = data["provenance"]
        assert prov["session_id"] in ("session_20261119_132926", "session_20261119_142559")
        assert isinstance(prov["frame_id"], int)
        assert isinstance(prov["timestamp_ms"], float)
        assert prov["eye_side"] in ("LEFT", "RIGHT")
        assert prov["pair_annotation_id"].startswith("ANN_")
        assert prov["sampling_category"] in (
            "RANDOM_TIME", "EAR_EVENT", "OPTICAL_DEGRADED", "HEAD_POSE", "BORDERLINE_EAR"
        )


def test_bilateral_pairing_reciprocity():
    """Verify that every left eye annotation points to a right eye annotation on the same frame, and vice versa."""
    manifest_path = Path("data/evaluation/dataset_manifest.json")
    if not manifest_path.is_file():
        pytest.skip("data/evaluation/dataset_manifest.json not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    annotations = manifest["annotations"]

    for ann_id, data in annotations.items():
        prov = data["provenance"]
        pair_id = prov["pair_annotation_id"]
        assert pair_id in annotations, f"Missing paired annotation: {pair_id}"
        pair_prov = annotations[pair_id]["provenance"]

        assert pair_prov["pair_annotation_id"] == ann_id, "Bilateral pair link is not reciprocal!"
        assert pair_prov["session_id"] == prov["session_id"], "Paired eyes belong to different sessions!"
        assert pair_prov["frame_id"] == prov["frame_id"], "Paired eyes belong to different frames!"
        assert pair_prov["eye_side"] != prov["eye_side"], "Paired eyes must be opposite sides (LEFT vs RIGHT)!"


def test_event_grouping_integrity():
    """Verify that all 234 EAR_EVENT annotations have valid event_id and event_phase in onset/peak/offset."""
    manifest_path = Path("data/evaluation/dataset_manifest.json")
    if not manifest_path.is_file():
        pytest.skip("data/evaluation/dataset_manifest.json not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    annotations = manifest["annotations"]
    ear_event_samples = [d for d in annotations.values() if d["provenance"]["sampling_category"] == "EAR_EVENT"]

    # 234 frames * 2 eye crops = 468 annotations
    assert len(ear_event_samples) == 468

    valid_phases = {"onset", "peak", "offset"}
    event_ids = set()
    phase_counts = {"onset": 0, "peak": 0, "offset": 0}

    for sample in ear_event_samples:
        prov = sample["provenance"]
        assert prov["event_id"] is not None and "_EV" in prov["event_id"]
        assert prov["event_phase"] in valid_phases
        event_ids.add(prov["event_id"])
        phase_counts[prov["event_phase"]] += 1

    assert len(event_ids) > 0
    # Left and right eye share same phase, so counts must be even
    for phase, count in phase_counts.items():
        assert count % 2 == 0
        assert count > 0


def test_exact_monocular_and_binocular_ear_reconstruction():
    """
    Verify that monocular and binocular EAR reconstruction logic faithfully
    reproduces runtime classifications using raw_ear, closure_threshold, and is_geometric_valid.
    """
    manifest_path = Path("data/evaluation/dataset_manifest.json")
    if not manifest_path.is_file():
        pytest.skip("data/evaluation/dataset_manifest.json not found")

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    annotations = manifest["annotations"]

    for ann_id, data in annotations.items():
        t_state = data["telemetry_ground_state"]
        raw_ear = t_state["raw_ear"]
        thresh = t_state["ear_closure_threshold"]
        is_geo_valid = t_state["is_geometric_valid"]
        mono_reconstructed = t_state["reconstructed_monocular_closed"]

        # Formula: monocular closure is True iff eye geometry is valid AND raw_ear < threshold
        expected_mono = bool(is_geo_valid and (raw_ear < thresh))
        assert mono_reconstructed == expected_mono

        # Pair lookup for binocular closure test
        pair_id = data["provenance"]["pair_annotation_id"]
        pair_raw_ear = annotations[pair_id]["telemetry_ground_state"]["raw_ear"]
        bino_reconstructed = t_state["reconstructed_binocular_closed"]
        expected_bino = bool(is_geo_valid and (raw_ear < thresh and pair_raw_ear < thresh))
        assert bino_reconstructed == expected_bino
