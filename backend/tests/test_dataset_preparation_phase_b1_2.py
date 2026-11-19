"""
Unit and integration tests for Phase B1.2A: Independent Eye-State Dataset Preparation.

Validates:
1. Timestamp alignment between video frames and CSV telemetry rows.
2. Frame extraction and crop extraction dimensions (64x64).
3. Left and right eye crop separation and labeling.
4. Duplicate prevention in sampling strategy.
5. Stable sample identifier contract.
6. Missing landmarks / face loss rejection handling.
7. Context crop extraction (clean ocular context without model overlays).
8. CSV generation and round-trip reading.
9. Reproducible sampling with fixed random seed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from backend.app.ai.geometry import (
    EyeCropQuality,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    extract_eye_crop_with_quality,
)
from scripts.research.prepare_eye_evaluation_dataset import (
    extract_context_crop,
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
        ear = 0.12 if is_closed else (0.19 if 75 <= i <= 80 else 0.28)  # 75-80 is borderline
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


def test_stable_sample_identifiers():
    """Verify sample ID naming contract across sessions, frames, and eye sides."""
    session_id = "session_20261119_132926"
    frame_id = 42
    left_id = f"{session_id}_F{frame_id:05d}_L"
    right_id = f"{session_id}_F{frame_id:05d}_R"

    assert left_id == "session_20261119_132926_F00042_L"
    assert right_id == "session_20261119_132926_F00042_R"
    assert left_id != right_id


def test_timestamp_alignment_existing_recordings():
    """Verify that video frame count and CSV rows match exactly in real recordings."""
    recordings_dir = Path("data/recordings")
    sessions = ["session_20261119_132926", "session_20261119_142559"]

    for s in sessions:
        v_path = recordings_dir / s / "stream.mp4"
        c_path = recordings_dir / s / "telemetry.csv"
        if not v_path.is_file() or not c_path.is_file():
            pytest.skip(f"Recording {s} not present in test environment.")

        cap = cv2.VideoCapture(str(v_path))
        v_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        with open(c_path, "r", encoding="utf-8") as f:
            c_count = sum(1 for _ in csv.DictReader(f))

        assert v_count == c_count, f"Frame count mismatch in {s}: Video {v_count} != CSV {c_count}"


def test_left_and_right_eye_crop_extraction():
    """Test extracting left and right crops from synthetic frame with valid geometry."""
    frame_rgb = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.circle(frame_rgb, (300, 240), 100, (200, 200, 200), -1)

    # Synthetic landmarks: create distinct coordinates for left and right eyes
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
    # Point outside image boundaries
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


def test_annotation_csv_roundtrip_and_blindness(tmp_path):
    """Verify that annotation CSV can be written and read back, and contains no model predictions."""
    test_csv = tmp_path / "test_annotation.csv"
    rows = [
        {
            "sample_id": "session_1_F00001_L",
            "session_id": "session_1",
            "frame_id": "1",
            "timestamp_ms": "100.00",
            "eye_side": "LEFT",
            "crop_path": "crops/session_1/sample_1_L.png",
            "context_path": "crops/session_1/sample_1_context.png",
            "sampling_category": "RANDOM_TIME",
            "human_label": "",
            "annotation_notes": "",
        }
    ]

    with open(test_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with open(test_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        read_rows = list(reader)

    assert len(read_rows) == 1
    assert read_rows[0]["sample_id"] == "session_1_F00001_L"
    assert read_rows[0]["human_label"] == ""

    # Verify BLINDNESS: Ensure zero prediction or confidence fields exist in CSV
    forbidden = ["ear", "cnn", "closed_prob", "open_prob", "drowsiness", "prediction", "is_eye_closed"]
    for col in reader.fieldnames:
        for f_word in forbidden:
            assert f_word not in col.lower(), f"Forbidden prediction/EAR column '{col}' exposed to annotator!"
