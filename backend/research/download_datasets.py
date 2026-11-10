"""
DriveGuard — Dataset Ingestion, Download & Integrity Verification Guide
======================================================================
Final-Year Computer Engineering Project:
Personalised Adaptive AI Driver Drowsiness Detection and Early Warning System
Using Computer Vision.

Author: Michael Umoize

Manages ingestion and verification for the two benchmark datasets used in DriveGuard:
  1. MRL Eye Dataset (84,898 eye crops across 37 subjects; glasses, reflections, lighting)
     - Used for: Training & verifying MobileNetV3-Small neural eye closure verifier.
  2. UTA-RLDD (University of Texas at Arlington Real-Life Drowsiness Dataset)
     - 60 participants × 3 vigilance classes (0: Alert, 5: Low Vigilance, 10: Drowsy)
     - Used for: Cross-participant evaluation of 4-Zone PDBP vs Fixed EAR.

Usage:
  # Verify local MRL Eye dataset directory:
  python -m backend.research.download_datasets --dataset mrl --path "C:/datasets/mrlEyes_2018_01"

  # Verify local UTA-RLDD dataset directory:
  python -m backend.research.download_datasets --dataset utarldd --path "C:/datasets/UTA-RLDD"

  # Generate synthetic sample dataset for offline pipeline smoke-testing:
  python -m backend.research.download_datasets --generate-synthetic-sample
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


MRL_DATASET_SPECS = {
    "name": "MRL Eye Dataset",
    "total_subjects": 37,
    "expected_files_approx": 84898,
    "kaggle_slug": "dheerajperumandla/drowsiness-dataset",
    "official_url": "http://mrl.cs.vsb.cz/eyedataset",
    "filename_format": "s<subject>_<index>_<gender>_<glasses>_<eyeState>_<reflections>_<lighting>_<sensor>.png",
}

UTARLDD_DATASET_SPECS = {
    "name": "UTA Real-Life Drowsiness Dataset (UTA-RLDD)",
    "total_participants": 60,
    "vigilance_classes": {
        "0": "Alert (Baseline driving, no sleep deprivation)",
        "5": "Low Vigilance (Subtle drowsiness, slow eye closures)",
        "10": "Drowsy (Extreme sleep deprivation, microsleeps)",
    },
    "folds": ["Fold1_part1", "Fold1_part2", "Fold2_part1", "Fold2_part2", "Fold3_part1", "Fold3_part2"],
    "official_url": "https://www.uta.edu/faculty/herrera/driving-dataset.html",
}


def verify_mrl_dataset(data_path: Path) -> Tuple[bool, Dict[str, int]]:
    """Scan and verify the integrity of an extracted MRL Eye dataset directory."""
    print(f"\n[Verifying MRL Eye Dataset] Directory: {data_path.resolve()}")
    if not data_path.is_dir():
        print(f"❌ Error: Path does not exist: {data_path}")
        return False, {}

    png_files = list(data_path.rglob("*.png"))
    total_found = len(png_files)
    print(f"Found {total_found:,} image files.")

    subjects: set[str] = set()
    open_count = 0
    closed_count = 0
    glasses_count = 0
    reflections_count = 0

    for p in png_files:
        name_parts = p.stem.split("_")
        # Official format has at least 8 parts (s0001_00001_0_0_0_0_0_01)
        if len(name_parts) >= 8 and name_parts[0].startswith("s"):
            subjects.add(name_parts[0])
            eye_state = name_parts[4]
            if eye_state == "0":
                open_count += 1
            elif eye_state == "1":
                closed_count += 1

            if name_parts[3] == "1":
                glasses_count += 1
            if name_parts[5] in ("1", "2"):
                reflections_count += 1
        else:
            # Fallback Kaggle layout (Open_Eyes/Closed_Eyes folders)
            if "open" in p.parent.name.lower():
                open_count += 1
            elif "closed" in p.parent.name.lower():
                closed_count += 1

    stats = {
        "total_images": total_found,
        "subjects_identified": len(subjects),
        "open_eyes": open_count,
        "closed_eyes": closed_count,
        "with_glasses": glasses_count,
        "with_glare_reflections": reflections_count,
    }

    print("\n--- MRL Dataset Verification Summary ---")
    for k, v in stats.items():
        print(f"  {k:<26}: {v:,}")

    is_valid = total_found > 1000
    if is_valid:
        print("[PASS] MRL Dataset verification PASSED.")
    else:
        print("[WARNING] MRL image count appears incomplete.")

    return is_valid, stats


def verify_utarldd_dataset(data_path: Path) -> Tuple[bool, Dict[str, int]]:
    """Scan and verify UTA-RLDD participant directories and class recordings."""
    print(f"\n[Verifying UTA-RLDD Dataset] Directory: {data_path.resolve()}")
    if not data_path.is_dir():
        print(f"[ERROR] Path does not exist: {data_path}")
        return False, {}

    video_exts = {".mov", ".mp4", ".avi", ".mkv"}
    participant_folders: set[str] = set()
    class_counts = {"0": 0, "5": 0, "10": 0}

    for p in data_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in video_exts:
            parent_name = p.parent.name
            stem = p.stem.split(".")[0]
            if stem in class_counts:
                class_counts[stem] += 1
                participant_folders.add(parent_name)

    stats = {
        "participants_found": len(participant_folders),
        "alert_videos_class_0": class_counts["0"],
        "low_vigilance_videos_class_5": class_counts["5"],
        "drowsy_videos_class_10": class_counts["10"],
    }

    print("\n--- UTA-RLDD Dataset Verification Summary ---")
    for k, v in stats.items():
        print(f"  {k:<28}: {v}")

    is_valid = len(participant_folders) > 0
    if is_valid:
        print("[PASS] UTA-RLDD Dataset verification PASSED.")
    else:
        print("[WARNING] No participant video folders matched (expected folders containing 0.*, 5.*, 10.*).")

    return is_valid, stats


def generate_synthetic_samples(target_dir: Path) -> None:
    """Create a minimal synthetic sample dataset structure for offline testing."""
    import numpy as np
    import cv2

    print(f"\n[Generating Synthetic Sample Datasets] Target: {target_dir.resolve()}")
    mrl_dir = target_dir / "mrl_sample"
    mrl_dir.mkdir(parents=True, exist_ok=True)

    # Generate 4 synthetic participants, 20 open and 20 closed crops each
    for s_idx in range(1, 5):
        s_id = f"s000{s_idx}"
        for i in range(1, 11):
            # Open eye crop
            fn_open = f"{s_id}_{i:05d}_0_0_0_0_0_01.png"
            img_open = np.full((64, 64, 3), 150, dtype=np.uint8)
            cv2.circle(img_open, (32, 32), 12, (30, 30, 30), -1)
            cv2.circle(img_open, (32, 32), 4, (240, 240, 240), -1)
            cv2.imwrite(str(mrl_dir / fn_open), img_open)

            # Closed eye crop
            fn_closed = f"{s_id}_{i+50:05d}_0_0_1_0_0_01.png"
            img_closed = np.full((64, 64, 3), 150, dtype=np.uint8)
            cv2.line(img_closed, (12, 32), (52, 32), (40, 40, 40), 2)
            cv2.imwrite(str(mrl_dir / fn_closed), img_closed)

    print(f"[SUCCESS] Generated 80 synthetic MRL eye crops in: {mrl_dir}")


def print_download_instructions() -> None:
    """Print complete instructions for acquiring official research benchmark datasets."""
    print("=" * 78)
    print("   [DriveGuard] Benchmark Datasets Acquisition Guide")
    print("=" * 78)
    print("""
1. MRL Eye Dataset (Used for MobileNetV3-Small Eye Verification):
   -----------------------------------------------------------------
   Option A (Kaggle CLI - Fastest):
     $ pip install kaggle
     $ kaggle datasets download -d dheerajperumandla/drowsiness-dataset
     $ unzip drowsiness-dataset.zip -d backend/datasets/mrl/

   Option B (Official Academic Mirror):
     Download from: http://mrl.cs.vsb.cz/eyedataset
     Extract archives to: backend/datasets/mrl/

2. UTA-RLDD Dataset (Used for Cross-Participant PDBP vs. Fixed EAR):
   -----------------------------------------------------------------
   Request access or download from official University of Texas Arlington repository:
     URL: https://www.uta.edu/faculty/herrera/driving-dataset.html
     Mirror: https://github.com/rezaghoddoosian/UTA-RLDD
   Extract Fold1_part1, Fold1_part2, etc. to: backend/datasets/utarldd/

3. Verification:
   Once downloaded, run:
     $ python -m backend.research.download_datasets --dataset mrl --path backend/datasets/mrl
     $ python -m backend.research.download_datasets --dataset utarldd --path backend/datasets/utarldd
""")


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Dataset Ingestion & Verification")
    parser.add_argument("--dataset", choices=["mrl", "utarldd"], help="Which dataset to verify")
    parser.add_argument("--path", type=str, help="Directory path of the downloaded dataset")
    parser.add_argument("--generate-synthetic-sample", action="store_true", help="Generate minimal synthetic test dataset")
    parser.add_argument("--guide", action="store_true", help="Print download instructions")
    args = parser.parse_args()

    if args.guide or (not args.dataset and not args.generate_synthetic_sample):
        print_download_instructions()
        if not args.generate_synthetic_sample:
            return

    if args.generate_synthetic_sample:
        target = Path("backend/datasets")
        generate_synthetic_samples(target)

    if args.dataset == "mrl" and args.path:
        verify_mrl_dataset(Path(args.path))
    elif args.dataset == "utarldd" and args.path:
        verify_utarldd_dataset(Path(args.path))


if __name__ == "__main__":
    main()
