"""
DriveGuard Cross-Language Direct Parity Generator.
Generates canonical facial landmark fixtures and runs the authoritative
Python detection algorithms to generate reference ground truth.
Outputs JSON fixtures to `scripts/validation/parity_fixtures.json`.
"""

import json
import math
import os
import sys

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.app.features.ear import extract_eye_aspect_ratio
from backend.app.features.mar import calculate_mar_with_validity
from backend.app.features.head_pose import estimate_head_pose, classify_pose_zone, PoseZone
from backend.app.vision.face_landmarker import NormalizedLandmark


def create_base_landmarks():
    """Create a 478-landmark array initialized at (0.5, 0.5, 0.0)."""
    return [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(478)]


def build_fixture_set():
    fixtures = {}

    # 1. Neutral Frontal Face
    neutral = create_base_landmarks()
    
    # Rigids for head pose
    neutral[1] = {"x": 0.50, "y": 0.55, "z": -0.05}  # nose tip
    neutral[6] = {"x": 0.50, "y": 0.45, "z": 0.02}   # sellion
    neutral[10] = {"x": 0.50, "y": 0.35, "z": 0.03}  # glabella
    neutral[152] = {"x": 0.50, "y": 0.70, "z": 0.01} # chin
    neutral[33] = {"x": 0.35, "y": 0.45, "z": 0.02}  # right eye outer
    neutral[263] = {"x": 0.65, "y": 0.45, "z": 0.02} # left eye outer
    neutral[61] = {"x": 0.42, "y": 0.62, "z": 0.02}  # right mouth corner
    neutral[291] = {"x": 0.58, "y": 0.62, "z": 0.02} # left mouth corner

    # Set right eye contour (ear ~ 0.285)
    neutral[133] = {"x": 0.45, "y": 0.45, "z": 0.02}
    neutral[160] = {"x": 0.38, "y": 0.435, "z": 0.02}
    neutral[158] = {"x": 0.42, "y": 0.4365, "z": 0.02}
    neutral[144] = {"x": 0.38, "y": 0.465, "z": 0.02}
    neutral[153] = {"x": 0.42, "y": 0.4635, "z": 0.02}

    # Set left eye contour (ear ~ 0.285)
    neutral[362] = {"x": 0.55, "y": 0.45, "z": 0.02}
    neutral[385] = {"x": 0.58, "y": 0.435, "z": 0.02}
    neutral[387] = {"x": 0.62, "y": 0.4365, "z": 0.02}
    neutral[380] = {"x": 0.58, "y": 0.465, "z": 0.02}
    neutral[373] = {"x": 0.62, "y": 0.4635, "z": 0.02}

    # Mouth closed (MAR ~ 0.15)
    neutral[13] = {"x": 0.50, "y": 0.608, "z": 0.02}
    neutral[14] = {"x": 0.50, "y": 0.632, "z": 0.02}
    neutral[81] = {"x": 0.46, "y": 0.608, "z": 0.02}
    neutral[178] = {"x": 0.46, "y": 0.632, "z": 0.02}
    neutral[311] = {"x": 0.54, "y": 0.608, "z": 0.02}
    neutral[402] = {"x": 0.54, "y": 0.632, "z": 0.02}

    fixtures["neutral"] = neutral

    # 2. Blink Onset (narrowed eyes, EAR ~ 0.21)
    blink_onset = [dict(p) for p in neutral]
    blink_onset[160] = {"x": 0.38, "y": 0.439, "z": 0.02}
    blink_onset[158] = {"x": 0.42, "y": 0.440, "z": 0.02}
    blink_onset[144] = {"x": 0.38, "y": 0.461, "z": 0.02}
    blink_onset[153] = {"x": 0.42, "y": 0.460, "z": 0.02}
    blink_onset[385] = {"x": 0.58, "y": 0.439, "z": 0.02}
    blink_onset[387] = {"x": 0.62, "y": 0.440, "z": 0.02}
    blink_onset[380] = {"x": 0.58, "y": 0.461, "z": 0.02}
    blink_onset[373] = {"x": 0.62, "y": 0.460, "z": 0.02}
    fixtures["blink_onset"] = blink_onset

    # 3. Fully Closed Eyes (microsleep, EAR ~ 0.10)
    closed = [dict(p) for p in neutral]
    closed[160] = {"x": 0.38, "y": 0.445, "z": 0.02}
    closed[158] = {"x": 0.42, "y": 0.445, "z": 0.02}
    closed[144] = {"x": 0.38, "y": 0.455, "z": 0.02}
    closed[153] = {"x": 0.42, "y": 0.455, "z": 0.02}
    closed[385] = {"x": 0.58, "y": 0.445, "z": 0.02}
    closed[387] = {"x": 0.62, "y": 0.445, "z": 0.02}
    closed[380] = {"x": 0.58, "y": 0.455, "z": 0.02}
    closed[373] = {"x": 0.62, "y": 0.455, "z": 0.02}
    fixtures["closed_eyes"] = closed

    # 4. Yawn (wide mouth, MAR ~ 0.75)
    yawn = [dict(p) for p in neutral]
    yawn[13] = {"x": 0.50, "y": 0.56, "z": 0.02}
    yawn[14] = {"x": 0.50, "y": 0.68, "z": 0.02}
    yawn[81] = {"x": 0.46, "y": 0.56, "z": 0.02}
    yawn[178] = {"x": 0.46, "y": 0.68, "z": 0.02}
    yawn[311] = {"x": 0.54, "y": 0.56, "z": 0.02}
    yawn[402] = {"x": 0.54, "y": 0.68, "z": 0.02}
    fixtures["yawn"] = yawn

    return fixtures


def run_python_reference(fixtures):
    results = {}
    
    for name, lms_raw in fixtures.items():
        # Convert to NormalizedLandmark objects
        lms = [NormalizedLandmark(x=p["x"], y=p["y"], z=p["z"]) for p in lms_raw]
        
        # 1. EAR
        ear_res = extract_eye_aspect_ratio(
            landmarks=lms,
            yaw_deg=0.0,
            pitch_deg=0.0,
            aspect_ratio=1.0,
            is_mirrored=False,
            is_normalized=True
        )
        
        # 2. MAR
        mar_val, mar_valid = calculate_mar_with_validity(
            landmarks=lms,
            aspect_ratio=1.0,
            is_normalized=True
        )
        
        # 3. Head Pose & Zone
        pose_res = estimate_head_pose(
            landmarks=lms,
            frame_width=640,
            frame_height=480,
            is_mirrored=False
        )
        
        results[name] = {
            "left_ear": round(float(ear_res.left_ear), 5),
            "right_ear": round(float(ear_res.right_ear), 5),
            "weighted_ear": round(float(ear_res.weighted_ear), 5),
            "compensated_ear": round(float(ear_res.compensated_ear), 5),
            "is_closed": bool(ear_res.is_closed),
            "mar": round(float(mar_val), 5),
            "is_mar_valid": bool(mar_valid),
            "pitch_deg": round(float(pose_res.pitch_deg), 2),
            "yaw_deg": round(float(pose_res.yaw_deg), 2),
            "roll_deg": round(float(pose_res.roll_deg), 2),
            "zone": str(pose_res.zone.value if hasattr(pose_res.zone, "value") else pose_res.zone),
            "is_valid_pose": bool(pose_res.is_valid),
        }
        
    return results


def main():
    print("[DriveGuard] Generating Cross-Language Parity Test Fixtures...")
    fixtures = build_fixture_set()
    py_results = run_python_reference(fixtures)
    
    output_path = os.path.join(os.path.dirname(__file__), "parity_fixtures.json")
    payload = {
        "fixtures": fixtures,
        "python_reference": py_results,
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        
    print(f"[DriveGuard] Parity fixtures written to: {output_path}")
    print("\n--- Python Reference Calculated Metrics ---")
    for name, res in py_results.items():
        print(f"Fixture: {name:<12} | EAR: {res['weighted_ear']:.4f} (closed={res['is_closed']}) | MAR: {res['mar']:.4f} | Zone: {res['zone']} | Pitch: {res['pitch_deg']:.1f}° | Yaw: {res['yaw_deg']:.1f}°")


if __name__ == "__main__":
    main()
