"""
DriveGuard Cross-Language Golden Parity Generator (Phase 6).

Generates canonical facial landmark fixtures, rotated 6-DoF head-pose fixtures,
a snake_case calibrated DriverBehaviourProfile DTO, and multi-frame physiological
sequences, then executes the authoritative Python pipeline
(`DriverFeatureExtractor` + `DrowsinessEngine`) across `HIGH`, `STANDARD`, and
`RELAXED` sensitivity presets and `mirror_suppression_enabled` modes.

Outputs JSON fixtures to `scripts/validation/parity_fixtures.json`.
"""

from __future__ import annotations

import json
import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add repository root to Python path
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    ZoneBaseline,
)
from backend.app.features.ear import extract_eye_aspect_ratio
from backend.app.features.feature_extractor import DriverFeatureExtractor
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    SOLVEPNP_LANDMARK_INDICES,
    PoseZone,
    estimate_head_pose,
)
from backend.app.features.mar import calculate_mar_with_validity
from backend.app.intelligence.drowsiness_engine import DrowsinessEngine
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)

FRAME_W = 640
FRAME_H = 480


def build_rotation_matrix_yxz(
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    *,
    is_mirrored: bool = False,
) -> np.ndarray:
    """
    Build camera-space rotation matrix R = Ry(psi) @ Rx(theta) @ Rz(phi)
    matching DriveGuard's Cardan Y-X-Z Euler sign convention.
    """
    cam_yaw = yaw_deg if is_mirrored else -yaw_deg
    cam_pitch = -pitch_deg
    cam_roll = -roll_deg if is_mirrored else roll_deg

    psi = math.radians(cam_yaw)
    theta = math.radians(cam_pitch)
    phi = math.radians(cam_roll)

    ry = np.array(
        [
            [math.cos(psi), 0.0, math.sin(psi)],
            [0.0, 1.0, 0.0],
            [-math.sin(psi), 0.0, math.cos(psi)],
        ],
        dtype=np.float64,
    )
    rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(theta), -math.sin(theta)],
            [0.0, math.sin(theta), math.cos(theta)],
        ],
        dtype=np.float64,
    )
    rz = np.array(
        [
            [math.cos(phi), -math.sin(phi), 0.0],
            [math.sin(phi), math.cos(phi), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ry @ rx @ rz


def create_face_landmarks(
    *,
    ear_target: float = 0.285,
    mar_target: float = 0.15,
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
    is_mirrored: bool = False,
) -> Tuple[List[Dict[str, float]], List[float]]:
    """
    Construct a realistic 478-landmark face at exact desired EAR, MAR, and 6-DoF
    head pose, together with a matching 4x4 row-major facial transformation matrix.
    """
    lms = [{"x": 0.5, "y": 0.5, "z": 0.0} for _ in range(478)]

    # 1. Rigid cranial + mandible anchor points via exact pinhole projection of CANONICAL_3D_FACE_POINTS
    rot = build_rotation_matrix_yxz(yaw_deg, pitch_deg, roll_deg, is_mirrored=is_mirrored)
    z0_mm = 900.0
    fx = float(FRAME_W)
    fy = float(FRAME_W)
    cx = FRAME_W / 2.0
    cy = FRAME_H / 2.0

    for i, lm_idx in enumerate(SOLVEPNP_LANDMARK_INDICES):
        p_rot = rot @ CANONICAL_3D_FACE_POINTS[i]
        p_cam_x = float(p_rot[0])
        p_cam_y = float(p_rot[1])
        p_cam_z = float(p_rot[2] + z0_mm)

        u = fx * (p_cam_x / p_cam_z) + cx
        v = fy * (p_cam_y / p_cam_z) + cy
        lms[lm_idx] = {
            "x": round(u / FRAME_W, 7),
            "y": round(v / FRAME_H, 7),
            "z": round(float(p_rot[2]) / z0_mm, 7),
        }

    # 2. Construct eye contours anchored to projected outer corners 33 & 263
    # Using isotropic horizontal width (0.10 in normalized x at aspect_ratio=FRAME_W/FRAME_H)
    aspect_ratio = FRAME_W / float(FRAME_H)
    eye_w_norm = 0.10
    # EAR = (d1 + d2) / (2 * d3) with d3 = eye_w_norm * aspect_ratio => d_vert_y = ear_target * eye_w_norm * aspect_ratio
    half_v = 0.5 * ear_target * eye_w_norm * aspect_ratio

    # Right eye (indices 33, 160, 158, 133, 153, 144) — keep index 33 at projected skull position!
    r_outer = lms[33]
    r_cx = r_outer["x"] + eye_w_norm * 0.5
    r_cy = r_outer["y"]
    r_z = r_outer["z"]
    lms[133] = {"x": round(r_outer["x"] + eye_w_norm, 7), "y": r_cy, "z": r_z}
    lms[160] = {"x": round(r_cx - eye_w_norm * 0.2, 7), "y": round(r_cy - half_v, 7), "z": r_z}
    lms[158] = {"x": round(r_cx + eye_w_norm * 0.2, 7), "y": round(r_cy - half_v, 7), "z": r_z}
    lms[144] = {"x": round(r_cx - eye_w_norm * 0.2, 7), "y": round(r_cy + half_v, 7), "z": r_z}
    lms[153] = {"x": round(r_cx + eye_w_norm * 0.2, 7), "y": round(r_cy + half_v, 7), "z": r_z}

    # Left eye (indices 362, 385, 387, 263, 373, 380) — keep index 263 at projected skull position!
    l_outer = lms[263]
    l_cx = l_outer["x"] - eye_w_norm * 0.5
    l_cy = l_outer["y"]
    l_z = l_outer["z"]
    lms[362] = {"x": round(l_outer["x"] - eye_w_norm, 7), "y": l_cy, "z": l_z}
    lms[385] = {"x": round(l_cx - eye_w_norm * 0.2, 7), "y": round(l_cy - half_v, 7), "z": l_z}
    lms[387] = {"x": round(l_cx + eye_w_norm * 0.2, 7), "y": round(l_cy - half_v, 7), "z": l_z}
    lms[380] = {"x": round(l_cx - eye_w_norm * 0.2, 7), "y": round(l_cy + half_v, 7), "z": l_z}
    lms[373] = {"x": round(l_cx + eye_w_norm * 0.2, 7), "y": round(l_cy + half_v, 7), "z": l_z}

    # 3. Construct inner-lip pairs relative to outer corners 61 & 291 to hit exact mar_target
    p61 = lms[61]
    p291 = lms[291]
    mouth_horiz = math.hypot((p61["x"] - p291["x"]) * aspect_ratio, p61["y"] - p291["y"])
    half_mouth_v = 0.5 * mar_target * mouth_horiz
    m_cx = 0.5 * (p61["x"] + p291["x"])
    m_cy = 0.5 * (p61["y"] + p291["y"])
    m_z = 0.5 * (p61["z"] + p291["z"])
    dx = 0.25 * (p291["x"] - p61["x"])

    lms[13] = {"x": round(m_cx, 7), "y": round(m_cy - half_mouth_v, 7), "z": m_z}
    lms[14] = {"x": round(m_cx, 7), "y": round(m_cy + half_mouth_v, 7), "z": m_z}
    lms[81] = {"x": round(m_cx - dx, 7), "y": round(m_cy - half_mouth_v, 7), "z": m_z}
    lms[178] = {"x": round(m_cx - dx, 7), "y": round(m_cy + half_mouth_v, 7), "z": m_z}
    lms[311] = {"x": round(m_cx + dx, 7), "y": round(m_cy - half_mouth_v, 7), "z": m_z}
    lms[402] = {"x": round(m_cx + dx, 7), "y": round(m_cy + half_mouth_v, 7), "z": m_z}

    # Row-major 4x4 transformation matrix for Float32Array parity verification
    mat4 = [
        float(rot[0, 0]), float(rot[0, 1]), float(rot[0, 2]), 0.0,
        float(rot[1, 0]), float(rot[1, 1]), float(rot[1, 2]), 0.0,
        float(rot[2, 0]), float(rot[2, 1]), float(rot[2, 2]), z0_mm,
        0.0, 0.0, 0.0, 1.0,
    ]
    return lms, mat4


def build_calibrated_profile() -> Tuple[DriverBehaviourProfile, Dict[str, Any]]:
    """Create a canonical calibrated 4-zone profile and its snake_case DTO dict."""
    from backend.app.ai.profile_adapter import profile_to_dto

    identity = np.eye(3, dtype=np.float64)
    zones: Dict[str, ZoneBaseline] = {
        PoseZone.CENTER.value: ZoneBaseline(
            zone=PoseZone.CENTER.value,
            ear_median=0.285,
            ear_mad_scaled=0.015,
            pitch_median_deg=0.0,
            yaw_median_deg=0.0,
            sample_count=45,
        ),
        PoseZone.LEFT_MIRROR.value: ZoneBaseline(
            zone=PoseZone.LEFT_MIRROR.value,
            ear_median=0.275,
            ear_mad_scaled=0.015,
            pitch_median_deg=0.0,
            yaw_median_deg=24.0,
            sample_count=40,
        ),
        PoseZone.RIGHT_MIRROR.value: ZoneBaseline(
            zone=PoseZone.RIGHT_MIRROR.value,
            ear_median=0.275,
            ear_mad_scaled=0.015,
            pitch_median_deg=0.0,
            yaw_median_deg=-24.0,
            sample_count=40,
        ),
        PoseZone.DASHBOARD.value: ZoneBaseline(
            zone=PoseZone.DASHBOARD.value,
            ear_median=0.275,
            ear_mad_scaled=0.015,
            pitch_median_deg=-15.0,
            yaw_median_deg=0.0,
            sample_count=40,
        ),
    }
    profile = DriverBehaviourProfile(
        driver_id="parity_driver",
        calibration_mode="FOUR_ZONE",
        center_ear_median=0.285,
        center_ear_mad_scaled=0.015,
        blink_duration_median_ms=235.0,
        blink_duration_mad_ms=30.0,
        blink_rcr_median=0.95,
        blink_rcr_mad=0.12,
        blink_freq_median_per_min=15.0,
        blink_freq_mad_per_min=4.0,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        calibration_reliability=0.95,
        zone_baselines=zones,
        active_center_ear_median=0.285,
        neutral_rot_mat=identity,
        calibration_is_mirrored=False,
    )
    dto = profile_to_dto(profile).model_dump()
    return profile, dto


def build_static_fixtures() -> Dict[str, Dict[str, Any]]:
    specs = {
        "neutral": dict(ear_target=0.285, mar_target=0.15, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        "blink_onset": dict(ear_target=0.210, mar_target=0.15, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        "closed_eyes": dict(ear_target=0.100, mar_target=0.15, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        "yawn": dict(ear_target=0.285, mar_target=0.75, yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0),
        "yaw_left_25": dict(ear_target=0.275, mar_target=0.15, yaw_deg=25.0, pitch_deg=0.0, roll_deg=0.0),
        "yaw_right_25": dict(ear_target=0.275, mar_target=0.15, yaw_deg=-25.0, pitch_deg=0.0, roll_deg=0.0),
        "pitch_down_20": dict(ear_target=0.275, mar_target=0.15, yaw_deg=0.0, pitch_deg=-20.0, roll_deg=0.0),
        "roll_plus_15": dict(ear_target=0.285, mar_target=0.15, yaw_deg=0.0, pitch_deg=0.0, roll_deg=15.0),
        "roll_minus_15": dict(ear_target=0.285, mar_target=0.15, yaw_deg=0.0, pitch_deg=0.0, roll_deg=-15.0),
        "compound_pose": dict(ear_target=0.275, mar_target=0.15, yaw_deg=25.0, pitch_deg=-20.0, roll_deg=15.0),
    }
    out: Dict[str, Dict[str, Any]] = {}
    for name, kw in specs.items():
        lms, mat4 = create_face_landmarks(**kw)
        out[name] = {
            "landmarks": lms,
            "facial_matrix_flat": mat4,
            "expected_pose_input": {
                "yaw_deg": kw["yaw_deg"],
                "pitch_deg": kw["pitch_deg"],
                "roll_deg": kw["roll_deg"],
            },
        }
    return out


def run_python_static_reference(static_fixtures: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    aspect_ratio = FRAME_W / float(FRAME_H)
    results: Dict[str, Any] = {}
    for name, item in static_fixtures.items():
        lms = [NormalizedLandmark(x=p["x"], y=p["y"], z=p["z"]) for p in item["landmarks"]]
        pose_res = estimate_head_pose(
            landmarks=lms,
            frame_width=FRAME_W,
            frame_height=FRAME_H,
            is_mirrored=False,
        )
        ear_res = extract_eye_aspect_ratio(
            landmarks=lms,
            yaw_deg=pose_res.yaw_deg,
            pitch_deg=pose_res.pitch_deg,
            aspect_ratio=aspect_ratio,
            is_mirrored=False,
            is_normalized=True,
        )
        mar_val, mar_valid = calculate_mar_with_validity(
            landmarks=lms,
            aspect_ratio=aspect_ratio,
            is_normalized=True,
        )
        results[name] = {
            "left_ear": round(float(ear_res.left_ear), 6),
            "right_ear": round(float(ear_res.right_ear), 6),
            "weighted_ear": round(float(ear_res.weighted_ear), 6),
            "compensated_ear": round(float(ear_res.compensated_ear), 6),
            "is_closed": bool(ear_res.is_closed),
            "mar": round(float(mar_val), 6),
            "is_mar_valid": bool(mar_valid),
            "pitch_deg": round(float(pose_res.pitch_deg), 3),
            "yaw_deg": round(float(pose_res.yaw_deg), 3),
            "roll_deg": round(float(pose_res.roll_deg), 3),
            "zone": str(pose_res.zone.value),
            "is_valid_pose": bool(pose_res.is_valid),
        }
    return results


def _run_python_sequence(
    frames: List[Dict[str, Any]],
    profile: DriverBehaviourProfile,
    sensitivity_level: str = "STANDARD",
    mirror_suppression_enabled: bool = True,
) -> Dict[str, Any]:
    extractor = DriverFeatureExtractor(profile=profile)
    engine = DrowsinessEngine(
        sensitivity_level=sensitivity_level,
        mirror_suppression_enabled=mirror_suppression_enabled,
    )
    frame_outputs: List[Dict[str, Any]] = []
    for f in frames:
        lms = tuple(NormalizedLandmark(x=p["x"], y=p["y"], z=p["z"]) for p in f["landmarks"])
        lm_res = FaceLandmarkResult(
            has_face=True,
            landmarks=lms,
            presence_confidence=0.95,
            timestamp_ms=int(f["timestamp_ms"]),
            status=FaceLandmarkStatus.TRACKING,
            inference_latency_ms=2.0,
            reasons=("Tracking",),
        )
        features = extractor.extract(lm_res, frame_width=FRAME_W, frame_height=FRAME_H, is_mirrored=False)
        assessment = engine.evaluate(features, quality=None, is_profile_calibrated=True, is_calibrating=False)
        frame_outputs.append(
            {
                "timestamp_ms": f["timestamp_ms"],
                "ear": round(float(features.ear.compensated_ear), 6),
                "mar": round(float(features.mar.mar), 6),
                "perclos_pct": round(float(features.perclos.perclos_pct), 6),
                "pitch_deg": round(float(features.head_pose.pitch_deg), 3),
                "yaw_deg": round(float(features.head_pose.yaw_deg), 3),
                "roll_deg": round(float(features.head_pose.roll_deg), 3),
                "zone": features.head_pose.zone.value,
                "total_blinks": int(features.blink.total_blink_count),
                "total_yawns": int(features.mar.total_yawn_count),
                "is_yawn_in_progress": bool(features.mar.is_yawn_in_progress),
                "is_nodding_microsleep": bool(assessment.temporal_metrics.is_nodding_microsleep),
                "operational_state": assessment.operational_state.value,
                "state": assessment.state.value,
                "primary_reason": assessment.primary_reason.code.value if assessment.primary_reason else None,
            }
        )
    return {
        "sensitivity_level": sensitivity_level,
        "mirror_suppression_enabled": mirror_suppression_enabled,
        "frames": frame_outputs,
    }


def _make_seq_frame(ts_ms: int, **kwargs: Any) -> Dict[str, Any]:
    lms, mat4 = create_face_landmarks(**kwargs)
    return {
        "timestamp_ms": ts_ms,
        "landmarks": lms,
        "facial_matrix_flat": mat4,
    }


def build_sequences(profile: DriverBehaviourProfile) -> Dict[str, Any]:
    sequences: Dict[str, Any] = {}

    # Sequence 1: Normal 200ms binocular blink (step = 100ms)
    blink_frames = [
        _make_seq_frame(0, ear_target=0.285),
        _make_seq_frame(100, ear_target=0.285),
        _make_seq_frame(200, ear_target=0.120),
        _make_seq_frame(300, ear_target=0.110),
        _make_seq_frame(400, ear_target=0.285),
        _make_seq_frame(500, ear_target=0.285),
    ]
    sequences["normal_blink"] = {
        "frames": blink_frames,
        "runs": [_run_python_sequence(blink_frames, profile, "STANDARD", True)],
    }

    # Sequence 2: 500ms speech burst (rejected) followed by 1800ms authentic yawn (accepted)
    yawn_frames: List[Dict[str, Any]] = []
    ts = 0
    for _ in range(3):
        yawn_frames.append(_make_seq_frame(ts, ear_target=0.285, mar_target=0.15))
        ts += 100
    # 500ms speech burst (MAR = 0.65)
    for _ in range(5):
        yawn_frames.append(_make_seq_frame(ts, ear_target=0.285, mar_target=0.65))
        ts += 100
    # Close mouth for 400ms
    for _ in range(4):
        yawn_frames.append(_make_seq_frame(ts, ear_target=0.285, mar_target=0.15))
        ts += 100
    # 1800ms authentic yawn (MAR = 0.72 -> 19 frames @ 100ms = 1800ms observed)
    for _ in range(19):
        yawn_frames.append(_make_seq_frame(ts, ear_target=0.285, mar_target=0.72))
        ts += 100
    # Close mouth to complete yawn
    for _ in range(2):
        yawn_frames.append(_make_seq_frame(ts, ear_target=0.285, mar_target=0.15))
        ts += 100
    sequences["speech_vs_yawn"] = {
        "frames": yawn_frames,
        "runs": [_run_python_sequence(yawn_frames, profile, "STANDARD", True)],
    }

    # Sequence 3: 4200ms Left Mirror Glance across STANDARD (suppression ON/OFF), HIGH (suppression OFF), RELAXED
    mirror_frames: List[Dict[str, Any]] = []
    ts = 0
    for _ in range(2):
        mirror_frames.append(_make_seq_frame(ts, ear_target=0.285, yaw_deg=0.0))
        ts += 100
    for _ in range(43):  # 4200ms sustained left mirror glance
        mirror_frames.append(_make_seq_frame(ts, ear_target=0.275, yaw_deg=24.0))
        ts += 100
    sequences["mirror_distraction_3500ms"] = {
        "frames": mirror_frames,
        "runs": [
            _run_python_sequence(mirror_frames, profile, "STANDARD", True),
            _run_python_sequence(mirror_frames, profile, "HIGH", False),
            _run_python_sequence(mirror_frames, profile, "RELAXED", True),
        ],
    }

    # Sequence 4: Nodding Microsleep (eyes closed >= 800ms + pitch slump -18 deg)
    nod_frames: List[Dict[str, Any]] = []
    ts = 0
    for _ in range(3):
        nod_frames.append(_make_seq_frame(ts, ear_target=0.285, pitch_deg=0.0))
        ts += 100
    for _ in range(10):  # 900ms closed + pitched down -18 deg
        nod_frames.append(_make_seq_frame(ts, ear_target=0.110, pitch_deg=-18.0))
        ts += 100
    sequences["nodding_microsleep"] = {
        "frames": nod_frames,
        "runs": [_run_python_sequence(nod_frames, profile, "STANDARD", True)],
    }

    # Sequence 5: 1.9s Sustained Eye Closure across HIGH (1200ms), STANDARD (1500ms), RELAXED (1800ms)
    closure_frames: List[Dict[str, Any]] = []
    ts = 0
    for _ in range(2):
        closure_frames.append(_make_seq_frame(ts, ear_target=0.285))
        ts += 100
    for _ in range(21):  # 2000ms closure
        closure_frames.append(_make_seq_frame(ts, ear_target=0.110))
        ts += 100
    runs_closure = [
        _run_python_sequence(closure_frames, profile, level, True)
        for level in ("HIGH", "STANDARD", "RELAXED")
    ]

    # Verify Python-side sensitivity feature invariance (diff == 0.0 across all 3 levels)
    base_frames = runs_closure[0]["frames"]
    for other_run in runs_closure[1:]:
        for f0, f1 in zip(base_frames, other_run["frames"]):
            for key in ("ear", "mar", "perclos_pct", "pitch_deg", "yaw_deg", "roll_deg"):
                assert f0[key] == f1[key], (
                    f"Python feature invariance violated for {key}: {f0[key]} != {f1[key]}"
                )

    sequences["sustained_closure_presets"] = {
        "frames": closure_frames,
        "runs": runs_closure,
    }

    return sequences


def main() -> None:
    print("[DriveGuard] Generating Phase 6 Cross-Language Golden Parity Fixtures...")
    profile, profile_dto = build_calibrated_profile()
    static_fixtures = build_static_fixtures()
    py_static_ref = run_python_static_reference(static_fixtures)
    sequences = build_sequences(profile)

    output_path = os.path.join(os.path.dirname(__file__), "parity_fixtures.json")
    payload = {
        "frame_width": FRAME_W,
        "frame_height": FRAME_H,
        "calibrated_profile": profile_dto,
        "fixtures": {k: v["landmarks"] for k, v in static_fixtures.items()},
        "static_fixtures": static_fixtures,
        "python_reference": py_static_ref,
        "sequences": sequences,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"[DriveGuard] Golden parity fixtures written to: {output_path}")
    print("\n--- Static & Rotated 6-DoF Python Reference ---")
    for name, res in py_static_ref.items():
        print(
            f"  {name:<15} | EAR={res['weighted_ear']:.4f} | MAR={res['mar']:.4f} | "
            f"Yaw={res['yaw_deg']:+6.2f}° Pitch={res['pitch_deg']:+6.2f}° Roll={res['roll_deg']:+6.2f}° | Zone={res['zone']}"
        )
    print(f"\n--- Multi-Frame Sequences Generated: {len(sequences)} ---")
    for seq_name, seq_data in sequences.items():
        print(f"  {seq_name:<28} | frames={len(seq_data['frames']):2d} | preset_runs={len(seq_data['runs'])}")


if __name__ == "__main__":
    main()
