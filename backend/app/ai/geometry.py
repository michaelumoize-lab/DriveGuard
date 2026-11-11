"""
Geometric Feature Extraction, Pose Zone Classification, and Eye Crop Quality Assessment.

Implements:
- 3D/2D Eye Aspect Ratio (EAR) from MediaPipe 478 facial landmarks
- Mouth Aspect Ratio (MAR) for yawning analysis
- Head Pose Euler Angles (Pitch, Yaw, Roll) from transformation matrix or solvePnP
- 4-Zone Pose Classification (CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD)
- Eye Crop Extraction & Visual Reliability Scoring (sharpness, exposure, glare check)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class PoseZone(str, Enum):
    """Discrete driver gaze/head orientation zones for pose-conditioned calibration."""

    CENTER = "CENTER"
    LEFT_MIRROR = "LEFT_MIRROR"
    RIGHT_MIRROR = "RIGHT_MIRROR"
    DASHBOARD = "DASHBOARD"
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"


# MediaPipe Face Mesh 6-point eye contour indices: [p1(outer), p2(top1), p3(top2), p4(inner), p5(bot2), p6(bot1)]
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)

# MediaPipe Mouth contour indices for MAR: [left_corner, right_corner, top_inner, bot_inner, top_left, bot_left, top_right, bot_right]
MOUTH_OUTER_CORNERS = (61, 291)
MOUTH_VERTICAL_PAIRS = ((13, 14), (81, 178), (311, 402))

# Canonical 3D face model points for solvePnP fallback (in mm)
CANONICAL_3D_FACE_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (idx 1)
        (0.0, -63.6, -12.5),      # Chin (idx 152)
        (-43.3, 32.7, -26.0),     # Left eye outer corner (idx 33)
        (43.3, 32.7, -26.0),      # Right eye outer corner (idx 263)
        (-28.9, -28.9, -24.1),    # Left mouth corner (idx 61)
        (28.9, -28.9, -24.1),     # Right mouth corner (idx 291)
    ],
    dtype=np.float64,
)
SOLVEPNP_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


@dataclass
class EyeCropQuality:
    """Visual reliability metrics for an extracted eye image crop."""

    crop_rgb: Optional[np.ndarray]
    sharpness_score: float       # Normalized [0, 1] from Laplacian variance
    exposure_score: float        # Normalized [0, 1] penalizing extreme dark or glare clipping
    glare_fraction: float        # Fraction of saturated specular highlight pixels
    overall_reliability: float   # Combined visual quality in [0, 1]


@dataclass
class FrameGeometryFeatures:
    """Per-frame geometric measurements extracted from MediaPipe landmarks."""

    timestamp_ms: float
    face_detected: bool
    landmark_confidence: float
    left_ear: float
    right_ear: float
    yaw_weighted_ear: float
    blendshape_blink_left: Optional[float]
    blendshape_blink_right: Optional[float]
    blendshape_blink_mean: Optional[float]
    mar: float
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    pose_zone: PoseZone
    geometric_reliability: float
    blendshape_brow_down: Optional[float] = None
    blendshape_eye_squint: Optional[float] = None
    left_eye_crop: Optional[EyeCropQuality] = None
    right_eye_crop: Optional[EyeCropQuality] = None
    tracking_mode: str = "MEDIAPIPE_TASKS"


def compute_single_eye_ear(landmarks_xyz: np.ndarray, indices: Tuple[int, ...], eps: float = 1e-6) -> float:
    """
    Calculate the Eye Aspect Ratio (EAR) from 6 3D/2D landmark coordinates.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4|| + eps)
    """
    pts = landmarks_xyz[list(indices), :2]  # Use normalized image-plane (x, y) scaled to aspect
    v1 = float(np.linalg.norm(pts[1] - pts[5]))
    v2 = float(np.linalg.norm(pts[2] - pts[4]))
    horiz = float(np.linalg.norm(pts[0] - pts[3]))
    if horiz < eps:
        return 0.0
    return (v1 + v2) / (2.0 * horiz + eps)


def compute_yaw_weighted_ear(left_ear: float, right_ear: float, yaw_deg: float) -> float:
    """
    Combine left and right EAR while handling partial occlusion when the head turns.
    In front-facing webcam frames:
      - Positive yaw (> 15 deg) indicates head turned toward driver's LEFT (right eye is more frontal).
      - Negative yaw (< -15 deg) indicates head turned toward driver's RIGHT (left eye is more frontal).
    """
    if yaw_deg > 15.0:
        # Head turned left -> right eye is closer/more frontal to camera
        w_left, w_right = 0.25, 0.75
    elif yaw_deg < -15.0:
        # Head turned right -> left eye is closer/more frontal to camera
        w_left, w_right = 0.75, 0.25
    else:
        w_left, w_right = 0.50, 0.50
    return w_left * left_ear + w_right * right_ear


def compute_pose_compensated_ear(
    left_ear: float,
    right_ear: float,
    yaw_deg: float,
    pitch_deg: float = 0.0,
    neutral_pitch_deg: float = 0.0,
) -> float:
    """
    Combine left and right EAR with yaw perspective weighting and downward pitch compensation.
    When the driver glances downward towards the dashboard/instruments (pitch < -4 deg),
    eyelids naturally foreshorten from the camera's perspective. Compensating by up to 18%
    prevents normal road/cockpit scanning from falsely triggering closure alarms.
    """
    base_ear = compute_yaw_weighted_ear(left_ear, right_ear, yaw_deg)
    rel_pitch = pitch_deg - neutral_pitch_deg
    if rel_pitch < -4.0:
        pitch_boost = 1.0 + min(0.18, max(0.0, (-rel_pitch - 4.0) * 0.012))
        return float(base_ear * pitch_boost)
    return float(base_ear)


def compute_mar(landmarks_xyz: np.ndarray, eps: float = 1e-6) -> float:
    """
    Calculate the Mouth Aspect Ratio (MAR) using 3 vertical inner-lip pairs and outer corners.
    """
    left_c = landmarks_xyz[MOUTH_OUTER_CORNERS[0], :2]
    right_c = landmarks_xyz[MOUTH_OUTER_CORNERS[1], :2]
    horiz = float(np.linalg.norm(left_c - right_c))
    if horiz < eps:
        return 0.0

    vert_sum = 0.0
    for top_idx, bot_idx in MOUTH_VERTICAL_PAIRS:
        vert_sum += float(np.linalg.norm(landmarks_xyz[top_idx, :2] - landmarks_xyz[bot_idx, :2]))
    return (vert_sum / len(MOUTH_VERTICAL_PAIRS)) / (horiz + eps)


def estimate_head_pose_from_matrix(transformation_matrix: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract (pitch_deg, yaw_deg, roll_deg) from MediaPipe's 4x4 facial transformation matrix.
    """
    rot = transformation_matrix[:3, :3]
    sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
    singular = sy < 1e-6

    if not singular:
        pitch = float(np.degrees(np.arctan2(rot[2, 1], rot[2, 2])))
        yaw = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
        roll = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
    else:
        pitch = float(np.degrees(np.arctan2(-rot[1, 2], rot[1, 1])))
        yaw = float(np.degrees(np.arctan2(-rot[2, 0], sy)))
        roll = 0.0

    return pitch, yaw, roll


def estimate_head_pose_solvepnp(
    landmarks_pixel_xyz: np.ndarray, frame_width: int, frame_height: int
) -> Tuple[float, float, float]:
    """
    Fallback head-pose estimation using OpenCV solvePnP when transformation matrix is unavailable.
    """
    image_pts = np.array(
        [landmarks_pixel_xyz[idx, :2] for idx in SOLVEPNP_LANDMARK_INDICES],
        dtype=np.float64,
    )
    focal_length = float(frame_width)
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [[focal_length, 0.0, center[0]], [0.0, focal_length, center[1]], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rvec, tvec = cv2.solvePnP(
        CANONICAL_3D_FACE_POINTS,
        image_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rvec)
    proj_matrix = np.hstack((rot_mat, tvec))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)
    pitch = float(euler_angles[0, 0])
    yaw = float(euler_angles[1, 0])
    roll = float(euler_angles[2, 0])

    # Normalize pitch wrap-around near +/-180 deg
    if pitch > 90.0:
        pitch -= 180.0
    elif pitch < -90.0:
        pitch += 180.0

    return pitch, yaw, roll


def classify_pose_zone(
    pitch_deg: float,
    yaw_deg: float,
    neutral_pitch_deg: float = 0.0,
    neutral_yaw_deg: float = 0.0,
) -> PoseZone:
    """
    Assign the driver's instantaneous head orientation to one of the 4 calibrated zones
    (or OUT_OF_BOUNDS if severely turned away).

    Relative angles are measured from the driver's neutral center posture.
    """
    rel_pitch = pitch_deg - neutral_pitch_deg
    rel_yaw = yaw_deg - neutral_yaw_deg

    if abs(rel_yaw) > 48.0 or abs(rel_pitch) > 38.0:
        return PoseZone.OUT_OF_BOUNDS

    # Mirror-corrected driver perspective (front-facing webcam):
    # Driver turning left yields positive relative yaw; turning right yields negative relative yaw.
    if rel_yaw >= 12.0:
        return PoseZone.LEFT_MIRROR
    if rel_yaw <= -12.0:
        return PoseZone.RIGHT_MIRROR
    if rel_pitch <= -8.0:
        return PoseZone.DASHBOARD

    return PoseZone.CENTER


def extract_eye_crop_with_quality(
    frame_rgb: np.ndarray,
    landmarks_pixel_xyz: np.ndarray,
    eye_indices: Tuple[int, ...],
    output_size: Tuple[int, int] = (64, 64),
    padding_ratio: float = 0.45,
) -> EyeCropQuality:
    """
    Extract a padded square/rectangular eye crop and evaluate its visual reliability
    (sharpness via Laplacian variance, exposure, and glare saturation fraction).
    """
    h, w = frame_rgb.shape[:2]
    pts = landmarks_pixel_xyz[list(eye_indices), :2]

    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    eye_w = max(float(max_x - min_x), 8.0)
    eye_h = max(float(max_y - min_y), 6.0)

    cx = 0.5 * (min_x + max_x)
    cy = 0.5 * (min_y + max_y)
    half_side = 0.5 * max(eye_w, eye_h * 1.8) * (1.0 + padding_ratio)

    x1 = int(max(0, np.floor(cx - half_side)))
    y1 = int(max(0, np.floor(cy - half_side)))
    x2 = int(min(w, np.ceil(cx + half_side)))
    y2 = int(min(h, np.ceil(cy + half_side)))

    if (x2 - x1) < 10 or (y2 - y1) < 8:
        return EyeCropQuality(
            crop_rgb=None,
            sharpness_score=0.0,
            exposure_score=0.0,
            glare_fraction=1.0,
            overall_reliability=0.0,
        )

    raw_crop = frame_rgb[y1:y2, x1:x2]
    resized_crop = cv2.resize(raw_crop, output_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized_crop, cv2.COLOR_RGB2GRAY)

    # 1. Sharpness via variance of Laplacian (typical sharp 64x64 eye crop > 80)
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    sharpness_score = float(np.clip(lap_var / 100.0, 0.0, 1.0))

    # 2. Brightness & exposure score (ideal mean intensity in [60, 195])
    mean_intensity = float(np.mean(gray))
    if mean_intensity < 30.0:
        exposure_score = max(0.0, mean_intensity / 30.0)
    elif mean_intensity > 225.0:
        exposure_score = max(0.0, (255.0 - mean_intensity) / 30.0)
    else:
        exposure_score = 1.0

    # 3. Specular glare fraction (glasses reflection check: pixels >= 245)
    glare_fraction = float(np.mean(gray >= 245))
    glare_penalty = float(np.clip(1.0 - (glare_fraction / 0.18), 0.0, 1.0))

    overall_reliability = float(
        np.clip(0.40 * sharpness_score + 0.35 * exposure_score + 0.25 * glare_penalty, 0.0, 1.0)
    )

    return EyeCropQuality(
        crop_rgb=resized_crop,
        sharpness_score=sharpness_score,
        exposure_score=exposure_score,
        glare_fraction=glare_fraction,
        overall_reliability=overall_reliability,
    )


def assess_geometric_reliability(
    face_detected: bool,
    landmark_confidence: float,
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float,
    left_ear: float,
    right_ear: float,
    blendshape_blink_mean: Optional[float] = None,
    reference_open_ear: float = 0.28,
) -> Tuple[float, float]:
    """
    Evaluate how reliable and internally consistent the geometric landmark measurements are.

    Returns:
        (geometric_reliability in [0, 1], disagreement_score in [0, 1])
    """
    if not face_detected:
        return 0.0, 1.0

    # 1. Pose penalty: extreme head rotation reduces 2D landmark reliability
    yaw_penalty = float(np.clip(1.0 - max(0.0, abs(yaw_deg) - 18.0) / 35.0, 0.1, 1.0))
    pitch_penalty = float(np.clip(1.0 - max(0.0, abs(pitch_deg) - 15.0) / 30.0, 0.1, 1.0))
    roll_penalty = float(np.clip(1.0 - max(0.0, abs(roll_deg) - 20.0) / 35.0, 0.2, 1.0))
    pose_factor = 0.45 * yaw_penalty + 0.35 * pitch_penalty + 0.20 * roll_penalty

    # 2. Left-right eye symmetry consistency (when head is near frontal)
    ear_diff = abs(left_ear - right_ear)
    symmetry_factor = float(np.clip(1.0 - (ear_diff / 0.14), 0.2, 1.0)) if abs(yaw_deg) < 15.0 else 0.85

    # 3. Disagreement between geometric EAR closure and MediaPipe blendshape blink score
    mean_ear = 0.5 * (left_ear + right_ear)
    implied_geom_closure = float(
        np.clip((reference_open_ear - mean_ear) / max(reference_open_ear * 0.65, 0.08), 0.0, 1.0)
    )
    if blendshape_blink_mean is not None:
        disagreement = abs(implied_geom_closure - float(blendshape_blink_mean))
    else:
        disagreement = 0.0

    consistency_factor = float(np.clip(1.0 - disagreement, 0.0, 1.0))
    reliability = float(
        np.clip(
            landmark_confidence * (0.45 * pose_factor + 0.25 * symmetry_factor + 0.30 * consistency_factor),
            0.0,
            1.0,
        )
    )
    return reliability, float(disagreement)
