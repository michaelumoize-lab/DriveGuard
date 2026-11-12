"""
DriveGuard — Head Pose Estimation & Cockpit Zone Classifier (Phase 5).

1. What it does:
   Estimates driver 3D head orientation (Pitch, Yaw, Roll in degrees) using
   OpenCV's `cv2.solvePnP` algorithm with 6 canonical facial anthropometric landmarks.
   Classifies head orientation into discrete cockpit gaze zones.

2. Why it exists:
   Head pose provides two vital drowsiness & distraction indicators:
   - Pitch: Detects nodding microsleep episodes (head dropping forward).
   - Yaw & Zone: Identifies mirror checks and road distraction, allowing the
     decision engine to avoid false eye closure alerts when the driver looks at mirrors.

3. What data it receives:
   - Sequence of `NormalizedLandmark` points (or NumPy pixel coordinates).
   - Video frame dimensions (`frame_width`, `frame_height`).

4. What data it produces:
   - `HeadPoseResult` dataclass with `pitch_deg`, `yaw_deg`, `roll_deg`,
     `zone`, and `is_looking_away` boolean.

5. Modules that depend on it:
   - `backend.app.features.ear` (yaw-weighted and pitch-compensated EAR)
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine (Phase 6) and Personalization (Phase 7)

6. Failure behavior:
   - Returns 0.0 Euler angles with zone `OUT_OF_BOUNDS` if landmarks are missing or solvePnP fails.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from backend.app.vision.face_landmarker import NormalizedLandmark


class PoseZone(str, Enum):
    """Discrete driver gaze/head orientation cockpit zones."""
    CENTER = "CENTER"              # Forward road view (primary driving orientation)
    LEFT_MIRROR = "LEFT_MIRROR"    # Head turned towards driver's side mirror (yaw > +18°)
    RIGHT_MIRROR = "RIGHT_MIRROR"  # Head turned towards center/passenger mirror (yaw < -18°)
    DASHBOARD = "DASHBOARD"        # Head tilted downward towards instruments (pitch < -14°)
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"# Extreme head turn or tilt outside normal cabin view


# Canonical 3D anthropometric facial model points (in mm, centered at nose tip)
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

# Corresponding MediaPipe landmark indices
SOLVEPNP_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291)


@dataclass(frozen=True)
class HeadPoseResult:
    """
    Structured 6-DoF head pose estimation result.
    """
    pitch_deg: float      # Positive = tilting back/up, Negative = nodding down
    yaw_deg: float        # Positive = turned to driver's left, Negative = turned to right
    roll_deg: float       # Positive = tilted toward right shoulder, Negative = left shoulder
    zone: PoseZone
    is_looking_away: bool
    is_valid: bool
    rel_pitch_deg: float = 0.0  # Pitch relative to neutral baseline
    rel_yaw_deg: float = 0.0    # Yaw relative to neutral baseline


def classify_pose_zone(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float = 0.0,
    neutral_pitch_deg: float = 0.0,
    neutral_yaw_deg: float = 0.0,
) -> PoseZone:
    """
    Classify continuous 3D Euler angles into discrete cockpit zones relative to driver neutral posture.

    Calibrated according to Euro NCAP DMS 2026 and SAE J941 automotive ergonomics:
    - OUT_OF_BOUNDS: |rel_yaw| > 50° OR |rel_pitch| > 38° OR |roll| > 48° (extreme turn / slumped)
    - DASHBOARD: rel_pitch <= -9.0° (natural glance at instrument cluster)
    - LEFT_MIRROR: rel_yaw >= 14.0° (driver-side wing mirror check, 14° to 50°)
    - RIGHT_MIRROR: rel_yaw <= -14.0° (center rearview or passenger mirror check, -50° to -14°)
    - CENTER: forward road view through windshield
    """
    rel_pitch = pitch_deg - neutral_pitch_deg
    rel_yaw = yaw_deg - neutral_yaw_deg

    if abs(rel_yaw) > 50.0 or abs(rel_pitch) > 38.0 or abs(roll_deg) > 48.0:
        return PoseZone.OUT_OF_BOUNDS
    if rel_pitch <= -9.0:
        return PoseZone.DASHBOARD
    if rel_yaw >= 14.0:
        return PoseZone.LEFT_MIRROR
    if rel_yaw <= -14.0:
        return PoseZone.RIGHT_MIRROR
    return PoseZone.CENTER


def estimate_head_pose(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    frame_width: int = 1280,
    frame_height: int = 720,
    neutral_pitch_deg: float = 0.0,
    neutral_yaw_deg: float = 0.0,
) -> HeadPoseResult:
    """
    Compute 6-DoF head pose Euler angles and cockpit zone using cv2.solvePnP.

    Args:
        landmarks: Sequence of NormalizedLandmark points or (N, 2)/(N, 3) NumPy array.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.
        neutral_pitch_deg: Baseline resting pitch angle (default: 0.0°).
        neutral_yaw_deg: Baseline resting yaw angle (default: 0.0°).

    Returns:
        HeadPoseResult detailing Pitch, Yaw, Roll, relative angles, and classified PoseZone.
    """
    max_idx = max(SOLVEPNP_LANDMARK_INDICES)
    if len(landmarks) <= max_idx:
        return HeadPoseResult(
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            zone=PoseZone.OUT_OF_BOUNDS,
            is_looking_away=True,
            is_valid=False,
            rel_pitch_deg=0.0,
            rel_yaw_deg=0.0,
        )

    # Extract 2D image points in pixel space
    if isinstance(landmarks, np.ndarray):
        if landmarks.max() <= 1.5:
            # Normalized coordinates
            image_pts = np.array(
                [
                    (landmarks[idx, 0] * frame_width, landmarks[idx, 1] * frame_height)
                    for idx in SOLVEPNP_LANDMARK_INDICES
                ],
                dtype=np.float64,
            )
        else:
            image_pts = np.array(
                [landmarks[idx, :2] for idx in SOLVEPNP_LANDMARK_INDICES],
                dtype=np.float64,
            )
    else:
        image_pts = np.array(
            [
                (landmarks[idx].x * frame_width, landmarks[idx].y * frame_height)
                for idx in SOLVEPNP_LANDMARK_INDICES
            ],
            dtype=np.float64,
        )

    # Approximate pinhole camera intrinsic matrix
    focal_length = float(frame_width)
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve Perspective-n-Point pose problem
    success, rvec, tvec = cv2.solvePnP(
        CANONICAL_3D_FACE_POINTS,
        image_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return HeadPoseResult(
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            zone=PoseZone.OUT_OF_BOUNDS,
            is_looking_away=True,
            is_valid=False,
        )

    # Convert rotation vector to 3x3 rotation matrix
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

    rel_pitch = pitch - neutral_pitch_deg
    rel_yaw = yaw - neutral_yaw_deg

    zone = classify_pose_zone(
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=roll,
        neutral_pitch_deg=neutral_pitch_deg,
        neutral_yaw_deg=neutral_yaw_deg,
    )
    is_looking_away = zone != PoseZone.CENTER

    return HeadPoseResult(
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=roll,
        zone=zone,
        is_looking_away=is_looking_away,
        is_valid=True,
        rel_pitch_deg=rel_pitch,
        rel_yaw_deg=rel_yaw,
    )
