"""
DriveGuard — Eye Aspect Ratio (EAR) Feature Extraction Module.

1. What it does:
   Calculates the Eye Aspect Ratio (EAR) independently for left and right eyes
   from 3D normalized facial landmarks. Computes perspective-weighted EAR
   conditioned on head yaw angle and pitch angle.

2. Why it exists:
   EAR is the primary geometric metric for detecting eyelid closure and blinks.
   Decoupling left and right eyes allows asymmetrical closure detection (winks vs blinks),
   while perspective weighting prevents false closures when the head turns to check mirrors.

3. What data it receives:
   - Tuple or List of `NormalizedLandmark` points (or NumPy array).
   - Head orientation angles (yaw and pitch in degrees).

4. What data it produces:
   - `EyeAspectRatioResult` dataclass containing `left_ear`, `right_ear`,
     `weighted_ear`, `compensated_ear`, and `is_closed` boolean.

5. Modules that depend on it:
   - `backend.app.features.blink_detector`
   - `backend.app.features.perclos`
   - `backend.app.features.feature_extractor`

6. Failure behavior:
   - Returns EAR = 0.0 with `is_closed=True` if landmarks are missing, empty, or degenerate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from backend.app.vision.face_landmarker import NormalizedLandmark

# MediaPipe 6-point canonical eye contour indices
# Format: (outer_corner, top_1, top_2, inner_corner, bottom_2, bottom_1)
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)

# Default geometric closure threshold (calibrated baseline overrides this in Phase 7)
DEFAULT_EAR_CLOSURE_THRESHOLD = 0.20


@dataclass(frozen=True)
class EyeAspectRatioResult:
    """
    Immutable structured evaluation of eye opening measurements.
    """
    left_ear: float
    right_ear: float
    weighted_ear: float
    compensated_ear: float
    closure_threshold: float
    is_left_closed: bool
    is_right_closed: bool
    is_closed: bool
    is_valid: bool = True


def calculate_single_eye_ear(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    indices: Tuple[int, int, int, int, int, int],
    aspect_ratio: float = 16.0 / 9.0,
    eps: float = 1e-6,
) -> float:
    """
    Compute Eye Aspect Ratio for a single eye using 6 eyelid landmarks:
    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4|| + eps)

    Args:
        landmarks: Sequence of NormalizedLandmark or NumPy array (N, 2) or (N, 3).
        indices: 6 landmark indices for the eye contour.
        aspect_ratio: Frame width/height ratio used to correct pixel aspect ratio.
        eps: Small epsilon preventing division by zero.

    Returns:
        Float scalar representing the eye aspect ratio (typically 0.25 - 0.38 when open, < 0.18 when closed).
    """
    if len(landmarks) <= max(indices):
        return 0.0

    if isinstance(landmarks, np.ndarray):
        pts = landmarks[list(indices), :2]
        # Scale x coordinate by aspect ratio to compute isotropic Euclidean distances
        p1, p2, p3, p4, p5, p6 = pts[0], pts[1], pts[2], pts[3], pts[4], pts[5]
        v1 = float(math.hypot((p2[0] - p6[0]) * aspect_ratio, p2[1] - p6[1]))
        v2 = float(math.hypot((p3[0] - p5[0]) * aspect_ratio, p3[1] - p5[1]))
        horiz = float(math.hypot((p1[0] - p4[0]) * aspect_ratio, p1[1] - p4[1]))
    else:
        # Sequence of NormalizedLandmark objects
        lms = [landmarks[i] for i in indices]
        p1, p2, p3, p4, p5, p6 = lms[0], lms[1], lms[2], lms[3], lms[4], lms[5]
        v1 = math.hypot((p2.x - p6.x) * aspect_ratio, p2.y - p6.y)
        v2 = math.hypot((p3.x - p5.x) * aspect_ratio, p3.y - p5.y)
        horiz = math.hypot((p1.x - p4.x) * aspect_ratio, p1.y - p4.y)

    if horiz < eps:
        return 0.0

    return float((v1 + v2) / (2.0 * horiz + eps))


def calculate_yaw_weighted_ear(left_ear: float, right_ear: float, yaw_deg: float) -> float:
    """
    Combine left and right EAR with yaw perspective weighting.

    Coordinate & Anatomical Mapping:
    - left_ear is computed from LEFT_EYE_INDICES (MediaPipe landmark 33), which is anatomically
      the driver's RIGHT eye (located on the left side of the camera image).
    - right_ear is computed from RIGHT_EYE_INDICES (MediaPipe landmark 362), which is anatomically
      the driver's LEFT eye (located on the right side of the camera image).

    Perspective Physics:
    - Positive yaw (> 15 deg, turned to driver's left): The driver's right eye (left_ear / idx 33)
      directly faces the camera lens with full openness, while the driver's left eye (right_ear / idx 362)
      turns away, foreshortening vertically. Therefore, left_ear receives 75% weight and right_ear receives 25%.
    - Negative yaw (< -15 deg, turned to driver's right): The driver's left eye (right_ear / idx 362)
      directly faces the camera lens, while the driver's right eye (left_ear / idx 33) turns away.
      Therefore, right_ear receives 75% weight and left_ear receives 25%.
    """
    if yaw_deg > 15.0:
        w_left, w_right = 0.75, 0.25
    elif yaw_deg < -15.0:
        w_left, w_right = 0.25, 0.75
    else:
        w_left, w_right = 0.50, 0.50

    return float(w_left * left_ear + w_right * right_ear)


def calculate_pitch_compensated_ear(
    base_ear: float,
    pitch_deg: float,
    neutral_pitch_deg: float = 0.0,
) -> float:
    """
    Compensate EAR when the driver glances downward towards the instrument cluster/dashboard.
    Glancing down (pitch < neutral - 4 deg) naturally foreshortens the eyelid opening from
    the camera's viewpoint. Applying a mild compensation factor (up to 18%) prevents false
    closure alerts during dashboard scanning.
    """
    rel_pitch = pitch_deg - neutral_pitch_deg
    if rel_pitch < -4.0:
        pitch_boost = 1.0 + min(0.18, max(0.0, (-rel_pitch - 4.0) * 0.012))
        return float(base_ear * pitch_boost)
    return float(base_ear)


def extract_eye_aspect_ratio(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    aspect_ratio: float = 16.0 / 9.0,
    closure_threshold: float = DEFAULT_EAR_CLOSURE_THRESHOLD,
) -> EyeAspectRatioResult:
    """
    Complete EAR pipeline extracting left, right, yaw-weighted, and pitch-compensated EAR.
    """
    left_ear = calculate_single_eye_ear(landmarks, LEFT_EYE_INDICES, aspect_ratio=aspect_ratio)
    right_ear = calculate_single_eye_ear(landmarks, RIGHT_EYE_INDICES, aspect_ratio=aspect_ratio)

    weighted = calculate_yaw_weighted_ear(left_ear, right_ear, yaw_deg)
    compensated = calculate_pitch_compensated_ear(weighted, pitch_deg)

    is_valid = (left_ear > 0.0) and (right_ear > 0.0)
    if not is_valid:
        is_left_closed = False
        is_right_closed = False
        is_closed = False
    else:
        is_left_closed = left_ear < closure_threshold
        is_right_closed = right_ear < closure_threshold
        # Binocular closure when compensated EAR is below the closure threshold
        is_closed = compensated < closure_threshold

    return EyeAspectRatioResult(
        left_ear=left_ear,
        right_ear=right_ear,
        weighted_ear=weighted,
        compensated_ear=compensated,
        closure_threshold=closure_threshold,
        is_left_closed=is_left_closed,
        is_right_closed=is_right_closed,
        is_closed=is_closed,
        is_valid=is_valid,
    )
