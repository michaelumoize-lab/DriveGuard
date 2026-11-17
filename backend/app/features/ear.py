"""
DriveGuard — Eye Aspect Ratio (EAR) Feature Extraction Module (Phase 5).

1. What it does:
   Calculates the Eye Aspect Ratio (EAR) independently for left and right eyes
   from 2D/3D facial landmarks. Computes C1-continuous perspective-weighted EAR
   conditioned on head yaw angle, maintains raw uncompensated measurements,
   and provides safe pitch compensation for open-eye downward glances without
   masking genuine eye closures.

2. Why it exists:
   EAR is the primary geometric metric for detecting eyelid closure and blinks (Soukupova & Cech, 2016).
   Decoupling left and right eyes allows asymmetrical closure detection (winks vs blinks,
   unilateral occlusions, ptosis), while continuous perspective weighting prevents false
   closures when the head turns to check mirrors.

3. Coordinate-System Contract:
   - Explicit configuration (`is_normalized=True` or `False`) is the preferred contract.
   - Normalized Coordinates (`is_normalized=True`, default):
     x and y are in [0, 1] relative to image width and height. Horizontal distances (dx)
     are scaled by `aspect_ratio = frame_width / frame_height` to ensure isotropic
     Euclidean distance calculation in non-square camera frames.
   - Pixel Coordinates (`is_normalized=False`):
     x and y are in isotropic pixel units (e.g. [0, W] and [0, H]). No aspect ratio
     multiplier is applied (dx = x2 - x1).
   - Automatic Detection Fallback (`is_normalized=None`):
     Examines strictly the planar (x, y) coordinates of the eye landmarks (never the z coordinate).
     If any coordinate exceeds 2.0, pixel space is inferred; otherwise normalized space is assumed.

4. Yaw Coordinate Convention & Perspective Geometry:
   - Looking forward: `yaw_deg = 0.0` (50% left_ear, 50% right_ear).
   - Positive yaw (> 0°, driver turned to driver's left, e.g. checking left mirror):
     The driver's right eye (MediaPipe landmark 33 / camera image left / `left_ear`)
     directly faces the camera lens with full openness, while the driver's left eye
     (MediaPipe landmark 362 / camera image right / `right_ear`) turns away and foreshortens.
     Therefore, positive yaw smoothly shifts weighting toward `left_ear` (up to 80% at +30°).
   - Negative yaw (< 0°, driver turned to driver's right, e.g. checking right mirror):
     The driver's left eye (MediaPipe landmark 362 / camera image right / `right_ear`)
     directly faces the camera lens.
     Therefore, negative yaw smoothly shifts weighting toward `right_ear` (up to 80% at -30°).
   - Camera Mirroring (`is_mirrored=True`):
     When the camera frame is horizontally mirrored (selfie mode), image coordinates are
     reflected, so the camera-facing eye appears at image right (`right_ear`).
     Inverting the effective yaw (`-yaw_deg`) ensures the camera-facing eye receives the higher weight.
   - Smoothstep Transition:
     Weights follow a C1-continuous Hermite smoothstep curve over [-30°, +30°], ensuring both
     the weight and its derivative are continuous with zero derivative at the ±30° boundaries.

5. What data it receives:
   - Landmarks: Sequence of `NormalizedLandmark` or NumPy array of shape (N, 2) or (N, 3).
   - Head orientation angles (yaw and pitch in degrees).
   - Optional aspect ratio, closure threshold, and coordinate space indicators.

6. What data it produces:
   - `EyeAspectRatioResult` dataclass containing:
     - `left_ear`: EAR for left landmark eye contour (MediaPipe idx 33 / camera left).
     - `right_ear`: EAR for right landmark eye contour (MediaPipe idx 362 / camera right).
     - `weighted_ear`: C1-continuous perspective-weighted EAR based on yaw angle and eye validity.
     - `compensated_ear`: Experimental pitch-compensated EAR (only active for open eyes during downward pitch).
     - `closure_threshold`: Effective EAR closure threshold (nominal 0.20 or personalized baseline).
     - `is_left_closed`: Boolean closure state of left eye.
     - `is_right_closed`: Boolean closure state of right eye.
     - `is_closed`: Binocular closure state (both valid eyes closed, or single valid eye closed).
     - `is_valid`: Boolean flag (True if at least one eye is measurable).
     - `raw_ear`: Unweighted, uncompensated arithmetic mean of all valid eyes.
     - `is_left_valid`: Boolean validity of left eye landmarks.
     - `is_right_valid`: Boolean validity of right eye landmarks.

7. Safety & Validity Contract:
   - A genuinely closed eye can have a very low EAR (e.g. 0.01 - 0.08) and is valid as long as
     landmark geometry is sound (finite coordinates, non-collapsed eye contour width).
   - If one eye is occluded or out-of-frame, the valid eye is NOT invalidated or halved;
     its measurement is preserved with 100% weight.
   - Pitch compensation NEVER boosts an EAR that is at or below the closure threshold.
   - Closure classification (`is_closed`) uses strictly uncompensated EAR.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from backend.app.vision.face_landmarker import NormalizedLandmark

logger = logging.getLogger(__name__)

# MediaPipe 6-point canonical eye contour indices
# Format: (outer_corner, top_1, top_2, inner_corner, bottom_2, bottom_1)
LEFT_EYE_INDICES = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_INDICES = (362, 385, 387, 263, 373, 380)

# Default nominal geometric closure threshold.
# NOTE: 0.20 is a generic population heuristic (Soukupova & Cech, 2016), NOT a direct measurement
# of 80% eyelid coverage. True physiological P80 requires driver-specific calibration (PDBP):
# EAR_P80 = 0.20 * EAR_open + 0.80 * EAR_closed. Nominal 0.20 serves as the uncalibrated surrogate.
DEFAULT_EAR_CLOSURE_THRESHOLD = 0.20


@dataclass(frozen=True)
class EyeAspectRatioResult:
    """
    Immutable structured evaluation of eye opening measurements.
    Preserves raw measurements, independent eye validity, and uncompensated closure states.
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
    raw_ear: float = 0.0
    is_left_valid: bool = True
    is_right_valid: bool = True


def calculate_single_eye_ear_with_validity(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    indices: Tuple[int, int, int, int, int, int],
    aspect_ratio: float = 16.0 / 9.0,
    is_normalized: Optional[bool] = None,
    eps: float = 1e-6,
) -> Tuple[float, bool]:
    """
    Compute Eye Aspect Ratio for a single eye and evaluate landmark validity.

    EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4|| + eps)

    Args:
        landmarks: Sequence of NormalizedLandmark or NumPy array (N, 2) or (N, 3).
        indices: 6 landmark indices for the eye contour.
        aspect_ratio: Frame width/height ratio used to correct pixel aspect ratio in normalized space.
        is_normalized: True if coordinates are in [0, 1], False if in pixel space. Auto-detected if None.
        eps: Small epsilon preventing division by zero.

    Returns:
        Tuple of (ear_value: float, is_valid: bool).
        A genuinely closed eye with small vertical heights returns a small EAR and is_valid=True.
        Malformed, collapsed, non-finite, or missing landmarks return (0.0, False).
    """
    if len(landmarks) <= max(indices):
        return 0.0, False

    try:
        if isinstance(landmarks, np.ndarray):
            pts = landmarks[list(indices), :2]
            p = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(6)]
        else:
            p = [(float(landmarks[i].x), float(landmarks[i].y)) for i in indices]
    except (IndexError, AttributeError, ValueError, TypeError):
        return 0.0, False

    # 1. Finite coordinate check
    for x, y in p:
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0, False

    # 2. Coordinate space determination
    # Explicit configuration (is_normalized=True or False) is the preferred contract.
    # Fallback auto-detection examines strictly horizontal (x) and vertical (y) bounds, never z.
    if is_normalized is None:
        max_coord = max(max(abs(x), abs(y)) for x, y in p)
        norm_mode = max_coord <= 2.0
    else:
        norm_mode = is_normalized

    ar_factor = float(aspect_ratio) if norm_mode else 1.0

    p1, p2, p3, p4, p5, p6 = p[0], p[1], p[2], p[3], p[4], p[5]

    # Horizontal eye width (between outer and inner corners p1 and p4)
    horiz = math.hypot((p1[0] - p4[0]) * ar_factor, p1[1] - p4[1])

    # Eye contour must have measurable horizontal width (not collapsed to a single point)
    if horiz < eps:
        return 0.0, False

    # Vertical eyelid spans
    v1 = math.hypot((p2[0] - p6[0]) * ar_factor, p2[1] - p6[1])
    v2 = math.hypot((p3[0] - p5[0]) * ar_factor, p3[1] - p5[1])

    ear = float((v1 + v2) / (2.0 * horiz + eps))

    # Plausibility check: EAR cannot be negative or anatomically impossible
    if ear < 0.0 or ear > 1.5:
        return 0.0, False

    return ear, True


def calculate_single_eye_ear(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    indices: Tuple[int, int, int, int, int, int],
    aspect_ratio: float = 16.0 / 9.0,
    is_normalized: Optional[bool] = None,
    eps: float = 1e-6,
) -> float:
    """
    Compute Eye Aspect Ratio for a single eye using 6 eyelid landmarks.

    Maintains backward compatibility by returning float scalar.
    """
    ear, _ = calculate_single_eye_ear_with_validity(
        landmarks=landmarks,
        indices=indices,
        aspect_ratio=aspect_ratio,
        is_normalized=is_normalized,
        eps=eps,
    )
    return ear


def calculate_yaw_weighted_ear(
    left_ear: float,
    right_ear: float,
    yaw_deg: float,
    left_valid: bool = True,
    right_valid: bool = True,
    is_mirrored: bool = False,
) -> float:
    """
    Combine left and right EAR with smoothstep (C1-continuous) yaw perspective weighting.

    Coordinate & Anatomical Mapping:
    - left_ear is computed from LEFT_EYE_INDICES (MediaPipe landmark 33), which is anatomically
      the driver's RIGHT eye (located on the left side of the camera image).
    - right_ear is computed from RIGHT_EYE_INDICES (MediaPipe landmark 362), which is anatomically
      the driver's LEFT eye (located on the right side of the camera image).

    Perspective Geometry & Coordinate Convention:
    - Yaw = 0.0° (neutral forward): 50% left_ear, 50% right_ear.
    - Positive yaw (driver turned left, e.g. checking left mirror):
      The driver's right eye (left_ear / idx 33) directly faces the camera lens with full openness,
      while the driver's left eye (right_ear / idx 362) turns away and is foreshortened.
      Weight smoothly ramps toward left_ear, reaching 80% at +30.0° and above.
    - Negative yaw (driver turned right, e.g. checking right mirror):
      The driver's left eye (right_ear / idx 362) faces the camera lens.
      Weight smoothly ramps toward right_ear, reaching 80% at -30.0° and below.
    - Mirrored View:
      If is_mirrored is True, the image is horizontally reflected so that the camera-facing eye
      appears on the opposite side of the frame; effective yaw is inverted to maintain physical accuracy.
    - Smoothstep Transition:
      Uses a cubic Hermite polynomial S(u) = 3u^2 - 2u^3 mapped over [-30°, +30°].
      Both the weighting value and its derivative are continuous (C1 continuity), with zero
      derivative at the ±30° boundaries to eliminate velocity artifacts.
    - Unilateral Fallback:
      If only one eye is valid (e.g. unilateral occlusion), the valid eye receives 100% weight.
    - If neither eye is valid, returns 0.0.
    """
    if not left_valid and not right_valid:
        return 0.0
    if left_valid and not right_valid:
        return float(left_ear)
    if right_valid and not left_valid:
        return float(right_ear)

    # In mirrored mode, the camera-facing eye is reflected across the horizontal axis
    effective_yaw = -float(yaw_deg) if is_mirrored else float(yaw_deg)

    # Both eyes valid: smoothstep interpolation over [-30°, +30°]
    yaw_clamped = max(-30.0, min(30.0, effective_yaw))
    u = (yaw_clamped + 30.0) / 60.0  # normalized to [0.0, 1.0]

    # Hermite smoothstep: S(u) = 3u^2 - 2u^3
    # S(0) = 0, S(0.5) = 0.5, S(1) = 1, S'(0) = 0, S'(1) = 0
    s_u = u * u * (3.0 - 2.0 * u)

    # Bounded between 20% and 80%
    w_left = 0.20 + 0.60 * s_u
    w_right = 1.0 - w_left

    return float(w_left * left_ear + w_right * right_ear)


def calculate_pitch_compensated_ear(
    base_ear: float,
    pitch_deg: float,
    neutral_pitch_deg: float = 0.0,
    closure_threshold: float = DEFAULT_EAR_CLOSURE_THRESHOLD,
) -> float:
    """
    Compensate EAR when the driver glances downward towards the instrument cluster/dashboard.

    Safeguards:
    - Only applies when base_ear is strictly above the closure threshold. If the eye is closed
      or near the closure threshold, pitch compensation is INACTIVE (factor = 1.0).
      This prevents downward head nodding during microsleeps or genuine fatigue from being
      artificially boosted into an open-eye measurement.
    - Preserved independently for open-eye baseline tracking; does NOT determine closure state.
    """
    if base_ear <= 0.0 or base_ear <= closure_threshold:
        return float(base_ear)

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
    is_normalized: Optional[bool] = None,
    is_mirrored: bool = False,
) -> EyeAspectRatioResult:
    """
    Complete EAR pipeline extracting left, right, raw, yaw-weighted, and pitch-compensated EAR.

    Closure Classification:
    - Uses uncompensated EAR metrics.
    - When both eyes are valid: is_left_closed = left_ear < closure_threshold,
      is_right_closed = right_ear < closure_threshold,
      is_closed = is_left_closed and is_right_closed (binocular closure).
    - When only one eye is valid: uses the valid eye's closure state for is_closed.
    - When neither eye is valid: is_closed = False, is_valid = False.
    """
    left_ear, left_valid = calculate_single_eye_ear_with_validity(
        landmarks, LEFT_EYE_INDICES, aspect_ratio=aspect_ratio, is_normalized=is_normalized
    )
    right_ear, right_valid = calculate_single_eye_ear_with_validity(
        landmarks, RIGHT_EYE_INDICES, aspect_ratio=aspect_ratio, is_normalized=is_normalized
    )

    is_valid = left_valid or right_valid

    # Raw uncompensated arithmetic mean of valid eyes
    if left_valid and right_valid:
        raw_ear = (left_ear + right_ear) / 2.0
    elif left_valid:
        raw_ear = left_ear
    elif right_valid:
        raw_ear = right_ear
    else:
        raw_ear = 0.0

    weighted = calculate_yaw_weighted_ear(
        left_ear=left_ear,
        right_ear=right_ear,
        yaw_deg=yaw_deg,
        left_valid=left_valid,
        right_valid=right_valid,
        is_mirrored=is_mirrored,
    )

    compensated = calculate_pitch_compensated_ear(
        base_ear=weighted,
        pitch_deg=pitch_deg,
        neutral_pitch_deg=0.0,
        closure_threshold=closure_threshold,
    )

    # Independent closure states for each eye
    is_left_closed = bool(left_valid and (left_ear < closure_threshold))
    is_right_closed = bool(right_valid and (right_ear < closure_threshold))

    # Overall closure: binocular closure if both eyes valid, or single-eye closure if one eye valid
    if left_valid and right_valid:
        is_closed = is_left_closed and is_right_closed
    elif left_valid:
        is_closed = is_left_closed
    elif right_valid:
        is_closed = is_right_closed
    else:
        is_closed = False

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
        raw_ear=raw_ear,
        is_left_valid=left_valid,
        is_right_valid=right_valid,
    )
