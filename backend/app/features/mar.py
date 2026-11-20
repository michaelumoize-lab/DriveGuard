"""
DriveGuard — Mouth Aspect Ratio (MAR) & Yawn Detection Module (Phase 5).

1. What it does:
   Calculates the Mouth Aspect Ratio (MAR) from 2D facial landmarks and tracks
   sustained mouth-opening episodes over time. Employs dual-threshold Schmitt trigger
   hysteresis (open threshold 0.55, hold threshold 0.45) to prevent boundary flutter from
   lip tremor, tracks baseline closed-mouth ratios, and incorporates a 200 ms dropout grace
   period for transient tracking loss without crediting unobserved time to mouth opening.

2. Why it exists:
   Yawning is an early behavioral indicator of drowsiness. However, ordinary talking, singing,
   and verbal reactions frequently cause temporary mouth openings.
   IMPORTANT SCIENTIFIC DISTINCTION:
   MAR is a mouth-opening measurement, not a definitive yawn detector. A large mouth opening
   sustained for >= 1500 ms is evidence of a possible candidate yawn, not clinical proof of fatigue.
   The `is_authentic_yawn` field is retained as a backward-compatible indicator for sustained
   candidate openings, while definitive drowsiness decisions are reserved for downstream
   multi-factor fusion (risk engine).

3. Coordinate-System Contract:
   - Explicit configuration (`is_normalized=True` or `False`) is the preferred contract.
   - Normalized Coordinates (`is_normalized=True`, default):
     x and y are in [0, 1] relative to image width and height. Horizontal distances (dx)
     are scaled by `aspect_ratio = frame_width / frame_height` to ensure isotropic
     Euclidean distance calculation in non-square camera frames.
   - Pixel Coordinates (`is_normalized=False`):
     x and y are in isotropic pixel units. No aspect ratio multiplier is applied (dx = x2 - x1).
   - Automatic Detection Fallback (`is_normalized=None`):
     Examines strictly the planar (x, y) coordinates of the mouth landmarks (never z).
     If any coordinate exceeds 2.0, pixel space is inferred; otherwise normalized space is assumed.

4. Temporal Dynamics & Tracking Grace:
   - Hysteresis: An episode initiates when MAR >= 0.55 and remains active as long as MAR >= 0.45.
   - Dropout Grace: Temporary invalid observations (< 200 ms) maintain an in-flight candidate
     without corrupting state, but unobserved time is NOT added to measured open-mouth duration.
   - Prolonged Gap: Tracking gaps exceeding 500 ms cleanly terminate any in-flight candidate.
   - Timestamp Validation: Non-finite, negative, duplicate, or backwards timestamps are rejected
     without corrupting previous valid detector state.

5. Baseline & Expansion Tracking:
   - Tracks rolling resting baseline (`baseline_mar`) strictly from stable, valid, closed-mouth
     observations (MAR <= 0.28) using exponential smoothing.
   - Does NOT update baseline during active openings, tracking dropouts, or significant head turns.
   - Exposes `opening_ratio = mar / baseline_mar` for future personalized calibration.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

from backend.app.vision.face_landmarker import NormalizedLandmark

logger = logging.getLogger(__name__)

# MediaPipe Canonical Mouth Landmark Indices
# Outer lip corners: (left_corner, right_corner)
MOUTH_OUTER_CORNERS = (61, 291)

# Inner lip vertical pairs: ((top_center, bottom_center), (top_left, bottom_left), (top_right, bottom_right))
MOUTH_VERTICAL_PAIRS = ((13, 14), (81, 178), (311, 402))

# Default Hysteresis Thresholds
DEFAULT_MAR_OPEN_THRESHOLD: float = 0.55        # Mouth opening onset trigger threshold
DEFAULT_MAR_HOLD_THRESHOLD: float = 0.45        # Sustained opening floor threshold (hysteresis)

# Default Duration Criteria
DEFAULT_MIN_YAWN_DURATION_MS: float = 1500.0    # Baseline minimum duration for candidate yawn event
DEFAULT_MAX_YAWN_DURATION_MS: float = 8000.0    # Maximum realistic single yawn duration (8s)

# Tracking Gap & Dropout Thresholds
DEFAULT_DROPOUT_GRACE_MS: float = 200.0         # Maximum tolerance for transient landmark loss
DEFAULT_MAX_SAMPLING_GAP_MS: float = 250.0      # Inter-frame gap threshold triggering event reset (aligned with Blink/PERCLOS/Temporal)

# Baseline Parameters
DEFAULT_CLOSED_BASELINE_THRESHOLD: float = 0.28  # Maximum MAR to qualify as closed for baseline update
DEFAULT_RESTING_BASELINE_MAR: float = 0.15      # Default nominal resting baseline MAR
DEFAULT_BASELINE_ALPHA: float = 0.02            # Exponential smoothing factor for baseline tracking


@dataclass(frozen=True)
class YawnEvent:
    """
    Record of a completed candidate mouth opening (yawn candidate event).
    Preserves all existing fields and positional order for backward compatibility.
    """
    start_ts_ms: float
    end_ts_ms: float
    duration_ms: float
    peak_mar: float
    observed_duration_ms: float = 0.0
    wall_clock_duration_ms: float = 0.0
    mean_mar: float = 0.0
    opening_ratio: float = 1.0


@dataclass(frozen=True)
class MouthAspectRatioResult:
    """
    Per-frame structured evaluation of mouth opening and candidate yawn state.
    Preserves all existing fields and positional order for backward compatibility.
    """
    mar: float
    is_mouth_open: bool
    is_yawn_in_progress: bool
    current_yawn_duration_ms: float
    is_authentic_yawn: bool
    completed_yawn: Optional[YawnEvent]
    total_yawn_count: int
    is_valid: bool = True
    baseline_mar: float = DEFAULT_RESTING_BASELINE_MAR
    opening_ratio: float = 1.0
    peak_mar: float = 0.0
    observed_yawn_duration_ms: float = 0.0


def calculate_mar_with_validity(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    aspect_ratio: float = 16.0 / 9.0,
    is_normalized: Optional[bool] = None,
    eps: float = 1e-6,
) -> Tuple[float, bool]:
    """
    Compute Mouth Aspect Ratio (MAR) and evaluate geometric landmark validity.

    MAR = (mean vertical inner-lip distance) / (outer corner horizontal width + eps)

    Args:
        landmarks: Sequence of NormalizedLandmark or NumPy array (N, 2) or (N, 3).
        aspect_ratio: Frame width/height ratio used to correct non-square normalized space.
        is_normalized: True if coordinates are in [0, 1], False if in pixel space. Auto-detected if None.
        eps: Small epsilon preventing division by zero.

    Returns:
        Tuple of (mar_value: float, is_valid: bool).
        Malformed, collapsed, non-finite, or missing landmarks return (0.0, False).
    """
    all_needed_indices = (
        MOUTH_OUTER_CORNERS[0],
        MOUTH_OUTER_CORNERS[1],
        MOUTH_VERTICAL_PAIRS[0][0],
        MOUTH_VERTICAL_PAIRS[0][1],
        MOUTH_VERTICAL_PAIRS[1][0],
        MOUTH_VERTICAL_PAIRS[1][1],
        MOUTH_VERTICAL_PAIRS[2][0],
        MOUTH_VERTICAL_PAIRS[2][1],
    )
    if len(landmarks) <= max(all_needed_indices):
        return 0.0, False

    try:
        if isinstance(landmarks, np.ndarray):
            pts = landmarks[list(all_needed_indices), :2]
            coords = [(float(pts[i, 0]), float(pts[i, 1])) for i in range(len(all_needed_indices))]
        else:
            coords = [(float(landmarks[idx].x), float(landmarks[idx].y)) for idx in all_needed_indices]
    except (IndexError, AttributeError, ValueError, TypeError):
        return 0.0, False

    # 1. Finite coordinate check
    for x, y in coords:
        if not (math.isfinite(x) and math.isfinite(y)):
            return 0.0, False

    # 2. Coordinate space determination
    # Explicit configuration (is_normalized=True or False) is preferred.
    # Fallback auto-detection inspects strictly planar (x, y) coordinates, never z.
    if is_normalized is None:
        max_coord = max(max(abs(x), abs(y)) for x, y in coords)
        norm_mode = max_coord <= 2.0
    else:
        norm_mode = is_normalized

    ar_factor = float(aspect_ratio) if norm_mode else 1.0

    # Horizontal mouth width (between outer lip corners 61 and 291)
    p_left = coords[0]
    p_right = coords[1]
    horiz = math.hypot((p_left[0] - p_right[0]) * ar_factor, p_left[1] - p_right[1])
    if horiz < eps:
        return 0.0, False

    # Vertical inner-lip distances (3 canonical pairs)
    vert_sum = 0.0
    for i in range(len(MOUTH_VERTICAL_PAIRS)):
        p_top = coords[2 + 2 * i]
        p_bot = coords[3 + 2 * i]
        v = math.hypot((p_top[0] - p_bot[0]) * ar_factor, p_top[1] - p_bot[1])
        vert_sum += v

    mean_vert = vert_sum / float(len(MOUTH_VERTICAL_PAIRS))
    mar = float(mean_vert / (horiz + eps))

    # Plausibility check: MAR cannot be negative or anatomically absurd (> 2.5)
    if mar < 0.0 or mar > 2.5:
        return 0.0, False

    return mar, True


def calculate_mar(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    aspect_ratio: float = 16.0 / 9.0,
    is_normalized: Optional[bool] = None,
    eps: float = 1e-6,
) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) using 3 vertical inner-lip pairs and outer corners:
    MAR = (mean vertical distance) / (horizontal width + eps)

    Maintains backward compatibility by returning float scalar.
    """
    mar, _ = calculate_mar_with_validity(
        landmarks=landmarks,
        aspect_ratio=aspect_ratio,
        is_normalized=is_normalized,
        eps=eps,
    )
    return mar


class YawnDetector:
    """
    Temporal tracker that accumulates sustained mouth opening over time to identify
    candidate yawn events while rejecting brief speech bursts and singing.

    Employs dual-threshold hysteresis, tracks resting closed-mouth baseline,
    handles transient tracking dropouts with a 200ms grace window, and enforces
    strict timestamp monotonicity.
    """

    def __init__(
        self,
        mar_open_threshold: float = DEFAULT_MAR_OPEN_THRESHOLD,
        min_yawn_duration_ms: float = DEFAULT_MIN_YAWN_DURATION_MS,
        max_yawn_duration_ms: float = DEFAULT_MAX_YAWN_DURATION_MS,
        mar_hold_threshold: float = DEFAULT_MAR_HOLD_THRESHOLD,
        dropout_grace_ms: float = DEFAULT_DROPOUT_GRACE_MS,
        max_sampling_gap_ms: float = DEFAULT_MAX_SAMPLING_GAP_MS,
        closed_baseline_threshold: float = DEFAULT_CLOSED_BASELINE_THRESHOLD,
        baseline_alpha: float = DEFAULT_BASELINE_ALPHA,
    ) -> None:
        """
        Initialize the temporal mouth-opening tracker.

        Args:
            mar_open_threshold: MAR threshold to trigger an opening episode (default: 0.55).
            min_yawn_duration_ms: Minimum duration for candidate yawn event (default: 1500.0 ms).
            max_yawn_duration_ms: Maximum realistic yawn duration (default: 8000.0 ms).
            mar_hold_threshold: MAR threshold to sustain an opening episode (default: 0.45).
            dropout_grace_ms: Grace window for transient tracking loss (default: 200.0 ms).
            max_sampling_gap_ms: Gap threshold triggering episode cancellation (default: 500.0 ms).
            closed_baseline_threshold: Upper limit for closed-mouth baseline updates (default: 0.28).
            baseline_alpha: Exponential smoothing factor for baseline tracking (default: 0.02).
        """
        self.mar_open_threshold = float(mar_open_threshold)
        self.min_yawn_duration_ms = float(min_yawn_duration_ms)
        self.max_yawn_duration_ms = float(max_yawn_duration_ms)
        self.mar_hold_threshold = float(mar_hold_threshold)
        self.dropout_grace_ms = float(dropout_grace_ms)
        self.max_sampling_gap_ms = float(max_sampling_gap_ms)
        self.closed_baseline_threshold = float(closed_baseline_threshold)
        self.baseline_alpha = float(baseline_alpha)

        # In-flight state tracking
        self._is_opening: bool = False
        self._opening_exceeded_max: bool = False
        self._opening_start_ts: Optional[float] = None
        self._last_open_valid_ts: Optional[float] = None
        self._observed_open_ms: float = 0.0
        self._open_mar_samples: List[float] = []
        self._peak_mar: float = 0.0
        self._total_yawns: int = 0

        # Monotonic timestamp and dropout tracking
        self._last_valid_ts: Optional[float] = None
        self._dropout_start_ts: Optional[float] = None

        # Rolling baseline tracking
        self._baseline_mar: float = DEFAULT_RESTING_BASELINE_MAR

    @property
    def baseline_mar(self) -> float:
        """Current resting closed-mouth baseline estimate."""
        return self._baseline_mar

    def update(
        self,
        landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
        timestamp_ms: float,
        aspect_ratio: float = 16.0 / 9.0,
        has_face: bool = True,
        is_normalized: Optional[bool] = None,
        is_looking_away: Optional[bool] = None,
    ) -> MouthAspectRatioResult:
        """
        Evaluate frame landmarks and update temporal mouth opening state.

        Args:
            landmarks: Sequence of NormalizedLandmark or NumPy array.
            timestamp_ms: Monotonic frame timestamp in milliseconds.
            aspect_ratio: Video frame aspect ratio (width / height).
            has_face: True if face landmarker locked onto a face.
            is_normalized: True if normalized coords, False if pixel coords, None for auto-detect.

        Returns:
            MouthAspectRatioResult describing current MAR, aperture state, and candidate yawn events.
        """
        # 1. Monotonic Timestamp Validation
        ts = float(timestamp_ms)
        if not math.isfinite(ts) or ts < 0.0:
            logger.warning("YawnDetector: rejected non-finite or negative timestamp %s", timestamp_ms)
            return self._build_result(
                mar=0.0,
                is_mouth_open=False,
                is_valid=False,
                current_ts=self._last_valid_ts or 0.0,
            )

        if self._last_valid_ts is not None:
            dt = ts - self._last_valid_ts
            if dt <= 0.0:
                logger.warning(
                    "YawnDetector: rejected non-monotonic timestamp %s <= last %s",
                    ts,
                    self._last_valid_ts,
                )
                return self._build_result(
                    mar=0.0,
                    is_mouth_open=False,
                    is_valid=False,
                    current_ts=self._last_valid_ts,
                )

            # Sampling gap check (> 500 ms): reset any in-flight opening as unobserved
            if dt > self.max_sampling_gap_ms:
                self._abort_opening()
                self._dropout_start_ts = None

        # 2. Geometric Validity & Face Tracking Check
        mar = 0.0
        is_valid = False
        if has_face and len(landmarks) > 0:
            mar, is_valid = calculate_mar_with_validity(
                landmarks=landmarks,
                aspect_ratio=aspect_ratio,
                is_normalized=is_normalized,
            )

        # 3. Handle Invalid Observation or Tracking Dropout
        if not is_valid:
            if self._is_opening:
                # Active opening in flight: check dropout grace period
                if self._dropout_start_ts is None:
                    self._dropout_start_ts = ts

                dropout_elapsed = ts - self._dropout_start_ts
                if dropout_elapsed <= self.dropout_grace_ms:
                    # Within grace period: hold in-flight state without adding unobserved time
                    self._last_valid_ts = ts
                    return self._build_result(
                        mar=0.0,
                        is_mouth_open=False,
                        is_valid=False,
                        current_ts=ts,
                    )
                else:
                    # Grace period exceeded: cancel in-flight episode cleanly
                    self._abort_opening()
                    self._dropout_start_ts = None

            self._last_valid_ts = ts
            return self._build_result(
                mar=0.0,
                is_mouth_open=False,
                is_valid=False,
                current_ts=ts,
            )

        # Observation is valid: check if recovering from dropout
        was_in_dropout = False
        if self._dropout_start_ts is not None:
            if (ts - self._dropout_start_ts) > self.dropout_grace_ms:
                self._abort_opening()
            else:
                was_in_dropout = True
            self._dropout_start_ts = None

        # 4. Temporal Dynamics & Hysteresis State Machine
        completed_yawn: Optional[YawnEvent] = None
        is_mouth_open = False

        if not self._is_opening:
            # Check opening onset trigger (open threshold: 0.55)
            if mar >= self.mar_open_threshold:
                self._is_opening = True
                self._opening_start_ts = ts
                self._last_open_valid_ts = ts
                self._observed_open_ms = 0.0
                self._open_mar_samples = [mar]
                self._peak_mar = mar
                is_mouth_open = True
            else:
                is_mouth_open = False
                # Closed/stable mouth: update resting baseline
                # Exclude active head turns if is_looking_away is provided
                if mar <= self.closed_baseline_threshold and not (is_looking_away is True):
                    self._baseline_mar = (
                        (1.0 - self.baseline_alpha) * self._baseline_mar
                        + self.baseline_alpha * mar
                    )
                    self._baseline_mar = max(0.05, min(0.35, self._baseline_mar))
        else:
            # Opening is in flight: evaluate sustained hold floor (hold threshold: 0.45)
            if mar >= self.mar_hold_threshold:
                is_mouth_open = True
                if self._last_open_valid_ts is not None and not was_in_dropout:
                    step_open_dt = ts - self._last_open_valid_ts
                    self._observed_open_ms += step_open_dt

                self._last_open_valid_ts = ts
                self._peak_mar = max(self._peak_mar, mar)
                self._open_mar_samples.append(mar)
                if self._observed_open_ms > self.max_yawn_duration_ms:
                    self._opening_exceeded_max = True
            else:
                # Mouth has dropped below hold floor (episode terminates)
                is_mouth_open = False
                if self._opening_start_ts is not None:
                    wall_clock_duration = ts - self._opening_start_ts
                    observed_duration = self._observed_open_ms

                    # Check candidate yawn duration criteria (1500ms <= duration <= 8000ms)
                    if self.min_yawn_duration_ms <= observed_duration <= self.max_yawn_duration_ms:
                        mean_mar = (
                            sum(self._open_mar_samples) / float(len(self._open_mar_samples))
                            if self._open_mar_samples
                            else self._peak_mar
                        )
                        completed_yawn = YawnEvent(
                            start_ts_ms=self._opening_start_ts,
                            end_ts_ms=ts,
                            duration_ms=observed_duration,
                            peak_mar=self._peak_mar,
                            observed_duration_ms=observed_duration,
                            wall_clock_duration_ms=wall_clock_duration,
                            mean_mar=mean_mar,
                            opening_ratio=self._peak_mar / max(self._baseline_mar, 1e-6),
                        )
                        self._total_yawns += 1

                self._abort_opening()

        self._last_valid_ts = ts

        return self._build_result(
            mar=mar,
            is_mouth_open=is_mouth_open,
            is_valid=True,
            current_ts=ts,
            completed_yawn=completed_yawn,
        )

    def _abort_opening(self) -> None:
        """Reset active opening tracking state."""
        self._is_opening = False
        self._opening_exceeded_max = False
        self._opening_start_ts = None
        self._last_open_valid_ts = None
        self._observed_open_ms = 0.0
        self._open_mar_samples = []
        self._peak_mar = 0.0

    def _build_result(
        self,
        mar: float,
        is_mouth_open: bool,
        is_valid: bool,
        current_ts: float,
        completed_yawn: Optional[YawnEvent] = None,
    ) -> MouthAspectRatioResult:
        """Construct structured MouthAspectRatioResult maintaining field order."""
        if (
            is_valid
            and self._is_opening
            and not self._opening_exceeded_max
            and self._opening_start_ts is not None
        ):
            wall_clock_duration = current_ts - self._opening_start_ts
            observed_duration = self._observed_open_ms
        else:
            wall_clock_duration = 0.0
            observed_duration = 0.0

        active_within_bounds = (
            is_valid
            and self._is_opening
            and not self._opening_exceeded_max
            and observed_duration <= self.max_yawn_duration_ms
        )
        is_yawn_in_progress = bool(
            active_within_bounds and (observed_duration >= (self.min_yawn_duration_ms * 0.5))
        )
        is_authentic_yawn = bool(
            active_within_bounds and (observed_duration >= self.min_yawn_duration_ms)
        )
        opening_ratio = mar / max(self._baseline_mar, 1e-6)

        return MouthAspectRatioResult(
            mar=mar,
            is_mouth_open=is_mouth_open,
            is_yawn_in_progress=is_yawn_in_progress,
            current_yawn_duration_ms=wall_clock_duration,
            is_authentic_yawn=is_authentic_yawn,
            completed_yawn=completed_yawn,
            total_yawn_count=self._total_yawns,
            is_valid=is_valid,
            baseline_mar=self._baseline_mar,
            opening_ratio=opening_ratio,
            peak_mar=self._peak_mar if (is_valid and self._is_opening) else mar,
            observed_yawn_duration_ms=observed_duration,
        )

    def reset(self) -> None:
        """Reset temporal state for a new driving session."""
        self._abort_opening()
        self._total_yawns = 0
        self._last_valid_ts = None
        self._dropout_start_ts = None
        self._baseline_mar = DEFAULT_RESTING_BASELINE_MAR
