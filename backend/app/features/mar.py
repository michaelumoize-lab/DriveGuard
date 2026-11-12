"""
DriveGuard — Mouth Aspect Ratio (MAR) & Yawn Detection Module (Phase 5).

1. What it does:
   Calculates the Mouth Aspect Ratio (MAR) from facial landmarks and tracks
   yawn events over time, distinguishing authentic prolonged yawns from brief speech.

2. Why it exists:
   Yawning is an early behavioral indicator of fatigue. Normal conversation creates
   brief mouth openings (< 800 ms). An authentic yawn requires continuous opening
   exceeding the MAR threshold for >= 1.5 seconds.

3. What data it receives:
   - Sequence of `NormalizedLandmark` points.
   - Monotonic frame timestamp in milliseconds.

4. What data it produces:
   - `MouthAspectRatioResult` dataclass with `mar`, `is_mouth_open`,
     `is_yawn_in_progress`, `yawn_duration_ms`, `is_authentic_yawn`, and completed `yawn_count`.

5. Modules that depend on it:
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine (Phase 6)

6. Failure behavior:
   - Returns MAR = 0.0 with `is_mouth_open=False` on missing or degenerate landmarks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import numpy as np

from backend.app.vision.face_landmarker import NormalizedLandmark

# MediaPipe Mouth Landmark Indices
MOUTH_OUTER_CORNERS = (61, 291)
MOUTH_VERTICAL_PAIRS = ((13, 14), (81, 178), (311, 402))

# Thresholds
DEFAULT_MAR_OPEN_THRESHOLD = 0.55      # Above this, mouth is significantly open
DEFAULT_MIN_YAWN_DURATION_MS = 1500.0  # Must be open >= 1.5s to distinguish from speech
DEFAULT_MAX_YAWN_DURATION_MS = 8000.0  # Maximum realistic single yawn duration (8s)


@dataclass(frozen=True)
class YawnEvent:
    """
    Record of a completed authentic yawn.
    """
    start_ts_ms: float
    end_ts_ms: float
    duration_ms: float
    peak_mar: float


@dataclass(frozen=True)
class MouthAspectRatioResult:
    """
    Per-frame evaluation of mouth opening and yawn state.
    """
    mar: float
    is_mouth_open: bool
    is_yawn_in_progress: bool
    current_yawn_duration_ms: float
    is_authentic_yawn: bool
    completed_yawn: Optional[YawnEvent]
    total_yawn_count: int


def calculate_mar(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    aspect_ratio: float = 16.0 / 9.0,
    eps: float = 1e-6,
) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) using 3 vertical inner-lip pairs and outer corners:
    MAR = (mean vertical distance) / (horizontal width + eps)
    """
    needed_indices = [MOUTH_OUTER_CORNERS[0], MOUTH_OUTER_CORNERS[1]]
    for t_idx, b_idx in MOUTH_VERTICAL_PAIRS:
        needed_indices.extend([t_idx, b_idx])

    if len(landmarks) <= max(needed_indices):
        return 0.0

    if isinstance(landmarks, np.ndarray):
        left_c = landmarks[MOUTH_OUTER_CORNERS[0], :2]
        right_c = landmarks[MOUTH_OUTER_CORNERS[1], :2]
        horiz = float(math.hypot((left_c[0] - right_c[0]) * aspect_ratio, left_c[1] - right_c[1]))
        if horiz < eps:
            return 0.0

        vert_sum = 0.0
        for top_idx, bot_idx in MOUTH_VERTICAL_PAIRS:
            p_top = landmarks[top_idx, :2]
            p_bot = landmarks[bot_idx, :2]
            vert_sum += float(math.hypot((p_top[0] - p_bot[0]) * aspect_ratio, p_top[1] - p_bot[1]))
    else:
        left_c = landmarks[MOUTH_OUTER_CORNERS[0]]
        right_c = landmarks[MOUTH_OUTER_CORNERS[1]]
        horiz = math.hypot((left_c.x - right_c.x) * aspect_ratio, left_c.y - right_c.y)
        if horiz < eps:
            return 0.0

        vert_sum = 0.0
        for top_idx, bot_idx in MOUTH_VERTICAL_PAIRS:
            p_top = landmarks[top_idx]
            p_bot = landmarks[bot_idx]
            vert_sum += math.hypot((p_top.x - p_bot.x) * aspect_ratio, p_top.y - p_bot.y)

    mean_vert = vert_sum / len(MOUTH_VERTICAL_PAIRS)
    return float(mean_vert / (horiz + eps))


class YawnDetector:
    """
    Temporal tracker that accumulates mouth opening over time to distinguish
    authentic yawns from conversation speech.
    """

    def __init__(
        self,
        mar_open_threshold: float = DEFAULT_MAR_OPEN_THRESHOLD,
        min_yawn_duration_ms: float = DEFAULT_MIN_YAWN_DURATION_MS,
        max_yawn_duration_ms: float = DEFAULT_MAX_YAWN_DURATION_MS,
    ) -> None:
        self.mar_open_threshold = mar_open_threshold
        self.min_yawn_duration_ms = min_yawn_duration_ms
        self.max_yawn_duration_ms = max_yawn_duration_ms

        self._is_opening: bool = False
        self._opening_start_ts: Optional[float] = None
        self._peak_mar: float = 0.0
        self._total_yawns: int = 0
        self._last_ts: Optional[float] = None

    def update(
        self,
        landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
        timestamp_ms: float,
        aspect_ratio: float = 16.0 / 9.0,
        has_face: bool = True,
    ) -> MouthAspectRatioResult:
        """
        Evaluate frame landmarks and update temporal yawn state.
        """
        # If face is lost, cancel active in-flight opening without registering a completed yawn
        if not has_face or len(landmarks) == 0:
            self._is_opening = False
            self._opening_start_ts = None
            self._peak_mar = 0.0
            self._last_ts = timestamp_ms
            return MouthAspectRatioResult(
                mar=0.0,
                is_mouth_open=False,
                is_yawn_in_progress=False,
                current_yawn_duration_ms=0.0,
                is_authentic_yawn=False,
                completed_yawn=None,
                total_yawn_count=self._total_yawns,
            )

        mar = calculate_mar(landmarks, aspect_ratio=aspect_ratio)
        is_mouth_open = mar >= self.mar_open_threshold

        completed_yawn: Optional[YawnEvent] = None
        current_yawn_duration_ms = 0.0
        is_yawn_in_progress = False
        is_authentic_yawn = False

        # Check for unexpected sampling gap (> 500ms) to reset in-flight yawn
        if self._last_ts is not None and (timestamp_ms - self._last_ts) > 500.0:
            self._is_opening = False
            self._opening_start_ts = None
            self._peak_mar = 0.0

        self._last_ts = timestamp_ms

        if is_mouth_open:
            if not self._is_opening:
                # Mouth opening initiated
                self._is_opening = True
                self._opening_start_ts = timestamp_ms
                self._peak_mar = mar
            else:
                self._peak_mar = max(self._peak_mar, mar)

            if self._opening_start_ts is not None:
                current_yawn_duration_ms = timestamp_ms - self._opening_start_ts

            is_yawn_in_progress = current_yawn_duration_ms >= (self.min_yawn_duration_ms * 0.5)
            is_authentic_yawn = current_yawn_duration_ms >= self.min_yawn_duration_ms
        else:
            if self._is_opening and self._opening_start_ts is not None:
                duration_ms = timestamp_ms - self._opening_start_ts
                # Check if opening met authentic yawn criteria
                if self.min_yawn_duration_ms <= duration_ms <= self.max_yawn_duration_ms:
                    completed_yawn = YawnEvent(
                        start_ts_ms=self._opening_start_ts,
                        end_ts_ms=timestamp_ms,
                        duration_ms=duration_ms,
                        peak_mar=self._peak_mar,
                    )
                    self._total_yawns += 1

            self._is_opening = False
            self._opening_start_ts = None
            self._peak_mar = 0.0

        return MouthAspectRatioResult(
            mar=mar,
            is_mouth_open=is_mouth_open,
            is_yawn_in_progress=is_yawn_in_progress,
            current_yawn_duration_ms=current_yawn_duration_ms,
            is_authentic_yawn=is_authentic_yawn,
            completed_yawn=completed_yawn,
            total_yawn_count=self._total_yawns,
        )

    def reset(self) -> None:
        """Reset temporal state for a new driving session."""
        self._is_opening = False
        self._opening_start_ts = None
        self._peak_mar = 0.0
        self._total_yawns = 0
        self._last_ts = None
