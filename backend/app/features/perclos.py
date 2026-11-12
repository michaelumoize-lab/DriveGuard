"""
DriveGuard — PERCLOS (Percentage of Eye Closure) Accumulator (Phase 5).

1. What it does:
   Calculates the gold-standard PERCLOS drowsiness metric over a sliding time window
   (default: 60 seconds). Tracks observation coverage and handles sampling gaps.

2. Why it exists:
   PERCLOS (established by the FHWA and NHTSA) is the single most validated physiological
   metric for cumulative driver fatigue. A driver whose eyes are closed >= 80% for more than
   12% to 15% of a 60-second window is severely impaired.

3. What data it receives:
   - Millisecond frame timestamp.
   - Boolean flag indicating if eye is closed (`is_closed`).
   - Boolean flag indicating if face is actively tracked (`has_face`).

4. What data it produces:
   - `PerclosResult` dataclass containing `perclos_pct`, `coverage_pct`,
     `is_reliable`, `closed_duration_sec`, and `observed_duration_sec`.

5. Modules that depend on it:
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine (Phase 6)

6. Failure behavior:
   - If observation coverage is < 50% (e.g. face was absent for more than 30s of the 60s window),
     `is_reliable` is flagged False and PERCLOS is excluded from triggering high-confidence warnings.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple


@dataclass(frozen=True)
class PerclosResult:
    """
    Structured rolling PERCLOS evaluation over a sliding time window.
    """
    perclos_pct: float             # Cumulative eye closure percentage [0.0, 100.0]
    coverage_pct: float            # Percentage of window actively observed [0.0, 100.0]
    is_reliable: bool              # True if coverage >= min_coverage_pct (default: 50%)
    closed_duration_sec: float     # Total seconds eyes were closed in window
    observed_duration_sec: float   # Total seconds valid face was observed in window
    window_sec: float              # Window duration (default: 60.0s)


class PerclosCalculator:
    """
    Sliding-window time-weighted PERCLOS accumulator with sampling-gap awareness.
    """

    def __init__(
        self,
        window_sec: float = 60.0,
        min_coverage_pct: float = 50.0,
        max_valid_dt_sec: float = 0.25,
    ) -> None:
        """
        Initialize the PERCLOS calculator.

        Args:
            window_sec: Sliding window length in seconds (default: 60.0s).
            min_coverage_pct: Minimum observation percentage required for reliability (default: 50.0%).
            max_valid_dt_sec: Maximum inter-frame step before gap detection (default: 0.25s).
        """
        if window_sec <= 0.0:
            raise ValueError("window_sec must be positive.")
        if not (0.0 < min_coverage_pct <= 100.0):
            raise ValueError("min_coverage_pct must be in (0.0, 100.0]")

        self.window_sec = window_sec
        self.min_coverage_pct = min_coverage_pct
        self.max_valid_dt_sec = max_valid_dt_sec

        # Ring buffer storing tuples of: (timestamp_ms, dt_sec, is_closed)
        self._samples: Deque[Tuple[float, float, bool]] = deque()
        self._prev_ts_ms: Optional[float] = None

    def update(
        self,
        timestamp_ms: float,
        is_closed: bool,
        has_face: bool = True,
    ) -> PerclosResult:
        """
        Record a new observation and compute rolling PERCLOS.
        """
        # Calculate time delta since previous frame
        if self._prev_ts_ms is not None:
            dt_sec = max(0.0, (timestamp_ms - self._prev_ts_ms) / 1000.0)
        else:
            dt_sec = 0.0

        self._prev_ts_ms = timestamp_ms

        # Only accumulate valid observation steps when face is present and step is reasonable
        if has_face and 0.0 < dt_sec <= self.max_valid_dt_sec:
            self._samples.append((timestamp_ms, dt_sec, is_closed))

        # Evict samples older than sliding window threshold
        window_cutoff_ms = timestamp_ms - (self.window_sec * 1000.0)
        while self._samples and self._samples[0][0] < window_cutoff_ms:
            self._samples.popleft()

        # Aggregate totals in the current active window
        total_observed_sec = 0.0
        total_closed_sec = 0.0

        for _, sample_dt, sample_closed in self._samples:
            total_observed_sec += sample_dt
            if sample_closed:
                total_closed_sec += sample_dt

        # Coverage calculation relative to configured window length
        coverage_pct = min(100.0, (total_observed_sec / self.window_sec) * 100.0)
        is_reliable = coverage_pct >= self.min_coverage_pct

        # PERCLOS = (closed duration / observed duration) * 100
        if total_observed_sec > 0.001:
            perclos_pct = min(100.0, (total_closed_sec / total_observed_sec) * 100.0)
        else:
            perclos_pct = 0.0

        return PerclosResult(
            perclos_pct=perclos_pct,
            coverage_pct=coverage_pct,
            is_reliable=is_reliable,
            closed_duration_sec=total_closed_sec,
            observed_duration_sec=total_observed_sec,
            window_sec=self.window_sec,
        )

    def reset(self) -> None:
        """Reset internal accumulator for a new driving session."""
        self._samples.clear()
        self._prev_ts_ms = None
