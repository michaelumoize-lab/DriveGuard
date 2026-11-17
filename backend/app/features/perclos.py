"""
DriveGuard — PERCLOS (Percentage of Eye Closure) Accumulator (Phase 5).

1. What it does:
   Calculates the gold-standard PERCLOS drowsiness metric over a sliding time window
   (default: 60.0 seconds) using exact elapsed-time integration. Tracks active observation
   coverage, clips rolling-window boundaries, handles sampling gaps, and rejects invalid
   or non-monotonic timestamps without corrupting internal state.

2. Why it exists:
   PERCLOS is the single most validated psychophysiological metric for driver vigilance
   and cumulative fatigue, standardized by the Federal Highway Administration (FHWA) and
   National Highway Traffic Safety Administration (NHTSA) (Wierwille et al., 1994;
   Dinges & Grace, 1998).

3. PERCLOS Variant Implemented:
   This module implements **P80**: the percentage of time that the driver's eyes are
   at least 80% closed over the observation window (Dinges & Grace, 1998).
   The caller passes this state via `is_closed: Optional[bool]`.

   Scientific & Geometric Distinction:
   - Physiological P80 is defined as >= 80% pupil coverage by the eyelids.
   - Fixed EAR thresholds (e.g., 0.20) are nominal population heuristics (Soukupova & Cech, 2016)
     and do NOT directly measure 80% eyelid coverage for any individual driver.
   - Personalized P80 calibration (Phase 6 / PDBP) calculates an individual's geometric proxy:
     EAR_P80 = 0.20 * EAR_open + 0.80 * EAR_closed.
   - `PerclosCalculator` accumulates the elapsed time where `is_closed=True`, agnostic to
     whether `is_closed` was evaluated against a nominal or personalized threshold.

4. What data it receives:
   - `timestamp_ms`: Current monotonic frame timestamp in milliseconds.
   - `is_closed`: Optional boolean indicating P80 closure (None if measurement is unavailable).
   - `has_face`: Boolean indicating whether a face is actively tracked in the frame.
   - `is_valid`: Boolean indicating whether eye landmarks are valid/unoccluded for measurement.

5. What data it produces:
   - `PerclosResult` dataclass containing:
     - `perclos_pct`: Raw calculated closure ratio [0.0, 100.0] for diagnostics and telemetry.
     - `effective_perclos_pct`: Gated PERCLOS ratio (None when observation is unreliable).
     - `coverage_pct`: Proportion of the window actively observed [0.0, 100.0].
     - `is_reliable`: True when observation coverage meets or exceeds `min_coverage_pct`.
     - `closed_duration_sec`: Seconds eyes were observed closed in the window.
     - `observed_duration_sec`: Seconds valid observations were acquired in the window.
     - `unobserved_duration_sec`: Seconds unobserved (gaps, face-loss, or invalid data) in the window.
     - `window_sec`: Configured sliding window duration (default: 60.0s).

6. Downstream dependencies:
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine & Temporal Analyzer

7. Failure & Reliability Contract:
   - Missing, occluded, or unmeasurable eyes are treated as unobserved time. They are
     NEVER converted into open eyes (which would dangerously dilute PERCLOS) or closed
     eyes (which would artificially inflate PERCLOS).
   - Non-finite, negative, duplicate, or backwards timestamps are rejected without
     corrupting the previous valid timestamp.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import math
from typing import Deque, Optional, Tuple

logger = logging.getLogger(__name__)

# PERCLOS Variant: P80 (eyelids >= 80% closed).
# Standardized by FHWA / NHTSA (Wierwille et al., 1994; Dinges & Grace, 1998).
PERCLOS_VARIANT: str = "P80"

# Literature-referenced default parameters.
# Note: Thresholds are configurable baseline indicators and require personalized calibration.
DEFAULT_PERCLOS_WINDOW_SEC: float = 60.0
DEFAULT_MIN_COVERAGE_PCT: float = 50.0
DEFAULT_MAX_VALID_DT_SEC: float = 0.25


@dataclass(frozen=True)
class PerclosResult:
    """
    Structured rolling PERCLOS evaluation over a sliding time window.
    """
    perclos_pct: float             # Raw cumulative eye closure percentage [0.0, 100.0]
    coverage_pct: float            # Percentage of window actively observed [0.0, 100.0]
    is_reliable: bool              # True if coverage >= min_coverage_pct (default: 50%)
    closed_duration_sec: float     # Total seconds eyes were closed in window
    observed_duration_sec: float   # Total seconds valid face/eye was observed in window
    window_sec: float              # Window duration (default: 60.0s)
    unobserved_duration_sec: float = 0.0  # Total unobserved seconds in window

    @property
    def effective_perclos_pct(self) -> Optional[float]:
        """
        Gated PERCLOS percentage: returns perclos_pct if is_reliable is True, else None.

        Returning None prevents downstream alert engines or telemetry dashboards from
        mistaking unrepresentative, low-coverage windows (e.g. session warm-up or severe
        tracking loss) for an alert or drowsy state.
        """
        return self.perclos_pct if self.is_reliable else None


class PerclosCalculator:
    """
    Sliding-window time-weighted PERCLOS (P80) accumulator with boundary clipping,
    sampling-gap awareness, and monotonic timestamp validation.
    """

    def __init__(
        self,
        window_sec: float = DEFAULT_PERCLOS_WINDOW_SEC,
        min_coverage_pct: float = DEFAULT_MIN_COVERAGE_PCT,
        max_valid_dt_sec: float = DEFAULT_MAX_VALID_DT_SEC,
    ) -> None:
        """
        Initialize the PERCLOS calculator.

        Args:
            window_sec: Sliding window length in seconds (default: 60.0s).
            min_coverage_pct: Minimum observation percentage required for reliability (default: 50.0%).
            max_valid_dt_sec: Maximum inter-frame step before gap detection (default: 0.25s).
        """
        if not math.isfinite(window_sec) or window_sec <= 0.0:
            raise ValueError("window_sec must be positive.")
        if not math.isfinite(min_coverage_pct) or not (0.0 < min_coverage_pct <= 100.0):
            raise ValueError("min_coverage_pct must be in (0.0, 100.0]")
        if not math.isfinite(max_valid_dt_sec) or max_valid_dt_sec <= 0.0:
            raise ValueError("max_valid_dt_sec must be positive.")

        self.window_sec = float(window_sec)
        self.min_coverage_pct = float(min_coverage_pct)
        self.max_valid_dt_sec = float(max_valid_dt_sec)

        # Ring buffer storing valid observation intervals as:
        # (start_ts_ms: float, end_ts_ms: float, is_closed: bool)
        self._samples: Deque[Tuple[float, float, bool]] = deque()
        self._prev_ts_ms: Optional[float] = None
        self._prev_valid: bool = False

    def update(
        self,
        timestamp_ms: float,
        is_closed: Optional[bool] = None,
        has_face: bool = True,
        is_valid: bool = True,
    ) -> PerclosResult:
        """
        Record a new observation and compute rolling PERCLOS.

        Args:
            timestamp_ms: Monotonic timestamp of the frame in milliseconds.
            is_closed: Whether the eyes are closed according to P80 criterion.
                       Pass None if eye state is unknown or unmeasurable.
            has_face: Whether a face is detected in the frame.
            is_valid: Whether landmark measurements are reliable (e.g. not occluded).

        Returns:
            PerclosResult with rolling closure metrics, coverage, and reliability.
        """
        # 1. Timestamp sanity validation
        if not math.isfinite(timestamp_ms) or timestamp_ms < 0.0:
            logger.warning("Rejected non-finite or negative timestamp_ms: %s", timestamp_ms)
            return self._evaluate(self._prev_ts_ms)

        # 2. Monotonicity validation
        if self._prev_ts_ms is not None:
            if timestamp_ms < self._prev_ts_ms:
                logger.warning(
                    "Rejected backwards timestamp: %s ms < prev %s ms",
                    timestamp_ms,
                    self._prev_ts_ms,
                )
                return self._evaluate(self._prev_ts_ms)
            if timestamp_ms == self._prev_ts_ms:
                logger.debug("Duplicate timestamp %s ms; skipping interval accumulation", timestamp_ms)
                return self._evaluate(self._prev_ts_ms)

        # 3. Inter-frame delta and validity determination
        dt_sec = (timestamp_ms - self._prev_ts_ms) / 1000.0 if self._prev_ts_ms is not None else 0.0

        # An observation is valid if face is present, landmark quality is valid,
        # and closure measurement is available (not None).
        current_valid = bool(has_face and is_valid and (is_closed is not None))

        # An interval is validly observed only if both boundary frames were valid
        # and inter-frame duration did not exceed sampling gap threshold.
        was_observed = current_valid and self._prev_valid
        if was_observed and 0.0 < dt_sec <= self.max_valid_dt_sec:
            # Record interval [start_ts, end_ts, is_closed]
            assert self._prev_ts_ms is not None
            self._samples.append((self._prev_ts_ms, timestamp_ms, bool(is_closed)))
        elif dt_sec > self.max_valid_dt_sec:
            logger.debug(
                "Sampling gap detected: dt=%.3fs > max_valid_dt=%.3fs. Time treated as unobserved.",
                dt_sec,
                self.max_valid_dt_sec,
            )

        # Update tracking state for next iteration
        self._prev_ts_ms = timestamp_ms
        self._prev_valid = current_valid

        return self._evaluate(timestamp_ms)

    def _evaluate(self, reference_ts_ms: Optional[float]) -> PerclosResult:
        """
        Evaluate rolling PERCLOS metrics at the given reference timestamp,
        pruning expired intervals and clipping the oldest interval at the window boundary.
        """
        if reference_ts_ms is None or reference_ts_ms <= 0.0:
            return PerclosResult(
                perclos_pct=0.0,
                coverage_pct=0.0,
                is_reliable=False,
                closed_duration_sec=0.0,
                observed_duration_sec=0.0,
                window_sec=self.window_sec,
                unobserved_duration_sec=self.window_sec,
            )

        window_ms = self.window_sec * 1000.0
        window_cutoff_ms = reference_ts_ms - window_ms

        # Evict samples that ended at or before the window cutoff
        while self._samples and self._samples[0][1] <= window_cutoff_ms:
            self._samples.popleft()

        total_observed_sec = 0.0
        total_closed_sec = 0.0

        # Aggregate intervals within the sliding window, applying boundary clipping
        for start_ts, end_ts, sample_closed in self._samples:
            # Clip interval start if it crossed the window boundary
            eff_start = max(start_ts, window_cutoff_ms)
            eff_dt = max(0.0, (end_ts - eff_start) / 1000.0)
            total_observed_sec += eff_dt
            if sample_closed:
                total_closed_sec += eff_dt

        # Ensure totals stay within physical window constraints
        total_observed_sec = min(self.window_sec, total_observed_sec)
        total_closed_sec = min(total_observed_sec, total_closed_sec)

        unobserved_duration_sec = max(0.0, self.window_sec - total_observed_sec)
        coverage_pct = min(100.0, max(0.0, (total_observed_sec / self.window_sec) * 100.0))
        is_reliable = coverage_pct >= self.min_coverage_pct

        if total_observed_sec > 1e-4:
            perclos_pct = min(100.0, max(0.0, (total_closed_sec / total_observed_sec) * 100.0))
        else:
            perclos_pct = 0.0

        return PerclosResult(
            perclos_pct=perclos_pct,
            coverage_pct=coverage_pct,
            is_reliable=is_reliable,
            closed_duration_sec=total_closed_sec,
            observed_duration_sec=total_observed_sec,
            window_sec=self.window_sec,
            unobserved_duration_sec=unobserved_duration_sec,
        )

    def reset(self) -> None:
        """Reset internal accumulator for a new driving session."""
        self._samples.clear()
        self._prev_ts_ms = None
        self._prev_valid = False
