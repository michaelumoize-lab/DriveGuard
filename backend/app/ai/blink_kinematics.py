"""
Temporal Blink Kinematics, Sub-Frame Timestamp Interpolation, and Rolling PERCLOS Engine.

Implements:
- Exponential Moving Average (EMA) signal smoothing
- Sub-frame linear interpolation for blink entry and exit timestamps
- Blink duration (ms), closing velocity, reopening velocity, and Reopening-to-Closing Ratio (RCR)
- Rolling time-weighted PERCLOS (%) over configurable observation windows (default 60s)
- Yawn event duration and frequency tracking
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional


@dataclass
class BlinkEvent:
    """Completed blink event with full kinematic measurements."""

    start_ts_ms: float
    min_ts_ms: float
    end_ts_ms: float
    duration_ms: float
    open_ear_before: float
    min_ear: float
    amplitude: float
    closing_velocity_per_s: float
    reopening_velocity_per_s: float
    rcr: float  # Reopening-to-Closing Velocity Ratio


@dataclass
class YawnEvent:
    """Completed prolonged mouth-opening event consistent with yawning."""

    start_ts_ms: float
    end_ts_ms: float
    duration_ms: float
    peak_mar: float


@dataclass
class TemporalKinematicsSnapshot:
    """Current rolling temporal indicators at frame timestamp t."""

    smoothed_ear: float
    smoothed_closure: float
    is_eye_closed: bool
    current_closure_duration_ms: float
    perclos_60s_pct: float
    blink_rate_per_min: float
    recent_blink_duration_ms: Optional[float]
    recent_blink_rcr: Optional[float]
    completed_blink: Optional[BlinkEvent]
    yawn_count_5min: int
    is_yawning: bool


class TemporalKinematicsTracker:
    """
    Per-session stateful tracker for blink waveforms, sub-frame duration interpolation,
    reopening velocity kinematics, and rolling time-weighted PERCLOS.
    """

    def __init__(
        self,
        ema_alpha: float = 0.60,
        perclos_window_sec: float = 60.0,
        min_blink_duration_ms: float = 55.0,
        max_blink_duration_ms: float = 2500.0,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.perclos_window_ms = perclos_window_sec * 1000.0
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = max_blink_duration_ms

        self._smoothed_ear: Optional[float] = None
        self._smoothed_closure: Optional[float] = None
        self._prev_ts_ms: Optional[float] = None
        self._prev_ear: Optional[float] = None

        # Active blink state tracking
        self._in_blink: bool = False
        self._blink_enter_ts_ms: float = 0.0
        self._blink_open_ear_ref: float = 0.28
        self._blink_min_ear: float = 1.0
        self._blink_min_ts_ms: float = 0.0

        # Active yawn state tracking
        self._in_yawn: bool = False
        self._yawn_enter_ts_ms: float = 0.0
        self._yawn_peak_mar: float = 0.0

        # Rolling history buffers: (timestamp_ms, dt_ms, is_closed_bool)
        self._closure_samples: Deque[tuple[float, float, bool]] = deque()
        self.blink_history: Deque[BlinkEvent] = deque(maxlen=120)
        self.yawn_history: Deque[YawnEvent] = deque(maxlen=50)

    def reset(self) -> None:
        """Clear all temporal state for a new session."""
        self._smoothed_ear = None
        self._smoothed_closure = None
        self._prev_ts_ms = None
        self._prev_ear = None
        self._in_blink = False
        self._in_yawn = False
        self._closure_samples.clear()
        self.blink_history.clear()
        self.yawn_history.clear()

    @staticmethod
    def _interpolate_crossing_ts(
        t1: float, v1: float, t2: float, v2: float, threshold: float
    ) -> float:
        """Linearly interpolate the exact timestamp when signal crosses threshold between (t1, v1) and (t2, v2)."""
        dv = v2 - v1
        if abs(dv) < 1e-6 or t2 <= t1:
            return 0.5 * (t1 + t2)
        frac = float((threshold - v1) / dv)
        frac = max(0.0, min(1.0, frac))
        return t1 + frac * (t2 - t1)

    def update(
        self,
        timestamp_ms: float,
        ear_value: float,
        fused_closure_score: float,
        mar_value: float,
        ear_close_threshold: float,
        mar_yawn_threshold: float = 0.55,
    ) -> TemporalKinematicsSnapshot:
        """
        Process a new valid observation at `timestamp_ms` and return updated rolling kinematics.
        """
        if self._smoothed_ear is None:
            self._smoothed_ear = ear_value
            self._smoothed_closure = fused_closure_score
        else:
            self._smoothed_ear = (
                self.ema_alpha * ear_value + (1.0 - self.ema_alpha) * self._smoothed_ear
            )
            self._smoothed_closure = (
                self.ema_alpha * fused_closure_score
                + (1.0 - self.ema_alpha) * (self._smoothed_closure or fused_closure_score)
            )

        s_ear = float(self._smoothed_ear)
        s_closure = float(self._smoothed_closure)

        dt_ms = (
            max(1.0, min(250.0, timestamp_ms - self._prev_ts_ms))
            if self._prev_ts_ms is not None
            else 33.3
        )

        # Eye closure is governed by the cascade-arbitrated closure score (or geometric EAR when consistent)
        is_closed = (s_closure >= 0.62) or (s_ear < ear_close_threshold and s_closure >= 0.45)

        completed_blink: Optional[BlinkEvent] = None

        # --- Blink Waveform State Machine with Sub-Frame Timestamp Interpolation ---
        if not self._in_blink:
            if is_closed:
                self._in_blink = True
                if self._prev_ts_ms is not None and self._prev_ear is not None:
                    self._blink_enter_ts_ms = self._interpolate_crossing_ts(
                        self._prev_ts_ms, self._prev_ear, timestamp_ms, s_ear, ear_close_threshold
                    )
                    self._blink_open_ear_ref = max(self._prev_ear, ear_close_threshold + 0.03)
                else:
                    self._blink_enter_ts_ms = timestamp_ms
                    self._blink_open_ear_ref = ear_close_threshold + 0.05
                self._blink_min_ear = s_ear
                self._blink_min_ts_ms = timestamp_ms
        else:
            if is_closed:
                if s_ear < self._blink_min_ear:
                    self._blink_min_ear = s_ear
                    self._blink_min_ts_ms = timestamp_ms
            else:
                # Eyes reopened -> finalize blink event
                exit_ts_ms = (
                    self._interpolate_crossing_ts(
                        self._prev_ts_ms, self._prev_ear, timestamp_ms, s_ear, ear_close_threshold
                    )
                    if (self._prev_ts_ms is not None and self._prev_ear is not None)
                    else timestamp_ms
                )
                duration_ms = max(1.0, exit_ts_ms - self._blink_enter_ts_ms)
                if self.min_blink_duration_ms <= duration_ms <= self.max_blink_duration_ms:
                    amp_close = max(0.01, self._blink_open_ear_ref - self._blink_min_ear)
                    amp_open = max(0.01, s_ear - self._blink_min_ear)
                    t_close_s = max(0.015, (self._blink_min_ts_ms - self._blink_enter_ts_ms) / 1000.0)
                    t_open_s = max(0.015, (exit_ts_ms - self._blink_min_ts_ms) / 1000.0)

                    v_close = amp_close / t_close_s
                    v_open = amp_open / t_open_s
                    rcr = float(v_open / max(v_close, 1e-4))

                    completed_blink = BlinkEvent(
                        start_ts_ms=self._blink_enter_ts_ms,
                        min_ts_ms=self._blink_min_ts_ms,
                        end_ts_ms=exit_ts_ms,
                        duration_ms=duration_ms,
                        open_ear_before=self._blink_open_ear_ref,
                        min_ear=self._blink_min_ear,
                        amplitude=0.5 * (amp_close + amp_open),
                        closing_velocity_per_s=v_close,
                        reopening_velocity_per_s=v_open,
                        rcr=rcr,
                    )
                    self.blink_history.append(completed_blink)

                self._in_blink = False

        current_closure_duration_ms = (
            max(0.0, timestamp_ms - self._blink_enter_ts_ms) if self._in_blink else 0.0
        )

        # --- Time-Weighted Rolling PERCLOS (60s window) ---
        self._closure_samples.append((timestamp_ms, dt_ms, is_closed))
        cutoff_60s = timestamp_ms - self.perclos_window_ms
        while self._closure_samples and self._closure_samples[0][0] < cutoff_60s:
            self._closure_samples.popleft()

        total_dt = sum(item[1] for item in self._closure_samples)
        closed_dt = sum(item[1] for item in self._closure_samples if item[2])
        perclos_60s_pct = (closed_dt / total_dt * 100.0) if total_dt > 0.0 else 0.0

        # --- Rolling Blink Rate & Recent Blink Kinematics ---
        blinks_in_60s = [b for b in self.blink_history if b.end_ts_ms >= cutoff_60s]
        observed_window_min = max(0.25, min(1.0, total_dt / 60000.0))
        blink_rate_per_min = len(blinks_in_60s) / observed_window_min

        recent_blinks: List[BlinkEvent] = list(self.blink_history)[-3:]
        recent_blink_duration_ms = (
            float(sum(b.duration_ms for b in recent_blinks) / len(recent_blinks))
            if recent_blinks
            else None
        )
        recent_blink_rcr = (
            float(sum(b.rcr for b in recent_blinks) / len(recent_blinks))
            if recent_blinks
            else None
        )

        # --- Yawning Temporal Analysis (distinguish sustained yawn >= 1.8s from speech) ---
        if not self._in_yawn:
            if mar_value >= mar_yawn_threshold:
                self._in_yawn = True
                self._yawn_enter_ts_ms = timestamp_ms
                self._yawn_peak_mar = mar_value
        else:
            if mar_value >= mar_yawn_threshold * 0.90:
                self._yawn_peak_mar = max(self._yawn_peak_mar, mar_value)
            else:
                yawn_dur_ms = timestamp_ms - self._yawn_enter_ts_ms
                if 1800.0 <= yawn_dur_ms <= 8500.0:
                    self.yawn_history.append(
                        YawnEvent(
                            start_ts_ms=self._yawn_enter_ts_ms,
                            end_ts_ms=timestamp_ms,
                            duration_ms=yawn_dur_ms,
                            peak_mar=self._yawn_peak_mar,
                        )
                    )
                self._in_yawn = False

        cutoff_5min = timestamp_ms - 300_000.0
        yawn_count_5min = sum(1 for y in self.yawn_history if y.end_ts_ms >= cutoff_5min)
        is_actively_yawning = self._in_yawn and (timestamp_ms - self._yawn_enter_ts_ms >= 1500.0)

        self._prev_ts_ms = timestamp_ms
        self._prev_ear = s_ear

        return TemporalKinematicsSnapshot(
            smoothed_ear=s_ear,
            smoothed_closure=s_closure,
            is_eye_closed=is_closed,
            current_closure_duration_ms=current_closure_duration_ms,
            perclos_60s_pct=perclos_60s_pct,
            blink_rate_per_min=blink_rate_per_min,
            recent_blink_duration_ms=recent_blink_duration_ms,
            recent_blink_rcr=recent_blink_rcr,
            completed_blink=completed_blink,
            yawn_count_5min=yawn_count_5min,
            is_yawning=is_actively_yawning,
        )
