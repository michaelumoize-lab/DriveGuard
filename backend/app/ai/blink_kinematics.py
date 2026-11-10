"""
Temporal Blink Kinematics, Sub-Frame Timestamp Interpolation, and Rolling PERCLOS Engine.

Phase 5 Implementation:
- Formal Blink State Machine (OPEN, CLOSING, CLOSED, REOPENING, PROLONGED_CLOSURE, UNCERTAIN_GAP)
- Deterministic blink classifications (NORMAL, PROLONGED, REPEATED_ABNORMAL, MICROSLEEP_CANDIDATE, UNCERTAIN)
- Sub-frame linear crossing interpolation and velocity kinematics (closing velocity, reopening velocity, RCR)
- Sampling-gap awareness & explicit face-loss handling (handle_gap, handle_face_lost)
- Observation coverage tracking & reliable rolling time-weighted PERCLOS (%)
- Independent left/right eye tracking and asymmetry protection
- Personalized blink-duration baseline integration (median and scaled MAD)
- Yawn event duration and frequency tracking
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Deque, List, Optional, Tuple


class BlinkState(str, Enum):
    """Explicit states of the temporal blink state machine."""

    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    REOPENING = "REOPENING"
    PROLONGED_CLOSURE = "PROLONGED_CLOSURE"
    UNCERTAIN_GAP = "UNCERTAIN_GAP"


class BlinkClassification(str, Enum):
    """Deterministic classification of observed blinks and closures."""

    NORMAL = "NORMAL"
    PROLONGED = "PROLONGED"
    REPEATED_ABNORMAL = "REPEATED_ABNORMAL"
    MICROSLEEP_CANDIDATE = "MICROSLEEP_CANDIDATE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class BlinkEvent:
    """Completed blink event with full kinematic and asymmetric measurements."""

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
    classification: BlinkClassification = BlinkClassification.NORMAL
    is_binocular: bool = True
    left_ear_min: Optional[float] = None
    right_ear_min: Optional[float] = None


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
    blink_state: BlinkState = BlinkState.OPEN
    blink_classification: BlinkClassification = BlinkClassification.NORMAL
    perclos_coverage_pct: float = 100.0
    perclos_reliable: bool = True
    sampling_gap_detected: bool = False
    left_eye_closed: bool = False
    right_eye_closed: bool = False
    abnormal_blink_cluster_detected: bool = False


class TemporalKinematicsTracker:
    """
    Per-session stateful tracker for blink waveforms, sub-frame duration interpolation,
    reopening velocity kinematics, sampling-gap awareness, and rolling time-weighted PERCLOS.
    """

    def __init__(
        self,
        ema_alpha: float = 0.60,
        perclos_window_sec: float = 60.0,
        min_blink_duration_ms: float = 55.0,
        prolonged_blink_duration_ms: float = 400.0,
        microsleep_duration_ms: float = 800.0,
        min_perclos_coverage_pct: float = 50.0,
        max_sampling_gap_ms: float = 250.0,
        personalized_blink_median_ms: Optional[float] = None,
        personalized_blink_mad_ms: Optional[float] = None,
    ) -> None:
        self.ema_alpha = ema_alpha
        self.perclos_window_ms = perclos_window_sec * 1000.0
        self.min_blink_duration_ms = min_blink_duration_ms
        self.max_blink_duration_ms = microsleep_duration_ms  # backward compatibility alias
        self.prolonged_blink_duration_ms = prolonged_blink_duration_ms
        self.microsleep_duration_ms = microsleep_duration_ms
        self.min_perclos_coverage_pct = min_perclos_coverage_pct
        self.max_sampling_gap_ms = max_sampling_gap_ms
        self.personalized_blink_median_ms = personalized_blink_median_ms
        self.personalized_blink_mad_ms = personalized_blink_mad_ms

        self._smoothed_ear: Optional[float] = None
        self._smoothed_closure: Optional[float] = None
        self._prev_ts_ms: Optional[float] = None
        self._prev_ear: Optional[float] = None

        # State machine tracking
        self._state: BlinkState = BlinkState.OPEN
        self._blink_enter_ts_ms: float = 0.0
        self._blink_open_ear_ref: float = 0.28
        self._blink_min_ear: float = 1.0
        self._blink_min_ts_ms: float = 0.0
        self._left_min_ear: Optional[float] = None
        self._right_min_ear: Optional[float] = None
        self._sampling_gap_detected: bool = False

        # Active yawn state tracking
        self._in_yawn: bool = False
        self._yawn_enter_ts_ms: float = 0.0
        self._yawn_peak_mar: float = 0.0

        # Rolling observation history for PERCLOS:
        # Tuple of: (timestamp_ms, dt_ms, is_valid: bool, is_closed: bool)
        self._observation_samples: Deque[Tuple[float, float, bool, bool]] = deque()
        self.blink_history: Deque[BlinkEvent] = deque(maxlen=120)
        self.yawn_history: Deque[YawnEvent] = deque(maxlen=50)

    @property
    def _in_blink(self) -> bool:
        """Backward compatibility property for older inspection checks."""
        return self._state in (BlinkState.CLOSING, BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE)

    def set_personalized_baseline(
        self,
        median_duration_ms: Optional[float],
        mad_duration_ms: Optional[float],
    ) -> None:
        """Update personalized driver blink statistics from PDBP calibration profile."""
        if median_duration_ms is not None and median_duration_ms > 0:
            self.personalized_blink_median_ms = median_duration_ms
        if mad_duration_ms is not None and mad_duration_ms > 0:
            self.personalized_blink_mad_ms = mad_duration_ms

    def reset(self) -> None:
        """Clear all temporal state for a new session."""
        self._smoothed_ear = None
        self._smoothed_closure = None
        self._prev_ts_ms = None
        self._prev_ear = None
        self._state = BlinkState.OPEN
        self._blink_enter_ts_ms = 0.0
        self._blink_open_ear_ref = 0.28
        self._blink_min_ear = 1.0
        self._blink_min_ts_ms = 0.0
        self._left_min_ear = None
        self._right_min_ear = None
        self._sampling_gap_detected = False
        self._in_yawn = False
        self._yawn_enter_ts_ms = 0.0
        self._yawn_peak_mar = 0.0
        self._observation_samples.clear()
        self.blink_history.clear()
        self.yawn_history.clear()

    def handle_gap(self, timestamp_ms: float, gap_ms: Optional[float] = None) -> None:
        """
        Explicitly handle an observation sampling gap (network drop, frame drop, or inference stall).
        Invalidates any incomplete in-flight blink, records unobserved interval, and sets UNCERTAIN_GAP state.
        """
        if gap_ms is None and self._prev_ts_ms is not None:
            gap_duration = max(0.0, timestamp_ms - self._prev_ts_ms)
        elif gap_ms is not None:
            gap_duration = max(0.0, gap_ms)
        else:
            gap_duration = self.max_sampling_gap_ms

        # Invalidate in-flight blink to prevent fabricating false prolonged closure across the gap
        if self._state in (
            BlinkState.CLOSING,
            BlinkState.CLOSED,
            BlinkState.PROLONGED_CLOSURE,
            BlinkState.REOPENING,
        ):
            self._state = BlinkState.UNCERTAIN_GAP

        self._sampling_gap_detected = True

        # Record unobserved time interval for PERCLOS observation coverage
        if gap_duration > 0:
            self._observation_samples.append((timestamp_ms, gap_duration, False, False))

        # Dampen previous timestamps and reset EMA history
        self._prev_ts_ms = timestamp_ms
        self._smoothed_ear = None
        self._smoothed_closure = None

    def handle_face_lost(self, timestamp_ms: float) -> None:
        """
        Explicitly handle face detection loss.
        Prevents in-flight blinks from becoming stuck as prolonged closures.
        """
        gap_dur = (
            max(0.0, timestamp_ms - self._prev_ts_ms)
            if self._prev_ts_ms is not None
            else self.max_sampling_gap_ms
        )
        self.handle_gap(timestamp_ms=timestamp_ms, gap_ms=gap_dur)

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
        left_ear: Optional[float] = None,
        right_ear: Optional[float] = None,
        left_threshold: Optional[float] = None,
        right_threshold: Optional[float] = None,
        landmark_confidence: Optional[float] = None,
        head_yaw_deg: float = 0.0,
        mar_yawn_threshold: float = 0.55,
    ) -> TemporalKinematicsSnapshot:
        """
        Process a new valid observation at `timestamp_ms` and return updated rolling kinematics.
        """
        # --- 1. Detect & Handle Sampling Gaps ---
        sampling_gap_this_frame = False
        if self._prev_ts_ms is not None:
            raw_dt = timestamp_ms - self._prev_ts_ms
            if raw_dt > self.max_sampling_gap_ms:
                self.handle_gap(timestamp_ms=timestamp_ms, gap_ms=raw_dt)
                sampling_gap_this_frame = True
                dt_ms = 33.3  # Nominal step after gap
            elif raw_dt <= 0:
                # Monotonic safeguard
                dt_ms = 1.0
            else:
                dt_ms = raw_dt
        else:
            dt_ms = 33.3

        # --- 2. EMA Signal Smoothing ---
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

        # --- 3. Independent Left/Right Eye Evaluation & Asymmetry Protection ---
        l_thresh = left_threshold if left_threshold is not None else ear_close_threshold
        r_thresh = right_threshold if right_threshold is not None else ear_close_threshold

        l_val = left_ear if left_ear is not None else s_ear
        r_val = right_ear if right_ear is not None else s_ear

        left_eye_closed = l_val < l_thresh
        right_eye_closed = r_val < r_thresh

        # Dual confirmation: requires BOTH geometric EAR closure AND cascade/CNN closure score >= 0.55
        # Deep physical/neural closure overrides protect edge cases (e.g. eyes squeezed tightly shut)
        is_closed = (
            (s_ear < ear_close_threshold and s_closure >= 0.55)
            or (s_ear < ear_close_threshold * 0.78)
            or (s_closure >= 0.85 and s_ear < ear_close_threshold * 1.08)
        )

        completed_blink: Optional[BlinkEvent] = None

        # --- 4. Formal Blink State Machine ---
        if self._state == BlinkState.UNCERTAIN_GAP:
            # First valid observation following an interruption
            self._sampling_gap_detected = False
            if is_closed:
                self._state = BlinkState.CLOSED
                self._blink_enter_ts_ms = timestamp_ms
                self._blink_open_ear_ref = max(s_ear, ear_close_threshold + 0.05)
                self._blink_min_ear = s_ear
                self._blink_min_ts_ms = timestamp_ms
                self._left_min_ear = l_val
                self._right_min_ear = r_val
            else:
                self._state = BlinkState.OPEN

        elif self._state == BlinkState.OPEN:
            if is_closed:
                # Closure initiated
                self._state = BlinkState.CLOSED
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
                self._left_min_ear = l_val
                self._right_min_ear = r_val

        elif self._state in (BlinkState.CLOSING, BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE):
            if is_closed:
                if s_ear < self._blink_min_ear:
                    self._blink_min_ear = s_ear
                    self._blink_min_ts_ms = timestamp_ms
                    self._left_min_ear = l_val
                    self._right_min_ear = r_val

                closure_dur = max(0.0, timestamp_ms - self._blink_enter_ts_ms)
                if closure_dur >= self.prolonged_blink_duration_ms:
                    self._state = BlinkState.PROLONGED_CLOSURE
            else:
                # Eyes reopened -> Reopening & Finalization
                self._state = BlinkState.REOPENING
                exit_ts_ms = (
                    self._interpolate_crossing_ts(
                        self._prev_ts_ms, self._prev_ear, timestamp_ms, s_ear, ear_close_threshold
                    )
                    if (self._prev_ts_ms is not None and self._prev_ear is not None)
                    else timestamp_ms
                )
                duration_ms = max(1.0, exit_ts_ms - self._blink_enter_ts_ms)

                # Kinematic Velocities & RCR
                amp_close = max(0.01, self._blink_open_ear_ref - self._blink_min_ear)
                amp_open = max(0.01, s_ear - self._blink_min_ear)
                t_close_s = max(0.015, (self._blink_min_ts_ms - self._blink_enter_ts_ms) / 1000.0)
                t_open_s = max(0.015, (exit_ts_ms - self._blink_min_ts_ms) / 1000.0)
                v_close = amp_close / t_close_s
                v_open = amp_open / t_open_s
                rcr = float(v_open / max(v_close, 1e-4))

                # Personalized prolonged cutoff calculation
                if (
                    self.personalized_blink_median_ms is not None
                    and self.personalized_blink_mad_ms is not None
                ):
                    prolonged_cutoff = max(
                        self.prolonged_blink_duration_ms,
                        self.personalized_blink_median_ms + 2.5 * self.personalized_blink_mad_ms,
                    )
                else:
                    prolonged_cutoff = self.prolonged_blink_duration_ms

                # Blink Classification
                if duration_ms < self.min_blink_duration_ms:
                    cls_type = BlinkClassification.UNCERTAIN
                elif duration_ms >= self.microsleep_duration_ms:
                    cls_type = BlinkClassification.MICROSLEEP_CANDIDATE
                elif duration_ms >= prolonged_cutoff:
                    cls_type = BlinkClassification.PROLONGED
                else:
                    cls_type = BlinkClassification.NORMAL

                # Binocular coordination check (account for yaw perspective)
                is_binocular = True
                if left_ear is not None and right_ear is not None and abs(head_yaw_deg) < 15.0:
                    if abs(l_val - r_val) > 0.12:
                        is_binocular = False

                if cls_type != BlinkClassification.UNCERTAIN:
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
                        classification=cls_type,
                        is_binocular=is_binocular,
                        left_ear_min=self._left_min_ear,
                        right_ear_min=self._right_min_ear,
                    )
                    self.blink_history.append(completed_blink)
                else:
                    completed_blink = None

                self._state = BlinkState.OPEN

        elif self._state == BlinkState.REOPENING:
            self._state = BlinkState.OPEN

        current_closure_duration_ms = (
            max(0.0, timestamp_ms - self._blink_enter_ts_ms)
            if self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE)
            else 0.0
        )

        # --- 5. Rolling Time-Weighted PERCLOS & Observation Coverage ---
        self._observation_samples.append((timestamp_ms, dt_ms, True, is_closed))
        cutoff_window = timestamp_ms - self.perclos_window_ms
        while self._observation_samples and self._observation_samples[0][0] < cutoff_window:
            self._observation_samples.popleft()

        valid_observed_ms = sum(s[1] for s in self._observation_samples if s[2])
        observed_closed_ms = sum(s[1] for s in self._observation_samples if s[2] and s[3])
        unobserved_ms = sum(s[1] for s in self._observation_samples if not s[2])
        total_tracked_ms = valid_observed_ms + unobserved_ms

        perclos_coverage_pct = (
            min(100.0, (valid_observed_ms / self.perclos_window_ms) * 100.0)
            if self.perclos_window_ms > 0
            else 0.0
        )

        # PERCLOS = Observed closed duration / Valid observed duration
        if valid_observed_ms > 0.0:
            perclos_60s_pct = (observed_closed_ms / valid_observed_ms) * 100.0
        else:
            perclos_60s_pct = 0.0

        perclos_reliable = (
            perclos_coverage_pct >= self.min_perclos_coverage_pct
            and valid_observed_ms >= 15_000.0
        )

        # --- 6. Rolling Blink Frequency & Clusters ---
        blinks_in_window = [b for b in self.blink_history if b.end_ts_ms >= cutoff_window]
        observed_window_min = max(0.25, min(1.0, valid_observed_ms / 60000.0))
        blink_rate_per_min = len(blinks_in_window) / observed_window_min

        # Use median over recent 30s binocular blinks so a single outlier or stale blink does not stick
        cutoff_recent_30s = timestamp_ms - 30_000.0
        recent_valid_blinks: List[BlinkEvent] = [
            b for b in blinks_in_window if b.end_ts_ms >= cutoff_recent_30s and b.is_binocular
        ][-4:]
        if not recent_valid_blinks and blinks_in_window:
            recent_valid_blinks = [b for b in blinks_in_window if b.end_ts_ms >= cutoff_recent_30s][-3:]

        if recent_valid_blinks:
            sorted_durs = sorted(b.duration_ms for b in recent_valid_blinks)
            sorted_rcrs = sorted(b.rcr for b in recent_valid_blinks)
            mid = len(recent_valid_blinks) // 2
            if len(recent_valid_blinks) % 2 == 1:
                recent_blink_duration_ms = float(sorted_durs[mid])
                recent_blink_rcr = float(sorted_rcrs[mid])
            else:
                recent_blink_duration_ms = float(0.5 * (sorted_durs[mid - 1] + sorted_durs[mid]))
                recent_blink_rcr = float(0.5 * (sorted_rcrs[mid - 1] + sorted_rcrs[mid]))
        else:
            recent_blink_duration_ms = None
            recent_blink_rcr = None

        # Repeated abnormal blink cluster detection (e.g. 2 of last 4 blinks prolonged or RCR < 0.45)
        recent_last_4 = blinks_in_window[-4:] if len(blinks_in_window) >= 2 else []
        prolonged_count = sum(
            1
            for b in recent_last_4
            if b.classification in (BlinkClassification.PROLONGED, BlinkClassification.MICROSLEEP_CANDIDATE)
        )
        sluggish_count = sum(1 for b in recent_last_4 if b.rcr < 0.45)
        abnormal_cluster = prolonged_count >= 2 or (len(recent_last_4) >= 3 and sluggish_count >= 2)

        # --- 7. Yawning Temporal Analysis (sustained >= 1.8s) ---
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

        # Determine active blink classification for snapshot
        if self._state == BlinkState.UNCERTAIN_GAP:
            active_classification = BlinkClassification.UNCERTAIN
        elif self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE):
            if current_closure_duration_ms >= self.microsleep_duration_ms:
                active_classification = BlinkClassification.MICROSLEEP_CANDIDATE
            elif current_closure_duration_ms >= self.prolonged_blink_duration_ms:
                active_classification = BlinkClassification.PROLONGED
            else:
                active_classification = BlinkClassification.NORMAL
        elif abnormal_cluster:
            active_classification = BlinkClassification.REPEATED_ABNORMAL
        elif completed_blink is not None:
            active_classification = completed_blink.classification
        else:
            active_classification = BlinkClassification.NORMAL

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
            blink_state=self._state,
            blink_classification=active_classification,
            perclos_coverage_pct=round(perclos_coverage_pct, 1),
            perclos_reliable=perclos_reliable,
            sampling_gap_detected=sampling_gap_this_frame,
            left_eye_closed=left_eye_closed,
            right_eye_closed=right_eye_closed,
            abnormal_blink_cluster_detected=abnormal_cluster,
        )
