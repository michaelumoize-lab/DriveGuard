"""
DriveGuard — Temporal Blink Kinematics & Sub-Frame Event Detector (Phase 5).

1. What it does:
   Detects blink events, tracks eyelid closing/reopening waveforms using a formal
   state machine, applies sub-frame linear crossing interpolation, and calculates
   kinematics (closing velocity, reopening velocity, and RCR).

2. Why it exists:
   Blink duration and velocity are critical physiological indicators of drowsiness:
   - Alert blinks: Snappy, short (100–250 ms), with high reopening-to-closing ratio.
   - Drowsy blinks: Sluggish eyelid reopening (prolonged duration > 350 ms, low RCR).
   - Microsleeps: Unbroken closure >= 800 ms.
   Sub-frame interpolation prevents quantisation error from discrete frame rates.

3. What data it receives:
   - Left, right, and uncompensated perspective-weighted EAR measurements.
   - Millisecond frame timestamp.
   - Face tracking state (`has_face`).

4. What data it produces:
   - `BlinkSnapshot` containing current blink state, active closure duration,
     completed `BlinkEvent` (if any), recent kinematics (duration, RCR), and totals.

5. Modules that depend on it:
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine & Temporal Analyzer (Phase 6)

6. Failure behavior:
   - Face loss or sampling gap (> 250 ms) invalidates in-flight blinks to `UNCERTAIN_GAP`.
   - On tracking recovery with closed eyes, begins tracking from the recovery timestamp
     with left-censoring: contributes to sustained-closure monitoring but emits no fabricated
     completed blink upon reopening.
   - Non-monotonic timestamps, duplicate timestamps, and NaN/Inf values are rejected safely.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from typing import Optional, Tuple


class BlinkState(str, Enum):
    """Explicit states of the eyelid kinematic state machine."""
    OPEN = "OPEN"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    REOPENING = "REOPENING"
    PROLONGED_CLOSURE = "PROLONGED_CLOSURE"
    UNCERTAIN_GAP = "UNCERTAIN_GAP"


class BlinkClassification(str, Enum):
    """Deterministic classification of observed blinks and closures."""
    NORMAL = "NORMAL"                         # 40ms <= duration <= 350ms
    PROLONGED = "PROLONGED"                   # 350ms < duration < 800ms
    MICROSLEEP_CANDIDATE = "MICROSLEEP"       # Unbroken closure >= 800ms
    UNCERTAIN = "UNCERTAIN"                   # Incomplete or contaminated by gap


@dataclass(frozen=True)
class BlinkEvent:
    """
    Completed blink event with kinematic measurements.
    Preserves exact positional order of fields 0 through 11 for backward compatibility.
    """
    start_ts_ms: float
    min_ts_ms: float
    end_ts_ms: float
    duration_ms: float
    open_ear_before: float
    min_ear: float
    amplitude: float
    closing_velocity_per_s: float
    reopening_velocity_per_s: float
    rcr: float  # Reopening-to-Closing Velocity Ratio (v_reopen / v_close)
    classification: BlinkClassification
    is_binocular: bool
    is_kinematics_reliable: bool = True


@dataclass(frozen=True)
class BlinkSnapshot:
    """
    Current real-time blink state and rolling kinematic indicators.
    """
    state: BlinkState
    is_eye_closed: bool
    current_closure_duration_ms: float
    completed_blink: Optional[BlinkEvent]
    recent_blink_duration_ms: Optional[float]
    recent_blink_rcr: Optional[float]
    total_blink_count: int
    prolonged_blink_count: int

    @property
    def sampling_gap_detected(self) -> bool:
        """Whether a sampling gap or tracking uncertainty was flagged on this frame."""
        return self.state == BlinkState.UNCERTAIN_GAP


class BlinkKinematicsDetector:
    """
    Stateful detector tracking eyelid kinetics with sub-frame crossing interpolation.
    """

    def __init__(
        self,
        closure_threshold: float = 0.205,
        reopen_threshold: float = 0.240,
        min_blink_duration_ms: float = 40.0,
        prolonged_duration_ms: float = 350.0,
        microsleep_duration_ms: float = 800.0,
        min_amplitude: float = 0.04,
        max_sampling_gap_ms: float = 250.0,
    ) -> None:
        """
        Initialize the blink kinematics detector.
        
        Note on provisional thresholds:
        - min_blink_duration_ms (default 40.0 ms): A provisional filter against sub-frame
          landmark flutter. Configurable and subject to real-world calibration.
        - min_amplitude (default 0.04): A provisional excursion floor between open EAR baseline
          and closed nadir. Configurable and subject to real-world calibration.
        """
        try:
            val_prolonged = float(prolonged_duration_ms)
            val_microsleep = float(microsleep_duration_ms)
        except (TypeError, ValueError) as err:
            raise ValueError(f"Invalid duration thresholds: {err}") from err

        if not (math.isfinite(val_prolonged) and val_prolonged > 0.0):
            raise ValueError(
                f"prolonged_duration_ms must be a finite positive number, got {prolonged_duration_ms}"
            )
        if not (math.isfinite(val_microsleep) and val_microsleep > val_prolonged):
            raise ValueError(
                f"microsleep_duration_ms ({microsleep_duration_ms}) must be finite and strictly "
                f"greater than prolonged_duration_ms ({prolonged_duration_ms})"
            )

        self.closure_threshold = closure_threshold
        self.reopen_threshold = reopen_threshold
        self.min_blink_duration_ms = min_blink_duration_ms
        self.prolonged_duration_ms = val_prolonged
        self.microsleep_duration_ms = val_microsleep
        self.min_amplitude = min_amplitude
        self.max_sampling_gap_ms = max_sampling_gap_ms

        self._state = BlinkState.OPEN
        self._prev_ts: Optional[float] = None
        self._prev_ear: Optional[float] = None

        # Active in-flight blink tracking
        self._blink_start_ts: Optional[float] = None
        self._closure_start_ts: Optional[float] = None
        self._min_ear: float = 1.0
        self._min_ear_ts: Optional[float] = None
        self._open_ear_before: float = 0.30
        self._was_left_closed: bool = False
        self._was_right_closed: bool = False
        self._is_left_censored: bool = False

        # Totals and historical metrics
        self._total_blinks: int = 0
        self._prolonged_blinks: int = 0
        self._recent_duration_ms: Optional[float] = None
        self._recent_rcr: Optional[float] = None

    def update_thresholds(self, closure_threshold: float, reopen_threshold: float) -> None:
        """
        Dynamically update closure and reopen thresholds from calibrated or rolling baseline.
        """
        self.closure_threshold = float(closure_threshold)
        self.reopen_threshold = float(reopen_threshold)

    def _interpolate_crossing(
        self,
        t_prev: float,
        ear_prev: float,
        t_curr: float,
        ear_curr: float,
        threshold: float,
    ) -> float:
        """
        Linearly interpolate exact sub-frame timestamp where EAR crossed threshold:
        t* = t_prev + (t_curr - t_prev) * (threshold - ear_prev) / (ear_curr - ear_prev)
        """
        denom = ear_curr - ear_prev
        if abs(denom) < 1e-6:
            return t_curr
        ratio = (threshold - ear_prev) / denom
        ratio = max(0.0, min(1.0, ratio))
        return t_prev + (t_curr - t_prev) * ratio

    def update(
        self,
        left_ear: Optional[float],
        right_ear: Optional[float],
        ear: Optional[float] = None,
        timestamp_ms: float = 0.0,
        has_face: bool = True,
        compensated_ear: Optional[float] = None,
        left_valid: bool = True,
        right_valid: bool = True,
    ) -> BlinkSnapshot:
        """
        Process a new frame measurement and update blink kinematics state machine.
        
        Args:
            left_ear: Left eye aspect ratio.
            right_ear: Right eye aspect ratio.
            ear: Uncompensated, perspective-weighted EAR measurement.
            timestamp_ms: Monotonic frame timestamp in milliseconds.
            has_face: Whether face tracking and eye geometry are valid.
            compensated_ear: Legacy keyword alias for `ear` (backward compatibility).
            left_valid: Whether left eye landmarks are valid.
            right_valid: Whether right eye landmarks are valid.
        """
        # Resolve uncompensated EAR with backward-compatible keyword alias
        ear_val = ear if ear is not None else (compensated_ear if compensated_ear is not None else 0.0)

        # 1. Input Validation: Guard against NaN, Inf, and negative values
        if (
            math.isnan(ear_val)
            or math.isinf(ear_val)
            or ear_val < 0.0
            or math.isnan(timestamp_ms)
            or math.isinf(timestamp_ms)
            or timestamp_ms < 0.0
            or (left_ear is not None and (math.isnan(left_ear) or math.isinf(left_ear)))
            or (right_ear is not None and (math.isnan(right_ear) or math.isinf(right_ear)))
        ):
            # Degenerate input: do not corrupt internal state
            is_currently_closed = self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE)
            cur_dur = (
                timestamp_ms - self._closure_start_ts
                if (self._closure_start_ts is not None and is_currently_closed)
                else 0.0
            )
            return BlinkSnapshot(
                state=self._state,
                is_eye_closed=is_currently_closed,
                current_closure_duration_ms=cur_dur,
                completed_blink=None,
                recent_blink_duration_ms=self._recent_duration_ms,
                recent_blink_rcr=self._recent_rcr,
                total_blink_count=self._total_blinks,
                prolonged_blink_count=self._prolonged_blinks,
            )

        # 2. Face Loss Handling
        if not has_face or (not left_valid and not right_valid):
            self._handle_face_lost()
            return BlinkSnapshot(
                state=self._state,
                is_eye_closed=False,
                current_closure_duration_ms=0.0,
                completed_blink=None,
                recent_blink_duration_ms=self._recent_duration_ms,
                recent_blink_rcr=self._recent_rcr,
                total_blink_count=self._total_blinks,
                prolonged_blink_count=self._prolonged_blinks,
            )

        # 3. Timestamp Monotonicity & Sampling Gap Handling
        if self._prev_ts is not None:
            gap_ms = timestamp_ms - self._prev_ts
            if gap_ms <= 0.0:
                # Duplicate or backwards timestamp: ignore without mutating state
                is_currently_closed = self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE)
                cur_dur = (
                    timestamp_ms - self._closure_start_ts
                    if (self._closure_start_ts is not None and is_currently_closed)
                    else 0.0
                )
                return BlinkSnapshot(
                    state=self._state,
                    is_eye_closed=is_currently_closed,
                    current_closure_duration_ms=cur_dur,
                    completed_blink=None,
                    recent_blink_duration_ms=self._recent_duration_ms,
                    recent_blink_rcr=self._recent_rcr,
                    total_blink_count=self._total_blinks,
                    prolonged_blink_count=self._prolonged_blinks,
                )

            if gap_ms > self.max_sampling_gap_ms:
                # Large sampling gap detected: invalidate in-flight waveform and record left-censored state
                self._handle_sampling_gap()
                self._prev_ts = timestamp_ms
                self._prev_ear = ear_val
                return BlinkSnapshot(
                    state=BlinkState.UNCERTAIN_GAP,
                    is_eye_closed=False,
                    current_closure_duration_ms=0.0,
                    completed_blink=None,
                    recent_blink_duration_ms=self._recent_duration_ms,
                    recent_blink_rcr=self._recent_rcr,
                    total_blink_count=self._total_blinks,
                    prolonged_blink_count=self._prolonged_blinks,
                )

        completed_blink: Optional[BlinkEvent] = None
        current_closure_duration_ms = 0.0

        # Threshold comparisons
        is_below_closed = ear_val < self.closure_threshold
        is_above_reopen = ear_val >= self.reopen_threshold

        # Track whether individual eyes reached valid overlapping closure during this in-flight event
        l_val = left_ear if left_ear is not None else ear_val
        r_val = right_ear if right_ear is not None else ear_val
        left_closed_now = bool(left_valid and (l_val < self.closure_threshold))
        right_closed_now = bool(right_valid and (r_val < self.closure_threshold))
        if left_valid and right_valid:
            valid_eye_closed = left_closed_now and right_closed_now
        elif left_valid:
            valid_eye_closed = left_closed_now
        elif right_valid:
            valid_eye_closed = right_closed_now
        else:
            valid_eye_closed = False

        if is_below_closed or self._state in (BlinkState.CLOSING, BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE):
            if left_closed_now and right_closed_now:
                self._was_left_closed = True
                self._was_right_closed = True

        # 4. State Machine Evaluation
        if self._state == BlinkState.UNCERTAIN_GAP:
            # Recovery from tracking loss or sampling gap
            if is_below_closed:
                # Discovered eye already closed or closure underway:
                # Track closure from recovery timestamp without backdating or fabricating duration
                self._state = BlinkState.CLOSED
                self._closure_start_ts = timestamp_ms
                self._blink_start_ts = None
                self._is_left_censored = True
                self._min_ear = ear_val
                self._min_ear_ts = timestamp_ms
                if left_closed_now and right_closed_now:
                    self._was_left_closed = True
                    self._was_right_closed = True
            elif is_above_reopen:
                self._state = BlinkState.OPEN
                self._is_left_censored = False
                self._closure_start_ts = None
                self._blink_start_ts = None
            else:
                # Hysteresis zone: safely transition to OPEN to prevent deadlock
                self._state = BlinkState.OPEN
                self._is_left_censored = False
                self._closure_start_ts = None
                self._blink_start_ts = None

        elif self._prev_ear is not None and self._prev_ts is not None:
            if self._state == BlinkState.OPEN:
                if is_below_closed:
                    # Direct drop below closed threshold in a single frame step
                    t_cross_close = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, ear_val, self.closure_threshold
                    )
                    t_cross_open = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, ear_val, self.reopen_threshold
                    )
                    self._blink_start_ts = t_cross_open
                    self._closure_start_ts = t_cross_close
                    self._open_ear_before = self._prev_ear
                    self._min_ear = ear_val
                    self._min_ear_ts = timestamp_ms
                    self._is_left_censored = False
                    self._state = BlinkState.CLOSED
                elif ear_val < self._prev_ear and ear_val < self.reopen_threshold:
                    # Starting to close through reopen threshold
                    t_cross_open = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, ear_val, self.reopen_threshold
                    )
                    self._blink_start_ts = t_cross_open
                    self._open_ear_before = self._prev_ear
                    self._is_left_censored = False
                    self._state = BlinkState.CLOSING

            elif self._state == BlinkState.CLOSING:
                if is_below_closed:
                    t_cross_close = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, ear_val, self.closure_threshold
                    )
                    self._closure_start_ts = t_cross_close
                    self._min_ear = ear_val
                    self._min_ear_ts = timestamp_ms
                    self._state = BlinkState.CLOSED
                elif ear_val >= self.reopen_threshold:
                    # Incomplete false closure, aborted
                    self._state = BlinkState.OPEN
                    self._blink_start_ts = None
                    self._closure_start_ts = None
                elif self._blink_start_ts is not None and (timestamp_ms - self._blink_start_ts) > 1000.0:
                    # Trapped in closing timeout (e.g. drooping eyelids)
                    self._state = BlinkState.OPEN
                    self._blink_start_ts = None
                    self._closure_start_ts = None

            elif self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE):
                if ear_val < self._min_ear:
                    self._min_ear = ear_val
                    self._min_ear_ts = timestamp_ms

                if self._closure_start_ts is not None:
                    current_closure_duration_ms = timestamp_ms - self._closure_start_ts

                # Check transition to prolonged closure / microsleep candidate
                if current_closure_duration_ms >= self.prolonged_duration_ms:
                    self._state = BlinkState.PROLONGED_CLOSURE

                # Check if reopening started
                if ear_val > self.closure_threshold:
                    self._state = BlinkState.REOPENING

            if self._state == BlinkState.REOPENING:
                if is_above_reopen:
                    # Eyelid has reopened above reopen_threshold
                    if self._is_left_censored:
                        # Left-censored event: onset was unobserved in gap.
                        # Do NOT emit a completed blink with fabricated duration.
                        self._state = BlinkState.OPEN
                        self._is_left_censored = False
                        self._blink_start_ts = None
                        self._closure_start_ts = None
                        self._min_ear = 1.0
                        self._min_ear_ts = None
                        self._was_left_closed = False
                        self._was_right_closed = False
                    else:
                        t_end = self._interpolate_crossing(
                            self._prev_ts, self._prev_ear, timestamp_ms, ear_val, self.reopen_threshold
                        )
                        start_ts = (
                            self._blink_start_ts
                            if self._blink_start_ts is not None
                            else (self._closure_start_ts if self._closure_start_ts is not None else self._prev_ts)
                        )
                        min_ts = self._min_ear_ts or (start_ts + (t_end - start_ts) * 0.5)

                        measured_duration_ms = t_end - start_ts
                        if measured_duration_ms <= 0.0:
                            # Non-positive interval: invalid timing, abort without creating a completed blink
                            classification = BlinkClassification.UNCERTAIN
                            completed_blink = None
                        else:
                            duration_ms = measured_duration_ms
                            amplitude = max(0.0, self._open_ear_before - self._min_ear)

                            # Compute kinematics without arbitrary duration clamping
                            close_dt_sec = (min_ts - start_ts) / 1000.0
                            reopen_dt_sec = (t_end - min_ts) / 1000.0

                            # Kinematic measurement reliability:
                            # Intervals <= 5 ms indicate insufficient camera sampling to resolve
                            # eyelid velocity physically. In such cases, mark kinematics unreliable
                            # and set velocities/RCR to 0.0 rather than exposing extreme/manufactured values.
                            if close_dt_sec <= 0.005 or reopen_dt_sec <= 0.005:
                                is_kinematics_reliable = False
                                v_close = 0.0
                                v_reopen = 0.0
                                rcr = 0.0
                            else:
                                is_kinematics_reliable = True
                                v_close = amplitude / close_dt_sec
                                v_reopen = amplitude / reopen_dt_sec
                                rcr = v_reopen / max(1e-6, v_close)

                            is_binocular = self._was_left_closed and self._was_right_closed

                            # Classify event against provisional noise thresholds
                            if duration_ms < self.min_blink_duration_ms or amplitude < self.min_amplitude:
                                classification = BlinkClassification.UNCERTAIN
                            elif duration_ms <= self.prolonged_duration_ms:
                                classification = BlinkClassification.NORMAL
                            elif duration_ms < self.microsleep_duration_ms:
                                classification = BlinkClassification.PROLONGED
                            else:
                                classification = BlinkClassification.MICROSLEEP_CANDIDATE

                            # Filter noise: UNCERTAIN flutter is suppressed from completed blink emission
                            if classification == BlinkClassification.UNCERTAIN:
                                completed_blink = None
                            else:
                                completed_blink = BlinkEvent(
                                    start_ts_ms=start_ts,
                                    min_ts_ms=min_ts,
                                    end_ts_ms=t_end,
                                    duration_ms=duration_ms,
                                    open_ear_before=self._open_ear_before,
                                    min_ear=self._min_ear,
                                    amplitude=amplitude,
                                    closing_velocity_per_s=v_close,
                                    reopening_velocity_per_s=v_reopen,
                                    rcr=rcr,
                                    classification=classification,
                                    is_binocular=is_binocular,
                                    is_kinematics_reliable=is_kinematics_reliable,
                                )

                                # Only valid binocular blinks increment binocular counters
                                if is_binocular:
                                    self._total_blinks += 1
                                    if classification in (
                                        BlinkClassification.PROLONGED,
                                        BlinkClassification.MICROSLEEP_CANDIDATE,
                                    ):
                                        self._prolonged_blinks += 1
                                    self._recent_duration_ms = duration_ms
                                    if is_kinematics_reliable:
                                        self._recent_rcr = rcr

                        # Reset state to open
                        self._state = BlinkState.OPEN
                        self._blink_start_ts = None
                        self._closure_start_ts = None
                        self._min_ear = 1.0
                        self._min_ear_ts = None
                        self._was_left_closed = False
                        self._was_right_closed = False

                elif ear_val < self.closure_threshold:
                    # Dropped back into closed
                    self._closure_start_ts = timestamp_ms
                    self._state = BlinkState.CLOSED

        self._prev_ts = timestamp_ms
        self._prev_ear = ear_val

        is_currently_closed = self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE)
        if self._closure_start_ts is not None and is_currently_closed:
            current_closure_duration_ms = timestamp_ms - self._closure_start_ts

        return BlinkSnapshot(
            state=self._state,
            is_eye_closed=is_currently_closed and valid_eye_closed,
            current_closure_duration_ms=current_closure_duration_ms,
            completed_blink=completed_blink,
            recent_blink_duration_ms=self._recent_duration_ms,
            recent_blink_rcr=self._recent_rcr,
            total_blink_count=self._total_blinks,
            prolonged_blink_count=self._prolonged_blinks,
        )

    def _handle_face_lost(self) -> None:
        """Cancel in-flight blink on face loss to avoid false prolonged closure."""
        self._state = BlinkState.UNCERTAIN_GAP
        self._blink_start_ts = None
        self._closure_start_ts = None
        self._min_ear = 1.0
        self._min_ear_ts = None
        self._was_left_closed = False
        self._was_right_closed = False
        self._prev_ts = None
        self._prev_ear = None
        self._is_left_censored = True

    def _handle_sampling_gap(self) -> None:
        """Cancel in-flight blink if camera or frame pipeline suffered a sampling drop."""
        self._state = BlinkState.UNCERTAIN_GAP
        self._blink_start_ts = None
        self._closure_start_ts = None
        self._min_ear = 1.0
        self._min_ear_ts = None
        self._was_left_closed = False
        self._was_right_closed = False
        self._prev_ts = None
        self._prev_ear = None
        self._is_left_censored = True

    def reset(self) -> None:
        """Reset state for a new session."""
        self._state = BlinkState.OPEN
        self._prev_ts = None
        self._prev_ear = None
        self._blink_start_ts = None
        self._closure_start_ts = None
        self._min_ear = 1.0
        self._min_ear_ts = None
        self._was_left_closed = False
        self._was_right_closed = False
        self._is_left_censored = False
        self._total_blinks = 0
        self._prolonged_blinks = 0
        self._recent_duration_ms = None
        self._recent_rcr = None
