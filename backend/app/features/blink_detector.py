"""
DriveGuard — Temporal Blink Kinematics & Sub-Frame Event Detector (Phase 5).

1. What it does:
   Detects blink events, tracks eyelid closing/reopening waveforms using a formal
   state machine, applies sub-frame linear crossing interpolation, and calculates
   vital kinematics (closing velocity, reopening velocity, and RCR).

2. Why it exists:
   Blink duration and velocity are critical physiological indicators of drowsiness:
   - Alert blinks: Snappy, short (100–250 ms), with high reopening-to-closing ratio (RCR >= 0.85).
   - Drowsy blinks: Sluggish eyelid reopening (prolonged duration > 350 ms, low RCR < 0.60).
   - Microsleeps: Unbroken closure >= 800 ms.
   Sub-frame interpolation prevents quantisation error from discrete frame rates.

3. What data it receives:
   - Left, right, and compensated EAR measurements.
   - Millisecond frame timestamp.
   - Face tracking state (`has_face`).

4. What data it produces:
   - `BlinkSnapshot` containing current blink state, active closure duration,
     completed `BlinkEvent` (if any), recent kinematics (duration, RCR), and totals.

5. Modules that depend on it:
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine (Phase 6)

6. Failure behavior:
   - Face loss or sampling gap (> 250 ms) invalidates in-flight blinks to `UNCERTAIN_GAP`
     to avoid fabricating artificial multi-second closures.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
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
    NORMAL = "NORMAL"                         # 70ms <= duration <= 350ms
    PROLONGED = "PROLONGED"                   # 350ms < duration < 800ms
    MICROSLEEP_CANDIDATE = "MICROSLEEP"       # Unbroken closure >= 800ms
    UNCERTAIN = "UNCERTAIN"                   # Incomplete or contaminated by gap


@dataclass(frozen=True)
class BlinkEvent:
    """
    Completed blink event with kinematic measurements.
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


class BlinkKinematicsDetector:
    """
    Stateful detector tracking eyelid kinetics with sub-frame crossing interpolation.
    """

    def __init__(
        self,
        closure_threshold: float = 0.20,
        reopen_threshold: float = 0.23,
        min_blink_duration_ms: float = 60.0,
        prolonged_duration_ms: float = 350.0,
        microsleep_duration_ms: float = 800.0,
        max_sampling_gap_ms: float = 250.0,
    ) -> None:
        self.closure_threshold = closure_threshold
        self.reopen_threshold = reopen_threshold
        self.min_blink_duration_ms = min_blink_duration_ms
        self.prolonged_duration_ms = prolonged_duration_ms
        self.microsleep_duration_ms = microsleep_duration_ms
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

        # Totals and historical metrics
        self._total_blinks: int = 0
        self._prolonged_blinks: int = 0
        self._recent_duration_ms: Optional[float] = None
        self._recent_rcr: Optional[float] = None

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
        left_ear: float,
        right_ear: float,
        compensated_ear: float,
        timestamp_ms: float,
        has_face: bool = True,
    ) -> BlinkSnapshot:
        """
        Process a new frame measurement and update blink kinematics state machine.
        """
        completed_blink: Optional[BlinkEvent] = None
        current_closure_duration_ms = 0.0

        # 1. Face Loss or Sampling Gap Handling
        if not has_face:
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

        if self._prev_ts is not None:
            gap_ms = timestamp_ms - self._prev_ts
            if gap_ms > self.max_sampling_gap_ms:
                # Large sampling gap detected: invalidate in-flight waveform
                self._handle_sampling_gap()

        # 2. State Machine Evaluation
        is_below_closed = compensated_ear < self.closure_threshold
        is_above_reopen = compensated_ear >= self.reopen_threshold

        if self._prev_ear is not None and self._prev_ts is not None:
            if self._state == BlinkState.OPEN:
                if is_below_closed:
                    # Direct drop below closed threshold
                    t_cross = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, compensated_ear, self.closure_threshold
                    )
                    self._blink_start_ts = t_cross
                    self._closure_start_ts = t_cross
                    self._open_ear_before = self._prev_ear
                    self._min_ear = compensated_ear
                    self._min_ear_ts = timestamp_ms
                    self._state = BlinkState.CLOSED
                elif compensated_ear < self._prev_ear and compensated_ear < self.reopen_threshold:
                    # Starting to close
                    self._blink_start_ts = self._prev_ts
                    self._open_ear_before = self._prev_ear
                    self._state = BlinkState.CLOSING

            elif self._state == BlinkState.CLOSING:
                if is_below_closed:
                    t_cross = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, compensated_ear, self.closure_threshold
                    )
                    self._closure_start_ts = t_cross
                    self._min_ear = compensated_ear
                    self._min_ear_ts = timestamp_ms
                    self._state = BlinkState.CLOSED
                elif compensated_ear >= self.reopen_threshold:
                    # Incomplete false closure, aborted
                    self._state = BlinkState.OPEN

            elif self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE):
                if compensated_ear < self._min_ear:
                    self._min_ear = compensated_ear
                    self._min_ear_ts = timestamp_ms

                if self._closure_start_ts is not None:
                    current_closure_duration_ms = timestamp_ms - self._closure_start_ts

                # Check transition to prolonged closure / microsleep candidate
                if current_closure_duration_ms >= self.prolonged_duration_ms:
                    self._state = BlinkState.PROLONGED_CLOSURE

                # Check if reopening started
                if compensated_ear > self.closure_threshold:
                    self._state = BlinkState.REOPENING

            if self._state == BlinkState.REOPENING:
                if is_above_reopen:
                    # Blink completed!
                    t_end = self._interpolate_crossing(
                        self._prev_ts, self._prev_ear, timestamp_ms, compensated_ear, self.reopen_threshold
                    )
                    start_ts = self._blink_start_ts or self._prev_ts
                    min_ts = self._min_ear_ts or (start_ts + (t_end - start_ts) * 0.5)

                    duration_ms = max(1.0, t_end - start_ts)
                    amplitude = max(0.01, self._open_ear_before - self._min_ear)

                    # Compute kinematics
                    close_dt_sec = max(0.001, (min_ts - start_ts) / 1000.0)
                    reopen_dt_sec = max(0.001, (t_end - min_ts) / 1000.0)
                    v_close = amplitude / close_dt_sec
                    v_reopen = amplitude / reopen_dt_sec
                    rcr = v_reopen / max(1e-6, v_close)

                    # Classify duration
                    if duration_ms < self.min_blink_duration_ms:
                        classification = BlinkClassification.UNCERTAIN
                    elif duration_ms <= self.prolonged_duration_ms:
                        classification = BlinkClassification.NORMAL
                    elif duration_ms < self.microsleep_duration_ms:
                        classification = BlinkClassification.PROLONGED
                        self._prolonged_blinks += 1
                    else:
                        classification = BlinkClassification.MICROSLEEP_CANDIDATE
                        self._prolonged_blinks += 1

                    is_binocular = (left_ear < self.reopen_threshold) and (right_ear < self.reopen_threshold)

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
                    )

                    self._total_blinks += 1
                    self._recent_duration_ms = duration_ms
                    self._recent_rcr = rcr

                    # Reset to open
                    self._state = BlinkState.OPEN
                    self._blink_start_ts = None
                    self._closure_start_ts = None
                    self._min_ear = 1.0

                elif compensated_ear < self.closure_threshold:
                    # Dropped back into closed
                    self._state = BlinkState.CLOSED

            elif self._state == BlinkState.UNCERTAIN_GAP:
                if is_above_reopen:
                    self._state = BlinkState.OPEN

        self._prev_ts = timestamp_ms
        self._prev_ear = compensated_ear

        is_currently_closed = self._state in (BlinkState.CLOSED, BlinkState.PROLONGED_CLOSURE)
        if self._closure_start_ts is not None and is_currently_closed:
            current_closure_duration_ms = timestamp_ms - self._closure_start_ts

        return BlinkSnapshot(
            state=self._state,
            is_eye_closed=is_currently_closed,
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
        self._prev_ts = None
        self._prev_ear = None

    def _handle_sampling_gap(self) -> None:
        """Cancel in-flight blink if camera or frame pipeline suffered a sampling drop."""
        self._state = BlinkState.UNCERTAIN_GAP
        self._blink_start_ts = None
        self._closure_start_ts = None
        self._min_ear = 1.0

    def reset(self) -> None:
        """Reset state for a new session."""
        self._state = BlinkState.OPEN
        self._prev_ts = None
        self._prev_ear = None
        self._blink_start_ts = None
        self._closure_start_ts = None
        self._min_ear = 1.0
        self._min_ear_ts = None
        self._total_blinks = 0
        self._prolonged_blinks = 0
        self._recent_duration_ms = None
        self._recent_rcr = None
