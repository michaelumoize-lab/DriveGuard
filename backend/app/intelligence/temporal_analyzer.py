"""
DriveGuard — Multi-Window Temporal Event Analyzer (Phase 6).

Aggregates instantaneous physiological observations across short (10s), intermediate (30s),
and long (60s) rolling windows to detect patterns of driver fatigue, attentional drift,
and acute drowsiness episodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
from typing import Deque, Optional, Tuple

from backend.app.features.blink_detector import BlinkClassification, BlinkEvent, BlinkState
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import PoseZone
from backend.app.features.mar import YawnEvent


@dataclass(frozen=True)
class TemporalMetrics:
    """
    Consolidated temporal indicators across all observation windows.
    """
    timestamp_ms: float
    active_closure_duration_ms: float
    active_off_road_dwell_ms: float
    is_mirror_check_context: bool
    is_dashboard_check_context: bool
    is_nodding_microsleep: bool
    
    # 60-Second & 30-Second Rolling Blink Kinematics
    blink_count_60s: int
    blink_count_30s: int
    mean_blink_duration_ms_60s: float
    prolonged_blink_count_60s: int
    severe_blink_count_60s: int
    mean_rcr_60s: float

    # Rolling Yawn Kinematics
    yawn_count_60s: int
    yawn_count_30s: int
    active_yawn_duration_ms: float

    # Cumulative PERCLOS & Sensor Continuity
    perclos_pct: float
    perclos_coverage_pct: float
    perclos_is_reliable: bool
    has_face: bool
    face_lost_duration_ms: float
    is_eye_closed: Optional[bool] = None
    completed_blink: Optional[BlinkEvent] = None
    downward_dwell_ms: float = 0.0

    @property
    def is_eye_verified_open(self) -> bool:
        """True only when eye tracking is valid and confirmed open."""
        return self.is_eye_closed is False

    @property
    def is_eye_verified_closed(self) -> bool:
        """True only when eye tracking is valid and confirmed closed."""
        return self.is_eye_closed is True

    @property
    def effective_perclos_pct(self) -> Optional[float]:
        """Gated PERCLOS percentage: returns perclos_pct if perclos_is_reliable, else None."""
        return self.perclos_pct if self.perclos_is_reliable else None


class TemporalAnalyzer:
    """
    Maintains rolling time-ordered histories of physiological events.
    """

    def __init__(
        self,
        window_60s_ms: float = 60_000.0,
        window_30s_ms: float = 30_000.0,
        nodding_pitch_drop_deg: float = -15.0,
        nodding_min_closure_ms: float = 800.0,
        max_sampling_gap_ms: float = 250.0,
    ) -> None:
        self.window_60s_ms = window_60s_ms
        self.window_30s_ms = window_30s_ms
        self.nodding_pitch_drop_deg = nodding_pitch_drop_deg
        self.nodding_min_closure_ms = nodding_min_closure_ms
        self.max_sampling_gap_ms = max_sampling_gap_ms

        # Rolling Event Buffers
        self._blinks: Deque[BlinkEvent] = deque()
        self._yawns: Deque[YawnEvent] = deque()
        self._pitch_history: Deque[Tuple[float, float]] = deque()  # (ts_ms, rel_pitch_deg)

        # Active Temporal States
        self._prev_ts_ms: Optional[float] = None
        self._closure_start_ts_ms: Optional[float] = None
        self._off_road_start_ts_ms: Optional[float] = None
        self._downward_start_ts_ms: Optional[float] = None
        self._face_lost_start_ts_ms: Optional[float] = None
        self._last_processed_blink_end_ts: Optional[float] = None
        self._last_processed_yawn_end_ts: Optional[float] = None
        self._current_closure_duration_ms: float = 0.0
        self._current_off_road_duration_ms: float = 0.0
        self._current_downward_dwell_ms: float = 0.0
        self._last_metrics: Optional[TemporalMetrics] = None

    @property
    def current_closure_duration_ms(self) -> float:
        """Active continuous eye-closure duration in milliseconds."""
        return self._current_closure_duration_ms

    @property
    def current_off_road_duration_ms(self) -> float:
        """Active off-road gaze dwell in milliseconds."""
        return self._current_off_road_duration_ms

    @property
    def current_downward_dwell_ms(self) -> float:
        """Active downward head posture dwell in milliseconds."""
        return self._current_downward_dwell_ms

    @property
    def pitch_history(self) -> Tuple[Tuple[float, float], ...]:
        """Snapshot of rolling pitch history tuples (ts_ms, rel_pitch_deg)."""
        return tuple(self._pitch_history)

    def _make_default_metrics(self) -> TemporalMetrics:
        """Construct safe neutral metrics for uninitialized or invalid frame updates."""
        return TemporalMetrics(
            timestamp_ms=0.0,
            active_closure_duration_ms=0.0,
            active_off_road_dwell_ms=0.0,
            is_mirror_check_context=False,
            is_dashboard_check_context=False,
            is_nodding_microsleep=False,
            blink_count_60s=0,
            blink_count_30s=0,
            mean_blink_duration_ms_60s=0.0,
            prolonged_blink_count_60s=0,
            severe_blink_count_60s=0,
            mean_rcr_60s=1.0,
            yawn_count_60s=0,
            yawn_count_30s=0,
            active_yawn_duration_ms=0.0,
            perclos_pct=0.0,
            perclos_coverage_pct=0.0,
            perclos_is_reliable=False,
            has_face=False,
            face_lost_duration_ms=0.0,
            downward_dwell_ms=0.0,
        )

    def update(self, features: DriverFeatures, is_reliable: bool = True) -> TemporalMetrics:
        """
        Ingest current frame's DriverFeatures and return consolidated TemporalMetrics.
        """
        ts_ms = features.timestamp_ms

        # ------------------------------------------------------------------
        # 0. Timestamp Validation & Integrity Guard
        # ------------------------------------------------------------------
        if not math.isfinite(ts_ms) or ts_ms < 0.0:
            # Reject non-finite or negative timestamps without mutating state
            return self._last_metrics or self._make_default_metrics()

        if self._prev_ts_ms is not None:
            dt_ms = ts_ms - self._prev_ts_ms
            if dt_ms <= 0.0:
                # Reject duplicate or backwards timestamps without mutating state
                return self._last_metrics or self._make_default_metrics()
        else:
            dt_ms = 0.0

        # Decouple actual camera sampling gap from ocular tracking gap
        is_camera_sampling_gap = (self._prev_ts_ms is not None) and (dt_ms > self.max_sampling_gap_ms)
        blink_gap_signal = getattr(features.blink, "sampling_gap_detected", False) or (
            getattr(features.blink, "state", None) == BlinkState.UNCERTAIN_GAP
        )
        is_ocular_gap = is_camera_sampling_gap or blink_gap_signal

        # ------------------------------------------------------------------
        # 1. Face Loss Tracking
        # ------------------------------------------------------------------
        if not features.has_face:
            if self._face_lost_start_ts_ms is None:
                self._face_lost_start_ts_ms = ts_ms
            face_lost_ms = ts_ms - self._face_lost_start_ts_ms
            self._closure_start_ts_ms = None
            self._off_road_start_ts_ms = None
            self._pitch_history.clear()
        else:
            self._face_lost_start_ts_ms = None
            face_lost_ms = 0.0

        # ------------------------------------------------------------------
        # 2. Continuous Eye Closure Tracking
        # ------------------------------------------------------------------
        if not is_reliable:
            is_eye_closed_val: Optional[bool] = None
            self._closure_start_ts_ms = None
            active_closure_ms = 0.0
        else:
            ear_valid = features.has_face and features.ear.is_valid
            blink_valid = (
                features.has_face
                and not features.blink.sampling_gap_detected
                and features.blink.state != BlinkState.UNCERTAIN_GAP
            )

            ear_closed = ear_valid and features.ear.is_closed
            blink_closed = blink_valid and features.blink.is_eye_closed

            if ear_closed or blink_closed:
                is_eye_closed_val = True
            elif ear_valid and blink_valid:
                is_eye_closed_val = False
            elif ear_valid and not features.ear.is_closed and not blink_valid:
                is_eye_closed_val = False
            elif blink_valid and not features.blink.is_eye_closed and not ear_valid:
                is_eye_closed_val = False
            else:
                is_eye_closed_val = None

            is_currently_closed = is_eye_closed_val is True

            if is_ocular_gap:
                # Camera sampling gap or ocular tracking uncertainty: unobserved or unverified
                # time must never be counted as closure. If the driver is closed on the first
                # post-gap observation, establish a new closure start at this frame without
                # backdating or carrying previous duration.
                if is_currently_closed:
                    self._closure_start_ts_ms = ts_ms
                else:
                    self._closure_start_ts_ms = None
                active_closure_ms = 0.0
            else:
                if is_currently_closed:
                    if self._closure_start_ts_ms is None:
                        self._closure_start_ts_ms = ts_ms
                    raw_closure_duration = ts_ms - self._closure_start_ts_ms
                    # Blend with kinematic detector's sub-frame interpolated closure duration
                    active_closure_ms = max(
                        raw_closure_duration,
                        features.blink.current_closure_duration_ms,
                    )
                else:
                    self._closure_start_ts_ms = None
                    active_closure_ms = 0.0

        # ------------------------------------------------------------------
        # 3. Off-Road Gaze Dwell & Cockpit Context
        # ------------------------------------------------------------------
        zone = features.head_pose.zone
        is_off_road = (
            features.has_face
            and features.head_pose.is_valid
            and zone not in (PoseZone.CENTER, PoseZone.UNKNOWN)
        )

        if is_camera_sampling_gap:
            # Actual camera sampling gap: unobserved camera time must never be counted as off-road gaze.
            # Ocular tracking gaps do NOT reset valid head-pose off-road dwell.
            if is_off_road:
                self._off_road_start_ts_ms = ts_ms
            else:
                self._off_road_start_ts_ms = None
            active_off_road_ms = 0.0
        else:
            if is_off_road:
                if self._off_road_start_ts_ms is None:
                    self._off_road_start_ts_ms = ts_ms
                active_off_road_ms = ts_ms - self._off_road_start_ts_ms
            else:
                self._off_road_start_ts_ms = None
                active_off_road_ms = 0.0

        # Euro NCAP / SAE J941 Mirror Glance Context
        is_mirror_check = (
            zone in (PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR)
            and active_off_road_ms <= 3000.0
        )
        is_dashboard_check = (
            zone == PoseZone.DASHBOARD
            and active_off_road_ms <= 2000.0
        )

        # ------------------------------------------------------------------
        # 3b. Downward Head Posture Dwell Tracking
        # ------------------------------------------------------------------
        # Tracks continuous downward posture (DASHBOARD zone, rel_pitch <= -12°, |rel_yaw| < 14°).
        # Camera sampling gaps reset/re-anchor downward dwell; ocular tracking gaps do NOT.
        is_downward_posture = (
            features.has_face
            and features.head_pose.is_valid
            and features.head_pose.zone == PoseZone.DASHBOARD
            and features.head_pose.rel_pitch_deg <= -12.0
            and abs(features.head_pose.rel_yaw_deg) < 14.0
        )

        if is_camera_sampling_gap:
            # Actual camera sampling gap: unobserved camera time must never be credited.
            # Ocular tracking gaps (e.g. UNCERTAIN_GAP) do NOT reset valid head-pose downward dwell.
            if is_downward_posture:
                self._downward_start_ts_ms = ts_ms
            else:
                self._downward_start_ts_ms = None
            downward_dwell_ms = 0.0
        else:
            if is_downward_posture:
                if self._downward_start_ts_ms is None:
                    self._downward_start_ts_ms = ts_ms
                downward_dwell_ms = ts_ms - self._downward_start_ts_ms
            else:
                self._downward_start_ts_ms = None
                downward_dwell_ms = 0.0

        self._current_downward_dwell_ms = downward_dwell_ms

        # ------------------------------------------------------------------
        # 4. Nodding Microsleep Detection
        # ------------------------------------------------------------------
        # Pitch history is cleared on actual camera sampling gaps, face loss,
        # or invalid/UNKNOWN head pose. Ocular tracking gaps (e.g. unmeasurable EAR)
        # or overall confidence degradation do NOT clear valid head-pose pitch history.
        is_pose_invalid = not features.has_face or not features.head_pose.is_valid or features.head_pose.zone == PoseZone.UNKNOWN
        if is_camera_sampling_gap or is_pose_invalid:
            self._pitch_history.clear()

        is_nodding_microsleep = False
        if features.has_face and features.head_pose.is_valid and features.head_pose.zone != PoseZone.UNKNOWN:
            rel_pitch = features.head_pose.rel_pitch_deg
            self._pitch_history.append((ts_ms, rel_pitch))
            # Expire pitch older than 2.5s
            while self._pitch_history and (ts_ms - self._pitch_history[0][0]) > 2500.0:
                self._pitch_history.popleft()

            # Nodding condition 1: Sustained closure + downward slump
            if (
                active_closure_ms >= self.nodding_min_closure_ms
                and rel_pitch <= self.nodding_pitch_drop_deg
            ):
                is_nodding_microsleep = True

            # Nodding condition 2: Sudden downward pitch drop of > 16° while eyes are closed
            if len(self._pitch_history) >= 2 and active_closure_ms >= 500.0:
                max_pitch_recent = max(p for _, p in self._pitch_history)
                if (max_pitch_recent - rel_pitch) >= 16.0 and rel_pitch <= -12.0:
                    is_nodding_microsleep = True

        # ------------------------------------------------------------------
        # 5. Blink History Accumulation & Kinematic Aggregation
        # ------------------------------------------------------------------
        completed_blink = features.blink.completed_blink
        new_completed_blink: Optional[BlinkEvent] = None
        if (
            completed_blink is not None
            and completed_blink.end_ts_ms != self._last_processed_blink_end_ts
        ):
            self._blinks.append(completed_blink)
            self._last_processed_blink_end_ts = completed_blink.end_ts_ms
            new_completed_blink = completed_blink

        # Expire blinks older than 60s
        while self._blinks and (ts_ms - self._blinks[0].end_ts_ms) > self.window_60s_ms:
            self._blinks.popleft()

        cutoff_30s = ts_ms - self.window_30s_ms
        blinks_30s = [b for b in self._blinks if b.end_ts_ms >= cutoff_30s]

        # Only valid binocular blinks contribute to rolling binocular fatigue statistics.
        # Unilateral events are preserved in self._blinks for diagnostics but excluded here.
        binocular_blinks_60s = [
            b for b in self._blinks
            if b.is_binocular and b.classification != BlinkClassification.UNCERTAIN
        ]
        binocular_blinks_30s = [
            b for b in blinks_30s
            if b.is_binocular and b.classification != BlinkClassification.UNCERTAIN
        ]

        blink_count_60s = len(binocular_blinks_60s)
        blink_count_30s = len(binocular_blinks_30s)

        if blink_count_60s > 0:
            durations = [b.duration_ms for b in binocular_blinks_60s]
            mean_dur_60s = float(sum(durations) / blink_count_60s)
            prolonged_count_60s = sum(
                1 for b in binocular_blinks_60s
                if b.classification in (
                    BlinkClassification.PROLONGED,
                    BlinkClassification.MICROSLEEP_CANDIDATE,
                )
            )
            severe_count_60s = sum(1 for d in durations if d >= 500.0)
            reliable_rcrs = [
                b.rcr for b in binocular_blinks_60s
                if getattr(b, "is_kinematics_reliable", True) and b.rcr > 0.0
            ]
            mean_rcr_60s = float(sum(reliable_rcrs) / len(reliable_rcrs)) if reliable_rcrs else 1.0
        else:
            mean_dur_60s = 0.0
            prolonged_count_60s = 0
            severe_count_60s = 0
            mean_rcr_60s = 1.0

        # ------------------------------------------------------------------
        # 6. Yawn History Accumulation
        # ------------------------------------------------------------------
        completed_yawn = features.mar.completed_yawn
        if (
            completed_yawn is not None
            and completed_yawn.end_ts_ms != self._last_processed_yawn_end_ts
        ):
            self._yawns.append(completed_yawn)
            self._last_processed_yawn_end_ts = completed_yawn.end_ts_ms

        # Expire yawns older than 60s
        while self._yawns and (ts_ms - self._yawns[0].end_ts_ms) > self.window_60s_ms:
            self._yawns.popleft()

        yawns_30s = [y for y in self._yawns if y.end_ts_ms >= cutoff_30s]
        yawn_count_60s = len(self._yawns)
        yawn_count_30s = len(yawns_30s)
        active_yawn_ms = (
            features.mar.observed_yawn_duration_ms if features.mar.is_yawn_in_progress else 0.0
        )

        self._current_closure_duration_ms = active_closure_ms
        self._current_off_road_duration_ms = active_off_road_ms
        self._prev_ts_ms = ts_ms

        metrics = TemporalMetrics(
            timestamp_ms=ts_ms,
            active_closure_duration_ms=active_closure_ms,
            active_off_road_dwell_ms=active_off_road_ms,
            is_mirror_check_context=is_mirror_check,
            is_dashboard_check_context=is_dashboard_check,
            is_nodding_microsleep=is_nodding_microsleep,
            blink_count_60s=blink_count_60s,
            blink_count_30s=blink_count_30s,
            mean_blink_duration_ms_60s=mean_dur_60s,
            prolonged_blink_count_60s=prolonged_count_60s,
            severe_blink_count_60s=severe_count_60s,
            mean_rcr_60s=mean_rcr_60s,
            yawn_count_60s=yawn_count_60s,
            yawn_count_30s=yawn_count_30s,
            active_yawn_duration_ms=active_yawn_ms,
            perclos_pct=features.perclos.perclos_pct,
            perclos_coverage_pct=features.perclos.coverage_pct,
            perclos_is_reliable=features.perclos.is_reliable,
            has_face=features.has_face,
            face_lost_duration_ms=face_lost_ms,
            is_eye_closed=is_eye_closed_val,
            completed_blink=new_completed_blink,
            downward_dwell_ms=downward_dwell_ms,
        )
        self._last_metrics = metrics
        return metrics

    def reset(self) -> None:
        """Reset all temporal history for a fresh monitoring session."""
        self._blinks.clear()
        self._yawns.clear()
        self._pitch_history.clear()
        self._prev_ts_ms = None
        self._closure_start_ts_ms = None
        self._off_road_start_ts_ms = None
        self._downward_start_ts_ms = None
        self._face_lost_start_ts_ms = None
        self._last_processed_blink_end_ts = None
        self._last_processed_yawn_end_ts = None
        self._current_closure_duration_ms = 0.0
        self._current_off_road_duration_ms = 0.0
        self._current_downward_dwell_ms = 0.0
        self._last_metrics = None
