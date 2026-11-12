"""
DriveGuard — Multi-Window Temporal Event Analyzer (Phase 6).

Aggregates instantaneous physiological observations across short (10s), intermediate (30s),
and long (60s) rolling windows to detect patterns of driver fatigue, attentional drift,
and acute drowsiness episodes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Optional, Tuple

from backend.app.features.blink_detector import BlinkEvent
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
    ) -> None:
        self.window_60s_ms = window_60s_ms
        self.window_30s_ms = window_30s_ms
        self.nodding_pitch_drop_deg = nodding_pitch_drop_deg
        self.nodding_min_closure_ms = nodding_min_closure_ms

        # Rolling Event Buffers
        self._blinks: Deque[BlinkEvent] = deque()
        self._yawns: Deque[YawnEvent] = deque()
        self._pitch_history: Deque[Tuple[float, float]] = deque()  # (ts_ms, rel_pitch_deg)

        # Active Temporal States
        self._closure_start_ts_ms: Optional[float] = None
        self._off_road_start_ts_ms: Optional[float] = None
        self._face_lost_start_ts_ms: Optional[float] = None
        self._last_processed_blink_end_ts: Optional[float] = None
        self._last_processed_yawn_end_ts: Optional[float] = None

    def update(self, features: DriverFeatures) -> TemporalMetrics:
        """
        Ingest current frame's DriverFeatures and return consolidated TemporalMetrics.
        """
        ts_ms = features.timestamp_ms

        # ------------------------------------------------------------------
        # 1. Face Loss Tracking
        # ------------------------------------------------------------------
        if not features.has_face:
            if self._face_lost_start_ts_ms is None:
                self._face_lost_start_ts_ms = ts_ms
            face_lost_ms = ts_ms - self._face_lost_start_ts_ms
            self._closure_start_ts_ms = None
            self._off_road_start_ts_ms = None
        else:
            self._face_lost_start_ts_ms = None
            face_lost_ms = 0.0

        # ------------------------------------------------------------------
        # 2. Continuous Eye Closure Tracking
        # ------------------------------------------------------------------
        if features.has_face and features.ear.is_valid and features.ear.is_closed:
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
        if features.has_face and features.head_pose.is_valid and zone != PoseZone.CENTER:
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
        # 4. Nodding Microsleep Detection
        # ------------------------------------------------------------------
        is_nodding_microsleep = False
        if features.has_face and features.head_pose.is_valid:
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
        if (
            completed_blink is not None
            and completed_blink.end_ts_ms != self._last_processed_blink_end_ts
        ):
            self._blinks.append(completed_blink)
            self._last_processed_blink_end_ts = completed_blink.end_ts_ms

        # Expire blinks older than 60s
        while self._blinks and (ts_ms - self._blinks[0].end_ts_ms) > self.window_60s_ms:
            self._blinks.popleft()

        cutoff_30s = ts_ms - self.window_30s_ms
        blinks_30s = [b for b in self._blinks if b.end_ts_ms >= cutoff_30s]

        blink_count_60s = len(self._blinks)
        blink_count_30s = len(blinks_30s)

        if blink_count_60s > 0:
            durations = [b.duration_ms for b in self._blinks]
            mean_dur_60s = float(sum(durations) / blink_count_60s)
            prolonged_count_60s = sum(1 for d in durations if d >= 300.0)
            severe_count_60s = sum(1 for d in durations if d >= 500.0)
            rcr_values = [b.rcr for b in self._blinks]
            mean_rcr_60s = float(sum(rcr_values) / blink_count_60s)
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
            features.mar.current_yawn_duration_ms if features.mar.is_yawn_in_progress else 0.0
        )

        return TemporalMetrics(
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
        )

    def reset(self) -> None:
        """Reset all temporal history for a fresh monitoring session."""
        self._blinks.clear()
        self._yawns.clear()
        self._pitch_history.clear()
        self._closure_start_ts_ms = None
        self._off_road_start_ts_ms = None
        self._face_lost_start_ts_ms = None
        self._last_processed_blink_end_ts = None
        self._last_processed_yawn_end_ts = None
