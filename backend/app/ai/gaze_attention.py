"""
Euro NCAP 2026 Gaze Attention, Visual Scanpath Entropy, and Dwell Time Watchdog.

Implements:
1. First-Order Transition Gaze Entropy (H_t):
   - Computes Markovian transition probabilities across driver attention zones
     (CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD, OUT_OF_BOUNDS).
   - Quantifies visual scanning breadth vs. cognitive tunneling / highway hypnosis.
     Healthy attentive driving: H_t in [0.9, 1.8].
     Fatigued / cognitive tunnel vision: H_t < 0.35 (rigid staring).

2. Euro NCAP 2026 Dwell Time Watchdog:
   - Long Off-Road Gaze Dwell: continuous non-center gaze >= 3.0s (or >= 2.0s high sensitivity).
   - Cognitive Tunneling / Blank Stare: continuous forward gaze >= 5.0s with low saccadic variance.

3. Percentage Road Center Fixation (PRCF):
   - Rolling percentage of observation time spent looking at forward roadway.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Deque, Dict, List, Optional, Tuple
import numpy as np

from backend.app.ai.geometry import PoseZone


@dataclass
class GazeAttentionSnapshot:
    """Current rolling visual attention metrics for frame t."""

    timestamp_ms: float
    current_zone: PoseZone
    current_zone_dwell_ms: float
    transition_entropy_ht: float
    prcf_pct: float  # Percentage Road Center Fixation (%)
    is_off_road_dwell_exceeded: bool
    is_blank_stare_detected: bool
    off_road_dwell_threshold_ms: float = 3000.0
    recent_zone_transitions_count: int = 0


class GazeAttentionTracker:
    """
    Per-session stateful tracker for Euro NCAP 2026 visual attention,
    dwell time limits, and Markov transition entropy.
    """

    ZONES = (
        PoseZone.CENTER,
        PoseZone.LEFT_MIRROR,
        PoseZone.RIGHT_MIRROR,
        PoseZone.DASHBOARD,
        PoseZone.OUT_OF_BOUNDS,
    )

    def __init__(
        self,
        window_duration_sec: float = 30.0,
        off_road_dwell_threshold_ms: float = 3000.0,
        blank_stare_dwell_threshold_ms: float = 5000.0,
        saccade_threshold_deg: float = 2.0,
    ) -> None:
        self.window_duration_ms = window_duration_sec * 1000.0
        self.off_road_dwell_threshold_ms = off_road_dwell_threshold_ms
        self.blank_stare_dwell_threshold_ms = blank_stare_dwell_threshold_ms
        self.saccade_threshold_deg = saccade_threshold_deg

        # Zone history tuples: (timestamp_ms, dt_ms, zone)
        self._zone_samples: Deque[Tuple[float, float, PoseZone]] = deque()
        # Discrete zone transition history tuples: (timestamp_ms, from_zone, to_zone)
        self._transitions: Deque[Tuple[float, PoseZone, PoseZone]] = deque(maxlen=200)

        self._prev_ts_ms: Optional[float] = None
        self._current_zone: PoseZone = PoseZone.CENTER
        self._zone_enter_ts_ms: float = 0.0

        # Head/Gaze pose micro-movement tracking for blank stare detection
        self._center_poses: Deque[Tuple[float, float, float]] = deque()  # (ts, pitch, yaw)

    def reset(self) -> None:
        """Clear state history."""
        self._zone_samples.clear()
        self._transitions.clear()
        self._center_poses.clear()
        self._prev_ts_ms = None
        self._current_zone = PoseZone.CENTER
        self._zone_enter_ts_ms = 0.0

    def update(
        self,
        timestamp_ms: float,
        zone: PoseZone,
        pitch_deg: float = 0.0,
        yaw_deg: float = 0.0,
        dt_ms: Optional[float] = None,
    ) -> GazeAttentionSnapshot:
        """
        Ingest current frame's gaze zone and return updated attention snapshot.
        """
        if dt_ms is None:
            if self._prev_ts_ms is not None:
                step_dt = max(1.0, min(250.0, timestamp_ms - self._prev_ts_ms))
            else:
                step_dt = 33.3
        else:
            step_dt = max(1.0, dt_ms)

        # 1. Detect zone transitions
        if zone != self._current_zone:
            if self._prev_ts_ms is not None:
                self._transitions.append((timestamp_ms, self._current_zone, zone))
            self._current_zone = zone
            self._zone_enter_ts_ms = timestamp_ms
            self._center_poses.clear()

        dwell_ms = max(0.0, timestamp_ms - self._zone_enter_ts_ms)

        # 2. Append rolling sample
        self._zone_samples.append((timestamp_ms, step_dt, zone))
        cutoff_window = timestamp_ms - self.window_duration_ms

        while self._zone_samples and self._zone_samples[0][0] < cutoff_window:
            self._zone_samples.popleft()
        while self._transitions and self._transitions[0][0] < cutoff_window:
            self._transitions.popleft()

        # 3. Percentage Road Center Fixation (PRCF)
        total_time_ms = sum(s[1] for s in self._zone_samples)
        center_time_ms = sum(s[1] for s in self._zone_samples if s[2] == PoseZone.CENTER)
        prcf_pct = (center_time_ms / total_time_ms * 100.0) if total_time_ms > 0 else 100.0

        # 4. Euro NCAP Off-Road Dwell Limit Check
        # Non-center zones sustained beyond threshold constitute distraction
        is_off_road = zone != PoseZone.CENTER
        is_off_road_dwell_exceeded = is_off_road and (dwell_ms >= self.off_road_dwell_threshold_ms)

        # 5. Euro NCAP Blank Stare / Cognitive Tunneling Check
        # Driver is staring rigidly forward for >= blank_stare_dwell_threshold_ms with near-zero angular variation
        is_blank_stare = False
        if zone == PoseZone.CENTER:
            self._center_poses.append((timestamp_ms, pitch_deg, yaw_deg))
            cutoff_5s = timestamp_ms - self.blank_stare_dwell_threshold_ms
            while self._center_poses and self._center_poses[0][0] < cutoff_5s:
                self._center_poses.popleft()

            if dwell_ms >= self.blank_stare_dwell_threshold_ms and len(self._center_poses) >= 15:
                pitches = [p[1] for p in self._center_poses]
                yaws = [p[2] for p in self._center_poses]
                pitch_std = float(np.std(pitches))
                yaw_std = float(np.std(yaws))
                # Rigid stare: standard deviation of head/gaze is virtually stationary (< 0.8 deg)
                if pitch_std < 0.80 and yaw_std < 0.80:
                    is_blank_stare = True
        else:
            self._center_poses.clear()

        # 6. First-Order Transition Gaze Entropy (H_t)
        ht = self._compute_transition_entropy()

        self._prev_ts_ms = timestamp_ms

        return GazeAttentionSnapshot(
            timestamp_ms=timestamp_ms,
            current_zone=zone,
            current_zone_dwell_ms=dwell_ms,
            transition_entropy_ht=round(ht, 3),
            prcf_pct=round(prcf_pct, 1),
            is_off_road_dwell_exceeded=is_off_road_dwell_exceeded,
            is_blank_stare_detected=is_blank_stare,
            off_road_dwell_threshold_ms=self.off_road_dwell_threshold_ms,
            recent_zone_transitions_count=len(self._transitions),
        )

    def _compute_transition_entropy(self) -> float:
        """
        Calculate Shannon entropy of the first-order Markov transition matrix:
          H_t = - sum_i p(i) sum_j P(j | i) log2 P(j | i)
        Returns a value in [0.0, log2(N_zones)].
        """
        if len(self._transitions) < 2:
            # Baseline resting entropy for forward attentive driving
            return 1.20

        zone_indices = {z: i for i, z in enumerate(self.ZONES)}
        n_zones = len(self.ZONES)
        counts = np.zeros((n_zones, n_zones), dtype=np.float64)

        for _, z_from, z_to in self._transitions:
            if z_from in zone_indices and z_to in zone_indices:
                counts[zone_indices[z_from], zone_indices[z_to]] += 1.0

        # Row sums
        row_sums = counts.sum(axis=1, keepdims=True)
        # Avoid division by zero: only compute for active departure zones
        active_rows = (row_sums.squeeze() > 0)
        if not np.any(active_rows):
            return 1.20

        # Marginal distribution p(i) over transitions
        total_transitions = counts.sum()
        p_i = counts.sum(axis=1) / max(total_transitions, 1e-4)

        # Transition matrix P(j | i)
        p_j_given_i = np.zeros_like(counts)
        for i in range(n_zones):
            if row_sums[i, 0] > 0:
                p_j_given_i[i, :] = counts[i, :] / row_sums[i, 0]

        # Conditional entropy H(S_t | S_{t-1})
        h_t = 0.0
        for i in range(n_zones):
            if p_i[i] > 1e-6:
                for j in range(n_zones):
                    prob = p_j_given_i[i, j]
                    if prob > 1e-6:
                        h_t -= p_i[i] * prob * np.log2(prob)

        return float(max(0.0, min(np.log2(n_zones), h_t)))
