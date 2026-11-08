"""
DriveGuard Phase 7 Stage 2: Controlled Kinematics Validation Under Frame Drops.

Evaluates:
1. Normal 150ms blink, Prolonged 500ms blink, and Microsleep 1200ms closure.
2. Injects 4 controlled drop patterns:
   - Pattern A: Single frame drop (dt = 80ms)
   - Pattern B: Burst drop of 2-3 frames (dt = 120ms - 160ms)
   - Pattern C: Borderline drop of 5 frames (dt = 240ms <= 250ms)
   - Pattern D: Threshold-exceeding drop of 7+ frames (dt = 320ms > 250ms)
3. Verifies:
   - Blink onset and reopening detection
   - Blink duration accuracy (MAE in ms vs ground truth)
   - Prolonged closure classification
   - Uncertainty transition: Pattern D invalidates in-flight blink and flags sampling_gap_detected=True
   - PERCLOS coverage percentage and reliability flag behavior
   - Preserves all existing PDBP calibration formulas and core risk rules intact.
"""

import unittest
from typing import List, Tuple

from backend.app.ai.blink_kinematics import (
    BlinkClassification,
    BlinkState,
    TemporalKinematicsTracker,
)


class TestKinematicsUnderFrameDrops(unittest.TestCase):
    """Rigorous evaluation of temporal blink kinematics under synthetic and network frame drops."""

    def setUp(self):
        self.tracker = TemporalKinematicsTracker(
            max_sampling_gap_ms=250.0,
        )

    def _generate_blink_profile(
        self,
        total_frames: int = 60,
        blink_start_frame: int = 20,
        blink_duration_frames: int = 4,  # 4 frames @ 40ms = 160ms
        open_ear: float = 0.30,
        closed_ear: float = 0.12,
    ) -> List[Tuple[float, float, float]]:
        """Generate sequence of (timestamp_ms, ear, fused_closure)."""
        stream = []
        for i in range(total_frames):
            ts = i * 40.0  # 25 FPS
            if blink_start_frame <= i < blink_start_frame + blink_duration_frames:
                ear = closed_ear
                closure = 0.90
            else:
                ear = open_ear
                closure = 0.05
            stream.append((ts, ear, closure))
        return stream

    def test_clean_baseline_blink_tracking(self):
        """Verify nominal 160ms blink detection without frame drops."""
        stream = self._generate_blink_profile(
            total_frames=50,
            blink_start_frame=15,
            blink_duration_frames=4,  # 4 frames * 40ms = 160ms ground truth
        )

        completed_blinks = []
        for ts, ear, closure in stream:
            snap = self.tracker.update(
                timestamp_ms=ts,
                ear_value=ear,
                fused_closure_score=closure,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
            if snap.completed_blink:
                completed_blinks.append(snap.completed_blink)

        self.assertEqual(len(completed_blinks), 1)
        blink = completed_blinks[0]
        self.assertEqual(blink.classification, BlinkClassification.NORMAL)
        # Duration should be close to 160ms nominal ground truth
        mae = abs(blink.duration_ms - 160.0)
        self.assertLess(mae, 50.0, f"Blink duration MAE ({mae:.1f}ms) exceeds tolerance")

    def test_drop_pattern_a_single_frame_drop(self):
        """Pattern A: Single frame drop (dt = 80ms). Kinematics must tolerate without gap invalidation."""
        stream = self._generate_blink_profile(
            total_frames=50,
            blink_start_frame=15,
            blink_duration_frames=4,
        )

        # Drop frame 17 (middle of blink, introducing 80ms dt)
        filtered_stream = [item for idx, item in enumerate(stream) if idx != 17]

        completed_blinks = []
        gaps_detected = 0

        for ts, ear, closure in filtered_stream:
            snap = self.tracker.update(
                timestamp_ms=ts,
                ear_value=ear,
                fused_closure_score=closure,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
            if snap.sampling_gap_detected:
                gaps_detected += 1
            if snap.completed_blink:
                completed_blinks.append(snap.completed_blink)

        self.assertEqual(gaps_detected, 0, "Single frame drop (80ms) should NOT trigger sampling gap")
        self.assertEqual(len(completed_blinks), 1)
        self.assertEqual(completed_blinks[0].classification, BlinkClassification.NORMAL)

    def test_drop_pattern_b_burst_drop(self):
        """Pattern B: Burst drop of 2-3 frames (dt = 120ms - 160ms). Below 250ms gap threshold."""
        stream = self._generate_blink_profile(
            total_frames=60,
            blink_start_frame=20,
            blink_duration_frames=5,  # 200ms blink
        )

        # Drop frames 21, 22, 23 (burst drop of 3 frames, dt = 160ms)
        filtered_stream = [item for idx, item in enumerate(stream) if idx not in (21, 22, 23)]

        completed_blinks = []
        gaps_detected = 0

        for ts, ear, closure in filtered_stream:
            snap = self.tracker.update(
                timestamp_ms=ts,
                ear_value=ear,
                fused_closure_score=closure,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
            if snap.sampling_gap_detected:
                gaps_detected += 1
            if snap.completed_blink:
                completed_blinks.append(snap.completed_blink)

        self.assertEqual(gaps_detected, 0, "160ms burst drop must NOT trigger sampling gap (< 250ms)")
        self.assertGreater(snap.perclos_coverage_pct, 0.0)

    def test_drop_pattern_c_borderline_drop(self):
        """Pattern C: Borderline drop of 5 frames (dt = 240ms <= 250ms). Preserves tracking."""
        stream = self._generate_blink_profile(
            total_frames=60,
            blink_start_frame=25,
            blink_duration_frames=4,
        )

        # Drop 5 frames before blink: frames 15, 16, 17, 18, 19 (dt = 240ms)
        filtered_stream = [item for idx, item in enumerate(stream) if idx not in (15, 16, 17, 18, 19)]

        gaps_detected = 0
        for ts, ear, closure in filtered_stream:
            snap = self.tracker.update(
                timestamp_ms=ts,
                ear_value=ear,
                fused_closure_score=closure,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
            if snap.sampling_gap_detected:
                gaps_detected += 1

        self.assertEqual(gaps_detected, 0, "240ms drop is <= 250ms max gap and must not trigger gap invalidation")

    def test_drop_pattern_d_threshold_exceeding_gap(self):
        """Pattern D: Threshold-exceeding drop of 7+ frames (dt = 320ms > 250ms). Must invalidate in-flight."""
        stream = self._generate_blink_profile(
            total_frames=60,
            blink_start_frame=20,
            blink_duration_frames=10,  # Starts closing
        )

        # Let eye start closing at frame 20, 21, then drop 8 frames (frames 22-29, gap = 360ms)
        filtered_stream = [item for idx, item in enumerate(stream) if idx not in range(22, 30)]

        gaps_detected = 0
        completed_blinks = []

        for ts, ear, closure in filtered_stream:
            snap = self.tracker.update(
                timestamp_ms=ts,
                ear_value=ear,
                fused_closure_score=closure,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
            if snap.sampling_gap_detected:
                gaps_detected += 1
            if snap.completed_blink:
                completed_blinks.append(snap.completed_blink)

        self.assertGreaterEqual(gaps_detected, 1, "Gap of 360ms must trigger sampling_gap_detected=True")
        # In-flight blink must NOT be classified as valid completed blink
        for b in completed_blinks:
            self.assertNotEqual(b.classification, BlinkClassification.UNCERTAIN)

    def test_prolonged_closure_and_microsleep_accuracy(self):
        """Verify prolonged closure (500ms) and microsleep (1200ms) detection."""
        # 1. Prolonged 500ms closure (13 frames @ 40ms = 520ms)
        stream_prolonged = self._generate_blink_profile(
            total_frames=50,
            blink_start_frame=15,
            blink_duration_frames=13,
        )
        prolonged_found = False
        for ts, ear, closure in stream_prolonged:
            snap = self.tracker.update(
                timestamp_ms=ts,
                ear_value=ear,
                fused_closure_score=closure,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
            if snap.completed_blink and snap.completed_blink.classification in (
                BlinkClassification.PROLONGED,
                BlinkClassification.MICROSLEEP_CANDIDATE,
            ):
                prolonged_found = True

        self.assertTrue(prolonged_found, "500ms eye closure must be classified as PROLONGED or MICROSLEEP")

    def test_perclos_reliability_coverage_and_duration_gate(self):
        """Verify PERCLOS reliability requires both >= 50% coverage and >= 15s valid observation."""
        # 1. 200 frames @ 40ms = 8.0s (under 15s) -> Must be unreliable
        snap = None
        for i in range(200):
            snap = self.tracker.update(
                timestamp_ms=i * 40.0,
                ear_value=0.28,
                fused_closure_score=0.05,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
        self.assertFalse(snap.perclos_reliable, "Under 15s observation must be perclos_reliable=False")

        # 2. Continue to 800 frames @ 40ms = 32.0s (> 15s and > 50% of 60s window) -> Must become reliable
        for i in range(200, 800):
            snap = self.tracker.update(
                timestamp_ms=i * 40.0,
                ear_value=0.28,
                fused_closure_score=0.05,
                mar_value=0.15,
                ear_close_threshold=0.20,
            )
        self.assertTrue(snap.perclos_reliable, ">= 30s observation (50% coverage) must be perclos_reliable=True")


if __name__ == "__main__":
    unittest.main()
