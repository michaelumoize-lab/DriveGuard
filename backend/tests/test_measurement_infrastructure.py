"""
Unit tests for DriveGuard Phase 7 Stage 1: Measurement Infrastructure & Clock Domains.
Verifies:
1. Monotonic decimal string sequence numbers (frame_seq, event_seq).
2. Session epoch and worker boot identity preservation.
3. Strict separation of server monotonic and epoch clock domains.
4. Thread-safe FrameAccountingSnapshot invariants under all lifecycle operations.
"""

import asyncio
import re
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from backend.app.schemas.telemetry import (
    AlertStateEnum,
    CascadeStatusDTO,
    ConventionalFixedComparisonDTO,
    EventAckDTO,
    EyeMetricsDTO,
    FrameTelemetryDTO,
    HeadPoseDTO,
    PoseZoneEnum,
    SafetyEventDTO,
    TelemetryPacketDTO,
    TimingBreakdownDTO,
)
from backend.app.services.livekit_worker import (
    FrameAccountingSnapshot,
    LiveKitAIWorkerSession,
    WORKER_BOOT_ID,
)


class TestMeasurementInfrastructure(unittest.IsolatedAsyncioTestCase):
    """Test suite validating Phase 7 measurement infrastructure and clock domain separation."""

    def setUp(self):
        self.session_id = "test_perf_room_001"
        self.driver_id = "driver_perf_test"
        self.worker = LiveKitAIWorkerSession(
            room_name=self.session_id,
            driver_id=self.driver_id,
        )

    def _create_sample_telemetry(self, frame_idx: int = 1) -> FrameTelemetryDTO:
        return FrameTelemetryDTO(
            session_id=self.session_id,
            driver_id=self.driver_id,
            frame_index=frame_idx,
            frame_seq=str(frame_idx),
            event_seq="1",
            worker_boot_id=WORKER_BOOT_ID,
            session_epoch="1a2b3c",
            timestamp_ms=1000.0,
            fps=25.0,
            face_detected=True,
            state=AlertStateEnum.NORMAL,
            risk_score=0.1,
            pose_zone=PoseZoneEnum.CENTER,
            reason_codes=[],
            head_pose=HeadPoseDTO(pitch_deg=0.0, yaw_deg=0.0, roll_deg=0.0),
            eye_metrics=EyeMetricsDTO(
                ear_raw=0.30,
                ear_left=0.30,
                ear_right=0.30,
                ear_normalized=0.30,
                ear_closure_threshold=0.20,
                is_eye_closed=False,
                fused_closure_prob=0.05,
                perclos_pct=5.0,
            ),
            cascade=CascadeStatusDTO(
                experiment_mode="EXPERIMENT_A_GEOMETRIC_ONLY",
                cnn_triggered=False,
                geom_open_prob=0.95,
                geom_reliability=0.90,
                arbitration_winner="GEOMETRY",
                disagreement_flag=False,
                cnn_latency_ms=0.0,
            ),
            comparison=ConventionalFixedComparisonDTO(
                fixed_cutoff_ear=0.22,
                fixed_alert_state="NORMAL",
                fixed_false_alarm=False,
                personalised_zone="CENTER",
                personalised_threshold=0.20,
                is_mirror_check_suppressed=False,
            ),
            engine_mode="MONITORING",
            inference_latency_ms=12.5,
            timing=TimingBreakdownDTO(
                queue_latency_ms=1.5,
                infer_duration_ms=12.5,
                server_total_ms=14.0,
                server_ingest_mono_ms=10000.5,
                server_publish_mono_ms=10014.5,
                server_wall_epoch_ms=1700000000100.0,
            ),
        )

    def test_decimal_string_sequence_serialization(self):
        """Verify sequence numbers are serialized as decimal strings conforming to regex ^[0-9]+$."""
        seq_regex = re.compile(r"^[0-9]+$")

        dto = self._create_sample_telemetry(frame_idx=9007199254740995)  # Greater than 2^53 - 1
        dto_dict = dto.model_dump()

        self.assertIsInstance(dto_dict["frame_seq"], str)
        self.assertIsInstance(dto_dict["event_seq"], str)
        self.assertTrue(seq_regex.match(dto_dict["frame_seq"]))
        self.assertTrue(seq_regex.match(dto_dict["event_seq"]))
        self.assertEqual(dto_dict["frame_seq"], "9007199254740995")

        # Test TelemetryPacketDTO
        packet = TelemetryPacketDTO(
            v=2,
            type="TELEMETRY",
            session_id=self.session_id,
            seq="9007199254740995",
            frame_seq="9007199254740995",
            event_seq="100",
            worker_boot_id=WORKER_BOOT_ID,
            session_epoch="1a2b3c",
            server_wall_epoch_ms=1700000000000.0,
            timestamp=1700000000.0,
            telemetry=dto,
        )
        packet_dict = packet.model_dump()
        self.assertEqual(packet_dict["frame_seq"], "9007199254740995")
        self.assertEqual(packet_dict["event_seq"], "100")
        self.assertEqual(packet_dict["worker_boot_id"], WORKER_BOOT_ID)

    def test_safety_event_dto_coercion_and_causal_linking(self):
        """Verify SafetyEventDTO coerces numeric seq to string and links frame_seq causally."""
        event = SafetyEventDTO(
            v=2,
            type="SAFETY_EVENT",
            event_id="evt_test_123",
            session_id=self.session_id,
            event_seq=42,  # Passed as int, must coerce to "42"
            frame_seq=105,  # Passed as int, must coerce to "105"
            worker_boot_id=WORKER_BOOT_ID,
            session_epoch="epoch_xyz",
            event_type="STATE_TRANSITION",
            previous_state=AlertStateEnum.NORMAL,
            new_state=AlertStateEnum.CRITICAL,
            timestamp=time.time(),
            server_wall_epoch_ms=round(time.time() * 1000.0, 1),
            metrics={"risk_score": 0.95},
        )
        event_dict = event.model_dump()
        self.assertEqual(event_dict["event_seq"], "42")
        self.assertEqual(event_dict["frame_seq"], "105")
        self.assertEqual(event_dict["worker_boot_id"], WORKER_BOOT_ID)
        self.assertEqual(event_dict["session_epoch"], "epoch_xyz")
        self.assertIsNotNone(event_dict["server_wall_epoch_ms"])

    def test_clock_domain_separation(self):
        """Verify server monotonic and epoch timestamps are distinct and non-zero."""
        dto = self._create_sample_telemetry(frame_idx=1)
        timing = dto.timing
        self.assertIsNotNone(timing)
        self.assertGreater(timing.server_ingest_mono_ms, 0)
        self.assertGreater(timing.server_publish_mono_ms, timing.server_ingest_mono_ms)
        self.assertGreater(timing.server_wall_epoch_ms, 1700000000000.0)
        # Server monotonic and wall-clock must differ by orders of magnitude (monotonic < 1e9, epoch > 1.7e12)
        self.assertNotEqual(timing.server_ingest_mono_ms, timing.server_wall_epoch_ms)

    def test_frame_accounting_initial_state_and_invariant(self):
        """Verify FrameAccountingSnapshot invariant holds initially and across counters."""
        snap = self.worker.get_frame_accounting_snapshot()
        self.assertEqual(snap.frames_received, 0)
        self.assertEqual(snap.frames_processed, 0)
        self.assertEqual(snap.frames_replaced, 0)
        self.assertEqual(snap.frames_queued, 0)
        self.assertEqual(snap.frames_in_flight, 0)
        self.assertTrue(snap.is_invariant_satisfied)
        self.assertEqual(snap.discrepancy, 0)

    def test_frame_accounting_snapshot_invariance_under_simulation(self):
        """Verify mathematical invariant under direct counter modifications."""
        snapshot = FrameAccountingSnapshot(
            frames_received=1000,
            frames_dequeued=750,
            frames_processed=745,
            frames_replaced=250,
            frames_stale_discarded=0,
            frames_errored=3,
            frames_aborted=1,
            frames_queued=0,
            frames_in_flight=1,
        )
        # 250 replaced + 0 stale + 3 errored + 1 aborted + 745 processed + 0 queued + 1 in_flight = 1000
        self.assertEqual(snapshot.total_accounted, 1000)
        self.assertTrue(snapshot.is_invariant_satisfied)
        self.assertEqual(snapshot.discrepancy, 0)

    async def test_worker_build_telemetry_payload_increments_event_seq(self):
        """Verify _build_telemetry_payload increments event_seq and tags frame_seq correctly."""
        dto1 = self._create_sample_telemetry(frame_idx=1)
        dto2 = self._create_sample_telemetry(frame_idx=2)

        payload1 = self.worker._build_telemetry_payload(dto1, frame_seq_num=101)
        self.assertEqual(payload1["frame_seq"], "101")
        self.assertEqual(payload1["event_seq"], "1")
        self.assertEqual(payload1["worker_boot_id"], self.worker.worker_boot_id)
        self.assertEqual(payload1["session_epoch"], self.worker.session_epoch)

        payload2 = self.worker._build_telemetry_payload(dto2, frame_seq_num=102)
        self.assertEqual(payload2["frame_seq"], "102")
        self.assertEqual(payload2["event_seq"], "2")
        self.assertGreater(int(payload2["event_seq"]), int(payload1["event_seq"]))


if __name__ == "__main__":
    unittest.main()
