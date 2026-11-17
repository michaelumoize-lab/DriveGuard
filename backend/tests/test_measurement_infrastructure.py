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
    generate_session_epoch,
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

    def tearDown(self):
        if hasattr(self, "worker") and self.worker and self.worker._executor:
            self.worker._executor.shutdown(wait=False)

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

    def test_session_epoch_128bit_uniqueness_and_format(self):
        """Verify session_epoch incorporates a 128-bit UUID4 hex string and has zero collisions across 10k instances."""
        epoch_regex = re.compile(r"^[0-9]+_[0-9a-f]{32}$")
        self.assertTrue(
            epoch_regex.match(self.worker.session_epoch),
            f"session_epoch '{self.worker.session_epoch}' does not match expected format <timestamp>_<32hex>",
        )

        # Uniqueness test across 10,000 generated epochs
        epochs = {generate_session_epoch() for _ in range(10000)}
        self.assertEqual(len(epochs), 10000, "Collision detected in 128-bit session epoch generation!")

    async def test_inference_cancellation_preserves_in_flight_during_thread_execution(self):
        """
        Verify that cancelling an awaiting inference coroutine does NOT prematurely zero in_flight frames
        while the underlying OS thread is still executing process_frame().
        Verifies thread-safe event loop callback reconciliation and conservation invariant preservation.
        """
        worker = LiveKitAIWorkerSession(room_name="test_cancel_semantics")
        worker.is_running = True

        try:
            # Mock engine that sleeps for 80ms in background thread
            thread_started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def slow_process_frame(norm_img, recv_mono_ms):
                loop.call_soon_threadsafe(thread_started.set)
                time.sleep(0.08)
                return self._create_sample_telemetry(frame_idx=1)

            mock_engine = MagicMock()
            mock_engine.process_frame.side_effect = slow_process_frame
            mock_engine.get_calibration_status.return_value = None
            worker.engine = mock_engine

            # Mock WebRTC room publish_data
            worker.room = MagicMock()
            worker.room.local_participant = MagicMock()
            worker.room.local_participant.publish_data = AsyncMock()

            # Mock VideoFrame
            mock_frame = MagicMock()
            mock_buffer = MagicMock()
            mock_buffer.data = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
            mock_buffer.width = 640
            mock_buffer.height = 480
            mock_frame.convert.return_value = mock_buffer

            # Queue one frame
            worker._frames_received = 1
            await worker.frame_queue.put((mock_frame, time.perf_counter() * 1000.0, 1))

            # Launch inference loop
            task = asyncio.create_task(worker._inference_loop())

            # Wait until thread actually starts executing
            await asyncio.wait_for(thread_started.wait(), timeout=1.0)

            # At this point, the thread is actively executing process_frame
            snap_during = worker.get_frame_accounting_snapshot()
            self.assertEqual(snap_during.frames_in_flight, 1)
            self.assertEqual(snap_during.threads_executing, 1)
            self.assertEqual(snap_during.frames_aborted, 0)
            self.assertTrue(snap_during.is_invariant_satisfied)

            # CANCEL the inference task while thread is still sleeping
            task.cancel()
            await task

            # IMMEDIATELY AFTER COROUTINE CANCELLATION:
            # The thread is STILL physically running. Verify frames_in_flight is NOT zero!
            snap_post_cancel = worker.get_frame_accounting_snapshot()
            self.assertEqual(
                snap_post_cancel.frames_in_flight,
                1,
                "CRITICAL BUG: frames_in_flight prematurely reported 0 while worker thread was still executing!",
            )
            self.assertEqual(snap_post_cancel.threads_executing, 1)
            self.assertEqual(snap_post_cancel.frames_aborted, 0)
            self.assertTrue(snap_post_cancel.is_invariant_satisfied)

            # Wait for thread to finish sleeping and for the event-loop threadsafe callback to fire
            await asyncio.sleep(0.12)

            # AFTER THREAD TERMINATION:
            # In-flight must now be 0, aborted must be 1, zero double counting, invariant satisfied
            snap_final = worker.get_frame_accounting_snapshot()
            self.assertEqual(snap_final.frames_in_flight, 0)
            self.assertEqual(snap_final.threads_executing, 0)
            self.assertEqual(snap_final.frames_aborted, 1)
            self.assertEqual(snap_final.frames_processed, 0)
            self.assertTrue(snap_final.is_invariant_satisfied)
            self.assertEqual(snap_final.discrepancy, 0)
        finally:
            if worker._executor:
                worker._executor.shutdown(wait=False)

    async def test_stop_worker_preserves_in_flight_until_thread_done_without_double_counting(self):
        """Verify worker.stop() preserves in-flight status if thread is running and avoids double-counting."""
        worker = LiveKitAIWorkerSession(room_name="test_stop_semantics")
        worker.is_running = True

        try:
            thread_started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def slow_process_frame(norm_img, recv_mono_ms):
                loop.call_soon_threadsafe(thread_started.set)
                time.sleep(0.08)
                return self._create_sample_telemetry(frame_idx=1)

            mock_engine = MagicMock()
            mock_engine.process_frame.side_effect = slow_process_frame
            mock_engine.get_calibration_status.return_value = None
            worker.engine = mock_engine

            worker.room = MagicMock()
            worker.room.disconnect = AsyncMock()
            worker.room.local_participant = MagicMock()
            worker.room.local_participant.publish_data = AsyncMock()

            mock_frame = MagicMock()
            mock_buffer = MagicMock()
            mock_buffer.data = np.zeros((480, 640, 3), dtype=np.uint8).tobytes()
            mock_buffer.width = 640
            mock_buffer.height = 480
            mock_frame.convert.return_value = mock_buffer

            worker._frames_received = 2
            # Put first frame into queue
            await worker.frame_queue.put((mock_frame, time.perf_counter() * 1000.0, 1))

            worker._inference_task = asyncio.create_task(worker._inference_loop())
            await asyncio.wait_for(thread_started.wait(), timeout=1.0)

            # Queue is now empty (frame 1 dequeued into thread), so queue frame 2 without blocking maxsize=1
            await worker.frame_queue.put((mock_frame, time.perf_counter() * 1000.0, 2))

            # Call worker.stop() while thread is running
            await worker.stop()

            # In Phase 1 Workstream D, stop() safely awaits the active thread up to bounded timeout.
            # Upon return, the thread has completed, queued frames were aborted, and in-flight is reconciled:
            snap_after_stop = worker.get_frame_accounting_snapshot()
            self.assertEqual(snap_after_stop.frames_in_flight, 0)
            self.assertEqual(snap_after_stop.threads_executing, 0)
            self.assertEqual(snap_after_stop.frames_aborted, 2)
            self.assertTrue(snap_after_stop.is_invariant_satisfied)

            # Wait for thread to terminate and callback to reconcile
            await asyncio.sleep(0.12)

            snap_final = worker.get_frame_accounting_snapshot()
            self.assertEqual(snap_final.frames_in_flight, 0)
            self.assertEqual(snap_final.threads_executing, 0)
            self.assertEqual(snap_final.frames_aborted, 2)  # both frames now aborted
            self.assertEqual(snap_final.frames_processed, 0)
            self.assertTrue(snap_final.is_invariant_satisfied)
            self.assertEqual(snap_final.discrepancy, 0)
        finally:
            if worker._executor:
                worker._executor.shutdown(wait=False)


if __name__ == "__main__":
    unittest.main()
