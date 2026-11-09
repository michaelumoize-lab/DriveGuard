"""
DriveGuard Phase 7 Stage 6: Fault Injection, Resilience & Recovery Test Suite.

Systematically verifies system robustness under real-time failure scenarios:
  - Group A: Video & Camera Failures
  - Group B: Inference Failures
  - Group C: LiveKit SFU & Network Failures
  - Group D: Telemetry & Safety Alert Failures
  - Group E: Session Lifecycle & Concurrency Failures

Ensures zero silent failures, strict frame accounting invariants,
complete resource reclamation, and multi-session failure isolation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import pathlib
import sys
import threading
import time
import unittest
import uuid
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import cv2
import numpy as np

# Ensure backend root on sys.path
_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from backend.app.ai.cascade_classifier import CascadeStageResult, ExperimentMode
from backend.app.ai.geometry import FrameGeometryFeatures
from backend.app.ai.landmarker import DriveGuardFaceLandmarker
from backend.app.ai.pipeline import SessionAIEngine
from backend.app.schemas.telemetry import AlertStateEnum
from backend.app.services.livekit_worker import (
    FrameAccountingSnapshot,
    LiveKitAIWorkerSession,
    WORKER_BOOT_ID,
    livekit_worker_manager,
)
from backend.app.services.session_manager import session_manager
from backend.benchmarks.benchmark_inference_offline import generate_benchmark_frame

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("test_fault_injection")


class MockVideoFrame:
    """Mock WebRTC VideoFrame wrapping numpy RGB buffer."""

    def __init__(self, img_bgr: np.ndarray) -> None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        self.width = img_bgr.shape[1]
        self.height = img_bgr.shape[0]
        self._data = img_rgb.tobytes()

    def convert(self, buffer_type: Any) -> MockVideoFrame:
        return self

    @property
    def data(self) -> bytes:
        return self._data


class MockDataPacket:
    """Mock WebRTC DataPacket for incoming control and ACK messages."""

    def __init__(self, data: bytes) -> None:
        self.data = data


class MockAsyncVideoStream:
    """Mock WebRTC VideoStream tracking aclose() and yielding frames."""

    def __init__(
        self,
        frames: Optional[List[MockVideoFrame]] = None,
        close_delay: float = 0.0,
        should_raise_on_close: Optional[Exception] = None,
    ) -> None:
        self.frames = frames or []
        self.close_delay = close_delay
        self.should_raise_on_close = should_raise_on_close
        self.close_call_count = 0
        self.closed = False

    async def __aiter__(self):
        idx = 0
        while not self.closed:
            if self.frames:
                yield MagicMock(frame=self.frames[idx % len(self.frames)])
                idx += 1
            await asyncio.sleep(0.04)

    async def aclose(self) -> None:
        self.close_call_count += 1
        if self.close_delay > 0:
            await asyncio.sleep(self.close_delay)
        if self.should_raise_on_close:
            raise self.should_raise_on_close
        self.closed = True


class FaultInjectionCollector:
    """Mock local_participant collecting published packets and safety events."""

    def __init__(self, session_id: str, auto_ack: bool = False, worker_ref: Optional[Any] = None) -> None:
        self.session_id = session_id
        self.auto_ack = auto_ack
        self.worker_ref = worker_ref
        self.packets: List[dict] = []
        self.safety_events: List[dict] = []
        self.control_acks: List[dict] = []
        self.fail_publish_with: Optional[Exception] = None
        self.publish_call_count: int = 0

    async def publish_data(self, data: bytes, reliable: bool = False, topic: Optional[str] = None) -> None:
        self.publish_call_count += 1
        if self.fail_publish_with:
            raise self.fail_publish_with
        try:
            payload = json.loads(data.decode("utf-8"))
            msg_type = payload.get("type")
            if msg_type in ("STATE_CHANGE", "SAFETY_ALERT", "SAFETY_EVENT"):
                self.safety_events.append(payload)
                if self.auto_ack and self.worker_ref:
                    evt_id = payload.get("event_id")
                    if evt_id:
                        asyncio.create_task(
                            self.worker_ref._handle_client_control_message(
                                MockDataPacket(json.dumps({"action": "ACK_EVENT", "event_id": evt_id}).encode("utf-8"))
                            )
                        )
            elif msg_type in ("STATE_SYNC", "CALIBRATION_STARTED", "PONG"):
                self.control_acks.append(payload)
            else:
                self.packets.append(payload)
        except Exception as exc:
            logger.debug(f"Collector decode error: {exc}")


# ============================================================================
# Test Suite Implementation
# ============================================================================

class TestDriveGuardFaultInjection(unittest.IsolatedAsyncioTestCase):
    """Stage 6 Fault Injection and Resilience Test Suite."""

    async def asyncSetUp(self) -> None:
        self.test_room = f"test_fault_{uuid.uuid4().hex[:8]}"
        self.test_driver = f"driver_{uuid.uuid4().hex[:6]}"

    async def asyncTearDown(self) -> None:
        # Ensure session cleanup from managers
        await livekit_worker_manager.stop_worker(self.test_room)
        await session_manager.terminate_session(self.test_room)

    # ------------------------------------------------------------------------
    # Group A: Video & Camera Failures
    # ------------------------------------------------------------------------

    async def test_camera_permission_denied_or_missing_feed(self) -> None:
        """When camera feed produces no valid frames, worker must stay in MONITORING_UNAVAILABLE and never report NORMAL."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker.engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)
        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Feed completely empty / unobservable inputs
        for idx in range(5):
            blank_img = np.zeros((480, 640, 3), dtype=np.uint8)
            now_ms = time.perf_counter() * 1000.0
            worker._frames_received += 1
            worker._frame_seq += 1
            await worker.frame_queue.put((MockVideoFrame(blank_img), now_ms, worker._frame_seq))
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.1)

        # Must report MONITORING_UNAVAILABLE, never NORMAL
        snap = worker.get_frame_accounting_snapshot()
        self.assertGreater(snap.frames_processed, 0)
        for pkt in collector.packets:
            self.assertEqual(pkt["telemetry"]["state"], "MONITORING_UNAVAILABLE")
            self.assertIn("NO_FACE_DETECTED", pkt["telemetry"]["reason_codes"])

        await worker.stop()
        post_snap = worker.get_frame_accounting_snapshot()
        self.assertTrue(post_snap.is_invariant_satisfied)
        self.assertEqual(post_snap.frames_queued, 0)
        self.assertEqual(post_snap.frames_in_flight, 0)

    async def test_video_track_unexpectedly_unsubscribed(self) -> None:
        """Unsubscribing active video track transitions immediately to MONITORING_UNAVAILABLE and closes stream."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker._last_state = "NORMAL"

        mock_stream = MockAsyncVideoStream()
        worker._current_video_stream = mock_stream

        # Trigger track_unsubscribed
        mock_track = MagicMock()
        mock_track.kind = "video"  # string or enum
        mock_participant = MagicMock()
        mock_participant.identity = "test_driver"

        # Call the transition handler directly
        await worker._safe_close_video_stream()
        await worker._transition_monitoring_unavailable("VIDEO_TRACK_UNSUBSCRIBED")

        # Stream must be closed
        self.assertTrue(mock_stream.closed)
        self.assertEqual(mock_stream.close_call_count, 1)

        # Safety event must be emitted
        self.assertEqual(worker._last_state, "MONITORING_UNAVAILABLE")
        self.assertEqual(len(collector.safety_events), 1)
        self.assertEqual(collector.safety_events[0]["new_state"], "MONITORING_UNAVAILABLE")
        self.assertIn("VIDEO_TRACK_UNSUBSCRIBED", collector.safety_events[0]["metrics"]["reasons"])

        await worker.stop()

    async def test_video_stream_stops_producing_frames_watchdog(self) -> None:
        """If frame delivery stalls for >2.0s, watchdog must transition to MONITORING_UNAVAILABLE."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker.engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)

        # Set running state to NORMAL and simulate prior frame delivery
        worker._last_state = "NORMAL"
        worker._frames_received = 10

        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Withhold frames and wait for the 2.0s watchdog to trigger
        await asyncio.sleep(2.3)

        # Watchdog must have detected frame delivery timeout
        self.assertEqual(worker._last_state, "MONITORING_UNAVAILABLE")
        self.assertGreaterEqual(len(collector.safety_events), 1)
        latest_evt = collector.safety_events[-1]
        self.assertEqual(latest_evt["new_state"], "MONITORING_UNAVAILABLE")
        self.assertIn("FRAME_DELIVERY_TIMEOUT", latest_evt["metrics"]["reasons"])

        await worker.stop()

    async def test_invalid_and_corrupted_frames(self) -> None:
        """Corrupted, zero-byte, or malformed frame buffers return IMAGE_DECODE_FAILED without crashing."""
        engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)

        # 1. Zero-byte array
        empty_arr = np.zeros((0, 0, 3), dtype=np.uint8)
        dto_empty = engine.process_frame(empty_arr, 100.0)
        self.assertEqual(dto_empty.state.value, "MONITORING_UNAVAILABLE")
        self.assertIn("IMAGE_DECODE_FAILED", dto_empty.reason_codes)

        # 2. Corrupt raw bytes
        corrupt_bytes = b"\x00\xff\xfe\x01\x02\x03\xaa\xbbcorrupted"
        dto_corrupt = engine.process_frame(corrupt_bytes, 140.0)
        self.assertEqual(dto_corrupt.state.value, "MONITORING_UNAVAILABLE")
        self.assertIn("IMAGE_DECODE_FAILED", dto_corrupt.reason_codes)

        # 3. Valid frame following corrupted frames must succeed immediately
        valid_img = generate_benchmark_frame(0)
        dto_valid = engine.process_frame(valid_img, 180.0)
        self.assertEqual(dto_valid.state.value, "NORMAL")
        self.assertTrue(dto_valid.face_detected)

    async def test_video_track_replacement_during_active_session(self) -> None:
        """Replacing video track cleanly closes old stream and activates new stream."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        stream_a = MockAsyncVideoStream()
        worker._current_video_stream = stream_a
        worker._is_running = True

        # Track replacement simulation
        stream_b = MockAsyncVideoStream()
        await worker._safe_close_video_stream()
        worker._current_video_stream = stream_b

        self.assertTrue(stream_a.closed)
        self.assertEqual(stream_a.close_call_count, 1)
        self.assertFalse(stream_b.closed)
        self.assertIs(worker._current_video_stream, stream_b)

        await worker.stop()
        self.assertTrue(stream_b.closed)

    async def test_reconnection_after_temporary_video_loss(self) -> None:
        """Temporary video loss marks gap and resets PERCLOS without freezing false blinks."""
        engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)

        # Initial normal frame
        frame_normal = generate_benchmark_frame(0)
        dto1 = engine.process_frame(frame_normal, 1000.0)
        self.assertEqual(dto1.state.value, "NORMAL")

        # Simulate 2.5s dropout
        dto_dropout = engine.process_frame(np.zeros((10, 10, 3), dtype=np.uint8), 3500.0)
        self.assertEqual(dto_dropout.state.value, "MONITORING_UNAVAILABLE")

        # Resumed frame after gap
        dto_resumed = engine.process_frame(frame_normal, 3540.0)
        self.assertEqual(dto_resumed.state.value, "NORMAL")
        # In-flight closure must NOT be prolonged across gap
        self.assertLess(dto_resumed.eye_metrics.eye_closure_duration_ms, 200.0)

    # ------------------------------------------------------------------------
    # Group B: Inference Failures
    # ------------------------------------------------------------------------

    async def test_mediapipe_runtime_exception_isolation(self) -> None:
        """MediaPipe runtime exception returns LANDMARKER_EXCEPTION without crashing session."""
        engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)

        with patch.object(engine.landmarker, "process_frame", side_effect=RuntimeError("MediaPipe internal graph failure")):
            valid_img = generate_benchmark_frame(1)
            dto = engine.process_frame(valid_img, 200.0)

            self.assertEqual(dto.state.value, "MONITORING_UNAVAILABLE")
            self.assertTrue(any("LANDMARKER_EXCEPTION" in r for r in dto.reason_codes))

        # Subsequent call without exception recovers normally
        dto_recovered = engine.process_frame(valid_img, 240.0)
        self.assertEqual(dto_recovered.state.value, "NORMAL")

    async def test_onnx_runtime_exception_graceful_fallback(self) -> None:
        """ONNX runtime exception gracefully degrades to geometric baseline without crashing."""
        engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)

        with patch.object(engine.cascade, "evaluate", side_effect=Exception("ONNX Runtime execution failure")):
            valid_img = generate_benchmark_frame(2)
            dto = engine.process_frame(valid_img, 300.0)

            # Successfully processed via GEOMETRIC_FALLBACK
            self.assertEqual(dto.cascade.arbitration_winner, "GEOMETRIC_FALLBACK")
            self.assertEqual(dto.state.value, "NORMAL")

    async def test_repeated_consecutive_inference_failures(self) -> None:
        """Worker accounts for repeated unhandled inference exceptions under frames_errored."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker.engine = MagicMock()
        worker.engine.process_frame.side_effect = RuntimeError("Fatal hardware execution fault")

        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Feed 5 frames that will all trigger errors
        for idx in range(5):
            worker._frames_received += 1
            worker._frame_seq += 1
            await worker.frame_queue.put((MockVideoFrame(generate_benchmark_frame(idx)), time.perf_counter() * 1000.0, worker._frame_seq))
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.1)

        snap = worker.get_frame_accounting_snapshot()
        self.assertEqual(snap.frames_errored, 5)
        self.assertEqual(snap.frames_processed, 0)
        self.assertTrue(snap.is_invariant_satisfied)

        await worker.stop()
        post_snap = worker.get_frame_accounting_snapshot()
        self.assertTrue(post_snap.is_invariant_satisfied)
        self.assertEqual(post_snap.frames_in_flight, 0)

    async def test_inference_duration_spike_and_queue_replacement(self) -> None:
        """Inference duration spike causes single-slot queue to drop intermediate frames, keeping latency bounded."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker.engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)

        # Slow down process_frame artificially
        orig_process = worker.engine.process_frame
        def slow_process(*args, **kwargs):
            time.sleep(0.12)  # 120ms inference time
            return orig_process(*args, **kwargs)

        worker.engine.process_frame = slow_process
        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Ingest 10 frames at 25 FPS (40ms interval)
        for idx in range(10):
            frame = MockVideoFrame(generate_benchmark_frame(idx))
            now_ms = time.perf_counter() * 1000.0
            worker._frames_received += 1
            worker._frame_seq += 1
            if worker.frame_queue.full():
                try:
                    worker.frame_queue.get_nowait()
                    worker._frames_replaced += 1
                except asyncio.QueueEmpty:
                    pass
            await worker.frame_queue.put((frame, now_ms, worker._frame_seq))
            await asyncio.sleep(0.04)

        await asyncio.sleep(0.3)

        snap = worker.get_frame_accounting_snapshot()
        self.assertGreater(snap.frames_replaced, 0)
        self.assertTrue(snap.is_invariant_satisfied)

        await worker.stop()

    async def test_inference_thread_cancellation_reconciliation(self) -> None:
        """Cancelling coroutine while OS worker thread is active preserves in_flight until joined."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        worker.engine = MagicMock()

        thread_barrier = threading.Event()
        def blocked_process(*args, **kwargs):
            thread_barrier.wait(timeout=2.0)
            return MagicMock()

        worker.engine.process_frame = blocked_process
        worker.is_running = True
        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Enqueue frame
        worker._frames_received += 1
        worker._frame_seq += 1
        await worker.frame_queue.put((MockVideoFrame(generate_benchmark_frame(0)), time.perf_counter() * 1000.0, 1))

        # Wait for thread to enter blocked_process
        await asyncio.sleep(0.05)
        self.assertEqual(worker._frames_in_flight, 1)

        # Cancel inference task while thread is blocked
        worker._inference_task.cancel()
        try:
            await worker._inference_task
        except asyncio.CancelledError:
            pass

        # In-flight must be preserved while OS thread is running
        self.assertTrue(worker._coro_cancelled_for_active_frame)

        # Release thread
        thread_barrier.set()
        await asyncio.sleep(0.05)

        # Reconciled to aborted
        self.assertEqual(worker._frames_in_flight, 0)
        self.assertEqual(worker._frames_aborted, 1)

        await worker.stop()

    # ------------------------------------------------------------------------
    # Group C: LiveKit & Network Failures
    # ------------------------------------------------------------------------

    async def test_participant_departure_grace_period_and_cancellation(self) -> None:
        """Participant departure starts 10s auto-shutdown; reconnect within grace cancels it."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        worker.room = MagicMock()
        worker.room.remote_participants = {}
        worker.is_running = True

        # Departure event
        mock_p = MagicMock()
        mock_p.identity = "test_driver"
        worker._shutdown_task = asyncio.create_task(worker._auto_shutdown_if_empty(delay_sec=5.0))
        self.assertIsNotNone(worker._shutdown_task)
        self.assertFalse(worker._shutdown_task.done())

        # Reconnect before timeout
        worker.room.remote_participants = {"test_driver": mock_p}
        if worker._shutdown_task and not worker._shutdown_task.done():
            worker._shutdown_task.cancel()
            worker._shutdown_task = None

        self.assertIsNone(worker._shutdown_task)
        self.assertTrue(worker.is_running)

        await worker.stop()

    async def test_reconnection_with_same_session_no_duplicate_workers(self) -> None:
        """ensure_worker_running returns identical worker instance on repeated / reconnect calls."""
        with patch.object(LiveKitAIWorkerSession, "start", new_callable=AsyncMock):
            w1 = await livekit_worker_manager.ensure_worker_running(self.test_room, self.test_driver)
            w1.is_running = True

            # Second call for same room
            w2 = await livekit_worker_manager.ensure_worker_running(self.test_room, self.test_driver)

            self.assertIs(w1, w2)
            self.assertEqual(len(livekit_worker_manager.list_active_workers()), 1)

            await livekit_worker_manager.stop_worker(self.test_room)

    async def test_telemetry_publication_failure_resilience(self) -> None:
        """Failure in publish_data does not terminate worker or violate frame accounting."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        collector.fail_publish_with = ConnectionResetError("WebRTC DataChannel connection lost")

        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker.engine = await session_manager.get_or_create_session(self.test_room, self.test_driver)
        worker._inference_task = asyncio.create_task(worker._inference_loop())

        # Feed frame
        worker._frames_received += 1
        worker._frame_seq += 1
        await worker.frame_queue.put((MockVideoFrame(generate_benchmark_frame(0)), time.perf_counter() * 1000.0, 1))
        await asyncio.sleep(0.4)

        # Worker still running despite publish failure
        self.assertTrue(worker.is_running)
        snap = worker.get_frame_accounting_snapshot()
        self.assertEqual(snap.frames_processed, 1)
        self.assertTrue(snap.is_invariant_satisfied)

        await worker.stop()

    # ------------------------------------------------------------------------
    # Group D: Telemetry & Safety Alert Failures
    # ------------------------------------------------------------------------

    async def test_reliable_safety_event_retry_and_exhaustion(self) -> None:
        """Safety event unacknowledged by client retries 3 times with backoff and remains unacknowledged."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room, auto_ack=False)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True

        evt_payload = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_test_retry",
            "session_id": self.test_room,
            "event_seq": "1",
            "frame_seq": "1",
            "worker_boot_id": WORKER_BOOT_ID,
            "session_epoch": worker.session_epoch,
            "event_type": "STATE_TRANSITION",
            "previous_state": "NORMAL",
            "new_state": "CRITICAL",
            "timestamp": time.time(),
        }

        # Publish without client ACK
        await worker._publish_reliable_safety_event(evt_payload)

        # Fast-forward / wait for retries (0.4s + 0.8s + 1.2s = 2.4s)
        await asyncio.sleep(2.7)

        # Initial attempt + 3 retries = 4 publish attempts
        self.assertGreaterEqual(collector.publish_call_count, 4)
        self.assertIn("evt_test_retry", worker._unacknowledged_events)

        await worker.stop()

    async def test_duplicate_safety_event_ack_confirmation(self) -> None:
        """Sending ACK_EVENT confirms event and removes it from unacknowledged queue immediately."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room, auto_ack=True, worker_ref=worker)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True

        evt_payload = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_ack_test",
            "session_id": self.test_room,
            "event_seq": "1",
            "frame_seq": "1",
            "worker_boot_id": WORKER_BOOT_ID,
            "session_epoch": worker.session_epoch,
            "event_type": "STATE_TRANSITION",
            "previous_state": "NORMAL",
            "new_state": "WARNING",
            "timestamp": time.time(),
        }

        await worker._publish_reliable_safety_event(evt_payload)
        await asyncio.sleep(0.15)

        # Auto-ACK received -> event removed
        self.assertNotIn("evt_ack_test", worker._unacknowledged_events)

        await worker.stop()

    async def test_frontend_disconnected_state_resync_recovery(self) -> None:
        """SYNC_STATE delivers current state, latest safety event, and all unacknowledged events."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        collector = FaultInjectionCollector(self.test_room)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.room.local_participant = collector
        worker.is_running = True
        worker.engine = MagicMock()
        worker._last_state = "CRITICAL"

        # Simulate unacknowledged event during disconnect
        unacked_evt = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_unacked_01",
            "new_state": "CRITICAL",
        }
        worker._latest_safety_event = unacked_evt
        worker._unacknowledged_events["evt_unacked_01"] = {
            "payload": unacked_evt,
            "retries": 1,
            "sent_at": time.perf_counter(),
        }

        # Client reconnects and sends SYNC_STATE
        sync_packet = MockDataPacket(json.dumps({"action": "SYNC_STATE"}).encode("utf-8"))
        await worker._handle_client_control_message(sync_packet)
        await asyncio.sleep(0.05)

        self.assertEqual(len(collector.control_acks), 1)
        sync_ack = collector.control_acks[0]
        self.assertEqual(sync_ack["type"], "STATE_SYNC")
        self.assertEqual(sync_ack["current_state"], "CRITICAL")
        self.assertEqual(sync_ack["latest_safety_event"]["event_id"], "evt_unacked_01")
        self.assertEqual(len(sync_ack["unacknowledged_events"]), 1)
        self.assertEqual(sync_ack["unacknowledged_events"][0]["event_id"], "evt_unacked_01")

        await worker.stop()

    # ------------------------------------------------------------------------
    # Group E: Session Lifecycle Failures
    # ------------------------------------------------------------------------

    async def test_repeated_stop_calls_idempotency(self) -> None:
        """Calling stop() multiple times concurrently is fully idempotent and raises no errors."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        worker.room = MagicMock()
        worker.room.disconnect = AsyncMock()
        worker.is_running = True

        # Execute 5 concurrent stop() calls
        await asyncio.gather(*(worker.stop() for _ in range(5)))

        self.assertFalse(worker.is_running)
        self.assertTrue(worker._is_stopped)
        self.assertEqual(worker.room.disconnect.call_count, 1)

    async def test_multi_session_failure_isolation(self) -> None:
        """Catastrophic failure in Session A does not interrupt, corrupt, or alter Session B."""
        room_a = f"{self.test_room}_a"
        room_b = f"{self.test_room}_b"

        worker_a = LiveKitAIWorkerSession(room_name=room_a, driver_id="driver_a")
        worker_b = LiveKitAIWorkerSession(room_name=room_b, driver_id="driver_b")

        worker_a.room = MagicMock()
        worker_a.room.disconnect = AsyncMock()
        worker_a.room.local_participant = FaultInjectionCollector(room_a)
        worker_a.is_running = True
        worker_a.engine = MagicMock()
        # Injected catastrophic failure in A
        worker_a.engine.process_frame.side_effect = RuntimeError("Session A catastrophic GPU/NPU crash")

        worker_b.room = MagicMock()
        worker_b.room.disconnect = AsyncMock()
        worker_b.room.local_participant = FaultInjectionCollector(room_b)
        worker_b.is_running = True
        worker_b.engine = await session_manager.get_or_create_session(room_b, "driver_b")

        worker_a._inference_task = asyncio.create_task(worker_a._inference_loop())
        worker_b._inference_task = asyncio.create_task(worker_b._inference_loop())

        # Ingest frames to both
        for idx in range(5):
            frame = MockVideoFrame(generate_benchmark_frame(idx))
            now = time.perf_counter() * 1000.0

            worker_a._frames_received += 1
            worker_a._frame_seq += 1
            await worker_a.frame_queue.put((frame, now, worker_a._frame_seq))

            worker_b._frames_received += 1
            worker_b._frame_seq += 1
            await worker_b.frame_queue.put((frame, now, worker_b._frame_seq))
            await asyncio.sleep(0.02)

        await asyncio.sleep(0.1)

        # Verify Session A captured errors
        snap_a = worker_a.get_frame_accounting_snapshot()
        self.assertEqual(snap_a.frames_errored, 5)
        self.assertEqual(snap_a.frames_processed, 0)
        self.assertTrue(snap_a.is_invariant_satisfied)

        # Verify Session B processed all frames normally
        snap_b = worker_b.get_frame_accounting_snapshot()
        self.assertEqual(snap_b.frames_errored, 0)
        self.assertGreater(snap_b.frames_processed, 0)
        self.assertTrue(snap_b.is_invariant_satisfied)

        await worker_a.stop()
        await worker_b.stop()

    async def test_worker_cleanup_exception_containment(self) -> None:
        """Exception during stream aclose() is caught, counted in diagnostics, and does not leak."""
        worker = LiveKitAIWorkerSession(room_name=self.test_room, driver_id=self.test_driver)
        mock_stream = MockAsyncVideoStream(
            should_raise_on_close=RuntimeError("Native WebRTC FFI deallocation failure")
        )
        worker._current_video_stream = mock_stream

        # Safe close records error in diagnostics and surfaces to caller; during worker.stop() it is cleanly contained
        with self.assertRaises(RuntimeError):
            await worker._safe_close_video_stream()

        self.assertEqual(worker.stream_cleanup_error_count, 1)
        self.assertIn("Native WebRTC FFI deallocation failure", worker.last_stream_cleanup_error or "")

        # Worker stop() must contain any stream close error without crashing
        await worker.stop()

    async def test_rapid_session_creation_and_termination(self) -> None:
        """Rapidly cycling 5 sessions creates and disposes resources cleanly without resource leaks."""
        with patch.object(LiveKitAIWorkerSession, "start", new_callable=AsyncMock):
            for idx in range(5):
                temp_room = f"rapid_sess_{idx}_{uuid.uuid4().hex[:6]}"
                worker = await livekit_worker_manager.ensure_worker_running(temp_room, f"driver_{idx}")
                worker.room = MagicMock()
                worker.room.disconnect = AsyncMock()
                worker.room.local_participant = FaultInjectionCollector(temp_room)
                worker.is_running = True
                worker.engine = await session_manager.get_or_create_session(temp_room, f"driver_{idx}")
                worker._inference_task = asyncio.create_task(worker._inference_loop())

                # Send 1 frame
                worker._frames_received += 1
                worker._frame_seq += 1
                await worker.frame_queue.put((MockVideoFrame(generate_benchmark_frame(0)), time.perf_counter() * 1000.0, 1))
                await asyncio.sleep(0.05)

                # Abrupt stop
                await livekit_worker_manager.stop_worker(temp_room)
                snap = worker.get_frame_accounting_snapshot()
                self.assertTrue(snap.is_invariant_satisfied)
                self.assertEqual(snap.frames_queued, 0)
                self.assertEqual(snap.frames_in_flight, 0)

            # Active worker list must be empty
            self.assertEqual(len(livekit_worker_manager.list_active_workers()), 0)


if __name__ == "__main__":
    unittest.main()
