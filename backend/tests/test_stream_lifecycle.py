"""
Unit tests for DriveGuard Phase 7 Part B Stage 4:
Stream lifecycle management, race-safe teardown, cancellation safety,
and hybrid latency history bounded memory.
"""

import asyncio
from collections import deque
from typing import Optional
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np

from backend.app.ai.pipeline import SessionAIEngine
from backend.app.services.livekit_worker import LiveKitAIWorkerSession


class MockAsyncVideoStream:
    """Mock WebRTC VideoStream tracking aclose() invocations and simulated delay."""

    def __init__(self, close_delay: float = 0.01, should_raise: Optional[Exception] = None):
        self.close_delay = close_delay
        self.close_call_count = 0
        self.closed = False
        self.should_raise = should_raise
        self._close_event = asyncio.Event()

    async def __aiter__(self):
        while not self.closed:
            await asyncio.sleep(0.02)
            if False:
                yield None

    async def aclose(self) -> None:
        self.close_call_count += 1
        if self.close_delay > 0:
            await asyncio.sleep(self.close_delay)
        if self.should_raise:
            raise self.should_raise
        self.closed = True
        self._close_event.set()


class TestStreamLifecycleAndLatency(unittest.IsolatedAsyncioTestCase):
    """Test suite for VideoStream cleanup invariants and SessionAIEngine memory stability."""

    async def asyncSetUp(self):
        self.worker = LiveKitAIWorkerSession(
            room_name="test_stream_lifecycle_room",
            driver_id="test_driver",
        )
        self.worker.room = MagicMock()
        self.worker.room.disconnect = AsyncMock()

    async def asyncTearDown(self):
        if self.worker:
            if self.worker._executor:
                self.worker._executor.shutdown(wait=False, cancel_futures=True)
            self.worker = None

    async def test_stream_cleanup_idempotency(self):
        """Repeated calls to _safe_close_video_stream must execute aclose() exactly once."""
        mock_stream = MockAsyncVideoStream()
        self.worker._current_video_stream = mock_stream

        await self.worker._safe_close_video_stream()
        self.assertTrue(mock_stream.closed)
        self.assertEqual(mock_stream.close_call_count, 1)
        self.assertIsNone(self.worker._current_video_stream)

        # Second cleanup call on already closed stream
        await self.worker._safe_close_video_stream()
        self.assertEqual(mock_stream.close_call_count, 1)

    async def test_concurrent_unsubscription_and_stop(self):
        """Concurrent track unsubscription and stop() must coordinate on single aclose() execution."""
        mock_stream = MockAsyncVideoStream(close_delay=0.03)
        self.worker._current_video_stream = mock_stream
        self.worker.is_running = True

        # Simulate concurrent unsub task and worker stop()
        task_unsub = asyncio.create_task(self.worker._safe_close_video_stream())
        task_stop = asyncio.create_task(self.worker.stop())

        await asyncio.gather(task_unsub, task_stop)

        self.assertTrue(mock_stream.closed)
        self.assertEqual(mock_stream.close_call_count, 1)
        self.assertFalse(self.worker.is_running)

    async def test_cancellation_during_cleanup_preserves_native_disposal(self):
        """Cancellation of the caller awaiting cleanup must not abandon native stream aclose()."""
        mock_stream = MockAsyncVideoStream(close_delay=0.05)
        self.worker._current_video_stream = mock_stream

        cleanup_caller_task = asyncio.create_task(self.worker._safe_close_video_stream())
        await asyncio.sleep(0.01)

        # Cancel caller coroutine
        cleanup_caller_task.cancel()

        with self.assertRaises(asyncio.CancelledError):
            await cleanup_caller_task

        # Underlying close task protected by shield must complete
        await self.worker._stream_close_task
        self.assertTrue(mock_stream.closed)
        self.assertEqual(mock_stream.close_call_count, 1)

    async def test_stop_awaits_stream_cleanup_before_disconnect(self):
        """stop() must await stream.aclose() to complete before calling room.disconnect()."""
        call_order = []

        mock_stream = MockAsyncVideoStream(close_delay=0.02)
        orig_aclose = mock_stream.aclose

        async def tracked_aclose():
            call_order.append("stream_aclose_start")
            await orig_aclose()
            call_order.append("stream_aclose_done")

        mock_stream.aclose = tracked_aclose
        self.worker._current_video_stream = mock_stream
        self.worker.is_running = True

        async def tracked_disconnect():
            call_order.append("room_disconnect")

        self.worker.room.disconnect = tracked_disconnect

        await self.worker.stop()

        self.assertEqual(
            call_order,
            ["stream_aclose_start", "stream_aclose_done", "room_disconnect"],
            "stream.aclose() must complete before room.disconnect() is invoked",
        )

    def test_hybrid_latency_history_memory_boundedness(self):
        """SessionAIEngine latency history must cap memory to 1500 samples while maintaining exact lifetime sum."""
        mock_landmarker = MagicMock()
        mock_features = MagicMock()
        mock_features.face_detected = True
        mock_features.mesh_landmarks_2d = np.ones((478, 2), dtype=np.float32)
        mock_features.iris_landmarks = np.ones((10, 2), dtype=np.float32)
        mock_landmarker.detect_from_bgr.return_value = mock_features

        engine = SessionAIEngine(
            session_id="test_latency_session",
            driver_id="driver_test",
            landmarker=mock_landmarker,
        )

        # Manually record 5,000 frames into hybrid accumulators
        synthetic_latencies = [20.0 + (i % 25) * 0.5 for i in range(5000)]
        expected_sum = sum(synthetic_latencies)
        expected_mean = expected_sum / len(synthetic_latencies)

        for lat in synthetic_latencies:
            engine._lifetime_latency_count += 1
            engine._lifetime_latency_sum_ms += lat
            engine._recent_latencies_ms.append(lat)
            engine.frame_index += 1

        # Check bounds
        self.assertEqual(len(engine._recent_latencies_ms), 1500)
        self.assertEqual(len(engine.latencies_ms), 1500)
        self.assertEqual(len(engine.get_recent_latencies()), 1500)
        self.assertEqual(engine._lifetime_latency_count, 5000)

        # Check summation precision
        self.assertAlmostEqual(engine._lifetime_latency_sum_ms, expected_sum, places=4)

        # Check session summary uses exact lifetime mean
        summary = engine.get_session_summary()
        self.assertEqual(summary["total_frames_processed"], 5000)
        self.assertAlmostEqual(summary["mean_inference_latency_ms"], round(expected_mean, 2), places=2)

        # Check close releases resources
        engine.close()
        self.assertTrue(engine._is_closed)
        self.assertIsNone(engine.cascade)

    async def test_stream_aclose_exception_recording(self):
        """aclose() raising an exception must be recorded in error diagnostics and surfaced."""
        err = RuntimeError("Native WebRTC FFI deallocation failure")
        mock_stream = MockAsyncVideoStream(close_delay=0.01, should_raise=err)
        self.worker._current_video_stream = mock_stream

        with self.assertRaises(RuntimeError) as ctx:
            await self.worker._safe_close_video_stream()
        self.assertIn("Native WebRTC FFI deallocation failure", str(ctx.exception))
        self.assertEqual(self.worker.stream_cleanup_error_count, 1)
        self.assertIsNotNone(self.worker.last_stream_cleanup_error)
        self.assertIn("RuntimeError", self.worker.last_stream_cleanup_error)
        self.assertEqual(mock_stream.close_call_count, 1)

        # Subsequent cleanup on the same stream does not re-invoke aclose()
        # and should not raise or re-record error
        await self.worker._safe_close_video_stream(mock_stream)
        self.assertEqual(mock_stream.close_call_count, 1)

    async def test_concurrent_explicit_cleanup_of_same_stream(self):
        """Concurrent calls to _safe_close_video_stream explicitly passing the same stream invoke aclose() once."""
        mock_stream = MockAsyncVideoStream(close_delay=0.03)

        task1 = asyncio.create_task(self.worker._safe_close_video_stream(mock_stream))
        task2 = asyncio.create_task(self.worker._safe_close_video_stream(mock_stream))

        await asyncio.gather(task1, task2)

        self.assertEqual(mock_stream.close_call_count, 1)
        self.assertTrue(mock_stream.closed)

    async def test_rapid_track_resubscription_while_closing(self):
        """Rapid track re-subscription while previous stream is closing waits and replaces stream safely."""
        stream_a = MockAsyncVideoStream(close_delay=0.03)
        stream_b = MockAsyncVideoStream(close_delay=0.01)

        self.worker.is_running = True
        self.worker._current_video_stream = stream_a

        track_b = MagicMock()
        track_b.kind = MagicMock()
        track_b.sid = "track_b_sid"
        participant = MagicMock()
        participant.identity = "driver_client"

        with patch("livekit.rtc.VideoStream", return_value=stream_b):
            task_resub = asyncio.create_task(self.worker._handle_track_subscribed(track_b, participant))
            await task_resub

        self.assertEqual(stream_a.close_call_count, 1)
        self.assertTrue(stream_a.closed)
        self.assertIs(self.worker._current_video_stream, stream_b)

        await self.worker.stop()
        self.assertEqual(stream_b.close_call_count, 1)
        self.assertTrue(stream_b.closed)


if __name__ == "__main__":
    unittest.main()
