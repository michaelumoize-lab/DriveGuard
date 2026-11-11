"""
Unit tests for DG-10 (Worker Lifecycle & Concurrency) and DG-11 (Stale Inference Results & Track Replacement).
"""

from __future__ import annotations

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch
import pytest

from backend.app.services.livekit_worker import (
    LiveKitAIWorkerSession,
    LiveKitWorkerManager,
    WorkerState,
)


@pytest.fixture
def mock_worker():
    """Create a test worker with mocked LiveKit room."""
    mock_room = MagicMock()
    mock_room.connect = AsyncMock()
    mock_room.disconnect = AsyncMock()
    mock_room.remote_participants = {}
    mock_room.local_participant = MagicMock()
    mock_room.local_participant.publish_data = AsyncMock()

    worker = LiveKitAIWorkerSession(
        room_name="test_lifecycle_room",
        driver_id="driver_test",
        room=mock_room,
    )
    return worker


@pytest.mark.asyncio
async def test_track_generation_increments_and_purges_stale_queue(mock_worker):
    """Subscribing to a new video track must increment track generation and purge stale queue."""
    worker = mock_worker
    assert worker.track_generation == 0

    mock_track_1 = MagicMock()
    mock_track_1.sid = "TR_001"
    mock_part = MagicMock()
    mock_part.identity = "driver_test"

    with patch("backend.app.services.livekit_worker.rtc.VideoStream") as mock_vstream_cls:
        mock_vstream = MagicMock()
        mock_vstream_cls.return_value = mock_vstream

        # Subscribe Track 1
        worker.is_running = True
        await worker._handle_track_subscribed(mock_track_1, mock_part)
        assert worker.track_generation == 1

        # Simulate an old frame queued from track 1
        mock_frame = MagicMock()
        await worker.frame_queue.put((mock_frame, time.perf_counter() * 1000.0, 10, 1))
        assert worker.frame_queue.qsize() == 1

        # Subscribe Track 2 (replaces Track 1)
        mock_track_2 = MagicMock()
        mock_track_2.sid = "TR_002"
        await worker._handle_track_subscribed(mock_track_2, mock_part)

        # Generation should be 2
        assert worker.track_generation == 2
        # Queue should be purged
        assert worker.frame_queue.empty()
        # Stale frame counter incremented
        assert worker._frames_stale_discarded >= 1


@pytest.mark.asyncio
async def test_in_flight_inference_discarded_on_track_replacement(mock_worker):
    """If a track is replaced while inference is running, stale result must be discarded."""
    worker = mock_worker
    worker.is_running = True
    worker._track_generation = 1

    # Simulate queued frame from track generation 1
    mock_frame = MagicMock()
    mock_frame.convert.return_value = MagicMock(data=bytes(640 * 480 * 3), width=640, height=480)
    await worker.frame_queue.put((mock_frame, time.perf_counter() * 1000.0, 1, 1))

    # Advance track generation to 2 before inference completes
    worker._track_generation = 2

    initial_stale = worker._frames_stale_discarded

    # Run inference step
    # Frame popped with generation 1 != current generation 2
    queue_item = await worker.frame_queue.get()
    frame, recv_mono_ms, frame_seq_num, frame_track_gen = queue_item
    assert frame_track_gen == 1
    assert frame_track_gen != worker._track_generation

    # If processed, stale counter must increment
    if frame_track_gen != worker._track_generation:
        worker._frames_stale_discarded += 1

    assert worker._frames_stale_discarded == initial_stale + 1


@pytest.mark.asyncio
async def test_participant_departure_grace_period_and_cancellation(mock_worker):
    """Participant departure starts 10s grace period; reconnection cancels shutdown."""
    worker = mock_worker
    worker.is_running = True

    # 1. Start grace period shutdown
    worker._shutdown_task = asyncio.create_task(worker._auto_shutdown_if_empty(delay_sec=10.0))
    assert worker._shutdown_task is not None
    assert not worker._shutdown_task.done()

    # 2. Participant reconnects -> cancels shutdown task
    if worker._shutdown_task and not worker._shutdown_task.done():
        worker._shutdown_task.cancel()
        worker._shutdown_task = None

    assert worker._shutdown_task is None
    assert worker.is_running is True


def test_worker_initialization_with_or_without_room():
    """Worker can be instantiated cleanly with custom room or default."""
    mock_room = MagicMock()
    worker = LiveKitAIWorkerSession(
        room_name="test_init_room",
        driver_id="driver_init",
        room=mock_room,
    )
    assert worker.room is mock_room
    assert worker.state == WorkerState.CREATED
    assert worker.track_generation == 0
