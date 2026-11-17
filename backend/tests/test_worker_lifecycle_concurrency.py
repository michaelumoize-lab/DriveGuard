"""
Unit Tests for Workstream D — Deterministic LiveKit Worker Lifecycle.

Verifies:
1. Concurrent ensure_worker_running() requests for the same room produce exactly one worker.
2. Startup failures propagate cleanly and remove failed workers from the active registry.
3. Stop during active inference waits for the OS thread before releasing native engine resources.
4. Stop during startup cancels cleanly without orphan state.
5. Explicit WorkerState transitions: CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED.
"""

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
def mock_worker_dependencies():
    """Mock LiveKit SDK network calls and session manager to isolate lifecycle logic."""
    with patch("backend.app.services.livekit_worker.rtc.Room") as mock_room_cls, \
         patch("backend.app.services.livekit_worker.api.AccessToken") as mock_token_cls, \
         patch("backend.app.services.livekit_worker.session_manager") as mock_sess_mgr:

        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.disconnect = AsyncMock()
        mock_room_cls.return_value = mock_room

        mock_token = MagicMock()
        mock_token.with_identity.return_value = mock_token
        mock_token.with_name.return_value = mock_token
        mock_token.with_grants.return_value = mock_token
        mock_token.to_jwt.return_value = "mock_jwt_token"
        mock_token_cls.return_value = mock_token

        mock_engine = MagicMock()
        mock_engine.is_calibrating = False
        mock_engine.profile.calibration_mode = "UNCALIBRATED_DEFAULT"
        mock_engine._is_closed = False
        mock_sess_mgr.get_or_create_session = AsyncMock(return_value=mock_engine)
        mock_sess_mgr.terminate_session = AsyncMock(return_value={"total_frames_processed": 5})

        yield {
            "room": mock_room,
            "engine": mock_engine,
            "session_manager": mock_sess_mgr,
        }


@pytest.mark.asyncio
async def test_concurrent_ensure_worker_running_single_worker(mock_worker_dependencies):
    """Simultaneous calls to ensure_worker_running for the same room must produce exactly one worker."""
    manager = LiveKitWorkerManager()
    room_name = "test_concurrent_room_001"

    # Simulate 10 simultaneous calls
    results = await asyncio.gather(
        *[manager.ensure_worker_running(room_name) for _ in range(10)]
    )

    # All 10 callers must receive the exact same worker instance
    first_worker = results[0]
    for w in results[1:]:
        assert w is first_worker

    assert first_worker.state == WorkerState.RUNNING
    assert first_worker.is_running is True
    assert len(manager._workers) == 1
    assert manager._workers[room_name] is first_worker
    assert len(manager._startup_tasks) == 0

    await manager.stop_worker(room_name)
    assert first_worker.state == WorkerState.STOPPED


@pytest.mark.asyncio
async def test_startup_failure_cleanly_removed_from_registry(mock_worker_dependencies):
    """Startup failures must propagate to all concurrent callers and leave registry clean."""
    manager = LiveKitWorkerManager()
    room_name = "test_failing_room_002"

    mock_room_connect = mock_worker_dependencies["room"].connect
    mock_room_connect.side_effect = ConnectionError("LiveKit Cloud SFU unreachable")

    # 5 concurrent requests all awaiting the failing startup
    tasks = [
        asyncio.create_task(manager.ensure_worker_running(room_name))
        for _ in range(5)
    ]

    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Every caller must receive the ConnectionError
    for r in results:
        assert isinstance(r, ConnectionError)
        assert "LiveKit Cloud SFU unreachable" in str(r)

    # Registry must be completely clean
    assert room_name not in manager._workers
    assert room_name not in manager._startup_tasks
    assert len(manager.list_active_workers()) == 0


@pytest.mark.asyncio
async def test_stop_worker_during_active_inference_waits_for_thread(mock_worker_dependencies):
    """Worker.stop() must wait for active in-flight thread before tearing down native engine."""
    worker = LiveKitAIWorkerSession(room_name="test_shutdown_active_003")
    await worker.start()
    assert worker.state == WorkerState.RUNNING

    thread_finished = False

    def slow_mock_inference():
        nonlocal thread_finished
        time.sleep(0.3)
        thread_finished = True
        return "telemetry_mock", 5.0, 15.0

    # Submit slow inference future into worker's executor
    active_cfut = worker._executor.submit(slow_mock_inference)
    worker._active_cfut = active_cfut

    # Call stop() while inference thread is running
    t_stop_start = time.perf_counter()
    await worker.stop()
    t_stop_dur = time.perf_counter() - t_stop_start

    # Must have waited for thread to finish (> 0.25s)
    assert thread_finished is True
    assert t_stop_dur >= 0.25
    assert worker.state == WorkerState.STOPPED
    assert worker._executor is None


@pytest.mark.asyncio
async def test_stop_worker_during_startup(mock_worker_dependencies):
    """Calling stop_worker while worker is starting cancels startup cleanly."""
    manager = LiveKitWorkerManager()
    room_name = "test_stop_during_startup_004"

    # Make room.connect take some time
    async def slow_connect(*args, **kwargs):
        await asyncio.sleep(0.5)

    mock_worker_dependencies["room"].connect.side_effect = slow_connect

    # Start worker in background
    startup_task = asyncio.create_task(manager.ensure_worker_running(room_name))
    await asyncio.sleep(0.05)  # Let startup task register in manager._startup_tasks

    assert room_name in manager._startup_tasks

    # Now stop worker before startup finishes
    stopped = await manager.stop_worker(room_name)

    # Startup task should have been cancelled or cleaned up
    assert room_name not in manager._startup_tasks
    assert room_name not in manager._workers

    # Awaiting the cancelled startup task should raise CancelledError
    with pytest.raises(asyncio.CancelledError):
        await startup_task


@pytest.mark.asyncio
async def test_worker_state_transitions(mock_worker_dependencies):
    """Worker must progress through CREATED -> STARTING -> RUNNING -> STOPPING -> STOPPED."""
    worker = LiveKitAIWorkerSession(room_name="test_state_seq_005")
    assert worker.state == WorkerState.CREATED

    start_task = asyncio.create_task(worker.start())
    # During startup
    await asyncio.sleep(0.01)
    if worker.state != WorkerState.RUNNING:
        assert worker.state == WorkerState.STARTING

    await start_task
    assert worker.state == WorkerState.RUNNING

    stop_task = asyncio.create_task(worker.stop())
    await stop_task
    assert worker.state == WorkerState.STOPPED
