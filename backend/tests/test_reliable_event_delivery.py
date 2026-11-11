"""
Unit Tests for Workstream E — Reliable Safety-Event Delivery & Bounded Retries.

Verifies:
1. Client EVENT_ACK immediately cancels background retry task and confirms delivery.
2. Unacknowledged events are retried up to 3 bounded attempts, then marked exhausted without looping.
3. Memory bounding: unacknowledged events dict is capped at max_unacked_events (100).
4. TTL eviction: events older than unacked_ttl_sec are pruned.
5. Event IDs remain strictly stable across retries with monotonic sequence numbers.
"""

import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.app.services.livekit_worker import (
    LiveKitAIWorkerSession,
    WorkerState,
)


@pytest.fixture
def mock_worker():
    """Create an isolated LiveKitAIWorkerSession with mocked network transport."""
    with patch("backend.app.services.livekit_worker.rtc.Room") as mock_room_cls:
        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.disconnect = AsyncMock()
        mock_room.local_participant = MagicMock()
        mock_room.local_participant.publish_data = AsyncMock()
        mock_room_cls.return_value = mock_room

        worker = LiveKitAIWorkerSession(room_name="test_event_delivery_room")
        worker.room = mock_room
        worker.is_running = True
        worker.state = WorkerState.RUNNING

        yield worker


@pytest.mark.asyncio
async def test_event_ack_cancels_retry_immediately(mock_worker):
    """Client EVENT_ACK must immediately cancel background retry task and pop unacked event."""
    event_payload = {
        "v": 2,
        "type": "SAFETY_EVENT",
        "event_id": "evt_test_001",
        "session_id": mock_worker.room_name,
        "event_seq": "1",
        "previous_state": "NORMAL",
        "new_state": "WARNING",
    }

    await mock_worker._publish_reliable_safety_event(event_payload)

    event_id = "evt_test_001"
    assert event_id in mock_worker._unacknowledged_events
    assert event_id in mock_worker._retry_tasks
    retry_task = mock_worker._retry_tasks[event_id]
    assert not retry_task.done()

    # Simulate client sending EVENT_ACK message
    ack_msg = {
        "action": "EVENT_ACK",
        "event_id": event_id,
        "event_seq": "1",
    }
    await mock_worker._handle_client_control_message(ack_msg)
    await asyncio.sleep(0)  # Yield to event loop so cancelled task processes cancellation

    # Event popped and retry task cancelled
    assert event_id not in mock_worker._unacknowledged_events
    assert event_id not in mock_worker._retry_tasks
    assert retry_task.cancelled() or retry_task.done()


@pytest.mark.asyncio
async def test_retry_exhaustion_after_3_attempts(mock_worker):
    """Unacknowledged events must retry exactly 3 times and terminate without looping."""
    event_payload = {
        "v": 2,
        "type": "SAFETY_EVENT",
        "event_id": "evt_test_exhaust",
        "session_id": mock_worker.room_name,
        "event_seq": "2",
        "previous_state": "NORMAL",
        "new_state": "CRITICAL",
    }

    # Speed up retry sleep intervals for fast deterministic test execution
    with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
        await mock_worker._publish_reliable_safety_event(event_payload)
        retry_task = mock_worker._retry_tasks["evt_test_exhaust"]
        await retry_task

    entry = mock_worker._unacknowledged_events.get("evt_test_exhaust")
    assert entry is not None
    assert entry["retries"] == 3
    assert entry.get("exhausted") is True
    # Initial publish + 3 retries = 4 publish_data calls
    assert mock_worker.room.local_participant.publish_data.call_count == 4
    assert "evt_test_exhaust" not in mock_worker._retry_tasks


@pytest.mark.asyncio
async def test_unacknowledged_events_bounded_capacity(mock_worker):
    """Unacknowledged events dict must cap at _max_unacked_events (100) and evict oldest."""
    mock_worker._max_unacked_events = 10  # small bound for unit test

    with patch("asyncio.sleep", new_callable=AsyncMock):
        for i in range(15):
            payload = {
                "v": 2,
                "type": "SAFETY_EVENT",
                "event_id": f"evt_bound_{i}",
                "session_id": mock_worker.room_name,
                "event_seq": str(i),
                "previous_state": "NORMAL",
                "new_state": "WARNING",
            }
            await mock_worker._publish_reliable_safety_event(payload)

    # Size must not exceed capacity
    assert len(mock_worker._unacknowledged_events) <= 10
    # Earliest events (0, 1, 2, 3, 4) should have been evicted
    assert "evt_bound_0" not in mock_worker._unacknowledged_events
    assert "evt_bound_14" in mock_worker._unacknowledged_events


@pytest.mark.asyncio
async def test_unacknowledged_events_ttl_eviction(mock_worker):
    """Expired unacknowledged events exceeding _unacked_ttl_sec must be pruned."""
    mock_worker._unacked_ttl_sec = 1.0  # 1s TTL

    with patch("asyncio.sleep", new_callable=AsyncMock):
        payload1 = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_ttl_old",
            "session_id": mock_worker.room_name,
            "event_seq": "10",
            "previous_state": "NORMAL",
            "new_state": "WARNING",
        }
        await mock_worker._publish_reliable_safety_event(payload1)
        assert "evt_ttl_old" in mock_worker._unacknowledged_events

        # Simulate time aging beyond TTL
        mock_worker._unacknowledged_events["evt_ttl_old"]["sent_at"] = time.perf_counter() - 5.0

        payload2 = {
            "v": 2,
            "type": "SAFETY_EVENT",
            "event_id": "evt_ttl_new",
            "session_id": mock_worker.room_name,
            "event_seq": "11",
            "previous_state": "WARNING",
            "new_state": "CRITICAL",
        }
        await mock_worker._publish_reliable_safety_event(payload2)

    # Old event must have been evicted by TTL
    assert "evt_ttl_old" not in mock_worker._unacknowledged_events
    assert "evt_ttl_new" in mock_worker._unacknowledged_events


@pytest.mark.asyncio
async def test_stable_event_id_across_retries(mock_worker):
    """Event IDs and sequences must remain identical across all transmission retries."""
    event_payload = {
        "v": 2,
        "type": "SAFETY_EVENT",
        "event_id": "evt_stable_id_123",
        "session_id": mock_worker.room_name,
        "event_seq": "42",
        "previous_state": "NORMAL",
        "new_state": "WARNING",
    }

    published_payloads = []

    async def capture_publish(payload_bytes, reliable=True):
        data = json.loads(payload_bytes.decode("utf-8"))
        published_payloads.append(data)

    mock_worker.room.local_participant.publish_data.side_effect = capture_publish

    with patch("asyncio.sleep", new_callable=AsyncMock):
        await mock_worker._publish_reliable_safety_event(event_payload)
        retry_task = mock_worker._retry_tasks["evt_stable_id_123"]
        await retry_task

    assert len(published_payloads) == 4  # 1 initial + 3 retries
    for p in published_payloads:
        assert p["event_id"] == "evt_stable_id_123"
        assert p["event_seq"] == "42"
        assert p["new_state"] == "WARNING"
