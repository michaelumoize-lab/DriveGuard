"""
Integration test connecting directly to LiveKit Cloud SFU.
"""

import asyncio
import sys
from pathlib import Path
import pytest

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import settings
from livekit import api, rtc


@pytest.mark.asyncio
async def test_livekit_cloud_connection():
    if not settings.LIVEKIT_URL or not settings.LIVEKIT_API_KEY or not settings.LIVEKIT_API_SECRET:
        pytest.skip("LiveKit Cloud credentials not configured in environment.")

    print(f"[TEST] Connecting to LiveKit Cloud: {settings.LIVEKIT_URL}...")
    
    room_name = "dg_healthcheck_room"
    token = (
        api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity("test_worker_probe")
        .with_grants(api.VideoGrants(room_join=True, room=room_name, can_publish_data=True))
        .to_jwt()
    )

    room = rtc.Room()

    try:
        await asyncio.wait_for(room.connect(settings.LIVEKIT_URL, token), timeout=10.0)
        print(f"[SUCCESS] Connected to LiveKit Cloud SFU! Room Name: {room.name}")
        print(f"[SUCCESS] Local Participant Identity: {room.local_participant.identity}")
        
        # Test publishing a tiny data ping over the WebRTC data channel
        await room.local_participant.publish_data(b'{"ping": true}', reliable=True)
        print("[SUCCESS] WebRTC Data Channel publish confirmed!")
        
    finally:
        await room.disconnect()
        print("[SUCCESS] Disconnected cleanly.")


if __name__ == "__main__":
    asyncio.run(test_livekit_cloud_connection())
