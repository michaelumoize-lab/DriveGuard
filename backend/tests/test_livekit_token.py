"""
Unit verification test for LiveKit Cloud credentials and token generation.
"""

import sys
from pathlib import Path
import jwt

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from backend.app.core.config import settings
from livekit import api


def test_token_minting():
    print(f"[TEST] LiveKit URL: {settings.LIVEKIT_URL}")
    print(f"[TEST] LiveKit API Key: {settings.LIVEKIT_API_KEY[:6]}...")
    
    assert settings.LIVEKIT_URL, "LIVEKIT_URL must be defined"
    assert settings.LIVEKIT_API_KEY, "LIVEKIT_API_KEY must be defined"
    assert settings.LIVEKIT_API_SECRET, "LIVEKIT_API_SECRET must be defined"

    room_name = "test_room_123"
    driver_id = "test_driver"

    grant = api.VideoGrants(
        room_join=True,
        room=room_name,
        can_publish=True,
        can_subscribe=True,
        can_publish_data=True,
    )
    token_jwt = (
        api.AccessToken(settings.LIVEKIT_API_KEY, settings.LIVEKIT_API_SECRET)
        .with_identity(f"driver_{driver_id}")
        .with_grants(grant)
        .to_jwt()
    )

    assert isinstance(token_jwt, str) and len(token_jwt) > 20, "JWT must be a valid non-empty string"

    # Decode without verification to inspect claims
    decoded = jwt.decode(token_jwt, options={"verify_signature": False})
    print(f"[TEST] Decoded JWT Claims: {decoded}")
    
    assert decoded["sub"] == f"driver_{driver_id}"
    video_grants = decoded.get("video", {})
    assert video_grants.get("room") == room_name
    assert video_grants.get("roomJoin") is True

    print("\n[SUCCESS] LiveKit Token generation verified successfully!")


if __name__ == "__main__":
    test_token_minting()
