"""
Stage 8 Backend <-> Frontend Contract & LiveKit Synchronization Tests.

Verifies:
1. FastAPI OpenAPI schema matches every REST route documented in docs/FRONTEND_BACKEND_CONTRACT.md.
2. Every authoritative ReasonCode in backend/app/intelligence/reason_codes.py is cataloged in frontend/types/telemetry.ts.
3. DriverProfileDTO accepts exact snake_case JSON payloads including blink_rcr_median, blink_rcr_mad,
   neutral_rot_mat, and calibration_is_mirrored.
4. LiveKitAIWorkerSession does NOT unexpectedly auto-start 1-zone calibration for `cal_*` rooms,
   handles explicit START_CALIBRATION (including include_blink_baseline) and RECALIBRATE_ZONE ("BLINK_CADENCE"),
   and cleans up both worker and session_manager session on authenticated stop-worker.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from backend.app.core.auth import create_backend_access_token
from backend.app.intelligence.reason_codes import ReasonCode
from backend.app.main import app
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.livekit_worker import LiveKitAIWorkerSession, livekit_worker_manager
from backend.app.services.session_manager import session_manager


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_openapi_exposes_all_documented_contract_endpoints():
    schema = app.openapi()
    paths = schema["paths"]

    required_routes = [
        ("/", "get"),
        ("/health", "get"),
        ("/health/liveness", "get"),
        ("/health/readiness", "get"),
        ("/api/v1/sessions", "get"),
        ("/api/v1/sessions", "post"),
        ("/api/v1/sessions/{session_id}", "get"),
        ("/api/v1/sessions/{session_id}", "delete"),
        ("/api/v1/sessions/{session_id}/experiment", "post"),
        ("/api/v1/sessions/{session_id}/frame", "post"),
        ("/api/v1/sessions/{session_id}/telemetry", "get"),
        ("/api/v1/profiles", "get"),
        ("/api/v1/profiles", "post"),
        ("/api/v1/profiles/{driver_id}", "get"),
        ("/api/v1/profiles/{driver_id}", "delete"),
        ("/api/v1/calibration/start", "post"),
        ("/api/v1/calibration/cancel/{session_id}", "post"),
        ("/api/v1/calibration/{session_id}/status", "get"),
        ("/api/v1/livekit/token", "get"),
        ("/api/v1/livekit/stop-worker", "post"),
    ]

    for route, method in required_routes:
        assert route in paths, f"Missing route in OpenAPI schema: {route}"
        assert method in paths[route], f"Missing method {method.upper()} on {route}"


def test_all_backend_reason_codes_present_in_frontend_catalog():
    ts_path = REPO_ROOT / "frontend" / "types" / "telemetry.ts"
    ts_text = ts_path.read_text(encoding="utf-8")
    for rc in ReasonCode:
        assert f"{rc.value}:" in ts_text, (
            f"Authoritative ReasonCode '{rc.value}' missing from frontend REASON_CODE_CATALOG!"
        )


def test_profile_dto_accepts_frontend_snake_case_payload_with_pose_metadata():
    payload = {
        "version": 1,
        "driver_id": "driver_sync_test",
        "calibrated_at": "2026-11-20T05:00:00Z",
        "is_calibrated": True,
        "calibration_mode": "FOUR_ZONE",
        "center_ear_median": 0.295,
        "blink_duration_median_ms": 225.0,
        "blink_duration_mad_ms": 38.0,
        "blink_rcr_median": 1.02,
        "blink_rcr_mad": 0.17,
        "blink_freq_median_per_min": 16.5,
        "blink_freq_mad_per_min": 4.2,
        "baseline_perclos_pct": 3.1,
        "blink_amplitude_median": 0.145,
        "temporal_blinks_sampled": 28,
        "temporal_window_sec": 120.0,
        "sensitivity_level": "STANDARD",
        "neutral_pitch_deg": -1.5,
        "neutral_yaw_deg": 0.8,
        "neutral_rot_mat": [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        "calibration_is_mirrored": True,
        "calibration_reliability": 0.93,
        "zone_baselines": {
            "CENTER": {
                "zone": "CENTER",
                "ear_median": 0.295,
                "ear_mad_scaled": 0.017,
                "dynamic_closure_threshold": 0.212,
                "pitch_median_deg": -1.5,
                "yaw_median_deg": 0.8,
                "sample_count": 35,
            }
        },
    }

    dto = DriverProfileDTO(**payload)
    assert dto.blink_rcr_median == pytest.approx(1.02)
    assert dto.blink_rcr_mad == pytest.approx(0.17)
    assert dto.neutral_rot_mat == [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert dto.calibration_is_mirrored is True
    assert dto.zone_baselines["CENTER"].dynamic_closure_threshold == pytest.approx(0.212)


@pytest.mark.asyncio
async def test_livekit_cal_room_does_not_auto_start_and_handles_explicit_calibration_commands():
    room_name = "cal_sync_test_room"
    driver_id = "driver_cal_sync"
    worker = LiveKitAIWorkerSession(
        room_name=room_name,
        driver_id=driver_id,
        owner_id=driver_id,
    )

    mock_room = MagicMock()
    mock_room.connect = AsyncMock()
    mock_room.disconnect = AsyncMock()
    mock_room.local_participant = MagicMock()
    mock_room.local_participant.publish_data = AsyncMock()
    worker.room = mock_room

    try:
        with patch("backend.app.services.livekit_worker.settings") as mock_settings:
            mock_settings.LIVEKIT_API_KEY = "API_TEST_KEY"
            mock_settings.LIVEKIT_API_SECRET = "SUPER_SECRET_TEST_KEY_LONG_ENOUGH"
            mock_settings.LIVEKIT_URL = "wss://test.livekit.cloud"
            await worker.start()

        # Verify cal_ prefix no longer races by auto-starting 1-zone calibration
        assert worker.engine is not None
        assert worker.engine.is_calibrating is False
        assert worker.engine.get_calibration_status() is None

        # Now send explicit START_CALIBRATION with 4-zone + 2-min blink baseline
        await worker._handle_client_control_message({
            "action": "START_CALIBRATION",
            "four_zone": True,
            "zone_duration_sec": 5.0,
            "include_blink_baseline": True,
            "blink_baseline_duration_sec": 120.0,
        })
        cal_status = worker.engine.get_calibration_status()
        assert cal_status is not None
        assert cal_status.current_zone == "CENTER"
        assert cal_status.total_phases == 5

        # Send RECALIBRATE_ZONE for BLINK_CADENCE directly
        await worker._handle_client_control_message({
            "action": "RECALIBRATE_ZONE",
            "zone": "BLINK_CADENCE",
            "blink_baseline_duration_sec": 60.0,
        })
        blink_status = worker.engine.get_calibration_status()
        assert blink_status is not None
        assert blink_status.current_zone == "BLINK_CADENCE"
        assert blink_status.phase_type == "BLINK_CADENCE"
    finally:
        await worker.stop()
        assert session_manager.get_session(room_name) is None


def test_livekit_token_and_stop_worker_rest_lifecycle_with_mocked_sfu():
    client = TestClient(app)
    token = create_backend_access_token(user_id="driver_lk_e2e", role="USER")
    headers = {"Authorization": f"Bearer {token}"}
    room_name = "live_e2e_sync_room"

    with (
        patch("backend.app.api.routes.livekit_token.settings") as route_settings,
        patch("backend.app.services.livekit_worker.settings") as worker_settings,
        patch("backend.app.services.livekit_worker.rtc.Room") as mock_room_cls,
    ):
        for s in (route_settings, worker_settings):
            s.LIVEKIT_API_KEY = "API_TEST_KEY"
            s.LIVEKIT_API_SECRET = "SUPER_SECRET_TEST_KEY_LONG_ENOUGH"
            s.LIVEKIT_URL = "wss://test.livekit.cloud"

        mock_room = MagicMock()
        mock_room.connect = AsyncMock()
        mock_room.disconnect = AsyncMock()
        mock_room.local_participant = MagicMock()
        mock_room.local_participant.publish_data = AsyncMock()
        mock_room_cls.return_value = mock_room

        # 1. Mint token & auto-start worker
        res = client.get(
            f"/api/v1/livekit/token?room_name={room_name}&driver_id=driver_lk_e2e&auto_start_worker=true",
            headers=headers,
        )
        assert res.status_code == 200, res.text
        data = res.json()
        assert data["room_name"] == room_name
        assert data["url"] == "wss://test.livekit.cloud"
        assert data["token"]
        assert livekit_worker_manager.get_worker(room_name) is not None
        assert session_manager.get_session(room_name) is not None

        # 2. Unauthorized stop-worker must be rejected (401)
        unauth_stop = client.post(f"/api/v1/livekit/stop-worker?room_name={room_name}")
        assert unauth_stop.status_code == 401

        # 3. Authenticated stop-worker cleans up worker and session
        stop_res = client.post(
            f"/api/v1/livekit/stop-worker?room_name={room_name}",
            headers=headers,
        )
        assert stop_res.status_code == 200
        assert stop_res.json()["worker_stopped"] is True
        assert livekit_worker_manager.get_worker(room_name) is None
        assert session_manager.get_session(room_name) is None
