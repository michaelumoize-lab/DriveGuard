"""
DriveGuard — Phase 4A Security Tests for Loopback-Only Local Camera API Guard.

Verifies:
- LOCAL_CAMERA_API_ENABLED=False returns 403 Forbidden on HTTP, MJPEG, and WebSocket
- Non-loopback client IPs (203.0.113.10, 192.168.1.50, 'testclient') return 403 Forbidden
- Reverse-proxy headers (X-Forwarded-For, Forwarded, X-Real-IP, CF-Connecting-IP) return 403 Forbidden
- Non-loopback Host or Origin headers return 403 Forbidden
- True loopback peer (127.0.0.1 / ::1) with loopback Host & Origin succeeds
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.core.config import settings
from backend.app.main import app


def test_local_api_blocked_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LOCAL_CAMERA_API_ENABLED", False)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))

    for method, path in [
        ("GET", "/api/v1/local/status"),
        ("GET", "/api/v1/local/devices"),
        ("POST", "/api/v1/local/start"),
        ("POST", "/api/v1/local/stop"),
        ("GET", "/api/v1/local/preview.mjpg"),
        ("GET", "/api/v1/local/sessions"),
    ]:
        res = client.request(method, path)
        assert res.status_code == 403, f"Expected 403 when disabled for {method} {path}, got {res.status_code}"

    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/ws/local/telemetry"):
            pass


@pytest.mark.parametrize("peer_ip", ["203.0.113.10", "192.168.1.50", "10.0.0.4", "testclient"])
def test_local_api_rejects_non_loopback_peers(monkeypatch: pytest.MonkeyPatch, peer_ip: str) -> None:
    monkeypatch.setattr(settings, "LOCAL_CAMERA_API_ENABLED", True)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=(peer_ip, 50000))

    res = client.get("/api/v1/local/status")
    assert res.status_code == 403

    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/ws/local/telemetry"):
            pass


@pytest.mark.parametrize(
    "header_name,header_val",
    [
        ("X-Forwarded-For", "127.0.0.1"),
        ("Forwarded", "for=127.0.0.1;proto=http"),
        ("X-Real-IP", "127.0.0.1"),
        ("CF-Connecting-IP", "198.51.100.7"),
        ("True-Client-IP", "127.0.0.1"),
    ],
)
def test_local_api_rejects_reverse_proxy_headers(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
    header_val: str,
) -> None:
    monkeypatch.setattr(settings, "LOCAL_CAMERA_API_ENABLED", True)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))

    res = client.get("/api/v1/local/status", headers={header_name: header_val})
    assert res.status_code == 403

    with pytest.raises((WebSocketDisconnect, Exception)):
        with client.websocket_connect("/ws/local/telemetry", headers={header_name: header_val}):
            pass


def test_local_api_rejects_non_loopback_host_and_origin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "LOCAL_CAMERA_API_ENABLED", True)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=("127.0.0.1", 50000))

    res_origin = client.get(
        "/api/v1/local/status",
        headers={"Origin": "https://drive-guard-mu.vercel.app"},
    )
    assert res_origin.status_code == 403

    res_evil_origin = client.get(
        "/api/v1/local/status",
        headers={"Origin": "https://evil.example.com"},
    )
    assert res_evil_origin.status_code == 403

    res_host = client.get(
        "/api/v1/local/status",
        headers={"Host": "evil.example.com:8000"},
    )
    assert res_host.status_code == 403


@pytest.mark.parametrize("peer_ip", ["127.0.0.1", "::1"])
def test_local_api_allows_genuine_loopback_with_local_origin(
    monkeypatch: pytest.MonkeyPatch,
    peer_ip: str,
) -> None:
    monkeypatch.setattr(settings, "LOCAL_CAMERA_API_ENABLED", True)
    client = TestClient(app, base_url="http://127.0.0.1:8000", client=(peer_ip, 50000))

    res = client.get(
        "/api/v1/local/status",
        headers={"Origin": "http://localhost:3000"},
    )
    assert res.status_code == 200
    body = res.json()
    assert body["engine_type"] == "PYTHON"
    assert "running" in body

    with client.websocket_connect(
        "/ws/local/telemetry",
        headers={"Host": "127.0.0.1:8000", "Origin": "http://localhost:3000"},
    ) as ws:
        initial = ws.receive_json()
        assert initial["type"] == "STATUS"
        assert initial["engine_type"] == "PYTHON"
