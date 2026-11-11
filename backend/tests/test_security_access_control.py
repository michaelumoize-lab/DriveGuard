"""
Unit and Integration Tests for Security and Access Control (Phase 1, DG-01, DG-20, DG-21).

Verifies:
1. All protected FastAPI routes reject unauthenticated requests with 401 Unauthorized.
2. Expired, forged, or malformed tokens are rejected with 401 Unauthorized.
3. Cross-user profile operations are blocked with 403 Forbidden.
4. Cross-user session operations are blocked with 403 Forbidden.
5. Unauthorized worker termination is blocked with 403 Forbidden.
6. WebSocket connections without valid token or with cross-user identity are rejected with code 1008.
7. Global exception handler suppresses stack traces and returns correlation_id.
"""

import time
import unittest
import jwt
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from backend.app.core.config import settings
from backend.app.main import app
from backend.app.services.session_manager import session_manager
from backend.app.services.livekit_worker import livekit_worker_manager, LiveKitAIWorkerSession


def make_token(
    user_id: str,
    role: str = "USER",
    expires_in: int = 300,
    secret: str = settings.AUTH_SECRET,
    issuer: str = settings.AUTH_TOKEN_ISSUER,
    audience: str = settings.AUTH_TOKEN_AUDIENCE,
) -> str:
    """Helper to mint test HS256 JWT tokens."""
    now = int(time.time())
    payload = {
        "sub": user_id,
        "role": role,
        "iss": issuer,
        "aud": audience,
        "iat": now,
        "exp": now + expires_in,
        "scope": ["*"],
    }
    return jwt.encode(payload, secret, algorithm="HS256")


class TestSecurityAccessControl(unittest.TestCase):
    """Test security boundaries across REST routes, WebSockets, and exceptions."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app, raise_server_exceptions=False)
        cls.alice_id = "user_alice_123"
        cls.bob_id = "user_bob_456"
        cls.alice_token = make_token(cls.alice_id)
        cls.bob_token = make_token(cls.bob_id)
        cls.admin_token = make_token("admin_user", role="ADMIN")

    def test_public_endpoints_accessible_without_auth(self) -> None:
        """Root pointers and health check remain public without tokens."""
        r_root = self.client.get("/")
        self.assertEqual(r_root.status_code, 200)

        r_health = self.client.get("/health")
        self.assertEqual(r_health.status_code, 200)

    def test_protected_routes_reject_unauthenticated(self) -> None:
        """Every protected route must return 401 when Authorization header is absent."""
        endpoints = [
            ("GET", "/api/v1/sessions"),
            ("POST", "/api/v1/sessions"),
            ("GET", "/api/v1/profiles"),
            ("POST", "/api/v1/profiles"),
            ("GET", f"/api/v1/profiles/{self.alice_id}"),
            ("DELETE", f"/api/v1/profiles/{self.alice_id}"),
            ("POST", "/api/v1/calibration/start"),
            ("GET", "/api/v1/calibration/sess_123/status"),
            ("POST", "/api/v1/calibration/cancel/sess_123"),
            ("GET", "/api/v1/livekit/token?room_name=test_room"),
            ("POST", "/api/v1/livekit/stop-worker?room_name=test_room"),
        ]
        for method, path in endpoints:
            with self.subTest(method=method, path=path):
                if method == "GET":
                    resp = self.client.get(path)
                elif method == "POST":
                    resp = self.client.post(path, json={})
                elif method == "DELETE":
                    resp = self.client.delete(path)
                self.assertEqual(resp.status_code, 401, f"{method} {path} must require auth")

    def test_reject_expired_token(self) -> None:
        """Expired tokens must be rejected with 401."""
        expired_token = make_token(self.alice_id, expires_in=-60)
        headers = {"Authorization": f"Bearer {expired_token}"}
        resp = self.client.get("/api/v1/sessions", headers=headers)
        self.assertEqual(resp.status_code, 401)
        self.assertIn("expired", resp.json().get("detail", "").lower())

    def test_reject_invalid_signature(self) -> None:
        """Tokens signed with wrong secret must be rejected with 401."""
        bad_token = make_token(self.alice_id, secret="wrong-secret-key-32-bytes-long-padding!!")
        headers = {"Authorization": f"Bearer {bad_token}"}
        resp = self.client.get("/api/v1/sessions", headers=headers)
        self.assertEqual(resp.status_code, 401)

    def test_reject_invalid_issuer_and_audience(self) -> None:
        """Tokens with mismatched issuer or audience must be rejected with 401."""
        bad_iss = make_token(self.alice_id, issuer="untrusted-issuer")
        self.assertEqual(
            self.client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {bad_iss}"}).status_code,
            401,
        )

        bad_aud = make_token(self.alice_id, audience="wrong-audience")
        self.assertEqual(
            self.client.get("/api/v1/sessions", headers={"Authorization": f"Bearer {bad_aud}"}).status_code,
            401,
        )

    def test_cross_user_profile_access_forbidden(self) -> None:
        """User Alice cannot read, write, or delete Bob's driver profile."""
        alice_headers = {"Authorization": f"Bearer {self.alice_token}"}

        # Alice cannot GET Bob's profile
        resp_get = self.client.get(f"/api/v1/profiles/{self.bob_id}", headers=alice_headers)
        self.assertEqual(resp_get.status_code, 403)

        # Alice cannot save a profile claiming to be Bob
        profile_payload = {
            "version": 1,
            "driver_id": self.bob_id,
            "calibrated_at": "2026-11-11T00:00:00Z",
            "is_calibrated": False,
            "calibration_mode": "UNCALIBRATED_DEFAULT",
            "zone_baselines": {},
        }
        resp_post = self.client.post("/api/v1/profiles", json=profile_payload, headers=alice_headers)
        self.assertEqual(resp_post.status_code, 403)

        # Alice cannot DELETE Bob's profile
        resp_del = self.client.delete(f"/api/v1/profiles/{self.bob_id}", headers=alice_headers)
        self.assertEqual(resp_del.status_code, 403)

    def test_cross_user_session_access_forbidden(self) -> None:
        """User Alice cannot create sessions for Bob, nor access Bob's active sessions."""
        alice_headers = {"Authorization": f"Bearer {self.alice_token}"}
        bob_headers = {"Authorization": f"Bearer {self.bob_token}"}

        # Alice attempts to create session for driver Bob
        resp_create = self.client.post(
            "/api/v1/sessions",
            json={"session_id": "sess_alice_for_bob", "driver_id": self.bob_id},
            headers=alice_headers,
        )
        self.assertEqual(resp_create.status_code, 403)

        # Bob creates a valid session
        bob_sess_id = "sess_bob_valid_01"
        r_bob = self.client.post(
            "/api/v1/sessions",
            json={"session_id": bob_sess_id, "driver_id": self.bob_id},
            headers=bob_headers,
        )
        self.assertEqual(r_bob.status_code, 200)

        # Alice attempts to GET Bob's session details -> 403
        r_steal = self.client.get(f"/api/v1/sessions/{bob_sess_id}", headers=alice_headers)
        self.assertEqual(r_steal.status_code, 403)

        # Alice attempts to terminate Bob's session -> 403
        r_kill = self.client.delete(f"/api/v1/sessions/{bob_sess_id}", headers=alice_headers)
        self.assertEqual(r_kill.status_code, 403)

        # Alice listing sessions does not show Bob's session
        r_list = self.client.get("/api/v1/sessions", headers=alice_headers)
        self.assertEqual(r_list.status_code, 200)
        alice_session_ids = [s["session_id"] for s in r_list.json()]
        self.assertNotIn(bob_sess_id, alice_session_ids)

        # Clean up Bob's session with Bob's token
        self.client.delete(f"/api/v1/sessions/{bob_sess_id}", headers=bob_headers)

    def test_cross_user_worker_stop_forbidden(self) -> None:
        """User Alice cannot stop a LiveKit worker owned by User Bob."""
        alice_headers = {"Authorization": f"Bearer {self.alice_token}"}

        # Register a mock active worker session owned by Bob
        bob_room = "room_bob_private"
        mock_worker = LiveKitAIWorkerSession(
            room_name=bob_room,
            driver_id=self.bob_id,
            owner_id=self.bob_id,
        )
        livekit_worker_manager._workers[bob_room] = mock_worker

        try:
            # Alice attempts to stop Bob's worker
            resp = self.client.post(
                f"/api/v1/livekit/stop-worker?room_name={bob_room}",
                headers=alice_headers,
            )
            self.assertEqual(resp.status_code, 403)
            self.assertIn("owned by", resp.json().get("detail", "").lower())
        finally:
            livekit_worker_manager._workers.pop(bob_room, None)

    def test_websocket_unauthenticated_rejected(self) -> None:
        """WebSocket connection without token or with invalid token must be rejected with 1008."""
        with self.client.websocket_connect("/ws/v1/session/sess_ws_unauth") as ws:
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "FATAL_ERROR")
            self.assertIn("unauthorized", msg["detail"].lower())
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 1008)

        bad_token = make_token(self.alice_id, secret="wrong-secret-32-bytes-padding-long!")
        with self.client.websocket_connect(f"/ws/v1/session/sess_ws_unauth?token={bad_token}") as ws:
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "FATAL_ERROR")
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 1008)

    def test_websocket_cross_user_rejected(self) -> None:
        """Connecting with Alice's token while asserting Bob's driver_id must be rejected with 1008."""
        with self.client.websocket_connect(
            f"/ws/v1/session/sess_ws_impersonate?token={self.alice_token}&driver_id={self.bob_id}"
        ) as ws:
            msg = ws.receive_json()
            self.assertEqual(msg["type"], "FATAL_ERROR")
            self.assertIn("forbidden", msg["detail"].lower())
            with self.assertRaises(WebSocketDisconnect) as ctx:
                ws.receive_json()
            self.assertEqual(ctx.exception.code, 1008)

    def test_global_exception_handler_sanitizes_traceback(self) -> None:
        """Server exceptions return structured JSON with correlation_id and no raw tracebacks in response."""
        # Querying nonexistent endpoint with invalid syntax or triggering 500
        # Check global exception handler response contract directly
        from fastapi import Request
        from backend.app.main import global_exception_handler
        import asyncio

        scope = {
            "type": "http",
            "method": "GET",
            "path": "/api/v1/simulated-crash",
            "headers": [],
        }
        fake_req = Request(scope)
        fake_exc = RuntimeError("Simulated internal database crash with secret db://pass@host:5432")

        res = asyncio.run(global_exception_handler(fake_req, fake_exc))
        self.assertEqual(res.status_code, 500)
        import json
        body = json.loads(res.body.decode("utf-8"))
        self.assertEqual(body["status"], "ERROR")
        self.assertIn("correlation_id", body)
        self.assertNotIn("traceback", body)
        self.assertNotIn("db://pass@host", body["message"])


if __name__ == "__main__":
    unittest.main()
