"""
DriveGuard — Strict Loopback Access Guard for Local Python Camera & Telemetry API.

Security guarantees (Decision B1):
1. Hard opt-in flag (`settings.LOCAL_CAMERA_API_ENABLED`) that defaults to False
   in cloud/production deployments (e.g. Render).
2. Unconditional rejection of reverse-proxied requests carrying forwarding headers
   (`X-Forwarded-For`, `Forwarded`, `X-Real-IP`, `CF-Connecting-IP`, `True-Client-IP`,
   `X-Client-IP`).
3. Strict peer IP verification: only true IPv4/IPv6 loopback peers (`127.0.0.1`, `::1`,
   `::ffff:127.0.0.1`) are permitted.
4. Strict `Host` and `Origin` header validation preventing DNS-rebinding or cross-site
   hijacking of the local camera hardware from remote web pages.
"""

from __future__ import annotations

import ipaddress
import logging
from typing import Mapping, Optional
from urllib.parse import urlparse

from fastapi import HTTPException, Request, WebSocket, status

from backend.app.core.config import settings

logger = logging.getLogger("driveguard.local_guard")

FORBIDDEN_PROXY_HEADERS = frozenset(
    {
        "x-forwarded-for",
        "forwarded",
        "x-real-ip",
        "cf-connecting-ip",
        "true-client-ip",
        "x-client-ip",
    }
)

ALLOWED_LOOPBACK_HOSTNAMES = frozenset({"localhost", "127.0.0.1", "::1"})


def is_loopback_ip(host: Optional[str]) -> bool:
    """Return True strictly if `host` parses as a valid IPv4/IPv6 loopback address."""
    if not host or not isinstance(host, str):
        return False
    candidate = host.strip()
    if candidate.startswith("[") and candidate.endswith("]"):
        candidate = candidate[1:-1]
    try:
        addr = ipaddress.ip_address(candidate)
        if isinstance(addr, ipaddress.IPv6Address) and addr.ipv4_mapped is not None:
            return addr.ipv4_mapped.is_loopback
        return addr.is_loopback
    except ValueError:
        return False


def _extract_hostname_from_host_header(host_header: str) -> Optional[str]:
    """Parse hostname (without port) from an HTTP Host header."""
    raw = host_header.strip().lower()
    if not raw:
        return None
    if raw.startswith("["):
        end_bracket = raw.find("]")
        if end_bracket == -1:
            return None
        return raw[1:end_bracket]
    if ":" in raw:
        # IPv4/hostname with optional :port (multiple colons without brackets is invalid Host)
        if raw.count(":") > 1:
            return None
        return raw.split(":", 1)[0]
    return raw


def _is_allowed_hostname(hostname: Optional[str]) -> bool:
    if not hostname:
        return False
    normalized = hostname.strip().lower()
    if normalized in ALLOWED_LOOPBACK_HOSTNAMES:
        return True
    return is_loopback_ip(normalized)


def assert_loopback_access(
    client_host: Optional[str],
    headers: Mapping[str, str],
) -> None:
    """
    Validate all loopback security invariants or raise HTTPException(403).
    Used identically for HTTP REST, MJPEG streaming, and WebSocket handshakes.
    """
    if not settings.LOCAL_CAMERA_API_ENABLED:
        logger.warning("Blocked local camera API access: LOCAL_CAMERA_API_ENABLED is False.")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local camera API is disabled in this environment.",
        )

    # Reject any reverse-proxy forwarding header unconditionally
    lowered_headers = {k.lower(): v for k, v in headers.items()}
    for forbidden in FORBIDDEN_PROXY_HEADERS:
        if forbidden in lowered_headers and str(lowered_headers[forbidden]).strip():
            logger.warning(
                "Blocked local camera API request containing proxy header '%s'.",
                forbidden,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Reverse-proxied requests are forbidden on local camera endpoints.",
            )

    # Verify peer IP is strictly loopback
    if not is_loopback_ip(client_host):
        logger.warning(
            "Blocked local camera API request from non-loopback peer '%s'.",
            client_host,
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Local camera endpoints are restricted to loopback clients (127.0.0.1 / ::1).",
        )

    # Validate Host header (prevents DNS rebinding)
    host_header = lowered_headers.get("host")
    if host_header:
        hostname = _extract_hostname_from_host_header(host_header)
        if not _is_allowed_hostname(hostname):
            logger.warning(
                "Blocked local camera API request with non-loopback Host header '%s'.",
                host_header,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Host header must target a loopback interface.",
            )

    # Validate Origin header if present (prevents cross-origin browser hijacking)
    origin_header = lowered_headers.get("origin")
    if origin_header:
        parsed = urlparse(origin_header)
        if parsed.scheme not in ("http", "https") or not _is_allowed_hostname(parsed.hostname):
            logger.warning(
                "Blocked local camera API request from non-loopback Origin '%s'.",
                origin_header,
            )
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Origin header must be a local loopback origin.",
            )


def require_loopback_http(request: Request) -> None:
    """FastAPI dependency enforcing loopback-only access for HTTP and MJPEG routes."""
    client_host = request.client.host if request.client else None
    assert_loopback_access(client_host=client_host, headers=request.headers)


def require_loopback_ws(websocket: WebSocket) -> None:
    """FastAPI dependency enforcing loopback-only access prior to WebSocket accept()."""
    client_host = websocket.client.host if websocket.client else None
    assert_loopback_access(client_host=client_host, headers=websocket.headers)
