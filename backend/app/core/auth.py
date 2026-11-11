"""
DriveGuard — Backend Authentication & Access Control Module.

Implements:
1. Cryptographically verified backend access tokens (HS256 JWT) minted by Next.js server.
2. Claim verification: signature, issuer ('driveguard-frontend'), audience ('driveguard-backend'),
   expiration timestamp, and non-empty subject ('sub' = verified user identity).
3. FastAPI dependency injection (`get_current_user`) for strict REST route protection.
4. Ownership validation (`verify_resource_ownership`) preventing cross-user access to
   driver profiles, monitoring sessions, LiveKit rooms, and workers.
5. WebSocket token authentication and connection authorization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import jwt
from fastapi import Depends, Header, HTTPException, Query, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from backend.app.core.config import settings

logger = logging.getLogger("driveguard.auth")
security_scheme = HTTPBearer(auto_error=False)


@dataclass(frozen=True)
class AuthenticatedUser:
    """Authenticated user identity and authorization claims extracted from backend token."""

    user_id: str
    role: str = "USER"
    scopes: List[str] = field(default_factory=lambda: ["*"])
    claims: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_admin(self) -> bool:
        return self.role.upper() == "ADMIN"


def verify_backend_token(token: str) -> AuthenticatedUser:
    """
    Verify and decode a signed backend access token.
    Raises HTTPException(401) on invalid signature, expiration, or claim mismatches.
    """
    if not token or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token is missing or empty.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    clean_token = token.strip()
    if clean_token.lower().startswith("bearer "):
        clean_token = clean_token[7:].strip()

    secret = settings.AUTH_SECRET
    if not secret:
        logger.error("[Auth] AUTH_SECRET is not configured on the backend.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication secret is misconfigured.",
        )

    try:
        payload = jwt.decode(
            clean_token,
            secret,
            algorithms=["HS256"],
            audience=settings.AUTH_TOKEN_AUDIENCE,
            issuer=settings.AUTH_TOKEN_ISSUER,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidIssuerError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has invalid issuer.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidAudienceError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token has invalid audience.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except jwt.InvalidTokenError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid authentication token: {exc}",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as exc:
        logger.error(f"[Auth] Unexpected error during token validation: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication token validation failed.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    sub = payload.get("sub")
    if not sub or not isinstance(sub, str) or not sub.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token 'sub' claim must contain a valid user identity.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    role = str(payload.get("role", "USER"))
    raw_scopes = payload.get("scope", ["*"])
    scopes = raw_scopes if isinstance(raw_scopes, list) else [str(raw_scopes)]

    return AuthenticatedUser(
        user_id=sub.strip(),
        role=role,
        scopes=scopes,
        claims=payload,
    )


async def get_current_user(
    auth_header: Optional[HTTPAuthorizationCredentials] = Depends(security_scheme),
    raw_header: Optional[str] = Header(None, alias="Authorization"),
) -> AuthenticatedUser:
    """
    FastAPI dependency that enforces authentication on protected endpoints.
    Extracts Bearer token from HTTP Authorization header.
    """
    token: Optional[str] = None
    if auth_header and auth_header.credentials:
        token = auth_header.credentials
    elif raw_header:
        token = raw_header

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Expected 'Bearer <token>'.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return verify_backend_token(token)


def verify_resource_ownership(
    current_user: AuthenticatedUser,
    resource_owner_id: str,
    resource_name: str = "resource",
) -> None:
    """
    Verify that the authenticated user owns the given resource or holds ADMIN privilege.
    Raises HTTPException(403) on ownership violation.
    """
    if current_user.is_admin:
        return

    if current_user.user_id != resource_owner_id:
        logger.warning(
            f"[Auth] Ownership violation: User '{current_user.user_id}' attempted to access "
            f"{resource_name} owned by '{resource_owner_id}'."
        )
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Forbidden: You do not have permission to access or modify this {resource_name}.",
        )


def verify_ws_token(token: Optional[str]) -> AuthenticatedUser:
    """
    Verify authentication token for WebSocket connections.
    Raises ValueError on authentication failure.
    """
    if not token or not token.strip():
        raise ValueError("WebSocket authentication token is required.")

    try:
        return verify_backend_token(token)
    except HTTPException as exc:
        raise ValueError(exc.detail)
