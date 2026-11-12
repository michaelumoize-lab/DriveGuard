"""
DriveGuard — Camera Configuration Module.

1. What it does:
   Defines configuration schemas and parameter validations for optical camera
   acquisition devices (e.g., laptop webcam, external UVC USB camera, phone stream).

2. Why it exists:
   Decouples hardware settings (device index, resolution, target FPS, Windows video
   backend API) from the capture implementation. Allows seamless switching between
   the HP EliteBook built-in webcam, external Arducam IMX462, and mobile phone feeds.

3. What data it receives:
   User-specified or default configuration arguments (integer indices, resolution tuples,
   backend preferences).

4. What data it produces:
   Immutable, validated `CameraConfig` objects consumed by camera adapters.

5. Modules that depend on it:
   - `backend.app.camera.base`
   - `backend.app.camera.opencv_camera`
   - `backend.app.camera.phone_camera`
   - `scripts.run_camera_test`

6. Failure behavior:
   Raises `ValueError` immediately on invalid dimensions, negative frame rates, or
   unsupported backend identifiers before any hardware resource is allocated.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Tuple, Union
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def sanitize_source_identifier(source: Union[int, str]) -> str:
    """
    Produce a log- and display-safe string identifier from a camera device index or stream URL.
    Redacts user credentials and sensitive query parameters to prevent leaking secrets into logs.
    """
    if not isinstance(source, str):
        return str(source)

    try:
        parts = urlsplit(source)
    except Exception:
        return str(source)

    if not parts.scheme:
        return source

    netloc = parts.netloc
    if parts.username or parts.password:
        host_port = parts.hostname or ""
        if parts.port:
            host_port = f"{host_port}:{parts.port}"
        if parts.username and parts.password:
            netloc = f"{parts.username}:***@{host_port}"
        elif parts.password:
            netloc = f"***@{host_port}"
        elif parts.username:
            netloc = f"{parts.username}@{host_port}"

    query = parts.query
    if query:
        sensitive_keys = {
            "token",
            "auth",
            "key",
            "secret",
            "password",
            "pass",
            "pwd",
            "access_token",
            "api_key",
            "apikey",
        }
        sanitized_pairs = []
        for k, v in parse_qsl(query, keep_blank_values=True):
            if k.lower() in sensitive_keys or any(
                s in k.lower() for s in ("token", "secret", "pass", "auth")
            ):
                sanitized_pairs.append((k, "***"))
            else:
                sanitized_pairs.append((k, v))
        query = urlencode(sanitized_pairs, safe="*")

    return urlunsplit((parts.scheme, netloc, parts.path, query, parts.fragment))


def make_filename_safe(identifier: str) -> str:
    """Sanitize identifier to contain only safe alphanumeric and underscore characters for filenames."""
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", identifier).strip("_")
    return sanitized[:40] if sanitized else "source"


class CameraBackend(str, Enum):
    """
    Supported Windows video capture backends.
    
    DirectShow (DSHOW) is strongly preferred on Windows laptops and USB UVC cameras
    because it avoids the Media Foundation (MSMF) initialization freeze (3-8 seconds)
    and enforces precise frame rate negotiation.
    """
    DSHOW = "DSHOW"
    MSMF = "MSMF"
    ANY = "ANY"


@dataclass(frozen=True)
class CameraConfig:
    """
    Immutable configuration for a video capture source.
    """
    device_index: int = 0
    target_width: int = 1280
    target_height: int = 720
    target_fps: int = 30
    backend: CameraBackend = CameraBackend.DSHOW
    buffer_size: int = 1
    auto_reconnect: bool = True
    reconnect_timeout_sec: float = 5.0
    warmup_frames: int = 5
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        """Validate configuration parameters upon instantiation."""
        if not isinstance(self.backend, CameraBackend):
            raise ValueError(
                f"backend must be an instance of CameraBackend enum, got {self.backend!r}"
            )
        if self.device_index < 0:
            raise ValueError(f"device_index must be non-negative, got {self.device_index}")
        if self.target_width <= 0 or self.target_height <= 0:
            raise ValueError(
                f"Resolution dimensions must be positive, got {self.target_width}x{self.target_height}"
            )
        if self.target_fps <= 0 or self.target_fps > 120:
            raise ValueError(
                f"target_fps must be between 1 and 120 FPS, got {self.target_fps}"
            )
        if self.buffer_size != 1:
            raise ValueError(
                f"DriveGuard latest-frame policy requires buffer_size=1, got {self.buffer_size}"
            )

    @property
    def resolution(self) -> Tuple[int, int]:
        """Convenience property returning (width, height)."""
        return self.target_width, self.target_height
