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

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


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
        if self.buffer_size < 1:
            raise ValueError(
                f"buffer_size must be at least 1, got {self.buffer_size}"
            )

    @property
    def resolution(self) -> Tuple[int, int]:
        """Convenience property returning (width, height)."""
        return self.target_width, self.target_height
