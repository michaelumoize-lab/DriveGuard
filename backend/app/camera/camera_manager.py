"""
DriveGuard — Camera Manager & Source Factory.

1. What it does:
   Provides factory instantiation (`create_camera`) and discovery utilities
   for video capture devices, unifying laptop webcams, USB cameras, and phone streams.

2. Why it exists:
   Shields downstream application code from hardware-specific adapter details.
   A caller simply passes an identifier (e.g., `0` for laptop webcam, or
   `"http://192.168.1.50:8080/video"` for a phone stream) and receives an
   instantiated `BaseCamera` compliant with the standard contract.

3. What data it receives:
   A device identifier (integer index or stream URL string) along with optional
   configuration overrides.

4. What data it produces:
   An instantiated `BaseCamera` implementation (`OpenCVCamera` or `PhoneCamera`).

5. Modules that depend on it:
   - `backend.app.main`
   - `scripts.run_camera_test`
   - Downstream vision controllers

6. Failure behavior:
   Raises `ValueError` for unsupported stream protocols or invalid device identifiers
   prior to allocating hardware resources.
"""

from __future__ import annotations

import logging
from typing import Any, List, Union

import cv2

from backend.app.camera.base import BaseCamera
from backend.app.camera.camera_config import (
    CameraBackend,
    CameraConfig,
    sanitize_source_identifier,
)
from backend.app.camera.opencv_camera import OpenCVCamera
from backend.app.camera.phone_camera import PhoneCamera, PhoneCameraConfig

logger = logging.getLogger("driveguard.camera.manager")


def create_camera(
    source: Union[int, str] = 0,
    target_width: int = 1280,
    target_height: int = 720,
    target_fps: int = 30,
    backend: str = "DSHOW",
    **kwargs: Any,
) -> BaseCamera:
    """
    Factory function to instantiate the appropriate camera adapter.
    
    Args:
        source: Integer device index (0 for laptop webcam) or URL string for phone stream.
        target_width: Desired resolution width (default: 1280).
        target_height: Desired resolution height (default: 720).
        target_fps: Desired frame rate (default: 30).
        backend: Windows capture backend ('DSHOW', 'MSMF', 'ANY').
        **kwargs: Additional adapter-specific options (e.g. timeout_sec for PhoneCamera).
        
    Returns:
        An instantiated BaseCamera subclass ready for .open().
    """
    # Check if source is a URL (Phone Camera Stream)
    if isinstance(source, str) and (
        source.lower().startswith("http://")
        or source.lower().startswith("https://")
        or source.lower().startswith("rtsp://")
    ):
        safe_url = sanitize_source_identifier(source)
        logger.info("Factory selecting PhoneCamera adapter for URL: %s", safe_url)
        phone_config = PhoneCameraConfig(
            stream_url=source,
            target_fps=target_fps,
            timeout_sec=kwargs.get("timeout_sec", 3.0),
            auto_reconnect=kwargs.get("auto_reconnect", True),
            reconnect_interval_sec=kwargs.get("reconnect_interval_sec", 2.0),
            buffer_size=kwargs.get("buffer_size", 1),
            flip_horizontal=kwargs.get("flip_horizontal", True),
            flip_vertical=kwargs.get("flip_vertical", False),
        )
        return PhoneCamera(phone_config)

    # Check if source is a local webcam device index
    try:
        device_index = int(source)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid camera source '{source}'. Must be an integer device index (e.g. 0) "
            f"or a valid network stream URL (http://... or rtsp://...)"
        ) from exc

    backend_enum = CameraBackend(backend.upper())
    logger.info(
        "Factory selecting OpenCVCamera adapter for device index %d (backend: %s)",
        device_index,
        backend_enum.value,
    )
    config = CameraConfig(
        device_index=device_index,
        target_width=target_width,
        target_height=target_height,
        target_fps=target_fps,
        backend=backend_enum,
        buffer_size=kwargs.get("buffer_size", 1),
        auto_reconnect=kwargs.get("auto_reconnect", True),
        flip_horizontal=kwargs.get("flip_horizontal", True),
        flip_vertical=kwargs.get("flip_vertical", False),
    )
    return OpenCVCamera(config)


def discover_local_devices(max_devices_to_probe: int = 3) -> List[int]:
    """
    Probes available DirectShow video capture indices on the local Windows machine.
    
    Args:
        max_devices_to_probe: Maximum device index numbers to test (default: 3).
        
    Returns:
        List of integer device indices that opened successfully.
    """
    available_indices = []
    logger.info("Probing local video capture devices (0 to %d)...", max_devices_to_probe - 1)

    for index in range(max_devices_to_probe):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            available_indices.append(index)
            cap.release()

    logger.info("Discovered available camera devices: %s", available_indices)
    return available_indices
