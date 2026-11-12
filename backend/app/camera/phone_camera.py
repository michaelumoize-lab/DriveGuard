"""
DriveGuard — Phone Camera Network Stream Capture Adapter.

1. What it does:
   Acquires live video frames from a mobile phone (Android or iPhone) broadcasting an
   HTTP MJPEG or RTSP video stream across a local Wi-Fi or USB tethered network.

2. Why it exists:
   Enforces Camera Independence (Rule 9). Allows a driver's mobile phone to serve as an
   alternative high-definition camera without modifying downstream vision, feature
   extraction, or drowsiness decision modules. Operates entirely over local network sockets
   without routing video data through external cloud servers.

3. What data it receives:
   A validated `PhoneCameraConfig` containing the local network stream URL
   (e.g., `http://192.168.1.50:8080/video` or `rtsp://...`), connection timeouts, and reconnect parameters.

4. What data it produces:
   Standardized `FramePacket` instances via `read()`, identical in contract to `OpenCVCamera`.

5. Modules that depend on it:
   - `backend.app.camera.camera_manager`
   - `scripts.run_camera_test`
   - Downstream vision modules (Phase 4+)

6. Failure behavior:
   Implements a dedicated watchdog timer that monitors frame delivery. If the phone screen locks,
   Wi-Fi drops, or no frame arrives within `timeout_sec`, it marks the camera as stalled,
   logs a diagnostic warning, and initiates an automatic reconnection state machine.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

import cv2
import numpy as np

from backend.app.camera.base import BaseCamera, FramePacket
from backend.app.camera.frame_buffer import FrameBuffer

logger = logging.getLogger("driveguard.camera.phone")


@dataclass(frozen=True)
class PhoneCameraConfig:
    """
    Configuration settings for a phone network video stream.
    """
    stream_url: str
    target_fps: int = 30
    timeout_sec: float = 3.0
    auto_reconnect: bool = True
    reconnect_interval_sec: float = 2.0
    buffer_size: int = 1
    flip_horizontal: bool = False
    flip_vertical: bool = False

    def __post_init__(self) -> None:
        """Validate stream URL and parameters."""
        if not self.stream_url or not isinstance(self.stream_url, str):
            raise ValueError("stream_url must be a non-empty string.")

        parsed = urlparse(self.stream_url)
        if parsed.scheme.lower() not in ("http", "https", "rtsp"):
            raise ValueError(
                f"Unsupported stream scheme '{parsed.scheme}'. Must be 'http', 'https', or 'rtsp'."
            )
        if not parsed.netloc:
            raise ValueError(f"Invalid stream URL format: '{self.stream_url}'. Host and port are required.")

        if self.target_fps <= 0 or self.target_fps > 120:
            raise ValueError(f"target_fps must be between 1 and 120, got {self.target_fps}")
        if self.timeout_sec <= 0.5:
            raise ValueError(f"timeout_sec must be at least 0.5 seconds, got {self.timeout_sec}")
        if self.buffer_size < 1:
            raise ValueError(f"buffer_size must be at least 1, got {self.buffer_size}")


class PhoneCamera(BaseCamera):
    """
    Adapter for phone IP/RTSP/MJPEG streams using a low-latency threaded consumer.
    """

    def __init__(self, config: PhoneCameraConfig) -> None:
        """
        Initialize the PhoneCamera adapter.
        
        Args:
            config: Validated PhoneCameraConfig instance.
        """
        self._config = config
        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer = FrameBuffer(capacity=self._config.buffer_size)

        # Thread synchronization
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Telemetry & Diagnostics
        self._is_connected: bool = False
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = float(self._config.target_fps)
        self._ingestion_fps: float = 0.0
        self._frame_count: int = 0
        self._flip_horizontal: bool = self._config.flip_horizontal
        self._flip_vertical: bool = self._config.flip_vertical

        # Rolling FPS window
        self._fps_window_start: float = time.monotonic()
        self._fps_window_frames: int = 0
        self._last_frame_received_time: float = 0.0

    def open(self) -> bool:
        """
        Connect to the phone video stream and start the background frame pump thread.
        """
        with self._lock:
            if self._is_connected and self._cap is not None and self._cap.isOpened():
                logger.info("Phone camera stream already connected to %s", self._config.stream_url)
                return True

            logger.info("Connecting to phone video stream at %s...", self._config.stream_url)

            # Enforce TCP transport for RTSP to prevent packet-loss stalls on Wi-Fi
            if self._config.stream_url.lower().startswith("rtsp"):
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

            try:
                self._cap = cv2.VideoCapture(self._config.stream_url, cv2.CAP_FFMPEG)
            except Exception as exc:
                logger.error("Failed to initialize VideoCapture for phone stream: %s", exc)
                return False

            if not self._cap.isOpened():
                logger.error("Could not open phone stream at %s. Verify phone IP and port.", self._config.stream_url)
                self._cap.release()
                self._cap = None
                return False

            # Set buffer size to minimum to discard network stream backlog
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

            # Query negotiated stream dimensions
            self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            if reported_fps > 0:
                self._actual_fps = reported_fps

            logger.info(
                "Phone stream connected: %dx%d @ reported %.1f FPS.",
                self._actual_width,
                self._actual_height,
                self._actual_fps,
            )

            # Start acquisition worker
            self._stop_event.clear()
            self._is_connected = True
            now = time.monotonic()
            self._fps_window_start = now
            self._last_frame_received_time = now
            self._fps_window_frames = 0

            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                name=f"PhoneCaptureWorker-{urlparse(self._config.stream_url).netloc}",
                daemon=True,
            )
            self._capture_thread.start()
            return True

    def _capture_worker(self) -> None:
        """
        Background loop continuously draining the network stream buffer.
        """
        source_id = f"phone:{self._config.stream_url}"

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.05)
                continue

            # grab() flushes the frame from the network buffer without heavy decoding
            grabbed = self._cap.grab()
            capture_time_ns = time.monotonic_ns()

            if not grabbed:
                now_sec = time.monotonic()
                # Check watchdog timeout for dropped connection
                if (now_sec - self._last_frame_received_time) > self._config.timeout_sec:
                    logger.warning(
                        "Phone stream timed out (> %.1fs without frame). Marking disconnected.",
                        self._config.timeout_sec,
                    )
                    self._is_connected = False
                    if self._config.auto_reconnect:
                        self._attempt_reconnect()
                    else:
                        break
                time.sleep(0.01)
                continue

            # Decode the grabbed frame
            ret, frame = self._cap.retrieve()
            if not ret or frame is None or frame.size == 0:
                time.sleep(0.01)
                continue

            # Apply orientation/mirror transforms if requested
            if self._flip_horizontal and self._flip_vertical:
                frame = cv2.flip(frame, -1)
            elif self._flip_horizontal:
                frame = cv2.flip(frame, 1)
            elif self._flip_vertical:
                frame = cv2.flip(frame, 0)

            # Update dimensions dynamically if phone changed orientation
            h, w = frame.shape[:2]
            self._actual_width = w
            self._actual_height = h

            now_sec = time.monotonic()
            self._last_frame_received_time = now_sec
            self._is_connected = True
            self._frame_count += 1
            self._fps_window_frames += 1

            # Update rolling measured ingestion FPS every 0.5s
            elapsed_sec = now_sec - self._fps_window_start
            if elapsed_sec >= 0.5:
                self._ingestion_fps = self._fps_window_frames / elapsed_sec
                self._fps_window_start = now_sec
                self._fps_window_frames = 0

            # Wrap in FramePacket and push to bounded buffer
            packet = FramePacket(
                frame=frame,
                timestamp_ns=capture_time_ns,
                frame_index=self._frame_count,
                source_id=source_id,
            )
            self._buffer.push(packet)

        logger.info("Phone camera worker thread stopped for %s.", source_id)

    def _attempt_reconnect(self) -> None:
        """Attempt to reconnect to the phone stream URL after network interruption."""
        logger.info("Attempting phone stream reconnect to %s...", self._config.stream_url)
        if self._cap is not None:
            self._cap.release()
            self._cap = None

        time.sleep(self._config.reconnect_interval_sec)
        try:
            new_cap = cv2.VideoCapture(self._config.stream_url, cv2.CAP_FFMPEG)
            if new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)
                self._cap = new_cap
                self._is_connected = True
                self._last_frame_received_time = time.monotonic()
                logger.info("Phone stream reconnected successfully.")
        except Exception as exc:
            logger.error("Phone stream reconnect attempt failed: %s", exc)

    def read(self, timeout_sec: float = 0.1) -> Tuple[bool, Optional[FramePacket]]:
        """
        Pull the latest frame packet from the buffer.
        """
        if not self._is_connected:
            return False, None
        packet = self._buffer.pop(timeout_sec=timeout_sec)
        if packet is None:
            return False, None
        return True, packet

    def close(self) -> None:
        """
        Safely stop worker thread and release network capture descriptors.
        """
        with self._lock:
            self._stop_event.set()
            if self._capture_thread is not None and self._capture_thread.is_alive():
                self._capture_thread.join(timeout=2.0)
                self._capture_thread = None

            if self._cap is not None:
                self._cap.release()
                self._cap = None

            self._is_connected = False
            self._buffer.clear()
            logger.info("Phone stream connection to %s closed cleanly.", self._config.stream_url)

    def is_opened(self) -> bool:
        """Returns True if the stream connection is currently active."""
        return self._is_connected and self._cap is not None and self._cap.isOpened()

    def get_actual_fps(self) -> float:
        """Return the measured hardware ingestion FPS from the phone stream."""
        return self._ingestion_fps

    def get_actual_resolution(self) -> Tuple[int, int]:
        """Return the actual video frame resolution (width, height)."""
        return self._actual_width, self._actual_height

    def toggle_flip_horizontal(self) -> bool:
        """Toggle horizontal mirroring dynamically at runtime."""
        self._flip_horizontal = not self._flip_horizontal
        logger.info("Phone camera horizontal mirroring toggled to: %s", self._flip_horizontal)
        return self._flip_horizontal

    def get_buffer_stats(self) -> dict:
        """Return buffer performance statistics."""
        return self._buffer.get_stats()
