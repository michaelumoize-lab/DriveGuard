"""
DriveGuard — Threaded OpenCV Camera Capture Adapter.

1. What it does:
   Implements hardware video acquisition via OpenCV (`cv2.VideoCapture`) with a
   dedicated background capture thread, Windows DirectShow optimization, and bounded
   buffer management.

2. Why it exists:
   OpenCV's standard `cap.read()` is a blocking I/O operation tied to the camera sensor's
   internal hardware clock. If frame acquisition and AI processing run on a single thread,
   any processing delay creates an escalating backlog of stale frames in the OS video buffer.
   By decoupling capture onto a dedicated thread that pushes exclusively into a 1-slot buffer,
   this adapter guarantees that downstream modules always receive the newest optical frame
   with minimum latency (< 30 ms).

3. What data it receives:
   A validated `CameraConfig` object defining device index, requested resolution, target FPS,
   and Windows video backend (`DSHOW` / `MSMF`).

4. What data it produces:
   Standardized `FramePacket` instances emitted via `read()`, along with live telemetry
   (measured ingestion FPS, negotiated resolution, dropped frame counts).

5. Modules that depend on it:
   - `backend.app.camera.camera_manager`
   - `backend.app.main`
   - `scripts.run_camera_test`

6. Failure behavior:
   Catches OpenCV capture failures and camera disconnects cleanly. If frame grabbing fails,
   it logs a warning, sets connection status to False, and (if configured) initiates an
   automatic reconnection loop without crashing the host application.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.app.camera.base import BaseCamera, FramePacket
from backend.app.camera.camera_config import CameraBackend, CameraConfig
from backend.app.camera.frame_buffer import FrameBuffer

logger = logging.getLogger("driveguard.camera.opencv")


class OpenCVCamera(BaseCamera):
    """
    Thread-safe OpenCV camera adapter supporting DirectShow, Media Foundation,
    and automatic reconnection.
    """

    def __init__(self, config: Optional[CameraConfig] = None) -> None:
        """
        Initialize the camera adapter.
        
        Args:
            config: Camera configuration settings (defaults to HP webcam index 0 via DirectShow).
        """
        self._config = config or CameraConfig()
        self._cap: Optional[cv2.VideoCapture] = None
        self._buffer = FrameBuffer(capacity=self._config.buffer_size)
        
        # Thread synchronization
        self._capture_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Hardware diagnostics
        self._is_connected: bool = False
        self._actual_width: int = 0
        self._actual_height: int = 0
        self._actual_fps: float = 0.0
        self._ingestion_fps: float = 0.0
        self._frame_count: int = 0
        
        # Rolling FPS calculation window
        self._fps_window_start: float = time.monotonic()
        self._fps_window_frames: int = 0

    def _resolve_backend_flag(self) -> int:
        """Map configuration enum to OpenCV backend constant."""
        if self._config.backend == CameraBackend.DSHOW:
            return cv2.CAP_DSHOW
        elif self._config.backend == CameraBackend.MSMF:
            return cv2.CAP_MSMF
        return cv2.CAP_ANY

    def open(self) -> bool:
        """
        Open the camera device, negotiate resolution/FPS, and start the capture thread.
        """
        with self._lock:
            if self._is_connected and self._cap is not None and self._cap.isOpened():
                logger.info("Camera already opened on device index %d.", self._config.device_index)
                return True

            cv_backend = self._resolve_backend_flag()
            logger.info(
                "Opening camera device %d using backend %s (requested: %dx%d @ %d FPS)...",
                self._config.device_index,
                self._config.backend.value,
                self._config.target_width,
                self._config.target_height,
                self._config.target_fps,
            )

            try:
                self._cap = cv2.VideoCapture(self._config.device_index, cv_backend)
            except Exception as exc:
                logger.error("Failed to instantiate VideoCapture: %s", exc)
                return False

            if not self._cap.isOpened():
                logger.error(
                    "Could not open camera device %d with backend %s.",
                    self._config.device_index,
                    self._config.backend.value,
                )
                self._cap.release()
                self._cap = None
                return False

            # Request hardware properties
            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.target_width))
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.target_height))
            self._cap.set(cv2.CAP_PROP_FPS, float(self._config.target_fps))
            
            # Minimize internal OS buffer queue to prevent lag
            self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1.0)

            # Read back actual negotiated properties
            self._actual_width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self._actual_height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            reported_fps = float(self._cap.get(cv2.CAP_PROP_FPS))
            self._actual_fps = reported_fps if reported_fps > 0 else float(self._config.target_fps)

            logger.info(
                "Camera device %d successfully opened. Negotiated: %dx%d (reported FPS: %.1f).",
                self._config.device_index,
                self._actual_width,
                self._actual_height,
                self._actual_fps,
            )

            # Warmup: discard initial dark frames while exposure and white balance adjust
            for _ in range(self._config.warmup_frames):
                self._cap.read()

            # Start background acquisition thread
            self._stop_event.clear()
            self._is_connected = True
            self._fps_window_start = time.monotonic()
            self._fps_window_frames = 0
            
            self._capture_thread = threading.Thread(
                target=self._capture_worker,
                name=f"CameraCapture-{self._config.device_index}",
                daemon=True,
            )
            self._capture_thread.start()
            return True

    def _capture_worker(self) -> None:
        """
        Background loop continuously draining the camera hardware buffer.
        """
        source_id = f"opencv:{self._config.device_index}"
        consecutive_read_failures = 0
        max_consecutive_failures = 30  # Approximately 1 second of failed frames at 30 FPS

        while not self._stop_event.is_set():
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.05)
                continue

            grabbed, frame = self._cap.read()
            capture_time_ns = time.monotonic_ns()

            if not grabbed or frame is None or frame.size == 0:
                consecutive_read_failures += 1
                if consecutive_read_failures >= max_consecutive_failures:
                    logger.warning(
                        "Camera %s stopped delivering frames (%d consecutive read errors).",
                        source_id,
                        consecutive_read_failures,
                    )
                    self._is_connected = False
                    if self._config.auto_reconnect:
                        self._attempt_reconnect()
                        consecutive_read_failures = 0
                    else:
                        break
                time.sleep(0.01)
                continue

            # Reset failure count on valid frame
            consecutive_read_failures = 0
            self._is_connected = True
            self._frame_count += 1
            self._fps_window_frames += 1

            # Update rolling measured ingestion FPS every 0.5s
            now_sec = time.monotonic()
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

        logger.info("Camera capture worker thread stopped for %s.", source_id)

    def _attempt_reconnect(self) -> None:
        """Attempt to reopen a disconnected camera device."""
        logger.info("Attempting camera reconnect on device %d...", self._config.device_index)
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        
        time.sleep(1.0)
        cv_backend = self._resolve_backend_flag()
        try:
            new_cap = cv2.VideoCapture(self._config.device_index, cv_backend)
            if new_cap.isOpened():
                new_cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.target_width))
                new_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.target_height))
                self._cap = new_cap
                self._is_connected = True
                logger.info("Camera reconnected successfully.")
        except Exception as exc:
            logger.error("Camera reconnect failed: %s", exc)

    def read(self, timeout_sec: float = 0.1) -> Tuple[bool, Optional[FramePacket]]:
        """
        Pull the newest frame packet from the buffer.
        """
        if not self._is_connected:
            return False, None
        packet = self._buffer.pop(timeout_sec=timeout_sec)
        if packet is None:
            return False, None
        return True, packet

    def close(self) -> None:
        """
        Safely stop the capture worker and release camera hardware descriptors.
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
            logger.info("Camera device %d released cleanly.", self._config.device_index)

    def is_opened(self) -> bool:
        """Returns True if the camera device is currently acquiring frames."""
        return self._is_connected and self._cap is not None and self._cap.isOpened()

    def get_actual_fps(self) -> float:
        """Return the actual measured hardware ingestion FPS."""
        return self._ingestion_fps

    def get_actual_resolution(self) -> Tuple[int, int]:
        """Return the negotiated video frame resolution (width, height)."""
        return self._actual_width, self._actual_height

    def get_buffer_stats(self) -> dict:
        """Return buffer performance statistics."""
        return self._buffer.get_stats()
