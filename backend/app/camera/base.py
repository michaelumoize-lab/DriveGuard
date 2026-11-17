"""
DriveGuard — Abstract Camera Interface & Frame Packet Definition.

1. What it does:
   Defines the hardware-agnostic contract (`BaseCamera`) and the standardized
   `FramePacket` data structure for all video sources (laptop webcam, phone stream,
   external UVC camera).

2. Why it exists:
   Enforces Camera Independence (Rule 9). The vision pipeline, feature extraction,
   and drowsiness classifier interact strictly with `BaseCamera` and receive `FramePacket`
   instances. Swapping from the HP EliteBook webcam to an Arducam or mobile phone
   requires zero modifications to downstream AI modules.

3. What data it receives:
   Hardware-agnostic method calls (`open`, `read`, `close`).

4. What data it produces:
   Standardized `FramePacket` objects encapsulating raw BGR image data, monotonic
   timestamps, frame index, and source identifier.

5. Modules that depend on it:
   - `backend.app.camera.opencv_camera`
   - `backend.app.camera.phone_camera`
   - `backend.app.camera.camera_manager`
   - `backend.app.vision.face_landmarker`

6. Failure behavior:
   Methods return explicit boolean status flags (`is_opened`, success in `read`) rather
   than crashing, allowing callers to handle camera disconnections gracefully.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FramePacket:
    """
    Standardized video frame container with microsecond-precision metadata.
    
    Attributes:
        frame: OpenCV BGR NumPy array (uint8).
        timestamp_ns: Monotonic capture timestamp in nanoseconds.
        frame_index: Sequential integer counter from the camera source.
        source_id: Unique string identifying the capture device.
    """
    frame: np.ndarray
    timestamp_ns: int
    frame_index: int
    source_id: str

    @property
    def timestamp_sec(self) -> float:
        """Capture timestamp in fractional seconds."""
        return self.timestamp_ns / 1_000_000_000.0

    @property
    def timestamp_ms(self) -> float:
        """Capture timestamp in milliseconds."""
        return self.timestamp_ns / 1_000_000.0

    @property
    def age_ms(self) -> float:
        """
        Elapsed time in milliseconds between optical capture and current moment.
        Used to measure video latency and detect stale buffered frames.
        """
        elapsed_ns = time.monotonic_ns() - self.timestamp_ns
        return max(0.0, elapsed_ns / 1_000_000.0)

    @property
    def resolution(self) -> Tuple[int, int]:
        """Returns (width, height) of the image frame."""
        height, width = self.frame.shape[:2]
        return width, height


class BaseCamera(abc.ABC):
    """
    Abstract Base Class for all DriveGuard camera adapters.
    
    Subclasses must implement hardware initialization, frame retrieval,
    safe release, and live telemetry reporting. Supports Python context manager
    (`with` statement) for guaranteed resource cleanup.
    """

    @abc.abstractmethod
    def open(self) -> bool:
        """
        Initialize the camera hardware/stream and start capture.
        
        Returns:
            True if the camera opened and began acquiring frames successfully; False otherwise.
        """
        pass

    @abc.abstractmethod
    def read(self) -> Tuple[bool, Optional[FramePacket]]:
        """
        Retrieve the latest captured frame packet from the buffer.
        
        Returns:
            Tuple of (success: bool, frame_packet: Optional[FramePacket]).
            If success is False, frame_packet will be None.
        """
        pass

    @abc.abstractmethod
    def close(self) -> None:
        """
        Stop capture threads and safely release all underlying hardware resources.
        Must be idempotent (safe to call multiple times).
        """
        pass

    @abc.abstractmethod
    def is_opened(self) -> bool:
        """
        Check if the camera is currently active and acquiring frames.
        
        Returns:
            True if healthy and open; False otherwise.
        """
        pass

    @abc.abstractmethod
    def get_actual_fps(self) -> float:
        """
        Return the measured hardware ingestion frame rate (FPS).
        """
        pass

    @abc.abstractmethod
    def get_actual_resolution(self) -> Tuple[int, int]:
        """
        Return the actual negotiated video resolution (width, height).
        """
        pass

    @property
    def is_flip_horizontal(self) -> bool:
        """Return True if horizontal mirroring/flipping is currently active."""
        return False

    def __enter__(self) -> BaseCamera:
        """Context manager entry."""
        if not self.open():
            raise RuntimeError(f"Failed to open camera ({self.__class__.__name__}).")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit with guaranteed resource release."""
        self.close()
