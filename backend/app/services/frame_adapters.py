"""
DriveGuard — Input Frame Adapters for Authoritative Monitoring Pipeline.

Normalizes heterogeneous video sources into a single canonical BGR `AdaptedFrame`
consumed by `LocalMonitoringService`:
  - `CameraFrameAdapter`: Local OpenCV / Phone IP camera `FramePacket` streams.
  - `EncodedBytesFrameAdapter`: Encoded JPEG/PNG/WebP bytes from FastAPI WebSocket
    binary frames or HTTP frame uploads.
  - `ArrayFrameAdapter`: Pre-decoded RGB24 (e.g., LiveKit WebRTC `rtc.VideoFrame`)
    or BGR24 NumPy arrays from future video sources or offline replay harnesses.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Optional

import cv2
import numpy as np

from backend.app.camera.base import FramePacket


class FrameDecodeError(ValueError):
    """Raised when an incoming frame payload cannot be decoded into a valid BGR image."""


@dataclass(frozen=True)
class AdaptedFrame:
    """Canonical BGR frame envelope accepted by `LocalMonitoringService`."""

    frame_bgr: np.ndarray
    timestamp_ms: float
    frame_width: int
    frame_height: int
    frame_index: int = 0
    source_id: str = "unknown"
    is_mirrored: bool = True
    packet: Optional[FramePacket] = None


def _validate_bgr_array(frame: np.ndarray, source_id: str) -> np.ndarray:
    if (
        not isinstance(frame, np.ndarray)
        or frame.size == 0
        or frame.ndim != 3
        or frame.shape[2] != 3
    ):
        raise FrameDecodeError(
            f"Invalid frame array from '{source_id}': expected non-empty HxWx3 uint8 array."
        )
    if frame.dtype != np.uint8:
        raise FrameDecodeError(
            f"Invalid frame dtype from '{source_id}': expected uint8, got {frame.dtype}."
        )
    return frame


class CameraFrameAdapter:
    """Adapts local capture `FramePacket` objects into canonical `AdaptedFrame`s."""

    @staticmethod
    def adapt(packet: FramePacket, *, is_mirrored: bool = True) -> AdaptedFrame:
        frame = _validate_bgr_array(packet.frame, packet.source_id)
        h, w = frame.shape[:2]
        return AdaptedFrame(
            frame_bgr=frame,
            timestamp_ms=float(packet.timestamp_ms),
            frame_width=int(w),
            frame_height=int(h),
            frame_index=int(packet.frame_index),
            source_id=packet.source_id,
            is_mirrored=is_mirrored,
            packet=packet,
        )


class EncodedBytesFrameAdapter:
    """Adapts raw encoded image bytes (JPEG/PNG/WebP) from WebSocket or HTTP uploads."""

    @staticmethod
    def adapt(
        frame_bytes: bytes,
        *,
        timestamp_ms: Optional[float] = None,
        frame_index: int = 0,
        source_id: str = "encoded_bytes",
        is_mirrored: bool = True,
        max_bytes: Optional[int] = None,
    ) -> AdaptedFrame:
        if not isinstance(frame_bytes, (bytes, bytearray)) or len(frame_bytes) == 0:
            raise FrameDecodeError("Empty image payload.")
        if max_bytes is not None and len(frame_bytes) > max_bytes:
            raise FrameDecodeError(
                f"Frame payload ({len(frame_bytes)} bytes) exceeds limit ({max_bytes} bytes)."
            )

        nparr = np.frombuffer(frame_bytes, dtype=np.uint8)
        decoded = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0:
            raise FrameDecodeError("Unable to decode image bytes into a valid BGR frame.")

        h, w = decoded.shape[:2]
        ts = float(timestamp_ms) if timestamp_ms is not None else (time.perf_counter() * 1000.0)
        return AdaptedFrame(
            frame_bgr=decoded,
            timestamp_ms=ts,
            frame_width=int(w),
            frame_height=int(h),
            frame_index=int(frame_index),
            source_id=source_id,
            is_mirrored=is_mirrored,
            packet=None,
        )


class ArrayFrameAdapter:
    """Adapts in-memory RGB24 (e.g., WebRTC/LiveKit) or BGR24 NumPy arrays."""

    @staticmethod
    def adapt_bgr(
        frame_bgr: np.ndarray,
        *,
        timestamp_ms: Optional[float] = None,
        frame_index: int = 0,
        source_id: str = "bgr_array",
        is_mirrored: bool = True,
    ) -> AdaptedFrame:
        frame = _validate_bgr_array(frame_bgr, source_id)
        h, w = frame.shape[:2]
        ts = float(timestamp_ms) if timestamp_ms is not None else (time.perf_counter() * 1000.0)
        return AdaptedFrame(
            frame_bgr=frame,
            timestamp_ms=ts,
            frame_width=int(w),
            frame_height=int(h),
            frame_index=int(frame_index),
            source_id=source_id,
            is_mirrored=is_mirrored,
            packet=None,
        )

    @staticmethod
    def adapt_rgb(
        frame_rgb: np.ndarray,
        *,
        timestamp_ms: Optional[float] = None,
        frame_index: int = 0,
        source_id: str = "rgb_array",
        is_mirrored: bool = True,
    ) -> AdaptedFrame:
        frame = _validate_bgr_array(frame_rgb, source_id)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        h, w = frame_bgr.shape[:2]
        ts = float(timestamp_ms) if timestamp_ms is not None else (time.perf_counter() * 1000.0)
        return AdaptedFrame(
            frame_bgr=frame_bgr,
            timestamp_ms=ts,
            frame_width=int(w),
            frame_height=int(h),
            frame_index=int(frame_index),
            source_id=source_id,
            is_mirrored=is_mirrored,
            packet=None,
        )
