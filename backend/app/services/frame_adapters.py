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


class AdaptedFrame:
    """
    Canonical frame envelope accepted by `LocalMonitoringService`.

    Supports native `"BGR"` (OpenCV / decoded JPEG) and zero-copy `"RGB"`
    (LiveKit WebRTC RGB24) passthrough via `.frame` and `.color_format`,
    eliminating redundant `RGB -> BGR -> RGB` color conversions while keeping
    `.frame_bgr` backward-compatible via lazy conversion when requested.
    """

    __slots__ = (
        "_frame",
        "_cached_bgr",
        "timestamp_ms",
        "frame_width",
        "frame_height",
        "frame_index",
        "source_id",
        "is_mirrored",
        "packet",
        "color_format",
    )

    def __init__(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        timestamp_ms: float = 0.0,
        frame_width: int = 0,
        frame_height: int = 0,
        frame_index: int = 0,
        source_id: str = "unknown",
        is_mirrored: bool = True,
        packet: Optional[FramePacket] = None,
        *,
        frame: Optional[np.ndarray] = None,
        color_format: str = "BGR",
    ) -> None:
        raw = frame if frame is not None else frame_bgr
        if raw is None:
            raise FrameDecodeError("AdaptedFrame requires either 'frame' or 'frame_bgr'.")
        fmt = color_format.upper()
        if fmt not in ("BGR", "RGB"):
            raise ValueError(f"Unsupported color_format '{color_format}'; expected 'BGR' or 'RGB'.")
        object.__setattr__(self, "_frame", raw)
        object.__setattr__(self, "_cached_bgr", raw if fmt == "BGR" else None)
        object.__setattr__(self, "timestamp_ms", float(timestamp_ms))
        object.__setattr__(self, "frame_width", int(frame_width))
        object.__setattr__(self, "frame_height", int(frame_height))
        object.__setattr__(self, "frame_index", int(frame_index))
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "is_mirrored", bool(is_mirrored))
        object.__setattr__(self, "packet", packet)
        object.__setattr__(self, "color_format", fmt)

    def __setattr__(self, name: str, value) -> None:
        raise AttributeError("AdaptedFrame is immutable.")

    @property
    def frame(self) -> np.ndarray:
        """Native frame array in `self.color_format` ('BGR' or 'RGB') with zero conversion."""
        return self._frame

    @property
    def frame_bgr(self) -> np.ndarray:
        """Backward-compatible BGR accessor (zero-cost for BGR; lazy convert for RGB)."""
        if self._cached_bgr is None:
            converted = cv2.cvtColor(self._frame, cv2.COLOR_RGB2BGR)
            object.__setattr__(self, "_cached_bgr", converted)
        return self._cached_bgr


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
        h, w = frame.shape[:2]
        ts = float(timestamp_ms) if timestamp_ms is not None else (time.perf_counter() * 1000.0)
        return AdaptedFrame(
            frame=frame,
            color_format="RGB",
            timestamp_ms=ts,
            frame_width=int(w),
            frame_height=int(h),
            frame_index=int(frame_index),
            source_id=source_id,
            is_mirrored=is_mirrored,
            packet=None,
        )
