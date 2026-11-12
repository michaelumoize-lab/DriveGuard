"""
DriveGuard — Thread-Safe Bounded Latest-Frame Buffer.

1. What it does:
   Provides a bounded, thread-safe queue designed specifically for real-time video
   streaming. Stores the latest frame packet and automatically drops older unconsumed
   frames to prevent video latency accumulation.

2. Why it exists:
   In real-time driver monitoring, an old frame from 500 ms ago is dangerous to process.
   Traditional unbounded FIFO queues cause "buffer bloat", where detection results lag
   behind the driver's actual physical state. This buffer guarantees that consumers
   (the vision pipeline) always receive the most recent optical frame.

3. What data it receives:
   `FramePacket` instances pushed by the camera acquisition thread.

4. What data it produces:
   The freshest available `FramePacket` popped by the consumer thread.

5. Modules that depend on it:
   - `backend.app.camera.opencv_camera`
   - `backend.app.camera.phone_camera`
   - `backend.app.camera.camera_manager`

6. Failure behavior:
   Thread-safe under concurrent access. If the buffer is empty when `pop` is called,
   it returns `None` (or blocks up to an optional timeout) rather than raising an unhandled exception.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from backend.app.camera.base import FramePacket


class FrameBuffer:
    """
    Thread-safe 1-slot latest-frame buffer with dropped-frame tracking.
    """

    def __init__(self, capacity: int = 1) -> None:
        """
        Initialize the bounded buffer.
        
        Args:
            capacity: Maximum number of frames to retain (default: 1 for latest-frame strategy).
        """
        if capacity < 1:
            raise ValueError(f"capacity must be at least 1, got {capacity}")
        self._capacity = capacity
        self._lock = threading.Lock()
        self._new_frame_event = threading.Event()
        self._latest_packet: Optional[FramePacket] = None
        self._total_pushed: int = 0
        self._total_dropped: int = 0
        self._total_consumed: int = 0

    def push(self, packet: FramePacket) -> None:
        """
        Push a new frame packet into the buffer.
        If an existing unread frame is present, it is dropped to guarantee zero latency.
        """
        with self._lock:
            if self._latest_packet is not None:
                self._total_dropped += 1
            self._latest_packet = packet
            self._total_pushed += 1
            self._new_frame_event.set()

    def pop(self, timeout_sec: Optional[float] = None) -> Optional[FramePacket]:
        """
        Pop the latest frame packet from the buffer.
        
        Args:
            timeout_sec: Maximum seconds to wait for a frame if currently empty.
            
        Returns:
            The latest FramePacket, or None if timeout expired or buffer empty.
        """
        # Wait for new frame event if requested
        if timeout_sec is not None:
            self._new_frame_event.wait(timeout=timeout_sec)

        with self._lock:
            if self._latest_packet is None:
                return None
            packet = self._latest_packet
            self._latest_packet = None
            self._total_consumed += 1
            self._new_frame_event.clear()
            return packet

    def peek(self) -> Optional[FramePacket]:
        """
        Inspect the latest frame without removing it.
        """
        with self._lock:
            return self._latest_packet

    def clear(self) -> None:
        """
        Clear all frames and reset the new-frame signal.
        """
        with self._lock:
            self._latest_packet = None
            self._new_frame_event.clear()

    def get_stats(self) -> Dict[str, Any]:
        """
        Return operational telemetry for diagnostic reporting.
        """
        with self._lock:
            return {
                "capacity": self._capacity,
                "total_pushed": self._total_pushed,
                "total_consumed": self._total_consumed,
                "total_dropped": self._total_dropped,
                "is_empty": self._latest_packet is None,
            }
