"""
Unit tests for DriveGuard Camera Subsystem (Phase 1).

Covers:
  - CameraConfig validation and error handling
  - FramePacket metadata and latency calculations
  - FrameBuffer thread-safety, 1-slot latest-frame drop policy
  - Mocked OpenCVCamera lifecycle (open, threaded frame consumption, close)
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.camera.base import FramePacket
from backend.app.camera.camera_config import CameraBackend, CameraConfig
from backend.app.camera.frame_buffer import FrameBuffer
from backend.app.camera.opencv_camera import OpenCVCamera


def test_camera_config_defaults() -> None:
    """Verifies default settings are configured for laptop webcam with DirectShow."""
    config = CameraConfig()
    assert config.device_index == 0
    assert config.target_width == 1280
    assert config.target_height == 720
    assert config.target_fps == 30
    assert config.backend == CameraBackend.DSHOW
    assert config.buffer_size == 1
    assert config.resolution == (1280, 720)


@pytest.mark.parametrize("invalid_w,invalid_h", [(0, 720), (-1280, 720), (1280, 0), (1280, -720)])
def test_camera_config_invalid_dimensions(invalid_w: int, invalid_h: int) -> None:
    """Verifies ValueError is raised for non-positive resolutions."""
    with pytest.raises(ValueError, match="Resolution dimensions must be positive"):
        CameraConfig(target_width=invalid_w, target_height=invalid_h)


@pytest.mark.parametrize("invalid_fps", [0, -10, 150])
def test_camera_config_invalid_fps(invalid_fps: int) -> None:
    """Verifies ValueError is raised for out-of-range FPS targets."""
    with pytest.raises(ValueError, match="target_fps must be between 1 and 120 FPS"):
        CameraConfig(target_fps=invalid_fps)


def test_frame_packet_metadata() -> None:
    """Verifies FramePacket calculations for timestamp, age, and resolution."""
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    now_ns = time.monotonic_ns()
    
    packet = FramePacket(
        frame=dummy_frame,
        timestamp_ns=now_ns,
        frame_index=42,
        source_id="test:0",
    )

    assert packet.resolution == (640, 480)
    assert packet.frame_index == 42
    assert packet.source_id == "test:0"
    assert packet.age_ms >= 0.0
    assert abs(packet.timestamp_sec - (now_ns / 1_000_000_000.0)) < 1e-6


def test_frame_buffer_latest_frame_drop_policy() -> None:
    """
    Verifies that the 1-slot bounded buffer drops older unconsumed frames
    to strictly prevent video latency buildup.
    """
    buffer = FrameBuffer(capacity=1)
    frame1 = np.ones((10, 10, 3), dtype=np.uint8)
    frame2 = np.ones((10, 10, 3), dtype=np.uint8) * 2

    packet1 = FramePacket(frame=frame1, timestamp_ns=time.monotonic_ns(), frame_index=1, source_id="test")
    packet2 = FramePacket(frame=frame2, timestamp_ns=time.monotonic_ns(), frame_index=2, source_id="test")

    # Push first frame
    buffer.push(packet1)
    stats1 = buffer.get_stats()
    assert stats1["total_pushed"] == 1
    assert stats1["total_dropped"] == 0

    # Push second frame without popping first -> frame1 must be dropped
    buffer.push(packet2)
    stats2 = buffer.get_stats()
    assert stats2["total_pushed"] == 2
    assert stats2["total_dropped"] == 1

    # Pop should yield frame2 (the freshest frame)
    popped = buffer.pop(timeout_sec=0.1)
    assert popped is not None
    assert popped.frame_index == 2
    assert np.array_equal(popped.frame, frame2)

    # Next pop should be None
    assert buffer.pop(timeout_sec=0.01) is None


def test_frame_buffer_clear() -> None:
    """Verifies clearing the buffer resets content."""
    buffer = FrameBuffer(capacity=1)
    packet = FramePacket(
        frame=np.zeros((5, 5, 3), dtype=np.uint8),
        timestamp_ns=time.monotonic_ns(),
        frame_index=1,
        source_id="test",
    )
    buffer.push(packet)
    assert buffer.peek() is not None
    buffer.clear()
    assert buffer.peek() is None


def test_opencv_camera_mocked_lifecycle() -> None:
    """
    Verifies OpenCVCamera opens, starts capture worker, reads frames, and closes cleanly
    using a mock VideoCapture backend.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    
    # Simulate valid frames returned by read()
    dummy_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.read.return_value = (True, dummy_img)
    mock_cap.get.side_effect = lambda prop: {
        3: 1280.0,  # cv2.CAP_PROP_FRAME_WIDTH
        4: 720.0,   # cv2.CAP_PROP_FRAME_HEIGHT
        5: 30.0,    # cv2.CAP_PROP_FPS
    }.get(prop, 0.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        camera = OpenCVCamera(CameraConfig(device_index=0, warmup_frames=1))
        
        opened = camera.open()
        assert opened is True
        assert camera.is_opened() is True

        # Wait briefly for capture worker to push at least one frame
        time.sleep(0.1)
        
        success, packet = camera.read(timeout_sec=0.5)
        assert success is True
        assert packet is not None
        assert packet.resolution == (1280, 720)

        # Clean close
        camera.close()
        assert camera.is_opened() is False
        mock_cap.release.assert_called_once()
