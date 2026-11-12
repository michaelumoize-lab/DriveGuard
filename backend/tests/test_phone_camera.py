"""
Unit tests for DriveGuard Phone Camera Adapter & Camera Manager (Phase 2).

Covers:
  - PhoneCameraConfig URL scheme validation (HTTP, HTTPS, RTSP)
  - Timeout and parameter boundary checks
  - CameraManager factory dispatch (OpenCVCamera vs PhoneCamera)
  - Mocked PhoneCamera network stream lifecycle and watchdog timeout
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.camera.camera_manager import create_camera
from backend.app.camera.opencv_camera import OpenCVCamera
from backend.app.camera.phone_camera import PhoneCamera, PhoneCameraConfig


def test_phone_camera_config_valid_schemes() -> None:
    """Verifies that HTTP, HTTPS, and RTSP stream URLs are accepted."""
    cfg_http = PhoneCameraConfig("http://192.168.1.50:8080/video")
    assert cfg_http.stream_url == "http://192.168.1.50:8080/video"
    assert cfg_http.target_fps == 30

    cfg_rtsp = PhoneCameraConfig("rtsp://192.168.1.50:554/live")
    assert cfg_rtsp.stream_url == "rtsp://192.168.1.50:554/live"

    cfg_https = PhoneCameraConfig("https://192.168.1.50:8443/feed")
    assert cfg_https.stream_url == "https://192.168.1.50:8443/feed"


@pytest.mark.parametrize("invalid_url", ["", "ftp://192.168.1.50/video", "file:///tmp/video.mp4", "not_a_url"])
def test_phone_camera_config_invalid_url(invalid_url: str) -> None:
    """Verifies that invalid or unsupported stream URLs raise ValueError."""
    with pytest.raises(ValueError):
        PhoneCameraConfig(invalid_url)


@pytest.mark.parametrize("invalid_timeout", [0.0, 0.2, -1.0])
def test_phone_camera_config_invalid_timeout(invalid_timeout: float) -> None:
    """Verifies that non-viable watchdog timeouts raise ValueError."""
    with pytest.raises(ValueError, match="timeout_sec must be at least 0.5"):
        PhoneCameraConfig("http://192.168.1.50:8080/video", timeout_sec=invalid_timeout)


@pytest.mark.parametrize("invalid_interval", [-0.1, -1.0])
def test_phone_camera_config_invalid_reconnect_interval(invalid_interval: float) -> None:
    """Verifies that negative reconnect intervals raise ValueError."""
    with pytest.raises(ValueError, match="reconnect_interval_sec must be non-negative"):
        PhoneCameraConfig("http://192.168.1.50:8080/video", reconnect_interval_sec=invalid_interval)


@pytest.mark.parametrize("invalid_buffer", [0, 2, 5])
def test_phone_camera_config_invalid_buffer_size(invalid_buffer: int) -> None:
    """Verifies that buffer_size != 1 is rejected per latest-frame policy."""
    with pytest.raises(ValueError, match="latest-frame policy requires buffer_size=1"):
        PhoneCameraConfig("http://192.168.1.50:8080/video", buffer_size=invalid_buffer)


def test_camera_manager_factory_dispatch() -> None:
    """
    Verifies that create_camera automatically selects OpenCVCamera for device indices
    and PhoneCamera for stream URLs.
    """
    # Local webcam dispatch
    cam_webcam = create_camera(0, target_width=1280, target_height=720)
    assert isinstance(cam_webcam, OpenCVCamera)

    cam_webcam_str = create_camera("1")
    assert isinstance(cam_webcam_str, OpenCVCamera)

    # Phone stream dispatch
    cam_phone = create_camera("http://192.168.1.25:8080/video")
    assert isinstance(cam_phone, PhoneCamera)

    cam_rtsp = create_camera("rtsp://10.0.0.5:554/stream")
    assert isinstance(cam_rtsp, PhoneCamera)

    # Invalid source
    with pytest.raises(ValueError, match="Invalid camera source"):
        create_camera("unknown_source_string")


def test_phone_camera_mocked_lifecycle() -> None:
    """
    Verifies PhoneCamera connects, grabs/retrieves frames in background worker,
    redacts credentials in source_id, and closes cleanly using a mock VideoCapture backend.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.grab.return_value = True

    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    mock_cap.retrieve.return_value = (True, dummy_frame)
    mock_cap.get.side_effect = lambda prop: {
        3: 1280.0,  # cv2.CAP_PROP_FRAME_WIDTH
        4: 720.0,   # cv2.CAP_PROP_FRAME_HEIGHT
        5: 30.0,    # cv2.CAP_PROP_FPS
    }.get(prop, 0.0)

    with patch("cv2.VideoCapture", return_value=mock_cap) as mock_vc_ctor:
        # Pass stream URL with sensitive credentials
        raw_url = "http://admin:secretPass123@192.168.1.50:8080/video?token=myToken"
        config = PhoneCameraConfig(raw_url)
        camera = PhoneCamera(config)

        opened = camera.open()
        assert opened is True
        assert camera.is_opened() is True

        # Verify underlying VideoCapture was opened with the real functional URL
        assert mock_vc_ctor.call_args[0][0] == raw_url

        # Allow background thread to process a frame
        time.sleep(0.1)

        success, packet = camera.read(timeout_sec=0.5)
        assert success is True
        assert packet is not None
        assert packet.resolution == (1280, 720)
        # Verify credentials and sensitive tokens are strictly redacted in source_id
        assert "secretPass123" not in packet.source_id
        assert "myToken" not in packet.source_id
        assert packet.source_id == "phone:http://admin:***@192.168.1.50:8080/video?token=***"

        # Clean close
        camera.close()
        assert camera.is_opened() is False
        mock_cap.release.assert_called_once()


def test_phone_camera_watchdog_timeout_on_stream_loss() -> None:
    """
    Verifies that when a network stream drops (grab returns False),
    the watchdog detects timeout and marks connection inactive.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    # Simulate network packet drop: grab fails
    mock_cap.grab.return_value = False

    with patch("cv2.VideoCapture", return_value=mock_cap):
        # Short timeout for rapid test
        config = PhoneCameraConfig("http://192.168.1.50:8080/video", timeout_sec=0.6, auto_reconnect=False)
        camera = PhoneCamera(config)
        camera.open()

        # Initially opened
        assert camera.is_opened() is True

        # Wait past watchdog timeout (0.6s)
        time.sleep(0.8)

        # Watchdog should flag disconnected
        assert camera.is_opened() is False

        camera.close()


def test_phone_camera_watchdog_timeout_on_repeated_retrieve_failure() -> None:
    """
    Verifies that when grab succeeds but retrieve repeatedly fails (e.g. corrupt decode),
    the watchdog detects timeout and marks connection inactive.
    """
    mock_cap = MagicMock()
    mock_cap.isOpened.return_value = True
    mock_cap.grab.return_value = True
    # Simulate decode failure: retrieve returns False / empty
    mock_cap.retrieve.return_value = (False, None)
    mock_cap.get.side_effect = lambda prop: {3: 1280.0, 4: 720.0, 5: 30.0}.get(prop, 0.0)

    with patch("cv2.VideoCapture", return_value=mock_cap):
        config = PhoneCameraConfig("http://192.168.1.50:8080/video", timeout_sec=0.6, auto_reconnect=False)
        camera = PhoneCamera(config)
        camera.open()

        assert camera.is_opened() is True

        # Wait past watchdog timeout (0.6s)
        time.sleep(0.8)

        # Watchdog must flag disconnected even though grab() succeeded
        assert camera.is_opened() is False

        camera.close()


def test_phone_camera_open_stops_prior_worker() -> None:
    """Verifies that calling open() coordinates with and stops any existing worker thread."""
    mock_cap1 = MagicMock()
    mock_cap1.isOpened.return_value = True
    mock_cap1.grab.return_value = True
    mock_cap1.retrieve.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_cap1.get.side_effect = lambda prop: {3: 1280.0, 4: 720.0, 5: 30.0}.get(prop, 0.0)

    mock_cap2 = MagicMock()
    mock_cap2.isOpened.return_value = True
    mock_cap2.grab.return_value = True
    mock_cap2.retrieve.return_value = (True, np.zeros((720, 1280, 3), dtype=np.uint8))
    mock_cap2.get.side_effect = lambda prop: {3: 1280.0, 4: 720.0, 5: 30.0}.get(prop, 0.0)

    with patch("cv2.VideoCapture", side_effect=[mock_cap1, mock_cap2]):
        config = PhoneCameraConfig("http://192.168.1.50:8080/video")
        camera = PhoneCamera(config)
        assert camera.open() is True
        old_thread = camera._capture_thread
        assert old_thread is not None and old_thread.is_alive()

        # Simulate connection drop so subsequent open() proceeds with reopening
        camera._is_connected = False
        assert camera.open() is True

        # Prior thread must be cleanly joined
        assert not old_thread.is_alive()
        mock_cap1.release.assert_called()

        camera.close()
