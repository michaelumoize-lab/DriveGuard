"""
DriveGuard — Camera Subsystem Package.

Exposes hardware acquisition abstractions, configurations, and concrete capture adapters.
"""

from backend.app.camera.base import BaseCamera, FramePacket
from backend.app.camera.camera_config import CameraBackend, CameraConfig
from backend.app.camera.frame_buffer import FrameBuffer
from backend.app.camera.opencv_camera import OpenCVCamera

__all__ = [
    "BaseCamera",
    "FramePacket",
    "CameraBackend",
    "CameraConfig",
    "FrameBuffer",
    "OpenCVCamera",
]
