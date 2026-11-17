"""
DriveGuard — Camera Subsystem Package.

Exposes hardware acquisition abstractions, configurations, concrete capture adapters,
and the unified camera factory.
"""

from backend.app.camera.base import BaseCamera, FramePacket
from backend.app.camera.camera_config import CameraBackend, CameraConfig
from backend.app.camera.camera_manager import create_camera, discover_local_devices
from backend.app.camera.frame_buffer import FrameBuffer
from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityAnalyzer,
    FrameQualityResult,
    LightingCondition,
    QualityStatus,
)
from backend.app.camera.opencv_camera import OpenCVCamera
from backend.app.camera.phone_camera import PhoneCamera, PhoneCameraConfig

__all__ = [
    "BaseCamera",
    "FramePacket",
    "CameraBackend",
    "CameraConfig",
    "FrameBuffer",
    "BlurCondition",
    "FrameQualityAnalyzer",
    "FrameQualityResult",
    "LightingCondition",
    "QualityStatus",
    "OpenCVCamera",
    "PhoneCamera",
    "PhoneCameraConfig",
    "create_camera",
    "discover_local_devices",
]
