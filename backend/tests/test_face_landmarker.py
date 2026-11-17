"""
Unit tests for DriveGuard Face Landmarker & Vision Perception (Phase 4).

Covers:
  - FaceLandmarkerConfig parameter validation
  - Empty/corrupted frame handling (CAMERA_FAULT)
  - Synthetic blank frame inference (no false face detections)
  - Monotonic timestamp enforcement (prevents MediaPipe C++ runtime crashes)
  - Face-loss grace state machine (FACE_LOST_GRACE -> INSUFFICIENT_DATA)
  - Normalized to pixel coordinate mapping
  - Clean lifecycle management and resource release
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.app.camera.base import FramePacket
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkerConfig,
    FaceLandmarkResult,
    FaceLandmarkStatus,
    NormalizedLandmark,
)


def test_face_landmarker_config_valid() -> None:
    """Verifies that valid configurations instantiate successfully."""
    cfg = FaceLandmarkerConfig(
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
        face_loss_grace_timeout_sec=0.5,
        delegate="CPU",
    )
    assert cfg.min_face_detection_confidence == 0.6
    assert cfg.delegate == "CPU"


@pytest.mark.parametrize(
    "invalid_conf",
    [0.0, -0.5, 1.1, 2.0],
)
def test_face_landmarker_config_invalid_confidence(invalid_conf: float) -> None:
    """Verifies that out-of-range confidence thresholds raise ValueError."""
    with pytest.raises(ValueError):
        FaceLandmarkerConfig(min_face_detection_confidence=invalid_conf)


def test_face_landmarker_config_invalid_delegate() -> None:
    """Verifies that unsupported delegates raise ValueError."""
    with pytest.raises(ValueError, match="delegate must be 'CPU' or 'GPU'"):
        FaceLandmarkerConfig(delegate="TPU")


def test_face_landmarker_empty_frame_camera_fault() -> None:
    """Verifies that None or empty image arrays immediately return CAMERA_FAULT."""
    landmarker = DriveGuardFaceLandmarker()
    try:
        empty_frame = np.empty((0, 0, 3), dtype=np.uint8)
        result = landmarker.detect(empty_frame, timestamp_ms=100)

        assert result.has_face is False
        assert result.status == FaceLandmarkStatus.CAMERA_FAULT
        assert len(result.landmarks) == 0
        assert any("Empty or corrupted" in r for r in result.reasons)
    finally:
        landmarker.close()


def test_face_landmarker_blank_image_no_face() -> None:
    """Verifies that featureless blank images do not generate phantom faces."""
    landmarker = DriveGuardFaceLandmarker()
    try:
        # 640x480 pitch-black image
        black_img = np.zeros((480, 640, 3), dtype=np.uint8)
        result = landmarker.detect(black_img, timestamp_ms=100)

        assert result.has_face is False
        assert result.status in (FaceLandmarkStatus.INSUFFICIENT_DATA, FaceLandmarkStatus.FACE_LOST_GRACE)
        assert len(result.landmarks) == 0
        assert result.presence_confidence == 0.0
    finally:
        landmarker.close()


def test_face_landmarker_monotonic_timestamp_enforcement() -> None:
    """
    MediaPipe detect_for_video() raises a hard C++ exception if timestamps are
    duplicate or out-of-order. Verifies that DriveGuard enforces monotonicity.
    """
    landmarker = DriveGuardFaceLandmarker()
    try:
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)

        # Send timestamps: 100, duplicate 100, backwards 90, normal 120
        res1 = landmarker.detect(dummy_img, timestamp_ms=100)
        assert res1.timestamp_ms == 100

        res2 = landmarker.detect(dummy_img, timestamp_ms=100)
        assert res2.timestamp_ms == 101  # Clamped and incremented

        res3 = landmarker.detect(dummy_img, timestamp_ms=90)
        assert res3.timestamp_ms == 102  # Clamped and incremented

        res4 = landmarker.detect(dummy_img, timestamp_ms=120)
        assert res4.timestamp_ms == 120  # Normal forward progression preserved
    finally:
        landmarker.close()


def test_face_landmarker_frame_packet_integration() -> None:
    """Verifies that detect() seamlessly unpacks FramePacket instances."""
    landmarker = DriveGuardFaceLandmarker()
    try:
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        packet = FramePacket(
            frame=dummy_img,
            timestamp_ns=500_000_000,  # 500 ms
            frame_index=1,
            source_id="camera:0",
        )

        result = landmarker.detect(packet)
        assert result.timestamp_ms == 500
        assert isinstance(result, FaceLandmarkResult)
    finally:
        landmarker.close()


def test_face_landmarker_pixel_coordinate_conversion() -> None:
    """Verifies conversion from normalized coordinates [0.0, 1.0] to integer pixel values."""
    lm = NormalizedLandmark(x=0.5, y=0.25, z=-0.05)
    res = FaceLandmarkResult(
        has_face=True,
        landmarks=(lm,),
        presence_confidence=1.0,
        timestamp_ms=100,
        status=FaceLandmarkStatus.TRACKING,
        inference_latency_ms=5.0,
        reasons=("Tracking",),
    )

    px, py = res.get_pixel_coords(0, frame_width=1280, frame_height=720)
    assert px == 640
    assert py == 180

    # Out of bounds index
    assert res.get_pixel_coords(999, 1280, 720) is None


def test_face_landmarker_grace_period_fsm() -> None:
    """
    Verifies that when a face is lost:
    1. During grace period (< 0.3s), status is FACE_LOST_GRACE.
    2. Once grace period expires (>= 0.3s), status transitions to INSUFFICIENT_DATA.
    """
    cfg = FaceLandmarkerConfig(face_loss_grace_timeout_sec=0.3)
    landmarker = DriveGuardFaceLandmarker(cfg)
    try:
        dummy_img = np.zeros((240, 320, 3), dtype=np.uint8)
        mock_empty_result = MagicMock(face_landmarks=[])

        with (
            patch.object(landmarker._landmarker, "detect_for_video", return_value=mock_empty_result),
            patch("backend.app.vision.face_landmarker.time.monotonic", side_effect=[1000.1, 1000.4]),
        ):
            # Simulate a mock detection hit at mocked time 1000.0
            landmarker._last_face_seen_time = 1000.0

            # Immediate next frame without face (elapsed 0.1s < 0.3s)
            res_grace = landmarker.detect(dummy_img, timestamp_ms=100)
            assert res_grace.has_face is False
            assert res_grace.status == FaceLandmarkStatus.FACE_LOST_GRACE
            assert any("momentarily lost" in r for r in res_grace.reasons)

            # Frame after grace timeout (elapsed 0.4s >= 0.3s)
            res_expired = landmarker.detect(dummy_img, timestamp_ms=200)
            assert res_expired.has_face is False
            assert res_expired.status == FaceLandmarkStatus.INSUFFICIENT_DATA
            assert any("exceeds" in r for r in res_expired.reasons)
    finally:
        landmarker.close()


def test_face_landmarker_clean_close() -> None:
    """Verifies that calling close() releases resources and rejects further detect() calls."""
    landmarker = DriveGuardFaceLandmarker()
    assert landmarker._landmarker is not None

    landmarker.close()
    assert landmarker._is_closed is True
    assert landmarker._landmarker is None

    # Repeated close is idempotent
    landmarker.close()

    # Calling detect on closed landmarker raises RuntimeError
    with pytest.raises(RuntimeError, match="closed or not initialized"):
        landmarker.detect(np.zeros((100, 100, 3), dtype=np.uint8))
