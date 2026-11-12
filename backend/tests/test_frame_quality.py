"""
Unit tests for DriveGuard Frame Quality Diagnostics (Phase 3).

Covers:
  - Threshold validation and error handling
  - Extreme dark frame detection (underexposure / pitch-black tunnel)
  - Extreme bright frame detection (overexposure / sunlight glare)
  - Defocus and motion blur detection via Laplacian variance
  - Sharp, well-lit optimal frame classification
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from backend.app.camera.frame_quality import (
    BlurCondition,
    FrameQualityAnalyzer,
    LightingCondition,
    QualityStatus,
)


def test_analyzer_threshold_validation() -> None:
    """Verifies that invalid thresholds raise ValueError."""
    with pytest.raises(ValueError, match="strictly less"):
        FrameQualityAnalyzer(underexposed_threshold=200, overexposed_threshold=100)

    with pytest.raises(ValueError, match="blur_threshold must be positive"):
        FrameQualityAnalyzer(blur_threshold=-5.0)


def test_empty_or_none_frame() -> None:
    """Verifies safe handling of None and zero-sized images."""
    analyzer = FrameQualityAnalyzer()
    res_none = analyzer.analyze(None)
    assert res_none.status == QualityStatus.SEVERELY_DEGRADED
    assert res_none.is_viable_for_inference is False

    res_empty = analyzer.analyze(np.array([], dtype=np.uint8))
    assert res_empty.status == QualityStatus.SEVERELY_DEGRADED
    assert res_empty.is_viable_for_inference is False


def test_underexposed_dark_frame() -> None:
    """Verifies that low-luminance images (e.g., night cabin without lights) are flagged."""
    analyzer = FrameQualityAnalyzer(underexposed_threshold=35.0)
    # Mean intensity = 20 (dim/underexposed)
    dark_frame = np.full((100, 100, 3), 20, dtype=np.uint8)

    result = analyzer.analyze(dark_frame)
    assert result.lighting == LightingCondition.UNDEREXPOSED
    assert result.mean_luminance == pytest.approx(20.0, abs=0.5)


def test_pitch_black_frame_non_viable() -> None:
    """Verifies that pitch-black images (< 15.0) are marked non-viable for inference."""
    analyzer = FrameQualityAnalyzer()
    black_frame = np.zeros((100, 100, 3), dtype=np.uint8)

    result = analyzer.analyze(black_frame)
    assert result.lighting == LightingCondition.UNDEREXPOSED
    assert result.status == QualityStatus.SEVERELY_DEGRADED
    assert result.is_viable_for_inference is False
    assert any("Cabin illumination too low" in r for r in result.reasons)


def test_overexposed_glare_frame() -> None:
    """Verifies that washed-out glare images (> 225.0) are flagged as OVEREXPOSED."""
    analyzer = FrameQualityAnalyzer(overexposed_threshold=225.0)
    glare_frame = np.full((100, 100, 3), 245, dtype=np.uint8)

    result = analyzer.analyze(glare_frame)
    assert result.lighting == LightingCondition.OVEREXPOSED
    assert result.mean_luminance == pytest.approx(245.0, abs=0.5)
    assert any("glare or overexposure" in r for r in result.reasons)


def test_blurry_frame_detection() -> None:
    """Verifies that low-texture/blurred frames produce low Laplacian variance."""
    analyzer = FrameQualityAnalyzer(blur_threshold=80.0)
    # Uniform gray image has zero edge transitions (Laplacian var = 0)
    uniform_frame = np.full((100, 100, 3), 128, dtype=np.uint8)

    result = analyzer.analyze(uniform_frame)
    assert result.blur == BlurCondition.BLURRY
    assert result.blur_variance == pytest.approx(0.0, abs=1.0)
    assert any("unfocused or smudged" in r for r in result.reasons)


def test_sharp_optimal_frame() -> None:
    """Verifies that high-contrast textured images with balanced lighting classify as EXCELLENT."""
    analyzer = FrameQualityAnalyzer(underexposed_threshold=35.0, overexposed_threshold=225.0, blur_threshold=80.0)
    
    # Create sharp checkerboard pattern with balanced average luminance (~127)
    checkerboard = np.indices((100, 100)).sum(axis=0) % 2 * 255
    sharp_frame = cv2.cvtColor(checkerboard.astype(np.uint8), cv2.COLOR_GRAY2BGR)

    result = analyzer.analyze(sharp_frame)
    assert result.lighting == LightingCondition.OPTIMAL
    assert result.blur == BlurCondition.SHARP
    assert result.status == QualityStatus.EXCELLENT
    assert result.is_viable_for_inference is True
    assert result.blur_variance > 500.0  # Sharp checkerboard edges produce very high variance
