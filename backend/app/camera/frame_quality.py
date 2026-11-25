"""
DriveGuard — Optical Frame Quality & Environmental Diagnostics.

1. What it does:
   Analyzes video frames for optical and environmental degradation:
   - Luminance / Exposure: Detects severe darkness (night driving, tunnels) or sun glare.
   - Sharpness / Defocus: Calculates the variance of the Laplacian to identify blurred
     or smudged lenses.

2. Why it exists:
   In real-world automotive driver monitoring systems (DMS), sudden environmental changes
   (e.g., entering an unlit road or direct headlight glare) degrade computer vision reliability.
   Attempting facial landmark extraction on pitch-black or severely blurry frames causes the AI
   to produce erratic landmark coordinates, which leads to false drowsiness alarms.
   By pre-filtering frames through this diagnostic module, the system can enter an explicit
   `INSUFFICIENT_DATA` or `LOW_LIGHT` warning state rather than falsely declaring the driver asleep.

3. What data it receives:
   A raw BGR NumPy image array from a `FramePacket`.

4. What data it produces:
   A structured, immutable `FrameQualityResult` containing mean luminance, blur variance score,
   lighting classification, sharpness classification, overall quality rating, and explainable reason codes.

5. Modules that depend on it:
   - `backend.app.vision.face_quality`
   - `backend.app.intelligence.confidence`
   - `scripts.run_camera_test`
   - Unified AI pipeline (Phase 4+)

6. Failure behavior:
   Guards against empty, malformed, or non-uint8 images by raising `ValueError` or returning
   `QualityStatus.SEVERELY_DEGRADED` with explicit diagnostic reason strings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np

logger = logging.getLogger("driveguard.camera.quality")


class LightingCondition(str, Enum):
    """Categorical classification of in-cabin illumination."""
    UNDEREXPOSED = "UNDEREXPOSED"  # Mean luminance < lower threshold (e.g., pitch-black cabin)
    OPTIMAL = "OPTIMAL"            # Balanced illumination suitable for landmark extraction
    OVEREXPOSED = "OVEREXPOSED"    # Direct sunlight or high-beam glare


class BlurCondition(str, Enum):
    """Categorical classification of optical focus and motion blur."""
    BLURRY = "BLURRY"              # Laplacian variance < threshold (out of focus or smeared)
    SHARP = "SHARP"                # Sufficient high-frequency edges for reliable landmarking


class QualityStatus(str, Enum):
    """Overall optical viability assessment."""
    EXCELLENT = "EXCELLENT"        # Optimal lighting and high sharpness
    ACCEPTABLE = "ACCEPTABLE"      # Minor degradation, still viable for inference
    POOR_LIGHTING = "POOR_LIGHTING"# Lighting compromised; landmark tracking may be degraded
    BLURRY = "BLURRY"              # Focus degraded; eye aspect ratio calculations unreliable
    SEVERELY_DEGRADED = "SEVERELY_DEGRADED"  # Non-viable for safety assessment (insufficient data)


@dataclass(frozen=True)
class FrameQualityResult:
    """
    Immutable structured evaluation of optical frame health.
    """
    mean_luminance: float
    blur_variance: float
    lighting: LightingCondition
    blur: BlurCondition
    status: QualityStatus
    is_viable_for_inference: bool
    reasons: Tuple[str, ...]

    @property
    def summary_text(self) -> str:
        """Brief human-readable diagnostic tag for HUD overlay."""
        return f"{self.status.value} (Lum: {self.mean_luminance:.1f}, Blur: {self.blur_variance:.1f})"


class FrameQualityAnalyzer:
    """
    Fast, mathematically grounded optical frame quality evaluator.
    Designed for real-time edge execution with minimal CPU overhead (< 1.5 ms per frame).
    """

    def __init__(
        self,
        underexposed_threshold: float = 35.0,
        overexposed_threshold: float = 225.0,
        blur_threshold: float = 80.0,
        severe_blur_threshold: float = 20.0,
    ) -> None:
        """
        Initialize analyzer with customizable thresholds.
        
        Args:
            underexposed_threshold: Mean grayscale luminance below which image is too dark (default: 35.0).
            overexposed_threshold: Mean grayscale luminance above which image is washed out (default: 225.0).
            blur_threshold: Laplacian variance below which image is considered blurry (default: 80.0).
            severe_blur_threshold: Laplacian variance at or below which blur is completely non-viable (default: 20.0).
        """
        if underexposed_threshold >= overexposed_threshold:
            raise ValueError("underexposed_threshold must be strictly less than overexposed_threshold.")
        if blur_threshold <= 0:
            raise ValueError("blur_threshold must be positive.")
        if severe_blur_threshold < 0 or severe_blur_threshold >= blur_threshold:
            raise ValueError(
                f"severe_blur_threshold must be non-negative and strictly less than blur_threshold ({blur_threshold}), got {severe_blur_threshold}"
            )

        self.underexposed_threshold = underexposed_threshold
        self.overexposed_threshold = overexposed_threshold
        self.blur_threshold = blur_threshold
        self.severe_blur_threshold = severe_blur_threshold

    def analyze(self, frame: np.ndarray, *, color_format: str = "BGR") -> FrameQualityResult:
        """
        Evaluate optical luminance and edge sharpness on a BGR or RGB image frame.
        
        Args:
            frame: OpenCV BGR or RGB NumPy image array (uint8).
            color_format: Input channel order ('BGR' or 'RGB').
            
        Returns:
            FrameQualityResult detailing exposure, blur, and viability.
        """
        if frame is None or frame.size == 0 or getattr(frame, "dtype", None) != np.uint8:
            return FrameQualityResult(
                mean_luminance=0.0,
                blur_variance=0.0,
                lighting=LightingCondition.UNDEREXPOSED,
                blur=BlurCondition.BLURRY,
                status=QualityStatus.SEVERELY_DEGRADED,
                is_viable_for_inference=False,
                reasons=("Empty, corrupted, or unsupported frame data type (expected uint8).",),
            )

        # Convert to single-channel 8-bit grayscale for photometric calculations
        if len(frame.shape) == 3:
            code = cv2.COLOR_RGB2GRAY if color_format.upper() == "RGB" else cv2.COLOR_BGR2GRAY
            gray = cv2.cvtColor(frame, code)
        else:
            gray = frame

        # 1. Photometric Luminance Calculation (SIMD-accelerated OpenCV mean)
        mean_lum = float(cv2.mean(gray)[0])
        reasons: List[str] = []

        if mean_lum < self.underexposed_threshold:
            lighting = LightingCondition.UNDEREXPOSED
            reasons.append(
                f"Cabin illumination too low ({mean_lum:.1f} < {self.underexposed_threshold:.1f})."
            )
        elif mean_lum > self.overexposed_threshold:
            lighting = LightingCondition.OVEREXPOSED
            reasons.append(
                f"Severe glare or overexposure detected ({mean_lum:.1f} > {self.overexposed_threshold:.1f})."
            )
        else:
            lighting = LightingCondition.OPTIMAL

        # 2. Sharpness / Blur Estimation via Laplacian Variance: Var(Laplacian(I))
        # Uses CV_32F + cv2.meanStdDev (mathematically equivalent to CV_64F.var() with ~5-10x lower latency).
        laplacian_kernel = cv2.Laplacian(gray, cv2.CV_32F)
        _, stddev = cv2.meanStdDev(laplacian_kernel)
        std_val = float(stddev[0, 0])
        blur_var = std_val * std_val

        if blur_var < self.blur_threshold:
            blur = BlurCondition.BLURRY
            reasons.append(
                f"Camera lens is unfocused or smudged (sharpness {blur_var:.1f} < {self.blur_threshold:.1f})."
            )
        else:
            blur = BlurCondition.SHARP

        # 3. Aggregate Quality Status & Inference Viability Determination
        if lighting == LightingCondition.OPTIMAL and blur == BlurCondition.SHARP:
            status = QualityStatus.EXCELLENT
            is_viable = True
        elif blur == BlurCondition.BLURRY and lighting != LightingCondition.OPTIMAL:
            status = QualityStatus.SEVERELY_DEGRADED
            is_viable = False
        elif lighting == LightingCondition.UNDEREXPOSED and mean_lum < 15.0:
            # Extreme darkness (pitch black) cannot yield facial landmarks
            status = QualityStatus.SEVERELY_DEGRADED
            is_viable = False
        elif lighting != LightingCondition.OPTIMAL:
            status = QualityStatus.POOR_LIGHTING
            is_viable = True  # Marginal lighting may still locate high-contrast facial mesh
        else:
            if blur_var <= self.severe_blur_threshold:
                # Severe blur (e.g., zero variance uniform image or extreme defocus) cannot yield facial landmarks
                status = QualityStatus.SEVERELY_DEGRADED
                is_viable = False
            else:
                status = QualityStatus.BLURRY
                is_viable = True  # Mild blur can still track general head posture

        return FrameQualityResult(
            mean_luminance=mean_lum,
            blur_variance=blur_var,
            lighting=lighting,
            blur=blur,
            status=status,
            is_viable_for_inference=is_viable,
            reasons=tuple(reasons) if reasons else ("Optimal optical frame quality.",),
        )
