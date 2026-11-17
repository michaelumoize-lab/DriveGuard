"""
DriveGuard — Multi-Signal Reliability and Confidence Scoring (Phase 6).

Evaluates whether the upstream video capture, optical quality, and facial landmark
tracking are sufficiently reliable to support safety-critical drowsiness arbitration.
Prevents false alarms and ghost activations caused by sensor occlusions, extreme lighting
deficits, or motion blur.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from backend.app.camera.frame_quality import FrameQualityResult, QualityStatus
from backend.app.features.blink_detector import BlinkState
from backend.app.features.feature_extractor import DriverFeatures
from backend.app.features.head_pose import PoseZone
from backend.app.vision.face_landmarker import FaceLandmarkStatus


class ConfidenceTier(str, Enum):
    """Discrete reliability classification."""
    HIGH = "HIGH"
    ACCEPTABLE = "ACCEPTABLE"
    DEGRADED = "DEGRADED"
    UNAVAILABLE = "UNAVAILABLE"


@dataclass(frozen=True)
class ConfidenceAssessment:
    """
    Evaluated reliability of upstream sensing pipeline for the current frame.
    """
    overall_confidence: float
    tier: ConfidenceTier
    is_reliable: bool
    reasons: List[str] = field(default_factory=list)
    is_head_pose_available: Optional[bool] = None
    is_ocular_available: Optional[bool] = None
    is_yawn_available: Optional[bool] = None

    @property
    def head_pose_viable(self) -> bool:
        """Explicit head-pose availability if evaluated; falls back to is_reliable for legacy mocks."""
        if self.is_head_pose_available is not None:
            return self.is_head_pose_available
        return self.is_reliable

    @property
    def ocular_viable(self) -> bool:
        """Explicit ocular availability if evaluated; falls back to is_reliable for legacy mocks."""
        if self.is_ocular_available is not None:
            return self.is_ocular_available
        return self.is_reliable

    @property
    def yawn_viable(self) -> bool:
        """Explicit yawn availability if evaluated; falls back to is_reliable for legacy mocks."""
        if self.is_yawn_available is not None:
            return self.is_yawn_available
        return self.is_reliable

    @property
    def summary_text(self) -> str:
        status = "RELIABLE" if self.is_reliable else "UNRELIABLE"
        return f"[{self.tier.value} | {self.overall_confidence:.2f} | {status}]"


class SignalConfidenceEvaluator:
    """
    Evaluates optical, landmark, and geometric signals to produce a normalized
    confidence score [0.0, 1.0].
    """

    def __init__(
        self,
        min_reliable_confidence: float = 0.50,
        unreliable_grace_period_ms: float = 1000.0,
    ) -> None:
        self.min_reliable_confidence = min_reliable_confidence
        self.unreliable_grace_period_ms = unreliable_grace_period_ms
        self._unreliable_since_ms: Optional[float] = None

    def evaluate(
        self,
        features: DriverFeatures,
        quality: Optional[FrameQualityResult] = None,
    ) -> ConfidenceAssessment:
        """
        Compute multi-factor reliability score for current frame.
        """
        reasons: List[str] = []
        score = 1.0

        # 1. Face Landmark Status
        if not features.has_face:
            score = 0.0
            reasons.append(f"No face detected ({features.face_status.value})")
            return ConfidenceAssessment(
                overall_confidence=0.0,
                tier=ConfidenceTier.UNAVAILABLE,
                is_reliable=False,
                reasons=reasons,
                is_head_pose_available=False,
                is_ocular_available=False,
                is_yawn_available=False,
            )

        if features.face_status == FaceLandmarkStatus.OCCLUDED_OR_MARGINAL:
            score *= 0.5
            reasons.append("Landmark visibility compromised (marginal tracking)")
        elif features.face_status in (FaceLandmarkStatus.CAMERA_FAULT, FaceLandmarkStatus.INSUFFICIENT_DATA):
            score = 0.0
            reasons.append(f"Vision status: {features.face_status.value}")
            return ConfidenceAssessment(
                overall_confidence=0.0,
                tier=ConfidenceTier.UNAVAILABLE,
                is_reliable=False,
                reasons=reasons,
                is_head_pose_available=False,
                is_ocular_available=False,
                is_yawn_available=False,
            )

        # 2. Geometric Validity Checks
        if not features.ear.is_valid:
            score *= 0.6
            reasons.append("Eye landmarks degenerate or occluded")

        if not features.head_pose.is_valid:
            score *= 0.7
            reasons.append("SolvePnP head pose convergence failed")

        # 3. Optical Frame Quality (if provided from Phase 3)
        if quality is not None:
            if not quality.is_viable_for_inference:
                score *= 0.3
                reasons.append(f"Optical quality unviable: {quality.status.value}")
            elif quality.status == QualityStatus.POOR_LIGHTING:
                score *= 0.75
                reasons.append(f"Suboptimal illumination: {quality.lighting.value}")
            elif quality.status == QualityStatus.BLURRY:
                score *= 0.70
                reasons.append(f"Motion blur detected: {quality.blur.value}")

        score = max(0.0, min(1.0, score))

        # Determine Tier
        if score >= 0.80:
            tier = ConfidenceTier.HIGH
        elif score >= 0.50:
            tier = ConfidenceTier.ACCEPTABLE
        elif score >= 0.25:
            tier = ConfidenceTier.DEGRADED
        else:
            tier = ConfidenceTier.UNAVAILABLE

        is_reliable = score >= self.min_reliable_confidence

        # Channel availability evaluation (independent signal channels)
        is_face_viable = (
            features.has_face
            and features.face_status in (
                FaceLandmarkStatus.TRACKING,
                FaceLandmarkStatus.OCCLUDED_OR_MARGINAL,
            )
        )
        is_optical_viable = (
            quality is None or quality.is_viable_for_inference
        )
        is_head_pose_available = (
            is_face_viable
            and is_optical_viable
            and features.head_pose.is_valid
            and features.head_pose.zone != PoseZone.UNKNOWN
        )
        blink_gap = getattr(features.blink, "sampling_gap_detected", False) or (
            getattr(features.blink, "state", None) == BlinkState.UNCERTAIN_GAP
        )
        is_ocular_available = (
            is_face_viable
            and is_optical_viable
            and features.ear.is_valid
            and not blink_gap
        )
        is_yawn_available = (
            is_face_viable
            and is_optical_viable
            and features.mar.is_valid
        )

        return ConfidenceAssessment(
            overall_confidence=score,
            tier=tier,
            is_reliable=is_reliable,
            reasons=reasons,
            is_head_pose_available=is_head_pose_available,
            is_ocular_available=is_ocular_available,
            is_yawn_available=is_yawn_available,
        )

    def reset(self) -> None:
        """Reset temporal tracking state."""
        self._unreliable_since_ms = None
