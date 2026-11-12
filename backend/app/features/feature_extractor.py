"""
DriveGuard — Unified Driver Feature Extractor (Phase 5).

1. What it does:
   Fuses the individual feature modules (`ear`, `head_pose`, `mar`, `blink_detector`, `perclos`)
   into a single, coordinated, per-frame pipeline pass.

2. Why it exists:
   Provides a clean, unified interface between raw facial landmark localization (Phase 4)
   and the multi-indicator decision engine (Phase 6). Ensures all physiological
   features share consistent timestamps and synchronized temporal history.

3. What data it receives:
   - `FaceLandmarkResult` from `DriveGuardFaceLandmarker`.
   - Frame dimensions (width, height).

4. What data it produces:
   - Immutable `DriverFeatures` record detailing all instantaneous and rolling indicators.

5. Modules that depend on it:
   - Downstream Decision Engine (Phase 6)
   - Diagnostic HUD & telemetry streaming (Phase 11)

6. Failure behavior:
   - Gracefully handles face loss by marking features invalid or uncertain without
     triggering false closure alerts or crashing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from backend.app.features.blink_detector import (
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import (
    EyeAspectRatioResult,
    extract_eye_aspect_ratio,
)
from backend.app.features.head_pose import (
    HeadPoseResult,
    PoseZone,
    estimate_head_pose,
)
from backend.app.features.mar import (
    MouthAspectRatioResult,
    YawnDetector,
)
from backend.app.features.perclos import (
    PerclosCalculator,
    PerclosResult,
)
from backend.app.vision.face_landmarker import (
    FaceLandmarkResult,
    FaceLandmarkStatus,
)


@dataclass(frozen=True)
class DriverFeatures:
    """
    Unified immutable record of all physiological driver measurements for one frame.
    """
    timestamp_ms: float
    has_face: bool
    face_status: FaceLandmarkStatus
    ear: EyeAspectRatioResult
    head_pose: HeadPoseResult
    mar: MouthAspectRatioResult
    blink: BlinkSnapshot
    perclos: PerclosResult

    @property
    def summary_text(self) -> str:
        """Brief human-readable diagnostic tag for HUD overlay."""
        if not self.has_face:
            return f"FACE: {self.face_status.value}"
        return (
            f"EAR: {self.ear.compensated_ear:.2f} | "
            f"PERCLOS: {self.perclos.perclos_pct:.1f}% | "
            f"Blinks: {self.blink.total_blink_count} | "
            f"Pose: {self.head_pose.zone.value} | "
            f"MAR: {self.mar.mar:.2f}"
        )


class DriverFeatureExtractor:
    """
    Coordinated multi-feature pipeline transforming 478 3D landmarks into DriverFeatures.
    """

    def __init__(
        self,
        ear_closure_threshold: float = 0.20,
        perclos_window_sec: float = 60.0,
        min_yawn_duration_ms: float = 1500.0,
        neutral_pitch_deg: float = 0.0,
        neutral_yaw_deg: float = 0.0,
        is_mirrored: bool = False,
    ) -> None:
        """
        Initialize the feature extraction pipeline.
        """
        self.ear_closure_threshold = ear_closure_threshold
        self.neutral_pitch_deg = neutral_pitch_deg
        self.neutral_yaw_deg = neutral_yaw_deg
        self.is_mirrored = is_mirrored

        self._yawn_detector = YawnDetector(min_yawn_duration_ms=min_yawn_duration_ms)
        self._blink_detector = BlinkKinematicsDetector(closure_threshold=ear_closure_threshold)
        self._perclos_calculator = PerclosCalculator(window_sec=perclos_window_sec)

    def extract(
        self,
        landmark_result: FaceLandmarkResult,
        frame_width: int = 1280,
        frame_height: int = 720,
        is_mirrored: Optional[bool] = None,
    ) -> DriverFeatures:
        """
        Extract all geometric, kinematic, and temporal features from a FaceLandmarkResult.
        """
        ts_ms = float(landmark_result.timestamp_ms)
        has_face = landmark_result.has_face and len(landmark_result.landmarks) >= 468
        aspect_ratio = frame_width / max(1.0, float(frame_height))
        mirrored = self.is_mirrored if is_mirrored is None else is_mirrored

        if has_face:
            # 1. 6-DoF Head Pose Estimation & Cockpit Zone
            pose_res = estimate_head_pose(
                landmarks=landmark_result.landmarks,
                frame_width=frame_width,
                frame_height=frame_height,
                neutral_pitch_deg=self.neutral_pitch_deg,
                neutral_yaw_deg=self.neutral_yaw_deg,
                is_mirrored=mirrored,
            )

            # 2. Eye Aspect Ratio (Yaw-weighted & Pitch-compensated relative to neutral)
            ear_res = extract_eye_aspect_ratio(
                landmarks=landmark_result.landmarks,
                yaw_deg=pose_res.rel_yaw_deg,
                pitch_deg=pose_res.rel_pitch_deg,
                aspect_ratio=aspect_ratio,
                closure_threshold=self.ear_closure_threshold,
            )

            # 3. Mouth Aspect Ratio & Yawn Tracking
            mar_res = self._yawn_detector.update(
                landmarks=landmark_result.landmarks,
                timestamp_ms=ts_ms,
                aspect_ratio=aspect_ratio,
                has_face=has_face,
            )

            # 4. Blink Kinematics FSM
            valid_eye_tracking = has_face and ear_res.is_valid
            blink_res = self._blink_detector.update(
                left_ear=ear_res.left_ear,
                right_ear=ear_res.right_ear,
                compensated_ear=ear_res.compensated_ear,
                timestamp_ms=ts_ms,
                has_face=valid_eye_tracking,
            )

            # 5. Sliding-Window PERCLOS
            perclos_res = self._perclos_calculator.update(
                timestamp_ms=ts_ms,
                is_closed=ear_res.is_closed,
                has_face=valid_eye_tracking,
            )

        else:
            # Face Loss Branch — gracefully freeze/invalidate without false alarms
            pose_res = HeadPoseResult(
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
                zone=PoseZone.OUT_OF_BOUNDS,
                is_looking_away=True,
                is_valid=False,
            )
            ear_res = EyeAspectRatioResult(
                left_ear=0.0,
                right_ear=0.0,
                weighted_ear=0.0,
                compensated_ear=0.0,
                closure_threshold=self.ear_closure_threshold,
                is_left_closed=False,
                is_right_closed=False,
                is_closed=False,
                is_valid=False,
            )
            mar_res = self._yawn_detector.update(
                landmarks=(),
                timestamp_ms=ts_ms,
                aspect_ratio=aspect_ratio,
                has_face=False,
            )
            blink_res = self._blink_detector.update(
                left_ear=0.0,
                right_ear=0.0,
                compensated_ear=0.0,
                timestamp_ms=ts_ms,
                has_face=False,
            )
            perclos_res = self._perclos_calculator.update(
                timestamp_ms=ts_ms,
                is_closed=False,
                has_face=False,
            )

        return DriverFeatures(
            timestamp_ms=ts_ms,
            has_face=has_face,
            face_status=landmark_result.status,
            ear=ear_res,
            head_pose=pose_res,
            mar=mar_res,
            blink=blink_res,
            perclos=perclos_res,
        )

    def reset(self) -> None:
        """Reset internal temporal states for a new monitoring session."""
        self._yawn_detector.reset()
        self._blink_detector.reset()
        self._perclos_calculator.reset()
