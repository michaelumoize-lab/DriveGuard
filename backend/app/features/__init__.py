"""
DriveGuard — Physiological Feature Extraction Package (Phase 5).

Exposes core physiological algorithms and unified pipelines:
- Eye Aspect Ratio (EAR) with 3D Euclidean distances, yaw-weighting, and pitch compensation.
- 6-DoF SolvePnP Head Pose Estimation and cockpit gaze zone classification.
- 3-pair inner-lip Mouth Aspect Ratio (MAR) and authentic yawn detection.
- Sub-frame kinematic blink detector (closing/reopening velocities, RCR).
- 60-second sliding-window PERCLOS accumulator with coverage tracking.
- Unified DriverFeatureExtractor coordinating all feature streams.
"""

from backend.app.features.blink_detector import (
    BlinkClassification,
    BlinkEvent,
    BlinkKinematicsDetector,
    BlinkSnapshot,
    BlinkState,
)
from backend.app.features.ear import (
    DEFAULT_EAR_CLOSURE_THRESHOLD,
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeAspectRatioResult,
    calculate_pitch_compensated_ear,
    calculate_single_eye_ear,
    calculate_yaw_weighted_ear,
    extract_eye_aspect_ratio,
)
from backend.app.features.head_pose import (
    CANONICAL_3D_FACE_POINTS,
    MANDIBLE_SOLVEPNP_INDICES,
    RIGID_SOLVEPNP_INDICES,
    SOLVEPNP_LANDMARK_INDICES,
    HeadPoseResult,
    PoseZone,
    build_neutral_matrix,
    classify_pose_zone,
    estimate_head_pose,
    extract_euler_yxz,
    is_valid_rotation_matrix,
    validate_pnp_solution,
)
from backend.app.features.mar import (
    DEFAULT_MAR_OPEN_THRESHOLD,
    DEFAULT_MIN_YAWN_DURATION_MS,
    MouthAspectRatioResult,
    YawnDetector,
    YawnEvent,
    calculate_mar,
)
from backend.app.features.perclos import (
    PerclosCalculator,
    PerclosResult,
)
from backend.app.features.feature_extractor import (
    DriverFeatureExtractor,
    DriverFeatures,
)

__all__ = [
    # EAR
    "DEFAULT_EAR_CLOSURE_THRESHOLD",
    "LEFT_EYE_INDICES",
    "RIGHT_EYE_INDICES",
    "EyeAspectRatioResult",
    "calculate_single_eye_ear",
    "calculate_yaw_weighted_ear",
    "calculate_pitch_compensated_ear",
    "extract_eye_aspect_ratio",
    # Head Pose
    "CANONICAL_3D_FACE_POINTS",
    "MANDIBLE_SOLVEPNP_INDICES",
    "RIGID_SOLVEPNP_INDICES",
    "SOLVEPNP_LANDMARK_INDICES",
    "HeadPoseResult",
    "PoseZone",
    "build_neutral_matrix",
    "classify_pose_zone",
    "estimate_head_pose",
    "extract_euler_yxz",
    "is_valid_rotation_matrix",
    "validate_pnp_solution",
    # MAR
    "DEFAULT_MAR_OPEN_THRESHOLD",
    "DEFAULT_MIN_YAWN_DURATION_MS",
    "MouthAspectRatioResult",
    "YawnDetector",
    "YawnEvent",
    "calculate_mar",
    # Blink Kinematics
    "BlinkClassification",
    "BlinkEvent",
    "BlinkKinematicsDetector",
    "BlinkSnapshot",
    "BlinkState",
    # PERCLOS
    "PerclosCalculator",
    "PerclosResult",
    # Unified Extractor
    "DriverFeatureExtractor",
    "DriverFeatures",
]
