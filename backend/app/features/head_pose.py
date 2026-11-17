"""
DriveGuard — Head Pose Estimation & Cockpit Zone Classifier (Phase 5).

1. What it does:
   Estimates driver 3D head orientation (Pitch, Yaw, Roll in degrees) using
   OpenCV's `cv2.solvePnP` algorithm with canonical facial anthropometric landmarks.
   Cranial rigid landmarks (Sellion, Glabella, Nose Tip, Eye Corners) are utilized to
   render pose estimation invariant to jaw movement (speech, laughing, and yawning).
   Euler angles are extracted via closed-form intrinsic Y-X-Z Cardan/Tait-Bryan
   decomposition with full 3D relative rotation decoupling (R_rel = R_neutral.T @ R_pnp).
   Classifies head orientation into discrete cockpit gaze zones with transition hysteresis.

2. Why it exists:
   Head pose provides two vital drowsiness & distraction indicators:
   - Pitch: Detects nodding microsleep episodes (head dropping forward).
   - Yaw & Zone: Identifies mirror checks and road distraction, allowing the
     decision engine to avoid false eye closure alerts when the driver looks at mirrors.

3. What data it receives:
   - Sequence of `NormalizedLandmark` points (or NumPy pixel coordinates).
   - Video frame dimensions (`frame_width`, `frame_height`).
   - Calibrated neutral pitch, yaw, or 3x3 neutral rotation matrix.
   - Camera horizontal mirroring flag (`is_mirrored`).
   - Previous cockpit zone for hysteresis deadband filtering.

4. What data it produces:
   - `HeadPoseResult` dataclass with absolute `pitch_deg`, `yaw_deg`, `roll_deg`,
     relative `rel_pitch_deg`, `rel_yaw_deg`, `rel_roll_deg`, `zone`, `rot_mat`,
     and `is_looking_away` boolean.

5. Modules that depend on it:
   - `backend.app.features.ear` (yaw-weighted and pitch-compensated EAR)
   - `backend.app.features.feature_extractor`
   - Downstream Decision Engine and Temporal Analyzer

6. Failure behavior:
   - Returns is_valid=False, is_looking_away=False, and zone=UNKNOWN on missing face,
     non-finite coordinates, solver failure, or reprojection validation failure.
     Face loss or tracking interruption is NEVER classified as looking away.
   - OUT_OF_BOUNDS is reserved exclusively for authentic extreme head poses
     (|rel_yaw| > 50° or |rel_pitch| > 38°).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from backend.app.vision.face_landmarker import NormalizedLandmark


class PoseZone(str, Enum):
    """Discrete driver gaze/head orientation cockpit zones."""
    CENTER = "CENTER"              # Forward road view (primary driving orientation)
    LEFT_MIRROR = "LEFT_MIRROR"    # Head turned towards driver's side mirror (yaw >= +14°)
    RIGHT_MIRROR = "RIGHT_MIRROR"  # Head turned towards center/passenger mirror (yaw <= -14°)
    DASHBOARD = "DASHBOARD"        # Head tilted downward towards instruments (pitch <= -9° within forward cone)
    OUT_OF_BOUNDS = "OUT_OF_BOUNDS"# Extreme head turn or tilt outside normal cabin view (|yaw| > 50° or |pitch| > 38°)
    UNKNOWN = "UNKNOWN"            # Invalid pose, tracking loss, insufficient landmarks, or PnP failure


# Canonical 3D anthropometric facial model points (in mm, centered at nose tip)
# Coordinates are aligned with the OpenCV pinhole camera convention at neutral frontal pose (R = I):
# X: lateral (+X = image right / landmark 263, -X = image left / landmark 33)
# Y: vertical (+Y = image down / chin, -Y = image up / forehead & sellion)
# Z: sagittal depth (+Z = recessed away from camera lens into skull, 0 = nose tip)
CANONICAL_3D_FACE_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),          # Nose tip (idx 1)
        (0.0, 63.6, 12.5),        # Chin (idx 152)
        (-43.3, -32.7, 26.0),     # Right eye outer corner in mirrored / left in canonical (idx 33)
        (43.3, -32.7, 26.0),      # Left eye outer corner in mirrored / right in canonical (idx 263)
        (-28.9, 28.9, 24.1),      # Right mouth corner in mirrored (idx 61)
        (28.9, 28.9, 24.1),      # Left mouth corner in mirrored (idx 291)
        (0.0, -31.0, 22.0),       # Sellion / nose bridge between eyes (idx 6)
        (0.0, -68.0, 35.0),       # Glabella / lower forehead midpoint (idx 10)
    ],
    dtype=np.float64,
)

# MediaPipe landmark indices corresponding to CANONICAL_3D_FACE_POINTS
SOLVEPNP_LANDMARK_INDICES = (1, 152, 33, 263, 61, 291, 6, 10)

# Indices within CANONICAL_3D_FACE_POINTS that reside on the rigid skull / cranial bone
# (Indices: 0: nose tip, 2: eye 33, 3: eye 263, 6: sellion 6, 7: glabella 10).
# Excludes mandible / jaw points (152, 61, 291) to guarantee zero pitch shift during yawning/speech.
RIGID_SOLVEPNP_INDICES = (0, 2, 3, 6, 7)

# Indices within CANONICAL_3D_FACE_POINTS that reside on the mandible / jaw
# (Indices: 1: chin 152, 4: mouth corner 61, 5: mouth corner 291).
# Evaluated for residual consistency to detect jaw movement (speech, laughing, and yawning).
MANDIBLE_SOLVEPNP_INDICES = (1, 4, 5)


@dataclass(frozen=True)
class HeadPoseResult:
    """
    Structured 6-DoF head pose estimation result.
    """
    pitch_deg: float            # Positive = tilting back/up, Negative = nodding down
    yaw_deg: float              # Positive = turned to driver's left, Negative = turned to right
    roll_deg: float             # Positive = tilted toward right shoulder, Negative = left shoulder
    zone: PoseZone
    is_looking_away: bool       # True if driver is actively looking away from road (False on tracking loss)
    is_valid: bool              # True if solvePnP converged with valid orthogonal rotation matrix
    rel_pitch_deg: float = 0.0  # Pitch relative to calibrated neutral posture
    rel_yaw_deg: float = 0.0    # Yaw relative to calibrated neutral posture
    rel_roll_deg: float = 0.0   # Roll relative to calibrated neutral posture
    rot_mat: Optional[np.ndarray] = None  # 3x3 orthonormal rotation matrix


def is_valid_rotation_matrix(rot_mat: np.ndarray, tol: float = 1e-2) -> bool:
    """
    Validate that rot_mat is finite, 3x3, orthogonal (R^T R ~ I), and special orthogonal (det(R) ~ 1.0).
    """
    if not isinstance(rot_mat, np.ndarray) or rot_mat.shape != (3, 3):
        return False
    if not np.all(np.isfinite(rot_mat)):
        return False
    ident = np.eye(3, dtype=np.float64)
    if not np.allclose(rot_mat.T @ rot_mat, ident, atol=tol):
        return False
    det = float(np.linalg.det(rot_mat))
    if abs(det - 1.0) > tol:
        return False
    return True


def validate_pnp_solution(
    pts_3d: np.ndarray,
    pts_2d: np.ndarray,
    rvec: Optional[np.ndarray],
    tvec: Optional[np.ndarray],
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    d_ref_px: float,
    max_norm_mean_res: float = 0.12,
    max_norm_max_res: float = 0.25,
    min_depth_mm: float = 100.0,
    max_depth_mm: float = 3000.0,
) -> Tuple[bool, float, float]:
    """
    Validate Perspective-n-Point solution against physical depth and normalized reprojection error.

    Validation requirements:
    1. Finite check: rvec and tvec must contain finite numeric values.
    2. Depth check: camera-space depth (tvec[2]) must be positive and within [min_depth_mm, max_depth_mm] (0.1m - 3.0m).
    3. Rotation matrix: Rodrigues rotation matrix must be strictly orthonormal with det(R) ~ 1.0.
    4. Reprojection error: mean residual <= max_norm_mean_res * d_ref and max residual <= max_norm_max_res * d_ref,
       where d_ref is the outer interocular distance in pixels.

    Returns:
        (is_valid, mean_residual_px, max_residual_px)
    """
    if rvec is None or tvec is None:
        return False, float("inf"), float("inf")

    if not (np.all(np.isfinite(rvec)) and np.all(np.isfinite(tvec))):
        return False, float("inf"), float("inf")

    # Depth validation: tvec[2] is camera Z (distance in front of lens in mm)
    depth_z = float(tvec[2, 0] if tvec.ndim > 1 else tvec[2])
    if depth_z < min_depth_mm or depth_z > max_depth_mm:
        return False, float("inf"), float("inf")

    # Rotation matrix orthonormality validation
    try:
        rot_mat, _ = cv2.Rodrigues(rvec)
    except cv2.error:
        return False, float("inf"), float("inf")

    if not is_valid_rotation_matrix(rot_mat):
        return False, float("inf"), float("inf")

    # Explicit point-wise camera-space depth validation: P_camera = R @ P_model + t
    # Every 3D model point must have finite, strictly positive camera-space depth (Z_camera > 0)
    try:
        if not np.all(np.isfinite(pts_3d)):
            return False, float("inf"), float("inf")
        pts_cam = (rot_mat @ pts_3d.T).T + tvec.reshape(1, 3)
        if not np.all(np.isfinite(pts_cam)):
            return False, float("inf"), float("inf")
        if np.any(pts_cam[:, 2] <= 0.0):
            return False, float("inf"), float("inf")
    except (ValueError, TypeError):
        return False, float("inf"), float("inf")

    # Reprojection residual validation
    try:
        projected, _ = cv2.projectPoints(pts_3d, rvec, tvec, camera_matrix, dist_coeffs)
        projected_2d = projected.reshape(-1, 2)
        residuals = np.linalg.norm(pts_2d - projected_2d, axis=1)
        mean_res = float(np.mean(residuals))
        max_res = float(np.max(residuals))
    except (cv2.error, ValueError):
        return False, float("inf"), float("inf")

    ref_scale = max(20.0, float(d_ref_px))
    norm_mean = mean_res / ref_scale
    norm_max = max_res / ref_scale

    if norm_mean > max_norm_mean_res or norm_max > max_norm_max_res:
        return False, mean_res, max_res

    return True, mean_res, max_res


def build_neutral_matrix(
    neutral_pitch_deg: float = 0.0,
    neutral_yaw_deg: float = 0.0,
    neutral_rot_mat: Optional[np.ndarray] = None,
    is_mirrored: bool = False,
) -> np.ndarray:
    """
    Construct an orthonormal 3x3 neutral rotation matrix from calibrated angles or matrix.

    Architecture & Minimal Persistence Contract:
    To preserve clean backward compatibility without modifying the SQLite database schema or
    DriverProfileDTO persistence layer, R_neutral is dynamically reconstructed from the persisted
    scalar median angles (neutral_pitch_deg, neutral_yaw_deg) via:
        R_neutral = R_y(neutral_yaw_cam) @ R_x(neutral_pitch_cam)
    according to camera coordinate conventions:
    - Mirrored: driver left turn (+yaw) corresponds to +Y camera rotation.
    - Unmirrored: driver left turn (+yaw) corresponds to -Y camera rotation.

    If an explicit 3x3 neutral_rot_mat is supplied (e.g. from an in-memory session), it is validated
    for orthonormality and used directly.
    """
    if neutral_rot_mat is not None and is_valid_rotation_matrix(neutral_rot_mat):
        return neutral_rot_mat.astype(np.float64)

    cam_yaw = neutral_yaw_deg if is_mirrored else -neutral_yaw_deg
    cam_pitch = neutral_pitch_deg
    psi = float(np.radians(cam_yaw))
    theta = float(np.radians(cam_pitch))

    Ry = np.array(
        [
            [np.cos(psi), 0.0, np.sin(psi)],
            [0.0, 1.0, 0.0],
            [-np.sin(psi), 0.0, np.cos(psi)],
        ],
        dtype=np.float64,
    )
    Rx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ],
        dtype=np.float64,
    )
    return Ry @ Rx


def extract_euler_yxz(rot_mat: np.ndarray) -> Tuple[float, float, float]:
    """
    Extract intrinsic Y-X-Z (Yaw -> Pitch -> Roll) Cardan/Tait-Bryan Euler angles in degrees
    from a 3x3 rotation matrix R = R_y(yaw) @ R_x(pitch) @ R_z(roll).

    Matrix structure:
        R[0, 2] =  sin(yaw)*cos(pitch)
        R[1, 2] = -sin(pitch)
        R[2, 2] =  cos(yaw)*cos(pitch)
        R[1, 0] =  cos(pitch)*sin(roll)
        R[1, 1] =  cos(pitch)*cos(roll)

    Singularity (Gimbal lock):
        When cos(pitch) ~ 0 (pitch ~ +/-90°), roll is set to 0.0 and yaw is resolved safely.

    Returns:
        (raw_yaw_deg, raw_pitch_deg, raw_roll_deg)
    """
    sin_pitch = float(np.clip(-rot_mat[1, 2], -1.0, 1.0))
    pitch_rad = float(np.arcsin(sin_pitch))

    cos_pitch = float(np.sqrt(rot_mat[1, 0] ** 2 + rot_mat[1, 1] ** 2))
    if cos_pitch > 1e-4:
        # Standard non-singular branch
        yaw_rad = float(np.arctan2(rot_mat[0, 2], rot_mat[2, 2]))
        roll_rad = float(np.arctan2(rot_mat[1, 0], rot_mat[1, 1]))
    else:
        # Gimbal lock singularity branch at pitch ~ +/-90°
        yaw_rad = float(np.arctan2(-rot_mat[2, 0], rot_mat[0, 0]))
        roll_rad = 0.0

    return (
        float(np.degrees(yaw_rad)),
        float(np.degrees(pitch_rad)),
        float(np.degrees(roll_rad)),
    )


def classify_pose_zone(
    pitch_deg: float,
    yaw_deg: float,
    roll_deg: float = 0.0,
    neutral_pitch_deg: float = 0.0,
    neutral_yaw_deg: float = 0.0,
    current_zone: Optional[PoseZone] = None,
    hysteresis_deg: float = 1.5,
) -> PoseZone:
    """
    Classify continuous 3D Euler angles into discrete cockpit zones relative to driver neutral posture.

    Calibrated according to Euro NCAP DMS 2026 and SAE J941 automotive ergonomics with strict priority:
    1. OUT_OF_BOUNDS: |rel_yaw| > 50° OR |rel_pitch| > 38° (extreme head turn outside cabin envelope)
    2. LEFT_MIRROR: rel_yaw >= 14.0° (driver-side wing mirror check, taking precedence over downward tilt)
    3. RIGHT_MIRROR: rel_yaw <= -14.0° (center rearview or passenger mirror check, -50° to -14°)
    4. DASHBOARD: rel_pitch <= -9.0° CONSTRAINED to forward yaw cone (|rel_yaw| < 14.0°)
    5. CENTER: forward road view through windshield

    Includes transition hysteresis deadband (default 1.5°) to prevent boundary chattering when
    current_zone is supplied.
    """
    rel_pitch = pitch_deg - neutral_pitch_deg
    rel_yaw = yaw_deg - neutral_yaw_deg

    # 1. OUT_OF_BOUNDS: Extreme head orientation check
    oob_h = hysteresis_deg if current_zone == PoseZone.OUT_OF_BOUNDS else 0.0
    if abs(rel_yaw) > (50.0 - oob_h) or abs(rel_pitch) > (38.0 - oob_h):
        return PoseZone.OUT_OF_BOUNDS

    # 2. LEFT_MIRROR: Lateral driver-side glance
    left_h = hysteresis_deg if current_zone == PoseZone.LEFT_MIRROR else 0.0
    if rel_yaw >= (14.0 - left_h):
        return PoseZone.LEFT_MIRROR

    # 3. RIGHT_MIRROR: Lateral passenger/center rearview glance
    right_h = hysteresis_deg if current_zone == PoseZone.RIGHT_MIRROR else 0.0
    if rel_yaw <= (-14.0 + right_h):
        return PoseZone.RIGHT_MIRROR

    # 4. DASHBOARD: Downward glance CONSTRAINED within the forward road yaw cone
    dash_h = hysteresis_deg if current_zone == PoseZone.DASHBOARD else 0.0
    if rel_pitch <= (-9.0 + dash_h) and abs(rel_yaw) < 14.0:
        return PoseZone.DASHBOARD

    # 5. CENTER: Nominal forward road orientation
    return PoseZone.CENTER


def estimate_head_pose(
    landmarks: Union[Sequence[NormalizedLandmark], np.ndarray],
    frame_width: int = 1280,
    frame_height: int = 720,
    neutral_pitch_deg: float = 0.0,
    neutral_yaw_deg: float = 0.0,
    neutral_rot_mat: Optional[np.ndarray] = None,
    is_mirrored: bool = False,
    current_zone: Optional[PoseZone] = None,
    enable_mandible_refinement: bool = False,
) -> HeadPoseResult:
    """
    Compute 6-DoF head pose Euler angles and cockpit zone using cv2.solvePnP with rigid cranial landmarks.

    Args:
        landmarks: Sequence of NormalizedLandmark points or (N, 2)/(N, 3) NumPy array.
        frame_width: Video frame width in pixels.
        frame_height: Video frame height in pixels.
        neutral_pitch_deg: Baseline resting pitch angle (default: 0.0°).
        neutral_yaw_deg: Baseline resting yaw angle (default: 0.0°).
        neutral_rot_mat: Baseline resting 3x3 rotation matrix (optional, takes precedence if valid).
        is_mirrored: True if the video frame is horizontally mirrored (selfie/webcam view).
        current_zone: Previous classified PoseZone for transition hysteresis.
        enable_mandible_refinement: If True, conditionally refines pose with mandible points when stationary.

    Returns:
        HeadPoseResult detailing Pitch, Yaw, Roll, relative angles, rotation matrix, and classified PoseZone.
    """
    invalid_result = HeadPoseResult(
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        zone=PoseZone.UNKNOWN,
        is_looking_away=False,  # CRITICAL: Face loss or invalid measurement is NOT looking away
        is_valid=False,
        rel_pitch_deg=0.0,
        rel_yaw_deg=0.0,
        rel_roll_deg=0.0,
        rot_mat=None,
    )

    if landmarks is None:
        return invalid_result

    # Determine available landmarks count and indices
    try:
        num_landmarks = len(landmarks)
    except (TypeError, AttributeError):
        return invalid_result

    min_required = max(SOLVEPNP_LANDMARK_INDICES) + 1
    if num_landmarks < min_required:
        return invalid_result

    # Extract 2D image points in pixel space
    try:
        if isinstance(landmarks, np.ndarray):
            if landmarks.max() <= 1.5:
                # Normalized coordinates in [0, 1]
                image_pts = np.array(
                    [
                        (landmarks[idx, 0] * frame_width, landmarks[idx, 1] * frame_height)
                        for idx in SOLVEPNP_LANDMARK_INDICES
                    ],
                    dtype=np.float64,
                )
            else:
                image_pts = np.array(
                    [landmarks[idx, :2] for idx in SOLVEPNP_LANDMARK_INDICES],
                    dtype=np.float64,
                )
        else:
            image_pts = np.array(
                [
                    (landmarks[idx].x * frame_width, landmarks[idx].y * frame_height)
                    for idx in SOLVEPNP_LANDMARK_INDICES
                ],
                dtype=np.float64,
            )
    except (IndexError, AttributeError, ValueError):
        return invalid_result

    # Numerical validation: finite coordinates
    if not np.all(np.isfinite(image_pts)):
        return invalid_result

    # Bounding box span validation: reject collapsed/degenerate landmark sets (< 5 pixels)
    span = np.ptp(image_pts, axis=0)
    if span[0] < 5.0 or span[1] < 5.0:
        return invalid_result

    # Facial reference scale: outer interocular distance in pixels (landmarks 33 and 263)
    d_ref_px = float(np.linalg.norm(image_pts[2] - image_pts[3]))

    # Approximate pinhole camera intrinsic matrix
    focal_length = float(frame_width)
    center = (frame_width / 2.0, frame_height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    # Solve Perspective-n-Point pose problem
    # Primary: SQPnP on the 5 rigid cranial landmarks (1, 33, 263, 6, 10).
    # Mandible/jaw points (152, 61, 291) are excluded from the baseline solve to render pose
    # estimation invariant to jaw movement (yawning, speech, laughter).
    primary_success = False
    rvec: Optional[np.ndarray] = None
    tvec: Optional[np.ndarray] = None
    mean_res = float("inf")

    if len(image_pts) >= len(SOLVEPNP_LANDMARK_INDICES):
        rigid_3d = CANONICAL_3D_FACE_POINTS[list(RIGID_SOLVEPNP_INDICES)]
        rigid_2d = image_pts[list(RIGID_SOLVEPNP_INDICES)]
        try:
            solve_ok, rvec_cand, tvec_cand = cv2.solvePnP(
                rigid_3d,
                rigid_2d,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_SQPNP,
            )
            if solve_ok:
                valid, m_res, _ = validate_pnp_solution(
                    rigid_3d,
                    rigid_2d,
                    rvec_cand,
                    tvec_cand,
                    camera_matrix,
                    dist_coeffs,
                    d_ref_px,
                )
                if valid:
                    primary_success = True
                    rvec, tvec = rvec_cand, tvec_cand
                    mean_res = m_res
        except cv2.error:
            primary_success = False

        # Conditional Mandible Consistency & Refinement Investigation
        if primary_success and rvec is not None and tvec is not None:
            mandible_3d = CANONICAL_3D_FACE_POINTS[list(MANDIBLE_SOLVEPNP_INDICES)]
            mandible_2d = image_pts[list(MANDIBLE_SOLVEPNP_INDICES)]
            try:
                proj_mandible, _ = cv2.projectPoints(mandible_3d, rvec, tvec, camera_matrix, dist_coeffs)
                resids_mandible = np.linalg.norm(mandible_2d - proj_mandible.reshape(-1, 2), axis=1)
                norm_mandible_max = float(np.max(resids_mandible)) / max(20.0, d_ref_px)

                # Only attempt refinement if enabled AND jaw is stationary (max mandible residual <= 0.06 * d_ref)
                if enable_mandible_refinement and norm_mandible_max <= 0.06:
                    ref_ok, rvec_ref, tvec_ref = cv2.solvePnP(
                        CANONICAL_3D_FACE_POINTS,
                        image_pts,
                        camera_matrix,
                        dist_coeffs,
                        rvec=rvec.copy(),
                        tvec=tvec.copy(),
                        useExtrinsicGuess=True,
                        flags=cv2.SOLVEPNP_ITERATIVE,
                    )
                    if ref_ok:
                        ref_valid, _, _ = validate_pnp_solution(
                            CANONICAL_3D_FACE_POINTS,
                            image_pts,
                            rvec_ref,
                            tvec_ref,
                            camera_matrix,
                            dist_coeffs,
                            d_ref_px,
                        )
                        proj_rigid_ref, _ = cv2.projectPoints(rigid_3d, rvec_ref, tvec_ref, camera_matrix, dist_coeffs)
                        rigid_mean_ref = float(np.mean(np.linalg.norm(rigid_2d - proj_rigid_ref.reshape(-1, 2), axis=1)))
                        if ref_valid and rigid_mean_ref <= mean_res * 1.20:
                            rvec, tvec = rvec_ref, tvec_ref
            except (cv2.error, ValueError):
                pass  # Keep pure rigid solve on any projection or refinement issue

    if not primary_success or rvec is None or tvec is None:
        # Fallback: Iterative Levenberg-Marquardt solver over all available points
        pts_3d = CANONICAL_3D_FACE_POINTS[:len(image_pts)]
        try:
            fallback_ok, rvec_cand, tvec_cand = cv2.solvePnP(
                pts_3d,
                image_pts,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if fallback_ok:
                valid, _, _ = validate_pnp_solution(
                    pts_3d,
                    image_pts,
                    rvec_cand,
                    tvec_cand,
                    camera_matrix,
                    dist_coeffs,
                    d_ref_px,
                )
                if valid:
                    rvec, tvec = rvec_cand, tvec_cand
                else:
                    return invalid_result
            else:
                return invalid_result
        except cv2.error:
            return invalid_result

    # Convert rotation vector to 3x3 rotation matrix
    rot_mat, _ = cv2.Rodrigues(rvec)
    if not is_valid_rotation_matrix(rot_mat):
        return invalid_result

    # Build calibrated neutral matrix: R_neutral
    R_neutral = build_neutral_matrix(
        neutral_pitch_deg=neutral_pitch_deg,
        neutral_yaw_deg=neutral_yaw_deg,
        neutral_rot_mat=neutral_rot_mat,
        is_mirrored=is_mirrored,
    )

    # Compute 3D relative rotation matrix: R_relative = R_neutral.T @ R_pnp
    R_relative = R_neutral.T @ rot_mat
    if not is_valid_rotation_matrix(R_relative):
        R_relative = rot_mat.copy()

    # Closed-form intrinsic Y-X-Z Euler angle extraction
    abs_raw_yaw, abs_raw_pitch, abs_raw_roll = extract_euler_yxz(rot_mat)
    rel_raw_yaw, rel_raw_pitch, rel_raw_roll = extract_euler_yxz(R_relative)

    # Convert Euler yaw and roll to driver-centric convention:
    # Turning to driver's LEFT is positive (+yaw), turning to RIGHT is negative (-yaw).
    # Tilting toward driver's RIGHT shoulder is positive (+roll).
    # Nodding DOWN is negative (-pitch), looking UP is positive (+pitch).
    # On an unmirrored camera, OpenCV camera frame has +X to right, so left turn is -Y rotation.
    # On a mirrored camera (selfie/flip), the horizontal axis is already inverted.
    if is_mirrored:
        yaw = abs_raw_yaw
        pitch = abs_raw_pitch
        roll = -abs_raw_roll
        rel_yaw = rel_raw_yaw
        rel_pitch = rel_raw_pitch
        rel_roll = -rel_raw_roll
    else:
        yaw = -abs_raw_yaw
        pitch = abs_raw_pitch
        roll = abs_raw_roll
        rel_yaw = -rel_raw_yaw
        rel_pitch = rel_raw_pitch
        rel_roll = rel_raw_roll

    zone = classify_pose_zone(
        pitch_deg=rel_pitch,
        yaw_deg=rel_yaw,
        roll_deg=rel_roll,
        neutral_pitch_deg=0.0,
        neutral_yaw_deg=0.0,
        current_zone=current_zone,
    )
    is_looking_away = zone != PoseZone.CENTER

    return HeadPoseResult(
        pitch_deg=pitch,
        yaw_deg=yaw,
        roll_deg=roll,
        zone=zone,
        is_looking_away=is_looking_away,
        is_valid=True,
        rel_pitch_deg=rel_pitch,
        rel_yaw_deg=rel_yaw,
        rel_roll_deg=rel_roll,
        rot_mat=rot_mat,
    )
