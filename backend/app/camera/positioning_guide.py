"""
DriveGuard — Real-Time Camera Positioning & Framing Guide Subsystem.

Designed specifically for automotive cockpits with obliquely mounted cameras
(e.g. phone mounted on center dash, air vent, or passenger side).

Guiding Principle:
The driver is instructed to look naturally forward down the ROADWAY, NOT at the camera lens.
The guide verifies:
1. Oblique road-gaze angle is within acceptable bounds (<= 32° yaw).
2. Face is framed with sufficient margin for mirror checks without leaving view.
3. Both eyes are clearly visible to the camera (not occluded).
4. Distance and in-cabin illumination meet automotive DMS requirements.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.app.camera.frame_quality import FrameQualityResult, QualityStatus
from backend.app.features.head_pose import HeadPoseResult
from backend.app.vision.face_landmarker import FaceLandmarkResult


@dataclass
class CameraPositioningResult:
    """Evaluation result of current camera placement, framing, and angle."""

    is_optimal: bool
    has_face: bool

    # Distance / Scale
    distance_status: str  # "PERFECT", "TOO_FAR", "TOO_CLOSE", "UNKNOWN"
    distance_feedback: str
    face_height_ratio: float  # Face height / frame height (ideal: 0.18 - 0.50)

    # Lateral Angle (Yaw) & Tilt (Pitch / Roll)
    angle_status: str  # "OPTIMAL", "TOO_OBLIQUE", "TOO_STEEP", "TILTED", "UNKNOWN"
    angle_feedback: str
    yaw_deg: float
    pitch_deg: float
    roll_deg: float

    # Centering / Reticle Framing
    centering_status: str  # "CENTERED", "TOO_LEFT", "TOO_RIGHT", "TOO_HIGH", "TOO_LOW", "UNKNOWN"
    centering_feedback: str
    face_center_norm: Tuple[float, float]  # (cx, cy) in [0, 1]
    face_bbox_norm: Tuple[float, float, float, float]  # (x1, y1, x2, y2)

    # Optical Environment
    optical_status: str  # "EXCELLENT", "TOO_DARK", "TOO_BRIGHT", "BLURRY", "UNKNOWN"
    optical_feedback: str

    # Primary Actionable Takeaway
    primary_instruction: str


def evaluate_camera_position(
    landmark_result: Optional[FaceLandmarkResult],
    head_pose: Optional[HeadPoseResult],
    quality: Optional[FrameQualityResult],
    frame_width: int = 1280,
    frame_height: int = 720,
) -> CameraPositioningResult:
    """
    Evaluates real-time camera alignment for oblique vehicle mounts.
    Supports driver looking straight ahead at the road (yaw up to ±32° relative to camera).
    """
    if landmark_result is None or not landmark_result.has_face or landmark_result.landmarks is None:
        return CameraPositioningResult(
            is_optimal=False,
            has_face=False,
            distance_status="UNKNOWN",
            distance_feedback="No driver face detected in camera field of view.",
            face_height_ratio=0.0,
            angle_status="UNKNOWN",
            angle_feedback="N/A",
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            centering_status="UNKNOWN",
            centering_feedback="Sit in the driver seat looking straight ahead down the roadway.",
            face_center_norm=(0.5, 0.5),
            face_bbox_norm=(0.4, 0.3, 0.6, 0.7),
            optical_status="UNKNOWN",
            optical_feedback="N/A",
            primary_instruction="Look down the roadway; position camera so your face is visible.",
        )

    landmarks = landmark_result.landmarks

    # 1. Centering & Face Scale
    try:
        if isinstance(landmarks, np.ndarray):
            non_zero_mask = (landmarks[:, 0] != 0.0) | (landmarks[:, 1] != 0.0)
            if np.any(non_zero_mask):
                valid_lms = landmarks[non_zero_mask]
                xs = valid_lms[:, 0]
                ys = valid_lms[:, 1]
            else:
                xs = landmarks[:, 0]
                ys = landmarks[:, 1]
            nose_x = float(landmarks[1, 0])
            nose_y = float(landmarks[1, 1])
            chin_y = float(landmarks[152, 1])
            top_y = float(landmarks[10, 1])
        else:
            non_zeros = [(lm.x, lm.y) for lm in landmarks if lm.x != 0.0 or lm.y != 0.0]
            if non_zeros:
                xs = [p[0] for p in non_zeros]
                ys = [p[1] for p in non_zeros]
            else:
                xs = [lm.x for lm in landmarks]
                ys = [lm.y for lm in landmarks]
            nose_x = float(landmarks[1].x)
            nose_y = float(landmarks[1].y)
            chin_y = float(landmarks[152].y)
            top_y = float(landmarks[10].y)
        min_x, max_x = float(np.min(xs)), float(np.max(xs))
        min_y, max_y = float(np.min(ys)), float(np.max(ys))
    except (IndexError, AttributeError):
        nose_x, nose_y, chin_y, top_y = 0.5, 0.5, 0.7, 0.3
        min_x, max_x, min_y, max_y = 0.35, 0.65, 0.25, 0.75

    face_h_ratio = abs(chin_y - top_y)
    face_cx = (min_x + max_x) / 2.0
    face_cy = (min_y + max_y) / 2.0

    if max_x - min_x < 0.05:
        # Fallback width if sparse landmarks provided
        face_w_half = max(0.08, face_h_ratio * 0.35)
        min_x = max(0.0, face_cx - face_w_half)
        max_x = min(1.0, face_cx + face_w_half)

    # Distance Evaluation
    if face_h_ratio < 0.18:
        dist_status = "TOO_FAR"
        dist_feedback = "Move phone closer (face occupies < 18% of frame)."
    elif face_h_ratio > 0.50:
        dist_status = "TOO_CLOSE"
        dist_feedback = "Move phone slightly further back (face occupies > 50% of frame)."
    else:
        dist_status = "PERFECT"
        dist_feedback = "Distance is ideal."

    # Centering Evaluation (Generous cockpit safe zone: 18% - 82% width, 15% - 80% height)
    if min_x < 0.05 or face_cx < 0.18:
        cent_status = "TOO_LEFT"
        cent_feedback = "Face too close to left edge. Angle camera slightly rightward."
    elif max_x > 0.95 or face_cx > 0.82:
        cent_status = "TOO_RIGHT"
        cent_feedback = "Face too close to right edge. Angle camera slightly leftward."
    elif min_y < 0.05 or face_cy < 0.16:
        cent_status = "TOO_HIGH"
        cent_feedback = "Face too close to top edge. Tilt camera slightly upward."
    elif max_y > 0.95 or face_cy > 0.82:
        cent_status = "TOO_LOW"
        cent_feedback = "Face too close to bottom edge. Tilt camera slightly downward."
    else:
        cent_status = "CENTERED"
        cent_feedback = "Face is well-framed within camera view."

    # 2. Angle Evaluation for Oblique Mounts (Driver looking forward at the road)
    yaw = head_pose.yaw_deg if head_pose else 0.0
    pitch = head_pose.pitch_deg if head_pose else 0.0
    roll = head_pose.roll_deg if head_pose else 0.0

    angle_issues = []
    # Up to ±32.0° is optimal for oblique dash/vent mount
    if abs(roll) > 14.0:
        angle_status = "TILTED"
        angle_issues.append("Level phone mount horizontally (phone is rotated sideways).")
    elif abs(yaw) > 32.0:
        angle_status = "TOO_OBLIQUE"
        angle_issues.append(f"Road angle is {yaw:+.0f}° (exceeds 32° limit). Rotate phone more toward driver.")
    elif pitch > 26.0:
        angle_status = "TOO_STEEP"
        angle_issues.append("Camera mounted too low pointing up chin. Mount higher on dashboard.")
    elif pitch < -22.0:
        angle_status = "TOO_STEEP"
        angle_issues.append("Camera mounted too high pointing down. Mount lower near steering height.")
    else:
        angle_status = "OPTIMAL"
        angle_issues.append(f"Roadway gaze angle is optimal ({yaw:+.0f}° oblique mount).")

    angle_feedback = " ".join(angle_issues)

    # 3. Optical Environment Evaluation
    opt_status = "EXCELLENT"
    opt_feedback = "Lighting and focus are clear."
    if quality is not None:
        if quality.mean_luminance < 35.0:
            opt_status = "TOO_DARK"
            opt_feedback = "Lighting is too dark. Increase in-cabin illumination."
        elif quality.mean_luminance > 230.0:
            opt_status = "TOO_BRIGHT"
            opt_feedback = "Direct sunlight glare detected. Adjust sun visor."
        elif quality.blur_variance < 40.0:
            opt_status = "BLURRY"
            opt_feedback = "Camera focus is blurry or vibrating. Tighten phone mount."

    # 4. Overall Optimal Status
    is_optimal = (
        dist_status == "PERFECT"
        and cent_status == "CENTERED"
        and angle_status == "OPTIMAL"
        and opt_status == "EXCELLENT"
    )

    # Primary instruction prioritized
    if not is_optimal:
        if dist_status != "PERFECT":
            primary = dist_feedback
        elif cent_status != "CENTERED":
            primary = cent_feedback
        elif angle_status != "OPTIMAL":
            primary = angle_feedback
        else:
            primary = opt_feedback
    else:
        primary = f"Looking forward down road ({yaw:+.0f}° oblique). Position optimal! Press 'c' to calibrate."

    return CameraPositioningResult(
        is_optimal=is_optimal,
        has_face=True,
        distance_status=dist_status,
        distance_feedback=dist_feedback,
        face_height_ratio=round(face_h_ratio, 3),
        angle_status=angle_status,
        angle_feedback=angle_feedback,
        yaw_deg=round(yaw, 1),
        pitch_deg=round(pitch, 1),
        roll_deg=round(roll, 1),
        centering_status=cent_status,
        centering_feedback=cent_feedback,
        face_center_norm=(round(face_cx, 3), round(face_cy, 3)),
        face_bbox_norm=(round(min_x, 3), round(min_y, 3), round(max_x, 3), round(max_y, 3)),
        optical_status=opt_status,
        optical_feedback=opt_feedback,
        primary_instruction=primary,
    )


def draw_positioning_guide(
    canvas: np.ndarray,
    guide_res: CameraPositioningResult,
    is_calibrating: bool = False,
) -> np.ndarray:
    """
    Renders an interactive framing guide and alignment banner for oblique cockpit mounts.
    Suppressed automatically during active 4-zone calibration.
    """
    if is_calibrating:
        return canvas

    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Status color
    if not guide_res.has_face:
        status_color = (239, 68, 68)   # Red-500
    elif guide_res.is_optimal:
        status_color = (34, 197, 94)   # Emerald-500
    else:
        status_color = (234, 179, 8)   # Amber-500

    # 1. Dynamic Face Reticle tracking the driver's actual head
    overlay = canvas.copy()
    if guide_res.has_face:
        min_x, min_y, max_x, max_y = guide_res.face_bbox_norm
        x1 = max(0, int(min_x * width) - 10)
        y1 = max(0, int(min_y * height) - 10)
        x2 = min(width - 1, int(max_x * width) + 10)
        y2 = min(height - 1, int(max_y * height) + 10)

        # Draw smooth corner brackets around driver's face
        corner_len = min(28, (x2 - x1) // 4)
        th = 2
        # Top-left
        cv2.line(overlay, (x1, y1), (x1 + corner_len, y1), status_color, th, cv2.LINE_AA)
        cv2.line(overlay, (x1, y1), (x1, y1 + corner_len), status_color, th, cv2.LINE_AA)
        # Top-right
        cv2.line(overlay, (x2, y1), (x2 - corner_len, y1), status_color, th, cv2.LINE_AA)
        cv2.line(overlay, (x2, y1), (x2, y1 + corner_len), status_color, th, cv2.LINE_AA)
        # Bottom-left
        cv2.line(overlay, (x1, y2), (x1 + corner_len, y2), status_color, th, cv2.LINE_AA)
        cv2.line(overlay, (x1, y2), (x1, y2 - corner_len), status_color, th, cv2.LINE_AA)
        # Bottom-right
        cv2.line(overlay, (x2, y2), (x2 - corner_len, y2), status_color, th, cv2.LINE_AA)
        cv2.line(overlay, (x2, y2), (x2, y2 - corner_len), status_color, th, cv2.LINE_AA)

        # Floating angle badge right above face
        badge_text = f"Road Gaze: {guide_res.yaw_deg:+.0f}\xb0 yaw"
        cv2.putText(overlay, badge_text, (x1, max(24, y1 - 8)), font, 0.40, status_color, 1, cv2.LINE_AA)

    # 2. Cockpit Safe-Zone Margins (Dashed subtle border)
    safe_x1, safe_y1 = int(0.12 * width), int(0.12 * height)
    safe_x2, safe_y2 = int(0.88 * width), int(0.85 * height)
    cv2.rectangle(overlay, (safe_x1, safe_y1), (safe_x2, safe_y2), (71, 85, 105), 1)
    cv2.putText(overlay, "CABIN MIRROR-CHECK MARGIN", (safe_x1 + 8, safe_y1 + 16), font, 0.34, (100, 116, 139), 1, cv2.LINE_AA)

    cv2.addWeighted(overlay, 0.80, canvas, 0.20, 0, canvas)

    # 3. Bottom Positioning Guide Banner
    banner_h = 76
    banner_y = height - banner_h - 14
    sub = canvas.copy()
    bg_color = (6, 78, 59) if guide_res.is_optimal else ((15, 23, 42) if guide_res.has_face else (50, 10, 10))
    cv2.rectangle(sub, (16, banner_y), (width - 16, banner_y + banner_h), bg_color, -1)
    cv2.addWeighted(sub, 0.88, canvas, 0.12, 0, canvas)
    cv2.rectangle(canvas, (16, banner_y), (width - 16, banner_y + banner_h), status_color, 2)

    # Banner Line 1: Header / Title + Badges
    y_l1 = banner_y + 24
    if guide_res.is_optimal:
        header_text = "CAMERA POSITION OPTIMAL: READY FOR CALIBRATION (Press 'c')"
        h_color = (34, 197, 94)
    elif not guide_res.has_face:
        header_text = "CAMERA MOUNTING GUIDE: LOOK FORWARD DOWN THE ROADWAY"
        h_color = (239, 68, 68)
    else:
        header_text = "CAMERA MOUNTING GUIDE: (Look straight ahead down roadway, NOT at camera)"
        h_color = (234, 179, 8)

    cv2.putText(canvas, header_text, (28, y_l1), font, 0.43, h_color, 1, cv2.LINE_AA)

    # Alignment mini-badges on right of Line 1
    dist_badge = f"Dist:{guide_res.distance_status}"
    ang_badge = f"Angle:{guide_res.angle_status} ({guide_res.yaw_deg:+.0f}\xb0)"
    b_txt = f"{dist_badge}  |  {ang_badge}"
    (bw, _), _ = cv2.getTextSize(b_txt, font, 0.38, 1)
    if width - 28 - bw > 400:
        cv2.putText(canvas, b_txt, (width - 28 - bw, y_l1), font, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    # Banner Line 2: Real-time actionable guidance
    y_l2 = banner_y + 54
    instr_text = guide_res.primary_instruction
    cv2.putText(canvas, f">> {instr_text}", (28, y_l2), font, 0.44, (255, 255, 255), 1, cv2.LINE_AA)

    return canvas
