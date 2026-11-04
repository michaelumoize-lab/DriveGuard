"""
MediaPipe Face Landmarker Wrapper with Automatic Model Asset Management.

Supports:
1. MediaPipe Tasks Vision API (`mp.tasks.vision.FaceLandmarker`) with 478 3D landmarks,
   52 facial blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`), and 4x4 facial transformation matrix.
2. Automatic download of Google's official `face_landmarker.task` bundle into `backend/weights/`
   if not already present on disk.
3. Legacy `mp.solutions.face_mesh` fallback if Tasks API is unavailable.
"""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.app.ai.geometry import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
     FrameGeometryFeatures,
    PoseZone,
    assess_geometric_reliability,
    classify_pose_zone,
    compute_mar,
    compute_single_eye_ear,
    compute_yaw_weighted_ear,
    estimate_head_pose_from_matrix,
    estimate_head_pose_solvepnp,
    extract_eye_crop_with_quality,
)

FACE_LANDMARKER_TASK_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
DEFAULT_TASK_MODEL_PATH = (
    Path(__file__).resolve().parents[2] / "weights" / "face_landmarker.task"
)


def ensure_face_landmarker_model(model_path: Path = DEFAULT_TASK_MODEL_PATH) -> Path:
    """Download the official MediaPipe `face_landmarker.task` model if not already present."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.is_file() or model_path.stat().st_size < 100_000:
        urllib.request.urlretrieve(FACE_LANDMARKER_TASK_URL, str(model_path))
    return model_path


class DriveGuardFaceLandmarker:
    """
    Extracts 478 3D facial landmarks, blendshape eye-blink scores, head pose, and eye crops
    from RGB frames using MediaPipe Face Landmarker.
    """

    def __init__(self, model_path: Optional[Path] = None) -> None:
        self.model_path = model_path or DEFAULT_TASK_MODEL_PATH
        self._tasks_landmarker = None
        self._legacy_face_mesh = None
        self._use_tasks_api = False
        self._mp = None

        self._initialize_mediapipe()

    def _initialize_mediapipe(self) -> None:
        import mediapipe as mp

        self._mp = mp
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            resolved_model = ensure_face_landmarker_model(self.model_path)
            base_options = mp_python.BaseOptions(model_asset_path=str(resolved_model))
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.50,
                min_face_presence_confidence=0.50,
                min_tracking_confidence=0.50,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._tasks_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._use_tasks_api = True
        except Exception as exc:
            print(f"[DriveGuardFaceLandmarker WARNING] Tasks API initialization failed ({exc}). Attempting legacy FaceMesh...")
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                try:
                    self._legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=False,
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.50,
                        min_tracking_confidence=0.50,
                    )
                    self._use_tasks_api = False
                    print("[DriveGuardFaceLandmarker] Legacy FaceMesh initialized successfully.")
                except Exception as legacy_exc:
                    raise RuntimeError(f"Could not initialize Tasks API ({exc}) nor legacy FaceMesh ({legacy_exc})")
            else:
                raise RuntimeError(f"Could not initialize MediaPipe Tasks API: {exc}")

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: float,
        neutral_pitch_deg: float = 0.0,
        neutral_yaw_deg: float = 0.0,
        reference_open_ear: float = 0.28,
        extract_crops: bool = True,
    ) -> Tuple[FrameGeometryFeatures, Optional[np.ndarray]]:
        """
        Process a BGR video frame and return `(FrameGeometryFeatures, landmarks_pixel_xyz)`.
        """
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        landmarks_pixel_xyz: Optional[np.ndarray] = None
        blendshape_left: Optional[float] = None
        blendshape_right: Optional[float] = None
        trans_matrix: Optional[np.ndarray] = None

        if self._use_tasks_api and self._tasks_landmarker is not None:
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)
            result = self._tasks_landmarker.detect(mp_image)
            if result.face_landmarks:
                lms = result.face_landmarks[0]
                landmarks_pixel_xyz = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float64
                )
                if result.face_blendshapes:
                    for cat in result.face_blendshapes[0]:
                        if cat.category_name == "eyeBlinkLeft":
                            blendshape_left = float(cat.score)
                        elif cat.category_name == "eyeBlinkRight":
                            blendshape_right = float(cat.score)
                if result.facial_transformation_matrixes:
                    trans_matrix = np.array(
                        result.facial_transformation_matrixes[0], dtype=np.float64
                    )
        elif self._legacy_face_mesh is not None:
            result = self._legacy_face_mesh.process(frame_rgb)
            if result.multi_face_landmarks:
                lms = result.multi_face_landmarks[0].landmark
                landmarks_pixel_xyz = np.array(
                    [[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float64
                )

        if landmarks_pixel_xyz is None:
            return (
                FrameGeometryFeatures(
                    timestamp_ms=timestamp_ms,
                    face_detected=False,
                    landmark_confidence=0.0,
                    left_ear=0.0,
                    right_ear=0.0,
                    yaw_weighted_ear=0.0,
                    blendshape_blink_left=None,
                    blendshape_blink_right=None,
                    blendshape_blink_mean=None,
                    mar=0.0,
                    pitch_deg=0.0,
                    yaw_deg=0.0,
                    roll_deg=0.0,
                    pose_zone=PoseZone.OUT_OF_BOUNDS,
                    geometric_reliability=0.0,
                ),
                None,
            )

        left_ear = compute_single_eye_ear(landmarks_pixel_xyz, LEFT_EYE_INDICES)
        right_ear = compute_single_eye_ear(landmarks_pixel_xyz, RIGHT_EYE_INDICES)
        mar = compute_mar(landmarks_pixel_xyz)

        if trans_matrix is not None:
            pitch_deg, yaw_deg, roll_deg = estimate_head_pose_from_matrix(trans_matrix)
        else:
            pitch_deg, yaw_deg, roll_deg = estimate_head_pose_solvepnp(landmarks_pixel_xyz, w, h)

        yaw_weighted_ear = compute_yaw_weighted_ear(left_ear, right_ear, yaw_deg)

        bs_mean: Optional[float] = None
        if blendshape_left is not None and blendshape_right is not None:
            bs_mean = 0.5 * (blendshape_left + blendshape_right)

        pose_zone = classify_pose_zone(pitch_deg, yaw_deg, neutral_pitch_deg, neutral_yaw_deg)
        geom_rel, _ = assess_geometric_reliability(
            face_detected=True,
            landmark_confidence=0.92,
            pitch_deg=pitch_deg - neutral_pitch_deg,
            yaw_deg=yaw_deg - neutral_yaw_deg,
            roll_deg=roll_deg,
            left_ear=left_ear,
            right_ear=right_ear,
            blendshape_blink_mean=bs_mean,
            reference_open_ear=reference_open_ear,
        )

        left_crop = None
        right_crop = None
        if extract_crops:
            left_crop = extract_eye_crop_with_quality(
                frame_rgb, landmarks_pixel_xyz, LEFT_EYE_INDICES
            )
            right_crop = extract_eye_crop_with_quality(
                frame_rgb, landmarks_pixel_xyz, RIGHT_EYE_INDICES
            )

        features = FrameGeometryFeatures(
            timestamp_ms=timestamp_ms,
            face_detected=True,
            landmark_confidence=0.92,
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            yaw_weighted_ear=round(yaw_weighted_ear, 4),
            blendshape_blink_left=blendshape_left,
            blendshape_blink_right=blendshape_right,
            blendshape_blink_mean=bs_mean,
            mar=round(mar, 4),
            pitch_deg=round(pitch_deg, 2),
            yaw_deg=round(yaw_deg, 2),
            roll_deg=round(roll_deg, 2),
            pose_zone=pose_zone,
            geometric_reliability=round(geom_rel, 3),
            left_eye_crop=left_crop,
            right_eye_crop=right_crop,
        )
        return features, landmarks_pixel_xyz

    def close(self) -> None:
        if self._tasks_landmarker is not None:
            self._tasks_landmarker.close()
        if self._legacy_face_mesh is not None:
            self._legacy_face_mesh.close()
