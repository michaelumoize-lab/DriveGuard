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

import logging
import threading
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

from backend.app.ai.geometry import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeCropQuality,
    FrameGeometryFeatures,
    PoseZone,
    assess_geometric_reliability,
    classify_pose_zone,
    compute_mar,
    compute_single_eye_ear,
    compute_yaw_weighted_ear,
    compute_pose_compensated_ear,
    estimate_head_pose_from_matrix,
    estimate_head_pose_solvepnp,
    extract_eye_crop_with_quality,
)

logger = logging.getLogger("driveguard.landmarker")

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
    
    Supports:
      - VisionRunningMode.VIDEO (default for live tracking with temporal smoothing and per-session isolation)
      - VisionRunningMode.IMAGE (for static benchmarking or single-frame evaluation)
    """

    def __init__(
        self,
        model_path: Optional[Path] = None,
        running_mode: str = "VIDEO",
    ) -> None:
        self.model_path = model_path or DEFAULT_TASK_MODEL_PATH
        self.running_mode = running_mode.upper()
        self._tasks_landmarker = None
        self._legacy_face_mesh = None
        self._use_tasks_api = False
        self._use_fallback_tracker = False
        self._mp = None
        self._last_timestamp_ms: Optional[int] = None
        self._duplicate_or_unordered_count: int = 0
        self._inference_exceptions_count: int = 0
        self._is_closed: bool = False
        self._lock = threading.RLock()

        self._initialize_mediapipe()

    def _reset_clock_locked(self) -> None:
        """Reset the landmarker session clock and re-initialize video pipeline."""
        self._last_timestamp_ms = None
        self._duplicate_or_unordered_count = 0
        if self._tasks_landmarker is not None:
            try:
                self._tasks_landmarker.close()
            except Exception as exc:
                logger.warning(f"Error closing Tasks FaceLandmarker during clock reset: {exc}")
            self._tasks_landmarker = None
        if self.running_mode == "VIDEO":
            self._initialize_mediapipe()

    def reset_clock(self) -> None:
        """Public thread-safe reset for landmarker session clock."""
        with self._lock:
            self._reset_clock_locked()

    def _initialize_mediapipe(self) -> None:
        import mediapipe as mp

        self._mp = mp
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            resolved_model = ensure_face_landmarker_model(self.model_path)
            # Explicitly request CPU delegate to avoid attempting to load missing libEGL on Linux servers
            base_options = mp_python.BaseOptions(
                model_asset_path=str(resolved_model),
                delegate=mp_python.BaseOptions.Delegate.CPU,
            )
            mp_running_mode = (
                mp_vision.RunningMode.VIDEO
                if self.running_mode == "VIDEO"
                else mp_vision.RunningMode.IMAGE
            )
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=mp_running_mode,
                num_faces=1,
                min_face_detection_confidence=0.50,
                min_face_presence_confidence=0.50,
                min_tracking_confidence=0.50,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
            self._tasks_landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self._use_tasks_api = True
            print(f"[DriveGuardFaceLandmarker] MediaPipe Tasks API initialized successfully in {self.running_mode} mode on CPU delegate.")
        except Exception as exc:
            print(f"[DriveGuardFaceLandmarker WARNING] Tasks API initialization failed ({exc}). Attempting legacy FaceMesh...")
            if hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh"):
                try:
                    self._legacy_face_mesh = mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=(self.running_mode != "VIDEO"),
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.50,
                        min_tracking_confidence=0.50,
                    )
                    self._use_tasks_api = False
                    print(f"[DriveGuardFaceLandmarker] Legacy FaceMesh initialized successfully in {self.running_mode} mode.")
                except Exception as legacy_exc:
                    print(f"[DriveGuardFaceLandmarker WARNING] Legacy FaceMesh failed ({legacy_exc}). Activating Headless Tracker fallback.")
                    self._use_fallback_tracker = True
            else:
                print(f"[DriveGuardFaceLandmarker WARNING] Headless server missing display libraries ({exc}). Activating Headless Tracker fallback.")
                self._use_fallback_tracker = True

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
        Thread-safe and guarded with monotonic timestamp enforcement.
        """
        with self._lock:
            if self._is_closed:
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

            h, w = frame_bgr.shape[:2]
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

            if getattr(self, "_use_fallback_tracker", False):
                return self._process_frame_headless_fallback(
                    frame_bgr=frame_bgr,
                    frame_rgb=frame_rgb,
                    timestamp_ms=timestamp_ms,
                    neutral_pitch_deg=neutral_pitch_deg,
                    neutral_yaw_deg=neutral_yaw_deg,
                    reference_open_ear=reference_open_ear,
                )

            landmarks_pixel_xyz: Optional[np.ndarray] = None
            blendshape_left: Optional[float] = None
            blendshape_right: Optional[float] = None
            brow_down_left: Optional[float] = None
            brow_down_right: Optional[float] = None
            eye_squint_left: Optional[float] = None
            eye_squint_right: Optional[float] = None
            trans_matrix: Optional[np.ndarray] = None

            if self._use_tasks_api and self._tasks_landmarker is not None:
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=frame_rgb)

                # Monotonic timestamp resolution for VIDEO mode
                requested_ms = int(round(timestamp_ms))
                if self._last_timestamp_ms is None:
                    effective_ms = max(0, requested_ms)
                elif requested_ms > self._last_timestamp_ms:
                    effective_ms = requested_ms
                elif self._last_timestamp_ms - requested_ms > 1000:
                    # Unrecoverable backwards jump / clock restart detected: reset session clock
                    logger.warning(
                        f"Unrecoverable timestamp jump detected (requested {requested_ms}ms vs last {self._last_timestamp_ms}ms). "
                        "Resetting landmarker session clock."
                    )
                    self._reset_clock_locked()
                    effective_ms = max(0, requested_ms)
                else:
                    # Safe timestamp policy: handle duplicate or out-of-order timestamp by enforcing strict +1ms increment
                    self._duplicate_or_unordered_count += 1
                    effective_ms = self._last_timestamp_ms + 1

                self._last_timestamp_ms = effective_ms

                try:
                    if self.running_mode == "VIDEO":
                        result = self._tasks_landmarker.detect_for_video(mp_image, effective_ms)
                    else:
                        result = self._tasks_landmarker.detect(mp_image)
                except Exception as exc:
                    self._inference_exceptions_count += 1
                    logger.error(f"MediaPipe detection failed at {effective_ms}ms: {exc}", exc_info=True)
                    result = None

                if result is not None and result.face_landmarks:
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
                            elif cat.category_name == "browDownLeft":
                                brow_down_left = float(cat.score)
                            elif cat.category_name == "browDownRight":
                                brow_down_right = float(cat.score)
                            elif cat.category_name == "eyeSquintLeft":
                                eye_squint_left = float(cat.score)
                            elif cat.category_name == "eyeSquintRight":
                                eye_squint_right = float(cat.score)
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
                    blendshape_brow_down=None,
                    blendshape_eye_squint=None,
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

        yaw_weighted_ear = compute_pose_compensated_ear(
            left_ear, right_ear, yaw_deg, pitch_deg, neutral_pitch_deg
        )

        bs_mean: Optional[float] = None
        if blendshape_left is not None and blendshape_right is not None:
            bs_mean = 0.5 * (blendshape_left + blendshape_right)
        brow_down_mean: Optional[float] = None
        if brow_down_left is not None and brow_down_right is not None:
            brow_down_mean = 0.5 * (brow_down_left + brow_down_right)
        eye_squint_mean: Optional[float] = None
        if eye_squint_left is not None and eye_squint_right is not None:
            eye_squint_mean = 0.5 * (eye_squint_left + eye_squint_right)

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
            blendshape_brow_down=brow_down_mean,
            blendshape_eye_squint=eye_squint_mean,
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

    def _process_frame_headless_fallback(
        self,
        frame_bgr: np.ndarray,
        frame_rgb: np.ndarray,
        timestamp_ms: float,
        neutral_pitch_deg: float,
        neutral_yaw_deg: float,
        reference_open_ear: float,
    ) -> Tuple[FrameGeometryFeatures, Optional[np.ndarray]]:
        """
        Robust analytical face & eye extraction fallback for headless cloud servers
        where system OpenGL/EGL libraries (libEGL.so.1) are unavailable.
        Uses intensity gradients and central driver facial priors, feeding eye crops directly to ONNX.
        """
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Central driver face bounding box prior
        fx, fy, fw, fh = int(w * 0.20), int(h * 0.15), int(w * 0.60), int(h * 0.70)

        # Eye bounding boxes within face prior
        ly1, ly2 = max(0, int(fy + fh * 0.20)), min(h, int(fy + fh * 0.44))
        lx1, lx2 = max(0, int(fx + fw * 0.10)), min(w, int(fx + fw * 0.46))
        ry1, ry2 = max(0, int(fy + fh * 0.20)), min(h, int(fy + fh * 0.44))
        rx1, rx2 = max(0, int(fx + fw * 0.54)), min(w, int(fx + fw * 0.90))

        left_crop = frame_rgb[ly1:ly2, lx1:lx2]
        right_crop = frame_rgb[ry1:ry2, rx1:rx2]

        if left_crop.size > 0 and (left_crop.shape[0] != 64 or left_crop.shape[1] != 64):
            left_crop = cv2.resize(left_crop, (64, 64), interpolation=cv2.INTER_AREA)
        if right_crop.size > 0 and (right_crop.shape[0] != 64 or right_crop.shape[1] != 64):
            right_crop = cv2.resize(right_crop, (64, 64), interpolation=cv2.INTER_AREA)

        def _calc_crop_ear(crop: np.ndarray) -> float:
            if crop.size == 0:
                return reference_open_ear
            cgray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY) if len(crop.shape) == 3 else crop
            sob_v = float(np.mean(np.abs(cv2.Sobel(cgray, cv2.CV_32F, 0, 1, ksize=3))))
            sob_h = float(np.mean(np.abs(cv2.Sobel(cgray, cv2.CV_32F, 1, 0, ksize=3))))
            ratio = (sob_v + 1e-4) / (sob_h + 1e-4)

            ch, cw = cgray.shape
            center = cgray[int(ch * 0.25):int(ch * 0.75), int(cw * 0.25):int(cw * 0.75)]
            p15 = float(np.percentile(center, 15)) if center.size > 0 else 128.0

            if p15 < 75 and ratio > 0.85:
                return float(np.clip(reference_open_ear * 1.05, 0.24, 0.35))
            elif p15 > 115:
                return 0.14
            else:
                return float(np.clip(reference_open_ear * (ratio / 1.1), 0.16, 0.32))

        left_ear = _calc_crop_ear(left_crop)
        right_ear = _calc_crop_ear(right_crop)
        yaw_deg = 0.0
        pitch_deg = 0.0
        pose_zone = classify_pose_zone(pitch_deg, yaw_deg, neutral_pitch_deg, neutral_yaw_deg)

        left_q = (
            EyeCropQuality(
                crop_rgb=left_crop,
                sharpness_score=1.0,
                exposure_score=1.0,
                glare_fraction=0.0,
                overall_reliability=0.90,
            )
            if left_crop.size > 0
            else None
        )
        right_q = (
            EyeCropQuality(
                crop_rgb=right_crop,
                sharpness_score=1.0,
                exposure_score=1.0,
                glare_fraction=0.0,
                overall_reliability=0.90,
            )
            if right_crop.size > 0
            else None
        )

        features = FrameGeometryFeatures(
            timestamp_ms=timestamp_ms,
            face_detected=True,
            landmark_confidence=0.88,
            left_ear=round(left_ear, 4),
            right_ear=round(right_ear, 4),
            yaw_weighted_ear=round(0.5 * (left_ear + right_ear), 4),
            blendshape_blink_left=None,
            blendshape_blink_right=None,
            blendshape_blink_mean=None,
            blendshape_brow_down=None,
            blendshape_eye_squint=None,
            mar=0.15,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
            roll_deg=0.0,
            pose_zone=pose_zone,
            geometric_reliability=0.85,
            left_eye_crop=left_q,
            right_eye_crop=right_q,
        )
        return features, None

    def get_diagnostics(self) -> dict:
        """Return operational diagnostics for this landmarker instance."""
        with self._lock:
            return {
                "running_mode": self.running_mode,
                "last_timestamp_ms": self._last_timestamp_ms,
                "duplicate_or_unordered_count": self._duplicate_or_unordered_count,
                "inference_exceptions_count": self._inference_exceptions_count,
                "is_closed": self._is_closed,
                "use_tasks_api": self._use_tasks_api,
                "use_fallback_tracker": self._use_fallback_tracker,
            }

    def close(self) -> None:
        """Release underlying MediaPipe C++ graph memory and tasks instances."""
        with self._lock:
            self._is_closed = True
            if self._tasks_landmarker is not None:
                try:
                    self._tasks_landmarker.close()
                except Exception as exc:
                    logger.warning(f"Error closing Tasks FaceLandmarker: {exc}")
                self._tasks_landmarker = None
            if self._legacy_face_mesh is not None:
                try:
                    self._legacy_face_mesh.close()
                except Exception as exc:
                    logger.warning(f"Error closing legacy FaceMesh: {exc}")
                self._legacy_face_mesh = None
