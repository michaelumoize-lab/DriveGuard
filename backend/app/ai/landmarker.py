"""
MediaPipe Face Landmarker Wrapper with Automatic Model Asset Management.

Supports:
1. MediaPipe Tasks Vision API (`mp.tasks.vision.FaceLandmarker`) with 478 3D landmarks,
   52 facial blendshapes (`eyeBlinkLeft`, `eyeBlinkRight`), and 4x4 facial transformation matrix.
2. Automatic download of Google's official `face_landmarker.task` bundle into `backend/weights/`
   if not already present on disk.
3. Legacy `mp.solutions.face_mesh` fallback if Tasks API is unavailable.
4. Explicit degraded/unavailable monitoring mode if MediaPipe cannot be initialized.
   Never fabricates face presence, central head pose, or synthetic landmarks.
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
        logger.info(f"Downloading face landmarker model from {FACE_LANDMARKER_TASK_URL} to {model_path}...")
        urllib.request.urlretrieve(FACE_LANDMARKER_TASK_URL, str(model_path))
    return model_path


class DriveGuardFaceLandmarker:
    """
    Extracts 478 3D facial landmarks, blendshape eye-blink scores, head pose, and eye crops
    from RGB frames using MediaPipe Face Landmarker.
    
    Supports:
      - VisionRunningMode.VIDEO (default for live tracking with temporal smoothing and per-session isolation)
      - VisionRunningMode.IMAGE (for static benchmarking or single-frame evaluation)
      - Explicit DEGRADED_UNAVAILABLE operating mode when landmarkers cannot be initialized
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

        self.active_implementation: str = "DEGRADED_UNAVAILABLE"
        self._is_initialized: bool = False
        self._init_error: Optional[str] = None

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
        if self.running_mode == "VIDEO" and not self._use_fallback_tracker:
            self._initialize_mediapipe()

    def reset_clock(self) -> None:
        """Public thread-safe reset for landmarker session clock."""
        with self._lock:
            self._reset_clock_locked()

    def _initialize_mediapipe(self) -> None:
        try:
            import mediapipe as mp
            self._mp = mp
        except Exception as exc:
            self._mp = None
            self._init_error = f"MediaPipe import failed: {exc}"
            self._use_fallback_tracker = True
            self.active_implementation = "DEGRADED_UNAVAILABLE"
            self._is_initialized = False
            logger.warning(f"[DriveGuardFaceLandmarker] MediaPipe unavailable ({exc}). Active mode: DEGRADED_UNAVAILABLE.")
            return

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
            self.active_implementation = "MEDIAPIPE_TASKS"
            self._is_initialized = True
            self._init_error = None
            logger.info(f"[DriveGuardFaceLandmarker] MediaPipe Tasks API initialized successfully in {self.running_mode} mode on CPU delegate.")
        except Exception as exc:
            self._init_error = f"Tasks API failed: {exc}"
            logger.warning(f"[DriveGuardFaceLandmarker] Tasks API initialization failed ({exc}). Attempting legacy FaceMesh...")
            if hasattr(self._mp, "solutions") and hasattr(self._mp.solutions, "face_mesh"):
                try:
                    self._legacy_face_mesh = self._mp.solutions.face_mesh.FaceMesh(
                        static_image_mode=(self.running_mode != "VIDEO"),
                        max_num_faces=1,
                        refine_landmarks=True,
                        min_detection_confidence=0.50,
                        min_tracking_confidence=0.50,
                    )
                    self._use_tasks_api = False
                    self.active_implementation = "MEDIAPIPE_LEGACY_FACEMESH"
                    self._is_initialized = True
                    self._init_error = None
                    logger.info(f"[DriveGuardFaceLandmarker] Legacy FaceMesh initialized successfully in {self.running_mode} mode.")
                except Exception as legacy_exc:
                    self._init_error = f"Tasks API ({exc}) and Legacy FaceMesh ({legacy_exc}) failed"
                    logger.warning(f"[DriveGuardFaceLandmarker] Legacy FaceMesh failed ({legacy_exc}). Activating DEGRADED_UNAVAILABLE mode.")
                    self._use_fallback_tracker = True
                    self.active_implementation = "DEGRADED_UNAVAILABLE"
                    self._is_initialized = False
            else:
                self._use_fallback_tracker = True
                self.active_implementation = "DEGRADED_UNAVAILABLE"
                self._is_initialized = False
                logger.warning(f"[DriveGuardFaceLandmarker] FaceMesh solutions unavailable ({exc}). Activating DEGRADED_UNAVAILABLE mode.")

    def process_frame(
        self,
        frame_bgr: Optional[np.ndarray] = None,
        timestamp_ms: float = 0.0,
        neutral_pitch_deg: float = 0.0,
        neutral_yaw_deg: float = 0.0,
        reference_open_ear: float = 0.28,
        extract_crops: bool = True,
        *,
        frame_rgb: Optional[np.ndarray] = None,
        is_rgb: bool = False,
    ) -> Tuple[FrameGeometryFeatures, Optional[np.ndarray]]:
        """
        Process a video frame and return `(FrameGeometryFeatures, landmarks_pixel_xyz)`.
        Thread-safe and guarded with monotonic timestamp enforcement.

        EXP-1A RGB Pass-Through: Accepts either pre-decoded RGB directly via `frame_rgb`
        (eliminating redundant color conversions) or legacy BGR via `frame_bgr`.
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
                        tracking_mode=self.active_implementation,
                    ),
                    None,
                )

            # EXP-1A RGB Pass-Through: bypass cv2.cvtColor if frame_rgb or is_rgb is supplied
            if frame_rgb is not None:
                rgb = frame_rgb
            elif frame_bgr is not None:
                rgb = frame_bgr if is_rgb else cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            else:
                raise ValueError("Either frame_rgb or frame_bgr must be provided to process_frame.")

            h, w = rgb.shape[:2]

            if getattr(self, "_use_fallback_tracker", False) or self.active_implementation == "DEGRADED_UNAVAILABLE":
                return self._process_frame_headless_fallback(
                    frame_rgb=rgb,
                    timestamp_ms=timestamp_ms,
                    neutral_pitch_deg=neutral_pitch_deg,
                    neutral_yaw_deg=neutral_yaw_deg,
                    reference_open_ear=reference_open_ear,
                    frame_bgr=frame_bgr,
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
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)

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
                result = self._legacy_face_mesh.process(rgb)
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
                        left_eye_crop=None,
                        right_eye_crop=None,
                        tracking_mode=self.active_implementation,
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
                    rgb, landmarks_pixel_xyz, LEFT_EYE_INDICES
                )
                right_crop = extract_eye_crop_with_quality(
                    rgb, landmarks_pixel_xyz, RIGHT_EYE_INDICES
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
                tracking_mode=self.active_implementation,
            )
            return features, landmarks_pixel_xyz

    def _process_frame_headless_fallback(
        self,
        frame_rgb: np.ndarray,
        timestamp_ms: float,
        neutral_pitch_deg: float,
        neutral_yaw_deg: float,
        reference_open_ear: float,
        frame_bgr: Optional[np.ndarray] = None,
    ) -> Tuple[FrameGeometryFeatures, Optional[np.ndarray]]:
        """
        Explicit degraded/unavailable monitoring fallback when face tracking models are uninitialized.
        Does NOT report fabricated face detections, synthetic landmarks, or artificial confidence scores.
        """
        features = FrameGeometryFeatures(
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
            left_eye_crop=None,
            right_eye_crop=None,
            tracking_mode="DEGRADED_UNAVAILABLE",
        )
        return features, None

    def get_diagnostics(self) -> dict:
        """Return operational diagnostics for this landmarker instance."""
        with self._lock:
            return {
                "active_implementation": self.active_implementation,
                "running_mode": self.running_mode,
                "is_initialized": self._is_initialized,
                "initialization_error": self._init_error,
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
