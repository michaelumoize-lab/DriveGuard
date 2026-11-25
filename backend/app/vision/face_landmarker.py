"""
DriveGuard — Face Landmark Tracking Module.

1. What it does:
   Wraps Google's MediaPipe Face Landmarker Tasks API (`face_landmarker.task`)
   to detect driver faces and track 478 3D landmarks (face mesh, eye contours,
   and refined iris centers) in real time.

2. Why it exists:
   Decouples raw video ingestion from downstream physiological feature calculations
   (EAR, MAR, blink kinematics, and head pose). Provides a robust, crash-proof
   interface that enforces strict timestamp monotonicity and graceful face-loss
   state handling without ever making false drowsiness predictions.

3. What data it receives:
   - `FramePacket` instances (or raw NumPy BGR image frames with monotonic timestamps).

4. What data it produces:
   - Immutable `FaceLandmarkResult` dataclasses containing normalized 3D landmarks,
     presence confidence, detection status, inference latency, and diagnostic reasons.

5. Modules that depend on it:
   - `backend.app.features.ear` (Phase 5)
   - `backend.app.features.blink` (Phase 5)
   - `backend.app.features.mar` (Phase 5)
   - `backend.app.features.head_pose` (Phase 5)
   - `scripts.run_camera_test` (diagnostic HUD visualization)

6. Failure behavior:
   - Missing or corrupted model asset: Automatically downloads `face_landmarker.task`
     from Google Cloud Storage if not found on disk.
   - Lost face or occlusion: Emits `FaceLandmarkStatus.FACE_LOST_GRACE` during brief
     drops (< 0.4s), transitioning to `FaceLandmarkStatus.INSUFFICIENT_DATA` once
     the timeout is exceeded.
   - Non-monotonic timestamps: Automatically clamps and increments timestamps to
     prevent MediaPipe C++ runtime crashes.
"""

from __future__ import annotations

import logging
import math
import threading
import time
import urllib.request
import uuid
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

from backend.app.camera.base import FramePacket

logger = logging.getLogger("driveguard.vision.landmarker")

# Canonical MediaPipe Face Mesh Landmark Indices
LEFT_EYE_CONTOUR = (33, 160, 158, 133, 153, 144)
RIGHT_EYE_CONTOUR = (362, 385, 387, 263, 373, 380)

# Iris landmarks (requires refine_landmarks=True / face_landmarker.task bundle)
LEFT_IRIS_CENTER = 468
LEFT_IRIS_POINTS = (468, 469, 470, 471, 472)
RIGHT_IRIS_CENTER = 473
RIGHT_IRIS_POINTS = (473, 474, 475, 476, 477)

# Lips contour
LIPS_OUTER_CONTOUR = (
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78
)
LIPS_INNER_CONTOUR = (
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308,
    324, 318, 402, 317, 14, 87, 178, 88, 95
)

# 6-DoF Head Pose Canonical Anchor Points (used by SolvePnP in Phase 5)
POSE_ANCHORS = {
    "nose_tip": 1,
    "chin": 199,
    "left_eye_corner": 33,
    "right_eye_corner": 263,
    "left_mouth_corner": 61,
    "right_mouth_corner": 291,
}

# Official Google Cloud Storage URL for Face Landmarker Task Bundle
OFFICIAL_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)


class FaceLandmarkStatus(str, Enum):
    """Categorical classification of facial tracking quality."""
    TRACKING = "TRACKING"                    # Face locked with 478 3D landmarks
    OCCLUDED_OR_MARGINAL = "MARGINAL"        # Face present but marginal confidence (< 0.5)
    FACE_LOST_GRACE = "FACE_LOST_GRACE"      # Brief face loss within grace window (< grace_timeout)
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"  # Driver not visible or prolonged occlusion
    CAMERA_FAULT = "CAMERA_FAULT"            # Empty or invalid video frame


@dataclass(frozen=True)
class NormalizedLandmark:
    """
    3D normalized coordinate point.
    x, y are in range [0.0, 1.0] relative to frame width and height.
    z represents depth relative to the midpoint between the eyes.
    """
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class FaceLandmarkResult:
    """
    Structured, immutable result of real-time facial landmark localization.
    """
    has_face: bool
    landmarks: Tuple[NormalizedLandmark, ...]
    presence_confidence: float
    timestamp_ms: int
    status: FaceLandmarkStatus
    inference_latency_ms: float
    reasons: Tuple[str, ...]

    def get_landmark(self, index: int) -> Optional[NormalizedLandmark]:
        """Safely retrieve a landmark by index, or None if unavailable."""
        if self.has_face and 0 <= index < len(self.landmarks):
            return self.landmarks[index]
        return None

    def get_pixel_coords(self, index: int, frame_width: int, frame_height: int) -> Optional[Tuple[int, int]]:
        """Convert normalized [0.0, 1.0] coordinates to integer pixel (x, y)."""
        lm = self.get_landmark(index)
        if lm is None:
            return None
        px = int(np.clip(lm.x * frame_width, 0, frame_width - 1))
        py = int(np.clip(lm.y * frame_height, 0, frame_height - 1))
        return px, py


@dataclass(frozen=True)
class FaceLandmarkerConfig:
    """
    Runtime configuration for MediaPipe Face Landmarker.
    """
    model_path: Optional[Path] = None
    min_face_detection_confidence: float = 0.5
    min_face_presence_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    face_loss_grace_timeout_sec: float = 0.4
    delegate: str = "CPU"

    def __post_init__(self) -> None:
        """Validate landmarker confidence and timeout thresholds."""
        if not (0.0 < self.min_face_detection_confidence <= 1.0):
            raise ValueError("min_face_detection_confidence must be in range (0.0, 1.0]")
        if not (0.0 < self.min_face_presence_confidence <= 1.0):
            raise ValueError("min_face_presence_confidence must be in range (0.0, 1.0]")
        if not (0.0 < self.min_tracking_confidence <= 1.0):
            raise ValueError("min_tracking_confidence must be in range (0.0, 1.0]")
        if self.face_loss_grace_timeout_sec <= 0.0:
            raise ValueError("face_loss_grace_timeout_sec must be positive.")
        if self.delegate.upper() not in ("CPU", "GPU"):
            raise ValueError(f"delegate must be 'CPU' or 'GPU', got {self.delegate}")


class DriveGuardFaceLandmarker:
    """
    Thread-safe MediaPipe Face Landmarker wrapper optimized for local real-time driver monitoring.
    """

    def __init__(self, config: Optional[FaceLandmarkerConfig] = None) -> None:
        """
        Initialize the face landmarker adapter.
        
        Args:
            config: Optional FaceLandmarkerConfig instance (uses defaults if omitted).
        """
        self._config = config or FaceLandmarkerConfig()
        self._lock = threading.Lock()
        self._landmarker = None
        self._is_closed: bool = False

        # Monotonic timestamp tracking (prevents MediaPipe C++ crashes)
        self._last_timestamp_ms: Optional[int] = None
        self._duplicate_or_unordered_count: int = 0

        # Face-loss tracking state machine
        self._last_face_seen_ms: Optional[int] = None
        self._last_face_seen_time: Optional[float] = None
        self._total_inferences: int = 0
        self._total_face_hits: int = 0

        self._resolved_model_path = self._resolve_model_path(self._config.model_path)
        self._initialize_mediapipe()

    @property
    def active_implementation(self) -> str:
        """Human-readable name of the active MediaPipe implementation."""
        return "MEDIAPIPE_TASKS"

    def get_diagnostics(self) -> dict:
        """Return thread-safe runtime diagnostics for session monitoring and health checks."""
        with self._lock:
            return {
                "is_closed": self._is_closed,
                "running_mode": "VIDEO",
                "use_tasks_api": True,
                "active_implementation": self.active_implementation,
                "last_timestamp_ms": self._last_timestamp_ms,
                "duplicate_or_unordered_count": self._duplicate_or_unordered_count,
                "total_inferences": self._total_inferences,
                "total_face_hits": self._total_face_hits,
                "model_path": str(self._resolved_model_path),
            }

    @staticmethod
    def _resolve_model_path(user_path: Optional[Path]) -> Path:
        """
        Locate or download the official `face_landmarker.task` bundle.
        """
        candidate_paths: List[Path] = []
        if user_path is not None:
            candidate_paths.append(Path(user_path))

        # Standard repository search locations
        repo_root = Path(__file__).resolve().parents[3]
        candidate_paths.extend([
            repo_root / "backend" / "weights" / "face_landmarker.task",
            repo_root / "weights" / "face_landmarker.task",
            Path.cwd() / "backend" / "weights" / "face_landmarker.task",
            Path.cwd() / "weights" / "face_landmarker.task",
        ])

        for path in candidate_paths:
            if path.is_file() and path.stat().st_size >= 100_000:
                logger.info("Found Face Landmarker model at: %s", path)
                return path

        # If not found, download to standard weights directory atomically
        target_path = repo_root / "backend" / "weights" / "face_landmarker.task"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = target_path.with_name(f"{target_path.stem}.{uuid.uuid4().hex}.tmp")
        logger.info("Model not found locally. Downloading official MediaPipe Face Landmarker bundle to %s...", target_path)
        try:
            req = urllib.request.Request(
                OFFICIAL_MODEL_URL,
                headers={"User-Agent": "DriveGuard-ModelDownloader/1.0"},
            )
            with urllib.request.urlopen(req, timeout=30.0) as resp, open(temp_path, "wb") as out_f:
                while chunk := resp.read(65536):
                    out_f.write(chunk)

            if temp_path.stat().st_size < 100_000:
                raise ValueError("Downloaded model file is suspiciously small or corrupted.")

            temp_path.replace(target_path)
            logger.info("Model download complete (%d bytes).", target_path.stat().st_size)
            return target_path
        except Exception as exc:
            if temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass
            raise FileNotFoundError(
                f"Failed to locate or download face_landmarker.task model. "
                f"Please place 'face_landmarker.task' in '{target_path}'."
            ) from exc

    def _initialize_mediapipe(self) -> None:
        """
        Instantiate the MediaPipe Tasks FaceLandmarker in VIDEO running mode.
        """
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (
            FaceLandmarker,
            FaceLandmarkerOptions,
            RunningMode,
        )

        delegate_enum = (
            BaseOptions.Delegate.GPU
            if self._config.delegate.upper() == "GPU"
            else BaseOptions.Delegate.CPU
        )

        base_options = BaseOptions(
            model_asset_path=str(self._resolved_model_path),
            delegate=delegate_enum,
        )

        options = FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=self._config.min_face_detection_confidence,
            min_face_presence_confidence=self._config.min_face_presence_confidence,
            min_tracking_confidence=self._config.min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )

        self._landmarker = FaceLandmarker.create_from_options(options)
        self._mp = mp
        logger.info(
            "MediaPipe Face Landmarker initialized in VIDEO mode (delegate: %s, model: %s).",
            self._config.delegate.upper(),
            self._resolved_model_path.name,
        )

    @staticmethod
    def _estimate_landmark_confidence(raw_landmarks: Sequence) -> float:
        """
        Compute a non-hardcoded landmark quality & framing confidence in [0.0, 1.0].

        MediaPipe Tasks FaceLandmarker does not always populate scalar per-face presence
        scores in Python bindings, so we combine any populated native presence scores with
        geometric framing/scale health across 16 canonical facial anchor points.
        """
        if len(raw_landmarks) < 468:
            return 0.0

        anchor_indices = (1, 6, 10, 13, 14, 33, 61, 133, 145, 152, 159, 263, 291, 362, 374, 386)
        bounds_sum = 0.0
        native_presences: List[float] = []

        for idx in anchor_indices:
            lm = raw_landmarks[idx]
            x = float(lm.x)
            y = float(lm.y)
            if not (math.isfinite(x) and math.isfinite(y)):
                continue
            if 0.02 <= x <= 0.98 and 0.02 <= y <= 0.98:
                bounds_sum += 1.0
            elif 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0:
                bounds_sum += 0.75

            p = getattr(lm, "presence", None)
            if isinstance(p, (int, float)) and not isinstance(p, bool) and math.isfinite(p) and 0.0 < float(p) <= 1.0:
                native_presences.append(float(p))

        bounds_score = bounds_sum / float(len(anchor_indices))

        # Geometric scale checks (interocular span and vertical face height)
        lm_l = raw_landmarks[33]
        lm_r = raw_landmarks[263]
        lm_top = raw_landmarks[10]
        lm_bot = raw_landmarks[152]
        if not all(
            math.isfinite(v)
            for v in (lm_l.x, lm_l.y, lm_r.x, lm_r.y, lm_top.y, lm_bot.y)
        ):
            return 0.0

        iod = math.hypot(float(lm_r.x) - float(lm_l.x), float(lm_r.y) - float(lm_l.y))
        span_y = abs(float(lm_bot.y) - float(lm_top.y))
        scale_x = min(1.0, iod / 0.05)
        scale_y = min(1.0, span_y / 0.08)
        scale_score = min(scale_x, scale_y)

        native_factor = (
            (sum(native_presences) / float(len(native_presences)))
            if len(native_presences) == len(anchor_indices)
            else 1.0
        )
        return float(np.clip(bounds_score * scale_score * native_factor, 0.0, 1.0))

    def detect(
        self,
        input_data: Union[FramePacket, np.ndarray],
        timestamp_ms: Optional[int] = None,
        *,
        color_format: str = "BGR",
    ) -> FaceLandmarkResult:
        """
        Execute facial landmark localization on a frame.
        
        Args:
            input_data: A FramePacket object or raw BGR/RGB NumPy image array.
            timestamp_ms: Optional millisecond timestamp (extracted automatically if FramePacket is provided).
            color_format: Channel order ('BGR' default; 'RGB' bypasses BGR->RGB cv2.cvtColor).
            
        Returns:
            FaceLandmarkResult containing 478 3D landmarks, confidence, and status.
        """
        if self._is_closed or self._landmarker is None:
            raise RuntimeError("DriveGuardFaceLandmarker is closed or not initialized.")

        # Extract frame and timestamp
        if isinstance(input_data, FramePacket):
            bgr_frame = input_data.frame
            inferred_ms = int(input_data.timestamp_ns // 1_000_000)
        elif isinstance(input_data, np.ndarray):
            bgr_frame = input_data
            inferred_ms = timestamp_ms if timestamp_ms is not None else int(time.monotonic() * 1000)
        else:
            raise TypeError(f"Expected FramePacket or np.ndarray, got {type(input_data)}")

        if (
            bgr_frame is None
            or bgr_frame.size == 0
            or getattr(bgr_frame, "dtype", None) != np.uint8
            or len(getattr(bgr_frame, "shape", ())) != 3
            or bgr_frame.shape[2] != 3
        ):
            return FaceLandmarkResult(
                has_face=False,
                landmarks=(),
                presence_confidence=0.0,
                timestamp_ms=inferred_ms,
                status=FaceLandmarkStatus.CAMERA_FAULT,
                inference_latency_ms=0.0,
                reasons=("Empty or corrupted input video frame.",),
            )

        with self._lock:
            # Enforce strict timestamp monotonicity required by MediaPipe VIDEO mode
            if self._last_timestamp_ms is not None and inferred_ms <= self._last_timestamp_ms:
                self._duplicate_or_unordered_count += 1
                inferred_ms = self._last_timestamp_ms + 1
            self._last_timestamp_ms = inferred_ms

            t_start = time.perf_counter()

            try:
                # MediaPipe Tasks expects C-contiguous SRGB format; skip cv2.cvtColor when already RGB
                if color_format.upper() == "RGB":
                    rgb_frame = np.ascontiguousarray(bgr_frame)
                else:
                    rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb_frame)
                detection_result = self._landmarker.detect_for_video(mp_image, inferred_ms)
            except Exception as exc:
                logger.error("MediaPipe inference exception: %s", exc)
                return FaceLandmarkResult(
                    has_face=False,
                    landmarks=(),
                    presence_confidence=0.0,
                    timestamp_ms=inferred_ms,
                    status=FaceLandmarkStatus.CAMERA_FAULT,
                    inference_latency_ms=0.0,
                    reasons=(f"MediaPipe inference error: {exc}",),
                )

            latency_ms = (time.perf_counter() - t_start) * 1000.0
            self._total_inferences += 1
            now_sec = time.monotonic()

            # Check if at least one face was detected
            if detection_result.face_landmarks and len(detection_result.face_landmarks) > 0:
                raw_landmarks = detection_result.face_landmarks[0]
                landmarks_tuple = tuple(
                    NormalizedLandmark(x=lm.x, y=lm.y, z=lm.z) for lm in raw_landmarks
                )
                self._last_face_seen_ms = inferred_ms
                self._last_face_seen_time = now_sec
                self._total_face_hits += 1

                # Derive landmark quality & framing confidence (never hardcoded 1.0)
                confidence = self._estimate_landmark_confidence(raw_landmarks)

                if confidence < self._config.min_face_presence_confidence:
                    status = FaceLandmarkStatus.OCCLUDED_OR_MARGINAL
                    reasons = (f"Marginal face presence confidence ({confidence:.2f}).",)
                else:
                    status = FaceLandmarkStatus.TRACKING
                    reasons = ("Driver face locked and tracked.",)

                return FaceLandmarkResult(
                    has_face=True,
                    landmarks=landmarks_tuple,
                    presence_confidence=confidence,
                    timestamp_ms=inferred_ms,
                    status=status,
                    inference_latency_ms=latency_ms,
                    reasons=reasons,
                )

            # Face lost branch — evaluate grace state machine using frame timestamps
            # (with fallback to wall-clock monotonic time if only _last_face_seen_time was set)
            if self._last_face_seen_ms is not None:
                elapsed_lost_sec = max(0.0, (inferred_ms - self._last_face_seen_ms) / 1000.0)
            elif self._last_face_seen_time is not None:
                elapsed_lost_sec = max(0.0, now_sec - self._last_face_seen_time)
            else:
                elapsed_lost_sec = float("inf")

            if elapsed_lost_sec < self._config.face_loss_grace_timeout_sec:
                status = FaceLandmarkStatus.FACE_LOST_GRACE
                reasons = (
                    f"Face momentarily lost ({elapsed_lost_sec * 1000:.0f}ms < "
                    f"{self._config.face_loss_grace_timeout_sec * 1000:.0f}ms grace).",
                )
            else:
                status = FaceLandmarkStatus.INSUFFICIENT_DATA
                reasons = (
                    f"Driver face not detected for {elapsed_lost_sec:.1f}s "
                    f"(exceeds {self._config.face_loss_grace_timeout_sec:.1f}s threshold).",
                )

            return FaceLandmarkResult(
                has_face=False,
                landmarks=(),
                presence_confidence=0.0,
                timestamp_ms=inferred_ms,
                status=status,
                inference_latency_ms=latency_ms,
                reasons=reasons,
            )

    def draw_landmarks(
        self,
        frame: np.ndarray,
        result: FaceLandmarkResult,
        draw_mesh: bool = False,
        draw_eyes: bool = True,
        draw_irises: bool = True,
    ) -> np.ndarray:
        """
        Render facial landmark overlays directly onto a BGR frame.
        
        Args:
            frame: OpenCV BGR NumPy image array.
            result: FaceLandmarkResult returned by .detect().
            draw_mesh: If True, draws all 468 facial mesh points.
            draw_eyes: If True, draws left and right eye contours in cyan.
            draw_irises: If True, draws left and right iris centers in bright green.
            
        Returns:
            The annotated BGR frame.
        """
        if not result.has_face or len(result.landmarks) == 0:
            return frame.copy()

        canvas = frame.copy()
        h, w = canvas.shape[:2]

        # 1. Full Mesh Wireframe (optional toggle)
        if draw_mesh:
            for i, lm in enumerate(result.landmarks[:468]):
                px = int(lm.x * w)
                py = int(lm.y * h)
                cv2.circle(canvas, (px, py), 1, (180, 180, 180), -1, cv2.LINE_AA)

        # 2. Eye Contours
        if draw_eyes:
            for contour_indices, color in ((LEFT_EYE_CONTOUR, (255, 255, 0)), (RIGHT_EYE_CONTOUR, (255, 255, 0))):
                pts = [result.get_pixel_coords(idx, w, h) for idx in contour_indices]
                valid_pts = [p for p in pts if p is not None]
                if len(valid_pts) == len(contour_indices):
                    poly = np.array(valid_pts, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(canvas, [poly], isClosed=True, color=color, thickness=1, lineType=cv2.LINE_AA)
                    for px, py in valid_pts:
                        cv2.circle(canvas, (px, py), 2, (0, 200, 255), -1, cv2.LINE_AA)

        # 3. Iris Refinement Points
        if draw_irises and len(result.landmarks) >= 478:
            for iris_center_idx in (LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
                pt = result.get_pixel_coords(iris_center_idx, w, h)
                if pt is not None:
                    cv2.circle(canvas, pt, 3, (0, 255, 0), -1, cv2.LINE_AA)
                    cv2.circle(canvas, pt, 5, (0, 255, 0), 1, cv2.LINE_AA)

        return canvas

    def close(self) -> None:
        """Release underlying MediaPipe C++ runtime descriptors cleanly."""
        with self._lock:
            if not self._is_closed:
                if self._landmarker is not None:
                    try:
                        self._landmarker.close()
                    except Exception as exc:
                        logger.warning("Exception during MediaPipe landmarker close: %s", exc)
                    self._landmarker = None
                self._is_closed = True
                logger.info("DriveGuardFaceLandmarker closed cleanly.")

    def __enter__(self) -> DriveGuardFaceLandmarker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def ensure_face_landmarker_model(user_path: Optional[Path] = None) -> Path:
    """Verify or download the authoritative MediaPipe `face_landmarker.task` bundle."""
    return DriveGuardFaceLandmarker._resolve_model_path(user_path)
