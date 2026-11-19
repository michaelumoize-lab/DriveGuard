"""
DriveGuard — Session Recorder Subsystem.

Provides synchronized dual-stream session recording:
1. Native .mp4 video stream (ground truth at camera native resolution and FPS).
2. Millisecond-accurate per-frame physiological telemetry (.csv and .jsonl).
3. Session metadata JSON capturing configuration, driver profile, and performance metrics.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

logger = logging.getLogger("driveguard.recorder")

CSV_COLUMNS = [
    "frame_id",
    "timestamp_ms",
    "face_detected",
    "face_status",
    "presence_confidence",
    "pose_zone",
    "pitch_deg",
    "yaw_deg",
    "roll_deg",
    "rel_pitch_deg",
    "rel_yaw_deg",
    "rel_roll_deg",
    "ear_left",
    "ear_right",
    "ear_compensated",
    "ear_closure_threshold",
    "is_eye_closed",
    "is_geometric_valid",
    "mar",
    "is_yawning",
    "total_blinks",
    "total_yawns",
    "perclos_pct",
    "drowsiness_state",
    "drowsiness_severity",
    "primary_reason",
    "luminance",
    "blur_variance",
    "optical_status",
    "inference_latency_ms",
]


class SessionRecorder:
    """
    Manages synchronized video and telemetry recording for live driving or benchmarking sessions.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/recordings",
        session_id: Optional[str] = None,
        driver_id: str = "driver_default",
        frame_width: int = 1280,
        frame_height: int = 720,
        fps: float = 30.0,
        profile_metadata: Optional[Dict[str, Any]] = None,
        source_identifier: str = "default",
    ) -> None:
        self.base_dir = Path(base_dir)
        self.session_id = session_id or datetime.now(timezone.utc).strftime("session_%Y%m%d_%H%M%S")
        self.driver_id = driver_id
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.fps = max(1.0, float(fps))
        self.profile_metadata = profile_metadata or {}
        self.source_identifier = str(source_identifier)

        self.session_dir: Path = self.base_dir / self.session_id
        self.video_path: Path = self.session_dir / "stream.mp4"
        self.csv_path: Path = self.session_dir / "telemetry.csv"
        self.jsonl_path: Path = self.session_dir / "telemetry.jsonl"
        self.meta_path: Path = self.session_dir / "session_metadata.json"

        self._is_recording: bool = False
        self._frame_count: int = 0
        self._start_time: float = 0.0
        self._start_iso: str = ""

        self._video_writer: Optional[cv2.VideoWriter] = None
        self._csv_file = None
        self._csv_writer = None
        self._jsonl_file = None

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def elapsed_sec(self) -> float:
        if not self._is_recording:
            return 0.0
        return time.monotonic() - self._start_time

    def start(self) -> bool:
        """Initializes output directory, video writer, and telemetry files."""
        if self._is_recording:
            logger.warning("SessionRecorder already running.")
            return True

        try:
            self.session_dir.mkdir(parents=True, exist_ok=True)
            self._start_iso = datetime.now(timezone.utc).isoformat()
            self._start_time = time.monotonic()
            self._frame_count = 0

            # 1. Initialize Video Writer
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._video_writer = cv2.VideoWriter(
                str(self.video_path),
                fourcc,
                self.fps,
                (self.frame_width, self.frame_height),
            )
            if not self._video_writer.isOpened():
                # Fallback to AVI with XVID if mp4v fails
                logger.warning("Failed to open MP4 video writer. Falling back to XVID AVI.")
                self.video_path = self.session_dir / "stream.avi"
                fourcc_avi = cv2.VideoWriter_fourcc(*"XVID")
                self._video_writer = cv2.VideoWriter(
                    str(self.video_path),
                    fourcc_avi,
                    self.fps,
                    (self.frame_width, self.frame_height),
                )
                if not self._video_writer.isOpened():
                    logger.error("Could not initialize VideoWriter for session %s", self.session_id)
                    return False

            # 2. Initialize Telemetry CSV
            self._csv_file = open(self.csv_path, mode="w", newline="", encoding="utf-8")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=CSV_COLUMNS)
            self._csv_writer.writeheader()
            self._csv_file.flush()

            # 3. Initialize Telemetry JSONL
            self._jsonl_file = open(self.jsonl_path, mode="w", encoding="utf-8")

            # 4. Write Initial Metadata JSON
            meta = {
                "session_id": self.session_id,
                "driver_id": self.driver_id,
                "source_identifier": self.source_identifier,
                "created_at": self._start_iso,
                "target_resolution": [self.frame_width, self.frame_height],
                "target_fps": self.fps,
                "status": "RECORDING",
                "profile_metadata": self.profile_metadata,
            }
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(meta, f, indent=2)

            self._is_recording = True
            logger.info("Started session recording in: %s", self.session_dir)
            return True

        except Exception as e:
            logger.error("Failed to start session recording: %s", e)
            self.stop()
            return False

    def record_frame(
        self,
        frame: np.ndarray,
        timestamp_ms: float,
        telemetry: Dict[str, Any],
    ) -> None:
        """Records a video frame and its matching telemetry row."""
        if not self._is_recording:
            return

        self._frame_count += 1
        frame_id = self._frame_count

        # 1. Write video frame
        if self._video_writer is not None:
            # Ensure frame size matches video writer dimensions
            if (frame.shape[1], frame.shape[0]) != (self.frame_width, self.frame_height):
                f_resized = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                self._video_writer.write(f_resized)
            else:
                self._video_writer.write(frame)

        # 2. Write CSV row
        if self._csv_writer is not None and self._csv_file is not None:
            row: Dict[str, Any] = {"frame_id": frame_id, "timestamp_ms": round(float(timestamp_ms), 2)}
            for col in CSV_COLUMNS:
                if col in ("frame_id", "timestamp_ms"):
                    continue
                val = telemetry.get(col, "")
                if isinstance(val, float):
                    row[col] = round(val, 4)
                else:
                    row[col] = val
            self._csv_writer.writerow(row)
            if frame_id % 30 == 0:
                self._csv_file.flush()

        # 3. Write JSONL line
        if self._jsonl_file is not None:
            json_data = dict(telemetry)
            json_data["frame_id"] = frame_id
            json_data["timestamp_ms"] = round(float(timestamp_ms), 2)
            self._jsonl_file.write(json.dumps(json_data) + "\n")
            if frame_id % 30 == 0:
                self._jsonl_file.flush()

    def stop(self) -> Dict[str, Any]:
        """Finalizes and closes all file handles, updating session metadata."""
        if not self._is_recording and self._video_writer is None:
            return {}

        self._is_recording = False
        elapsed = time.monotonic() - self._start_time if self._start_time > 0 else 0.0
        actual_fps = (self._frame_count / elapsed) if elapsed > 0 else 0.0
        end_iso = datetime.now(timezone.utc).isoformat()

        # Close Video Writer
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception as e:
                logger.warning("Error releasing VideoWriter: %s", e)
            self._video_writer = None

        # Close CSV File
        if self._csv_file is not None:
            try:
                self._csv_file.flush()
                self._csv_file.close()
            except Exception as e:
                logger.warning("Error closing CSV file: %s", e)
            self._csv_file = None
            self._csv_writer = None

        # Close JSONL File
        if self._jsonl_file is not None:
            try:
                self._jsonl_file.flush()
                self._jsonl_file.close()
            except Exception as e:
                logger.warning("Error closing JSONL file: %s", e)
            self._jsonl_file = None

        # Update metadata JSON
        summary = {
            "session_id": self.session_id,
            "driver_id": self.driver_id,
            "source_identifier": self.source_identifier,
            "created_at": self._start_iso,
            "completed_at": end_iso,
            "duration_sec": round(elapsed, 2),
            "total_frames": self._frame_count,
            "average_fps": round(actual_fps, 2),
            "video_file": self.video_path.name,
            "csv_file": self.csv_path.name,
            "jsonl_file": self.jsonl_path.name,
            "status": "COMPLETED",
            "profile_metadata": self.profile_metadata,
        }
        try:
            with open(self.meta_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)
        except Exception as e:
            logger.warning("Failed to write final session metadata: %s", e)

        logger.info(
            "Session recording completed: %d frames (%.1fs, %.1f FPS) saved in %s",
            self._frame_count,
            elapsed,
            actual_fps,
            self.session_dir,
        )
        return summary
