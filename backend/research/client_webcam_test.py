"""
DriveGuard — Live WebSocket Webcam Client Test & Diagnostic HUD
================================================================
Streams live webcam frames to the FastAPI WebSocket server
(`ws://127.0.0.1:8000/ws/v1/session/{session_id}`) and displays real-time
adaptive AI telemetry with an interactive OpenCV cockpit HUD.

Features:
  - Real-time 4-Zone Pose-Conditioned Gaze & Personalized Baselines (PDBP)
  - Sub-frame Kinematic Eye Closure & Rolling 60s PERCLOS
  - Optalert Amplitude-to-Velocity Ratio (AVR) with >35.0 Drowsiness Warning
  - Coefficient of Variation of Inter-Blink Intervals (CV_IBI) Autonomic Fatigue
  - Euro NCAP 2026 Gaze Scanpath Entropy (H_t) & Cabin Transition Tracking
  - Corrugator Supercilii (AU4) Brow Lowerer Facial Strain Detection
  - Authentic Yawn Temporal Envelope Filtering (Anti-Speech False Alarm)
  - Confidence-Gated Cascade Decision (Geometry vs. MobileNetV3)
  - Live Side-by-Side Comparison (Conventional Static vs. DriveGuard Adaptive)
  - Dynamic Resolution Presets (640x480, 1280x720) with Hardware Negotiation Query
  - Latency Waterfall Breakdown (Queue, Inference, Server Total, Client RTT)
  - Headless Automated Self-Test Verification Mode (--test-frames N / --verify)

Interactive Keyboard Controls:
  'c' -> Start 4-Zone Calibration Wizard
  'm' -> Toggle Audio Alert Buzzer (Mute / Unmute)
  '1' -> Switch to Experiment A (Geometric EAR Only)
  '2' -> Switch to Experiment B (MobileNetV3 CNN Always)
  '3' -> Switch to Experiment C (Confidence-Gated Cascade)
  's' -> Save Calibrated Driver Profile to Server
  't' -> Toggle Latency Waterfall & Timing Breakdown HUD
  'e' -> Toggle Advanced DMS Metrics Panel (Euro NCAP / Optalert)
  'f' -> Toggle Horizontal Mirror Flip
  'q' or ESC -> Quit

Usage:
    python backend/research/client_webcam_test.py
    python backend/research/client_webcam_test.py --camera 0 --fps 25 --resolution 640x480
    python backend/research/client_webcam_test.py --camera 0 --fps 30 --resolution 1280x720
    python backend/research/client_webcam_test.py --synthetic --fps 25
    python backend/research/client_webcam_test.py --verify
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root in sys.path
ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

VENV_PYTHON_WIN = ROOT_DIR / ".venv" / "Scripts" / "python.exe"
VENV_PYTHON_POSIX = ROOT_DIR / ".venv" / "bin" / "python"


def is_running_in_project_venv() -> bool:
    try:
        active_exec = Path(sys.executable).resolve()
        venv_dir = (ROOT_DIR / ".venv").resolve()
        return venv_dir in active_exec.parents or active_exec.is_relative_to(venv_dir)
    except (ValueError, AttributeError):
        return str(ROOT_DIR / ".venv") in str(sys.executable)


def auto_switch_to_venv_if_needed() -> None:
    if is_running_in_project_venv():
        return
    target_python = VENV_PYTHON_WIN if os.name == "nt" else VENV_PYTHON_POSIX
    if target_python.is_file():
        cmd = [str(target_python), str(Path(__file__).resolve())] + sys.argv[1:]
        ret = subprocess.call(cmd)
        sys.exit(ret)


# Auto-switch to virtual environment before importing heavy dependencies
auto_switch_to_venv_if_needed()

import cv2
import numpy as np


class NonBlockingAudioAlarm:
    """
    Non-blocking background audio alarm synthesizer for real-time driver fatigue alerts.
    Dispatches beeps in a daemon worker thread so webcam capture never blocks.
    """

    def __init__(self, enabled: bool = True) -> None:
        self.enabled: bool = enabled
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        self._thread: threading.Thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()
        self._last_critical_time: float = 0.0
        self._last_warning_time: float = 0.0

    def play_critical(self) -> None:
        """Trigger urgent 2-tone alarm (880Hz -> 660Hz) with cooldown."""
        if not self.enabled:
            return
        now = time.time()
        if (now - self._last_critical_time) < 1.3:
            return
        self._last_critical_time = now
        try:
            self._queue.put_nowait("CRITICAL")
        except queue.Full:
            pass

    def play_warning(self) -> None:
        """Trigger gentle advisory alert tone (520Hz) with cooldown."""
        if not self.enabled:
            return
        now = time.time()
        if (now - self._last_warning_time) < 2.5:
            return
        self._last_warning_time = now
        try:
            self._queue.put_nowait("WARNING")
        except queue.Full:
            pass

    def play_chime(self, freq: int = 1100, dur_ms: int = 90) -> None:
        """Short affirmative notification chime (e.g. mode change or profile saved)."""
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(("CHIME", freq, dur_ms))
        except queue.Full:
            pass

    def _worker(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if os.name == "nt":
                    import winsound

                    if item == "CRITICAL":
                        winsound.Beep(880, 160)
                        time.sleep(0.04)
                        winsound.Beep(660, 200)
                    elif item == "WARNING":
                        winsound.Beep(520, 150)
                    elif isinstance(item, tuple) and item[0] == "CHIME":
                        winsound.Beep(item[1], item[2])
                else:
                    sys.stdout.write("\a")
                    sys.stdout.flush()
            except Exception:
                pass
            finally:
                self._queue.task_done()


class LiveWebcamHUDClient:
    """Manages OpenCV webcam capture, WebSocket streaming, and cockpit HUD rendering."""

    def __init__(
        self,
        server_ws_url: str,
        session_id: str = "webcam_live_demo",
        camera_idx: int = 0,
        target_fps: int = 25,
        width: int = 640,
        height: int = 480,
        jpeg_quality: int = 70,
        synthetic: bool = False,
        audio_enabled: bool = True,
        mirror: bool = True,
        show_diagnostics: bool = False,
        show_advanced_metrics: bool = True,
        test_frames: Optional[int] = None,
        headless: bool = False,
    ) -> None:
        self.server_ws_url = server_ws_url
        self.session_id = session_id
        self.camera_idx = camera_idx
        self.target_fps = target_fps
        self.width = width
        self.height = height
        self.jpeg_quality = jpeg_quality
        self.synthetic = synthetic
        self.mirror = mirror
        self.show_diagnostics = show_diagnostics
        self.show_advanced_metrics = show_advanced_metrics
        self.test_frames = test_frames
        self.headless = headless

        # Negotiated hardware properties
        self.actual_width: int = width
        self.actual_height: int = height
        self.actual_camera_fps: float = float(target_fps)

        # Non-blocking audio synthesizer
        self.audio_alarm = NonBlockingAudioAlarm(enabled=audio_enabled)

        # State storage from server
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_calibration: Optional[Dict[str, Any]] = None
        self.connected: bool = False
        self.running: bool = True
        self.last_roundtrip_ms: float = 0.0

        # Performance counters
        self.total_frames_sent: int = 0
        self.total_telemetry_received: int = 0
        self.rtt_history_ms: List[float] = []

        # UI Notification Banner
        self.notification_msg: str = "Connected to DriveGuard Backend. Press 'c' to Calibrate."
        self.notification_expiry: float = time.time() + 5.0

    def generate_synthetic_frame(self, frame_idx: int) -> np.ndarray:
        """Create an animated driver face canvas dynamically sized to target resolution."""
        w, h = self.width, self.height
        frame = np.full((h, w, 3), 28, dtype=np.uint8)

        cx, cy = w // 2, int(h * 0.48)
        face_rx = int(w * 0.17)
        face_ry = int(h * 0.31)

        # Draw driver face oval
        cv2.ellipse(frame, (cx, cy), (face_rx, face_ry), 0, 0, 360, (180, 150, 120), -1)
        cv2.ellipse(frame, (cx, cy), (face_rx, face_ry), 0, 0, 360, (220, 190, 160), 2)

        # Periodic blink simulation (closed for 6 frames every 75 frames)
        is_blink = (frame_idx % 75) < 6
        eye_h = max(2, int(h * 0.005 if is_blink else h * 0.028))
        eye_rx = int(w * 0.034)
        eye_offset_x = int(w * 0.070)
        eye_y = int(cy - h * 0.05)

        # Left Eye
        cv2.ellipse(frame, (cx - eye_offset_x, eye_y), (eye_rx, eye_h), 0, 0, 360, (255, 255, 255), -1)
        if not is_blink:
            cv2.circle(frame, (cx - eye_offset_x, eye_y), max(3, int(w * 0.011)), (60, 40, 20), -1)

        # Right Eye
        cv2.ellipse(frame, (cx + eye_offset_x, eye_y), (eye_rx, eye_h), 0, 0, 360, (255, 255, 255), -1)
        if not is_blink:
            cv2.circle(frame, (cx + eye_offset_x, eye_y), max(3, int(w * 0.011)), (60, 40, 20), -1)

        # Eyebrows (simulating AU4 brow furrow slightly when blink approaches)
        brow_y = eye_y - int(h * 0.04)
        cv2.line(frame, (cx - eye_offset_x - eye_rx, brow_y), (cx - eye_offset_x + eye_rx, brow_y - 2), (70, 50, 40), 2)
        cv2.line(frame, (cx + eye_offset_x - eye_rx, brow_y - 2), (cx + eye_offset_x + eye_rx, brow_y), (70, 50, 40), 2)

        # Mouth (with subtle movement)
        mouth_y = cy + int(h * 0.14)
        mouth_rx = int(w * 0.055)
        mouth_ry = int(h * 0.020)
        cv2.ellipse(frame, (cx, mouth_y), (mouth_rx, mouth_ry), 0, 0, 360, (120, 70, 70), -1)

        return frame

    def set_notification(self, msg: str, duration_sec: float = 3.5) -> None:
        self.notification_msg = msg
        self.notification_expiry = time.time() + duration_sec

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render informative real-time cockpit HUD overlays onto the webcam frame."""
        h, w = frame.shape[:2]
        hud = frame.copy()

        # 1. Semi-transparent top dashboard bar & bottom control bar
        overlay = hud.copy()
        top_bar_h = 75
        cv2.rectangle(overlay, (0, 0), (w, top_bar_h), (14, 16, 20), -1)
        cv2.rectangle(overlay, (0, h - 34), (w, h), (14, 16, 20), -1)
        cv2.addWeighted(overlay, 0.82, hud, 0.18, 0, hud)

        # 2. Title Banner
        cv2.putText(
            hud,
            "DRIVEGUARD | COCKPIT HUD",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.56,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        t = self.latest_telemetry
        cal = self.latest_calibration

        # Hardware Resolution & Mirror Tag
        res_text = f"CAM: {self.actual_width}x{self.actual_height}"
        cv2.putText(
            hud,
            res_text,
            (240, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (180, 190, 205),
            1,
            cv2.LINE_AA,
        )

        # Audio Alarm Status Pill
        audio_on = self.audio_alarm.enabled
        aud_color = (0, 220, 100) if audio_on else (120, 120, 130)
        aud_text = "VOL: ON" if audio_on else "VOL: OFF"
        cv2.rectangle(hud, (380, 7), (455, 27), (28, 30, 38), -1)
        cv2.rectangle(hud, (380, 7), (455, 27), aud_color, 1)
        cv2.putText(
            hud,
            aud_text,
            (387, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            aud_color,
            1,
            cv2.LINE_AA,
        )

        # Connection & Latency Info
        conn_text = "WS: ONLINE" if self.connected else "WS: CONNECTING..."
        conn_color = (0, 220, 0) if self.connected else (0, 165, 255)
        cv2.putText(
            hud,
            conn_text,
            (w - 180, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            conn_color,
            1,
            cv2.LINE_AA,
        )

        # Subtle Central Driver Framing Reticle
        cx, cy = w // 2, h // 2
        reticle_col = (0, 220, 100) if (t and t.get("face_detected")) else (80, 80, 90)
        ret_w = max(55, int(w * 0.10))
        ret_h = max(65, int(h * 0.15))
        cv2.line(hud, (cx - ret_w, cy - ret_h), (cx - ret_w + 14, cy - ret_h), reticle_col, 2)
        cv2.line(hud, (cx - ret_w, cy - ret_h), (cx - ret_w, cy - ret_h + 14), reticle_col, 2)
        cv2.line(hud, (cx + ret_w, cy - ret_h), (cx + ret_w - 14, cy - ret_h), reticle_col, 2)
        cv2.line(hud, (cx + ret_w, cy - ret_h), (cx + ret_w, cy - ret_h + 14), reticle_col, 2)
        cv2.line(hud, (cx - ret_w, cy + ret_h), (cx - ret_w + 14, cy + ret_h), reticle_col, 2)
        cv2.line(hud, (cx - ret_w, cy + ret_h), (cx - ret_w, cy + ret_h - 14), reticle_col, 2)
        cv2.line(hud, (cx + ret_w, cy + ret_h), (cx + ret_w - 14, cy + ret_h), reticle_col, 2)
        cv2.line(hud, (cx + ret_w, cy + ret_h), (cx + ret_w, cy + ret_h - 14), reticle_col, 2)
        cv2.line(hud, (cx - 6, cy), (cx + 6, cy), reticle_col, 1)
        cv2.line(hud, (cx, cy - 6), (cx, cy + 6), reticle_col, 1)

        if t is not None:
            # 1. Alert State Badge
            state = t.get("state", "NORMAL")
            state_colors = {
                "NORMAL": (0, 200, 0),
                "WARNING": (0, 215, 255),
                "CRITICAL": (0, 0, 255),
                "MONITORING_UNAVAILABLE": (128, 128, 128),
            }
            badge_color = state_colors.get(state, (0, 200, 0))

            cv2.rectangle(hud, (14, 34), (145, 66), badge_color, -1)
            cv2.putText(
                hud,
                state[:12],
                (20, 56),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 0, 0) if state in ("NORMAL", "WARNING") else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 2. Gaze Zone Pill
            zone = t.get("pose_zone", "CENTER")
            zone_color = (255, 180, 50) if zone != "CENTER" else (220, 220, 220)
            cv2.rectangle(hud, (155, 34), (320, 66), (24, 26, 33), -1)
            cv2.rectangle(hud, (155, 34), (320, 66), (65, 70, 80), 1)
            cv2.putText(
                hud,
                f"ZONE: {zone}",
                (165, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                zone_color,
                2,
                cv2.LINE_AA,
            )

            # 3. Mirror Check Protection Status
            comp = t.get("comparison", {})
            if comp.get("is_mirror_check_suppressed", False):
                cv2.rectangle(hud, (330, 34), (495, 66), (255, 140, 0), -1)
                cv2.putText(
                    hud,
                    "MIRROR PROTECTED",
                    (338, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.42,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif comp.get("fixed_false_alarm", False):
                cv2.rectangle(hud, (330, 34), (495, 66), (0, 0, 220), -1)
                cv2.putText(
                    hud,
                    "STATIC: FALSE ALARM!",
                    (336, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.40,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 4. Latency & FPS
            fps = t.get("fps", 0.0)
            inf_ms = t.get("inference_latency_ms", 0.0)
            cv2.putText(
                hud,
                f"{fps:.1f} FPS | Infer: {inf_ms:.1f}ms",
                (w - 210, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                f"RTT: {self.last_roundtrip_ms:.0f} ms",
                (w - 210, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 220, 150) if self.last_roundtrip_ms < 40 else (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

            # -------------------------------------------------------------
            # 5. LEFT COCKPIT PANEL (EYE KINEMATICS & ADVANCED METRICS)
            # -------------------------------------------------------------
            l_x, l_y = 12, 82
            l_w = min(230, int(w * 0.34))
            l_h = h - l_y - 42
            l_overlay = hud.copy()
            cv2.rectangle(l_overlay, (l_x, l_y), (l_x + l_w, l_y + l_h), (12, 14, 18), -1)
            cv2.addWeighted(l_overlay, 0.84, hud, 0.16, 0, hud)
            cv2.rectangle(hud, (l_x, l_y), (l_x + l_w, l_y + l_h), (50, 55, 65), 1)

            cv2.putText(hud, "EYE & BLINK METRICS", (l_x + 10, l_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 225, 255), 1, cv2.LINE_AA)

            eye = t.get("eye_metrics", {})
            ear = eye.get("ear_raw", 0.0)
            thresh = eye.get("ear_closure_threshold", 0.22)
            perclos = eye.get("perclos_pct", 0.0)
            is_closed = eye.get("is_eye_closed", False)

            # EAR Section
            cv2.putText(hud, f"EAR: {ear:.3f} (Thresh: {thresh:.3f})", (l_x + 10, l_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1)
            bar_w = l_w - 24
            fill_w = int(np.clip(ear / 0.45, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (l_x + 10, l_y + 43), (l_x + 10 + bar_w, l_y + 51), (35, 38, 45), -1)
            ear_col = (0, 0, 255) if is_closed else (0, 220, 0)
            cv2.rectangle(hud, (l_x + 10, l_y + 43), (l_x + 10 + fill_w, l_y + 51), ear_col, -1)
            th_x = l_x + 10 + int(np.clip(thresh / 0.45, 0.0, 1.0) * bar_w)
            cv2.line(hud, (th_x, l_y + 41), (th_x, l_y + 53), (0, 240, 255), 2)

            # PERCLOS Section
            cv2.putText(hud, f"PERCLOS (60s): {perclos:.1f}%", (l_x + 10, l_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (230, 230, 230), 1)
            p_fill_w = int(np.clip(perclos / 100.0, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (l_x + 10, l_y + 75), (l_x + 10 + bar_w, l_y + 83), (35, 38, 45), -1)
            p_col = (0, 220, 0) if perclos < 15 else ((0, 215, 255) if perclos < 25 else (0, 0, 255))
            cv2.rectangle(hud, (l_x + 10, l_y + 75), (l_x + 10 + p_fill_w, l_y + 83), p_col, -1)

            # Kinematics Detail
            blink_dur = eye.get("blink_duration_ms")
            dur_str = f"{blink_dur:.0f} ms" if blink_dur else "--"
            rcr = eye.get("blink_rcr")
            rcr_str = f"{rcr:.2f}" if rcr else "--"
            rate = eye.get("blink_rate_per_min", 0.0)

            cv2.putText(hud, f"Blink Dur:   {dur_str}", (l_x + 10, l_y + 104), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)
            cv2.putText(hud, f"Kinematic RCR: {rcr_str}", (l_x + 10, l_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)
            cv2.putText(hud, f"Blink Freq:  {rate:.1f} /min", (l_x + 10, l_y + 136), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)

            # ADVANCED METRICS: Optalert AVR & CV_IBI
            avr = eye.get("blink_avr")
            cv_ibi = eye.get("blink_cv_ibi")
            avr_str = f"{avr:.1f}" if avr is not None else "--"
            cv_ibi_str = f"{cv_ibi:.2f}" if cv_ibi is not None else "--"

            avr_color = (0, 0, 255) if (avr is not None and avr > 35.0) else (190, 230, 200)
            cv2.putText(hud, f"Optalert AVR:  {avr_str} {'>35.0 [HIGH]' if (avr and avr > 35.0) else ''}", (l_x + 10, l_y + 154), cv2.FONT_HERSHEY_SIMPLEX, 0.36, avr_color, 1)
            cv2.putText(hud, f"Autonomic CV_IBI: {cv_ibi_str}", (l_x + 10, l_y + 170), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)

            # Sustained Eye Closure & Classification
            closed_dur = eye.get("eye_closure_duration_ms", 0.0)
            b_class = eye.get("blink_classification", "NORMAL")
            class_col = (0, 0, 255) if "MICROSLEEP" in b_class else ((0, 215, 255) if "PROLONGED" in b_class else (180, 220, 180))
            cv2.putText(hud, f"Blink Class: {b_class[:16]}", (l_x + 10, l_y + 188), cv2.FONT_HERSHEY_SIMPLEX, 0.36, class_col, 1)
            if is_closed and closed_dur > 0:
                cv2.putText(hud, f"Closure Dur: {closed_dur:.0f} ms", (l_x + 10, l_y + 204), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (0, 100, 255), 1)

            # Cascade Decision Section
            cv2.line(hud, (l_x + 10, l_y + 218), (l_x + l_w - 10, l_y + 218), (40, 45, 55), 1)
            casc = t.get("cascade", {})
            cnn_on = casc.get("cnn_triggered", False)
            winner = casc.get("arbitration_winner", "GEOMETRY")
            mode = casc.get("experiment_mode", "EXP_C")

            cv2.putText(hud, "CASCADE GATE", (l_x + 10, l_y + 236), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 225, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Mode: {mode.split('_')[-1]}", (l_x + 10, l_y + 254), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)
            gate_col = (0, 220, 255) if cnn_on else (140, 140, 140)
            cv2.putText(hud, f"CNN: {'INVOKED' if cnn_on else 'BYPASS (Saved)'}", (l_x + 10, l_y + 270), cv2.FONT_HERSHEY_SIMPLEX, 0.36, gate_col, 1)
            cv2.putText(hud, f"Decision: {winner}", (l_x + 10, l_y + 286), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)

            # -------------------------------------------------------------
            # 6. RIGHT COCKPIT PANEL (HEAD POSE, EURO NCAP & RISK)
            # -------------------------------------------------------------
            r_w = min(230, int(w * 0.34))
            r_x = w - r_w - 12
            r_y = 82
            r_h = h - r_y - 42
            r_overlay = hud.copy()
            cv2.rectangle(r_overlay, (r_x, r_y), (r_x + r_w, r_y + r_h), (12, 14, 18), -1)
            cv2.addWeighted(r_overlay, 0.84, hud, 0.16, 0, hud)
            cv2.rectangle(hud, (r_x, r_y), (r_x + r_w, r_y + r_h), (50, 55, 65), 1)

            cv2.putText(hud, "HEAD POSE & GAZE", (r_x + 10, r_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 225, 255), 1, cv2.LINE_AA)

            hp = t.get("head_pose", {})
            pitch = hp.get("pitch_deg", 0.0)
            yaw = hp.get("yaw_deg", 0.0)
            roll = hp.get("roll_deg", 0.0)

            pitch_status = "NOD" if pitch <= -12.0 else ("UP" if pitch >= 14.0 else "LEVEL")
            yaw_status = "RIGHT" if yaw <= -12.0 else ("LEFT" if yaw >= 12.0 else "CTR")

            cv2.putText(hud, f"Pitch (P): {pitch:+5.1f} [{pitch_status}]", (r_x + 10, r_y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
            cv2.putText(hud, f"Yaw   (Y): {yaw:+5.1f} [{yaw_status}]", (r_x + 10, r_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (220, 220, 220), 1)
            cv2.putText(hud, f"Roll  (R): {roll:+5.1f} deg", (r_x + 10, r_y + 70), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (200, 200, 200), 1)

            # Horizontal Interactive Yaw Balance Meter
            cv2.putText(hud, "L <--  ROTATION METER  --> R", (r_x + 10, r_y + 90), cv2.FONT_HERSHEY_SIMPLEX, 0.30, (150, 150, 160), 1)
            r_bar_w = r_w - 24
            cv2.rectangle(hud, (r_x + 10, r_y + 96), (r_x + 10 + r_bar_w, r_y + 104), (35, 38, 45), -1)
            cv2.line(hud, (r_x + 10 + r_bar_w // 2, r_y + 94), (r_x + 10 + r_bar_w // 2, r_y + 106), (120, 120, 120), 1)
            yaw_ratio = np.clip((-yaw + 45.0) / 90.0, 0.0, 1.0)
            yaw_cursor_x = r_x + 10 + int(yaw_ratio * r_bar_w)
            cursor_col = (0, 220, 255) if abs(yaw) < 12.0 else (255, 180, 50)
            cv2.circle(hud, (yaw_cursor_x, r_y + 100), 4, cursor_col, -1)

            # ADVANCED METRICS: Euro NCAP 2026 Gaze Entropy & AU4 Strain
            gaze_h_t = eye.get("gaze_entropy")
            au4_strain = eye.get("brow_furrow_au4")
            yawn_conf = eye.get("yawn_envelope_confidence")

            cv2.line(hud, (r_x + 10, r_y + 116), (r_x + r_w - 10, r_y + 116), (40, 45, 55), 1)
            cv2.putText(hud, "EURO NCAP & BIOMETRICS", (r_x + 10, r_y + 132), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 225, 255), 1, cv2.LINE_AA)

            ht_str = f"{gaze_h_t:.2f} bits" if gaze_h_t is not None else "--"
            ht_col = (0, 215, 255) if (gaze_h_t and (gaze_h_t < 0.20 or gaze_h_t > 1.60)) else (190, 230, 200)
            cv2.putText(hud, f"Gaze Entropy H_t: {ht_str}", (r_x + 10, r_y + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.36, ht_col, 1)

            au4_str = f"{au4_strain:.2f}" if au4_strain is not None else "--"
            cv2.putText(hud, f"AU4 Brow Strain:  {au4_str}", (r_x + 10, r_y + 166), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)

            yawn_str = f"{yawn_conf * 100:.0f}%" if yawn_conf is not None else "--"
            cv2.putText(hud, f"Authentic Yawn:   {yawn_str}", (r_x + 10, r_y + 182), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (190, 190, 190), 1)

            # Fatigue Risk Score
            cv2.line(hud, (r_x + 10, r_y + 196), (r_x + r_w - 10, r_y + 196), (40, 45, 55), 1)
            risk = t.get("risk_score", 0.0)
            cv2.putText(hud, f"FATIGUE RISK: {risk * 100:.0f}%", (r_x + 10, r_y + 214), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 225, 255), 1, cv2.LINE_AA)
            r_fill_w = int(np.clip(risk, 0.0, 1.0) * r_bar_w)
            cv2.rectangle(hud, (r_x + 10, r_y + 220), (r_x + 10 + r_bar_w, r_y + 228), (35, 38, 45), -1)
            r_col = (0, 200, 0) if risk < 0.35 else ((0, 215, 255) if risk < 0.65 else (0, 0, 255))
            cv2.rectangle(hud, (r_x + 10, r_y + 220), (r_x + 10 + r_fill_w, r_y + 228), r_col, -1)

            # Active Explainable Reason Codes
            reasons = t.get("reason_codes", [])
            cv2.putText(hud, "Explainable Rationale:", (r_x + 10, r_y + 248), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (160, 160, 170), 1)
            if reasons:
                for idx, r_code in enumerate(reasons[:3]):
                    short_r = r_code.replace("PROLONGED_", "").replace("SUSTAINED_", "").replace("_", " ")[:20]
                    cv2.putText(hud, f"- {short_r}", (r_x + 10, r_y + 266 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 220, 255), 1)
            else:
                cv2.putText(hud, "- ALERT (BASELINE NORMAL)", (r_x + 10, r_y + 266), cv2.FONT_HERSHEY_SIMPLEX, 0.33, (0, 200, 0), 1)

            # -------------------------------------------------------------
            # 7. OPTIONAL LATENCY WATERFALL BREAKDOWN (TOGGLED VIA 't')
            # -------------------------------------------------------------
            if self.show_diagnostics:
                timing = t.get("timing") or {}
                q_ms = timing.get("queue_latency_ms", 0.0)
                inf_d_ms = timing.get("infer_duration_ms", inf_ms)
                srv_ms = timing.get("server_total_ms", inf_ms)
                rtt = self.last_roundtrip_ms

                diag_w = min(420, w - 24)
                diag_h = 75
                diag_x = (w - diag_w) // 2
                diag_y = h - diag_h - 40

                d_overlay = hud.copy()
                cv2.rectangle(d_overlay, (diag_x, diag_y), (diag_x + diag_w, diag_y + diag_h), (10, 12, 16), -1)
                cv2.addWeighted(d_overlay, 0.90, hud, 0.10, 0, hud)
                cv2.rectangle(hud, (diag_x, diag_y), (diag_x + diag_w, diag_y + diag_h), (0, 215, 255), 1)

                cv2.putText(hud, "LATENCY WATERFALL BREAKDOWN [EXP-1A/3A]", (diag_x + 10, diag_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 240, 255), 1)
                cv2.putText(
                    hud,
                    f"Queue: {q_ms:.1f}ms | Infer: {inf_d_ms:.1f}ms | Server: {srv_ms:.1f}ms | RTT: {rtt:.0f}ms",
                    (diag_x + 10, diag_y + 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.38,
                    (255, 255, 255),
                    1,
                )
                cv2.putText(
                    hud,
                    f"End-to-End Camera-to-Alert Latency: ~{rtt + srv_ms:.1f}ms (Fast-Path Active)",
                    (diag_x + 10, diag_y + 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.35,
                    (0, 255, 160),
                    1,
                )

        # 8. Guided Calibration Wizard Overlay (Centered if calibrating)
        is_calibrating = cal is not None and not cal.get("is_completed", False)
        if is_calibrating:
            step = cal.get("phase_index", 1)
            total = cal.get("total_phases", 4)
            zone_name = cal.get("current_zone", "CENTER")
            instruction = cal.get("instruction", f"LOOK AT {zone_name}")
            prog = cal.get("zone_progress_pct", 0.0)
            total_prog = cal.get("total_progress_pct", 0.0)
            feedback = cal.get("stability_feedback", "Holding Steady")
            in_zone = cal.get("is_head_in_target_zone", True)

            cal_w = min(560, w - 40)
            cal_h = 130
            cal_x = (w - cal_w) // 2
            cal_y = (h - cal_h) // 2

            overlay_cal = hud.copy()
            bg_color = (45, 15, 60) if in_zone else (20, 20, 70)
            cv2.rectangle(overlay_cal, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), bg_color, -1)
            cv2.addWeighted(overlay_cal, 0.88, hud, 0.12, 0, hud)
            border_color = (220, 100, 255) if in_zone else (0, 215, 255)
            cv2.rectangle(hud, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), border_color, 2)

            cv2.putText(
                hud,
                f"4-ZONE CALIBRATION | STEP {step} OF {total} (Total: {total_prog:.0f}%)",
                (cal_x + 18, cal_y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 220, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                instruction,
                (cal_x + 18, cal_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.60,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            feed_color = (0, 255, 150) if in_zone else (0, 215, 255)
            cv2.putText(
                hud,
                f"Status: {feedback} | Target: {zone_name}",
                (cal_x + 18, cal_y + 88),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.46,
                feed_color,
                1,
                cv2.LINE_AA,
            )

            bar_w = cal_w - 36
            fill = int((prog / 100.0) * bar_w)
            cv2.rectangle(hud, (cal_x + 18, cal_y + 100), (cal_x + 18 + bar_w, cal_y + 116), (50, 25, 70), -1)
            cv2.rectangle(hud, (cal_x + 18, cal_y + 100), (cal_x + 18 + fill, cal_y + 116), (220, 100, 255), -1)
            cv2.putText(
                hud,
                f"{prog:.0f}%",
                (cal_x + 22 + fill if fill < bar_w - 40 else cal_x + fill - 35, cal_y + 113),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

        # 9. Notification Banner & Controls Helper
        if time.time() < self.notification_expiry:
            cv2.putText(
                hud,
                self.notification_msg,
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            cv2.putText(
                hud,
                "Keys: [c] Calibrate | [m] Audio | [t] Latency Waterfall | [f] Flip | [1,2,3] Exp Modes | [s] Save | [q] Quit",
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (160, 165, 175),
                1,
                cv2.LINE_AA,
            )

        return hud

    def handle_server_message(self, resp_data: Dict[str, Any]) -> None:
        """Update local telemetry and notification state from server response."""
        msg_type = resp_data.get("type")

        if msg_type == "TELEMETRY":
            self.total_telemetry_received += 1
            self.latest_telemetry = resp_data.get("telemetry")
            cal = resp_data.get("calibration")
            if cal is not None:
                self.latest_calibration = cal
                if cal.get("is_completed", False):
                    self.audio_alarm.play_chime(1400, 140)
                    self.set_notification("Calibration complete! Personal baseline active.", duration_sec=4.0)
            elif self.latest_calibration and not self.latest_calibration.get("is_completed", False):
                self.latest_calibration = None
                self.set_notification("Calibration finished! Personal baseline active.", duration_sec=4.0)

            # Trigger synthesized audio fatigue alerts
            if self.latest_telemetry:
                st = self.latest_telemetry.get("state")
                if st == "CRITICAL":
                    self.audio_alarm.play_critical()
                elif st == "WARNING":
                    self.audio_alarm.play_warning()

        elif msg_type == "CALIBRATION_STARTED":
            self.audio_alarm.play_chime(1100, 90)
            self.latest_calibration = resp_data.get("calibration")
            self.set_notification("Calibration started! Follow target instructions in center of screen.", duration_sec=4.0)
        elif msg_type == "CALIBRATION_CANCELLED":
            self.latest_calibration = None
            self.set_notification("Calibration cancelled.", duration_sec=3.0)
        elif msg_type == "PROFILE_SAVED":
            self.audio_alarm.play_chime(1500, 120)
            prof = resp_data.get("profile", {})
            is_cal = prof.get("is_calibrated", False)
            mode = prof.get("calibration_mode", "UNKNOWN")
            center_ear = prof.get("center_ear_median", 0.0)
            if is_cal:
                self.set_notification(f"Calibrated profile saved to disk! Mode: {mode} (EAR: {center_ear:.3f})", duration_sec=5.0)
            else:
                self.set_notification(f"Profile saved ({mode}). Press 'c' to run 4-zone calibration first!", duration_sec=5.0)
        elif msg_type == "EXPERIMENT_SWITCHED":
            self.audio_alarm.play_chime(1200, 70)
            active_mode = resp_data.get("active_mode", "UNKNOWN")
            self.set_notification(f"Experiment mode switched: {active_mode.split('_')[-1]}", duration_sec=3.5)

    async def send_control_command(self, ws: Any, cmd: Dict[str, Any]) -> None:
        """Send a JSON control command and immediately read & process the server's reply."""
        await ws.send(json.dumps(cmd))
        raw_reply = await ws.recv()
        resp_data = json.loads(raw_reply)
        self.handle_server_message(resp_data)

    async def run(self) -> None:
        """Main async loop: captures frames, sends to WebSocket, receives telemetry, renders HUD."""
        import websockets

        cap = None
        if not self.synthetic:
            cap = cv2.VideoCapture(self.camera_idx)
            if not cap.isOpened():
                print(f"\n[Client ERROR] Unable to open webcam at index {self.camera_idx}.")
                print("If you do not have a physical webcam connected, you can run in simulated mode:")
                print("    python backend/research/client_webcam_test.py --synthetic\n")
                return

            # Request desired capture resolution
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            cap.set(cv2.CAP_PROP_FPS, self.target_fps)

            # Query actual negotiated hardware camera properties
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)

            self.actual_width = actual_w if actual_w > 0 else self.width
            self.actual_height = actual_h if actual_h > 0 else self.height
            self.actual_camera_fps = actual_fps if actual_fps > 0 else float(self.target_fps)

            print(
                f"[Client Webcam] Requested: {self.width}x{self.height} @ {self.target_fps} FPS | "
                f"Negotiated Hardware: {self.actual_width}x{self.actual_height} @ {self.actual_camera_fps:.1f} FPS"
            )
        else:
            self.actual_width = self.width
            self.actual_height = self.height
            self.actual_camera_fps = float(self.target_fps)
            print(f"[Client Webcam] Running in SYNTHETIC simulation mode ({self.width}x{self.height} @ {self.target_fps} FPS).")

        print(f"[Client] Connecting to WebSocket: {self.server_ws_url}")
        print(f"[Client] Session ID: {self.session_id}")

        frame_seq = 0
        try:
            async with websockets.connect(self.server_ws_url, max_size=16 * 1024 * 1024) as ws:
                self.connected = True
                print("[Client] WebSocket connection established successfully!")
                self.set_notification("Connected to server! Starting real-time stream...")

                frame_interval = 1.0 / self.target_fps

                while self.running:
                    loop_start = time.time()
                    frame_seq += 1

                    if self.synthetic:
                        frame = self.generate_synthetic_frame(frame_seq)
                    else:
                        ret, frame = cap.read()
                        if not ret:
                            print("[Client WARNING] Failed to read frame from webcam.")
                            await asyncio.sleep(0.05)
                            continue

                    if self.mirror:
                        frame = cv2.flip(frame, 1)

                    # Encode to JPEG
                    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality]
                    success, buffer = cv2.imencode(".jpg", frame, encode_param)
                    if not success:
                        continue

                    jpeg_bytes = buffer.tobytes()

                    # Send binary frame over WebSocket
                    t0 = time.time()
                    await ws.send(jpeg_bytes)
                    self.total_frames_sent += 1

                    # Wait for telemetry response
                    raw_resp = await ws.recv()
                    t1 = time.time()
                    rtt_ms = (t1 - t0) * 1000.0
                    self.last_roundtrip_ms = rtt_ms
                    self.rtt_history_ms.append(rtt_ms)

                    resp_data = json.loads(raw_resp)
                    self.handle_server_message(resp_data)

                    # Render HUD
                    display_frame = self.draw_hud(frame)

                    if not self.headless:
                        cv2.imshow("DriveGuard Live Client Telemetry (WebSocket)", display_frame)

                        # Keyboard Handling
                        key = cv2.waitKey(1) & 0xFF
                        if key in (ord("q"), 27):  # 'q' or ESC
                            print("[Client] Quit requested by user.")
                            self.running = False
                            break
                        elif key == ord("m"):
                            self.audio_alarm.enabled = not self.audio_alarm.enabled
                            aud_st = "ENABLED" if self.audio_alarm.enabled else "MUTED"
                            print(f"[Client] Audio Alert Buzzer: {aud_st}")
                            if self.audio_alarm.enabled:
                                self.audio_alarm.play_chime(1200, 80)
                            self.set_notification(f"Audio Buzzer: {aud_st}", duration_sec=2.5)
                        elif key == ord("t"):
                            self.show_diagnostics = not self.show_diagnostics
                            diag_st = "ON" if self.show_diagnostics else "OFF"
                            self.set_notification(f"Timing Waterfall: {diag_st}", duration_sec=2.0)
                        elif key == ord("e"):
                            self.show_advanced_metrics = not self.show_advanced_metrics
                            adv_st = "ON" if self.show_advanced_metrics else "OFF"
                            self.set_notification(f"Euro NCAP / Optalert Panel: {adv_st}", duration_sec=2.0)
                        elif key == ord("f"):
                            self.mirror = not self.mirror
                            m_st = "ON" if self.mirror else "OFF"
                            self.set_notification(f"Horizontal Mirror: {m_st}", duration_sec=2.0)
                        elif key == ord("c"):
                            print("[Client] Requesting 4-Zone Calibration...")
                            self.audio_alarm.play_chime(1000, 80)
                            await self.send_control_command(
                                ws,
                                {
                                    "action": "START_CALIBRATION",
                                    "four_zone": True,
                                    "zone_duration_sec": 4.0,
                                },
                            )
                        elif key == ord("1"):
                            print("[Client] Switching to Experiment A (Geometric EAR Only)...")
                            self.audio_alarm.play_chime(900, 70)
                            await self.send_control_command(
                                ws,
                                {
                                    "action": "SWITCH_EXPERIMENT",
                                    "mode": "EXPERIMENT_A_GEOMETRIC_ONLY",
                                },
                            )
                        elif key == ord("2"):
                            print("[Client] Switching to Experiment B (MobileNetV3 Always)...")
                            self.audio_alarm.play_chime(950, 70)
                            await self.send_control_command(
                                ws,
                                {
                                    "action": "SWITCH_EXPERIMENT",
                                    "mode": "EXPERIMENT_B_CNN_ALWAYS",
                                },
                            )
                        elif key == ord("3"):
                            print("[Client] Switching to Experiment C (Confidence-Gated Cascade)...")
                            self.audio_alarm.play_chime(1000, 70)
                            await self.send_control_command(
                                ws,
                                {
                                    "action": "SWITCH_EXPERIMENT",
                                    "mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
                                },
                            )
                        elif key == ord("s"):
                            print("[Client] Saving calibrated profile to backend...")
                            self.audio_alarm.play_chime(1300, 100)
                            await self.send_control_command(
                                ws,
                                {
                                    "action": "SAVE_PROFILE",
                                    "driver_name": "Michael",
                                    "vehicle_id": "DriveGuard-01",
                                },
                            )

                    # Automated test termination check
                    if self.test_frames is not None and frame_seq >= self.test_frames:
                        print(f"\n[Client Test] Reached target test frame count ({self.test_frames}). Exiting cleanly.")
                        self.running = False
                        break

                    # Maintain target frame pacing
                    elapsed = time.time() - loop_start
                    sleep_time = max(0.001, frame_interval - elapsed)
                    await asyncio.sleep(sleep_time)

        except websockets.exceptions.ConnectionClosed:
            print("[Client] WebSocket connection closed by server.")
        except ConnectionRefusedError:
            print(f"\n[Client ERROR] Could not connect to {self.server_ws_url}.")
            print("Is the DriveGuard backend running? Start it first with: python run.py\n")
        except Exception as exc:
            print(f"[Client EXCEPTION] {exc}")
        finally:
            if cap is not None:
                cap.release()
            cv2.destroyAllWindows()
            print("[Client] Cleaned up camera and OpenCV windows.")

            # Print test statistics summary if test frames were requested
            if self.total_telemetry_received > 0:
                avg_rtt = sum(self.rtt_history_ms) / len(self.rtt_history_ms) if self.rtt_history_ms else 0.0
                print("\n======================================================================")
                print("   [DriveGuard Client Webcam Test Summary]")
                print("======================================================================")
                print(f"  Frames Sent:           {self.total_frames_sent}")
                print(f"  Telemetry Received:    {self.total_telemetry_received}")
                print(f"  Average RTT:           {avg_rtt:.1f} ms")
                print(f"  Negotiated Resolution: {self.actual_width}x{self.actual_height}")
                print(f"  Negotiated FPS:        {self.actual_camera_fps:.1f}")
                if self.latest_telemetry:
                    st = self.latest_telemetry.get("state")
                    ear = self.latest_telemetry.get("eye_metrics", {}).get("ear_raw", 0.0)
                    avr = self.latest_telemetry.get("eye_metrics", {}).get("blink_avr")
                    ht = self.latest_telemetry.get("eye_metrics", {}).get("gaze_entropy")
                    print(f"  Final State:           {st}")
                    print(f"  Final EAR:             {ear:.3f}")
                    print(f"  Final AVR:             {avr}")
                    print(f"  Final Gaze Entropy:    {ht}")
                print("======================================================================\n")


def parse_resolution(res_str: str) -> Tuple[int, int]:
    """Parse '640x480' or '1280x720' into (width, height)."""
    try:
        parts = res_str.lower().split("x")
        if len(parts) == 2:
            return int(parts[0]), int(parts[1])
    except Exception:
        pass
    return 640, 480


def main() -> None:
    auto_switch_to_venv_if_needed()

    parser = argparse.ArgumentParser(
        description="DriveGuard Live WebSocket Webcam Client Test & Diagnostic Cockpit HUD",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", type=str, default=None, help="Direct WebSocket URL. Overrides --host/--port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="FastAPI server host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--session-id", type=str, default="live_webcam_demo", help="Session ID")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (0 = default)")
    parser.add_argument("--fps", type=int, default=25, help="Target capture FPS (e.g. 15, 20, 25, 30)")
    parser.add_argument("--resolution", type=str, default="640x480", help="Resolution preset (640x480, 1280x720, etc.)")
    parser.add_argument("--width", type=int, default=None, help="Frame width in pixels (overrides --resolution)")
    parser.add_argument("--height", type=int, default=None, help="Frame height in pixels (overrides --resolution)")
    parser.add_argument("--quality", type=int, default=70, help="JPEG compression quality (1-100)")
    parser.add_argument("--synthetic", action="store_true", help="Run with simulated synthetic driver frames (no webcam required)")
    parser.add_argument("--mute", action="store_true", help="Start client with audio alerts muted")
    parser.add_argument("--no-mirror", action="store_true", help="Disable horizontal mirror flip")
    parser.add_argument("--show-diagnostics", action="store_true", help="Start with latency waterfall breakdown visible")
    parser.add_argument("--test-frames", type=int, default=None, help="Headless test execution: exit cleanly after N frames")
    parser.add_argument("--verify", action="store_true", help="Run automated 20-frame headless self-verification test")
    args = parser.parse_args()

    # Resolution resolution
    if args.width and args.height:
        width, height = args.width, args.height
    else:
        width, height = parse_resolution(args.resolution)

    # Verification shortcut
    if args.verify:
        args.synthetic = True
        args.mute = True
        if args.test_frames is None:
            args.test_frames = 20

    ws_url = args.url if args.url else f"ws://{args.host}:{args.port}/ws/v1/session/{args.session_id}"

    client = LiveWebcamHUDClient(
        server_ws_url=ws_url,
        session_id=args.session_id,
        camera_idx=args.camera,
        target_fps=args.fps,
        width=width,
        height=height,
        jpeg_quality=args.quality,
        synthetic=args.synthetic,
        audio_enabled=not args.mute,
        mirror=not args.no_mirror,
        show_diagnostics=args.show_diagnostics,
        test_frames=args.test_frames,
        headless=(args.test_frames is not None),
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n[Client] Terminated by user.")


if __name__ == "__main__":
    main()
