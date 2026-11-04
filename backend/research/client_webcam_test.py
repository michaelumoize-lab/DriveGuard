"""
DriveGuard — Live WebSocket Webcam Client Test & Diagnostic HUD
================================================================
Streams live webcam frames to the FastAPI WebSocket server
(`ws://127.0.0.1:8000/ws/v1/session/{session_id}`) and displays the real-time
adaptive AI telemetry with an interactive OpenCV HUD.

Features:
  - Real-time 4-Zone Pose-Conditioned Gaze & Baselines
  - Sub-frame Kinematic Eye Closure & Rolling 60s PERCLOS
  - Confidence-Gated Cascade Decision (Geometry vs. MobileNetV3)
  - Live Side-by-Side Comparison (Conventional Static vs. DriveGuard Adaptive)
  - Interactive Keyboard Controls:
      'c' -> Start 4-Zone Calibration Wizard
      '1' -> Switch to Experiment A (Geometric EAR Only)
      '2' -> Switch to Experiment B (MobileNetV3 CNN Always)
      '3' -> Switch to Experiment C (Confidence-Gated Cascade)
      's' -> Save Calibrated Profile
      'q' -> Quit

Usage:
    python backend/research/client_webcam_test.py
    python backend/research/client_webcam_test.py --camera 0 --fps 25
    python backend/research/client_webcam_test.py --session-id test_driver_01
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
from typing import Any, Dict, Optional

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
        self._queue: queue.Queue = queue.Queue(maxsize=4)
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
    """Manages OpenCV webcam capture, WebSocket streaming, and HUD rendering."""

    def __init__(
        self,
        server_ws_url: str,
        session_id: str = "webcam_live_demo",
        camera_idx: int = 0,
        target_fps: int = 25,
        jpeg_quality: int = 70,
        synthetic: bool = False,
        audio_enabled: bool = True,
    ) -> None:
        self.server_ws_url = server_ws_url
        self.session_id = session_id
        self.camera_idx = camera_idx
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.synthetic = synthetic

        # Non-blocking audio synthesizer
        self.audio_alarm = NonBlockingAudioAlarm(enabled=audio_enabled)

        # State storage from server
        self.latest_telemetry: Optional[Dict[str, Any]] = None
        self.latest_calibration: Optional[Dict[str, Any]] = None
        self.connected: bool = False
        self.running: bool = True
        self.last_roundtrip_ms: float = 0.0

        # UI Notification Banner
        self.notification_msg: str = "Connected to DriveGuard Backend. Press 'c' to Calibrate."
        self.notification_expiry: float = time.time() + 5.0

    def generate_synthetic_frame(self, frame_idx: int) -> np.ndarray:
        """Create an animated driver face canvas when running without physical webcam."""
        frame = np.full((480, 640, 3), 30, dtype=np.uint8)
        # Draw face oval
        center = (320, 240)
        cv2.ellipse(frame, center, (110, 150), 0, 0, 360, (180, 150, 120), -1)
        cv2.ellipse(frame, center, (110, 150), 0, 0, 360, (220, 190, 160), 2)

        # Periodic blink simulation (closed for 6 frames every 75 frames)
        is_blink = (frame_idx % 75) < 6
        eye_h = 2 if is_blink else 14

        # Eyes
        cv2.ellipse(frame, (275, 215), (22, eye_h), 0, 0, 360, (255, 255, 255), -1)
        if not is_blink:
            cv2.circle(frame, (275, 215), 7, (60, 40, 20), -1)
        cv2.ellipse(frame, (365, 215), (22, eye_h), 0, 0, 360, (255, 255, 255), -1)
        if not is_blink:
            cv2.circle(frame, (365, 215), 7, (60, 40, 20), -1)

        # Mouth
        cv2.ellipse(frame, (320, 310), (35, 10), 0, 0, 360, (120, 70, 70), -1)
        return frame

    def set_notification(self, msg: str, duration_sec: float = 3.5) -> None:
        self.notification_msg = msg
        self.notification_expiry = time.time() + duration_sec

    def draw_hud(self, frame: np.ndarray) -> np.ndarray:
        """Render informative real-time HUD overlays onto the webcam frame."""
        h, w = frame.shape[:2]
        hud = frame.copy()

        # 1. Semi-transparent top dashboard bar & bottom control bar
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 75), (14, 16, 20), -1)
        cv2.rectangle(overlay, (0, h - 32), (w, h), (14, 16, 20), -1)
        cv2.addWeighted(overlay, 0.80, hud, 0.20, 0, hud)

        # 2. Title Banner
        cv2.putText(
            hud,
            "DRIVEGUARD | COCKPIT HUD",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        t = self.latest_telemetry
        cal = self.latest_calibration

        # 3. Audio Alarm Status Pill
        audio_on = self.audio_alarm.enabled
        aud_color = (0, 220, 100) if audio_on else (120, 120, 130)
        aud_text = "VOL: ON" if audio_on else "VOL: OFF"
        cv2.rectangle(hud, (240, 6), (315, 26), (28, 30, 38), -1)
        cv2.rectangle(hud, (240, 6), (315, 26), aud_color, 1)
        cv2.putText(
            hud,
            aud_text,
            (248, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            aud_color,
            1,
            cv2.LINE_AA,
        )

        # 4. Connection & Latency Info
        conn_text = "WS: 127.0.0.1:8000" if self.connected else "WS: CONNECTING..."
        conn_color = (0, 220, 0) if self.connected else (0, 165, 255)
        cv2.putText(
            hud,
            conn_text,
            (w - 200, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            conn_color,
            1,
            cv2.LINE_AA,
        )

        # Subtle Central Driver Framing Reticle
        cx, cy = w // 2, h // 2
        reticle_col = (0, 220, 100) if (t and t.get("face_detected")) else (80, 80, 90)
        ret_w, ret_h = 65, 75
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
                state,
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
                f"{fps:.1f} FPS | {inf_ms:.1f}ms",
                (w - 200, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                f"RTT: {self.last_roundtrip_ms:.0f} ms",
                (w - 200, 66),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                (0, 220, 150) if self.last_roundtrip_ms < 35 else (0, 180, 255),
                1,
                cv2.LINE_AA,
            )

            # -------------------------------------------------------------
            # 5. LEFT COCKPIT PANEL (EYE & KINEMATICS)
            # -------------------------------------------------------------
            l_x, l_y = 12, 82
            l_w, l_h = 205, 345
            l_overlay = hud.copy()
            cv2.rectangle(l_overlay, (l_x, l_y), (l_x + l_w, l_y + l_h), (12, 14, 18), -1)
            cv2.addWeighted(l_overlay, 0.82, hud, 0.18, 0, hud)
            cv2.rectangle(hud, (l_x, l_y), (l_x + l_w, l_y + l_h), (50, 55, 65), 1)

            cv2.putText(hud, "EYE & BLINK METRICS", (l_x + 10, l_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 225, 255), 1, cv2.LINE_AA)

            eye = t.get("eye_metrics", {})
            ear = eye.get("ear_raw", 0.0)
            thresh = eye.get("ear_closure_threshold", 0.22)
            perclos = eye.get("perclos_pct", 0.0)
            is_closed = eye.get("is_eye_closed", False)

            # EAR Section
            cv2.putText(hud, f"EAR: {ear:.3f} (Thresh: {thresh:.3f})", (l_x + 10, l_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
            bar_w = 180
            fill_w = int(np.clip(ear / 0.45, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (l_x + 10, l_y + 50), (l_x + 10 + bar_w, l_y + 60), (35, 38, 45), -1)
            ear_col = (0, 0, 255) if is_closed else (0, 220, 0)
            cv2.rectangle(hud, (l_x + 10, l_y + 50), (l_x + 10 + fill_w, l_y + 60), ear_col, -1)
            th_x = l_x + 10 + int(np.clip(thresh / 0.45, 0.0, 1.0) * bar_w)
            cv2.line(hud, (th_x, l_y + 48), (th_x, l_y + 62), (0, 240, 255), 2)

            # PERCLOS Section
            cv2.putText(hud, f"PERCLOS (60s): {perclos:.1f}%", (l_x + 10, l_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (230, 230, 230), 1)
            p_fill_w = int(np.clip(perclos / 100.0, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (l_x + 10, l_y + 88), (l_x + 10 + bar_w, l_y + 98), (35, 38, 45), -1)
            p_col = (0, 220, 0) if perclos < 15 else ((0, 215, 255) if perclos < 25 else (0, 0, 255))
            cv2.rectangle(hud, (l_x + 10, l_y + 88), (l_x + 10 + p_fill_w, l_y + 98), p_col, -1)

            # Kinematics Detail
            blink_dur = eye.get("blink_duration_ms")
            dur_str = f"{blink_dur:.0f} ms" if blink_dur else "--"
            rcr = eye.get("blink_rcr")
            rcr_str = f"{rcr:.2f}" if rcr else "--"
            rate = eye.get("blink_rate_per_min", 0.0)

            cv2.putText(hud, f"Blink Duration: {dur_str}", (l_x + 10, l_y + 122), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
            cv2.putText(hud, f"Kinematic RCR:  {rcr_str}", (l_x + 10, l_y + 142), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)
            cv2.putText(hud, f"Blink Freq:     {rate:.1f} /min", (l_x + 10, l_y + 162), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (190, 190, 190), 1)

            # Cascade Decision Section
            cv2.line(hud, (l_x + 10, l_y + 180), (l_x + l_w - 10, l_y + 180), (40, 45, 55), 1)
            casc = t.get("cascade", {})
            cnn_on = casc.get("cnn_triggered", False)
            winner = casc.get("arbitration_winner", "GEOMETRY")
            mode = casc.get("experiment_mode", "EXP_C")

            cv2.putText(hud, "CASCADE GATE", (l_x + 10, l_y + 202), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 225, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Mode: {mode.split('_')[-1]}", (l_x + 10, l_y + 224), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)
            gate_col = (0, 220, 255) if cnn_on else (140, 140, 140)
            cv2.putText(hud, f"CNN: {'INVOKED' if cnn_on else 'BYPASS (Save Power)'}", (l_x + 10, l_y + 244), cv2.FONT_HERSHEY_SIMPLEX, 0.38, gate_col, 1)
            cv2.putText(hud, f"Decision: {winner}", (l_x + 10, l_y + 264), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 255, 255), 1)

            # -------------------------------------------------------------
            # 6. RIGHT COCKPIT PANEL (3D HEAD POSE & RISK)
            # -------------------------------------------------------------
            r_x, r_y = w - 217, 82
            r_w, r_h = 205, 345
            r_overlay = hud.copy()
            cv2.rectangle(r_overlay, (r_x, r_y), (r_x + r_w, r_y + r_h), (12, 14, 18), -1)
            cv2.addWeighted(r_overlay, 0.82, hud, 0.18, 0, hud)
            cv2.rectangle(hud, (r_x, r_y), (r_x + r_w, r_y + r_h), (50, 55, 65), 1)

            cv2.putText(hud, "3D HEAD POSE & GAZE", (r_x + 10, r_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 225, 255), 1, cv2.LINE_AA)

            hp = t.get("head_pose", {})
            pitch = hp.get("pitch_deg", 0.0)
            yaw = hp.get("yaw_deg", 0.0)
            roll = hp.get("roll_deg", 0.0)

            pitch_status = "NOD" if pitch <= -12.0 else ("UP" if pitch >= 14.0 else "LEVEL")
            yaw_status = "RIGHT" if yaw <= -12.0 else ("LEFT" if yaw >= 12.0 else "CTR")

            cv2.putText(hud, f"Pitch (P): {pitch:+5.1f} deg [{pitch_status}]", (r_x + 10, r_y + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1)
            cv2.putText(hud, f"Yaw   (Y): {yaw:+5.1f} deg [{yaw_status}]", (r_x + 10, r_y + 64), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1)
            cv2.putText(hud, f"Roll  (R): {roll:+5.1f} deg", (r_x + 10, r_y + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (200, 200, 200), 1)

            # Horizontal Interactive Yaw Balance Meter
            cv2.putText(hud, "L <--  ROTATION METER  --> R", (r_x + 10, r_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (150, 150, 160), 1)
            cv2.rectangle(hud, (r_x + 10, r_y + 114), (r_x + 10 + bar_w, r_y + 124), (35, 38, 45), -1)
            # Center tick
            cv2.line(hud, (r_x + 10 + bar_w // 2, r_y + 112), (r_x + 10 + bar_w // 2, r_y + 126), (120, 120, 120), 1)
            # Yaw cursor position (-45 deg to +45 deg)
            yaw_ratio = np.clip((-yaw + 45.0) / 90.0, 0.0, 1.0)
            yaw_cursor_x = r_x + 10 + int(yaw_ratio * bar_w)
            cursor_col = (0, 220, 255) if abs(yaw) < 12.0 else (255, 180, 50)
            cv2.circle(hud, (yaw_cursor_x, r_y + 119), 5, cursor_col, -1)

            # Fatigue Risk Score
            cv2.line(hud, (r_x + 10, r_y + 140), (r_x + r_w - 10, r_y + 140), (40, 45, 55), 1)
            risk = t.get("risk_score", 0.0)
            cv2.putText(hud, "FATIGUE RISK SCORE", (r_x + 10, r_y + 162), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (0, 225, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Risk: {risk * 100:.0f}%", (r_x + 10, r_y + 184), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 240), 1)
            r_fill_w = int(np.clip(risk, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (r_x + 10, r_y + 190), (r_x + 10 + bar_w, r_y + 200), (35, 38, 45), -1)
            r_col = (0, 200, 0) if risk < 0.35 else ((0, 215, 255) if risk < 0.65 else (0, 0, 255))
            cv2.rectangle(hud, (r_x + 10, r_y + 190), (r_x + 10 + r_fill_w, r_y + 200), r_col, -1)

            # Active Reason Codes
            reasons = t.get("reason_codes", [])
            cv2.putText(hud, "Active Reason Codes:", (r_x + 10, r_y + 224), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160, 160, 170), 1)
            if reasons:
                for idx, r_code in enumerate(reasons[:4]):
                    short_r = r_code.replace("PROLONGED_", "").replace("SUSTAINED_", "").replace("_", " ")[:20]
                    cv2.putText(hud, f"- {short_r}", (r_x + 10, r_y + 244 + idx * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 220, 255), 1)
            else:
                cv2.putText(hud, "- NORMAL (ALERT)", (r_x + 10, r_y + 244), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 200, 0), 1)

        # 6. Guided Calibration Wizard Overlay (Centered if calibrating)
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

            # Translucent backdrop
            overlay_cal = hud.copy()
            bg_color = (45, 15, 60) if in_zone else (20, 20, 70)
            cv2.rectangle(overlay_cal, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), bg_color, -1)
            cv2.addWeighted(overlay_cal, 0.88, hud, 0.12, 0, hud)
            border_color = (220, 100, 255) if in_zone else (0, 215, 255)
            cv2.rectangle(hud, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), border_color, 2)

            # Step title
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
            # Target instruction
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
            # Feedback string
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

            # Progress Bar for active step
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

        # 7. Notification Banner
        if time.time() < self.notification_expiry:
            cv2.putText(
                hud,
                self.notification_msg,
                (15, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            # Controls helper
            cv2.putText(
                hud,
                "Keys: [c] Calibrate | [m] Toggle Audio | [1,2,3] Exp A/B/C | [s] Save Profile | [q] Quit",
                (15, h - 11),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (160, 165, 175),
                1,
                cv2.LINE_AA,
            )

        return hud

    def handle_server_message(self, resp_data: Dict[str, Any]) -> None:
        """Update local telemetry and notification state from server response."""
        msg_type = resp_data.get("type")

        if msg_type == "TELEMETRY":
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
        """Main async loop: captures frames, sends to WebSocket, receives telemetry."""
        import websockets

        cap = None
        if not self.synthetic:
            cap = cv2.VideoCapture(self.camera_idx)
            if not cap.isOpened():
                print(f"\n[Client ERROR] Unable to open webcam at index {self.camera_idx}.")
                print("If you do not have a physical webcam connected, you can run in simulated mode:")
                print("    python backend/research/client_webcam_test.py --synthetic\n")
                return

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        else:
            print("[Client] Running in SYNTHETIC simulation mode (no physical camera needed).")

        print(f"[Client] Connecting to WebSocket: {self.server_ws_url}")
        print(f"[Client] Session ID: {self.session_id}")

        frame_seq = 0
        try:
            async with websockets.connect(self.server_ws_url, max_size=10 * 1024 * 1024) as ws:
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
                        # Mirror for intuitive natural gaze
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

                    # Wait for telemetry response
                    raw_resp = await ws.recv()
                    t1 = time.time()
                    self.last_roundtrip_ms = (t1 - t0) * 1000.0

                    resp_data = json.loads(raw_resp)
                    self.handle_server_message(resp_data)

                    # Render HUD
                    display_frame = self.draw_hud(frame)
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


def main() -> None:
    auto_switch_to_venv_if_needed()

    parser = argparse.ArgumentParser(
        description="DriveGuard Live WebSocket Webcam Client Test",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--url", type=str, default=None, help="Direct WebSocket URL (e.g. wss://driveguard.onrender.com/ws/v1/session/live_01). Overrides --host/--port.")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="FastAPI server host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--session-id", type=str, default="live_webcam_demo", help="Session ID")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (0 = default)")
    parser.add_argument("--fps", type=int, default=25, help="Target capture FPS")
    parser.add_argument("--quality", type=int, default=70, help="JPEG compression quality (1-100)")
    parser.add_argument("--synthetic", action="store_true", help="Run with simulated synthetic driver frames (no webcam required)")
    parser.add_argument("--mute", action="store_true", help="Start client with audio alerts muted")
    args = parser.parse_args()

    ws_url = args.url if args.url else f"ws://{args.host}:{args.port}/ws/v1/session/{args.session_id}"

    client = LiveWebcamHUDClient(
        server_ws_url=ws_url,
        session_id=args.session_id,
        camera_idx=args.camera,
        target_fps=args.fps,
        jpeg_quality=args.quality,
        synthetic=args.synthetic,
        audio_enabled=not args.mute,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n[Client] Terminated by user.")


if __name__ == "__main__":
    main()
