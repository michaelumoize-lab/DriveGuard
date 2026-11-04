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
import subprocess
import sys
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
    ) -> None:
        self.server_ws_url = server_ws_url
        self.session_id = session_id
        self.camera_idx = camera_idx
        self.target_fps = target_fps
        self.jpeg_quality = jpeg_quality
        self.synthetic = synthetic

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

        # Semi-transparent top dashboard bar
        overlay = hud.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (20, 20, 24), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 24), -1)
        cv2.addWeighted(overlay, 0.75, hud, 0.25, 0, hud)

        # Title Banner
        cv2.putText(
            hud,
            "DRIVEGUARD | ADAPTIVE AI MONITORING",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Connection Status
        conn_text = "WS: CONNECTED" if self.connected else "WS: CONNECTING..."
        conn_color = (0, 220, 0) if self.connected else (0, 165, 255)
        cv2.putText(
            hud,
            conn_text,
            (w - 180, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            conn_color,
            1,
            cv2.LINE_AA,
        )

        t = self.latest_telemetry
        cal = self.latest_calibration

        if t is not None:
            # 1. Alert State Badge
            state = t.get("state", "NORMAL")
            state_colors = {
                "NORMAL": (0, 200, 0),
                "WARNING": (0, 215, 255),
                "CRITICAL": (0, 0, 255),
                "MONITORING_UNAVAILABLE": (128, 128, 128),
            }
            color = state_colors.get(state, (0, 200, 0))

            cv2.rectangle(hud, (15, 38), (170, 72), color, -1)
            cv2.putText(
                hud,
                state,
                (25, 62),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 0, 0) if state in ("NORMAL", "WARNING") else (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # 2. Gaze Zone Pill
            zone = t.get("pose_zone", "CENTER")
            zone_color = (255, 180, 50) if zone != "CENTER" else (200, 200, 200)
            cv2.putText(
                hud,
                f"Zone: {zone}",
                (185, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                zone_color,
                2,
                cv2.LINE_AA,
            )

            # 3. Mirror Check Protection Status
            comp = t.get("comparison", {})
            if comp.get("is_mirror_check_suppressed", False):
                cv2.rectangle(hud, (360, 38), (620, 72), (255, 140, 0), -1)
                cv2.putText(
                    hud,
                    "MIRROR-CHECK PROTECTED",
                    (370, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif comp.get("fixed_false_alarm", False):
                cv2.rectangle(hud, (360, 38), (620, 72), (0, 0, 220), -1)
                cv2.putText(
                    hud,
                    "STATIC METHOD: FALSE ALARM!",
                    (370, 62),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # 4. Latency & FPS
            fps = t.get("fps", 0.0)
            inf_ms = t.get("inference_latency_ms", 0.0)
            cv2.putText(
                hud,
                f"{fps:.1f} FPS | {inf_ms:.1f} ms",
                (w - 180, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
                cv2.LINE_AA,
            )

            # 5. Side Telemetry Panel (Left Hand Side)
            panel_x = 15
            panel_y = 110

            # Eye Metrics
            eye = t.get("eye_metrics", {})
            ear = eye.get("ear_raw", 0.0)
            thresh = eye.get("ear_closure_threshold", 0.22)
            perclos = eye.get("perclos_pct", 0.0)
            is_closed = eye.get("is_eye_closed", False)

            # Background panel
            panel_w = 260
            panel_h = 240
            cv2.rectangle(hud, (panel_x - 5, panel_y - 20), (panel_x + panel_w, panel_y + panel_h), (15, 15, 18), -1)
            cv2.rectangle(hud, (panel_x - 5, panel_y - 20), (panel_x + panel_w, panel_y + panel_h), (60, 60, 70), 1)

            cv2.putText(hud, "EYE & KINEMATICS", (panel_x, panel_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)

            # EAR Bar
            cv2.putText(hud, f"EAR: {ear:.3f} (Thresh: {thresh:.3f})", (panel_x, panel_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
            bar_w = 180
            fill_w = int(np.clip(ear / 0.45, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (panel_x, panel_y + 32), (panel_x + bar_w, panel_y + 44), (40, 40, 50), -1)
            ear_bar_color = (0, 0, 255) if is_closed else (0, 220, 0)
            cv2.rectangle(hud, (panel_x, panel_y + 32), (panel_x + fill_w, panel_y + 44), ear_bar_color, -1)
            # Threshold line
            thresh_x = panel_x + int(np.clip(thresh / 0.45, 0.0, 1.0) * bar_w)
            cv2.line(hud, (thresh_x, panel_y + 30), (thresh_x, panel_y + 46), (0, 255, 255), 2)

            # PERCLOS Bar
            cv2.putText(hud, f"PERCLOS (60s): {perclos:.1f}%", (panel_x, panel_y + 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (230, 230, 230), 1)
            p_fill_w = int(np.clip(perclos / 100.0, 0.0, 1.0) * bar_w)
            cv2.rectangle(hud, (panel_x, panel_y + 75), (panel_x + bar_w, panel_y + 87), (40, 40, 50), -1)
            p_color = (0, 220, 0) if perclos < 15 else ((0, 215, 255) if perclos < 25 else (0, 0, 255))
            cv2.rectangle(hud, (panel_x, panel_y + 75), (panel_x + p_fill_w, panel_y + 87), p_color, -1)

            # Blink Kinematics
            blink_dur = eye.get("blink_duration_ms")
            dur_str = f"{blink_dur:.0f} ms" if blink_dur else "N/A"
            rcr = eye.get("blink_rcr")
            rcr_str = f"{rcr:.2f}" if rcr else "N/A"
            rate = eye.get("blink_rate_per_min", 0.0)

            cv2.putText(hud, f"Blink Dur:   {dur_str}", (panel_x, panel_y + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.putText(hud, f"Kinematic RCR: {rcr_str}", (panel_x, panel_y + 130), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.putText(hud, f"Blink Rate:  {rate:.1f} /min", (panel_x, panel_y + 148), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

            # Cascade Decision
            casc = t.get("cascade", {})
            cnn_on = casc.get("cnn_triggered", False)
            winner = casc.get("arbitration_winner", "N/A")
            mode = casc.get("experiment_mode", "EXP_C")

            cv2.putText(hud, "CASCADE ARBITRATION", (panel_x, panel_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA)
            cv2.putText(hud, f"Mode:   {mode.split('_')[-1]}", (panel_x, panel_y + 195), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
            cv2.putText(
                hud,
                f"CNN Gate: {'INVOKED' if cnn_on else 'BYPASS (Save Power)'}",
                (panel_x, panel_y + 213),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (0, 220, 255) if cnn_on else (150, 150, 150),
                1,
            )
            cv2.putText(hud, f"Decision: {winner}", (panel_x, panel_y + 231), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)

        # 6. Guided Calibration Wizard Overlay (Centered if calibrating)
        if cal is not None and cal.get("is_calibrating", False):
            phase = cal.get("phase_name", "CALIBRATING")
            step = cal.get("phase_idx", 1)
            rem = cal.get("remaining_sec", 0.0)
            prog = cal.get("step_progress_pct", 0.0)

            cal_w = 460
            cal_h = 100
            cal_x = (w - cal_w) // 2
            cal_y = h // 2 - 50

            cv2.rectangle(hud, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), (35, 10, 50), -1)
            cv2.rectangle(hud, (cal_x, cal_y), (cal_x + cal_w, cal_y + cal_h), (200, 100, 255), 2)

            cv2.putText(
                hud,
                f"4-ZONE CALIBRATION (Step {step}/4)",
                (cal_x + 20, cal_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                hud,
                f"{phase} ({rem:.1f}s)",
                (cal_x + 20, cal_y + 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )

            # Progress Bar
            fill = int((prog / 100.0) * (cal_w - 40))
            cv2.rectangle(hud, (cal_x + 20, cal_y + 72), (cal_x + cal_w - 20, cal_y + 86), (60, 30, 80), -1)
            cv2.rectangle(hud, (cal_x + 20, cal_y + 72), (cal_x + 20 + fill, cal_y + 86), (200, 100, 255), -1)

        # 7. Notification Banner
        if time.time() < self.notification_expiry:
            cv2.putText(
                hud,
                self.notification_msg,
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (0, 255, 255),
                1,
                cv2.LINE_AA,
            )
        else:
            # Controls helper
            cv2.putText(
                hud,
                "Keys: [c] Calibrate 4-Zones | [1,2,3] Exp A/B/C | [s] Save Profile | [q] Quit",
                (15, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (160, 160, 160),
                1,
                cv2.LINE_AA,
            )

        return hud

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
                    msg_type = resp_data.get("type")

                    if msg_type == "TELEMETRY":
                        self.latest_telemetry = resp_data.get("telemetry")
                        self.latest_calibration = resp_data.get("calibration")
                    elif msg_type == "CALIBRATION_STARTED":
                        self.latest_calibration = resp_data.get("calibration")
                        self.set_notification("Calibration started! Follow on-screen instructions.")
                    elif msg_type == "CALIBRATION_CANCELLED":
                        self.latest_calibration = None
                        self.set_notification("Calibration cancelled.")
                    elif msg_type == "PROFILE_SAVED":
                        prof = resp_data.get("profile", {})
                        self.set_notification(f"Profile saved successfully! Mode: {prof.get('calibration_mode')}")
                    elif msg_type == "EXPERIMENT_SWITCHED":
                        active_mode = resp_data.get("active_mode", "UNKNOWN")
                        self.set_notification(f"Switched mode: {active_mode.split('_')[-1]}")

                    # Render HUD
                    display_frame = self.draw_hud(frame)
                    cv2.imshow("DriveGuard Live Client Telemetry (WebSocket)", display_frame)

                    # Keyboard Handling
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):  # 'q' or ESC
                        print("[Client] Quit requested by user.")
                        self.running = False
                        break
                    elif key == ord("c"):
                        print("[Client] Requesting 4-Zone Calibration...")
                        self.set_notification("Starting 4-Zone Calibration...")
                        await ws.send(
                            json.dumps({
                                "action": "START_CALIBRATION",
                                "four_zone": True,
                                "zone_duration_sec": 4.0,
                            })
                        )
                    elif key == ord("1"):
                        print("[Client] Switching to Experiment A (Geometric EAR Only)...")
                        await ws.send(
                            json.dumps({
                                "action": "SWITCH_EXPERIMENT",
                                "mode": "EXPERIMENT_A_GEOMETRIC_ONLY",
                            })
                        )
                    elif key == ord("2"):
                        print("[Client] Switching to Experiment B (MobileNetV3 Always)...")
                        await ws.send(
                            json.dumps({
                                "action": "SWITCH_EXPERIMENT",
                                "mode": "EXPERIMENT_B_CNN_ALWAYS",
                            })
                        )
                    elif key == ord("3"):
                        print("[Client] Switching to Experiment C (Confidence-Gated Cascade)...")
                        await ws.send(
                            json.dumps({
                                "action": "SWITCH_EXPERIMENT",
                                "mode": "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
                            })
                        )
                    elif key == ord("s"):
                        print("[Client] Saving calibrated profile to backend...")
                        await ws.send(
                            json.dumps({
                                "action": "SAVE_PROFILE",
                                "driver_name": "Michael",
                                "vehicle_id": "DriveGuard-01",
                            })
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
    parser.add_argument("--host", type=str, default="127.0.0.1", help="FastAPI server host")
    parser.add_argument("--port", type=int, default=8000, help="FastAPI server port")
    parser.add_argument("--session-id", type=str, default="live_webcam_demo", help="Session ID")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (0 = default)")
    parser.add_argument("--fps", type=int, default=25, help="Target capture FPS")
    parser.add_argument("--quality", type=int, default=70, help="JPEG compression quality (1-100)")
    parser.add_argument("--synthetic", action="store_true", help="Run with simulated synthetic driver frames (no webcam required)")
    args = parser.parse_args()

    ws_url = f"ws://{args.host}:{args.port}/ws/v1/session/{args.session_id}"

    client = LiveWebcamHUDClient(
        server_ws_url=ws_url,
        session_id=args.session_id,
        camera_idx=args.camera,
        target_fps=args.fps,
        jpeg_quality=args.quality,
        synthetic=args.synthetic,
    )

    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        print("\n[Client] Terminated by user.")


if __name__ == "__main__":
    main()
