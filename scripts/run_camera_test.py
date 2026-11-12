"""
DriveGuard — Camera Acquisition Benchmark & Verification Runner (Phase 1 & Phase 2).

What it does:
  Initializes a video source (laptop webcam, external USB UVC camera, or phone stream URL),
  captures frames at the requested resolution and FPS, measures true hardware ingestion FPS
  versus consumer display loop FPS, and renders a diagnostic HUD overlay.

Usage:
  # Test built-in HP EliteBook webcam (Phase 1)
  python scripts/run_camera_test.py --device 0

  # Test phone camera stream (Phase 2)
  python scripts/run_camera_test.py --url http://192.168.1.50:8080/video

Controls:
  'q' or ESC : Exit gracefully and display performance summary
  's'        : Save diagnostic snapshot PNG to disk
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Add project root to sys.path to allow running directly from anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from backend.app.camera.camera_manager import create_camera, discover_local_devices
from backend.app.camera.frame_quality import FrameQualityAnalyzer, FrameQualityResult, QualityStatus

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("driveguard.benchmark")


def draw_diagnostic_hud(
    frame: np.ndarray,
    source_id: str,
    actual_res: tuple[int, int],
    ingestion_fps: float,
    display_fps: float,
    frame_age_ms: float,
    buffer_stats: dict,
    quality: Optional[FrameQualityResult] = None,
) -> np.ndarray:
    """
    Renders a semi-transparent diagnostic overlay with real-time performance indicators
    and optical quality metrics (Phase 3).
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # Create semi-transparent top banner
    banner_height = 105
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), -1)  # Slate-900
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Color indicator for FPS health
    if ingestion_fps >= 25.0:
        fps_color = (34, 197, 94)    # Emerald-500 (Green)
        fps_text = "STABLE"
    elif ingestion_fps >= 15.0:
        fps_color = (234, 179, 8)    # Amber-500 (Yellow)
        fps_text = "JITTER"
    else:
        fps_color = (239, 68, 68)    # Red-500 (Alert)
        fps_text = "LOW FPS"

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title & FPS Status Badge
    cv2.putText(canvas, "DriveGuard — Camera & Quality Diagnostics (Phase 3)", (16, 24), font, 0.60, (255, 255, 255), 2, cv2.LINE_AA)
    
    fps_badge = f"[{fps_text} | {ingestion_fps:.1f} FPS]"
    cv2.putText(canvas, fps_badge, (width - 240, 24), font, 0.55, fps_color, 2, cv2.LINE_AA)

    # Quality Badge
    if quality is not None:
        if quality.status == QualityStatus.EXCELLENT:
            qual_color = (34, 197, 94)
            qual_text = "[OPTICAL: EXCELLENT]"
        elif quality.status in (QualityStatus.POOR_LIGHTING, QualityStatus.BLURRY):
            qual_color = (234, 179, 8)
            qual_text = f"[OPTICAL: {quality.status.value}]"
        else:
            qual_color = (239, 68, 68)
            qual_text = "[OPTICAL: DEGRADED]"
        cv2.putText(canvas, qual_text, (width - 480, 24), font, 0.55, qual_color, 2, cv2.LINE_AA)

    # Line 2: Telemetry Metrics
    w, h = actual_res
    telemetry_line_1 = (
        f"Source: {source_id} | "
        f"Resolution: {w}x{h} | "
        f"Ingestion: {ingestion_fps:.1f} FPS | "
        f"Display: {display_fps:.1f} FPS"
    )
    cv2.putText(canvas, telemetry_line_1, (16, 48), font, 0.44, (226, 232, 240), 1, cv2.LINE_AA)

    # Line 3: Optical Quality Metrics (Phase 3)
    if quality is not None:
        telemetry_line_2 = (
            f"Luminance: {quality.mean_luminance:.1f} ({quality.lighting.value}) | "
            f"Sharpness: {quality.blur_variance:.1f} ({quality.blur.value}) | "
            f"Frame Latency: {frame_age_ms:.1f} ms"
        )
    else:
        telemetry_line_2 = f"Frame Latency: {frame_age_ms:.1f} ms"
    cv2.putText(canvas, telemetry_line_2, (16, 70), font, 0.44, (148, 163, 184), 1, cv2.LINE_AA)

    # Line 4: Controls & Warnings
    dropped = buffer_stats.get("total_dropped", 0)
    controls_line = (
        f"Dropped Frames: {dropped} | "
        f"Press 'q'/ESC: exit | 'm': toggle mirror | 's': snapshot"
    )
    cv2.putText(canvas, controls_line, (16, 92), font, 0.40, (100, 116, 139), 1, cv2.LINE_AA)

    return canvas


def run_benchmark(
    source_identifier: str | int = 0,
    target_width: int = 1280,
    target_height: int = 720,
    target_fps: int = 30,
    backend_str: str = "DSHOW",
    duration_sec: float = 0.0,
    headless: bool = False,
    flip_h: bool = False,
    flip_v: bool = False,
) -> int:
    """
    Executes the camera capture verification loop for webcams or phone streams.
    """
    logger.info("Initializing camera adapter for source '%s'...", source_identifier)

    camera = create_camera(
        source=source_identifier,
        target_width=target_width,
        target_height=target_height,
        target_fps=target_fps,
        backend=backend_str,
        flip_horizontal=flip_h,
        flip_vertical=flip_v,
    )

    if not camera.open():
        logger.error("Failed to open camera source '%s'.", source_identifier)
        return 1

    actual_res = camera.get_actual_resolution()
    window_name = f"DriveGuard Diagnostics [{source_identifier}]"

    if not headless:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        # Avoid zero or invalid dimensions
        w = actual_res[0] if actual_res[0] > 0 else target_width
        h = actual_res[1] if actual_res[1] > 0 else target_height
        cv2.resizeWindow(window_name, w, h)

    quality_analyzer = FrameQualityAnalyzer()

    # Performance tracking metrics
    start_time = time.monotonic()
    total_frames_displayed = 0
    display_fps = 0.0
    fps_timer = time.monotonic()
    fps_frame_count = 0
    latencies_ms = []
    latest_quality: Optional[FrameQualityResult] = None

    logger.info("Starting live capture loop. Press 'q' or ESC to terminate.")

    try:
        while True:
            loop_now = time.monotonic()

            # Auto-exit if duration is set (used for automated testing)
            if duration_sec > 0 and (loop_now - start_time) >= duration_sec:
                logger.info("Target duration of %.1fs reached.", duration_sec)
                break

            success, packet = camera.read(timeout_sec=0.1)
            if not success or packet is None:
                time.sleep(0.005)
                continue

            total_frames_displayed += 1
            fps_frame_count += 1
            frame_age = packet.age_ms
            latencies_ms.append(frame_age)

            # Analyze optical quality (Phase 3)
            latest_quality = quality_analyzer.analyze(packet.frame)

            # Measure consumer display FPS every 0.5s
            elapsed = loop_now - fps_timer
            if elapsed >= 0.5:
                display_fps = fps_frame_count / elapsed
                fps_timer = loop_now
                fps_frame_count = 0

            # Render overlay if running in GUI mode
            if not headless:
                hud_frame = draw_diagnostic_hud(
                    frame=packet.frame,
                    source_id=packet.source_id,
                    actual_res=camera.get_actual_resolution(),
                    ingestion_fps=camera.get_actual_fps(),
                    display_fps=display_fps,
                    frame_age_ms=frame_age,
                    buffer_stats=camera.get_buffer_stats(),
                    quality=latest_quality,
                )
                cv2.imshow(window_name, hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.info("User requested termination.")
                    break
                elif key in (ord("m"), ord("M")):
                    if hasattr(camera, "toggle_flip_horizontal"):
                        camera.toggle_flip_horizontal()
                elif key in (ord("s"), ord("S")):
                    sanitized_name = str(source_identifier).replace("://", "_").replace("/", "_").replace(":", "_")
                    snapshot_filename = f"snapshot_{sanitized_name}_{int(time.time())}.png"
                    cv2.imwrite(snapshot_filename, packet.frame)
                    logger.info("Snapshot saved to %s", snapshot_filename)

    finally:
        total_time = time.monotonic() - start_time
        camera.close()
        if not headless:
            cv2.destroyAllWindows()

    # Benchmark Summary
    avg_display_fps = total_frames_displayed / max(0.001, total_time)
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    buffer_stats = camera.get_buffer_stats()

    print("\n" + "=" * 60)
    print("      DRIVEGUARD CAMERA BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Camera Source         : {source_identifier}")
    print(f"Negotiated Resolution : {actual_res[0]}x{actual_res[1]}")
    print(f"Run Duration          : {total_time:.2f} seconds")
    print(f"Total Captured Frames : {buffer_stats['total_pushed']}")
    print(f"Total Rendered Frames : {total_frames_displayed}")
    print(f"Dropped Backlog Frames: {buffer_stats['total_dropped']}")
    print(f"Average Display FPS   : {avg_display_fps:.2f} FPS")
    print(f"Hardware Ingestion FPS: {camera.get_actual_fps():.2f} FPS")
    print(f"Mean Frame Latency    : {avg_latency:.2f} ms")
    print(f"95th Percentile Age   : {p95_latency:.2f} ms")
    if latest_quality is not None:
        print(f"Optical Quality Status: {latest_quality.status.value}")
        print(f"Mean Luminance        : {latest_quality.mean_luminance:.1f} ({latest_quality.lighting.value})")
        print(f"Sharpness Score       : {latest_quality.blur_variance:.1f} ({latest_quality.blur.value})")
        print(f"Inference Viability   : {'VIABLE' if latest_quality.is_viable_for_inference else 'UNRELIABLE / DEGRADED'}")
    print("=" * 60 + "\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Camera Benchmark (Webcam & Phone Stream)")
    parser.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Phone stream URL (e.g. http://192.168.1.50:8080/video or rtsp://...)",
    )
    parser.add_argument("--width", type=int, default=1280, help="Requested width (default: 1280)")
    parser.add_argument("--height", type=int, default=720, help="Requested height (default: 720)")
    parser.add_argument("--fps", type=int, default=30, help="Requested FPS (default: 30)")
    parser.add_argument(
        "--backend",
        type=str,
        default="DSHOW",
        choices=["DSHOW", "MSMF", "ANY"],
        help="Windows capture backend for webcams (default: DSHOW)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 for indefinite until 'q' pressed)",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying GUI window (for automated benchmarking)",
    )
    parser.add_argument(
        "--flip-h",
        action="store_true",
        help="Horizontally flip / mirror video (corrects selfie camera flip)",
    )
    parser.add_argument(
        "--flip-v",
        action="store_true",
        help="Vertically flip video (corrects upside down phone mounts)",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="Probe and list all available local video capture devices on Windows",
    )

    args = parser.parse_args()

    if args.probe:
        devices = discover_local_devices(max_devices_to_probe=4)
        print(f"Available local capture device indices: {devices}")
        return

    source = args.url if args.url else args.device

    sys.exit(
        run_benchmark(
            source_identifier=source,
            target_width=args.width,
            target_height=args.height,
            target_fps=args.fps,
            backend_str=args.backend,
            duration_sec=args.duration,
            headless=args.headless,
            flip_h=args.flip_h,
            flip_v=args.flip_v,
        )
    )


if __name__ == "__main__":
    main()
