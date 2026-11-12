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

from backend.app.camera.camera_config import make_filename_safe, sanitize_source_identifier
from backend.app.camera.camera_manager import create_camera, discover_local_devices
from backend.app.camera.frame_quality import FrameQualityAnalyzer, FrameQualityResult, QualityStatus
from backend.app.features import DriverFeatureExtractor, DriverFeatures
from backend.app.intelligence import (
    DrowsinessAssessment,
    DrowsinessEngine,
    DrowsinessState,
)
from backend.app.vision.face_landmarker import (
    DriveGuardFaceLandmarker,
    FaceLandmarkResult,
    FaceLandmarkStatus,
)

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
    landmark_result: Optional[FaceLandmarkResult] = None,
    driver_features: Optional[DriverFeatures] = None,
    assessment: Optional[DrowsinessAssessment] = None,
    draw_mesh: bool = False,
    draw_eyes: bool = True,
) -> np.ndarray:
    """
    Renders a semi-transparent diagnostic overlay with real-time performance indicators,
    optical quality metrics (Phase 3), facial landmark tracking (Phase 4),
    physiological feature extraction (Phase 5), and drowsiness intelligence state (Phase 6).
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]

    # Create semi-transparent top banner (height 155 to accommodate Phase 6 intelligence)
    banner_height = 155
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), -1)  # Slate-900
    cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)

    # Color indicator for FPS health
    if ingestion_fps >= 25.0:
        fps_color = (34, 197, 94)  # Emerald-500 (Green)
        fps_text = "STABLE"
    elif ingestion_fps >= 15.0:
        fps_color = (234, 179, 8)  # Amber-500 (Yellow)
        fps_text = "JITTER"
    else:
        fps_color = (239, 68, 68)  # Red-500 (Alert)
        fps_text = "LOW FPS"

    font = cv2.FONT_HERSHEY_SIMPLEX

    # Title
    cv2.putText(
        canvas,
        "DriveGuard — Drowsiness Intelligence Engine (Phase 6)",
        (16, 24),
        font,
        0.56,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Top-right status badges
    fps_badge = f"[{fps_text} | {ingestion_fps:.1f} FPS]"
    cv2.putText(canvas, fps_badge, (width - 230, 24), font, 0.50, fps_color, 2, cv2.LINE_AA)

    # Optical Quality Badge
    if quality is not None:
        if quality.status == QualityStatus.EXCELLENT:
            qual_color = (34, 197, 94)
            qual_text = "[OPTICAL: OK]"
        elif quality.status in (QualityStatus.POOR_LIGHTING, QualityStatus.BLURRY):
            qual_color = (234, 179, 8)
            qual_text = f"[{quality.status.value}]"
        else:
            qual_color = (239, 68, 68)
            qual_text = "[DEGRADED]"
        cv2.putText(canvas, qual_text, (width - 390, 24), font, 0.50, qual_color, 2, cv2.LINE_AA)

    # Landmark Tracking Badge (Phase 4)
    if landmark_result is not None:
        if landmark_result.status == FaceLandmarkStatus.TRACKING:
            lm_color = (34, 197, 94)
            lm_text = f"[AI: TRACKING | {landmark_result.inference_latency_ms:.1f}ms]"
        elif landmark_result.status == FaceLandmarkStatus.FACE_LOST_GRACE:
            lm_color = (234, 179, 8)
            lm_text = "[AI: FACE LOST (GRACE)]"
        else:
            lm_color = (239, 68, 68)
            lm_text = f"[AI: {landmark_result.status.value}]"
        cv2.putText(canvas, lm_text, (width - 660, 24), font, 0.50, lm_color, 2, cv2.LINE_AA)

    # Drowsiness State Badge (Phase 6)
    if assessment is not None:
        state_color = assessment.status_color_bgr
        state_badge = f"[{assessment.state.value}]"
        cv2.putText(canvas, state_badge, (width - 860, 24), font, 0.50, state_color, 2, cv2.LINE_AA)

    # Line 2: Telemetry Metrics
    w, h = actual_res
    telemetry_line_1 = (
        f"Source: {source_id} | "
        f"Resolution: {w}x{h} | "
        f"Ingestion: {ingestion_fps:.1f} FPS | "
        f"Display: {display_fps:.1f} FPS"
    )
    cv2.putText(canvas, telemetry_line_1, (16, 46), font, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

    # Line 3: Optical Quality & Landmark Latency
    ai_info = f"AI Latency: {landmark_result.inference_latency_ms:.1f}ms" if landmark_result else "AI: N/A"
    if quality is not None:
        telemetry_line_2 = (
            f"Luminance: {quality.mean_luminance:.1f} ({quality.lighting.value}) | "
            f"Sharpness: {quality.blur_variance:.1f} ({quality.blur.value}) | "
            f"Latency: {frame_age_ms:.1f}ms | {ai_info}"
        )
    else:
        telemetry_line_2 = f"Frame Latency: {frame_age_ms:.1f}ms | {ai_info}"
    cv2.putText(canvas, telemetry_line_2, (16, 66), font, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    # Line 4: Physiological Features (Phase 5)
    if driver_features is not None and driver_features.has_face:
        ear = driver_features.ear
        pose = driver_features.head_pose
        mar = driver_features.mar
        blink = driver_features.blink
        perclos = driver_features.perclos

        telemetry_line_3 = (
            f"EAR: {ear.compensated_ear:.2f} (L:{ear.left_ear:.2f} R:{ear.right_ear:.2f}) | "
            f"PERCLOS: {perclos.perclos_pct:.1f}% | "
            f"MAR: {mar.mar:.2f} (Yawns: {mar.total_yawn_count}) | "
            f"Pose: {pose.zone.value} (P:{pose.pitch_deg:+.0f}\xb0 Y:{pose.yaw_deg:+.0f}\xb0) | "
            f"Blinks: {blink.total_blink_count}"
        )
        features_color = (56, 189, 248)  # Sky-400
    elif driver_features is not None:
        telemetry_line_3 = f"Physiological Features: [NO FACE DETECTED — {driver_features.face_status.value}]"
        features_color = (148, 163, 184)  # Slate-400
    else:
        telemetry_line_3 = "Physiological Features: N/A"
        features_color = (148, 163, 184)
    cv2.putText(canvas, telemetry_line_3, (16, 88), font, 0.42, features_color, 1, cv2.LINE_AA)

    # Line 5: Drowsiness Intelligence (Phase 6)
    if assessment is not None:
        state_color = assessment.status_color_bgr
        reason_text = assessment.primary_reason_text
        intel_line = (
            f"Drowsiness: {assessment.state.value} (Sev {assessment.severity}) | "
            f"Confidence: {assessment.confidence.overall_confidence:.2f} ({assessment.confidence.tier.value}) | "
            f"Reason: {reason_text}"
        )
        cv2.putText(canvas, intel_line, (16, 110), font, 0.42, state_color, 1, cv2.LINE_AA)
    else:
        intel_line = "Drowsiness Intelligence: N/A"
        cv2.putText(canvas, intel_line, (16, 110), font, 0.42, (148, 163, 184), 1, cv2.LINE_AA)

    # Line 6: Controls
    dropped = buffer_stats.get("total_dropped", 0)
    mesh_str = "ON" if draw_mesh else "OFF"
    eyes_str = "ON" if draw_eyes else "OFF"
    controls_line = (
        f"Dropped: {dropped} | "
        f"'q'/ESC: exit | 'm': mirror | 'w': wireframe ({mesh_str}) | 'e': eyes ({eyes_str}) | 's': snapshot"
    )
    cv2.putText(canvas, controls_line, (16, 132), font, 0.40, (100, 116, 139), 1, cv2.LINE_AA)

    # Subtle status border around the full viewport
    border_color = assessment.status_color_bgr if assessment else (34, 197, 94)
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), border_color, 4)

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
    safe_identifier = sanitize_source_identifier(source_identifier)
    logger.info("Initializing camera adapter for source '%s'...", safe_identifier)

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
        logger.error("Failed to open camera source '%s'.", safe_identifier)
        return 1

    start_time = time.monotonic()
    try:
        actual_res = camera.get_actual_resolution()
        window_name = f"DriveGuard Diagnostics [{safe_identifier}]"

        if not headless:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            # Avoid zero or invalid dimensions
            w = actual_res[0] if actual_res[0] > 0 else target_width
            h = actual_res[1] if actual_res[1] > 0 else target_height
            cv2.resizeWindow(window_name, w, h)

        quality_analyzer = FrameQualityAnalyzer()
        landmarker = DriveGuardFaceLandmarker()
        feature_extractor = DriverFeatureExtractor()
        drowsiness_engine = DrowsinessEngine()

        # Visualization toggles (Phase 4)
        draw_mesh: bool = False
        draw_eyes: bool = True

        # Performance tracking metrics
        start_time = time.monotonic()
        total_frames_displayed = 0
        display_fps = 0.0
        fps_timer = time.monotonic()
        fps_frame_count = 0
        latencies_ms = []
        inference_latencies_ms = []
        total_face_hits = 0
        latest_quality: Optional[FrameQualityResult] = None
        latest_landmark: Optional[FaceLandmarkResult] = None
        latest_features: Optional[DriverFeatures] = None
        latest_assessment: Optional[DrowsinessAssessment] = None

        logger.info("Starting live capture loop. Press 'q' or ESC to terminate.")

        while True:
            loop_now = time.monotonic()

            # Auto-exit if duration is set (used for automated testing)
            if duration_sec > 0 and (loop_now - start_time) >= duration_sec:
                logger.info("Target duration of %.1fs reached.", duration_sec)
                break

            success, packet = camera.read(timeout_sec=0.1)
            if not success or packet is None:
                if not headless:
                    key = cv2.waitKey(1) & 0xFF
                    if key in (27, ord("q"), ord("Q")):
                        logger.info("User requested termination.")
                        break
                time.sleep(0.005)
                continue

            total_frames_displayed += 1
            fps_frame_count += 1
            frame_age = packet.age_ms
            latencies_ms.append(frame_age)

            # Analyze optical quality (Phase 3)
            latest_quality = quality_analyzer.analyze(packet.frame)

            # Extract 478 3D facial landmarks (Phase 4)
            latest_landmark = landmarker.detect(packet)
            inference_latencies_ms.append(latest_landmark.inference_latency_ms)
            if latest_landmark.has_face:
                total_face_hits += 1

            # Check mirror status from camera adapter
            is_mirrored = getattr(camera, "is_flip_horizontal", False)

            # Extract physiological driver features (Phase 5)
            f_w = actual_res[0] if actual_res[0] > 0 else packet.frame.shape[1]
            f_h = actual_res[1] if actual_res[1] > 0 else packet.frame.shape[0]
            latest_features = feature_extractor.extract(
                latest_landmark,
                frame_width=f_w,
                frame_height=f_h,
                is_mirrored=is_mirrored,
            )

            # Evaluate Drowsiness Intelligence Engine (Phase 6)
            latest_assessment = drowsiness_engine.evaluate(
                features=latest_features,
                quality=latest_quality,
            )

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
                    landmark_result=latest_landmark,
                    driver_features=latest_features,
                    assessment=latest_assessment,
                    draw_mesh=draw_mesh,
                    draw_eyes=draw_eyes,
                )
                hud_frame = landmarker.draw_landmarks(
                    frame=hud_frame,
                    result=latest_landmark,
                    draw_mesh=draw_mesh,
                    draw_eyes=draw_eyes,
                    draw_irises=True,
                )
                cv2.imshow(window_name, hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.info("User requested termination.")
                    break
                elif key in (ord("m"), ord("M")):
                    if hasattr(camera, "toggle_flip_horizontal"):
                        camera.toggle_flip_horizontal()
                elif key in (ord("w"), ord("W")):
                    draw_mesh = not draw_mesh
                    logger.info("Face mesh wireframe toggled to: %s", draw_mesh)
                elif key in (ord("e"), ord("E")):
                    draw_eyes = not draw_eyes
                    logger.info("Eye contour overlay toggled to: %s", draw_eyes)
                elif key in (ord("s"), ord("S")):
                    short_safe_tag = make_filename_safe(safe_identifier)
                    snapshot_filename = f"snapshot_{short_safe_tag}_{int(time.time() * 1000)}.png"
                    save_ok = cv2.imwrite(snapshot_filename, hud_frame)
                    if save_ok:
                        logger.info("Diagnostic HUD snapshot saved to %s", snapshot_filename)
                    else:
                        logger.error(
                            "Failed to save snapshot to %s (cv2.imwrite returned False).",
                            snapshot_filename,
                        )

    finally:
        total_time = time.monotonic() - start_time
        buffer_stats = camera.get_buffer_stats()
        hardware_fps = camera.get_actual_fps()
        camera.close()
        if "landmarker" in locals() and landmarker is not None:
            landmarker.close()
        if not headless:
            cv2.destroyAllWindows()

    # Benchmark Summary
    avg_display_fps = total_frames_displayed / max(0.001, total_time)
    avg_latency = float(np.mean(latencies_ms)) if latencies_ms else 0.0
    p95_latency = float(np.percentile(latencies_ms, 95)) if latencies_ms else 0.0
    avg_ai_ms = float(np.mean(inference_latencies_ms)) if inference_latencies_ms else 0.0
    p95_ai_ms = float(np.percentile(inference_latencies_ms, 95)) if inference_latencies_ms else 0.0
    face_hit_rate = (total_face_hits / max(1, total_frames_displayed)) * 100.0

    print("\n" + "=" * 60)
    print("      DRIVEGUARD CAMERA & AI BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"Camera Source         : {safe_identifier}")
    print(f"Negotiated Resolution : {actual_res[0]}x{actual_res[1]}")
    print(f"Run Duration          : {total_time:.2f} seconds")
    print(f"Total Captured Frames : {buffer_stats['total_pushed']}")
    print(f"Total Rendered Frames : {total_frames_displayed}")
    print(f"Dropped Backlog Frames: {buffer_stats['total_dropped']}")
    print(f"Average Display FPS   : {avg_display_fps:.2f} FPS")
    print(f"Hardware Ingestion FPS: {hardware_fps:.2f} FPS")
    print(f"Mean Frame Latency    : {avg_latency:.2f} ms")
    print(f"95th Percentile Age   : {p95_latency:.2f} ms")
    if latest_quality is not None:
        print(f"Optical Quality Status: {latest_quality.status.value}")
        print(f"Mean Luminance        : {latest_quality.mean_luminance:.1f} ({latest_quality.lighting.value})")
        print(f"Sharpness Score       : {latest_quality.blur_variance:.1f} ({latest_quality.blur.value})")
        print(f"Inference Viability   : {'VIABLE' if latest_quality.is_viable_for_inference else 'UNRELIABLE / DEGRADED'}")
    if latest_landmark is not None:
        print(f"Face Tracking Status  : {latest_landmark.status.value}")
        print(f"Face Detection Rate   : {face_hit_rate:.1f}% ({total_face_hits}/{total_frames_displayed} frames)")
        print(f"Mean AI Inference Lat : {avg_ai_ms:.2f} ms")
        print(f"95th Percentile AI Lat: {p95_ai_ms:.2f} ms")
    if latest_features is not None:
        print("-" * 60)
        print("      PHYSIOLOGICAL FEATURES (PHASE 5)")
        print("-" * 60)
        print(f"Compensated EAR       : {latest_features.ear.compensated_ear:.3f} (Left: {latest_features.ear.left_ear:.3f}, Right: {latest_features.ear.right_ear:.3f})")
        print(f"Eye Closure State     : {'CLOSED' if latest_features.ear.is_closed else 'OPEN'}")
        print(f"Rolling PERCLOS (60s) : {latest_features.perclos.perclos_pct:.1f}% ({'RELIABLE' if latest_features.perclos.is_reliable else 'UNRELIABLE - INSUFFICIENT COVERAGE'})")
        print(f"Mouth Aspect Ratio    : {latest_features.mar.mar:.3f} (Open: {latest_features.mar.is_mouth_open})")
        print(f"Total Authentic Yawns : {latest_features.mar.total_yawn_count}")
        print(f"Blink Tracking        : {latest_features.blink.total_blink_count} blinks ({latest_features.blink.prolonged_blink_count} prolonged)")
        if latest_features.blink.recent_blink_rcr is not None:
            print(f"Recent Blink Kinematics: Duration={latest_features.blink.recent_blink_duration_ms:.1f}ms, RCR={latest_features.blink.recent_blink_rcr:.2f}")
        print(f"Head Pose Orientation : Zone={latest_features.head_pose.zone.value} (Pitch={latest_features.head_pose.pitch_deg:+.1f}\xb0, Yaw={latest_features.head_pose.yaw_deg:+.1f}\xb0, Roll={latest_features.head_pose.roll_deg:+.1f}\xb0)")
    print("=" * 60 + "\n")

    if total_frames_displayed == 0:
        logger.error("No frames were captured or displayed from source '%s'.", safe_identifier)
        return 1

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
