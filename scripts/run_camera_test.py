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
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

# Add project root to sys.path to allow running directly from anywhere
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import cv2
import numpy as np

from backend.app.ai.blink_kinematics import (
    BlinkClassification as AIBlinkClassification,
    BlinkEvent as AIBlinkEvent,
)
from backend.app.ai.geometry import PoseZone
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
    create_immutable_profile_snapshot,
)
from backend.app.ai.eye_shadow_observer import (
    BinocularShadowTelemetry,
    EyeCNNInvocationPolicy,
    EyeCNNShadowWorker,
)
from backend.app.ai.profile_adapter import dto_to_profile, profile_to_dto
from backend.app.alerts import (
    AlertDispatcher,
    AlertState,
    AlertStateManager,
    AudioWarningEngine,
    audio_settings_store,
    map_drowsiness_state_to_alert_state,
)
from backend.app.camera.camera_config import make_filename_safe, sanitize_source_identifier
from backend.app.camera.camera_manager import create_camera, discover_local_devices
from backend.app.camera.frame_quality import FrameQualityAnalyzer, FrameQualityResult, QualityStatus
from backend.app.camera.positioning_guide import draw_positioning_guide, evaluate_camera_position
from backend.app.camera.session_recorder import SessionRecorder
from backend.app.features import DriverFeatureExtractor, DriverFeatures
from backend.app.intelligence import (
    DrowsinessAssessment,
    DrowsinessEngine,
    DrowsinessState,
)
from backend.app.schemas.profile import DriverProfileDTO
from backend.app.services.profile_store import profile_store
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


def truncate_middle(text: str, max_chars: int = 32) -> str:
    """Truncates text in the middle if it exceeds max_chars."""
    if len(text) <= max_chars:
        return text
    half = max(1, (max_chars - 3) // 2)
    return text[:half] + "..." + text[-half:]


def put_fitted_text(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    font: int,
    scale: float,
    color: tuple,
    max_w: int,
    thickness: int = 1,
) -> int:
    """
    Renders text guaranteed not to exceed max_w pixels.
    If text exceeds max_w, dynamically truncates with ellipsis.
    Returns rendered text width.
    """
    (tw, _), _ = cv2.getTextSize(text, font, scale, thickness)
    if tw <= max_w:
        cv2.putText(canvas, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)
        return tw

    ellipsis_w, _ = cv2.getTextSize("...", font, scale, thickness)[0]
    avail_w = max_w - ellipsis_w
    if avail_w <= 10:
        return 0

    low, high = 1, len(text)
    best_len = 1
    while low <= high:
        mid = (low + high) // 2
        sub_w = cv2.getTextSize(text[:mid], font, scale, thickness)[0][0]
        if sub_w <= avail_w:
            best_len = mid
            low = mid + 1
        else:
            high = mid - 1

    truncated = text[:best_len] + "..."
    cv2.putText(canvas, truncated, (x, y), font, scale, color, thickness, cv2.LINE_AA)
    return cv2.getTextSize(truncated, font, scale, thickness)[0][0]


def commit_and_activate_profile(
    profile: DriverBehaviourProfile,
    feature_extractor: DriverFeatureExtractor,
    store: Optional[ProfileStore] = None,
) -> Tuple[bool, str]:
    """
    Coordinated transaction: validates feasibility, persists atomically to disk,
    and activates on the runtime feature extractor with automatic disk rollback on activation failure.
    If a valid profile already existed, rolls back to exact previous content; otherwise unlinks new file.
    Rejects any new commit while another profile transaction is pending safe-latch activation.
    """
    target_store = store or profile_store

    # 1. Reject conflicting recalibration transactions while one is pending
    if getattr(feature_extractor, "has_pending_profile", False):
        return False, (
            f"Cannot commit profile for driver '{profile.driver_id}': "
            "another profile transaction is currently pending safe-latch activation."
        )

    # 2. Validate feasibility
    is_feasible, reason = feature_extractor.validate_profile_feasibility(profile)
    if not is_feasible:
        return False, f"Infeasible profile: {reason}"

    # 3. Immutable snapshot and DTO preparation
    try:
        snapshot = create_immutable_profile_snapshot(profile)
        dto = profile_to_dto(profile)
    except Exception as prep_err:
        return False, f"Profile preparation error: {prep_err}"

    # 4. Read previous persisted content if it exists, to support exact restoration
    # Since any pending switch was rejected above, existing disk content is true baseline
    previous_content: Optional[str] = None
    try:
        target_path = target_store._file_path(profile.driver_id)
        if target_path.is_file():
            previous_content = target_path.read_text(encoding="utf-8")
    except Exception as read_err:
        logger.warning(f"Could not read previous profile for '{profile.driver_id}': {read_err}")

    # 5. Disk persistence
    try:
        target_store.save_profile(dto)
    except Exception as save_err:
        return False, f"Persistence error: {save_err}"

    # One-time rollback helper
    rollback_executed = False

    def do_rollback() -> None:
        nonlocal rollback_executed
        if not rollback_executed:
            rollback_executed = True
            target_store.rollback_new_profile(profile.driver_id, previous_content=previous_content)

    # 6. Runtime activation with fingerprint-bound disk rollback on failure
    try:
        # Register fingerprint-bound rollback callback for deferred safe-latch activation
        feature_extractor.set_pending_rollback_callback(
            do_rollback,
            fingerprint=snapshot.fingerprint,
        )
        activated = feature_extractor.set_profile(snapshot)
        if not activated:
            do_rollback()
            feature_extractor.set_pending_rollback_callback(None)
            return False, "Runtime activation rejected profile; disk persistence rolled back."
    except Exception as act_err:
        try:
            do_rollback()
        except Exception as rb_err:
            logger.critical(f"Critical failure during persistence rollback: {rb_err}")
            feature_extractor.set_pending_rollback_callback(None)
            return False, f"Activation error: {act_err}; Rollback failed: {rb_err}"
        feature_extractor.set_pending_rollback_callback(None)
        return False, f"Runtime activation error: {act_err}; disk persistence rolled back."

    if getattr(feature_extractor, "has_pending_profile", False):
        return True, "Profile persisted and queued for deferred activation at next safe latch."

    return True, "Profile successfully persisted and activated."


def draw_pill(
    canvas: np.ndarray,
    text: str,
    x: int,
    y: int,
    fg_color: tuple,
    bg_color: tuple,
    border_color: Optional[tuple] = None,
    scale: float = 0.40,
    thickness: int = 1,
    pad_x: int = 7,
    pad_y: int = 3,
    align: str = "right",
) -> Tuple[int, int, int, int]:
    """
    Renders a styled pill badge and returns its (x1, y1, x2, y2) bounds.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    pill_w = tw + pad_x * 2
    pill_h = th + pad_y * 2

    if align == "right":
        x2 = x
        x1 = x - pill_w
    else:
        x1 = x
        x2 = x + pill_w

    y1 = y - th - pad_y
    y2 = y1 + pill_h

    cv2.rectangle(canvas, (x1, y1), (x2, y2), bg_color, -1)
    if border_color is not None:
        cv2.rectangle(canvas, (x1, y1), (x2, y2), border_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x1 + pad_x, y), font, scale, fg_color, thickness, cv2.LINE_AA)
    return x1, y1, x2, y2


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
    calibration_info: Optional[tuple] = None,
    calibration_banner: Optional[str] = None,
    calibrated_profile_name: Optional[str] = None,
    recording_info: Optional[Tuple[bool, float, int]] = None,
    show_position_guide: bool = False,
    alert_state: Optional[AlertState] = None,
    is_recovering: bool = False,
    audio_warnings_enabled: bool = True,
    audio_feedback_banner: Optional[str] = None,
    shadow_telemetry: Optional[BinocularShadowTelemetry] = None,
    shadow_enabled: bool = False,
) -> np.ndarray:
    """
    Renders a semi-transparent diagnostic overlay with real-time performance indicators,
    optical quality metrics (Phase 3), facial landmark tracking (Phase 4),
    physiological feature extraction (Phase 5), drowsiness intelligence state (Phase 6),
    alert state machine / audio warning indicators (Phase 8),
    and passive MobileNetV3 eye CNN shadow observer diagnostics (Phase B1).
    Guarantees responsive layout with zero text overlaps across all resolutions.
    """
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX

    # 1. Top Panel (Banner Height 192px with Glassmorphic Slate-900 Overlay)
    banner_height = 192
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, banner_height), (15, 23, 42), -1)  # Slate-900
    cv2.addWeighted(overlay, 0.82, canvas, 0.18, 0, canvas)
    # Bottom border line of top banner
    cv2.line(canvas, (0, banner_height), (width, banner_height), (51, 65, 85), 1)

    # -------------------------------------------------------------
    # ROW 1 (Y = 22): Title (Left) + System Status Badges (Right)
    # -------------------------------------------------------------
    y_row1 = 22
    # Left: App Title & Subtitle
    title_text = "DriveGuard AI" if width >= 700 else "DriveGuard"
    (tw, th), _ = cv2.getTextSize(title_text, font, 0.54, 2)
    cv2.putText(canvas, title_text, (16, y_row1), font, 0.54, (255, 255, 255), 2, cv2.LINE_AA)

    version_text = "Phase 8"
    (vw, vh), _ = cv2.getTextSize(version_text, font, 0.38, 1)
    vx = 16 + tw + 8
    cv2.putText(canvas, version_text, (vx, y_row1), font, 0.38, (148, 163, 184), 1, cv2.LINE_AA)
    title_reserved_x = vx + vw + 16

    # Dynamic Right-to-Left Status Badges
    cur_x = width - 16

    # 0. Recording Badge (if recording active)
    if recording_info is not None and recording_info[0]:
        rec_active, rec_elapsed, rec_frames = recording_info
        rec_m = int(rec_elapsed // 60)
        rec_s = int(rec_elapsed % 60)
        rec_text = f"[REC {rec_m:02d}:{rec_s:02d} | {rec_frames}f]" if width >= 800 else f"[REC {rec_m:02d}:{rec_s:02d}]"
        rec_color = (239, 68, 68)  # Red-500
        rec_bg = (50, 10, 10)
        bx0, _, _, _ = draw_pill(canvas, rec_text, cur_x, y_row1, rec_color, rec_bg, rec_color, 0.40, 1, 7, 3, "right")
        cur_x = bx0 - 8

    # 1. FPS Badge
    if ingestion_fps >= 25.0:
        fps_color = (34, 197, 94)  # Emerald-500
        fps_bg = (12, 35, 25)
        fps_text = "[STABLE | " + f"{ingestion_fps:.1f} FPS]" if width >= 800 else f"[{ingestion_fps:.1f} FPS]"
    elif ingestion_fps >= 15.0:
        fps_color = (234, 179, 8)  # Amber-500
        fps_bg = (35, 30, 12)
        fps_text = "[JITTER | " + f"{ingestion_fps:.1f} FPS]" if width >= 800 else f"[{ingestion_fps:.1f} FPS]"
    else:
        fps_color = (239, 68, 68)  # Red-500
        fps_bg = (40, 15, 15)
        fps_text = "[LOW | " + f"{ingestion_fps:.1f} FPS]" if width >= 800 else f"[{ingestion_fps:.1f} FPS]"

    bx1, _, _, _ = draw_pill(canvas, fps_text, cur_x, y_row1, fps_color, fps_bg, fps_color, 0.40, 1, 7, 3, "right")
    cur_x = bx1 - 8

    # 2. Optical Quality Badge
    if quality is not None:
        if quality.status == QualityStatus.EXCELLENT:
            qual_color = (34, 197, 94)
            qual_bg = (12, 35, 25)
            qual_text = "[OPTICAL: OK]"
        elif quality.status in (QualityStatus.POOR_LIGHTING, QualityStatus.BLURRY):
            qual_color = (234, 179, 8)
            qual_bg = (35, 30, 12)
            qual_text = f"[{quality.status.value}]"
        else:
            qual_color = (239, 68, 68)
            qual_bg = (40, 15, 15)
            qual_text = "[DEGRADED]"
        (qw, _), _ = cv2.getTextSize(qual_text, font, 0.40, 1)
        if cur_x - (qw + 22) > title_reserved_x + 220:
            bx1, _, _, _ = draw_pill(canvas, qual_text, cur_x, y_row1, qual_color, qual_bg, qual_color, 0.40, 1, 7, 3, "right")
            cur_x = bx1 - 8

    # 3. Landmark Tracking Badge (Phase 4)
    if landmark_result is not None:
        if landmark_result.status == FaceLandmarkStatus.TRACKING:
            lm_color = (34, 197, 94)
            lm_bg = (12, 35, 25)
            lm_text = f"[AI: TRACKING {landmark_result.inference_latency_ms:.0f}ms]" if width >= 720 else "[AI: OK]"
        elif landmark_result.status == FaceLandmarkStatus.FACE_LOST_GRACE:
            lm_color = (234, 179, 8)
            lm_bg = (35, 30, 12)
            lm_text = "[AI: GRACE]"
        else:
            lm_color = (239, 68, 68)
            lm_bg = (40, 15, 15)
            lm_text = f"[AI: {landmark_result.status.value}]"
        (lw, _), _ = cv2.getTextSize(lm_text, font, 0.40, 1)
        if cur_x - (lw + 22) > title_reserved_x + 140:
            bx1, _, _, _ = draw_pill(canvas, lm_text, cur_x, y_row1, lm_color, lm_bg, lm_color, 0.40, 1, 7, 3, "right")
            cur_x = bx1 - 8

    # 3b. Audio Warnings Badge (Phase 8)
    if audio_warnings_enabled:
        audio_color = (34, 197, 94)  # Emerald-500
        audio_bg = (12, 35, 25)
        audio_text = "[AUDIO: ON]"
    else:
        audio_color = (148, 163, 184)  # Slate-400
        audio_bg = (30, 41, 59)
        audio_text = "[AUDIO: OFF]"
    (aw, _), _ = cv2.getTextSize(audio_text, font, 0.40, 1)
    if cur_x - (aw + 22) > title_reserved_x + 80:
        bx1, _, _, _ = draw_pill(canvas, audio_text, cur_x, y_row1, audio_color, audio_bg, audio_color, 0.40, 1, 7, 3, "right")
        cur_x = bx1 - 8

    # 3c. CNN Shadow Badge (Phase B1)
    if shadow_enabled:
        sh_color = (56, 189, 248)  # Sky-400
        sh_bg = (15, 35, 55)
        sh_text = "[SHADOW: ON]"
    else:
        sh_color = (148, 163, 184)  # Slate-400
        sh_bg = (30, 41, 59)
        sh_text = "[SHADOW: OFF]"
    (sw_w, _), _ = cv2.getTextSize(sh_text, font, 0.40, 1)
    if cur_x - (sw_w + 22) > title_reserved_x + 80:
        bx1, _, _, _ = draw_pill(canvas, sh_text, cur_x, y_row1, sh_color, sh_bg, sh_color, 0.40, 1, 7, 3, "right")
        cur_x = bx1 - 8

    # 4. Alert State Badge (Phase 8)
    effective_alert = alert_state
    if effective_alert is None and assessment is not None:
        effective_alert = map_drowsiness_state_to_alert_state(assessment.state)
    if effective_alert is None:
        effective_alert = AlertState.NORMAL

    state_color = effective_alert.hud_color_bgr
    rec_suffix = " | RECOVERING" if is_recovering else ""
    state_badge = f"[{effective_alert.value}{rec_suffix}]"
    state_bg = (
        (6, 78, 59) if effective_alert == AlertState.NORMAL
        else ((35, 28, 10) if effective_alert == AlertState.ATTENTION
        else ((45, 20, 10) if effective_alert == AlertState.WARNING
        else (60, 10, 10)))
    )
    (sw, _), _ = cv2.getTextSize(state_badge, font, 0.42, 1)
    if cur_x - (sw + 22) >= title_reserved_x:
        draw_pill(canvas, state_badge, cur_x, y_row1, state_color, state_bg, state_color, 0.42, 1, 8, 4, "right")

    # -------------------------------------------------------------
    # ROW 2 (Y = 47): Ocular Profile Status (Left) & Stream Info (Right)
    # -------------------------------------------------------------
    y_row2 = 47
    if calibrated_profile_name:
        prof_badge = f"[PROFILE: {calibrated_profile_name} — OCULAR ACTIVE]" if width >= 850 else f"[PROFILE: {calibrated_profile_name}]"
        p_fg = (56, 189, 248)  # Sky-400
        p_bg = (15, 35, 55)
        p_border = (56, 189, 248)
    else:
        prof_badge = "[OCULAR MONITORING OFF — CALIBRATION REQUIRED (Press 'c')]" if width >= 800 else "[OCULAR OFF: CALIBRATE]"
        p_fg = (234, 179, 8)  # Amber-500
        p_bg = (35, 28, 10)
        p_border = (234, 179, 8)

    _, _, px2, _ = draw_pill(canvas, prof_badge, 16, y_row2, p_fg, p_bg, p_border, 0.40, 1, 7, 3, "left")
    prof_end_x = px2 + 16

    # Stream Telemetry on Right of Row 2
    w_res, h_res = actual_res
    avail_stream_w = width - 16 - prof_end_x
    if avail_stream_w > 100:
        clean_src = truncate_middle(str(source_id), 18 if width < 1000 else 28)
        stream_str = f"Source: {clean_src} | {w_res}x{h_res} | Ingestion: {ingestion_fps:.1f}fps | Display: {display_fps:.1f}fps"
        (stw, sth), _ = cv2.getTextSize(stream_str, font, 0.39, 1)
        if stw > avail_stream_w:
            stream_str = f"{w_res}x{h_res} | {ingestion_fps:.1f}fps | Latency: {frame_age_ms:.0f}ms"
            (stw, sth), _ = cv2.getTextSize(stream_str, font, 0.39, 1)
        if stw <= avail_stream_w:
            sx = width - 16 - stw
            cv2.putText(canvas, stream_str, (sx, y_row2), font, 0.39, (148, 163, 184), 1, cv2.LINE_AA)

    # Subtle horizontal divider line
    cv2.line(canvas, (16, 57), (width - 16, 57), (30, 41, 59), 1)

    # -------------------------------------------------------------
    # ROW 3 (Y = 76): Physiological Features (Phase 5)
    # -------------------------------------------------------------
    y_row3 = 76
    if driver_features is not None and driver_features.has_face:
        ear = driver_features.ear
        pose = driver_features.head_pose
        mar = driver_features.mar
        blink = driver_features.blink
        perclos = driver_features.perclos

        if calibrated_profile_name:
            pose_str = f"Pose: {pose.zone.value} (rel: P:{pose.rel_pitch_deg:+.0f}\xb0 Y:{pose.rel_yaw_deg:+.0f}\xb0 | raw: P:{pose.pitch_deg:+.0f}\xb0 Y:{pose.yaw_deg:+.0f}\xb0)"
        else:
            pose_str = f"Pose: {pose.zone.value} (P:{pose.pitch_deg:+.0f}\xb0 Y:{pose.yaw_deg:+.0f}\xb0)"

        telemetry_line_3 = (
            f"EAR: {ear.compensated_ear:.2f} (thr:{ear.closure_threshold:.2f} L:{ear.left_ear:.2f} R:{ear.right_ear:.2f}) | "
            f"PERCLOS: {perclos.perclos_pct:.1f}% | "
            f"Blinks: {blink.total_blink_count} | "
            f"MAR: {mar.mar:.2f} (Yawns: {mar.total_yawn_count}) | "
            f"{pose_str}"
        )
        features_color = (56, 189, 248)  # Sky-400
    elif driver_features is not None:
        telemetry_line_3 = f"Physiological Features: [NO FACE DETECTED — {driver_features.face_status.value}]"
        features_color = (148, 163, 184)
    else:
        telemetry_line_3 = "Physiological Features: N/A"
        features_color = (148, 163, 184)

    put_fitted_text(canvas, telemetry_line_3, 16, y_row3, font, 0.41, features_color, width - 32, 1)

    # -------------------------------------------------------------
    # ROW 4 (Y = 98): Alert State & Drowsiness Intelligence (Phase 8)
    # -------------------------------------------------------------
    y_row4 = 98
    effective_alert = alert_state
    if effective_alert is None and assessment is not None:
        effective_alert = map_drowsiness_state_to_alert_state(assessment.state)
    if effective_alert is None:
        effective_alert = AlertState.NORMAL

    state_color = effective_alert.hud_color_bgr
    rec_str = "[RECOVERING] " if is_recovering else ""
    if assessment is not None:
        reason_text = assessment.primary_reason_text
        intel_line = (
            f"Alert State: {effective_alert.value} (Sev {effective_alert.severity}) | "
            f"{rec_str}Conf: {assessment.confidence.overall_confidence:.2f} [{assessment.confidence.tier.value}] | "
            f"Reason: {reason_text}"
        )
        put_fitted_text(canvas, intel_line, 16, y_row4, font, 0.41, state_color, width - 32, 1)
    else:
        intel_line = f"Alert State: {effective_alert.value} (Sev {effective_alert.severity}) | {rec_str}Nominal Monitoring"
        put_fitted_text(canvas, intel_line, 16, y_row4, font, 0.41, (148, 163, 184), width - 32, 1)

    # -------------------------------------------------------------
    # ROW 5 (Y = 120): Optical Quality & Diagnostics
    # -------------------------------------------------------------
    y_row5 = 120
    ai_info = f"AI Latency: {landmark_result.inference_latency_ms:.1f}ms" if landmark_result else "AI: N/A"
    dropped = buffer_stats.get("total_dropped", 0)
    if quality is not None:
        telemetry_line_2 = (
            f"Luminance: {quality.mean_luminance:.1f} ({quality.lighting.value}) | "
            f"Sharpness: {quality.blur_variance:.1f} ({quality.blur.value}) | "
            f"Frame Latency: {frame_age_ms:.1f}ms | {ai_info} | Dropped: {dropped}"
        )
    else:
        telemetry_line_2 = f"Frame Latency: {frame_age_ms:.1f}ms | {ai_info} | Dropped: {dropped}"

    put_fitted_text(canvas, telemetry_line_2, 16, y_row5, font, 0.40, (148, 163, 184), width - 32, 1)

    # -------------------------------------------------------------
    # ROW 6 (Y = 144): CNN Shadow Diagnostics (Phase B1 - Experimental Observer)
    # -------------------------------------------------------------
    y_row6 = 144
    if not shadow_enabled:
        shadow_str = "[EXP] CNN SHADOW: OFF (Press 'x' to enable shadow observer)"
        shadow_color = (100, 116, 139)  # Slate-500
    elif shadow_telemetry is None:
        shadow_str = "[EXP] CNN SHADOW: INITIALIZING / IDLE..."
        shadow_color = (234, 179, 8)  # Amber-500
    elif shadow_telemetry.status_label == "STALE":
        shadow_str = f"[EXP] CNN SHADOW: STALE (>200ms) | Age: {frame_age_ms:.0f}ms | Drops: {shadow_telemetry.worker_queue_depth}"
        shadow_color = (234, 179, 8)
    elif shadow_telemetry.status_label == "ERROR":
        shadow_str = "[EXP] CNN SHADOW: INFERENCE ERROR (Webcam unaffected)"
        shadow_color = (239, 68, 68)  # Red-500
    elif shadow_telemetry.status_label == "QUALITY_REJECTED":
        shadow_str = f"[EXP] CNN SHADOW: CROPS REJECTED (Low quality / occlusion) | Latency: {shadow_telemetry.inference_latency_ms:.1f}ms"
        shadow_color = (148, 163, 184)
    else:
        # ACTIVE
        l = shadow_telemetry.left_eye
        r = shadow_telemetry.right_eye
        l_pred_txt = f"{l.cnn_predicted_class or 'N/A'}"
        if l.cnn_open_prob is not None:
            l_prob = max(l.cnn_open_prob, l.cnn_closed_prob or 0.0)
            l_pred_txt += f"({l_prob:.2f})"
        r_pred_txt = f"{r.cnn_predicted_class or 'N/A'}"
        if r.cnn_open_prob is not None:
            r_prob = max(r.cnn_open_prob, r.cnn_closed_prob or 0.0)
            r_pred_txt += f"({r_prob:.2f})"

        l_agree = "AGREE" if l.is_agreement is True else ("DIFF" if l.is_agreement is False else "N/A")
        r_agree = "AGREE" if r.is_agreement is True else ("DIFF" if r.is_agreement is False else "N/A")
        binoc_tag = "CONSENSUS" if shadow_telemetry.binocular_agreement is True else ("DIVERGENT" if shadow_telemetry.binocular_agreement is False else "PARTIAL")

        shadow_str = (
            f"[EXP] CNN SHADOW: [{binoc_tag}] | "
            f"L: {l.geometric_state} vs CNN:{l_pred_txt} [{l_agree}] | "
            f"R: {r.geometric_state} vs CNN:{r_pred_txt} [{r_agree}] | "
            f"CNN: {shadow_telemetry.inference_latency_ms:.1f}ms (Tot:{shadow_telemetry.total_latency_ms:.1f}ms)"
        )
        shadow_color = (56, 189, 248) if shadow_telemetry.binocular_agreement is True else (234, 179, 8)

    put_fitted_text(canvas, shadow_str, 16, y_row6, font, 0.39, shadow_color, width - 32, 1)

    # -------------------------------------------------------------
    # ROW 7 (Y = 168): Controls Guide
    # -------------------------------------------------------------
    y_row7 = 168
    mesh_str = "ON" if draw_mesh else "OFF"
    eyes_str = "ON" if draw_eyes else "OFF"
    rec_str = "ON" if (recording_info and recording_info[0]) else "OFF"
    guide_str = "ON" if show_position_guide else "OFF"
    audio_str = "ON" if audio_warnings_enabled else "OFF"
    shadow_sw = "ON" if shadow_enabled else "OFF"
    controls_line = (
        f"Controls: [q] Exit  [a] Audio:{audio_str}  [x] Shadow:{shadow_sw}  [c] Calibrate  [p] Guide:{guide_str}  [r] Rec:{rec_str}  [m] Mirror  [w] Mesh:{mesh_str}  [e] Eyes:{eyes_str}  [s] Snapshot"
    )
    put_fitted_text(canvas, controls_line, 16, y_row7, font, 0.38, (100, 116, 139), width - 32, 1)

    # -------------------------------------------------------------
    # BOTTOM CALIBRATION BOX (Phase 7 Guided Calibration)
    # -------------------------------------------------------------
    if calibration_info is not None:
        step_num, total_steps, progress_ratio, instruction, is_aligned, feedback = calibration_info
        if step_num == 1:
            instruction = (
                "Sit naturally, look straight ahead. Keep eyes fully open (do not squint or droop). Blink normally."
            )

        box_h = 104
        box_y = height - 118
        b_color = (34, 197, 94) if is_aligned else (234, 179, 8)

        # Background overlay
        sub = canvas.copy()
        cv2.rectangle(sub, (16, box_y), (width - 16, box_y + box_h), (15, 23, 42), -1)
        cv2.addWeighted(sub, 0.90, canvas, 0.10, 0, canvas)
        cv2.rectangle(canvas, (16, box_y), (width - 16, box_y + box_h), b_color, 2)

        # Line 1: Step Pill + Target Zone (Left) & Dedicated Progress Bar (Right)
        step_y = box_y + 24
        step_pill_text = f"STEP {step_num}/{total_steps}"
        step_bg = (6, 78, 59) if is_aligned else (35, 28, 10)
        _, _, bx2, _ = draw_pill(canvas, step_pill_text, 28, step_y, (255, 255, 255), step_bg, b_color, 0.42, 1, 8, 3, "left")

        # Determine target zone title
        zone_titles = {
            1: "CENTER ZONE POSTURE",
            2: "LEFT MIRROR GLANCE (~20 deg)",
            3: "RIGHT MIRROR GLANCE (~20 deg)",
            4: "DASHBOARD / INSTRUMENT CLUSTER (~10 deg)",
        }
        zone_title = zone_titles.get(step_num, "CALIBRATION ZONE")
        cv2.putText(canvas, zone_title, (bx2 + 10, step_y), font, 0.48, (255, 255, 255), 2, cv2.LINE_AA)

        # Progress Bar & Percentage on the right
        bar_w = min(220, max(100, int(width * 0.22)))
        bar_x = (width - 28) - bar_w - 55
        bar_y = box_y + 11
        bar_h = 16
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (51, 65, 85), -1)
        fill_w = int(bar_w * max(0.0, min(1.0, progress_ratio)))
        if fill_w > 0:
            cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), b_color, -1)
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (148, 163, 184), 1)

        pct_x = bar_x + bar_w + 8
        pct_y = bar_y + 13
        cv2.putText(canvas, f"{int(progress_ratio * 100)}%", (pct_x, pct_y), font, 0.42, (255, 255, 255), 1, cv2.LINE_AA)

        # Line 2: Full-Width Instruction Text
        inst_y = box_y + 54
        put_fitted_text(canvas, instruction, 28, inst_y, font, 0.46, (255, 255, 255), width - 56, 1)

        # Line 3: Live Feedback Status (Left) & Cancel Hint (Right)
        feed_y = box_y + 84
        feed_prefix = "[HOLDING STEADY] " if is_aligned else "[ALIGNING] "
        feed_str = f"{feed_prefix}{feedback}"
        cancel_hint = "Press 'c' or ESC to cancel"
        (cw, ch), _ = cv2.getTextSize(cancel_hint, font, 0.38, 1)
        avail_feed_w = (width - 28 - cw - 16) - 28
        put_fitted_text(canvas, feed_str, 28, feed_y, font, 0.44, b_color, avail_feed_w, 1)

        cx = width - 28 - cw
        cv2.putText(canvas, cancel_hint, (cx, feed_y), font, 0.38, (148, 163, 184), 1, cv2.LINE_AA)

    elif calibration_banner:
        box_y = height - 76
        box_h = 56
        sub = canvas.copy()
        is_failure = "FAILED" in calibration_banner
        bg_color = (30, 27, 75) if is_failure else (6, 78, 59)
        border_color = (239, 68, 68) if is_failure else (34, 197, 94)
        cv2.rectangle(sub, (16, box_y), (width - 16, box_y + box_h), bg_color, -1)
        cv2.addWeighted(sub, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (16, box_y), (width - 16, box_y + box_h), border_color, 2)

        # Auto-scale font if message is long
        (tw, th), _ = cv2.getTextSize(calibration_banner, font, 0.52, 2)
        b_scale = 0.52
        b_thick = 2
        if tw > width - 64:
            b_scale = max(0.35, 0.52 * ((width - 64) / max(1, tw)))
            b_thick = 2 if b_scale > 0.45 else 1
            (tw, th), _ = cv2.getTextSize(calibration_banner, font, b_scale, b_thick)

        bx = max(24, (width - tw) // 2)
        by = box_y + (box_h + th) // 2
        cv2.putText(canvas, calibration_banner, (bx, by), font, b_scale, (255, 255, 255), b_thick, cv2.LINE_AA)

    elif audio_feedback_banner:
        box_y = height - 76
        box_h = 56
        sub = canvas.copy()
        is_muted = "MUTED" in audio_feedback_banner or "OFF" in audio_feedback_banner
        bg_color = (40, 30, 20) if is_muted else (6, 78, 59)
        border_color = (234, 179, 8) if is_muted else (34, 197, 94)
        cv2.rectangle(sub, (16, box_y), (width - 16, box_y + box_h), bg_color, -1)
        cv2.addWeighted(sub, 0.85, canvas, 0.15, 0, canvas)
        cv2.rectangle(canvas, (16, box_y), (width - 16, box_y + box_h), border_color, 2)

        (tw, th), _ = cv2.getTextSize(audio_feedback_banner, font, 0.52, 2)
        bx = max(24, (width - tw) // 2)
        by = box_y + (box_h + th) // 2
        cv2.putText(canvas, audio_feedback_banner, (bx, by), font, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

    # Subtle status border around the full viewport (Phase 8 alert-aware)
    border_color = effective_alert.hud_color_bgr
    cv2.rectangle(canvas, (0, 0), (width - 1, height - 1), border_color, 3)

    return canvas


def run_benchmark(
    source_identifier: str | int = 0,
    target_width: int = 1280,
    target_height: int = 720,
    target_fps: int = 30,
    backend_str: str = "DSHOW",
    duration_sec: float = 0.0,
    headless: bool = False,
    flip_h: bool = True,
    flip_v: bool = False,
    profile_arg: Optional[str] = None,
    start_calibrating: bool = False,
    driver_id: str = "driver_default",
    record_session: bool = False,
    cnn_shadow: bool = False,
    cnn_shadow_hz: float = 15.0,
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

        # Profile loading & FeatureExtractor initialization (Phase 7 Task 2)
        profile_loaded: Optional[DriverBehaviourProfile] = None
        calibrated_label: Optional[str] = None
        if profile_arg:
            p_path = Path(profile_arg)
            if p_path.is_file():
                try:
                    with open(p_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    if isinstance(data, dict) and "schema_version" in data and "zone_baselines" in data:
                        dto = DriverProfileDTO.from_dict(data)
                        profile_loaded = dto_to_profile(dto)
                    else:
                        profile_loaded = DriverBehaviourProfile.load_json(p_path)
                    calibrated_label = profile_loaded.driver_id
                    logger.info("Loaded driver profile from file: %s (driver_id: %s)", p_path, calibrated_label)
                except Exception as e:
                    logger.warning("Could not parse profile file %s: %s", p_path, e)
            if profile_loaded is None:
                profile_dto = profile_store.get_profile(profile_arg)
                if profile_dto is not None:
                    try:
                        profile_loaded = dto_to_profile(profile_dto)
                        calibrated_label = profile_dto.driver_id
                        logger.info("Loaded driver profile '%s' from profile store via adapter.", profile_arg)
                    except Exception as e:
                        logger.warning("Failed to convert profile DTO '%s': %s", profile_arg, e)

        # Auto-load existing driver profile from profile store if not explicitly supplied
        if profile_loaded is None and driver_id:
            profile_dto = profile_store.get_profile(driver_id)
            if profile_dto is not None:
                try:
                    profile_loaded = dto_to_profile(profile_dto)
                    calibrated_label = profile_dto.driver_id
                    logger.info("Auto-loaded saved driver profile for '%s' from profile store.", driver_id)
                except Exception as e:
                    logger.warning("Failed to auto-convert profile DTO for '%s': %s", driver_id, e)

        # Initialize feature extractor with profile (evaluates mathematical feasibility)
        feature_extractor = DriverFeatureExtractor(profile=profile_loaded)
        if profile_loaded is not None:
            if not feature_extractor.is_profile_feasible:
                logger.warning(
                    "Loaded profile '%s' is mathematically infeasible. Ocular classification remains disabled.",
                    calibrated_label,
                )
                calibrated_label = None
            else:
                logger.info(
                    "Profile '%s' active: Center c=%.3f, r=%.3f (span=%.3f)",
                    calibrated_label,
                    feature_extractor.active_closure_threshold,
                    feature_extractor.active_reopening_threshold,
                    feature_extractor.active_reopening_threshold - feature_extractor.active_closure_threshold,
                )

        quality_analyzer = FrameQualityAnalyzer()
        landmarker = DriveGuardFaceLandmarker()

        drowsiness_engine = DrowsinessEngine()

        # Phase 8: Alert State Machine & Audio Warning System
        alert_manager = AlertStateManager()
        audio_engine = AudioWarningEngine()
        alert_dispatcher = AlertDispatcher(
            audio_engine=audio_engine,
            settings_store=audio_settings_store,
        )
        audio_feedback_banner_text: Optional[str] = None
        audio_feedback_banner_until: float = 0.0

        # Phase B1: Passive Eye CNN Shadow Observer (Decoupled Diagnostic)
        shadow_policy = EyeCNNInvocationPolicy(
            enabled=cnn_shadow,
            max_frequency_hz=cnn_shadow_hz,
            max_staleness_ms=200.0,
        )
        shadow_worker = EyeCNNShadowWorker(policy=shadow_policy)
        shadow_worker.start()
        latest_shadow_obs: Optional[BinocularShadowTelemetry] = None
        if cnn_shadow:
            logger.info("Eye CNN Shadow Observer active at max %.1f Hz (passive diagnostic).", cnn_shadow_hz)

        # Guided Calibration Session (Phase 7)
        calibration_session: Optional[GuidedCalibrationSession] = None
        calibration_banner_text: Optional[str] = None
        calibration_banner_until: float = 0.0

        if start_calibrating:
            calibration_session = GuidedCalibrationSession(
                driver_id=driver_id,
                four_zone_mode=True,
                step_duration_sec=6.0,
            )
            calibration_session.start()
            logger.info("Started guided 4-zone calibration for driver '%s'.", driver_id)

        # Visualization toggles (Phase 4)
        draw_mesh: bool = False
        draw_eyes: bool = True
        show_position_guide: bool = not start_calibrating

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

        logger.info("Starting live capture loop. Press 'q' or ESC to terminate, 'c' to calibrate.")

        # Synchronized Session Recorder (Video + Telemetry)
        session_recorder: Optional[SessionRecorder] = None
        if record_session:
            f_w = actual_res[0] if actual_res[0] > 0 else target_width
            f_h = actual_res[1] if actual_res[1] > 0 else target_height
            session_recorder = SessionRecorder(
                driver_id=driver_id,
                frame_width=f_w,
                frame_height=f_h,
                fps=camera.get_actual_fps() or float(target_fps),
                source_identifier=safe_identifier,
                profile_metadata={"profile_label": calibrated_label},
            )
            if session_recorder.start():
                logger.info("Auto-started session recording on launch in: %s", session_recorder.session_dir)
            else:
                logger.warning("Failed to auto-start session recorder.")

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
            if hasattr(camera, "is_flip_horizontal"):
                is_mirrored = camera.is_flip_horizontal() if callable(camera.is_flip_horizontal) else bool(camera.is_flip_horizontal)
            else:
                is_mirrored = False

            # Extract physiological driver features (Phase 5)
            f_w = actual_res[0] if actual_res[0] > 0 else packet.frame.shape[1]
            f_h = actual_res[1] if actual_res[1] > 0 else packet.frame.shape[0]
            latest_features = feature_extractor.extract(
                latest_landmark,
                frame_width=f_w,
                frame_height=f_h,
                is_mirrored=is_mirrored,
            )

            # Eye CNN Passive Shadow Observer Ingestion (Phase B1 - Decoupled Diagnostic)
            try:
                should_shadow, _ = shadow_policy.should_invoke(
                    timestamp_ms=packet.timestamp_ms,
                    has_face=latest_landmark.has_face if latest_landmark else False,
                    ear_valid=latest_features.ear.is_geometric_valid if latest_features else False,
                )
                if should_shadow and latest_landmark is not None and latest_landmark.has_face:
                    lms_xyz = np.array(
                        [[lm.x * f_w, lm.y * f_h, lm.z * f_w] for lm in latest_landmark.landmarks],
                        dtype=np.float64,
                    )
                    shadow_worker.submit_frame(
                        frame_bgr=packet.frame,
                        landmarks_pixel_xyz=lms_xyz,
                        timestamp_ms=packet.timestamp_ms,
                        frame_id=getattr(packet, "frame_index", total_frames_displayed),
                        geom_left_closed=latest_features.ear.is_left_closed if latest_features else None,
                        geom_right_closed=latest_features.ear.is_right_closed if latest_features else None,
                        raw_left_ear=latest_features.ear.left_ear if latest_features else 0.0,
                        raw_right_ear=latest_features.ear.right_ear if latest_features else 0.0,
                        ear_threshold=latest_features.ear.closure_threshold if latest_features else 0.20,
                    )

                # Retrieve freshest non-stale shadow telemetry observation
                latest_shadow_obs = shadow_worker.get_latest_observation(packet.timestamp_ms)
            except Exception as shadow_exc:
                logger.warning("Error during eye shadow observer ingestion: %s", shadow_exc)

            # Ingest frame into calibration session if active
            calib_info = None
            if calibration_session is not None and calibration_session.is_running:
                calib_info = calibration_session.get_progress(packet.timestamp_ms)
                if latest_landmark is not None and latest_features is not None:
                    fb = latest_features.blink.completed_blink
                    adapted_blink: Optional[AIBlinkEvent] = None
                    if fb is not None:
                        cls_name = getattr(fb.classification, "name", "NORMAL")
                        ai_cls = getattr(AIBlinkClassification, cls_name, AIBlinkClassification.NORMAL)
                        adapted_blink = AIBlinkEvent(
                            start_ts_ms=fb.start_ts_ms,
                            min_ts_ms=fb.min_ts_ms,
                            end_ts_ms=fb.end_ts_ms,
                            duration_ms=fb.duration_ms,
                            open_ear_before=fb.open_ear_before,
                            min_ear=fb.min_ear,
                            amplitude=fb.amplitude,
                            closing_velocity_per_s=fb.closing_velocity_per_s,
                            reopening_velocity_per_s=fb.reopening_velocity_per_s,
                            rcr=fb.rcr,
                            classification=ai_cls,
                            is_binocular=fb.is_binocular,
                        )
                    # Decouple calibration data collection from ocular classification
                    is_closed_for_calib = (
                        latest_features.ear.is_closed
                        if feature_extractor.is_profile_feasible
                        else None
                    )
                    completed_profile = calibration_session.ingest_frame(
                        timestamp_ms=packet.timestamp_ms,
                        face_detected=latest_landmark.has_face,
                        geometric_reliability=0.95 if latest_landmark.has_face else 0.0,
                        ear_value=latest_features.ear.compensated_ear,
                        pitch_deg=latest_features.head_pose.pitch_deg,
                        yaw_deg=latest_features.head_pose.yaw_deg,
                        is_eye_currently_closed=is_closed_for_calib,
                        completed_blink=adapted_blink,
                        left_ear=latest_features.ear.left_ear,
                        right_ear=latest_features.ear.right_ear,
                        landmark_confidence=latest_landmark.presence_confidence,
                        rot_mat=latest_features.head_pose.rot_mat,
                        ear_geometric_valid=latest_features.ear.is_geometric_valid,
                    )
                    if completed_profile is not None:
                        success, status_msg = commit_and_activate_profile(
                            profile=completed_profile,
                            feature_extractor=feature_extractor,
                            store=profile_store,
                        )
                        if success:
                            calibrated_label = completed_profile.driver_id
                            c_ctr = feature_extractor.active_closure_threshold
                            r_ctr = feature_extractor.active_reopening_threshold
                            calibration_banner_text = (
                                f"CALIBRATION COMPLETE: '{completed_profile.driver_id}' active! "
                                f"Thresholds: Center c={c_ctr:.3f}/r={r_ctr:.3f}"
                            )
                            calibration_banner_until = loop_now + 5.0
                            logger.info(
                                "Calibration successfully completed, persisted, and hot-reloaded for driver '%s'.",
                                completed_profile.driver_id,
                            )
                        else:
                            logger.warning(
                                "Calibration commit/activation failed for '%s': %s",
                                completed_profile.driver_id,
                                status_msg,
                            )
                            calibration_banner_text = f"CALIBRATION FAILED: {status_msg}"
                            calibration_banner_until = loop_now + 5.0
                        calibration_session = None
                    elif not calibration_session.is_running:
                        # Session ended without a completed profile (rejected or timed out)
                        reason = getattr(calibration_session, "rejection_reason", "Session terminated")
                        logger.warning("Calibration ended without completed profile: %s", reason)
                        calibration_banner_text = f"CALIBRATION FAILED: {reason}"
                        calibration_banner_until = loop_now + 5.0
                        calibration_session = None

            # Evaluate Drowsiness Intelligence Engine (Phase 6)
            latest_assessment = drowsiness_engine.evaluate(
                features=latest_features,
                quality=latest_quality,
            )

            # Evaluate Alert State Manager & Dispatcher (Phase 8)
            alert_transition = alert_manager.update(latest_assessment)
            alert_dispatcher.dispatch(
                transition=alert_transition,
                current_state=alert_manager.current_state,
                timestamp_ms=packet.timestamp_ms,
                is_recovering=alert_manager.is_recovering,
            )

            # Record synchronized video frame and telemetry row
            if session_recorder is not None and session_recorder.is_recording:
                t_dict = {
                    "face_detected": latest_landmark.has_face if latest_landmark else False,
                    "face_status": latest_landmark.status.value if latest_landmark else "UNKNOWN",
                    "presence_confidence": latest_landmark.presence_confidence if latest_landmark else 0.0,
                    "pose_zone": latest_features.head_pose.zone.value if latest_features else "UNKNOWN",
                    "pitch_deg": latest_features.head_pose.pitch_deg if latest_features else 0.0,
                    "yaw_deg": latest_features.head_pose.yaw_deg if latest_features else 0.0,
                    "roll_deg": latest_features.head_pose.roll_deg if latest_features else 0.0,
                    "rel_pitch_deg": latest_features.head_pose.rel_pitch_deg if latest_features else 0.0,
                    "rel_yaw_deg": latest_features.head_pose.rel_yaw_deg if latest_features else 0.0,
                    "rel_roll_deg": latest_features.head_pose.rel_roll_deg if latest_features else 0.0,
                    "ear_left": latest_features.ear.left_ear if latest_features else 0.0,
                    "ear_right": latest_features.ear.right_ear if latest_features else 0.0,
                    "ear_compensated": latest_features.ear.compensated_ear if latest_features else 0.0,
                    "ear_closure_threshold": latest_features.ear.closure_threshold if latest_features else 0.20,
                    "is_eye_closed": latest_features.ear.is_closed if latest_features else False,
                    "is_geometric_valid": latest_features.ear.is_geometric_valid if latest_features else False,
                    "mar": latest_features.mar.mar if latest_features else 0.0,
                    "is_yawning": bool(getattr(latest_features.mar, "is_yawn_in_progress", False)) if latest_features else False,
                    "total_blinks": latest_features.blink.total_blink_count if latest_features else 0,
                    "total_yawns": latest_features.mar.total_yawn_count if latest_features else 0,
                    "perclos_pct": latest_features.perclos.perclos_pct if latest_features else 0.0,
                    "drowsiness_state": latest_assessment.state.value if latest_assessment else "NORMAL",
                    "drowsiness_severity": latest_assessment.severity if latest_assessment else 0,
                    "primary_reason": latest_assessment.primary_reason_text if latest_assessment is not None else "NONE",
                    "alert_state": alert_manager.current_state.value,
                    "alert_severity": alert_manager.severity,
                    "is_recovering": alert_manager.is_recovering,
                    "audio_warnings_enabled": alert_dispatcher.is_audio_enabled,
                    "luminance": latest_quality.mean_luminance if latest_quality else 0.0,
                    "blur_variance": latest_quality.blur_variance if latest_quality else 0.0,
                    "optical_status": latest_quality.status.value if latest_quality else "UNKNOWN",
                    "inference_latency_ms": latest_landmark.inference_latency_ms if latest_landmark else 0.0,
                    "shadow_status": latest_shadow_obs.status_label if latest_shadow_obs else "DISABLED",
                    "shadow_left_cnn": latest_shadow_obs.left_eye.cnn_predicted_class if latest_shadow_obs else None,
                    "shadow_right_cnn": latest_shadow_obs.right_eye.cnn_predicted_class if latest_shadow_obs else None,
                    "shadow_binoc_agreement": latest_shadow_obs.binocular_agreement if latest_shadow_obs else None,
                    "shadow_latency_ms": latest_shadow_obs.inference_latency_ms if latest_shadow_obs else 0.0,
                }
                session_recorder.record_frame(
                    frame=packet.frame,
                    timestamp_ms=packet.timestamp_ms,
                    telemetry=t_dict,
                )

            # Measure consumer display FPS every 0.5s
            elapsed = loop_now - fps_timer
            if elapsed >= 0.5:
                display_fps = fps_frame_count / elapsed
                fps_timer = loop_now
                fps_frame_count = 0

            # Render overlay if running in GUI mode
            if loop_now >= calibration_banner_until:
                calibration_banner_text = None
            if loop_now >= audio_feedback_banner_until:
                audio_feedback_banner_text = None

            if not headless:
                # 1. Render facial landmarks directly onto raw video frame
                rendered_frame = landmarker.draw_landmarks(
                    frame=packet.frame,
                    result=latest_landmark,
                    draw_mesh=draw_mesh,
                    draw_eyes=draw_eyes,
                    draw_irises=True,
                )

                # Optional Camera Positioning & Framing Guide Overlay
                if show_position_guide and (calibration_session is None or not calibration_session.is_running):
                    pos_res = evaluate_camera_position(
                        landmark_result=latest_landmark,
                        head_pose=latest_features.head_pose if latest_features else None,
                        quality=latest_quality,
                        frame_width=f_w,
                        frame_height=f_h,
                    )
                    rendered_frame = draw_positioning_guide(
                        canvas=rendered_frame,
                        guide_res=pos_res,
                        is_calibrating=False,
                    )

                # 2. Render diagnostic HUD cleanly on top of landmarks with zero overlap
                hud_frame = draw_diagnostic_hud(
                    frame=rendered_frame,
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
                    calibration_info=calib_info,
                    calibration_banner=calibration_banner_text,
                    calibrated_profile_name=calibrated_label,
                    recording_info=(
                        session_recorder.is_recording,
                        session_recorder.elapsed_sec,
                        session_recorder.frame_count,
                    ) if session_recorder is not None else None,
                    show_position_guide=show_position_guide,
                    alert_state=alert_manager.current_state,
                    is_recovering=alert_manager.is_recovering,
                    audio_warnings_enabled=alert_dispatcher.is_audio_enabled,
                    audio_feedback_banner=audio_feedback_banner_text,
                    shadow_telemetry=latest_shadow_obs,
                    shadow_enabled=shadow_policy.enabled,
                )
                cv2.imshow(window_name, hud_frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q"), ord("Q")):
                    logger.info("User requested termination.")
                    break
                elif key in (ord("x"), ord("X")):
                    shadow_policy.enabled = not shadow_policy.enabled
                    audio_feedback_banner_text = (
                        "CNN SHADOW: ENABLED" if shadow_policy.enabled else "CNN SHADOW: MUTED"
                    )
                    audio_feedback_banner_until = loop_now + 2.5
                    logger.info(
                        "Eye CNN Shadow Observer toggled to: %s",
                        "ENABLED" if shadow_policy.enabled else "MUTED",
                    )
                elif key in (ord("a"), ord("A")):
                    new_audio_state = alert_dispatcher.toggle_audio(
                        active_state=alert_manager.current_state,
                        timestamp_ms=packet.timestamp_ms,
                    )
                    audio_feedback_banner_text = (
                        "AUDIO WARNINGS: ENABLED" if new_audio_state else "AUDIO WARNINGS: MUTED"
                    )
                    audio_feedback_banner_until = loop_now + 2.5
                    logger.info(
                        "Audio warnings toggled to: %s",
                        "ENABLED" if new_audio_state else "MUTED",
                    )
                elif key in (ord("c"), ord("C")):
                    show_position_guide = False
                    if calibration_session is None or not calibration_session.is_running:
                        calibration_session = GuidedCalibrationSession(
                            driver_id=driver_id,
                            four_zone_mode=True,
                            step_duration_sec=6.0,
                        )
                        calibration_session.start(timestamp_ms=packet.timestamp_ms)
                        logger.info("Started interactive 4-zone guided calibration for driver '%s'.", driver_id)
                    else:
                        calibration_session = None
                        calibration_banner_text = "CALIBRATION FAILED: Cancelled by user"
                        calibration_banner_until = loop_now + 5.0
                        logger.info("Cancelled ongoing calibration.")
                elif key in (ord("p"), ord("P")):
                    show_position_guide = not show_position_guide
                    logger.info("Camera positioning guide toggled: %s", show_position_guide)
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
                elif key in (ord("r"), ord("R")):
                    if session_recorder is None or not session_recorder.is_recording:
                        f_w = actual_res[0] if actual_res[0] > 0 else target_width
                        f_h = actual_res[1] if actual_res[1] > 0 else target_height
                        session_recorder = SessionRecorder(
                            driver_id=driver_id,
                            frame_width=f_w,
                            frame_height=f_h,
                            fps=camera.get_actual_fps() or float(target_fps),
                            source_identifier=safe_identifier,
                            profile_metadata={"profile_label": calibrated_label},
                        )
                        if session_recorder.start():
                            logger.info("Started session recording in: %s", session_recorder.session_dir)
                    else:
                        summary = session_recorder.stop()
                        logger.info(
                            "Stopped session recording: %d frames saved in %s",
                            summary.get("total_frames", 0),
                            session_recorder.session_dir,
                        )

    finally:
        if "shadow_worker" in locals() and shadow_worker is not None:
            shadow_worker.stop()
        if "alert_dispatcher" in locals() and alert_dispatcher is not None:
            alert_dispatcher.shutdown()
        if "session_recorder" in locals() and session_recorder is not None and session_recorder.is_recording:
            session_recorder.stop()
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
    if "shadow_worker" in locals() and shadow_worker is not None and shadow_worker._total_inferences_count > 0:
        print("-" * 60)
        print("      EYE CNN SHADOW OBSERVER (PHASE B1)")
        print("-" * 60)
        print(f"Shadow Worker Status  : {'AVAILABLE' if shadow_worker.is_available else 'UNAVAILABLE'}")
        print(f"Total Inferences Run  : {shadow_worker._total_inferences_count}")
        print(f"Dropped Frames        : {shadow_worker._dropped_jobs_count}")
        print(f"Inference Errors      : {shadow_worker._error_count}")
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
        dest="flip_h",
        action="store_true",
        default=True,
        help="Horizontally flip / mirror video (default: True, mirrored)",
    )
    parser.add_argument(
        "--no-flip-h",
        dest="flip_h",
        action="store_false",
        help="Disable horizontal flip / mirror video (use raw camera orientation)",
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
    parser.add_argument(
        "--profile",
        type=str,
        default=None,
        help="Path to driver profile JSON or driver ID in profile store",
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Start 4-zone guided driver calibration immediately on launch",
    )
    parser.add_argument(
        "--driver-id",
        type=str,
        default="driver_default",
        help="Driver identifier for calibration profile saving (default: driver_default)",
    )
    parser.add_argument(
        "--record",
        dest="record_session",
        action="store_true",
        help="Record synchronized video (.mp4) and per-frame telemetry (.csv/.jsonl)",
    )
    parser.add_argument(
        "--cnn-shadow",
        dest="cnn_shadow",
        action="store_true",
        default=False,
        help="Enable MobileNetV3 eye-state CNN passive shadow observer (Phase B1 diagnostic)",
    )
    parser.add_argument(
        "--cnn-shadow-hz",
        type=float,
        default=15.0,
        help="Maximum inference frequency for CNN shadow observer in Hz (default: 15.0)",
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
            profile_arg=args.profile,
            start_calibrating=args.calibrate,
            driver_id=args.driver_id,
            record_session=args.record_session,
            cnn_shadow=args.cnn_shadow,
            cnn_shadow_hz=args.cnn_shadow_hz,
        )
    )


if __name__ == "__main__":
    main()
