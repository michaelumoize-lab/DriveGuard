"""
DriveGuard Phase 1 Deliverable: Real-Time Webcam Prototype & Cascade Benchmark Suite.

Usage:
  1. Live Interactive Webcam Prototype:
       python -m backend.research.prototype_webcam_cli --camera 0

  2. Headless Experimental Benchmark (Compares Fixed vs. 4-Zone PDBP and Experiments A/B/C):
       python -m backend.research.prototype_webcam_cli --benchmark-simulation

Interactive Controls (Webcam Window):
  [C] : Start 4-Zone Guided Calibration (Center -> Left Mirror -> Right Mirror -> Dashboard)
  [1] : Start Quick 1-Zone (Center-Only) Calibration
  [A] : Switch to Experiment A (MediaPipe EAR + Blendshapes Only)
  [B] : Switch to Experiment B (MobileNetV3-Small Eye Verifier Always On)
  [G] : Switch to Experiment C (Confidence-Gated Cascade - Recommended)
  [S] : Export Driver Profile & Session Telemetry JSON
  [Q] or [ESC] : Quit
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import psutil

from backend.app.ai.blink_kinematics import TemporalKinematicsTracker
from backend.app.ai.cascade_classifier import ConfidenceGatedEyeCascade, ExperimentMode
from backend.app.ai.geometry import (
    LEFT_EYE_INDICES,
    RIGHT_EYE_INDICES,
    EyeCropQuality,
    FrameGeometryFeatures,
    PoseZone,
    classify_pose_zone,
)
from backend.app.ai.pdbp_calibration import (
    DriverBehaviourProfile,
    GuidedCalibrationSession,
)
from backend.app.ai.risk_engine import AlertState, PersonalisedAdaptiveRiskEngine

STATE_COLORS_BGR = {
    AlertState.NORMAL: (80, 210, 60),               # Green
    AlertState.WARNING: (0, 195, 255),              # Amber/Yellow
    AlertState.CRITICAL: (40, 40, 245),             # Red
    AlertState.MONITORING_UNAVAILABLE: (160, 160, 160),  # Gray
}


def draw_hud_overlay(
    frame_bgr: np.ndarray,
    landmarks_xyz: Optional[np.ndarray],
    geom: FrameGeometryFeatures,
    cascade_res,
    kinematics_res,
    risk_res,
    calibration: GuidedCalibrationSession,
    fps: float,
) -> np.ndarray:
    """Render an informative engineering HUD on the live webcam frame."""
    canvas = frame_bgr.copy()
    h, w = canvas.shape[:2]

    # 1. Draw Eye Contours if landmarks present
    if landmarks_xyz is not None:
        for indices in (LEFT_EYE_INDICES, RIGHT_EYE_INDICES):
            pts = landmarks_xyz[list(indices), :2].astype(np.int32)
            cv2.polylines(canvas, [pts], isClosed=True, color=(0, 255, 180), thickness=1)

    # 2. Top Status Banner
    banner_color = STATE_COLORS_BGR.get(risk_res.state, (120, 120, 120))
    cv2.rectangle(canvas, (0, 0), (w, 52), (22, 22, 26), -1)
    cv2.rectangle(canvas, (0, 48), (w, 52), banner_color, -1)

    cv2.putText(
        canvas,
        f"DRIVEGUARD [{risk_res.state.value}]  |  Risk: {risk_res.risk_score:.2f}  |  Zone: {geom.pose_zone.value}",
        (14, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        banner_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Mode: {cascade_res.experiment_mode.value}  |  Gate: {'CNN_ON' if cascade_res.cnn_invoked else 'GEOM_FAST'} ({cascade_res.winning_authority})  |  {fps:.1f} FPS",
        (14, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (215, 215, 220),
        1,
        cv2.LINE_AA,
    )

    # 3. Calibration Progress Banner (if active)
    if calibration.is_running:
        step_num, total_steps, ratio, instr = calibration.get_progress(geom.timestamp_ms)
        cv2.rectangle(canvas, (12, 62), (w - 12, 118), (45, 35, 15), -1)
        cv2.rectangle(canvas, (12, 62), (w - 12, 118), (0, 200, 255), 1)
        cv2.putText(
            canvas,
            instr,
            (22, 84),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (255, 240, 180),
            1,
            cv2.LINE_AA,
        )
        bar_w = int((w - 48) * ratio)
        cv2.rectangle(canvas, (22, 95), (w - 26, 108), (70, 70, 75), -1)
        cv2.rectangle(canvas, (22, 95), (22 + bar_w, 108), (0, 215, 255), -1)

    # 4. Left Telemetry Panel (Metrics & Robust z-scores)
    panel_y = 128 if calibration.is_running else 64
    cv2.rectangle(canvas, (12, panel_y), (330, panel_y + 210), (24, 24, 30), -1)
    cv2.rectangle(canvas, (12, panel_y), (330, panel_y + 210), (75, 75, 85), 1)

    lines = [
        f"EAR (Smoothed): {kinematics_res.smoothed_ear:.3f} (Base: {risk_res.deviations.zone_ear_baseline:.3f})",
        f"EAR Cut Threshold: {risk_res.deviations.zone_closure_threshold:.3f}  [z_EAR: {risk_res.deviations.z_ear:.2f}]",
        f"PERCLOS (60s): {kinematics_res.perclos_60s_pct:.1f}%  [z_PERC: {risk_res.deviations.z_perclos:.2f}]",
        f"Blink Dur: {(kinematics_res.recent_blink_duration_ms or 0.0):.0f} ms  [z_Dur: {risk_res.deviations.z_blink_duration:.2f}]",
        f"Blink RCR (Vel): {(kinematics_res.recent_blink_rcr or 0.0):.2f}  [z_RCR: {risk_res.deviations.z_blink_rcr:.2f}]",
        f"Blink Rate: {kinematics_res.blink_rate_per_min:.1f}/min  |  MAR: {geom.mar:.2f}",
        f"Head Pose: Pitch {geom.pitch_deg:+.1f}  Yaw {geom.yaw_deg:+.1f}",
        f"Geom Rel: {geom.geometric_reliability:.2f}  |  Fused Closure: {cascade_res.fused_closure_score:.2f}",
        f"Fixed (EAR<0.22) State: {risk_res.conventional_fixed_state.value}",
    ]
    for idx, text in enumerate(lines):
        color = (225, 225, 230)
        if idx == len(lines) - 1:
            color = STATE_COLORS_BGR.get(risk_res.conventional_fixed_state, (200, 200, 200))
        cv2.putText(
            canvas,
            text,
            (20, panel_y + 22 + idx * 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            color,
            1,
            cv2.LINE_AA,
        )

    # 5. Bottom Explainable Reasons & Controls Footer
    cv2.rectangle(canvas, (0, h - 54), (w, h), (20, 20, 24), -1)
    reasons_str = ", ".join(risk_res.reasons) if risk_res.reasons else "NONE (ALERT_NORMAL_BEHAVIOUR)"
    cv2.putText(
        canvas,
        f"Reasons: {reasons_str[:95]}",
        (14, h - 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.44,
        banner_color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Keys: [C] 4-Zone Calib  [1] 1-Zone Calib  [A] Exp-A  [B] Exp-B  [G] Exp-C Cascade  [S] Save  [Q] Quit",
        (14, h - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (175, 175, 185),
        1,
        cv2.LINE_AA,
    )

    return canvas


def run_benchmark_simulation() -> Dict[str, object]:
    """
    Execute a deterministic 5-stage driver evaluation comparing:
      1. Conventional Fixed EAR Threshold (EAR < 0.22) vs. DriveGuard 4-Zone Personalised PDBP
      2. Experiment A (Geometric Only) vs. Experiment B (CNN Always) vs. Experiment C (Confidence-Gated Cascade)

    Simulated Driver Profile:
      - Driver has naturally narrower eyes: Open-eye EAR in CENTER = 0.218 (below standard 0.22 fixed cutoff!)
      - Left/Right mirror checks reduce apparent 2D EAR to 0.198
      - Normal blink duration = 180 ms, normal RCR = 0.94
    """
    print("=" * 88)
    print("DRIVEGUARD PHASE 1 — EXPERIMENTAL SIMULATION & CASCADE BENCHMARK")
    print("=" * 88)

    # Build a synthetic eye crop helper for deterministic testing
    def make_synthetic_crop(is_open: bool, has_glare: bool = False) -> EyeCropQuality:
        img = np.full((64, 64, 3), 155, dtype=np.uint8)
        if is_open:
            # Draw dark iris/pupil disk with strong vertical edges
            cv2.circle(img, (32, 32), 13, (28, 28, 28), -1)
            cv2.circle(img, (32, 32), 6, (10, 10, 10), -1)
        else:
            # Draw horizontal closed-eyelid crease
            cv2.line(img, (10, 32), (54, 32), (55, 55, 55), 3)
        glare_frac = 0.01
        if has_glare:
            cv2.rectangle(img, (18, 18), (30, 30), (252, 252, 252), -1)
            glare_frac = 0.09
        return EyeCropQuality(
            crop_rgb=img,
            sharpness_score=0.88,
            exposure_score=0.92,
            glare_fraction=glare_frac,
            overall_reliability=0.84 if not has_glare else 0.72,
        )

    # 1. Run 4-Zone Calibration for Narrow-Eyed Driver
    cal = GuidedCalibrationSession(driver_id="narrow_eye_test_driver", four_zone_mode=True)
    ts_ms = 0.0
    cal.start(ts_ms)
    profile: Optional[DriverBehaviourProfile] = None

    # Feed 36 seconds of 4-zone calibration at 20 FPS (dt = 50 ms)
    zone_sim_params = {
        PoseZone.CENTER: (0.218, 0.0, 0.0),
        PoseZone.LEFT_MIRROR: (0.198, 0.0, -22.0),
        PoseZone.RIGHT_MIRROR: (0.199, 0.0, +22.0),
        PoseZone.DASHBOARD: (0.192, -13.0, 0.0),
    }

    rng = np.random.default_rng(42)
    kin_cal = TemporalKinematicsTracker()

    while cal.is_running:
        step_zone = cal.current_step.zone
        base_ear, pitch, yaw = zone_sim_params[step_zone]
        # Inject a natural 180ms blink every 4 seconds
        in_natural_blink = (int(ts_ms) % 4000) in range(1000, 1180)
        ear = 0.110 if in_natural_blink else float(base_ear + rng.normal(0.0, 0.005))
        snap = kin_cal.update(
            timestamp_ms=ts_ms,
            ear_value=ear,
            fused_closure_score=0.90 if in_natural_blink else 0.08,
            mar_value=0.15,
            ear_close_threshold=0.165,
        )
        profile = cal.ingest_frame(
            timestamp_ms=ts_ms,
            face_detected=True,
            geometric_reliability=0.90,
            ear_value=ear,
            pitch_deg=pitch,
            yaw_deg=yaw,
            is_eye_currently_closed=in_natural_blink,
            completed_blink=snap.completed_blink,
        )
        ts_ms += 50.0

    assert profile is not None, f"Calibration failed: {cal.rejection_reason}"
    print(f"\n[1] 4-Zone Personalised Calibration Profile Created ({profile.driver_id}):")
    for zk, zb in profile.zone_baselines.items():
        print(
            f"    - Zone {zk:<13}: Open EAR Median = {zb.ear_median:.3f} (+/-{zb.ear_mad_scaled:.3f}) "
            f"| Dynamic Closure Cutoff = {zb.dynamic_closure_threshold:.3f}"
        )
    print(
        f"    - Blink Baseline    : Duration = {profile.blink_duration_median_ms:.1f} ms "
        f"| Reopening Ratio (RCR) = {profile.blink_rcr_median:.2f}"
    )

    # 2. Compare Experiments A, B, and C across a 90-second Driving Scenario (1800 frames @ 20 FPS)
    #    - 0-30s: Alert driving + regular mirror checks (Ground Truth: NORMAL)
    #    - 30-60s: Glasses glare + borderline eye openness (Ground Truth: NORMAL)
    #    - 60-90s: Progressive fatigue + sluggish blinks + microsleep (Ground Truth: WARNING/CRITICAL)
    results_by_mode: Dict[str, Dict[str, float]] = {}

    for mode in (
        ExperimentMode.A_GEOMETRIC_ONLY,
        ExperimentMode.B_CNN_ALWAYS,
        ExperimentMode.C_CONFIDENCE_GATED_CASCADE,
    ):
        cascade = ConfidenceGatedEyeCascade(mode=mode)
        kinematics = TemporalKinematicsTracker()
        risk_engine = PersonalisedAdaptiveRiskEngine(profile=profile, fixed_ear_threshold=0.22)

        sim_ts = 0.0
        false_alarms_pdbp = 0
        false_alarms_fixed = 0
        fatigue_detected_frames = 0
        total_fatigue_gt_frames = 0
        latencies_ms: List[float] = []

        for frame_idx in range(1800):
            sec = sim_ts / 1000.0
            has_glare = False
            bs_blink = 0.08

            if sec < 30.0:
                # Phase I: Alert driving with periodic 2-second left/right mirror checks
                gt_drowsy = False
                if 8.0 <= (sec % 12.0) < 10.0:
                    pitch, yaw = 0.0, -22.0
                    ear = 0.197
                else:
                    pitch, yaw = 0.0, 0.0
                    in_blink = (frame_idx % 80) in (10, 11, 12)  # 150ms crisp blink
                    ear = 0.115 if in_blink else 0.218
                    bs_blink = 0.88 if in_blink else 0.08
                crop_open = ear > 0.155
                geom_rel = 0.90
            elif sec < 60.0:
                # Phase II: Alert driving under severe glasses glare (t=36..44s & 50..56s)
                # Glare causes landmark jitter that falsely depresses geometric EAR to 0.142,
                # while the eye is actually open (blendshape=0.14 and eye crop shows open iris).
                gt_drowsy = False
                pitch, yaw = 0.0, 0.0
                in_glare_burst = (36.0 <= sec <= 44.0) or (50.0 <= sec <= 56.0)
                has_glare = in_glare_burst
                in_blink = (frame_idx % 80) in (10, 11, 12)
                if in_blink:
                    ear = 0.112
                    bs_blink = 0.88
                    crop_open = False
                    geom_rel = 0.75
                elif in_glare_burst:
                    ear = 0.142  # Falsely depressed below closure threshold (0.166)
                    bs_blink = 0.14  # Blendshape disagrees with depressed EAR
                    crop_open = True
                    geom_rel = 0.42  # Low geometric reliability under glare
                else:
                    ear = 0.216
                    bs_blink = 0.08
                    crop_open = True
                    geom_rel = 0.88
            else:
                # Phase III: Drowsy driving (prolonged 600ms sluggish blinks + 1.5s microsleep at t=74s)
                gt_drowsy = True
                pitch, yaw = -8.0, 0.0
                in_long_blink = (frame_idx % 45) in range(10, 23)  # 650ms slow blink
                in_microsleep = 74.0 <= sec <= 75.6
                is_closed_now = in_long_blink or in_microsleep
                ear = 0.105 if is_closed_now else 0.188
                bs_blink = 0.92 if is_closed_now else 0.32
                crop_open = not is_closed_now
                geom_rel = 0.85

            zone = classify_pose_zone(pitch, yaw, profile.neutral_pitch_deg, profile.neutral_yaw_deg)
            crop = make_synthetic_crop(is_open=crop_open, has_glare=has_glare)

            geom = FrameGeometryFeatures(
                timestamp_ms=sim_ts,
                face_detected=True,
                landmark_confidence=0.92,
                left_ear=ear,
                right_ear=ear,
                yaw_weighted_ear=ear,
                blendshape_blink_left=bs_blink,
                blendshape_blink_right=bs_blink,
                blendshape_blink_mean=bs_blink,
                mar=0.16,
                pitch_deg=pitch,
                yaw_deg=yaw,
                roll_deg=0.0,
                pose_zone=zone,
                geometric_reliability=geom_rel,
                left_eye_crop=crop,
                right_eye_crop=crop,
            )

            t_start = time.perf_counter()
            zb = profile.get_zone_baseline(zone)
            c_res = cascade.evaluate(
                geom=geom,
                reference_open_ear=zb.ear_median,
                close_threshold_ear=zb.dynamic_closure_threshold,
            )
            k_res = kinematics.update(
                timestamp_ms=sim_ts,
                ear_value=ear,
                fused_closure_score=c_res.fused_closure_score,
                mar_value=geom.mar,
                ear_close_threshold=zb.dynamic_closure_threshold,
            )
            r_res = risk_engine.evaluate(geom=geom, cascade=c_res, kinematics=k_res)
            latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

            if not gt_drowsy:
                if r_res.state in (AlertState.WARNING, AlertState.CRITICAL):
                    false_alarms_pdbp += 1
                if r_res.conventional_fixed_state in (AlertState.WARNING, AlertState.CRITICAL):
                    false_alarms_fixed += 1
            else:
                total_fatigue_gt_frames += 1
                if r_res.state in (AlertState.WARNING, AlertState.CRITICAL):
                    fatigue_detected_frames += 1

            sim_ts += 50.0

        recall_pct = (fatigue_detected_frames / max(1, total_fatigue_gt_frames)) * 100.0
        results_by_mode[mode.value] = {
            "fixed_false_alarm_frames": float(false_alarms_fixed),
            "pdbp_false_alarm_frames": float(false_alarms_pdbp),
            "drowsy_recall_pct": round(recall_pct, 1),
            "cnn_trigger_rate_pct": round(cascade.cnn_trigger_rate * 100.0, 1),
            "mean_stage_latency_ms": round(float(np.mean(latencies_ms)), 3),
            "disagreement_audits": float(len(cascade.disagreement_events)),
        }

    print("\n[2] Experimental Comparison Table (1,800 frames / 90s Multi-Condition Scenario):")
    print("-" * 88)
    print(
        f"{'Experiment Mode':<38} | {'Fixed FA':<8} | {'PDBP FA':<8} | {'Recall':<7} | {'CNN Gate %':<10} | {'Latency'}"
    )
    print("-" * 88)
    for mode_name, stats in results_by_mode.items():
        print(
            f"{mode_name:<38} | {int(stats['fixed_false_alarm_frames']):<8} | "
            f"{int(stats['pdbp_false_alarm_frames']):<8} | {stats['drowsy_recall_pct']:>5.1f}% | "
            f"{stats['cnn_trigger_rate_pct']:>8.1f}% | {stats['mean_stage_latency_ms']:.3f} ms"
        )
    print("-" * 88)

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile.save_json(out_dir / "sample_calibrated_profile.json")
    (out_dir / "phase1_benchmark_results.json").write_text(
        json.dumps(results_by_mode, indent=2), encoding="utf-8"
    )
    print(f"\nSaved calibrated profile and benchmark artifacts to: {out_dir}")
    return results_by_mode


def run_live_webcam(camera_index: int = 0) -> None:
    """Launch the interactive real-time webcam prototype."""
    from backend.app.ai.landmarker import DriveGuardFaceLandmarker

    print("Initializing MediaPipe Face Landmarker & DriveGuard Adaptive Engine...")
    landmarker = DriveGuardFaceLandmarker()
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    profile_path = out_dir / "live_driver_profile.json"

    if profile_path.is_file():
        loaded = DriverBehaviourProfile.load_json(profile_path)
        if loaded.calibration_mode != "UNCALIBRATED_DEFAULT":
            profile = loaded
            print(
                f"Loaded saved calibrated profile ({profile.calibration_mode}): "
                f"Center EAR={profile.center_ear_median:.3f}, Blink Dur={profile.blink_duration_median_ms:.0f}ms"
            )
        else:
            profile = DriverBehaviourProfile.create_default_uncalibrated(driver_id="live_driver")
    else:
        profile = DriverBehaviourProfile.create_default_uncalibrated(driver_id="live_driver")

    calibration = GuidedCalibrationSession(driver_id="live_driver", four_zone_mode=True)
    cascade = ConfidenceGatedEyeCascade(mode=ExperimentMode.C_CONFIDENCE_GATED_CASCADE)
    kinematics = TemporalKinematicsTracker()
    risk_engine = PersonalisedAdaptiveRiskEngine(profile=profile)
    recorded_telemetry: List[Dict[str, object]] = []

    cap = cv2.VideoCapture(camera_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print(f"ERROR: Could not open camera index {camera_index}.")
        return

    fps_ema = 25.0
    prev_wall = time.perf_counter()
    process = psutil.Process()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            now_wall = time.perf_counter()
            dt = max(1e-3, now_wall - prev_wall)
            prev_wall = now_wall
            fps_ema = 0.90 * fps_ema + 0.10 * (1.0 / dt)
            ts_ms = now_wall * 1000.0

            geom, landmarks_xyz = landmarker.process_frame(
                frame_bgr=frame_bgr,
                timestamp_ms=ts_ms,
                neutral_pitch_deg=risk_engine.profile.neutral_pitch_deg,
                neutral_yaw_deg=risk_engine.profile.neutral_yaw_deg,
                reference_open_ear=risk_engine.profile.center_ear_median,
                extract_crops=True,
            )

            zb = risk_engine.profile.get_zone_baseline(geom.pose_zone)
            c_res = cascade.evaluate(
                geom=geom,
                reference_open_ear=zb.ear_median,
                close_threshold_ear=zb.dynamic_closure_threshold,
            )
            k_res = kinematics.update(
                timestamp_ms=ts_ms,
                ear_value=geom.yaw_weighted_ear,
                fused_closure_score=c_res.fused_closure_score,
                mar_value=geom.mar,
                ear_close_threshold=zb.dynamic_closure_threshold,
            )

            if calibration.is_running:
                new_profile = calibration.ingest_frame(
                    timestamp_ms=ts_ms,
                    face_detected=geom.face_detected,
                    geometric_reliability=geom.geometric_reliability,
                    ear_value=geom.yaw_weighted_ear,
                    pitch_deg=geom.pitch_deg,
                    yaw_deg=geom.yaw_deg,
                    is_eye_currently_closed=k_res.is_eye_closed,
                    completed_blink=k_res.completed_blink,
                )
                if new_profile is not None:
                    risk_engine.set_profile(new_profile)
                    new_profile.save_json(profile_path)
                    print(
                        f"\n[CALIBRATION COMPLETE & AUTO-SAVED] Mode: {new_profile.calibration_mode} "
                        f"| Center Open EAR: {new_profile.center_ear_median:.3f} "
                        f"| Blink Dur: {new_profile.blink_duration_median_ms:.0f} ms\n"
                        f"Saved to: {profile_path}"
                    )
                elif not calibration.is_running and calibration.rejection_reason:
                    print(f"\n[CALIBRATION REJECTED] {calibration.rejection_reason}")
                    calibration.rejection_reason = None

            r_res = risk_engine.evaluate(geom=geom, cascade=c_res, kinematics=k_res)

            recorded_telemetry.append(
                {
                    "timestamp_ms": round(ts_ms, 1),
                    "face_detected": geom.face_detected,
                    "ear": geom.yaw_weighted_ear,
                    "smoothed_ear": round(k_res.smoothed_ear, 4),
                    "mar": geom.mar,
                    "pitch_deg": geom.pitch_deg,
                    "yaw_deg": geom.yaw_deg,
                    "pose_zone": geom.pose_zone.value,
                    "geom_reliability": geom.geometric_reliability,
                    "cnn_invoked": c_res.cnn_invoked,
                    "fused_closure": round(c_res.fused_closure_score, 3),
                    "perclos_60s": round(k_res.perclos_60s_pct, 2),
                    "blink_dur_ms": k_res.recent_blink_duration_ms,
                    "blink_rcr": k_res.recent_blink_rcr,
                    "risk_score": r_res.risk_score,
                    "pdbp_state": r_res.state.value,
                    "fixed_state": r_res.conventional_fixed_state.value,
                    "reasons": r_res.reasons,
                }
            )

            hud = draw_hud_overlay(
                frame_bgr=frame_bgr,
                landmarks_xyz=landmarks_xyz,
                geom=geom,
                cascade_res=c_res,
                kinematics_res=k_res,
                risk_res=r_res,
                calibration=calibration,
                fps=fps_ema,
            )
            cv2.imshow("DriveGuard — Phase 1 Personalised Adaptive Prototype", hud)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            elif key in (ord("c"), ord("C")):
                calibration = GuidedCalibrationSession(driver_id="live_driver", four_zone_mode=True)
                calibration.start(ts_ms)
            elif key == ord("1"):
                calibration = GuidedCalibrationSession(driver_id="live_driver", four_zone_mode=False)
                calibration.start(ts_ms)
            elif key in (ord("a"), ord("A")):
                cascade.mode = ExperimentMode.A_GEOMETRIC_ONLY
            elif key in (ord("b"), ord("B")):
                cascade.mode = ExperimentMode.B_CNN_ALWAYS
            elif key in (ord("g"), ord("G")):
                cascade.mode = ExperimentMode.C_CONFIDENCE_GATED_CASCADE
            elif key in (ord("s"), ord("S")):
                risk_engine.profile.save_json(profile_path)
                (out_dir / "live_session_telemetry.json").write_text(
                    json.dumps(recorded_telemetry[-1800:], indent=2), encoding="utf-8"
                )
                print(
                    f"Saved profile to {profile_path} and {len(recorded_telemetry[-1800:])} telemetry frames to {out_dir / 'live_session_telemetry.json'}"
                )
    finally:
        cap.release()
        cv2.destroyAllWindows()
        landmarker.close()
        if recorded_telemetry:
            (out_dir / "live_session_telemetry.json").write_text(
                json.dumps(recorded_telemetry[-1800:], indent=2), encoding="utf-8"
            )
        print(
            f"Session Summary: Total Frames={cascade.total_frames}, "
            f"CNN Gate Trigger Rate={cascade.cnn_trigger_rate:.1%}, "
            f"Disagreement Audits={len(cascade.disagreement_events)}, "
            f"Memory RSS={process.memory_info().rss / (1024 * 1024):.1f} MB"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="DriveGuard Phase 1 Webcam Prototype & Benchmark")
    parser.add_argument("--camera", type=int, default=0, help="Webcam device index (default: 0)")
    parser.add_argument(
        "--benchmark-simulation",
        action="store_true",
        help="Run headless experimental benchmark across 4-zone calibration and Experiments A/B/C",
    )
    args = parser.parse_args()

    if args.benchmark_simulation:
        run_benchmark_simulation()
    else:
        run_live_webcam(camera_index=args.camera)


if __name__ == "__main__":
    main()
