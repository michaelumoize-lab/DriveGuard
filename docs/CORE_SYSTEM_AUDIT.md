# DriveGuard — Comprehensive Core System Audit Report

**Project:** DriveGuard — Personalized AI Driver Drowsiness Detection System  
**Branch:** `core-engine`  
**Audit Date:** 2026-11-20  
**Audit Mode:** Strictly Read-Only (inspection, verification, and test execution only)  
**Overall Verdict:** **FAIL** *(at initial audit — all 28 findings subsequently remediated and verified in [docs/CORE_AUDIT_REMEDIATION.md](./CORE_AUDIT_REMEDIATION.md); 668/668 automated tests passing)*

---

## Quick Fix-Tracking Checklist

Use this checklist to verify that every discovered defect has been resolved and covered by regression tests before re-auditing or advancing to Phase 9 hardware work.

### Safety-Critical (`CRITICAL`)
- [x] **CRITICAL-1:** False-safe `NORMAL` state during sustained eye closure when `head_pose.is_valid == False` + mild `QualityStatus.BLURRY` (`drowsiness_engine.py:118-121`, `temporal_analyzer.py:201-204`, `confidence.py:128-187`, `decision_engine.py:163-166`)
- [x] **CRITICAL-2:** Zero-hysteresis microsleep reset on single-frame EAR jitter into `(closure_threshold, reopen_threshold)` (`blink_detector.py:536-537,653-661`, `ear.py:364-376`, `temporal_analyzer.py:240-251`)

### High Severity (`HIGH`)
- [x] **HIGH-1:** Pitch-compensated `compensated_ear` passed to `GuidedCalibrationSession.ingest_frame()`, contaminating DASHBOARD / oblique-CENTER baselines (`scripts/run_camera_test.py:1024`)
- [x] **HIGH-2:** Unbounded `BlinkState.REOPENING` deadlock (freezes profile/threshold switching and fabricates multi-second `MICROSLEEP` blinks) (`blink_detector.py:539-662`)
- [x] **HIGH-3:** `DecisionEngine` Timer 1 & `AlertStateManager.is_recovering` desynchronization on closed-eye reacquisition (Scenario L) and eye occlusion (Scenario I), falsely suppressing CRITICAL audio reminders (`decision_engine.py:455-488`, `alert_state_manager.py:152-153`, `alert_dispatcher.py:165`)
- [x] **HIGH-4:** `DriverFeatureExtractor` accepts `UNCALIBRATED_DEFAULT` (`CENTER.sample_count == 0`) as a valid calibrated profile (`feature_extractor.py:660-765`)
- [x] **HIGH-5:** Calibration quality gates bypassed via hardcoded `geometric_reliability=0.95`, hardcoded `presence_confidence=1.0`, missing `FrameQualityResult` check, and silent completion when $SO(3)$ neutral rotation aggregation rejects (`scripts/run_camera_test.py:1020-1034`, `vision/face_landmarker.py:375`, `pdbp_calibration.py:1810-1824`)
- [x] **HIGH-6:** `head_pose.py` 2D profile fallback fabricates `rot_mat = np.eye(3)` with `is_valid=True` and confuses camera-relative `nose_ratio` with neutral-relative yaw (`head_pose.py:563-618`)
- [x] **HIGH-7:** Persisted `neutral_rot_mat` breaks on mirror mode toggle (`'m'` / `--no-flip-h`) because `is_mirrored` is not stored or normalized (`head_pose.py:233-236,620-657`, `schemas/profile.py:72-75`)

### Medium Severity (`MEDIUM`)
- [x] **MEDIUM-1:** `AlertEventLogger` formats monotonic uptime via `time.gmtime()`, logging 1970 ISO timestamps (`alerts/event_logger.py:88`)
- [x] **MEDIUM-2:** `YawnDetector` uses `max_sampling_gap_ms = 500 ms` (vs `250 ms` elsewhere), lacks `<= max_yawn_duration_ms` cap on active yawn flags, keeps yawn flags `True` during `is_valid=False` dropout grace, and `DecisionEngine` ignores `confidence.yawn_viable` while firing `AUTHENTIC_YAWN` on uncompleted candidates (`mar.py:78,365-370,491-496`, `decision_engine.py:376-385`)
- [x] **MEDIUM-3:** `BlinkKinematicsDetector` returns `NaN` / negative `current_closure_duration_ms` on rejected invalid/backward timestamps (`blink_detector.py:287-291,323-327`)
- [x] **MEDIUM-4:** `run_camera_test.py` deferred calibration UI claims immediate activation with old thresholds, never reverts `calibrated_label` on deferred rollback, omits `existing_profile` on `'c'` recalibration, and retains broken `DriverProfileDTO.from_dict()` branch (`scripts/run_camera_test.py:785-787,1042-1048,1234`)
- [x] **MEDIUM-5:** Contradictory `primary_reason` (`ALERT_NORMAL`) emitted during elevated-state hysteresis cooldown hold (`decision_engine.py:546`, `scripts/run_camera_test.py:524`)
- [x] **MEDIUM-6:** `AlertStateManager.force_recovery()` passes `code=DrowsinessState.NORMAL` instead of `ReasonCode.ALERT_NORMAL` (`alerts/alert_state_manager.py:232`)
- [x] **MEDIUM-7:** Head-pose proxy terminology labeled as `"gaze"` / `"off-road gaze dwell"` without caveat (`head_pose.py:57`, `temporal_analyzer.py:29,117`, `decision_engine.py:341,426`)
- [x] **MEDIUM-8:** Asymmetric closure discrepancy where `BlinkKinematicsDetector` enters `CLOSED`/`PROLONGED_CLOSURE` on `weighted_ear < c` while `ear.is_closed == False`, `blink.is_eye_closed == False`, and `PERCLOS` records OPEN (`blink_detector.py:361,677`)
- [x] **MEDIUM-9:** `CLOSING` stall re-anchor updates `_blink_start_ts` without updating `_open_ear_before`, biasing `v_close` high and `rcr` low (`blink_detector.py:504,512`)
- [x] **MEDIUM-10:** Unbounded memory lists (`latencies_ms`, `inference_latencies_ms`, `_completed_inference_timestamps_ms`) and redundant full-resolution frame copies / hot-loop disk I/O (`scripts/run_camera_test.py:885-886`, `eye_shadow_observer.py:210,290,428`, `frame_quality.py:172`, `session_recorder.py:194-225`)

### Low Severity (`LOW`)
- [x] **LOW-1:** Explicitly pass `is_normalized=True` from `feature_extractor.py` to `extract_eye_aspect_ratio` and `YawnDetector.update`
- [x] **LOW-2:** Replace falsy `self._min_ear_ts or ...` with `if self._min_ear_ts is not None` (`blink_detector.py:566`)
- [x] **LOW-3:** Align face-loss `closure_threshold` reporting in `feature_extractor.py:1250` with uncalibrated (`0.0`) / active threshold
- [x] **LOW-4:** Default unseen `_last_announced_ts_ms` in `alert_dispatcher.py:195` to `None` instead of `0.0`, and log periodic CRITICAL reminders to `AlertEventLogger`
- [x] **LOW-5:** Use frame `timestamp_ms` instead of wall-clock `time.monotonic()` for face-loss grace in `vision/face_landmarker.py:363,396`
- [x] **LOW-6:** Expose `severe_perclos_threshold_pct` in `DrowsinessEngine.__init__` and emit the actual triggering closure threshold (`500 ms` vs `800 ms`) in `NODDING_MICROSLEEP`
- [x] **LOW-7:** Guard `EyeCNNShadowWorker` diagnostic counters and `OpenCVCamera._is_connected`/`_cap` under locks; avoid mutating `FramePacket.frame` in place in `draw_landmarks()`

### End-to-End Regression Tests & Architecture Consolidation
- [x] **TEST-1:** Add full pipeline integration tests (`FrameQualityAnalyzer → DriverFeatureExtractor → DrowsinessEngine → AlertStateManager → AlertDispatcher` and `GuidedCalibrationSession → commit_and_activate_profile → DriverFeatureExtractor`) covering CRITICAL-1/2, HIGH-1..7, and Scenarios C, D, E, F, H, I, J, K, L
- [x] **ARCH-1:** Extract authoritative local pipeline orchestration out of `scripts/run_camera_test.py` into a reusable backend service exposing processed telemetry, and deprecate/isolate legacy raw-video `SessionAIEngine` paths

---

## A. Executive Summary

DriveGuard's modular Phase 5–8 core intelligence pipeline demonstrates **strong individual-module engineering**: clean dataclasses, C1-continuous yaw-weighted EAR with mathematically valid mirror symmetry, boundary-clipped sliding-window PERCLOS integration, rigid cranial SQPnP head pose with full 3D $SO(3)$ neutral-frame decoupling ($R_{\text{relative}} = R_{\text{neutral}}^T R_{\text{pnp}}$), geodesic medoid $SO(3)$ calibration aggregation, canonical SHA-256 profile fingerprinting with transactional rollback, and a well-structured Phase 8 alert dispatch stack. All **583** default-collected automated tests pass.

However, **cross-module reasoning and end-to-end execution tracing reveal severe safety and integration defects that isolated synthetic unit tests did not catch**:

1. **Verified False-Safe Pathway (CRITICAL-1):** When head pose fails alongside mild blur (`QualityStatus.BLURRY`, still viable for inference), scalar `overall_confidence` drops to `0.49 < 0.50` (`is_reliable = False`) while `ocular_viable` remains `True`. Because `backend/app/intelligence/drowsiness_engine.py:118-121` passes scalar `is_reliable` rather than channel-specific `ocular_viable` to `TemporalAnalyzer`, eye-closure tracking is silently zeroed (`active_closure_duration_ms = 0.0`) while `DecisionEngine` sees `is_primary_viable = True` and reports **`NORMAL` indefinitely during sustained 2,000+ ms eye closure**.
2. **Verified Microsleep Reset Under Single-Frame Jitter (CRITICAL-2):** `BlinkKinematicsDetector` exits `CLOSED`/`PROLONGED_CLOSURE` as soon as `ear > closure_threshold` (ignoring `reopen_threshold` hysteresis) and resets `_closure_start_ts` upon re-entering `CLOSED` from `REOPENING`. A single noisy frame bouncing `EAR` to `closure_threshold + 0.001` (still well below `reopen_threshold`) wipes out a 1,400 ms continuous eye closure back to ~33 ms in both `BlinkKinematicsDetector` and `TemporalAnalyzer`, delaying or preventing `CRITICAL_WARNING`.
3. **High-Severity Calibration, State-Machine, and Continuity Defects:** Pitch-compensated EAR contaminates personalized calibration in `scripts/run_camera_test.py:1024`; `BlinkState.REOPENING` has no timeout (causing indefinite threshold-latch deadlock and fabricated multi-second blinks); `AlertStateManager` falsely sets `is_recovering = True` (suppressing CRITICAL audio reminders) immediately after prolonged face loss even when the driver reappears with **eyes closed**; `DriverFeatureExtractor` accepts `UNCALIBRATED_DEFAULT` (`sample_count = 0`) as a valid calibrated profile; calibration quality gates are effectively bypassed by hardcoded `geometric_reliability = 0.95` and `presence_confidence = 1.0`.
4. **Architectural Split:** The authoritative local pipeline exists **only** inside `scripts/run_camera_test.py`, while the FastAPI server (`backend/app/main.py`) still serves the legacy raw-video WebSocket and LiveKit `SessionAIEngine` stack backed by duplicated, divergent implementations of landmarker, geometry, blink kinematics, PERCLOS, and risk scoring.

---

## B. Actual Architecture

### Verified Authoritative Local Processing Pipeline

The real entry point for the authoritative local webcam pipeline is **`main()` → `run_benchmark()`** in `scripts/run_camera_test.py` (lines 727–1489):

```text
Local Camera (OpenCVCamera / PhoneCamera via camera_manager.create_camera)
    |  FramePacket(frame: BGR np.ndarray, timestamp_ns, frame_index, source_id)
    v
FrameQualityAnalyzer.analyze(packet.frame)              [backend/app/camera/frame_quality.py]
    |  -> FrameQualityResult (luminance, Laplacian blur_variance, status, is_viable_for_inference)
    v
DriveGuardFaceLandmarker.detect(packet)                 [backend/app/vision/face_landmarker.py]
    |  -> FaceLandmarkResult (478 NormalizedLandmark, timestamp_ms: int, status, hardcoded conf=1.0)
    v
DriverFeatureExtractor.extract(landmark_result, w, h, is_mirrored)  [backend/app/features/feature_extractor.py]
    |
    +--> 1. estimate_head_pose(...)                     [backend/app/features/head_pose.py]
    |         -> HeadPoseResult (abs/rel P,Y,R, rot_mat, zone, is_looking_away, is_valid)
    +--> 2. YawnDetector.update(...)                    [backend/app/features/mar.py]
    |         -> MouthAspectRatioResult (mar, is_mouth_open, completed_yawn, baseline_mar, is_valid)
    +--> 3. PDBP Safe-Latch & Threshold Selection       [feature_extractor.py:1100-1134]
    |         (applies pending disable/switch/zone thresholds only when not in active blink)
    +--> 4. extract_eye_aspect_ratio(...)               [backend/app/features/ear.py]
    |         (camera-relative yaw_deg, neutral-relative rel_pitch_deg, active_c)
    |         -> EyeAspectRatioResult (left/right/raw/weighted/compensated EAR, is_closed, is_valid)
    +--> 5. BlinkKinematicsDetector.update(...)         [backend/app/features/blink_detector.py]
    |         (receives ear_res.weighted_ear, active_c, active_r)
    |         -> BlinkSnapshot (state, is_eye_closed, current_closure_duration_ms, completed_blink)
    +--> 6. PerclosCalculator.update(...)               [backend/app/features/perclos.py]
    |         (receives ear_res.is_closed if ear_res.is_valid else None)
    |         -> PerclosResult (perclos_pct, coverage_pct, is_reliable, effective_perclos_pct)
    |
    |  -> DriverFeatures (unified immutable record)
    v
[Side-branch A: Passive Experimental] EyeCNNShadowWorker.submit_frame(...)  [backend/app/ai/eye_shadow_observer.py]
[Side-branch B: Interactive Calibration] GuidedCalibrationSession.ingest_frame(...) [backend/app/ai/pdbp_calibration.py]
    |  (on completion: commit_and_activate_profile -> ProfileStore + feature_extractor.set_profile)
    v
DrowsinessEngine.evaluate(features, quality)            [backend/app/intelligence/drowsiness_engine.py]
    |
    +--> 1. SignalConfidenceEvaluator.evaluate(...)     [backend/app/intelligence/confidence.py]
    |         -> ConfidenceAssessment (overall_confidence, is_reliable, head_pose/ocular/yawn_viable)
    +--> 2. TemporalAnalyzer.update(features, is_reliable=confidence.is_reliable)  [temporal_analyzer.py]
    |         -> TemporalMetrics (active_closure, off_road/downward dwell, nodding, 30s/60s blinks/yawns)
    +--> 3. DecisionEngine.evaluate(metrics, confidence) [backend/app/intelligence/decision_engine.py]
    |         -> DecisionResult (state, severity, reasons, alert_active, unresolved_alert_state)
    |
    |  -> DrowsinessAssessment
    v
AlertStateManager.update(assessment)                    [backend/app/alerts/alert_state_manager.py]
    |  -> Optional[AlertTransition]
    v
AlertDispatcher.dispatch(transition, current_state, packet.timestamp_ms, is_recovering) [alert_dispatcher.py]
    +--> AudioWarningEngine (SoundDeviceAudioBackend, priority queue)  [audio_warning_engine.py]
    +--> AlertEventLogger (ring buffer + data/logs/alert_events.jsonl) [event_logger.py]
    +--> Visual HUD (draw_diagnostic_hud) & SessionRecorder (.mp4 + .csv/.jsonl)
```

### Architectural Inconsistencies with the Intended Local-First Direction

1. **Active Raw-Video Streaming Endpoints Still Mounted:** `backend/app/main.py:173-182` mounts `/ws/v1/session/{session_id}` (`backend/app/api/ws_telemetry.py:25-120`), which accepts up to 2 MB raw JPEG/PNG binary video frames over WebSockets, and `/api/v1/livekit/*` (`backend/app/services/livekit_worker.py`) for server-side raw WebRTC video ingestion—both wired exclusively to the legacy `SessionAIEngine` (`backend/app/ai/pipeline.py`).
2. **Authoritative Pipeline Not Exposed as a Service:** The authoritative Phase 5–8 intelligence pipeline is orchestrated only inside the CLI/OpenCV script `scripts/run_camera_test.py`; no FastAPI route or local telemetry publisher exposes `DriverFeatureExtractor` / `DrowsinessEngine` / `AlertStateManager` outputs to the frontend.
3. **Cross-Layer Coupling to Legacy Types:** Shared `backend/app/ai/pdbp_calibration.py:30` imports `BlinkEvent` from legacy `backend/app/ai/blink_kinematics.py`, forcing `scripts/run_camera_test.py:995-1013` to manually adapt `features.blink_detector.BlinkEvent` into `ai.blink_kinematics.BlinkEvent` during calibration.

---

## C. Module-by-Module Findings

### 1. Frame Quality & Face Landmarker (`backend/app/camera/frame_quality.py`, `backend/app/vision/face_landmarker.py`)
- **What it does:** `FrameQualityAnalyzer` computes grayscale mean luminance and full-frame float64 Laplacian variance; `DriveGuardFaceLandmarker` runs MediaPipe Tasks `FaceLandmarker` in `RunningMode.VIDEO` returning 478 normalized landmarks and a 0.4 s face-loss grace state.
- **Correctness & Integration:**
  - `DriveGuardFaceLandmarker.detect()` hardcodes `confidence = 1.0` (`vision/face_landmarker.py:375`), making `FaceLandmarkStatus.OCCLUDED_OR_MARGINAL` unreachable in production and making `landmark_confidence >= 0.35` in calibration always true.
  - Face-loss grace uses wall-clock `time.monotonic()` (`vision/face_landmarker.py:363,396`) rather than packet `timestamp_ms`, breaking deterministic replay on recorded video.
  - Timestamps are truncated to integer ms (`timestamp_ns // 1_000_000`) and incremented by +1 on duplicates (`vision/face_landmarker.py:312,338-340`), creating a minor timestamp-base split vs. float `packet.timestamp_ms` used by `AlertDispatcher`, `GuidedCalibrationSession`, `SessionRecorder`, and `EyeCNNShadowWorker`.
  - `draw_landmarks()` (`vision/face_landmarker.py:423-475`) mutates `packet.frame` in place despite `FramePacket` being `frozen=True`.
- **Test coverage:** Unit tested with synthetic frames/mocks (`test_frame_quality.py`, `test_face_landmarker.py`); no test checks real confidence extraction or offline deterministic grace timing.

### 2. EAR & Eye-State Processing (`backend/app/features/ear.py`)
- **What it does:** Extracts 6-point canonical MediaPipe eye contours (`LEFT_EYE_INDICES = (33,160,158,133,153,144)`, `RIGHT_EYE_INDICES = (362,385,387,263,373,380)`), computes aspect-ratio-corrected single-eye EAR with finite/collapse/plausibility (`0.0 <= ear <= 1.5`) checks, C1-continuous Hermite smoothstep yaw weighting over `[-30°, +30°]` bounded in `[0.20, 0.80]`, unilateral 100% fallback, and bounded downward-pitch compensation (`<= +18%`, inactive when `base_ear <= closure_threshold`).
- **Mathematical & Orientation Verification:**
  - **Landmark math & validity:** Correct Euclidean distances with `ar_factor = aspect_ratio` in normalized space and `1.0` in pixel space. Genuinely closed eyes (`ear` near `0.0`) remain `is_valid = True`.
  - **Mirror convention:** Verified geometrically—when `is_mirrored=False`, turning to driver's left (`+yaw`) faces landmark `33` (`left_ear`) toward the camera (`w_left -> 0.80`); when `is_mirrored=True` (`cv2.flip(frame, 1)`), MediaPipe labels the reflected camera-facing eye as `263` (`right_ear`), `head_pose.py` still outputs `+yaw`, and `ear.py:265` negates `effective_yaw` so `w_right -> 0.80`. **Consistent.**
  - **Camera-relative vs neutral-relative orientation:** `DriverFeatureExtractor` properly passes camera-relative `pose_res.yaw_deg` for optical EAR weighting and neutral-relative `pose_res.rel_pitch_deg` for pitch compensation (`feature_extractor.py:1187-1188`).
- **Limitations / Defects:**
  - `extract_eye_aspect_ratio()` is called without explicit `is_normalized=True` (`feature_extractor.py:1139,1185`); off-frame extrapolated normalized coordinates `> 2.0` would silently flip to pixel mode (`ar_factor = 1.0`).
  - Collapsed-width floor `eps = 1e-6` (`ear.py:182`) allows sub-pixel eye widths (`0.01 px`).
  - **No hysteresis in `ear.is_closed`** (`left_ear < c and right_ear < c`) vs. `BlinkKinematicsDetector` driving state transitions on `weighted_ear < c` and `reopen_threshold`. Asymmetric closure where one eye `< c`, the other `>= c`, and `weighted_ear < c` drives `BlinkKinematicsDetector` into `CLOSED`/`PROLONGED_CLOSURE` while `ear.is_closed == False`, `blink.is_eye_closed == False`, and `PERCLOS` records OPEN.
  - **Calibration contamination:** `compensated_ear` is passed to `GuidedCalibrationSession.ingest_frame()` in `scripts/run_camera_test.py:1024`, violating the pitch-compensation isolation contract.

### 3. PERCLOS (`backend/app/features/perclos.py`)
- **What it does:** Rolling 60 s time-weighted P80 eye-closure accumulator using `(start_ts_ms, end_ts_ms, is_closed)` intervals with exact window-boundary clipping, monotonic timestamp enforcement, sampling-gap exclusion (`dt > 0.25 s`), and `min_coverage_pct = 50%` reliability gating via `effective_perclos_pct`.
- **Correctness & Integration:**
  - Mathematical integration, boundary clipping, unobserved gap exclusion, and invalid-EAR exclusion (`is_closed=None, is_valid=False`) are **correct**.
  - `DecisionEngine` properly checks `metrics.perclos_is_reliable` before every PERCLOS threshold comparison.
- **Limitations:**
  - HUD (`scripts/run_camera_test.py:496`), `SessionRecorder` (`scripts/run_camera_test.py:1109`), and `AlertStateManager` details (`alert_state_manager.py:198`) display/log raw `perclos_pct` (`0.0%` when uncalibrated or during warmup) without indicating unavailability.
  - Duplicate, inconsistent PERCLOS calculation exists in legacy `backend/app/ai/blink_kinematics.py:459-484` (no boundary clipping, fabricates `33.3 ms` post-gap intervals).

### 4. MAR & Yawn Detection (`backend/app/features/mar.py`)
- **What it does:** Computes 3-pair inner-lip vertical mean divided by outer-corner width `(61, 291)` with Schmitt hysteresis (`open >= 0.55`, `hold >= 0.45`), 200 ms dropout grace, 500 ms max sampling gap, `1500–8000 ms` observed duration window, and resting baseline EMA (`mar <= 0.28`, gated against `is_looking_away`).
- **Correctness & Integration:**
  - Geometry math and baseline isolation from active yawns/head turns/invalid frames are sound (minor: quiet speech with `mar <= 0.28` can gently drift diagnostic `baseline_mar`, though thresholds are fixed).
  - **Defects:**
    1. `max_sampling_gap_ms = 500.0` (`mar.py:78`) is double the `250.0 ms` gap threshold used by `BlinkKinematicsDetector`, `PerclosCalculator`, `TemporalAnalyzer`, and `DecisionEngine`, allowing a 251–500 ms camera freeze to be credited as observed open-mouth duration (`mar.py:421-423`).
    2. `is_authentic_yawn` and `is_yawn_in_progress` (`mar.py:491-496`) lack an upper bound `<= max_yawn_duration_ms` (`8000 ms`) and remain `True` during dropout grace even when `is_valid=False` (`mar.py:365-370`).
    3. `DecisionEngine` triggers `ReasonCode.AUTHENTIC_YAWN` ("Authentic yawn detected") on an **uncompleted candidate** (`active_yawn_duration_ms >= 1500.0`, `decision_engine.py:376-385`) and **never checks `confidence.yawn_viable`**.
    4. `_dropout_start_ts` is anchored at the first invalid frame rather than the last valid frame, extending effective dropout grace by up to 1 frame interval.

### 5. Blink Detection & Kinematics (`backend/app/features/blink_detector.py`)
- **What it does:** 6-state FSM (`OPEN`, `CLOSING`, `CLOSED`, `REOPENING`, `PROLONGED_CLOSURE`, `UNCERTAIN_GAP`) with sub-frame linear threshold crossing interpolation, stall/hover re-anchoring in `CLOSING`, left-censoring across gaps, and velocity/RCR reliability gating (`<= 5 ms` intervals marked unreliable with `0.0`).
- **Correctness & Integration Defects:**
  - **CRITICAL-2:** Exiting `CLOSED`/`PROLONGED_CLOSURE` at `ear_val > closure_threshold` (`blink_detector.py:536`) bypasses `reopen_threshold` hysteresis; re-entering `CLOSED` from `REOPENING` overwrites `self._closure_start_ts` and resets `self._state = BlinkState.CLOSED` (`blink_detector.py:653-661`), wiping out accumulated closure duration on a single noisy frame.
  - **HIGH-2:** `BlinkState.REOPENING` has **no timeout** (unlike `CLOSING`'s 1,000 ms timeout at line 483). Prolonged hovering in `[closure_threshold, reopen_threshold)` (e.g., drooping eyelids) deadlocks the detector in `REOPENING`, blocks profile/threshold switching (`_is_in_active_blink() == True`), and fabricates a multi-second `MICROSLEEP` `BlinkEvent` upon eventual rise above `reopen_threshold`.
  - **MEDIUM-3:** Rejected invalid (`NaN`, `Inf`, negative) or backwards timestamps while `CLOSED` return corrupted `current_closure_duration_ms = timestamp_ms - _closure_start_ts` (`NaN` or negative!) at `blink_detector.py:287-291` and `323-327`.
  - **MEDIUM-9:** When `CLOSING` stalls and re-anchors `_blink_start_ts` to `_prev_ts` (`blink_detector.py:504,512`), `_open_ear_before` is not updated to `_closing_onset_ear`, biasing `v_close` high and `rcr` low.
  - **LOW-2:** `min_ts = self._min_ear_ts or ...` (`blink_detector.py:566`) treats `0.0` as falsy; on the exact `CLOSED -> REOPENING` transition frame, `current_closure_duration_ms` leaks the pre-transition duration despite `is_eye_closed=False` (`blink_detector.py:529,537,677`).

### 6. Head Pose & Visual Gaze (`backend/app/features/head_pose.py`)
- **What it does:** Rigid 5-point cranial `SOLVEPNP_SQPNP` (jaw-invariant) with `validate_pnp_solution()` (finite, depth `[100, 3000]` mm, $SO(3)$ check, per-point positive depth, normalized reprojection residual `<= 0.12 / 0.25 * d_ref`), 8-point iterative fallback, 2D profile fallback, 3D neutral decoupling $R_{\text{relative}} = R_{\text{neutral}}^T R_{\text{pnp}}$, closed-form intrinsic Y-X-Z Euler decomposition, and 6-zone classification with `1.5°` hysteresis.
- **Mathematical Verification:**
  - $R_{\text{relative}} = R_{\text{neutral}}^T R_{\text{pnp}}$ multiplication order and `extract_euler_yxz()` formulas (including gimbal-lock branch) are **mathematically exact**, preserving nonzero roll cleanly.
- **Defects & Limitations:**
  - **HIGH-6:** The 2D profile fallback (`head_pose.py:563-618`) treats camera-relative `nose_ratio` as neutral-relative `approx_rel_yaw = ±28°`, adds `neutral_yaw_deg` onto it, misapplies `neutral_pitch_deg` inside `classify_pose_zone(pitch_deg=0.0, ...)` while returning `rel_pitch_deg=0.0`, and returns a **fabricated `rot_mat = np.eye(3)` with `is_valid=True`** (which can even be ingested as a valid rotation sample during fresh CENTER calibration because `28° <= 32°`).
  - **HIGH-7:** `neutral_rot_mat` is stored in the camera coordinate frame without recording `is_mirrored`. Toggling mirror mode (`'m'` in `run_camera_test.py`) or launching with `--no-flip-h` flips the sign of `solvePnP`'s yaw rotation while `neutral_rot_mat` remains unflipped, introducing a $2 \times \text{neutral\_yaw}$ error (e.g., 50° error on an oblique 25° mount).
  - **MEDIUM-7:** Terminology in `head_pose.py`, `temporal_analyzer.py`, and `decision_engine.py` repeatedly refers to `"gaze"` and `"off-road gaze dwell"` despite measuring strictly **head pose** (iris landmarks 468–477 are only drawn, never analyzed).

### 7. Driver Feature Extractor (`backend/app/features/feature_extractor.py`)
- **What it does:** Orchestrates head pose → MAR/yawn → PDBP threshold safe-latch → EAR → blink detector → PERCLOS with atomic `RuntimeSnapshot` rollback and fingerprint-bound persistence rollback callbacks.
- **Correctness & Integration:**
  - Processing order, uncalibrated ocular deactivation (without fabricating face loss or open eyes), and mid-blink threshold latching are well designed.
  - **HIGH-4:** `validate_profile()` and `_evaluate_profile_feasibility()` (`feature_extractor.py:660-765`) **never reject `calibration_mode == "UNCALIBRATED_DEFAULT"` or `CENTER.sample_count == 0`**. Passing `DriverBehaviourProfile.create_default_uncalibrated()` activates ocular classification as if personally calibrated (`is_profile_feasible == True`).
  - **LOW-3:** Face-loss branch (`feature_extractor.py:1250`) reports `closure_threshold = self.ear_closure_threshold` (`0.20`) instead of `0.0` (uncalibrated) or the active personalized threshold.
  - **Dead state:** `_open_ear_history`, `open_ear_baseline`, `auto_adaptive_thresholds`, and `set_calibrated_baseline()` (`feature_extractor.py:229-231,1056`) are initialized/maintained but never used in `extract()`.

### 8. Signal Confidence Evaluator (`backend/app/intelligence/confidence.py`)
- **What it does:** Computes scalar `overall_confidence` (`[0, 1]`), tier (`HIGH`/`ACCEPTABLE`/`DEGRADED`/`UNAVAILABLE`), `is_reliable` (`>= 0.50`), and independent channel flags (`is_head_pose_available`, `is_ocular_available`, `is_yawn_available`).
- **Correctness & Integration:**
  - Independent channel computation is clean, **except** `DrowsinessEngine` fails to pass `ocular_viable` to `TemporalAnalyzer` (causing **CRITICAL-1**) and `DecisionEngine` never reads `yawn_viable`.
  - `_unreliable_since_ms` and `unreliable_grace_period_ms` (`confidence.py:80-84,206`) are dead state (grace timing is handled in `DecisionEngine`).
  - Does not distinguish measurement confidence from physiological certainty (`presence_confidence` is always `1.0`).

### 9. Temporal Analyzer (`backend/app/intelligence/temporal_analyzer.py`)
- **What it does:** Maintains 30 s / 60 s rolling blink & yawn histories (deduplicated by `end_ts_ms`), 2.5 s pitch history for nodding microsleep, continuous closure duration, off-road dwell, downward posture dwell (`rel_pitch <= -12°`, `|rel_yaw| < 14°`), and face-loss duration.
- **Correctness & Integration:**
  - Monotonic timestamp guard, event deduplication, binocular-only blink filtering, and decoupling of camera sampling gaps (`dt > 250 ms`) from ocular tracking gaps (`BlinkState.UNCERTAIN_GAP`) for head-pose pitch/dwell preservation are well implemented.
  - **Defects:**
    1. `if not is_reliable:` (`temporal_analyzer.py:201-204`) kills eye-closure tracking when scalar confidence drops `< 0.50` due to unrelated head-pose failure + mild blur (**CRITICAL-1**).
    2. `elif blink_valid and not features.blink.is_eye_closed and not ear_valid:` (`temporal_analyzer.py:222-223`) converts invalid EAR into `is_eye_closed = False` (verified open) if `blink.state != UNCERTAIN_GAP`.
    3. `active_yawn_ms` (`temporal_analyzer.py:427-429`) ignores `features.mar.is_valid` and has no `<= 8000 ms` upper cap.
    4. Dead/unconsumed metrics: `mean_rcr_60s`, `blink_count_30s`, and `severe_blink_count_60s` (which also inconsistently uses `d >= 500.0` vs. `> 500.0` for `PROLONGED`) are calculated every frame and never consumed by `DecisionEngine`.

### 10. Decision Engine & Drowsiness Engine (`backend/app/intelligence/decision_engine.py`, `backend/app/intelligence/drowsiness_engine.py`)
- **What it does:** 5-level deterministic priority FSM (`CRITICAL_WARNING > DROWSINESS_WARNING > ATTENTION > NORMAL`, plus `INSUFFICIENT_DATA`) with dual recovery timers: Timer 1 (state hysteresis cooldown) and Timer 2 (unresolved `WARNING`/`CRITICAL` alert continuity requiring verified open eyes).
- **Correctness & Integration Defects:**
  - **CRITICAL-1:** `DrowsinessEngine.evaluate()` passes `is_reliable=confidence_res.is_reliable` to `TemporalAnalyzer` instead of `ocular_viable`, creating a false-safe `NORMAL` during closed-eye microsleeps under `head_pose invalid + BLURRY`.
  - **HIGH-3 (Timer 1 vs Timer 2 Divergence):**
    - After prolonged face loss (`INSUFFICIENT_DATA`, `severity = -1`), on the very first frame of face reacquisition `candidate_sev >= 0 > -1` immediately sets `self.current_state = NORMAL` (`decision_engine.py:455-456`)—even when `alert_active == True` and eyes are closed.
    - During eye occlusion while head pose remains viable, Timer 1 downgrades `CRITICAL_WARNING` or `ATTENTION` to `NORMAL` without checking `metrics.is_eye_closed is False` or `confidence.ocular_viable` (`decision_engine.py:461-488`). For `ATTENTION` (`EXTENDED_CLOSURE`), this clears the alert and triggers a false audio recovery announcement while eyes remain occluded.
  - **MEDIUM-5:** During hysteresis hold (`current_state == CRITICAL_WARNING`, `candidate_state == NORMAL`), `primary_reason` is set to `reasons[0]` = `ALERT_NORMAL` (`decision_engine.py:546`), producing contradictory telemetry (`CRITICAL_WARNING` with reason `"Driver alert..."`).
  - **LOW-6:** `DrowsinessEngine.__init__` omits `severe_perclos_threshold_pct`; `NODDING_MICROSLEEP` hardcodes `threshold=800.0` even when Condition 2 triggers at `500 ms`; `_stable_face_start_ts_ms` (`decision_engine.py:95,244-245`) is written but never read.

### 11. Personalized Driver Calibration / PDBP (`backend/app/ai/pdbp_calibration.py`, `backend/app/ai/profile_adapter.py`, `backend/app/services/profile_store.py`)
- **What it does:** 4-zone guided calibration (`CENTER`, `LEFT_MIRROR`, `RIGHT_MIRROR`, `DASHBOARD`) supporting oblique camera mounts (`±32°` yaw, `[-18°, +14°]` pitch), bilateral 10% EAR trimming, median/scaled-MAD baselines, $SO(3)$ geodesic medoid + inlier SVD projection (`aggregate_neutral_rotation_matrix`), immutable snapshots, canonical SHA-256 fingerprinting, atomic file persistence with quarantine of corrupt files, and transactional disk rollback (`commit_and_activate_profile`).
- **Correctness & Integration Defects:**
  - **HIGH-1:** `scripts/run_camera_test.py:1024` feeds `latest_features.ear.compensated_ear` instead of `weighted_ear` into `GuidedCalibrationSession.ingest_frame()`, inflating DASHBOARD (and elevated-camera CENTER) baselines by up to +18% (reproduced: `+13.2%` at `-15°` pitch, shifting `c` from `0.1824` to `0.2065`).
  - **HIGH-5:** `scripts/run_camera_test.py:1023` hardcodes `geometric_reliability = 0.95` and ignores `latest_quality.is_viable_for_inference`; combined with hardcoded `presence_confidence = 1.0`, all three calibration quality gates are bypassed in live webcam operation. Additionally, `GuidedCalibrationSession._finalize()` (`pdbp_calibration.py:1810-1824`) still marks calibration **completed** and returns a profile with `neutral_rot_mat = None` even when `aggregate_neutral_rotation_matrix()` rejects due to `EXCESSIVE_ANGULAR_DISPERSION` or `INSUFFICIENT_INLIER_RATIO`, silently falling back to noisy scalar medians without surfacing the warning in `run_camera_test.py`.
  - **MEDIUM-4:** Pressing `'c'` to recalibrate (`scripts/run_camera_test.py:1234`) never passes `existing_profile`, bypassing recalibration's stricter `is_eye_currently_closed is False` filter. When `commit_and_activate_profile()` queues a deferred switch during an active blink, `scripts/run_camera_test.py:1042-1048` immediately labels the new profile active with the *old* thresholds and never reverts `calibrated_label` if deferred safe-latch activation subsequently fails and rolls back. Dead/broken `DriverProfileDTO.from_dict(data)` branch exists at `scripts/run_camera_test.py:785-787` (Pydantic v2 uses `model_validate`, and field is `version`, not `schema_version`).
  - **Physiological limitation:** Unsupervised initial calibration accepts `ear_value > 0.12` without an open-eye verification step; sustained squinting around `EAR = 0.175` passes feasibility (`span = 0.042 >= 0.040`) and can be learned as the driver's open baseline.

### 12. Alert System & Outputs (`backend/app/alerts/*`)
- **What it does:** Translates `DrowsinessAssessment` into 4 user-facing states (`NORMAL`, `ATTENTION`, `WARNING`, `CRITICAL`) via `AlertStateManager`, dispatches priority-ordered tone+voice WAVs asynchronously via `AlertDispatcher` → `AudioWarningEngine`, persists toggle state via `AudioSettingsStore`, and logs JSONL events via `AlertEventLogger`.
- **Correctness & Integration:**
  - Audio muting cleanly silences playback without disabling detection; CRITICAL preempts lower playback and is protected from lower-priority interruption; shutdown joins worker thread cleanly.
  - **HIGH-3:** `alerts/alert_state_manager.py:153` infers `self._is_recovering = (candidate_alert == AlertState.NORMAL)` while `alert_active == True`, falsely setting `is_recovering = True` (and suppressing `AlertDispatcher`'s periodic CRITICAL reminders at `alerts/alert_dispatcher.py:165`) when `DecisionEngine.state` drops to `NORMAL` on face reacquisition with **eyes closed** or during eye occlusion.
  - **MEDIUM-1:** `alerts/event_logger.py:88` formats monotonic `timestamp_ms` using `time.gmtime(timestamp_ms / 1000.0)`, recording bogus **1970-01-xx** ISO timestamps on every event (reproduced: `1970-01-03T11:38:31`).
  - **MEDIUM-6:** `alerts/alert_state_manager.py:232` passes `code=DrowsinessState.NORMAL` instead of `ReasonCode.ALERT_NORMAL` to `ExplainableReason` in `force_recovery()`.
  - **LOW-4:** `alerts/alert_dispatcher.py:195` defaults unseen `_last_announced_ts_ms` to `0.0` (suppressing non-forced alerts if `timestamp_ms < cooldown`), and periodic CRITICAL reminders are not written to `AlertEventLogger`. HUD (`scripts/run_camera_test.py:524`) prints `assessment.primary_reason_text` instead of `alert_manager.active_reason`.

---

## D. Cross-Module Findings & Scenario A–L End-to-End Trace

| Scenario | Expected Behavior | Actual Source-Code Behavior | Test Coverage | Status / Defect |
|---|---|---|---|---|
| **A. Normal driving** | `NORMAL`, steady open eyes, valid pose/MAR, PERCLOS reliable after 30 s. | Works when calibrated; HUD shows `PERCLOS: 0.0%` during 30 s warmup without `(UNRELIABLE)` tag. | Covered (`test_features.py`, `test_drowsiness_engine.py`). | Minor HUD observability gap. |
| **B. Natural blink** | `OPEN→CLOSING→CLOSED→REOPENING→OPEN`; single binocular `BlinkEvent`, kinematics, PERCLOS updated. | Works cleanly if EAR crosses `r` cleanly. If EAR lingers in `[c, r)`, `REOPENING` never times out. | Covered (`test_features.py`, `test_blink_prolonged_alignment.py`). | Vulnerable to `REOPENING` hang (**HIGH-2**). |
| **C. Prolonged eye closure** | `>= 500 ms` → `PROLONGED_CLOSURE` & `ATTENTION`; `>= 1500 ms` → `CRITICAL_WARNING`. | Works on smooth synthetic EAR. **Fails if a single noisy frame bounces EAR to `c + 0.001`** (`_closure_start_ts` resets to 0 in both `BlinkKinematicsDetector` and `TemporalAnalyzer`). | Smooth synthetic only; **jitter inside `[c, r)` untested**. | **CRITICAL-2**. |
| **D. Microsleep-like event** | Long closure → `CRITICAL_WARNING`; if interrupted by gap, left-censored without fabricated blink; reacquisition tracks closure from recovery. | Left-censoring works properly. **Fails if `head_pose` fails + mild `BLURRY`** (`DrowsinessEngine` reports `NORMAL` indefinitely during 2 s+ closed eyes). | Gap tested (`test_temporal_sampling_gap.py`); **head-pose+blur combo untested**. | **CRITICAL-1**. |
| **E. Driver looks at mirror** | Oblique camera: neutral-relative `LEFT/RIGHT_MIRROR`, camera-relative yaw EAR weighting, mirror context `<= 3 s`. | `estimate_head_pose` + `ear.py` math is correct (both mirrored & unmirrored). **Breaks if PnP fails on oblique mount** (2D fallback misinterprets camera-relative `nose_ratio` as neutral-relative) or if user toggles `'m'` with saved `neutral_rot_mat`. | Covered for PnP; **untested for mirror toggle + `neutral_rot_mat` & oblique 2D fallback**. | **HIGH-6, HIGH-7**. |
| **F. Driver looks at dashboard** | `DASHBOARD` zone (`rel_pitch <= -6°`), downward dwell (`rel_pitch <= -12°`), `SUSPECTED_HEAD_SLUMP` if ocular unavailable `> 2 s`. | Works at runtime, **except DASHBOARD calibration baseline was inflated by up to +18%** via `compensated_ear` in `run_camera_test.py:1024`, causing false closures when looking down with open eyes. | Runtime covered (`test_downward_head_monitoring_phase7.py`); **calibration-to-runtime loop with `run_camera_test` untested**. | **HIGH-1**. |
| **G. Face temporarily disappears** | `< 1 s`: holds state, freezes/cancels recovery; `>= 1 s`: `INSUFFICIENT_DATA`, PERCLOS marks unobserved, no false distraction. | Works cleanly during face loss. `DriveGuardFaceLandmarker` grace uses wall-clock `time.monotonic()` rather than frame timestamp. | Covered (`test_drowsiness_engine.py`, `test_alert_continuity_phase6.py`). | Minor clock-domain split in landmarker (**LOW-5**). |
| **H. Camera sampling interruption** | `dt > 250 ms` invalidates in-flight blink/recovery, no fabricated closure/PERCLOS/dwell. | Blink, PERCLOS, TemporalAnalyzer, DecisionEngine handle `> 250 ms` properly. **`YawnDetector` uses `500 ms`**, crediting 251–500 ms freezes as open-mouth time. | Covered for ocular/pose (`test_temporal_sampling_gap.py`); **untested for `YawnDetector` 250–500 ms gap**. | **MEDIUM-2**. |
| **I. Eye occlusion while face visible** | `head_pose_viable=True`, `ocular_viable=False`, `yawn_viable=True`; no open-eye fabrication; head pose/slump still active. | Channels decouple properly. **HOWEVER**, Timer 1 in `DecisionEngine` still downgrades `ATTENTION`/`CRITICAL` to `NORMAL` while eyes are occluded (`is_eye_closed=None`), clearing `ATTENTION` with a false recovery announcement and setting `is_recovering=True` on `CRITICAL`. | Tested for pitch/dwell preservation (`test_ocular_gap_head_pose_independence.py`); **untested for Timer 1 recovery during eye occlusion**. | **HIGH-3**. |
| **J. Calibration changes during active blink** | Deferred safe-latch activation until `OPEN`/`UNCERTAIN_GAP`; rollback on failure; rejects concurrent pending switches. | Core `DriverFeatureExtractor` latch & rollback logic is strong. **`run_camera_test.py:1042-1048`** immediately announces profile "active" with old thresholds and never updates `calibrated_label` if deferred switch fails. | Core extractor tested (`test_pdbp_pipeline_integration_phase7.py`); **`run_camera_test` UI state untested**. | **MEDIUM-4**. |
| **K. Corrupted or missing calibration profile** | Corrupt file quarantined; ocular monitoring disabled cleanly (`is_ocular_active=False`); head pose & yawns continue. | `ProfileStore` quarantines corrupt JSON cleanly; `profile=None` disables ocular safely. **HOWEVER**, `DriverFeatureExtractor` accepts `DriverBehaviourProfile.create_default_uncalibrated()` (`sample_count=0`) as calibrated. | Missing/corrupt tested (`test_profile_persistence.py`, `test_pdbp_pipeline_integration_phase7.py`); **`UNCALIBRATED_DEFAULT` untested**. | **HIGH-4**. |
| **L. Critical drowsiness followed by face loss** | `CRITICAL` preserved across grace and `INSUFFICIENT_DATA`; upon reacquisition, only clears after 2,000 ms verified open eyes. | `AlertStateManager` holds `AlertState.CRITICAL` during face loss (good). **ON REACQUISITION WITH EYES CLOSED**: `DecisionEngine.state` immediately jumps `INSUFFICIENT_DATA → NORMAL`, causing `AlertStateManager` to set **`is_recovering=True`** (`[CRITICAL \| RECOVERING]`) and **suppress periodic CRITICAL audio reminders while eyes remain closed**. | Tested for `alert_active=True` in `DecisionEngine` (`test_alert_continuity_phase6.py`), **never tested end-to-end with `AlertStateManager` + `AlertDispatcher` on closed-eye reacquisition**. | **HIGH-3**. |

---

## E. Detailed Defect Register

### CRITICAL

#### CRITICAL-1: False-Safe `NORMAL` State During Sustained Eye Closure When Head Pose Fails + Mild Blur
- **Location:** `backend/app/intelligence/drowsiness_engine.py:118-121`, `backend/app/intelligence/confidence.py:128-187`, `backend/app/intelligence/temporal_analyzer.py:201-204`, `backend/app/intelligence/decision_engine.py:163-166`
- **Actual Behaviour:** When `head_pose.is_valid == False` (`score *= 0.70`) and optical quality is `QualityStatus.BLURRY` (`is_viable_for_inference == True`, `score *= 0.70`), `overall_confidence = 0.49 < 0.50` (`is_reliable = False`) while `is_ocular_available = True` (`ocular_viable = True`). `DrowsinessEngine.evaluate()` passes scalar `is_reliable=False` to `TemporalAnalyzer.update()`, which executes `active_closure_ms = 0.0; is_eye_closed_val = None`. Downstream, `DecisionEngine` checks `is_primary_viable = head_pose_viable or ocular_viable` (`True`), skips `INSUFFICIENT_DATA`, and outputs **`DrowsinessState.NORMAL`** ("Driver alert; gaze and eye metrics within nominal ranges"). The same occurs under `OCCLUDED_OR_MARGINAL` + `POOR_LIGHTING` (`0.50 * 0.75 = 0.375`).
- **Why It Is Critical:** Direct safety-critical false-safe pathway: a driver experiencing a genuine microsleep while head pose fails under mild blur receives **neither** a drowsiness warning **nor** an `INSUFFICIENT_DATA` sensor fault warning—the system actively declares them `NORMAL`.
- **Reproducible Evidence (Verified in `.venv311`):**
  > `After 2000ms closed eyes (ear=0.10, head_pose.is_valid=False, QualityStatus.BLURRY): state=NORMAL, conf=0.49, is_reliable=False, ocular_viable=True, active_closure_ms=0.0`
- **Recommended Correction:** Pass channel-specific `is_reliable=confidence_res.ocular_viable` (or pass `ConfidenceAssessment` directly) into `TemporalAnalyzer.update()`, and align `DecisionEngine`'s sensor-viability gate with the exact channel viability used by `TemporalAnalyzer` (ensuring `DecisionEngine` never declares `NORMAL` when all active channels are suppressed).

#### CRITICAL-2: Zero-Hysteresis Microsleep Reset on Single-Frame EAR Jitter Above `closure_threshold`
- **Location:** `backend/app/features/blink_detector.py:536-537,653-661`, `backend/app/features/ear.py:364-376`, `backend/app/intelligence/temporal_analyzer.py:240-251`
- **Actual Behaviour:** While in `CLOSED` or `PROLONGED_CLOSURE`, `BlinkKinematicsDetector` immediately transitions to `REOPENING` (`is_eye_closed = False`) as soon as `ear_val > closure_threshold` (ignoring `reopen_threshold`). If the next frame drops back below `closure_threshold`, lines 658–661 **overwrite `self._closure_start_ts`** with the new crossing timestamp and **reset `self._state = BlinkState.CLOSED`**. Simultaneously, `ear.is_closed` has no hysteresis, so `TemporalAnalyzer` resets `self._closure_start_ts_ms = None`.
- **Why It Is Critical:** Under real webcam landmark noise (easily `±0.01–0.02` EAR), a sleeping driver whose closed EAR hovers near `closure_threshold` will suffer single-frame bounces into `(c, r)` (e.g., `0.201 < 0.24`), repeatedly wiping out `active_closure_duration_ms` back to `~33 ms` and preventing `CRITICAL_WARNING` (`>= 1500 ms`).
- **Reproducible Evidence (Verified in `.venv311`):**
  > `At t=1400ms: state=PROLONGED_CLOSURE, dur=1337.5ms` → `At t=1500ms (ear=0.201 < reopen=0.24): state=REOPENING, is_eye_closed=False` → `At t=1533ms (ear=0.14): state=CLOSED, dur=32.5ms`
- **Recommended Correction:** Preserve `self._closure_start_ts` (and `PROLONGED_CLOSURE` eligibility) whenever re-entering `CLOSED` from `REOPENING` before `reopen_threshold` is reached, and apply true Schmitt-trigger hysteresis so sub-`reopen_threshold` jitter cannot reset continuous closure accumulation.

---

### HIGH

#### HIGH-1: Pitch-Compensated EAR Contaminates Personalized Calibration Baselines
- **Location:** `scripts/run_camera_test.py:1024`
- **Actual Behaviour:** `calibration_session.ingest_frame()` receives `ear_value=latest_features.ear.compensated_ear` instead of `latest_features.ear.weighted_ear`.
- **Why It Is a Problem:** During DASHBOARD calibration (`rel_pitch_deg ≈ -10° to -15°`) and fresh CENTER calibration on elevated camera mounts (`pitch_deg < -4°`), `compensated_ear` is boosted by up to **+18%**, inflating the stored `ear_median` and raising `dynamic_closure_threshold`, while runtime closure detection evaluates **uncompensated** `left_ear`/`right_ear`/`weighted_ear` against that inflated threshold—causing false closures during dashboard glances and elevated-mount driving.
- **Reproducible Evidence (Verified in `.venv311`):**
  > `DASHBOARD raw=0.2400 -> comp=0.2717 (+13.2%): c_raw=0.1824 vs c_comp=0.2065, r_raw=0.2074 vs r_comp=0.2315`
- **Recommended Correction:** Pass `ear_value=latest_features.ear.weighted_ear` into `calibration_session.ingest_frame()`.

#### HIGH-2: Unbounded `BlinkState.REOPENING` Deadlock & Fabricated Multi-Second Blink Events
- **Location:** `backend/app/features/blink_detector.py:539-662`, `backend/app/features/feature_extractor.py:795-802`
- **Actual Behaviour:** Unlike `CLOSING` (which times out after 1,000 ms), `REOPENING` has no timeout. If EAR remains in `[closure_threshold, reopen_threshold)` (e.g., drowsy drooping eyelids / ptosis), `BlinkKinematicsDetector` stays in `REOPENING` indefinitely, locking `_is_in_active_blink() == True` (deadlocking pending profile/zone switches), and upon eventually exceeding `reopen_threshold` emits a fabricated `MICROSLEEP` `BlinkEvent` spanning the entire hover duration.
- **Reproducible Evidence (Verified in `.venv311`):**
  > `After 10s hovering at 0.21: state=REOPENING` → `Upon rising to 0.28: completed_blink=MICROSLEEP, dur=10048.2ms`
- **Recommended Correction:** Add a bounded reopening timeout (or sustained droop/ptosis escalation rule) in `REOPENING` that cleanly exits to `OPEN` (or `UNCERTAIN_GAP`) without fabricating an extreme blink event.

#### HIGH-3: `DecisionEngine` Timer 1 & `AlertStateManager.is_recovering` Desynchronize on Closed-Eye Reacquisition and Eye Occlusion
- **Location:** `backend/app/intelligence/decision_engine.py:455-488`, `backend/app/alerts/alert_state_manager.py:152-153`, `backend/app/alerts/alert_dispatcher.py:165`
- **Actual Behaviour:**
  1. In Scenario L (`CRITICAL_WARNING` → prolonged face loss `INSUFFICIENT_DATA` → face reappears with **eyes closed**), `DecisionEngine.current_state` jumps immediately from `INSUFFICIENT_DATA` (`-1`) to `NORMAL` (`0`) because `0 > -1`. `AlertStateManager` then computes `_is_recovering = (candidate_alert == AlertState.NORMAL)` → **`True`**, displaying `[CRITICAL | RECOVERING]` and **suppressing `AlertDispatcher`'s periodic CRITICAL reminders** even though the driver's eyes are closed and Timer 2 never started.
  2. When eyes become occluded (`is_eye_closed = None`, `ocular_viable = False`) while head pose stays viable, Timer 1 downgrades `CRITICAL_WARNING` or `ATTENTION` to `NORMAL` without requiring verified open eyes—clearing `ATTENTION` (`EXTENDED_CLOSURE`) with a false audio recovery announcement while eyes are unobservable.
- **Reproducible Evidence (Verified in `.venv311`):**
  > `Reacquired EYES CLOSED: eng=NORMAL, alert_active=True, alert=CRITICAL, recovering=True`
- **Recommended Correction:** Expose an explicit `is_alert_recovering: bool` field from `DecisionEngine` (true **only** when `_alert_recovery_start_ts_ms is not None`), gate Timer 1 de-escalation for ocular alerts on verified open eyes (`metrics.is_eye_closed is False`), and restore `self.current_state` from `_unresolved_alert_state` rather than `NORMAL` upon exiting `INSUFFICIENT_DATA` while `alert_active` is `True`.

#### HIGH-4: `DriverFeatureExtractor` Accepts `UNCALIBRATED_DEFAULT` (`sample_count = 0`) as a Calibrated Feasible Profile
- **Location:** `backend/app/features/feature_extractor.py:660-765`
- **Actual Behaviour:** `validate_profile()` accepts any non-empty `calibration_mode` and `sample_count >= 0`, and `_evaluate_profile_feasibility()` checks `sample_count == 0` only for peripheral zones—not `CENTER`. Passing `DriverBehaviourProfile.create_default_uncalibrated()` activates ocular classification with population defaults (`is_profile_feasible = True`, `is_ocular_active = True`).
- **Reproducible Evidence (Verified in `.venv311`):**
  > `calibration_mode=UNCALIBRATED_DEFAULT, CENTER sample_count=0, is_profile_feasible=True, is_ocular_active=True, active_c=0.2166`
- **Recommended Correction:** Reject `calibration_mode == "UNCALIBRATED_DEFAULT"` and `center_zb.sample_count <= 0` in `validate_profile()` / `_evaluate_profile_feasibility()`.

#### HIGH-5: Calibration Quality Gates Bypassed in Authoritative Local Pipeline
- **Location:** `scripts/run_camera_test.py:1020-1034`, `backend/app/vision/face_landmarker.py:375`, `backend/app/ai/pdbp_calibration.py:1810-1824`
- **Actual Behaviour:** `run_camera_test.py` hardcodes `geometric_reliability = 0.95`, never checks `latest_quality.is_viable_for_inference`, and `DriveGuardFaceLandmarker` hardcodes `presence_confidence = 1.0`. Additionally, if `aggregate_neutral_rotation_matrix()` rejects noisy CENTER rotations (`EXCESSIVE_ANGULAR_DISPERSION`), `GuidedCalibrationSession._finalize()` still succeeds (`is_completed = True`) with `neutral_rot_mat = None` and silently falls back to noisy scalar medians without surfacing the warning in `run_camera_test.py`.
- **Why It Is a Problem:** Blurry, dark, or shaky calibration frames can produce an unreliable personalized baseline and claim `"CALIBRATION COMPLETE"`.
- **Recommended Correction:** Gate calibration frame ingestion on `latest_quality.is_viable_for_inference` and actual geometric reliability (reprojection error / pose quality), and fail or require retry when $SO(3)$ neutral rotation aggregation rejects during CENTER calibration.

#### HIGH-6: `head_pose.py` 2D Profile Fallback Fabricates `rot_mat = I` and Confuses Camera-Relative vs Neutral-Relative Yaw
- **Location:** `backend/app/features/head_pose.py:563-618`
- **Actual Behaviour:** When PnP fails, the 2D `nose_ratio` fallback estimates a camera-relative turn (`±28°`), labels it `approx_rel_yaw`, adds `neutral_yaw_deg` to produce `yaw_deg`, mismatches `neutral_pitch_deg` inside `classify_pose_zone(pitch_deg=0.0, ...)` vs `rel_pitch_deg=0.0`, and returns `is_valid=True` with a fabricated identity matrix `rot_mat = np.eye(3)`.
- **Why It Is a Problem:** On oblique mounts, straight-ahead driving during a PnP failure can be misclassified as `LEFT/RIGHT_MIRROR`, and during fresh CENTER calibration `28° <= 32°` allows the fabricated `np.eye(3)` to pollute `_candidate_rot_mats`.
- **Recommended Correction:** Return `rot_mat=None` in 2D fallback (so it can never pollute $SO(3)$ calibration), compute `approx_abs_yaw` vs `neutral_yaw_deg` consistently, and pass `pitch_deg=neutral_pitch_deg` into `classify_pose_zone` (or mark `is_valid=False` when 3D PnP fails).

#### HIGH-7: Persisted `neutral_rot_mat` Breaks on Mirror Mode Toggle (`'m'` / `--no-flip-h`)
- **Location:** `backend/app/features/head_pose.py:233-236,620-657`, `backend/app/schemas/profile.py:72-75`
- **Actual Behaviour:** `neutral_rot_mat` is captured in the camera coordinate frame (where horizontal mirroring reflects yaw/roll signs), but `is_mirrored` is not stored in the profile. If a user calibrates with default `flip_h=True` on an oblique mount (`+25°`) and later toggles `'m'` or runs `--no-flip-h`, `R_neutral.T @ rot_mat` produces a ~50° yaw error (`OUT_OF_BOUNDS`).
- **Recommended Correction:** Store `calibration_is_mirrored` (or convert `neutral_rot_mat` to a canonical unmirrored driver frame via similarity transform $M R M$ where $M = \operatorname{diag}(-1, 1, 1)$ when loading/applying under opposite mirror state).

---

### MEDIUM

- **MEDIUM-1: `AlertEventLogger` Emits 1970 ISO Timestamps** — `backend/app/alerts/event_logger.py:88` calls `time.gmtime(timestamp_ms / 1000.0)` on monotonic uptime (`time.monotonic_ns()`), logging `1970-01-03T11:38:31`. Use wall-clock `datetime.now(timezone.utc).isoformat()` for `iso_time` while preserving monotonic `timestamp_ms`.
- **MEDIUM-2: `YawnDetector` Gap Threshold (`500 ms`) & Unbounded Active Yawn** — `backend/app/features/mar.py:78,365-370,491-496`, `backend/app/intelligence/decision_engine.py:376-385`: Align `max_sampling_gap_ms` to `250 ms`, cap `is_authentic_yawn` at `<= max_yawn_duration_ms`, clear active yawn flags when `is_valid=False`, gate yawn rules on `confidence.yawn_viable`, and distinguish candidate mouth opening from completed yawn in `DecisionEngine`.
- **MEDIUM-3: `BlinkKinematicsDetector` Returns `NaN` / Negative `current_closure_duration_ms` on Rejected Timestamps** — `backend/app/features/blink_detector.py:287-291,323-327` (reproduced: `NaN` and `-555.6 ms`): Compute duration from `self._prev_ts` (clamped `>= 0.0`) instead of the rejected `timestamp_ms`.
- **MEDIUM-4: Deferred Calibration UI State & Recalibration `existing_profile` Omission in `run_camera_test.py`** — `scripts/run_camera_test.py:785-787,1042-1048,1234`: Distinguish queued/pending vs active activation in the HUD banner, revert `calibrated_label` on deferred rollback, pass `existing_profile` on recalibration, and fix dead `DriverProfileDTO.from_dict()` call.
- **MEDIUM-5: Contradictory `primary_reason` (`ALERT_NORMAL`) During Hysteresis Hold** — `backend/app/intelligence/decision_engine.py:546` and `scripts/run_camera_test.py:524`: Preserve elevated/historical reason when holding an elevated state during recovery cooldown, and render `alert_manager.active_reason` on the HUD.
- **MEDIUM-6: Wrong Enum Type in `AlertStateManager.force_recovery()`** — `backend/app/alerts/alert_state_manager.py:232` sets `code=DrowsinessState.NORMAL` instead of `ReasonCode.ALERT_NORMAL`.
- **MEDIUM-7: Head-Pose Proxy Labeled as "Gaze"** — `backend/app/features/head_pose.py:57`, `backend/app/intelligence/temporal_analyzer.py:29,117`, `backend/app/intelligence/decision_engine.py:341,426`: Rename or explicitly qualify telemetry/messages as head-orientation proxy, not eye gaze.
- **MEDIUM-8: Asymmetric Closure State Discrepancy Between `BlinkKinematicsDetector` (`weighted_ear`) and `ear.is_closed` / `PERCLOS`** — `backend/app/features/blink_detector.py:361,677`: Unify or document unilateral/asymmetric transition semantics so `BlinkState.CLOSED` cannot coexist with `is_eye_closed=False` and PERCLOS counting OPEN.
- **MEDIUM-9: `CLOSING` Stall Re-Anchor Velocity Bias** — `backend/app/features/blink_detector.py:504,512`: Update `_open_ear_before` consistently when re-anchoring `_blink_start_ts` after a stall.
- **MEDIUM-10: Unbounded Memory Growth & Per-Frame Full-Resolution Allocations** — `scripts/run_camera_test.py:885-886` (`latencies_ms`, `inference_latencies_ms`), `backend/app/ai/eye_shadow_observer.py:210` (`_completed_inference_timestamps_ms`), and up to 5–7 full 1280×720 copies/frame (HUD alpha blends, `frame_bgr.copy()`, full-res float64 Laplacian) plus synchronous `SessionRecorder` disk I/O on the capture loop.

---

### LOW

- **LOW-1:** Explicitly pass `is_normalized=True` from `feature_extractor.py` to `extract_eye_aspect_ratio` and `YawnDetector.update`.
- **LOW-2:** `blink_detector.py:566` should check `self._min_ear_ts is not None` instead of `or`.
- **LOW-3:** `feature_extractor.py:1250` face-loss branch should report `closure_threshold=0.0` when uncalibrated (matching line 1151) or active threshold when calibrated.
- **LOW-4:** `alert_dispatcher.py:195` should default missing `_last_announced_ts_ms` to `None` rather than `0.0`, and log periodic CRITICAL reminders to `AlertEventLogger`.
- **LOW-5:** `DriveGuardFaceLandmarker` (`vision/face_landmarker.py:363,396`) should use monotonic frame `timestamp_ms` for face-loss grace duration rather than `time.monotonic()`.
- **LOW-6:** `DrowsinessEngine.__init__` should expose `severe_perclos_threshold_pct`; `DecisionEngine` should use the actual triggering closure threshold (`500 ms` vs `800 ms`) in `NODDING_MICROSLEEP`.
- **LOW-7:** `EyeCNNShadowWorker` diagnostic counters and `OpenCVCamera._is_connected`/`_cap` should be consistently guarded under locks; `FramePacket.frame` should not be mutated in place by `draw_landmarks()`.

---

## F. Test and Validation Assessment

### Exact Backend Test Run Results

- **Command:** `.\.venv311\Scripts\pytest.exe backend/tests`
- **Python / Pytest:** Python 3.11.9, pytest 9.2.0 (`pytest.ini` applies `-v` and ignores 4 legacy LiveKit/session tests: `test_fault_injection.py`, `test_measurement_infrastructure.py`, `test_session_isolation.py`, `test_stream_lifecycle.py`)
- **Collected:** 583 tests
- **Passed:** 583 passed (plus 11 subtests passed)
- **Failed:** 0
- **Skipped:** 0
- **Warnings:** 1 pytest summary warning (`StarletteDeprecationWarning` in `fastapi/testclient.py`) + 1 startup `PytestDeprecationWarning` (`pytest-asyncio` `asyncio_default_fixture_loop_scope` unset)
- **Exit Code:** `0`

### Critical Assessment of Test Quality & Validation Gaps

1. **Software / Synthetic Unit-Test Correctness (Strong, Narrow):** Individual module tests (`test_features.py`, `test_drowsiness_engine.py`, `test_alert_continuity_phase6.py`, `test_pdbp_pipeline_integration_phase7.py`, `test_ocular_gap_head_pose_independence.py`, `test_recovery_cooldown_alignment.py`, `test_blink_prolonged_alignment.py`, `test_alert_*_phase8.py`) provide thorough unit coverage and strong regression protection for past fixes.
2. **Missing Cross-Module Integration Paths (Why CRITICAL/HIGH Bugs Escaped):**
   - `test_drowsiness_engine.py` and `test_alert_continuity_phase6.py` construct synthetic `DriverFeatures` and `TemporalMetrics` directly, bypassing `DriverFeatureExtractor`, `SignalConfidenceEvaluator` + `FrameQualityResult` combinations, and `AlertStateManager` + `AlertDispatcher` coupling.
   - No test executes the full authoritative chain: `FrameQualityAnalyzer` → `DriverFeatureExtractor` → `DrowsinessEngine` → `AlertStateManager` → `AlertDispatcher` → `GuidedCalibrationSession` (including `run_camera_test.py`'s glue logic).
   - No test exercises single-frame EAR jitter inside `(c, r)` during prolonged closure, prolonged hovering in `REOPENING`, `head_pose.is_valid=False + QualityStatus.BLURRY` during eye closure, closed-eye reacquisition through `AlertStateManager`, or `create_default_uncalibrated()` passed to `DriverFeatureExtractor`.
3. **Separation of Evidence Levels:**
   - **Software correctness:** Partially verified; undermined by CRITICAL-1, CRITICAL-2, and HIGH-1..7.
   - **Synthetic-test correctness:** High (583/583 passing), but synthetic landmarks are noiseless and do not model MediaPipe jitter, glasses glare, partial eyelid droop, or real oblique-mount PnP degradation.
   - **Real-camera performance:** Not proven by unit tests. Full-frame 1280×720 float64 Laplacian + MediaPipe CPU inference + 5–7 full-frame HUD/copy allocations + synchronous `SessionRecorder` disk writes on the capture loop can introduce CPU contention and frame drops on laptop CPUs, especially with `--cnn-shadow` enabled.
   - **Real-world physiological accuracy:** **Unvalidated.** Fixed MAR (`0.55`/`0.45`), provisional blink floors (`40 ms`, `0.04` amplitude), EAR-to-P80 proxy assumptions, downward-pitch compensation factor (`+18%`), and unsupervised open-eye calibration require real human-subject validation across diverse eye morphologies, eyewear, lighting (day/night/IR), oblique cabin mounts, and genuine fatigue datasets.

---

## G. Architecture, Technical Debt & Legacy / Experimental Classification

### Recommended Classification

| Status | Modules / Files | Notes |
|---|---|---|
| **Authoritative Core** | `backend/app/camera/{base,camera_config,camera_manager,frame_buffer,frame_quality,opencv_camera,phone_camera,positioning_guide,session_recorder}.py`, `backend/app/vision/face_landmarker.py`, `backend/app/features/{ear,mar,perclos,blink_detector,head_pose,feature_extractor}.py`, `backend/app/intelligence/{confidence,temporal_analyzer,decision_engine,drowsiness_engine,reason_codes}.py`, `backend/app/alerts/*`, `backend/app/ai/{pdbp_calibration,profile_adapter}.py`, `backend/app/schemas/profile.py`, `backend/app/services/profile_store.py` | Core local intelligence pipeline. `scripts/run_camera_test.py` currently acts as the sole orchestrator and should eventually delegate orchestration to a reusable backend runner/service. |
| **Experimental (Keep Explicitly Labelled)** | `backend/app/ai/eye_shadow_observer.py`, `scripts/benchmarks/*`, `scripts/research/*`, `scripts/verify_mobilenetv3_inference.py` | Passive diagnostic MobileNetV3 eye CNN observer; cleanly decoupled from alerts today, should stay strictly non-authoritative until real-world validation completes. |
| **Legacy / Candidates for Eventual Retirement** | `backend/app/ai/{pipeline,landmarker,geometry,blink_kinematics,cascade_classifier,gaze_attention,risk_engine}.py`, `backend/app/api/ws_telemetry.py`, `backend/app/services/{session_manager,livekit_worker}.py`, `backend/app/api/routes/{livekit_token,sessions,calibration}.py` (server-side raw-video path) | Duplicates every stage of the authoritative pipeline with incompatible math/thresholds (`ai/geometry.py` vs `features/head_pose.py` & `ear.py`; `ai/blink_kinematics.py` vs `features/blink_detector.py` & `perclos.py`; `ai/risk_engine.py` vs `intelligence/decision_engine.py`). Contradicts the local-first processed-telemetry architecture. |

### Dead / Unused Configuration & Telemetry Fields
- `backend/app/core/config.py`: `DEFAULT_EXPERIMENT_MODE`, `FIXED_EAR_CUTOFF`, `DEFAULT_ZONE_CALIBRATION_DURATION_SEC` (legacy-only).
- `backend/app/vision/face_landmarker.py`: `POSE_ANCHORS` (unused, also uses chin `199` vs `head_pose.py`'s `152`).
- `backend/app/features/feature_extractor.py`: `_open_ear_history`, `open_ear_baseline`, `auto_adaptive_thresholds`, `set_calibrated_baseline()`.
- `backend/app/intelligence/confidence.py`: `_unreliable_since_ms`, `unreliable_grace_period_ms`; `ConfidenceAssessment.yawn_viable` (never read by `DecisionEngine`).
- `backend/app/intelligence/temporal_analyzer.py`: `mean_rcr_60s`, `blink_count_30s`, `severe_blink_count_60s` (computed, never read by `DecisionEngine`).
- `backend/app/intelligence/decision_engine.py`: `_stable_face_start_ts_ms` (written, never read).

---

## H. Recommended Remediation Roadmap

1. **Priority 1 — Safety-Critical Correctness Defects:**
   - **Fix CRITICAL-1** (`drowsiness_engine.py`, `temporal_analyzer.py`, `decision_engine.py`): Pass channel-specific `confidence_res.ocular_viable` (not scalar `is_reliable`) to `TemporalAnalyzer`, ensuring unrelated head-pose/blur combinations can never blind eye-closure monitoring while reporting `NORMAL`.
   - **Fix CRITICAL-2** (`blink_detector.py`, `temporal_analyzer.py`, `ear.py`): Preserve `_closure_start_ts` across `REOPENING → CLOSED` dips below `reopen_threshold` and apply hysteresis so single-frame EAR jitter inside `(c, r)` cannot reset sustained microsleep duration.
   - **Fix HIGH-3** (`decision_engine.py`, `alert_state_manager.py`): Expose explicit `is_alert_recovering` from Timer 2, gate Timer 1 ocular recovery on verified open eyes, and preserve `unresolved_alert_state` on `INSUFFICIENT_DATA` exit so closed-eye reacquisition never reports `NORMAL` or `is_recovering=True`.

2. **Priority 2 — Core Mathematical & Temporal Correctness:**
   - **Fix HIGH-2** (`blink_detector.py`): Add bounded timeout / droop handling for `BlinkState.REOPENING`.
   - **Fix HIGH-6 & HIGH-7** (`head_pose.py`, `schemas/profile.py`): Remove fabricated `rot_mat = np.eye(3)` and neutral-yaw confusion from 2D fallback; record/transform mirror orientation for `neutral_rot_mat`.
   - **Fix MEDIUM-2, MEDIUM-3, MEDIUM-8, MEDIUM-9** (`mar.py`, `blink_detector.py`, `decision_engine.py`): Align `YawnDetector.max_sampling_gap_ms` to `250 ms`, bound active yawn duration, check `confidence.yawn_viable`, guard rejected timestamps in `BlinkKinematicsDetector`, and fix `CLOSING` re-anchor velocity baseline.

3. **Priority 3 — Calibration & Profile Reliability:**
   - **Fix HIGH-1, HIGH-4, HIGH-5, MEDIUM-4** (`run_camera_test.py`, `feature_extractor.py`, `pdbp_calibration.py`): Feed uncompensated `weighted_ear` to calibration; reject `UNCALIBRATED_DEFAULT` and `CENTER.sample_count == 0` in `validate_profile`; wire real optical/geometric quality gates into calibration ingestion; reject or warn prominently when $SO(3)$ neutral rotation aggregation fails; handle deferred activation accurately in HUD and pass `existing_profile` on recalibration.

4. **Priority 4 — Cross-Module Observability & Alert Hygiene:**
   - **Fix MEDIUM-1, MEDIUM-5, MEDIUM-6, MEDIUM-7, LOW-4** (`event_logger.py`, `decision_engine.py`, `alert_state_manager.py`, `alert_dispatcher.py`, `run_camera_test.py`): Fix 1970 `iso_time` using wall-clock UTC; fix `force_recovery()` `ReasonCode`; preserve elevated `primary_reason` during cooldown; display PERCLOS reliability on HUD; qualify head-pose proxy terminology accurately.

5. **Priority 5 — Missing End-to-End Regression Tests:**
   - Add a dedicated end-to-end integration test suite exercising `FrameQualityAnalyzer → DriverFeatureExtractor → DrowsinessEngine → AlertStateManager → AlertDispatcher` and `GuidedCalibrationSession → commit_and_activate_profile → DriverFeatureExtractor` covering all newly identified defects (`CRITICAL-1/2`, `HIGH-1..7`, Scenarios C, D, E, F, H, I, J, K, L).

6. **Priority 6 — Performance, Maintainability & Architecture Consolidation:**
   - Eliminate redundant full-frame copies (ROI alpha-blending for HUD, downsampled Laplacian or grayscale reuse, crop before BGR→RGB in `EyeCNNShadowWorker`), bound latency/timestamp lists with `deque(maxlen=...)`, move `SessionRecorder` writes off the hot loop, extract the authoritative local pipeline out of `scripts/run_camera_test.py` into a clean backend service exposing processed telemetry (rather than raw video), and formally deprecate/isolate legacy `SessionAIEngine` modules.

---

## I. Overall Engineering Verdict

### Verdict: **FAIL**

**Justification:** Although DriveGuard's modular Phase 5–8 architecture is well conceived and all 583 default-collected unit tests pass, this audit reproduced **two safety-critical defects** (**CRITICAL-1**: silent false-safe `NORMAL` during a 2,000+ ms eye closure under `head_pose invalid + BLURRY`; **CRITICAL-2**: single-frame EAR jitter inside the hysteresis band wiping out accumulated microsleep closure duration) alongside **seven HIGH defects** spanning calibration contamination (`compensated_ear`), `REOPENING` state deadlock, closed-eye reacquisition falsely suppressing CRITICAL audio reminders (`is_recovering=True`), uncalibrated defaults accepted as calibrated profiles, bypassed calibration quality gates, and oblique/mirror head-pose failures. These defects directly undermine core safety guarantees and must be resolved and regression-tested before the core engine can be certified as sound or advanced to hardware integration.
