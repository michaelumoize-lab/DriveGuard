# DriveGuard — Technical Debt Remediation Record

> **Execution Order:** All phases below were executed strictly **after** Stage 8 (Frontend ↔ FastAPI synchronisation & LiveKit integration tests) passed.

---

## Phase 9.1 — Unified `PoseZone` Enum & Invalid-Pose Safety

### Problem Identified
- `backend/app/schemas/telemetry.py` defined a duplicate `PoseZoneEnum(str, Enum)` alongside `backend/app/features/head_pose.py`'s canonical `PoseZone(str, Enum)`.
- `backend/app/ai/pipeline.py` carried a redundant `_map_pose_zone_to_dto()` translator.
- `backend/app/services/livekit_worker.py` reported `"pose_zone": "OUT_OF_BOUNDS"` on monitoring-unavailable transitions instead of `"UNKNOWN"`.
- **Safety Bug Found During Remediation:** `classify_pose_zone()` in `backend/app/features/head_pose.py` did not check `math.isfinite()` on input angles — `NaN` or `±Inf` angles evaluated `False` on every threshold check and silently fell through to `PoseZone.CENTER`!

### Remediation Applied
- `backend/app/schemas/telemetry.py`: aliased `PoseZoneEnum` directly to `backend.app.features.head_pose.PoseZone` (`PoseZoneEnum is CanonicalPoseZone is LegacyGeometryPoseZone`).
- `backend/app/ai/pipeline.py`: removed `_map_pose_zone_to_dto()` and assigned `features.head_pose.zone` directly.
- `backend/app/services/livekit_worker.py`: changed `_transition_monitoring_unavailable()` `"pose_zone"` to `"UNKNOWN"`.
- `backend/app/features/head_pose.py`: added explicit `math.isfinite()` guard in `classify_pose_zone()` returning `PoseZone.UNKNOWN` on non-finite angles.
- Verified in `backend/tests/test_technical_debt_remediation.py` (`test_phase_9_1_*`).

---

## Phase 9.2 — Production vs Legacy Research Module Separation

### Problem Identified
- Production runtime modules (`pipeline.py`, `session_manager.py`, `pdbp_calibration.py`, `profile_adapter.py`, `local_monitoring_service.py`) still imported `ExperimentMode`, `PoseZone`, or `eye_shadow_observer` from legacy `backend/app/ai/*` modules at module import time.
- Six legacy research modules (`geometry.py`, `landmarker.py`, `cascade_classifier.py`, `blink_kinematics.py`, `gaze_attention.py`, `risk_engine.py`) lived directly beside production `pipeline.py` and `pdbp_calibration.py`.

### Remediation Applied
- Extracted canonical `ExperimentMode` into `backend/app/core/experiment_mode.py`.
- Repointed `pdbp_calibration.py`, `profile_adapter.py`, `local_monitoring_service.py` to import `PoseZone` directly from `backend.app.features.head_pose`.
- Repointed `pipeline.py` and `session_manager.py` to import `ExperimentMode` from `backend.app.core.experiment_mode`.
- Lazy-imported `eye_shadow_observer` inside `LocalMonitoringService.__init__()` only when `enable_shadow_worker=True`.
- Relocated the 6 legacy research implementations into `backend/app/ai/legacy/` (fixing `parents[3] / "weights"` relative paths in `cascade_classifier.py` and `landmarker.py`) and retained thin compatibility shims in `backend/app/ai/<name>.py` so offline research scripts (including `scripts/research/prepare_eye_evaluation_dataset.py`) and legacy benchmark tests continue operating untouched.
- Enforced zero production imports via both AST inspection and fresh-process `sys.modules` verification (`test_phase_9_2_*`, `LOADED_LEGACY_AT_STARTUP: []`).

---

## Phase 9.3 — Explicit Optional Desktop Audio (`sounddevice`)

### Problem Identified
- `SoundDeviceAudioBackend` attempted lazy `import sounddevice` on every playback attempt, logging repetitive warnings on headless servers without exposing explicit availability status or documented optional installation instructions.

### Remediation Applied
- Added `check_sounddevice_availability()`, `NullAudioBackend`, `AudioWarningEngine.audio_available`, `AudioWarningEngine.audio_unavailable_reason`, and `AudioWarningEngine.get_audio_status()` in `backend/app/alerts/audio_warning_engine.py`.
- Added `requirements-desktop-audio.txt` (`sounddevice>=0.4.6`) and documented it as optional in `requirements.txt`.
- Exposed `"desktop_audio": {"optional": True, "available": ..., "web_client_audio": "BROWSER_WEB_AUDIO_API"}` in `/health/readiness` (`backend/app/main.py`) while keeping headless servers `200 READY`.
- Verified in `test_phase_9_3_*`.

---

## Phase 9.4 — Pipeline Performance Investigation (~12.1 FPS Bottleneck)

### Measurement Methodology
Created reproducible stage profiler `scripts/benchmarks/profile_authoritative_pipeline.py` measuring `jpeg_decode`, `rgb_adapt`, `frame_quality`, `mediapipe_landmarker`, `feature_extraction`, `drowsiness_engine`, `alert_manager`, `dto_json_serialize`, and full `SessionAIEngine.process_frame(..., is_rgb=True)` across 40 frames (6 warmup) at 640×480 and 1280×720.

### Root Cause Found (Measured)
`FrameQualityAnalyzer.analyze()` (`backend/app/camera/frame_quality.py`) claimed `< 1.5 ms` overhead in its docstring, but actually consumed **13.97 ms (56.5%)** at 640×480 and **17.91 ms (66.2%)** at 1280×720 because it computed a full-frame `CV_64F` (64-bit double) `cv2.Laplacian` followed by 2-pass NumPy `.var()` and `np.mean()`. Combined with full MediaPipe face-mesh + blendshapes on real faces (~30–40 ms) plus local OpenCV capture/HUD rendering (~15–20 ms), total frame time reached ~82 ms (~12.1 FPS).

### Safe Optimization Applied
Replaced `np.mean(gray)` + `cv2.Laplacian(gray, cv2.CV_64F).var()` with mathematically equivalent SIMD `cv2.mean(gray)[0]` + `cv2.Laplacian(gray, cv2.CV_32F)` + `cv2.meanStdDev(lap)[1][0, 0] ** 2`. Zero change to optical thresholds, blur viability rules, or drowsiness detection behavior (all 20/20 `test_frame_quality.py` + `test_end_to_end_authoritative_pipeline.py` tests pass).

### Measured Before vs After Results (Local Windows CPU, 40 frames, 6 warmup)

| Metric | Before (640×480) | After (640×480) | Before (1280×720) | After (1280×720) |
|---|---|---|---|---|
| `frame_quality` Mean (ms) | 13.971 | **2.642** (-81.1%) | 17.914 | **7.023** (-60.8%) |
| `frame_quality` P95 (ms) | 22.371 | **3.081** | 20.071 | **8.266** |
| `mediapipe_landmarker` Mean (ms) | 9.453 | 3.231 | 7.196 | 6.418 |
| `feature_extraction` Mean (ms) | 0.132 | 0.058 | 0.073 | 0.060 |
| `drowsiness_engine` Mean (ms) | 0.145 | 0.057 | 0.078 | 0.065 |
| `alert_manager` Mean (ms) | 0.007 | 0.003 | 0.004 | 0.003 |
| `dto_json_serialize` Mean (ms) | 0.195 | 0.068 | 0.053 | 0.047 |
| **Full `SessionAIEngine` Mean (ms)** | 27.387 | **6.292** | 27.719 | **16.275** |
| **Full `SessionAIEngine` P95 (ms)** | 49.907 | **6.887** | 33.349 | **18.198** |
| **Effective Throughput (FPS)** | 36.51 FPS | **158.93 FPS** | 36.08 FPS | **61.44 FPS** |

> *Note:* Synthetic textured frames do not contain an actual human face mesh; real webcam sessions with active 478-point mesh tracking + blendshapes add MediaPipe mesh inference time (~20–35 ms on CPU), so eliminating ~11 ms/frame from `frame_quality` directly recovers ~4–8 FPS of real-world webcam headroom toward the `>= 15–20 FPS` target.

---

## Phase 9.5 — Repeatable Real-World Validation Procedure

- Created `docs/REAL_WORLD_VALIDATION.md` specifying participant diversity, lighting/glare/night matrix, camera mount geometry, 7-phase standardized trial script, and explicit quantitative targets (FPR, FNR, detection latency, pose-zone accuracy, calibration convergence) with clear non-certification scope boundaries.
- Created `scripts/validation/evaluate_recorded_sessions.py` to compute all empirical metrics deterministically from recorded JSONL telemetry + ground-truth JSON annotations; verified via `--self-test`.

---

## Remaining Non-Blocking Follow-Ups

1. **Legacy Research Compatibility Shims:** Once offline research scripts (`scripts/research/prepare_eye_evaluation_dataset.py`, `backend/research/evaluate_experiments.py`, `evaluate_utarldd.py`) migrate to `backend.app.vision.face_landmarker` and `backend.app.features`, the 6 compatibility shims under `backend/app/ai/` and `backend/app/ai/legacy/` can be retired entirely.
2. **Real LiveKit Cloud SFU Browser Media Loop:** Automated tests verify FastAPI LiveKit token minting against configured credentials and full LiveKit worker/client lifecycle with mocked SFU rooms; a real browser-to-LiveKit-Cloud-SFU-to-Python-worker webcam pass should be executed during Phase 9.5 recorded field trials.
3. **FastAPI `HEAD` Route Operation ID Warning:** `@app.api_route(..., methods=["GET", "HEAD"])` in `backend/app/main.py` emits harmless FastAPI OpenAPI duplicate-operation-id warnings; can be silenced by splitting or disabling `include_in_schema` on `HEAD` probes in a future cleanup.
