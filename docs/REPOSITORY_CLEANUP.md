# DriveGuard — Repository Cleanup & Consolidation Log

**Date:** 2026-11-20  
**Scope:** FastAPI architecture consolidation onto the single authoritative `LocalMonitoringService` / `DrowsinessEngine` pipeline, dead-file removal, and dependency hygiene.  
**Preserved Audit History:** Detailed module-level audit findings and Phase 1–7 remediation records remain preserved in [CORE_SYSTEM_AUDIT.md](./CORE_SYSTEM_AUDIT.md) and [CORE_AUDIT_REMEDIATION.md](./CORE_AUDIT_REMEDIATION.md).

---

## 1. File Classification Summary

| Category | Scope |
|---|---|
| **KEEP** | Authoritative pipeline (`backend/app/vision/`, `backend/app/features/`, `backend/app/intelligence/`, `backend/app/alerts/`, `backend/app/camera/`, `backend/app/core/`, `backend/app/schemas/`, `backend/app/services/profile_store.py`), active frontend (`frontend/app/`, `frontend/components/`, `frontend/hooks/use-camera.ts`, `use-livekit-telemetry-stream.ts`, `use-driveguard-audio.ts`, `use-system-health.ts`, `use-mobile.ts`), active LiveKit & WebSocket transports, datasets/weights/recordings/evaluation crops (`backend/datasets/`, `backend/weights/`, `data/`), user-modified research preparation (`scripts/research/prepare_eye_evaluation_dataset.py`, `backend/tests/test_dataset_preparation_phase_b1_2.py`), research evaluation scripts & paper outputs (`backend/research/outputs/`, `backend/benchmarks/`), audit docs (`docs/CORE_SYSTEM_AUDIT.md`, `docs/CORE_AUDIT_REMEDIATION.md`). |
| **REFACTOR** | `backend/app/ai/pipeline.py` (`SessionAIEngine`), `backend/app/services/local_monitoring_service.py`, `backend/app/services/session_manager.py`, `backend/app/vision/face_landmarker.py`, `backend/app/ai/pdbp_calibration.py`, `backend/app/schemas/telemetry.py`, `backend/app/main.py`, `backend/app/api/routes/sessions.py`, `backend/app/api/ws_telemetry.py`, `backend/app/services/livekit_worker.py`, `scripts/run_camera_test.py`, and 9 legacy-mock test suites migrated to the authoritative pipeline. |
| **DELETE** | 9 demonstrably obsolete, duplicate, or leaked artifact files (enumerated below). |
| **RETAINED LEGACY** | 6 modules under `backend/app/ai/` disconnected from FastAPI runtime but retained for active research/benchmark/uncommitted-script consumers (enumerated in Section 4). |

---

## 2. Deleted Files

| Deleted Path | Reason Safe to Delete |
|---|---|
| `backend/requirements.txt` | Duplicate of root `requirements.txt` with zero references anywhere in the repository (`Dockerfile` and `render.yaml` both install from root `requirements.txt`). |
| `launchers/run_all.bat` | Byte-for-byte duplicate of root `run_all.bat`; unreferenced and omitted from `launchers/README.md`. |
| `launchers/run_backend.bat` | Byte-for-byte duplicate of root `run_backend.bat`; unreferenced and omitted from `launchers/README.md`. |
| `launchers/run_frontend.bat` | Byte-for-byte duplicate of root `run_frontend.bat`; unreferenced and omitted from `launchers/README.md`. |
| `frontend/hooks/use-ai-telemetry-stream.ts` | Unused legacy WebSocket React hook with zero imports across `frontend/`; superseded by `frontend/hooks/use-livekit-telemetry-stream.ts`. |
| `frontend/hooks/use-webcam.ts` | Unused legacy camera React hook with zero imports across `frontend/`; superseded by `frontend/hooks/use-camera.ts`. |
| `backend/research/prototype_webcam_cli.py` | Abandoned Phase 1 webcam CLI directly instantiating legacy `ai.risk_engine`/`ai.blink_kinematics`; zero code or test references; superseded by `scripts/run_camera_test.py` + `LocalMonitoringService`. |
| `backend/data/profiles/driver_cal_a_profile.json` | Untracked stray profile artifact leaked during previous runs of `test_session_isolation.py` (teardown cleanup added to prevent recurrence). |
| `frontend/tsconfig.tsbuildinfo` | Untracked stale TypeScript incremental build cache regenerated automatically by `tsc`. |

---

## 3. Refactored & Added Files

### Added
- `backend/app/services/frame_adapters.py` — Canonical `AdaptedFrame`, `FrameDecodeError`, `CameraFrameAdapter`, `EncodedBytesFrameAdapter`, and `ArrayFrameAdapter`.
- `backend/tests/test_fastapi_authoritative_integration.py` — End-to-end FastAPI + authoritative engine integration, parity, isolation, WebSocket cleanup, error handling, adapter, and AST no-duplicate-pipeline guard tests.
- `docs/ARCHITECTURE.md` — Consolidated authoritative system architecture and Mermaid diagram.
- `docs/REPOSITORY_CLEANUP.md` — This cleanup and migration record.

### Refactored (Runtime)
- `backend/app/ai/pipeline.py` — Refactored `SessionAIEngine` from a competing AI stack into a thin session/transport adapter wrapping `LocalMonitoringService`. Removed all runtime usage of `ai.landmarker`, `ai.blink_kinematics`, `ai.gaze_attention`, `ai.risk_engine`, and `ai.cascade_classifier.ConfidenceGatedEyeCascade`.
- `backend/app/services/local_monitoring_service.py` — Added headless mode (`enable_audio`, `enable_event_logging`, `auto_load_profile`, `persist_calibration_to_store`, `initial_profile`, `face_loss_grace_ms`), `process_adapted_frame()`, `set_profile()`, `skip_calibration_step()`, and removed the manual legacy `AIBlinkEvent` adapter.
- `backend/app/vision/face_landmarker.py` — Added `get_diagnostics()`, `active_implementation`, duplicate/out-of-order timestamp counters, and `ensure_face_landmarker_model()`.
- `backend/app/ai/pdbp_calibration.py` — Replaced legacy `from backend.app.ai.blink_kinematics import BlinkEvent` coupling with a structural `BlinkEventLike` `Protocol`.
- `backend/app/services/session_manager.py` — Switched model preloading from legacy `backend.app.ai.landmarker` to authoritative `backend.app.vision.face_landmarker.ensure_face_landmarker_model`.
- `backend/app/schemas/telemetry.py` — Added backward-compatible authoritative fields (`drowsiness_state`, `alert_state`, `is_recovering`, `authoritative_engine`) to `FrameTelemetryDTO`.
- `backend/app/api/routes/sessions.py` — Added `POST /api/v1/sessions/{session_id}/frame` and `GET /api/v1/sessions/{session_id}/telemetry` delegating directly to the session's authoritative monitoring service.
- `backend/app/api/ws_telemetry.py` & `backend/app/services/livekit_worker.py` — Removed unused `ExperimentMode` imports; both transports now execute the authoritative `LocalMonitoringService` through `SessionAIEngine`.
- `backend/app/main.py` — Updated lifespan startup logs, root metadata, and `/health/readiness` to report the authoritative `LocalMonitoringService` + `DrowsinessEngine` pipeline.
- `scripts/run_camera_test.py` — Routed camera packets through `CameraFrameAdapter` → `LocalMonitoringService.process_adapted_frame()`.

### Refactored (Tests)
- Migrated 9 existing test suites (`test_session_engine.py`, `test_face_loss_recovery.py`, `test_ai_integrity.py`, `test_serialized_engine_mutations.py`, `test_mediapipe_video_phase3.py`, `test_fault_injection.py`, `test_stream_lifecycle.py`, `test_ai_pipeline_experiments.py`, `test_advanced_metrics_phase6.py`) from legacy `SessionAIEngine` mock interfaces (`landmarker.process_frame`, `engine.cascade`, `engine.kinematics`) to the authoritative `FaceLandmarkResult` / `LocalMonitoringService` contract without weakening assertions.
- Added `profile_store.delete_profile("driver_cal_a")` teardown cleanup in `backend/tests/test_session_isolation.py`.

---

## 4. Retained Legacy Components & Why

None of the modules below are imported or executed by FastAPI, WebSocket, LiveKit, `SessionManager`, `SessionAIEngine`, `LocalMonitoringService`, or `scripts/run_camera_test.py` for drowsiness detection (enforced by AST guard test `test_no_duplicate_drowsiness_pipeline_imported_by_fastapi_runtime`). They are preserved solely to avoid breaking active research, benchmark, and uncommitted user code:

| Retained Legacy Module | Active Consumers Blocking Deletion |
|---|---|
| `backend/app/ai/geometry.py` | Active `eye_shadow_observer.py`, `pdbp_calibration.py`, `profile_adapter.py`, uncommitted `scripts/research/prepare_eye_evaluation_dataset.py` + `test_dataset_preparation_phase_b1_2.py`, Phase A/B1 eye CNN tests, and research evaluation scripts. |
| `backend/app/ai/landmarker.py` | Uncommitted user script `scripts/research/prepare_eye_evaluation_dataset.py`, `backend/research/benchmark_phase3_mediapipe.py`, `evaluate_experiments.py`, `evaluate_utarldd.py`, and historical `test_ai_integrity.py` / `test_mediapipe_video_phase3.py` unit tests. |
| `backend/app/ai/cascade_classifier.py` | Provides `ExperimentMode` enum metadata and `MobileNetV3EyeVerifier` / `ConfidenceGatedEyeCascade` used by `test_eye_cnn_phase_a.py`, `test_ai_pipeline_experiments.py`, `test_performance_instrumentation.py`, `backend/benchmarks/*`, and paper experiments (`evaluate_experiments.py`, `evaluate_utarldd.py`). |
| `backend/app/ai/blink_kinematics.py` | Used by historical research evaluation scripts (`evaluate_experiments.py`, `evaluate_utarldd.py`) and Phase 1/4/5/6/7 drop-kinematics tests (`test_kinematics_under_drops.py`, `test_temporal_kinematics_phase5.py`, etc.). |
| `backend/app/ai/gaze_attention.py` | Used by `risk_engine.py`, `test_advanced_metrics_phase6.py`, and `test_ai_integrity.py` (`GazeAttentionTracker` unit test). |
| `backend/app/ai/risk_engine.py` | Used by paper experiment scripts (`evaluate_experiments.py`, `evaluate_utarldd.py`) and historical Phase 1/4/5/6 risk-engine unit tests (`test_ai_pipeline.py`, `test_pdbp_calibration_phase4.py`, `test_temporal_kinematics_phase5.py`, `test_advanced_metrics_phase6.py`). |

---

## 5. Removed Dependencies (`requirements.txt`)

| Removed Package | Verification Evidence |
|---|---|
| `sqlalchemy>=2.0.0` | Zero Python imports anywhere in the repository; persistent database access lives in `frontend/` via Prisma ORM. |
| `pyserial>=3.5` | Zero Python imports anywhere in the repository; ESP32 serial hardware integration is deferred to future Phase 9. |
| `onnx>=1.16.0` | Zero Python imports in runtime or tests; ONNX model execution uses OpenCV `cv2.dnn.readNetFromONNX` and `onnxruntime` (both retained). |

All retained dependencies (`numpy`, `scipy`, `opencv-python`, `mediapipe`, `onnxruntime`, `fastapi`, `uvicorn[standard]`, `pydantic`, `pydantic-settings`, `psutil`, `httpx`, `websockets`, `pytest`, `pytest-asyncio`, `pyjwt`, `livekit`, `livekit-api`) were verified against active runtime, benchmark, or test imports and kept at their existing version specifiers.

---

## 6. Follow-Up Consolidation Completed (Frontend Sync & Technical Debt Remediation)

All four previously open architectural follow-ups have been addressed (full details in [TECHNICAL_DEBT.md](./TECHNICAL_DEBT.md) and [FRONTEND_BACKEND_CONTRACT.md](./FRONTEND_BACKEND_CONTRACT.md)):

1. **Legacy Research Module Isolation (Phase 9.2):** Relocated `geometry.py`, `landmarker.py`, `cascade_classifier.py`, `blink_kinematics.py`, `gaze_attention.py`, and `risk_engine.py` into `backend/app/ai/legacy/` with thin compatibility shims under `backend/app/ai/` for offline research scripts (`prepare_eye_evaluation_dataset.py`, `evaluate_experiments.py`, `evaluate_utarldd.py`). Extracted `ExperimentMode` to `backend/app/core/experiment_mode.py` and lazy-loaded `eye_shadow_observer`. Verified zero legacy modules are imported or loaded by production runtime startup.
2. **Unified `PoseZone` Enum (Phase 9.1):** Aliased `backend/app/schemas/telemetry.py:PoseZoneEnum` directly to `backend/app/features/head_pose.py:PoseZone`, removed `_map_pose_zone_to_dto()`, fixed `livekit_worker.py` unavailable pose to `"UNKNOWN"`, and added a `math.isfinite()` guard in `classify_pose_zone()` so invalid poses never default to `CENTER`.
3. **Frontend Duplication Removed (Stages 2–7):** Removed duplicate client-side blink detection, median/MAD/PERCLOS math, and fabricated baseline defaults from `guided-calibration-wizard.tsx`; removed client-side `<310ms` reason-code filtering from `use-livekit-telemetry-stream.ts`; consolidated duplicate profile DTO mapping into `frontend/lib/api/profile-mapper.ts`.
4. **Optional `sounddevice` Dependency (Phase 9.3):** Documented `sounddevice>=0.4.6` in `requirements-desktop-audio.txt`, added explicit `check_sounddevice_availability()` + `NullAudioBackend` headless fallback + `/health/readiness` status reporting.
