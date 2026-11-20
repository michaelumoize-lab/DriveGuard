# DriveGuard — Frontend/Backend Synchronisation & Technical Debt Tracker

This tracker records verification gates and progress across all stages of the DriveGuard frontend-backend synchronisation, LiveKit integration, and subsequent technical debt remediation.

---

## Baseline (Stage 0) — Recorded Before Changes

- [x] **Stage 0: Repository inspection and baseline**
  - Git status recorded; all existing uncommitted user changes preserved untouched.
  - Frontend type-check (`npx tsc --noEmit --incremental false`): **0 errors**.
  - Frontend unit/security tests (`test:audio`, `test:camera`, `test:security`): **46/46 passed** (19 audio + 20 camera + 7 security).
  - Backend targeted API/LiveKit/integration baseline: **46/46 passed** in 22.86s.
  - Backend full suite baseline: **684/684 passed** in 66.53s.

---

## Major Stage 1 — Frontend & FastAPI Synchronisation + LiveKit Integration

- [x] **Stage 1: Establish the actual FastAPI contract**
  - Documented all FastAPI REST, WebSocket, and LiveKit DataChannel contracts in `docs/FRONTEND_BACKEND_CONTRACT.md`.
- [x] **Stage 2: Make the frontend match FastAPI**
  - Aligned `frontend/types/telemetry.ts` with authoritative `FrameTelemetryDTO`, `DrowsinessState`, `AlertState`, `PoseZone`, `CalibrationPhaseDTO`, `DriverProfileDTO`, and all 16 authoritative `ReasonCode` values.
  - Created centralized `frontend/lib/api/driveguard-api.ts` and canonical `frontend/lib/api/profile-mapper.ts` (`buildDriverProfileDtoPayload()`).
  - Fixed snake_case / missing field mismatches (`blink_rcr_median`, `blink_rcr_mad`, `neutral_rot_mat`, `calibration_is_mirrored`) in `frontend/app/api/calibration/save/route.ts` and `frontend/app/api/v1/[...path]/route.ts`.
- [x] **Stage 3: Make LiveKit the primary real-time transport**
  - Hardened `backend/app/services/livekit_worker.py` (removed unexpected `cal_` auto-1-zone calibration race; enriched `SAFETY_EVENT` with `drowsiness_state`, `alert_state`, `is_recovering`, `mar_yawn`).
  - Refactored `frontend/hooks/use-livekit-telemetry-stream.ts` + `frontend/lib/livekit/livekit-session-client.ts` (authenticated `stop-worker` teardown, explicit track unpublish/duplicate-publish mutex, explicit `backendProcessingStatus`, `ERROR`/`PROFILE_SAVED` handling, zero client-side decision filtering).
- [x] **Stage 4: Synchronise calibration**
  - Refactored `frontend/components/calibration/guided-calibration-wizard.tsx` to use backend `GuidedCalibrationSession` exclusively (removed client-side blink detector, median/MAD math, and fabricated defaults; wired `BLINK_CADENCE` via `RECALIBRATE_ZONE`; surfaced `rejection_reason` and `quality_warnings` honestly).
- [x] **Stage 5: Synchronise monitoring dashboard**
  - Updated `unified-live-monitor.tsx`, `driver-cockpit-hud.tsx`, `live-telemetry-hud.tsx`, `explainable-risk-panel.tsx`, `readiness-checklist.tsx` to display authoritative `drowsiness_state`, `alert_state`, `is_recovering`, `mar_yawn`, `geom_reliability`, calibration requirement notice, and centralized reason explanations.
- [x] **Stage 6: User experience and error handling**
  - Eliminated false-green `NORMAL` / `STREAM ACTIVE` / `All Physiological Signals Nominal` states when `telemetry` is null, stale, or `MONITORING_UNAVAILABLE`; rendered honest `Unavailable` / `—` metrics when face is lost or uncalibrated.
- [x] **Stage 7: Remove frontend detection duplication**
  - Verified zero client-side EAR/blink/PERCLOS/drowsiness/alert decision logic remains in `frontend/`.
- [x] **Stage 8: Integration and end-to-end testing (MANDATORY GATE BEFORE STAGE 9)**
  - Frontend tests: **54/54 passed** (`test:audio` 19 + `test:camera` 20 + `test:security` 7 + `test:sync` 8) + `tsc --noEmit` (**0 errors**).
  - Backend contract, API, security, and LiveKit lifecycle tests: **51/51 passed**.
  - Live FastAPI server HTTP E2E check verified (`/health`, `/health/readiness`, sessions, frame ingest, telemetry, real LiveKit JWT minting, session termination).

---

## Major Stage 2 — Technical Debt Remediation (Strictly After Stage 8 Passes)

- [x] **Phase 9.1: Unify `PoseZone` across backend and frontend** (aliased `PoseZoneEnum` directly to `features.head_pose.PoseZone`, removed `_map_pose_zone_to_dto()`, fixed `NaN`/`Inf` finiteness guard so invalid poses always return `UNKNOWN`, fixed `livekit_worker` unavailable pose to `UNKNOWN`)
- [x] **Phase 9.2: Separate legacy research modules from production runtime** (extracted `backend/app/core/experiment_mode.py`, relocated 6 legacy research modules to `backend/app/ai/legacy/` with compatibility shims, lazy-imported `eye_shadow_observer`, verified `LOADED_LEGACY_AT_STARTUP: []`)
- [x] **Phase 9.3: Make desktop audio explicitly optional with clear headless fallback** (`requirements-desktop-audio.txt`, `NullAudioBackend`, `check_sounddevice_availability()`, `get_audio_status()`, `/health/readiness` reporting)
- [x] **Phase 9.4: Investigate and profile ~12.1 FPS pipeline performance** (`scripts/benchmarks/profile_authoritative_pipeline.py`; identified `FrameQualityAnalyzer` `CV_64F` Laplacian as 56–66% bottleneck and optimized to equivalent SIMD `CV_32F` + `cv2.meanStdDev`: 13.97ms → 2.64ms at 640×480, 17.91ms → 7.02ms at 1280×720)
- [x] **Phase 9.5: Establish repeatable real-world validation procedure** (`docs/REAL_WORLD_VALIDATION.md` + `scripts/validation/evaluate_recorded_sessions.py` with passing `--self-test`)

---

## Stage 10 & Stage 11 — Documentation & Final Verification

- [x] **Stage 10: Documentation updates** (`FRONTEND_BACKEND_CONTRACT.md`, `ARCHITECTURE.md`, `REPOSITORY_CLEANUP.md`, `TECHNICAL_DEBT.md`, `REAL_WORLD_VALIDATION.md`, `SYNC_AND_DEBT_TRACKER.md`)
- [x] **Stage 11: Final verification & 14-point report**
  - Backend full pytest: **695/695 passed** (baseline 684/684).
  - Python `compileall`: exit 0.
  - Frontend tests (`test:audio`, `test:camera`, `test:security`, `test:sync`): **54/54 passed** (baseline 46/46).
  - Frontend `npx tsc --noEmit --incremental false`: **0 errors**.
  - Frontend production build (`npm run build`): **Compiled successfully** (18/18 pages).
  - Live FastAPI HTTP E2E check & Phase 9.5 evaluator `--self-test`: **passed**.
