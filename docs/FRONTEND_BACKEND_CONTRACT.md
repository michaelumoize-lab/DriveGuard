# DriveGuard — Frontend & FastAPI Contract Specification

> **Source of Truth:** Derived directly from the FastAPI backend implementation under `backend/app/main.py`, `backend/app/api/routes/`, `backend/app/api/ws_telemetry.py`, `backend/app/services/livekit_worker.py`, and `backend/app/schemas/`.
>
> **Authoritative Processing Pipeline:** `SessionAIEngine` → `LocalMonitoringService` → `DrowsinessEngine` → `AlertStateManager`
>
> **Primary Browser Real-Time Transport:** LiveKit WebRTC (`GET /api/v1/livekit/token`, `POST /api/v1/livekit/stop-worker`, LiveKit VideoTrack publication, and LiveKit DataChannel telemetry/control packets).

---

## 1. Authentication & Authorization Model

All protected FastAPI endpoints require a signed HS256 JWT minted by the Next.js server (`GET /api/auth/backend-token` or injected server-side by the Next.js proxy `/api/v1/[...path]`):

- **Header (REST):** `Authorization: Bearer <jwt>`
- **Query Parameter (WebSocket):** `?token=<jwt>`
- **JWT Claims Verified (`backend/app/core/auth.py`):**
  - `sub` — Authenticated user ID (non-empty)
  - `role` — `"USER"` or `"ADMIN"`
  - `iss` — `driveguard-frontend`
  - `aud` — `driveguard-backend`
  - `exp` — Expiration Unix timestamp
- **Ownership Enforcement (`verify_resource_ownership`):**
  - Non-admin users may only access/create/modify/delete their own sessions, driver profiles, and LiveKit rooms/workers.
  - Cross-user access fails with `403 Forbidden`.
  - Missing, malformed, or expired tokens fail with `401 Unauthorized`.

---

## 2. Status & Health Endpoints

| Protocol / Method | URL | Auth | Request | Response Fields | Error Responses | Frontend Consumer | Status |
|---|---|---|---|---|---|---|---|
| `GET`, `HEAD` | `/` | Public | None | `service`, `version`, `status` (`"RUNNING"`), `authoritative_engine` (`"DrowsinessEngine"`), `orchestrator` (`"LocalMonitoringService"`), `docs_url`, `websocket_url` | `500` (global handler) | Diagnostic / manual | Diagnostic |
| `GET`, `HEAD` | `/health` | Public | None | `status` (`"HEALTHY"`), `uptime_sec`, `active_sessions`, `liveness`, `readiness`, `weights.{mobilenetv3_eye_onnx, face_landmarker_task}` | `500` | `frontend/app/api/dashboard/health/route.ts`, `frontend/hooks/use-system-health.ts` | Actively used |
| `GET`, `HEAD` | `/health/liveness` | Public | None | `status` (`"ALIVE"`), `service`, `uptime_sec` | `500` | Infrastructure / probes | Actively used |
| `GET`, `HEAD` | `/health/readiness` | Public | None | `status` (`"READY"` \| `"NOT_READY"`), `ready` (bool), `uptime_sec`, `components.{authoritative_pipeline, face_landmarker_task, eye_verifier, livekit}` | `503` when `face_landmarker.task` is missing | Infrastructure / readiness check | Available |

---

## 3. Session Management Endpoints (`/api/v1/sessions`)

Source: `backend/app/api/routes/sessions.py`

| Method | URL | Auth | Request Fields | Response Fields | Error Responses | Frontend Consumer | Status |
|---|---|---|---|---|---|---|---|
| `GET` | `/api/v1/sessions` | Bearer JWT | None | `Array<SessionSummary>` (scoped to `current_user.user_id`, or all for `ADMIN`) | `401` | `frontend/lib/api/driveguard-api.ts` | Available |
| `POST` | `/api/v1/sessions` | Bearer JWT | JSON `CreateSessionRequest`: `session_id?: string`, `driver_id?: string`, `experiment_mode?: string` (default `"EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"`), `profile?: DriverProfileDTO` | `{ session_id, driver_id, owner_id, experiment_mode, is_calibrated, calibration_mode }` | `401`, `403` (impersonation or session owned by another user), `422` (invalid DTO) | `frontend/components/monitor/unified-live-monitor.tsx` (via `driveguard-api.ts`) & proxy `/api/v1/[...path]` | Actively used |
| `GET` | `/api/v1/sessions/{session_id}` | Bearer JWT (owner/admin) | Path `session_id` | `SessionSummary`: `{ session_id, driver_id, owner_id, experiment_mode, authoritative_engine, total_frames_processed, avg_fps, is_calibrating, alert_state, is_recovering, face_landmarker_diagnostics, ... }` | `401`, `403`, `404` | `frontend/lib/api/driveguard-api.ts` | Available |
| `POST` | `/api/v1/sessions/{session_id}/experiment` | Bearer JWT (owner/admin) | JSON `{ experiment_mode: string }` | `{ session_id, experiment_mode, status: "SUCCESS" }` | `400` (invalid mode), `401`, `403`, `404` | `frontend/lib/api/driveguard-api.ts` (LiveKit uses DataChannel `SWITCH_EXPERIMENT`) | Available |
| `POST` | `/api/v1/sessions/{session_id}/frame` | Bearer JWT (owner/admin) | Raw body: JPEG/PNG/WebP bytes (`<= MAX_WS_FRAME_SIZE_BYTES`); optional query `?timestamp_ms=<float>` | `{ type: "TELEMETRY", telemetry: FrameTelemetryDTO, calibration: CalibrationPhaseDTO \| null }` | `400` (empty or undecodable image), `401`, `403`, `404`, `413` (payload too large) | Non-LiveKit single-frame HTTP clients / tests | Available (not primary browser transport) |
| `GET` | `/api/v1/sessions/{session_id}/telemetry` | Bearer JWT (owner/admin) | Path `session_id` | `{ session_id, driver_id, authoritative_engine, alert_state, is_recovering, is_calibrated, telemetry: dict \| null }` | `401`, `403`, `404` | Diagnostics / polling fallback | Available |
| `DELETE` | `/api/v1/sessions/{session_id}` | Bearer JWT (owner/admin) | Path `session_id` | Final `SessionSummary` dictionary | `401`, `403`, `404` | `frontend/lib/api/driveguard-api.ts` | Available |

---

## 4. Driver Profile Endpoints (`/api/v1/profiles`)

Source: `backend/app/api/routes/profiles.py`, schema `backend/app/schemas/profile.py`

| Method | URL | Auth | Request Fields | Response Fields | Error Responses | Frontend Consumer | Status |
|---|---|---|---|---|---|---|---|
| `GET` | `/api/v1/profiles` | Bearer JWT | None | `string[]` (driver IDs owned by user) | `401` | `frontend/lib/api/driveguard-api.ts` | Available |
| `GET` | `/api/v1/profiles/{driver_id}` | Bearer JWT (owner/admin) | Path `driver_id` | `DriverProfileDTO` | `401`, `403`, `404` | `frontend/lib/api/driveguard-api.ts` | Available |
| `POST` | `/api/v1/profiles` | Bearer JWT (owner/admin) | JSON `DriverProfileDTO` (strict `snake_case`, see below) | Saved `DriverProfileDTO` | `401`, `403`, `422` (validation failure, e.g. invalid `neutral_rot_mat` or contradictory `is_calibrated`/`calibration_mode`) | `frontend/app/api/calibration/save/route.ts`, `frontend/components/monitor/unified-live-monitor.tsx` | Actively used |
| `DELETE` | `/api/v1/profiles/{driver_id}` | Bearer JWT (owner/admin) | Path `driver_id` | `{ status: "DELETED", driver_id }` | `401`, `403`, `404` | Profile management | Available |

### Exact `DriverProfileDTO` Schema (`backend/app/schemas/profile.py`)

- `version: int` (default `1`)
- `driver_id: str`
- `calibrated_at: Optional[str]` (ISO-8601 timestamp)
- `is_calibrated: bool` (must be consistent with `calibration_mode`)
- `calibration_mode: Optional[str]` (`"4ZONE"`, `"FOUR_ZONE"`, `"THREE_ZONE"`, `"CENTER_ONLY"`, `"UNCALIBRATED_DEFAULT"`, etc.)
- `zone_baselines: Record<string, ZoneBaselineDTO>` where `ZoneBaselineDTO` has:
  - `zone`, `ear_median`, `ear_mad_scaled`, `dynamic_closure_threshold`, `pitch_median_deg`, `yaw_median_deg`, `sample_count`, `ear_left_median?`, `ear_right_median?`, `blendshape_open_median?`, `confidence_median?`
- `center_ear_median: float` (`[0.0, 1.0]`)
- `blink_duration_median_ms: float`, `blink_duration_mad_ms: float`
- `blink_rcr_median: float`, `blink_rcr_mad: float` (**strict snake_case**)
- `blink_freq_median_per_min: float`, `blink_freq_mad_per_min: float`
- `baseline_perclos_pct: float` (`[0.0, 100.0]`)
- `blink_amplitude_median: float` (`[0.0, 1.0]`)
- `temporal_blinks_sampled: int`, `temporal_window_sec: float`
- `sensitivity_level: str` (`"STANDARD"` \| `"HIGH"` \| `"RELAXED"`)
- `neutral_pitch_deg: float`, `neutral_yaw_deg: float`
- `neutral_rot_mat: Optional<number[3][3]>` (validated SO(3) orthonormal rotation matrix, `det(R) = 1.0 ± 1e-5`)
- `calibration_is_mirrored: Optional<boolean>`
- `calibration_reliability: float`

---

## 5. Calibration REST Endpoints (`/api/v1/calibration`)

Source: `backend/app/api/routes/calibration.py`, schema `backend/app/schemas/calibration.py`

> **Note:** In the LiveKit browser workflow, calibration control commands (`START_CALIBRATION`, `RECALIBRATE_ZONE`, `SKIP_CALIBRATION_STEP`, `CANCEL_CALIBRATION`) are sent over the LiveKit reliable DataChannel directly to the active room worker, and real-time `CalibrationPhaseDTO` progress arrives in both `CALIBRATION_STARTED` / `CALIBRATION_COMPLETED` / `CALIBRATION_REJECTED` packets and every `TELEMETRY` packet's `calibration` field. The REST endpoints below operate on the same `session_manager` session when using REST/WebSocket flows.

| Method | URL | Auth | Request Fields | Response Fields | Error Responses | Frontend Consumer | Status |
|---|---|---|---|---|---|---|---|
| `POST` | `/api/v1/calibration/start` | Bearer JWT (owner/admin) | JSON `{ session_id, four_zone_mode?: bool, zone_duration_sec?: float [1..30] }` | `CalibrationPhaseDTO` | `401`, `403`, `404`, `422` | REST/WS clients & tests | Available |
| `POST` | `/api/v1/calibration/cancel/{session_id}` | Bearer JWT (owner/admin) | Path `session_id` | `{ status: "CALIBRATION_CANCELLED", session_id }` | `401`, `403`, `404` | REST/WS clients & tests | Available |
| `GET` | `/api/v1/calibration/{session_id}/status` | Bearer JWT (owner/admin) | Path `session_id` | `CalibrationPhaseDTO` | `400` (not currently calibrating), `401`, `403`, `404` | REST/WS clients & tests | Available |

### Exact `CalibrationPhaseDTO` Schema (`backend/app/schemas/calibration.py`)

- `current_zone: string` (`"CENTER" | "LEFT_MIRROR" | "RIGHT_MIRROR" | "DASHBOARD" | "BLINK_CADENCE"`)
- `phase_index: number` (1-based)
- `total_phases: number`
- `phase_type: "SPATIAL_ZONE" | "BLINK_CADENCE"`
- `zone_progress_pct: number` (`0..100`)
- `total_progress_pct: number` (`0..100`)
- `stable_samples_count: number`
- `target_samples_count: number`
- `elapsed_sec: number`
- `target_duration_sec: number`
- `collected_blinks_count: number`
- `live_blink_rate_per_min: number`
- `last_blink_duration_ms: number | null`
- `live_median_blink_duration_ms: number | null`
- `live_median_blink_rcr: number | null`
- `instruction: string`
- `is_head_in_target_zone: boolean`
- `stability_feedback: string`
- `zone_quality_score: number`
- `is_completed: boolean`
- `rejection_reason: string | null`
- `quality_warnings: string[]`
- `profile: DriverProfileDTO | null`

---

## 6. LiveKit WebRTC Endpoints (`/api/v1/livekit`) — Primary Frontend Transport

Source: `backend/app/api/routes/livekit_token.py`, `backend/app/services/livekit_worker.py`

| Method | URL | Auth | Request Parameters | Response Fields | Error Responses | Frontend Consumer | Status |
|---|---|---|---|---|---|---|---|
| `GET` | `/api/v1/livekit/token` | Bearer JWT | Query: `room_name` (required, maps 1:1 to `session_id`), `driver_id?`, `participant_name?`, `experiment_mode?`, `auto_start_worker?: bool` (default `true`) | `LivekitTokenResponse`: `{ token, url, room_name, participant_identity }` | `401`, `403` (impersonating driver or room owned by another user), `500` (LiveKit credentials missing or worker startup failure) | `frontend/hooks/use-livekit-telemetry-stream.ts` (via `driveguard-api.ts` and `/api/v1/[...path]`) | **Primary active transport** |
| `POST` | `/api/v1/livekit/stop-worker` | Bearer JWT (owner/admin) | Query: `room_name` (required) | `{ status: "SUCCESS", room_name, worker_stopped: boolean }` | `401`, `403`, `422` | `frontend/hooks/use-livekit-telemetry-stream.ts` (via `driveguard-api.ts`) | **Primary active teardown** |

### Session-to-Room & Worker Lifecycle

1. Frontend optionally pre-syncs the driver's PostgreSQL profile to FastAPI (`POST /api/v1/profiles`) and/or initializes session state (`POST /api/v1/sessions` with `session_id = room_name`).
2. Frontend calls `GET /api/v1/livekit/token?room_name=<session_id>&driver_id=<userId>&auto_start_worker=true`.
   - When proxied through Next.js (`/api/v1/[...path]`), Next.js attaches the signed `Authorization: Bearer <jwt>` header and automatically hydrates the user's PostgreSQL `DriverProfile` into FastAPI `POST /api/v1/profiles` before minting the token.
   - When targeting a local backend directly (`local` mode), the frontend fetches a Bearer token from `/api/auth/backend-token` and passes `Authorization: Bearer <jwt>` on every FastAPI call (including `stop-worker`).
3. FastAPI mints a room-scoped LiveKit participant JWT (`can_publish=True, can_subscribe=True, can_publish_data=True`) and starts a `LiveKitAIWorkerSession` for `room_name`.
   - The worker joins the room as `ai_worker_<room_name>` (`can_subscribe=True, can_publish_data=True, can_publish=False`) and acquires `session_manager.get_or_create_session(session_id=room_name, driver_id=driver_id, ...)`.
4. Frontend joins the LiveKit room and publishes its browser `MediaStreamTrack` video track (`LocalVideoTrack`).
5. Worker subscribes to the video track, ingests RGB frames into `SessionAIEngine.process_frame_rgb()` → `LocalMonitoringService.process_adapted_frame()` → `DrowsinessEngine` → `AlertStateManager`, and publishes telemetry & safety events back over the LiveKit DataChannel.
6. On disconnect/stop/unmount, frontend explicitly unpublishes its camera track, disconnects from the LiveKit room, and calls authenticated `POST /api/v1/livekit/stop-worker?room_name=<room_name>`. The worker closes its `rtc.VideoStream`, waits for any in-flight inference thread to finish, disconnects from LiveKit, and calls `session_manager.terminate_session(room_name)`.

---

## 7. LiveKit DataChannel Protocol (v2)

Source: `backend/app/services/livekit_worker.py`

### 7.1 Server → Frontend Packets

| `type` | Reliability | Key Fields | Purpose |
|---|---|---|---|
| `TELEMETRY` | Unreliable (lossy UDP) | `v: 2`, `session_id`, `seq`, `frame_seq`, `event_seq`, `worker_boot_id`, `session_epoch`, `timestamp`, `server_wall_epoch_ms`, `telemetry: FrameTelemetryDTO`, `calibration: CalibrationPhaseDTO \| null`, `echo_client_ts`, `network_roundtrip_ms` | Per-frame physiological telemetry, pose zone, drowsiness state, alert state, reason codes, FPS, latency, and active calibration status |
| `SAFETY_EVENT` | Reliable (with server retry until `EVENT_ACK`) | `v: 2`, `event_id`, `session_id`, `event_seq`, `frame_seq`, `worker_boot_id`, `session_epoch`, `event_type: "STATE_TRANSITION"`, `previous_state`, `new_state`, `timestamp`, `server_wall_epoch_ms`, `metrics.{risk_score, ear, ear_raw, perclos, pose_zone, reasons, drowsiness_state?, alert_state?, is_recovering?}` | Authoritative state transition event requiring client `EVENT_ACK` |
| `STATE_SYNC` | Reliable | `v: 2`, `session_id`, `current_state`, `latest_safety_event`, `unacknowledged_events`, `server_time` | Re-syncs safety state after DataChannel reconnect |
| `PONG` | Reliable | `v: 2`, `client_ts`, `ping_id`, `server_time`, `queue_depth` | Network RTT measurement |
| `CALIBRATION_STARTED` | Reliable | `calibration: CalibrationPhaseDTO` | Confirms calibration session started |
| `CALIBRATION_STEP_SKIPPED` | Reliable | `calibration: CalibrationPhaseDTO \| null` | Confirms current calibration phase skipped |
| `CALIBRATION_CANCELLED` | Reliable | `{ type: "CALIBRATION_CANCELLED" }` | Confirms calibration cancelled |
| `CALIBRATION_COMPLETED` | Reliable | `calibration: CalibrationPhaseDTO`, `profile: DriverProfileDTO` | Confirms calibration succeeded and profile saved to backend store |
| `CALIBRATION_REJECTED` | Reliable | `calibration: CalibrationPhaseDTO`, `rejection_reason: string` | Reports calibration failed quality validation; profile NOT saved |
| `EXPERIMENT_SWITCHED` | Reliable | `mode: string` | Confirms experiment mode change |
| `PROFILE_SAVED` | Reliable | `driver_id: string` | Confirms manual profile save |
| `ERROR` | Reliable | `action: string`, `error: string` | Surfaces worker control-command errors to the client |

### 7.2 Frontend → Server Control Actions (Reliable DataChannel)

| `action` | Payload Fields | Backend Behavior |
|---|---|---|
| `PING` | `client_ts`, `ping_id` | Replies `PONG` + echoes `client_ts` in next `TELEMETRY` |
| `SYNC_STATE` | None | Replies `STATE_SYNC` with current state & unacked events |
| `EVENT_ACK` | `event_id`, `event_seq`, `client_timestamp` | Clears server retry timer for `event_id` |
| `START_CALIBRATION` | `four_zone?`, `three_zone?`, `zone_duration_sec?`, `include_blink_baseline?`, `blink_baseline_duration_sec?` | Starts authoritative `GuidedCalibrationSession`; replies `CALIBRATION_STARTED` |
| `RECALIBRATE_ZONE` | `zone` (`"CENTER" \| "LEFT_MIRROR" \| "RIGHT_MIRROR" \| "DASHBOARD" \| "BLINK_CADENCE"`), `zone_duration_sec?`, `blink_baseline_duration_sec?` | Recalibrates one spatial zone or the blink cadence baseline; replies `CALIBRATION_STARTED` |
| `SKIP_CALIBRATION_STEP` | None | Skips current optional calibration step; replies `CALIBRATION_STEP_SKIPPED` |
| `CANCEL_CALIBRATION` | None | Cancels active calibration; replies `CALIBRATION_CANCELLED` |
| `SWITCH_EXPERIMENT` | `mode` | Switches experiment mode on next frame boundary; replies `EXPERIMENT_SWITCHED` |
| `SAVE_PROFILE` | None | Exports and persists active engine profile; replies `PROFILE_SAVED` |

---

## 8. Monitoring Telemetry Schema (`FrameTelemetryDTO`)

Source: `backend/app/schemas/telemetry.py`

| Field | Type | Description |
|---|---|---|
| `session_id`, `driver_id` | `string` | Session and driver identity |
| `frame_index` | `number` | Monotonic processed frame counter |
| `frame_seq`, `event_seq` | `string` | Decimal-string monotonic sequence IDs |
| `worker_boot_id`, `session_epoch` | `string` | Worker boot/epoch markers for replay protection |
| `timestamp_ms` | `number` | Frame timestamp in ms |
| `fps` | `number` | Authoritative pipeline processing FPS |
| `face_detected`, `face_visible` | `boolean` | Face detection & visibility status |
| `grace_period_active` | `boolean` | True during short face-loss grace buffer |
| `last_seen_ms` | `number \| null` | Last timestamp face was observed |
| `tracking_mode` | `string \| null` | Active MediaPipe landmarking implementation |
| `state` | `AlertStateEnum` (`NORMAL \| WARNING \| CRITICAL \| MONITORING_UNAVAILABLE`) | Transport-compatible alert state |
| `drowsiness_state` | `DrowsinessState \| null` (`NORMAL \| ATTENTION \| DROWSINESS_WARNING \| CRITICAL_WARNING \| INSUFFICIENT_DATA`) | **Authoritative 5-level `DrowsinessEngine` state** |
| `alert_state` | `AuthoritativeAlertState \| null` (`NORMAL \| ATTENTION \| WARNING \| CRITICAL`) | **Authoritative 4-level `AlertStateManager` state** |
| `is_recovering` | `boolean` | True when alert hysteresis is tracking verified driver recovery |
| `authoritative_engine` | `string` | `"DrowsinessEngine"` |
| `monitoring_status` | `string` | `MONITORING_ACTIVE \| FACE_VISIBILITY_DEGRADED \| MONITORING_UNAVAILABLE \| MONITORING_RECOVERING` |
| `risk_score` | `number` (`0.0..1.0`) | Composite risk score |
| `pose_zone` | `PoseZoneEnum` | Spatial head-pose zone (`CENTER`, `LEFT_MIRROR`, `RIGHT_MIRROR`, `DASHBOARD`, `OFF_ROAD` / `OUT_OF_BOUNDS`, `UNKNOWN`) |
| `reason_codes` | `string[]` | Authoritative `ReasonCode` values + transport availability codes |
| `head_pose` | `{ pitch_deg, yaw_deg, roll_deg }` | 3D head pose Euler angles in degrees |
| `eye_metrics` | `EyeMetricsDTO` | `ear_raw`, `ear_left`, `ear_right`, `ear_normalized`, `ear_closure_threshold`, `is_eye_closed`, `fused_closure_prob`, `perclos_pct`, `perclos_coverage_pct`, `perclos_reliable`, `blink_duration_ms`, `blink_rcr`, `blink_rate_per_min`, `mar_yawn`, `blink_state`, `blink_classification`, `eye_closure_duration_ms`, `sampling_gap_detected`, `blink_avr`, `blink_cv_ibi`, `gaze_entropy`, `brow_furrow_au4`, `yawn_envelope_confidence` |
| `cascade` | `CascadeStatusDTO` | `experiment_mode`, `cnn_triggered`, `geom_open_prob`, `geom_reliability` (signal confidence), `cnn_open_prob`, `cnn_reliability`, `arbitration_winner`, `disagreement_flag`, `cnn_latency_ms`, `cnn_execution_mode`, `verifier_invoked`, `gate_trigger_reason`, `onnx_inference_executed`, `cnn_inference_duration_ms` |
| `comparison` | `ConventionalFixedComparisonDTO` | Static `EAR < 0.22` benchmark comparison fields |
| `engine_mode` | `"MONITORING" \| "CALIBRATING"` | Engine operational mode |
| `inference_latency_ms` | `number` | Processing latency for this frame in ms |
| `timing` | `TimingBreakdownDTO \| null` | Detailed queue/convert/preprocess/inference/publish stage latencies |

---

## 9. Secondary WebSocket Endpoint (`/ws/v1/session/{session_id}`)

Source: `backend/app/api/ws_telemetry.py`

| Protocol | URL | Auth | Inbound | Outbound | Frontend Consumer | Status |
|---|---|---|---|---|---|---|
| `WS` | `/ws/v1/session/{session_id}?token=<jwt>&driver_id=<id>&experiment_mode=<mode>` | Query JWT (`token`) | Binary JPEG/PNG frames (`<= MAX_WS_FRAME_SIZE_BYTES`) or JSON control actions (`START_CALIBRATION`, `CANCEL_CALIBRATION`, `SKIP_CALIBRATION_STEP`, `SWITCH_EXPERIMENT`, `SAVE_PROFILE`, `PING`) | `SESSION_READY`, `TELEMETRY`, `CALIBRATION_*`, `EXPERIMENT_SWITCHED`, `PROFILE_SAVED`, `PONG`, `ERROR`, `DISCONNECT`, `FATAL_ERROR` | Not used by Next.js browser UI (browser uses LiveKit) | Retained for non-LiveKit clients, edge devices, and backend tests |

---

## 10. Next.js BFF / Persistence Routes (`frontend/app/api/`)

| Method | URL | Role | Downstream |
|---|---|---|---|
| `GET` | `/api/auth/backend-token` | Mints short-lived (300s) FastAPI HS256 JWT for authenticated Next.js user (used in local-backend mode) | Local signing via `lib/backend-auth.ts` |
| `ALL` | `/api/v1/[...path]` | Authenticated proxy to FastAPI `/api/v1/*`; auto-hydrates PostgreSQL `DriverProfile` to FastAPI `POST /api/v1/profiles` on session/LiveKit requests | FastAPI `/api/v1/*` |
| `POST` | `/api/calibration/save` | Persists calibrated `DriverProfile` to Neon PostgreSQL via Prisma AND forwards snake_case `DriverProfileDTO` to FastAPI `POST /api/v1/profiles` | Prisma + FastAPI |
| `POST` | `/api/sessions/start` | Creates `DrivingSession` audit record in PostgreSQL | Prisma |
| `POST` | `/api/sessions/end` | Updates `DrivingSession` summary aggregates in PostgreSQL | Prisma |
| `POST` | `/api/alerts/log` | Logs `DrowsinessAlert` + increments session alert counters atomically | Prisma |
| `GET` | `/api/dashboard/health` | Probes Next.js, FastAPI `/health`, and PostgreSQL connectivity | FastAPI `/health` + Prisma |
