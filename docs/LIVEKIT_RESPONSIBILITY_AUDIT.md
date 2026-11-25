# DriveGuard — LiveKit Responsibility Audit

**Status**: Preserved & Active (Remote Cloud Python Mode)  
**Audit Phase**: Phase 4B — LiveKit Responsibility Audit & Cloud Profile Sync Hardening  
**Scope**: Complete code-verified inventory of every responsibility LiveKit currently fulfills across backend and frontend, plus explicit migration criteria and risks before any future deprecation or removal is ever considered.

---

## 1. Executive Summary

LiveKit is **not** required for:
- **Browser AI** (on-device MediaPipe WASM + TypeScript pipeline, IndexedDB profile store, in-tab offline monitoring — Phase 3).
- **Standalone Local Python AI** (direct OpenCV/Phone camera acquisition, loopback `/api/v1/local/*` REST/MJPEG + `/ws/local/telemetry`, local SQLite WAL persistence — Phase 4A).

LiveKit **is** currently the active transport and control plane for **Remote Cloud Python AI** (streaming a browser/mobile webcam over the public internet to a remote Render-hosted FastAPI worker and receiving real-time AI telemetry, calibration control, and reliable safety alerts back).

> **Decision N-01 Mandate**: Do not delete or weaken any LiveKit runtime path until every responsibility listed in Sections 3–5 has an explicit, tested replacement or product sign-off.

---

## 2. Code-Verified File & Route Inventory

### Backend Modules
| File | Key Symbols & Routes | Role |
|---|---|---|
| `backend/app/api/routes/livekit_token.py` | `GET /api/v1/livekit/token`, `POST /api/v1/livekit/stop-worker`, `LivekitTokenResponse` | Authenticates user via `get_current_user`, enforces driver ownership (`current_user.user_id == target_driver_id` unless admin), mints signed LiveKit `AccessToken` JWTs (`VideoGrants`: `room_join`, `can_publish`, `can_subscribe`, `can_publish_data`), and triggers `livekit_worker_manager.ensure_worker_running()` / `stop_worker()`. |
| `backend/app/services/livekit_worker.py` | `LiveKitAIWorkerSession`, `LiveKitWorkerManager`, `livekit_worker_manager`, `WorkerState`, `FrameAccountingSnapshot` | Autonomous Python WebRTC participant joining LiveKit Cloud SFU rooms, subscribing to remote video tracks via `rtc.VideoStream`, enforcing a latest-frame buffer (`maxsize=1`), running `SessionAIEngine(is_mirrored=False)` on a dedicated single-thread `ThreadPoolExecutor`, publishing lossy telemetry + reliable safety/calibration DataChannel packets, and guaranteeing bounded shutdown teardown. |
| `backend/app/core/config.py` | `LIVEKIT_URL`, `LIVEKIT_API_KEY`, `LIVEKIT_API_SECRET` | Configuration aliases for LiveKit Cloud credentials (`LIVEKIT_URL` / `NEXT_PUBLIC_LIVEKIT_URL`). |
| `backend/app/main.py` | `api_v1_router.include_router(livekit_router)`, `/health/readiness` (`components.livekit.configured`) | Mounts LiveKit REST routes and exposes configuration readiness. |

### Frontend Modules
| File | Key Symbols & Routes | Role |
|---|---|---|
| `frontend/app/api/v1/[...path]/route.ts` | `GET/POST/PUT/DELETE/PATCH /api/v1/[...path]` | BFF proxy authenticating Better Auth sessions into short-lived backend Bearer JWTs, synchronously hydrating PostgreSQL `PYTHON` calibration profiles to FastAPI (`POST /api/v1/profiles`) before forwarding `livekit/token` or `sessions`. |
| `frontend/lib/api/driveguard-api.ts` | `fetchBackendAccessToken`, `requestLiveKitToken`, `stopLiveKitWorker`, `syncDriverProfileToFastApi` | Authenticated HTTP clients targeting either Next.js BFF proxy (`cloud`) or local FastAPI (`local`). |
| `frontend/lib/livekit/livekit-session-client.ts` | `establishLiveKitSession`, `teardownLiveKitSession`, `processDataChannelMessage`, `computeBackendProcessingStatus`, `createProtocolSequenceTracker` | Framework-agnostic LiveKit session lifecycle (`Room.connect`, single non-duplicated `publishTrack`, non-destructive `unpublishTrack(..., false)` + `disconnect(false)`, authenticated `stopLiveKitWorker`) and inbound DataChannel protocol parser (`v: 2`, sequence monotonic dedupe, `EVENT_ACK` generation). |
| `frontend/hooks/use-livekit-telemetry-stream.ts` | `useLivekitTelemetryStream` | React hook managing `livekit-client` `Room`, video track publication (`maxBitrate: 650000`, `maxFramerate: 25`, `simulcast: false`), RTT `PING`/`PONG` heartbeat, stale-stream watchdog (`3500 ms` stale / `10000 ms` initial timeout), and bidirectional control actions (`startCalibration`, `recalibrateZone`, `skipCalibrationStep`, `cancelCalibration`, `switchExperiment`). |
| `frontend/components/monitor/unified-live-monitor.tsx` | Remote AI runtime mode | Consumes `useLivekitTelemetryStream` when user switches from Browser AI to Remote LiveKit AI. |
| `frontend/components/calibration/guided-calibration-wizard.tsx` | Python Local / LiveKit Cloud selector | Consumes `useLivekitTelemetryStream` when calibrating a remote Python AI profile (`engineType: "PYTHON"`). |

---

## 3. Core Responsibilities Fulfilled by LiveKit

### 3.1 NAT / Firewall Traversal & WebRTC Media Transport
- Mobile phones and laptops behind symmetric NATs, cellular carriers, or corporate firewalls cannot receive inbound connections and suffer severe head-of-line blocking when streaming raw JPEG frames over TCP WebSockets to a remote Render cloud instance.
- LiveKit Cloud SFU provides ICE/STUN/TURN negotiation, adaptive bitrate VP8/H.264 video transport (`650 kbps`, `25 fps`), and sub-100 ms media ingestion into `rtc.VideoStream` on the Python worker.
- `LiveKitAIWorkerSession` enforces a `maxsize=1` latest-frame `asyncio.Queue` with `FrameAccountingSnapshot` invariants and `normalize_frame_width` (`<= 640px`), preventing multi-second backlog drift on congested cellular links.

### 3.2 Dual-Channel Telemetry & Reliable Safety/Control Protocol (`v: 2`)
LiveKit DataChannels split traffic cleanly between **lossy/unreliable** high-frequency telemetry and **reliable/ordered** safety & control messages:

#### Worker → Client Messages
| `type` | Delivery Mode | Purpose |
|---|---|---|
| `TELEMETRY` | Unreliable / high-frequency (`15–25 Hz`) | Carries `FrameTelemetryDTO`, `frame_seq`, `event_seq`, `worker_boot_id`, `session_epoch`, `calibration` progress snapshot, `echo_client_ts`, and `network_roundtrip_ms`. Monotonically deduplicated by `BigInt` `frame_seq` on the client. |
| `SAFETY_EVENT` | **Reliable + ACK/Retry** (up to 3 retries, `0.4s * attempt` backoff, 15 s TTL, max 100 unacked) | Authoritative state transitions (`WARNING`, `CRITICAL`, `MONITORING_UNAVAILABLE`, recovery) keyed by `event_id` & `event_seq` for immediate Web Audio alerting and PostgreSQL alert logging. |
| `STATE_SYNC` | Reliable | Reconciliation snapshot (`current_state`, `latest_safety_event`, `unacknowledged_events`) sent in response to `SYNC_STATE` after reconnect. |
| `PONG` | Reliable | Echoes `client_ts`, `ping_id`, `server_time`, and `queue_depth` for client RTT & latency diagnostics. |
| `CALIBRATION_STARTED`, `CALIBRATION_STEP_SKIPPED`, `CALIBRATION_COMPLETED`, `CALIBRATION_REJECTED`, `CALIBRATION_CANCELLED` | Reliable | Authoritative 4-zone & single-zone calibration lifecycle events and final `CalibratedProfilePayload`. |
| `EXPERIMENT_SWITCHED`, `PROFILE_SAVED`, `ERROR` | Reliable | Control RPC acknowledgements and structured worker error reporting. |

#### Client → Worker Control Actions (`action`)
| `action` | Purpose |
|---|---|
| `EVENT_ACK` / `ACK_EVENT` | Confirms receipt of `SAFETY_EVENT` (`event_id`, `event_seq`), cancelling the worker's retry timer. |
| `SYNC_STATE` | Requests authoritative current state and any unacknowledged safety events on reconnect. |
| `PING` | Measures network round-trip latency and updates `_last_client_ping_ts`. |
| `START_CALIBRATION` | Starts 4-zone / 3-zone guided calibration on `SessionAIEngine`. |
| `RECALIBRATE_ZONE` | Starts single-zone recalibration (`CENTER`, `LEFT_MIRROR`, `RIGHT_MIRROR`, `DASHBOARD`). |
| `SKIP_CALIBRATION_STEP` / `CANCEL_CALIBRATION` | Advances past or aborts an active calibration session. |
| `SWITCH_EXPERIMENT` | Deprecated experiment-mode control hook (retained for legacy protocol compatibility). |
| `SAVE_PROFILE` | Persists active `SessionAIEngine` profile DTO to `ProfileStore`. |

### 3.3 Worker Lifecycle, Concurrency & Room Ownership Auth
- `LiveKitWorkerManager` guarantees atomic single-worker startup per `room_name` (`_startup_tasks` + `asyncio.Lock`), preventing duplicate workers when a client reconnects or refreshes.
- Room ownership checks (`owner_id == current_user.user_id` unless `is_admin`) block cross-tenant hijacking or unauthorized termination of another driver's worker (`403 Forbidden` via `PermissionError`).
- `LiveKitAIWorkerSession.stop()` guarantees race-safe `rtc.VideoStream.aclose()` (`asyncio.shield`), bounded wait (`2.5 s`) for any in-flight inference future, `ThreadPoolExecutor.shutdown(wait=True)`, `room.disconnect()`, and `session_manager.terminate_session()` cleanup without leaking MediaPipe C++ handles.

---

## 4. What Breaks If LiveKit Is Removed Prematurely

Removing LiveKit before a replacement exists would immediately break:
1. **Remote Cloud Python Monitoring**: Any browser or mobile device running the Vercel-hosted dashboard cannot stream webcam video to the Render-hosted Python `SessionAIEngine` with low latency.
2. **Mobile-to-Cloud Phone Testing**: Phone webcams (iOS Safari / Android Chrome) over LTE/5G cannot reach a local loopback Python server and rely on LiveKit Cloud SFU for NAT traversal.
3. **Cloud Python 4-Zone Calibration**: Remote calibration of a `PYTHON` profile on Render relies on LiveKit video upload + reliable DataChannel `CALIBRATION_*` events.
4. **Reliable Cloud Safety Event Delivery**: The `SAFETY_EVENT` / `EVENT_ACK` retry loop and `STATE_SYNC` reconnect recovery over WebRTC DataChannels would be lost.
5. **Existing Contract & Worker Tests**: `frontend/__tests__/livekit-and-contract-sync.test.ts` and backend LiveKit worker/lifecycle tests validate these contracts directly.

---

## 5. Explicit Migration Criteria (Required Before Any Future Removal)

LiveKit should **only** be considered for deprecation/removal if ALL of the following criteria are satisfied and approved:
1. **Product Decision**: Either remote cloud Python video processing is formally retired (in favor of Browser AI on mobile/web + Local Python AI on edge/desktop hardware), **OR** an alternative WebRTC/WHEP/WHIP or binary WebSocket transport with proven TURN/NAT traversal and sub-250 ms p95 latency on cellular networks is implemented and load-tested.
2. **Reliable Event Parity**: Any replacement cloud transport must preserve `event_id` idempotency, `EVENT_ACK`/retry semantics, and `STATE_SYNC` reconnect reconciliation.
3. **Auth & Tenant Isolation Parity**: Room/session ownership enforcement (`owner_id` / Better Auth JWT) and clean server-side resource teardown on disconnect must be preserved with equivalent test coverage.
4. **Zero Impact on Dual-Engine Isolation**: Browser AI (`engineType="BROWSER"`) and Local Python AI (`engineType="PYTHON"`, loopback `/api/v1/local/*`) must remain completely independent of any cloud transport changes.
