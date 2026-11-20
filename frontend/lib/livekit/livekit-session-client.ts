/**
 * Pure, framework-agnostic LiveKit session lifecycle & DataChannel protocol helpers.
 *
 * Used by `useLivekitTelemetryStream` in the browser and directly exercised by
 * automated Node integration tests with mocked LiveKit Room + FastAPI fetch.
 */

import {
  BackendTargetMode,
  fetchBackendAccessToken,
  requestLiveKitToken,
  stopLiveKitWorker,
} from "@/lib/api/driveguard-api";
import type {
  AlertState,
  BackendProcessingStatus,
  CalibrationPhase,
  EventAck,
  ExperimentMode,
  FrameTelemetry,
  LiveKitTokenResponse,
  SafetyEvent,
} from "@/types/telemetry";

export interface LiveKitTrackLike {
  id?: string;
  label?: string;
  readyState?: string;
}

export interface LiveKitPublishedTrackLike {
  track?: unknown;
}

export interface LiveKitLocalParticipantLike {
  publishTrack: (track: any, options?: Record<string, unknown>) => Promise<LiveKitPublishedTrackLike>;
  unpublishTrack?: (track: any, stopOnUnpublish?: boolean) => Promise<unknown>;
  publishData: (payload: Uint8Array, options?: { reliable?: boolean }) => Promise<void>;
}

export interface LiveKitRoomLike {
  state: string;
  localParticipant: LiveKitLocalParticipantLike;
  connect: (url: string, token: string) => Promise<void>;
  disconnect: (stopTracks?: boolean) => Promise<void>;
  on: (event: string, listener: (...args: any[]) => void) => unknown;
}

export interface DataChannelPacketResult {
  kind:
    | "TELEMETRY"
    | "SAFETY_EVENT"
    | "STATE_SYNC"
    | "PONG"
    | "CALIBRATION_EVENT"
    | "CALIBRATION_CANCELLED"
    | "EXPERIMENT_SWITCHED"
    | "PROFILE_SAVED"
    | "ERROR"
    | "IGNORED";
  telemetry?: FrameTelemetry;
  calibration?: CalibrationPhase | null;
  safetyEvent?: SafetyEvent;
  ackToSend?: EventAck;
  stateSync?: {
    currentState?: AlertState;
    latestSafetyEvent?: SafetyEvent;
    sessionEpoch?: string;
  };
  pongClientTs?: number;
  experimentMode?: ExperimentMode;
  driverId?: string;
  errorMessage?: string;
  action?: string;
  calibrationEventType?: string;
}

export interface ProtocolSequenceTracker {
  lastTelemetrySeq: bigint;
  lastSafetyEventSeq: bigint;
  lastSessionEpoch: string | null;
  processedEventIds: Set<string>;
}

export function createProtocolSequenceTracker(): ProtocolSequenceTracker {
  return {
    lastTelemetrySeq: BigInt(-1),
    lastSafetyEventSeq: BigInt(-1),
    lastSessionEpoch: null,
    processedEventIds: new Set<string>(),
  };
}

export function resetProtocolSequenceTracker(tracker: ProtocolSequenceTracker): void {
  tracker.lastTelemetrySeq = BigInt(-1);
  tracker.lastSafetyEventSeq = BigInt(-1);
  tracker.lastSessionEpoch = null;
  tracker.processedEventIds.clear();
}

function parseBigIntSeq(raw: unknown): bigint {
  try {
    return BigInt(String(raw ?? 0));
  } catch {
    return BigInt(0);
  }
}

/**
 * Parse and validate an inbound LiveKit DataChannel message from `livekit_worker.py`.
 * Performs zero client-side drowsiness decision filtering — backend outputs are authoritative.
 */
export function processDataChannelMessage(
  msg: Record<string, any>,
  tracker: ProtocolSequenceTracker,
  options: { latestRttMs?: number; nowMs?: number } = {}
): DataChannelPacketResult {
  if (!msg || typeof msg !== "object" || typeof msg.type !== "string") {
    return { kind: "IGNORED" };
  }

  if (msg.type === "TELEMETRY" && msg.telemetry) {
    const rawSeq = msg.frame_seq ?? msg.seq ?? msg.telemetry.frame_index ?? 0;
    const currentSeq = parseBigIntSeq(rawSeq);
    if (currentSeq <= tracker.lastTelemetrySeq) {
      return { kind: "IGNORED" };
    }
    tracker.lastTelemetrySeq = currentSeq;

    const tel = msg.telemetry as FrameTelemetry;
    const measuredRtt =
      options.latestRttMs !== undefined && options.latestRttMs > 0
        ? options.latestRttMs
        : undefined;

    // Preserve backend reason_codes verbatim (no client-side threshold/blink filtering)
    const enriched: FrameTelemetry = {
      ...tel,
      reason_codes: Array.isArray(tel.reason_codes) ? [...tel.reason_codes] : [],
      network_roundtrip_ms: measuredRtt,
    };

    return {
      kind: "TELEMETRY",
      telemetry: enriched,
      calibration: (msg.calibration as CalibrationPhase | null | undefined) ?? null,
    };
  }

  if (msg.type === "SAFETY_EVENT") {
    const safetyEvent = msg as SafetyEvent;
    const currentEventSeq = parseBigIntSeq(safetyEvent.event_seq);

    // Always ACK so server clears retry timer even for duplicate/reordered deliveries
    const ackToSend: EventAck = {
      action: "EVENT_ACK",
      event_id: safetyEvent.event_id,
      event_seq: String(safetyEvent.event_seq),
      client_timestamp: options.nowMs ?? Date.now(),
    };

    if (safetyEvent.session_epoch && safetyEvent.session_epoch !== tracker.lastSessionEpoch) {
      tracker.lastSessionEpoch = safetyEvent.session_epoch;
      tracker.lastSafetyEventSeq = BigInt(-1);
      tracker.processedEventIds.clear();
    }

    if (currentEventSeq <= tracker.lastSafetyEventSeq) {
      return { kind: "IGNORED", ackToSend };
    }

    if (tracker.processedEventIds.has(safetyEvent.event_id)) {
      return { kind: "IGNORED", ackToSend };
    }

    tracker.processedEventIds.add(safetyEvent.event_id);
    tracker.lastSafetyEventSeq = currentEventSeq;
    if (tracker.processedEventIds.size > 200) {
      const oldest = tracker.processedEventIds.values().next().value;
      if (oldest) tracker.processedEventIds.delete(oldest);
    }

    return {
      kind: "SAFETY_EVENT",
      safetyEvent,
      ackToSend,
    };
  }

  if (msg.type === "STATE_SYNC") {
    if (msg.session_epoch && msg.session_epoch !== tracker.lastSessionEpoch) {
      tracker.lastSessionEpoch = msg.session_epoch;
      tracker.lastSafetyEventSeq = BigInt(-1);
      tracker.processedEventIds.clear();
    }
    const latestSafetyEvent = msg.latest_safety_event as SafetyEvent | undefined;
    if (latestSafetyEvent) {
      const syncSeq = parseBigIntSeq(latestSafetyEvent.event_seq);
      if (syncSeq > tracker.lastSafetyEventSeq) {
        tracker.lastSafetyEventSeq = syncSeq;
      }
    }
    return {
      kind: "STATE_SYNC",
      stateSync: {
        currentState: msg.current_state as AlertState | undefined,
        latestSafetyEvent,
        sessionEpoch: msg.session_epoch as string | undefined,
      },
    };
  }

  if (msg.type === "PONG") {
    if (msg.client_ts !== undefined && msg.client_ts !== null) {
      return { kind: "PONG", pongClientTs: Number(msg.client_ts) };
    }
    return { kind: "IGNORED" };
  }

  if (
    msg.type === "CALIBRATION_STARTED" ||
    msg.type === "CALIBRATION_STEP_SKIPPED" ||
    msg.type === "CALIBRATION_COMPLETED" ||
    msg.type === "CALIBRATION_REJECTED"
  ) {
    const cal = msg.calibration ? ({ ...msg.calibration } as CalibrationPhase) : null;
    if (cal) {
      if (msg.profile) {
        cal.profile = msg.profile;
      }
      if (msg.rejection_reason && !cal.rejection_reason) {
        cal.rejection_reason = msg.rejection_reason;
      }
    }
    return {
      kind: "CALIBRATION_EVENT",
      calibration: cal,
      calibrationEventType: msg.type,
    };
  }

  if (msg.type === "CALIBRATION_CANCELLED") {
    return { kind: "CALIBRATION_CANCELLED" };
  }

  if (msg.type === "EXPERIMENT_SWITCHED" && msg.mode) {
    return {
      kind: "EXPERIMENT_SWITCHED",
      experimentMode: msg.mode as ExperimentMode,
    };
  }

  if (msg.type === "PROFILE_SAVED") {
    return {
      kind: "PROFILE_SAVED",
      driverId: typeof msg.driver_id === "string" ? msg.driver_id : undefined,
    };
  }

  if (msg.type === "ERROR") {
    return {
      kind: "ERROR",
      action: typeof msg.action === "string" ? msg.action : "UNKNOWN",
      errorMessage: String(msg.error || msg.message || "Backend worker control error"),
    };
  }

  return { kind: "IGNORED" };
}

/**
 * Determine honest backend processing status from connection state and telemetry timestamps.
 */
export function computeBackendProcessingStatus(params: {
  connectionStatus: "disconnected" | "connecting" | "connected" | "reconnecting" | "error";
  connectedAtMs: number | null;
  lastTelemetryAtMs: number | null;
  nowMs: number;
  staleAfterMs?: number;
  initialFrameTimeoutMs?: number;
}): { status: BackendProcessingStatus; isStale: boolean } {
  const {
    connectionStatus,
    connectedAtMs,
    lastTelemetryAtMs,
    nowMs,
    staleAfterMs = 3500,
    initialFrameTimeoutMs = 10000,
  } = params;

  if (connectionStatus === "error") {
    return { status: "error", isStale: false };
  }
  if (connectionStatus === "disconnected") {
    return { status: "idle", isStale: false };
  }
  if (connectionStatus === "connecting" || connectionStatus === "reconnecting") {
    return { status: "connecting", isStale: false };
  }

  // Connected to LiveKit SFU
  if (lastTelemetryAtMs !== null && lastTelemetryAtMs > 0) {
    if (nowMs - lastTelemetryAtMs > staleAfterMs) {
      return { status: "stale", isStale: true };
    }
    return { status: "processing", isStale: false };
  }

  if (connectedAtMs !== null && nowMs - connectedAtMs > initialFrameTimeoutMs) {
    return { status: "stale", isStale: true };
  }
  return { status: "awaiting_telemetry", isStale: false };
}

export interface ConnectSessionDeps {
  roomName: string;
  driverId: string;
  experimentMode: ExperimentMode;
  backendMode: BackendTargetMode;
  localBaseUrl?: string;
  directCloudBaseUrl?: string;
  fetchImpl?: typeof fetch;
  createRoom: () => LiveKitRoomLike;
  videoTrack?: LiveKitTrackLike | null;
}

export interface ActiveLiveKitConnection {
  room: LiveKitRoomLike;
  tokenResponse: LiveKitTokenResponse;
  bearerToken: string | null;
  publishedTrack: unknown | null;
}

/**
 * Acquire auth token + LiveKit token, connect Room, and publish camera track once without duplication.
 */
export async function establishLiveKitSession(
  deps: ConnectSessionDeps
): Promise<ActiveLiveKitConnection> {
  const fetchImpl = deps.fetchImpl ?? fetch;

  let bearerToken: string | null = null;
  try {
    bearerToken = await fetchBackendAccessToken(fetchImpl);
  } catch {
    if (deps.backendMode === "local") {
      throw new Error("Authentication required: failed to acquire backend access token for local FastAPI.");
    }
  }

  let tokenResponse: LiveKitTokenResponse;
  try {
    tokenResponse = await requestLiveKitToken(
      {
        roomName: deps.roomName,
        driverId: deps.driverId,
        experimentMode: deps.experimentMode,
        autoStartWorker: true,
      },
      {
        mode: deps.backendMode,
        localBaseUrl: deps.localBaseUrl,
        fetchImpl,
        bearerToken,
      }
    );
  } catch (err) {
    // Fallback to direct cloud URL only in cloud mode if Next.js proxy failed and we have a bearer token
    if (deps.backendMode === "cloud" && deps.directCloudBaseUrl && bearerToken) {
      tokenResponse = await requestLiveKitToken(
        {
          roomName: deps.roomName,
          driverId: deps.driverId,
          experimentMode: deps.experimentMode,
          autoStartWorker: true,
        },
        {
          mode: "local",
          localBaseUrl: deps.directCloudBaseUrl,
          fetchImpl,
          bearerToken,
        }
      );
    } else {
      throw err;
    }
  }

  const room = deps.createRoom();
  await room.connect(tokenResponse.url, tokenResponse.token);

  let publishedTrack: unknown | null = null;
  if (deps.videoTrack) {
    const pub = await room.localParticipant.publishTrack(deps.videoTrack, {
      name: "driver-camera",
      simulcast: false,
      videoEncoding: {
        maxBitrate: 650000,
        maxFramerate: 25,
      },
    });
    publishedTrack = pub?.track ?? deps.videoTrack;
  }

  return {
    room,
    tokenResponse,
    bearerToken,
    publishedTrack,
  };
}

/**
 * Unpublish camera track (without killing browser MediaStreamTrack), disconnect LiveKit room,
 * and stop the FastAPI worker with proper Authorization header.
 */
export async function teardownLiveKitSession(params: {
  room: LiveKitRoomLike | null;
  publishedTrack: unknown | null;
  roomName: string;
  backendMode: BackendTargetMode;
  localBaseUrl?: string;
  bearerToken?: string | null;
  fetchImpl?: typeof fetch;
  keepalive?: boolean;
}): Promise<void> {
  const {
    room,
    publishedTrack,
    roomName,
    backendMode,
    localBaseUrl,
    bearerToken,
    fetchImpl = fetch,
    keepalive = true,
  } = params;

  if (room) {
    try {
      if (publishedTrack && room.localParticipant?.unpublishTrack) {
        await room.localParticipant.unpublishTrack(publishedTrack, false);
      }
    } catch {
      // Non-fatal during teardown
    }

    try {
      await room.disconnect(false);
    } catch {
      // Non-fatal during teardown
    }
  }

  await stopLiveKitWorker(
    roomName,
    {
      mode: backendMode,
      localBaseUrl,
      bearerToken,
      fetchImpl,
    },
    { keepalive }
  ).catch(() => {
    // Non-fatal if server already cleaned up or browser is unloading
  });
}
