/**
 * Centralized FastAPI & Next.js BFF API Client for DriveGuard.
 *
 * Handles:
 * - Cloud mode (authenticated Next.js proxy `/api/v1/*`) and Local mode
 *   (`http://127.0.0.1:8000/api/v1/*` with signed Bearer token from `/api/auth/backend-token`).
 * - Structured error parsing (`DriveGuardApiError`).
 * - Session lifecycle (`createSession`, `getSession`, `terminateSession`).
 * - LiveKit token minting & authenticated worker teardown (`requestLiveKitToken`, `stopLiveKitWorker`).
 * - Driver profile synchronization (`syncDriverProfile`).
 */

import type {
  ApiErrorPayload,
  CalibratedProfilePayload,
  ExperimentMode,
  LiveKitTokenResponse,
  SessionCreateRequest,
  SessionCreateResponse,
  StopWorkerResponse,
} from "@/types/telemetry";

export type BackendTargetMode = "cloud" | "local";

export class DriveGuardApiError extends Error {
  public readonly status: number;
  public readonly endpoint: string;
  public readonly detail: string;

  constructor(endpoint: string, status: number, detail: string) {
    super(detail);
    this.name = "DriveGuardApiError";
    this.status = status;
    this.endpoint = endpoint;
    this.detail = detail;
  }
}

export interface ApiClientConfig {
  mode: BackendTargetMode;
  localBaseUrl?: string;
  fetchImpl?: typeof fetch;
  /** Optional pre-fetched Bearer token (used in local mode or tests). */
  bearerToken?: string | null;
}

const DEFAULT_LOCAL_BASE_URL = "http://127.0.0.1:8000";

export function resolveApiPrefix(mode: BackendTargetMode, localBaseUrl = DEFAULT_LOCAL_BASE_URL): string {
  if (mode === "local") {
    return localBaseUrl.replace(/\/+$/, "");
  }
  return "";
}

/**
 * Fetch a short-lived signed FastAPI Bearer token from Next.js `/api/auth/backend-token`.
 */
export async function fetchBackendAccessToken(fetchImpl: typeof fetch = fetch): Promise<string> {
  const res = await fetchImpl("/api/auth/backend-token", {
    method: "GET",
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res, "Failed to mint FastAPI authentication token");
    throw new DriveGuardApiError("/api/auth/backend-token", res.status, detail);
  }
  const data = (await res.json()) as { token?: string };
  if (!data?.token) {
    throw new DriveGuardApiError(
      "/api/auth/backend-token",
      500,
      "Backend token endpoint returned an empty token."
    );
  }
  return data.token;
}

async function extractErrorDetail(res: Response, fallback: string): Promise<string> {
  try {
    const body = (await res.json()) as ApiErrorPayload;
    return body.detail || body.error || body.message || `${fallback} (HTTP ${res.status})`;
  } catch {
    return `${fallback} (HTTP ${res.status})`;
  }
}

async function buildAuthHeaders(config: ApiClientConfig, baseHeaders: Record<string, string> = {}): Promise<Record<string, string>> {
  const headers: Record<string, string> = { ...baseHeaders };
  if (config.mode === "local") {
    const token = config.bearerToken || (await fetchBackendAccessToken(config.fetchImpl ?? fetch));
    headers["Authorization"] = `Bearer ${token}`;
  } else if (config.bearerToken) {
    headers["Authorization"] = `Bearer ${config.bearerToken}`;
  }
  return headers;
}

export async function createMonitoringSession(
  payload: SessionCreateRequest,
  config: ApiClientConfig
): Promise<SessionCreateResponse> {
  const fetchFn = config.fetchImpl ?? fetch;
  const endpoint = `${resolveApiPrefix(config.mode, config.localBaseUrl)}/api/v1/sessions`;
  const headers = await buildAuthHeaders(config, { "Content-Type": "application/json" });
  const res = await fetchFn(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res, "Failed to initialize monitoring session");
    throw new DriveGuardApiError(endpoint, res.status, detail);
  }
  return (await res.json()) as SessionCreateResponse;
}

export async function terminateMonitoringSession(
  sessionId: string,
  config: ApiClientConfig
): Promise<Record<string, unknown> | null> {
  const fetchFn = config.fetchImpl ?? fetch;
  const endpoint = `${resolveApiPrefix(config.mode, config.localBaseUrl)}/api/v1/sessions/${encodeURIComponent(sessionId)}`;
  const headers = await buildAuthHeaders(config);
  const res = await fetchFn(endpoint, {
    method: "DELETE",
    headers,
  });
  if (res.status === 404) {
    return null;
  }
  if (!res.ok) {
    const detail = await extractErrorDetail(res, "Failed to terminate monitoring session");
    throw new DriveGuardApiError(endpoint, res.status, detail);
  }
  return (await res.json()) as Record<string, unknown>;
}

export async function syncDriverProfileToFastApi(
  profileDto: CalibratedProfilePayload,
  config: ApiClientConfig
): Promise<CalibratedProfilePayload> {
  const fetchFn = config.fetchImpl ?? fetch;
  const endpoint = `${resolveApiPrefix(config.mode, config.localBaseUrl)}/api/v1/profiles`;
  const headers = await buildAuthHeaders(config, { "Content-Type": "application/json" });
  const res = await fetchFn(endpoint, {
    method: "POST",
    headers,
    body: JSON.stringify(profileDto),
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res, "Failed to sync driver profile to FastAPI");
    throw new DriveGuardApiError(endpoint, res.status, detail);
  }
  return (await res.json()) as CalibratedProfilePayload;
}

export interface RequestLiveKitTokenParams {
  roomName: string;
  driverId: string;
  participantName?: string;
  experimentMode?: ExperimentMode;
  autoStartWorker?: boolean;
}

export async function requestLiveKitToken(
  params: RequestLiveKitTokenParams,
  config: ApiClientConfig
): Promise<LiveKitTokenResponse> {
  const fetchFn = config.fetchImpl ?? fetch;
  const qs = new URLSearchParams({
    room_name: params.roomName,
    driver_id: params.driverId,
    auto_start_worker: String(params.autoStartWorker ?? true),
  });
  if (params.participantName) qs.set("participant_name", params.participantName);
  if (params.experimentMode) qs.set("experiment_mode", params.experimentMode);

  const endpoint = `${resolveApiPrefix(config.mode, config.localBaseUrl)}/api/v1/livekit/token?${qs.toString()}`;
  const headers = await buildAuthHeaders(config);
  const res = await fetchFn(endpoint, {
    method: "GET",
    headers,
    cache: "no-store",
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res, "Failed to retrieve LiveKit WebRTC token");
    throw new DriveGuardApiError(endpoint, res.status, detail);
  }
  return (await res.json()) as LiveKitTokenResponse;
}

export async function stopLiveKitWorker(
  roomName: string,
  config: ApiClientConfig,
  options: { keepalive?: boolean } = {}
): Promise<StopWorkerResponse> {
  const fetchFn = config.fetchImpl ?? fetch;
  const qs = new URLSearchParams({ room_name: roomName });
  const endpoint = `${resolveApiPrefix(config.mode, config.localBaseUrl)}/api/v1/livekit/stop-worker?${qs.toString()}`;
  const headers = await buildAuthHeaders(config);
  const res = await fetchFn(endpoint, {
    method: "POST",
    headers,
    keepalive: options.keepalive,
  });
  if (!res.ok) {
    const detail = await extractErrorDetail(res, "Failed to stop LiveKit AI worker");
    throw new DriveGuardApiError(endpoint, res.status, detail);
  }
  return (await res.json()) as StopWorkerResponse;
}
