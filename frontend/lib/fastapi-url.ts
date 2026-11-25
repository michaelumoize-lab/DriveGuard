/**
 * Centralized FastAPI Backend URL Resolution
 * ==========================================
 * - In development / non-production (`NODE_ENV !== "production"`), defaults to
 *   `LOCAL_FASTAPI_URL` (`http://127.0.0.1:8000`) when no explicit env URL is configured.
 * - In production (`NODE_ENV === "production"`), falls back to the canonical
 *   production Render service (`https://driveguard-x048.onrender.com`) and strips
 *   stale Render PR preview URLs (`driveguard-pr-*.onrender.com`) or localhost values.
 */

export const PRODUCTION_FASTAPI_URL = "https://driveguard-x048.onrender.com";
export const PRODUCTION_FASTAPI_WS_URL = "wss://driveguard-x048.onrender.com";
export const LOCAL_FASTAPI_URL = "http://127.0.0.1:8000";
export const LOCAL_FASTAPI_WS_URL = "ws://127.0.0.1:8000";

const STALE_PREVIEW_HOST_PATTERN = /driveguard-pr-\d+\.onrender\.com/i;

export function sanitizeFastApiUrl(rawUrl: string | undefined | null, fallback: string): string {
  if (!rawUrl || !rawUrl.trim()) {
    return fallback;
  }
  const cleaned = rawUrl.trim().replace(/\/+$/, "");
  if (STALE_PREVIEW_HOST_PATTERN.test(cleaned)) {
    return fallback;
  }
  return cleaned;
}

/**
 * Resolves the authoritative FastAPI HTTP base URL.
 * Local-first (`http://127.0.0.1:8000`) when `NODE_ENV !== "production"` and env is unset;
 * production Render fallback strictly reserved for `NODE_ENV === "production"`.
 */
export function resolveCloudFastApiUrl(): string {
  const isProd = process.env.NODE_ENV === "production";
  const defaultFallback = isProd ? PRODUCTION_FASTAPI_URL : LOCAL_FASTAPI_URL;

  const candidate =
    process.env.FASTAPI_URL ||
    process.env.NEXT_PUBLIC_FASTAPI_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL;

  const resolved = sanitizeFastApiUrl(candidate, defaultFallback);
  if (isProd && (resolved.includes("localhost") || resolved.includes("127.0.0.1"))) {
    return PRODUCTION_FASTAPI_URL;
  }
  return resolved;
}

/**
 * Resolves the authoritative FastAPI WebSocket base URL.
 */
export function resolveCloudFastApiWsUrl(): string {
  const isProd = process.env.NODE_ENV === "production";
  const defaultFallback = isProd ? PRODUCTION_FASTAPI_WS_URL : LOCAL_FASTAPI_WS_URL;
  const candidate = process.env.NEXT_PUBLIC_FASTAPI_WS_URL;
  const resolved = sanitizeFastApiUrl(candidate, defaultFallback);
  if (isProd && (resolved.includes("localhost") || resolved.includes("127.0.0.1"))) {
    return PRODUCTION_FASTAPI_WS_URL;
  }
  return resolved;
}
