/**
 * Centralized FastAPI Backend URL Resolution
 * ==========================================
 * Ensures that dead Render PR preview URLs (e.g. `driveguard-pr-1.onrender.com`,
 * which return HTTP 404 `x-render-routing: no-server` without CORS headers)
 * automatically resolve to the live production Render service (`https://driveguard-x048.onrender.com`).
 */

export const PRODUCTION_FASTAPI_URL = "https://driveguard-x048.onrender.com";
export const PRODUCTION_FASTAPI_WS_URL = "wss://driveguard-x048.onrender.com";
export const LOCAL_FASTAPI_URL = "http://127.0.0.1:8000";

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
 * Resolves the authoritative Cloud FastAPI HTTP base URL.
 */
export function resolveCloudFastApiUrl(): string {
  const candidate =
    process.env.FASTAPI_URL ||
    process.env.NEXT_PUBLIC_FASTAPI_URL ||
    process.env.NEXT_PUBLIC_API_BASE_URL;

  const resolved = sanitizeFastApiUrl(candidate, PRODUCTION_FASTAPI_URL);
  if (
    process.env.NODE_ENV === "production" &&
    (resolved.includes("localhost") || resolved.includes("127.0.0.1"))
  ) {
    return PRODUCTION_FASTAPI_URL;
  }
  return resolved;
}

/**
 * Resolves the authoritative Cloud FastAPI WebSocket base URL.
 */
export function resolveCloudFastApiWsUrl(): string {
  const candidate = process.env.NEXT_PUBLIC_FASTAPI_WS_URL;
  return sanitizeFastApiUrl(candidate, PRODUCTION_FASTAPI_WS_URL);
}
