/**
 * DriveGuard — Upstream FastAPI path resolution for Next.js /api/v1/[...path] BFF proxy.
 *
 * FastAPI exposes operational health routes at root (`/health`, `/health/liveness`,
 * `/health/readiness`) while versioned application endpoints live under `/api/v1/*`.
 */

const PUBLIC_ROOT_HEALTH_SUBPATHS = new Set([
  "health",
  "health/liveness",
  "health/readiness",
]);

export function isPublicHealthSubPath(subPath: string): boolean {
  const normalized = subPath.replace(/^\/+|\/+$/g, "");
  return normalized === "" || PUBLIC_ROOT_HEALTH_SUBPATHS.has(normalized);
}

export function resolveFastApiUpstreamPath(subPath: string): string {
  const normalized = subPath.replace(/^\/+|\/+$/g, "");
  if (PUBLIC_ROOT_HEALTH_SUBPATHS.has(normalized)) {
    return `/${normalized}`;
  }
  return `/api/v1/${normalized}`;
}
