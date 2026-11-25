// DriveGuard Service Worker (Native Next.js 16 PWA)
// Strict security & privacy invariants:
// 1. NEVER cache authenticated/dynamic routes (/dashboard*, /settings*, /onboarding*, /auth*, /api*, /).
// 2. NEVER cache Next.js RSC flight payloads (_rsc / RSC header) for authenticated or user-specific routes.
// 3. NEVER force skipWaiting() or client reloads mid-monitoring; updates wait cleanly for explicit idle trigger or next session.

const CACHE_VERSION = "v1";
const SHELL_CACHE = `driveguard-shell-${CACHE_VERSION}`;
const ASSET_CACHE = `driveguard-assets-${CACHE_VERSION}`;
const MODEL_CACHE = `driveguard-mediapipe-${CACHE_VERSION}`;

const PRECACHE_SHELL_URLS = [
  "/offline",
  "/local-monitor",
  "/manifest.webmanifest",
  "/icons/icon-192.png",
  "/icons/icon-512.png",
  "/icons/icon-maskable-192.png",
  "/icons/icon-maskable-512.png",
  "/apple-icon.png",
];

const PRECACHE_MODEL_URLS = ["/models/face_landmarker.task"];

const NEVER_CACHE_PREFIXES = [
  "/api/",
  "/auth/",
  "/dashboard",
  "/settings",
  "/onboarding",
  "/_next/image",
];

function isNeverCachedPath(pathname) {
  if (pathname === "/") return true;
  return NEVER_CACHE_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(prefix)
  );
}

function isNextRscOrPrefetch(request, url) {
  return (
    url.searchParams.has("_rsc") ||
    request.headers.get("RSC") === "1" ||
    request.headers.get("Next-Router-Prefetch") === "1"
  );
}

function isStaticAssetPath(pathname) {
  return pathname.startsWith("/_next/static/");
}

function isMediaPipeAssetPath(pathname) {
  return (
    pathname.startsWith("/models/") ||
    pathname.startsWith("/wasm/")
  );
}

function isShellIconOrManifest(pathname) {
  return (
    pathname === "/manifest.webmanifest" ||
    pathname === "/apple-icon.png" ||
    pathname.startsWith("/icons/")
  );
}

function isCacheableShellRoute(pathname) {
  return pathname === "/offline" || pathname === "/local-monitor";
}

async function warmOfflineAssets(offlineResponse) {
  try {
    const html = await offlineResponse.clone().text();
    const matches = html.match(/\/_next\/static\/[^"'\s)\\>]+/g);
    if (!matches || matches.length === 0) return;
    const uniqueUrls = Array.from(new Set(matches));
    const assetCache = await caches.open(ASSET_CACHE);
    await Promise.allSettled(
      uniqueUrls.map(async (assetUrl) => {
        const existing = await assetCache.match(assetUrl);
        if (!existing) {
          const res = await fetch(assetUrl, { cache: "no-cache" });
          if (res.ok) {
            await assetCache.put(assetUrl, res);
          }
        }
      })
    );
  } catch {
    // Non-fatal: runtime asset caching also populates ASSET_CACHE on first load
  }
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    (async () => {
      const shellCache = await caches.open(SHELL_CACHE);
      await Promise.allSettled(
        PRECACHE_SHELL_URLS.map(async (url) => {
          try {
            const res = await fetch(url, { cache: "no-cache" });
            if (res.ok) {
              await shellCache.put(url, res.clone());
              if (url === "/offline" || url === "/local-monitor") {
                await warmOfflineAssets(res);
              }
            }
          } catch {
            // Allow SW installation to complete gracefully if a precache fetch fails transiently
          }
        })
      );

      const modelCache = await caches.open(MODEL_CACHE);
      await Promise.allSettled(
        PRECACHE_MODEL_URLS.map(async (url) => {
          try {
            const existing = await modelCache.match(url);
            if (!existing) {
              const res = await fetch(url);
              if (res.ok) {
                await modelCache.put(url, res);
              }
            }
          } catch {
            // Will runtime-cache on first MediaPipe initialization
          }
        })
      );
    })()
  );
});

self.addEventListener("activate", (event) => {
  const validCaches = new Set([SHELL_CACHE, ASSET_CACHE, MODEL_CACHE]);
  event.waitUntil(
    (async () => {
      const keys = await caches.keys();
      await Promise.all(
        keys
          .filter((key) => key.startsWith("driveguard-") && !validCaches.has(key))
          .map((key) => caches.delete(key))
      );
      await self.clients.claim();
    })()
  );
});

self.addEventListener("message", (event) => {
  if (event.data && event.data.type === "SKIP_WAITING") {
    self.skipWaiting();
  }
});

async function cacheFirst(request, cacheName) {
  const cache = await caches.open(cacheName);
  const cached = await cache.match(request);
  if (cached) {
    return cached;
  }
  const response = await fetch(request);
  if (response && response.ok && response.type === "basic") {
    await cache.put(request, response.clone());
  }
  return response;
}

async function networkFirstShell(request) {
  const cache = await caches.open(SHELL_CACHE);
  try {
    const response = await fetch(request);
    if (response && response.ok && response.type === "basic") {
      await cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await cache.match(request, { ignoreSearch: true });
    if (cached) return cached;
    const offlineFallback = await cache.match("/offline", { ignoreSearch: true });
    return (
      offlineFallback ||
      new Response("DriveGuard is currently offline.", {
        status: 503,
        statusText: "Service Unavailable",
        headers: { "Content-Type": "text/plain; charset=utf-8" },
      })
    );
  }
}

async function networkOnlyWithOfflineNavigationFallback(request) {
  try {
    return await fetch(request);
  } catch {
    if (request.mode === "navigate") {
      const shellCache = await caches.open(SHELL_CACHE);
      const offlinePage = await shellCache.match("/offline", { ignoreSearch: true });
      if (offlinePage) {
        return offlinePage;
      }
    }
    return new Response("Network unavailable.", {
      status: 503,
      statusText: "Service Unavailable",
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }
}

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return;

  // Never cache RSC flight payloads or private/authenticated paths
  if (isNextRscOrPrefetch(request, url) || isNeverCachedPath(url.pathname)) {
    if (request.mode === "navigate") {
      event.respondWith(networkOnlyWithOfflineNavigationFallback(request));
    }
    return;
  }

  if (isMediaPipeAssetPath(url.pathname)) {
    event.respondWith(cacheFirst(request, MODEL_CACHE));
    return;
  }

  if (isStaticAssetPath(url.pathname)) {
    event.respondWith(cacheFirst(request, ASSET_CACHE));
    return;
  }

  if (isShellIconOrManifest(url.pathname)) {
    event.respondWith(cacheFirst(request, SHELL_CACHE));
    return;
  }

  if (isCacheableShellRoute(url.pathname)) {
    event.respondWith(networkFirstShell(request));
    return;
  }

  if (request.mode === "navigate") {
    event.respondWith(networkOnlyWithOfflineNavigationFallback(request));
  }
});
