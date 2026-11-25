"use client";

import { useEffect } from "react";

/**
 * Registers the DriveGuard service worker safely without forcing page reloads
 * or interrupting active camera monitoring sessions on updates.
 */
export function SwRegister() {
  useEffect(() => {
    if (typeof window === "undefined" || !("serviceWorker" in navigator)) {
      return;
    }

    const shouldRegister =
      process.env.NODE_ENV === "production" ||
      process.env.NEXT_PUBLIC_ENABLE_DEV_SW === "true";

    if (!shouldRegister) {
      return;
    }

    let cancelled = false;

    const register = async () => {
      try {
        const registration = await navigator.serviceWorker.register("/sw.js", {
          scope: "/",
          updateViaCache: "none",
        });

        if (cancelled) return;

        registration.addEventListener("updatefound", () => {
          const installing = registration.installing;
          if (!installing) return;

          installing.addEventListener("statechange", () => {
            if (
              installing.state === "installed" &&
              navigator.serviceWorker.controller
            ) {
              // Notify interested UI without calling skipWaiting() or reload()
              // mid-monitoring; updates take effect cleanly on next cold launch.
              window.dispatchEvent(
                new CustomEvent("driveguard:sw-update-available", {
                  detail: { registration },
                })
              );
            }
          });
        });
      } catch (err) {
        if (process.env.NODE_ENV !== "production") {
          console.warn("[DriveGuard PWA] Service worker registration failed:", err);
        }
      }
    };

    if (document.readyState === "complete") {
      void register();
    } else {
      window.addEventListener("load", register, { once: true });
    }

    return () => {
      cancelled = true;
      window.removeEventListener("load", register);
    };
  }, []);

  return null;
}
