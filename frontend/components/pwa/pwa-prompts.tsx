"use client";

import { usePathname } from "next/navigation";
import { PwaInstallBanner } from "./pwa-install-prompt";
import { SwUpdateNotice } from "./sw-update-notice";

const SUPPRESSED_PREFIXES = [
  "/dashboard/live-monitor",
  "/dashboard/calibration",
  "/dashboard/camera-simulation",
  "/local-monitor",
  "/onboarding",
  "/auth",
  "/offline",
];

/**
 * Coordinates non-disruptive PWA install and update prompts, staying hidden on
 * active camera monitoring, calibration, auth, and offline fallback routes.
 */
export function PwaPrompts() {
  const pathname = usePathname() || "/";

  const isSuppressedRoute = SUPPRESSED_PREFIXES.some(
    (prefix) => pathname === prefix || pathname.startsWith(`${prefix}/`)
  );

  if (isSuppressedRoute) {
    return null;
  }

  return (
    <div className="fixed bottom-[max(1rem,env(safe-area-inset-bottom,0px))] right-[max(0.75rem,env(safe-area-inset-right,0px))] left-[max(0.75rem,env(safe-area-inset-left,0px))] sm:left-auto sm:w-96 z-40 flex flex-col gap-2 pointer-events-none">
      <div className="pointer-events-auto space-y-2">
        <SwUpdateNotice />
        <PwaInstallBanner />
      </div>
    </div>
  );
}
