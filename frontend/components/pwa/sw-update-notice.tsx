"use client";

import { useCallback, useEffect, useState } from "react";
import { RefreshCw, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface SwUpdateEventDetail {
  registration?: ServiceWorkerRegistration;
}

export function SwUpdateNotice() {
  const [waitingRegistration, setWaitingRegistration] =
    useState<ServiceWorkerRegistration | null>(null);
  const [dismissed, setDismissed] = useState<boolean>(false);
  const [isApplying, setIsApplying] = useState<boolean>(false);

  useEffect(() => {
    const onUpdateAvailable = (event: Event) => {
      const custom = event as CustomEvent<SwUpdateEventDetail>;
      if (custom.detail?.registration) {
        setWaitingRegistration(custom.detail.registration);
        setDismissed(false);
      }
    };

    window.addEventListener("driveguard:sw-update-available", onUpdateAvailable);
    return () => {
      window.removeEventListener("driveguard:sw-update-available", onUpdateAvailable);
    };
  }, []);

  const handleApplyUpdate = useCallback(() => {
    if (!waitingRegistration?.waiting) {
      window.location.reload();
      return;
    }

    setIsApplying(true);
    let reloaded = false;
    const triggerReload = () => {
      if (reloaded) return;
      reloaded = true;
      window.location.reload();
    };

    navigator.serviceWorker.addEventListener("controllerchange", triggerReload, {
      once: true,
    });
    waitingRegistration.waiting.postMessage({ type: "SKIP_WAITING" });

    // Safety fallback if controllerchange already fired
    setTimeout(triggerReload, 1500);
  }, [waitingRegistration]);

  if (!waitingRegistration || dismissed) {
    return null;
  }

  return (
    <div
      role="status"
      aria-live="polite"
      className="rounded-2xl border border-primary/40 bg-card/95 backdrop-blur-md shadow-xl p-3.5 text-xs space-y-2.5"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="space-y-0.5">
          <p className="font-semibold text-foreground text-sm leading-tight">
            App Update Ready
          </p>
          <p className="text-muted-foreground leading-relaxed">
            A newer DriveGuard offline shell is cached and ready when you are idle.
          </p>
        </div>
        <Button
          type="button"
          size="icon"
          variant="ghost"
          onClick={() => setDismissed(true)}
          aria-label="Dismiss update notification"
          className="size-11 sm:size-8 shrink-0 text-muted-foreground hover:text-foreground -mr-1.5 -mt-1.5"
        >
          <X className="size-4" aria-hidden="true" />
        </Button>
      </div>

      <div className="flex items-center gap-2">
        <Button
          type="button"
          size="sm"
          onClick={handleApplyUpdate}
          disabled={isApplying}
          className="min-h-11 sm:h-9 sm:min-h-0 flex-1 gap-1.5 font-semibold"
        >
          <RefreshCw
            className={`size-3.5 ${isApplying ? "animate-spin motion-reduce:animate-none" : ""}`}
            aria-hidden="true"
          />
          {isApplying ? "Updating…" : "Refresh to Update"}
        </Button>
        <Button
          type="button"
          size="sm"
          variant="outline"
          onClick={() => setDismissed(true)}
          className="min-h-11 sm:h-9 sm:min-h-0 px-3 text-xs"
        >
          Later
        </Button>
      </div>
    </div>
  );
}
