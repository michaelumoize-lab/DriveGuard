"use client";

import { useCallback, useEffect, useState } from "react";
import Link from "next/link";
import { Download, Share, Smartphone, X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface BeforeInstallPromptEvent extends Event {
  readonly platforms: string[];
  readonly userChoice: Promise<{
    outcome: "accepted" | "dismissed";
    platform: string;
  }>;
  prompt(): Promise<void>;
}

const DISMISS_STORAGE_KEY = "driveguard_pwa_install_dismissed_v1";
const DISMISS_COOLDOWN_MS = 7 * 24 * 60 * 60 * 1000;

export interface UsePwaInstallResult {
  canInstallPrompt: boolean;
  isIosSafari: boolean;
  isStandalone: boolean;
  isDismissed: boolean;
  promptInstall: () => Promise<boolean>;
  dismissPrompt: () => void;
}

export function usePwaInstall(): UsePwaInstallResult {
  const [deferredPrompt, setDeferredPrompt] = useState<BeforeInstallPromptEvent | null>(null);
  const [isIosSafari, setIsIosSafari] = useState<boolean>(false);
  const [isStandalone, setIsStandalone] = useState<boolean>(false);
  const [isDismissed, setIsDismissed] = useState<boolean>(false);

  useEffect(() => {
    if (typeof window === "undefined") return;

    const standalone =
      window.matchMedia("(display-mode: standalone)").matches ||
      Boolean((navigator as { standalone?: boolean }).standalone);

    const ua = navigator.userAgent || "";
    const isIos = /iphone|ipad|ipod/i.test(ua);
    const isSafari = /safari/i.test(ua) && !/crios|fxios|edgios|opios/i.test(ua);

    let dismissedRecently = false;
    try {
      const raw = window.localStorage.getItem(DISMISS_STORAGE_KEY);
      if (raw) {
        const ts = Number(raw);
        if (Number.isFinite(ts) && Date.now() - ts < DISMISS_COOLDOWN_MS) {
          dismissedRecently = true;
        }
      }
    } catch {
      // Ignore storage access errors
    }

    queueMicrotask(() => {
      setIsStandalone(standalone);
      setIsIosSafari(isIos && isSafari && !standalone);
      setIsDismissed(dismissedRecently);
    });

    const onBeforeInstallPrompt = (e: Event) => {
      e.preventDefault();
      setDeferredPrompt(e as BeforeInstallPromptEvent);
    };

    const onAppInstalled = () => {
      setDeferredPrompt(null);
      setIsStandalone(true);
    };

    window.addEventListener("beforeinstallprompt", onBeforeInstallPrompt);
    window.addEventListener("appinstalled", onAppInstalled);
    return () => {
      window.removeEventListener("beforeinstallprompt", onBeforeInstallPrompt);
      window.removeEventListener("appinstalled", onAppInstalled);
    };
  }, []);

  const promptInstall = useCallback(async (): Promise<boolean> => {
    if (!deferredPrompt) return false;
    try {
      await deferredPrompt.prompt();
      const { outcome } = await deferredPrompt.userChoice;
      setDeferredPrompt(null);
      if (outcome === "accepted") {
        setIsStandalone(true);
        return true;
      }
      return false;
    } catch {
      return false;
    }
  }, [deferredPrompt]);

  const dismissPrompt = useCallback(() => {
    setIsDismissed(true);
    try {
      window.localStorage.setItem(DISMISS_STORAGE_KEY, String(Date.now()));
    } catch {
      // Ignore storage errors
    }
  }, []);

  return {
    canInstallPrompt: Boolean(deferredPrompt),
    isIosSafari,
    isStandalone,
    isDismissed,
    promptInstall,
    dismissPrompt,
  };
}

export function PwaInstallBanner() {
  const {
    canInstallPrompt,
    isIosSafari,
    isStandalone,
    isDismissed,
    promptInstall,
    dismissPrompt,
  } = usePwaInstall();

  if (isStandalone || isDismissed || (!canInstallPrompt && !isIosSafari)) {
    return null;
  }

  return (
    <aside
      aria-label="Install DriveGuard app"
      className="rounded-2xl border border-border bg-card/95 backdrop-blur-md shadow-xl p-3.5 text-xs space-y-2.5"
    >
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-start gap-2.5 min-w-0">
          <div className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-primary/15 text-primary border border-primary/25">
            <Smartphone className="size-4" aria-hidden="true" />
          </div>
          <div className="min-w-0 space-y-0.5">
            <p className="font-semibold text-foreground text-sm leading-tight">
              Install DriveGuard
            </p>
            {canInstallPrompt ? (
              <p className="text-muted-foreground leading-relaxed text-pretty">
                Add to your home screen for full-screen cockpit access and offline{" "}
                <span className="font-mono text-foreground">/local-monitor</span>.
              </p>
            ) : (
              <p className="text-muted-foreground leading-relaxed text-pretty">
                In Safari, tap <Share className="inline size-3.5 mx-0.5 text-foreground" aria-hidden="true" />
                <strong className="text-foreground">Share</strong> then{" "}
                <strong className="text-foreground">Add to Home Screen</strong>. Camera
                monitoring runs only while DriveGuard is open in the foreground.
              </p>
            )}
          </div>
        </div>

        <Button
          type="button"
          size="icon"
          variant="ghost"
          onClick={dismissPrompt}
          aria-label="Dismiss install prompt"
          className="size-11 sm:size-8 shrink-0 text-muted-foreground hover:text-foreground -mr-1.5 -mt-1.5"
        >
          <X className="size-4" aria-hidden="true" />
        </Button>
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {canInstallPrompt && (
          <Button
            type="button"
            size="sm"
            onClick={() => void promptInstall()}
            className="min-h-11 sm:h-9 sm:min-h-0 flex-1 gap-1.5 font-semibold"
          >
            <Download className="size-3.5" aria-hidden="true" />
            Install App
          </Button>
        )}
        <Button
          asChild
          size="sm"
          variant="outline"
          className="min-h-11 sm:h-9 sm:min-h-0 flex-1 text-xs"
        >
          <Link href="/local-monitor">Offline Monitor</Link>
        </Button>
      </div>
    </aside>
  );
}
