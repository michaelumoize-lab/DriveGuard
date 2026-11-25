"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import { Activity, Home, RefreshCw, Wifi, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";

export function OfflineActions() {
  const [isOnline, setIsOnline] = useState<boolean>(false);

  useEffect(() => {
    const sync = () => setIsOnline(navigator.onLine);
    sync();
    window.addEventListener("online", sync);
    window.addEventListener("offline", sync);
    return () => {
      window.removeEventListener("online", sync);
      window.removeEventListener("offline", sync);
    };
  }, []);

  return (
    <div className="space-y-4">
      <div
        role="status"
        aria-live="polite"
        className={`flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-medium ${
          isOnline
            ? "border-emerald-500/40 bg-emerald-500/10 text-emerald-700 dark:text-emerald-300"
            : "border-amber-500/40 bg-amber-500/10 text-amber-700 dark:text-amber-300"
        }`}
      >
        {isOnline ? (
          <>
            <Wifi className="size-4 shrink-0" aria-hidden="true" />
            <span>Connection detected. You can reload to resume online features.</span>
          </>
        ) : (
          <>
            <WifiOff className="size-4 shrink-0" aria-hidden="true" />
            <span>Device is currently offline.</span>
          </>
        )}
      </div>

      <div className="space-y-2.5">
        <Button asChild className="min-h-11 w-full gap-2">
          <Link href="/local-monitor">
            <Activity className="size-4" aria-hidden="true" />
            Launch Offline Local Monitor
          </Link>
        </Button>

        <div className="flex flex-col sm:flex-row gap-2.5">
          <Button
            type="button"
            variant="secondary"
            onClick={() => window.location.reload()}
            className="min-h-11 flex-1 gap-2"
          >
            <RefreshCw className="size-4" aria-hidden="true" />
            Retry Connection
          </Button>

          <Button asChild variant="outline" className="min-h-11 flex-1 gap-2">
            <Link href="/">
              <Home className="size-4" aria-hidden="true" />
              Return Home
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}
