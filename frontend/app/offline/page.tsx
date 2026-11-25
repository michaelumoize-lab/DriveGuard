import type { Metadata } from "next";
import { ShieldAlert, WifiOff } from "lucide-react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { OfflineActions } from "@/components/pwa/offline-actions";

export const metadata: Metadata = {
  title: "Offline Mode",
  description:
    "Network connectivity is unavailable. Review offline DriveGuard limitations and retry when reconnected.",
};

export default function OfflinePage() {
  return (
    <main
      id="main-content"
      className="min-h-svh flex items-center justify-center bg-background text-foreground px-4 py-8 pt-safe pb-safe"
    >
      <Card className="w-full max-w-md border-border/80 shadow-lg">
        <CardHeader className="space-y-3">
          <div className="flex items-center justify-between gap-2">
            <div className="flex size-11 items-center justify-center rounded-xl bg-amber-500/15 border border-amber-500/30 text-amber-500">
              <WifiOff className="size-5" aria-hidden="true" />
            </div>
            <Badge
              variant="outline"
              className="text-[11px] font-mono border-amber-500/40 text-amber-600 dark:text-amber-400"
            >
              OFFLINE FALLBACK
            </Badge>
          </div>
          <CardTitle className="text-xl font-bold tracking-tight text-balance">
            No Network Connection
          </CardTitle>
          <CardDescription className="text-xs sm:text-sm leading-relaxed">
            Authenticated dashboard routes, cloud session sync, and Remote LiveKit AI require an
            active internet connection and are not cached offline for account privacy.
          </CardDescription>
        </CardHeader>

        <CardContent className="space-y-4">
          <div className="rounded-xl border border-border bg-muted/35 p-3.5 text-xs text-muted-foreground space-y-2 leading-relaxed">
            <p className="font-semibold text-foreground flex items-center gap-1.5">
              <ShieldAlert className="size-3.5 text-amber-500 shrink-0" aria-hidden="true" />
              Safety &amp; Privacy Notice
            </p>
            <ul className="list-disc pl-4 space-y-1">
              <li>Raw camera frames never leave your device.</li>
              <li>
                Live monitoring only runs while DriveGuard stays open in the foreground; mobile
                browsers pause camera access when locked or backgrounded.
              </li>
            </ul>
          </div>

          <OfflineActions />
        </CardContent>
      </Card>
    </main>
  );
}
