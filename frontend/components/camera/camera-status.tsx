"use client";

import React from "react";
import { CameraResolution, CameraState } from "@/types/camera";
import { Badge } from "@/components/ui/badge";
import { Clock, ShieldCheck, Video } from "lucide-react";

interface CameraStatusProps {
  cameraState: CameraState;
  resolution: CameraResolution | null;
  facingMode: "user" | "environment";
  durationSec: number;
  activeDeviceLabel?: string;
  runtimeLabel?: string;
}

export const CameraStatus = React.memo(function CameraStatus({
  cameraState,
  resolution,
  facingMode,
  durationSec,
  runtimeLabel = "On-Device Privacy AI",
}: CameraStatusProps) {
  const formatDuration = (totalSeconds: number) => {
    const mins = Math.floor(totalSeconds / 60);
    const secs = totalSeconds % 60;
    return `${mins}m ${secs < 10 ? "0" : ""}${secs}s`;
  };

  const getStatusBadge = () => {
    switch (cameraState) {
      case "active":
        return (
          <Badge
            variant="outline"
            className="border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 text-xs font-medium flex items-center gap-1.5 px-2.5 py-0.5"
          >
            <span className="size-2 rounded-full bg-emerald-500" aria-hidden="true" />
            <span>Live stream</span>
          </Badge>
        );
      case "paused":
        return (
          <Badge
            variant="outline"
            className="border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400 text-xs font-medium flex items-center gap-1.5 px-2.5 py-0.5"
          >
            <span className="size-2 rounded-full bg-amber-500" aria-hidden="true" />
            <span>Paused</span>
          </Badge>
        );
      case "requesting-permission":
      case "permission-granted":
        return (
          <Badge
            variant="outline"
            className="border-blue-500/40 bg-blue-500/10 text-blue-500 text-xs font-medium flex items-center gap-1.5 px-2.5 py-0.5"
          >
            <span className="size-2 rounded-full bg-blue-500" aria-hidden="true" />
            <span>Connecting…</span>
          </Badge>
        );
      case "stopped":
      case "not-initialized":
        return (
          <Badge
            variant="outline"
            className="border-border bg-muted/40 text-muted-foreground text-xs font-medium flex items-center gap-1.5 px-2.5 py-0.5"
          >
            <span className="size-2 rounded-full bg-muted-foreground/50" aria-hidden="true" />
            <span>Standby</span>
          </Badge>
        );
      default:
        return (
          <Badge
            variant="outline"
            className="border-rose-500/40 bg-rose-500/10 text-rose-500 text-xs font-medium flex items-center gap-1.5 px-2.5 py-0.5"
          >
            <span className="size-2 rounded-full bg-rose-500" aria-hidden="true" />
            <span>Camera error</span>
          </Badge>
        );
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 px-3.5 py-2.5 rounded-2xl bg-card/70 border border-border/80 text-xs">
      <div className="flex flex-wrap items-center gap-2">
        {getStatusBadge()}

        {resolution ? (
          <Badge variant="secondary" className="font-mono tabular-nums text-[11px] gap-1 px-2 py-0.5">
            <Video className="size-3 text-primary" aria-hidden="true" />
            <span>
              {resolution.width}×{resolution.height}
              {resolution.frameRate ? ` · ${resolution.frameRate}\u00A0fps` : ""}
            </span>
          </Badge>
        ) : (
          <Badge variant="secondary" className="font-mono tabular-nums text-[11px] text-muted-foreground px-2 py-0.5">
            640×480 target
          </Badge>
        )}

        <Badge variant="secondary" className="text-[11px] px-2 py-0.5">
          {facingMode === "user" ? "Front camera" : "Rear camera"}
        </Badge>
      </div>

      <div className="flex flex-wrap items-center gap-2 sm:gap-3 text-xs text-muted-foreground">
        <div className="flex items-center gap-1.5 font-mono tabular-nums">
          <Clock className="size-3.5 text-primary" aria-hidden="true" />
          <span>{formatDuration(durationSec)}</span>
        </div>

        <div className="flex items-center gap-1 text-[11px] sm:border-l sm:border-border sm:pl-3">
          <ShieldCheck className="size-3.5 text-emerald-500 shrink-0" aria-hidden="true" />
          <span>{runtimeLabel}</span>
        </div>
      </div>
    </div>
  );
});
