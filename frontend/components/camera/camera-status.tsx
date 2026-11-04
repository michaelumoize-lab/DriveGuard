"use client";

import React from "react";
import { CameraResolution, CameraState } from "@/types/camera";
import { Badge } from "@/components/ui/badge";
import { Activity, Clock, Cpu, Eye, ShieldCheck, Video } from "lucide-react";

interface CameraStatusProps {
  cameraState: CameraState;
  resolution: CameraResolution | null;
  facingMode: "user" | "environment";
  durationSec: number;
  activeDeviceLabel?: string;
}

export function CameraStatus({
  cameraState,
  resolution,
  facingMode,
  durationSec,
  activeDeviceLabel,
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
          <Badge variant="outline" className="border-emerald-500/40 bg-emerald-500/10 text-emerald-500 text-xs font-mono flex items-center gap-1.5 px-2.5 py-1">
            <span className="size-2 rounded-full bg-emerald-500 animate-pulse" />
            <span>ACTIVE STREAM</span>
          </Badge>
        );
      case "paused":
        return (
          <Badge variant="outline" className="border-amber-500/40 bg-amber-500/10 text-amber-500 text-xs font-mono flex items-center gap-1.5 px-2.5 py-1">
            <span className="size-2 rounded-full bg-amber-500" />
            <span>STREAM PAUSED</span>
          </Badge>
        );
      case "requesting-permission":
      case "permission-granted":
        return (
          <Badge variant="outline" className="border-blue-500/40 bg-blue-500/10 text-blue-400 text-xs font-mono flex items-center gap-1.5 px-2.5 py-1">
            <span className="size-2 rounded-full bg-blue-400 animate-ping" />
            <span>CONNECTING</span>
          </Badge>
        );
      case "stopped":
      case "not-initialized":
        return (
          <Badge variant="outline" className="border-border bg-muted/40 text-muted-foreground text-xs font-mono flex items-center gap-1.5 px-2.5 py-1">
            <span className="size-2 rounded-full bg-muted-foreground/50" />
            <span>STANDBY</span>
          </Badge>
        );
      default:
        return (
          <Badge variant="outline" className="border-rose-500/40 bg-rose-500/10 text-rose-500 text-xs font-mono flex items-center gap-1.5 px-2.5 py-1">
            <span className="size-2 rounded-full bg-rose-500" />
            <span>HARDWARE ERROR</span>
          </Badge>
        );
    }
  };

  return (
    <div className="flex flex-wrap items-center justify-between gap-2 p-3 rounded-2xl bg-card/60 backdrop-blur-xs border border-border">
      <div className="flex flex-wrap items-center gap-2">
        {getStatusBadge()}

        {/* Resolution Badge */}
        {resolution ? (
          <Badge variant="secondary" className="font-mono text-[11px] gap-1 px-2.5 py-1">
            <Video className="size-3 text-primary" />
            <span>
              {resolution.width} × {resolution.height}
              {resolution.frameRate ? ` @ ${resolution.frameRate}fps` : ""}
            </span>
          </Badge>
        ) : (
          <Badge variant="secondary" className="font-mono text-[11px] text-muted-foreground px-2.5 py-1">
            Resolution: --
          </Badge>
        )}

        {/* Facing Mode Badge */}
        <Badge variant="secondary" className="font-mono text-[11px] capitalize px-2.5 py-1">
          {facingMode === "user" ? "Front (Driver)" : "Rear (Cabin)"}
        </Badge>
      </div>

      <div className="flex items-center gap-3 text-xs font-mono text-muted-foreground">
        {/* Stream Duration */}
        <div className="flex items-center gap-1.5">
          <Clock className="size-3.5 text-primary" />
          <span>{formatDuration(durationSec)}</span>
        </div>

        {/* Local Simulation Mode Chip */}
        <div className="hidden sm:flex items-center gap-1 text-[10px] text-muted-foreground/80 border-l border-border pl-3">
          <ShieldCheck className="size-3 text-emerald-500" />
          <span>Local Simulation</span>
        </div>
      </div>
    </div>
  );
}
