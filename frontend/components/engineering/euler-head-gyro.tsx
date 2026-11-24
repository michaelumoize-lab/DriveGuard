"use client";

import React, { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Compass, Eye } from "lucide-react";

interface EulerHeadGyroProps {
  pitchDeg?: number;
  yawDeg?: number;
  rollDeg?: number;
  zone?: string;
  hasTelemetry?: boolean;
  className?: string;
}

export function EulerHeadGyro({
  pitchDeg = 0,
  yawDeg = 0,
  rollDeg = 0,
  zone = "CENTER",
  hasTelemetry = true,
  className = "",
}: EulerHeadGyroProps) {
  const yawOffsetPx = hasTelemetry ? Math.max(-48, Math.min(48, (yawDeg / 40) * 48)) : 0;
  const pitchOffsetPx = hasTelemetry ? Math.max(-36, Math.min(36, (-pitchDeg / 30) * 36)) : 0;

  const zoneColor = useMemo(() => {
    if (!hasTelemetry) {
      return "text-muted-foreground border-border bg-muted/40";
    }
    switch (zone) {
      case "LEFT_MIRROR":
        return "text-sky-400 border-sky-400/40 bg-sky-400/10";
      case "RIGHT_MIRROR":
        return "text-indigo-400 border-indigo-400/40 bg-indigo-400/10";
      case "DASHBOARD":
        return "text-amber-400 border-amber-400/40 bg-amber-400/10";
      default:
        return "text-emerald-400 border-emerald-400/40 bg-emerald-400/10";
    }
  }, [hasTelemetry, zone]);

  return (
    <div className={`p-4 rounded-2xl bg-card border border-border/80 shadow-xs space-y-3 ${className}`}>
      <div className="flex items-center justify-between border-b border-border/60 pb-2 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Compass className="size-4 text-primary shrink-0" aria-hidden="true" />
          <span className="text-xs font-semibold text-foreground truncate">
            3D Head Pose Gyroscope
          </span>
        </div>
        <Badge variant="outline" className={`text-[10px] font-mono ${zoneColor}`}>
          {hasTelemetry ? zone : "STANDBY"}
        </Badge>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center">
        <div
          aria-hidden="true"
          className="sm:col-span-5 relative w-36 h-36 mx-auto rounded-full border-2 border-border/80 bg-black/40 flex items-center justify-center overflow-hidden shadow-inner"
        >
          <div className="absolute inset-0 border-b border-dashed border-zinc-700/60 top-1/2 -translate-y-1/2 pointer-events-none" />
          <div className="absolute inset-0 border-r border-dashed border-zinc-700/60 left-1/2 -translate-x-1/2 pointer-events-none" />

          <span className="absolute top-1 text-[9px] font-mono text-zinc-500">UP</span>
          <span className="absolute bottom-1 text-[9px] font-mono text-amber-500/80">DASH</span>
          <span className="absolute left-1.5 text-[9px] font-mono text-sky-400/80">LEFT</span>
          <span className="absolute right-1.5 text-[9px] font-mono text-indigo-400/80">RIGHT</span>

          <div
            style={{
              transform: `translate(${yawOffsetPx}px, ${pitchOffsetPx}px) rotate(${hasTelemetry ? rollDeg : 0}deg)`,
            }}
            className={`relative size-10 rounded-full border-2 shadow-lg flex items-center justify-center transition-transform duration-100 ease-out motion-reduce:transition-none ${
              hasTelemetry
                ? "border-primary bg-primary/20 text-primary"
                : "border-muted-foreground/40 bg-muted/20 text-muted-foreground"
            }`}
          >
            <Eye className="size-4" />
            <div
              className={`absolute -top-1 w-1.5 h-1.5 rounded-full ${
                hasTelemetry ? "bg-primary" : "bg-muted-foreground/50"
              }`}
            />
          </div>
        </div>

        <div className="sm:col-span-7 grid grid-cols-3 sm:grid-cols-1 gap-2 text-xs">
          <div className="p-2 rounded-lg bg-muted/25 border border-border/60 flex items-center justify-between gap-2">
            <span className="text-[11px] text-muted-foreground">Yaw (turn)</span>
            <span
              className={`font-mono font-bold tabular-nums ${
                hasTelemetry && Math.abs(yawDeg) > 20 ? "text-sky-400" : "text-foreground"
              }`}
            >
              {!hasTelemetry ? "—" : yawDeg > 0 ? `+${yawDeg.toFixed(1)}°` : `${yawDeg.toFixed(1)}°`}
            </span>
          </div>

          <div className="p-2 rounded-lg bg-muted/25 border border-border/60 flex items-center justify-between gap-2">
            <span className="text-[11px] text-muted-foreground">Pitch (nod)</span>
            <span
              className={`font-mono font-bold tabular-nums ${
                hasTelemetry && pitchDeg < -12 ? "text-amber-400" : "text-foreground"
              }`}
            >
              {!hasTelemetry ? "—" : pitchDeg > 0 ? `+${pitchDeg.toFixed(1)}°` : `${pitchDeg.toFixed(1)}°`}
            </span>
          </div>

          <div className="p-2 rounded-lg bg-muted/25 border border-border/60 flex items-center justify-between gap-2">
            <span className="text-[11px] text-muted-foreground">Roll (tilt)</span>
            <span className="font-mono font-bold tabular-nums text-foreground">
              {!hasTelemetry ? "—" : rollDeg > 0 ? `+${rollDeg.toFixed(1)}°` : `${rollDeg.toFixed(1)}°`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
