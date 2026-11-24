"use client";

import React, { useMemo } from "react";
import { Badge } from "@/components/ui/badge";
import { Compass, Eye } from "lucide-react";

interface EulerHeadGyroProps {
  pitchDeg?: number;
  yawDeg?: number;
  rollDeg?: number;
  zone?: string;
  className?: string;
}

export function EulerHeadGyro({
  pitchDeg = 0,
  yawDeg = 0,
  rollDeg = 0,
  zone = "CENTER",
  className = "",
}: EulerHeadGyroProps) {
  // Normalize angles to clamped 2D visualization offsets
  const yawOffsetPx = Math.max(-48, Math.min(48, (yawDeg / 40) * 48));
  const pitchOffsetPx = Math.max(-36, Math.min(36, (-pitchDeg / 30) * 36));

  const zoneColor = useMemo(() => {
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
  }, [zone]);

  return (
    <div className={`p-4 rounded-2xl bg-card border border-border/80 shadow-xs space-y-3 ${className}`}>
      <div className="flex items-center justify-between border-b border-border/60 pb-2">
        <div className="flex items-center gap-2">
          <Compass className="size-4 text-primary" />
          <span className="text-xs font-bold uppercase tracking-wider text-foreground">
            3D Spatial Head Gyroscope
          </span>
        </div>
        <Badge variant="outline" className={`text-[10px] font-mono ${zoneColor}`}>
          ZONE: {zone}
        </Badge>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-12 gap-3 items-center">
        {/* Visual Euler Sphere Compass */}
        <div className="sm:col-span-5 relative w-36 h-36 mx-auto rounded-full border-2 border-border/80 bg-black/40 flex items-center justify-center overflow-hidden shadow-inner">
          {/* Pitch & Yaw Grid Lines */}
          <div className="absolute inset-0 border-b border-dashed border-zinc-700/60 top-1/2 -translate-y-1/2 pointer-events-none" />
          <div className="absolute inset-0 border-r border-dashed border-zinc-700/60 left-1/2 -translate-x-1/2 pointer-events-none" />

          {/* 4-Zone Peripheral Reticles */}
          <span className="absolute top-1 text-[9px] font-mono text-zinc-500 uppercase tracking-widest">
            UP
          </span>
          <span className="absolute bottom-1 text-[9px] font-mono text-amber-500/80 uppercase tracking-widest">
            DASH
          </span>
          <span className="absolute left-1.5 text-[9px] font-mono text-sky-400/80 uppercase">
            LEFT
          </span>
          <span className="absolute right-1.5 text-[9px] font-mono text-indigo-400/80 uppercase">
            RIGHT
          </span>

          {/* Head Indicator Reticle */}
          <div
            style={{
              transform: `translate(${yawOffsetPx}px, ${pitchOffsetPx}px) rotate(${rollDeg}deg)`,
              transition: "transform 100ms ease-out",
            }}
            className="relative size-10 rounded-full border-2 border-primary bg-primary/20 shadow-lg flex items-center justify-center text-primary"
          >
            <Eye className="size-4" />
            <div className="absolute -top-1 w-1.5 h-1.5 bg-primary rounded-full" />
          </div>
        </div>

        {/* Numeric Euler Angle Readouts */}
        <div className="sm:col-span-7 grid grid-cols-3 sm:grid-cols-1 gap-2 font-mono text-xs">
          <div className="p-2 rounded-lg bg-muted/20 border border-border/60 flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground">YAW (TURNING)</span>
            <span
              className={`font-bold tabular-nums ${
                Math.abs(yawDeg) > 20 ? "text-sky-400" : "text-foreground"
              }`}
            >
              {yawDeg > 0 ? `+${yawDeg.toFixed(1)}°` : `${yawDeg.toFixed(1)}°`}
            </span>
          </div>

          <div className="p-2 rounded-lg bg-muted/20 border border-border/60 flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground">PITCH (NODDING)</span>
            <span
              className={`font-bold tabular-nums ${
                pitchDeg < -12 ? "text-amber-400" : "text-foreground"
              }`}
            >
              {pitchDeg > 0 ? `+${pitchDeg.toFixed(1)}°` : `${pitchDeg.toFixed(1)}°`}
            </span>
          </div>

          <div className="p-2 rounded-lg bg-muted/20 border border-border/60 flex items-center justify-between">
            <span className="text-[10px] text-muted-foreground">ROLL (TILT)</span>
            <span className="font-bold tabular-nums text-foreground">
              {rollDeg > 0 ? `+${rollDeg.toFixed(1)}°` : `${rollDeg.toFixed(1)}°`}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
