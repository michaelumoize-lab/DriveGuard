"use client";

import React from "react";
import { FrameTelemetry } from "@/types/telemetry";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Activity,
  AlertTriangle,
  Compass,
  Cpu,
  Eye,
  Flame,
  Gauge,
  Sparkles,
  Zap,
} from "lucide-react";

interface LiveTelemetryHUDProps {
  telemetry: FrameTelemetry | null;
  className?: string;
}

export function LiveTelemetryHUD({ telemetry, className = "" }: LiveTelemetryHUDProps) {
  if (!telemetry) {
    return (
      <div className={`p-4 rounded-2xl bg-card border border-border text-center text-xs text-muted-foreground ${className}`}>
        <Activity className="size-5 mx-auto mb-2 text-muted-foreground/60 animate-pulse" />
        <p>Awaiting live driver telemetry packet...</p>
      </div>
    );
  }

  const {
    state,
    risk_score,
    pose_zone,
    fps,
    inference_latency_ms,
    network_roundtrip_ms,
    eye_metrics,
    head_pose,
    cascade,
  } = telemetry;

  const ear = eye_metrics?.ear_raw ?? 0;
  const threshold = eye_metrics?.ear_closure_threshold ?? 0.22;
  const isEyeClosed = eye_metrics?.is_eye_closed ?? false;
  const perclos = eye_metrics?.perclos_pct ?? 0;
  const blinkDur = eye_metrics?.blink_duration_ms;
  const blinkRcr = eye_metrics?.blink_rcr;
  const blinkRate = eye_metrics?.blink_rate_per_min ?? 0;

  // Determine state banner colors
  const getStateColor = () => {
    switch (state) {
      case "CRITICAL":
        return "bg-rose-500 text-white animate-pulse";
      case "WARNING":
        return "bg-amber-500 text-black";
      case "NORMAL":
        return "bg-emerald-500 text-white";
      default:
        return "bg-muted text-muted-foreground";
    }
  };

  return (
    <div className={`space-y-3 ${className}`}>
      {/* Top Cockpit Telemetry Strip: State & Risk Score */}
      <div className="flex items-center justify-between p-3 rounded-2xl bg-card border border-border shadow-xs">
        <div className="flex items-center gap-2">
          <div className={`px-3 py-1 rounded-full text-xs font-black uppercase tracking-wider ${getStateColor()}`}>
            {state}
          </div>
          <span className="text-xs text-muted-foreground font-mono">
            Risk: <strong>{(risk_score * 100).toFixed(0)}%</strong>
          </span>
        </div>

        <div className="flex items-center gap-2 text-xs font-mono">
          <Badge variant="outline" className="border-border text-[11px] gap-1">
            <Gauge className="size-3 text-primary" />
            <span>{fps.toFixed(1)} FPS</span>
          </Badge>
          {network_roundtrip_ms !== undefined && (
            <Badge variant="outline" className="border-border text-[11px] text-muted-foreground">
              RTT: {network_roundtrip_ms}ms
            </Badge>
          )}
        </div>
      </div>

      {/* 2-Column Gauge Grid: EAR on Left, PERCLOS on Right */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {/* EAR Gauge Box */}
        <div className="p-3.5 rounded-2xl bg-card border border-border space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold flex items-center gap-1.5 text-foreground">
              <Eye className="size-3.5 text-primary" />
              <span>Eye Aspect Ratio (EAR)</span>
            </span>
            <span
              className={`text-xs font-mono font-bold px-2 py-0.5 rounded-md ${
                isEyeClosed ? "bg-rose-500/20 text-rose-400" : "bg-emerald-500/20 text-emerald-400"
              }`}
            >
              {ear.toFixed(3)}
            </span>
          </div>

          {/* Visual EAR Bar with Cutoff Marker */}
          <div className="space-y-1">
            <div className="relative h-3 w-full bg-muted/60 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-100 rounded-full ${
                  isEyeClosed ? "bg-rose-500" : "bg-emerald-500"
                }`}
                style={{ width: `${Math.min(100, Math.max(0, (ear / 0.45) * 100))}%` }}
              />
              {/* Dynamic Threshold Cutoff Line */}
              <div
                className="absolute top-0 bottom-0 w-1 bg-amber-400 shadow-xs z-10"
                style={{ left: `${Math.min(95, Math.max(5, (threshold / 0.45) * 100))}%` }}
                title={`Closure Cutoff: ${threshold.toFixed(3)}`}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
              <span>Closed (&lt;0.18)</span>
              <span className="text-amber-400">Cutoff ({threshold.toFixed(3)})</span>
              <span>Open (&gt;0.30)</span>
            </div>
          </div>
        </div>

        {/* PERCLOS (60s Window) Gauge Box */}
        <div className="p-3.5 rounded-2xl bg-card border border-border space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold flex items-center gap-1.5 text-foreground">
              <Activity className="size-3.5 text-primary" />
              <span>PERCLOS (60s Window)</span>
            </span>
            <span
              className={`text-xs font-mono font-bold px-2 py-0.5 rounded-md ${
                perclos > 25
                  ? "bg-rose-500/20 text-rose-400"
                  : perclos > 15
                  ? "bg-amber-500/20 text-amber-400"
                  : "bg-emerald-500/20 text-emerald-400"
              }`}
            >
              {perclos.toFixed(1)}%
            </span>
          </div>

          {/* Visual PERCLOS Bar */}
          <div className="space-y-1">
            <div className="relative h-3 w-full bg-muted/60 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-150 rounded-full ${
                  perclos > 25 ? "bg-rose-500" : perclos > 15 ? "bg-amber-500" : "bg-emerald-500"
                }`}
                style={{ width: `${Math.min(100, Math.max(0, perclos))}%` }}
              />
            </div>
            <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
              <span className="text-emerald-400">Alert (&lt;8%)</span>
              <span className="text-amber-400">Drowsy (15%)</span>
              <span className="text-rose-400">Critical (&gt;25%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* 3D Gaze Zone & Sub-frame Kinematics */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
        {/* Active Zone */}
        <div className="p-2.5 rounded-xl bg-card border border-border">
          <div className="text-[10px] text-muted-foreground uppercase flex items-center gap-1">
            <Compass className="size-3 text-primary" />
            <span>3D Gaze Zone</span>
          </div>
          <div className="font-bold text-foreground mt-0.5 truncate">{pose_zone}</div>
        </div>

        {/* Head Pose Pitch / Yaw */}
        <div className="p-2.5 rounded-xl bg-card border border-border">
          <div className="text-[10px] text-muted-foreground uppercase">Head Pose</div>
          <div className="font-bold text-foreground mt-0.5">
            P: {head_pose ? Math.round(head_pose.pitch_deg) : 0}° | Y: {head_pose ? Math.round(head_pose.yaw_deg) : 0}°
          </div>
        </div>

        {/* Blink Rate */}
        <div className="p-2.5 rounded-xl bg-card border border-border">
          <div className="text-[10px] text-muted-foreground uppercase">Blink Frequency</div>
          <div className="font-bold text-foreground mt-0.5">
            {blinkRate > 0 ? `${blinkRate.toFixed(1)} /min` : "--"}
          </div>
        </div>

        {/* Kinematic RCR */}
        <div className="p-2.5 rounded-xl bg-card border border-border">
          <div className="text-[10px] text-muted-foreground uppercase">Kinematic RCR</div>
          <div className="font-bold text-foreground mt-0.5">
            {blinkRcr ? blinkRcr.toFixed(2) : blinkDur ? `${Math.round(blinkDur)}ms` : "N/A"}
          </div>
        </div>
      </div>

      {/* Confidence Cascade Architecture Status */}
      <div className="p-3 rounded-2xl bg-card border border-border flex items-center justify-between text-xs">
        <div className="flex items-center gap-2">
          <Cpu className="size-4 text-primary shrink-0" />
          <span className="text-muted-foreground">Cascade Decision Authority:</span>
          <Badge variant="secondary" className="font-mono text-[11px] font-bold">
            {cascade?.arbitration_winner || "GEOMETRY"}
          </Badge>
        </div>

        <div className="flex items-center gap-1.5">
          {cascade?.cnn_triggered ? (
            <Badge className="bg-primary/20 text-primary border-primary/40 text-[10px] font-mono animate-pulse">
              <Zap className="size-3 mr-1 fill-primary" />
              CNN ACTIVE
            </Badge>
          ) : (
            <Badge variant="outline" className="border-border text-muted-foreground text-[10px] font-mono">
              GEOMETRIC FAST-PATH
            </Badge>
          )}
        </div>
      </div>
    </div>
  );
}
