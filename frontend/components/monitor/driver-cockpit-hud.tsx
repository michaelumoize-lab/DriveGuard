"use client";

import React from "react";
import { FrameTelemetry } from "@/types/telemetry";
import { CameraState } from "@/types/camera";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  ShieldCheck,
  AlertTriangle,
  Flame,
  EyeOff,
  Compass,
  Volume2,
  VolumeX,
  Volume1,
  SlidersHorizontal,
  Car,
  Eye,
  CheckCircle2,
  Radio,
  Zap,
} from "lucide-react";

interface DriverCockpitHUDProps {
  telemetry: FrameTelemetry | null;
  cameraState?: CameraState;
  isStale?: boolean;
  className?: string;
  isAudioUnlocked?: boolean;
  isMuted?: boolean;
  volume?: number;
  audioStatus?: "ready" | "muted" | "blocked" | "unavailable";
  onToggleMute?: () => void;
  onSetVolume?: (volume: number) => void;
  onUnlockAudio?: () => Promise<boolean>;
  onSwitchToDiagnostics?: () => void;
}

export function DriverCockpitHUD({
  telemetry,
  cameraState = "not-initialized",
  isStale = false,
  className = "",
  isAudioUnlocked = false,
  isMuted = false,
  volume = 0.75,
  audioStatus,
  onToggleMute,
  onSetVolume,
  onUnlockAudio,
  onSwitchToDiagnostics,
}: DriverCockpitHUDProps) {
  const isCameraActive = cameraState === "active";

  // Standby or waiting for camera
  if (!telemetry || !isCameraActive) {
    return (
      <div className={`p-5 rounded-2xl bg-card border border-border/80 shadow-xs space-y-4 ${className}`}>
        <div className="flex items-center justify-between border-b border-border/60 pb-3">
          <div className="flex items-center gap-2">
            <Car className="size-4 text-primary" />
            <span className="text-xs font-bold uppercase tracking-wider text-foreground">
              Driver Safety Cockpit
            </span>
          </div>
          <Badge variant="outline" className="text-[10px] font-mono border-border text-muted-foreground">
            {isCameraActive ? "INITIALIZING CAM" : "STANDBY"}
          </Badge>
        </div>

        <div className="py-6 text-center space-y-3">
          <div className="size-14 rounded-2xl bg-primary/10 border border-primary/20 flex items-center justify-center mx-auto text-primary">
            <Eye className="size-7" />
          </div>
          <div className="space-y-1 max-w-sm mx-auto">
            <h3 className="text-sm font-bold text-foreground">
              {isCameraActive ? "Aligning Facial Baseline…" : "Driver Safety Guard Offline"}
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {isCameraActive
                ? "Calibrating 3D head pose and baseline eyelid dynamics. Look straight ahead at the roadway."
                : "Start the camera stream to activate real-time fatigue monitoring, mirror-glance suppression, and audio alarms."}
            </p>
          </div>
        </div>

        {onSwitchToDiagnostics && (
          <div className="pt-2 border-t border-border/50 flex justify-end">
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onSwitchToDiagnostics}
              className="text-xs text-muted-foreground hover:text-foreground gap-1.5 h-8"
              aria-label="Switch to engineering diagnostics view"
            >
              <SlidersHorizontal className="size-3.5" />
              <span>Inspect Raw Engineering Diagnostics</span>
            </Button>
          </div>
        )}
      </div>
    );
  }

  const { state, risk_score, pose_zone, eye_metrics, comparison, reason_codes } = telemetry;
  const riskPct = Math.round(risk_score * 100);
  const alertnessPct = Math.max(0, Math.min(100, 100 - riskPct));

  // Visual status banner styling
  const getBannerState = () => {
    switch (state) {
      case "CRITICAL":
        return {
          pillBg: "bg-rose-600 text-white shadow-lg shadow-rose-600/30",
          icon: <Flame className="size-5 animate-pulse" />,
          title: "CRITICAL FATIGUE WARNING",
          sublabel: "Pull over immediately — severe eye closure detected",
          cardBorder: "border-rose-500/50 ring-2 ring-rose-500/20",
          alertnessColor: "text-rose-500",
          progressColor: "bg-rose-500",
        };
      case "WARNING":
        return {
          pillBg: "bg-amber-500 text-slate-950 shadow-md shadow-amber-500/20",
          icon: <AlertTriangle className="size-5" />,
          title: "FATIGUE ADVISORY",
          sublabel: "Drowsiness pattern detected — rest recommended",
          cardBorder: "border-amber-500/40 ring-1 ring-amber-500/20",
          alertnessColor: "text-amber-500",
          progressColor: "bg-amber-500",
        };
      case "MONITORING_UNAVAILABLE":
        return {
          pillBg: "bg-slate-900 text-amber-300 border border-amber-400/80 border-dashed",
          icon: <EyeOff className="size-5 text-amber-400" />,
          title: "FACE TRACKING UNAVAILABLE",
          sublabel: "Face obstructed or out of camera frame",
          cardBorder: "border-amber-500/40 border-dashed",
          alertnessColor: "text-muted-foreground",
          progressColor: "bg-muted-foreground",
        };
      case "NORMAL":
      default:
        return {
          pillBg: "bg-emerald-600 text-white shadow-md shadow-emerald-600/20",
          icon: <ShieldCheck className="size-5" />,
          title: "ATTENTIVE & ROAD FOCUSED",
          sublabel: "Eyelid dynamics and head posture within safe baseline",
          cardBorder: "border-border",
          alertnessColor: "text-emerald-500",
          progressColor: "bg-emerald-500",
        };
    }
  };

  const banner = getBannerState();

  // Natural human-readable driver advisory
  const getDriverAdvisory = () => {
    if (state === "CRITICAL") {
      if (reason_codes?.includes("MICRO_SLEEP") || reason_codes?.includes("MICROSLEEP")) {
        return "Critical hazard: Microsleep episode detected! Sounding audio warning. Bring vehicle to a safe stop.";
      }
      return "Prolonged eyelid closure detected. Immediate rest required before continuing to drive.";
    }

    if (state === "WARNING") {
      if (reason_codes?.includes("HIGH_PERCLOS")) {
        return "Cumulative eye closure rate is elevated. Your alertness is declining — plan a rest stop.";
      }
      return "Drowsiness indicators detected. Cabin siren chime armed.";
    }

    if (state === "MONITORING_UNAVAILABLE") {
      return "Align your face with the camera viewport. Avoid bright direct glare or obstructing the lens.";
    }

    if (comparison?.is_mirror_check_suppressed) {
      return "Natural mirror check recognized (under 3.0s). False alarm suppressed.";
    }

    return "Driver awareness nominal. Maintain safe following distance and speed.";
  };

  const getZoneLabel = (zone: string) => {
    switch (zone) {
      case "CENTER":
        return { label: "Road Center Ahead", hint: "Safe Road Focus", color: "text-emerald-500" };
      case "LEFT_MIRROR":
        return { label: "Left Side Mirror", hint: "Side Check (Safe)", color: "text-blue-400" };
      case "RIGHT_MIRROR":
        return { label: "Right Side Mirror", hint: "Side Check (Safe)", color: "text-amber-400" };
      case "DASHBOARD":
        return { label: "Dashboard Cluster", hint: "Speed/Instrument Check", color: "text-purple-400" };
      default:
        return { label: zone, hint: "Averted Gaze", color: "text-rose-400" };
    }
  };

  const zoneDisplay = getZoneLabel(pose_zone);

  return (
    <div className={`space-y-3.5 ${isStale ? "opacity-75" : ""} ${className}`}>
      {/* 1. Primary Driver State Card */}
      <div className={`p-4 sm:p-5 rounded-2xl bg-card border ${banner.cardBorder} shadow-sm space-y-4`}>
        {/* Top Status & Alertness Gauge Row */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-xl ${banner.pillBg} shrink-0`}>
              {banner.icon}
            </div>
            <div>
              <h2 className="text-base sm:text-lg font-extrabold text-foreground tracking-tight">
                {banner.title}
              </h2>
              <p className="text-xs text-muted-foreground">{banner.sublabel}</p>
            </div>
          </div>

          {/* Alertness Readiness Badge */}
          <div className="inline-flex items-center gap-3 px-3 py-1.5 rounded-xl bg-muted/40 border border-border/80 self-start sm:self-center">
            <div className="text-left">
              <span className="text-[10px] uppercase font-mono text-muted-foreground block">
                Driver Alertness
              </span>
              <span className={`text-base font-extrabold font-mono tabular-nums ${banner.alertnessColor}`}>
                {alertnessPct}%
              </span>
            </div>
            <div className="w-16 h-2 bg-muted/80 rounded-full overflow-hidden">
              <div
                className={`h-full transition-all duration-300 rounded-full ${banner.progressColor}`}
                style={{ width: `${alertnessPct}%` }}
              />
            </div>
          </div>
        </div>

        {/* Actionable Human-Centric Driver Advisory */}
        <div className="p-3 rounded-xl bg-muted/30 border border-border/60 flex items-start gap-2.5 text-xs">
          <CheckCircle2 className="size-4 text-primary shrink-0 mt-0.5" />
          <div className="space-y-0.5 min-w-0">
            <span className="font-semibold text-foreground">Safety Advisory:</span>
            <p className="text-muted-foreground leading-relaxed">{getDriverAdvisory()}</p>
          </div>
        </div>

        {/* Simplified In-Cabin Spatial & Audio Strip */}
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2.5 pt-1 text-xs">
          {/* Cabin Gaze Zone */}
          <div className="p-2.5 rounded-xl bg-background border border-border flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Compass className="size-4 text-primary shrink-0" />
              <div>
                <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                  Cabin Gaze Target
                </span>
                <span className="font-semibold text-foreground text-xs">{zoneDisplay.label}</span>
              </div>
            </div>
            <span className={`text-[11px] font-medium ${zoneDisplay.color}`}>{zoneDisplay.hint}</span>
          </div>

          {/* Audio Chime Control */}
          <div className="p-2.5 rounded-xl bg-background border border-border flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              {audioStatus === "blocked" ? (
                <VolumeX className="size-4 text-amber-500 shrink-0" />
              ) : isMuted ? (
                <VolumeX className="size-4 text-rose-500 shrink-0" />
              ) : (
                <Volume2 className="size-4 text-emerald-500 shrink-0" />
              )}
              <div>
                <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                  Cabin Audio Siren
                </span>
                <span className="font-semibold text-foreground text-xs">
                  {audioStatus === "blocked"
                    ? "Click to Unlock"
                    : isMuted
                    ? "Muted"
                    : `Active (${Math.round(volume * 100)}%)`}
                </span>
              </div>
            </div>

            <div className="flex items-center gap-1.5">
              {audioStatus === "blocked" && onUnlockAudio ? (
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onUnlockAudio}
                  className="h-7 text-xs border-amber-500/40 text-amber-500 hover:bg-amber-500/10"
                  aria-label="Unlock browser audio autoplay"
                >
                  Enable Audio
                </Button>
              ) : onToggleMute ? (
                <Button
                  size="sm"
                  variant={isMuted ? "default" : "outline"}
                  onClick={onToggleMute}
                  className="h-7 text-xs"
                  aria-label={isMuted ? "Unmute cabin audio alerts" : "Mute cabin audio alerts"}
                >
                  {isMuted ? "Unmute" : "Mute"}
                </Button>
              ) : null}
            </div>
          </div>
        </div>

        {/* Footer: One-click jump to full engineering diagnostics */}
        {onSwitchToDiagnostics && (
          <div className="pt-2 border-t border-border/50 flex items-center justify-between gap-2 flex-wrap">
            <span className="text-[11px] text-muted-foreground">
              Driver Cockpit Mode active • Uncluttered in-vehicle layout
            </span>
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onSwitchToDiagnostics}
              className="text-xs text-primary hover:text-primary/90 gap-1.5 h-8 font-medium cursor-pointer"
              aria-label="Expand raw engineering telemetry diagnostics inspector"
            >
              <SlidersHorizontal className="size-3.5" />
              <span>Show Engineering Diagnostics Inspector</span>
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
