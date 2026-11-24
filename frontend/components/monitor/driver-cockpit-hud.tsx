"use client";

import React from "react";
import Link from "next/link";
import {
  FrameTelemetry,
  explainReasonCode,
  resolveEffectiveDisplayState,
} from "@/types/telemetry";
import { CameraState } from "@/types/camera";
import { Button } from "@/components/ui/button";
import {
  ShieldCheck,
  AlertTriangle,
  Flame,
  EyeOff,
  Compass,
  Volume2,
  VolumeX,
  SlidersHorizontal,
  Car,
  Eye,
  CheckCircle2,
  Radio,
  ArrowRight,
  Clock,
  ShieldAlert,
} from "lucide-react";

interface DriverCockpitHUDProps {
  telemetry: FrameTelemetry | null;
  cameraState?: CameraState;
  isStale?: boolean;
  isCalibrated?: boolean;
  runtimeMode?: "browser" | "remote";
  sessionDurationSec?: number;
  className?: string;
  isAudioUnlocked?: boolean;
  isMuted?: boolean;
  volume?: number;
  audioStatus?: "ready" | "muted" | "blocked" | "unavailable";
  onStartMonitoring?: () => void;
  onCalibrateClick?: () => void;
  onToggleMute?: () => void;
  onSetVolume?: (volume: number) => void;
  onUnlockAudio?: () => Promise<boolean>;
  onSwitchToDiagnostics?: () => void;
}

function formatDuration(totalSeconds: number): string {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins}:${secs < 10 ? "0" : ""}${secs}`;
}

export function DriverCockpitHUD({
  telemetry,
  cameraState = "not-initialized",
  isStale = false,
  isCalibrated = true,
  runtimeMode = "browser",
  sessionDurationSec = 0,
  className = "",
  isMuted = false,
  volume = 0.75,
  audioStatus,
  onStartMonitoring,
  onCalibrateClick,
  onToggleMute,
  onSetVolume,
  onUnlockAudio,
  onSwitchToDiagnostics,
}: DriverCockpitHUDProps) {
  const isCameraActive = cameraState === "active";

  // 1. Standby / Pre-Drive View
  if (!telemetry || !isCameraActive) {
    return (
      <div className={`rounded-2xl bg-card border border-border/80 divide-y divide-border/60 ${className}`}>
        {/* Top Greeting */}
        <div className="p-5 space-y-4">
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-2">
              <Car className="size-4 text-primary" aria-hidden="true" />
              <h2 className="text-sm font-semibold text-foreground">
                Driver Safety Cockpit
              </h2>
            </div>
            <span className="inline-flex items-center rounded-full border border-border bg-muted/40 px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
              {isCameraActive ? "Starting…" : "Standby"}
            </span>
          </div>

          <div className="flex items-start gap-3.5 pt-1">
            <div className="size-11 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 text-primary">
              <Eye className="size-5" aria-hidden="true" />
            </div>
            <div className="space-y-1 min-w-0">
              <h3 className="text-base font-bold text-foreground tracking-tight text-balance">
                {isCameraActive ? "Aligning your face baseline…" : "Ready to monitor your drive"}
              </h3>
              <p className="text-xs text-muted-foreground leading-relaxed max-w-prose">
                {isCameraActive
                  ? "Look straight ahead for a couple of seconds while tracking locks onto your eyes."
                  : "Start monitoring when you're seated. DriveGuard watches for heavy eyelids and microsleeps without interrupting normal mirror checks."}
              </p>
            </div>
          </div>

          {!isCameraActive && onStartMonitoring && (
            <Button
              type="button"
              onClick={onStartMonitoring}
              className="w-full h-11 rounded-xl font-semibold text-sm"
            >
              Start Monitoring
            </Button>
          )}
        </div>

        {/* Simple Pre-Drive Steps */}
        <div className="p-5 space-y-2.5">
          <p className="text-xs font-medium text-muted-foreground">
            Quick setup before you drive
          </p>
          <div className="space-y-2 text-xs">
            <div className="flex items-center justify-between gap-2 rounded-xl bg-muted/30 px-3 py-2.5 border border-border/50">
              <div className="flex items-center gap-2.5 min-w-0">
                <Compass
                  className={`size-4 shrink-0 ${isCalibrated ? "text-emerald-500" : "text-amber-500"}`}
                  aria-hidden="true"
                />
                <div className="min-w-0">
                  <div className="font-medium text-foreground">Personal eye calibration</div>
                  <div className="text-[11px] text-muted-foreground truncate">
                    {isCalibrated
                      ? "Calibrated to your natural eye shape"
                      : "Takes 15 seconds for accurate drowsiness alerts"}
                  </div>
                </div>
              </div>
              {onCalibrateClick && (
                <Button
                  type="button"
                  size="sm"
                  variant={isCalibrated ? "ghost" : "outline"}
                  onClick={onCalibrateClick}
                  className="h-7 text-xs shrink-0"
                >
                  {isCalibrated ? "Redo" : "Calibrate"}
                </Button>
              )}
            </div>

            <div className="flex items-center justify-between gap-2 rounded-xl bg-muted/30 px-3 py-2.5 border border-border/50">
              <div className="flex items-center gap-2.5 min-w-0">
                {audioStatus === "blocked" || isMuted ? (
                  <VolumeX className="size-4 text-amber-500 shrink-0" aria-hidden="true" />
                ) : (
                  <Volume2 className="size-4 text-emerald-500 shrink-0" aria-hidden="true" />
                )}
                <div className="min-w-0">
                  <div className="font-medium text-foreground">Cabin alert sound</div>
                  <div className="text-[11px] text-muted-foreground truncate">
                    {audioStatus === "blocked"
                      ? "Browser needs one tap to allow audio"
                      : isMuted
                        ? "Currently muted — visual alerts only"
                        : `Armed at ${Math.round(volume * 100)}% volume`}
                  </div>
                </div>
              </div>
              {audioStatus === "blocked" && onUnlockAudio ? (
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={onUnlockAudio}
                  className="h-7 text-xs shrink-0 border-amber-500/40 text-amber-600 dark:text-amber-400"
                >
                  Enable
                </Button>
              ) : onToggleMute ? (
                <Button
                  type="button"
                  size="sm"
                  variant="ghost"
                  onClick={onToggleMute}
                  className="h-7 text-xs shrink-0"
                >
                  {isMuted ? "Unmute" : "Mute"}
                </Button>
              ) : null}
            </div>
          </div>
        </div>

        {/* Navigation & Mode Switcher */}
        <div className="px-5 py-3.5 flex items-center justify-between gap-2 flex-wrap bg-muted/20 rounded-b-2xl">
          <div className="flex items-center gap-3 text-xs">
            <Link
              href="/dashboard/sessions"
              className="inline-flex items-center gap-1 font-medium text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
            >
              <Clock className="size-3.5" aria-hidden="true" />
              <span>Sessions</span>
            </Link>
            <Link
              href="/dashboard/alerts"
              className="inline-flex items-center gap-1 font-medium text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
            >
              <ShieldAlert className="size-3.5" aria-hidden="true" />
              <span>Alerts</span>
            </Link>
          </div>

          {onSwitchToDiagnostics && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onSwitchToDiagnostics}
              className="text-xs text-muted-foreground hover:text-foreground gap-1.5 h-8"
            >
              <SlidersHorizontal className="size-3.5" aria-hidden="true" />
              <span>Open Inspector</span>
            </Button>
          )}
        </div>
      </div>
    );
  }

  // 2. Active Monitoring View
  const { risk_score, pose_zone, comparison, reason_codes } = telemetry;
  const effectiveState = resolveEffectiveDisplayState(telemetry, {
    isConnected: isCameraActive,
    isStale,
  });
  const isTrackingValid =
    effectiveState !== "STALE" && effectiveState !== "MONITORING_UNAVAILABLE";
  const riskPct = Math.round(risk_score * 100);
  const alertnessPct = Math.max(0, Math.min(100, 100 - riskPct));

  const getBannerState = () => {
    switch (effectiveState) {
      case "STALE":
        return {
          pillBg: "bg-violet-950/90 text-violet-200 border border-violet-400/60 border-dashed",
          icon: <Radio className="size-5 text-violet-300" aria-hidden="true" />,
          title: "Connection Stalled",
          sublabel: "No fresh data received — check your connection",
          cardBorder: "border-violet-500/40",
          alertnessColor: "text-muted-foreground",
          progressColor: "bg-violet-400",
        };
      case "CRITICAL":
        return {
          pillBg: "bg-rose-600 text-white shadow-sm",
          icon: <Flame className="size-5" aria-hidden="true" />,
          title: "Critical Fatigue — Pull Over",
          sublabel: "Prolonged eye closure detected. Stop safely to rest.",
          cardBorder: "border-rose-500/60 ring-2 ring-rose-500/20",
          alertnessColor: "text-rose-500",
          progressColor: "bg-rose-500",
        };
      case "WARNING":
        return {
          pillBg: "bg-amber-500 text-slate-950",
          icon: <AlertTriangle className="size-5" aria-hidden="true" />,
          title: "Fatigue Warning",
          sublabel: "Signs of drowsiness detected — consider taking a break.",
          cardBorder: "border-amber-500/50",
          alertnessColor: "text-amber-500",
          progressColor: "bg-amber-500",
        };
      case "ATTENTION":
        return {
          pillBg: "bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/40",
          icon: <AlertTriangle className="size-5" aria-hidden="true" />,
          title: "Early Fatigue Cue",
          sublabel: "Slight drop in alertness — stay focused on the road.",
          cardBorder: "border-amber-500/30",
          alertnessColor: "text-amber-500",
          progressColor: "bg-amber-400",
        };
      case "MONITORING_UNAVAILABLE":
        return {
          pillBg: "bg-muted text-amber-600 dark:text-amber-300 border border-amber-500/40 border-dashed",
          icon: <EyeOff className="size-5 text-amber-500" aria-hidden="true" />,
          title: "Face Not Visible",
          sublabel: "Center your face in the camera view to resume.",
          cardBorder: "border-amber-500/40",
          alertnessColor: "text-muted-foreground",
          progressColor: "bg-muted-foreground",
        };
      case "NORMAL":
      default:
        return {
          pillBg: "bg-emerald-600 text-white",
          icon: <ShieldCheck className="size-5" aria-hidden="true" />,
          title: telemetry.is_recovering ? "Alertness Recovering" : "Attentive & Focused",
          sublabel: "Eyes and head posture are within your normal range.",
          cardBorder: "border-border/80",
          alertnessColor: "text-emerald-600 dark:text-emerald-400",
          progressColor: "bg-emerald-500",
        };
    }
  };

  const banner = getBannerState();

  const getDriverAdvisory = () => {
    if (effectiveState === "STALE") {
      return "Live telemetry paused for more than 3.5 seconds. Switch to On-Device AI or retry your connection.";
    }
    if (effectiveState === "MONITORING_UNAVAILABLE") {
      return "Adjust your camera angle or lighting so your face stays clearly visible inside the frame.";
    }

    const activeReasons = (reason_codes ?? []).filter(
      (c) => c !== "ALERT_NORMAL" && c !== "MIRROR_CHECK_CONTEXT" && c !== "DASHBOARD_CHECK_CONTEXT"
    );

    if (activeReasons.length > 0) {
      const top = explainReasonCode(activeReasons[0]);
      return `${top.title}: ${top.description}`;
    }

    if (
      comparison?.is_mirror_check_suppressed ||
      reason_codes?.includes("MIRROR_CHECK_CONTEXT")
    ) {
      return "Mirror check recognized — normal glance alerts are suppressed automatically.";
    }

    if (telemetry.is_recovering) {
      return "Your alertness is recovering steadily. Keep your eyes on the road ahead.";
    }

    return "Everything looks normal. Drive safely and keep a comfortable following distance.";
  };

  const getZoneLabel = (zone: string) => {
    switch (zone) {
      case "CENTER":
        return { label: "Road Ahead", hint: "Focused", color: "text-emerald-600 dark:text-emerald-400" };
      case "LEFT_MIRROR":
        return { label: "Left Mirror", hint: "Safe check", color: "text-sky-600 dark:text-sky-400" };
      case "RIGHT_MIRROR":
        return { label: "Right Mirror", hint: "Safe check", color: "text-sky-600 dark:text-sky-400" };
      case "DASHBOARD":
        return { label: "Dashboard", hint: "Instrument check", color: "text-amber-600 dark:text-amber-400" };
      case "OFF_ROAD":
      case "OUT_OF_BOUNDS":
        return { label: "Looking Away", hint: "Eyes off road", color: "text-rose-500" };
      case "UNKNOWN":
        return { label: "Unavailable", hint: "Face lost", color: "text-muted-foreground" };
      default:
        return { label: zone, hint: "Glance", color: "text-amber-500" };
    }
  };

  const zoneDisplay = getZoneLabel(pose_zone);

  return (
    <div className={`rounded-2xl bg-card border ${banner.cardBorder} divide-y divide-border/60 ${isStale ? "opacity-80" : ""} ${className}`}>
      {/* 1. Primary Driver Status & Alertness Gauge */}
      <div className="p-5 space-y-4">
        <div className="flex items-start justify-between gap-3 flex-wrap">
          <div className="flex items-center gap-3 min-w-0">
            <div className={`p-2.5 rounded-xl shrink-0 ${banner.pillBg}`}>
              {banner.icon}
            </div>
            <div className="min-w-0">
              <h2 className="text-base sm:text-lg font-bold text-foreground tracking-tight text-balance">
                {banner.title}
              </h2>
              <p className="text-xs text-muted-foreground">{banner.sublabel}</p>
            </div>
          </div>

          <div className="text-right shrink-0">
            <div className="text-[11px] font-medium text-muted-foreground">Alertness</div>
            <div className={`text-xl font-bold font-mono tabular-nums ${banner.alertnessColor}`}>
              {isTrackingValid ? `${alertnessPct}%` : "—"}
            </div>
          </div>
        </div>

        {/* Clean Alertness Bar */}
        <div
          className="h-2 w-full bg-muted rounded-full overflow-hidden"
          role="progressbar"
          aria-label="Driver alertness score"
          aria-valuenow={isTrackingValid ? alertnessPct : 0}
          aria-valuemin={0}
          aria-valuemax={100}
        >
          <div
            className={`h-full transition-transform duration-300 origin-left rounded-full ${banner.progressColor}`}
            style={{ width: `${isTrackingValid ? alertnessPct : 0}%` }}
          />
        </div>

        {/* Plain-Language Advisory */}
        <div className="p-3.5 rounded-xl bg-muted/35 border border-border/60 flex items-start gap-2.5 text-xs">
          <CheckCircle2 className="size-4 text-primary shrink-0 mt-0.5" aria-hidden="true" />
          <p className="text-foreground/90 leading-relaxed">{getDriverAdvisory()}</p>
        </div>

        {/* Calibration Reminder only when uncalibrated */}
        {!isCalibrated && onCalibrateClick && (
          <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/35 flex items-center justify-between gap-2 text-xs">
            <span className="text-foreground">
              Calibrate your eyes for personalized fatigue thresholds.
            </span>
            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={onCalibrateClick}
              className="h-7 text-xs shrink-0 border-amber-500/40"
            >
              Calibrate Now
            </Button>
          </div>
        )}
      </div>

      {/* 2. Gaze Target & Audio Controls */}
      <div className="p-5 grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
        <div className="p-3 rounded-xl bg-muted/30 border border-border/60 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            <Compass className="size-4 text-primary shrink-0" aria-hidden="true" />
            <div className="min-w-0">
              <div className="text-[11px] text-muted-foreground">Gaze direction</div>
              <div className="font-semibold text-foreground truncate">{zoneDisplay.label}</div>
            </div>
          </div>
          <span className={`text-[11px] font-medium shrink-0 ${zoneDisplay.color}`}>
            {zoneDisplay.hint}
          </span>
        </div>

        <div className="p-3 rounded-xl bg-muted/30 border border-border/60 flex items-center justify-between gap-2">
          <div className="flex items-center gap-2.5 min-w-0">
            {audioStatus === "blocked" ? (
              <VolumeX className="size-4 text-amber-500 shrink-0" aria-hidden="true" />
            ) : isMuted ? (
              <VolumeX className="size-4 text-rose-500 shrink-0" aria-hidden="true" />
            ) : (
              <Volume2 className="size-4 text-emerald-500 shrink-0" aria-hidden="true" />
            )}
            <div className="min-w-0">
              <div className="text-[11px] text-muted-foreground">Alert audio</div>
              <div className="font-semibold text-foreground font-mono tabular-nums truncate">
                {audioStatus === "blocked"
                  ? "Blocked"
                  : isMuted
                    ? "Muted"
                    : `${Math.round(volume * 100)}%`}
              </div>
            </div>
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            {audioStatus === "blocked" && onUnlockAudio ? (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={onUnlockAudio}
                className="h-7 text-xs border-amber-500/40 text-amber-600 dark:text-amber-400"
              >
                Enable
              </Button>
            ) : onToggleMute ? (
              <Button
                type="button"
                size="sm"
                variant={isMuted ? "default" : "outline"}
                onClick={onToggleMute}
                className="h-7 text-xs"
              >
                {isMuted ? "Unmute" : "Mute"}
              </Button>
            ) : null}
          </div>
        </div>

        {onSetVolume && !isMuted && audioStatus === "ready" && (
          <div className="sm:col-span-2 flex items-center justify-between gap-3 px-1">
            <label htmlFor="driver-alert-volume" className="text-[11px] text-muted-foreground">
              Alert chime volume
            </label>
            <div className="flex items-center gap-2">
              <input
                id="driver-alert-volume"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={volume}
                onChange={(e) => onSetVolume(parseFloat(e.target.value))}
                className="w-28 accent-primary cursor-pointer"
              />
              <span className="text-[11px] font-mono tabular-nums text-muted-foreground w-8 text-right">
                {Math.round(volume * 100)}%
              </span>
            </div>
          </div>
        )}
      </div>

      {/* 3. Quiet Session Meta & Inspector Jump */}
      <div className="px-5 py-3 flex items-center justify-between gap-2 flex-wrap bg-muted/20 rounded-b-2xl text-xs">
        <div className="flex items-center gap-3 text-muted-foreground font-mono tabular-nums">
          <span>{formatDuration(sessionDurationSec)}</span>
          <span aria-hidden="true">·</span>
          <span className="font-sans">
            {runtimeMode === "browser" ? "On-device AI" : "Remote AI"}
          </span>
        </div>

        <div className="flex items-center gap-3">
          <Link
            href="/dashboard/sessions"
            className="inline-flex items-center gap-1 text-muted-foreground hover:text-foreground transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
          >
            <span>Sessions</span>
            <ArrowRight className="size-3" aria-hidden="true" />
          </Link>
          {onSwitchToDiagnostics && (
            <Button
              type="button"
              variant="ghost"
              size="sm"
              onClick={onSwitchToDiagnostics}
              className="text-xs text-primary hover:text-primary/90 gap-1.5 h-7 px-2"
            >
              <SlidersHorizontal className="size-3.5" aria-hidden="true" />
              <span>Inspector</span>
            </Button>
          )}
        </div>
      </div>
    </div>
  );
}
