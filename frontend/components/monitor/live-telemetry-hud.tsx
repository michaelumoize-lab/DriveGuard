"use client";

import React from "react";
import {
  FrameTelemetry,
  LatencyDiagnostics,
  resolveEffectiveDisplayState,
} from "@/types/telemetry";
import { CameraState } from "@/types/camera";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Activity,
  AlertTriangle,
  Camera,
  Car,
  Compass,
  Cpu,
  Crosshair,
  Eye,
  EyeOff,
  Gauge,
  Radio,
  ShieldAlert,
  ShieldCheck,
  Timer,
  Volume1,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
} from "lucide-react";

interface LiveTelemetryHUDProps {
  telemetry: FrameTelemetry | null;
  cameraState?: CameraState;
  backendType?: "cloud" | "local";
  latencyDiagnostics?: LatencyDiagnostics | null;
  isStale?: boolean;
  className?: string;
  isAudioUnlocked?: boolean;
  isMuted?: boolean;
  volume?: number;
  audioStatus?: "ready" | "muted" | "blocked" | "unavailable";
  onToggleMute?: () => void;
  onSetVolume?: (volume: number) => void;
  onUnlockAudio?: () => Promise<boolean>;
  onTestAudioTier?: (tier: "WARNING" | "DROWSY" | "CRITICAL" | "RECOVERY") => void;
  onSwitchToCockpit?: () => void;
  showEngineStrip?: boolean;
  showComparisonStrip?: boolean;
  showAudioTestBar?: boolean;
}

export function LiveTelemetryHUD({
  telemetry,
  cameraState = "not-initialized",
  backendType = "local",
  latencyDiagnostics = null,
  isStale = false,
  className = "",
  isAudioUnlocked = false,
  isMuted = false,
  volume = 0.75,
  audioStatus,
  onToggleMute,
  onSetVolume,
  onUnlockAudio,
  onTestAudioTier,
  onSwitchToCockpit,
  showEngineStrip = true,
  showComparisonStrip = true,
  showAudioTestBar = true,
}: LiveTelemetryHUDProps) {
  const isCameraActive = cameraState === "active";

  // 1. Standby / Connecting State (never display stale readings once camera is stopped/offline)
  if (!telemetry || cameraState === "stopped" || cameraState === "not-initialized") {
    return (
      <div
        className={`p-5 rounded-2xl bg-card/90 backdrop-blur-md border border-border/80 shadow-md space-y-4 ${className}`}
      >
        <div className="flex items-center justify-between border-b border-border/60 pb-3 flex-wrap gap-2">
          <div className="flex items-center gap-2">
            <Radio className="size-4 text-primary animate-pulse" />
            <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">
              Engineering Telemetry HUD
            </h3>
          </div>
          <div className="flex items-center gap-2">
            {onSwitchToCockpit && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onSwitchToCockpit}
                className="h-7 text-xs text-muted-foreground hover:text-foreground gap-1 px-2 cursor-pointer"
                title="Switch to driver safety cockpit view"
                aria-label="Switch to driver safety cockpit view"
              >
                <Car className="size-3.5 text-primary" />
                <span>Driver View</span>
              </Button>
            )}
            {audioStatus && (
              <Badge
                variant="outline"
                className={`text-[10px] font-mono border-border ${
                  audioStatus === "ready"
                    ? "text-emerald-500 border-emerald-500/30"
                    : audioStatus === "blocked"
                    ? "text-amber-500 border-amber-500/30"
                    : audioStatus === "muted"
                    ? "text-rose-400 border-rose-500/30"
                    : "text-muted-foreground"
                }`}
              >
                {audioStatus === "ready"
                  ? "AUDIO READY"
                  : audioStatus === "blocked"
                  ? "AUDIO BLOCKED"
                  : audioStatus === "muted"
                  ? "MUTED"
                  : "NO AUDIO"}
              </Badge>
            )}
            <Badge
              variant="outline"
              className="text-[10px] font-mono border-border text-muted-foreground"
            >
              {isCameraActive ? "INITIALIZING" : "STANDBY"}
            </Badge>
          </div>
        </div>

        {isCameraActive ? (
          <div className="py-4 text-center space-y-3">
            <div className="relative size-12 mx-auto">
              <div className="size-12 rounded-full border-2 border-primary/20 border-t-primary animate-spin" />
              <Activity className="size-5 text-primary absolute inset-0 m-auto animate-pulse" />
            </div>
            <div className="space-y-1">
              <p className="text-xs font-semibold text-foreground">
                Synthesizing Facial Landmarks &amp; Gaze Vector…
              </p>
              <p className="text-[11px] text-muted-foreground max-w-xs mx-auto">
                Connected to {backendType === "local" ? "Local AI Engine (127.0.0.1:8000)" : "DriveGuard LiveKit Engine"}. Awaiting first telemetry frame.
              </p>
            </div>
          </div>
        ) : (
          <div className="py-4 text-center space-y-3">
            <div className="size-12 rounded-2xl bg-muted/40 border border-border flex items-center justify-center mx-auto text-muted-foreground/70">
              <Camera className="size-6" />
            </div>
            <div className="space-y-1">
              <p className="text-xs font-semibold text-foreground">
                Driver Telemetry Offline
              </p>
              <p className="text-[11px] text-muted-foreground max-w-xs mx-auto">
                Click &quot;Start Camera&quot; to begin sub-frame eye aspect ratio (EAR), PERCLOS, 3D gaze tracking, and cascade AI arbitration.
              </p>
            </div>
          </div>
        )}

        {/* Standby Baseline Spec Preview */}
        <div className="grid grid-cols-3 gap-2 pt-1 border-t border-border/40 text-center font-mono">
          <div className="p-2 rounded-xl bg-muted/30 border border-border/50">
            <div className="text-[9px] uppercase text-muted-foreground">EAR Cutoff</div>
            <div className="text-xs font-bold text-foreground mt-0.5">0.220</div>
          </div>
          <div className="p-2 rounded-xl bg-muted/30 border border-border/50">
            <div className="text-[9px] uppercase text-muted-foreground">PERCLOS Win</div>
            <div className="text-xs font-bold text-foreground mt-0.5">60s</div>
          </div>
          <div className="p-2 rounded-xl bg-muted/30 border border-border/50">
            <div className="text-[9px] uppercase text-muted-foreground">Cascade Mode</div>
            <div className="text-xs font-bold text-foreground mt-0.5">Adaptive</div>
          </div>
        </div>
      </div>
    );
  }

  // 2. Active Telemetry Unpack
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
    reason_codes,
  } = telemetry;

  const ear = eye_metrics?.ear_raw ?? 0;
  const earLeft = eye_metrics?.ear_left;
  const earRight = eye_metrics?.ear_right;
  const threshold = eye_metrics?.ear_closure_threshold ?? 0.22;
  const isEyeClosed = eye_metrics?.is_eye_closed ?? false;
  const perclos = eye_metrics?.perclos_pct ?? 0;
  const marYawn = eye_metrics?.mar_yawn ?? 0;
  const blinkDur = eye_metrics?.blink_duration_ms;
  const blinkRcr = eye_metrics?.blink_rcr;
  const blinkRate = eye_metrics?.blink_rate_per_min ?? 0;

  const effectiveState = resolveEffectiveDisplayState(telemetry, {
    isConnected: isCameraActive,
    isStale,
  });
  const hasLiveFace =
    telemetry.face_detected &&
    effectiveState !== "STALE" &&
    effectiveState !== "MONITORING_UNAVAILABLE";

  // Determine state banner visuals (never shows green ATTENTIVE when stale or unavailable)
  const getStateVisuals = () => {
    switch (effectiveState) {
      case "STALE":
        return {
          pillBg: "bg-purple-900/90 text-purple-200 border border-dashed border-purple-400/70",
          icon: <WifiOff className="size-4 text-purple-300" />,
          label: "TELEMETRY STALE",
          sublabel: "No recent backend frames",
          cardBorder: "border-purple-500/40 border-dashed",
          riskColor: "text-muted-foreground",
        };
      case "CRITICAL":
        return {
          pillBg: "bg-rose-500 text-white shadow-md shadow-rose-500/25",
          icon: <ShieldAlert className="size-4" aria-hidden="true" />,
          label: "CRITICAL FATIGUE",
          sublabel: "Immediate rest required",
          cardBorder: "border-rose-500/40 ring-1 ring-rose-500/20",
          riskColor: "text-rose-500",
        };
      case "WARNING":
        return {
          pillBg: "bg-amber-500 text-black shadow-md shadow-amber-500/20",
          icon: <AlertTriangle className="size-4" />,
          label: "DROWSY / WARNING",
          sublabel: "Elevated drowsiness indicators",
          cardBorder: "border-amber-500/40 ring-1 ring-amber-500/20",
          riskColor: "text-amber-500",
        };
      case "ATTENTION":
        return {
          pillBg: "bg-amber-500/20 text-amber-500 border border-amber-500/40",
          icon: <AlertTriangle className="size-4" />,
          label: "ATTENTION / EARLY SIGNAL",
          sublabel: "Early fatigue cue detected",
          cardBorder: "border-amber-500/30",
          riskColor: "text-amber-500",
        };
      case "NORMAL":
        return {
          pillBg: "bg-emerald-500 text-white shadow-md shadow-emerald-500/20",
          icon: <ShieldCheck className="size-4" />,
          label: telemetry.is_recovering ? "ATTENTIVE (RECOVERING)" : "ATTENTIVE & SAFE",
          sublabel: "Normal eyelid & gaze profile",
          cardBorder: "border-border",
          riskColor: "text-emerald-500",
        };
      case "MONITORING_UNAVAILABLE":
        return {
          pillBg: "bg-slate-800 text-amber-300 border border-dashed border-amber-400/70",
          icon: <EyeOff className="size-4 text-amber-400" />,
          label: "MONITORING UNAVAILABLE",
          sublabel: "Face lost or sensor degraded",
          cardBorder: "border-amber-500/40 border-dashed",
          riskColor: "text-muted-foreground",
        };
      default:
        return {
          pillBg: "bg-muted text-muted-foreground",
          icon: <EyeOff className="size-4" />,
          label: state || "UNKNOWN",
          sublabel: "Facial detection degraded",
          cardBorder: "border-border",
          riskColor: "text-muted-foreground",
        };
    }
  };

  const stateStyle = getStateVisuals();
  const riskPct = Math.round(risk_score * 100);

  // Authoritative 3D Zone display
  const getZoneDisplay = (zone: string) => {
    switch (zone) {
      case "CENTER":
        return { label: "Center Ahead", hint: "Road Focus", status: "safe" };
      case "LEFT_MIRROR":
        return { label: "Left Mirror", hint: "Side Check", status: "caution" };
      case "RIGHT_MIRROR":
        return { label: "Right Mirror", hint: "Side Check", status: "caution" };
      case "DASHBOARD":
        return { label: "Dashboard", hint: "Cluster Check", status: "caution" };
      case "OFF_ROAD":
      case "OUT_OF_BOUNDS":
        return { label: "Off Road", hint: "Averted Gaze", status: "critical" };
      case "UNKNOWN":
        return { label: "Tracking Unavailable", hint: "Face tracking lost", status: "caution" };
      default:
        return { label: zone, hint: "Averted Gaze", status: "critical" };
    }
  };

  const zoneInfo = getZoneDisplay(pose_zone);

  return (
    <div className={`space-y-3 transition-opacity ${isStale ? "opacity-75" : ""} ${className}`}>
      {isStale && (
        <div
          role="alert"
          className="px-3 py-2 rounded-xl border border-dashed border-purple-500/50 bg-purple-500/10 text-purple-600 dark:text-purple-300 text-xs flex items-center justify-between gap-2"
        >
          <div className="flex items-center gap-2 font-medium">
            <WifiOff className="size-4 shrink-0" />
            <span>Telemetry Stale (&gt;3.5s) — Values below reflect last received frame.</span>
          </div>
          <Badge variant="outline" className="text-[10px] font-mono border-purple-500/40 text-purple-400">
            STALE
          </Badge>
        </div>
      )}

      {/* 1. Top Cockpit Telemetry Banner: State, Risk, & Stream Health (2 Clean Non-Overlapping Rows) */}
      <div
        className={`p-3.5 rounded-2xl bg-card border ${stateStyle.cardBorder} shadow-xs space-y-2.5`}
      >
        {/* Row 1: Dynamic Status Pill + Risk Index + View Switcher */}
        <div className="flex items-center justify-between gap-2 flex-wrap">
          <div
            className={`px-3 py-1.5 rounded-xl text-xs font-black uppercase tracking-wider inline-flex items-center gap-1.5 whitespace-nowrap ${stateStyle.pillBg}`}
          >
            {stateStyle.icon}
            <span>{stateStyle.label}</span>
          </div>

          <div className="flex items-center gap-2">
            {onSwitchToCockpit && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={onSwitchToCockpit}
                className="h-7 text-xs text-muted-foreground hover:text-foreground gap-1 px-2 cursor-pointer"
                title="Switch to Driver Cockpit View"
                aria-label="Switch to Driver Cockpit View"
              >
                <Car className="size-3.5 text-primary" />
                <span className="hidden sm:inline">Driver View</span>
              </Button>
            )}

            <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded-xl bg-muted/50 border border-border/80">
              <span className="text-[10px] uppercase font-mono text-muted-foreground whitespace-nowrap">
                Risk Index
              </span>
              <span className={`text-sm font-black font-mono tabular-nums ${stateStyle.riskColor}`}>
                {riskPct}%
              </span>
            </div>
          </div>
        </div>

        {/* Row 2: Stream Health Metrics Strip (FPS, RTT, Queue, AI, Server, Audio) */}
        <div className="flex items-center gap-1.5 flex-wrap font-mono text-[11px] pt-2 border-t border-border/50">
          {/* FPS Badge with warning color if dropped */}
          <Badge
            variant="outline"
            className={`px-2 py-0.5 rounded-lg border gap-1 whitespace-nowrap ${
              fps < 10
                ? "border-amber-500/40 text-amber-500 bg-amber-500/10"
                : "border-border text-foreground"
            }`}
            title="Video Frames Processed per Second"
          >
            <Gauge className="size-3 text-primary" />
            <span>{fps.toFixed(1)} FPS</span>
          </Badge>

          {/* WebRTC DataChannel RTT Badge (Exact Measurement) */}
          {(() => {
            const rawRtt =
              latencyDiagnostics?.datachannel_rtt_ms && latencyDiagnostics.datachannel_rtt_ms > 0
                ? latencyDiagnostics.datachannel_rtt_ms
                : typeof network_roundtrip_ms === "number" && network_roundtrip_ms > 0
                ? network_roundtrip_ms
                : 0;
            const selectedRtt = Math.round(rawRtt);

            if (selectedRtt <= 0) return null;

            return (
              <Badge
                variant="outline"
                className={`px-2 py-0.5 rounded-lg border gap-1 whitespace-nowrap ${
                  selectedRtt > 300
                    ? "border-rose-500/40 text-rose-500 bg-rose-500/10"
                    : selectedRtt > 120
                    ? "border-amber-500/40 text-amber-500 bg-amber-500/10"
                    : "border-border text-foreground"
                }`}
                title={`Exact WebRTC DataChannel RTT (p50: ${latencyDiagnostics?.rtt_p50 && latencyDiagnostics.rtt_p50 > 0 ? latencyDiagnostics.rtt_p50 : selectedRtt}ms, p95: ${latencyDiagnostics?.rtt_p95 && latencyDiagnostics.rtt_p95 > 0 ? latencyDiagnostics.rtt_p95 : selectedRtt}ms)`}
              >
                <Wifi className="size-3 text-primary" />
                <span>RTT: {selectedRtt}ms</span>
              </Badge>
            );
          })()}

          {/* Server Ingest Queue Latency */}
          {telemetry.timing?.queue_latency_ms !== undefined && (
            <Badge
              variant="outline"
              className="px-2 py-0.5 rounded-lg border-border text-muted-foreground whitespace-nowrap"
              title="Server-side async frame queue dwell time before thread pool dispatch"
            >
              Q: {telemetry.timing.queue_latency_ms.toFixed(0)}ms
            </Badge>
          )}

          {/* AI Inference Latency Badge */}
          {(telemetry.timing?.infer_duration_ms != null || (inference_latency_ms != null && inference_latency_ms > 0)) && (
            <Badge
              variant="outline"
              className="px-2 py-0.5 rounded-lg border-border text-muted-foreground whitespace-nowrap"
              title="Authoritative LocalMonitoringService + DrowsinessEngine inference latency"
            >
              AI: {(telemetry.timing?.infer_duration_ms != null ? telemetry.timing.infer_duration_ms : inference_latency_ms ?? 0).toFixed(0)}ms
            </Badge>
          )}

          {/* Audio Subsystem Indicator & Controls */}
          {audioStatus && (
            <button
              type="button"
              onClick={() => {
                if (audioStatus === "blocked" && onUnlockAudio) {
                  onUnlockAudio();
                } else if (onToggleMute) {
                  onToggleMute();
                }
              }}
              className={`ml-auto inline-flex items-center gap-1 px-2 py-0.5 rounded-lg border text-[11px] font-mono tabular-nums whitespace-nowrap transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                audioStatus === "blocked"
                  ? "border-amber-500/40 text-amber-500 bg-amber-500/10 hover:bg-amber-500/20 cursor-pointer"
                  : audioStatus === "muted" || isMuted
                  ? "border-rose-500/40 text-rose-400 bg-rose-500/10 hover:bg-rose-500/20 cursor-pointer"
                  : audioStatus === "ready"
                  ? "border-emerald-500/30 text-emerald-400 bg-emerald-500/10 hover:bg-emerald-500/20 cursor-pointer"
                  : "border-border text-muted-foreground cursor-not-allowed"
              }`}
              title={
                audioStatus === "blocked"
                  ? "Audio autoplay blocked by browser. Click to enable sound alerts."
                  : audioStatus === "muted" || isMuted
                  ? "Audio muted. Click to unmute."
                  : `Audio active (${Math.round(volume * 100)}%). Click to mute.`
              }
            >
              {audioStatus === "blocked" ? (
                <>
                  <VolumeX className="size-3 text-amber-500" />
                  <span>Enable Audio</span>
                </>
              ) : audioStatus === "muted" || isMuted ? (
                <>
                  <VolumeX className="size-3 text-rose-400" />
                  <span>Muted</span>
                </>
              ) : audioStatus === "ready" ? (
                <>
                  {volume > 0.5 ? (
                    <Volume2 className="size-3 text-emerald-400" />
                  ) : (
                    <Volume1 className="size-3 text-emerald-400" />
                  )}
                  <span>Audio {Math.round(volume * 100)}%</span>
                </>
              ) : (
                <>
                  <VolumeX className="size-3 text-muted-foreground" />
                  <span>Audio N/A</span>
                </>
              )}
            </button>
          )}
        </div>
      </div>

      {/* 2. Hero Gauge Grid: EAR (Left) vs PERCLOS (Right) — Stacked until xl so cards never overlap */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-3">
        {/* Card A: Eye Aspect Ratio (EAR) */}
        <div className="p-3.5 rounded-2xl bg-card border border-border shadow-xs space-y-2.5">
          {/* Header Row 1: Title + Status Pill */}
          <div className="flex items-center justify-between gap-2">
            <div className="flex items-center gap-1.5 min-w-0">
              <Eye className="size-4 text-primary shrink-0" />
              <span className="text-xs font-bold text-foreground truncate">
                Eye Aspect Ratio (EAR)
              </span>
            </div>

            <span
              className={`text-xs font-mono tabular-nums font-bold px-2 py-0.5 rounded-lg inline-flex items-center gap-1 shrink-0 whitespace-nowrap ${
                !hasLiveFace
                  ? "bg-muted text-muted-foreground border border-border"
                  : isEyeClosed
                    ? "bg-rose-500/20 text-rose-500 border border-rose-500/30"
                    : "bg-emerald-500/20 text-emerald-500 border border-emerald-500/30"
              }`}
            >
              {!hasLiveFace ? (
                <span>Unavailable</span>
              ) : (
                <>
                  {isEyeClosed ? <EyeOff className="size-3" /> : <Eye className="size-3" />}
                  <span>{ear.toFixed(3)}</span>
                  <span className="text-[10px] uppercase font-sans">
                    {isEyeClosed ? "Closed" : "Open"}
                  </span>
                </>
              )}
            </span>
          </div>

          {/* Header Row 2: Dedicated full-width sub-metrics row (never wraps vertically) */}
          <div className="flex items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-muted-foreground">
            {earLeft !== undefined && earRight !== undefined ? (
              <span>
                L: {earLeft.toFixed(3)} &bull; R: {earRight.toFixed(3)}
              </span>
            ) : (
              <span>Dynamic facial geometry</span>
            )}
            <span className="text-amber-500 font-semibold">
              Cutoff: {threshold.toFixed(3)}
            </span>
          </div>

          {/* Visual EAR Bar with Cutoff Marker */}
          <div className="space-y-1 pt-0.5">
            <div className="relative h-3 w-full bg-muted/60 rounded-full overflow-hidden">
              <div
                className={`h-full transition-[width] duration-100 rounded-full ${
                  isEyeClosed ? "bg-rose-500" : "bg-emerald-500"
                }`}
                style={{
                  width: `${Math.min(100, Math.max(0, (ear / 0.45) * 100))}%`,
                }}
              />
              <div
                className="absolute top-0 bottom-0 w-1 bg-amber-400 shadow-sm z-10"
                style={{
                  left: `${Math.min(95, Math.max(5, (threshold / 0.45) * 100))}%`,
                }}
                title={`Closure Cutoff Threshold: ${threshold.toFixed(3)}`}
              />
            </div>

            <div className="flex justify-between text-[10px] font-mono text-muted-foreground">
              <span>0.00 (Closed)</span>
              <span>0.45 (Open)</span>
            </div>
          </div>
        </div>

        {/* Card B: PERCLOS (Percentage Eye Closure) */}
        {(() => {
          const formattedCoveragePct = Math.round(eye_metrics?.perclos_coverage_pct ?? 100);
          return (
            <div className="p-3.5 rounded-2xl bg-card border border-border shadow-xs space-y-2.5">
              {/* Header Row 1: Title + Status Pill */}
              <div className="flex items-center justify-between gap-2">
                <div className="flex items-center gap-1.5 min-w-0">
                  <Activity className="size-4 text-primary shrink-0" />
                  <span className="text-xs font-bold text-foreground truncate">
                    PERCLOS (60s)
                  </span>
                </div>

                <span
                  className={`text-xs font-mono tabular-nums font-bold px-2 py-0.5 rounded-lg inline-flex items-center gap-1 shrink-0 whitespace-nowrap ${
                    eye_metrics?.perclos_reliable === false
                      ? "bg-muted text-muted-foreground border border-border"
                      : perclos > 25
                      ? "bg-rose-500/20 text-rose-500 border border-rose-500/30"
                      : perclos > 15
                      ? "bg-amber-500/20 text-amber-500 border border-amber-500/30"
                      : "bg-emerald-500/20 text-emerald-500 border border-emerald-500/30"
                  }`}
                >
                  <span>{perclos.toFixed(1)}%</span>
                  <span className="text-[10px] uppercase font-sans">
                    {eye_metrics?.perclos_reliable === false
                      ? "Warmup"
                      : perclos > 25
                      ? "Critical"
                      : perclos > 15
                      ? "Drowsy"
                      : "Alert"}
                  </span>
                </span>
              </div>

              {/* Header Row 2: Dedicated full-width sub-metrics row */}
              <div className="flex items-center justify-between gap-2 text-[11px] font-mono tabular-nums text-muted-foreground">
                <span>Time eyes closed &ge; 80%</span>
                <span
                  className={
                    eye_metrics?.perclos_reliable === false
                      ? "text-amber-500 font-medium"
                      : "text-emerald-500 font-medium"
                  }
                >
                  Window Coverage: {formattedCoveragePct}%
                </span>
              </div>

              {/* Visual Segmented PERCLOS Bar */}
              <div className="space-y-1 pt-0.5">
                <div className="relative h-3 w-full bg-muted/60 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-[width] duration-150 rounded-full ${
                      perclos > 25
                        ? "bg-rose-500"
                        : perclos > 15
                        ? "bg-amber-500"
                        : "bg-emerald-500"
                    }`}
                    style={{ width: `${Math.min(100, Math.max(0, perclos))}%` }}
                  />
                </div>

                <div className="flex justify-between text-[10px] font-mono">
                  <span className="text-emerald-500 font-medium">&lt;15% Safe</span>
                  <span className="text-amber-500 font-medium">15–25% Warn</span>
                  <span className="text-rose-500 font-medium">&gt;25% Crit</span>
                </div>
              </div>
            </div>
          );
        })()}
      </div>

      {/* 4. Cockpit Kinematics & Spatial Tracking (4 Tile Instruments) */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono tabular-nums">
        {/* Instrument 1: 3D Gaze Zone & Euro NCAP Entropy */}
        <div className="p-2.5 rounded-xl bg-card border border-border shadow-2xs space-y-1">
          <div className="text-[10px] text-muted-foreground uppercase flex items-center justify-between">
            <span className="flex items-center gap-1 truncate">
              <Compass className="size-3 text-primary shrink-0" />
              <span>3D Gaze Target</span>
            </span>
            {eye_metrics?.gaze_entropy !== null && eye_metrics?.gaze_entropy !== undefined && (
              <span className="text-[9px] text-primary font-bold">
                Ht: {eye_metrics.gaze_entropy.toFixed(2)}
              </span>
            )}
          </div>
          <div className="font-bold text-foreground text-xs truncate">
            {pose_zone}
          </div>
          <div className="text-[10px] text-muted-foreground font-sans truncate">
            {zoneInfo.hint}
          </div>
        </div>

        {/* Instrument 2: Head Pose Angles */}
        <div className="p-2.5 rounded-xl bg-card border border-border shadow-2xs space-y-1">
          <div className="text-[10px] text-muted-foreground uppercase flex items-center gap-1">
            <Crosshair className="size-3 text-primary shrink-0" />
            <span className="truncate">Head Pose</span>
          </div>
          <div className="font-bold text-foreground text-xs truncate">
            {hasLiveFace && head_pose
              ? `P: ${Math.round(head_pose.pitch_deg)}° | Y: ${Math.round(head_pose.yaw_deg)}°`
              : "Unavailable"}
          </div>
          <div className="text-[10px] text-muted-foreground font-sans truncate">
            {hasLiveFace && head_pose ? `Roll: ${Math.round(head_pose.roll_deg)}°` : "Awaiting face lock"}
          </div>
        </div>

        {/* Instrument 3: Blink Rate */}
        <div className="p-2.5 rounded-xl bg-card border border-border shadow-2xs space-y-1">
          <div className="text-[10px] text-muted-foreground uppercase flex items-center gap-1">
            <Eye className="size-3 text-primary shrink-0" />
            <span className="truncate">Blink Frequency</span>
          </div>
          <div className="font-bold text-foreground text-xs truncate">
            {blinkRate > 0 ? `${blinkRate.toFixed(1)} /min` : "--"}
          </div>
          <div className="text-[10px] text-muted-foreground font-sans truncate">
            {blinkDur ? `Last: ${Math.round(blinkDur)}ms` : "Normal cadence"}
          </div>
        </div>

        {/* Instrument 4: Yawn MAR & Kinematics */}
        <div className="p-2.5 rounded-xl bg-card border border-border shadow-2xs space-y-1">
          <div className="text-[10px] text-muted-foreground uppercase flex items-center justify-between">
            <span className="flex items-center gap-1 truncate">
              <Timer className="size-3 text-primary shrink-0" />
              <span>Yawn MAR / RCR</span>
            </span>
          </div>
          <div className="font-bold text-foreground text-xs truncate">
            {hasLiveFace
              ? `MAR: ${marYawn.toFixed(2)}${blinkRcr ? ` · RCR ${blinkRcr.toFixed(2)}` : ""}`
              : "Unavailable"}
          </div>
          <div className="text-[10px] text-muted-foreground font-sans truncate">
            {reason_codes?.includes("AUTHENTIC_YAWN") || reason_codes?.includes("FREQUENT_YAWNING")
              ? "Active yawn detected"
              : "Mouth aperture nominal"}
          </div>
        </div>
      </div>

      {/* 5. Authoritative Backend Engine Authority Panel */}
      {showEngineStrip && (
        <div className="p-3 rounded-2xl bg-card border border-border flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 text-xs">
          <div className="flex items-center gap-2 flex-wrap">
            <Cpu className="size-4 text-primary shrink-0" aria-hidden="true" />
            <span className="text-muted-foreground font-medium text-[11px] sm:text-xs">
              Engine:
            </span>
            <Badge variant="secondary" className="font-mono text-[11px] font-bold">
              {telemetry.authoritative_engine || "DrowsinessEngine"}
            </Badge>
            {telemetry.drowsiness_state && (
              <Badge variant="outline" className="font-mono text-[10px]">
                State: {telemetry.drowsiness_state}
              </Badge>
            )}
          </div>

          <div className="flex items-center gap-2 font-mono tabular-nums flex-wrap">
            {telemetry.alert_state && (
              <Badge variant="outline" className="border-border text-foreground text-[10px]">
                Alert: {telemetry.alert_state}
              </Badge>
            )}
            {cascade?.geom_reliability !== undefined && (
              <span className="text-[10px] text-muted-foreground">
                Confidence: {(cascade.geom_reliability * 100).toFixed(0)}%
              </span>
            )}
          </div>
        </div>
      )}

      {/* 6. Personalised 4-Zone vs Conventional Static (0.22) Benchmark Strip */}
      {showComparisonStrip && telemetry.comparison && (
        <div className="p-3 rounded-2xl bg-card border border-border flex flex-wrap items-center justify-between gap-2 text-xs">
          <div className="flex items-center gap-2">
            <ShieldCheck className="size-4 text-emerald-500 shrink-0" aria-hidden="true" />
            <span className="font-medium text-foreground">PDBP vs Fixed 0.22 Rule:</span>
            <span className="font-mono tabular-nums text-muted-foreground">
              Zone Thresh {telemetry.comparison.personalised_threshold.toFixed(3)} vs Fixed {telemetry.comparison.fixed_cutoff_ear.toFixed(2)}
            </span>
          </div>
          <div className="flex items-center gap-1.5 font-mono">
            {telemetry.comparison.fixed_false_alarm ? (
              <Badge className="bg-emerald-500/20 text-emerald-500 border border-emerald-500/40 text-[10px]">
                FALSE ALARM PREVENTED
              </Badge>
            ) : telemetry.comparison.is_mirror_check_suppressed ? (
              <Badge variant="outline" className="border-primary/40 text-primary text-[10px]">
                MIRROR GLANCE SUPPRESSED
              </Badge>
            ) : (
              <Badge variant="outline" className="text-[10px] text-muted-foreground">
                Fixed Rule: {telemetry.comparison.fixed_alert_state}
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* 7. Audio Alert Synthesizer Control & Test Bar */}
      {showAudioTestBar && onTestAudioTier && (
        <div className="p-3 rounded-2xl bg-card border border-border space-y-2 text-xs">
          <div className="flex items-center justify-between gap-2">
            <span className="font-semibold text-foreground flex items-center gap-1.5">
              <Volume2 className="size-3.5 text-primary" aria-hidden="true" />
              Cabin Audio Siren Control &amp; Test
            </span>
            {onSetVolume && (
              <div className="flex items-center gap-2">
                <label htmlFor="hud-alert-volume" className="text-[11px] font-mono tabular-nums text-muted-foreground">
                  Vol {Math.round(volume * 100)}%
                </label>
                <input
                  id="hud-alert-volume"
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={volume}
                  aria-label="Alert volume"
                  onChange={(e) => onSetVolume(parseFloat(e.target.value))}
                  className="w-20 accent-primary cursor-pointer"
                />
              </div>
            )}
          </div>
          <div className="grid grid-cols-4 gap-1.5 font-mono text-[11px]">
            <button
              type="button"
              onClick={() => onTestAudioTier("WARNING")}
              className="py-1.5 px-2 rounded-lg border border-amber-500/40 bg-amber-500/10 text-amber-500 hover:bg-amber-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Warn Beep
            </button>
            <button
              type="button"
              onClick={() => onTestAudioTier("DROWSY")}
              className="py-1.5 px-2 rounded-lg border border-orange-500/40 bg-orange-500/10 text-orange-500 hover:bg-orange-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Drowsy Pulse
            </button>
            <button
              type="button"
              onClick={() => onTestAudioTier("CRITICAL")}
              className="py-1.5 px-2 rounded-lg border border-rose-500/40 bg-rose-500/10 text-rose-500 hover:bg-rose-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Crit Siren
            </button>
            <button
              type="button"
              onClick={() => onTestAudioTier("RECOVERY")}
              className="py-1.5 px-2 rounded-lg border border-emerald-500/40 bg-emerald-500/10 text-emerald-500 hover:bg-emerald-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              Recovery
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
