"use client";

import React from "react";
import {
  Camera,
  ScanFace,
  Compass,
  Radio,
  Volume2,
  VolumeX,
  CheckCircle2,
  AlertTriangle,
  CircleDashed,
} from "lucide-react";

export interface ReadinessChecklistProps {
  cameraActive: boolean;
  cameraResolutionLabel?: string;
  faceDetected: boolean;
  isCalibrated: boolean;
  centerEarBaseline?: number;
  livekitStatus: "disconnected" | "connecting" | "connected" | "reconnecting" | "error";
  isStale: boolean;
  rttMs?: number;
  audioStatus: string;
  audioMuted: boolean;
  onUnlockAudio?: () => void;
  onCalibrateClick?: () => void;
}

export const ReadinessChecklist = React.memo(function ReadinessChecklist({
  cameraActive,
  cameraResolutionLabel,
  faceDetected,
  isCalibrated,
  centerEarBaseline,
  livekitStatus,
  isStale,
  rttMs,
  audioStatus,
  audioMuted,
  onUnlockAudio,
  onCalibrateClick,
}: ReadinessChecklistProps) {
  const items: Array<{
    id: string;
    label: string;
    value: string;
    state: "ready" | "warning" | "idle";
    icon: React.ComponentType<{ className?: string }>;
    onClick?: () => void;
  }> = [
    {
      id: "camera",
      label: "Camera Feed",
      value: cameraActive ? cameraResolutionLabel || "Active" : "Standby",
      state: cameraActive ? "ready" : "idle",
      icon: Camera,
    },
    {
      id: "face",
      label: "Face Tracking",
      value: !cameraActive
        ? "Awaiting Feed"
        : faceDetected
          ? "478 Landmarks Locked"
          : "Face Not Detected",
      state: !cameraActive ? "idle" : faceDetected ? "ready" : "warning",
      icon: ScanFace,
    },
    {
      id: "calibration",
      label: "4-Zone Baseline",
      value: isCalibrated
        ? `Calibrated (${(centerEarBaseline ?? 0.285).toFixed(3)})`
        : "Uncalibrated Default",
      state: isCalibrated ? "ready" : "warning",
      icon: Compass,
      onClick: !isCalibrated ? onCalibrateClick : undefined,
    },
    {
      id: "worker",
      label: "LiveKit AI Worker",
      value:
        livekitStatus === "connected" && !isStale
          ? `Online${rttMs && rttMs > 0 ? ` (${rttMs} ms)` : ""}`
          : isStale
            ? "Telemetry Stale"
            : livekitStatus === "connecting" || livekitStatus === "reconnecting"
              ? "Handshaking…"
              : livekitStatus === "error"
                ? "Connection Error"
                : "Standby",
      state:
        livekitStatus === "connected" && !isStale
          ? "ready"
          : isStale || livekitStatus === "error"
            ? "warning"
            : "idle",
      icon: Radio,
    },
    {
      id: "audio",
      label: "Cabin Siren",
      value: audioMuted
        ? "Muted"
        : audioStatus === "ready"
          ? "Armed (4-Tier)"
          : "Tap to Unlock",
      state: audioMuted ? "warning" : audioStatus === "ready" ? "ready" : "idle",
      icon: audioMuted ? VolumeX : Volume2,
      onClick: audioStatus !== "ready" && !audioMuted ? onUnlockAudio : undefined,
    },
  ];

  return (
    <div
      role="region"
      aria-label="System Readiness Checklist"
      className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2"
    >
      {items.map((item) => {
        const Icon = item.icon;
        const isReady = item.state === "ready";
        const isWarn = item.state === "warning";

        return (
          <div
            key={item.id}
            onClick={item.onClick}
            role={item.onClick ? "button" : undefined}
            tabIndex={item.onClick ? 0 : undefined}
            onKeyDown={
              item.onClick
                ? (e) => {
                    if (e.key === "Enter" || e.key === " ") {
                      e.preventDefault();
                      item.onClick?.();
                    }
                  }
                : undefined
            }
            className={`flex items-center justify-between gap-2 px-3 py-2 rounded-xl border text-xs transition-colors outline-none ${
              item.onClick ? "cursor-pointer hover:bg-muted/60 focus-visible:ring-2 focus-visible:ring-primary" : ""
            } ${
              isReady
                ? "border-emerald-500/30 bg-emerald-500/5 text-foreground"
                : isWarn
                  ? "border-amber-500/40 bg-amber-500/10 text-foreground"
                  : "border-border/80 bg-card/60 text-muted-foreground"
            }`}
          >
            <div className="flex items-center gap-2 min-w-0">
              <Icon
                className={`size-3.5 shrink-0 ${
                  isReady
                    ? "text-emerald-500"
                    : isWarn
                      ? "text-amber-500"
                      : "text-muted-foreground"
                }`}
              />
              <div className="min-w-0">
                <div className="text-[11px] font-medium text-muted-foreground truncate">
                  {item.label}
                </div>
                <div className="font-mono tabular-nums font-semibold text-xs text-foreground truncate">
                  {item.value}
                </div>
              </div>
            </div>

            {isReady ? (
              <CheckCircle2 className="size-3.5 text-emerald-500 shrink-0" aria-hidden="true" />
            ) : isWarn ? (
              <AlertTriangle className="size-3.5 text-amber-500 shrink-0" aria-hidden="true" />
            ) : (
              <CircleDashed className="size-3.5 text-muted-foreground/60 shrink-0" aria-hidden="true" />
            )}
          </div>
        );
      })}
    </div>
  );
});
