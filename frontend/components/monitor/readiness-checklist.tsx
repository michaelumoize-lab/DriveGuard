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

import type { BackendProcessingStatus } from "@/types/telemetry";

export interface ReadinessChecklistProps {
  cameraActive: boolean;
  cameraResolutionLabel?: string;
  faceDetected: boolean;
  isCalibrated: boolean;
  centerEarBaseline?: number;
  livekitStatus: "disconnected" | "connecting" | "connected" | "reconnecting" | "error";
  backendProcessingStatus?: BackendProcessingStatus;
  isStale: boolean;
  rttMs?: number;
  audioStatus: string;
  audioMuted: boolean;
  runtimeMode?: "browser" | "remote";
  layout?: "grid" | "list";
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
  backendProcessingStatus,
  isStale,
  rttMs,
  audioStatus,
  audioMuted,
  runtimeMode = "browser",
  layout = "grid",
  onUnlockAudio,
  onCalibrateClick,
}: ReadinessChecklistProps) {
  const engineValue =
    runtimeMode === "browser"
      ? backendProcessingStatus === "processing"
        ? "On-device active"
        : backendProcessingStatus === "connecting"
          ? "Loading model…"
          : "On-device ready"
      : backendProcessingStatus === "processing" && !isStale
        ? `Connected${rttMs && rttMs > 0 ? ` (${rttMs}\u00A0ms)` : ""}`
        : backendProcessingStatus === "awaiting_telemetry"
          ? "Awaiting frames…"
          : isStale
            ? "Stream stalled"
            : livekitStatus === "connecting" || livekitStatus === "reconnecting"
              ? "Connecting…"
              : livekitStatus === "error"
                ? "Connection error"
                : livekitStatus === "connected"
                  ? "Connected"
                  : "Standby";

  const engineState: "ready" | "warning" | "idle" =
    runtimeMode === "browser"
      ? backendProcessingStatus === "processing"
        ? "ready"
        : "idle"
      : backendProcessingStatus === "processing" && !isStale
        ? "ready"
        : isStale || livekitStatus === "error"
          ? "warning"
          : "idle";

  const items: Array<{
    id: string;
    label: string;
    value: string;
    state: "ready" | "warning" | "idle";
    icon: React.ComponentType<{ className?: string }>;
    actionHint?: string;
    onClick?: () => void;
  }> = [
    {
      id: "camera",
      label: "Camera feed",
      value: cameraActive ? cameraResolutionLabel || "Active" : "Standby",
      state: cameraActive ? "ready" : "idle",
      icon: Camera,
    },
    {
      id: "face",
      label: "Face tracking",
      value: !cameraActive
        ? "Waiting for camera"
        : faceDetected
          ? "Face locked"
          : "Face not visible",
      state: !cameraActive ? "idle" : faceDetected ? "ready" : "warning",
      icon: ScanFace,
    },
    {
      id: "calibration",
      label: "Eye calibration",
      value: isCalibrated
        ? `Personalised (${centerEarBaseline !== undefined ? centerEarBaseline.toFixed(3) : "Active"})`
        : "Needs 15s calibration",
      state: isCalibrated ? "ready" : "warning",
      icon: Compass,
      actionHint: !isCalibrated && onCalibrateClick ? "Calibrate" : undefined,
      onClick: !isCalibrated ? onCalibrateClick : undefined,
    },
    {
      id: "worker",
      label: runtimeMode === "browser" ? "On-device AI" : "Remote AI worker",
      value: engineValue,
      state: engineState,
      icon: Radio,
    },
    {
      id: "audio",
      label: "Alert audio",
      value: audioMuted
        ? "Muted"
        : audioStatus === "ready"
          ? "Armed"
          : "Tap to enable",
      state: audioMuted ? "warning" : audioStatus === "ready" ? "ready" : "idle",
      icon: audioMuted ? VolumeX : Volume2,
      actionHint: audioStatus !== "ready" && !audioMuted && onUnlockAudio ? "Enable" : undefined,
      onClick: audioStatus !== "ready" && !audioMuted ? onUnlockAudio : undefined,
    },
  ];

  const containerClass =
    layout === "list"
      ? "flex flex-col gap-1.5"
      : "grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-2";

  return (
    <div role="region" aria-label="System Readiness Checklist" className={containerClass}>
      {items.map((item) => {
        const Icon = item.icon;
        const isReady = item.state === "ready";
        const isWarn = item.state === "warning";

        const baseClasses = `flex w-full items-center justify-between gap-2.5 px-3 py-2 rounded-xl border text-left text-xs transition-colors ${
          isReady
            ? "border-emerald-500/25 bg-emerald-500/5 text-foreground"
            : isWarn
              ? "border-amber-500/35 bg-amber-500/10 text-foreground"
              : "border-border/80 bg-muted/30 text-muted-foreground"
        }`;

        const inner = (
          <>
            <div className="flex items-center gap-2.5 min-w-0">
              <Icon
                aria-hidden="true"
                className={`size-4 shrink-0 ${
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

            <div className="flex items-center gap-1.5 shrink-0">
              {item.actionHint && (
                <span className="text-[11px] font-semibold text-primary underline-offset-2 group-hover:underline">
                  {item.actionHint}
                </span>
              )}
              {isReady ? (
                <CheckCircle2 className="size-4 text-emerald-500 shrink-0" aria-hidden="true" />
              ) : isWarn ? (
                <AlertTriangle className="size-4 text-amber-500 shrink-0" aria-hidden="true" />
              ) : (
                <CircleDashed className="size-4 text-muted-foreground/60 shrink-0" aria-hidden="true" />
              )}
            </div>
          </>
        );

        if (item.onClick) {
          return (
            <button
              key={item.id}
              type="button"
              onClick={item.onClick}
              className={`group ${baseClasses} cursor-pointer hover:bg-muted/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring`}
            >
              {inner}
            </button>
          );
        }

        return (
          <div key={item.id} className={baseClasses}>
            {inner}
          </div>
        );
      })}
    </div>
  );
});
