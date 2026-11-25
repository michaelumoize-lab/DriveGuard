"use client";

import React from "react";
import Link from "next/link";
import {
  Car,
  ChevronDown,
  Compass,
  SlidersHorizontal,
  CheckCircle2,
  AlertTriangle,
  CircleDashed,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import {
  Popover,
  PopoverContent,
  PopoverHeader,
  PopoverTitle,
  PopoverDescription,
  PopoverTrigger,
} from "@/components/ui/popover";
import { ReadinessChecklist } from "./readiness-checklist";
import type { BackendProcessingStatus } from "@/types/telemetry";

export interface MonitorHeaderBarProps {
  consoleMode: "cockpit" | "diagnostics";
  onConsoleModeChange: (mode: "cockpit" | "diagnostics") => void;
  runtimeMode: "browser" | "python-local" | "remote";
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
  browserSyncStatus?: "synced" | "offline-buffer" | string;
  showInlineCalibration: boolean;
  onToggleCalibration: () => void;
  onUnlockAudio: () => void;
}

export const MonitorHeaderBar = React.memo(function MonitorHeaderBar({
  consoleMode,
  onConsoleModeChange,
  runtimeMode,
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
  browserSyncStatus,
  showInlineCalibration,
  onToggleCalibration,
  onUnlockAudio,
}: MonitorHeaderBarProps) {
  // Compute a clean 5-point readiness summary so the header stays compact
  const engineReady =
    runtimeMode === "browser"
      ? backendProcessingStatus === "processing"
      : backendProcessingStatus === "processing" && !isStale;
  const audioReady = !audioMuted && audioStatus === "ready";
  const checks = [cameraActive, faceDetected, isCalibrated, engineReady, audioReady];
  const readyCount = checks.filter(Boolean).length;
  const hasAttention = !isCalibrated || (cameraActive && !faceDetected) || isStale || livekitStatus === "error" || audioMuted;

  return (
    <header className="flex flex-col gap-2.5 sm:flex-row sm:items-center sm:justify-between pb-3 border-b border-border/70">
      {/* Left: Title & Quiet Runtime Context */}
      <div className="flex items-center gap-3 min-w-0">
        <div className="min-w-0">
          <div className="flex items-center gap-2.5 flex-wrap">
            <h1 className="text-lg sm:text-xl font-bold tracking-tight text-foreground text-balance">
              Live Driver Monitor
            </h1>
            <span className="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/40 px-2.5 py-0.5 text-xs font-medium text-muted-foreground">
              <span
                aria-hidden="true"
                className={`size-1.5 rounded-full ${
                  cameraActive && !isStale
                    ? "bg-emerald-500"
                    : isStale
                      ? "bg-amber-500"
                      : "bg-muted-foreground/60"
                }`}
              />
              <span>
                {runtimeMode === "browser"
                  ? "On-device AI"
                  : runtimeMode === "python-local"
                    ? "Local Python AI"
                    : "Remote server AI"}
              </span>
              {runtimeMode === "browser" && browserSyncStatus === "offline-buffer" && (
                <span className="text-amber-500">· Offline buffer</span>
              )}
            </span>
          </div>
          <p className="text-xs text-muted-foreground text-pretty mt-0.5">
            {consoleMode === "cockpit"
              ? "Calm road-ready view with real-time fatigue alerts and mirror-glance protection."
              : "Engineering inspector with live physiological metrics, waveforms, and testbench."}
          </p>
        </div>
      </div>

      {/* Right: Mode Toggle + Readiness Popover + Quick Calibrate */}
      <div className="flex items-center gap-2 flex-wrap shrink-0">
        {/* Driver / Inspector Segmented Control */}
        <div
          className="inline-flex items-center rounded-xl bg-muted/60 p-1 border border-border/80"
          role="group"
          aria-label="Monitor view mode"
        >
          <button
            type="button"
            onClick={() => onConsoleModeChange("cockpit")}
            aria-pressed={consoleMode === "cockpit"}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 min-h-10 sm:min-h-7 py-1.5 text-xs font-semibold transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
              consoleMode === "cockpit"
                ? "bg-background text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <Car className="size-3.5" aria-hidden="true" />
            <span>Driver</span>
          </button>
          <button
            type="button"
            onClick={() => onConsoleModeChange("diagnostics")}
            aria-pressed={consoleMode === "diagnostics"}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 min-h-10 sm:min-h-7 py-1.5 text-xs font-semibold transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
              consoleMode === "diagnostics"
                ? "bg-background text-foreground shadow-xs"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            <SlidersHorizontal className="size-3.5" aria-hidden="true" />
            <span>Inspector</span>
          </button>
        </div>

        {/* Collapsible System Readiness Summary Pill */}
        <Popover>
          <PopoverTrigger asChild>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className={`min-h-11 sm:h-8 sm:min-h-0 rounded-xl gap-1.5 text-xs font-medium tabular-nums ${
                readyCount === 5
                  ? "border-emerald-500/35 bg-emerald-500/5 text-foreground"
                  : hasAttention
                    ? "border-amber-500/40 bg-amber-500/10 text-foreground"
                    : ""
              }`}
              aria-label={`System readiness: ${readyCount} of 5 checks ready`}
            >
              {readyCount === 5 ? (
                <CheckCircle2 className="size-3.5 text-emerald-500 shrink-0" aria-hidden="true" />
              ) : hasAttention ? (
                <AlertTriangle className="size-3.5 text-amber-500 shrink-0" aria-hidden="true" />
              ) : (
                <CircleDashed className="size-3.5 text-muted-foreground shrink-0" aria-hidden="true" />
              )}
              <span>
                {readyCount === 5 ? "System Ready" : `Checklist ${readyCount}/5`}
              </span>
              <ChevronDown className="size-3 text-muted-foreground" aria-hidden="true" />
            </Button>
          </PopoverTrigger>
          <PopoverContent align="end" className="w-[min(22rem,calc(100vw-1.5rem))] sm:w-96 p-3.5 flex flex-col gap-3 overscroll-contain">
            <PopoverHeader>
              <PopoverTitle className="text-sm font-semibold">
                Pre-drive system checklist
              </PopoverTitle>
              <PopoverDescription className="text-xs">
                Verify camera, eye calibration, AI engine, and sound alerts before driving.
              </PopoverDescription>
            </PopoverHeader>

            <ReadinessChecklist
              layout="list"
              runtimeMode={runtimeMode}
              cameraActive={cameraActive}
              cameraResolutionLabel={cameraResolutionLabel}
              faceDetected={faceDetected}
              isCalibrated={isCalibrated}
              centerEarBaseline={centerEarBaseline}
              livekitStatus={livekitStatus}
              backendProcessingStatus={backendProcessingStatus}
              isStale={isStale}
              rttMs={rttMs}
              audioStatus={audioStatus}
              audioMuted={audioMuted}
              onUnlockAudio={onUnlockAudio}
              onCalibrateClick={onToggleCalibration}
            />

            <div className="pt-2 border-t border-border/60 flex items-center justify-between gap-2 text-xs">
              <Link
                href="/dashboard/calibration"
                className="min-h-11 sm:min-h-0 inline-flex items-center font-medium text-primary hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
              >
                Open full calibration wizard
              </Link>
              <Link
                href="/dashboard/sessions"
                className="min-h-11 sm:min-h-0 inline-flex items-center text-muted-foreground hover:text-foreground hover:underline focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded"
              >
                Drive history
              </Link>
            </div>
          </PopoverContent>
        </Popover>

        {/* Quick Calibrate Toggle */}
        <Button
          type="button"
          size="sm"
          variant={showInlineCalibration ? "default" : !isCalibrated ? "secondary" : "outline"}
          onClick={onToggleCalibration}
          className="min-h-11 sm:h-8 sm:min-h-0 rounded-xl gap-1.5 text-xs font-medium"
        >
          <Compass className="size-3.5" aria-hidden="true" />
          <span>{showInlineCalibration ? "Hide Calibration" : isCalibrated ? "Recalibrate" : "Calibrate"}</span>
        </Button>
      </div>
    </header>
  );
});
