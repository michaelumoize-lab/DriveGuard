"use client";

import React from "react";
import { CameraView } from "@/components/camera/camera-view";
import { CameraSelector } from "@/components/camera/camera-selector";
import { CameraErrorDisplay } from "@/components/camera/camera-error";
import { Card, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Camera,
  CheckCircle2,
  Cpu,
  FastForward,
  FlipHorizontal,
  Maximize2,
  Minimize2,
  PowerOff,
  Server,
  Square,
  SwitchCamera,
  Wifi,
  Cloud,
} from "lucide-react";
import type {
  CameraDevice,
  CameraError,
  CameraResolution,
  CameraState,
  OpticalFrameQuality,
} from "@/types/camera";
import type { BackendProcessingStatus, CalibrationPhase } from "@/types/telemetry";
import { formatZoneLabel, getZoneMeta } from "./calibration-zones";

interface CalibrationViewportProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraState: CameraState;
  previewMode?: "video" | "mjpeg";
  mjpegUrl?: string | null;
  cameraError: CameraError | null;
  devices: CameraDevice[];
  selectedDeviceId: string;
  isMirrored: boolean;
  resolution: CameraResolution | null;
  opticalQuality: OpticalFrameQuality | null;
  calibrationEngine: "browser" | "python";
  backendType: "cloud" | "local";
  onSelectEngine: (engine: "browser" | "python", pythonTarget?: "cloud" | "local") => void;
  connectionStatus: "disconnected" | "connecting" | "connected" | "reconnecting" | "error";
  backendProcessingStatus: BackendProcessingStatus;
  workerError: string | null;
  isStale: boolean;
  calState: CalibrationPhase | null;
  isCalibrating: boolean;
  isSuccess: boolean;
  isSavingProfile: boolean;
  isFullscreen: boolean;
  embeddedInOnboarding: boolean;
  isFaceVisible: boolean | null;
  onStartCameraCheck: () => void;
  onRetryCamera: () => void;
  onStopCamera: () => void;
  onSelectDevice: (deviceId: string) => void;
  onRefreshDevices: () => void;
  onSwitchFacing: () => void;
  onToggleMirror: () => void;
  onToggleFullscreen: () => void;
  onSkipZone: () => void;
  onStopCalibration: () => void;
}

export function CalibrationViewport({
  videoRef,
  cameraState,
  previewMode = "video",
  mjpegUrl = null,
  cameraError,
  devices,
  selectedDeviceId,
  isMirrored,
  resolution,
  opticalQuality,
  calibrationEngine,
  backendType,
  onSelectEngine,
  connectionStatus,
  backendProcessingStatus,
  workerError,
  isStale,
  calState,
  isCalibrating,
  isSuccess,
  isSavingProfile,
  isFullscreen,
  embeddedInOnboarding,
  isFaceVisible,
  onStartCameraCheck,
  onRetryCamera,
  onStopCamera,
  onSelectDevice,
  onRefreshDevices,
  onSwitchFacing,
  onToggleMirror,
  onToggleFullscreen,
  onSkipZone,
  onStopCalibration,
}: CalibrationViewportProps) {
  const isCameraActive = cameraState === "active";
  const isCameraErrorState =
    cameraState === "permission-denied" ||
    cameraState === "no-camera-available" ||
    cameraState === "camera-already-in-use" ||
    cameraState === "unsupported" ||
    cameraState === "connection-error";

  const activeZoneMeta = getZoneMeta(calState?.current_zone);

  return (
    <Card
      className={`overflow-hidden border-border/80 flex flex-col ${
        isFullscreen ? "border-0 rounded-none bg-black h-full" : "lg:col-span-7"
      }`}
    >
      {/* Top Viewport Bar: Camera Title + Engine Selector (Local / Cloud) + Status + Fullscreen */}
      <CardHeader className="py-3 px-4 flex flex-row flex-wrap items-center justify-between gap-2 border-b border-border bg-card">
        <div className="flex items-center gap-2 min-w-0">
          <Camera className="size-4 text-primary shrink-0" aria-hidden="true" />
          <CardTitle className="text-sm font-semibold truncate">
            {isCalibrating
              ? "Step 2: 4-Zone Gaze Capture"
              : isCameraActive
                ? "Step 1: Camera & Framing Check"
                : "Camera Preview"}
          </CardTitle>
          {resolution && isCameraActive && (
            <span className="hidden sm:inline text-[11px] font-mono tabular-nums text-muted-foreground">
              {resolution.width}×{resolution.height}
              {resolution.frameRate ? ` · ${resolution.frameRate}\u00A0fps` : ""}
            </span>
          )}
        </div>

        <div className="flex items-center gap-2 flex-wrap">
          {/* Calibration AI Engine Selector: Browser AI (Default) vs Python Local vs LiveKit Cloud */}
          <div
            role="group"
            aria-label="Calibration AI engine"
            className="flex flex-wrap items-center bg-muted/60 rounded-lg p-0.5 text-[11px] font-medium border border-border max-w-full"
          >
            <button
              type="button"
              aria-pressed={calibrationEngine === "browser"}
              onClick={() => onSelectEngine("browser")}
              className={`inline-flex items-center gap-1 px-2.5 min-h-10 sm:min-h-7 py-1 rounded-md transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                calibrationEngine === "browser"
                  ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              title="Use On-Device Browser AI (MediaPipe WASM — works offline, zero backend required)"
            >
              <Cpu className="size-3" aria-hidden="true" />
              <span>Browser AI</span>
            </button>
            <button
              type="button"
              aria-pressed={calibrationEngine === "python" && backendType === "local"}
              onClick={() => onSelectEngine("python", "local")}
              className={`inline-flex items-center gap-1 px-2.5 min-h-10 sm:min-h-7 py-1 rounded-md transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                calibrationEngine === "python" && backendType === "local"
                  ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              title="Use Local Python AI Engine (127.0.0.1:8000)"
            >
              <Server className="size-3" aria-hidden="true" />
              <span>Python Local</span>
            </button>
            <button
              type="button"
              aria-pressed={calibrationEngine === "python" && backendType === "cloud"}
              onClick={() => onSelectEngine("python", "cloud")}
              className={`inline-flex items-center gap-1 px-2.5 min-h-10 sm:min-h-7 py-1 rounded-md transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                calibrationEngine === "python" && backendType === "cloud"
                  ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                  : "text-muted-foreground hover:text-foreground"
              }`}
              title="Use Remote LiveKit Cloud Python AI Worker"
            >
              <Cloud className="size-3" aria-hidden="true" />
              <span>LiveKit Cloud</span>
            </button>
          </div>

          {/* Connection / Engine Status Badge */}
          <Badge
            variant="outline"
            className={`text-[11px] font-mono flex items-center gap-1 ${
              workerError || connectionStatus === "error"
                ? "border-rose-500/40 text-rose-500 bg-rose-500/10"
                : isStale
                  ? "border-amber-500/40 text-amber-500 bg-amber-500/10"
                  : isCalibrating
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : backendProcessingStatus === "processing"
                      ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10"
                      : "text-muted-foreground"
            }`}
          >
            {workerError || connectionStatus === "error" ? (
              "ENGINE ERROR"
            ) : isStale ? (
              "STALE STREAM"
            ) : isCalibrating ? (
              <>
                <Wifi className="size-3" aria-hidden="true" />
                <span>CALIBRATING</span>
              </>
            ) : backendProcessingStatus === "connecting" ||
              backendProcessingStatus === "awaiting_telemetry" ? (
              "CONNECTING…"
            ) : backendProcessingStatus === "processing" ? (
              "READY"
            ) : (
              "STANDBY"
            )}
          </Badge>

          {!embeddedInOnboarding && (
            <Button
              type="button"
              size="sm"
              variant="ghost"
              onClick={onToggleFullscreen}
              className="size-11 sm:size-8 p-0 text-muted-foreground hover:text-foreground"
              title={isFullscreen ? "Exit fullscreen" : "Fullscreen view"}
              aria-label={isFullscreen ? "Exit fullscreen view" : "Enter fullscreen view"}
            >
              {isFullscreen ? (
                <Minimize2 className="size-4" aria-hidden="true" />
              ) : (
                <Maximize2 className="size-4" aria-hidden="true" />
              )}
            </Button>
          )}
        </div>
      </CardHeader>

      {/* Camera Video Stage */}
      <div className="relative bg-black flex-1 flex items-center justify-center overflow-hidden min-h-[240px] sm:min-h-[320px]">
        {isCameraErrorState && cameraError ? (
          <div className="w-full p-4 sm:p-6">
            <CameraErrorDisplay error={cameraError} onRetry={onRetryCamera} />
          </div>
        ) : (
          <CameraView
            videoRef={videoRef}
            cameraState={cameraState}
            previewMode={previewMode}
            mjpegUrl={mjpegUrl}
            isMirrored={isMirrored}
            framingMode="cover"
            viewOrientation="landscape"
            isFullscreen={isFullscreen}
            opticalQuality={opticalQuality}
            onStartClick={onStartCameraCheck}
          >
            {/* Step 1 Overlay: Positioning & Framing Check Badges (only when active & not calibrating) */}
            {isCameraActive && !isCalibrating && (
              <div className="absolute top-3.5 left-3.5 right-3.5 flex flex-wrap items-center justify-between gap-2 pointer-events-none z-20">
                <div className="px-3 py-1 rounded-full bg-black/75 backdrop-blur-md border border-white/15 text-xs text-white flex items-center gap-2">
                  <span
                    aria-hidden="true"
                    className={`size-2 rounded-full ${
                      isSuccess
                        ? "bg-emerald-400"
                        : isFaceVisible === true
                          ? "bg-emerald-400"
                          : isFaceVisible === false
                            ? "bg-amber-400"
                            : "bg-sky-400"
                    }`}
                  />
                  <span className="font-medium">
                    {isSuccess
                      ? "Calibration complete — profile active"
                      : isFaceVisible === true
                        ? "Face detected — ready for calibration"
                        : isFaceVisible === false
                          ? "No face detected — center your face"
                          : backendProcessingStatus === "connecting" ||
                              backendProcessingStatus === "awaiting_telemetry"
                            ? "Connecting to AI engine…"
                            : "Camera active — checking framing"}
                  </span>
                </div>

                {opticalQuality && (
                  <div className="px-2.5 py-1 rounded-full bg-black/75 backdrop-blur-md border border-white/15 text-[11px] font-mono text-zinc-200 flex items-center gap-1.5">
                    <span
                      aria-hidden="true"
                      className={`size-1.5 rounded-full ${
                        opticalQuality.overallQuality === "EXCELLENT" ||
                        opticalQuality.overallQuality === "ACCEPTABLE"
                          ? "bg-emerald-400"
                          : "bg-amber-400"
                      }`}
                    />
                    <span>
                      {opticalQuality.lightingCondition === "UNDEREXPOSED"
                        ? "Low light"
                        : opticalQuality.lightingCondition === "OVEREXPOSED"
                          ? "Too bright"
                          : opticalQuality.blurCondition === "BLURRY"
                            ? "Blurry lens"
                            : "Lighting good"}
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* Step 2 Overlay: Active 4-Zone HUD Banner */}
            {isCameraActive && isCalibrating && calState && !isSuccess && (
              <div className="absolute top-3.5 left-3.5 right-3.5 flex flex-wrap items-start justify-between gap-2 pointer-events-none z-20">
                <div className="px-3 py-1.5 rounded-lg bg-black/80 backdrop-blur-md border border-white/15 text-xs text-white flex items-center gap-2">
                  <span className="font-mono text-zinc-300 tabular-nums">
                    Zone {calState.phase_index}/{calState.total_phases}:
                  </span>
                  <strong className="text-primary font-semibold">
                    {formatZoneLabel(calState.current_zone)}
                  </strong>
                </div>

                <Badge
                  variant={calState.is_head_in_target_zone ? "outline" : "destructive"}
                  className={`text-[11px] font-mono shadow-md ${
                    calState.is_head_in_target_zone
                      ? "border-emerald-500/60 text-emerald-400 bg-black/80"
                      : "bg-rose-600/90 text-white"
                  }`}
                >
                  {isFaceVisible === false
                    ? "NO FACE DETECTED"
                    : calState.is_head_in_target_zone
                      ? "ALIGNED · CAPTURING"
                      : "TURN HEAD TO TARGET"}
                </Badge>
              </div>
            )}

            {/* Fullscreen Calibration Companion Bar (keeps instructions & controls usable in fullscreen) */}
            {isFullscreen && (
              <div className="absolute left-[max(1rem,env(safe-area-inset-left,0px))] right-[max(1rem,env(safe-area-inset-right,0px))] bottom-[max(1rem,env(safe-area-inset-bottom,0px))] z-30 rounded-2xl bg-black/85 backdrop-blur-md border border-white/15 p-3.5 text-white space-y-2.5">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-semibold truncate">
                      {isCalibrating
                        ? activeZoneMeta.activePrompt
                        : isSuccess
                          ? "4-Zone profile saved!"
                          : "Position your face inside the reticle"}
                    </p>
                    {isCalibrating && calState && (
                      <p className="text-xs text-zinc-300 truncate">
                        {calState.stability_feedback || calState.instruction}
                      </p>
                    )}
                  </div>
                  <div className="flex flex-wrap items-center gap-2 shrink-0">
                    {isCalibrating && (
                      <>
                        <Button
                          type="button"
                          size="sm"
                          variant="outline"
                          onClick={onSkipZone}
                          disabled={isSavingProfile}
                          className="min-h-11 sm:h-8 sm:min-h-0 text-xs border-white/25 bg-white/10 text-white hover:bg-white/20"
                        >
                          <FastForward className="size-3.5 mr-1" aria-hidden="true" />
                          Skip Zone
                        </Button>
                        <Button
                          type="button"
                          size="sm"
                          variant="destructive"
                          onClick={onStopCalibration}
                          className="min-h-11 sm:h-8 sm:min-h-0 text-xs"
                        >
                          <Square className="size-3.5 mr-1" aria-hidden="true" />
                          Cancel
                        </Button>
                      </>
                    )}
                    <Button
                      type="button"
                      size="sm"
                      variant="secondary"
                      onClick={onToggleFullscreen}
                      className="min-h-11 sm:h-8 sm:min-h-0 text-xs"
                    >
                      <Minimize2 className="size-3.5 mr-1" aria-hidden="true" />
                      Exit Fullscreen
                    </Button>
                  </div>
                </div>
                {isCalibrating && calState && (
                  <Progress
                    value={calState.zone_progress_pct || 0}
                    aria-label="Current zone capture progress"
                    className="h-1.5 bg-white/20"
                  />
                )}
              </div>
            )}
          </CameraView>
        )}
      </div>

      {/* Camera Positioning & Device Controls Toolbar */}
      {!isFullscreen && (
        <div className="p-3 border-t border-border bg-muted/20 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5">
          <div className="flex-1 min-w-0">
            <CameraSelector
              devices={devices}
              selectedDeviceId={selectedDeviceId}
              onSelectDevice={onSelectDevice}
              onRefreshDevices={onRefreshDevices}
              disabled={isCalibrating}
            />
          </div>

          <div className="flex items-center gap-1.5 shrink-0">
            <Button
              type="button"
              size="icon"
              variant="outline"
              onClick={onSwitchFacing}
              disabled={isCalibrating}
              className="min-h-[44px] min-w-[44px] rounded-xl border-border"
              title="Switch front or rear camera"
              aria-label="Switch between front and rear camera"
            >
              <SwitchCamera className="size-4" aria-hidden="true" />
            </Button>

            <Button
              type="button"
              size="icon"
              variant="outline"
              onClick={onToggleMirror}
              aria-pressed={isMirrored}
              className={`min-h-[44px] min-w-[44px] rounded-xl border-border ${
                isMirrored ? "bg-primary/10 border-primary/40 text-primary" : ""
              }`}
              title={isMirrored ? "Disable mirror preview" : "Enable mirror preview"}
              aria-label={isMirrored ? "Disable mirror preview" : "Enable mirror preview"}
            >
              <FlipHorizontal className="size-4" aria-hidden="true" />
            </Button>

            {isCameraActive && !isCalibrating && (
              <Button
                type="button"
                size="sm"
                variant="outline"
                onClick={onStopCamera}
                className="min-h-[44px] rounded-xl border-border text-xs gap-1.5 px-3"
                title="Turn off camera preview"
                aria-label="Turn off camera preview"
              >
                <PowerOff className="size-3.5" aria-hidden="true" />
                <span className="hidden sm:inline">Turn Off Camera</span>
              </Button>
            )}

            {isSuccess && !isCalibrating && (
              <span className="inline-flex items-center gap-1 text-xs font-medium text-emerald-600 dark:text-emerald-400 pl-1">
                <CheckCircle2 className="size-3.5" aria-hidden="true" />
                <span>Saved</span>
              </span>
            )}
          </div>
        </div>
      )}
    </Card>
  );
}
