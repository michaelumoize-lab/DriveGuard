"use client";

import React from "react";
import { CameraState, OpticalFrameQuality, PositioningGuidance } from "@/types/camera";
import { AlertState } from "@/types/telemetry";
import { Camera, Eye, Pause, ShieldCheck, Crosshair } from "lucide-react";

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraState: CameraState;
  previewMode?: "video" | "mjpeg";
  mjpegUrl?: string | null;
  alertState?: AlertState;
  isStale?: boolean;
  isMirrored: boolean;
  framingMode: "cover" | "fit";
  viewOrientation: "portrait" | "landscape";
  isFullscreen?: boolean;
  opticalQuality?: OpticalFrameQuality | null;
  positioningGuidance?: PositioningGuidance | null;
  onStartClick?: () => void;
  children?: React.ReactNode;
}

export function CameraView({
  videoRef,
  cameraState,
  previewMode = "video",
  mjpegUrl = null,
  alertState = "NORMAL",
  isStale = false,
  isMirrored,
  framingMode,
  viewOrientation,
  isFullscreen = false,
  opticalQuality,
  positioningGuidance,
  onStartClick,
  children,
}: CameraViewProps) {
  const isPortrait = viewOrientation === "portrait";
  const isActive = cameraState === "active";
  const isPaused = cameraState === "paused";
  const isLoading =
    cameraState === "requesting-permission" || cameraState === "permission-granted";
  const isIdle =
    cameraState === "not-initialized" || cameraState === "stopped";

  const reticleBorderColor = isStale
    ? "border-purple-400/80 border-dashed"
    : alertState === "CRITICAL"
      ? "border-rose-500"
      : alertState === "WARNING"
        ? "border-amber-400"
        : alertState === "MONITORING_UNAVAILABLE"
          ? "border-slate-400/80 border-dashed"
          : "border-emerald-500/70";

  const crosshairBgColor = isStale
    ? "bg-purple-400"
    : alertState === "CRITICAL"
      ? "bg-rose-500"
      : alertState === "WARNING"
        ? "bg-amber-400"
        : alertState === "MONITORING_UNAVAILABLE"
          ? "bg-slate-400"
          : "bg-emerald-400";

  return (
    <div
      className={`relative bg-black rounded-2xl overflow-hidden border border-border/80 shadow-xl transition-colors duration-200 flex items-center justify-center ${
        isFullscreen
          ? "fixed inset-0 z-50 rounded-none border-0 w-screen h-[100dvh]"
          : isPortrait
          ? "w-full max-w-[420px] mx-auto aspect-[9/16] min-h-[480px] sm:min-h-[540px]"
          : "w-full aspect-video min-h-[360px]"
      }`}
    >
      {/* Native HTML5 Video Element (Browser / LiveKit WebRTC) OR Loopback MJPEG Feed (Python Local) */}
      {previewMode === "mjpeg" ? (
        mjpegUrl && isActive ? (
          // eslint-disable-next-line @next/next/no-img-element
          <img
            src={mjpegUrl}
            alt="Python Local AI live camera stream"
            className={`w-full h-full select-none ${
              framingMode === "cover"
                ? "object-cover object-[center_35%]"
                : "object-contain"
            }`}
          />
        ) : (
          <div className="w-full h-full bg-black" aria-hidden="true" />
        )
      ) : (
        <video
          ref={videoRef}
          playsInline
          autoPlay
          muted
          className={`w-full h-full transition-transform duration-200 select-none ${
            isMirrored ? "scale-x-[-1]" : ""
          } ${
            framingMode === "cover"
              ? "object-cover object-[center_35%]"
              : "object-contain"
          }`}
        />
      )}

      {/* Cockpit HUD Target Reticle & Corner Accents (Active Feed) */}
      {isActive && (
        <div className="absolute inset-0 pointer-events-none p-4 flex flex-col justify-between">
          <div className="flex justify-between items-start">
            <div className={`w-6 h-6 border-t-2 border-l-2 ${reticleBorderColor} rounded-tl-sm`} />
            <div className={`w-6 h-6 border-t-2 border-r-2 ${reticleBorderColor} rounded-tr-sm`} />
          </div>

          {/* Central Eye-Level Driver Focus Guide (Subtle) */}
          <div className="self-center flex flex-col items-center gap-1 opacity-25">
            <div className={`w-12 h-0.5 ${crosshairBgColor}`} />
            <div className={`h-12 w-0.5 ${crosshairBgColor}`} />
          </div>

          {/* Positioning Guidance Pill if not optimal */}
          {positioningGuidance && positioningGuidance.hasFace && !positioningGuidance.isOptimal && (
            <div className="self-center pointer-events-auto bg-black/85 backdrop-blur-md px-3 py-1.5 rounded-full border border-amber-500/50 text-amber-300 text-[11px] font-mono flex items-center gap-1.5 shadow-lg">
              <Crosshair className="size-3.5 shrink-0" aria-hidden="true" />
              <span>
                {positioningGuidance.distanceStatus !== "PERFECT"
                  ? positioningGuidance.distanceFeedback
                  : positioningGuidance.centeringStatus !== "CENTERED"
                  ? positioningGuidance.centeringFeedback
                  : positioningGuidance.angleFeedback}
              </span>
            </div>
          )}

          {/* Optical Environment Telemetry Tag (Default bottom-left positioning if no children overlay is provided) */}
          {opticalQuality && !children && (
            <div
              className={`absolute ${
                isFullscreen ? "bottom-16 left-4" : "bottom-3 left-4"
              } pointer-events-auto flex items-center gap-1.5 bg-black/75 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/15 text-[10px] font-mono text-zinc-300 shadow-md`}
            >
              <span
                className={`size-1.5 rounded-full ${
                  opticalQuality.overallQuality === "EXCELLENT" || opticalQuality.overallQuality === "ACCEPTABLE"
                    ? "bg-emerald-400"
                    : "bg-amber-400"
                }`}
              />
              <span>LUM: {Math.round(opticalQuality.meanLuminance)}</span>
              <span className="text-zinc-500">•</span>
              <span className={opticalQuality.blurCondition === "BLURRY" ? "text-rose-400" : "text-zinc-300"}>
                {opticalQuality.blurCondition} ({Math.round(opticalQuality.blurVarianceScore)})
              </span>
            </div>
          )}

          <div className="flex justify-between items-end">
            <div className={`w-6 h-6 border-b-2 border-l-2 ${reticleBorderColor} rounded-bl-sm`} />
            <div className={`w-6 h-6 border-b-2 border-r-2 ${reticleBorderColor} rounded-br-sm`} />
          </div>
        </div>
      )}

      {/* Paused State Overlay */}
      {isPaused && (
        <div className="absolute inset-0 bg-black/75 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center z-20">
          <div className="p-3.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 mb-3">
            <Pause className="size-8" aria-hidden="true" />
          </div>
          <p className="text-base font-semibold text-white">Monitoring Paused</p>
          <p className="text-xs text-muted-foreground mt-1 max-w-xs">
            Camera stream is paused. Select Resume to continue monitoring.
          </p>
        </div>
      )}

      {/* Loading & Requesting Permission Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-background/90 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-20 space-y-3">
          <div className="relative">
            <div className="size-12 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
            <Camera className="size-5 text-primary absolute inset-0 m-auto" aria-hidden="true" />
          </div>
          <div>
            <p className="text-sm font-semibold text-foreground">
              {previewMode === "mjpeg"
                ? "Connecting to local Python OpenCV camera…"
                : cameraState === "requesting-permission"
                  ? "Requesting camera permission…"
                  : "Starting video stream…"}
            </p>
            <p className="text-xs text-muted-foreground mt-1 max-w-xs">
              {previewMode === "mjpeg"
                ? "Python AI is acquiring the local webcam directly via http://127.0.0.1:8000."
                : "Click “Allow” in your browser prompt to grant webcam access."}
            </p>
          </div>
        </div>
      )}

      {/* Idle / Stopped Standby Overlay */}
      {isIdle && (
        <div className="absolute inset-0 bg-background/95 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center z-10 space-y-4">
          <div className="p-4 rounded-2xl bg-primary/10 border border-primary/20 text-primary shadow-inner">
            <Camera className="size-9" aria-hidden="true" />
          </div>
          <div className="space-y-1 max-w-xs">
            <h3 className="font-bold text-foreground text-base tracking-tight">
              Driver Camera Ready
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {isPortrait
                ? "Portrait camera view ready for front-facing driver monitoring."
                : "Video stays private on your device by default with audio recording disabled."}
            </p>
          </div>

          {onStartClick && (
            <button
              type="button"
              onClick={onStartClick}
              className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-sm shadow-sm hover:bg-primary/90 transition-colors min-h-[44px] cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2"
            >
              <Eye className="size-4" aria-hidden="true" />
              <span>Start Camera</span>
            </button>
          )}

          <div className="flex items-center gap-2 text-[11px] text-muted-foreground">
            <ShieldCheck className="size-3.5 text-emerald-500" aria-hidden="true" />
            <span>Private feed · Microphone off</span>
          </div>
        </div>
      )}

      {/* HUD Overlays / Gauges passed as children */}
      {children}
    </div>
  );
}
