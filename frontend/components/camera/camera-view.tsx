"use client";

import React from "react";
import { CameraState } from "@/types/camera";
import { AlertState } from "@/types/telemetry";
import { Camera, Eye, Pause, ShieldCheck } from "lucide-react";

interface CameraViewProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  cameraState: CameraState;
  alertState?: AlertState;
  isStale?: boolean;
  isMirrored: boolean;
  framingMode: "cover" | "fit";
  viewOrientation: "portrait" | "landscape";
  isFullscreen?: boolean;
  onStartClick?: () => void;
  children?: React.ReactNode;
}

export function CameraView({
  videoRef,
  cameraState,
  alertState = "NORMAL",
  isStale = false,
  isMirrored,
  framingMode,
  viewOrientation,
  isFullscreen = false,
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
      className={`relative bg-black rounded-2xl overflow-hidden border border-border/80 shadow-2xl transition-all duration-300 flex items-center justify-center ${
        isFullscreen
          ? "fixed inset-0 z-50 rounded-none border-0 w-screen h-[100dvh]"
          : isPortrait
          ? "w-full max-w-[420px] mx-auto aspect-[9/16] min-h-[480px] sm:min-h-[540px]"
          : "w-full aspect-video min-h-[360px]"
      }`}
    >
      {/* Native HTML5 Video Element */}
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

          <div className="flex justify-between items-end">
            <div className={`w-6 h-6 border-b-2 border-l-2 ${reticleBorderColor} rounded-bl-sm`} />
            <div className={`w-6 h-6 border-b-2 border-r-2 ${reticleBorderColor} rounded-br-sm`} />
          </div>
        </div>
      )}

      {/* Paused State Overlay */}
      {isPaused && (
        <div className="absolute inset-0 bg-black/75 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center z-20">
          <div className="p-3.5 rounded-full bg-amber-500/20 text-amber-400 border border-amber-500/30 mb-3 animate-pulse">
            <Pause className="size-8" />
          </div>
          <p className="text-base font-semibold text-white">Monitoring Paused</p>
          <p className="text-xs text-muted-foreground mt-1 max-w-xs">
            Webcam stream tracks are frozen. Click Resume to re-enable camera feed.
          </p>
        </div>
      )}

      {/* Loading & Requesting Permission Overlay */}
      {isLoading && (
        <div className="absolute inset-0 bg-background/90 backdrop-blur-md flex flex-col items-center justify-center p-6 text-center z-20 space-y-3">
          <div className="relative">
            <div className="size-12 rounded-full border-2 border-primary/30 border-t-primary animate-spin" />
            <Camera className="size-5 text-primary absolute inset-0 m-auto" />
          </div>
          <div>
            <p className="text-sm font-semibold text-foreground">
              {cameraState === "requesting-permission"
                ? "Requesting Camera Permission..."
                : "Initializing Video Stream..."}
            </p>
            <p className="text-xs text-muted-foreground mt-1 max-w-xs">
              Please click &quot;Allow&quot; in the browser prompt to grant webcam access.
            </p>
          </div>
        </div>
      )}

      {/* Idle / Stopped Standby Overlay */}
      {isIdle && (
        <div className="absolute inset-0 bg-background/95 backdrop-blur-xs flex flex-col items-center justify-center p-6 text-center z-10 space-y-4">
          <div className="p-4 rounded-2xl bg-primary/10 border border-primary/20 text-primary shadow-inner">
            <Camera className="size-10" />
          </div>
          <div className="space-y-1 max-w-xs">
            <h3 className="font-bold text-foreground text-base tracking-tight">
              DriveGuard Cabin Telemetry Feed
            </h3>
            <p className="text-xs text-muted-foreground leading-relaxed">
              {isPortrait
                ? "Full-bleed mobile portrait monitoring. Access front camera for driver facial analysis."
                : "Real-time WebRTC camera stream for personalised 4-zone driver drowsiness monitoring."}
            </p>
          </div>

          {onStartClick && (
            <button
              onClick={onStartClick}
              className="inline-flex items-center justify-center gap-2 px-6 py-2.5 rounded-xl bg-primary text-primary-foreground font-semibold text-sm shadow-lg hover:bg-primary/90 transition-all min-h-[44px] cursor-pointer"
            >
              <Eye className="size-4" />
              <span>Start Camera Stream</span>
            </button>
          )}

          <div className="flex items-center gap-2 text-[11px] text-muted-foreground/75 font-mono">
            <ShieldCheck className="size-3.5 text-emerald-500" />
            <span>Local WebRTC • Audio Disabled • Privacy Secured</span>
          </div>
        </div>
      )}

      {/* HUD Overlays / Gauges passed as children */}
      {children}
    </div>
  );
}
