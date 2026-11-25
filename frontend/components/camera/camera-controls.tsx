"use client";

import React from "react";
import { CameraState } from "@/types/camera";
import { Button } from "@/components/ui/button";
import {
  Camera,
  Play,
  Square,
  Pause,
  RefreshCw,
  Maximize2,
  Minimize2,
  SwitchCamera,
  Crop,
  FlipHorizontal,
  Volume2,
  VolumeX,
} from "lucide-react";

interface CameraControlsProps {
  cameraState: CameraState;
  isMirrored: boolean;
  framingMode: "cover" | "fit";
  isFullscreen: boolean;
  onStart: () => void;
  onStop: () => void;
  onPause: () => void;
  onResume: () => void;
  onSwitchFacing: () => void;
  onCaptureSnapshot: () => void;
  onToggleFraming: () => void;
  onToggleFullscreen: () => void;
  onToggleMirror: () => void;
  isCapturing?: boolean;
  audioStatus?: "ready" | "muted" | "blocked" | "unavailable";
  isMuted?: boolean;
  onToggleMute?: () => void;
  onUnlockAudio?: () => void;
}

export const CameraControls = React.memo(function CameraControls({
  cameraState,
  isMirrored,
  framingMode,
  isFullscreen,
  onStart,
  onStop,
  onPause,
  onResume,
  onSwitchFacing,
  onCaptureSnapshot,
  onToggleFraming,
  onToggleFullscreen,
  onToggleMirror,
  isCapturing = false,
  audioStatus,
  isMuted = false,
  onToggleMute,
  onUnlockAudio,
}: CameraControlsProps) {
  const isActive = cameraState === "active";
  const isPaused = cameraState === "paused";
  const isRunning = isActive || isPaused;
  const isLoading =
    cameraState === "requesting-permission" || cameraState === "permission-granted";

  return (
    <div className="flex flex-wrap items-center justify-between gap-2.5 p-2.5 sm:p-3 rounded-2xl bg-card border border-border/80 shadow-xs">
      {/* Primary Action Controls (Start/Stop, Pause, Audio Mute) */}
      <div className="flex flex-wrap items-center gap-2">
        {!isRunning ? (
          <Button
            type="button"
            size="default"
            onClick={onStart}
            disabled={isLoading}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-4 sm:px-5 rounded-xl min-h-11 shadow-xs flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <RefreshCw className="size-4 animate-spin motion-reduce:animate-none" aria-hidden="true" />
                <span>Connecting…</span>
              </>
            ) : (
              <>
                <Play className="size-4 fill-current" aria-hidden="true" />
                <span>Start Monitoring</span>
              </>
            )}
          </Button>
        ) : (
          <Button
            type="button"
            size="default"
            variant="destructive"
            onClick={onStop}
            className="font-semibold px-4 sm:px-5 rounded-xl min-h-11 shadow-xs flex items-center gap-2"
          >
            <Square className="size-4 fill-current" aria-hidden="true" />
            <span>Stop</span>
          </Button>
        )}

        {/* Pause / Resume Button */}
        {isRunning && (
          <Button
            type="button"
            size="icon"
            variant="outline"
            onClick={isPaused ? onResume : onPause}
            className="size-11 rounded-xl border-border"
            title={isPaused ? "Resume video feed" : "Pause video feed"}
            aria-label={isPaused ? "Resume camera stream" : "Pause camera stream"}
          >
            {isPaused ? (
              <Play className="size-4 text-emerald-500 fill-emerald-500" aria-hidden="true" />
            ) : (
              <Pause className="size-4 text-amber-500" aria-hidden="true" />
            )}
          </Button>
        )}

        {/* Unified Audio Mute / Unlock Toggle */}
        {audioStatus && (onToggleMute || onUnlockAudio) && (
          <Button
            type="button"
            size="icon"
            variant={audioStatus === "blocked" ? "secondary" : "outline"}
            onClick={() => {
              if (audioStatus === "blocked" && onUnlockAudio) {
                onUnlockAudio();
              } else if (onToggleMute) {
                onToggleMute();
              }
            }}
            className={`size-11 rounded-xl border-border ${
              audioStatus === "blocked"
                ? "border-amber-500/40 bg-amber-500/10 text-amber-500"
                : isMuted
                  ? "border-rose-500/35 bg-rose-500/10 text-rose-500"
                  : ""
            }`}
            title={
              audioStatus === "blocked"
                ? "Enable alert audio"
                : isMuted
                  ? "Unmute alert audio"
                  : "Mute alert audio"
            }
            aria-label={
              audioStatus === "blocked"
                ? "Enable alert audio"
                : isMuted
                  ? "Unmute alert audio"
                  : "Mute alert audio"
            }
          >
            {audioStatus === "blocked" || isMuted ? (
              <VolumeX className="size-4" aria-hidden="true" />
            ) : (
              <Volume2 className="size-4 text-emerald-500" aria-hidden="true" />
            )}
          </Button>
        )}
      </div>

      {/* Auxiliary Camera Utilities */}
      <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
        <Button
          type="button"
          size="icon"
          variant="secondary"
          onClick={onCaptureSnapshot}
          disabled={!isActive || isCapturing}
          className="size-11 rounded-xl border border-border bg-secondary/80 hover:bg-secondary"
          title="Capture frame snapshot (JPEG)"
          aria-label="Capture current frame snapshot"
        >
          {isCapturing ? (
            <RefreshCw className="size-4 animate-spin motion-reduce:animate-none text-primary" aria-hidden="true" />
          ) : (
            <Camera className="size-4 text-foreground" aria-hidden="true" />
          )}
        </Button>

        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={onSwitchFacing}
          className="size-11 rounded-xl border-border"
          title="Switch front or rear camera"
          aria-label="Switch between front and rear facing camera"
        >
          <SwitchCamera className="size-4 text-foreground" aria-hidden="true" />
        </Button>

        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={onToggleMirror}
          aria-pressed={isMirrored}
          className={`size-11 rounded-xl border-border ${
            isMirrored ? "bg-primary/10 border-primary/40 text-primary" : ""
          }`}
          title={isMirrored ? "Disable mirror view" : "Enable mirror view"}
          aria-label={isMirrored ? "Disable camera mirror mode" : "Enable camera mirror mode"}
        >
          <FlipHorizontal className="size-4" aria-hidden="true" />
        </Button>

        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={onToggleFraming}
          aria-pressed={framingMode === "cover"}
          className={`size-11 rounded-xl border-border ${
            framingMode === "cover" ? "bg-primary/10 border-primary/40 text-primary" : ""
          }`}
          title={framingMode === "cover" ? "Switch to uncropped fit" : "Switch to fill viewport"}
          aria-label={framingMode === "cover" ? "Switch to uncropped fit view" : "Switch to viewport cover view"}
        >
          <Crop className="size-4" aria-hidden="true" />
        </Button>

        <Button
          type="button"
          size="icon"
          variant="outline"
          onClick={onToggleFullscreen}
          className="size-11 rounded-xl border-border"
          title={isFullscreen ? "Exit fullscreen" : "Enter fullscreen"}
          aria-label={isFullscreen ? "Exit fullscreen monitor" : "Enter fullscreen monitor"}
        >
          {isFullscreen ? (
            <Minimize2 className="size-4" aria-hidden="true" />
          ) : (
            <Maximize2 className="size-4" aria-hidden="true" />
          )}
        </Button>
      </div>
    </div>
  );
});
