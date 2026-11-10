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
  Sparkles,
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
}

export function CameraControls({
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
}: CameraControlsProps) {
  const isActive = cameraState === "active";
  const isPaused = cameraState === "paused";
  const isRunning = isActive || isPaused;
  const isLoading =
    cameraState === "requesting-permission" || cameraState === "permission-granted";

  return (
    <div className="flex flex-wrap items-center justify-between gap-2.5 p-3 rounded-2xl bg-card border border-border shadow-md">
      {/* Primary Action Button (Start / Stop) */}
      <div className="flex items-center gap-2">
        {!isRunning ? (
          <Button
            size="default"
            onClick={onStart}
            disabled={isLoading}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold px-5 rounded-xl min-h-[44px] shadow-sm flex items-center gap-2"
          >
            {isLoading ? (
              <>
                <RefreshCw className="size-4 animate-spin" />
                <span>Connecting…</span>
              </>
            ) : (
              <>
                <Play className="size-4 fill-current" />
                <span>Start Camera</span>
              </>
            )}
          </Button>
        ) : (
          <Button
            size="default"
            variant="destructive"
            onClick={onStop}
            className="font-semibold px-5 rounded-xl min-h-[44px] shadow-sm flex items-center gap-2"
          >
            <Square className="size-4 fill-current" />
            <span>Stop Camera</span>
          </Button>
        )}

        {/* Pause / Resume Button (When active) */}
        {isRunning && (
          <Button
            size="icon"
            variant="outline"
            onClick={isPaused ? onResume : onPause}
            className="min-h-[44px] min-w-[44px] rounded-xl border-border"
            title={isPaused ? "Resume video feed" : "Pause video feed"}
            aria-label={isPaused ? "Resume camera stream" : "Pause camera stream"}
          >
            {isPaused ? (
              <Play className="size-4 text-emerald-500 fill-emerald-500" />
            ) : (
              <Pause className="size-4 text-amber-500" />
            )}
          </Button>
        )}
      </div>

      {/* Auxiliary Controls (Snapshot, Flip, Framing, Fullscreen) */}
      <div className="flex items-center gap-1.5 sm:gap-2">
        {/* Snapshot Capture Button */}
        <Button
          size="icon"
          variant="secondary"
          onClick={onCaptureSnapshot}
          disabled={!isActive || isCapturing}
          className="min-h-[44px] min-w-[44px] rounded-xl border border-border bg-secondary/80 hover:bg-secondary relative"
          title="Capture frame snapshot (JPEG)"
          aria-label="Capture current frame snapshot"
        >
          {isCapturing ? (
            <RefreshCw className="size-4 animate-spin text-primary" />
          ) : (
            <Camera className="size-4 text-foreground" />
          )}
        </Button>

        {/* Switch Front / Back Camera */}
        <Button
          size="icon"
          variant="outline"
          onClick={onSwitchFacing}
          className="min-h-[44px] min-w-[44px] rounded-xl border-border"
          title="Switch front / rear camera"
          aria-label="Switch between front and rear facing camera"
        >
          <SwitchCamera className="size-4 text-foreground" />
        </Button>

        {/* Mirror Toggle */}
        <Button
          size="icon"
          variant="outline"
          onClick={onToggleMirror}
          className={`min-h-[44px] min-w-[44px] rounded-xl border-border ${
            isMirrored ? "bg-primary/10 border-primary/40 text-primary" : ""
          }`}
          title={isMirrored ? "Disable mirror view" : "Enable mirror view"}
          aria-label={isMirrored ? "Disable camera mirror mode" : "Enable camera mirror mode"}
        >
          <FlipHorizontal className="size-4" />
        </Button>

        {/* Framing Mode Toggle (Cover vs Fit) */}
        <Button
          size="icon"
          variant="outline"
          onClick={onToggleFraming}
          className={`min-h-[44px] min-w-[44px] rounded-xl border-border ${
            framingMode === "cover" ? "bg-primary/10 border-primary/40 text-primary" : ""
          }`}
          title={framingMode === "cover" ? "Switch to Wide Fit (No crop)" : "Switch to Fullscreen Cover"}
          aria-label={framingMode === "cover" ? "Switch to wide aspect fit view" : "Switch to viewport cover view"}
        >
          <Crop className="size-4" />
        </Button>

        {/* Fullscreen Toggle */}
        <Button
          size="icon"
          variant="outline"
          onClick={onToggleFullscreen}
          className="min-h-[44px] min-w-[44px] rounded-xl border-border"
          title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
          aria-label={isFullscreen ? "Exit fullscreen monitor" : "Enter fullscreen monitor"}
        >
          {isFullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
        </Button>
      </div>
    </div>
  );
}
