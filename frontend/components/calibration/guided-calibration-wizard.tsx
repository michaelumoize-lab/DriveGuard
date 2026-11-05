"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useLivekitTelemetryStream } from "@/hooks/use-livekit-telemetry-stream";
import { CameraView } from "@/components/camera/camera-view";
import { CalibrationPhase } from "@/types/telemetry";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Eye,
  Camera,
  Play,
  RotateCcw,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  ArrowDown,
  Navigation,
  Square,
  Wifi,
  FastForward,
  Maximize2,
  Minimize2,
} from "lucide-react";

export function GuidedCalibrationWizard() {
  const containerRef = useRef<HTMLDivElement>(null);

  // Layout & UI States (matching Live Monitor)
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("landscape");
  const [framingMode, setFramingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [calibrationMode, setCalibrationMode] = useState<"3_ZONE" | "4_ZONE">("3_ZONE");
  const [isSuccess, setIsSuccess] = useState(false);
  const [calState, setCalState] = useState<CalibrationPhase | null>(null);

  // Stable Session ID
  const stableSessionId = useMemo(
    () => `cal_${Math.random().toString(36).slice(2, 9)}`,
    []
  );

  // 1. Unified Camera Hook (Exact same camera engine as Live Monitor)
  const {
    videoRef,
    cameraState,
    isMirrored,
    setIsMirrored,
    startCamera,
    stopCamera,
  } = useCamera({
    idealWidth: 1280,
    idealHeight: 720,
    idealFps: 30,
    defaultFacingMode: "user",
    mirrored: true,
  });

  // 2. Real-Time LiveKit WebRTC Streaming & Calibration Hook
  const {
    isConnected: isLivekitConnected,
    connectionStatus: livekitStatus,
    calibration: livekitCalibration,
    startStreamingLoop,
    stopStreamingLoop,
    startCalibration: triggerLivekitCalibration,
    cancelCalibration: triggerLivekitCancel,
    skipCalibrationStep: triggerLivekitSkip,
    backendType,
    switchBackend,
  } = useLivekitTelemetryStream({
    sessionId: stableSessionId,
    driverId: "michael_driver",
    onCalibration: (cal) => {
      setCalState(cal);
      if (cal.is_completed) {
        setIsSuccess(true);
        toast.success("Calibration Completed Successfully! Personal baselines activated.");

        // Persist calibrated baseline to PostgreSQL
        const prof = (cal as unknown as { profile?: Record<string, unknown> })?.profile;
        fetch("/api/calibration/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            calibrationMode,
            centerEarMedian: (prof?.center_ear_median as number) ?? 0.285,
            centerEarMadScaled: (prof?.center_ear_mad_scaled as number) ?? 0.018,
            blinkDurationMedianMs: (prof?.blink_duration_median_ms as number) ?? 185.0,
            blinkDurationMadMs: (prof?.blink_duration_mad_ms as number) ?? 35.0,
            blinkRcrMedian: (prof?.blink_rcr_median as number) ?? 0.92,
            blinkRcrMad: (prof?.blink_rcr_mad as number) ?? 0.14,
            neutralPitchDeg: (prof?.neutral_pitch_deg as number) ?? 0.0,
            neutralYawDeg: (prof?.neutral_yaw_deg as number) ?? 0.0,
            calibrationReliability: (prof?.calibration_reliability as number) ?? 0.90,
            zoneBaselines: prof?.zone_baselines ?? cal,
          }),
        }).catch((e) => console.warn("Could not save profile to DB:", e));
      }
    },
  });

  // Keep calState in sync with incoming LiveKit calibration data
  useEffect(() => {
    if (livekitCalibration) {
      setCalState(livekitCalibration);
      if (livekitCalibration.is_completed) {
        setIsSuccess(true);
      }
    }
  }, [livekitCalibration]);

  // Reactive Video Streaming: Link camera activation with WebRTC publisher
  useEffect(() => {
    if (cameraState === "active" && videoRef.current) {
      startStreamingLoop(videoRef.current, isMirrored);
    } else if (cameraState === "stopped") {
      stopStreamingLoop();
    }
  }, [cameraState, isMirrored, startStreamingLoop, stopStreamingLoop, videoRef]);

  // Fullscreen Handler
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const toggleFullscreen = async () => {
    try {
      if (!document.fullscreenElement) {
        if (containerRef.current) await containerRef.current.requestFullscreen();
      } else {
        await document.exitFullscreen();
      }
    } catch {
      toast.error("Could not toggle fullscreen.");
    }
  };

  // User Actions
  const handleStartCalibration = async () => {
    setIsSuccess(false);
    setCalState(null);

    if (cameraState !== "active") {
      await startCamera();
    }

    const isFourZone = calibrationMode === "4_ZONE";
    triggerLivekitCalibration(isFourZone, 5.0);
    toast.info(`Starting ${isFourZone ? "4-Zone" : "3-Zone"} Guided Baseline Calibration...`);
  };

  const handleStopCalibration = () => {
    triggerLivekitCancel();
    stopCamera();
    stopStreamingLoop();
    setCalState(null);
  };

  const handleSkipZone = () => {
    triggerLivekitSkip();
    toast.info("Advancing past current spatial zone...");
  };

  // Directional guidance helpers
  const currentZoneInfo = useMemo(() => {
    const zone = calState?.current_zone || "CENTER";
    switch (zone) {
      case "LEFT_MIRROR":
        return {
          label: "TURN HEAD LEFT (DRIVER MIRROR)",
          icon: <ArrowLeft className="size-6 text-blue-400 animate-pulse" />,
          color: "text-blue-400",
        };
      case "RIGHT_MIRROR":
        return {
          label: "TURN HEAD RIGHT (PASSENGER MIRROR)",
          icon: <ArrowRight className="size-6 text-amber-400 animate-pulse" />,
          color: "text-amber-400",
        };
      case "DASHBOARD":
        return {
          label: "GLANCE DOWN (DASHBOARD)",
          icon: <ArrowDown className="size-6 text-purple-400 animate-pulse" />,
          color: "text-purple-400",
        };
      case "CENTER":
      default:
        return {
          label: "LOOK STRAIGHT AHEAD (CENTER ROAD)",
          icon: <Navigation className="size-6 text-emerald-400" />,
          color: "text-emerald-400",
        };
    }
  }, [calState?.current_zone]);

  const isCalibrating = cameraState === "active" && isLivekitConnected && Boolean(calState && !calState.is_completed);

  return (
    <div
      ref={containerRef}
      className={`space-y-6 transition-all ${
        isFullscreen ? "fixed inset-0 z-50 bg-black p-4 flex flex-col justify-between overflow-hidden" : ""
      }`}
    >
      <div className={`grid gap-6 ${isFullscreen ? "h-full grid-cols-1" : "grid-cols-1 lg:grid-cols-3"}`}>
        {/* Video Viewport (Using Exact Same CameraView as Live Monitor) */}
        <Card
          className={`overflow-hidden border-border/80 flex flex-col ${
            isFullscreen ? "border-0 rounded-none bg-black h-full" : "lg:col-span-2"
          }`}
        >
          <CardHeader className="py-3 px-4 flex flex-row items-center justify-between border-b border-border">
            <div className="flex items-center gap-2">
              <Camera className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">Webcam Spatial Alignment</CardTitle>
            </div>

            <div className="flex items-center gap-2">
              {/* Target Inference Selector */}
              <div className="flex items-center bg-muted/60 rounded-md p-0.5 text-[11px] font-mono border border-border">
                <button
                  type="button"
                  onClick={() => switchBackend("local")}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    backendType === "local"
                      ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="Local AI Engine (127.0.0.1:8000)"
                >
                  ⚡ Local
                </button>
                <button
                  type="button"
                  onClick={() => switchBackend("cloud")}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    backendType === "cloud"
                      ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="LiveKit Cloud SFU"
                >
                  ☁️ LiveKit Cloud
                </button>
              </div>

              <Badge
                variant="outline"
                className={`text-xs font-mono flex items-center gap-1 ${
                  isCalibrating
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : "text-muted-foreground"
                }`}
              >
                {isCalibrating ? (
                  <>
                    <Wifi className="size-3" />
                    CALIBRATING
                  </>
                ) : (
                  "STANDBY"
                )}
              </Badge>

              <Button
                size="sm"
                variant="ghost"
                onClick={toggleFullscreen}
                className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                title={isFullscreen ? "Exit Fullscreen" : "Fullscreen HUD"}
              >
                {isFullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
              </Button>
            </div>
          </CardHeader>

          {/* Consistent CameraView Component */}
          <div className="relative bg-black flex-1 flex items-center justify-center overflow-hidden">
            <CameraView
              videoRef={videoRef}
              cameraState={cameraState}
              isMirrored={isMirrored}
              framingMode={framingMode}
              viewOrientation={viewOrientation}
              isFullscreen={isFullscreen}
              onStartClick={handleStartCalibration}
            >
              {/* Standby Setup Overlay */}
              {cameraState !== "active" && !isSuccess && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-4 z-20">
                  <div className="p-3 rounded-full bg-primary/10 text-primary">
                    <Eye className="size-8" />
                  </div>
                  <div className="font-semibold text-foreground text-sm">
                    Personalised Baseline Calibration
                  </div>
                  <p className="text-xs text-muted-foreground max-w-md">
                    Establish custom EAR open-eye baselines across spatial zones. Progress advances automatically when your head is properly aligned with each target zone.
                  </p>

                  {/* Mode Selector */}
                  <div className="flex items-center gap-2 p-1 bg-muted/60 rounded-xl border border-border text-xs">
                    <button
                      type="button"
                      onClick={() => setCalibrationMode("3_ZONE")}
                      className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                        calibrationMode === "3_ZONE"
                          ? "bg-primary text-primary-foreground shadow-xs"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      3-Zone (Recommended)
                    </button>
                    <button
                      type="button"
                      onClick={() => setCalibrationMode("4_ZONE")}
                      className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                        calibrationMode === "4_ZONE"
                          ? "bg-primary text-primary-foreground shadow-xs"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      4-Zone (Includes Dashboard)
                    </button>
                  </div>

                  <p className="text-[11px] text-muted-foreground max-w-xs">
                    {calibrationMode === "3_ZONE"
                      ? "✨ Center, Left Mirror, and Right Mirror. Dashboard baseline is derived automatically to prevent eyelid occlusion stalls."
                      : "Center, Left Mirror, Right Mirror, and Dashboard."}
                  </p>

                  <Button size="sm" onClick={handleStartCalibration} className="bg-primary text-primary-foreground">
                    <Play className="size-4 mr-1.5" />
                    Start {calibrationMode === "3_ZONE" ? "3-Zone" : "4-Zone"} Calibration
                  </Button>
                </div>
              )}

              {/* In-Video Heads Up Status Banner */}
              {cameraState === "active" && calState && !isSuccess && (
                <div className="absolute top-4 left-4 right-4 flex items-start justify-between pointer-events-none z-20">
                  <div className="px-3 py-1 rounded-lg bg-black/80 backdrop-blur-md border border-white/15 text-xs text-white font-mono flex items-center gap-2">
                    <span>Phase {calState.phase_index}/{calState.total_phases}:</span>
                    <strong className="text-primary">{calState.current_zone}</strong>
                  </div>

                  <Badge
                    variant={calState.is_head_in_target_zone ? "outline" : "destructive"}
                    className={`text-[11px] font-mono shadow-md ${
                      calState.is_head_in_target_zone
                        ? "border-emerald-500/60 text-emerald-400 bg-black/80"
                        : "bg-rose-600/90 text-white"
                    }`}
                  >
                    {calState.is_head_in_target_zone ? "ALIGNED [ACQUIRING]" : "TURN HEAD TO TARGET"}
                  </Badge>
                </div>
              )}

              {/* Success Overlay */}
              {isSuccess && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-3 z-20">
                  <div className="p-3 rounded-full bg-emerald-500/10 text-emerald-500">
                    <CheckCircle2 className="size-10" />
                  </div>
                  <div className="font-bold text-foreground text-base">
                    Calibration Successfully Completed!
                  </div>
                  <p className="text-xs text-muted-foreground max-w-sm">
                    Your personalised baselines across spatial zones have been calculated and saved to your PostgreSQL profile.
                  </p>
                  <div className="flex items-center gap-2 pt-2">
                    <Button size="sm" onClick={handleStartCalibration} variant="outline">
                      <RotateCcw className="size-4 mr-1.5" />
                      Recalibrate
                    </Button>
                  </div>
                </div>
              )}
            </CameraView>
          </div>

          <CardFooter className="py-3 px-4 flex items-center justify-between border-t border-border bg-muted/20">
            <div className="flex items-center gap-2">
              {cameraState === "active" ? (
                <>
                  <Button size="sm" variant="destructive" onClick={handleStopCalibration}>
                    <Square className="size-3.5 mr-1.5" />
                    Stop Calibration
                  </Button>
                  <Button size="sm" variant="outline" onClick={handleSkipZone} title="Skip current zone if needed">
                    <FastForward className="size-3.5 mr-1.5 text-primary" />
                    Skip Zone
                  </Button>
                </>
              ) : (
                <Button size="sm" onClick={handleStartCalibration} className="bg-primary text-primary-foreground">
                  <Play className="size-3.5 mr-1.5" />
                  Start Calibration
                </Button>
              )}
            </div>
            <span className="text-xs text-muted-foreground font-mono">
              {calState ? `Phase ${calState.phase_index} of ${calState.total_phases}` : "Ready"}
            </span>
          </CardFooter>
        </Card>

        {/* Guidance & Target Alignment HUD */}
        {!isFullscreen && (
          <Card className="border-border/80">
            <CardHeader>
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Eye className="size-4 text-primary" />
                Guidance & Target Gating
              </CardTitle>
              <CardDescription className="text-xs">
                Follow the active directional instructions
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-xs">
              {/* Active Target Banner */}
              <div className="p-3.5 rounded-xl border border-primary/20 bg-primary/5 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10 shrink-0">
                  {currentZoneInfo.icon}
                </div>
                <div>
                  <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                    Target Gaze Zone
                  </span>
                  <span className="font-bold text-foreground text-sm">
                    {currentZoneInfo.label}
                  </span>
                </div>
              </div>

              {/* Spatial Alignment Status Indicator */}
              <div className="space-y-1.5">
                <div className="flex justify-between items-center">
                  <span className="text-muted-foreground">Spatial Alignment:</span>
                  <Badge
                    variant={calState?.is_head_in_target_zone ? "outline" : "destructive"}
                    className={`text-[10px] ${
                      calState?.is_head_in_target_zone
                        ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                        : ""
                    }`}
                  >
                    {calState?.is_head_in_target_zone ? "HEAD ALIGNED [OK]" : "PLEASE TURN HEAD"}
                  </Badge>
                </div>
                <p className="text-[11px] text-muted-foreground">
                  {calState?.instruction || "Waiting to begin calibration..."}
                </p>
              </div>

              {/* Active Zone Progress */}
              <div className="space-y-1.5 pt-2 border-t border-border">
                <div className="flex justify-between font-mono">
                  <span className="text-muted-foreground">Zone Sample Acquisition:</span>
                  <span className="font-semibold text-foreground">
                    {calState ? `${Math.round(calState.zone_progress_pct)}%` : "0%"}
                  </span>
                </div>
                <Progress value={calState?.zone_progress_pct || 0} className="h-2" />
                <div className="text-[10px] text-muted-foreground flex justify-between">
                  <span>0% (Paused if Looking Away)</span>
                  <span>100% (Locked)</span>
                </div>
              </div>

              {/* Overall Total Progress */}
              <div className="space-y-1.5 pt-2 border-t border-border">
                <div className="flex justify-between font-mono">
                  <span className="text-muted-foreground">Total Calibration:</span>
                  <span className="font-semibold text-foreground">
                    {calState ? `${Math.round(calState.total_progress_pct)}%` : "0%"}
                  </span>
                </div>
                <Progress value={calState?.total_progress_pct || 0} className="h-1.5" />
              </div>

              {/* Tips */}
              <div className="p-2.5 rounded-lg bg-muted/40 border border-border text-[11px] text-muted-foreground space-y-1">
                <span className="font-semibold text-foreground block">💡 Calibration Tip:</span>
                <span>
                  Keep your head turned naturally toward the mirror indicator until the zone progress bar fills to 100%. If needed, you can click &apos;Skip Zone&apos; to proceed.
                </span>
              </div>
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  );
}
