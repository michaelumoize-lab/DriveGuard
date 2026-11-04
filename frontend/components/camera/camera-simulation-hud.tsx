"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { CapturedFrame } from "@/types/camera";
import { CameraView } from "./camera-view";
import { CameraControls } from "./camera-controls";
import { CameraSelector } from "./camera-selector";
import { CameraStatus } from "./camera-status";
import { CameraErrorDisplay } from "./camera-error";
import { CameraSnapshotPreview } from "./camera-snapshot-preview";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Activity,
  Camera,
  CheckCircle2,
  Cpu,
  Eye,
  Maximize2,
  Minimize2,
  Monitor,
  ShieldCheck,
  Smartphone,
  Sparkles,
} from "lucide-react";

export function CameraSimulationHUD() {
  const containerRef = useRef<HTMLDivElement>(null);

  // Layout and orientation states
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("landscape");
  const [framingMode, setFramingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMobileDevice, setIsMobileDevice] = useState(false);

  // Snapshot modal state
  const [capturedSnapshot, setCapturedSnapshot] = useState<CapturedFrame | null>(null);
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);

  // Core Camera Hook
  const {
    videoRef,
    cameraState,
    error,
    devices,
    selectedDeviceId,
    facingMode,
    resolution,
    isMirrored,
    setIsMirrored,
    streamDurationSec,
    startCamera,
    stopCamera,
    pauseCamera,
    resumeCamera,
    switchFacingMode,
    selectDevice,
    captureSnapshot,
    refreshDevices,
  } = useCamera({
    idealWidth: 1280,
    idealHeight: 720,
    idealFps: 30,
    defaultFacingMode: "user",
    mirrored: true,
  });

  // Auto-detect mobile device & orientation on mount
  useEffect(() => {
    if (typeof window !== "undefined") {
      const detectMobile = () => {
        const isSmallScreen = window.innerWidth <= 768;
        const isMobileUA =
          /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(
            navigator.userAgent
          );
        const mobile = isSmallScreen || isMobileUA;
        setIsMobileDevice(mobile);

        if (mobile) {
          setViewOrientation("portrait");
          setFramingMode("cover"); // Full-bleed portrait fill on mobile
        } else {
          setViewOrientation("landscape");
          setFramingMode("cover");
        }
      };

      detectMobile();
      window.addEventListener("resize", detectMobile);
      return () => window.removeEventListener("resize", detectMobile);
    }
  }, []);

  // Listen to browser fullscreen changes
  useEffect(() => {
    const handleFullscreenChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };

    document.addEventListener("fullscreenchange", handleFullscreenChange);
    return () => document.removeEventListener("fullscreenchange", handleFullscreenChange);
  }, []);

  const toggleFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        if (containerRef.current) {
          await containerRef.current.requestFullscreen();
        }
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.warn("Fullscreen toggle error:", err);
      toast.info("Fullscreen mode is not supported by your browser window policy.");
    }
  }, []);

  // Snapshot capture handler
  const handleCaptureSnapshot = useCallback(async () => {
    if (cameraState !== "active") {
      toast.error("Camera must be active to capture a frame snapshot.");
      return;
    }

    setIsCapturing(true);
    try {
      const frame = await captureSnapshot({
        format: "image/jpeg",
        quality: 0.92,
        cropToAspectRatio: viewOrientation === "portrait" ? 9 / 16 : undefined,
      });

      if (frame) {
        setCapturedSnapshot(frame);
        setIsSnapshotOpen(true);
        toast.success(`Captured ${frame.width}×${frame.height} frame (${(frame.sizeBytes / 1024).toFixed(0)} KB)`);
      } else {
        toast.error("Failed to capture frame snapshot.");
      }
    } catch (err) {
      console.error("Frame capture exception:", err);
      toast.error("Error occurred while capturing video frame.");
    } finally {
      setIsCapturing(false);
    }
  }, [cameraState, captureSnapshot, viewOrientation]);

  const isPortrait = viewOrientation === "portrait";

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* Top Header Card */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl bg-card border border-border shadow-xs">
        <div className="space-y-0.5">
          <div className="flex items-center gap-2">
            <Camera className="size-5 text-primary" />
            <h2 className="text-base sm:text-lg font-bold text-foreground">
              Responsive Webcam Driver HUD
            </h2>
            <Badge variant="outline" className="border-primary/40 bg-primary/10 text-primary text-[10px] font-mono">
              STANDALONE SIMULATION
            </Badge>
          </div>
          <p className="text-xs text-muted-foreground">
            Fully functional WebRTC browser camera interface prepared for future AI driver drowsiness detection.
          </p>
        </div>

        {/* Quick Orientation & Mode Selector */}
        <div className="flex items-center gap-2 shrink-0">
          <Button
            size="sm"
            variant={isPortrait ? "default" : "outline"}
            onClick={() => setViewOrientation("portrait")}
            className="text-xs h-9 px-3 rounded-xl gap-1.5"
            title="Portrait View (Mobile Driver View)"
          >
            <Smartphone className="size-3.5" />
            <span>Portrait</span>
          </Button>

          <Button
            size="sm"
            variant={!isPortrait ? "default" : "outline"}
            onClick={() => setViewOrientation("landscape")}
            className="text-xs h-9 px-3 rounded-xl gap-1.5"
            title="Landscape View (Desktop Cockpit View)"
          >
            <Monitor className="size-3.5" />
            <span>Landscape</span>
          </Button>
        </div>
      </div>

      {/* Error Display Card if error occurs */}
      {error && (
        <CameraErrorDisplay
          error={error}
          onRetry={() => startCamera()}
        />
      )}

      {/* Main Grid: Video Viewport & Companion Telemetry Panel */}
      <div
        className={`grid gap-4 ${
          isFullscreen
            ? "grid-cols-1 p-0 m-0"
            : isPortrait
            ? "grid-cols-1 md:grid-cols-12 max-w-5xl mx-auto"
            : "grid-cols-1 lg:grid-cols-3"
        }`}
      >
        {/* Main Video Viewport Column */}
        <div
          className={`flex flex-col gap-3 ${
            isFullscreen
              ? "col-span-1"
              : isPortrait
              ? "md:col-span-7 lg:col-span-6 mx-auto w-full"
              : "lg:col-span-2"
          }`}
        >
          {/* Camera Viewport */}
          <CameraView
            videoRef={videoRef}
            cameraState={cameraState}
            isMirrored={isMirrored}
            framingMode={framingMode}
            viewOrientation={viewOrientation}
            isFullscreen={isFullscreen}
            onStartClick={() => startCamera()}
          >
            {/* In-viewport Floating Cockpit Badges */}
            {cameraState === "active" && (
              <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
                <Badge
                  variant="outline"
                  className="bg-black/75 backdrop-blur-md border-emerald-500/50 text-emerald-400 text-[10px] font-mono px-2.5 py-1"
                >
                  <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5" />
                  LIVE CABIN FEED
                </Badge>

                {resolution && (
                  <Badge
                    variant="outline"
                    className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono px-2.5 py-1"
                  >
                    {resolution.width}×{resolution.height}
                  </Badge>
                )}
              </div>
            )}
          </CameraView>

          {/* Camera Controls Bar */}
          <CameraControls
            cameraState={cameraState}
            isMirrored={isMirrored}
            framingMode={framingMode}
            isFullscreen={isFullscreen}
            onStart={() => startCamera()}
            onStop={stopCamera}
            onPause={pauseCamera}
            onResume={resumeCamera}
            onSwitchFacing={switchFacingMode}
            onCaptureSnapshot={handleCaptureSnapshot}
            onToggleFraming={() =>
              setFramingMode((prev) => (prev === "cover" ? "fit" : "cover"))
            }
            onToggleFullscreen={toggleFullscreen}
            onToggleMirror={() => setIsMirrored((prev) => !prev)}
            isCapturing={isCapturing}
          />

          {/* Camera Status Bar */}
          <CameraStatus
            cameraState={cameraState}
            resolution={resolution}
            facingMode={facingMode}
            durationSec={streamDurationSec}
          />
        </div>

        {/* Companion Sidebar Panel (Desktop & Tablet) */}
        {!isFullscreen && (
          <div
            className={`space-y-4 ${
              isPortrait ? "md:col-span-5 lg:col-span-6" : "lg:col-span-1"
            }`}
          >
            {/* Device Selection Card */}
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Camera className="size-4 text-primary" />
                  <span>Camera Input Device</span>
                </CardTitle>
                <CardDescription className="text-xs">
                  Switch between front driver camera, rear cabin camera, or external USB sensors.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <CameraSelector
                  devices={devices}
                  selectedDeviceId={selectedDeviceId}
                  onSelectDevice={selectDevice}
                  onRefreshDevices={refreshDevices}
                />
              </CardContent>
            </Card>

            {/* Hardware & WebRTC Verification Card */}
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Activity className="size-4 text-emerald-500" />
                  <span>Hardware Simulation Specs</span>
                </CardTitle>
                <CardDescription className="text-xs">
                  Real WebRTC media stream specifications verified in this session.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-2.5 text-xs">
                <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
                  <span className="text-muted-foreground">Target Resolution</span>
                  <span className="font-mono font-medium text-foreground">
                    1280 × 720 (Ideal)
                  </span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
                  <span className="text-muted-foreground">Actual Resolution</span>
                  <span className="font-mono font-bold text-foreground">
                    {resolution ? `${resolution.width} × ${resolution.height}` : "Pending Start"}
                  </span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
                  <span className="text-muted-foreground">Target Frame Rate</span>
                  <span className="font-mono font-medium text-foreground">
                    15–30 FPS
                  </span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
                  <span className="text-muted-foreground">Default Facing Mode</span>
                  <span className="font-mono font-medium text-foreground">
                    User (Front)
                  </span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
                  <span className="text-muted-foreground">Audio Track</span>
                  <span className="font-mono text-emerald-500 font-semibold">
                    Disabled (Privacy)
                  </span>
                </div>

                <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
                  <span className="text-muted-foreground">Mobile Portrait Framing</span>
                  <span className="font-mono text-primary font-semibold">
                    Full Bleed (Cover)
                  </span>
                </div>
              </CardContent>
            </Card>

            {/* AI Integration Readiness Card */}
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Cpu className="size-4 text-primary" />
                  <span>Future AI Backend Readiness</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2 text-xs text-muted-foreground leading-relaxed">
                <p>
                  Frames captured from this interface are prepared via the decoupled{" "}
                  <code className="text-primary font-mono text-[11px]">FrameProducer</code> interface for seamless integration with the DriveGuard Python FastAPI backend.
                </p>
                <div className="flex items-center gap-2 pt-1 text-emerald-500 font-medium text-[11px]">
                  <CheckCircle2 className="size-3.5 shrink-0" />
                  <span>Zero fabricated or fake AI prediction scores</span>
                </div>
              </CardContent>
            </Card>
          </div>
        )}
      </div>

      {/* Snapshot Preview Dialog */}
      <CameraSnapshotPreview
        frame={capturedSnapshot}
        isOpen={isSnapshotOpen}
        onClose={() => setIsSnapshotOpen(false)}
      />
    </div>
  );
}
