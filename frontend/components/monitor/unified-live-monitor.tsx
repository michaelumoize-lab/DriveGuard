"use client";

import React, { useState, useEffect, useRef, useCallback } from "react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useAiTelemetryStream } from "@/hooks/use-ai-telemetry-stream";
import { CapturedFrame } from "@/types/camera";
import { ExperimentMode, FrameTelemetry } from "@/types/telemetry";
import { CameraView } from "@/components/camera/camera-view";
import { CameraControls } from "@/components/camera/camera-controls";
import { CameraSelector } from "@/components/camera/camera-selector";
import { CameraStatus } from "@/components/camera/camera-status";
import { CameraErrorDisplay } from "@/components/camera/camera-error";
import { CameraSnapshotPreview } from "@/components/camera/camera-snapshot-preview";
import { LiveTelemetryHUD } from "./live-telemetry-hud";
import { ExperimentSelector } from "./experiment-selector";
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
  AlertTriangle,
  Camera,
  Cpu,
  Eye,
  Flame,
  Maximize2,
  Minimize2,
  Monitor,
  RefreshCw,
  ShieldAlert,
  ShieldCheck,
  Smartphone,
  Sparkles,
  Volume2,
  VolumeX,
  Wifi,
  WifiOff,
} from "lucide-react";

export function UnifiedLiveMonitor() {
  const containerRef = useRef<HTMLDivElement>(null);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Layout & UI States
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("landscape");
  const [framingMode, setFramingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [audioMuted, setAudioMuted] = useState(false);

  // Snapshot modal state
  const [capturedSnapshot, setCapturedSnapshot] = useState<CapturedFrame | null>(null);
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);

  // Session & Database Persistence Refs
  const activeSessionCodeRef = useRef<string | null>(null);
  const lastAlertTimeRef = useRef<number>(0);
  const sessionStatsRef = useRef({
    totalFrames: 0,
    maxPerclos: 0,
    criticalCount: 0,
  });

  // 1. Core WebRTC Camera Hook
  const {
    videoRef,
    cameraState,
    error: cameraError,
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

  // 2. Adaptive AI Telemetry WebSocket Streaming Hook
  const {
    isConnected: isWsConnected,
    connectionStatus: wsStatus,
    telemetry,
    roundtripLatencyMs,
    experimentMode,
    backendType,
    switchBackend,
    startStreamingLoop,
    stopStreamingLoop,
    switchExperiment,
    reconnect: reconnectWs,
  } = useAiTelemetryStream({
    sessionId: `dg_session_${Date.now().toString(36)}`,
    driverId: "michael_driver",
    targetFps: 20, // 20 FPS adaptive lock-step
  });

  // Synthesized Web Audio Buzzer on CRITICAL Fatigue
  const playAlertBuzzer = useCallback(() => {
    if (audioMuted) return;
    try {
      if (!audioContextRef.current) {
        const AudioCtx =
          window.AudioContext ||
          (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
        audioContextRef.current = new AudioCtx();
      }
      const ctx = audioContextRef.current;
      if (ctx.state === "suspended") {
        ctx.resume();
      }

      const osc = ctx.createOscillator();
      const gain = ctx.createGain();

      osc.type = "sawtooth";
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.25);

      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.linearRampToValueAtTime(0.01, ctx.currentTime + 0.25);

      osc.connect(gain);
      gain.connect(ctx.destination);

      osc.start();
      osc.stop(ctx.currentTime + 0.25);
    } catch (_) {}
  }, [audioMuted]);

  // Handle telemetry updates & trigger alerts
  useEffect(() => {
    if (!telemetry) return;

    sessionStatsRef.current.totalFrames++;
    const perclos = telemetry.eye_metrics?.perclos_pct ?? 0;
    if (perclos > sessionStatsRef.current.maxPerclos) {
      sessionStatsRef.current.maxPerclos = perclos;
    }

    if (telemetry.state === "CRITICAL") {
      sessionStatsRef.current.criticalCount++;
      playAlertBuzzer();

      // Log alert to database (with 5-second cooldown)
      const now = Date.now();
      if (now - lastAlertTimeRef.current > 5000) {
        lastAlertTimeRef.current = now;
        fetch("/api/alerts/log", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionCode: activeSessionCodeRef.current || "LIVE_TELEMETRY",
            type: "FATAL_FATIGUE",
            severity: "CRITICAL",
            message: `Severe fatigue detected: PERCLOS ${perclos.toFixed(1)}%, EAR ${telemetry.eye_metrics?.ear_raw.toFixed(3)}`,
            metadata: {
              riskScore: telemetry.risk_score,
              poseZone: telemetry.pose_zone,
              perclos,
            },
          }),
        }).catch(() => {});
      }
    }
  }, [telemetry, playAlertBuzzer]);

  // Auto-detect mobile device & portrait orientation on mount
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
      console.warn("Fullscreen toggle exception:", err);
      toast.info("Fullscreen mode is not supported by your browser window policy.");
    }
  }, []);

  // Start live monitoring (Hardware Camera + Adaptive WebSocket Stream)
  const handleStartMonitoring = useCallback(async () => {
    await startCamera();

    // Start database driving session
    try {
      const startRes = await fetch("/api/sessions/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: experimentMode,
          device: isMobileDevice ? "MOBILE_PORTRAIT" : "DESKTOP_HUD",
        }),
      });
      const data = await startRes.json();
      if (data?.sessionCode) {
        activeSessionCodeRef.current = data.sessionCode;
      }
    } catch (_) {}

    // Start adaptive WebSocket frame pumping once video starts
    setTimeout(() => {
      if (videoRef.current) {
        startStreamingLoop(videoRef.current, isMirrored);
      }
    }, 400);
  }, [startCamera, experimentMode, isMobileDevice, startStreamingLoop, isMirrored, videoRef]);

  // Stop live monitoring
  const handleStopMonitoring = useCallback(async () => {
    stopStreamingLoop();
    stopCamera();

    // Close database session
    if (activeSessionCodeRef.current) {
      try {
        await fetch("/api/sessions/end", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionCode: activeSessionCodeRef.current,
            durationSec: streamDurationSec,
            totalFrames: sessionStatsRef.current.totalFrames,
            maxPerclos: sessionStatsRef.current.maxPerclos,
            criticalAlerts: sessionStatsRef.current.criticalCount,
          }),
        });
      } catch (_) {}
      activeSessionCodeRef.current = null;
    }
  }, [stopStreamingLoop, stopCamera, streamDurationSec]);

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
      }
    } catch (_) {
      toast.error("Error occurred while capturing frame.");
    } finally {
      setIsCapturing(false);
    }
  }, [cameraState, captureSnapshot, viewOrientation]);

  const isPortrait = viewOrientation === "portrait";

  return (
    <div ref={containerRef} className="w-full space-y-4">
      {/* 1. Top Header Card (COMPLETELY HIDDEN IN FULLSCREEN VIEW AS REQUESTED) */}
      {!isFullscreen && (
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-4 rounded-2xl bg-card border border-border shadow-xs">
          <div className="space-y-0.5">
            <div className="flex items-center gap-2">
              <Activity className="size-5 text-primary" />
              <h2 className="text-base sm:text-lg font-bold text-foreground">
                Driver Safety Live Monitor
              </h2>
              <Badge
                variant="outline"
                className={`text-[10px] font-mono flex items-center gap-1 ${
                  isWsConnected
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : wsStatus === "connecting" || wsStatus === "reconnecting"
                    ? "border-amber-500/40 text-amber-500 bg-amber-500/10 animate-pulse"
                    : "border-border text-muted-foreground"
                }`}
              >
                <span
                  className={`size-1.5 rounded-full ${
                    isWsConnected
                      ? "bg-emerald-500"
                      : wsStatus === "connecting" || wsStatus === "reconnecting"
                      ? "bg-amber-500 animate-ping"
                      : "bg-muted-foreground"
                  }`}
                />
                <span>{isWsConnected ? "AI CLOUD CONNECTED" : "AI ENGINE STANDBY"}</span>
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              Real-time cabin monitoring powered by DriveGuard adaptive confidence cascade, 3D gaze tracking, and sub-frame blink kinematics.
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
      )}

      {/* Error Display Card if hardware error occurs */}
      {cameraError && (
        <CameraErrorDisplay
          error={cameraError}
          onRetry={handleStartMonitoring}
        />
      )}

      {/* Main Grid: Video Viewport & Companion Live Telemetry */}
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
            onStartClick={handleStartMonitoring}
          >
            {/* In-viewport Floating Cockpit HUD Overlays */}
            {cameraState === "active" && (
              <>
                {/* Top Floating Status Strip */}
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
                  <div className="flex items-center gap-1.5">
                    {telemetry ? (
                      <Badge
                        className={`text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 ${
                          telemetry.state === "CRITICAL"
                            ? "bg-rose-500 text-white animate-pulse"
                            : telemetry.state === "WARNING"
                            ? "bg-amber-500 text-black"
                            : "bg-emerald-500 text-white"
                        }`}
                      >
                        {telemetry.state}
                      </Badge>
                    ) : (
                      <Badge
                        variant="outline"
                        className="bg-black/75 backdrop-blur-md border-emerald-500/50 text-emerald-400 text-[10px] font-mono px-2.5 py-1"
                      >
                        <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse mr-1.5" />
                        STREAM ACTIVE
                      </Badge>
                    )}

                    {telemetry && (
                      <Badge
                        variant="outline"
                        className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono px-2 py-0.5"
                      >
                        Zone: {telemetry.pose_zone}
                      </Badge>
                    )}
                  </div>

                  {/* Latency & Resolution */}
                  <div className="flex items-center gap-1.5">
                    {roundtripLatencyMs > 0 && (
                      <Badge
                        variant="outline"
                        className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono px-2 py-0.5"
                      >
                        {roundtripLatencyMs}ms
                      </Badge>
                    )}
                  </div>
                </div>

                {/* Minimal Bottom Overlay in Fullscreen */}
                {isFullscreen && (
                  <div className="absolute bottom-4 left-4 right-4 flex items-center justify-between z-30 pointer-events-auto">
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={handleStopMonitoring}
                      className="rounded-xl h-10 px-4 text-xs font-semibold shadow-xl"
                    >
                      Stop Stream
                    </Button>

                    <div className="flex items-center gap-2">
                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={() => setAudioMuted((prev) => !prev)}
                        className="h-10 w-10 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                        title={audioMuted ? "Unmute Buzzer" : "Mute Buzzer"}
                      >
                        {audioMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4 text-primary" />}
                      </Button>

                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={() => setFramingMode((prev) => (prev === "cover" ? "fit" : "cover"))}
                        className="h-10 w-10 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                        title="Toggle Framing"
                      >
                        <Flame className="size-4 text-primary" />
                      </Button>

                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={toggleFullscreen}
                        className="h-10 w-10 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                        title="Exit Fullscreen"
                      >
                        <Minimize2 className="size-4" />
                      </Button>
                    </div>
                  </div>
                )}
              </>
            )}
          </CameraView>

          {/* Controls Bar (Visible in standard non-fullscreen view) */}
          {!isFullscreen && (
            <>
              <CameraControls
                cameraState={cameraState}
                isMirrored={isMirrored}
                framingMode={framingMode}
                isFullscreen={isFullscreen}
                onStart={handleStartMonitoring}
                onStop={handleStopMonitoring}
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

              <CameraStatus
                cameraState={cameraState}
                resolution={resolution}
                facingMode={facingMode}
                durationSec={streamDurationSec}
              />
            </>
          )}
        </div>

        {/* Companion Sidebar Panel: Real-Time Telemetry & Experiment Controls */}
        {!isFullscreen && (
          <div
            className={`space-y-4 ${
              isPortrait ? "md:col-span-5 lg:col-span-6" : "lg:col-span-1"
            }`}
          >
            {/* Live AI Telemetry HUD Gauges */}
            <LiveTelemetryHUD telemetry={telemetry} />

            {/* Experiment Mode Selector Card */}
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Sparkles className="size-4 text-primary" />
                  <span>Cascade Experiment Mode</span>
                </CardTitle>
                <CardDescription className="text-xs">
                  Switch the arbitration logic between Geometric EAR and MobileNetV3 CNN.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ExperimentSelector
                  currentMode={experimentMode}
                  onSelectMode={switchExperiment}
                />
              </CardContent>
            </Card>

            {/* Camera Input Device Card */}
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Camera className="size-4 text-primary" />
                  <span>Camera Input Sensor</span>
                </CardTitle>
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

            {/* Target Cloud / Engine Selector */}
            <Card className="border-border">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Cpu className="size-4 text-primary" />
                  <span>AI Inference Target</span>
                </CardTitle>
              </CardHeader>
              <CardContent className="space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">Server Endpoint</span>
                  <span className="font-mono text-foreground font-medium">
                    {backendType === "cloud" ? "Render Cloud AI" : "127.0.0.1:8000"}
                  </span>
                </div>
                <div className="flex items-center gap-2 pt-1">
                  <Button
                    size="sm"
                    variant={backendType === "cloud" ? "default" : "outline"}
                    onClick={() => switchBackend("cloud")}
                    className="text-xs h-8 flex-1 rounded-xl"
                  >
                    Cloud (Render)
                  </Button>
                  <Button
                    size="sm"
                    variant={backendType === "local" ? "default" : "outline"}
                    onClick={() => switchBackend("local")}
                    className="text-xs h-8 flex-1 rounded-xl"
                  >
                    Localhost (8000)
                  </Button>
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
