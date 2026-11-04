"use client";

import React, { useState, useRef, useEffect, useCallback } from "react";
import { toast } from "sonner";
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
  AlertCircle,
  ArrowRight,
  ArrowLeft,
  ArrowDown,
  Navigation,
  RefreshCw,
  Square,
  Wifi,
  WifiOff,
  FastForward,
  Maximize2,
  Minimize2,
  Compass,
  Check,
  Sliders,
} from "lucide-react";

export interface CalibrationState {
  current_zone: string;
  phase_index: number;
  total_phases: number;
  zone_progress_pct: number;
  total_progress_pct: number;
  stable_samples_count: number;
  target_samples_count: number;
  instruction: string;
  is_head_in_target_zone: boolean;
  stability_feedback: string;
  is_completed: boolean;
}

export function GuidedCalibrationWizard() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isFrameInFlightRef = useRef<boolean>(false);
  const lastFrameSentTimeRef = useRef<number>(0);

  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calState, setCalState] = useState<CalibrationState | null>(null);
  const [isSuccess, setIsSuccess] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected">("disconnected");
  const [calibrationMode, setCalibrationMode] = useState<"3_ZONE" | "4_ZONE">("3_ZONE");
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Backend target selector (default to Local if on localhost, or Cloud Render)
  const [backendType, setBackendType] = useState<"local" | "cloud">("local");

  const localWsUrl = "ws://127.0.0.1:8000";
  const cloudWsUrl = process.env.NEXT_PUBLIC_FASTAPI_WS_URL || "wss://driveguard-x048.onrender.com";
  const activeWsBase = backendType === "local" ? localWsUrl : cloudWsUrl;

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("driveguard_backend_type") as "local" | "cloud" | null;
      if (saved) {
        setBackendType(saved);
      } else {
        const isLocalHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
        setBackendType(isLocalHost ? "local" : "cloud");
      }
    }
  }, []);

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

  const changeBackendType = (type: "local" | "cloud") => {
    setBackendType(type);
    if (typeof window !== "undefined") {
      localStorage.setItem("driveguard_backend_type", type);
    }
  };

  const handleCalibrationUpdate = useCallback((cal: unknown) => {
    if (!cal || typeof cal !== "object") return;
    const c = cal as Record<string, unknown>;
    const update: CalibrationState = {
      current_zone: String(c.current_zone || "CENTER"),
      phase_index: Number(c.phase_index || 1),
      total_phases: Number(c.total_phases || (calibrationMode === "4_ZONE" ? 4 : 3)),
      zone_progress_pct: Number(c.zone_progress_pct || 0),
      total_progress_pct: Number(c.total_progress_pct || 0),
      stable_samples_count: Number(c.stable_samples_count || 0),
      target_samples_count: Number(c.target_samples_count || 15),
      instruction: String(c.instruction || ""),
      is_head_in_target_zone: Boolean(c.is_head_in_target_zone),
      stability_feedback: String(c.stability_feedback || ""),
      is_completed: Boolean(c.is_completed),
    };
    setCalState(update);

    if (update.is_completed) {
      setIsSuccess(true);
      setIsCalibrating(false);
      toast.success("Calibration Completed Successfully! Personal baselines activated.");
    }
  }, [calibrationMode]);

  const startCalibration = async () => {
    try {
      setIsSuccess(false);

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: "user" },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);

      setWsStatus("connecting");
      const sessionId = `cal_${Date.now()}`;
      const wsUrl = `${activeWsBase}/ws/v1/session/${sessionId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      const connectTimeout = setTimeout(() => {
        if (ws.readyState !== WebSocket.OPEN && backendType === "cloud") {
          ws.close();
          setBackendType("local");
          toast.warning("Cloud backend unreachable. Falling back to Local AI Engine (127.0.0.1:8000)...");
        }
      }, 4000);

      ws.onopen = () => {
        clearTimeout(connectTimeout);
        setWsStatus("connected");
        setIsCalibrating(true);

        const isFourZone = calibrationMode === "4_ZONE";
        ws.send(JSON.stringify({
          action: "START_CALIBRATION",
          four_zone: isFourZone,
          zone_duration_sec: 5.0,
        }));
        toast.info(`Starting ${isFourZone ? "4-Zone" : "3-Zone"} Guided Baseline Calibration...`);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.calibration) {
            handleCalibrationUpdate(data.calibration);
          } else if (data.phase) {
            handleCalibrationUpdate(data.phase);
          }
        } catch {
          // ignore
        } finally {
          isFrameInFlightRef.current = false;
        }
      };

      ws.onerror = () => {
        clearTimeout(connectTimeout);
        setWsStatus("disconnected");
        if (backendType === "cloud") {
          toast.warning("Cloud connection error. Switching to Local AI Engine...");
          setBackendType("local");
        } else {
          toast.error(`WebSocket connection failed on ${activeWsBase}. Verify Python backend is running.`);
        }
      };

      ws.onclose = () => {
        clearTimeout(connectTimeout);
        setWsStatus("disconnected");
        isFrameInFlightRef.current = false;
      };

      // Backpressure-Controlled Frame Grabber
      isFrameInFlightRef.current = false;
      lastFrameSentTimeRef.current = 0;

      intervalRef.current = setInterval(() => {
        const now = Date.now();
        if (now - lastFrameSentTimeRef.current < 45) return;
        if (isFrameInFlightRef.current && now - lastFrameSentTimeRef.current < 1500) return;

        if (
          !videoRef.current ||
          !canvasRef.current ||
          !ws ||
          ws.readyState !== WebSocket.OPEN ||
          ws.bufferedAmount > 0
        ) {
          return;
        }

        const video = videoRef.current;
        const canvas = canvasRef.current;
        if (video.videoWidth === 0 || video.videoHeight === 0) return;

        const ctx = canvas.getContext("2d");
        if (!ctx) return;

        canvas.width = 320;
        canvas.height = 240;
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              isFrameInFlightRef.current = true;
              lastFrameSentTimeRef.current = Date.now();
              blob.arrayBuffer().then((buf) => {
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send(buf);
                }
              });
            }
          },
          "image/jpeg",
          0.65
        );
      }, 45);

    } catch (err) {
      console.error("Calibration camera error:", err);
      toast.error("Unable to access camera. Please allow webcam permissions in your browser.");
      stopCalibration();
    }
  };

  const skipCurrentZone = () => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ action: "SKIP_CALIBRATION_STEP" }));
      toast.info("Advanced past current zone; synthetic baseline applied.");
    }
  };

  const stopCalibration = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current) {
      try {
        wsRef.current.send(JSON.stringify({ action: "CANCEL_CALIBRATION" }));
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setIsCalibrating(false);
    setWsStatus("disconnected");
    isFrameInFlightRef.current = false;
  };

  useEffect(() => {
    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    };
  }, []);

  const zoneDirections: Record<string, { label: string; icon: React.ReactNode }> = {
    CENTER: { label: "LOOK STRAIGHT AHEAD (CENTER)", icon: <Navigation className="size-6 text-primary rotate-0" /> },
    LEFT_MIRROR: { label: "TURN HEAD LEFT (LEFT MIRROR)", icon: <ArrowLeft className="size-6 text-primary" /> },
    RIGHT_MIRROR: { label: "TURN HEAD RIGHT (RIGHT MIRROR)", icon: <ArrowRight className="size-6 text-primary" /> },
    DASHBOARD: { label: "GLANCE DOWNWARD (DASHBOARD)", icon: <ArrowDown className="size-6 text-primary" /> },
  };

  const currentZoneInfo = calState?.current_zone
    ? zoneDirections[calState.current_zone] || { label: calState.current_zone, icon: <Eye className="size-6 text-primary" /> }
    : { label: "LOOK STRAIGHT AHEAD (CENTER)", icon: <Navigation className="size-6 text-primary" /> };

  return (
    <div
      ref={containerRef}
      className={`space-y-6 transition-all ${
        isFullscreen ? "fixed inset-0 z-50 bg-black p-4 flex flex-col justify-between overflow-hidden" : ""
      }`}
    >
      <div className={`grid gap-6 ${isFullscreen ? "h-full grid-cols-1" : "grid-cols-1 lg:grid-cols-3"}`}>
        {/* Video Viewport */}
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
              {/* Backend Selector */}
              <div className="flex items-center bg-muted/60 rounded-md p-0.5 text-[11px] font-mono border border-border">
                <button
                  type="button"
                  onClick={() => changeBackendType("local")}
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
                  onClick={() => changeBackendType("cloud")}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    backendType === "cloud"
                      ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="Cloud AI Engine (Render)"
                >
                  ☁️ Cloud
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

          <CardContent className="p-0 relative bg-black aspect-video flex-1 flex items-center justify-center overflow-hidden">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover -scale-x-100"
            />
            <canvas ref={canvasRef} className="hidden" />

            {/* Standby Overlay (Only shown when camera is inactive and not success) */}
            {!isCameraActive && !isSuccess && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-4 z-10">
                <div className="p-3 rounded-full bg-primary/10 text-primary">
                  <Eye className="size-8" />
                </div>
                <div className="font-semibold text-foreground text-sm">
                  Personalised Baseline Calibration
                </div>
                <p className="text-xs text-muted-foreground max-w-md">
                  Establish custom EAR open-eye baselines across spatial zones. Progress only advances when your head is properly aligned with the target zone.
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
                    ? "✨ Center, Left Mirror, Right Mirror. Dashboard baseline is derived automatically to prevent eyelid occlusion stalls."
                    : "Center, Left Mirror, Right Mirror, and Dashboard."}
                </p>

                <Button size="sm" onClick={startCalibration} className="bg-primary text-primary-foreground">
                  <Play className="size-4 mr-1.5" />
                  Start {calibrationMode === "3_ZONE" ? "3-Zone" : "4-Zone"} Calibration
                </Button>
              </div>
            )}

            {/* In-Video Heads Up Status Banner */}
            {isCameraActive && calState && !isSuccess && (
              <div className="absolute top-3 left-3 right-3 flex items-start justify-between pointer-events-none z-20">
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
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-3 z-10">
                <div className="p-3 rounded-full bg-emerald-500/10 text-emerald-500">
                  <CheckCircle2 className="size-10" />
                </div>
                <div className="font-bold text-foreground text-base">
                  Calibration Successfully Completed!
                </div>
                <p className="text-xs text-muted-foreground max-w-sm">
                  Your personalised baselines across all spatial zones have been calculated and saved to your driver profile.
                </p>
                <div className="flex items-center gap-2 pt-2">
                  <Button size="sm" onClick={startCalibration} variant="outline">
                    <RotateCcw className="size-4 mr-1.5" />
                    Recalibrate
                  </Button>
                </div>
              </div>
            )}
          </CardContent>

          <CardFooter className="py-3 px-4 flex items-center justify-between border-t border-border bg-muted/20">
            <div className="flex items-center gap-2">
              {isCameraActive ? (
                <>
                  <Button size="sm" variant="destructive" onClick={stopCalibration}>
                    <Square className="size-3.5 mr-1.5" />
                    Stop Calibration
                  </Button>
                  <Button size="sm" variant="outline" onClick={skipCurrentZone} title="Skip current zone if lighting is tricky">
                    <FastForward className="size-3.5 mr-1.5 text-primary" />
                    Skip Zone
                  </Button>
                </>
              ) : (
                <Button size="sm" onClick={startCalibration} className="bg-primary text-primary-foreground">
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
