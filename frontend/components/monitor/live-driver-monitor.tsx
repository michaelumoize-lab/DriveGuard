"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
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
  Activity,
  Camera,
  Play,
  Square,
  Volume2,
  VolumeX,
  Sparkles,
  ShieldAlert,
  AlertTriangle,
  Eye,
  RefreshCw,
  Cpu,
  Layers,
} from "lucide-react";

interface TelemetryData {
  session_id: string;
  frame_index: number;
  timestamp_ms: number;
  fps: number;
  face_detected: boolean;
  state: "NORMAL" | "WARNING" | "CRITICAL" | "MONITORING_UNAVAILABLE";
  risk_score: number;
  pose_zone: "CENTER" | "LEFT_MIRROR" | "RIGHT_MIRROR" | "DASHBOARD" | "OUT_OF_BOUNDS";
  reason_codes: string[];
  latency_ms: number;
  ear?: number;
  ear_threshold?: number;
  perclos_60s?: number;
  blink_rate?: number;
  blink_duration_ms?: number | null;
  blink_rcr?: number | null;
  cascade_mode?: string;
  cnn_invoked?: boolean;
  decision_authority?: string;
  dynamic_ear_threshold?: number;
}

export function LiveDriverMonitor() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);
  const [audioMuted, setAudioMuted] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<"CASCADE" | "GEOMETRIC" | "CNN">("CASCADE");
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionDurationSec, setSessionDurationSec] = useState(0);

  // Fallback to deployed Render cloud URL if env is not explicitly set
  const wsBaseUrl =
    process.env.NEXT_PUBLIC_FASTAPI_WS_URL || "wss://driveguard-x048.onrender.com";

  // Session duration timer
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isStreaming && sessionStartTime) {
      timer = setInterval(() => {
        setSessionDurationSec(Math.floor((Date.now() - sessionStartTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isStreaming, sessionStartTime]);

  const sendCommand = useCallback((cmd: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  const switchExperiment = (mode: "CASCADE" | "GEOMETRIC" | "CNN") => {
    setSelectedExperiment(mode);
    const modeKey =
      mode === "GEOMETRIC"
        ? "A_GEOMETRIC_ONLY"
        : mode === "CNN"
        ? "B_CNN_ALWAYS"
        : "C_CONFIDENCE_GATED_CASCADE";

    sendCommand({ command: "SWITCH_EXPERIMENT", mode: modeKey });
    toast.info(`Switched cascade mode to: ${mode}`);
  };

  const startStream = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const sessionId = `web_${Date.now()}`;
      const wsUrl = `${wsBaseUrl}/ws/v1/session/${sessionId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsWsConnected(true);
        setIsStreaming(true);
        setSessionStartTime(Date.now());
        toast.success("Connected to DriveGuard AI Engine!");

        // Start capture loop at ~20 FPS (every 50ms)
        frameIntervalRef.current = setInterval(() => {
          if (!videoRef.current || !canvasRef.current || ws.readyState !== WebSocket.OPEN) {
            return;
          }

          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;

          canvas.width = 320;
          canvas.height = 240;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          canvas.toBlob(
            (blob) => {
              if (blob && ws.readyState === WebSocket.OPEN) {
                blob.arrayBuffer().then((buffer) => {
                  ws.send(buffer);
                });
              }
            },
            "image/jpeg",
            0.65
          );
        }, 50);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "TELEMETRY" && data.telemetry) {
            setTelemetry(data.telemetry);
          } else if (data.frame_index !== undefined) {
            setTelemetry(data);
          }
        } catch {
          // ignore non-json messages
        }
      };

      ws.onerror = () => {
        toast.error("WebSocket connection failed. Ensure the AI engine is running.");
      };

      ws.onclose = () => {
        setIsWsConnected(false);
        setIsStreaming(false);
      };
    } catch (err) {
      console.error(err);
      toast.error("Could not access camera. Please allow camera permissions in browser.");
    }
  };

  const stopStream = () => {
    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((track) => track.stop());
      videoRef.current.srcObject = null;
    }

    setIsStreaming(false);
    setIsWsConnected(false);
    setTelemetry(null);
    toast.info("Monitoring session ended.");
  };

  const stateColor =
    telemetry?.state === "CRITICAL"
      ? "bg-rose-600 text-white animate-pulse"
      : telemetry?.state === "WARNING"
      ? "bg-amber-500 text-black font-bold"
      : "bg-emerald-600 text-white font-semibold";

  return (
    <div className="space-y-6">
      {/* Video & Real-time HUD Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Video Viewport */}
        <Card className="lg:col-span-2 overflow-hidden border-border/80">
          <CardHeader className="py-3 px-4 flex flex-row items-center justify-between border-b border-border">
            <div className="flex items-center gap-2">
              <Camera className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">Webcam Cabin Feed</CardTitle>
            </div>
            <div className="flex items-center gap-2">
              <Badge
                variant="outline"
                className={`text-[10px] font-mono ${
                  isWsConnected
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : "text-muted-foreground"
                }`}
              >
                {isWsConnected ? "WS: CONNECTED" : "OFFLINE"}
              </Badge>
              {telemetry && (
                <span className="text-[11px] font-mono text-muted-foreground">
                  {telemetry.fps ? `${telemetry.fps.toFixed(1)} FPS` : "--"} |{" "}
                  {telemetry.latency_ms ? `${telemetry.latency_ms.toFixed(1)} ms` : "--"}
                </span>
              )}
            </div>
          </CardHeader>

          <CardContent className="p-0 relative bg-black aspect-video flex items-center justify-center">
            {/* Native Video Element */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className="w-full h-full object-cover -scale-x-100"
            />
            {/* Hidden canvas for JPEG frame grabbing */}
            <canvas ref={canvasRef} className="hidden" />

            {/* Offline Overlay Placeholder */}
            {!isStreaming && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/90 backdrop-blur-xs text-center p-6 space-y-3">
                <div className="p-3 rounded-full bg-primary/10 text-primary">
                  <Activity className="size-8" />
                </div>
                <div className="font-semibold text-foreground text-sm">
                  Ready to Start Driver Monitoring
                </div>
                <p className="text-xs text-muted-foreground max-w-sm">
                  Click &apos;Start Live Stream&apos; below to enable your camera and connect to the edge AI cascade.
                </p>
                <Button size="sm" onClick={startStream} className="bg-primary text-primary-foreground">
                  <Play className="size-4 mr-1.5" />
                  Start Live Stream
                </Button>
              </div>
            )}

            {/* In-Video Heads-Up Overlay */}
            {isStreaming && telemetry && (
              <div className="absolute top-3 left-3 right-3 flex items-start justify-between pointer-events-none">
                <div className={`px-3 py-1.5 rounded-lg shadow-md text-xs uppercase tracking-wider ${stateColor}`}>
                  {telemetry.state}
                </div>
                <div className="px-2.5 py-1 rounded-md bg-black/75 backdrop-blur-md border border-white/10 text-[11px] font-mono text-white shadow-xs">
                  Zone: <strong>{telemetry.pose_zone}</strong>
                </div>
              </div>
            )}
          </CardContent>

          <CardFooter className="py-3 px-4 flex items-center justify-between border-t border-border bg-muted/20">
            <div className="flex items-center gap-2">
              {isStreaming ? (
                <Button size="sm" variant="destructive" onClick={stopStream}>
                  <Square className="size-3.5 mr-1.5" />
                  Stop Stream
                </Button>
              ) : (
                <Button size="sm" onClick={startStream} className="bg-primary text-primary-foreground">
                  <Play className="size-3.5 mr-1.5" />
                  Start Stream
                </Button>
              )}

              <Button
                size="sm"
                variant="outline"
                onClick={() => setAudioMuted((prev) => !prev)}
                className="h-9 w-9 p-0"
              >
                {audioMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
              </Button>
            </div>

            <div className="text-xs font-mono text-muted-foreground">
              Session Duration:{" "}
              <strong className="text-foreground">
                {Math.floor(sessionDurationSec / 60)}m {sessionDurationSec % 60}s
              </strong>
            </div>
          </CardFooter>
        </Card>

        {/* Live Kinematics Telemetry Panel */}
        <div className="space-y-4">
          <Card className="border-border/80">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Eye className="size-4 text-primary" />
                Eye & Kinematics Telemetry
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-4 text-xs">
              {/* EAR Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between font-mono">
                  <span className="text-muted-foreground">Eye Aspect Ratio (EAR):</span>
                  <span className="font-semibold text-foreground">
                    {telemetry?.ear !== undefined ? telemetry.ear.toFixed(3) : "0.000"}
                  </span>
                </div>
                <Progress
                  value={telemetry?.ear ? Math.min(100, (telemetry.ear / 0.40) * 100) : 0}
                  className="h-2"
                />
                <div className="text-[10px] text-muted-foreground font-mono flex justify-between">
                  <span>Closed (0.15)</span>
                  <span>Threshold ({telemetry?.dynamic_ear_threshold?.toFixed(3) || "0.217"})</span>
                  <span>Open (0.35)</span>
                </div>
              </div>

              {/* PERCLOS Bar */}
              <div className="space-y-1.5">
                <div className="flex justify-between font-mono">
                  <span className="text-muted-foreground">PERCLOS (60s Window):</span>
                  <span
                    className={`font-semibold ${
                      (telemetry?.perclos_60s || 0) > 15.0 ? "text-rose-500" : "text-foreground"
                    }`}
                  >
                    {telemetry?.perclos_60s !== undefined ? `${telemetry.perclos_60s.toFixed(1)}%` : "0.0%"}
                  </span>
                </div>
                <Progress
                  value={Math.min(100, ((telemetry?.perclos_60s || 0) / 40.0) * 100)}
                  className="h-2"
                />
                <div className="text-[10px] text-muted-foreground font-mono flex justify-between">
                  <span>Alert (&lt;8%)</span>
                  <span>Drowsy (15%)</span>
                  <span>Critical (&gt;25%)</span>
                </div>
              </div>

              {/* Blink Stats */}
              <div className="grid grid-cols-2 gap-2 pt-2 border-t border-border">
                <div className="p-2 rounded bg-muted/40 font-mono">
                  <span className="text-muted-foreground block text-[10px]">Blink Rate</span>
                  <span className="font-semibold text-sm">
                    {telemetry?.blink_rate !== undefined ? `${telemetry.blink_rate.toFixed(1)}/min` : "--"}
                  </span>
                </div>
                <div className="p-2 rounded bg-muted/40 font-mono">
                  <span className="text-muted-foreground block text-[10px]">Recent Blink Dur</span>
                  <span className="font-semibold text-sm">
                    {telemetry?.blink_duration_ms ? `${telemetry.blink_duration_ms.toFixed(0)} ms` : "N/A"}
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Cascade Mode Controller */}
          <Card className="border-border/80">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Cpu className="size-4 text-primary" />
                Cascade Stage Arbitration
              </CardTitle>
              <CardDescription className="text-xs">
                Switch real-time cascade arbitration experiment modes
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-2">
              {[
                { key: "CASCADE", title: "Exp C: Confidence Cascade", desc: "Adaptive Stage 2 gating (saves power)" },
                { key: "GEOMETRIC", title: "Exp A: Geometric Only", desc: "Fast EAR landmark mesh only" },
                { key: "CNN", title: "Exp B: CNN Always On", desc: "Continuous MobileNetV3 verifier" },
              ].map((exp) => (
                <div
                  key={exp.key}
                  onClick={() => switchExperiment(exp.key as "CASCADE" | "GEOMETRIC" | "CNN")}
                  className={`p-2.5 rounded-lg border cursor-pointer text-xs transition-colors ${
                    selectedExperiment === exp.key
                      ? "border-primary bg-primary/10 text-primary font-medium"
                      : "border-border hover:bg-muted/30 text-muted-foreground"
                  }`}
                >
                  <div className="font-semibold text-foreground">{exp.title}</div>
                  <div className="text-[11px] text-muted-foreground">{exp.desc}</div>
                </div>
              ))}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
