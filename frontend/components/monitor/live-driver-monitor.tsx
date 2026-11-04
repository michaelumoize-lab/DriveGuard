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
  Server,
  Radio,
  CheckCircle2,
  Wifi,
  WifiOff,
  Maximize2,
  Minimize2,
  Smartphone,
  Monitor,
  Flame,
  Clock,
  Compass,
} from "lucide-react";

export interface NormalizedTelemetry {
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
  ear: number;
  ear_threshold: number;
  perclos_60s: number;
  blink_rate: number;
  blink_duration_ms: number | null;
  blink_rcr: number | null;
  cascade_mode: string;
  cnn_invoked: boolean;
  decision_authority: string;
  pitch_deg: number;
  yaw_deg: number;
  is_eye_closed: boolean;
}

export function LiveDriverMonitor() {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const frameIntervalRef = useRef<NodeJS.Timeout | null>(null);
  const isFrameInFlightRef = useRef<boolean>(false);
  const lastFrameSentTimeRef = useRef<number>(0);
  const audioContextRef = useRef<AudioContext | null>(null);

  // Active Session & DB Logging Refs
  const activeSessionCodeRef = useRef<string | null>(null);
  const lastAlertLoggedTimeRef = useRef<number>(0);
  const sessionMetricsRef = useRef({
    totalFrames: 0,
    fpsSum: 0,
    latencySum: 0,
    perclosSum: 0,
    maxPerclos: 0,
    totalBlinks: 0,
    blinkRateSum: 0,
    normalCount: 0,
    warningCount: 0,
    criticalCount: 0,
  });

  // States
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "error">("disconnected");
  const [telemetry, setTelemetry] = useState<NormalizedTelemetry | null>(null);
  const [audioMuted, setAudioMuted] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<"CASCADE" | "GEOMETRIC" | "CNN">("CASCADE");
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionDurationSec, setSessionDurationSec] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Device & Layout Modes
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("landscape");

  // Backend target selector (default to Local if on localhost, or Cloud Render)
  const [backendType, setBackendType] = useState<"local" | "cloud">("local");

  const localWsUrl = "ws://127.0.0.1:8000";
  const cloudWsUrl = process.env.NEXT_PUBLIC_FASTAPI_WS_URL || "wss://driveguard-x048.onrender.com";
  const activeWsBase = backendType === "local" ? localWsUrl : cloudWsUrl;

  // Auto-detect mobile device & orientation
  useEffect(() => {
    if (typeof window !== "undefined") {
      const checkMobile = () => {
        const isSmallScreen = window.innerWidth <= 768;
        const isMobileUA = /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
        const mobile = isSmallScreen || isMobileUA;
        setIsMobileDevice(mobile);
        if (mobile) {
          setViewOrientation("portrait");
        }
      };

      checkMobile();
      window.addEventListener("resize", checkMobile);
      return () => window.removeEventListener("resize", checkMobile);
    }
  }, []);

  // Initialize preference from localStorage if available
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

  // Listen to browser fullscreen changes
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
        if (containerRef.current) {
          await containerRef.current.requestFullscreen();
        }
      } else {
        await document.exitFullscreen();
      }
    } catch (err) {
      console.error("Fullscreen error:", err);
      toast.error("Fullscreen mode could not be activated by browser.");
    }
  };

  const changeBackendType = (type: "local" | "cloud") => {
    setBackendType(type);
    if (typeof window !== "undefined") {
      localStorage.setItem("driveguard_backend_type", type);
    }
    toast.info(`Switched AI target to ${type === "local" ? "Local Engine (127.0.0.1:8000)" : "Cloud Engine (Render)"}`);
  };

  // Session duration timer
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isCameraActive && sessionStartTime) {
      timer = setInterval(() => {
        setSessionDurationSec(Math.floor((Date.now() - sessionStartTime) / 1000));
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [isCameraActive, sessionStartTime]);

  // Audio warning buzzer on CRITICAL state
  const playAlertBuzzer = useCallback(() => {
    if (audioMuted) return;
    try {
      if (!audioContextRef.current) {
        const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
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
      osc.frequency.exponentialRampToValueAtTime(440, ctx.currentTime + 0.18);
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.20);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.22);
    } catch {
      // Audio playback restricted by browser policy
    }
  }, [audioMuted]);

  useEffect(() => {
    if (telemetry?.state === "CRITICAL") {
      playAlertBuzzer();
    }
  }, [telemetry?.state, playAlertBuzzer]);

  // Safely normalize raw telemetry from backend FrameTelemetryDTO
  const parseTelemetry = (raw: Record<string, unknown>): NormalizedTelemetry => {
    const eye = (raw.eye_metrics || {}) as Record<string, unknown>;
    const cascade = (raw.cascade || {}) as Record<string, unknown>;
    const pose = (raw.head_pose || {}) as Record<string, unknown>;

    return {
      session_id: String(raw.session_id || ""),
      frame_index: Number(raw.frame_index || 0),
      timestamp_ms: Number(raw.timestamp_ms || Date.now()),
      fps: Number(raw.fps || 0),
      face_detected: Boolean(raw.face_detected),
      state: (raw.state as NormalizedTelemetry["state"]) || "MONITORING_UNAVAILABLE",
      risk_score: Number(raw.risk_score || 0),
      pose_zone: (raw.pose_zone as NormalizedTelemetry["pose_zone"]) || "CENTER",
      reason_codes: Array.isArray(raw.reason_codes) ? (raw.reason_codes as string[]) : [],
      latency_ms: Number(raw.inference_latency_ms ?? raw.latency_ms ?? 0),
      ear: Number(eye.ear_raw ?? raw.ear ?? 0),
      ear_threshold: Number(eye.ear_closure_threshold ?? raw.dynamic_ear_threshold ?? 0.22),
      perclos_60s: Number(eye.perclos_pct ?? raw.perclos_60s ?? 0),
      blink_rate: Number(eye.blink_rate_per_min ?? raw.blink_rate ?? 0),
      blink_duration_ms: eye.blink_duration_ms !== undefined && eye.blink_duration_ms !== null ? Number(eye.blink_duration_ms) : null,
      blink_rcr: eye.blink_rcr !== undefined && eye.blink_rcr !== null ? Number(eye.blink_rcr) : null,
      cascade_mode: String(cascade.experiment_mode ?? raw.cascade_mode ?? "CASCADE"),
      cnn_invoked: Boolean(cascade.cnn_triggered ?? raw.cnn_invoked),
      decision_authority: String(cascade.arbitration_winner ?? raw.decision_authority ?? "GEOMETRY"),
      pitch_deg: Number(pose.pitch_deg ?? 0),
      yaw_deg: Number(pose.yaw_deg ?? 0),
      is_eye_closed: Boolean(eye.is_eye_closed),
    };
  };

  // Background Database Alert Logger (throttled to at most once per 3.5s per event)
  const logAlertToDatabase = useCallback(async (t: NormalizedTelemetry) => {
    const sessionCode = activeSessionCodeRef.current;
    if (!sessionCode) return;
    if (t.state !== "WARNING" && t.state !== "CRITICAL") return;

    const now = Date.now();
    if (now - lastAlertLoggedTimeRef.current < 3500) return;
    lastAlertLoggedTimeRef.current = now;

    try {
      await fetch("/api/alerts/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          sessionCode,
          severity: t.state,
          riskScore: t.risk_score,
          earValue: t.ear,
          perclosPct: t.perclos_60s,
          poseZone: t.pose_zone,
          reasons: t.reason_codes,
          cascadeDecision: t.decision_authority,
          timestampMs: t.timestamp_ms,
        }),
      });
    } catch (err) {
      console.warn("Could not persist alert to database:", err);
    }
  }, []);

  const sendCommand = useCallback((cmd: Record<string, unknown>) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(cmd));
    }
  }, []);

  const switchExperiment = (mode: "CASCADE" | "GEOMETRIC" | "CNN") => {
    setSelectedExperiment(mode);
    const modeKey =
      mode === "GEOMETRIC"
        ? "EXPERIMENT_A_GEOMETRIC_ONLY"
        : mode === "CNN"
        ? "EXPERIMENT_B_CNN_ALWAYS"
        : "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE";

    sendCommand({ action: "SWITCH_EXPERIMENT", mode: modeKey });
    toast.info(`Switched cascade mode to: ${mode}`);
  };

  // Connect WebSocket to AI Engine with auto-fallback
  const connectWebSocket = useCallback((sessionId: string, fallbackToLocal: boolean = true) => {
    setWsStatus("connecting");
    const wsUrl = `${activeWsBase}/ws/v1/session/${sessionId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    const connectTimeout = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN && backendType === "cloud" && fallbackToLocal) {
        console.warn("Cloud connection timed out, attempting local fallback...");
        ws.close();
        setBackendType("local");
        toast.warning("Cloud backend unreachable. Auto-switching to Local AI Engine (127.0.0.1:8000) ⚡");
      }
    }, 4500);

    ws.onopen = () => {
      clearTimeout(connectTimeout);
      setIsWsConnected(true);
      setWsStatus("connected");
      toast.success(`Connected to AI Engine (${backendType === "local" ? "127.0.0.1:8000" : "Render Cloud"})!`);

      const modeKey =
        selectedExperiment === "GEOMETRIC"
          ? "EXPERIMENT_A_GEOMETRIC_ONLY"
          : selectedExperiment === "CNN"
          ? "EXPERIMENT_B_CNN_ALWAYS"
          : "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE";
      ws.send(JSON.stringify({ action: "SWITCH_EXPERIMENT", mode: modeKey }));
    };

    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        let parsed: NormalizedTelemetry | null = null;
        if (data.type === "TELEMETRY" && data.telemetry) {
          parsed = parseTelemetry(data.telemetry);
        } else if (data.frame_index !== undefined) {
          parsed = parseTelemetry(data);
        }

        if (parsed) {
          setTelemetry(parsed);

          // Update session aggregate metrics
          const m = sessionMetricsRef.current;
          m.totalFrames += 1;
          m.fpsSum += parsed.fps;
          m.latencySum += parsed.latency_ms;
          m.perclosSum += parsed.perclos_60s;
          m.maxPerclos = Math.max(m.maxPerclos, parsed.perclos_60s);
          if (parsed.blink_rate > 0) {
            m.blinkRateSum += parsed.blink_rate;
          }
          if (parsed.state === "CRITICAL") m.criticalCount += 1;
          else if (parsed.state === "WARNING") m.warningCount += 1;
          else m.normalCount += 1;

          // Auto-log warning or critical alerts to database
          if (parsed.state === "WARNING" || parsed.state === "CRITICAL") {
            logAlertToDatabase(parsed);
          }
        }
      } catch {
        // Non-JSON binary or malformed payload
      } finally {
        isFrameInFlightRef.current = false;
      }
    };

    ws.onerror = () => {
      clearTimeout(connectTimeout);
      setWsStatus("error");
      if (backendType === "cloud" && fallbackToLocal) {
        toast.warning("Cloud backend offline. Falling back to Local AI Engine (127.0.0.1:8000)...");
        setBackendType("local");
      } else {
        toast.error(`WebSocket connection failed on ${activeWsBase}. Make sure Python server is running.`);
      }
    };

    ws.onclose = () => {
      clearTimeout(connectTimeout);
      setIsWsConnected(false);
      setWsStatus("disconnected");
      isFrameInFlightRef.current = false;
    };

    return ws;
  }, [activeWsBase, backendType, selectedExperiment, logAlertToDatabase]);

  // Main Start Camera & Streaming Action (Mobile-First Portrait Optimized)
  const startStream = async () => {
    try {
      const isPortrait = viewOrientation === "portrait" || isMobileDevice;

      // Request device camera with mobile-first portrait aspect ratio when on portrait
      const videoConstraints: MediaTrackConstraints = isPortrait
        ? {
            facingMode: "user",
            aspectRatio: { ideal: 9 / 16 },
            width: { ideal: 720 },
            height: { ideal: 1280 },
          }
        : {
            facingMode: "user",
            aspectRatio: { ideal: 16 / 9 },
            width: { ideal: 640 },
            height: { ideal: 480 },
          };

      const stream = await navigator.mediaDevices.getUserMedia({
        video: videoConstraints,
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setIsCameraActive(true);
      setSessionStartTime(Date.now());

      // 1. Initialize Driving Session in PostgreSQL
      const sessionCode = `DRV-${Date.now().toString().slice(-6)}`;
      activeSessionCodeRef.current = sessionCode;
      sessionMetricsRef.current = {
        totalFrames: 0,
        fpsSum: 0,
        latencySum: 0,
        perclosSum: 0,
        maxPerclos: 0,
        totalBlinks: 0,
        blinkRateSum: 0,
        normalCount: 0,
        warningCount: 0,
        criticalCount: 0,
      };

      try {
        await fetch("/api/sessions/start", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sessionCode }),
        });
        toast.success(`Driving Session ${sessionCode} started & recording to DB!`);
      } catch (err) {
        console.warn("Could not register session in DB:", err);
      }

      // 2. Connect WebSocket
      const sessionId = `web_${Date.now()}`;
      const ws = connectWebSocket(sessionId);

      isFrameInFlightRef.current = false;
      lastFrameSentTimeRef.current = 0;

      // 3. Backpressure-Controlled Frame Grabber (respects portrait aspect ratio)
      frameIntervalRef.current = setInterval(() => {
        const now = Date.now();
        if (now - lastFrameSentTimeRef.current < 40) return;
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

        // Size canvas matching camera orientation for 0-distortion MediaPipe processing
        if (isPortrait) {
          canvas.width = 240;
          canvas.height = 320;
        } else {
          canvas.width = 320;
          canvas.height = 240;
        }

        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob(
          (blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              isFrameInFlightRef.current = true;
              lastFrameSentTimeRef.current = Date.now();
              blob.arrayBuffer().then((buffer) => {
                if (ws.readyState === WebSocket.OPEN) {
                  ws.send(buffer);
                }
              });
            }
          },
          "image/jpeg",
          0.65
        );
      }, 40);

    } catch (err) {
      console.error("Camera access failed:", err);
      toast.error("Could not access camera. Please allow webcam permissions in your browser.");
      stopStream();
    }
  };

  // Teardown everything cleanly & finalize session in DB
  const stopStream = async () => {
    const sessionCode = activeSessionCodeRef.current;
    const duration = sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
    const m = sessionMetricsRef.current;

    // 1. Finalize Driving Session in PostgreSQL
    if (sessionCode) {
      try {
        const avgFps = m.totalFrames > 0 ? Number((m.fpsSum / m.totalFrames).toFixed(1)) : 0;
        const avgLatency = m.totalFrames > 0 ? Number((m.latencySum / m.totalFrames).toFixed(1)) : 0;
        const avgPerclos = m.totalFrames > 0 ? Number((m.perclosSum / m.totalFrames).toFixed(1)) : 0;
        const avgBlinkRate = m.totalFrames > 0 ? Number((m.blinkRateSum / m.totalFrames).toFixed(1)) : 0;

        await fetch("/api/sessions/end", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionCode,
            durationSeconds: duration,
            averageFps: avgFps,
            averageLatencyMs: avgLatency,
            averagePerclosPct: avgPerclos,
            maxPerclosPct: Number(m.maxPerclos.toFixed(1)),
            averageBlinkRate: avgBlinkRate,
            warningAlertsCount: m.warningCount,
            criticalAlertsCount: m.criticalCount,
          }),
        });
        toast.success(`Trip saved to Database! Logged ${m.warningCount} warnings & ${m.criticalCount} critical events.`);
      } catch (err) {
        console.warn("Could not finalize session in DB:", err);
      }
    }

    if (frameIntervalRef.current) {
      clearInterval(frameIntervalRef.current);
      frameIntervalRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
      wsRef.current = null;
    }

    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject = null;
    }

    setIsCameraActive(false);
    setIsWsConnected(false);
    setWsStatus("disconnected");
    setTelemetry(null);
    setSessionStartTime(null);
    activeSessionCodeRef.current = null;
    isFrameInFlightRef.current = false;
  };

  const reconnectWebSocket = () => {
    if (!isCameraActive) return;
    if (wsRef.current) {
      try {
        wsRef.current.close();
      } catch {
        // ignore
      }
    }
    const sessionId = `web_${Date.now()}`;
    connectWebSocket(sessionId);
  };

  useEffect(() => {
    return () => {
      if (frameIntervalRef.current) clearInterval(frameIntervalRef.current);
      if (wsRef.current) wsRef.current.close();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
    };
  }, []);

  const stateColor =
    telemetry?.state === "CRITICAL"
      ? "bg-rose-600 text-white animate-pulse"
      : telemetry?.state === "WARNING"
      ? "bg-amber-500 text-black font-bold"
      : telemetry?.state === "MONITORING_UNAVAILABLE"
      ? "bg-muted text-muted-foreground"
      : "bg-emerald-600 text-white font-semibold";

  const isPortraitMode = viewOrientation === "portrait";

  return (
    <div
      ref={containerRef}
      className={`space-y-6 transition-all ${
        isFullscreen
          ? "fixed inset-0 z-50 bg-black p-0 sm:p-4 flex flex-col justify-between overflow-hidden"
          : ""
      }`}
    >
      {/* Top Mobile/Desktop Orientation Bar */}
      {!isFullscreen && (
        <div className="flex items-center justify-between bg-muted/40 p-2.5 rounded-xl border border-border text-xs">
          <div className="flex items-center gap-2">
            <span className="text-muted-foreground hidden sm:inline">Stream Orientation:</span>
            <div className="flex items-center bg-background rounded-lg p-0.5 border border-border">
              <button
                type="button"
                onClick={() => setViewOrientation("portrait")}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 ${
                  viewOrientation === "portrait"
                    ? "bg-primary text-primary-foreground shadow-xs"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Smartphone className="size-3.5" />
                <span>TikTok Portrait</span>
              </button>
              <button
                type="button"
                onClick={() => setViewOrientation("landscape")}
                className={`px-2.5 py-1 rounded-md text-xs font-medium transition-all flex items-center gap-1.5 ${
                  viewOrientation === "landscape"
                    ? "bg-primary text-primary-foreground shadow-xs"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                <Monitor className="size-3.5" />
                <span>Webcam Landscape</span>
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {activeSessionCodeRef.current && (
              <Badge variant="outline" className="text-[10px] font-mono border-primary/40 text-primary bg-primary/10">
                🔴 {activeSessionCodeRef.current}
              </Badge>
            )}
            <span className="text-[11px] text-muted-foreground hidden md:inline">
              {isMobileDevice ? "📱 Mobile Device Detected" : "💻 Desktop Mode"}
            </span>
          </div>
        </div>
      )}

      {/* Video & Real-time HUD Grid */}
      <div
        className={`grid gap-6 ${
          isFullscreen
            ? "h-full w-full grid-cols-1 p-0"
            : isPortraitMode
            ? "grid-cols-1 md:grid-cols-12 max-w-5xl mx-auto"
            : "grid-cols-1 lg:grid-cols-3"
        }`}
      >
        {/* Main Video Viewport (TikTok Style Portrait or Landscape) */}
        <Card
          className={`overflow-hidden border-border/80 flex flex-col relative ${
            isFullscreen
              ? "border-0 rounded-none bg-black h-full w-full"
              : isPortraitMode
              ? "md:col-span-7 lg:col-span-6 mx-auto w-full max-w-[420px] shadow-2xl rounded-3xl border-2"
              : "lg:col-span-2"
          }`}
        >
          {/* Card / HUD Header (Minimalist for Portrait) */}
          <CardHeader
            className={`py-3 px-4 flex flex-row items-center justify-between border-b border-border ${
              isFullscreen
                ? "bg-black/90 border-white/10 z-30"
                : isPortraitMode
                ? "bg-background/90 backdrop-blur-md"
                : ""
            }`}
          >
            <div className="flex items-center gap-2">
              <Camera className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold text-foreground">
                {isFullscreen
                  ? "DriveGuard Cockpit HUD"
                  : isPortraitMode
                  ? "TikTok Cabin Stream"
                  : "Webcam Cabin Feed"}
              </CardTitle>
            </div>

            {/* Target AI Engine Toggle & Status */}
            <div className="flex items-center gap-1.5 sm:gap-2">
              <div className="flex items-center bg-muted/60 rounded-md p-0.5 text-[11px] font-mono border border-border">
                <button
                  type="button"
                  onClick={() => changeBackendType("local")}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    backendType === "local"
                      ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="Local AI Engine (ws://127.0.0.1:8000)"
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

              {/* WS Live Indicator */}
              <Badge
                variant="outline"
                className={`text-[10px] font-mono flex items-center gap-1 ${
                  isWsConnected
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : wsStatus === "connecting"
                    ? "border-amber-500/40 text-amber-500 bg-amber-500/10 animate-pulse"
                    : "border-destructive/30 text-destructive bg-destructive/10"
                }`}
              >
                {isWsConnected ? (
                  <>
                    <Wifi className="size-3" />
                    LIVE
                  </>
                ) : wsStatus === "connecting" ? (
                  <>
                    <RefreshCw className="size-3 animate-spin" />
                    SYNC
                  </>
                ) : (
                  <>
                    <WifiOff className="size-3" />
                    OFFLINE
                  </>
                )}
              </Badge>

              {/* Fullscreen Button */}
              <Button
                size="sm"
                variant="ghost"
                onClick={toggleFullscreen}
                className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                title={isFullscreen ? "Exit Fullscreen" : "Enter Fullscreen"}
              >
                {isFullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
              </Button>
            </div>
          </CardHeader>

          {/* Video Viewport Content (9:16 aspect ratio on portrait) */}
          <CardContent
            className={`p-0 relative bg-black flex-1 flex items-center justify-center overflow-hidden ${
              isFullscreen
                ? "h-full w-full"
                : isPortraitMode
                ? "aspect-[9/16] min-h-[500px]"
                : "aspect-video"
            }`}
          >
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

            {/* Offline Overlay Placeholder (Only visible when camera is truly inactive) */}
            {!isCameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-4 z-10">
                <div className="p-3.5 rounded-full bg-primary/10 text-primary">
                  <Flame className="size-8 animate-pulse text-primary" />
                </div>
                <div className="font-bold text-foreground text-base">
                  {isPortraitMode ? "TikTok-Style Mobile Stream" : "Ready to Start Monitoring"}
                </div>
                <p className="text-xs text-muted-foreground max-w-xs">
                  {isPortraitMode
                    ? "Hold your phone vertically. Face detection, blink kinematics, and fatigue alerts stream in real-time."
                    : "Click below to enable your camera and connect to the edge AI cascade."}
                </p>
                <Button size="sm" onClick={startStream} className="bg-primary text-primary-foreground font-semibold px-6 py-2 rounded-xl shadow-md">
                  <Play className="size-4 mr-1.5" />
                  Start Live Stream
                </Button>
                <div className="text-[10px] font-mono text-muted-foreground/80">
                  Target: {backendType === "local" ? "127.0.0.1:8000 (Local Python)" : "driveguard-x048.onrender.com"}
                </div>
              </div>
            )}

            {/* Disconnected Notice if camera is active but WS dropped */}
            {isCameraActive && !isWsConnected && (
              <div className="absolute top-3 left-3 right-3 flex items-center justify-between p-2 rounded-xl bg-black/85 backdrop-blur-md border border-amber-500/40 text-amber-400 text-xs shadow-md z-20">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="size-4 animate-bounce shrink-0" />
                  <span className="truncate">AI Offline ({activeWsBase})</span>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
                  {backendType === "cloud" && (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => changeBackendType("local")}
                      className="h-7 text-xs border-primary/50 text-primary hover:bg-primary/20"
                    >
                      ⚡ Local
                    </Button>
                  )}
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={reconnectWebSocket}
                    className="h-7 text-xs border-amber-500/50 text-amber-300 hover:bg-amber-500/20"
                  >
                    <RefreshCw className="size-3 mr-1" />
                    Retry
                  </Button>
                </div>
              </div>
            )}

            {/* TikTok-Style Overlay HUD Elements (When Camera is Active) */}
            {isCameraActive && telemetry && (
              <>
                {/* Top Profile / Zone Chip */}
                <div className="absolute top-3 left-3 right-3 flex items-start justify-between pointer-events-none z-10">
                  <div className={`px-3 py-1.5 rounded-full shadow-lg text-xs uppercase tracking-wider font-extrabold ${stateColor}`}>
                    {telemetry.state}
                  </div>
                  <div className="flex flex-col items-end gap-1.5">
                    <div className="px-2.5 py-1 rounded-full bg-black/75 backdrop-blur-md border border-white/15 text-[11px] font-mono text-white shadow-xs">
                      Zone: <strong>{telemetry.pose_zone}</strong>
                    </div>
                    {telemetry.risk_score > 0 && (
                      <div className="px-2.5 py-0.5 rounded-full bg-black/70 border border-white/10 text-[10px] font-mono text-white/90">
                        Fatigue: {(telemetry.risk_score * 100).toFixed(0)}%
                      </div>
                    )}
                  </div>
                </div>

                {/* TikTok-Style Floating Left Live Metrics Bubbles */}
                <div className="absolute bottom-20 left-3 flex flex-col gap-1.5 pointer-events-none z-20">
                  {/* Bubble 1: EAR */}
                  <div className="px-3 py-1.5 rounded-xl bg-black/70 backdrop-blur-md border border-white/15 text-white flex items-center gap-2 shadow-lg w-fit">
                    <Eye className="size-3.5 text-emerald-400 shrink-0" />
                    <span className="text-[11px] font-mono">
                      EAR: <strong className="text-emerald-400">{telemetry.ear.toFixed(3)}</strong>
                    </span>
                  </div>

                  {/* Bubble 2: PERCLOS */}
                  <div className="px-3 py-1.5 rounded-xl bg-black/70 backdrop-blur-md border border-white/15 text-white flex items-center gap-2 shadow-lg w-fit">
                    <Activity className="size-3.5 text-primary shrink-0" />
                    <span className="text-[11px] font-mono">
                      PERCLOS: <strong className={telemetry.perclos_60s > 15 ? "text-rose-400 font-bold" : "text-white"}>
                        {telemetry.perclos_60s.toFixed(1)}%
                      </strong>
                    </span>
                  </div>

                  {/* Bubble 3: Blink Rate & Head Nod */}
                  <div className="px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md border border-white/10 text-white/90 text-[10px] font-mono w-fit">
                    {telemetry.blink_rate.toFixed(0)} blinks/min • Pitch: {telemetry.pitch_deg.toFixed(0)}°
                  </div>

                  {/* Bubble 4: Active Warning Reasons */}
                  {telemetry.reason_codes.map((reason) => (
                    <div
                      key={reason}
                      className="px-2.5 py-1 rounded-lg bg-rose-600/90 text-white text-[10px] font-mono font-bold animate-pulse w-fit shadow-md border border-rose-400/40"
                    >
                      ⚠️ {reason}
                    </div>
                  ))}
                </div>
              </>
            )}
          </CardContent>

          {/* Footer Controls (Floating or Pinned Dock) */}
          <CardFooter className="py-3 px-4 flex items-center justify-between border-t border-border bg-muted/20">
            <div className="flex items-center gap-2">
              {isCameraActive ? (
                <Button size="sm" variant="destructive" onClick={stopStream} className="rounded-xl">
                  <Square className="size-3.5 mr-1.5" />
                  Stop Stream
                </Button>
              ) : (
                <Button size="sm" onClick={startStream} className="bg-primary text-primary-foreground font-semibold rounded-xl">
                  <Play className="size-3.5 mr-1.5" />
                  Start Stream
                </Button>
              )}

              <Button
                size="sm"
                variant="outline"
                onClick={() => setAudioMuted((prev) => !prev)}
                className="h-9 w-9 p-0 rounded-xl"
                title={audioMuted ? "Unmute Buzzer" : "Mute Buzzer"}
              >
                {audioMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4 text-primary" />}
              </Button>
            </div>

            <div className="text-xs font-mono text-muted-foreground flex items-center gap-2">
              <Clock className="size-3.5" />
              <span>
                {Math.floor(sessionDurationSec / 60)}m {sessionDurationSec % 60}s
              </span>
            </div>
          </CardFooter>
        </Card>

        {/* Live Kinematics Telemetry Side Panel (Desktop Widescreen or Side Companion) */}
        {(!isFullscreen || !isPortraitMode) && (
          <div className={`space-y-4 ${isPortraitMode ? "md:col-span-5 lg:col-span-6" : ""}`}>
            <Card className="border-border/80">
              <CardHeader className="pb-3">
                <CardTitle className="text-sm font-semibold flex items-center gap-2">
                  <Eye className="size-4 text-primary" />
                  Live Sensor Telemetry
                </CardTitle>
                <CardDescription className="text-xs">
                  Sub-frame blink kinematics & database recording
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 text-xs">
                {/* EAR Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between font-mono">
                    <span className="text-muted-foreground">Eye Aspect Ratio (EAR):</span>
                    <span className="font-semibold text-foreground">
                      {isWsConnected && telemetry?.ear !== undefined
                        ? telemetry.ear.toFixed(3)
                        : isCameraActive
                        ? "-- (Waiting AI)"
                        : "0.000"}
                    </span>
                  </div>
                  <Progress
                    value={isWsConnected && telemetry?.ear ? Math.min(100, (telemetry.ear / 0.40) * 100) : 0}
                    className="h-2"
                  />
                  <div className="text-[10px] text-muted-foreground font-mono flex justify-between">
                    <span>Closed (&lt;0.18)</span>
                    <span>Dynamic Cutoff ({telemetry?.ear_threshold?.toFixed(3) || "0.220"})</span>
                    <span>Open (&gt;0.30)</span>
                  </div>
                </div>

                {/* PERCLOS Bar */}
                <div className="space-y-1.5">
                  <div className="flex justify-between font-mono">
                    <span className="text-muted-foreground">PERCLOS (60s Window):</span>
                    <span
                      className={`font-semibold ${
                        (telemetry?.perclos_60s || 0) > 15.0 ? "text-rose-500 font-bold" : "text-foreground"
                      }`}
                    >
                      {isWsConnected && telemetry?.perclos_60s !== undefined
                        ? `${telemetry.perclos_60s.toFixed(1)}%`
                        : "0.0%"}
                    </span>
                  </div>
                  <Progress
                    value={isWsConnected ? Math.min(100, ((telemetry?.perclos_60s || 0) / 40.0) * 100) : 0}
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
                  <div className="p-2 rounded-xl bg-muted/40 font-mono">
                    <span className="text-muted-foreground block text-[10px]">Blink Frequency</span>
                    <span className="font-semibold text-sm">
                      {isWsConnected && telemetry?.blink_rate !== undefined
                        ? `${telemetry.blink_rate.toFixed(1)}/min`
                        : "--"}
                    </span>
                  </div>
                  <div className="p-2 rounded-xl bg-muted/40 font-mono">
                    <span className="text-muted-foreground block text-[10px]">Recent Blink Dur</span>
                    <span className="font-semibold text-sm">
                      {isWsConnected && telemetry?.blink_duration_ms
                        ? `${telemetry.blink_duration_ms.toFixed(0)} ms`
                        : "N/A"}
                    </span>
                  </div>
                </div>

                {/* Database Recording Pill */}
                <div className="p-2.5 rounded-xl bg-primary/5 border border-primary/20 flex items-center justify-between text-[11px]">
                  <div className="flex items-center gap-2">
                    <span className={`size-2 rounded-full ${isCameraActive ? "bg-emerald-500 animate-ping" : "bg-muted-foreground"}`} />
                    <span className="font-medium text-foreground">PostgreSQL Session Audit:</span>
                  </div>
                  <span className="font-mono text-primary font-semibold">
                    {activeSessionCodeRef.current || "Ready to Record"}
                  </span>
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
                  { key: "CASCADE", title: "Exp C: Confidence Cascade", desc: "Adaptive Stage 2 gating (saves ~67% power)" },
                  { key: "GEOMETRIC", title: "Exp A: Geometric Only", desc: "Fast EAR landmark mesh only (~2.5ms)" },
                  { key: "CNN", title: "Exp B: CNN Always On", desc: "Continuous MobileNetV3 verifier" },
                ].map((exp) => (
                  <div
                    key={exp.key}
                    onClick={() => switchExperiment(exp.key as "CASCADE" | "GEOMETRIC" | "CNN")}
                    className={`p-2.5 rounded-xl border cursor-pointer text-xs transition-colors ${
                      selectedExperiment === exp.key
                        ? "border-primary bg-primary/10 text-primary font-medium"
                        : "border-border hover:bg-muted/30 text-muted-foreground"
                    }`}
                  >
                    <div className="font-semibold text-foreground flex items-center justify-between">
                      <span>{exp.title}</span>
                      {selectedExperiment === exp.key && (
                        <Badge variant="outline" className="text-[9px] border-primary text-primary">
                          ACTIVE
                        </Badge>
                      )}
                    </div>
                    <div className="text-[11px] text-muted-foreground">{exp.desc}</div>
                  </div>
                ))}
              </CardContent>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
