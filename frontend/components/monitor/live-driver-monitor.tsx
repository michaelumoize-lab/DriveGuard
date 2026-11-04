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
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { useSystemHealth } from "@/hooks/use-system-health";
import {
  Activity,
  Camera,
  Play,
  Square,
  Volume2,
  VolumeX,
  ShieldAlert,
  AlertTriangle,
  Eye,
  RefreshCw,
  Cpu,
  Wifi,
  WifiOff,
  Maximize2,
  Minimize2,
  Smartphone,
  Monitor,
  Flame,
  Clock,
  Crop,
  Sparkles,
  Zap,
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
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

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

  // Single Source of Truth: System Health from Navbar Hook
  const { health, refresh: refreshHealth } = useSystemHealth();
  const fastApiStatus = health?.services?.fastapi?.status;

  // States
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isWsConnected, setIsWsConnected] = useState(false);
  const [wsStatus, setWsStatus] = useState<"disconnected" | "connecting" | "connected" | "reconnecting" | "error">("disconnected");
  const [telemetry, setTelemetry] = useState<NormalizedTelemetry | null>(null);
  const [audioMuted, setAudioMuted] = useState(false);
  const [selectedExperiment, setSelectedExperiment] = useState<"CASCADE" | "GEOMETRIC" | "CNN">("CASCADE");
  const [sessionStartTime, setSessionStartTime] = useState<number | null>(null);
  const [sessionDurationSec, setSessionDurationSec] = useState(0);
  const [isFullscreen, setIsFullscreen] = useState(false);

  // Device & Layout Modes
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("landscape");
  // Framing Mode: "fit" (wide FOV, no crop) vs "cover" (fullscreen fill)
  const [framingMode, setFramingMode] = useState<"fit" | "cover">("fit");

  // Pre-Flight AI Warm-up Dialog State
  const [isWarmingUpAI, setIsWarmingUpAI] = useState(false);
  const [warmUpCountdown, setWarmUpCountdown] = useState(25);

  // Backend target selector (default to Cloud Render unless explicitly on localhost)
  const [backendType, setBackendType] = useState<"local" | "cloud">("cloud");

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
          // Mobile default is Wide FOV (fit) so camera is never over-zoomed
          setFramingMode("fit");
          // Mobile devices cannot run Python on 127.0.0.1, always use cloud
          setBackendType("cloud");
        }
      };

      checkMobile();
      window.addEventListener("resize", checkMobile);
      return () => window.removeEventListener("resize", checkMobile);
    }
  }, []);

  // Initialize preference from localStorage if available (on desktop localhost only)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const isLocalHost = window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
      if (isLocalHost) {
        const saved = localStorage.getItem("driveguard_backend_type") as "local" | "cloud" | null;
        if (saved) {
          setBackendType(saved);
        } else {
          setBackendType("local");
        }
      } else {
        setBackendType("cloud");
      }
    }
  }, []);

  // Screen Wake Lock API (keeps mobile screen awake while monitoring)
  const requestWakeLock = useCallback(async () => {
    try {
      if (typeof navigator !== "undefined" && "wakeLock" in navigator) {
        wakeLockRef.current = await navigator.wakeLock.request("screen");
      }
    } catch {
      // Browser policy or background tab restriction
    }
  }, []);

  const releaseWakeLock = useCallback(async () => {
    if (wakeLockRef.current) {
      try {
        await wakeLockRef.current.release();
      } catch {
        // ignore
      } finally {
        wakeLockRef.current = null;
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
    const isLocalHost = typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
    if (type === "local" && !isLocalHost) {
      toast.warning("Local engine (127.0.0.1:8000) is only accessible on the host computer running Python. Staying on Cloud AI.");
      return;
    }
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

  // Audio warning buzzer on CRITICAL state & offline chime
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

  // Connect WebSocket to AI Engine with robust retry and NO mobile localhost fallback
  const connectWebSocket = useCallback((sessionId: string) => {
    setWsStatus("connecting");
    const wsUrl = `${activeWsBase}/ws/v1/session/${sessionId}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    // Generous 12-second timeout for cloud cold starts
    const connectTimeout = setTimeout(() => {
      if (ws.readyState !== WebSocket.OPEN) {
        console.warn("Cloud connection taking longer than expected. Retrying...");
      }
    }, 12000);

    ws.onopen = () => {
      clearTimeout(connectTimeout);
      setIsWsConnected(true);
      setWsStatus("connected");
      toast.success(`Connected to AI Engine (${backendType === "local" ? "Local Engine" : "Cloud Engine"})!`);

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
        if (data.type === "SESSION_READY") {
          setIsWsConnected(true);
          setWsStatus("connected");
          return;
        }

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
      const isLocalHost = typeof window !== "undefined" && (window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1");
      if (backendType === "cloud" && isLocalHost) {
        toast.warning("Cloud backend unreachable. Falling back to Local AI Engine (127.0.0.1:8000)...");
        setBackendType("local");
      } else {
        toast.error(`AI connection interrupted on ${activeWsBase}. Retrying connection...`);
      }
    };

    ws.onclose = () => {
      clearTimeout(connectTimeout);
      setIsWsConnected(false);
      setWsStatus("disconnected");
      isFrameInFlightRef.current = false;

      // Exponential auto-reconnect if camera is still streaming
      if (isCameraActive) {
        setWsStatus("reconnecting");
        if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
        reconnectTimeoutRef.current = setTimeout(() => {
          if (isCameraActive) {
            const nextSessionId = `web_${Date.now()}`;
            connectWebSocket(nextSessionId);
          }
        }, 3000);
      }
    };

    return ws;
  }, [activeWsBase, backendType, selectedExperiment, logAlertToDatabase, isCameraActive]);

  // Execute actual camera capture & start streaming
  const executeCameraStreaming = async () => {
    try {
      const isPortrait = viewOrientation === "portrait" || isMobileDevice;

      // Video constraints: Natural sensor aspect ratio without artificial digital zoom
      const videoConstraints: MediaTrackConstraints = isPortrait
        ? {
            facingMode: { ideal: "user" },
            aspectRatio: { ideal: 9 / 16 },
            width: { ideal: 720 },
            height: { ideal: 1280 },
          }
        : {
            facingMode: { ideal: "user" },
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
      requestWakeLock();

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

      // 3. Backpressure-Controlled Frame Grabber (respects natural sensor aspect ratio)
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

  // Pre-Flight Check before starting camera stream
  const startStream = async () => {
    // If backend is cloud and navbar/health indicates offline, probe first
    if (backendType === "cloud" && fastApiStatus !== "online") {
      setIsWarmingUpAI(true);
      setWarmUpCountdown(25);
      return;
    }

    await executeCameraStreaming();
  };

  // Launch in offline preview mode (skips waiting for AI cloud)
  const launchOfflinePreviewMode = async () => {
    setIsWarmingUpAI(false);
    toast.info("Starting in Local Camera Preview Mode. AI features will connect once ready.");
    await executeCameraStreaming();
  };

  // Auto-poll probe while warming up AI dialog is open
  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (isWarmingUpAI) {
      timer = setInterval(async () => {
        setWarmUpCountdown((prev) => Math.max(0, prev - 1));
        const res = await refreshHealth();
        if (res?.services?.fastapi?.status === "online") {
          setIsWarmingUpAI(false);
          toast.success("AI Cloud Engine is online! Launching live stream...");
          executeCameraStreaming();
        }
      }, 3000);
    }
    return () => clearInterval(timer);
  }, [isWarmingUpAI, refreshHealth]);

  // Teardown everything cleanly & finalize session in DB
  const stopStream = async () => {
    const sessionCode = activeSessionCodeRef.current;
    const duration = sessionStartTime ? Math.floor((Date.now() - sessionStartTime) / 1000) : 0;
    releaseWakeLock();

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (sessionCode) {
      try {
        const m = sessionMetricsRef.current;
        const avgFps = m.totalFrames > 0 ? Number((m.fpsSum / m.totalFrames).toFixed(1)) : 0;
        const avgPerclos = m.totalFrames > 0 ? Number((m.perclosSum / m.totalFrames).toFixed(1)) : 0;
        const avgBlinkRate = m.totalFrames > 0 ? Number((m.blinkRateSum / m.totalFrames).toFixed(1)) : 0;

        await fetch("/api/sessions/end", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionCode,
            durationSeconds: duration,
            totalFrames: m.totalFrames,
            averageFps: avgFps,
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
      if (reconnectTimeoutRef.current) clearTimeout(reconnectTimeoutRef.current);
      if (wsRef.current) wsRef.current.close();
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((track) => track.stop());
      }
      releaseWakeLock();
    };
  }, [releaseWakeLock]);

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
      className={`space-y-4 sm:space-y-6 transition-all ${
        isFullscreen
          ? "fixed inset-0 z-50 bg-black p-0 sm:p-4 flex flex-col justify-between overflow-hidden h-[100dvh] w-screen"
          : ""
      }`}
    >
      {/* Top Mobile/Desktop Orientation & Framing Controls */}
      {!isFullscreen && (
        <div className="flex flex-wrap items-center justify-between gap-2 bg-muted/40 p-2 sm:p-2.5 rounded-xl border border-border text-xs">
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-muted-foreground hidden sm:inline">Orientation:</span>
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
                <span>Portrait HUD</span>
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
                <span>Landscape Cockpit</span>
              </button>
            </div>

            {/* Framing Mode Toggle (Wide FOV vs Fill) */}
            <button
              type="button"
              onClick={() => setFramingMode((prev) => (prev === "fit" ? "cover" : "fit"))}
              className="px-2.5 py-1 rounded-lg border border-border bg-background hover:bg-muted text-muted-foreground hover:text-foreground text-xs flex items-center gap-1.5 transition-colors"
              title="Toggle Camera Framing (Wide FOV vs Fill)"
            >
              <Crop className="size-3.5 text-primary" />
              <span>{framingMode === "fit" ? "Wide FOV (No Crop)" : "Fill Screen"}</span>
            </button>
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
        className={`grid gap-4 sm:gap-6 ${
          isFullscreen
            ? "h-full w-full grid-cols-1 p-0"
            : isPortraitMode
            ? "grid-cols-1 md:grid-cols-12 max-w-5xl mx-auto"
            : "grid-cols-1 lg:grid-cols-3"
        }`}
      >
        {/* Main Video Viewport (Portrait HUD or Landscape) */}
        <Card
          className={`overflow-hidden border-border/80 flex flex-col relative ${
            isFullscreen
              ? "border-0 rounded-none bg-black h-full w-full"
              : isPortraitMode
              ? "md:col-span-7 lg:col-span-6 mx-auto w-full max-w-[440px] shadow-2xl rounded-2xl sm:rounded-3xl border-2"
              : "lg:col-span-2"
          }`}
        >
          {/* Card / HUD Header */}
          <CardHeader
            className={`py-2.5 sm:py-3 px-3 sm:px-4 flex flex-row items-center justify-between border-b border-border ${
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
                  ? "Driver Focus Stream"
                  : "Cabin Camera Stream"}
              </CardTitle>
            </div>

            {/* Target AI Engine Toggle & Status */}
            <div className="flex items-center gap-1.5 sm:gap-2">
              {/* WS Live Indicator */}
              <Badge
                variant="outline"
                className={`text-[10px] font-mono flex items-center gap-1 ${
                  isWsConnected
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : wsStatus === "connecting" || wsStatus === "reconnecting"
                    ? "border-amber-500/40 text-amber-500 bg-amber-500/10 animate-pulse"
                    : "border-destructive/30 text-destructive bg-destructive/10"
                }`}
              >
                {isWsConnected ? (
                  <>
                    <Wifi className="size-3" />
                    LIVE
                  </>
                ) : wsStatus === "connecting" || wsStatus === "reconnecting" ? (
                  <>
                    <RefreshCw className="size-3 animate-spin" />
                    SYNCING
                  </>
                ) : (
                  <>
                    <WifiOff className="size-3" />
                    PREVIEW
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

          {/* Video Viewport Content */}
          <CardContent
            className={`p-0 relative bg-black flex-1 flex items-center justify-center overflow-hidden ${
              isFullscreen
                ? "h-full w-full"
                : isPortraitMode
                ? "aspect-[9/16] min-h-[460px] max-h-[75vh]"
                : "aspect-video"
            }`}
          >
            {/* Native Video Element */}
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              className={`w-full h-full -scale-x-100 transition-all ${
                framingMode === "cover" ? "object-cover" : "object-contain"
              }`}
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
                  {isPortraitMode ? "Mobile Driver Focus Stream" : "Ready to Start Monitoring"}
                </div>
                <p className="text-xs text-muted-foreground max-w-xs">
                  {isPortraitMode
                    ? "Mount your phone in portrait. Sub-frame blink kinematics, 3D gaze zones, and fatigue alerts stream in real-time."
                    : "Enable camera to stream cabin video to the DriveGuard confidence cascade."}
                </p>
                <Button size="sm" onClick={startStream} className="bg-primary text-primary-foreground font-semibold px-6 py-2 rounded-xl shadow-md min-h-[44px]">
                  <Play className="size-4 mr-1.5" />
                  Start Live Stream
                </Button>
                <div className="text-[10px] font-mono text-muted-foreground/80 flex items-center gap-1.5">
                  <span className={`size-1.5 rounded-full ${fastApiStatus === "online" ? "bg-emerald-500" : "bg-amber-500 animate-ping"}`} />
                  <span>Target: {backendType === "local" ? "127.0.0.1:8000 (Local Python)" : "driveguard-x048.onrender.com"}</span>
                </div>
              </div>
            )}

            {/* Disconnected Notice if camera is active but WS dropped */}
            {isCameraActive && !isWsConnected && (
              <div className="absolute top-3 left-3 right-3 flex items-center justify-between p-2 rounded-xl bg-black/85 backdrop-blur-md border border-amber-500/40 text-amber-400 text-xs shadow-md z-20">
                <div className="flex items-center gap-2">
                  <AlertTriangle className="size-4 animate-bounce shrink-0" />
                  <span className="truncate">
                    {wsStatus === "reconnecting" ? "Reconnecting to AI Cloud..." : "Camera in Local Preview Mode"}
                  </span>
                </div>
                <div className="flex items-center gap-1.5 shrink-0">
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

            {/* Floating Cockpit HUD Elements (When Camera is Active) */}
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

                {/* Live Telemetry Gauges (Left Stack) */}
                <div className="absolute bottom-20 left-3 flex flex-col gap-1.5 pointer-events-none z-20">
                  {/* Gauge 1: EAR */}
                  <div className="px-3 py-1.5 rounded-xl bg-black/70 backdrop-blur-md border border-white/15 text-white flex items-center gap-2 shadow-lg w-fit">
                    <Eye className="size-3.5 text-emerald-400 shrink-0" />
                    <span className="text-[11px] font-mono">
                      EAR: <strong className="text-emerald-400">{telemetry.ear.toFixed(3)}</strong>
                    </span>
                  </div>

                  {/* Gauge 2: PERCLOS */}
                  <div className="px-3 py-1.5 rounded-xl bg-black/70 backdrop-blur-md border border-white/15 text-white flex items-center gap-2 shadow-lg w-fit">
                    <Activity className="size-3.5 text-primary shrink-0" />
                    <span className="text-[11px] font-mono">
                      PERCLOS: <strong className={telemetry.perclos_60s > 15 ? "text-rose-400 font-bold" : "text-white"}>
                        {telemetry.perclos_60s.toFixed(1)}%
                      </strong>
                    </span>
                  </div>

                  {/* Gauge 3: Blink Rate & Head Nod */}
                  <div className="px-3 py-1 rounded-xl bg-black/70 backdrop-blur-md border border-white/10 text-white/90 text-[10px] font-mono w-fit">
                    {telemetry.blink_rate.toFixed(0)} blinks/min • Pitch: {telemetry.pitch_deg.toFixed(0)}° • Yaw: {telemetry.yaw_deg.toFixed(0)}°
                  </div>

                  {/* Gauge 4: Active Warning Reasons */}
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

          {/* Footer Controls */}
          <CardFooter className="py-2.5 sm:py-3 px-3 sm:px-4 flex items-center justify-between border-t border-border bg-muted/20">
            <div className="flex items-center gap-2">
              {isCameraActive ? (
                <Button size="sm" variant="destructive" onClick={stopStream} className="rounded-xl min-h-[40px]">
                  <Square className="size-3.5 mr-1.5" />
                  Stop Stream
                </Button>
              ) : (
                <Button size="sm" onClick={startStream} className="bg-primary text-primary-foreground font-semibold rounded-xl min-h-[40px]">
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

              {/* In-stream framing toggle */}
              <Button
                size="sm"
                variant="outline"
                onClick={() => setFramingMode((prev) => (prev === "fit" ? "cover" : "fit"))}
                className="h-9 w-9 p-0 rounded-xl"
                title={framingMode === "fit" ? "Switch to Fill Screen" : "Switch to Wide FOV"}
              >
                <Crop className="size-4 text-muted-foreground" />
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

        {/* Live Kinematics Telemetry Companion Panel */}
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

      {/* Pre-Flight AI Cloud Warm-Up Dialog */}
      <Dialog open={isWarmingUpAI} onOpenChange={setIsWarmingUpAI}>
        <DialogContent className="max-w-md p-6">
          <DialogHeader className="space-y-2 text-center items-center">
            <div className="p-3.5 rounded-full bg-primary/10 text-primary w-fit">
              <RefreshCw className="size-8 animate-spin text-primary" />
            </div>
            <DialogTitle className="text-lg font-bold">Waking Up DriveGuard AI Engine</DialogTitle>
            <DialogDescription className="text-xs text-muted-foreground">
              The Render Cloud AI microservice spins down after idle periods to save resources. It is currently initializing neural weights and MediaPipe components (~{warmUpCountdown}s).
            </DialogDescription>
          </DialogHeader>

          <div className="py-4 space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-muted-foreground">Container Status:</span>
              <span className="text-amber-500 font-semibold animate-pulse">Initializing...</span>
            </div>
            <Progress value={Math.max(10, ((25 - warmUpCountdown) / 25) * 100)} className="h-2" />
            <p className="text-[11px] text-muted-foreground text-center">
              Stream will automatically launch as soon as the handshake succeeds.
            </p>
          </div>

          <DialogFooter className="flex flex-col sm:flex-row gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={launchOfflinePreviewMode}
              className="w-full text-xs"
            >
              Start in Camera Preview Mode
            </Button>
            <Button
              variant="default"
              size="sm"
              disabled
              className="w-full text-xs bg-primary/80"
            >
              <RefreshCw className="size-3.5 mr-1.5 animate-spin" />
              Waiting for Cloud AI...
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
