"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  ExperimentMode,
  FrameTelemetry,
  ServerTelemetryMessage,
} from "@/types/telemetry";

export interface UseAiTelemetryStreamOptions {
  sessionId?: string;
  driverId?: string;
  defaultBackend?: "cloud" | "local";
  targetFps?: number; // Target 15-20 FPS (~50-65ms)
  onTelemetry?: (telemetry: FrameTelemetry) => void;
  onError?: (err: Error) => void;
}

export function useAiTelemetryStream(options: UseAiTelemetryStreamOptions = {}) {
  const {
    sessionId: initialSessionId,
    driverId = "driver_default",
    defaultBackend = "cloud",
    targetFps = 20,
    onTelemetry,
    onError,
  } = options;

  // Stable session identifier that does NOT regenerate on every render
  const stableSessionId = useMemo(
    () => initialSessionId || `dg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
    [initialSessionId]
  );

  const cloudWsBase =
    process.env.NEXT_PUBLIC_FASTAPI_WS_URL || "wss://driveguard-x048.onrender.com";
  const localWsBase = "ws://127.0.0.1:8000";

  const [backendType, setBackendType] = useState<"cloud" | "local">(defaultBackend);
  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  >("disconnected");
  const [telemetry, setTelemetry] = useState<FrameTelemetry | null>(null);
  const [roundtripLatencyMs, setRoundtripLatencyMs] = useState<number>(0);
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>(
    "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
  );

  const wsRef = useRef<WebSocket | null>(null);
  const offscreenCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const loopActiveRef = useRef<boolean>(false);
  const isFrameInFlightRef = useRef<boolean>(false);
  const lastSendTimeRef = useRef<number>(0);
  const reconnectAttemptsRef = useRef<number>(0);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);

  // Synchronized callback refs
  const onTelemetryRef = useRef(onTelemetry);
  useEffect(() => {
    onTelemetryRef.current = onTelemetry;
  }, [onTelemetry]);

  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  // Construct WebSocket target URL
  const getWsUrl = useCallback(() => {
    const base = backendType === "cloud" ? cloudWsBase : localWsBase;
    return `${base}/ws/v1/session/${stableSessionId}?driver_id=${driverId}&experiment_mode=${experimentMode}`;
  }, [backendType, cloudWsBase, stableSessionId, driverId, experimentMode]);

  // Clean disconnect
  const disconnectWebSocket = useCallback(() => {
    loopActiveRef.current = false;
    isFrameInFlightRef.current = false;

    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }

    if (wsRef.current) {
      try {
        wsRef.current.onopen = null;
        wsRef.current.onmessage = null;
        wsRef.current.onerror = null;
        wsRef.current.onclose = null;
        wsRef.current.close();
      } catch (_) {}
      wsRef.current = null;
    }

    setConnectionStatus("disconnected");
    setTelemetry(null);
  }, []);

  // Connect WebSocket
  const connectWebSocket = useCallback(() => {
    disconnectWebSocket();
    setConnectionStatus("connecting");

    const url = getWsUrl();
    try {
      const ws = new WebSocket(url);
      ws.binaryType = "arraybuffer";
      wsRef.current = ws;

      ws.onopen = () => {
        setConnectionStatus("connected");
        reconnectAttemptsRef.current = 0;
        isFrameInFlightRef.current = false;
      };

      ws.onmessage = (event) => {
        try {
          if (typeof event.data === "string") {
            const msg: ServerTelemetryMessage = JSON.parse(event.data);

            if (msg.type === "TELEMETRY" && msg.telemetry) {
              const now = performance.now();
              const rtt = Math.round(now - lastSendTimeRef.current);
              setRoundtripLatencyMs(rtt);

              const enriched: FrameTelemetry = {
                ...msg.telemetry,
                network_roundtrip_ms: rtt,
              };

              setTelemetry(enriched);
              if (onTelemetryRef.current) {
                onTelemetryRef.current(enriched);
              }

              // Lock-step ping-pong: release lock so next frame can be grabbed
              isFrameInFlightRef.current = false;
            } else if (msg.type === "SESSION_READY") {
              setConnectionStatus("connected");
              isFrameInFlightRef.current = false;
            } else if (msg.type === "FATAL_ERROR") {
              console.error("Backend AI fatal error:", msg.detail);
              setConnectionStatus("error");
            }
          }
        } catch (parseErr) {
          console.warn("Failed to parse server telemetry message:", parseErr);
          isFrameInFlightRef.current = false;
        }
      };

      ws.onerror = (err) => {
        console.warn("WebSocket telemetry error:", err);
        setConnectionStatus("error");
        isFrameInFlightRef.current = false;
      };

      ws.onclose = () => {
        if (loopActiveRef.current) {
          setConnectionStatus("reconnecting");
          isFrameInFlightRef.current = false;

          const delay = Math.min(1000 * Math.pow(1.5, reconnectAttemptsRef.current), 5000);
          reconnectAttemptsRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            if (loopActiveRef.current) {
              connectWebSocket();
            }
          }, delay);
        } else {
          setConnectionStatus("disconnected");
        }
      };
    } catch (wsInitErr) {
      console.error("WebSocket init error:", wsInitErr);
      setConnectionStatus("error");
    }
  }, [disconnectWebSocket, getWsUrl]);

  // Sends single frame to WebSocket with synchronous lock-step protection
  const sendFrameFromVideo = useCallback(
    (videoElement: HTMLVideoElement, isMirrored = true) => {
      const ws = wsRef.current;
      if (!ws || ws.readyState !== WebSocket.OPEN) return;
      if (isFrameInFlightRef.current) return; // Strict lock-step: never queue frames

      const vw = videoElement.videoWidth;
      const vh = videoElement.videoHeight;
      if (!vw || !vh || videoElement.readyState < 2) return;

      if (!offscreenCanvasRef.current) {
        offscreenCanvasRef.current = document.createElement("canvas");
      }
      const canvas = offscreenCanvasRef.current;

      // Maintain natural unwarped aspect ratio at 640px base width
      const targetWidth = 640;
      const targetHeight = Math.round((vh / vw) * targetWidth);

      canvas.width = targetWidth;
      canvas.height = targetHeight;

      const ctx = canvas.getContext("2d", { willReadFrequently: false });
      if (!ctx) return;

      ctx.save();
      if (isMirrored) {
        // Pre-inference horizontal flip matching cv2.flip(frame, 1) in client_webcam_test.py
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }

      ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
      ctx.restore();

      // Lock synchronously before calling async toBlob to prevent race conditions
      isFrameInFlightRef.current = true;
      lastSendTimeRef.current = performance.now();

      // Encode to JPEG at high fidelity (quality 0.80)
      canvas.toBlob(
        async (blob) => {
          if (!blob || !ws || ws.readyState !== WebSocket.OPEN) {
            isFrameInFlightRef.current = false;
            return;
          }

          try {
            const buffer = await blob.arrayBuffer();
            if (ws.readyState === WebSocket.OPEN) {
              ws.send(buffer);
            } else {
              isFrameInFlightRef.current = false;
            }
          } catch (_) {
            isFrameInFlightRef.current = false;
          }
        },
        "image/jpeg",
        0.8
      );
    },
    []
  );

  // Starts the adaptive ping-pong loop driven by requestAnimationFrame / lock-step pacing
  const startStreamingLoop = useCallback(
    (videoElement: HTMLVideoElement, isMirrored = true) => {
      loopActiveRef.current = true;
      connectWebSocket();

      const frameIntervalMs = 1000 / targetFps;
      let lastAttempt = 0;

      const pump = () => {
        if (!loopActiveRef.current) return;

        const now = performance.now();
        // Send next frame only if target interval elapsed AND previous frame returned
        if (now - lastAttempt >= frameIntervalMs && !isFrameInFlightRef.current) {
          lastAttempt = now;
          sendFrameFromVideo(videoElement, isMirrored);
        }

        // Heartbeat timeout: if a frame is stuck in flight for > 1500ms, unlock pipeline
        if (isFrameInFlightRef.current && now - lastSendTimeRef.current > 1500) {
          isFrameInFlightRef.current = false;
        }

        requestAnimationFrame(pump);
      };

      requestAnimationFrame(pump);
    },
    [connectWebSocket, sendFrameFromVideo, targetFps]
  );

  // Stop streaming loop
  const stopStreamingLoop = useCallback(() => {
    loopActiveRef.current = false;
    disconnectWebSocket();
  }, [disconnectWebSocket]);

  // Send control commands (switch experiment, calibration, save profile)
  const sendCommand = useCallback((cmd: Record<string, any>) => {
    const ws = wsRef.current;
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(cmd));
    }
  }, []);

  const switchExperiment = useCallback(
    (mode: ExperimentMode) => {
      setExperimentMode(mode);
      sendCommand({ action: "SWITCH_EXPERIMENT", mode });
    },
    [sendCommand]
  );

  const startCalibration = useCallback(
    (fourZone = true, durationSec = 4.0) => {
      sendCommand({
        action: "START_CALIBRATION",
        four_zone: fourZone,
        zone_duration_sec: durationSec,
      });
    },
    [sendCommand]
  );

  const cancelCalibration = useCallback(() => {
    sendCommand({ action: "CANCEL_CALIBRATION" });
  }, [sendCommand]);

  const saveProfile = useCallback(
    (driverName = "Driver", vehicleId = "DriveGuard-01") => {
      sendCommand({
        action: "SAVE_PROFILE",
        driver_name: driverName,
        vehicle_id: vehicleId,
      });
    },
    [sendCommand]
  );

  // Switch backend target between Cloud Render and Localhost
  const switchBackend = useCallback(
    (type: "cloud" | "local") => {
      setBackendType(type);
      if (loopActiveRef.current) {
        setTimeout(connectWebSocket, 100);
      }
    },
    [connectWebSocket]
  );

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      loopActiveRef.current = false;
      disconnectWebSocket();
    };
  }, [disconnectWebSocket]);

  return {
    isConnected: connectionStatus === "connected",
    connectionStatus,
    telemetry,
    roundtripLatencyMs,
    experimentMode,
    backendType,
    switchBackend,
    startStreamingLoop,
    stopStreamingLoop,
    sendCommand,
    switchExperiment,
    startCalibration,
    cancelCalibration,
    saveProfile,
    reconnect: connectWebSocket,
  };
}
