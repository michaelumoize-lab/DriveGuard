"use client";

/**
 * DriveGuard — Python Local AI Engine Hook (Phase 4C).
 *
 * Connects directly to the loopback-guarded FastAPI local camera API:
 *   - HTTP REST:  http://127.0.0.1:8000/api/v1/local/*
 *   - MJPEG Feed: http://127.0.0.1:8000/api/v1/local/preview.mjpg
 *   - WebSocket:  ws://127.0.0.1:8000/ws/local/telemetry
 *
 * Strictly NEVER calls `navigator.mediaDevices.getUserMedia`: Python acquires the
 * OpenCV/Phone camera directly and releases it synchronously on `POST /api/v1/local/stop`.
 * Reconciles Python's local JSON `ProfileStore` with PostgreSQL (`engineType="PYTHON"`)
 * asynchronously when online without ever blocking offline local operation.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import { LOCAL_FASTAPI_URL, LOCAL_FASTAPI_WS_URL } from "@/lib/fastapi-url";
import {
  normalizeLocalTelemetryPayload,
  PythonReconcileOutcome,
  reconcilePythonProfileWithServer,
} from "@/lib/monitor/engine-adapter";
import type {
  BackendProcessingStatus,
  CalibrationPhase,
  FrameTelemetry,
  SafetyEvent,
} from "@/types/telemetry";

export interface LocalPythonDevice {
  id: number;
  label: string;
  type: string;
  in_use: boolean;
}

export interface LocalPythonSessionRecord {
  session_id: string;
  session_code: string;
  driver_id: string;
  engine_type: string;
  camera_source: string;
  status: string;
  started_at: string;
  ended_at?: string | null;
  duration_seconds: number;
  total_frames: number;
  warning_alerts_count: number;
  critical_alerts_count: number;
}

export interface UsePythonLocalEngineOptions {
  driverId?: string;
  localBaseUrl?: string;
  localWsBaseUrl?: string;
  enabled?: boolean;
  onTelemetry?: (telemetry: FrameTelemetry) => void;
  onSafetyEvent?: (event: SafetyEvent) => void;
  onCalibration?: (calibration: CalibrationPhase) => void;
  onError?: (err: Error) => void;
}

export function usePythonLocalEngine(options: UsePythonLocalEngineOptions = {}) {
  const {
    driverId = "driver_default",
    localBaseUrl = LOCAL_FASTAPI_URL,
    localWsBaseUrl = LOCAL_FASTAPI_WS_URL,
    enabled = true,
    onTelemetry,
    onSafetyEvent,
    onCalibration,
    onError,
  } = options;

  const baseHttp = localBaseUrl.replace(/\/+$/, "");
  const baseWs = localWsBaseUrl.replace(/\/+$/, "");

  const [isRunning, setIsRunning] = useState(false);
  const [isStarting, setIsStarting] = useState(false);
  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  >("disconnected");
  const [backendProcessingStatus, setBackendProcessingStatus] =
    useState<BackendProcessingStatus>("idle");
  const [telemetry, setTelemetry] = useState<FrameTelemetry | null>(null);
  const [calibration, setCalibration] = useState<CalibrationPhase | null>(null);
  const [isCalibrated, setIsCalibrated] = useState(false);
  const [activeProfile, setActiveProfile] = useState<Record<string, any> | null>(null);
  const [profileSyncStatus, setProfileSyncStatus] = useState<string>("LOCAL_ONLY");
  const [activeSession, setActiveSession] = useState<LocalPythonSessionRecord | null>(null);
  const [mjpegPreviewUrl, setMjpegPreviewUrl] = useState<string | null>(null);
  const [devices, setDevices] = useState<LocalPythonDevice[]>([]);
  const [localSessions, setLocalSessions] = useState<LocalPythonSessionRecord[]>([]);
  const [error, setError] = useState<string | null>(null);

  const wsRef = useRef<WebSocket | null>(null);
  const onTelemetryRef = useRef(onTelemetry);
  const onSafetyEventRef = useRef(onSafetyEvent);
  const onCalibrationRef = useRef(onCalibration);
  const onErrorRef = useRef(onError);

  useEffect(() => {
    onTelemetryRef.current = onTelemetry;
    onSafetyEventRef.current = onSafetyEvent;
    onCalibrationRef.current = onCalibration;
    onErrorRef.current = onError;
  }, [onTelemetry, onSafetyEvent, onCalibration, onError]);

  const closeWebSocket = useCallback(() => {
    const ws = wsRef.current;
    wsRef.current = null;
    if (ws) {
      try {
        ws.onopen = null;
        ws.onmessage = null;
        ws.onerror = null;
        ws.onclose = null;
        if (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING) {
          ws.close();
        }
      } catch {
        // ignore close errors
      }
    }
  }, []);

  const reconcileProfile = useCallback(async (): Promise<PythonReconcileOutcome> => {
    const outcome = await reconcilePythonProfileWithServer({
      driverId,
      localBaseUrl: baseHttp,
    });
    if (outcome.localProfile) {
      setActiveProfile(outcome.localProfile);
      setIsCalibrated(Boolean(outcome.localProfile.is_calibrated));
    }
    setProfileSyncStatus(outcome.status);
    return outcome;
  }, [baseHttp, driverId]);

  const refreshDevices = useCallback(async () => {
    try {
      const res = await fetch(`${baseHttp}/api/v1/local/devices`, {
        method: "GET",
        headers: { Accept: "application/json" },
      });
      if (!res.ok) return [];
      const data = await res.json();
      const list = Array.isArray(data?.devices) ? (data.devices as LocalPythonDevice[]) : [];
      setDevices(list);
      return list;
    } catch {
      return [];
    }
  }, [baseHttp]);

  const refreshSessions = useCallback(async () => {
    try {
      const res = await fetch(
        `${baseHttp}/api/v1/local/sessions?driver_id=${encodeURIComponent(driverId)}&limit=20`,
        { method: "GET", headers: { Accept: "application/json" } }
      );
      if (!res.ok) return [];
      const data = await res.json();
      const list = Array.isArray(data?.sessions)
        ? (data.sessions as LocalPythonSessionRecord[])
        : [];
      setLocalSessions(list);
      return list;
    } catch {
      return [];
    }
  }, [baseHttp, driverId]);

  const connectWebSocket = useCallback(() => {
    closeWebSocket();
    setConnectionStatus("connecting");
    setBackendProcessingStatus("connecting");

    try {
      const ws = new WebSocket(`${baseWs}/ws/local/telemetry`);
      wsRef.current = ws;

      ws.onopen = () => {
        if (wsRef.current !== ws) return;
        setConnectionStatus("connected");
        setBackendProcessingStatus("awaiting_telemetry");
        setError(null);
      };

      ws.onmessage = (event) => {
        if (wsRef.current !== ws) return;
        try {
          const msg = JSON.parse(String(event.data));
          if (!msg || typeof msg !== "object") return;

          if (msg.type === "TELEMETRY") {
            const normalized = normalizeLocalTelemetryPayload(msg);
            setTelemetry(normalized);
            setBackendProcessingStatus("processing");
            if (typeof msg.is_calibrated === "boolean") {
              setIsCalibrated(msg.is_calibrated);
            }
            if (msg.calibration) {
              setCalibration(msg.calibration as CalibrationPhase);
              onCalibrationRef.current?.(msg.calibration as CalibrationPhase);
            }
            onTelemetryRef.current?.(normalized);
          } else if (msg.type === "SAFETY_EVENT") {
            const safetyEv: SafetyEvent = {
              type: "SAFETY_EVENT",
              event_id: String(msg.event_id ?? `evt_py_${Date.now()}`),
              event_seq: String(msg.event_seq ?? Date.now()),
              previous_state: msg.previous_state ?? "NORMAL",
              new_state: msg.new_state ?? "WARNING",
              timestamp: Number(msg.timestamp ?? Date.now() / 1000),
              metrics: msg.metrics ?? {},
            } as SafetyEvent;
            onSafetyEventRef.current?.(safetyEv);
          } else if (msg.type === "CALIBRATION" && msg.calibration) {
            const cal = msg.calibration as CalibrationPhase;
            setCalibration(cal);
            if (cal.is_completed && !cal.rejection_reason && cal.profile) {
              setIsCalibrated(true);
              setActiveProfile(cal.profile as unknown as Record<string, any>);
              void reconcileProfile();
            }
            onCalibrationRef.current?.(cal);
          } else if (msg.type === "STATUS") {
            if (typeof msg.running === "boolean") {
              setIsRunning(msg.running);
              if (!msg.running) {
                setMjpegPreviewUrl(null);
                setBackendProcessingStatus("idle");
              }
            }
            if (typeof msg.is_calibrated === "boolean") {
              setIsCalibrated(msg.is_calibrated);
            }
            if (msg.active_session !== undefined) {
              setActiveSession(msg.active_session ?? null);
            }
          }
        } catch {
          // ignore malformed frames
        }
      };

      ws.onerror = () => {
        if (wsRef.current !== ws) return;
        const msg =
          "Could not connect to local Python telemetry WebSocket (ws://127.0.0.1:8000/ws/local/telemetry). Ensure local FastAPI is running with LOCAL_CAMERA_API_ENABLED=true.";
        setConnectionStatus("error");
        setBackendProcessingStatus("error");
        setError(msg);
        onErrorRef.current?.(new Error(msg));
      };

      ws.onclose = () => {
        if (wsRef.current === ws) {
          wsRef.current = null;
          setConnectionStatus("disconnected");
        }
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : "WebSocket connection failed";
      setConnectionStatus("error");
      setBackendProcessingStatus("error");
      setError(msg);
      onErrorRef.current?.(new Error(msg));
    }
  }, [baseWs, closeWebSocket, reconcileProfile]);

  const start = useCallback(
    async (opts?: {
      source?: number | string;
      fps?: number;
      isMirrored?: boolean;
      enableAudio?: boolean;
      startCalibrating?: boolean;
    }): Promise<boolean> => {
      setIsStarting(true);
      setError(null);
      try {
        // Non-blocking profile reconciliation before starting (works offline cleanly)
        await reconcileProfile().catch(() => {});

        const res = await fetch(`${baseHttp}/api/v1/local/start`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            driver_id: driverId,
            source: opts?.source ?? 0,
            width: 640,
            height: 480,
            fps: opts?.fps ?? 20,
            is_mirrored: opts?.isMirrored ?? true,
            enable_audio: opts?.enableAudio ?? false,
            start_calibrating: opts?.startCalibrating ?? false,
          }),
        });

        if (!res.ok) {
          const body = await res.json().catch(() => ({}));
          const detail =
            body?.detail ||
            (res.status === 403
              ? "Local Python camera API returned 403 Forbidden. Start local backend with LOCAL_CAMERA_API_ENABLED=true."
              : `Local Python start failed (HTTP ${res.status}).`);
          throw new Error(detail);
        }

        const data = await res.json();
        setIsRunning(Boolean(data?.running));
        setIsCalibrated(Boolean(data?.is_calibrated));
        setActiveSession(data?.active_session ?? null);

        const sessionNonce = data?.active_session?.session_id ?? String(Date.now());
        setMjpegPreviewUrl(
          `${baseHttp}/api/v1/local/preview.mjpg?session=${encodeURIComponent(sessionNonce)}`
        );
        connectWebSocket();
        return true;
      } catch (err) {
        const msg =
          err instanceof Error
            ? err.message
            : "Could not reach local Python backend at http://127.0.0.1:8000.";
        setError(msg);
        setConnectionStatus("error");
        setBackendProcessingStatus("error");
        onErrorRef.current?.(new Error(msg));
        return false;
      } finally {
        setIsStarting(false);
      }
    },
    [baseHttp, connectWebSocket, driverId, reconcileProfile]
  );

  /**
   * Stop local Python monitoring and synchronously confirm `camera_released === true`
   * before resolving (guarantees clean exclusive camera handoff back to Browser AI).
   */
  const stop = useCallback(async (): Promise<boolean> => {
    setMjpegPreviewUrl(null);
    closeWebSocket();
    setTelemetry(null);
    setCalibration(null);
    setBackendProcessingStatus("idle");
    setConnectionStatus("disconnected");

    try {
      const res = await fetch(`${baseHttp}/api/v1/local/stop`, {
        method: "POST",
        headers: { Accept: "application/json" },
      });
      setIsRunning(false);
      setActiveSession(null);
      if (!res.ok) {
        return true;
      }
      const data = await res.json().catch(() => ({}));
      void refreshSessions();
      return Boolean(data?.camera_released ?? true);
    } catch {
      setIsRunning(false);
      setActiveSession(null);
      return true;
    }
  }, [baseHttp, closeWebSocket, refreshSessions]);

  const startCalibration = useCallback(
    async (fourZoneMode: boolean = true, stepDurationSec: number = 4.0) => {
      const res = await fetch(`${baseHttp}/api/v1/local/calibration/start`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          four_zone_mode: fourZoneMode,
          step_duration_sec: stepDurationSec,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Local calibration start failed (HTTP ${res.status})`);
      }
    },
    [baseHttp]
  );

  const skipCalibrationStep = useCallback(async () => {
    await fetch(`${baseHttp}/api/v1/local/calibration/skip`, { method: "POST" });
  }, [baseHttp]);

  const cancelCalibration = useCallback(async () => {
    await fetch(`${baseHttp}/api/v1/local/calibration/cancel`, { method: "POST" });
    setCalibration(null);
  }, [baseHttp]);

  const recalibrateZone = useCallback(
    async (zone: string, stepDurationSec: number = 4.0) => {
      const res = await fetch(`${baseHttp}/api/v1/local/calibration/recalibrate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          zone,
          step_duration_sec: stepDurationSec,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body?.detail || `Local zone recalibration failed (HTTP ${res.status})`);
      }
    },
    [baseHttp]
  );

  // When enabled, quietly hydrate local PYTHON profile & sessions without requiring camera start
  useEffect(() => {
    if (!enabled) return;
    void reconcileProfile();
    void refreshSessions();
  }, [enabled, reconcileProfile, refreshSessions]);

  // Clean up WebSocket on unmount
  useEffect(() => {
    return () => {
      closeWebSocket();
    };
  }, [closeWebSocket]);

  return {
    isRunning,
    isStarting,
    connectionStatus,
    backendProcessingStatus,
    telemetry,
    calibration,
    isCalibrated,
    activeProfile,
    profileSyncStatus,
    activeSession,
    mjpegPreviewUrl,
    devices,
    localSessions,
    error,
    start,
    stop,
    startCalibration,
    skipCalibrationStep,
    cancelCalibration,
    recalibrateZone,
    refreshDevices,
    refreshSessions,
    reconcileProfile,
  };
}
