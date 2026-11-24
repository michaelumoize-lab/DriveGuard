"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useLivekitTelemetryStream } from "@/hooks/use-livekit-telemetry-stream";
import { useBrowserDetection } from "@/hooks/use-browser-detection";
import { convertBrowserTelemetryToFrameTelemetry, DrowsinessState } from "@/lib/ai";
import { CapturedFrame } from "@/types/camera";
import {
  AlertState,
  ExperimentMode,
  SafetyEvent,
  SessionEndPayload,
  resolveEffectiveDisplayState,
} from "@/types/telemetry";
import { buildDriverProfileDtoPayload } from "@/lib/api/profile-mapper";
import { syncDriverProfileToFastApi } from "@/lib/api/driveguard-api";
import { LOCAL_FASTAPI_URL } from "@/lib/fastapi-url";
import { CameraView } from "@/components/camera/camera-view";
import { CameraControls } from "@/components/camera/camera-controls";
import { CameraSelector } from "@/components/camera/camera-selector";
import { CameraStatus } from "@/components/camera/camera-status";
import { CameraErrorDisplay } from "@/components/camera/camera-error";
import { CameraSnapshotPreview } from "@/components/camera/camera-snapshot-preview";
import { DriverCockpitHUD } from "./driver-cockpit-hud";
import { LiveTelemetryHUD } from "./live-telemetry-hud";
import { ExperimentSelector } from "./experiment-selector";
import { ReadinessChecklist } from "./readiness-checklist";
import { LiveTelemetryChart, TelemetryHistoryPoint } from "./live-telemetry-chart";
import { ExplainableRiskPanel } from "./explainable-risk-panel";
import { useDriveGuardAudio } from "@/hooks/use-driveguard-audio";
import { runPreFlightHandshake } from "@/lib/monitor/preflight-handshake";
import { FaultInjectionToolbar } from "@/components/engineering/fault-injection-toolbar";
import { EulerHeadGyro } from "@/components/engineering/euler-head-gyro";
import { KinematicsOscilloscope } from "@/components/engineering/kinematics-oscilloscope";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsList, TabsTrigger, TabsContent } from "@/components/ui/tabs";
import { useSidebar } from "@/components/ui/sidebar";
import {
  AlertTriangle,
  Camera,
  Car,
  Compass,
  Cpu,
  EyeOff,
  Flame,
  Minimize2,
  Monitor,
  Radio,
  RefreshCw,
  SlidersHorizontal,
  ShieldCheck,
  Smartphone,
  Sparkles,
  Volume2,
  VolumeX,
} from "lucide-react";

export interface InitialDriverProfileData {
  isCalibrated: boolean;
  calibrationMode: string;
  centerEarMedian: number;
  centerEarMadScaled: number;
  blinkDurationMedianMs: number;
  blinkDurationMadMs: number;
  blinkRcrMedian: number;
  blinkRcrMad: number;
  neutralPitchDeg: number;
  neutralYawDeg: number;
  calibrationReliability: number;
  zoneBaselines: Record<string, unknown> | null;
}

export interface InitialUserPreferencesData {
  audioAlertsEnabled: boolean;
  sensitivityLevel: string;
  mirrorSuppression: boolean;
  wearsGlasses: boolean;
}

export interface UnifiedLiveMonitorProps {
  driverId?: string;
  initialDriverProfile?: InitialDriverProfileData | null;
  initialPreferences?: InitialUserPreferencesData;
}

export function UnifiedLiveMonitor({
  driverId = "driver_default",
  initialDriverProfile = null,
  initialPreferences,
}: UnifiedLiveMonitorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { setOpen: setSidebarOpen } = useSidebar();

  // Auto-collapse the left navigation sidebar when entering the Live Monitoring Console
  // so the two-column cockpit has maximum horizontal breathing room (+256px).
  useEffect(() => {
    setSidebarOpen(false);
  }, [setSidebarOpen]);

  // Layout & UI States
  const [runtimeMode, setRuntimeMode] = useState<"browser" | "remote">("browser");
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("landscape");
  const [consoleMode, setConsoleMode] = useState<"cockpit" | "diagnostics">("diagnostics");
  const [framingMode, setFramingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [showInlineCalibration, setShowInlineCalibration] = useState(false);
  const [isCalibrated, setIsCalibrated] = useState<boolean>(
    Boolean(initialDriverProfile?.isCalibrated)
  );
  const [centerEarBaseline, setCenterEarBaseline] = useState<number>(
    initialDriverProfile?.centerEarMedian ?? 0.285
  );

  // Restore saved view mode and runtime mode preferences
  useEffect(() => {
    try {
      const savedMode = localStorage.getItem("driveguard_runtime_mode");
      if (savedMode === "browser" || savedMode === "remote") {
        setRuntimeMode(savedMode);
      }
      const saved = localStorage.getItem("driveguard_console_mode");
      if (saved === "cockpit" || saved === "diagnostics") {
        setConsoleMode(saved);
      }
    } catch {
      // Safe fallback if localStorage is disabled
    }
  }, []);

  const handleConsoleModeChange = useCallback((mode: "cockpit" | "diagnostics") => {
    setConsoleMode(mode);
    try {
      localStorage.setItem("driveguard_console_mode", mode);
    } catch {
      // Ignore
    }
  }, []);

  // Rolling 60s chart history & screen-reader live announcement
  const [chartHistory, setChartHistory] = useState<TelemetryHistoryPoint[]>([]);
  const lastChartSampleMsRef = useRef<number>(0);
  const [srAnnouncement, setSrAnnouncement] = useState<string>("");
  const lastAnnouncedStateRef = useRef<AlertState>("NORMAL");

  // Web Audio Alert & Sound Synthesizer Subsystem
  const {
    isAudioUnlocked,
    isMuted: audioMuted,
    volume: audioVolume,
    audioStatus,
    toggleMute,
    setVolume: setAudioVolume,
    unlockAudio,
    testAlert,
    engine: audioEngine,
  } = useDriveGuardAudio({
    masterVolume: 0.8,
    initialMuted: initialPreferences?.audioAlertsEnabled === false,
  });

  // Snapshot modal state
  const [capturedSnapshot, setCapturedSnapshot] = useState<CapturedFrame | null>(null);
  const [isSnapshotOpen, setIsSnapshotOpen] = useState(false);
  const [isCapturing, setIsCapturing] = useState(false);

  // Stable session identifier (guaranteed not to mutate on re-renders)
  const stableSessionId = useMemo(
    () => `live_${Math.random().toString(36).slice(2, 9)}`,
    []
  );

  // Comprehensive Session & Database Persistence Refs
  const activeSessionCodeRef = useRef<string | null>(null);
  const sessionStartWallMsRef = useRef<number>(0);
  const lastAlertTimeRef = useRef<number>(0);
  const handledEventIdsRef = useRef<Set<string>>(new Set());
  const calibrationToastShownRef = useRef<boolean>(false);
  const sessionStatsRef = useRef({
    totalFrames: 0,
    normalFramesCount: 0,
    fpsSum: 0,
    latencySum: 0,
    perclosSum: 0,
    maxPerclos: 0,
    totalBlinks: 0,
    blinkRateSum: 0,
    warningCount: 0,
    criticalCount: 0,
    mirrorSuppressions: 0,
    lastBlinkDurationSeen: null as number | null,
  });

  const resetSessionStats = useCallback(() => {
    sessionStartWallMsRef.current = Date.now();
    sessionStatsRef.current = {
      totalFrames: 0,
      normalFramesCount: 0,
      fpsSum: 0,
      latencySum: 0,
      perclosSum: 0,
      maxPerclos: 0,
      totalBlinks: 0,
      blinkRateSum: 0,
      warningCount: 0,
      criticalCount: 0,
      mirrorSuppressions: 0,
      lastBlinkDurationSeen: null,
    };
  }, []);

  const buildSessionEndPayload = useCallback(
    (sessionCode: string, fallbackDurationSec: number): SessionEndPayload => {
      const stats = sessionStatsRef.current;
      const elapsedSec =
        sessionStartWallMsRef.current > 0
          ? Math.max(1, Math.round((Date.now() - sessionStartWallMsRef.current) / 1000))
          : Math.max(1, fallbackDurationSec);
      const frames = Math.max(1, stats.totalFrames);

      return {
        sessionCode,
        durationSeconds: Math.max(fallbackDurationSec, elapsedSec),
        averageFps: Number((stats.fpsSum / frames).toFixed(1)),
        averageLatencyMs: Number((stats.latencySum / frames).toFixed(1)),
        averagePerclosPct: Number((stats.perclosSum / frames).toFixed(2)),
        maxPerclosPct: Number(stats.maxPerclos.toFixed(2)),
        totalBlinks: stats.totalBlinks,
        averageBlinkRate: Number((stats.blinkRateSum / frames).toFixed(1)),
        normalFramesCount: stats.normalFramesCount,
        warningAlertsCount: stats.warningCount,
        criticalAlertsCount: stats.criticalCount,
        mirrorSuppressions: stats.mirrorSuppressions,
        // Legacy aliases for backward compatibility
        durationSec: Math.max(fallbackDurationSec, elapsedSec),
        totalFrames: stats.totalFrames,
        maxPerclos: Number(stats.maxPerclos.toFixed(2)),
        criticalAlerts: stats.criticalCount,
      };
    },
    []
  );

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
    opticalQuality,
    hardwareDiagnostics,
    startCamera,
    stopCamera,
    pauseCamera,
    resumeCamera,
    switchFacingMode,
    selectDevice,
    captureSnapshot,
    refreshDevices,
  } = useCamera({
    idealWidth: 640,
    idealHeight: 480,
    idealFps: 25,
    defaultFacingMode: "user",
    mirrored: true,
  });

  // Reliable Safety Event Callback (Deterministic State Transitions & Fast-Path Audio Alerting)
  const handleSafetyEvent = useCallback(
    (event: SafetyEvent) => {
      if (handledEventIdsRef.current.has(event.event_id)) {
        return;
      }
      handledEventIdsRef.current.add(event.event_id);
      if (handledEventIdsRef.current.size > 200) {
        const [first] = handledEventIdsRef.current;
        handledEventIdsRef.current.delete(first);
      }

      const eventTsMs =
        typeof event.timestamp === "number" && !isNaN(event.timestamp)
          ? Math.round(event.timestamp * 1000)
          : Date.now();

      // Fast-path Web Audio alert dispatch
      audioEngine?.setRiskState(event.new_state, event.event_id, eventTsMs);

      // Update screen-reader assertive live region on significant state change
      if (event.new_state !== lastAnnouncedStateRef.current) {
        lastAnnouncedStateRef.current = event.new_state;
        if (event.new_state === "CRITICAL") {
          setSrAnnouncement("Critical drowsiness alert! Prolonged eye closure or high fatigue detected. Pull over safely.");
        } else if (event.new_state === "WARNING") {
          setSrAnnouncement("Drowsiness warning: elevated eyelid closure detected.");
        } else if (event.new_state === "MONITORING_UNAVAILABLE") {
          setSrAnnouncement("Monitoring unavailable: driver face not visible in camera frame.");
        } else if (event.new_state === "NORMAL") {
          setSrAnnouncement("Driver monitoring state returned to normal.");
        }
      }

      // Persist both WARNING and CRITICAL safety transitions to Neon PostgreSQL
      if (event.new_state === "CRITICAL" || event.new_state === "WARNING") {
        lastAlertTimeRef.current = Date.now();
        if (event.new_state === "CRITICAL") {
          sessionStatsRef.current.criticalCount++;
        } else {
          sessionStatsRef.current.warningCount++;
        }

        const sessionCode = activeSessionCodeRef.current;
        if (!sessionCode) return;

        const earNum =
          typeof event.metrics?.ear === "number" && !isNaN(event.metrics.ear)
            ? event.metrics.ear
            : 0.18;
        const perclosNum =
          typeof event.metrics?.perclos === "number" && !isNaN(event.metrics.perclos)
            ? event.metrics.perclos
            : event.new_state === "CRITICAL"
              ? 24.0
              : 14.0;

        fetch("/api/alerts/log", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionCode,
            severity: event.new_state,
            riskScore:
              typeof event.metrics?.risk_score === "number"
                ? event.metrics.risk_score
                : event.new_state === "CRITICAL"
                  ? 0.85
                  : 0.58,
            earValue: earNum,
            perclosPct: perclosNum,
            poseZone: String(event.metrics?.pose_zone ?? "CENTER"),
            reasons: Array.isArray(event.metrics?.reasons)
              ? event.metrics.reasons
              : [ `${event.new_state}_TRANSITION` ],
            cascadeDecision: "HYBRID_CASCADE",
            timestampMs: eventTsMs,
          }),
        }).catch(() => {});
      }
    },
    [audioEngine]
  );

  // 2. Adaptive AI Telemetry LiveKit WebRTC Streaming Hook
  const {
    isConnected: isWsConnected,
    connectionStatus: wsStatus,
    backendProcessingStatus,
    workerError,
    isStale,
    telemetry,
    calibration,
    latencyDiagnostics,
    roundtripLatencyMs,
    experimentMode,
    backendType,
    switchBackend,
    startStreamingLoop,
    stopStreamingLoop,
    switchExperiment,
    startCalibration,
    recalibrateZone,
    cancelCalibration,
    skipCalibrationStep,
    reconnect: reconnectWs,
  } = useLivekitTelemetryStream({
    sessionId: stableSessionId,
    driverId,
    onSafetyEvent: handleSafetyEvent,
    onCalibration: (cal) => {
      if (!cal.is_completed) {
        calibrationToastShownRef.current = false;
        return;
      }
      if (calibrationToastShownRef.current) {
        return;
      }
      calibrationToastShownRef.current = true;

      if (cal.is_completed && !cal.rejection_reason && cal.profile) {
        const p = cal.profile;
        setIsCalibrated(true);
        if (typeof p.center_ear_median === "number") {
          setCenterEarBaseline(p.center_ear_median);
        }
        fetch("/api/calibration/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            calibrationMode: p.calibration_mode || "FOUR_ZONE",
            centerEarMedian: p.center_ear_median,
            centerEarMadScaled:
              p.center_ear_mad_scaled ?? p.zone_baselines?.CENTER?.ear_mad_scaled ?? 0.018,
            blinkDurationMedianMs: p.blink_duration_median_ms,
            blinkDurationMadMs: p.blink_duration_mad_ms,
            blinkRcrMedian: p.blink_rcr_median,
            blinkRcrMad: p.blink_rcr_mad,
            blinkFreqMedianPerMin: p.blink_freq_median_per_min ?? 16.0,
            blinkFreqMadPerMin: p.blink_freq_mad_per_min ?? 4.5,
            baselinePerclosPct: p.baseline_perclos_pct ?? 3.5,
            blinkAmplitudeMedian: p.blink_amplitude_median ?? 0.14,
            temporalBlinksSampled: p.temporal_blinks_sampled ?? 0,
            temporalWindowSec: p.temporal_window_sec ?? 120.0,
            sensitivityLevel: p.sensitivity_level ?? "STANDARD",
            neutralPitchDeg: p.neutral_pitch_deg,
            neutralYawDeg: p.neutral_yaw_deg,
            neutralRotMat: p.neutral_rot_mat ?? null,
            calibrationIsMirrored: p.calibration_is_mirrored ?? null,
            calibrationReliability: p.calibration_reliability,
            zoneBaselines: p.zone_baselines ?? {},
          }),
        })
          .then((res) => {
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            toast.success("4-Zone Personal Calibration saved to PostgreSQL & AI Worker!", {
              id: "calibration-completed-toast",
            });
          })
          .catch((err) => {
            toast.error(`Failed to persist calibrated profile: ${err?.message || "Save error"}`);
          });
      } else if (cal.is_completed && cal.rejection_reason) {
        toast.error(`Calibration rejected: ${cal.rejection_reason}`, {
          id: "calibration-rejected-toast",
        });
      }
    },
  });

  // Handle safety alerts triggered by Browser Detection Engine
  const handleBrowserSafetyAlert = useCallback(
    (decision: any, t: any) => {
      const isCritical = decision.state === DrowsinessState.CRITICAL_WARNING;
      const isWarning = decision.state === DrowsinessState.DROWSINESS_WARNING;
      if (!isCritical && !isWarning) return;

      const safetyEvent: SafetyEvent = {
        event_id: `evt_browser_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`,
        event_type: "DROWSINESS_ALERT",
        severity: isCritical ? "CRITICAL" : "WARNING",
        new_state: isCritical ? "CRITICAL" : "WARNING",
        prev_state: "NORMAL",
        timestamp: (t.timestampMs || Date.now()) / 1000,
        metrics: {
          ear: t.ear,
          perclos: t.perclosPct,
          risk_score: isCritical ? 0.95 : 0.65,
          reasons: t.reasons,
          pose_zone: t.poseZone,
        },
      };
      handleSafetyEvent(safetyEvent);
    },
    [handleSafetyEvent]
  );

  // Browser-Side AI Detection Hook (Local MediaPipe Face Landmarker & TypeScript detection)
  const {
    engineStatus: browserEngineStatus,
    telemetry: browserTelemetry,
    calibration: browserCalibration,
    sessionCode: browserSessionCode,
    setSessionCode: setBrowserSessionCode,
    syncStatus: browserSyncStatus,
    startMonitoring: startBrowserMonitoring,
    pauseMonitoring: pauseBrowserMonitoring,
    resumeMonitoring: resumeBrowserMonitoring,
    stopMonitoring: stopBrowserMonitoring,
    startCalibration: startBrowserCalibration,
    cancelCalibration: cancelBrowserCalibration,
    queueSessionEnd: queueBrowserSessionEnd,
  } = useBrowserDetection({
    driverId,
    initialProfile: (initialDriverProfile as unknown) as any,
    isMirrored,
    targetInferenceFps: 12,
    autoStartSession: false,
    onSafetyAlert: handleBrowserSafetyAlert,
  });

  // Convert browser telemetry to FrameTelemetry for existing HUD components
  const convertedBrowserTelemetry = useMemo(() => {
    if (!browserTelemetry) return null;
    return convertBrowserTelemetryToFrameTelemetry(browserTelemetry);
  }, [browserTelemetry]);

  // Unified telemetry & processing state
  const effectiveTelemetry = runtimeMode === "browser" ? convertedBrowserTelemetry : telemetry;
  const effectiveProcessingStatus = runtimeMode === "browser"
    ? (browserEngineStatus === "running" ? "processing" : browserEngineStatus === "initializing" ? "connecting" : "idle")
    : backendProcessingStatus;

  // Sync browser session code
  useEffect(() => {
    if (runtimeMode === "browser" && browserSessionCode) {
      activeSessionCodeRef.current = browserSessionCode;
    }
  }, [runtimeMode, browserSessionCode]);

  // Sync saved PostgreSQL DriverProfile to FastAPI via canonical DTO mapper (Remote AI mode)
  const syncProfileToWorker = useCallback(async () => {
    if (runtimeMode !== "remote") return;
    if (!initialDriverProfile?.isCalibrated || !driverId) return;

    const fastApiProfileDto = buildDriverProfileDtoPayload(
      {
        ...initialDriverProfile,
        isCalibrated: true,
      },
      driverId
    );

    try {
      await syncDriverProfileToFastApi(fastApiProfileDto, {
        mode: backendType,
        localBaseUrl: LOCAL_FASTAPI_URL,
      });
    } catch {
      // Non-fatal if worker is still spinning up; Next.js /api/v1 proxy also hydrates on token minting
    }
  }, [backendType, driverId, initialDriverProfile, runtimeMode]);

  // Reactive Stream Trigger: automatically start/stop inference loop when camera becomes active
  useEffect(() => {
    if (cameraState === "active" && videoRef.current) {
      if (runtimeMode === "browser") {
        startBrowserMonitoring(videoRef.current);
      } else {
        syncProfileToWorker().finally(() => {
          if (videoRef.current) {
            startStreamingLoop(videoRef.current, isMirrored);
          }
        });
      }
    } else if (cameraState === "stopped" || cameraState === "paused") {
      if (runtimeMode === "browser") {
        stopBrowserMonitoring();
      } else {
        stopStreamingLoop();
      }
    }
  }, [cameraState, runtimeMode, isMirrored, startBrowserMonitoring, stopBrowserMonitoring, startStreamingLoop, stopStreamingLoop, syncProfileToWorker, videoRef]);

  // Handle telemetry updates: session accumulators, rolling 60s chart buffer, and audio sync
  useEffect(() => {
    if (!effectiveTelemetry) return;

    const stats = sessionStatsRef.current;
    stats.totalFrames++;
    if (effectiveTelemetry.state === "NORMAL") {
      stats.normalFramesCount++;
    }
    stats.fpsSum += effectiveTelemetry.fps || 0;
    const inferLat =
      effectiveTelemetry.timing?.infer_duration_ms ?? effectiveTelemetry.inference_latency_ms ?? 0;
    stats.latencySum += inferLat;

    const perclos = effectiveTelemetry.eye_metrics?.perclos_pct ?? 0;
    stats.perclosSum += perclos;
    if (perclos > stats.maxPerclos) {
      stats.maxPerclos = perclos;
    }

    const blinkRate = effectiveTelemetry.eye_metrics?.blink_rate_per_min ?? 0;
    stats.blinkRateSum += blinkRate;

    const blinkDur = effectiveTelemetry.eye_metrics?.blink_duration_ms ?? null;
    if (blinkDur !== null && blinkDur !== stats.lastBlinkDurationSeen) {
      stats.lastBlinkDurationSeen = blinkDur;
      stats.totalBlinks++;
    }

    if (effectiveTelemetry.comparison?.is_mirror_check_suppressed) {
      stats.mirrorSuppressions++;
    }

    // Append to 60s rolling chart buffer at ~2 Hz (max 120 points = 60 seconds)
    const nowMs = Date.now();
    if (nowMs - lastChartSampleMsRef.current >= 500) {
      lastChartSampleMsRef.current = nowMs;
      const point: TelemetryHistoryPoint = {
        timeLabel: new Date(nowMs).toLocaleTimeString([], {
          minute: "2-digit",
          second: "2-digit",
        }),
        ear: Number((effectiveTelemetry.eye_metrics?.ear_raw ?? 0).toFixed(3)),
        threshold: Number((effectiveTelemetry.eye_metrics?.ear_closure_threshold ?? 0.21).toFixed(3)),
        perclos: Number(perclos.toFixed(1)),
        riskPct: Math.round((effectiveTelemetry.risk_score ?? 0) * 100),
      };
      setChartHistory((prev) => {
        const next = [...prev, point];
        return next.length > 120 ? next.slice(next.length - 120) : next;
      });
    }

    // Synchronize audio subsystem with continuous telemetry state
    if (effectiveTelemetry.state) {
      audioEngine?.setRiskState(effectiveTelemetry.state, undefined, effectiveTelemetry.timestamp_ms);
    }

    // Fallback sustained CRITICAL logging if state remains CRITICAL for > 8s without a transition packet
    if (effectiveTelemetry.state === "CRITICAL" && activeSessionCodeRef.current) {
      if (nowMs - lastAlertTimeRef.current >= 8000) {
        lastAlertTimeRef.current = nowMs;
        stats.criticalCount++;
        const earVal =
          effectiveTelemetry.eye_metrics?.ear_normalized ?? (effectiveTelemetry.eye_metrics?.ear_raw ?? 0.15);
        fetch("/api/alerts/log", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sessionCode: activeSessionCodeRef.current,
            severity: "CRITICAL",
            riskScore: effectiveTelemetry.risk_score ?? 0.85,
            earValue: earVal,
            perclosPct: perclos,
            poseZone: effectiveTelemetry.pose_zone ?? "CENTER",
            reasons: effectiveTelemetry.reason_codes ?? ["CRITICAL_STATE"],
            cascadeDecision: effectiveTelemetry.cascade?.arbitration_winner || (runtimeMode === "browser" ? "BROWSER_AI" : "GEOMETRY"),
            timestampMs: nowMs,
          }),
        }).catch(() => {});
      }
    }
  }, [effectiveTelemetry, audioEngine, runtimeMode]);

  // Flush active session on pagehide / tab close / component unmount so no session is left with 0s duration
  useEffect(() => {
    const flushActiveSession = () => {
      const code = activeSessionCodeRef.current;
      if (!code) return;
      const payload = buildSessionEndPayload(code, 0);
      activeSessionCodeRef.current = null;
      try {
        fetch("/api/sessions/end", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
          keepalive: true,
        }).catch(() => {});
      } catch (_) {}
    };

    window.addEventListener("pagehide", flushActiveSession);
    return () => {
      window.removeEventListener("pagehide", flushActiveSession);
      flushActiveSession();
    };
  }, [buildSessionEndPayload]);

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
          setFramingMode("cover");
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
    } catch {
      toast.info("Fullscreen mode is not supported by your browser window policy.");
    }
  }, []);

  // Switch between Browser AI and Remote AI runtime modes
  const handleSwitchRuntimeMode = useCallback(
    async (targetMode: "browser" | "remote") => {
      if (targetMode === runtimeMode) return;
      if (cameraState === "active" && videoRef.current) {
        if (runtimeMode === "browser") {
          stopBrowserMonitoring();
        } else {
          stopStreamingLoop();
        }
      }
      setRuntimeMode(targetMode);
      try {
        localStorage.setItem("driveguard_runtime_mode", targetMode);
      } catch {}
      toast.info(
        targetMode === "browser"
          ? "Switched to Browser AI (Edge MediaPipe — zero server video streaming)."
          : "Switched to Remote AI (LiveKit WebRTC + Python DrowsinessEngine)."
      );
    },
    [runtimeMode, cameraState, stopBrowserMonitoring, stopStreamingLoop, videoRef]
  );

  const [isPreflightRunning, setIsPreflightRunning] = useState(false);
  const [activeFault, setActiveFault] = useState<string | null>(null);

  // Start live monitoring (Automotive Pre-Flight Handshake + Database Driving Session)
  const handleStartMonitoring = useCallback(async () => {
    unlockAudio();
    resetSessionStats();
    setChartHistory([]);
    setIsPreflightRunning(true);

    // 1. Ensure camera stream is initiated
    if (cameraState !== "active") {
      await startCamera();
    }

    // 2. Perform Pre-Flight Readiness Handshake
    const isAiReady =
      runtimeMode === "browser"
        ? browserEngineStatus === "running" || browserEngineStatus === "idle" || browserEngineStatus === "initializing"
        : isWsConnected;

    const handshake = await runPreFlightHandshake({
      videoElement: videoRef.current,
      opticalQuality,
      isMediaPipeReady: isAiReady,
      isCascadeReady: true,
      isAudioUnlocked: isAudioUnlocked,
    });

    setIsPreflightRunning(false);

    if (!handshake.canStart) {
      const reasons = handshake.failedSubsystems.join(", ");
      toast.error(`Pre-Flight Safety Interlock: ${reasons} failed readiness check. Please resolve before starting.`);
      return;
    }

    // 3. Authenticate & initialize backend driving session in database
    try {
      const startRes = await fetch("/api/sessions/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mode: experimentMode,
          device: isMobileDevice ? "MOBILE_PORTRAIT" : "DESKTOP_HUD",
        }),
      });

      if (!startRes.ok) {
        throw new Error(`Session start returned HTTP ${startRes.status}`);
      }

      const data = await startRes.json();
      if (data?.sessionCode) {
        activeSessionCodeRef.current = data.sessionCode;
        if (runtimeMode === "browser") {
          setBrowserSessionCode(data.sessionCode);
        }
        toast.success(`Monitoring Session Initialized: #${data.sessionCode}`);
      }
    } catch (err) {
      console.error("Failed to start session recording:", err);
      toast.error("Could not register session in backend database. Check connection.");
      return;
    }
  }, [
    startCamera,
    cameraState,
    videoRef,
    opticalQuality,
    runtimeMode,
    browserEngineStatus,
    isWsConnected,
    isAudioUnlocked,
    experimentMode,
    isMobileDevice,
    unlockAudio,
    resetSessionStats,
    setBrowserSessionCode,
  ]);

  // Stop live monitoring, persist full session summary & completely clear the cockpit
  const handleStopMonitoring = useCallback(async () => {
    audioEngine?.stopCurrentAlert();
    audioEngine?.setRiskState("NORMAL");
    if (runtimeMode === "browser") {
      stopBrowserMonitoring();
    } else {
      stopStreamingLoop();
    }
    stopCamera();
    setChartHistory([]);
    setCapturedSnapshot(null);
    setShowInlineCalibration(false);
    setSrAnnouncement("");
    lastAnnouncedStateRef.current = "NORMAL";

    const code = activeSessionCodeRef.current;
    if (code) {
      activeSessionCodeRef.current = null;
      const payload = buildSessionEndPayload(code, streamDurationSec);
      resetSessionStats();
      if (runtimeMode === "browser") {
        queueBrowserSessionEnd(streamDurationSec);
      }
      try {
        const res = await fetch("/api/sessions/end", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (res.ok) {
          toast.success(
            `Session ${code} ended & saved (${payload.durationSeconds}s, Avg PERCLOS ${payload.averagePerclosPct.toFixed(1)}%)`
          );
        }
      } catch (_) {}
    } else {
      resetSessionStats();
    }
  }, [audioEngine, runtimeMode, stopBrowserMonitoring, stopStreamingLoop, stopCamera, buildSessionEndPayload, streamDurationSec, resetSessionStats, queueBrowserSessionEnd]);

  // Switch between Cloud and Localhost engines: cleanly ends any active session, stops camera, and resets cockpit
  const handleSwitchBackend = useCallback(
    async (targetBackend: "cloud" | "local") => {
      if (targetBackend === backendType) return;
      const wasActive =
        cameraState === "active" ||
        cameraState === "paused" ||
        Boolean(activeSessionCodeRef.current);

      if (wasActive) {
        await handleStopMonitoring();
      } else {
        stopStreamingLoop();
        setChartHistory([]);
      }

      switchBackend(targetBackend);
      toast.info(
        wasActive
          ? `Session ended & cockpit cleared. Switched AI Engine to ${
              targetBackend === "cloud" ? "LiveKit Cloud" : "Localhost (127.0.0.1:8000)"
            }.`
          : `Switched AI Engine to ${
              targetBackend === "cloud" ? "LiveKit Cloud" : "Localhost (127.0.0.1:8000)"
            }.`
      );
    },
    [backendType, cameraState, handleStopMonitoring, stopStreamingLoop, switchBackend]
  );

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
    } catch {
      toast.error("Error occurred while capturing frame.");
    } finally {
      setIsCapturing(false);
    }
  }, [cameraState, captureSnapshot, viewOrientation]);

  const isPortrait = viewOrientation === "portrait";

  // Helper for distinct video overlay badge per authoritative safety state (never green before telemetry)
  const renderOverlayStateBadge = () => {
    if (runtimeMode === "browser") {
      if (browserEngineStatus === "idle" || browserEngineStatus === "stopped") {
        return (
          <Badge
            variant="outline"
            className="bg-black/75 backdrop-blur-md border-amber-400/50 text-amber-300 text-[10px] font-mono px-2.5 py-1"
          >
            STANDBY (BROWSER AI)
          </Badge>
        );
      }
      if (browserEngineStatus === "initializing") {
        return (
          <Badge
            variant="outline"
            className="bg-black/75 backdrop-blur-md border-primary/50 text-primary text-[10px] font-mono px-2.5 py-1"
          >
            <span className="size-1.5 rounded-full bg-primary animate-ping mr-1.5" />
            LOADING MEDIAPIPE MODEL…
          </Badge>
        );
      }
      if (effectiveTelemetry) {
        const state = (effectiveTelemetry.alert_state || effectiveTelemetry.state) as string;
        if (state === "CRITICAL") {
          return (
            <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-rose-600 text-white border border-rose-300 animate-pulse gap-1.5">
              <Flame className="size-3.5" />
              <span>CRITICAL FATIGUE</span>
            </Badge>
          );
        }
        if (state === "WARNING") {
          return (
            <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-amber-500 text-slate-950 gap-1.5">
              <AlertTriangle className="size-3.5" />
              <span>WARNING</span>
            </Badge>
          );
        }
        if (state === "ATTENTION") {
          return (
            <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-amber-500/80 text-slate-950 gap-1.5">
              <AlertTriangle className="size-3.5" />
              <span>ATTENTION</span>
            </Badge>
          );
        }
        if (state === "MONITORING_UNAVAILABLE") {
          return (
            <Badge
              variant="outline"
              className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-slate-900/90 text-amber-300 border-amber-400/80 border-dashed gap-1.5"
            >
              <EyeOff className="size-3.5 text-amber-400" />
              <span>MONITORING UNAVAILABLE</span>
            </Badge>
          );
        }
        return (
          <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-emerald-600 text-white gap-1.5">
            <ShieldCheck className="size-3.5" />
            <span>NORMAL</span>
          </Badge>
        );
      }
    }

    const effectiveState = resolveEffectiveDisplayState(telemetry, {
      isConnected: isWsConnected,
      isStale,
      backendStatus: backendProcessingStatus,
    });

    if (effectiveState === "STALE") {
      return (
        <Badge
          variant="outline"
          className="bg-purple-950/90 backdrop-blur-md border-purple-400/70 border-dashed text-purple-200 text-[10px] font-mono px-2.5 py-1"
        >
          TELEMETRY STALE
        </Badge>
      );
    }

    if (effectiveState === "AWAITING_TELEMETRY" || effectiveState === "IDLE") {
      return (
        <Badge
          variant="outline"
          className="bg-black/75 backdrop-blur-md border-amber-400/50 text-amber-300 text-[10px] font-mono px-2.5 py-1"
        >
          <span className="size-1.5 rounded-full bg-amber-400 animate-pulse mr-1.5" />
          AWAITING BACKEND TELEMETRY
        </Badge>
      );
    }

    if (effectiveState === "CRITICAL") {
      return (
        <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-rose-600 text-white border border-rose-300 animate-pulse gap-1.5">
          <Flame className="size-3.5" />
          <span>CRITICAL FATIGUE</span>
        </Badge>
      );
    }

    if (effectiveState === "WARNING") {
      return (
        <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-amber-500 text-slate-950 gap-1.5">
          <AlertTriangle className="size-3.5" />
          <span>WARNING</span>
        </Badge>
      );
    }

    if (effectiveState === "ATTENTION") {
      return (
        <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-amber-500/80 text-slate-950 gap-1.5">
          <AlertTriangle className="size-3.5" />
          <span>ATTENTION</span>
        </Badge>
      );
    }

    if (effectiveState === "MONITORING_UNAVAILABLE") {
      return (
        <Badge
          variant="outline"
          className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-slate-900/90 text-amber-300 border-amber-400/80 border-dashed gap-1.5"
        >
          <EyeOff className="size-3.5 text-amber-400" />
          <span>MONITORING UNAVAILABLE</span>
        </Badge>
      );
    }

    return (
      <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-emerald-600 text-white gap-1.5">
        <ShieldCheck className="size-3.5" />
        <span>NORMAL</span>
      </Badge>
    );
  };

  return (
    <div ref={containerRef} className="w-full space-y-3.5">
      {/* Centralized Screen-Reader Assertive Announcer for Safety State Changes */}
      <div role="alert" aria-live="assertive" aria-atomic="true" className="sr-only">
        {srAnnouncement}
      </div>

      {/* 1. Top Header Card & 5-Pill Readiness Checklist (Hidden in Fullscreen) */}
      {!isFullscreen && (
        <div className="space-y-3">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 sm:p-3.5 rounded-2xl bg-card border border-border/80 shadow-xs">
            <div className="space-y-1 min-w-0">
              <div className="flex items-center gap-2 flex-wrap">
                <div className="size-8 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shrink-0">
                  <Radio className="size-4 animate-pulse" />
                </div>
                <h1 className="text-base sm:text-lg font-bold text-foreground tracking-tight">
                  Real-Time Driver Monitoring Workspace
                </h1>

                {/* Runtime Mode Selector */}
                <div className="flex items-center p-0.5 rounded-xl bg-muted/60 border border-border/80">
                  <Button
                    size="sm"
                    variant={runtimeMode === "browser" ? "default" : "ghost"}
                    onClick={() => handleSwitchRuntimeMode("browser")}
                    className="text-xs h-6 px-2.5 rounded-lg gap-1 cursor-pointer font-semibold"
                    title="Browser AI: MediaPipe Edge execution in TypeScript. Zero video sent to servers."
                  >
                    <Sparkles className="size-3 text-primary" />
                    <span>Browser AI (Edge)</span>
                  </Button>
                  <Button
                    size="sm"
                    variant={runtimeMode === "remote" ? "default" : "ghost"}
                    onClick={() => handleSwitchRuntimeMode("remote")}
                    className="text-xs h-6 px-2.5 rounded-lg gap-1 cursor-pointer"
                    title="Remote AI: LiveKit WebRTC + Python FastAPI (Research / Academic mode)"
                  >
                    <Radio className="size-3" />
                    <span>Remote AI (FastAPI)</span>
                  </Button>
                </div>

                {/* Status Badges */}
                {runtimeMode === "browser" ? (
                  <>
                    <Badge
                      variant="outline"
                      className={`text-[10px] font-mono flex items-center gap-1.5 px-2.5 py-0.5 rounded-full ${
                        browserEngineStatus === "running"
                          ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                          : browserEngineStatus === "initializing"
                            ? "border-amber-500/40 text-amber-500 bg-amber-500/10 animate-pulse"
                            : "border-border text-muted-foreground bg-muted/40"
                      }`}
                    >
                      <span
                        className={`size-1.5 rounded-full ${
                          browserEngineStatus === "running"
                            ? "bg-emerald-500"
                            : browserEngineStatus === "initializing"
                              ? "bg-amber-500 animate-ping"
                              : "bg-muted-foreground/60"
                        }`}
                      />
                      <span>
                        {browserEngineStatus === "running"
                          ? "BROWSER AI ACTIVE (EDGE)"
                          : browserEngineStatus === "initializing"
                            ? "LOADING MEDIAPIPE…"
                            : "STANDBY (BROWSER AI)"}
                      </span>
                    </Badge>
                    <Badge
                      variant="outline"
                      className={`text-[10px] font-mono px-2 py-0.5 rounded-full ${
                        browserSyncStatus === "synced"
                          ? "border-emerald-500/30 text-emerald-600 bg-emerald-500/5"
                          : "border-amber-500/40 text-amber-500 bg-amber-500/10"
                      }`}
                    >
                      {browserSyncStatus === "synced" ? "SYNCED TO NEON" : "OFFLINE BUFFER"}
                    </Badge>
                  </>
                ) : (
                  <Badge
                    variant="outline"
                    className={`text-[10px] font-mono flex items-center gap-1.5 px-2.5 py-0.5 rounded-full ${
                      isWsConnected && !isStale
                        ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                        : isStale
                          ? "border-purple-500/40 text-purple-400 bg-purple-500/10"
                          : wsStatus === "connecting" || wsStatus === "reconnecting"
                            ? "border-amber-500/40 text-amber-500 bg-amber-500/10 animate-pulse"
                            : "border-border text-muted-foreground bg-muted/40"
                    }`}
                  >
                    <span
                      className={`size-1.5 rounded-full ${
                        isWsConnected && !isStale
                          ? "bg-emerald-500"
                          : isStale
                            ? "bg-purple-400"
                            : wsStatus === "connecting" || wsStatus === "reconnecting"
                              ? "bg-amber-500 animate-ping"
                              : "bg-muted-foreground/60"
                      }`}
                    />
                    <span>
                      {isWsConnected && !isStale
                        ? "LIVEKIT WEBRTC ACTIVE"
                        : isStale
                          ? "TELEMETRY STALE"
                          : wsStatus === "connecting" || wsStatus === "reconnecting"
                            ? "CONNECTING TO SFU…"
                            : "PIPELINE STANDBY"}
                    </span>
                  </Badge>
                )}
              </div>
              <p className="text-xs text-muted-foreground pl-10 hidden sm:block">
                {runtimeMode === "browser"
                  ? "Browser Edge MediaPipe & TypeScript: 478-landmark 3D head pose, 4-zone PDBP baselines, blink kinematics, PERCLOS, and MAR yawning. Completely private, offline-capable, zero video sent to servers."
                  : "Authoritative Python DrowsinessEngine over LiveKit WebRTC: 478-landmark 3D head pose, 4-zone PDBP baselines, blink kinematics, PERCLOS, and MAR yawning."}
              </p>
            </div>

            {/* Quick Actions: View Mode Switcher + Inline Calibration + Orientation */}
            <div className="flex items-center gap-2 shrink-0 self-end sm:self-center flex-wrap">
              {/* Console Mode Switcher (Driver vs Diagnostics) */}
              <div
                className="flex items-center p-1 rounded-xl bg-muted/50 border border-border/60"
                role="group"
                aria-label="Monitoring view mode"
              >
                <Button
                  size="sm"
                  variant={consoleMode === "cockpit" ? "default" : "ghost"}
                  onClick={() => handleConsoleModeChange("cockpit")}
                  className="text-xs h-7 px-2.5 rounded-lg gap-1.5 cursor-pointer"
                  title="Driver Safety View (Clean road-safety interface)"
                  aria-label="Switch to Driver Safety View"
                >
                  <Car className="size-3.5" />
                  <span>Driver View</span>
                </Button>

                <Button
                  size="sm"
                  variant={consoleMode === "diagnostics" ? "default" : "ghost"}
                  onClick={() => handleConsoleModeChange("diagnostics")}
                  className="text-xs h-7 px-2.5 rounded-lg gap-1.5 cursor-pointer"
                  title="Engineering Diagnostics View (Full telemetry and latencies)"
                  aria-label="Switch to Engineering Diagnostics View"
                >
                  <SlidersHorizontal className="size-3.5" />
                  <span>Diagnostics [Dev]</span>
                </Button>
              </div>

              <Button
                size="sm"
                variant={showInlineCalibration ? "default" : "outline"}
                onClick={() => setShowInlineCalibration((prev) => !prev)}
                className="text-xs h-8 gap-1.5 rounded-xl"
              >
                <Compass className="size-3.5" />
                <span>{showInlineCalibration ? "Hide Calibration" : "Quick Calibrate"}</span>
              </Button>

              <div className="flex items-center p-1 rounded-xl bg-muted/50 border border-border/60">
                <Button
                  size="sm"
                  variant={isPortrait ? "default" : "ghost"}
                  onClick={() => setViewOrientation("portrait")}
                  className="text-xs h-7 px-2.5 rounded-lg gap-1.5"
                  title="Portrait View (Mobile Driver View)"
                >
                  <Smartphone className="size-3.5" />
                  <span>Portrait</span>
                </Button>

                <Button
                  size="sm"
                  variant={!isPortrait ? "default" : "ghost"}
                  onClick={() => setViewOrientation("landscape")}
                  className="text-xs h-7 px-2.5 rounded-lg gap-1.5"
                  title="Landscape View (Desktop Cockpit View)"
                >
                  <Monitor className="size-3.5" />
                  <span>Landscape</span>
                </Button>
              </div>
            </div>
          </div>

          {/* 5-Pill Pre-Monitoring Readiness Bar */}
          <ReadinessChecklist
            cameraActive={cameraState === "active"}
            cameraResolutionLabel={
              resolution ? `${resolution.width}×${resolution.height}` : "640×480"
            }
            faceDetected={Boolean(effectiveTelemetry?.face_detected)}
            isCalibrated={isCalibrated}
            centerEarBaseline={centerEarBaseline}
            livekitStatus={runtimeMode === "browser" ? "connected" : wsStatus}
            backendProcessingStatus={effectiveProcessingStatus}
            isStale={runtimeMode === "browser" ? false : isStale}
            rttMs={runtimeMode === "browser" ? 0 : roundtripLatencyMs}
            audioStatus={audioStatus}
            audioMuted={audioMuted}
            onUnlockAudio={unlockAudio}
            onCalibrateClick={() => setShowInlineCalibration(true)}
          />

          {/* Honest Calibration Notice (HIGH-4: Authoritative backend pauses ocular closure until calibrated) */}
          {!isCalibrated && (
            <div className="px-3.5 py-2.5 rounded-2xl border border-amber-500/40 bg-amber-500/10 flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-xs">
              <div className="flex items-center gap-2">
                <Compass className="size-4 text-amber-500 shrink-0" />
                <span className="text-foreground">
                  <strong>Calibration required for ocular monitoring:</strong> Run 4-Zone calibration
                  so the engine activates personalised eye-closure and PERCLOS thresholds.
                </span>
              </div>
              <Button
                size="sm"
                variant="outline"
                onClick={() => setShowInlineCalibration(true)}
                className="h-7 text-xs shrink-0 border-amber-500/40"
              >
                Calibrate Now
              </Button>
            </div>
          )}
        </div>
      )}

      {/* Connection Recovery Assistant (Shown when Cloud Worker is cold-starting or connection errors in Remote AI mode) */}
      {!isFullscreen && runtimeMode === "remote" && (wsStatus === "error" || isStale || workerError) && (
        <div
          role="region"
          aria-label="Connection Recovery Assistant"
          className="p-3.5 rounded-2xl border border-amber-500/40 bg-amber-500/10 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
        >
          <div className="space-y-0.5">
            <div className="font-bold text-foreground flex items-center gap-1.5">
              <AlertTriangle className="size-4 text-amber-500 shrink-0" />
              <span>
                {wsStatus === "error" || workerError
                  ? "AI Worker Connection Interrupted"
                  : "Telemetry Stream Stalled (>3.5s)"}
              </span>
            </div>
            <p className="text-muted-foreground">
              {workerError
                ? workerError
                : backendType === "cloud"
                  ? "Render cloud instances may take 30–45 seconds to wake from cold standby. You can retry the connection or switch to Localhost (port 8000)."
                  : "Ensure the local FastAPI worker is running on http://127.0.0.1:8000, or switch to LiveKit Cloud."}
            </p>
          </div>
          <div className="flex items-center gap-2 shrink-0">
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (videoRef.current) reconnectWs(videoRef.current);
              }}
              className="h-8 text-xs gap-1.5"
            >
              <RefreshCw className="size-3.5" /> Retry Worker
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => handleSwitchBackend(backendType === "cloud" ? "local" : "cloud")}
              className="h-8 text-xs"
            >
              Switch to {backendType === "cloud" ? "Localhost" : "Cloud"}
            </Button>
          </div>
        </div>
      )}

      {/* Error Display Card if hardware camera error occurs */}
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
              : "grid-cols-1 lg:grid-cols-12"
        }`}
      >
        {/* Main Video Viewport Column */}
        <div
          className={`flex flex-col gap-3 ${
            isFullscreen
              ? "col-span-1"
              : isPortrait
                ? "md:col-span-7 lg:col-span-6 mx-auto w-full"
                : "lg:col-span-7"
          }`}
        >
          {/* Compact Inline 4-Zone Calibration Bar (Docked directly above video without pushing camera below fold) */}
          {!isFullscreen && showInlineCalibration && (
            <div className="p-3 rounded-2xl border border-primary/40 bg-primary/5 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Compass className="size-4 text-primary" />
                  <span className="text-xs font-bold text-foreground">
                    Inline 4-Zone PDBP Calibration
                  </span>
                  {calibration && !calibration.is_completed && (
                    <Badge className="text-[10px] font-mono bg-primary text-primary-foreground">
                      Zone {calibration.phase_index}/{calibration.total_phases}: {calibration.current_zone} ({Math.round(calibration.zone_progress_pct)}%)
                    </Badge>
                  )}
                </div>
                <div className="flex items-center gap-1.5">
                  <Button
                    size="sm"
                    onClick={() => {
                      if (cameraState !== "active") {
                        handleStartMonitoring().then(() => startCalibration(true, 4.0));
                      } else {
                        startCalibration(true, 4.0);
                      }
                    }}
                    className="h-7 px-2.5 text-xs"
                  >
                    Start All 4 Zones
                  </Button>
                  {calibration && !calibration.is_completed && (
                    <>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={skipCalibrationStep}
                        className="h-7 px-2 text-xs"
                      >
                        Skip Zone
                      </Button>
                      <Button
                        size="sm"
                        variant="ghost"
                        onClick={cancelCalibration}
                        className="h-7 px-2 text-xs text-rose-500"
                      >
                        Cancel
                      </Button>
                    </>
                  )}
                  <Button asChild size="sm" variant="ghost" className="h-7 px-2 text-xs">
                    <Link href="/dashboard/calibration">Full Wizard</Link>
                  </Button>
                </div>
              </div>

              {calibration && !calibration.is_completed && (
                <div className="text-xs text-foreground font-medium bg-background/70 px-3 py-1.5 rounded-lg border border-border/60 flex items-center justify-between gap-2">
                  <span>{calibration.instruction}</span>
                  <span className="font-mono text-[11px] text-muted-foreground shrink-0">
                    Samples: {calibration.stable_samples_count}/{calibration.target_samples_count}
                  </span>
                </div>
              )}

              {/* Single-Zone Recalibration Quick Buttons */}
              <div className="flex items-center gap-1.5 flex-wrap text-[11px] font-mono">
                <span className="text-muted-foreground mr-1">Recalibrate Single Zone:</span>
                {(["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"] as const).map((z) => (
                  <button
                    key={z}
                    type="button"
                    onClick={() => {
                      if (cameraState !== "active") {
                        handleStartMonitoring().then(() => recalibrateZone(z, 4.0));
                      } else {
                        recalibrateZone(z, 4.0);
                      }
                    }}
                    className="px-2 py-0.5 rounded-md border border-border bg-card hover:bg-muted text-foreground transition-colors cursor-pointer"
                  >
                    {z.replace("_", " ")}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Camera Viewport */}
          <CameraView
            videoRef={videoRef}
            cameraState={cameraState}
            alertState={effectiveTelemetry?.state ?? "MONITORING_UNAVAILABLE"}
            isStale={runtimeMode === "browser" ? false : isStale}
            isMirrored={isMirrored}
            framingMode={framingMode}
            viewOrientation={viewOrientation}
            isFullscreen={isFullscreen}
            opticalQuality={opticalQuality}
            onStartClick={handleStartMonitoring}
          >
            {/* In-viewport Floating Cockpit HUD Overlays */}
            {cameraState === "active" && (
              <>
                {/* Top Floating Status Strip */}
                <div className="absolute top-3 left-3 right-3 flex items-center justify-between pointer-events-none z-10">
                  <div className="flex items-center gap-1.5 flex-wrap">
                    {renderOverlayStateBadge()}

                    {effectiveTelemetry && (
                      <Badge
                        variant="outline"
                        className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono px-2 py-0.5"
                      >
                        Zone: {effectiveTelemetry.pose_zone}
                      </Badge>
                    )}
                  </div>

                  {/* Latency & Resolution */}
                  <div className="flex items-center gap-1.5">
                    {roundtripLatencyMs > 0 && runtimeMode === "remote" && (
                      <Badge
                        variant="outline"
                        className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono tabular-nums px-2 py-0.5"
                      >
                        RTT {roundtripLatencyMs}ms
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
                      {audioStatus === "blocked" ? (
                        <Button
                          size="icon"
                          variant="secondary"
                          onClick={() => unlockAudio()}
                          className="h-10 w-10 rounded-xl bg-amber-500/80 backdrop-blur-md border border-amber-400 text-white animate-pulse"
                          title="Browser blocked audio. Click to Enable Sound Alerts"
                        >
                          <VolumeX className="size-4" />
                        </Button>
                      ) : (
                        <Button
                          size="icon"
                          variant="secondary"
                          onClick={toggleMute}
                          className="h-10 w-10 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                          title={audioMuted ? "Unmute Audio Alerts" : "Mute Audio Alerts"}
                        >
                          {audioMuted ? <VolumeX className="size-4" /> : <Volume2 className="size-4 text-primary" />}
                        </Button>
                      )}

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

          {/* Controls Bar & Rolling 60s Waveform Chart (Visible in standard non-fullscreen view) */}
          {!isFullscreen && (
            <>
              <CameraControls
                cameraState={cameraState}
                isMirrored={isMirrored}
                framingMode={framingMode}
                isFullscreen={isFullscreen}
                onStart={handleStartMonitoring}
                onStop={handleStopMonitoring}
                onPause={() => {
                  audioEngine?.stopCurrentAlert();
                  pauseCamera();
                }}
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

              {/* Rolling 60-Second Live Telemetry Chart (Visible in Engineering Diagnostics mode) */}
              {consoleMode === "diagnostics" && (
                <LiveTelemetryChart data={chartHistory} />
              )}
            </>
          )}
        </div>

        {/* Companion Sidebar Panel: Real-Time Telemetry or Clean Driver Cockpit HUD */}
        {!isFullscreen && (
          <div
            className={`space-y-4 ${
              isPortrait ? "md:col-span-5 lg:col-span-6" : "lg:col-span-5"
            }`}
          >
            {consoleMode === "cockpit" ? (
              <DriverCockpitHUD
                telemetry={effectiveTelemetry}
                cameraState={cameraState}
                isStale={runtimeMode === "browser" ? false : isStale}
                isAudioUnlocked={isAudioUnlocked}
                isMuted={audioMuted}
                volume={audioVolume}
                audioStatus={audioStatus}
                onToggleMute={toggleMute}
                onSetVolume={setAudioVolume}
                onUnlockAudio={unlockAudio}
                onSwitchToDiagnostics={() => handleConsoleModeChange("diagnostics")}
              />
            ) : (
              <>
                {/* Live AI Telemetry HUD Gauges */}
                <LiveTelemetryHUD
                  telemetry={effectiveTelemetry}
                  cameraState={cameraState}
                  backendType={backendType}
                  latencyDiagnostics={latencyDiagnostics}
                  isStale={runtimeMode === "browser" ? false : isStale}
                  isAudioUnlocked={isAudioUnlocked}
                  isMuted={audioMuted}
                  volume={audioVolume}
                  audioStatus={audioStatus}
                  onToggleMute={toggleMute}
                  onSetVolume={setAudioVolume}
                  onUnlockAudio={unlockAudio}
                  onTestAudioTier={testAlert}
                  onSwitchToCockpit={() => handleConsoleModeChange("cockpit")}
                />

                {/* Structured Explainable Risk & Stage Latency Breakdown Panel */}
                <ExplainableRiskPanel
                  telemetry={effectiveTelemetry}
                  latencyDiagnostics={latencyDiagnostics}
                  isStale={runtimeMode === "browser" ? false : isStale}
                />

                {/* Sub-Frame Blink Kinematics Oscilloscope (Innovation 2) */}
                <KinematicsOscilloscope
                  currentEar={effectiveTelemetry?.eye_metrics?.ear_raw ?? 0.285}
                  closureThreshold={effectiveTelemetry?.comparison?.personalised_threshold ?? 0.20}
                  dEarDt={effectiveTelemetry?.eye_metrics?.blink_rcr ? -0.15 : 0}
                  rcr={effectiveTelemetry?.eye_metrics?.blink_rcr ?? undefined}
                />

                {/* 3D Spatial Head Gyroscope & Zone Cones (Innovation 1) */}
                <EulerHeadGyro
                  pitchDeg={effectiveTelemetry?.head_pose?.pitch_deg ?? 0}
                  yawDeg={effectiveTelemetry?.head_pose?.yaw_deg ?? 0}
                  rollDeg={effectiveTelemetry?.head_pose?.roll_deg ?? 0}
                  zone={effectiveTelemetry?.comparison?.personalised_zone ?? "CENTER"}
                />

                {/* Academic Viva & Fault Injection Testbench */}
                <FaultInjectionToolbar
                  activeFault={activeFault}
                  onInjectMirrorGlance={() => setActiveFault("MIRROR_GLANCE")}
                  onInjectSluggishReopen={() => setActiveFault("SLUGGISH_REOPEN")}
                  onInjectGlare={() => setActiveFault("GLARE_DEFOCUS")}
                  onInjectWatchdogTimeout={() => setActiveFault("WATCHDOG_TIMEOUT")}
                  onResetFaults={() => setActiveFault(null)}
                />

                {/* Hardware & Cascade Configuration Card */}
                <Card className="border-border/80 shadow-xs overflow-hidden">
                  <CardHeader className="p-3.5 pb-2 border-b border-border/50 bg-muted/20">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <SlidersHorizontal className="size-4 text-primary" />
                        <CardTitle className="text-xs font-bold uppercase tracking-wider text-foreground">
                          Configuration &amp; Sensors
                        </CardTitle>
                      </div>
                      <Badge variant="outline" className="text-[10px] font-mono text-muted-foreground border-border/60">
                        {backendType === "cloud" ? "Cloud AI" : "Local:8000"}
                      </Badge>
                    </div>
                  </CardHeader>
                  <CardContent className="p-3">
                    <Tabs defaultValue="cascade" className="w-full space-y-3">
                      <TabsList className="grid grid-cols-3 w-full h-9 bg-muted/50 p-1 rounded-xl">
                        <TabsTrigger
                          value="cascade"
                          className="text-xs font-medium rounded-lg data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-2xs"
                        >
                          <Sparkles className="size-3 mr-1 text-primary shrink-0" />
                          <span>Cascade</span>
                        </TabsTrigger>
                        <TabsTrigger
                          value="camera"
                          className="text-xs font-medium rounded-lg data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-2xs"
                        >
                          <Camera className="size-3 mr-1 text-primary shrink-0" />
                          <span>Camera</span>
                        </TabsTrigger>
                        <TabsTrigger
                          value="engine"
                          className="text-xs font-medium rounded-lg data-[state=active]:bg-card data-[state=active]:text-foreground data-[state=active]:shadow-2xs"
                        >
                          <Cpu className="size-3 mr-1 text-primary shrink-0" />
                          <span>Engine</span>
                        </TabsTrigger>
                      </TabsList>

                      {/* Tab 1: Experiment Selector */}
                      <TabsContent value="cascade" className="mt-0 focus-visible:outline-none">
                        <ExperimentSelector
                          currentMode={experimentMode}
                          onSelectMode={switchExperiment}
                        />
                      </TabsContent>

                      {/* Tab 2: Camera Selector */}
                      <TabsContent value="camera" className="mt-0 focus-visible:outline-none space-y-2">
                        <div className="text-[11px] text-muted-foreground">
                          Select active hardware optical sensor for driver eye and face tracking:
                        </div>
                        <CameraSelector
                          devices={devices}
                          selectedDeviceId={selectedDeviceId}
                          onSelectDevice={selectDevice}
                          onRefreshDevices={refreshDevices}
                        />
                      </TabsContent>

                      {/* Tab 3: AI Inference Target */}
                      <TabsContent value="engine" className="mt-0 focus-visible:outline-none space-y-2.5">
                        <div className="flex items-center justify-between text-xs">
                          <span className="text-muted-foreground">Active Server Endpoint:</span>
                          <span className="font-mono text-foreground font-semibold" suppressHydrationWarning>
                            {backendType === "cloud" ? "LiveKit Cloud (Render)" : "127.0.0.1:8000"}
                          </span>
                        </div>
                        <div className="flex items-center gap-2" suppressHydrationWarning>
                          <Button
                            size="sm"
                            variant={backendType === "cloud" ? "default" : "outline"}
                            onClick={() => handleSwitchBackend("cloud")}
                            className="text-xs h-9 flex-1 rounded-xl"
                          >
                            LiveKit Cloud
                          </Button>
                          <Button
                            size="sm"
                            variant={backendType === "local" ? "default" : "outline"}
                            onClick={() => handleSwitchBackend("local")}
                            className="text-xs h-9 flex-1 rounded-xl"
                          >
                            Localhost (8000)
                          </Button>
                        </div>
                      </TabsContent>
                    </Tabs>
                  </CardContent>
                </Card>
              </>
            )}
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
