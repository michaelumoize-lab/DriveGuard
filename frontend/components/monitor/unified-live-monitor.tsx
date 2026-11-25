"use client";

import React, { useState, useEffect, useRef, useCallback, useMemo, useId } from "react";
import Link from "next/link";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useLivekitTelemetryStream } from "@/hooks/use-livekit-telemetry-stream";
import { useBrowserDetection } from "@/hooks/use-browser-detection";
import { usePythonLocalEngine } from "@/hooks/use-python-local-engine";
import {
  executeTwoPhaseCameraHandoff,
  stopAndVerifyBrowserMediaStream,
} from "@/lib/monitor/engine-adapter";
import {
  AlertTransition,
  convertBrowserTelemetryToFrameTelemetry,
  DecisionResult,
  DetectionTelemetry,
  DriverBehaviourProfileData,
  DrowsinessState,
  PoseZone,
} from "@/lib/ai";
import { CapturedFrame } from "@/types/camera";
import {
  AlertState,
  SafetyEvent,
  SessionEndPayload,
  resolveEffectiveDisplayState,
} from "@/types/telemetry";
import { buildDriverProfileDtoPayload } from "@/lib/api/profile-mapper";
import { syncDriverProfileToFastApi } from "@/lib/api/driveguard-api";
import { LOCAL_FASTAPI_URL } from "@/lib/fastapi-url";
import { CameraView } from "@/components/camera/camera-view";
import { CameraControls } from "@/components/camera/camera-controls";
import { CameraStatus } from "@/components/camera/camera-status";
import { CameraErrorDisplay } from "@/components/camera/camera-error";
import { CameraSnapshotPreview } from "@/components/camera/camera-snapshot-preview";
import { DriverCockpitHUD } from "./driver-cockpit-hud";
import { MonitorHeaderBar } from "./monitor-header-bar";
import { InspectorWorkspace, InspectorTab } from "./inspector-workspace";
import { TelemetryHistoryPoint } from "./live-telemetry-chart";
import { useDriveGuardAudio } from "@/hooks/use-driveguard-audio";
import { runPreFlightHandshake } from "@/lib/monitor/preflight-handshake";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { useSidebar } from "@/components/ui/sidebar";
import {
  AlertTriangle,
  Compass,
  EyeOff,
  Flame,
  Minimize2,
  RefreshCw,
  ShieldCheck,
  Volume2,
  VolumeX,
} from "lucide-react";

export interface InitialDriverProfileData {
  engineType?: "BROWSER" | "PYTHON" | string;
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
  neutralRotMat?: number[][] | null;
  calibrationIsMirrored?: boolean | null;
  calibrationReliability: number;
  profileVersion?: number;
  profileHash?: string | null;
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
  initialPythonProfile?: InitialDriverProfileData | null;
  initialPreferences?: InitialUserPreferencesData;
  initialConsoleMode?: "cockpit" | "diagnostics";
  initialInspectorTab?: InspectorTab;
  initialRuntimeMode?: "browser" | "python-local" | "remote";
  initialOrientation?: "portrait" | "landscape";
}

export function UnifiedLiveMonitor({
  driverId = "driver_default",
  initialDriverProfile = null,
  initialPythonProfile = null,
  initialPreferences,
  initialConsoleMode,
  initialInspectorTab,
  initialRuntimeMode,
  initialOrientation,
}: UnifiedLiveMonitorProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const { setOpen: setSidebarOpen } = useSidebar();
  const hasAutoCollapsedRef = useRef(false);

  // Auto-collapse the left navigation sidebar once on mount when entering Live Monitoring
  // so the cockpit has maximum horizontal breathing room, while still allowing the user
  // to expand it freely via the sidebar toggle.
  useEffect(() => {
    if (hasAutoCollapsedRef.current) return;
    hasAutoCollapsedRef.current = true;
    setSidebarOpen(false);
  }, [setSidebarOpen]);

  // Layout & UI States (Default to calm Driver Mode for first-time understandability)
  const [runtimeMode, setRuntimeMode] = useState<"browser" | "python-local" | "remote">(
    initialRuntimeMode ?? "browser"
  );
  const [isSwitchingEngine, setIsSwitchingEngine] = useState(false);
  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">(
    initialOrientation ?? "landscape"
  );
  const [consoleMode, setConsoleMode] = useState<"cockpit" | "diagnostics">(
    initialConsoleMode ?? "cockpit"
  );
  const [inspectorTab, setInspectorTab] = useState<InspectorTab>(
    initialInspectorTab ?? "overview"
  );
  const [framingMode, setFramingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isMobileDevice, setIsMobileDevice] = useState(false);
  const [showInlineCalibration, setShowInlineCalibration] = useState(false);
  const [pythonIsCalibrated, setPythonIsCalibrated] = useState<boolean>(
    Boolean((initialPythonProfile ?? (initialDriverProfile?.engineType === "PYTHON" ? initialDriverProfile : null))?.isCalibrated)
  );
  const [pythonCenterEarBaseline, setPythonCenterEarBaseline] = useState<number>(
    (initialPythonProfile ?? (initialDriverProfile?.engineType === "PYTHON" ? initialDriverProfile : null))?.centerEarMedian ?? 0.285
  );

  // Restore saved preferences asynchronously after hydration when URL params didn't specify them
  useEffect(() => {
    let cancelled = false;
    Promise.resolve().then(() => {
      if (cancelled) return;
      try {
        if (!initialRuntimeMode) {
          const savedMode = localStorage.getItem("driveguard_runtime_mode");
          if (savedMode === "browser" || savedMode === "python-local" || savedMode === "remote") {
            setRuntimeMode(savedMode);
          }
        }
        if (!initialConsoleMode) {
          const saved = localStorage.getItem("driveguard_console_mode");
          if (saved === "cockpit" || saved === "diagnostics") {
            setConsoleMode(saved);
          }
        }
      } catch {
        // Safe fallback if localStorage is disabled
      }
    });
    return () => {
      cancelled = true;
    };
  }, [initialRuntimeMode, initialConsoleMode]);

  // Sync workspace state to URL search params so views are deep-linkable and survive refreshes
  useEffect(() => {
    if (typeof window === "undefined") return;
    const url = new URL(window.location.href);
    url.searchParams.set("mode", consoleMode === "cockpit" ? "driver" : "inspector");
    if (consoleMode === "diagnostics") {
      url.searchParams.set("tab", inspectorTab);
    } else {
      url.searchParams.delete("tab");
    }
    url.searchParams.set("runtime", runtimeMode);
    url.searchParams.set("view", viewOrientation);
    window.history.replaceState(null, "", url.pathname + url.search);
  }, [consoleMode, inspectorTab, runtimeMode, viewOrientation]);

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

  // Stable session identifier (deterministic React ID across renders)
  const reactId = useId();
  const stableSessionId = useMemo(
    () => `live_${reactId.replace(/[^a-zA-Z0-9_-]/g, "")}`,
    [reactId]
  );

  // Comprehensive Session & Database Persistence Refs
  const activeSessionCodeRef = useRef<string | null>(null);
  const isMonitoringAudioAllowedRef = useRef<boolean>(false);
  const [hasActiveSession, setHasActiveSession] = useState(false);
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

      // Fast-path Web Audio alert dispatch (strictly gated on active monitoring)
      if (isMonitoringAudioAllowedRef.current) {
        audioEngine?.setRiskState(event.new_state, event.event_id, eventTsMs);
      }

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
            eventId: event.event_id,
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
        setPythonIsCalibrated(true);
        if (typeof p.center_ear_median === "number") {
          setPythonCenterEarBaseline(p.center_ear_median);
        }
        fetch("/api/calibration/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            engineType: "PYTHON",
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

  // Handle safety alerts triggered by Browser Detection Engine (AlertStateManager transitions only).
  // Persistence is owned exclusively by BrowserDetectionEngine's OfflineEventQueue (single idempotent channel),
  // so this callback handles immediate Web Audio + screen-reader announcements without duplicate POSTs.
  const handleBrowserSafetyAlert = useCallback(
    (decision: DecisionResult, t: DetectionTelemetry, transition?: AlertTransition) => {
      const isCritical = decision.state === DrowsinessState.CRITICAL_WARNING;
      const isWarning = decision.state === DrowsinessState.DROWSINESS_WARNING;
      if (!isCritical && !isWarning) return;

      const alertState = isCritical ? "CRITICAL" : "WARNING";
      const eventId =
        transition?.eventId ??
        `evt_browser_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      const eventTsMs = t.timestampMs || Date.now();

      if (isMonitoringAudioAllowedRef.current) {
        audioEngine?.setRiskState(alertState, eventId, eventTsMs);
      }

      if (alertState !== lastAnnouncedStateRef.current) {
        lastAnnouncedStateRef.current = alertState;
        if (isCritical) {
          setSrAnnouncement(
            "Critical drowsiness alert! Prolonged eye closure or high fatigue detected. Pull over safely."
          );
        } else {
          setSrAnnouncement("Drowsiness warning: elevated eyelid closure detected.");
        }
      }

      lastAlertTimeRef.current = Date.now();
      if (isCritical) {
        sessionStatsRef.current.criticalCount++;
      } else {
        sessionStatsRef.current.warningCount++;
      }
    },
    [audioEngine]
  );

  // Browser-Side AI Detection Hook (Local MediaPipe Face Landmarker & TypeScript detection)
  // Note: isMirrored is explicitly false because raw browser webcam video pixels are NOT flipped
  // (CSS scale-x-[-1] preview mirroring must never invert head-pose yaw coordinates).
  const initialBrowserProfile =
    initialDriverProfile && initialDriverProfile.engineType !== "PYTHON"
      ? initialDriverProfile
      : null;

  const {
    engineStatus: browserEngineStatus,
    telemetry: browserTelemetry,
    calibrationPhase: browserCalibrationPhase,
    isCalibrated: browserIsCalibrated,
    activeProfile: browserActiveProfile,
    sessionCode: browserSessionCode,
    setSessionCode: setBrowserSessionCode,
    syncStatus: browserSyncStatus,
    startMonitoring: startBrowserMonitoring,
    stopMonitoring: stopBrowserMonitoring,
    recordedFrameCount: browserRecordedFrames,
    exportTelemetryJsonl: exportBrowserTelemetryJsonl,
    startCalibration: startBrowserCalibration,
    skipCalibrationStep: skipBrowserCalibrationStep,
    recalibrateZone: recalibrateBrowserZone,
    cancelCalibration: cancelBrowserCalibration,
    setOpticalQuality: setBrowserOpticalQuality,
    startOfflineSession,
    queueSessionEnd: queueBrowserSessionEnd,
  } = useBrowserDetection({
    driverId,
    initialProfile: (initialBrowserProfile as unknown) as DriverBehaviourProfileData | null,
    isMirrored: false,
    sensitivityLevel: initialPreferences?.sensitivityLevel ?? "STANDARD",
    mirrorSuppression: initialPreferences?.mirrorSuppression ?? true,
    targetInferenceFps: 12,
    autoStartSession: false,
    onSafetyAlert: handleBrowserSafetyAlert,
    onCalibrationCompleted: (profile) => {
      toast.success(
        `Browser AI 4-Zone Calibration complete (Center EAR: ${profile.centerEarMedian.toFixed(3)}) — saved locally & syncing.`,
        { id: "browser-calibration-completed-toast" }
      );
    },
    onCalibrationProgress: (progress) => {
      if (progress.isCompleted && progress.rejectedReason) {
        toast.error(`Calibration rejected: ${progress.rejectedReason}`, {
          id: "browser-calibration-rejected-toast",
        });
      }
    },
  });

  // 2C. Local Python AI Hook (Direct Loopback OpenCV Camera + SQLite — zero getUserMedia/LiveKit)
  const pythonLocal = usePythonLocalEngine({
    driverId,
    enabled: runtimeMode === "python-local",
    onSafetyEvent: handleSafetyEvent,
    onCalibration: (cal) => {
      if (cal.is_completed && !cal.rejection_reason && cal.profile) {
        setPythonIsCalibrated(true);
        if (typeof cal.profile.center_ear_median === "number") {
          setPythonCenterEarBaseline(cal.profile.center_ear_median);
        }
        toast.success("Local Python AI 4-Zone Calibration saved to local ProfileStore.", {
          id: "python-local-calibration-completed-toast",
        });
      } else if (cal.is_completed && cal.rejection_reason) {
        toast.error(`Calibration rejected: ${cal.rejection_reason}`, {
          id: "python-local-calibration-rejected-toast",
        });
      }
    },
  });

  // Forward live optical quality assessments to BrowserDetectionEngine SignalConfidenceEvaluator
  useEffect(() => {
    setBrowserOpticalQuality(opticalQuality ?? null);
  }, [opticalQuality, setBrowserOpticalQuality]);

  // Engine-scoped calibration truth: Browser and Python calibrations never bleed into each other
  const isCalibrated =
    runtimeMode === "browser"
      ? browserIsCalibrated
      : runtimeMode === "python-local"
        ? pythonLocal.isCalibrated || pythonIsCalibrated
        : pythonIsCalibrated;
  const centerEarBaseline =
    runtimeMode === "browser"
      ? (browserActiveProfile?.centerEarMedian ?? initialBrowserProfile?.centerEarMedian ?? 0.285)
      : runtimeMode === "python-local"
        ? (typeof pythonLocal.activeProfile?.center_ear_median === "number"
            ? pythonLocal.activeProfile.center_ear_median
            : pythonCenterEarBaseline)
        : pythonCenterEarBaseline;

  // Convert browser telemetry to FrameTelemetry for existing HUD components
  const convertedBrowserTelemetry = useMemo(() => {
    if (!browserTelemetry) return null;
    return convertBrowserTelemetryToFrameTelemetry(browserTelemetry);
  }, [browserTelemetry]);

  // Unified camera, telemetry & processing state (always cleared once session is stopped/idle)
  const effectiveCameraState =
    runtimeMode === "python-local"
      ? pythonLocal.isStarting
        ? "permission-granted"
        : pythonLocal.isRunning
          ? "active"
          : "stopped"
      : cameraState;

  const isSessionLive =
    runtimeMode === "python-local"
      ? pythonLocal.isRunning
      : cameraState === "active" || cameraState === "paused";
  const effectiveTelemetry = isSessionLive
    ? runtimeMode === "browser"
      ? convertedBrowserTelemetry
      : runtimeMode === "python-local"
        ? pythonLocal.telemetry
        : telemetry
    : null;
  const effectiveLatencyDiagnostics =
    isSessionLive && runtimeMode === "remote" ? latencyDiagnostics : null;
  const effectiveProcessingStatus = !isSessionLive
    ? "idle"
    : runtimeMode === "browser"
      ? (browserEngineStatus === "running" ? "processing" : browserEngineStatus === "initializing" ? "connecting" : "idle")
      : runtimeMode === "python-local"
        ? pythonLocal.backendProcessingStatus
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
    const pythonSource =
      initialPythonProfile ??
      (initialDriverProfile?.engineType === "PYTHON" ? initialDriverProfile : null);
    if (!pythonSource?.isCalibrated || !driverId) return;

    const fastApiProfileDto = buildDriverProfileDtoPayload(
      {
        ...pythonSource,
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
  }, [backendType, driverId, initialDriverProfile, initialPythonProfile, runtimeMode]);

  // Reactive Stream Trigger: automatically start/stop browser/remote inference loop when browser camera activates.
  // Strictly inactive in "python-local" mode (Python owns OpenCV camera directly — zero getUserMedia).
  useEffect(() => {
    if (runtimeMode === "python-local") {
      return;
    }
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

    // Synchronize audio subsystem with continuous telemetry state ONLY while actively monitoring
    if (
      effectiveTelemetry.state &&
      isMonitoringAudioAllowedRef.current &&
      effectiveCameraState === "active"
    ) {
      audioEngine?.setRiskState(effectiveTelemetry.state, undefined, effectiveTelemetry.timestamp_ms);
    }

    // Fallback sustained CRITICAL logging for Remote LiveKit mode only.
    // Browser AI persistence is owned solely by AlertStateManager + OfflineEventQueue.
    if (
      runtimeMode === "remote" &&
      effectiveTelemetry.state === "CRITICAL" &&
      activeSessionCodeRef.current
    ) {
      if (nowMs - lastAlertTimeRef.current >= 8000) {
        lastAlertTimeRef.current = nowMs;
        stats.criticalCount++;
        const earVal =
          effectiveTelemetry.eye_metrics?.ear_normalized ?? (effectiveTelemetry.eye_metrics?.ear_raw ?? 0.15);
        fetch("/api/alerts/log", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            eventId: `evt_remote_sustained_${activeSessionCodeRef.current}_${Math.floor(nowMs / 8000)}`,
            sessionCode: activeSessionCodeRef.current,
            severity: "CRITICAL",
            riskScore: effectiveTelemetry.risk_score ?? 0.85,
            earValue: earVal,
            perclosPct: perclos,
            poseZone: effectiveTelemetry.pose_zone ?? "CENTER",
            reasons: effectiveTelemetry.reason_codes ?? ["CRITICAL_STATE"],
            cascadeDecision: effectiveTelemetry.cascade?.arbitration_winner || "GEOMETRY",
            timestampMs: nowMs,
          }),
        }).catch(() => {});
      }
    }
  }, [effectiveTelemetry, audioEngine, runtimeMode]);

  // Refs so the unmount/pagehide cleanup NEVER re-runs when streamDurationSec ticks every 1s
  const runtimeModeRef = useRef(runtimeMode);
  const streamDurationSecRef = useRef(streamDurationSec);
  const buildSessionEndPayloadRef = useRef(buildSessionEndPayload);
  const queueBrowserSessionEndRef = useRef(queueBrowserSessionEnd);
  const audioEngineRef = useRef(audioEngine);

  useEffect(() => {
    runtimeModeRef.current = runtimeMode;
  }, [runtimeMode]);
  useEffect(() => {
    streamDurationSecRef.current = streamDurationSec;
  }, [streamDurationSec]);
  useEffect(() => {
    buildSessionEndPayloadRef.current = buildSessionEndPayload;
  }, [buildSessionEndPayload]);
  useEffect(() => {
    queueBrowserSessionEndRef.current = queueBrowserSessionEnd;
  }, [queueBrowserSessionEnd]);
  useEffect(() => {
    audioEngineRef.current = audioEngine;
  }, [audioEngine]);

  // Flush active session ONLY on true pagehide / tab close / component unmount
  useEffect(() => {
    const flushActiveSession = () => {
      const code = activeSessionCodeRef.current;
      if (!code) return;
      activeSessionCodeRef.current = null;
      const elapsedSec =
        sessionStartWallMsRef.current > 0
          ? Math.max(1, Math.round((Date.now() - sessionStartWallMsRef.current) / 1000))
          : Math.max(1, streamDurationSecRef.current);
      if (runtimeModeRef.current === "browser") {
        queueBrowserSessionEndRef.current(elapsedSec);
        return;
      }
      const payload = buildSessionEndPayloadRef.current(code, elapsedSec);
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
      isMonitoringAudioAllowedRef.current = false;
      audioEngineRef.current?.resetSilent();
      flushActiveSession();
    };
  }, []);

  // Auto-detect mobile device on mount (respects explicit initialOrientation if provided)
  useEffect(() => {
    if (typeof window !== "undefined") {
      const detectMobile = (applyOrientation: boolean) => {
        const isSmallScreen = window.innerWidth <= 768;
        const isMobileUA =
          /Android|iPhone|iPad|iPod|webOS|BlackBerry|IEMobile|Opera Mini/i.test(
            navigator.userAgent
          );
        const mobile = isSmallScreen || isMobileUA;
        setIsMobileDevice(mobile);

        if (applyOrientation && !initialOrientation) {
          setViewOrientation(mobile ? "portrait" : "landscape");
          setFramingMode("cover");
        }
      };

      detectMobile(true);
      const onResize = () => detectMobile(false);
      window.addEventListener("resize", onResize);
      return () => window.removeEventListener("resize", onResize);
    }
  }, [initialOrientation]);

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

  // Keyboard shortcuts for hands-free desktop navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const target = e.target as HTMLElement | null;
      if (
        target &&
        (target.tagName === "INPUT" ||
          target.tagName === "TEXTAREA" ||
          target.tagName === "SELECT" ||
          target.isContentEditable)
      ) {
        return;
      }

      const key = e.key.toLowerCase();
      if (key === "d") {
        handleConsoleModeChange(consoleMode === "cockpit" ? "diagnostics" : "cockpit");
      } else if (key === "m") {
        toggleMute();
      } else if (key === "f") {
        toggleFullscreen();
      } else if (consoleMode === "diagnostics") {
        if (key === "1") setInspectorTab("overview");
        else if (key === "2") setInspectorTab("signals");
        else if (key === "3") setInspectorTab("testbench");
        else if (key === "4") setInspectorTab("setup");
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [consoleMode, handleConsoleModeChange, toggleFullscreen, toggleMute]);

  // Switch between Browser AI, Local Python AI, and Remote LiveKit AI with strict
  // Two-Phase Exclusive Camera Handoff (release camera + verify ended/released + 200ms OS settle).
  const handleSwitchRuntimeMode = useCallback(
    async (targetMode: "browser" | "python-local" | "remote") => {
      if (isSwitchingEngine || targetMode === runtimeMode) return;
      setIsSwitchingEngine(true);

      const toAdapterId = (m: "browser" | "python-local" | "remote") =>
        m === "browser" ? "BROWSER" : m === "python-local" ? "PYTHON_LOCAL" : "PYTHON_REMOTE";

      try {
        isMonitoringAudioAllowedRef.current = false;
        audioEngine?.resetSilent();

        const handoff = await executeTwoPhaseCameraHandoff(
          {
            engineId: toAdapterId(runtimeMode),
            stop: async () => {
              const code = activeSessionCodeRef.current;
              activeSessionCodeRef.current = null;
              setHasActiveSession(false);

              if (runtimeMode === "browser") {
                if (code) queueBrowserSessionEnd(streamDurationSec);
                stopBrowserMonitoring();
              } else if (runtimeMode === "python-local") {
                await pythonLocal.stop();
              } else {
                stopStreamingLoop();
              }
              return true;
            },
            releaseCamera: async () => {
              if (runtimeMode === "python-local") {
                return await pythonLocal.stop();
              }
              const currentStream = (videoRef.current?.srcObject as MediaStream | null) ?? null;
              stopCamera();
              return stopAndVerifyBrowserMediaStream(currentStream);
            },
          },
          {
            engineId: toAdapterId(targetMode),
            start: async () => true,
          },
          { settleDelayMs: 200, autoStartTarget: false }
        );

        if (!handoff.previousReleased) {
          toast.error(handoff.error || "Could not release current camera handle cleanly.");
          return;
        }

        // Re-silence after async engine handoff completes
        isMonitoringAudioAllowedRef.current = false;
        audioEngine?.resetSilent();

        resetSessionStats();
        setChartHistory([]);
        setRuntimeMode(targetMode);
        try {
          localStorage.setItem("driveguard_runtime_mode", targetMode);
        } catch {}

        toast.info(
          targetMode === "browser"
            ? "Switched to Browser AI (On-device MediaPipe WASM — works offline)."
            : targetMode === "python-local"
              ? "Switched to Local Python AI (Direct OpenCV camera via 127.0.0.1:8000 — zero LiveKit)."
              : "Switched to Remote LiveKit AI (WebRTC + Python DrowsinessEngine)."
        );
      } finally {
        setIsSwitchingEngine(false);
      }
    },
    [
      isSwitchingEngine,
      runtimeMode,
      audioEngine,
      queueBrowserSessionEnd,
      streamDurationSec,
      stopBrowserMonitoring,
      pythonLocal,
      stopStreamingLoop,
      videoRef,
      stopCamera,
      resetSessionStats,
    ]
  );

  const [isPreflightRunning, setIsPreflightRunning] = useState(false);

  // Start live monitoring (Browser AI / Local Python AI / Remote LiveKit AI)
  const handleStartMonitoring = useCallback(async () => {
    if (isSwitchingEngine) return;
    unlockAudio();
    // Ensure clean silent audio baseline before starting a new session
    isMonitoringAudioAllowedRef.current = false;
    audioEngine?.resetSilent();

    resetSessionStats();
    setChartHistory([]);

    // Local Python AI acquires OpenCV/Phone camera directly on 127.0.0.1:8000 — NEVER invokes getUserMedia!
    if (runtimeMode === "python-local") {
      setIsPreflightRunning(true);
      const started = await pythonLocal.start({
        isMirrored: true,
        enableAudio: false,
      });
      setIsPreflightRunning(false);
      if (started) {
        isMonitoringAudioAllowedRef.current = true;
        setHasActiveSession(true);
        toast.success("Local Python AI started (direct OpenCV camera + SQLite persistence).");
      } else {
        toast.error(
          pythonLocal.error ||
            "Could not start Local Python AI. Ensure FastAPI is running on 127.0.0.1:8000 with LOCAL_CAMERA_API_ENABLED=true."
        );
      }
      return;
    }

    setIsPreflightRunning(true);

    // 1. Ensure browser camera stream is initiated (Browser AI & Remote LiveKit only)
    if (cameraState !== "active") {
      await startCamera();
    }

    // 2. Perform Pre-Flight Readiness Handshake
    const isAiReady =
      runtimeMode === "browser"
        ? browserEngineStatus === "running" || browserEngineStatus === "idle" || browserEngineStatus === "initializing"
        : isWsConnected;

    const handshake = await runPreFlightHandshake({
      runtimeMode,
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

    // 3. Authenticate & initialize driving session (with automatic offline fallback in Browser AI mode)
    const deviceLabel = isMobileDevice ? "MOBILE_PORTRAIT" : "DESKTOP_HUD";

    if (runtimeMode === "browser" && handshake.persistenceMode === "OFFLINE_QUEUED") {
      const provisional = startOfflineSession();
      if (provisional) {
        isMonitoringAudioAllowedRef.current = true;
        activeSessionCodeRef.current = provisional.sessionCode;
        setHasActiveSession(true);
        toast.info(
          `Offline mode active: local monitoring session #${provisional.sessionCode} started (queued for sync).`
        );
      }
      return;
    }

    try {
      const startRes = await fetch("/api/sessions/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engineType: runtimeMode === "browser" ? "BROWSER" : "PYTHON",
          mode: experimentMode,
          device: deviceLabel,
        }),
      });

      if (!startRes.ok) {
        throw new Error(`Session start returned HTTP ${startRes.status}`);
      }

      const data = await startRes.json();
      if (data?.sessionCode) {
        isMonitoringAudioAllowedRef.current = true;
        activeSessionCodeRef.current = data.sessionCode;
        setHasActiveSession(true);
        if (runtimeMode === "browser") {
          setBrowserSessionCode(data.sessionCode);
        }
        toast.success(`Monitoring Session Initialized: #${data.sessionCode}`);
      }
    } catch (err) {
      if (runtimeMode === "browser") {
        const provisional = startOfflineSession();
        if (provisional) {
          isMonitoringAudioAllowedRef.current = true;
          activeSessionCodeRef.current = provisional.sessionCode;
          setHasActiveSession(true);
          toast.info(
            `Offline mode active: local monitoring session #${provisional.sessionCode} started (queued for sync).`
          );
        }
        return;
      }
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
    startOfflineSession,
    isSwitchingEngine,
    pythonLocal,
  ]);

  // Stop live monitoring, persist full session summary ONCE & completely clear the cockpit
  const handleStopMonitoring = useCallback(async () => {
    // Disarm audio gate synchronously FIRST so no late frame/event can re-trigger alarms during teardown
    isMonitoringAudioAllowedRef.current = false;
    audioEngine?.resetSilent();

    const code = activeSessionCodeRef.current;
    if (code) {
      activeSessionCodeRef.current = null;
      setHasActiveSession(false);
    }

    if (runtimeMode === "python-local") {
      await pythonLocal.stop();
      setHasActiveSession(false);
      toast.success("Local Python AI stopped & hardware camera released.");
    } else if (runtimeMode === "browser") {
      // Single idempotent session finalization via OfflineEventQueue BEFORE stopping/resetting engine stats
      if (code) {
        const elapsedSec =
          sessionStartWallMsRef.current > 0
            ? Math.max(1, Math.round((Date.now() - sessionStartWallMsRef.current) / 1000))
            : Math.max(1, streamDurationSec);
        const resolvedDurationSec = Math.max(streamDurationSec, elapsedSec);
        queueBrowserSessionEnd(resolvedDurationSec);
        toast.success(`Session ${code} ended & queued for persistence (${resolvedDurationSec}s).`);
      }
      stopBrowserMonitoring();
      stopCamera();
    } else {
      stopStreamingLoop();
      if (code) {
        const payload = {
          ...buildSessionEndPayload(code, streamDurationSec),
          finalizeToken: `fin_${code}_${Date.now()}`,
        };
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
      }
      stopCamera();
    }

    // Re-silence after async teardown completes in case any in-flight callback arrived mid-stop
    isMonitoringAudioAllowedRef.current = false;
    audioEngine?.resetSilent();

    resetSessionStats();
    setChartHistory([]);
    lastChartSampleMsRef.current = 0;
    setCapturedSnapshot(null);
    setIsSnapshotOpen(false);
    setShowInlineCalibration(false);
    setSrAnnouncement("");
    lastAnnouncedStateRef.current = "NORMAL";
    lastAlertTimeRef.current = 0;
    handledEventIdsRef.current.clear();
  }, [audioEngine, runtimeMode, pythonLocal, stopBrowserMonitoring, stopStreamingLoop, stopCamera, buildSessionEndPayload, streamDurationSec, resetSessionStats, queueBrowserSessionEnd]);

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
  const handleExportBrowserTelemetry = useCallback(() => {
    const jsonl = exportBrowserTelemetryJsonl();
    if (!jsonl.trim()) {
      toast.info("No Browser AI telemetry frames recorded yet. Start monitoring first.");
      return;
    }
    const now = new Date();
    const pad = (n: number) => String(n).padStart(2, "0");
    const stamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
    const filename = `browser_session_${stamp}.jsonl`;
    const blob = new Blob([jsonl], { type: "application/x-ndjson;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    toast.success(`Exported ${browserRecordedFrames.toLocaleString()} frames to ${filename}`);
  }, [exportBrowserTelemetryJsonl, browserRecordedFrames]);

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

  const handleToggleCalibration = useCallback(() => {
    setShowInlineCalibration((prev) => !prev);
  }, []);

  const handleOpenInlineCalibration = useCallback(() => {
    setShowInlineCalibration(true);
  }, []);

  const handleSwitchToDiagnostics = useCallback(() => {
    handleConsoleModeChange("diagnostics");
  }, [handleConsoleModeChange]);

  const handleSwitchToCockpit = useCallback(() => {
    handleConsoleModeChange("cockpit");
  }, [handleConsoleModeChange]);

  const handlePauseCamera = useCallback(() => {
    isMonitoringAudioAllowedRef.current = false;
    audioEngine?.resetSilent();
    pauseCamera();
  }, [audioEngine, pauseCamera]);

  const handleResumeCamera = useCallback(() => {
    isMonitoringAudioAllowedRef.current = true;
    resumeCamera();
  }, [resumeCamera]);

  const handleToggleFraming = useCallback(() => {
    setFramingMode((prev) => (prev === "cover" ? "fit" : "cover"));
  }, []);

  const handleToggleMirror = useCallback(() => {
    setIsMirrored((prev) => !prev);
  }, [setIsMirrored]);

  const handleCloseSnapshot = useCallback(() => {
    setIsSnapshotOpen(false);
  }, []);

  const isPortrait = viewOrientation === "portrait";

  // Helper for distinct video overlay badge per authoritative safety & operational state
  // Always routes through resolveEffectiveDisplayState() so uncalibrated/degraded/face-lost
  // frames NEVER display a green NORMAL badge.
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
      if (browserEngineStatus === "error") {
        return (
          <Badge
            variant="outline"
            className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-rose-950/90 text-rose-200 border-rose-400/80 gap-1.5"
          >
            <AlertTriangle className="size-3.5 text-rose-400" />
            <span>ENGINE ERROR</span>
          </Badge>
        );
      }
    }

    const effectiveState = resolveEffectiveDisplayState(
      runtimeMode === "remote" ? telemetry : effectiveTelemetry,
      runtimeMode === "browser"
        ? {
            isConnected: browserEngineStatus === "running",
            isStale: false,
            backendStatus: effectiveProcessingStatus,
          }
        : runtimeMode === "python-local"
          ? {
              isConnected: pythonLocal.connectionStatus === "connected",
              isStale: false,
              backendStatus: effectiveProcessingStatus,
            }
          : {
              isConnected: isWsConnected,
              isStale,
              backendStatus: backendProcessingStatus,
            }
    );

    if (effectiveState === "ERROR") {
      return (
        <Badge
          variant="outline"
          className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-rose-950/90 text-rose-200 border-rose-400/80 gap-1.5"
        >
          <AlertTriangle className="size-3.5 text-rose-400" />
          <span>ENGINE ERROR</span>
        </Badge>
      );
    }

    if (effectiveState === "CALIBRATION_REQUIRED") {
      return (
        <Badge
          variant="outline"
          className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-amber-950/90 text-amber-200 border-amber-400/80 gap-1.5"
        >
          <Compass className="size-3.5 text-amber-400" />
          <span>CALIBRATION REQUIRED</span>
        </Badge>
      );
    }

    if (effectiveState === "CALIBRATING") {
      return (
        <Badge
          variant="outline"
          className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-sky-950/90 text-sky-200 border-sky-400/80 gap-1.5"
        >
          <Compass className="size-3.5 text-sky-400 animate-spin motion-reduce:animate-none" aria-hidden="true" />
          <span>CALIBRATING…</span>
        </Badge>
      );
    }

    if (effectiveState === "DEGRADED") {
      return (
        <Badge
          variant="outline"
          className="text-xs uppercase font-bold shadow-lg px-2.5 py-1 bg-slate-900/90 text-amber-300 border-amber-400/80 border-dashed gap-1.5"
        >
          <EyeOff className="size-3.5 text-amber-400" aria-hidden="true" />
          <span>DEGRADED / NO FACE</span>
        </Badge>
      );
    }

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
          <span className="size-1.5 rounded-full bg-amber-400 animate-pulse motion-reduce:animate-none mr-1.5" aria-hidden="true" />
          AWAITING BACKEND TELEMETRY
        </Badge>
      );
    }

    if (effectiveState === "CRITICAL") {
      return (
        <Badge className="text-xs uppercase font-extrabold shadow-lg px-2.5 py-1 bg-rose-600 text-white border border-rose-300 animate-pulse motion-reduce:animate-none gap-1.5">
          <Flame className="size-3.5" aria-hidden="true" />
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

      {/* 1. Compact Command Bar & Expandable Readiness Checklist (Hidden in Fullscreen) */}
      {!isFullscreen && (
        <MonitorHeaderBar
          consoleMode={consoleMode}
          onConsoleModeChange={handleConsoleModeChange}
          runtimeMode={runtimeMode}
          cameraActive={effectiveCameraState === "active"}
          cameraResolutionLabel={
            resolution ? `${resolution.width}×${resolution.height}` : "640×480"
          }
          faceDetected={Boolean(effectiveTelemetry?.face_detected)}
          isCalibrated={isCalibrated}
          centerEarBaseline={centerEarBaseline}
          livekitStatus={
            runtimeMode === "browser"
              ? "connected"
              : runtimeMode === "python-local"
                ? pythonLocal.connectionStatus
                : wsStatus
          }
          backendProcessingStatus={effectiveProcessingStatus}
          isStale={runtimeMode === "remote" ? isStale : false}
          rttMs={runtimeMode === "remote" ? roundtripLatencyMs : 0}
          audioStatus={audioStatus}
          audioMuted={audioMuted}
          browserSyncStatus={browserSyncStatus}
          showInlineCalibration={showInlineCalibration}
          onToggleCalibration={handleToggleCalibration}
          onUnlockAudio={unlockAudio}
        />
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
          <div className="flex flex-wrap items-center gap-2 shrink-0">
            <Button
              size="sm"
              variant="outline"
              onClick={() => {
                if (videoRef.current) reconnectWs(videoRef.current);
              }}
              className="min-h-11 sm:h-8 sm:min-h-0 text-xs gap-1.5"
            >
              <RefreshCw className="size-3.5" aria-hidden="true" /> Retry Worker
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => handleSwitchBackend(backendType === "cloud" ? "local" : "cloud")}
              className="min-h-11 sm:h-8 sm:min-h-0 text-xs"
            >
              Switch to {backendType === "cloud" ? "Localhost" : "Cloud"}
            </Button>
          </div>
        </div>
      )}

      {/* Local Python AI Connection Warning Banner */}
      {!isFullscreen && runtimeMode === "python-local" && pythonLocal.error && (
        <div
          role="region"
          aria-label="Local Python Telemetry Notice"
          className="p-3.5 rounded-2xl border border-amber-500/40 bg-amber-500/10 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs"
        >
          <div className="space-y-0.5">
            <div className="font-bold text-foreground flex items-center gap-1.5">
              <AlertTriangle className="size-4 text-amber-500 shrink-0" aria-hidden="true" />
              <span>Local Python Telemetry Stream Issue</span>
            </div>
            <p className="text-muted-foreground">{pythonLocal.error}</p>
          </div>
          {pythonLocal.isRunning && (
            <Button
              size="sm"
              variant="outline"
              onClick={() => pythonLocal.reconnectWebSocket()}
              className="min-h-11 sm:h-8 sm:min-h-0 text-xs gap-1.5 shrink-0"
            >
              <RefreshCw className="size-3.5" aria-hidden="true" /> Retry Telemetry WS
            </Button>
          )}
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
          {/* Compact Inline 4-Zone Calibration Bar */}
          {!isFullscreen && showInlineCalibration && (() => {
            const activeCal =
              runtimeMode === "browser"
                ? browserCalibrationPhase
                  ? {
                      is_completed: browserCalibrationPhase.isCompleted,
                      phase_index: browserCalibrationPhase.stepIndex,
                      total_phases: browserCalibrationPhase.totalSteps,
                      current_zone: browserCalibrationPhase.currentZone,
                      zone_progress_pct: browserCalibrationPhase.zoneProgressPct,
                      instruction: browserCalibrationPhase.guidanceText,
                      stable_samples_count: Math.round(browserCalibrationPhase.zoneProgressPct),
                      target_samples_count: 100,
                    }
                  : null
                : runtimeMode === "python-local"
                  ? pythonLocal.calibration
                  : calibration;

            const triggerFullCalibration = () => {
              const run = () => {
                if (runtimeMode === "browser") {
                  startBrowserCalibration();
                } else if (runtimeMode === "python-local") {
                  void pythonLocal.startCalibration(true, 4.0);
                } else {
                  startCalibration(true, 4.0);
                }
              };
              if (effectiveCameraState !== "active") {
                handleStartMonitoring().then(run);
              } else {
                run();
              }
            };

            const triggerSkip = () => {
              if (runtimeMode === "browser") {
                skipBrowserCalibrationStep();
              } else if (runtimeMode === "python-local") {
                void pythonLocal.skipCalibrationStep();
              } else {
                skipCalibrationStep();
              }
            };

            const triggerCancel = () => {
              if (runtimeMode === "browser") {
                cancelBrowserCalibration();
              } else if (runtimeMode === "python-local") {
                void pythonLocal.cancelCalibration();
              } else {
                cancelCalibration();
              }
            };

            const triggerSingleZone = (z: "CENTER" | "LEFT_MIRROR" | "RIGHT_MIRROR" | "DASHBOARD") => {
              const run = () => {
                if (runtimeMode === "browser") {
                  recalibrateBrowserZone(z as PoseZone);
                } else if (runtimeMode === "python-local") {
                  void pythonLocal.recalibrateZone(z, 4.0);
                } else {
                  recalibrateZone(z, 4.0);
                }
              };
              if (effectiveCameraState !== "active") {
                handleStartMonitoring().then(run);
              } else {
                run();
              }
            };

            return (
            <div className="p-3.5 rounded-2xl border border-primary/40 bg-primary/5 space-y-2.5">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <Compass className="size-4 text-primary" aria-hidden="true" />
                  <span className="text-xs font-semibold text-foreground">
                    4-Zone Eye Calibration ({runtimeMode === "browser" ? "Browser AI" : runtimeMode === "python-local" ? "Python Local AI" : "Python Remote AI"})
                  </span>
                  {activeCal && !activeCal.is_completed && (
                    <Badge className="text-[11px] font-mono tabular-nums bg-primary text-primary-foreground">
                      Zone {activeCal.phase_index}/{activeCal.total_phases}: {activeCal.current_zone} ({Math.round(activeCal.zone_progress_pct)}%)
                    </Badge>
                  )}
                </div>
                <div className="flex flex-wrap items-center gap-1.5">
                  <Button
                    type="button"
                    size="sm"
                    onClick={triggerFullCalibration}
                    className="min-h-11 sm:h-7 sm:min-h-0 px-3 sm:px-2.5 text-xs"
                  >
                    Start All 4 Zones
                  </Button>
                  {activeCal && !activeCal.is_completed && (
                    <>
                      <Button
                        type="button"
                        size="sm"
                        variant="outline"
                        onClick={triggerSkip}
                        className="min-h-11 sm:h-7 sm:min-h-0 px-3 sm:px-2 text-xs"
                      >
                        Skip Zone
                      </Button>
                      <Button
                        type="button"
                        size="sm"
                        variant="ghost"
                        onClick={triggerCancel}
                        className="min-h-11 sm:h-7 sm:min-h-0 px-3 sm:px-2 text-xs text-rose-500"
                      >
                        Cancel
                      </Button>
                    </>
                  )}
                  <Button asChild size="sm" variant="ghost" className="min-h-11 sm:h-7 sm:min-h-0 px-3 sm:px-2 text-xs">
                    <Link href="/dashboard/calibration">Full Wizard</Link>
                  </Button>
                </div>
              </div>

              {activeCal && !activeCal.is_completed && (
                <div
                  aria-live="polite"
                  className="text-xs text-foreground font-medium bg-background/80 px-3 py-1.5 rounded-lg border border-border/60 flex flex-wrap items-center justify-between gap-2"
                >
                  <span>{activeCal.instruction}</span>
                  <span className="font-mono tabular-nums text-[11px] text-muted-foreground shrink-0">
                    Samples: {activeCal.stable_samples_count}/{activeCal.target_samples_count}
                  </span>
                </div>
              )}

              {/* Single-Zone Recalibration Quick Buttons */}
              <div className="flex items-center gap-1.5 flex-wrap text-[11px]">
                <span className="text-muted-foreground mr-1">Single zone:</span>
                {(["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"] as const).map((z) => (
                  <button
                    key={z}
                    type="button"
                    onClick={() => triggerSingleZone(z)}
                    className="min-h-11 sm:min-h-7 px-2.5 sm:px-2 py-1 sm:py-0.5 rounded-md border border-border bg-card hover:bg-muted text-foreground font-mono transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                  >
                    {z.replace("_", " ")}
                  </button>
                ))}
              </div>
            </div>
            );
          })()}

          {/* Camera Viewport */}
          <CameraView
            videoRef={videoRef}
            cameraState={effectiveCameraState}
            previewMode={runtimeMode === "python-local" ? "mjpeg" : "video"}
            mjpegUrl={runtimeMode === "python-local" ? pythonLocal.mjpegPreviewUrl : null}
            alertState={effectiveTelemetry?.state ?? "MONITORING_UNAVAILABLE"}
            isStale={runtimeMode === "remote" ? isStale : false}
            isMirrored={isMirrored}
            framingMode={framingMode}
            viewOrientation={viewOrientation}
            isFullscreen={isFullscreen}
            opticalQuality={opticalQuality}
            onStartClick={handleStartMonitoring}
          >
            {/* In-viewport Floating Cockpit HUD Overlays */}
            {effectiveCameraState === "active" && (
              <>
                {/* Top Floating Status Strip */}
                <div
                  className={`absolute flex items-start justify-between pointer-events-none z-10 gap-2 ${
                    isFullscreen
                      ? "top-[max(0.75rem,env(safe-area-inset-top,0px))] left-[max(0.75rem,env(safe-area-inset-left,0px))] right-[max(0.75rem,env(safe-area-inset-right,0px))]"
                      : "top-3 left-3 right-3"
                  }`}
                >
                  <div className="flex items-center gap-1.5 flex-wrap pointer-events-auto">
                    {renderOverlayStateBadge()}

                    {effectiveTelemetry && (
                      <Badge
                        variant="outline"
                        className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono px-2 py-0.5"
                      >
                        {effectiveTelemetry.pose_zone}
                      </Badge>
                    )}
                  </div>

                  {/* Right side: Optical Quality & Latency (Shown in Inspector mode only to keep Driver mode calm) */}
                  {consoleMode === "diagnostics" && (
                    <div className="flex items-center gap-1.5 pointer-events-auto flex-wrap justify-end">
                      {opticalQuality && (
                        <div className="flex items-center gap-1.5 bg-black/75 backdrop-blur-md px-2.5 py-1 rounded-md border border-white/15 text-[10px] font-mono tabular-nums text-zinc-300 shadow-md">
                          <span
                            aria-hidden="true"
                            className={`size-1.5 rounded-full ${
                              opticalQuality.overallQuality === "EXCELLENT" || opticalQuality.overallQuality === "ACCEPTABLE"
                                ? "bg-emerald-400"
                                : "bg-amber-400"
                            }`}
                          />
                          <span>LUM: {Math.round(opticalQuality.meanLuminance)}</span>
                          <span className="text-zinc-500" aria-hidden="true">·</span>
                          <span className={opticalQuality.blurCondition === "BLURRY" ? "text-rose-400" : "text-zinc-300"}>
                            {opticalQuality.blurCondition} ({Math.round(opticalQuality.blurVarianceScore)})
                          </span>
                        </div>
                      )}

                      {roundtripLatencyMs > 0 && runtimeMode === "remote" && (
                        <Badge
                          variant="outline"
                          className="bg-black/75 backdrop-blur-md border-white/20 text-white text-[10px] font-mono tabular-nums px-2 py-0.5"
                        >
                          RTT {roundtripLatencyMs}&nbsp;ms
                        </Badge>
                      )}
                    </div>
                  )}
                </div>

                {/* Minimal Bottom Overlay in Fullscreen */}
                {isFullscreen && (
                  <div className="absolute bottom-[max(1rem,env(safe-area-inset-bottom,0px))] left-[max(1rem,env(safe-area-inset-left,0px))] right-[max(1rem,env(safe-area-inset-right,0px))] flex flex-wrap items-center justify-between gap-2 z-30 pointer-events-auto">
                    <Button
                      size="sm"
                      variant="destructive"
                      onClick={handleStopMonitoring}
                      className="rounded-xl min-h-11 px-4 text-xs font-semibold shadow-xl"
                    >
                      Stop Stream
                    </Button>

                    <div className="flex items-center gap-2">
                      {audioStatus === "blocked" ? (
                        <Button
                          size="icon"
                          variant="secondary"
                          onClick={() => unlockAudio()}
                          className="size-11 rounded-xl bg-amber-500/80 backdrop-blur-md border border-amber-400 text-white animate-pulse motion-reduce:animate-none"
                          title="Browser blocked audio. Click to Enable Sound Alerts"
                          aria-label="Enable sound alerts"
                        >
                          <VolumeX className="size-4" aria-hidden="true" />
                        </Button>
                      ) : (
                        <Button
                          size="icon"
                          variant="secondary"
                          onClick={toggleMute}
                          className="size-11 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                          title={audioMuted ? "Unmute Audio Alerts" : "Mute Audio Alerts"}
                          aria-label={audioMuted ? "Unmute audio alerts" : "Mute audio alerts"}
                        >
                          {audioMuted ? (
                            <VolumeX className="size-4" aria-hidden="true" />
                          ) : (
                            <Volume2 className="size-4 text-primary" aria-hidden="true" />
                          )}
                        </Button>
                      )}

                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={() => setFramingMode((prev) => (prev === "cover" ? "fit" : "cover"))}
                        className="size-11 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                        title={framingMode === "cover" ? "Switch to uncropped fit" : "Switch to viewport cover"}
                        aria-label={framingMode === "cover" ? "Switch to uncropped fit view" : "Switch to viewport cover view"}
                      >
                        <Flame className="size-4 text-primary" aria-hidden="true" />
                      </Button>

                      <Button
                        size="icon"
                        variant="secondary"
                        onClick={toggleFullscreen}
                        className="size-11 rounded-xl bg-black/80 backdrop-blur-md border border-white/20 text-white"
                        title="Exit Fullscreen"
                        aria-label="Exit fullscreen monitor"
                      >
                        <Minimize2 className="size-4" aria-hidden="true" />
                      </Button>
                    </div>
                  </div>
                )}
              </>
            )}
          </CameraView>

          {/* Unified Controls Bar & Camera Status (Visible in standard non-fullscreen view) */}
          {!isFullscreen && (
            <>
              <CameraControls
                cameraState={effectiveCameraState}
                isMirrored={isMirrored}
                framingMode={framingMode}
                isFullscreen={isFullscreen}
                onStart={handleStartMonitoring}
                onStop={handleStopMonitoring}
                onPause={handlePauseCamera}
                onResume={handleResumeCamera}
                onSwitchFacing={switchFacingMode}
                onCaptureSnapshot={handleCaptureSnapshot}
                onToggleFraming={handleToggleFraming}
                onToggleFullscreen={toggleFullscreen}
                onToggleMirror={handleToggleMirror}
                isCapturing={isCapturing}
                audioStatus={audioStatus}
                isMuted={audioMuted}
                onToggleMute={toggleMute}
                onUnlockAudio={unlockAudio}
              />

              <CameraStatus
                cameraState={effectiveCameraState}
                resolution={resolution}
                facingMode={facingMode}
                durationSec={streamDurationSec}
                runtimeLabel={
                  runtimeMode === "browser"
                    ? "On-Device Privacy AI"
                    : runtimeMode === "python-local"
                      ? "Local Python OpenCV AI"
                      : "Remote LiveKit AI"
                }
              />
            </>
          )}
        </div>

        {/* Companion Panel: Calm Driver Cockpit or 4-Tab Engineering Inspector */}
        {!isFullscreen && (
          <div
            className={`space-y-4 ${
              isPortrait ? "md:col-span-5 lg:col-span-6" : "lg:col-span-5"
            }`}
          >
            {consoleMode === "cockpit" ? (
              <DriverCockpitHUD
                telemetry={effectiveTelemetry}
                cameraState={effectiveCameraState}
                isStale={runtimeMode === "remote" ? isStale : false}
                isCalibrated={isCalibrated}
                runtimeMode={runtimeMode}
                sessionDurationSec={streamDurationSec}
                isAudioUnlocked={isAudioUnlocked}
                isMuted={audioMuted}
                volume={audioVolume}
                audioStatus={audioStatus}
                onStartMonitoring={handleStartMonitoring}
                onCalibrateClick={handleOpenInlineCalibration}
                onToggleMute={toggleMute}
                onSetVolume={setAudioVolume}
                onUnlockAudio={unlockAudio}
                onSwitchToDiagnostics={handleSwitchToDiagnostics}
              />
            ) : (
              <InspectorWorkspace
                activeTab={inspectorTab}
                onTabChange={setInspectorTab}
                telemetry={effectiveTelemetry}
                cameraState={effectiveCameraState}
                runtimeMode={runtimeMode}
                onSwitchRuntimeMode={handleSwitchRuntimeMode}
                isSwitchingEngine={isSwitchingEngine}
                backendType={backendType}
                onSwitchBackend={handleSwitchBackend}
                latencyDiagnostics={effectiveLatencyDiagnostics}
                isStale={runtimeMode === "remote" ? isStale : false}
                isAudioUnlocked={isAudioUnlocked}
                isMuted={audioMuted}
                volume={audioVolume}
                audioStatus={audioStatus}
                onToggleMute={toggleMute}
                onSetVolume={setAudioVolume}
                onUnlockAudio={unlockAudio}
                onTestAudioTier={testAlert}
                onSwitchToCockpit={handleSwitchToCockpit}
                chartHistory={chartHistory}
                devices={devices}
                selectedDeviceId={selectedDeviceId}
                onSelectDevice={selectDevice}
                onRefreshDevices={refreshDevices}
                viewOrientation={viewOrientation}
                onOrientationChange={setViewOrientation}
                hasActiveSession={hasActiveSession}
                browserRecordedFrames={browserRecordedFrames}
                onExportBrowserTelemetry={handleExportBrowserTelemetry}
              />
            )}
          </div>
        )}
      </div>

      {/* Snapshot Preview Dialog */}
      <CameraSnapshotPreview
        frame={capturedSnapshot}
        isOpen={isSnapshotOpen}
        onClose={handleCloseSnapshot}
      />
    </div>
  );
}
