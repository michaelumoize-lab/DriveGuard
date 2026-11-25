"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo, useId } from "react";
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
  CalibrationProgress,
  computeZoneThresholds,
  DriverBehaviourProfileData,
  PoseZone,
} from "@/lib/ai";
import {
  BackendProcessingStatus,
  CalibratedProfilePayload,
  CalibrationPhase,
  FrameTelemetry,
} from "@/types/telemetry";
import { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";
import {
  CalibrationJourneyStage,
  CalibrationStepper,
} from "./calibration-stepper";
import { CalibrationViewport } from "./calibration-viewport";
import { CalibrationGuidancePanel } from "./calibration-guidance-panel";
import { CalibrationProfileInspector } from "./calibration-profile-inspector";
import {
  formatZoneLabel,
  validateBackendProfilePayload,
} from "./calibration-zones";

export interface GuidedCalibrationWizardProps {
  driverId?: string;
  initialDriverProfile?: InitialDriverProfileData | null;
  initialPythonProfile?: InitialDriverProfileData | null;
  defaultEngine?: "browser" | "python";
  embeddedInOnboarding?: boolean;
  onCalibrationSaved?: (profile: InitialDriverProfileData) => void;
}

function browserProfileToInitialData(
  profile: DriverBehaviourProfileData
): InitialDriverProfileData {
  const mappedZones: Record<string, unknown> = {};
  for (const [zoneKey, zb] of Object.entries(profile.zoneBaselines || {})) {
    if (!zb || typeof zb !== "object") continue;
    const th = computeZoneThresholds(zb.earMedian, zb.earMadScaled);
    mappedZones[zoneKey] = {
      zone: zb.zone,
      ear_median: zb.earMedian,
      ear_mad_scaled: zb.earMadScaled,
      dynamic_closure_threshold: th.closureThreshold,
      dynamic_warning_threshold: th.reopenThreshold,
      sample_count: zb.sampleCount,
      pose_yaw_median: zb.yawMedianDeg,
      pose_pitch_median: zb.pitchMedianDeg,
    };
  }

  return {
    engineType: "BROWSER",
    isCalibrated: true,
    calibrationMode: profile.calibrationMode || "FOUR_ZONE",
    centerEarMedian: profile.centerEarMedian,
    centerEarMadScaled: profile.centerEarMadScaled,
    blinkDurationMedianMs: profile.blinkDurationMedianMs,
    blinkDurationMadMs: profile.blinkDurationMadMs,
    blinkRcrMedian: profile.blinkRcrMedian,
    blinkRcrMad: profile.blinkRcrMad,
    neutralPitchDeg: profile.neutralPitchDeg,
    neutralYawDeg: profile.neutralYawDeg,
    neutralRotMat: profile.neutralRotMat ?? null,
    calibrationIsMirrored: profile.calibrationIsMirrored ?? false,
    calibrationReliability: profile.calibrationReliability,
    profileVersion: profile.profileVersion,
    profileHash: profile.contentHash ?? null,
    zoneBaselines: mappedZones,
  };
}

function browserPhaseToUiCalibrationPhase(
  phase: CalibrationProgress | null
): CalibrationPhase | null {
  if (!phase) return null;
  return {
    current_zone: phase.currentZone,
    phase_index: phase.stepIndex,
    total_phases: phase.totalSteps,
    zone_progress_pct: phase.zoneProgressPct,
    total_progress_pct: phase.overallProgressPct,
    instruction: phase.guidanceText,
    is_completed: phase.isCompleted,
    rejection_reason: phase.rejectedReason ?? null,
    stable_samples_count: Math.round(phase.zoneProgressPct),
    target_samples_count: 100,
    is_head_in_target_zone: phase.inZone,
    stability_feedback: phase.guidanceText,
    quality_warnings: [],
  };
}

export function GuidedCalibrationWizard({
  driverId = "driver_default",
  initialDriverProfile = null,
  initialPythonProfile = null,
  defaultEngine = "browser",
  embeddedInOnboarding = false,
  onCalibrationSaved,
}: GuidedCalibrationWizardProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Default to Browser AI (local MediaPipe WASM — zero LiveKit/FastAPI required)
  const [calibrationEngine, setCalibrationEngine] = useState<"browser" | "python">(
    defaultEngine
  );
  const [isSwitchingEngine, setIsSwitchingEngine] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [pythonCalState, setPythonCalState] = useState<CalibrationPhase | null>(null);

  const [browserSavedProfile, setBrowserSavedProfile] =
    useState<InitialDriverProfileData | null>(
      initialDriverProfile && initialDriverProfile.engineType !== "PYTHON"
        ? initialDriverProfile
        : null
    );
  const [pythonSavedProfile, setPythonSavedProfile] =
    useState<InitialDriverProfileData | null>(
      initialPythonProfile ??
        (initialDriverProfile?.engineType === "PYTHON" ? initialDriverProfile : null)
    );

  const [latestPythonTelemetry, setLatestPythonTelemetry] =
    useState<FrameTelemetry | null>(null);
  const [hasPendingProfileRetry, setHasPendingProfileRetry] = useState(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState("");

  const hasHandledCompletionRef = useRef<boolean>(false);
  const pendingBackendProfileRef = useRef<CalibratedProfilePayload | null>(null);
  const lastAnnouncedZoneRef = useRef<string | null>(null);
  const pendingBrowserActionRef = useRef<
    { type: "full" } | { type: "zone"; zone: PoseZone } | null
  >(null);

  // Stable Session ID (deterministic across SSR/client hydration)
  const reactId = useId();
  const stableSessionId = useMemo(
    () => `cal_${reactId.replace(/[^a-zA-Z0-9_-]/g, "")}`,
    [reactId]
  );

  // 1. Unified Camera Hook
  const {
    videoRef,
    cameraState,
    error: cameraError,
    devices,
    selectedDeviceId,
    resolution,
    isMirrored,
    setIsMirrored,
    opticalQuality,
    startCamera,
    stopCamera,
    switchFacingMode,
    selectDevice,
    refreshDevices,
  } = useCamera({
    idealWidth: 640,
    idealHeight: 480,
    idealFps: 25,
    defaultFacingMode: "user",
    mirrored: true,
  });

  // 2A. On-Device Browser AI Detection & Calibration Hook (Default — zero server required)
  const {
    engineStatus: browserEngineStatus,
    telemetry: browserTelemetry,
    calibrationPhase: browserCalibrationPhase,
    isCalibrated: browserIsCalibrated,
    activeProfile: browserActiveProfile,
    profileSyncStatus,
    startMonitoring: startBrowserMonitoring,
    stopMonitoring: stopBrowserMonitoring,
    startCalibration: startBrowserCalibration,
    skipCalibrationStep: skipBrowserCalibrationStep,
    recalibrateZone: recalibrateBrowserZone,
    cancelCalibration: cancelBrowserCalibration,
    setOpticalQuality: setBrowserOpticalQuality,
  } = useBrowserDetection({
    driverId,
    initialProfile: (browserSavedProfile as unknown) as DriverBehaviourProfileData | null,
    isMirrored: false,
    targetInferenceFps: 15,
    autoStartSession: false,
    onCalibrationCompleted: (profile) => {
      const next = browserProfileToInitialData(profile);
      setBrowserSavedProfile(next);
      setIsSuccess(true);
      setSaveError(null);
      setLiveAnnouncement("Browser AI 4-zone calibration completed and saved locally.");
      toast.success("Browser AI 4-zone calibration saved locally & syncing.", {
        id: "browser-calibration-completed-toast",
      });
      if (onCalibrationSaved) {
        onCalibrationSaved(next);
      }
    },
    onCalibrationProgress: (progress) => {
      if (progress.isCompleted && progress.rejectedReason) {
        setIsSuccess(false);
        setLiveAnnouncement(
          "Calibration could not be completed. Check your lighting and retry."
        );
        toast.error(`Calibration rejected: ${progress.rejectedReason}`, {
          id: "browser-calibration-rejected-toast",
        });
      }
    },
  });

  // Hydrate browserSavedProfile if IndexedDB had a local profile when SSR did not
  useEffect(() => {
    if (browserIsCalibrated && browserActiveProfile && !browserSavedProfile?.isCalibrated) {
      setBrowserSavedProfile(browserProfileToInitialData(browserActiveProfile));
    }
  }, [browserIsCalibrated, browserActiveProfile, browserSavedProfile]);

  // Forward optical quality into BrowserDetectionEngine SignalConfidenceEvaluator
  useEffect(() => {
    setBrowserOpticalQuality(opticalQuality ?? null);
  }, [opticalQuality, setBrowserOpticalQuality]);

  // Persist authoritative Python backend-produced profile explicitly under engineType: "PYTHON"
  const persistBackendCalibratedProfile = useCallback(
    async (backendProfile: CalibratedProfilePayload) => {
      if (!validateBackendProfilePayload(backendProfile)) {
        const msg =
          "Backend completed calibration without returning a valid physiological profile.";
        setSaveError(msg);
        setIsSuccess(false);
        setHasPendingProfileRetry(false);
        pendingBackendProfileRef.current = null;
        toast.error(msg);
        return;
      }

      pendingBackendProfileRef.current = backendProfile;
      setHasPendingProfileRetry(true);
      setIsSavingProfile(true);
      setSaveError(null);

      const centerZone = backendProfile.zone_baselines?.CENTER;
      const centerEarMad =
        backendProfile.center_ear_mad_scaled ??
        centerZone?.ear_mad_scaled ??
        0.018;
      const freqMed = backendProfile.blink_freq_median_per_min ?? 16.0;
      const freqMad = backendProfile.blink_freq_mad_per_min ?? 4.5;
      const durMed = backendProfile.blink_duration_median_ms;
      const durMad = backendProfile.blink_duration_mad_ms;
      const rcrMed = backendProfile.blink_rcr_median;
      const rcrMad = backendProfile.blink_rcr_mad;
      const basePerclos = backendProfile.baseline_perclos_pct ?? 3.5;
      const ampMed = backendProfile.blink_amplitude_median ?? 0.14;
      const blinksSampled = backendProfile.temporal_blinks_sampled ?? 0;
      const windowSec = backendProfile.temporal_window_sec ?? 0;
      const modeLabel = backendProfile.calibration_mode || "FOUR_ZONE";

      const enrichedZones: Record<string, unknown> = {
        ...(backendProfile.zone_baselines ?? {}),
        _temporalBaseline: {
          blinkFreqMedianPerMin: freqMed,
          blinkFreqMadPerMin: freqMad,
          blinkDurationMedianMs: durMed,
          blinkDurationMadMs: durMad,
          blinkRcrMedian: rcrMed,
          blinkRcrMad: rcrMad,
          baselinePerclosPct: basePerclos,
          blinkAmplitudeMedian: ampMed,
          temporalBlinksSampled: blinksSampled,
          temporalWindowSec: windowSec,
          calibratedAt: backendProfile.calibrated_at ?? new Date().toISOString(),
        },
        _poseMetadata: {
          neutral_rot_mat: backendProfile.neutral_rot_mat ?? null,
          calibration_is_mirrored: backendProfile.calibration_is_mirrored ?? null,
        },
      };

      const nextProfile: InitialDriverProfileData = {
        engineType: "PYTHON",
        isCalibrated: true,
        calibrationMode: modeLabel,
        centerEarMedian: backendProfile.center_ear_median,
        centerEarMadScaled: centerEarMad,
        blinkDurationMedianMs: durMed,
        blinkDurationMadMs: durMad,
        blinkRcrMedian: rcrMed,
        blinkRcrMad: rcrMad,
        neutralPitchDeg: backendProfile.neutral_pitch_deg,
        neutralYawDeg: backendProfile.neutral_yaw_deg,
        neutralRotMat: backendProfile.neutral_rot_mat ?? null,
        calibrationIsMirrored: backendProfile.calibration_is_mirrored ?? null,
        calibrationReliability: backendProfile.calibration_reliability,
        zoneBaselines: enrichedZones,
      };

      try {
        const res = await fetch("/api/calibration/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            engineType: "PYTHON",
            calibrationMode: modeLabel,
            centerEarMedian: nextProfile.centerEarMedian,
            centerEarMadScaled: nextProfile.centerEarMadScaled,
            blinkDurationMedianMs: nextProfile.blinkDurationMedianMs,
            blinkDurationMadMs: nextProfile.blinkDurationMadMs,
            blinkRcrMedian: nextProfile.blinkRcrMedian,
            blinkRcrMad: nextProfile.blinkRcrMad,
            blinkFreqMedianPerMin: freqMed,
            blinkFreqMadPerMin: freqMad,
            baselinePerclosPct: basePerclos,
            blinkAmplitudeMedian: ampMed,
            temporalBlinksSampled: blinksSampled,
            temporalWindowSec: windowSec,
            sensitivityLevel: backendProfile.sensitivity_level ?? "STANDARD",
            neutralPitchDeg: nextProfile.neutralPitchDeg,
            neutralYawDeg: nextProfile.neutralYawDeg,
            neutralRotMat: backendProfile.neutral_rot_mat ?? null,
            calibrationIsMirrored: backendProfile.calibration_is_mirrored ?? null,
            calibrationReliability: nextProfile.calibrationReliability,
            zoneBaselines: backendProfile.zone_baselines ?? {},
          }),
        });

        if (!res.ok) {
          const errBody = await res.json().catch(() => ({}));
          throw new Error(
            errBody.error || `Profile persistence failed (HTTP ${res.status})`
          );
        }

        pendingBackendProfileRef.current = null;
        setHasPendingProfileRetry(false);
        setPythonSavedProfile(nextProfile);
        setIsSuccess(true);
        setLiveAnnouncement("Python AI 4-zone calibration completed and saved.");
        toast.success("Python AI 4-zone calibration profile saved.", {
          id: "calibration-completed-toast",
        });
        if (onCalibrationSaved) {
          onCalibrationSaved(nextProfile);
        }
      } catch (err: unknown) {
        const msg =
          err instanceof Error ? err.message : "Failed to save calibrated profile.";
        setSaveError(msg);
        setIsSuccess(false);
        setLiveAnnouncement(`Calibration save error: ${msg}`);
        toast.error(msg, { id: "calibration-save-error-toast" });
      } finally {
        setIsSavingProfile(false);
      }
    },
    [onCalibrationSaved]
  );

  // 2B. Real-Time LiveKit WebRTC Streaming & Python Backend Calibration Hook (Active ONLY when Python engine selected)
  const {
    isConnected: isLivekitConnected,
    connectionStatus: livekitConnectionStatus,
    backendProcessingStatus: livekitProcessingStatus,
    workerError: livekitWorkerError,
    isStale: livekitIsStale,
    startStreamingLoop,
    stopStreamingLoop,
    startCalibration: triggerLivekitCalibration,
    recalibrateZone: triggerLivekitRecalibrate,
    cancelCalibration: triggerLivekitCancel,
    skipCalibrationStep: triggerLivekitSkip,
    backendType,
    switchBackend,
  } = useLivekitTelemetryStream({
    sessionId: stableSessionId,
    driverId,
    onTelemetry: (t) => {
      setLatestPythonTelemetry(t);
    },
    onCalibration: (cal) => {
      setPythonCalState(cal);
      if (!cal.is_completed) {
        hasHandledCompletionRef.current = false;
        return;
      }
      if (hasHandledCompletionRef.current) {
        return;
      }
      hasHandledCompletionRef.current = true;

      if (cal.rejection_reason) {
        setIsSuccess(false);
        setLiveAnnouncement("Calibration could not be completed. Check your lighting and retry.");
        toast.error(`Calibration rejected: ${cal.rejection_reason}`, {
          id: "calibration-rejected-toast",
        });
        return;
      }

      if (!cal.profile) {
        const msg =
          "Backend marked calibration finished without a validated profile. Please retry.";
        setSaveError(msg);
        setIsSuccess(false);
        toast.error(msg);
        return;
      }

      void persistBackendCalibratedProfile(cal.profile);
    },
    onError: (err) => {
      if (calibrationEngine === "python" && backendType === "cloud") {
        toast.error(`Calibration connection error: ${err.message}`, {
          id: "calibration-connection-error",
        });
      }
    },
  });

  const isPythonLocalMode = calibrationEngine === "python" && backendType === "local";

  // 2C. Direct Local Python Camera & Calibration Hook (Zero getUserMedia / Zero LiveKit)
  const pythonLocal = usePythonLocalEngine({
    driverId,
    enabled: isPythonLocalMode,
    onCalibration: (cal) => {
      if (cal.is_completed && !cal.rejection_reason && cal.profile) {
        const p = cal.profile;
        const next: InitialDriverProfileData = {
          engineType: "PYTHON",
          isCalibrated: true,
          calibrationMode: p.calibration_mode || "FOUR_ZONE",
          centerEarMedian: p.center_ear_median,
          centerEarMadScaled:
            p.center_ear_mad_scaled ?? p.zone_baselines?.CENTER?.ear_mad_scaled ?? 0.018,
          blinkDurationMedianMs: p.blink_duration_median_ms,
          blinkDurationMadMs: p.blink_duration_mad_ms,
          blinkRcrMedian: p.blink_rcr_median,
          blinkRcrMad: p.blink_rcr_mad,
          neutralPitchDeg: p.neutral_pitch_deg,
          neutralYawDeg: p.neutral_yaw_deg,
          neutralRotMat: p.neutral_rot_mat ?? null,
          calibrationIsMirrored: p.calibration_is_mirrored ?? null,
          calibrationReliability: p.calibration_reliability,
          zoneBaselines: (p.zone_baselines as Record<string, unknown>) ?? {},
        };
        setPythonSavedProfile(next);
        setIsSuccess(true);
        setSaveError(null);
        setLiveAnnouncement("Local Python AI 4-zone calibration completed and saved.");
        toast.success("Python Local AI 4-zone calibration saved to local ProfileStore.", {
          id: "python-local-calibration-completed-toast",
        });
        if (onCalibrationSaved) {
          onCalibrationSaved(next);
        }
      } else if (cal.is_completed && cal.rejection_reason) {
        setIsSuccess(false);
        toast.error(`Calibration rejected: ${cal.rejection_reason}`, {
          id: "calibration-rejected-toast",
        });
      }
    },
  });

  // Reactive Engine Activation: link browser camera activation strictly to Browser AI or LiveKit Cloud.
  // Python Local mode acquires OpenCV camera directly — never uses browser getUserMedia!
  useEffect(() => {
    if (isPythonLocalMode) {
      stopBrowserMonitoring();
      stopStreamingLoop();
      return;
    }
    if (cameraState === "active" && videoRef.current) {
      if (calibrationEngine === "browser") {
        stopStreamingLoop();
        void startBrowserMonitoring(videoRef.current).then(() => {
          const pending = pendingBrowserActionRef.current;
          if (!pending) return;
          pendingBrowserActionRef.current = null;
          if (pending.type === "full") {
            startBrowserCalibration();
          } else {
            recalibrateBrowserZone(pending.zone);
          }
        });
      } else {
        stopBrowserMonitoring();
        startStreamingLoop(videoRef.current, isMirrored);
      }
    } else if (cameraState === "stopped") {
      stopBrowserMonitoring();
      stopStreamingLoop();
    }
  }, [
    cameraState,
    calibrationEngine,
    isPythonLocalMode,
    isMirrored,
    startBrowserMonitoring,
    stopBrowserMonitoring,
    startBrowserCalibration,
    recalibrateBrowserZone,
    startStreamingLoop,
    stopStreamingLoop,
    videoRef,
  ]);

  const effectiveCameraState = isPythonLocalMode
    ? pythonLocal.isStarting
      ? "permission-granted"
      : pythonLocal.isRunning
        ? "active"
        : "stopped"
    : cameraState;

  const effectiveCalState: CalibrationPhase | null = useMemo(() => {
    if (calibrationEngine === "browser") {
      return browserPhaseToUiCalibrationPhase(browserCalibrationPhase);
    }
    if (isPythonLocalMode) {
      return pythonLocal.calibration;
    }
    return pythonCalState;
  }, [calibrationEngine, isPythonLocalMode, browserCalibrationPhase, pythonLocal.calibration, pythonCalState]);

  const savedProfile =
    calibrationEngine === "browser" ? browserSavedProfile : pythonSavedProfile;

  const isCalibrating =
    effectiveCameraState === "active" &&
    (calibrationEngine === "browser"
      ? Boolean(browserCalibrationPhase && !browserCalibrationPhase.isCompleted)
      : isPythonLocalMode
        ? Boolean(pythonLocal.calibration && !pythonLocal.calibration.is_completed)
        : isLivekitConnected && Boolean(pythonCalState && !pythonCalState.is_completed));

  const effectiveConnectionStatus =
    calibrationEngine === "browser"
      ? browserEngineStatus === "running"
        ? "connected"
        : browserEngineStatus === "initializing"
          ? "connecting"
          : browserEngineStatus === "error"
            ? "error"
            : "disconnected"
      : isPythonLocalMode
        ? pythonLocal.connectionStatus
        : livekitConnectionStatus;

  const effectiveProcessingStatus: BackendProcessingStatus =
    calibrationEngine === "browser"
      ? browserEngineStatus === "running"
        ? "processing"
        : browserEngineStatus === "initializing"
          ? "connecting"
          : "idle"
      : isPythonLocalMode
        ? pythonLocal.backendProcessingStatus
        : livekitProcessingStatus;

  const effectiveWorkerError =
    calibrationEngine === "browser"
      ? browserEngineStatus === "error"
        ? "MediaPipe WASM failed to initialize in this browser."
        : null
      : isPythonLocalMode
        ? pythonLocal.error
        : livekitWorkerError;

  const isFaceVisible =
    calibrationEngine === "browser"
      ? browserTelemetry
        ? browserTelemetry.hasFace
        : null
      : isPythonLocalMode
        ? pythonLocal.telemetry
          ? pythonLocal.telemetry.face_detected
          : null
        : latestPythonTelemetry
          ? latestPythonTelemetry.face_detected
          : null;

  // Announce active zone transitions politely for screen readers
  useEffect(() => {
    if (!isCalibrating || !effectiveCalState?.current_zone) {
      lastAnnouncedZoneRef.current = null;
      return;
    }
    if (lastAnnouncedZoneRef.current !== effectiveCalState.current_zone) {
      lastAnnouncedZoneRef.current = effectiveCalState.current_zone;
      setLiveAnnouncement(
        `Zone ${effectiveCalState.phase_index} of ${effectiveCalState.total_phases}: ${formatZoneLabel(
          effectiveCalState.current_zone
        )}`
      );
    }
  }, [
    isCalibrating,
    effectiveCalState?.current_zone,
    effectiveCalState?.phase_index,
    effectiveCalState?.total_phases,
  ]);

  // Warn before accidentally leaving or closing tab mid-calibration
  useEffect(() => {
    if (!isCalibrating) return;
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      e.preventDefault();
      e.returnValue = "";
    };
    window.addEventListener("beforeunload", handleBeforeUnload);
    return () => window.removeEventListener("beforeunload", handleBeforeUnload);
  }, [isCalibrating]);

  // Fullscreen Handler
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

  // Step 1: Turn on camera for positioning & framing pre-check
  const handleStartCameraCheck = async () => {
    if (isSwitchingEngine) return;
    hasHandledCompletionRef.current = false;
    pendingBackendProfileRef.current = null;
    pendingBrowserActionRef.current = null;
    setHasPendingProfileRetry(false);
    setIsSuccess(false);
    setSaveError(null);
    setPythonCalState(null);
    if (isPythonLocalMode) {
      const ok = await pythonLocal.start({ isMirrored: true, enableAudio: false });
      if (!ok) {
        toast.error(
          pythonLocal.error ||
            "Could not start Local Python camera on 127.0.0.1:8000."
        );
      }
      return;
    }
    await startCamera();
  };

  // Step 2: Start 4-Zone spatial calibration on the selected engine
  const handleBeginCalibration = async () => {
    if (isSwitchingEngine) return;
    hasHandledCompletionRef.current = false;
    pendingBackendProfileRef.current = null;
    setHasPendingProfileRetry(false);
    setIsSuccess(false);
    setSaveError(null);
    setPythonCalState(null);

    if (calibrationEngine === "browser") {
      if (cameraState !== "active") {
        pendingBrowserActionRef.current = { type: "full" };
        await startCamera();
      } else {
        startBrowserCalibration();
      }
      toast.info("Starting Browser AI 4-Zone Calibration (~15s)…");
      return;
    }

    if (isPythonLocalMode) {
      if (!pythonLocal.isRunning) {
        const ok = await pythonLocal.start({
          isMirrored: true,
          enableAudio: false,
          startCalibrating: true,
        });
        if (!ok) {
          toast.error(
            pythonLocal.error ||
              "Could not start Local Python calibration on 127.0.0.1:8000."
          );
          return;
        }
      } else {
        await pythonLocal.startCalibration(true, 4.0);
      }
      toast.info("Starting Python Local AI 4-Zone Calibration (~15s)…");
      return;
    }

    if (cameraState !== "active") {
      await startCamera();
    }

    triggerLivekitCalibration(true, 4.0, false, 0);
    toast.info("Starting LiveKit Cloud Python 4-Zone Calibration (~15s)…");
  };

  // Cancel active calibration while keeping camera preview active for easy retry
  const handleCancelCalibration = () => {
    pendingBrowserActionRef.current = null;
    if (calibrationEngine === "browser") {
      cancelBrowserCalibration();
    } else if (isPythonLocalMode) {
      void pythonLocal.cancelCalibration();
    } else {
      triggerLivekitCancel();
      setPythonCalState(null);
    }
    setSaveError(null);
    setLiveAnnouncement("Calibration cancelled.");
  };

  // Explicit camera shutdown
  const handleTurnOffCamera = () => {
    pendingBrowserActionRef.current = null;
    if (isPythonLocalMode) {
      void pythonLocal.stop();
      setSaveError(null);
      return;
    }
    if (isCalibrating) {
      if (calibrationEngine === "browser") {
        cancelBrowserCalibration();
      } else {
        triggerLivekitCancel();
      }
    }
    stopCamera();
    stopBrowserMonitoring();
    stopStreamingLoop();
    setPythonCalState(null);
    setSaveError(null);
    setLatestPythonTelemetry(null);
  };

  const handleSkipZone = () => {
    if (calibrationEngine === "browser") {
      skipBrowserCalibrationStep();
    } else if (isPythonLocalMode) {
      void pythonLocal.skipCalibrationStep();
    } else {
      triggerLivekitSkip();
    }
    toast.info("Advancing past current calibration zone…");
  };

  const handleRecalibrateZone = async (zone: string) => {
    if (isSwitchingEngine) return;
    hasHandledCompletionRef.current = false;
    pendingBackendProfileRef.current = null;
    setHasPendingProfileRetry(false);
    setIsSuccess(false);
    setSaveError(null);
    setPythonCalState(null);

    if (calibrationEngine === "browser") {
      const target = zone as PoseZone;
      if (cameraState !== "active") {
        pendingBrowserActionRef.current = { type: "zone", zone: target };
        await startCamera();
      } else {
        recalibrateBrowserZone(target);
      }
      toast.info(`Recalibrating ${formatZoneLabel(zone)} in Browser AI…`);
      return;
    }

    if (isPythonLocalMode) {
      if (!pythonLocal.isRunning) {
        const ok = await pythonLocal.start({ isMirrored: true, enableAudio: false });
        if (!ok) {
          toast.error(pythonLocal.error || "Could not reach Local Python camera.");
          return;
        }
      }
      await pythonLocal.recalibrateZone(zone, 4.0);
      toast.info(`Recalibrating ${formatZoneLabel(zone)} in Python Local AI…`);
      return;
    }

    if (cameraState !== "active") {
      await startCamera();
    }
    triggerLivekitRecalibrate(zone, 4.0, 0);
    toast.info(`Recalibrating ${formatZoneLabel(zone)}…`);
  };

  const handleRetrySave = () => {
    if (pendingBackendProfileRef.current) {
      void persistBackendCalibratedProfile(pendingBackendProfileRef.current);
    }
  };

  const handleSelectEngine = async (
    engine: "browser" | "python",
    pythonTarget?: "cloud" | "local"
  ) => {
    if (isSwitchingEngine) return;
    const nextBackend = pythonTarget ?? backendType;
    if (engine === calibrationEngine && (engine === "browser" || nextBackend === backendType)) {
      return;
    }

    setIsSwitchingEngine(true);
    pendingBrowserActionRef.current = null;

    const currentId =
      calibrationEngine === "browser"
        ? "BROWSER"
        : backendType === "local"
          ? "PYTHON_LOCAL"
          : "PYTHON_REMOTE";
    const targetId =
      engine === "browser"
        ? "BROWSER"
        : nextBackend === "local"
          ? "PYTHON_LOCAL"
          : "PYTHON_REMOTE";

    try {
      await executeTwoPhaseCameraHandoff(
        {
          engineId: currentId,
          stop: async () => {
            if (isCalibrating) {
              if (calibrationEngine === "browser") cancelBrowserCalibration();
              else if (isPythonLocalMode) await pythonLocal.cancelCalibration();
              else triggerLivekitCancel();
            }
            if (calibrationEngine === "browser") stopBrowserMonitoring();
            else if (isPythonLocalMode) await pythonLocal.stop();
            else stopStreamingLoop();
            return true;
          },
          releaseCamera: async () => {
            if (isPythonLocalMode) {
              return await pythonLocal.stop();
            }
            const currentStream = (videoRef.current?.srcObject as MediaStream | null) ?? null;
            stopCamera();
            return stopAndVerifyBrowserMediaStream(currentStream);
          },
        },
        {
          engineId: targetId,
          start: async () => true,
        },
        { settleDelayMs: 200, autoStartTarget: false }
      );

      setPythonCalState(null);
      setSaveError(null);
      setIsSuccess(false);

      if (engine === "browser") {
        setCalibrationEngine("browser");
        toast.info("Switched to Browser AI (On-Device MediaPipe — zero server required).");
      } else {
        setCalibrationEngine("python");
        if (pythonTarget && pythonTarget !== backendType) {
          switchBackend(pythonTarget);
        }
        toast.info(
          nextBackend === "local"
            ? "Switched to Python Local AI (127.0.0.1:8000 direct OpenCV camera — zero LiveKit)."
            : "Switched to LiveKit Cloud Python AI Worker."
        );
      }
    } finally {
      setIsSwitchingEngine(false);
    }
  };

  const journeyStage: CalibrationJourneyStage = isSuccess
    ? "complete"
    : isCalibrating
      ? "calibrating"
      : effectiveCameraState === "active"
        ? "positioning"
        : "overview";

  return (
    <div
      ref={containerRef}
      className={`space-y-5 transition-colors ${
        isFullscreen
          ? "fixed inset-0 z-50 bg-black p-4 flex flex-col justify-between overflow-hidden"
          : ""
      }`}
    >
      {/* Centralized polite screen-reader announcement region */}
      <div aria-live="polite" aria-atomic="true" className="sr-only">
        {liveAnnouncement}
      </div>

      {/* 3-Step Workflow + 4-Zone Progress Strip */}
      {!isFullscreen && (
        <CalibrationStepper
          journeyStage={journeyStage}
          currentZone={effectiveCalState?.current_zone}
          isCalibrating={isCalibrating}
          isSuccess={isSuccess}
        />
      )}

      {/* Primary Interactive Workspace */}
      <div
        className={`grid gap-5 ${
          isFullscreen ? "h-full grid-cols-1" : "grid-cols-1 lg:grid-cols-12"
        }`}
      >
        <CalibrationViewport
          videoRef={videoRef}
          cameraState={effectiveCameraState}
          previewMode={isPythonLocalMode ? "mjpeg" : "video"}
          mjpegUrl={isPythonLocalMode ? pythonLocal.mjpegPreviewUrl : null}
          cameraError={cameraError}
          devices={devices}
          selectedDeviceId={selectedDeviceId}
          isMirrored={isMirrored}
          resolution={resolution}
          opticalQuality={opticalQuality}
          calibrationEngine={calibrationEngine}
          backendType={backendType}
          onSelectEngine={(eng, target) => void handleSelectEngine(eng, target)}
          connectionStatus={effectiveConnectionStatus}
          backendProcessingStatus={effectiveProcessingStatus}
          workerError={effectiveWorkerError}
          isStale={calibrationEngine === "browser" || isPythonLocalMode ? false : livekitIsStale}
          calState={effectiveCalState}
          isCalibrating={isCalibrating}
          isSuccess={isSuccess}
          isSavingProfile={isSavingProfile}
          isFullscreen={isFullscreen}
          embeddedInOnboarding={embeddedInOnboarding}
          isFaceVisible={isFaceVisible}
          onStartCameraCheck={() => void handleStartCameraCheck()}
          onRetryCamera={() => void handleStartCameraCheck()}
          onStopCamera={handleTurnOffCamera}
          onSelectDevice={selectDevice}
          onRefreshDevices={() => void refreshDevices()}
          onSwitchFacing={switchFacingMode}
          onToggleMirror={() => setIsMirrored((prev) => !prev)}
          onToggleFullscreen={() => void toggleFullscreen()}
          onSkipZone={handleSkipZone}
          onStopCalibration={handleCancelCalibration}
        />

        {!isFullscreen && (
          <CalibrationGuidancePanel
            cameraState={effectiveCameraState}
            resolution={resolution}
            opticalQuality={opticalQuality}
            calibrationEngine={calibrationEngine}
            profileSyncStatus={isPythonLocalMode ? pythonLocal.profileSyncStatus : profileSyncStatus}
            isLivekitConnected={
              calibrationEngine === "browser"
                ? browserEngineStatus === "running"
                : isPythonLocalMode
                  ? pythonLocal.connectionStatus === "connected"
                  : isLivekitConnected
            }
            connectionStatus={effectiveConnectionStatus}
            backendProcessingStatus={effectiveProcessingStatus}
            backendType={backendType}
            workerError={effectiveWorkerError}
            saveError={saveError}
            hasPendingProfileRetry={hasPendingProfileRetry}
            isCalibrating={isCalibrating}
            isSavingProfile={isSavingProfile}
            isSuccess={isSuccess}
            calState={effectiveCalState}
            savedProfile={savedProfile}
            isFaceVisible={isFaceVisible}
            embeddedInOnboarding={embeddedInOnboarding}
            onStartCameraCheck={() => void handleStartCameraCheck()}
            onBeginCalibration={() => void handleBeginCalibration()}
            onCancelCalibration={handleCancelCalibration}
            onSkipZone={handleSkipZone}
            onRetrySave={handleRetrySave}
            onSwitchToCloud={() => void handleSelectEngine("python", "cloud")}
          />
        )}
      </div>

      {/* Saved 4-Zone Baseline Overview + Collapsible Technical Details */}
      {!isFullscreen && (
        <CalibrationProfileInspector
          savedProfile={savedProfile}
          isCalibrating={isCalibrating}
          isSavingProfile={isSavingProfile}
          onRecalibrateZone={(zone) => void handleRecalibrateZone(zone)}
        />
      )}
    </div>
  );
}
