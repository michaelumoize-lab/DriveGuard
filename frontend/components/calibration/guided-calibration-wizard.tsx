"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo, useId } from "react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useLivekitTelemetryStream } from "@/hooks/use-livekit-telemetry-stream";
import {
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
  embeddedInOnboarding?: boolean;
  onCalibrationSaved?: (profile: InitialDriverProfileData) => void;
}

export function GuidedCalibrationWizard({
  driverId = "driver_default",
  initialDriverProfile = null,
  embeddedInOnboarding = false,
  onCalibrationSaved,
}: GuidedCalibrationWizardProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  const [isFullscreen, setIsFullscreen] = useState(false);
  const [isSuccess, setIsSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [calState, setCalState] = useState<CalibrationPhase | null>(null);
  const [savedProfile, setSavedProfile] = useState<InitialDriverProfileData | null>(
    initialDriverProfile
  );
  const [latestTelemetry, setLatestTelemetry] = useState<FrameTelemetry | null>(null);
  const [hasPendingProfileRetry, setHasPendingProfileRetry] = useState(false);
  const [liveAnnouncement, setLiveAnnouncement] = useState("");

  const hasHandledCompletionRef = useRef<boolean>(false);
  const pendingBackendProfileRef = useRef<CalibratedProfilePayload | null>(null);
  const lastAnnouncedZoneRef = useRef<string | null>(null);

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

  // Persist authoritative backend-produced profile only (never fabricate fallback baselines)
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
        calibrationReliability: backendProfile.calibration_reliability,
        zoneBaselines: enrichedZones,
      };

      try {
        const res = await fetch("/api/calibration/save", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
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
        setSavedProfile(nextProfile);
        setIsSuccess(true);
        setLiveAnnouncement("4-zone calibration completed and saved.");
        toast.success("4-zone calibration profile saved.", {
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

  // 2. Real-Time LiveKit WebRTC Streaming & Authoritative Backend Calibration Hook
  const {
    isConnected: isLivekitConnected,
    connectionStatus,
    backendProcessingStatus,
    workerError,
    isStale,
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
      setLatestTelemetry(t);
    },
    onCalibration: (cal) => {
      setCalState(cal);
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
      toast.error(`Calibration connection error: ${err.message}`, {
        id: "calibration-connection-error",
      });
    },
  });

  // Reactive Video Streaming: Link camera activation with LiveKit WebRTC publisher
  useEffect(() => {
    if (cameraState === "active" && videoRef.current) {
      startStreamingLoop(videoRef.current, isMirrored);
    } else if (cameraState === "stopped") {
      stopStreamingLoop();
    }
  }, [cameraState, isMirrored, startStreamingLoop, stopStreamingLoop, videoRef]);

  const isCalibrating =
    cameraState === "active" &&
    isLivekitConnected &&
    Boolean(calState && !calState.is_completed);

  // Announce active zone transitions politely for screen readers
  useEffect(() => {
    if (!isCalibrating || !calState?.current_zone) {
      lastAnnouncedZoneRef.current = null;
      return;
    }
    if (lastAnnouncedZoneRef.current !== calState.current_zone) {
      lastAnnouncedZoneRef.current = calState.current_zone;
      setLiveAnnouncement(
        `Zone ${calState.phase_index} of ${calState.total_phases}: ${formatZoneLabel(
          calState.current_zone
        )}`
      );
    }
  }, [isCalibrating, calState?.current_zone, calState?.phase_index, calState?.total_phases]);

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
    hasHandledCompletionRef.current = false;
    pendingBackendProfileRef.current = null;
    setHasPendingProfileRetry(false);
    setIsSuccess(false);
    setSaveError(null);
    setCalState(null);
    await startCamera();
  };

  // Step 2: Start 4-Zone spatial calibration (existing backend workflow untouched)
  const handleBeginCalibration = async () => {
    hasHandledCompletionRef.current = false;
    pendingBackendProfileRef.current = null;
    setHasPendingProfileRetry(false);
    setIsSuccess(false);
    setSaveError(null);
    setCalState(null);

    if (cameraState !== "active") {
      await startCamera();
    }

    triggerLivekitCalibration(true, 4.0, false, 0);
    toast.info("Starting 4-Zone Calibration (~15s)…");
  };

  // Cancel active calibration while keeping camera preview active for easy retry
  const handleCancelCalibration = () => {
    triggerLivekitCancel();
    setCalState(null);
    setSaveError(null);
    setLiveAnnouncement("Calibration cancelled.");
  };

  // Explicit camera shutdown
  const handleTurnOffCamera = () => {
    if (isCalibrating) {
      triggerLivekitCancel();
    }
    stopCamera();
    stopStreamingLoop();
    setCalState(null);
    setSaveError(null);
    setLatestTelemetry(null);
  };

  const handleSkipZone = () => {
    triggerLivekitSkip();
    toast.info("Advancing past current calibration zone…");
  };

  const handleRecalibrateZone = async (zone: string) => {
    hasHandledCompletionRef.current = false;
    pendingBackendProfileRef.current = null;
    setHasPendingProfileRetry(false);
    setIsSuccess(false);
    setSaveError(null);
    setCalState(null);
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

  const handleSwitchBackend = (target: "cloud" | "local") => {
    if (backendType === target) return;
    if (isCalibrating) {
      triggerLivekitCancel();
    }
    setCalState(null);
    setSaveError(null);
    switchBackend(target);
    toast.info(
      target === "local"
        ? "Switched AI Engine to Localhost (127.0.0.1:8000)."
        : "Switched AI Engine to LiveKit Cloud."
    );
  };

  const isFaceVisible = latestTelemetry ? latestTelemetry.face_detected : null;

  const journeyStage: CalibrationJourneyStage = isSuccess
    ? "complete"
    : isCalibrating
      ? "calibrating"
      : cameraState === "active"
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
          currentZone={calState?.current_zone}
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
          cameraState={cameraState}
          cameraError={cameraError}
          devices={devices}
          selectedDeviceId={selectedDeviceId}
          isMirrored={isMirrored}
          resolution={resolution}
          opticalQuality={opticalQuality}
          backendType={backendType}
          onSwitchBackend={handleSwitchBackend}
          connectionStatus={connectionStatus}
          backendProcessingStatus={backendProcessingStatus}
          workerError={workerError}
          isStale={isStale}
          calState={calState}
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
            cameraState={cameraState}
            resolution={resolution}
            opticalQuality={opticalQuality}
            isLivekitConnected={isLivekitConnected}
            connectionStatus={connectionStatus}
            backendProcessingStatus={backendProcessingStatus}
            backendType={backendType}
            workerError={workerError}
            saveError={saveError}
            hasPendingProfileRetry={hasPendingProfileRetry}
            isCalibrating={isCalibrating}
            isSavingProfile={isSavingProfile}
            isSuccess={isSuccess}
            calState={calState}
            savedProfile={savedProfile}
            isFaceVisible={isFaceVisible}
            embeddedInOnboarding={embeddedInOnboarding}
            onStartCameraCheck={() => void handleStartCameraCheck()}
            onBeginCalibration={() => void handleBeginCalibration()}
            onCancelCalibration={handleCancelCalibration}
            onSkipZone={handleSkipZone}
            onRetrySave={handleRetrySave}
            onSwitchToCloud={() => handleSwitchBackend("cloud")}
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
