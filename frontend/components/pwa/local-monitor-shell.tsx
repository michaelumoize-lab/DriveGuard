"use client";

import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Link from "next/link";
import {
  ArrowLeft,
  Compass,
  Download,
  EyeOff,
  FastForward,
  Flame,
  Minimize2,
  Plus,
  ShieldAlert,
  ShieldCheck,
  Square,
  Trash2,
  UserCheck,
  Wifi,
  WifiOff,
  AlertTriangle,
} from "lucide-react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useBrowserDetection } from "@/hooks/use-browser-detection";
import { useDriveGuardAudio } from "@/hooks/use-driveguard-audio";
import {
  AlertTransition,
  convertBrowserTelemetryToFrameTelemetry,
  DecisionResult,
  DetectionTelemetry,
  DrowsinessState,
} from "@/lib/ai";
import { resolveEffectiveDisplayState } from "@/types/telemetry";
import { CapturedFrame } from "@/types/camera";
import {
  clearAndRemoveLocalSlot,
  createLocalDriverSlot,
  DEFAULT_LOCAL_SLOT_ID,
  getActiveLocalSlotId,
  listLocalDriverSlots,
  LocalDriverSlot,
  localOnlyProfileStore,
  setActiveLocalSlotId,
} from "@/lib/pwa/local-profile-slots";
import { CameraView } from "@/components/camera/camera-view";
import { CameraControls } from "@/components/camera/camera-controls";
import { CameraStatus } from "@/components/camera/camera-status";
import { CameraErrorDisplay } from "@/components/camera/camera-error";
import { CameraSnapshotPreview } from "@/components/camera/camera-snapshot-preview";
import { DriverCockpitHUD } from "@/components/monitor/driver-cockpit-hud";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Switch } from "@/components/ui/switch";

const SENSITIVITY_OPTIONS = ["RELAXED", "STANDARD", "HIGH"] as const;
type LocalSensitivity = (typeof SENSITIVITY_OPTIONS)[number];

export function LocalMonitorShell() {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [slots, setSlots] = useState<LocalDriverSlot[]>([
    {
      id: DEFAULT_LOCAL_SLOT_ID,
      label: "Local Driver 1",
      createdAt: "2026-01-01T00:00:00.000Z",
    },
  ]);
  const [activeSlotId, setActiveSlotIdState] = useState<string>(DEFAULT_LOCAL_SLOT_ID);
  const [newSlotName, setNewSlotName] = useState<string>("");
  const [showNewSlotInput, setShowNewSlotInput] = useState<boolean>(false);

  const [sensitivityLevel, setSensitivityLevel] = useState<LocalSensitivity>("STANDARD");
  const [mirrorSuppression, setMirrorSuppression] = useState<boolean>(true);
  const [isOnline, setIsOnline] = useState<boolean>(true);
  const [backgroundSuspended, setBackgroundSuspended] = useState<boolean>(false);

  const [viewOrientation, setViewOrientation] = useState<"portrait" | "landscape">("portrait");
  const [framingMode, setFramingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState<boolean>(false);
  const [showCalibrationBar, setShowCalibrationBar] = useState<boolean>(false);
  const [srAnnouncement, setSrAnnouncement] = useState<string>("");
  const [capturedSnapshot, setCapturedSnapshot] = useState<CapturedFrame | null>(null);
  const [isSnapshotOpen, setIsSnapshotOpen] = useState<boolean>(false);
  const [isCapturing, setIsCapturing] = useState<boolean>(false);

  const isMonitoringAudioAllowedRef = useRef<boolean>(false);
  const lastAnnouncedStateRef = useRef<string>("NORMAL");

  // Hydrate client-only state safely post-mount
  useEffect(() => {
    const syncOnline = () => setIsOnline(navigator.onLine);
    queueMicrotask(() => {
      const loadedSlots = listLocalDriverSlots();
      setSlots(loadedSlots);
      setActiveSlotIdState(getActiveLocalSlotId(loadedSlots));
      syncOnline();
      if (typeof window !== "undefined" && window.innerWidth >= 1024) {
        setViewOrientation("landscape");
      }
    });

    window.addEventListener("online", syncOnline);
    window.addEventListener("offline", syncOnline);
    return () => {
      window.removeEventListener("online", syncOnline);
      window.removeEventListener("offline", syncOnline);
    };
  }, []);

  const {
    videoRef,
    cameraState,
    error: cameraError,
    devices,
    facingMode,
    resolution,
    isMirrored,
    setIsMirrored,
    streamDurationSec,
    opticalQuality,
    startCamera,
    stopCamera,
    pauseCamera,
    resumeCamera,
    switchFacingMode,
    captureSnapshot,
  } = useCamera({
    autoStart: false,
    defaultFacingMode: "user",
    mirrored: true,
    idealWidth: 640,
    idealHeight: 480,
    idealFps: 25,
  });

  const {
    isAudioUnlocked,
    isMuted: audioMuted,
    volume: audioVolume,
    audioStatus,
    toggleMute,
    setVolume: setAudioVolume,
    unlockAudio,
    engine: audioEngine,
  } = useDriveGuardAudio({
    autoUnlockOnGesture: true,
    initialMuted: false,
  });

  const audioEngineRef = useRef(audioEngine);
  useEffect(() => {
    audioEngineRef.current = audioEngine;
  }, [audioEngine]);

  const handleBrowserSafetyAlert = useCallback(
    (decision: DecisionResult, t: DetectionTelemetry, transition?: AlertTransition) => {
      const isCritical = decision.state === DrowsinessState.CRITICAL_WARNING;
      const isWarning = decision.state === DrowsinessState.DROWSINESS_WARNING;
      if (!isCritical && !isWarning) return;

      const alertState = isCritical ? "CRITICAL" : "WARNING";
      const eventId =
        transition?.eventId ??
        `evt_local_${Date.now()}_${Math.random().toString(36).slice(2, 6)}`;
      const eventTsMs = t.timestampMs || Date.now();

      if (isMonitoringAudioAllowedRef.current) {
        audioEngine?.setRiskState(alertState, eventId, eventTsMs);
      }

      if (alertState !== lastAnnouncedStateRef.current) {
        lastAnnouncedStateRef.current = alertState;
        setSrAnnouncement(
          isCritical
            ? "Critical drowsiness alert! Pull over safely when possible."
            : "Drowsiness warning: elevated eyelid closure detected."
        );
      }
    },
    [audioEngine]
  );

  const {
    engineStatus: browserEngineStatus,
    telemetry: browserTelemetry,
    calibration: browserCalibrationPhase,
    isCalibrated,
    activeProfile,
    storageMode,
    startMonitoring: startBrowserMonitoring,
    pauseMonitoring: pauseBrowserMonitoring,
    resumeMonitoring: resumeBrowserMonitoring,
    stopMonitoring: stopBrowserMonitoring,
    startCalibration: startBrowserCalibration,
    skipCalibrationStep: skipBrowserCalibrationStep,
    cancelCalibration: cancelBrowserCalibration,
    setOpticalQuality: pushBrowserOpticalQuality,
    recordedFrameCount: browserRecordedFrames,
    exportTelemetryJsonl,
  } = useBrowserDetection({
    driverId: activeSlotId,
    profileStore: localOnlyProfileStore,
    offlineOnly: true,
    autoStartSession: false,
    isMirrored: false,
    sensitivityLevel,
    mirrorSuppression,
    targetInferenceFps: 12,
    onSafetyAlert: handleBrowserSafetyAlert,
    onCalibrationCompleted: (profile) => {
      toast.success(
        `Saved 4-zone local calibration (EAR ${profile.centerEarMedian.toFixed(3)}) for ${
          slots.find((s) => s.id === activeSlotId)?.label ?? "local driver"
        }.`
      );
      setShowCalibrationBar(false);
    },
    onError: (err) => {
      toast.error(err.message || "Local Browser AI initialization error.");
    },
  });

  // Sync optical quality diagnostics into Browser AI confidence gate
  useEffect(() => {
    if (cameraState === "active") {
      pushBrowserOpticalQuality(opticalQuality);
    }
  }, [cameraState, opticalQuality, pushBrowserOpticalQuality]);

  // Convert telemetry for DriverCockpitHUD
  const convertedTelemetry = useMemo(() => {
    if (!browserTelemetry || (cameraState !== "active" && cameraState !== "paused")) {
      return null;
    }
    return convertBrowserTelemetryToFrameTelemetry(
      browserTelemetry,
      "local_offline_session",
      activeSlotId
    );
  }, [browserTelemetry, cameraState, activeSlotId]);

  // Keep continuous audio state synchronized ONLY while monitoring is active
  useEffect(() => {
    if (
      convertedTelemetry?.state &&
      isMonitoringAudioAllowedRef.current &&
      cameraState === "active"
    ) {
      audioEngine?.setRiskState(
        convertedTelemetry.state,
        undefined,
        convertedTelemetry.timestamp_ms
      );
    }
  }, [convertedTelemetry, cameraState, audioEngine]);

  // Pause camera & silence repeating audio cleanly if app is backgrounded or screen locked
  useEffect(() => {
    const handleVisibilityChange = () => {
      if (document.hidden && cameraState === "active") {
        isMonitoringAudioAllowedRef.current = false;
        audioEngineRef.current?.resetSilent();
        pauseBrowserMonitoring();
        pauseCamera();
        setBackgroundSuspended(true);
      }
    };
    document.addEventListener("visibilitychange", handleVisibilityChange);
    return () => {
      document.removeEventListener("visibilitychange", handleVisibilityChange);
    };
  }, [cameraState, pauseBrowserMonitoring, pauseCamera]);

  // Fullscreen listener
  useEffect(() => {
    const onFsChange = () => {
      setIsFullscreen(Boolean(document.fullscreenElement));
    };
    document.addEventListener("fullscreenchange", onFsChange);
    return () => document.removeEventListener("fullscreenchange", onFsChange);
  }, []);

  // Silent audio & engine teardown on unmount
  useEffect(() => {
    return () => {
      isMonitoringAudioAllowedRef.current = false;
      audioEngineRef.current?.resetSilent();
    };
  }, []);

  const handleStartMonitoring = useCallback(async () => {
    setBackgroundSuspended(false);
    void unlockAudio();

    if (cameraState !== "active") {
      await startCamera();
    }

    if (!videoRef.current || !videoRef.current.srcObject) {
      isMonitoringAudioAllowedRef.current = false;
      audioEngineRef.current?.resetSilent();
      return;
    }

    isMonitoringAudioAllowedRef.current = true;
    await startBrowserMonitoring(videoRef.current);
  }, [cameraState, startCamera, startBrowserMonitoring, unlockAudio, videoRef]);

  const handleStopMonitoring = useCallback(() => {
    isMonitoringAudioAllowedRef.current = false;
    audioEngineRef.current?.resetSilent();
    stopBrowserMonitoring();
    stopCamera();
    setBackgroundSuspended(false);
  }, [stopBrowserMonitoring, stopCamera]);

  const handlePauseCamera = useCallback(() => {
    isMonitoringAudioAllowedRef.current = false;
    audioEngineRef.current?.resetSilent();
    pauseBrowserMonitoring();
    pauseCamera();
  }, [pauseBrowserMonitoring, pauseCamera]);

  const handleResumeCamera = useCallback(() => {
    setBackgroundSuspended(false);
    isMonitoringAudioAllowedRef.current = true;
    resumeBrowserMonitoring();
    resumeCamera();
  }, [resumeBrowserMonitoring, resumeCamera]);

  const handleSelectSlot = useCallback(
    (nextSlotId: string) => {
      if (cameraState === "active" || cameraState === "paused") {
        handleStopMonitoring();
      }
      setActiveLocalSlotId(nextSlotId);
      setActiveSlotIdState(nextSlotId);
      toast.info(
        `Switched local profile slot to ${
          slots.find((s) => s.id === nextSlotId)?.label ?? "driver"
        }.`
      );
    },
    [cameraState, handleStopMonitoring, slots]
  );

  const handleCreateSlot = useCallback(
    (e: React.FormEvent) => {
      e.preventDefault();
      if (cameraState === "active" || cameraState === "paused") {
        handleStopMonitoring();
      }
      const { slots: updated, created } = createLocalDriverSlot(newSlotName);
      setSlots(updated);
      setActiveSlotIdState(created.id);
      setNewSlotName("");
      setShowNewSlotInput(false);
      toast.success(`Created isolated local profile "${created.label}".`);
    },
    [cameraState, handleStopMonitoring, newSlotName]
  );

  const handleClearActiveSlot = useCallback(async () => {
    if (cameraState === "active" || cameraState === "paused") {
      handleStopMonitoring();
    }
    const label = slots.find((s) => s.id === activeSlotId)?.label ?? "current slot";
    const { slots: nextSlots, activeId } = await clearAndRemoveLocalSlot(activeSlotId);
    setSlots(nextSlots);
    setActiveSlotIdState(activeId);
    toast.success(`Cleared calibration data for "${label}".`);
  }, [activeSlotId, cameraState, handleStopMonitoring, slots]);

  const handleExportTelemetry = useCallback(() => {
    const jsonl = exportTelemetryJsonl();
    if (!jsonl) {
      toast.error("No local telemetry frames recorded yet.");
      return;
    }
    const blob = new Blob([jsonl], { type: "application/x-ndjson" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `driveguard_local_telemetry_${Date.now()}.jsonl`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
    toast.success(`Exported ${browserRecordedFrames} local frames (.jsonl).`);
  }, [browserRecordedFrames, exportTelemetryJsonl]);

  const handleCaptureSnapshot = useCallback(async () => {
    if (cameraState !== "active") return;
    setIsCapturing(true);
    try {
      const frame = await captureSnapshot({
        format: "image/jpeg",
        quality: 0.92,
      });
      if (frame) {
        setCapturedSnapshot(frame);
        setIsSnapshotOpen(true);
      }
    } finally {
      setIsCapturing(false);
    }
  }, [cameraState, captureSnapshot]);

  const toggleFullscreen = useCallback(async () => {
    try {
      if (!document.fullscreenElement) {
        if (containerRef.current?.requestFullscreen) {
          await containerRef.current.requestFullscreen();
        } else {
          setIsFullscreen(true);
        }
      } else {
        await document.exitFullscreen();
      }
    } catch {
      setIsFullscreen((prev) => !prev);
    }
  }, []);

  const handleStartCalibration = useCallback(() => {
    const begin = () => {
      setShowCalibrationBar(true);
      startBrowserCalibration();
    };
    if (cameraState !== "active") {
      void handleStartMonitoring().then(begin);
    } else {
      begin();
    }
  }, [cameraState, handleStartMonitoring, startBrowserCalibration]);

  const effectiveDisplayState = resolveEffectiveDisplayState(convertedTelemetry);
  const activeSlotLabel =
    slots.find((s) => s.id === activeSlotId)?.label ?? "Local Driver 1";

  return (
    <div ref={containerRef} className="max-w-7xl mx-auto px-3 sm:px-6 py-4 sm:py-6 space-y-4">
      <div role="alert" aria-live="assertive" aria-atomic="true" className="sr-only">
        {srAnnouncement}
      </div>

      {/* Header */}
      {!isFullscreen && (
        <header className="rounded-2xl border border-border/80 bg-card/80 backdrop-blur-xs p-3.5 sm:p-4 space-y-3">
          <div className="flex flex-wrap items-center justify-between gap-2.5">
            <div className="flex items-center gap-2.5 min-w-0">
              <Button asChild variant="outline" size="sm" className="min-h-11 sm:h-9 sm:min-h-0 px-3 shrink-0">
                <Link href="/">
                  <ArrowLeft className="size-4 mr-1.5" aria-hidden="true" />
                  Home
                </Link>
              </Button>
              <div className="min-w-0">
                <h1 className="text-base sm:text-lg font-bold tracking-tight text-foreground truncate">
                  Offline Local Monitor
                </h1>
                <p className="text-xs text-muted-foreground text-pretty">
                  On-device MediaPipe WASM · Device-isolated IndexedDB storage ({storageMode})
                </p>
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              <Badge
                variant="outline"
                className={`text-[11px] font-mono gap-1 ${
                  isOnline
                    ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10"
                    : "border-amber-500/40 text-amber-600 dark:text-amber-400 bg-amber-500/10"
                }`}
              >
                {isOnline ? (
                  <>
                    <Wifi className="size-3" aria-hidden="true" />
                    ONLINE (LOCAL-ONLY MODE)
                  </>
                ) : (
                  <>
                    <WifiOff className="size-3" aria-hidden="true" />
                    OFFLINE MODE ACTIVE
                  </>
                )}
              </Badge>

              {isOnline && (
                <Button asChild variant="secondary" size="sm" className="min-h-11 sm:h-8 sm:min-h-0 text-xs">
                  <Link href="/dashboard/live-monitor">Cloud Cockpit</Link>
                </Button>
              )}
            </div>
          </div>

          {/* Local Driver Slot Selector & Privacy Isolation Controls */}
          <div className="pt-2.5 border-t border-border/60 flex flex-col lg:flex-row lg:items-center justify-between gap-3">
            <div className="flex flex-wrap items-center gap-2">
              <div className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground">
                <UserCheck className="size-4 text-primary shrink-0" aria-hidden="true" />
                <label htmlFor="local-slot-select">Driver Slot:</label>
              </div>
              <select
                id="local-slot-select"
                value={activeSlotId}
                onChange={(e) => handleSelectSlot(e.target.value)}
                className="min-h-11 sm:h-9 sm:min-h-0 rounded-lg border border-input bg-background px-2.5 text-xs font-medium text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {slots.map((slot) => (
                  <option key={slot.id} value={slot.id}>
                    {slot.label}
                  </option>
                ))}
              </select>

              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => setShowNewSlotInput((prev) => !prev)}
                className="min-h-11 sm:h-9 sm:min-h-0 text-xs gap-1"
              >
                <Plus className="size-3.5" aria-hidden="true" />
                New Slot
              </Button>

              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={handleClearActiveSlot}
                className="min-h-11 sm:h-9 sm:min-h-0 text-xs gap-1 text-destructive hover:text-destructive"
              >
                <Trash2 className="size-3.5" aria-hidden="true" />
                Clear Slot
              </Button>
            </div>

            <div className="flex flex-wrap items-center gap-2.5">
              <div
                role="group"
                aria-label="Sensitivity preset"
                className="inline-flex items-center rounded-lg border border-border bg-muted/50 p-0.5 text-[11px]"
              >
                {SENSITIVITY_OPTIONS.map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    aria-pressed={sensitivityLevel === preset}
                    onClick={() => setSensitivityLevel(preset)}
                    className={`min-h-10 sm:min-h-7 px-2.5 rounded-md font-medium transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                      sensitivityLevel === preset
                        ? "bg-primary text-primary-foreground shadow-xs"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {preset}
                  </button>
                ))}
              </div>

              <label className="flex items-center gap-2 text-xs text-muted-foreground cursor-pointer min-h-11 sm:min-h-0">
                <Switch
                  checked={mirrorSuppression}
                  onCheckedChange={setMirrorSuppression}
                  aria-label="Mirror glance dwell suppression"
                />
                <span>Mirror Suppression</span>
              </label>

              <Button
                type="button"
                variant={isCalibrated ? "outline" : "default"}
                size="sm"
                onClick={handleStartCalibration}
                className="min-h-11 sm:h-9 sm:min-h-0 text-xs gap-1.5"
              >
                <Compass className="size-3.5" aria-hidden="true" />
                {isCalibrated ? "Recalibrate 4-Zone" : "Calibrate 4-Zone"}
              </Button>
            </div>
          </div>

          {showNewSlotInput && (
            <form onSubmit={handleCreateSlot} className="flex flex-col sm:flex-row gap-2 pt-1">
              <Input
                type="text"
                value={newSlotName}
                onChange={(e) => setNewSlotName(e.target.value)}
                placeholder="Enter driver slot name (e.g. Commuter)"
                aria-label="New local driver slot name"
                className="min-h-11 sm:h-9 sm:min-h-0 text-xs sm:max-w-xs"
              />
              <Button type="submit" size="sm" className="min-h-11 sm:h-9 sm:min-h-0 text-xs">
                Create Local Slot
              </Button>
            </form>
          )}
        </header>
      )}

      {/* Truthful Foreground / Background OS Camera Notice */}
      {!isFullscreen && (
        <div
          role={backgroundSuspended ? "alert" : "note"}
          className={`rounded-xl border px-3.5 py-2.5 text-xs flex items-start gap-2.5 ${
            backgroundSuspended
              ? "border-amber-500/50 bg-amber-500/15 text-amber-800 dark:text-amber-200"
              : "border-border/80 bg-muted/35 text-muted-foreground"
          }`}
        >
          <ShieldAlert className="size-4 text-amber-500 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="space-y-0.5">
            {backgroundSuspended ? (
              <p className="font-semibold text-foreground">
                Monitoring paused when DriveGuard left the foreground. Mobile browsers suspend
                camera feeds when backgrounded or locked—tap Resume to continue.
              </p>
            ) : (
              <p>
                <strong className="text-foreground">Privacy &amp; Foreground Policy:</strong> Video
                stays on your device, and profiles are isolated under &ldquo;{activeSlotLabel}
                &rdquo;. Mobile browsers suspend camera processing if you lock the screen or switch
                apps.
              </p>
            )}
          </div>
        </div>
      )}

      {cameraError && (
        <CameraErrorDisplay error={cameraError} onRetry={handleStartMonitoring} />
      )}

      {/* Main Monitoring Workspace */}
      <div
        className={`grid gap-4 ${
          isFullscreen
            ? "grid-cols-1"
            : viewOrientation === "portrait"
              ? "grid-cols-1 md:grid-cols-12"
              : "grid-cols-1 lg:grid-cols-12"
        }`}
      >
        <div
          className={`flex flex-col gap-3 ${
            isFullscreen
              ? "col-span-1"
              : viewOrientation === "portrait"
                ? "md:col-span-7 lg:col-span-6"
                : "lg:col-span-7"
          }`}
        >
          {/* Inline 4-Zone Calibration Banner */}
          {!isFullscreen && (showCalibrationBar || (browserCalibrationPhase && !browserCalibrationPhase.isCompleted)) && (
            <div className="rounded-xl border border-primary/40 bg-primary/10 p-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2.5 text-xs">
              <div className="space-y-0.5 min-w-0">
                <p className="font-semibold text-foreground flex items-center gap-1.5">
                  <Compass className="size-3.5 text-primary shrink-0" aria-hidden="true" />
                  <span>
                    {browserCalibrationPhase && !browserCalibrationPhase.isCompleted
                      ? `Zone ${browserCalibrationPhase.stepIndex}/${browserCalibrationPhase.totalSteps}: ${browserCalibrationPhase.currentZone}`
                      : "4-Zone Local Calibration Ready"}
                  </span>
                </p>
                <p className="text-muted-foreground text-pretty">
                  {browserCalibrationPhase?.guidanceText ??
                    "Start monitoring, keep your eyes open naturally, and follow each gaze prompt."}
                </p>
              </div>
              <div className="flex flex-wrap items-center gap-2 shrink-0">
                {browserCalibrationPhase && !browserCalibrationPhase.isCompleted ? (
                  <>
                    <Button
                      type="button"
                      size="sm"
                      variant="outline"
                      onClick={skipBrowserCalibrationStep}
                      className="min-h-11 sm:h-8 sm:min-h-0 text-xs gap-1"
                    >
                      <FastForward className="size-3.5" aria-hidden="true" />
                      Skip Step
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant="destructive"
                      onClick={() => {
                        cancelBrowserCalibration();
                        setShowCalibrationBar(false);
                      }}
                      className="min-h-11 sm:h-8 sm:min-h-0 text-xs gap-1"
                    >
                      <Square className="size-3.5" aria-hidden="true" />
                      Cancel
                    </Button>
                  </>
                ) : (
                  <Button
                    type="button"
                    size="sm"
                    onClick={handleStartCalibration}
                    className="min-h-11 sm:h-8 sm:min-h-0 text-xs"
                  >
                    Start Calibration
                  </Button>
                )}
              </div>
            </div>
          )}

          <CameraView
            videoRef={videoRef}
            cameraState={cameraState}
            previewMode="video"
            isMirrored={isMirrored}
            framingMode={framingMode}
            viewOrientation={viewOrientation}
            isFullscreen={isFullscreen}
            alertState={convertedTelemetry?.state}
            opticalQuality={opticalQuality}
            onStartClick={handleStartMonitoring}
          >
            {cameraState === "active" && (
              <div
                className={`absolute left-3.5 right-3.5 flex items-center justify-between gap-2 pointer-events-none z-20 ${
                  isFullscreen ? "top-[max(0.75rem,env(safe-area-inset-top,0px))]" : "top-3.5"
                }`}
              >
                {effectiveDisplayState === "CRITICAL" ? (
                  <Badge className="bg-rose-600 text-white border border-rose-300 animate-pulse motion-reduce:animate-none gap-1.5 text-xs font-extrabold px-2.5 py-1">
                    <Flame className="size-3.5" aria-hidden="true" />
                    CRITICAL FATIGUE
                  </Badge>
                ) : effectiveDisplayState === "WARNING" || effectiveDisplayState === "ATTENTION" ? (
                  <Badge className="bg-amber-500 text-slate-950 gap-1.5 text-xs font-extrabold px-2.5 py-1">
                    <AlertTriangle className="size-3.5" aria-hidden="true" />
                    {effectiveDisplayState}
                  </Badge>
                ) : effectiveDisplayState === "MONITORING_UNAVAILABLE" ? (
                  <Badge variant="outline" className="bg-slate-900/90 text-amber-300 border-amber-400/70 gap-1.5 text-xs font-bold px-2.5 py-1">
                    <EyeOff className="size-3.5" aria-hidden="true" />
                    {isCalibrated ? "MONITORING UNAVAILABLE" : "CALIBRATION REQUIRED"}
                  </Badge>
                ) : (
                  <Badge className="bg-emerald-600 text-white gap-1.5 text-xs font-extrabold px-2.5 py-1">
                    <ShieldCheck className="size-3.5" aria-hidden="true" />
                    NORMAL
                  </Badge>
                )}

                {browserEngineStatus === "initializing" && (
                  <Badge variant="outline" className="bg-black/75 text-primary border-primary/50 text-[10px] font-mono">
                    LOADING MEDIAPIPE…
                  </Badge>
                )}
              </div>
            )}

            {isFullscreen && (
              <div className="absolute right-[max(1rem,env(safe-area-inset-right,0px))] bottom-[max(1rem,env(safe-area-inset-bottom,0px))] z-20 flex items-center gap-2">
                <Button
                  type="button"
                  size="sm"
                  variant="destructive"
                  onClick={handleStopMonitoring}
                  className="min-h-11 px-4 text-xs font-semibold gap-1.5"
                >
                  <Square className="size-3.5" aria-hidden="true" />
                  Stop
                </Button>
                <Button
                  type="button"
                  size="icon"
                  variant="secondary"
                  onClick={toggleFullscreen}
                  aria-label="Exit fullscreen mode"
                  className="size-11 rounded-xl"
                >
                  <Minimize2 className="size-4" aria-hidden="true" />
                </Button>
              </div>
            )}
          </CameraView>

          {!isFullscreen && (
            <>
              <CameraControls
                cameraState={cameraState}
                isMirrored={isMirrored}
                framingMode={framingMode}
                isFullscreen={isFullscreen}
                onStart={handleStartMonitoring}
                onStop={handleStopMonitoring}
                onPause={handlePauseCamera}
                onResume={handleResumeCamera}
                onSwitchFacing={switchFacingMode}
                onCaptureSnapshot={handleCaptureSnapshot}
                onToggleFraming={() =>
                  setFramingMode((prev) => (prev === "cover" ? "fit" : "cover"))
                }
                onToggleFullscreen={toggleFullscreen}
                onToggleMirror={() => setIsMirrored((prev) => !prev)}
                isCapturing={isCapturing}
                audioStatus={audioStatus}
                isMuted={audioMuted}
                onToggleMute={toggleMute}
                onUnlockAudio={unlockAudio}
              />

              <CameraStatus
                cameraState={cameraState}
                resolution={resolution}
                facingMode={facingMode}
                durationSec={streamDurationSec}
                runtimeLabel={`Local-Only Browser AI · ${activeSlotLabel}`}
              />
            </>
          )}
        </div>

        {!isFullscreen && (
          <div
            className={`space-y-4 ${
              viewOrientation === "portrait" ? "md:col-span-5 lg:col-span-6" : "lg:col-span-5"
            }`}
          >
            <DriverCockpitHUD
              telemetry={convertedTelemetry}
              cameraState={cameraState}
              isStale={false}
              isCalibrated={isCalibrated}
              runtimeMode="browser"
              sessionDurationSec={streamDurationSec}
              isAudioUnlocked={isAudioUnlocked}
              isMuted={audioMuted}
              volume={audioVolume}
              audioStatus={audioStatus}
              onStartMonitoring={handleStartMonitoring}
              onCalibrateClick={handleStartCalibration}
              onToggleMute={toggleMute}
              onSetVolume={setAudioVolume}
              onUnlockAudio={unlockAudio}
            />

            {/* Local Profile Status & On-Device Telemetry Export */}
            <div className="rounded-2xl border border-border/80 bg-card/80 p-4 space-y-3 text-xs">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="font-semibold text-foreground text-sm">
                    Device-Local Calibration &amp; Export
                  </h2>
                  <p className="text-muted-foreground">
                    {isCalibrated && activeProfile
                      ? `Calibrated baseline EAR ${activeProfile.centerEarMedian.toFixed(
                          3
                        )} stored locally for ${activeSlotLabel}.`
                      : `No 4-zone calibration saved yet for ${activeSlotLabel}.`}
                  </p>
                </div>
                <Badge variant="outline" className="font-mono text-[10px]">
                  {isCalibrated ? "LOCAL PROFILE READY" : "CALIBRATION REQUIRED"}
                </Badge>
              </div>

              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-border/60">
                <span className="font-mono text-muted-foreground tabular-nums">
                  {devices.length} camera(s) · {browserRecordedFrames} local frames buffered
                </span>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={handleExportTelemetry}
                  disabled={browserRecordedFrames === 0}
                  className="min-h-11 sm:h-8 sm:min-h-0 gap-1.5 text-xs"
                >
                  <Download className="size-3.5" aria-hidden="true" />
                  Export .jsonl
                </Button>
              </div>
            </div>
          </div>
        )}
      </div>

      <CameraSnapshotPreview
        frame={capturedSnapshot}
        isOpen={isSnapshotOpen}
        onClose={() => setIsSnapshotOpen(false)}
      />
    </div>
  );
}
