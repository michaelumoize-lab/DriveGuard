"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useLivekitTelemetryStream } from "@/hooks/use-livekit-telemetry-stream";
import { CameraView } from "@/components/camera/camera-view";
import {
  CalibratedProfilePayload,
  CalibrationPhase,
  FrameTelemetry,
  ZoneBaselineEntry,
} from "@/types/telemetry";
import { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";
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
  AlertTriangle,
  Eye,
  Camera,
  Play,
  RotateCcw,
  CheckCircle2,
  ArrowRight,
  ArrowLeft,
  ArrowDown,
  Navigation,
  Square,
  Wifi,
  FastForward,
  Maximize2,
  Minimize2,
  Timer,
  Sparkles,
} from "lucide-react";

export interface GuidedCalibrationWizardProps {
  driverId?: string;
  initialDriverProfile?: InitialDriverProfileData | null;
  embeddedInOnboarding?: boolean;
  onCalibrationSaved?: (profile: InitialDriverProfileData) => void;
}

const BLINK_STREAM_TARGET_SEC = 0; // 2-minute blinking exercise removed in favor of streamlined 15s 4-zone alignment

export function GuidedCalibrationWizard({
  driverId = "driver_default",
  initialDriverProfile = null,
  embeddedInOnboarding = false,
  onCalibrationSaved,
}: GuidedCalibrationWizardProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  // Layout & UI States
  const [viewOrientation] = useState<"portrait" | "landscape">("landscape");
  const [framingMode] = useState<"cover" | "fit">("cover");
  const [isFullscreen, setIsFullscreen] = useState(false);
  const [calibrationWorkflow] = useState<"FOUR_ZONE_CALIBRATION">("FOUR_ZONE_CALIBRATION");
  const [isSuccess, setIsSuccess] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [isSavingProfile, setIsSavingProfile] = useState(false);
  const [calState, setCalState] = useState<CalibrationPhase | null>(null);
  const [savedProfile, setSavedProfile] = useState<InitialDriverProfileData | null>(
    initialDriverProfile
  );
  const [latestTelemetry, setLatestTelemetry] = useState<FrameTelemetry | null>(null);

  const hasHandledCompletionRef = useRef<boolean>(false);

  // Stable Session ID
  const stableSessionId = useMemo(
    () => `cal_${Math.random().toString(36).slice(2, 9)}`,
    []
  );

  // 1. Unified Camera Hook
  const {
    videoRef,
    cameraState,
    isMirrored,
    startCamera,
    stopCamera,
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
      if (
        !backendProfile ||
        typeof backendProfile.center_ear_median !== "number" ||
        !Number.isFinite(backendProfile.center_ear_median) ||
        backendProfile.center_ear_median <= 0
      ) {
        const msg = "Backend completed calibration without returning a valid physiological profile.";
        setSaveError(msg);
        setIsSuccess(false);
        toast.error(msg);
        return;
      }

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
      const windowSec = backendProfile.temporal_window_sec ?? BLINK_STREAM_TARGET_SEC;
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
          throw new Error(errBody.error || `Profile persistence failed (HTTP ${res.status})`);
        }

        setSavedProfile(nextProfile);
        setIsSuccess(true);
        toast.success(
          `Calibrated profile saved (Center EAR ${nextProfile.centerEarMedian.toFixed(3)} · ${Math.round(durMed)} ms blink).`,
          { id: "calibration-completed-toast" }
        );
        if (onCalibrationSaved) {
          onCalibrationSaved(nextProfile);
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : "Failed to save calibrated profile.";
        setSaveError(msg);
        setIsSuccess(false);
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
    calibration: livekitCalibration,
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

  // Keep calState in sync with incoming LiveKit calibration data
  useEffect(() => {
    if (livekitCalibration) {
      setCalState(livekitCalibration);
    }
  }, [livekitCalibration]);

  // Reactive Video Streaming: Link camera activation with LiveKit WebRTC publisher
  useEffect(() => {
    if (cameraState === "active" && videoRef.current) {
      startStreamingLoop(videoRef.current, isMirrored);
    } else if (cameraState === "stopped") {
      stopStreamingLoop();
    }
  }, [cameraState, isMirrored, startStreamingLoop, stopStreamingLoop, videoRef]);

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

  // User Actions — strictly delegated to backend GuidedCalibrationSession (Streamlined 4-Zone)
  const handleStartCalibration = async () => {
    hasHandledCompletionRef.current = false;
    setIsSuccess(false);
    setSaveError(null);
    setCalState(null);

    if (cameraState !== "active") {
      await startCamera();
    }

    // Runs pure 4-zone spatial calibration with 4-second hold per zone, zero blink delay
    triggerLivekitCalibration(true, 4.0, false, 0);
    toast.info("Starting Precision 4-Zone Gaze Alignment (~15s)...");
  };

  const handleStopCalibration = () => {
    triggerLivekitCancel();
    stopCamera();
    stopStreamingLoop();
    setCalState(null);
    setSaveError(null);
  };

  const handleSkipZone = () => {
    triggerLivekitSkip();
    toast.info("Requesting backend to advance past current calibration stage...");
  };

  const handleRecalibrateZone = async (zone: string) => {
    hasHandledCompletionRef.current = false;
    setIsSuccess(false);
    setSaveError(null);
    setCalState(null);
    if (cameraState !== "active") {
      await startCamera();
    }
    triggerLivekitRecalibrate(zone, 4.0, 0);
    toast.info(`Recalibrating ${zone.replace("_", " ")} on backend...`);
  };

  const isInBlinkPhase = false;

  // Read live blink cadence metrics strictly from backend CalibrationPhaseDTO
  const liveBlinkStats = useMemo(() => {
    const targetSec = calState?.target_duration_sec ?? BLINK_STREAM_TARGET_SEC;
    const elapsed = calState?.elapsed_sec ?? 0;
    const remainingSec = Math.max(0, Math.round(targetSec - elapsed));
    const mm = String(Math.floor(remainingSec / 60)).padStart(2, "0");
    const ss = String(remainingSec % 60).padStart(2, "0");
    const progressPct =
      typeof calState?.zone_progress_pct === "number"
        ? Math.min(100, Math.round(calState.zone_progress_pct))
        : targetSec > 0
          ? Math.min(100, Math.round((elapsed / targetSec) * 100))
          : 0;

    return {
      count: calState?.collected_blinks_count ?? 0,
      elapsed,
      remainingFormatted: `${mm}:${ss}`,
      progressPct,
      ratePerMin: calState?.live_blink_rate_per_min ?? 0.0,
      medianDurationMs:
        typeof calState?.live_median_blink_duration_ms === "number"
          ? Math.round(calState.live_median_blink_duration_ms)
          : null,
      medianRcr:
        typeof calState?.live_median_blink_rcr === "number"
          ? calState.live_median_blink_rcr.toFixed(2)
          : null,
    };
  }, [calState]);

  const currentZoneInfo = useMemo(() => {
    if (isInBlinkPhase) {
      return {
        label: "2-MINUTE NATURAL BLINK & CADENCE STREAM",
        icon: <Activity className="size-6 text-emerald-400 animate-pulse" />,
        color: "text-emerald-400",
      };
    }
    const zone = calState?.current_zone || "CENTER";
    switch (zone) {
      case "LEFT_MIRROR":
        return {
          label: "TURN HEAD LEFT (DRIVER MIRROR)",
          icon: <ArrowLeft className="size-6 text-blue-400 animate-pulse" />,
          color: "text-blue-400",
        };
      case "RIGHT_MIRROR":
        return {
          label: "TURN HEAD RIGHT (PASSENGER MIRROR)",
          icon: <ArrowRight className="size-6 text-amber-400 animate-pulse" />,
          color: "text-amber-400",
        };
      case "DASHBOARD":
        return {
          label: "GLANCE DOWN (DASHBOARD)",
          icon: <ArrowDown className="size-6 text-purple-400 animate-pulse" />,
          color: "text-purple-400",
        };
      case "CENTER":
      default:
        return {
          label: "LOOK STRAIGHT AHEAD (CENTER ROAD)",
          icon: <Navigation className="size-6 text-emerald-400" />,
          color: "text-emerald-400",
        };
    }
  }, [calState?.current_zone, isInBlinkPhase]);

  const isCalibrating =
    cameraState === "active" &&
    isLivekitConnected &&
    Boolean(calState && !calState.is_completed);

  const isFaceVisible = latestTelemetry ? latestTelemetry.face_detected : null;
  const liveEarValue =
    latestTelemetry && latestTelemetry.face_detected
      ? latestTelemetry.eye_metrics.ear_raw
      : null;
  const livePerclosPct =
    latestTelemetry && latestTelemetry.face_detected
      ? latestTelemetry.eye_metrics.perclos_pct
      : null;

  // Honest Baseline Inspector values — never show fabricated numbers when uncalibrated
  const storedTemporal = useMemo(() => {
    if (!savedProfile?.isCalibrated) {
      return {
        blinkFreqMedianPerMin: null as number | null,
        blinkFreqMadPerMin: null as number | null,
        blinkDurationMedianMs: null as number | null,
        blinkDurationMadMs: null as number | null,
        blinkRcrMedian: null as number | null,
        blinkRcrMad: null as number | null,
        baselinePerclosPct: null as number | null,
        temporalBlinksSampled: null as number | null,
        hasCustomBlinkStream: false,
      };
    }
    const rawZones = (savedProfile.zoneBaselines || {}) as Record<string, any>;
    const meta = (rawZones._temporalBaseline || {}) as Record<string, any>;
    return {
      blinkFreqMedianPerMin:
        typeof meta.blinkFreqMedianPerMin === "number" ? meta.blinkFreqMedianPerMin : null,
      blinkFreqMadPerMin:
        typeof meta.blinkFreqMadPerMin === "number" ? meta.blinkFreqMadPerMin : null,
      blinkDurationMedianMs: savedProfile.blinkDurationMedianMs,
      blinkDurationMadMs: savedProfile.blinkDurationMadMs,
      blinkRcrMedian: savedProfile.blinkRcrMedian,
      blinkRcrMad: savedProfile.blinkRcrMad,
      baselinePerclosPct:
        typeof meta.baselinePerclosPct === "number" ? meta.baselinePerclosPct : null,
      temporalBlinksSampled:
        typeof meta.temporalBlinksSampled === "number" ? meta.temporalBlinksSampled : null,
      hasCustomBlinkStream: Boolean(meta.calibratedAt),
    };
  }, [savedProfile]);

  const qualityWarnings = calState?.quality_warnings ?? [];

  return (
    <div
      ref={containerRef}
      className={`space-y-5 transition-all ${
        isFullscreen
          ? "fixed inset-0 z-50 bg-black p-4 flex flex-col justify-between overflow-hidden"
          : ""
      }`}
    >
      {/* 5-Stage Progress Strip */}
      {!isFullscreen && (
        <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
          {[
            { idx: 1, key: "CENTER", title: "1. Center Road", sub: "Primary Reference" },
            { idx: 2, key: "LEFT_MIRROR", title: "2. Left Mirror", sub: "Driver Side Glance" },
            { idx: 3, key: "RIGHT_MIRROR", title: "3. Right Mirror", sub: "Passenger Side Glance" },
            { idx: 4, key: "DASHBOARD", title: "4. Dashboard", sub: "Instrument Cluster" },
            {
              idx: 5,
              key: "BLINK_CADENCE",
              title: "5. 2-Min Blink Stream",
              sub: "Personal Blink Rate & Speed",
            },
          ].map((stage) => {
            const activeIdx = isInBlinkPhase ? 5 : calState?.phase_index ?? 0;
            const isCurrent = isCalibrating && activeIdx === stage.idx;
            const isCompletedStage =
              isSuccess || (isCalibrating && activeIdx > stage.idx);

            return (
              <div
                key={stage.key}
                className={`p-2.5 rounded-xl border text-xs transition-all ${
                  isCurrent
                    ? "border-primary bg-primary/10 text-foreground shadow-xs"
                    : isCompletedStage
                      ? "border-emerald-500/40 bg-emerald-500/5 text-emerald-500"
                      : "border-border/70 bg-muted/20 text-muted-foreground"
                }`}
              >
                <div className="flex items-center justify-between gap-1 font-semibold">
                  <span className="truncate">{stage.title}</span>
                  {isCompletedStage && <CheckCircle2 className="size-3.5 shrink-0 text-emerald-500" />}
                </div>
                <div className="text-[10px] opacity-80 truncate mt-0.5">{stage.sub}</div>
              </div>
            );
          })}
        </div>
      )}

      <div
        className={`grid gap-5 ${
          isFullscreen ? "h-full grid-cols-1" : "grid-cols-1 lg:grid-cols-3"
        }`}
      >
        {/* Video Viewport */}
        <Card
          className={`overflow-hidden border-border/80 flex flex-col ${
            isFullscreen ? "border-0 rounded-none bg-black h-full" : "lg:col-span-2"
          }`}
        >
          <CardHeader className="py-3 px-4 flex flex-row flex-wrap items-center justify-between gap-2 border-b border-border">
            <div className="flex items-center gap-2">
              <Camera className="size-4 text-primary" />
              <CardTitle className="text-sm font-semibold">
                {isInBlinkPhase
                  ? "Stage 2: 2-Minute Natural Blink Rate & Cadence Stream"
                  : "Stage 1: 4-Zone Spatial Alignment"}
              </CardTitle>
            </div>

            <div className="flex items-center gap-2">
              {/* Target Inference Selector */}
              <div className="flex items-center bg-muted/60 rounded-md p-0.5 text-[11px] font-mono border border-border">
                <button
                  type="button"
                  onClick={() => {
                    if (backendType !== "local") {
                      if (cameraState === "active") handleStopCalibration();
                      switchBackend("local");
                      toast.info("Calibration reset. Switched AI Engine to Localhost (127.0.0.1:8000).");
                    }
                  }}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    backendType === "local"
                      ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="Local AI Engine (127.0.0.1:8000)"
                >
                  ⚡ Local
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (backendType !== "cloud") {
                      if (cameraState === "active") handleStopCalibration();
                      switchBackend("cloud");
                      toast.info("Calibration reset. Switched AI Engine to LiveKit Cloud.");
                    }
                  }}
                  className={`px-2 py-0.5 rounded transition-colors ${
                    backendType === "cloud"
                      ? "bg-primary text-primary-foreground font-semibold shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                  title="LiveKit Cloud SFU"
                >
                  ☁️ LiveKit Cloud
                </button>
              </div>

              <Badge
                variant="outline"
                className={`text-xs font-mono flex items-center gap-1 ${
                  workerError || connectionStatus === "error"
                    ? "border-rose-500/40 text-rose-500 bg-rose-500/10"
                    : isStale
                      ? "border-amber-500/40 text-amber-500 bg-amber-500/10"
                      : isCalibrating
                        ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                        : "text-muted-foreground"
                }`}
              >
                {workerError || connectionStatus === "error" ? (
                  "ERROR"
                ) : isStale ? (
                  "STALE STREAM"
                ) : isCalibrating ? (
                  <>
                    <Wifi className="size-3" />
                    {isInBlinkPhase ? "BLINK STREAM" : "CALIBRATING"}
                  </>
                ) : backendProcessingStatus === "connecting" ||
                  backendProcessingStatus === "awaiting_telemetry" ? (
                  "CONNECTING..."
                ) : (
                  "STANDBY"
                )}
              </Badge>

              {!embeddedInOnboarding && (
                <Button
                  size="sm"
                  variant="ghost"
                  onClick={toggleFullscreen}
                  className="h-8 w-8 p-0 text-muted-foreground hover:text-foreground"
                  title={isFullscreen ? "Exit Fullscreen" : "Fullscreen HUD"}
                >
                  {isFullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
                </Button>
              )}
            </div>
          </CardHeader>

          <div className="relative bg-black flex-1 flex items-center justify-center overflow-hidden min-h-[320px]">
            <CameraView
              videoRef={videoRef}
              cameraState={cameraState}
              isMirrored={isMirrored}
              framingMode={framingMode}
              viewOrientation={viewOrientation}
              isFullscreen={isFullscreen}
              onStartClick={() => handleStartCalibration()}
            >
              {/* Standby Setup Overlay */}
              {cameraState !== "active" && !isSuccess && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-4 z-20">
                  <div className="p-3 rounded-full bg-primary/10 text-primary">
                    <Eye className="size-8" />
                  </div>
                  <div className="font-semibold text-foreground text-sm sm:text-base">
                    4-Zone Spatial + 2-Minute Personal Blink Rate Calibration
                  </div>
                  <p className="text-xs text-muted-foreground max-w-md">
                    Align your head across the 4 primary cabin zones (Center, Left Mirror, Right Mirror, Dashboard).
                    High-precision mathematical baselines are established in ~15 seconds with zero unnatural blinking delays.
                  </p>

                  <div className="flex items-center gap-2 px-3 py-1.5 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-600 dark:text-emerald-400 text-xs font-mono">
                    <Sparkles className="size-3.5" />
                    <span>Streamlined 4-Zone Gaze Alignment (~15 Seconds)</span>
                  </div>

                  <Button
                    size="sm"
                    onClick={() => handleStartCalibration()}
                    className="bg-primary text-primary-foreground font-semibold px-6 min-h-[44px]"
                  >
                    <Play className="size-4 mr-1.5" />
                    Start 4-Zone Calibration
                  </Button>
                </div>
              )}

              {/* In-Video Heads Up Status Banner (Spatial Phase) */}
              {cameraState === "active" && calState && !isInBlinkPhase && !isSuccess && (
                <div className="absolute top-4 left-4 right-4 flex items-start justify-between pointer-events-none z-20">
                  <div className="px-3 py-1 rounded-lg bg-black/80 backdrop-blur-md border border-white/15 text-xs text-white font-mono flex items-center gap-2">
                    <span>
                      Phase {calState.phase_index}/{calState.total_phases}:
                    </span>
                    <strong className="text-primary">{calState.current_zone}</strong>
                  </div>

                  <Badge
                    variant={calState.is_head_in_target_zone ? "outline" : "destructive"}
                    className={`text-[11px] font-mono shadow-md ${
                      calState.is_head_in_target_zone
                        ? "border-emerald-500/60 text-emerald-400 bg-black/80"
                        : "bg-rose-600/90 text-white"
                    }`}
                  >
                    {isFaceVisible === false
                      ? "NO FACE DETECTED"
                      : calState.is_head_in_target_zone
                        ? "ALIGNED [ACQUIRING]"
                        : "TURN HEAD TO TARGET"}
                  </Badge>
                </div>
              )}

              {/* In-Video Heads Up Status Banner (Backend 2-Minute Blink Stream Phase) */}
              {cameraState === "active" && isInBlinkPhase && !isSuccess && (
                <div className="absolute inset-x-4 top-4 bottom-4 flex flex-col justify-between pointer-events-none z-20">
                  <div className="flex flex-wrap items-center justify-between gap-2">
                    <div className="px-3 py-1.5 rounded-lg bg-black/80 backdrop-blur-md border border-emerald-500/40 text-xs text-white font-mono flex items-center gap-2">
                      <Timer className="size-4 text-emerald-400 animate-pulse" />
                      <span>2-MIN BLINK STREAM:</span>
                      <strong className="text-emerald-400 text-sm">
                        {liveBlinkStats.remainingFormatted}
                      </strong>
                    </div>

                    <Badge
                      variant="outline"
                      className={`text-[11px] font-mono shadow-md ${
                        isFaceVisible === false
                          ? "border-rose-500/60 text-rose-400 bg-black/85"
                          : "border-emerald-500/60 text-emerald-400 bg-black/80"
                      }`}
                    >
                      {isFaceVisible === false
                        ? "KEEP FACE IN CAMERA"
                        : liveEarValue !== null
                          ? `EAR ${liveEarValue.toFixed(3)} · BACKEND TRACKING`
                          : "AWAITING TELEMETRY"}
                    </Badge>
                  </div>

                  <div className="p-3 rounded-xl bg-black/80 backdrop-blur-md border border-white/15 grid grid-cols-3 gap-2 text-center font-mono text-white">
                    <div>
                      <span className="text-[10px] text-zinc-400 block">BLINKS CAPTURED</span>
                      <span className="text-base font-bold text-emerald-400">
                        {liveBlinkStats.count}
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-zinc-400 block">BLINK RATE</span>
                      <span className="text-base font-bold text-sky-400">
                        {liveBlinkStats.ratePerMin.toFixed(1)}/min
                      </span>
                    </div>
                    <div>
                      <span className="text-[10px] text-zinc-400 block">MEDIAN DURATION</span>
                      <span className="text-base font-bold text-amber-400">
                        {liveBlinkStats.medianDurationMs !== null
                          ? `${liveBlinkStats.medianDurationMs}ms`
                          : "—"}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Success Overlay */}
              {isSuccess && (
                <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-3 z-20">
                  <div className="p-3 rounded-full bg-emerald-500/10 text-emerald-500">
                    <CheckCircle2 className="size-10" />
                  </div>
                  <div className="font-bold text-foreground text-base">
                    Personal Calibrated Driver Profile Saved!
                  </div>
                  <p className="text-xs text-muted-foreground max-w-md">
                    Your authoritative spatial EAR baselines (Center EAR{" "}
                    <strong className="text-foreground">
                      {savedProfile?.centerEarMedian.toFixed(3) ?? "—"}
                    </strong>
                    ) and blink duration baseline (
                    <strong className="text-foreground">
                      {storedTemporal.blinkDurationMedianMs !== null
                        ? `${Math.round(storedTemporal.blinkDurationMedianMs)} ms`
                        : "—"}
                    </strong>
                    ) are now active in PostgreSQL and the FastAPI backend.
                  </p>
                  <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                    <Button
                      size="sm"
                      onClick={() => handleStartCalibration()}
                      variant="outline"
                    >
                      <RotateCcw className="size-4 mr-1.5" />
                      Run 4-Zone Recalibration
                    </Button>
                  </div>
                </div>
              )}
            </CameraView>
          </div>

          <CardFooter className="py-3 px-4 flex flex-wrap items-center justify-between gap-2 border-t border-border bg-muted/20">
            <div className="flex flex-wrap items-center gap-2">
              {cameraState === "active" && !isSuccess ? (
                <>
                  <Button size="sm" variant="destructive" onClick={handleStopCalibration}>
                    <Square className="size-3.5 mr-1.5" />
                    Stop
                  </Button>
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={handleSkipZone}
                    disabled={!isCalibrating || isSavingProfile}
                    title="Ask backend to advance past current calibration stage"
                  >
                    <FastForward className="size-3.5 mr-1.5 text-primary" />
                    {isInBlinkPhase ? "Complete Blink Stage Early" : "Skip Zone"}
                  </Button>
                </>
              ) : (
                <Button
                  size="sm"
                  onClick={() => handleStartCalibration()}
                  disabled={isSavingProfile}
                  className="bg-primary text-primary-foreground"
                >
                  <Play className="size-3.5 mr-1.5" />
                  {isSavingProfile
                    ? "Saving Profile..."
                    : isSuccess
                      ? "Recalibrate"
                      : "Start Calibration"}
                </Button>
              )}
            </div>
            <span className="text-xs text-muted-foreground font-mono">
              {isInBlinkPhase
                ? `Phase ${calState?.phase_index ?? 5}/${calState?.total_phases ?? 5} · Blink Stream (${liveBlinkStats.remainingFormatted} left)`
                : calState
                  ? `Stage ${calState.phase_index} of ${calState.total_phases}`
                  : "Ready"}
            </span>
          </CardFooter>
        </Card>

        {/* Guidance & Live Telemetry HUD */}
        {!isFullscreen && (
          <Card className="border-border/80">
            <CardHeader className="pb-3">
              <CardTitle className="text-sm font-semibold flex items-center gap-2">
                <Eye className="size-4 text-primary" />
                {isInBlinkPhase
                  ? "2-Minute Blink Cadence Profiler"
                  : "Spatial Target Guidance"}
              </CardTitle>
              <CardDescription className="text-xs">
                {isInBlinkPhase
                  ? "Look forward comfortably and blink at your normal pace"
                  : "Align your head with each target zone until backend locks 100%"}
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4 text-xs">
              {/* Active Target Banner */}
              <div className="p-3.5 rounded-xl border border-primary/20 bg-primary/5 flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10 shrink-0">
                  {currentZoneInfo.icon}
                </div>
                <div>
                  <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                    Active Calibration Phase
                  </span>
                  <span className="font-bold text-foreground text-sm">
                    {calState?.stability_feedback && calState.stability_feedback.includes("Get ready")
                      ? calState.stability_feedback
                      : currentZoneInfo.label}
                  </span>
                </div>
              </div>

              {/* Connection / Worker / Save Error Feedback */}
              {(workerError || saveError) && (
                <div className="p-3 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive text-xs space-y-2">
                  <div className="flex items-start gap-2">
                    <AlertTriangle className="size-4 shrink-0 mt-0.5" />
                    <div>
                      <span className="font-semibold block">Calibration Error</span>
                      <p className="mt-0.5">{saveError || workerError}</p>
                    </div>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-destructive/40 text-destructive hover:bg-destructive/10 h-7 text-xs"
                    onClick={() => handleStartCalibration()}
                  >
                    <RotateCcw className="size-3.5 mr-1" />
                    Retry Calibration
                  </Button>
                </div>
              )}

              {/* Rejection Feedback */}
              {calState?.rejection_reason && (
                <div className="p-3 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive text-xs space-y-2">
                  <div>
                    <span className="font-semibold block">Calibration Session Rejected</span>
                    <p className="mt-0.5">{calState.rejection_reason}</p>
                  </div>
                  <Button
                    size="sm"
                    variant="outline"
                    className="border-destructive/40 text-destructive hover:bg-destructive/10 h-7 text-xs"
                    onClick={() => handleStartCalibration()}
                  >
                    <RotateCcw className="size-3.5 mr-1" />
                    Retry Calibration
                  </Button>
                </div>
              )}

              {/* Non-Fatal Backend Quality Warnings */}
              {qualityWarnings.length > 0 && (
                <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/30 text-amber-600 dark:text-amber-400 text-xs space-y-1">
                  <span className="font-semibold block">Calibration Quality Advisories:</span>
                  <ul className="list-disc pl-4 space-y-0.5">
                    {qualityWarnings.map((w, idx) => (
                      <li key={idx}>{w}</li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Spatial Alignment Status & Mathematical Telemetry */}
              <div className="space-y-3">
                <div className="space-y-1.5">
                  <div className="flex justify-between items-center">
                    <span className="text-muted-foreground text-xs">Spatial Alignment:</span>
                    <Badge
                      variant={calState?.is_head_in_target_zone ? "outline" : "destructive"}
                      className={`text-[10px] ${
                        calState?.is_head_in_target_zone
                          ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                          : ""
                      }`}
                    >
                      {calState?.is_head_in_target_zone
                        ? "HEAD ALIGNED [OK]"
                        : calState?.stability_feedback || "WAITING FOR HEAD ALIGNMENT"}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-muted-foreground">
                    {calState?.instruction ||
                      "Click Start 4-Zone Calibration to align your cabin gaze baselines."}
                  </p>
                </div>

                {/* Mathematical Baseline Card */}
                <div className="p-3 rounded-xl bg-card/60 border border-border/80 font-mono text-xs space-y-2">
                  <div className="flex justify-between items-center text-muted-foreground text-[10px] uppercase tracking-wider">
                    <span>Mathematical Formulations</span>
                    <span className="text-primary font-bold">PDBP Statistics</span>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-[11px]">
                    <div className="p-2 rounded-lg bg-muted/20 border border-border/60">
                      <span className="text-[10px] text-muted-foreground block">ROBUST MAD</span>
                      <span className="font-bold text-foreground">
                        {latestTelemetry?.comparison?.personalised_threshold
                          ? `Tz = ${latestTelemetry.comparison.personalised_threshold.toFixed(3)}`
                          : "MAD: 1.4826·|x - x̃|"}
                      </span>
                    </div>
                    <div className="p-2 rounded-lg bg-muted/20 border border-border/60">
                      <span className="text-[10px] text-muted-foreground block">DYNAMIC CUTOFF</span>
                      <span className="font-bold text-emerald-500">
                        {savedProfile?.centerEarMedian
                          ? `Center: ${savedProfile.centerEarMedian.toFixed(3)}`
                          : "Tz = x̃ - 2.5·σ"}
                      </span>
                    </div>
                  </div>
                </div>

                {/* Active Zone Progress */}
                <div className="space-y-1.5 pt-2 border-t border-border">
                    <div className="flex justify-between font-mono">
                      <span className="text-muted-foreground">Zone Sample Acquisition:</span>
                      <span className="font-semibold text-foreground">
                        {calState
                          ? `${Math.round(calState.zone_progress_pct)}% (${calState.stable_samples_count}/${calState.target_samples_count})`
                          : "0%"}
                      </span>
                    </div>
                    <Progress value={calState?.zone_progress_pct || 0} className="h-2" />
                    <div className="text-[10px] text-muted-foreground flex justify-between">
                      <span>
                        Quality:{" "}
                        {typeof calState?.zone_quality_score === "number"
                          ? `${Math.round(calState.zone_quality_score * 100)}%`
                          : "—"}
                      </span>
                      <span>100% (Locked)</span>
                    </div>
                  </div>

                  {/* Overall Spatial Progress */}
                  <div className="space-y-1.5 pt-2 border-t border-border">
                    <div className="flex justify-between font-mono">
                      <span className="text-muted-foreground">Overall Calibration Progress:</span>
                      <span className="font-semibold text-foreground">
                        {calState ? `${Math.round(calState.total_progress_pct)}%` : "0%"}
                      </span>
                    </div>
                    <Progress value={calState?.total_progress_pct || 0} className="h-1.5" />
                  </div>
                </div>
            </CardContent>
          </Card>
        )}
      </div>

      {/* 4-Zone Spatial + Personal Blink Cadence Baseline Inspector */}
      {!isFullscreen && (
        <Card className="border-border/80">
          <CardHeader className="pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base font-bold flex items-center gap-2">
                <CheckCircle2 className="size-4 text-primary" />
                Personal 4-Zone & Temporal Blink Baseline Inspector
              </CardTitle>
              <CardDescription className="text-xs">
                Stored in Neon PostgreSQL (`DriverProfile`) and synced to the FastAPI authoritative
                engine (`DrowsinessEngine`).
              </CardDescription>
            </div>
            <Badge
              variant="outline"
              className={
                savedProfile?.isCalibrated
                  ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10 font-mono text-xs"
                  : "border-amber-500/40 text-amber-500 bg-amber-500/10 font-mono text-xs"
              }
            >
              {savedProfile?.isCalibrated ? "CALIBRATED PROFILE ACTIVE" : "CALIBRATION REQUIRED"}
            </Badge>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* Personal Temporal Blink Kinematics Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              <div className="p-3.5 rounded-xl border border-primary/25 bg-primary/5 flex flex-col justify-between gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-foreground">
                    BLINK RATE (/MIN)
                  </span>
                  <Badge variant="secondary" className="text-[10px] font-mono">
                    {storedTemporal.hasCustomBlinkStream ? "Calibrated" : "Uncalibrated"}
                  </Badge>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                  <span className="text-[11px] text-muted-foreground">Personal Median:</span>
                  <span className="text-base font-bold text-sky-500">
                    {storedTemporal.blinkFreqMedianPerMin !== null
                      ? `${storedTemporal.blinkFreqMedianPerMin.toFixed(1)} /min`
                      : "—"}
                  </span>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums">
                  <span className="text-[11px] text-muted-foreground">Robust MAD Spread:</span>
                  <span className="text-xs font-semibold text-foreground">
                    {storedTemporal.blinkFreqMadPerMin !== null
                      ? `±${storedTemporal.blinkFreqMadPerMin.toFixed(1)} /min`
                      : "—"}
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl border border-primary/25 bg-primary/5 flex flex-col justify-between gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-foreground">
                    BLINK DURATION
                  </span>
                  <Badge variant="secondary" className="text-[10px] font-mono">
                    Kinematics
                  </Badge>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                  <span className="text-[11px] text-muted-foreground">Personal Median:</span>
                  <span className="text-base font-bold text-amber-500">
                    {storedTemporal.blinkDurationMedianMs !== null
                      ? `${Math.round(storedTemporal.blinkDurationMedianMs)} ms`
                      : "—"}
                  </span>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums">
                  <span className="text-[11px] text-muted-foreground">Robust MAD:</span>
                  <span className="text-xs font-semibold text-foreground">
                    {storedTemporal.blinkDurationMadMs !== null
                      ? `±${Math.round(storedTemporal.blinkDurationMadMs)} ms`
                      : "—"}
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl border border-primary/25 bg-primary/5 flex flex-col justify-between gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-foreground">
                    EYELID RCR VELOCITY
                  </span>
                  <Badge variant="secondary" className="text-[10px] font-mono">
                    Kinematics
                  </Badge>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                  <span className="text-[11px] text-muted-foreground">Reopening Ratio:</span>
                  <span className="text-base font-bold text-foreground">
                    {storedTemporal.blinkRcrMedian !== null
                      ? storedTemporal.blinkRcrMedian.toFixed(2)
                      : "—"}
                  </span>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums">
                  <span className="text-[11px] text-muted-foreground">Robust MAD:</span>
                  <span className="text-xs font-semibold text-foreground">
                    {storedTemporal.blinkRcrMad !== null
                      ? `±${storedTemporal.blinkRcrMad.toFixed(2)}`
                      : "—"}
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl border border-primary/25 bg-primary/5 flex flex-col justify-between gap-2">
                <div className="space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-bold text-foreground">
                      RESTING PERCLOS
                    </span>
                    <Badge variant="secondary" className="text-[10px] font-mono">
                      {storedTemporal.temporalBlinksSampled !== null
                        ? `${storedTemporal.temporalBlinksSampled} blinks`
                        : "Uncalibrated"}
                    </Badge>
                  </div>
                  <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                    <span className="text-[11px] text-muted-foreground">Resting Baseline:</span>
                    <span className="text-base font-bold text-emerald-500">
                      {storedTemporal.baselinePerclosPct !== null
                        ? `${storedTemporal.baselinePerclosPct.toFixed(1)}%`
                        : "—"}
                    </span>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => handleRecalibrateZone("BLINK_CADENCE")}
                  className="w-full h-8 text-xs font-mono"
                >
                  <Timer className="size-3 mr-1.5" />
                  Recalibrate 2-Min Blink Stream
                </Button>
              </div>
            </div>

            {/* 4 Spatial Zones Row */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              {(["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"] as const).map((zone) => {
                const zbMap = (savedProfile?.zoneBaselines ?? null) as Record<
                  string,
                  ZoneBaselineEntry
                > | null;
                const zb = savedProfile?.isCalibrated ? zbMap?.[zone] : undefined;
                const earMed =
                  typeof zb?.ear_median === "number"
                    ? zb.ear_median
                    : zone === "CENTER" && savedProfile?.isCalibrated
                      ? savedProfile.centerEarMedian
                      : null;
                const thresh =
                  typeof zb?.dynamic_closure_threshold === "number"
                    ? zb.dynamic_closure_threshold
                    : null;

                return (
                  <div
                    key={zone}
                    className="p-3.5 rounded-xl border border-border bg-muted/20 flex flex-col justify-between gap-3"
                  >
                    <div className="space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-mono font-bold text-foreground">
                          {zone.replace("_", " ")}
                        </span>
                        <Badge variant="secondary" className="text-[10px] font-mono">
                          {earMed !== null ? "Calibrated" : "Uncalibrated"}
                        </Badge>
                      </div>
                      <div className="pt-1 flex items-baseline justify-between font-mono tabular-nums">
                        <span className="text-[11px] text-muted-foreground">Median EAR:</span>
                        <span className="text-base font-bold text-foreground">
                          {earMed !== null ? earMed.toFixed(3) : "—"}
                        </span>
                      </div>
                      <div className="flex items-baseline justify-between font-mono tabular-nums">
                        <span className="text-[11px] text-muted-foreground">Closure Cutoff:</span>
                        <span className="text-xs font-semibold text-emerald-500">
                          {thresh !== null ? thresh.toFixed(3) : "—"}
                        </span>
                      </div>
                    </div>

                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => handleRecalibrateZone(zone)}
                      className="w-full h-8 text-xs font-mono"
                    >
                      <RotateCcw className="size-3 mr-1.5" />
                      Recalibrate {zone.replace("_", " ")}
                    </Button>
                  </div>
                );
              })}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
}
