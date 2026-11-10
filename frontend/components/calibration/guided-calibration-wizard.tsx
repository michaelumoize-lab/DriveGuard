"use client";

import React, { useState, useRef, useEffect, useCallback, useMemo } from "react";
import { toast } from "sonner";
import { useCamera } from "@/hooks/use-camera";
import { useLivekitTelemetryStream } from "@/hooks/use-livekit-telemetry-stream";
import { CameraView } from "@/components/camera/camera-view";
import { CalibrationPhase, FrameTelemetry, ZoneBaselineEntry } from "@/types/telemetry";
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

interface CapturedBlinkSample {
  timestampMs: number;
  durationMs: number;
  rcr: number;
  amplitude: number;
}

function computeMedianAndScaledMad(values: number[], minFloor: number): [number, number] {
  if (!values.length) return [0, minFloor];
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  const median =
    sorted.length % 2 === 1 ? sorted[mid] : 0.5 * (sorted[mid - 1] + sorted[mid]);
  const absDevs = sorted.map((v) => Math.abs(v - median)).sort((a, b) => a - b);
  const mad =
    absDevs.length % 2 === 1
      ? absDevs[mid]
      : 0.5 * (absDevs[mid - 1] + absDevs[mid]);
  const scaledMad = Math.max(minFloor, 1.4826 * mad);
  return [median, scaledMad];
}

const BLINK_STREAM_TARGET_SEC = 120;

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
  const [calibrationWorkflow, setCalibrationWorkflow] = useState<
    "FULL_4ZONE_AND_BLINK" | "QUICK_4ZONE_ONLY" | "BLINK_2MIN_ONLY"
  >("FULL_4ZONE_AND_BLINK");
  const [isSuccess, setIsSuccess] = useState(false);
  const [calState, setCalState] = useState<CalibrationPhase | null>(null);
  const [savedProfile, setSavedProfile] = useState<InitialDriverProfileData | null>(
    initialDriverProfile
  );

  // 2-Minute Natural Blink Rate & Cadence Stream State
  const [isBlinkStreamActive, setIsBlinkStreamActive] = useState(false);
  const [blinkElapsedSec, setBlinkElapsedSec] = useState(0);
  const [capturedBlinks, setCapturedBlinks] = useState<CapturedBlinkSample[]>([]);
  const [livePerclosPct, setLivePerclosPct] = useState<number>(0);
  const [liveEarValue, setLiveEarValue] = useState<number>(0.285);
  const [isFaceVisibleInStream, setIsFaceVisibleInStream] = useState<boolean>(true);

  const isBlinkStreamActiveRef = useRef(false);
  const blinkElapsedSecRef = useRef(0);
  const capturedBlinksRef = useRef<CapturedBlinkSample[]>([]);
  const pendingSpatialProfileRef = useRef<Record<string, any> | null>(null);
  const lastSeenBlinkDurRef = useRef<number | null>(null);
  const lastBlinkRecordedAtMsRef = useRef<number>(0);
  const eyeClosedStartMsRef = useRef<number | null>(null);
  const minEarDuringClosureRef = useRef<number>(0.3);
  const perclosSamplesRef = useRef<number[]>([]);
  const hasHandledCompletionRef = useRef<boolean>(false);
  const workflowRef = useRef(calibrationWorkflow);
  useEffect(() => {
    workflowRef.current = calibrationWorkflow;
  }, [calibrationWorkflow]);

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

  // Finalize & Persist Combined (4-Zone Spatial + 2-Min Blink Rate) Profile
  const finalizeAndPersistProfile = useCallback(
    (spatialProf: Record<string, any> | null, blinks: CapturedBlinkSample[], elapsedSec: number) => {
      isBlinkStreamActiveRef.current = false;
      setIsBlinkStreamActive(false);

      const centerEar =
        (spatialProf?.center_ear_median as number) ??
        savedProfile?.centerEarMedian ??
        0.285;
      const centerEarMad =
        (spatialProf?.center_ear_mad_scaled as number) ??
        savedProfile?.centerEarMadScaled ??
        0.018;
      const zoneMap =
        (spatialProf?.zone_baselines as Record<string, unknown> | null) ??
        savedProfile?.zoneBaselines ??
        null;

      // Compute personal blink kinematics from the 2-minute stream (or fallback to backend values)
      let durMed =
        (spatialProf?.blink_duration_median_ms as number) ??
        savedProfile?.blinkDurationMedianMs ??
        235.0;
      let durMad =
        (spatialProf?.blink_duration_mad_ms as number) ??
        savedProfile?.blinkDurationMadMs ??
        40.0;
      let rcrMed =
        (spatialProf?.blink_rcr_median as number) ??
        savedProfile?.blinkRcrMedian ??
        0.95;
      let rcrMad =
        (spatialProf?.blink_rcr_mad as number) ??
        savedProfile?.blinkRcrMad ??
        0.18;
      let freqMed =
        (spatialProf?.blink_freq_median_per_min as number) ?? 16.0;
      let freqMad =
        (spatialProf?.blink_freq_mad_per_min as number) ?? 4.5;
      let ampMed =
        (spatialProf?.blink_amplitude_median as number) ?? 0.14;

      if (blinks.length >= 2) {
        const durs = blinks.map((b) => Math.min(520, Math.max(80, b.durationMs)));
        const rcrs = blinks.map((b) => Math.min(3.0, Math.max(0.3, b.rcr)));
        const amps = blinks.map((b) => Math.min(0.35, Math.max(0.05, b.amplitude)));
        const [dMed, dMad] = computeMedianAndScaledMad(durs, 35.0);
        const [rMed, rMad] = computeMedianAndScaledMad(rcrs, 0.14);
        const [aMed] = computeMedianAndScaledMad(amps, 0.02);
        durMed = Number(dMed.toFixed(1));
        durMad = Number(dMad.toFixed(1));
        rcrMed = Number(rMed.toFixed(2));
        rcrMad = Number(rMad.toFixed(2));
        ampMed = Number(aMed.toFixed(3));

        const effectiveMin = Math.max(0.25, elapsedSec / 60.0);
        const overallRate = blinks.length / effectiveMin;
        const ibiRates: number[] = [];
        for (let i = 1; i < blinks.length; i++) {
          const ibiMs = blinks[i].timestampMs - blinks[i - 1].timestampMs;
          if (ibiMs >= 1200 && ibiMs <= 25000) {
            ibiRates.push(60000.0 / ibiMs);
          }
        }
        if (ibiRates.length >= 2) {
          const [ibiMed, ibiMad] = computeMedianAndScaledMad(ibiRates, 3.5);
          freqMed = Number(
            Math.min(38.0, Math.max(5.0, 0.65 * overallRate + 0.35 * ibiMed)).toFixed(1)
          );
          freqMad = Number(Math.min(10.0, Math.max(3.5, ibiMad)).toFixed(1));
        } else {
          freqMed = Number(Math.min(35.0, Math.max(6.0, overallRate)).toFixed(1));
          freqMad = 4.5;
        }
      } else if (blinks.length === 1) {
        durMed = Number(
          Math.min(310, Math.max(170, 0.6 * blinks[0].durationMs + 0.4 * 235.0)).toFixed(1)
        );
        durMad = 40.0;
        rcrMed = Number(
          Math.min(1.8, Math.max(0.55, 0.6 * blinks[0].rcr + 0.4 * 0.95)).toFixed(2)
        );
        rcrMad = 0.18;
      } else if (durMed <= 190) {
        // Prevent legacy 185ms fallback from causing false alarms on normal blinks
        durMed = 235.0;
        durMad = 40.0;
      }

      const perclosArr = perclosSamplesRef.current;
      const basePerclos =
        perclosArr.length >= 5
          ? Number(
              Math.min(
                9.5,
                Math.max(
                  1.2,
                  perclosArr.reduce((acc, v) => acc + v, 0) / perclosArr.length
                )
              ).toFixed(2)
            )
          : Number(
              Math.min(8.5, Math.max(1.5, ((freqMed * durMed) / 60000.0) * 100.0)).toFixed(2)
            );

      const enrichedZones: Record<string, unknown> = {
        ...(zoneMap && typeof zoneMap === "object" ? zoneMap : {}),
        _temporalBaseline: {
          blinkFreqMedianPerMin: freqMed,
          blinkFreqMadPerMin: freqMad,
          blinkDurationMedianMs: durMed,
          blinkDurationMadMs: durMad,
          blinkRcrMedian: rcrMed,
          blinkRcrMad: rcrMad,
          baselinePerclosPct: basePerclos,
          blinkAmplitudeMedian: ampMed,
          temporalBlinksSampled: blinks.length,
          temporalWindowSec: Math.round(elapsedSec || BLINK_STREAM_TARGET_SEC),
          calibratedAt: new Date().toISOString(),
        },
      };

      const modeLabel =
        workflowRef.current === "FULL_4ZONE_AND_BLINK"
          ? "FOUR_ZONE_PLUS_BLINK"
          : workflowRef.current === "BLINK_2MIN_ONLY"
          ? "FOUR_ZONE_PLUS_BLINK"
          : "4_ZONE";

      const nextProfile: InitialDriverProfileData = {
        isCalibrated: true,
        calibrationMode: modeLabel,
        centerEarMedian: centerEar,
        centerEarMadScaled: centerEarMad,
        blinkDurationMedianMs: durMed,
        blinkDurationMadMs: durMad,
        blinkRcrMedian: rcrMed,
        blinkRcrMad: rcrMad,
        neutralPitchDeg:
          (spatialProf?.neutral_pitch_deg as number) ??
          savedProfile?.neutralPitchDeg ??
          0.0,
        neutralYawDeg:
          (spatialProf?.neutral_yaw_deg as number) ??
          savedProfile?.neutralYawDeg ??
          0.0,
        calibrationReliability:
          (spatialProf?.calibration_reliability as number) ??
          savedProfile?.calibrationReliability ??
          0.92,
        zoneBaselines: enrichedZones,
      };

      setSavedProfile(nextProfile);
      setIsSuccess(true);

      fetch("/api/calibration/save", {
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
          temporalBlinksSampled: blinks.length,
          temporalWindowSec: Math.round(elapsedSec || BLINK_STREAM_TARGET_SEC),
          neutralPitchDeg: nextProfile.neutralPitchDeg,
          neutralYawDeg: nextProfile.neutralYawDeg,
          calibrationReliability: nextProfile.calibrationReliability,
          zoneBaselines: enrichedZones,
        }),
      })
        .then(() => {
          toast.success(
            `Personal 4-Zone + Blink Baseline Saved (${freqMed.toFixed(1)} blinks/min · ${Math.round(durMed)}ms)!`,
            { id: "calibration-completed-toast" }
          );
          if (onCalibrationSaved) {
            onCalibrationSaved(nextProfile);
          }
        })
        .catch((e) => console.warn("Could not save profile to DB:", e));
    },
    [onCalibrationSaved, savedProfile]
  );

  // Start the 2-Minute Natural Blink Rate & Cadence Stream
  const beginTwoMinuteBlinkStream = useCallback((spatialProfile: Record<string, any> | null) => {
    pendingSpatialProfileRef.current = spatialProfile;
    capturedBlinksRef.current = [];
    setCapturedBlinks([]);
    blinkElapsedSecRef.current = 0;
    setBlinkElapsedSec(0);
    lastSeenBlinkDurRef.current = null;
    lastBlinkRecordedAtMsRef.current = 0;
    eyeClosedStartMsRef.current = null;
    perclosSamplesRef.current = [];
    isBlinkStreamActiveRef.current = true;
    setIsBlinkStreamActive(true);
    toast.info(
      "4-Zone spatial alignment locked! Starting 2-Minute Natural Blink Rate & Cadence stream — look ahead and blink naturally.",
      { id: "blink-stream-start-toast" }
    );
  }, []);

  // Process incoming telemetry frames during the 2-Minute Natural Blink Rate Stream
  const handleLiveTelemetry = useCallback((t: FrameTelemetry) => {
    if (!isBlinkStreamActiveRef.current) return;

    setIsFaceVisibleInStream(t.face_detected);
    if (!t.face_detected) return;

    const em = t.eye_metrics;
    if (!em) return;

    setLiveEarValue(em.ear_raw);
    if (typeof em.perclos_pct === "number") {
      setLivePerclosPct(em.perclos_pct);
      perclosSamplesRef.current.push(em.perclos_pct);
    }

    const nowMs = Date.now();

    // Method A: Server-emitted completed blink event
    if (
      typeof em.blink_duration_ms === "number" &&
      em.blink_duration_ms >= 70 &&
      em.blink_duration_ms <= 650 &&
      em.blink_duration_ms !== lastSeenBlinkDurRef.current &&
      nowMs - lastBlinkRecordedAtMsRef.current >= 350
    ) {
      lastSeenBlinkDurRef.current = em.blink_duration_ms;
      lastBlinkRecordedAtMsRef.current = nowMs;
      const sample: CapturedBlinkSample = {
        timestampMs: nowMs,
        durationMs: em.blink_duration_ms,
        rcr: typeof em.blink_rcr === "number" && em.blink_rcr > 0 ? em.blink_rcr : 0.95,
        amplitude: Math.max(0.08, 0.285 - Math.min(0.22, em.ear_raw)),
      };
      const next = [...capturedBlinksRef.current, sample];
      capturedBlinksRef.current = next;
      setCapturedBlinks(next);
      eyeClosedStartMsRef.current = null;
      return;
    }

    // Method B: Client edge-transition detector if two consecutive blinks had identical rounded duration
    const isClosed = Boolean(em.is_eye_closed || em.ear_raw < em.ear_closure_threshold);
    if (isClosed) {
      if (eyeClosedStartMsRef.current === null) {
        eyeClosedStartMsRef.current = nowMs;
        minEarDuringClosureRef.current = em.ear_raw;
      } else {
        minEarDuringClosureRef.current = Math.min(minEarDuringClosureRef.current, em.ear_raw);
      }
    } else if (eyeClosedStartMsRef.current !== null) {
      const durMs = nowMs - eyeClosedStartMsRef.current;
      eyeClosedStartMsRef.current = null;
      if (durMs >= 75 && durMs <= 600 && nowMs - lastBlinkRecordedAtMsRef.current >= 400) {
        lastBlinkRecordedAtMsRef.current = nowMs;
        const sample: CapturedBlinkSample = {
          timestampMs: nowMs,
          durationMs: durMs,
          rcr: typeof em.blink_rcr === "number" && em.blink_rcr > 0 ? em.blink_rcr : 0.95,
          amplitude: Math.max(0.06, em.ear_raw - minEarDuringClosureRef.current),
        };
        const next = [...capturedBlinksRef.current, sample];
        capturedBlinksRef.current = next;
        setCapturedBlinks(next);
      }
    }
  }, []);

  // 1-Second Ticker for the 2-Minute Natural Blink Stream
  useEffect(() => {
    if (!isBlinkStreamActive) return;
    const interval = setInterval(() => {
      if (!isBlinkStreamActiveRef.current) return;
      const nextSec = blinkElapsedSecRef.current + 1;
      blinkElapsedSecRef.current = nextSec;
      setBlinkElapsedSec(nextSec);

      if (nextSec >= BLINK_STREAM_TARGET_SEC) {
        finalizeAndPersistProfile(
          pendingSpatialProfileRef.current,
          capturedBlinksRef.current,
          nextSec
        );
      }
    }, 1000);
    return () => clearInterval(interval);
  }, [isBlinkStreamActive, finalizeAndPersistProfile]);

  // 2. Real-Time LiveKit WebRTC Streaming & Calibration Hook
  const {
    isConnected: isLivekitConnected,
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
    onTelemetry: handleLiveTelemetry,
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

      if (cal.is_completed && !cal.rejection_reason) {
        const prof = (cal as unknown as { profile?: Record<string, unknown> })?.profile ?? null;
        // If user selected FULL_4ZONE_AND_BLINK and backend completed spatial zones in 4 steps
        // without running Phase 5 natively, seamlessly launch the 2-Minute Blink Stream!
        if (
          workflowRef.current === "FULL_4ZONE_AND_BLINK" &&
          (cal.total_phases <= 4 || !prof?.temporal_blinks_sampled)
        ) {
          beginTwoMinuteBlinkStream(prof);
          return;
        }

        // Otherwise finalize immediately
        finalizeAndPersistProfile(
          prof,
          capturedBlinksRef.current,
          cal.elapsed_sec || BLINK_STREAM_TARGET_SEC
        );
      } else if (cal.is_completed && cal.rejection_reason) {
        setIsSuccess(false);
      }
    },
  });

  // Keep calState in sync with incoming LiveKit calibration data
  useEffect(() => {
    if (livekitCalibration && !isBlinkStreamActive) {
      setCalState(livekitCalibration);
    }
  }, [livekitCalibration, isBlinkStreamActive]);

  // Reactive Video Streaming: Link camera activation with WebRTC publisher
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

  // User Actions
  const handleStartCalibration = async (
    overrideMode?: "FULL_4ZONE_AND_BLINK" | "QUICK_4ZONE_ONLY" | "BLINK_2MIN_ONLY"
  ) => {
    const modeToRun = overrideMode || calibrationWorkflow;
    if (overrideMode) {
      setCalibrationWorkflow(overrideMode);
      workflowRef.current = overrideMode;
    }

    hasHandledCompletionRef.current = false;
    setIsSuccess(false);
    setCalState(null);
    isBlinkStreamActiveRef.current = false;
    setIsBlinkStreamActive(false);

    if (cameraState !== "active") {
      await startCamera();
    }

    if (modeToRun === "BLINK_2MIN_ONLY") {
      beginTwoMinuteBlinkStream(null);
      return;
    }

    const includeBlink = modeToRun === "FULL_4ZONE_AND_BLINK";
    triggerLivekitCalibration(true, 5.0, includeBlink, BLINK_STREAM_TARGET_SEC);
    toast.info(
      includeBlink
        ? "Starting 4-Zone Spatial + 2-Minute Natural Blink Rate Calibration..."
        : "Starting Quick 4-Zone Spatial Calibration..."
    );
  };

  const handleStopCalibration = () => {
    isBlinkStreamActiveRef.current = false;
    setIsBlinkStreamActive(false);
    triggerLivekitCancel();
    stopCamera();
    stopStreamingLoop();
    setCalState(null);
  };

  const handleSkipZone = () => {
    if (isBlinkStreamActive) {
      finalizeAndPersistProfile(
        pendingSpatialProfileRef.current,
        capturedBlinksRef.current,
        blinkElapsedSecRef.current
      );
      return;
    }
    triggerLivekitSkip();
    toast.info("Advancing past current calibration stage...");
  };

  const handleRecalibrateZone = async (zone: string) => {
    hasHandledCompletionRef.current = false;
    setIsSuccess(false);
    isBlinkStreamActiveRef.current = false;
    setIsBlinkStreamActive(false);
    if (cameraState !== "active") {
      await startCamera();
    }
    if (zone === "BLINK_CADENCE") {
      setCalibrationWorkflow("BLINK_2MIN_ONLY");
      workflowRef.current = "BLINK_2MIN_ONLY";
      beginTwoMinuteBlinkStream(null);
      return;
    }
    triggerLivekitRecalibrate(zone, 5.0);
    toast.info(`Recalibrating ${zone} zone...`);
  };

  // Live Blink Rate & Duration Metrics during 2-Min Stream
  const liveBlinkStats = useMemo(() => {
    const count =
      isBlinkStreamActive
        ? capturedBlinks.length
        : calState?.collected_blinks_count ?? capturedBlinks.length;
    const elapsed =
      isBlinkStreamActive
        ? blinkElapsedSec
        : calState?.elapsed_sec ?? blinkElapsedSec;
    const elapsedMin = Math.max(0.15, elapsed / 60.0);
    const ratePerMin =
      isBlinkStreamActive
        ? count > 0
          ? Number(Math.min(55, count / elapsedMin).toFixed(1))
          : 0.0
        : calState?.live_blink_rate_per_min ?? 0.0;
    const durs = capturedBlinks.map((b) => b.durationMs);
    const [medDur] = durs.length ? computeMedianAndScaledMad(durs, 35) : [null];
    const rcrs = capturedBlinks.map((b) => b.rcr);
    const [medRcr] = rcrs.length ? computeMedianAndScaledMad(rcrs, 0.14) : [null];

    const remainingSec = Math.max(0, BLINK_STREAM_TARGET_SEC - Math.round(elapsed));
    const mm = String(Math.floor(remainingSec / 60)).padStart(2, "0");
    const ss = String(remainingSec % 60).padStart(2, "0");
    const progressPct = Math.min(100, Math.round((elapsed / BLINK_STREAM_TARGET_SEC) * 100));

    return {
      count,
      elapsed,
      remainingFormatted: `${mm}:${ss}`,
      progressPct,
      ratePerMin,
      medianDurationMs:
        medDur !== null
          ? Math.round(medDur)
          : calState?.live_median_blink_duration_ms
          ? Math.round(calState.live_median_blink_duration_ms)
          : null,
      medianRcr:
        medRcr !== null
          ? medRcr.toFixed(2)
          : calState?.live_median_blink_rcr
          ? calState.live_median_blink_rcr.toFixed(2)
          : null,
    };
  }, [blinkElapsedSec, calState, capturedBlinks, isBlinkStreamActive]);

  const isInBlinkPhase =
    isBlinkStreamActive ||
    calState?.current_zone === "BLINK_CADENCE" ||
    calState?.phase_type === "BLINK_CADENCE";

  // Directional guidance helpers
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
    (isBlinkStreamActive || Boolean(calState && !calState.is_completed));

  // Extract stored temporal baseline metadata for the Baseline Inspector
  const storedTemporal = useMemo(() => {
    const rawZones = (savedProfile?.zoneBaselines || {}) as Record<string, any>;
    const meta = (rawZones._temporalBaseline || {}) as Record<string, any>;
    return {
      blinkFreqMedianPerMin:
        typeof meta.blinkFreqMedianPerMin === "number" ? meta.blinkFreqMedianPerMin : 16.0,
      blinkFreqMadPerMin:
        typeof meta.blinkFreqMadPerMin === "number" ? meta.blinkFreqMadPerMin : 4.5,
      blinkDurationMedianMs: savedProfile?.blinkDurationMedianMs ?? 235.0,
      blinkDurationMadMs: savedProfile?.blinkDurationMadMs ?? 40.0,
      blinkRcrMedian: savedProfile?.blinkRcrMedian ?? 0.95,
      blinkRcrMad: savedProfile?.blinkRcrMad ?? 0.18,
      baselinePerclosPct:
        typeof meta.baselinePerclosPct === "number" ? meta.baselinePerclosPct : 3.5,
      temporalBlinksSampled:
        typeof meta.temporalBlinksSampled === "number" ? meta.temporalBlinksSampled : null,
      hasCustomBlinkStream: Boolean(meta.calibratedAt),
    };
  }, [savedProfile]);

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
                  isCalibrating
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : "text-muted-foreground"
                }`}
              >
                {isCalibrating ? (
                  <>
                    <Wifi className="size-3" />
                    {isInBlinkPhase ? "BLINK STREAM" : "CALIBRATING"}
                  </>
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
                    First, align your gaze across the 4 cabin zones (~20s). Then sit naturally for a{" "}
                    <strong className="text-foreground">2-minute stream</strong> while DriveGuard learns
                    your personal blink rate, blink duration, and eyelid reopening speed.
                  </p>

                  {/* Workflow Selector */}
                  <div className="flex flex-wrap items-center justify-center gap-1.5 p-1 bg-muted/60 rounded-xl border border-border text-xs">
                    <button
                      type="button"
                      onClick={() => setCalibrationWorkflow("FULL_4ZONE_AND_BLINK")}
                      className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                        calibrationWorkflow === "FULL_4ZONE_AND_BLINK"
                          ? "bg-primary text-primary-foreground shadow-xs"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      4-Zone + 2-Min Blink Stream (Recommended)
                    </button>
                    <button
                      type="button"
                      onClick={() => setCalibrationWorkflow("BLINK_2MIN_ONLY")}
                      className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                        calibrationWorkflow === "BLINK_2MIN_ONLY"
                          ? "bg-primary text-primary-foreground shadow-xs"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      2-Min Blink Rate Only
                    </button>
                    <button
                      type="button"
                      onClick={() => setCalibrationWorkflow("QUICK_4ZONE_ONLY")}
                      className={`px-3 py-1.5 rounded-lg transition-all font-medium ${
                        calibrationWorkflow === "QUICK_4ZONE_ONLY"
                          ? "bg-primary text-primary-foreground shadow-xs"
                          : "text-muted-foreground hover:text-foreground"
                      }`}
                    >
                      Quick 4-Zone Spatial Only
                    </button>
                  </div>

                  <Button
                    size="sm"
                    onClick={() => handleStartCalibration()}
                    className="bg-primary text-primary-foreground"
                  >
                    <Play className="size-4 mr-1.5" />
                    {calibrationWorkflow === "FULL_4ZONE_AND_BLINK"
                      ? "Start 4-Zone + 2-Min Blink Calibration"
                      : calibrationWorkflow === "BLINK_2MIN_ONLY"
                      ? "Start 2-Minute Blink Rate Stream"
                      : "Start Quick 4-Zone Calibration"}
                  </Button>
                </div>
              )}

              {/* In-Video Heads Up Status Banner (Spatial Phase) */}
              {cameraState === "active" && calState && !isInBlinkPhase && !isSuccess && (
                <div className="absolute top-4 left-4 right-4 flex items-start justify-between pointer-events-none z-20">
                  <div className="px-3 py-1 rounded-lg bg-black/80 backdrop-blur-md border border-white/15 text-xs text-white font-mono flex items-center gap-2">
                    <span>Zone {calState.phase_index}/4:</span>
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
                    {calState.is_head_in_target_zone ? "ALIGNED [ACQUIRING]" : "TURN HEAD TO TARGET"}
                  </Badge>
                </div>
              )}

              {/* In-Video Heads Up Status Banner (2-Minute Blink Stream Phase) */}
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
                        isFaceVisibleInStream
                          ? "border-emerald-500/60 text-emerald-400 bg-black/80"
                          : "border-rose-500/60 text-rose-400 bg-black/85"
                      }`}
                    >
                      {isFaceVisibleInStream
                        ? `EAR ${liveEarValue.toFixed(3)} · TRACKING BLINKS`
                        : "KEEP FACE IN CAMERA"}
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
                    Personal 4-Zone & Blink Cadence Profile Saved!
                  </div>
                  <p className="text-xs text-muted-foreground max-w-md">
                    Your 4-zone spatial EAR thresholds, personal blink rate (
                    <strong className="text-foreground">
                      {storedTemporal.blinkFreqMedianPerMin.toFixed(1)} blinks/min
                    </strong>
                    ), and personal blink duration (
                    <strong className="text-foreground">
                      {Math.round(storedTemporal.blinkDurationMedianMs)} ms
                    </strong>
                    ) are now active in PostgreSQL and the AI engine.
                  </p>
                  <div className="flex flex-wrap items-center justify-center gap-2 pt-2">
                    <Button
                      size="sm"
                      onClick={() => handleStartCalibration("FULL_4ZONE_AND_BLINK")}
                      variant="outline"
                    >
                      <RotateCcw className="size-4 mr-1.5" />
                      Run Full Recalibration
                    </Button>
                    <Button
                      size="sm"
                      onClick={() => handleStartCalibration("BLINK_2MIN_ONLY")}
                      variant="secondary"
                      className="text-xs"
                    >
                      <Timer className="size-3.5 mr-1.5" />
                      Repeat 2-Min Blink Stream
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
                  {isInBlinkPhase ? (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleSkipZone}
                      disabled={liveBlinkStats.elapsed < 15 && liveBlinkStats.count < 3}
                      title="Save captured blinks and complete calibration early"
                    >
                      <CheckCircle2 className="size-3.5 mr-1.5 text-emerald-500" />
                      {liveBlinkStats.elapsed < 15 && liveBlinkStats.count < 3
                        ? `Sampling Blinks (${Math.max(0, 15 - Math.round(liveBlinkStats.elapsed))}s min)...`
                        : "Finish & Save Blink Baseline Now"}
                    </Button>
                  ) : (
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={handleSkipZone}
                      title="Skip current spatial zone"
                    >
                      <FastForward className="size-3.5 mr-1.5 text-primary" />
                      Skip Zone
                    </Button>
                  )}
                </>
              ) : (
                <Button
                  size="sm"
                  onClick={() => handleStartCalibration()}
                  className="bg-primary text-primary-foreground"
                >
                  <Play className="size-3.5 mr-1.5" />
                  {isSuccess ? "Recalibrate" : "Start Calibration"}
                </Button>
              )}
            </div>
            <span className="text-xs text-muted-foreground font-mono">
              {isInBlinkPhase
                ? `Stage 5/5 · Blink Stream (${liveBlinkStats.remainingFormatted} left)`
                : calState
                ? `Stage ${calState.phase_index} of ${
                    calibrationWorkflow === "FULL_4ZONE_AND_BLINK" ? 5 : calState.total_phases
                  }`
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
                  : "Align your head with each target zone until 100% locked"}
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
                    {currentZoneInfo.label}
                  </span>
                </div>
              </div>

              {calState?.rejection_reason && (
                <div className="p-3 rounded-xl bg-destructive/10 border border-destructive/30 text-destructive text-xs space-y-2">
                  <div>
                    <span className="font-semibold block">⚠️ Calibration Session Rejected:</span>
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

              {isInBlinkPhase ? (
                <>
                  {/* 2-Minute Countdown Progress */}
                  <div className="space-y-1.5 pt-1">
                    <div className="flex justify-between font-mono">
                      <span className="text-muted-foreground">2-Min Stream Progress:</span>
                      <span className="font-semibold text-emerald-500">
                        {liveBlinkStats.progressPct}% ({liveBlinkStats.remainingFormatted} left)
                      </span>
                    </div>
                    <Progress value={liveBlinkStats.progressPct} className="h-2" />
                  </div>

                  {/* Live Blink Cadence Readouts */}
                  <div className="grid grid-cols-2 gap-2.5 pt-2 border-t border-border font-mono">
                    <div className="p-2.5 rounded-lg border border-border bg-muted/20">
                      <span className="text-[10px] text-muted-foreground block">
                        BLINKS SAMPLED
                      </span>
                      <span className="text-base font-bold text-foreground">
                        {liveBlinkStats.count}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg border border-border bg-muted/20">
                      <span className="text-[10px] text-muted-foreground block">
                        PERSONAL BLINK RATE
                      </span>
                      <span className="text-base font-bold text-sky-500">
                        {liveBlinkStats.ratePerMin.toFixed(1)} /min
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg border border-border bg-muted/20">
                      <span className="text-[10px] text-muted-foreground block">
                        MEDIAN DURATION
                      </span>
                      <span className="text-base font-bold text-amber-500">
                        {liveBlinkStats.medianDurationMs !== null
                          ? `${liveBlinkStats.medianDurationMs} ms`
                          : "Sampling..."}
                      </span>
                    </div>
                    <div className="p-2.5 rounded-lg border border-border bg-muted/20">
                      <span className="text-[10px] text-muted-foreground block">
                        MEDIAN RCR / PERCLOS
                      </span>
                      <span className="text-sm font-bold text-emerald-500">
                        {liveBlinkStats.medianRcr ?? "0.95"} · {livePerclosPct.toFixed(1)}%
                      </span>
                    </div>
                  </div>

                  <div className="p-2.5 rounded-lg bg-emerald-500/5 border border-emerald-500/20 text-[11px] text-muted-foreground space-y-1">
                    <span className="font-semibold text-foreground flex items-center gap-1.5">
                      <Sparkles className="size-3.5 text-emerald-500" />
                      Why 2 Minutes?
                    </span>
                    <span>
                      Normal blink rate varies from 8 to 28 blinks/min across individuals, and webcam
                      blink duration averages 210–280 ms. Profiling 2 minutes of natural blinking
                      eliminates false &ldquo;Blink Duration Above Personal Baseline&rdquo; alerts.
                    </span>
                  </div>
                </>
              ) : (
                <>
                  {/* Spatial Alignment Status Indicator */}
                  <div className="space-y-1.5">
                    <div className="flex justify-between items-center">
                      <span className="text-muted-foreground">Spatial Alignment:</span>
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
                          : "PLEASE TURN HEAD"}
                      </Badge>
                    </div>
                    <p className="text-[11px] text-muted-foreground">
                      {calState?.instruction ||
                        "Click Start Calibration to align your 4 cabin gaze zones and 2-minute blink baseline."}
                    </p>
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
                      <span>0% (Paused if Looking Away)</span>
                      <span>100% (Locked)</span>
                    </div>
                  </div>

                  {/* Overall Spatial Progress */}
                  <div className="space-y-1.5 pt-2 border-t border-border">
                    <div className="flex justify-between font-mono">
                      <span className="text-muted-foreground">4-Zone Spatial Progress:</span>
                      <span className="font-semibold text-foreground">
                        {calState ? `${Math.round(calState.total_progress_pct)}%` : "0%"}
                      </span>
                    </div>
                    <Progress value={calState?.total_progress_pct || 0} className="h-1.5" />
                  </div>

                  <div className="p-2.5 rounded-lg bg-muted/40 border border-border text-[11px] text-muted-foreground space-y-1">
                    <span className="font-semibold text-foreground block">💡 Calibration Tip:</span>
                    <span>
                      Turn your head naturally toward each target zone until the bar reaches 100%.
                      Right after Zone 4 (Dashboard), the 2-Minute Natural Blink Stream begins
                      automatically.
                    </span>
                  </div>
                </>
              )}
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
                Stored in Neon PostgreSQL (`DriverProfile`) and synced to the real-time AI Risk Engine.
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
              {savedProfile?.isCalibrated ? "CALIBRATED PROFILE ACTIVE" : "UNCALIBRATED DEFAULTS"}
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
                    {storedTemporal.hasCustomBlinkStream ? "2-Min Calibrated" : "Baseline"}
                  </Badge>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                  <span className="text-[11px] text-muted-foreground">Personal Median:</span>
                  <span className="text-base font-bold text-sky-500">
                    {storedTemporal.blinkFreqMedianPerMin.toFixed(1)} /min
                  </span>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums">
                  <span className="text-[11px] text-muted-foreground">Robust MAD Spread:</span>
                  <span className="text-xs font-semibold text-foreground">
                    ±{storedTemporal.blinkFreqMadPerMin.toFixed(1)} /min
                  </span>
                </div>
              </div>

              <div className="p-3.5 rounded-xl border border-primary/25 bg-primary/5 flex flex-col justify-between gap-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-bold text-foreground">
                    BLINK DURATION
                  </span>
                  <Badge variant="secondary" className="text-[10px] font-mono">
                    Personal Cutoff
                  </Badge>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                  <span className="text-[11px] text-muted-foreground">Personal Median:</span>
                  <span className="text-base font-bold text-amber-500">
                    {Math.round(storedTemporal.blinkDurationMedianMs)} ms
                  </span>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums">
                  <span className="text-[11px] text-muted-foreground">Alert Cutoff (2.2z):</span>
                  <span className="text-xs font-semibold text-emerald-500">
                    ≥{" "}
                    {Math.round(
                      Math.max(
                        310,
                        storedTemporal.blinkDurationMedianMs +
                          2.2 * Math.max(35, storedTemporal.blinkDurationMadMs)
                      )
                    )}{" "}
                    ms
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
                    {storedTemporal.blinkRcrMedian.toFixed(2)}
                  </span>
                </div>
                <div className="flex items-baseline justify-between font-mono tabular-nums">
                  <span className="text-[11px] text-muted-foreground">Robust MAD:</span>
                  <span className="text-xs font-semibold text-foreground">
                    ±{storedTemporal.blinkRcrMad.toFixed(2)}
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
                        : "Alert Anchor"}
                    </Badge>
                  </div>
                  <div className="flex items-baseline justify-between font-mono tabular-nums pt-1">
                    <span className="text-[11px] text-muted-foreground">Resting Baseline:</span>
                    <span className="text-base font-bold text-emerald-500">
                      {storedTemporal.baselinePerclosPct.toFixed(1)}%
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
                const zb = zbMap?.[zone];
                const earMed =
                  zb?.ear_median ??
                  (zone === "CENTER" && savedProfile?.isCalibrated
                    ? savedProfile.centerEarMedian
                    : null);
                const thresh =
                  zb?.dynamic_closure_threshold ??
                  (earMed ? Number((earMed * 0.72).toFixed(3)) : null);

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
                          {earMed !== null ? "Calibrated" : "Default"}
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
