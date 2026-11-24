"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  AlertTransition,
  BrowserDetectionEngine,
  CalibrationProgress,
  DecisionResult,
  DetectionTelemetry,
  DriverBehaviourProfileData,
  EngineStatus,
  PersistenceSyncStatus,
} from "@/lib/ai";

export interface UseBrowserDetectionOptions {
  driverId?: string;
  initialProfile?: DriverBehaviourProfileData | unknown | null;
  isMirrored?: boolean;
  targetInferenceFps?: number;
  autoStartSession?: boolean;
  onSafetyAlert?: (
    decision: DecisionResult,
    telemetry: DetectionTelemetry,
    transition?: AlertTransition
  ) => void;
  onCalibrationProgress?: (progress: CalibrationProgress) => void;
  onError?: (err: Error) => void;
}

export function useBrowserDetection(options: UseBrowserDetectionOptions = {}) {
  const {
    driverId = "driver_default",
    initialProfile = null,
    isMirrored = false,
    targetInferenceFps = 12,
    autoStartSession = false,
    onSafetyAlert,
    onCalibrationProgress,
    onError,
  } = options;

  const [engineStatus, setEngineStatus] = useState<EngineStatus>("idle");
  const [telemetry, setTelemetry] = useState<DetectionTelemetry | null>(null);
  const [calibration, setCalibration] = useState<CalibrationProgress | null>(null);
  const [sessionCode, setSessionCode] = useState<string | null>(null);
  const [syncStatus, setSyncStatus] = useState<PersistenceSyncStatus>("synced");

  const engineRef = useRef<BrowserDetectionEngine | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastStateUpdateMsRef = useRef<number>(0);
  const runIdRef = useRef<number>(0);

  // Initialize engine instance
  useEffect(() => {
    const engine = new BrowserDetectionEngine({
      driverId,
      profile: initialProfile,
      isMirrored,
      targetInferenceFps,
      onTelemetry: (t) => {
        // Ignore stray telemetry emitted after monitoring has stopped
        if (engineRef.current?.status !== "running") return;
        // Throttle React state updates to ~10 Hz to prevent React re-render churn
        const now = performance.now();
        if (now - lastStateUpdateMsRef.current >= 90) {
          lastStateUpdateMsRef.current = now;
          setTelemetry({ ...t });
          setSyncStatus(t.persistenceSyncStatus);
        }
      },
      onSafetyAlert: (decision, t, transition) => {
        if (engineRef.current?.status !== "running") return;
        // Immediate unthrottled alert dispatch (emitted only on AlertStateManager transitions/cooldowns)
        if (onSafetyAlert) {
          onSafetyAlert(decision, t, transition);
        }
      },
      onCalibrationProgress: (p) => {
        if (engineRef.current?.status !== "running") return;
        setCalibration(p);
        if (onCalibrationProgress) {
          onCalibrationProgress(p);
        }
      },
      onError: (err) => {
        setEngineStatus("error");
        if (onError) onError(err);
      },
    });

    engineRef.current = engine;

    return () => {
      engine.dispose();
      engineRef.current = null;
    };
  }, [driverId, initialProfile, isMirrored, targetInferenceFps, onSafetyAlert, onCalibrationProgress, onError]);

  // Sync profile updates to active engine
  useEffect(() => {
    if (engineRef.current && initialProfile) {
      engineRef.current.setProfile(initialProfile);
    }
  }, [initialProfile]);

  // Sync mirroring updates to active engine
  useEffect(() => {
    if (engineRef.current) {
      engineRef.current.setIsMirrored(isMirrored);
    }
  }, [isMirrored]);

  const startMonitoring = useCallback(
    async (videoElement: HTMLVideoElement) => {
      if (!engineRef.current) return;
      const currentRunId = ++runIdRef.current;
      videoRef.current = videoElement;
      setTelemetry(null);
      setCalibration(null);
      lastStateUpdateMsRef.current = 0;

      try {
        setEngineStatus("initializing");
        await engineRef.current.initialize();

        // Abort if stopMonitoring() was called while model initialization was in flight
        if (currentRunId !== runIdRef.current || !engineRef.current) {
          return;
        }

        // If autoStartSession is enabled, request session code from Next.js API
        let activeCode: string | null = null;
        if (autoStartSession) {
          try {
            const res = await fetch("/api/sessions/start", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ engineType: "BROWSER" }),
            });
            if (currentRunId !== runIdRef.current || !engineRef.current) {
              return;
            }
            if (res.ok) {
              const data = await res.json();
              activeCode = data.sessionCode;
              setSessionCode(activeCode);
              engineRef.current.setSessionCode(activeCode);
            }
          } catch (sessionErr) {
            console.warn("[useBrowserDetection] Failed to start backend session:", sessionErr);
          }
        }

        if (currentRunId !== runIdRef.current || !engineRef.current) {
          return;
        }

        engineRef.current.start(videoElement);
        setEngineStatus("running");
      } catch (err) {
        if (currentRunId !== runIdRef.current) {
          return;
        }
        setEngineStatus("error");
        if (onError) onError(err instanceof Error ? err : new Error(String(err)));
      }
    },
    [autoStartSession, onError]
  );

  const pauseMonitoring = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.pause();
      setEngineStatus("paused");
    }
  }, []);

  const resumeMonitoring = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.resume();
      setEngineStatus("running");
    }
  }, []);

  const stopMonitoring = useCallback(() => {
    runIdRef.current++;
    if (engineRef.current) {
      engineRef.current.stop();
    }
    setEngineStatus("stopped");
    setTelemetry(null);
    setCalibration(null);
    setSessionCode(null);
    lastStateUpdateMsRef.current = 0;
  }, []);

  const startCalibration = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.startCalibration();
    }
  }, []);

  const cancelCalibration = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.cancelCalibration();
      setCalibration(null);
    }
  }, []);

  const queueSessionEnd = useCallback((durationSec: number) => {
    if (engineRef.current) {
      engineRef.current.queueSessionEnd(durationSec);
    }
  }, []);

  const getMetrics = useCallback(() => {
    if (engineRef.current) {
      return engineRef.current.retrievePerformanceMetrics();
    }
    return null;
  }, []);

  const setExplicitSessionCode = useCallback((code: string | null) => {
    setSessionCode(code);
    if (engineRef.current) {
      engineRef.current.setSessionCode(code);
    }
  }, []);

  return {
    engineStatus,
    telemetry,
    calibration,
    sessionCode,
    setSessionCode: setExplicitSessionCode,
    syncStatus,
    startMonitoring,
    pauseMonitoring,
    resumeMonitoring,
    stopMonitoring,
    startCalibration,
    cancelCalibration,
    queueSessionEnd,
    getMetrics,
    engine: engineRef.current,
  };
}
