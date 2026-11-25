"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { OpticalFrameQuality } from "@/types/camera";
import {
  AlertTransition,
  BrowserDetectionEngine,
  browserProfileStore,
  BrowserProfileStorageMode,
  BrowserProfileSyncState,
  CalibrationProgress,
  DecisionResult,
  DetectionTelemetry,
  DriverBehaviourProfileData,
  EngineStatus,
  normalizeDriverProfile,
  OperationalState,
  PersistenceSyncStatus,
  PoseZone,
  ProfileSyncOutcome,
} from "@/lib/ai";

export interface UseBrowserDetectionOptions {
  driverId?: string;
  initialProfile?: DriverBehaviourProfileData | unknown | null;
  isMirrored?: boolean;
  sensitivityLevel?: string | null;
  mirrorSuppression?: boolean;
  targetInferenceFps?: number;
  autoStartSession?: boolean;
  onSafetyAlert?: (
    decision: DecisionResult,
    telemetry: DetectionTelemetry,
    transition?: AlertTransition
  ) => void;
  onCalibrationProgress?: (progress: CalibrationProgress) => void;
  onCalibrationCompleted?: (
    profile: DriverBehaviourProfileData,
    syncOutcome?: ProfileSyncOutcome
  ) => void;
  onError?: (err: Error) => void;
}

export function useBrowserDetection(options: UseBrowserDetectionOptions = {}) {
  const {
    driverId = "driver_default",
    initialProfile = null,
    isMirrored = false,
    sensitivityLevel = "STANDARD",
    mirrorSuppression = true,
    targetInferenceFps = 12,
    autoStartSession = false,
    onSafetyAlert,
    onCalibrationProgress,
    onCalibrationCompleted,
    onError,
  } = options;

  const [engineStatus, setEngineStatus] = useState<EngineStatus>("idle");
  const [telemetry, setTelemetry] = useState<DetectionTelemetry | null>(null);
  const [calibration, setCalibration] = useState<CalibrationProgress | null>(null);
  const [sessionCode, setSessionCode] = useState<string | null>(null);
  const [syncStatus, setSyncStatus] = useState<PersistenceSyncStatus>("synced");
  const [profileSyncStatus, setProfileSyncStatus] =
    useState<BrowserProfileSyncState>("LOCAL_ONLY");
  const [storageMode, setStorageMode] = useState<BrowserProfileStorageMode>(
    browserProfileStore.storageMode
  );
  const [activeProfile, setActiveProfile] = useState<DriverBehaviourProfileData | null>(null);
  const [isCalibrated, setIsCalibrated] = useState<boolean>(false);
  const [recordedFrameCount, setRecordedFrameCount] = useState<number>(0);

  const engineRef = useRef<BrowserDetectionEngine | null>(null);
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const lastStateUpdateMsRef = useRef<number>(0);
  const runIdRef = useRef<number>(0);

  // Stabilize all callbacks via refs so changing props/callbacks NEVER disposes the running engine!
  const onSafetyAlertRef = useRef(onSafetyAlert);
  const onCalibrationProgressRef = useRef(onCalibrationProgress);
  const onCalibrationCompletedRef = useRef(onCalibrationCompleted);
  const onErrorRef = useRef(onError);
  const initialProfileRef = useRef(initialProfile);
  const sensitivityLevelRef = useRef(sensitivityLevel);
  const mirrorSuppressionRef = useRef(mirrorSuppression);

  useEffect(() => {
    onSafetyAlertRef.current = onSafetyAlert;
    onCalibrationProgressRef.current = onCalibrationProgress;
    onCalibrationCompletedRef.current = onCalibrationCompleted;
    onErrorRef.current = onError;
  }, [onSafetyAlert, onCalibrationProgress, onCalibrationCompleted, onError]);

  useEffect(() => {
    sensitivityLevelRef.current = sensitivityLevel;
    engineRef.current?.setSensitivityLevel(sensitivityLevel);
  }, [sensitivityLevel]);

  useEffect(() => {
    mirrorSuppressionRef.current = mirrorSuppression;
    engineRef.current?.setMirrorSuppression(Boolean(mirrorSuppression));
  }, [mirrorSuppression]);

  // Instantiate engine strictly once per (driverId, targetInferenceFps)
  useEffect(() => {
    const engine = new BrowserDetectionEngine({
      driverId,
      profile: initialProfileRef.current,
      isMirrored: false,
      sensitivityLevel: sensitivityLevelRef.current,
      mirrorSuppression: mirrorSuppressionRef.current,
      targetInferenceFps,
      onTelemetry: (t) => {
        if (engineRef.current?.status !== "running") return;
        const now = performance.now();
        if (now - lastStateUpdateMsRef.current >= 90) {
          lastStateUpdateMsRef.current = now;
          setTelemetry({ ...t });
          setSyncStatus(t.persistenceSyncStatus);
          setRecordedFrameCount(engine.getRecordedFrameCount());
        }
      },
      onSafetyAlert: (decision, t, transition) => {
        if (engineRef.current?.status !== "running") return;
        onSafetyAlertRef.current?.(decision, t, transition);
      },
      onCalibrationProgress: (p) => {
        if (engineRef.current?.status !== "running") return;
        setCalibration(p);
        onCalibrationProgressRef.current?.(p);
      },
      onCalibrationCompleted: (profile, syncOutcome) => {
        setActiveProfile(profile);
        setIsCalibrated(true);
        if (syncOutcome?.record) {
          setProfileSyncStatus(syncOutcome.record.syncState);
        } else if (syncOutcome?.status === "SYNCED" || syncOutcome?.status === "UNCHANGED") {
          setProfileSyncStatus("SYNCED");
        } else if (syncOutcome?.status === "CONFLICT_SERVER_AUTHORITATIVE") {
          setProfileSyncStatus("CONFLICT_PENDING_REVIEW");
        } else {
          setProfileSyncStatus("PENDING_UPLOAD");
        }
        onCalibrationCompletedRef.current?.(profile, syncOutcome);
      },
      onError: (err) => {
        setEngineStatus("error");
        onErrorRef.current?.(err);
      },
    });

    engineRef.current = engine;
    setIsCalibrated(engine.isCalibrated);
    setActiveProfile(engine.activeProfile);

    // Hydrate immediately from IndexedDB (offline-ready) then reconcile asynchronously with Neon
    let cancelled = false;
    void (async () => {
      const localRecord = await browserProfileStore.getLocalProfile(driverId);
      if (cancelled || engineRef.current !== engine) return;
      setStorageMode(browserProfileStore.storageMode);

      if (localRecord?.profile) {
        engine.setProfile(localRecord.profile);
        setActiveProfile(engine.activeProfile);
        setIsCalibrated(engine.isCalibrated);
        setProfileSyncStatus(localRecord.syncState);
      } else if (initialProfileRef.current) {
        const norm = normalizeDriverProfile(initialProfileRef.current);
        if (norm.isCalibrated && norm.profile && norm.profile.engineType === "BROWSER") {
          engine.setProfile(norm.profile);
          setActiveProfile(norm.profile);
          setIsCalibrated(true);
          await browserProfileStore.saveLocalProfile(driverId, norm.profile, {
            profileUuid: norm.profile.profileUuid,
            profileVersion: norm.profile.profileVersion ?? 1,
            baseServerVersion: norm.profile.profileVersion ?? 1,
            contentHash: norm.profile.contentHash ?? null,
            syncState: "SYNCED",
          });
          setProfileSyncStatus("SYNCED");
        }
      }

      // Non-blocking background reconciliation with PostgreSQL BROWSER profile
      const reconciled = await browserProfileStore.reconcileWithServer(driverId);
      if (cancelled || engineRef.current !== engine) return;
      if (reconciled.record?.profile) {
        engine.setProfile(reconciled.record.profile);
        setActiveProfile(engine.activeProfile);
        setIsCalibrated(engine.isCalibrated);
        setProfileSyncStatus(reconciled.record.syncState);
      }
    })();

    return () => {
      cancelled = true;
      engine.dispose();
      engineRef.current = null;
    };
  }, [driverId, targetInferenceFps]);

  // Apply explicit profile updates without recreating engine
  useEffect(() => {
    initialProfileRef.current = initialProfile;
    if (!engineRef.current || !initialProfile) return;
    const norm = normalizeDriverProfile(initialProfile);
    if (norm.isCalibrated && norm.profile && norm.profile.engineType !== "PYTHON") {
      engineRef.current.setProfile(norm.profile);
      setActiveProfile(engineRef.current.activeProfile);
      setIsCalibrated(engineRef.current.isCalibrated);
    }
  }, [initialProfile]);

  // Apply mirror flag updates without recreating engine
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

        if (currentRunId !== runIdRef.current || !engineRef.current) {
          return;
        }

        if (autoStartSession) {
          let startedOnline = false;
          try {
            const res = await fetch("/api/sessions/start", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ engineType: "BROWSER" }),
              signal: AbortSignal.timeout(4000),
            });
            if (currentRunId !== runIdRef.current || !engineRef.current) {
              return;
            }
            if (res.ok) {
              const data = await res.json();
              const activeCode = data.sessionCode;
              setSessionCode(activeCode);
              engineRef.current.setSessionCode(activeCode, data.clientSessionId ?? null);
              startedOnline = true;
            }
          } catch {
            // Offline fallback below
          }

          if (!startedOnline && engineRef.current) {
            const provisional = engineRef.current.startOfflineSession();
            setSessionCode(provisional.sessionCode);
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
        onErrorRef.current?.(err instanceof Error ? err : new Error(String(err)));
      }
    },
    [autoStartSession]
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

  const startCalibration = useCallback(
    (calOpts?: { targetZone?: PoseZone | null; minSamplesPerZone?: number }) => {
      if (engineRef.current) {
        engineRef.current.startCalibration(calOpts);
      }
    },
    []
  );

  const skipCalibrationStep = useCallback(() => {
    if (engineRef.current) {
      return engineRef.current.skipCalibrationStep();
    }
    return null;
  }, []);

  const recalibrateZone = useCallback((zone: PoseZone) => {
    if (engineRef.current) {
      engineRef.current.recalibrateZone(zone);
    }
  }, []);

  const cancelCalibration = useCallback(() => {
    if (engineRef.current) {
      engineRef.current.cancelCalibration();
      setCalibration(null);
    }
  }, []);

  const setOpticalQuality = useCallback((quality: OpticalFrameQuality | null) => {
    if (engineRef.current) {
      engineRef.current.setOpticalQuality(quality);
    }
  }, []);

  const startOfflineSession = useCallback(() => {
    if (!engineRef.current) return null;
    const provisional = engineRef.current.startOfflineSession();
    setSessionCode(provisional.sessionCode);
    return provisional;
  }, []);

  const queueSessionEnd = useCallback((durationSec: number) => {
    if (engineRef.current) {
      return engineRef.current.queueSessionEnd(durationSec);
    }
    return false;
  }, []);

  const getMetrics = useCallback(() => {
    if (engineRef.current) {
      return engineRef.current.retrievePerformanceMetrics();
    }
    return null;
  }, []);

  const exportTelemetryJsonl = useCallback((): string => {
    return engineRef.current?.exportRecordedTelemetryJsonl() ?? "";
  }, []);

  const setExplicitSessionCode = useCallback(
    (code: string | null, clientSessionId?: string | null) => {
      setSessionCode(code);
      if (engineRef.current) {
        engineRef.current.setSessionCode(code, clientSessionId);
      }
    },
    []
  );

  const operationalState: OperationalState =
    engineStatus === "error"
      ? OperationalState.ERROR
      : telemetry?.operationalState ??
        (calibration && !calibration.isCompleted
          ? OperationalState.CALIBRATING
          : isCalibrated
            ? OperationalState.MONITORING_ACTIVE
            : OperationalState.CALIBRATION_REQUIRED);

  return {
    engineStatus,
    operationalState,
    telemetry,
    calibration,
    calibrationPhase: calibration,
    isCalibrated,
    activeProfile,
    sessionCode,
    setSessionCode: setExplicitSessionCode,
    startOfflineSession,
    syncStatus,
    profileSyncStatus,
    storageMode,
    startMonitoring,
    pauseMonitoring,
    resumeMonitoring,
    stopMonitoring,
    startCalibration,
    skipCalibrationStep,
    recalibrateZone,
    cancelCalibration,
    setOpticalQuality,
    queueSessionEnd,
    getMetrics,
    recordedFrameCount,
    exportTelemetryJsonl,
    engine: engineRef.current,
  };
}
