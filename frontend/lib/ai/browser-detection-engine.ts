/**
 * DriveGuard — Browser AI Detection Engine Orchestrator.
 *
 * Implements the core Target Architecture:
 * - Decouples smooth full-rate camera preview from controlled AI inference scheduling (~10-15 FPS)
 * - Executes MediaPipe FaceLandmarker locally in browser
 * - Runs TypeScript detection algorithms: EAR, MAR, Head Pose, Blink Kinematics, PERCLOS, Temporal Analysis, Decision Engine
 * - Enforces OperationalState calibration gate (CALIBRATION_REQUIRED when uncalibrated)
 * - Routes safety transitions through AlertStateManager (no per-frame alert storms)
 * - Buffers deduplicated safety events (with eventId) and session summaries (with finalizeToken) through OfflineEventQueue
 */

import {
  AlertStateManager,
  AlertTransition,
  ManagedAlertState,
} from "./alert-state-manager";
import { DecisionEngine } from "./decision-engine";
import { DriverFeatureExtractor } from "./feature-extractor";
import { MediaPipeService } from "./mediapipe-service";
import { OfflineEventQueue, PersistenceSyncStatus } from "./offline-event-queue";
import { GuidedCalibrationEngine } from "./pdbp-calibration";
import { TemporalAnalyzer } from "./temporal-analyzer";
import {
  CalibrationProgress,
  DecisionResult,
  DetectionTelemetry,
  DriverBehaviourProfileData,
  DriverFeatures,
  DrowsinessState,
  NormalizedLandmark,
  OperationalState,
  PoseZone,
  RuntimePerformanceMetrics,
} from "./types";

export interface DetectionEngineConfig {
  targetInferenceFps?: number;
  driverId?: string;
  sessionCode?: string;
  profile?: DriverBehaviourProfileData | unknown | null;
  /**
   * Whether the raw video pixel buffer itself is horizontally mirrored.
   * Defaults to false (raw browser webcam frames are unmirrored; CSS preview mirroring does not flip pixels).
   */
  isMirrored?: boolean;
  onTelemetry?: (telemetry: DetectionTelemetry) => void;
  onSafetyAlert?: (
    decision: DecisionResult,
    telemetry: DetectionTelemetry,
    transition?: AlertTransition
  ) => void;
  onAlertTransition?: (
    transition: AlertTransition,
    decision: DecisionResult,
    telemetry: DetectionTelemetry
  ) => void;
  onCalibrationProgress?: (progress: CalibrationProgress) => void;
  onError?: (err: Error) => void;
}

export type EngineStatus = "idle" | "initializing" | "running" | "paused" | "stopped" | "error";

export class BrowserDetectionEngine {
  private _status: EngineStatus = "idle";
  private _config: DetectionEngineConfig;
  private _videoElement: HTMLVideoElement | null = null;
  private _mediaPipeService: MediaPipeService;
  private _featureExtractor: DriverFeatureExtractor;
  private _temporalAnalyzer: TemporalAnalyzer;
  private _decisionEngine: DecisionEngine;
  private _alertStateManager: AlertStateManager;
  private _calibrationEngine: GuidedCalibrationEngine | null = null;
  private _offlineQueue: OfflineEventQueue;

  private _animationFrameId: number | null = null;
  private _intervalTimerId: ReturnType<typeof setInterval> | null = null;
  private _isProcessingFrame: boolean = false;
  private _lastInferenceWallMs: number = 0;
  private _frameIntervalMs: number = 80; // default ~12.5 FPS (1000 / 12.5)

  // Diagnostics and performance telemetry
  private _frameId: number = 0;
  private _fpsCounter: number = 0;
  private _lastFpsCalcWallMs: number = 0;
  private _currentInferenceFps: number = 0;
  private _latencySamples: number[] = [];
  private _lastLatencyMs: number = 0;
  private _droppedFrames: number = 0;
  private _totalFrames: number = 0;

  // Last telemetry snapshot
  private _latestTelemetry: DetectionTelemetry | null = null;
  private _syncStatus: PersistenceSyncStatus = "synced";
  private _syncUnsubscribe: (() => void) | null = null;

  // Active session tracking
  private _sessionCode: string | null = null;
  private _finalizedSessionCodes: Set<string> = new Set();
  private _sessionStartWallMs: number | null = null;
  private _sessionStats = {
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
  };

  constructor(config: DetectionEngineConfig = {}) {
    this._config = {
      targetInferenceFps: 12,
      driverId: "driver_default",
      isMirrored: false,
      ...config,
    };

    const targetFps = Math.max(5, Math.min(30, this._config.targetInferenceFps || 12));
    this._frameIntervalMs = 1000.0 / targetFps;

    this._mediaPipeService = MediaPipeService.getInstance();
    this._featureExtractor = new DriverFeatureExtractor({
      profile: this._config.profile ?? null,
      isMirrored: this._config.isMirrored ?? false,
    });
    this._temporalAnalyzer = new TemporalAnalyzer();
    this._decisionEngine = new DecisionEngine();
    this._alertStateManager = new AlertStateManager();
    this._offlineQueue = new OfflineEventQueue();

    this._syncUnsubscribe = this._offlineQueue.subscribe((status) => {
      this._syncStatus = status;
      if (this._latestTelemetry) {
        this._latestTelemetry.persistenceSyncStatus = status;
      }
    });

    if (config.sessionCode) {
      this._sessionCode = config.sessionCode;
    }
  }

  public get status(): EngineStatus {
    return this._status;
  }

  public get sessionCode(): string | null {
    return this._sessionCode;
  }

  public get isCalibrated(): boolean {
    return (
      this._featureExtractor.isProfileCalibrated &&
      this._featureExtractor.isProfileFeasible
    );
  }

  public get alertStateManager(): AlertStateManager {
    return this._alertStateManager;
  }

  public setSessionCode(code: string | null): void {
    this._sessionCode = code;
    if (code) {
      this._finalizedSessionCodes.delete(code);
    }
  }

  public setProfile(profile: DriverBehaviourProfileData | unknown | null): void {
    this._config.profile = profile;
    this._featureExtractor.setProfile(profile);
  }

  public setIsMirrored(mirrored: boolean): void {
    this._config.isMirrored = mirrored;
    this._featureExtractor.setIsMirrored(mirrored);
  }

  public async initialize(): Promise<void> {
    if (this._status === "running") return;
    this._status = "initializing";
    try {
      await this._mediaPipeService.initialize();
      this._status = "idle";
    } catch (err) {
      this._status = "error";
      if (this._config.onError) {
        this._config.onError(err instanceof Error ? err : new Error(String(err)));
      }
      throw err;
    }
  }

  public start(videoElement: HTMLVideoElement): void {
    if (this._status === "running") return;
    this._videoElement = videoElement;
    this._status = "running";
    this._lastInferenceWallMs = 0;
    this._lastFpsCalcWallMs = performance.now();
    this._sessionStartWallMs = Date.now();
    this._fpsCounter = 0;
    this._resetSessionStats();
    this._scheduleNextLoop();
  }

  public pause(): void {
    if (this._status !== "running") return;
    this._status = "paused";
    this._stopScheduling();
  }

  public resume(): void {
    if (this._status !== "paused" || !this._videoElement) return;
    this._status = "running";
    this._scheduleNextLoop();
  }

  /**
   * Stop active inference loop and reset volatile analyzers.
   * Decoupled from session finalization so callers explicitly invoke queueSessionEnd()
   * exactly once when ending a monitoring session (eliminating duplicate /api/sessions/end POSTs).
   */
  public stop(): void {
    this._stopScheduling();
    this._status = "stopped";
    this._videoElement = null;

    this._latestTelemetry = null;
    this._calibrationEngine = null;
    this._featureExtractor.reset();
    this._temporalAnalyzer.reset();
    this._decisionEngine.reset();
    this._alertStateManager.reset();
    this._frameId = 0;
    this._fpsCounter = 0;
    this._lastFpsCalcWallMs = 0;
    this._lastInferenceWallMs = 0;
    this._currentInferenceFps = 0;
    this._latencySamples = [];
    this._lastLatencyMs = 0;
    this._droppedFrames = 0;
    this._totalFrames = 0;
  }

  public startCalibration(): void {
    this._calibrationEngine = new GuidedCalibrationEngine(
      this._config.driverId || "driver_default",
      this._config.isMirrored ?? false
    );
  }

  public cancelCalibration(): void {
    this._calibrationEngine = null;
  }

  private _resetSessionStats(): void {
    this._sessionStats = {
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
    };
  }

  private _scheduleNextLoop(): void {
    const loop = () => {
      if (this._status !== "running") return;
      const now = performance.now();

      if (now - this._lastInferenceWallMs >= this._frameIntervalMs) {
        this._lastInferenceWallMs = now;
        this._processCurrentFrame(now);
      }

      this._animationFrameId = requestAnimationFrame(loop);
    };

    this._animationFrameId = requestAnimationFrame(loop);
  }

  private _stopScheduling(): void {
    if (this._animationFrameId !== null) {
      cancelAnimationFrame(this._animationFrameId);
      this._animationFrameId = null;
    }
    if (this._intervalTimerId !== null) {
      clearInterval(this._intervalTimerId);
      this._intervalTimerId = null;
    }
  }

  /**
   * Core deterministic frame pipeline decoupled from DOM/MediaPipe so unit tests can
   * exercise the full FeatureExtractor -> TemporalAnalyzer -> DecisionEngine -> AlertStateManager flow.
   */
  public evaluateFrameLandmarks(
    landmarks: ArrayLike<NormalizedLandmark> | null,
    timestampMs: number,
    frameWidth: number = 640,
    frameHeight: number = 480,
    facialMatrix?: ArrayLike<number> | number[][] | null,
    inferenceLatencyMs: number = 0
  ): {
    features: DriverFeatures;
    decision: DecisionResult;
    transition: AlertTransition | null;
    telemetry: DetectionTelemetry;
  } {
    this._frameId++;
    this._totalFrames++;

    const hasFace = Boolean(landmarks && landmarks.length >= 468);

    // 1. Feature Extraction Pass
    const features: DriverFeatures = this._featureExtractor.extract(
      landmarks,
      timestampMs,
      frameWidth,
      frameHeight,
      facialMatrix
    );

    // Calibration update if active
    if (this._calibrationEngine) {
      const calProgress = this._calibrationEngine.update(
        timestampMs,
        features.ear.weightedEar,
        features.headPose.pitchDeg,
        features.headPose.yawDeg,
        hasFace,
        hasFace && features.ear.weightedEar > 0,
        features.headPose.rotMat
      );
      if (this._config.onCalibrationProgress) {
        this._config.onCalibrationProgress(calProgress);
      }
      if (calProgress.isCompleted && calProgress.profile) {
        this.setProfile(calProgress.profile);
        this._calibrationEngine = null;
      }
    }

    // 2. Temporal Analysis Pass
    const temporal = this._temporalAnalyzer.update(features);

    // 3. Decision Engine Pass with strict OperationalState calibration gate
    const isCalibrating = Boolean(this._calibrationEngine);
    const isCalibrated =
      this._featureExtractor.isProfileCalibrated &&
      this._featureExtractor.isProfileFeasible;

    const decision = this._decisionEngine.evaluate(temporal, {
      overallConfidence: hasFace ? 0.90 : 0.0,
      ocularViable: hasFace && features.ear.isValid && isCalibrated,
      headPoseViable: hasFace && features.headPose.isValid,
      yawnViable: hasFace && features.mar.isValid,
      isReliable: hasFace && features.ear.isValid && isCalibrated,
      isOcularAvailable: hasFace && features.ear.isValid && isCalibrated,
      isCalibrated,
      isCalibrating,
    });

    const operationalState =
      decision.operationalState ??
      (isCalibrating
        ? OperationalState.CALIBRATING
        : !isCalibrated
          ? OperationalState.CALIBRATION_REQUIRED
          : hasFace
            ? OperationalState.MONITORING_ACTIVE
            : OperationalState.DEGRADED);

    // 4. Alert State Manager Pass (eliminates per-frame alert storms)
    const transition = this._alertStateManager.update(decision, temporal);

    const telemetry: DetectionTelemetry = {
      frameId: this._frameId,
      timestampMs,
      hasFace,
      operationalState,
      engineType: "BROWSER",
      alertState: decision.state,
      severity: decision.severity,
      primaryReason: decision.primaryReason
        ? decision.primaryReason.code
        : isCalibrated
          ? "ALERT_NORMAL"
          : "CALIBRATION_REQUIRED",
      reasons: decision.reasons.map((r) => r.code),
      ear: Number(features.ear.compensatedEar.toFixed(3)),
      rawEar: Number(features.ear.rawEar.toFixed(3)),
      leftEar: Number(features.ear.leftEar.toFixed(3)),
      rightEar: Number(features.ear.rightEar.toFixed(3)),
      earClosureThreshold: Number(features.ear.closureThreshold.toFixed(4)),
      perclosPct: Number(features.perclos.perclosPct.toFixed(1)),
      effectivePerclosPct: features.perclos.effectivePerclosPct,
      perclosCoveragePct: Number(features.perclos.coveragePct.toFixed(1)),
      perclosReliable: features.perclos.isReliable,
      isClosed: features.blink.isEyeClosed,
      closureDurationMs: Math.round(features.blink.currentClosureDurationMs),
      blinkCount: features.blink.totalBlinkCount,
      blinkCount60s: temporal.blinkCount60s,
      totalBlinkCount: features.blink.totalBlinkCount,
      recentBlinkDurationMs: features.blink.recentBlinkDurationMs,
      recentBlinkRcr: features.blink.recentBlinkRcr
        ? Number(features.blink.recentBlinkRcr.toFixed(2))
        : null,
      mar: Number(features.mar.mar.toFixed(3)),
      isYawning: features.mar.isAuthenticYawn || features.mar.isYawnInProgress,
      yawnCount: features.mar.totalYawnCount,
      pitchDeg: Number(features.headPose.pitchDeg.toFixed(1)),
      yawDeg: Number(features.headPose.yawDeg.toFixed(1)),
      rollDeg: Number(features.headPose.rollDeg.toFixed(1)),
      poseZone: features.headPose.zone,
      isLookingAway: features.headPose.isLookingAway,
      inferenceFps: this._currentInferenceFps || this._config.targetInferenceFps || 12,
      inferenceLatencyMs: Number(inferenceLatencyMs.toFixed(1)),
      runtimeMode: "browser",
      persistenceSyncStatus: this._syncStatus,
    };

    this._latestTelemetry = telemetry;

    // Update session statistics accurately
    this._sessionStats.totalFrames++;
    this._sessionStats.fpsSum += telemetry.inferenceFps;
    this._sessionStats.latencySum += telemetry.inferenceLatencyMs;
    this._sessionStats.perclosSum += telemetry.perclosPct;
    this._sessionStats.maxPerclos = Math.max(
      this._sessionStats.maxPerclos,
      telemetry.perclosPct
    );
    this._sessionStats.totalBlinks = features.blink.totalBlinkCount;
    this._sessionStats.blinkRateSum += temporal.blinkCount60s;
    if (temporal.isMirrorCheckContext) {
      this._sessionStats.mirrorSuppressions++;
    }
    if (
      operationalState === OperationalState.MONITORING_ACTIVE &&
      decision.state === DrowsinessState.NORMAL
    ) {
      this._sessionStats.normalFramesCount++;
    }

    // Dispatch transition callbacks & single-channel idempotent alert persistence
    if (transition) {
      if (this._config.onAlertTransition) {
        this._config.onAlertTransition(transition, decision, telemetry);
      }

      if (
        transition.newState === ManagedAlertState.CRITICAL ||
        transition.newState === ManagedAlertState.WARNING
      ) {
        if (transition.newState === ManagedAlertState.CRITICAL) {
          this._sessionStats.criticalCount++;
        } else {
          this._sessionStats.warningCount++;
        }

        if (this._config.onSafetyAlert) {
          this._config.onSafetyAlert(decision, telemetry, transition);
        }

        if (this._sessionCode) {
          this._offlineQueue.enqueueAlert({
            eventId: transition.eventId,
            sessionCode: this._sessionCode,
            severity: transition.newState === ManagedAlertState.CRITICAL ? "CRITICAL" : "WARNING",
            riskScore: transition.newState === ManagedAlertState.CRITICAL ? 0.95 : 0.65,
            earValue: telemetry.ear,
            perclosPct: telemetry.perclosPct,
            poseZone: telemetry.poseZone,
            reasons: telemetry.reasons,
            cascadeDecision: "GEOMETRIC_AUTHORITATIVE",
            timestampMs,
          });
        }
      }
    }

    return { features, decision, transition, telemetry };
  }

  private _processCurrentFrame(timestampMs: number): void {
    if (this._isProcessingFrame || !this._videoElement) {
      this._droppedFrames++;
      return;
    }

    const video = this._videoElement;
    if (
      video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA ||
      video.videoWidth === 0 ||
      video.videoHeight === 0
    ) {
      return;
    }

    this._isProcessingFrame = true;
    const tStart = performance.now();

    try {
      // 1. MediaPipe Inference
      const result = this._mediaPipeService.detectForVideo(video, timestampMs);

      const hasFace = Boolean(
        result &&
          result.faceLandmarks &&
          result.faceLandmarks.length > 0 &&
          result.faceLandmarks[0].length >= 468
      );

      const landmarks: NormalizedLandmark[] | null = hasFace ? result!.faceLandmarks[0] : null;

      // Preserve Float32Array directly so head-pose.ts consumes MediaPipe's 4x4 matrix natively
      let facialMatrix: ArrayLike<number> | null = null;
      if (
        result &&
        result.facialTransformationMatrixes &&
        result.facialTransformationMatrixes.length > 0
      ) {
        facialMatrix = result.facialTransformationMatrixes[0].data;
      }

      const tEnd = performance.now();
      this._lastLatencyMs = tEnd - tStart;
      this._latencySamples.push(this._lastLatencyMs);
      if (this._latencySamples.length > 30) this._latencySamples.shift();

      this._fpsCounter++;
      if (tEnd - this._lastFpsCalcWallMs >= 1000.0) {
        this._currentInferenceFps = Number(
          ((this._fpsCounter * 1000.0) / (tEnd - this._lastFpsCalcWallMs)).toFixed(1)
        );
        this._fpsCounter = 0;
        this._lastFpsCalcWallMs = tEnd;
      }

      const { telemetry } = this.evaluateFrameLandmarks(
        landmarks,
        timestampMs,
        video.videoWidth,
        video.videoHeight,
        facialMatrix,
        this._lastLatencyMs
      );

      // Guard against emitting telemetry if stop() was called mid-frame
      if (this._status !== "running") {
        return;
      }

      if (this._config.onTelemetry) {
        this._config.onTelemetry(telemetry);
      }
    } catch (err) {
      console.warn("[BrowserDetectionEngine] Process frame error:", err);
      if (this._config.onError) {
        this._config.onError(err instanceof Error ? err : new Error(String(err)));
      }
    } finally {
      this._isProcessingFrame = false;
    }
  }

  public retrieveCurrentState(): DetectionTelemetry | null {
    return this._latestTelemetry;
  }

  public retrievePerformanceMetrics(): RuntimePerformanceMetrics {
    const avgLatency =
      this._latencySamples.length > 0
        ? this._latencySamples.reduce((a, b) => a + b, 0) / this._latencySamples.length
        : 0;
    const minLatency = this._latencySamples.length > 0 ? Math.min(...this._latencySamples) : 0;
    const maxLatency = this._latencySamples.length > 0 ? Math.max(...this._latencySamples) : 0;

    return {
      inferenceFps: this._currentInferenceFps,
      cameraFps: 30,
      averageInferenceLatencyMs: Number(avgLatency.toFixed(1)),
      minInferenceLatencyMs: Number(minLatency.toFixed(1)),
      maxInferenceLatencyMs: Number(maxLatency.toFixed(1)),
      totalFramesProcessed: this._totalFrames,
      droppedFrames: this._droppedFrames,
    };
  }

  public getSessionStatsSnapshot(): typeof this._sessionStats {
    return { ...this._sessionStats };
  }

  public dispose(): void {
    this.stop();
    if (this._syncUnsubscribe) {
      this._syncUnsubscribe();
      this._syncUnsubscribe = null;
    }
    this._offlineQueue.dispose();
    this._featureExtractor.reset();
    this._temporalAnalyzer.reset();
    this._decisionEngine.reset();
    this._alertStateManager.reset();
    this._calibrationEngine = null;
  }

  /**
   * Explicitly finalize the current Browser AI driving session exactly once
   * through OfflineEventQueue with a unique finalizeToken.
   */
  public queueSessionEnd(durationSec?: number): boolean {
    const code = this._sessionCode;
    if (!code || this._finalizedSessionCodes.has(code)) {
      return false;
    }
    this._finalizedSessionCodes.add(code);

    const stats = this._sessionStats;
    const frames = Math.max(1, stats.totalFrames);
    const elapsedFallbackSec =
      this._sessionStartWallMs !== null
        ? Math.max(1, Math.round((Date.now() - this._sessionStartWallMs) / 1000))
        : Math.max(1, Math.round(frames * (this._frameIntervalMs / 1000.0)));
    const resolvedDuration =
      durationSec !== undefined && Number.isFinite(durationSec) && durationSec > 0
        ? Math.round(durationSec)
        : elapsedFallbackSec;

    const finalizeToken = `fin_${code}_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;

    this._offlineQueue.enqueueSessionEnd({
      sessionCode: code,
      finalizeToken,
      durationSeconds: resolvedDuration,
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
    });

    this._sessionCode = null;
    this._sessionStartWallMs = null;
    return true;
  }
}
