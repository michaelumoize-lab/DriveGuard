/**
 * DriveGuard — Browser AI Detection Engine Orchestrator.
 *
 * Implements the core Target Architecture (Section 1) and Runtime Abstraction (Section 12):
 * - Decouples smooth full-rate camera preview from controlled AI inference scheduling (~10-15 FPS)
 * - Executes MediaPipe FaceLandmarker locally in browser
 * - Runs TypeScript detection algorithms: EAR, MAR, Head Pose, Blink Kinematics, PERCLOS, Temporal Analysis, Decision Engine
 * - Triggers zero-latency immediate local audio alerts
 * - Buffers safety events and session summaries through OfflineEventQueue to Next.js APIs
 * - Exposes state subscription, performance metrics, and lifecycle controls
 */

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
  PoseZone,
  RuntimePerformanceMetrics,
} from "./types";

export interface DetectionEngineConfig {
  targetInferenceFps?: number;
  driverId?: string;
  sessionCode?: string;
  profile?: DriverBehaviourProfileData | null;
  isMirrored?: boolean;
  onTelemetry?: (telemetry: DetectionTelemetry) => void;
  onSafetyAlert?: (decision: DecisionResult, telemetry: DetectionTelemetry) => void;
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
      isMirrored: true,
      ...config,
    };

    const targetFps = Math.max(5, Math.min(30, this._config.targetInferenceFps || 12));
    this._frameIntervalMs = 1000.0 / targetFps;

    this._mediaPipeService = MediaPipeService.getInstance();
    this._featureExtractor = new DriverFeatureExtractor({
      profile: this._config.profile || null,
      isMirrored: this._config.isMirrored,
    });
    this._temporalAnalyzer = new TemporalAnalyzer();
    this._decisionEngine = new DecisionEngine();
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

  public setSessionCode(code: string | null): void {
    this._sessionCode = code;
  }

  public setProfile(profile: DriverBehaviourProfileData | null): void {
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
    this._fpsCounter = 0;

    // Reset session aggregates
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

  public stop(): void {
    this._stopScheduling();
    this._status = "stopped";
    this._videoElement = null;

    // Dispatch final session summary if session code is active
    if (this._sessionCode && this._sessionStats.totalFrames > 0) {
      const frames = Math.max(1, this._sessionStats.totalFrames);
      this._offlineQueue.enqueueSessionEnd({
        sessionCode: this._sessionCode,
        durationSeconds: Math.round(frames * (this._frameIntervalMs / 1000.0)),
        averageFps: Number((this._sessionStats.fpsSum / frames).toFixed(1)),
        averageLatencyMs: Number((this._sessionStats.latencySum / frames).toFixed(1)),
        averagePerclosPct: Number((this._sessionStats.perclosSum / frames).toFixed(2)),
        maxPerclosPct: Number(this._sessionStats.maxPerclos.toFixed(2)),
        totalBlinks: this._sessionStats.totalBlinks,
        averageBlinkRate: Number((this._sessionStats.blinkRateSum / frames).toFixed(1)),
        normalFramesCount: this._sessionStats.normalFramesCount,
        warningAlertsCount: this._sessionStats.warningCount,
        criticalAlertsCount: this._sessionStats.criticalCount,
        mirrorSuppressions: this._sessionStats.mirrorSuppressions,
      });
    }

    // Clear live telemetry snapshot and reset rolling analyzers/metrics so ended
    // sessions never retain stale readings or leak state into the next drive.
    this._latestTelemetry = null;
    this._calibrationEngine = null;
    this._featureExtractor.reset();
    this._temporalAnalyzer.reset();
    this._decisionEngine.reset();
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
      this._config.isMirrored || false
    );
  }

  public cancelCalibration(): void {
    this._calibrationEngine = null;
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
      this._frameId++;
      this._totalFrames++;

      // 1. MediaPipe Inference
      const result = this._mediaPipeService.detectForVideo(video, timestampMs);

      const hasFace = Boolean(
        result &&
        result.faceLandmarks &&
        result.faceLandmarks.length > 0 &&
        result.faceLandmarks[0].length >= 468
      );

      const landmarks: NormalizedLandmark[] | null = hasFace ? result!.faceLandmarks[0] : null;

      // Extract facial transformation matrix if provided
      let facialMatrix: number[][] | null = null;
      if (
        result &&
        result.facialTransformationMatrixes &&
        result.facialTransformationMatrixes.length > 0
      ) {
        facialMatrix = result.facialTransformationMatrixes[0].data as unknown as number[][];
      }

      // 2. Feature Extraction Pass
      const features: DriverFeatures = this._featureExtractor.extract(
        landmarks,
        timestampMs,
        video.videoWidth,
        video.videoHeight,
        facialMatrix
      );

      // Calibration update if active
      if (this._calibrationEngine) {
        const calProgress = this._calibrationEngine.update(
          timestampMs,
          features.ear.compensatedEar,
          features.headPose.pitchDeg,
          features.headPose.yawDeg,
          hasFace,
          features.ear.isValid,
          features.headPose.rotMat
        );
        if (this._config.onCalibrationProgress) {
          this._config.onCalibrationProgress(calProgress);
        }
        if (calProgress.isCompleted && calProgress.profile) {
          this.setProfile(calProgress.profile);
        }
      }

      // 3. Temporal Analysis Pass
      const temporal = this._temporalAnalyzer.update(features);

      // 4. Decision Engine Pass
      const decision = this._decisionEngine.evaluate(temporal);

      // Guard against emitting telemetry if stop() was called mid-frame
      if (this._status !== "running") {
        return;
      }

      // Calculate latency & FPS
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

      // Construct telemetry snapshot
      const telemetry: DetectionTelemetry = {
        frameId: this._frameId,
        timestampMs,
        hasFace,
        alertState: decision.state,
        severity: decision.severity,
        primaryReason: decision.primaryReason ? decision.primaryReason.code : "ALERT_NORMAL",
        reasons: decision.reasons.map((r) => r.code),
        ear: Number(features.ear.compensatedEar.toFixed(3)),
        rawEar: Number(features.ear.rawEar.toFixed(3)),
        leftEar: Number(features.ear.leftEar.toFixed(3)),
        rightEar: Number(features.ear.rightEar.toFixed(3)),
        perclosPct: Number(features.perclos.perclosPct.toFixed(1)),
        effectivePerclosPct: features.perclos.effectivePerclosPct,
        isClosed: features.blink.isEyeClosed,
        closureDurationMs: Math.round(features.blink.currentClosureDurationMs),
        blinkCount: features.blink.totalBlinkCount,
        recentBlinkDurationMs: features.blink.recentBlinkDurationMs,
        recentBlinkRcr: features.blink.recentBlinkRcr ? Number(features.blink.recentBlinkRcr.toFixed(2)) : null,
        mar: Number(features.mar.mar.toFixed(3)),
        isYawning: features.mar.isAuthenticYawn || features.mar.isYawnInProgress,
        yawnCount: features.mar.totalYawnCount,
        pitchDeg: Number(features.headPose.pitchDeg.toFixed(1)),
        yawDeg: Number(features.headPose.yawDeg.toFixed(1)),
        rollDeg: Number(features.headPose.rollDeg.toFixed(1)),
        poseZone: features.headPose.zone,
        isLookingAway: features.headPose.isLookingAway,
        inferenceFps: this._currentInferenceFps || this._config.targetInferenceFps || 12,
        inferenceLatencyMs: Number(this._lastLatencyMs.toFixed(1)),
        runtimeMode: "browser",
        persistenceSyncStatus: this._syncStatus,
      };

      this._latestTelemetry = telemetry;

      // Update session statistics
      this._sessionStats.totalFrames++;
      this._sessionStats.fpsSum += telemetry.inferenceFps;
      this._sessionStats.latencySum += telemetry.inferenceLatencyMs;
      this._sessionStats.perclosSum += telemetry.perclosPct;
      this._sessionStats.maxPerclos = Math.max(this._sessionStats.maxPerclos, telemetry.perclosPct);
      this._sessionStats.totalBlinks = telemetry.blinkCount;
      if (features.headPose.zone === PoseZone.LEFT_MIRROR || features.headPose.zone === PoseZone.RIGHT_MIRROR) {
        this._sessionStats.mirrorSuppressions++;
      }
      if (decision.state === DrowsinessState.NORMAL) {
        this._sessionStats.normalFramesCount++;
      }

      // Notify telemetry subscriber
      if (this._config.onTelemetry) {
        this._config.onTelemetry(telemetry);
      }

      // Handle safety alerts (WARNING or CRITICAL)
      if (decision.state === DrowsinessState.CRITICAL_WARNING || decision.state === DrowsinessState.DROWSINESS_WARNING) {
        if (decision.state === DrowsinessState.CRITICAL_WARNING) {
          this._sessionStats.criticalCount++;
        } else {
          this._sessionStats.warningCount++;
        }

        if (this._config.onSafetyAlert) {
          this._config.onSafetyAlert(decision, telemetry);
        }

        // Buffer alert for persistence
        if (this._sessionCode) {
          const eventId = `evt_${Date.now()}_${Math.random().toString(36).slice(2, 7)}`;
          const severityStr = decision.state === DrowsinessState.CRITICAL_WARNING ? "CRITICAL" : "WARNING";
          this._offlineQueue.enqueueAlert({
            eventId,
            sessionCode: this._sessionCode,
            severity: severityStr,
            riskScore: decision.state === DrowsinessState.CRITICAL_WARNING ? 0.95 : 0.65,
            earValue: telemetry.ear,
            perclosPct: telemetry.perclosPct,
            poseZone: telemetry.poseZone,
            reasons: telemetry.reasons,
            cascadeDecision: "BROWSER_AI_ENGINE",
            timestampMs,
          });
        }
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
    this._calibrationEngine = null;
  }

  public queueSessionEnd(durationSec: number): void {
    if (!this._sessionCode) return;
    const stats = this._sessionStats;
    const frames = Math.max(1, stats.totalFrames);
    this._offlineQueue.enqueueSessionEnd({
      sessionCode: this._sessionCode,
      durationSeconds: Math.max(1, Math.round(durationSec)),
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
  }
}
