/**
 * DriveGuard — Multi-Window Temporal Event Analyzer in TypeScript.
 *
 * Direct, verified 1:1 port of backend/app/intelligence/temporal_analyzer.py.
 * Computes:
 * - Multi-window event buffering across 10s, 30s, and 60s observation horizons
 * - Active continuous eye closure duration tracking with camera vs ocular gap decoupling
 * - Off-road attentional dwell tracking (mirror checks <= 3000ms, dashboard <= 2000ms)
 * - Downward posture dwell tracking (DASHBOARD zone, relPitch <= -12 deg, |relYaw| < 14 deg)
 * - Nodding microsleep detection (Condition 1: closure >= 800ms & relPitch <= -15 deg;
 *   Condition 2: closure >= 500ms & 2.5s pitch drop >= 16 deg & relPitch <= -12 deg)
 * - Rolling 60s and 30s binocular blink kinematics (blink counts, mean duration, RCR)
 * - Rolling yawn counts and valid observed active yawn duration
 */

import {
  BlinkClassification,
  BlinkEvent,
  BlinkState,
  DriverFeatures,
  PoseZone,
  TemporalMetrics,
  YawnEvent,
} from "./types";

export class TemporalAnalyzer {
  public window60sMs: number;
  public window30sMs: number;
  public noddingPitchDropDeg: number;
  public noddingMinClosureMs: number;
  public maxSamplingGapMs: number;
  public mirrorCheckMaxMs: number;
  public dashboardCheckMaxMs: number;

  private _blinks: BlinkEvent[] = [];
  private _yawns: YawnEvent[] = [];
  private _pitchHistory: [number, number][] = []; // [tsMs, relPitchDeg]

  private _prevTsMs: number | null = null;
  private _closureStartTsMs: number | null = null;
  private _offRoadStartTsMs: number | null = null;
  private _downwardStartTsMs: number | null = null;
  private _faceLostStartTsMs: number | null = null;
  private _lastProcessedBlinkEndTs: number | null = null;
  private _lastProcessedYawnEndTs: number | null = null;
  private _lastMetrics: TemporalMetrics | null = null;

  constructor(options?: {
    window60sMs?: number;
    window30sMs?: number;
    noddingPitchDropDeg?: number;
    noddingMinClosureMs?: number;
    maxSamplingGapMs?: number;
    mirrorCheckMaxMs?: number;
    dashboardCheckMaxMs?: number;
  }) {
    this.window60sMs = options?.window60sMs ?? 60_000.0;
    this.window30sMs = options?.window30sMs ?? 30_000.0;
    this.noddingPitchDropDeg = options?.noddingPitchDropDeg ?? -15.0;
    this.noddingMinClosureMs = options?.noddingMinClosureMs ?? 800.0;
    this.maxSamplingGapMs = options?.maxSamplingGapMs ?? 250.0;
    this.mirrorCheckMaxMs = options?.mirrorCheckMaxMs ?? 3000.0;
    this.dashboardCheckMaxMs = options?.dashboardCheckMaxMs ?? 2000.0;
  }

  private _makeDefaultMetrics(): TemporalMetrics {
    return {
      timestampMs: 0.0,
      activeClosureDurationMs: 0.0,
      activeOffRoadDwellMs: 0.0,
      isMirrorCheckContext: false,
      isDashboardCheckContext: false,
      isNoddingMicrosleep: false,
      blinkCount60s: 0,
      blinkCount30s: 0,
      meanBlinkDurationMs60s: 0.0,
      prolongedBlinkCount60s: 0,
      severeBlinkCount60s: 0,
      meanRcr60s: 1.0,
      yawnCount60s: 0,
      yawnCount30s: 0,
      activeYawnDurationMs: 0.0,
      perclosPct: 0.0,
      perclosCoveragePct: 0.0,
      perclosIsReliable: false,
      effectivePerclosPct: null,
      hasFace: false,
      faceLostDurationMs: 0.0,
      isEyeClosed: null,
      completedBlink: null,
      downwardDwellMs: 0.0,
    };
  }

  public update(
    features: DriverFeatures,
    isReliable: boolean = true,
    isOcularViable?: boolean | null
  ): TemporalMetrics {
    const ts = features.timestampMs;

    // ------------------------------------------------------------------
    // 0. Timestamp Validation & Integrity Guard
    // ------------------------------------------------------------------
    if (!Number.isFinite(ts) || ts < 0.0) {
      return this._lastMetrics ?? this._makeDefaultMetrics();
    }

    let dtMs = 0.0;
    if (this._prevTsMs !== null) {
      dtMs = ts - this._prevTsMs;
      if (dtMs <= 0.0) {
        return this._lastMetrics ?? this._makeDefaultMetrics();
      }
    }

    // Decouple actual camera sampling gap from ocular tracking gap
    const isCameraSamplingGap = this._prevTsMs !== null && dtMs > this.maxSamplingGapMs;
    const blinkGapSignal =
      Boolean(features.blink.samplingGapDetected) ||
      features.blink.state === BlinkState.UNCERTAIN_GAP;
    const isOcularGap = isCameraSamplingGap || blinkGapSignal;

    // ------------------------------------------------------------------
    // 1. Face Loss Tracking
    // ------------------------------------------------------------------
    let faceLostDurationMs = 0.0;
    if (!features.hasFace) {
      if (this._faceLostStartTsMs === null) {
        this._faceLostStartTsMs = ts;
      }
      faceLostDurationMs = Math.max(0.0, ts - this._faceLostStartTsMs);
      this._closureStartTsMs = null;
      this._offRoadStartTsMs = null;
      this._pitchHistory = [];
    } else {
      this._faceLostStartTsMs = null;
      faceLostDurationMs = 0.0;
    }

    // ------------------------------------------------------------------
    // 2. Continuous Eye Closure Tracking
    // ------------------------------------------------------------------
    const ocularChannelViable =
      isOcularViable === undefined || isOcularViable === null
        ? isReliable
        : Boolean(isOcularViable);

    let isEyeClosedVal: boolean | null = null;
    let activeClosureDurationMs = 0.0;

    if (!ocularChannelViable) {
      isEyeClosedVal = null;
      this._closureStartTsMs = null;
      activeClosureDurationMs = 0.0;
    } else {
      const earValid = features.hasFace && features.ear.isValid;
      const blinkValid =
        features.hasFace &&
        !features.blink.samplingGapDetected &&
        features.blink.state !== BlinkState.UNCERTAIN_GAP;

      const earClosed = earValid && features.ear.isClosed;
      const blinkClosed = blinkValid && features.blink.isEyeClosed;

      if (earClosed || blinkClosed) {
        isEyeClosedVal = true;
      } else if (earValid && blinkValid) {
        isEyeClosedVal = false;
      } else if (earValid && !features.ear.isClosed && !blinkValid) {
        isEyeClosedVal = false;
      } else {
        isEyeClosedVal = null;
      }

      const isCurrentlyClosed = isEyeClosedVal === true;

      if (isOcularGap) {
        if (isCurrentlyClosed) {
          this._closureStartTsMs = ts;
        } else {
          this._closureStartTsMs = null;
        }
        activeClosureDurationMs = 0.0;
      } else if (isCurrentlyClosed) {
        if (this._closureStartTsMs === null) {
          this._closureStartTsMs = ts;
        }
        const rawClosureDuration = ts - this._closureStartTsMs;
        activeClosureDurationMs = Math.max(
          rawClosureDuration,
          features.blink.currentClosureDurationMs
        );
      } else {
        this._closureStartTsMs = null;
        activeClosureDurationMs = 0.0;
      }
    }

    // ------------------------------------------------------------------
    // 3. Off-Road Gaze Dwell & Cockpit Context
    // ------------------------------------------------------------------
    const zone = features.headPose.zone;
    const isOffRoad =
      features.hasFace &&
      features.headPose.isValid &&
      zone !== PoseZone.CENTER &&
      zone !== PoseZone.UNKNOWN;

    let activeOffRoadDwellMs = 0.0;
    if (isCameraSamplingGap) {
      if (isOffRoad) {
        this._offRoadStartTsMs = ts;
      } else {
        this._offRoadStartTsMs = null;
      }
      activeOffRoadDwellMs = 0.0;
    } else if (isOffRoad) {
      if (this._offRoadStartTsMs === null) {
        this._offRoadStartTsMs = ts;
      }
      activeOffRoadDwellMs = Math.max(0.0, ts - this._offRoadStartTsMs);
    } else {
      this._offRoadStartTsMs = null;
      activeOffRoadDwellMs = 0.0;
    }

    const isMirrorCheckContext =
      (zone === PoseZone.LEFT_MIRROR || zone === PoseZone.RIGHT_MIRROR) &&
      activeOffRoadDwellMs <= this.mirrorCheckMaxMs;
    const isDashboardCheckContext =
      zone === PoseZone.DASHBOARD &&
      activeOffRoadDwellMs <= this.dashboardCheckMaxMs;

    // ------------------------------------------------------------------
    // 3b. Downward Head Posture Dwell Tracking (matches Python temporal_analyzer.py)
    // ------------------------------------------------------------------
    const isDownwardPosture =
      features.hasFace &&
      features.headPose.isValid &&
      features.headPose.zone === PoseZone.DASHBOARD &&
      features.headPose.relPitchDeg <= -12.0 &&
      Math.abs(features.headPose.relYawDeg) < 14.0;

    let downwardDwellMs = 0.0;
    if (isCameraSamplingGap) {
      if (isDownwardPosture) {
        this._downwardStartTsMs = ts;
      } else {
        this._downwardStartTsMs = null;
      }
      downwardDwellMs = 0.0;
    } else if (isDownwardPosture) {
      if (this._downwardStartTsMs === null) {
        this._downwardStartTsMs = ts;
      }
      downwardDwellMs = Math.max(0.0, ts - this._downwardStartTsMs);
    } else {
      this._downwardStartTsMs = null;
      downwardDwellMs = 0.0;
    }

    // ------------------------------------------------------------------
    // 4. Nodding Microsleep Detection (Conditions 1 & 2, 2.5s pitch window)
    // ------------------------------------------------------------------
    const isPoseInvalid =
      !features.hasFace ||
      !features.headPose.isValid ||
      features.headPose.zone === PoseZone.UNKNOWN;
    if (isCameraSamplingGap || isPoseInvalid) {
      this._pitchHistory = [];
    }

    let isNoddingMicrosleep = false;
    if (
      features.hasFace &&
      features.headPose.isValid &&
      features.headPose.zone !== PoseZone.UNKNOWN
    ) {
      const relPitch = features.headPose.relPitchDeg;
      this._pitchHistory.push([ts, relPitch]);
      // Expire pitch older than 2.5s
      while (this._pitchHistory.length > 0 && ts - this._pitchHistory[0][0] > 2500.0) {
        this._pitchHistory.shift();
      }

      // Nodding Condition 1: Sustained closure >= 800 ms + downward slump <= -15 deg
      if (
        activeClosureDurationMs >= this.noddingMinClosureMs &&
        relPitch <= this.noddingPitchDropDeg
      ) {
        isNoddingMicrosleep = true;
      }

      // Nodding Condition 2: Sudden downward pitch drop >= 16 deg while eyes closed >= 500 ms
      if (this._pitchHistory.length >= 2 && activeClosureDurationMs >= 500.0) {
        let maxPitchRecent = -Infinity;
        for (let i = 0; i < this._pitchHistory.length; i++) {
          if (this._pitchHistory[i][1] > maxPitchRecent) {
            maxPitchRecent = this._pitchHistory[i][1];
          }
        }
        if (maxPitchRecent - relPitch >= 16.0 && relPitch <= -12.0) {
          isNoddingMicrosleep = true;
        }
      }
    }

    // ------------------------------------------------------------------
    // 5. Blink History Accumulation & Binocular Kinematic Aggregation
    // ------------------------------------------------------------------
    const completedBlink = features.blink.completedBlink;
    let newCompletedBlink: BlinkEvent | null = null;
    if (
      completedBlink !== null &&
      completedBlink.endTsMs !== this._lastProcessedBlinkEndTs
    ) {
      this._blinks.push(completedBlink);
      this._lastProcessedBlinkEndTs = completedBlink.endTsMs;
      newCompletedBlink = completedBlink;
    }

    // Expire blinks older than 60s
    while (
      this._blinks.length > 0 &&
      ts - this._blinks[0].endTsMs > this.window60sMs
    ) {
      this._blinks.shift();
    }

    const cutoff30s = ts - this.window30sMs;
    const blinks30s = this._blinks.filter((b) => b.endTsMs >= cutoff30s);

    // Only valid binocular blinks contribute to rolling binocular fatigue statistics
    const binocularBlinks60s = this._blinks.filter(
      (b) => b.isBinocular && b.classification !== BlinkClassification.UNCERTAIN
    );
    const binocularBlinks30s = blinks30s.filter(
      (b) => b.isBinocular && b.classification !== BlinkClassification.UNCERTAIN
    );

    const blinkCount60s = binocularBlinks60s.length;
    const blinkCount30s = binocularBlinks30s.length;

    let meanBlinkDurationMs60s = 0.0;
    let prolongedBlinkCount60s = 0;
    let severeBlinkCount60s = 0;
    let rcrSum = 0.0;
    let rcrCount = 0;

    if (blinkCount60s > 0) {
      for (const b of binocularBlinks60s) {
        meanBlinkDurationMs60s += b.durationMs;
        if (
          b.classification === BlinkClassification.PROLONGED ||
          b.classification === BlinkClassification.MICROSLEEP_CANDIDATE
        ) {
          prolongedBlinkCount60s++;
        }
        if (b.durationMs >= 500.0) {
          severeBlinkCount60s++;
        }
        if (b.isKinematicsReliable && b.rcr > 0.0) {
          rcrSum += b.rcr;
          rcrCount++;
        }
      }
      meanBlinkDurationMs60s /= blinkCount60s;
    }
    const meanRcr60s = rcrCount > 0 ? rcrSum / rcrCount : 1.0;

    // ------------------------------------------------------------------
    // 6. Yawn History Accumulation
    // ------------------------------------------------------------------
    const completedYawn = features.mar.completedYawn;
    if (
      completedYawn !== null &&
      completedYawn.endTsMs !== this._lastProcessedYawnEndTs
    ) {
      this._yawns.push(completedYawn);
      this._lastProcessedYawnEndTs = completedYawn.endTsMs;
    }

    while (
      this._yawns.length > 0 &&
      ts - this._yawns[0].endTsMs > this.window60sMs
    ) {
      this._yawns.shift();
    }

    const yawns30s = this._yawns.filter((y) => y.endTsMs >= cutoff30s);
    const activeYawnDurationMs =
      features.mar.isValid && features.mar.isYawnInProgress
        ? features.mar.observedYawnDurationMs
        : 0.0;

    this._prevTsMs = ts;

    const metrics: TemporalMetrics = {
      timestampMs: ts,
      activeClosureDurationMs,
      activeOffRoadDwellMs,
      isMirrorCheckContext,
      isDashboardCheckContext,
      isNoddingMicrosleep,
      blinkCount60s,
      blinkCount30s,
      meanBlinkDurationMs60s,
      prolongedBlinkCount60s,
      severeBlinkCount60s,
      meanRcr60s,
      yawnCount60s: this._yawns.length,
      yawnCount30s: yawns30s.length,
      activeYawnDurationMs,
      perclosPct: features.perclos.perclosPct,
      perclosCoveragePct: features.perclos.coveragePct,
      perclosIsReliable: features.perclos.isReliable,
      effectivePerclosPct: features.perclos.effectivePerclosPct,
      hasFace: features.hasFace,
      faceLostDurationMs,
      isEyeClosed: isEyeClosedVal,
      completedBlink: newCompletedBlink,
      downwardDwellMs,
    };
    this._lastMetrics = metrics;
    return metrics;
  }

  public reset(): void {
    this._blinks = [];
    this._yawns = [];
    this._pitchHistory = [];
    this._prevTsMs = null;
    this._closureStartTsMs = null;
    this._offRoadStartTsMs = null;
    this._downwardStartTsMs = null;
    this._faceLostStartTsMs = null;
    this._lastProcessedBlinkEndTs = null;
    this._lastProcessedYawnEndTs = null;
    this._lastMetrics = null;
  }
}
