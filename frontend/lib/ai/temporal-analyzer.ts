/**
 * DriveGuard — Multi-Window Temporal Event Analyzer in TypeScript.
 *
 * Direct, verified port of backend/app/intelligence/temporal_analyzer.py.
 * Computes:
 * - Multi-window event buffering across 10s, 30s, and 60s observation horizons
 * - Active continuous eye closure duration tracking
 * - Off-road attentional dwell tracking (mirror checks vs persistent distraction)
 * - Nodding microsleep detection (acute pitch drop >= 15° with closure >= 800 ms)
 * - Rolling 60s and 30s blink kinematics (blink counts, mean duration, RCR)
 * - Rolling yawn counts and active yawn duration
 */

import {
  BlinkClassification,
  BlinkEvent,
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

  private _blinks: BlinkEvent[] = [];
  private _yawns: YawnEvent[] = [];
  private _pitchHistory: [number, number][] = []; // [tsMs, relPitchDeg]

  private _prevTsMs: number | null = null;
  private _closureStartTsMs: number | null = null;
  private _offRoadStartTsMs: number | null = null;
  private _downwardStartTsMs: number | null = null;
  private _faceLostStartTsMs: number | null = null;

  constructor(options?: {
    window60sMs?: number;
    window30sMs?: number;
    noddingPitchDropDeg?: number;
    noddingMinClosureMs?: number;
    maxSamplingGapMs?: number;
  }) {
    this.window60sMs = options?.window60sMs ?? 60_000.0;
    this.window30sMs = options?.window30sMs ?? 30_000.0;
    this.noddingPitchDropDeg = options?.noddingPitchDropDeg ?? -15.0;
    this.noddingMinClosureMs = options?.noddingMinClosureMs ?? 800.0;
    this.maxSamplingGapMs = options?.maxSamplingGapMs ?? 250.0;
  }

  public update(features: DriverFeatures): TemporalMetrics {
    const ts = features.timestampMs;

    // Sampling gap check
    if (this._prevTsMs !== null && ts - this._prevTsMs > this.maxSamplingGapMs) {
      this._closureStartTsMs = null;
      this._offRoadStartTsMs = null;
      this._downwardStartTsMs = null;
    }

    // 1. Ingest completed events
    if (features.blink.completedBlink) {
      this._blinks.push(features.blink.completedBlink);
    }
    if (features.mar.completedYawn) {
      this._yawns.push(features.mar.completedYawn);
    }

    // 2. Track relative pitch history
    if (features.hasFace && features.headPose.isValid) {
      this._pitchHistory.push([ts, features.headPose.relPitchDeg]);
    }

    // 3. Prune expired events outside 60s window
    const cutoff60s = ts - this.window60sMs;
    const cutoff30s = ts - this.window30sMs;

    this._blinks = this._blinks.filter((b) => b.endTsMs > cutoff60s);
    this._yawns = this._yawns.filter((y) => y.endTsMs > cutoff60s);
    this._pitchHistory = this._pitchHistory.filter(([t]) => t > cutoff60s);

    // 4. Active closure tracking
    let activeClosureDurationMs = 0.0;
    if (features.hasFace && features.blink.isEyeClosed) {
      if (this._closureStartTsMs === null) {
        this._closureStartTsMs = ts;
      }
      activeClosureDurationMs = Math.max(
        features.blink.currentClosureDurationMs,
        ts - this._closureStartTsMs
      );
    } else {
      this._closureStartTsMs = null;
    }

    // 5. Active off-road glance tracking
    let activeOffRoadDwellMs = 0.0;
    const isLookingAway = features.hasFace && features.headPose.isLookingAway;
    if (isLookingAway) {
      if (this._offRoadStartTsMs === null) {
        this._offRoadStartTsMs = ts;
      }
      activeOffRoadDwellMs = Math.max(0.0, ts - this._offRoadStartTsMs);
    } else {
      this._offRoadStartTsMs = null;
    }

    // 6. Downward dwell tracking
    let downwardDwellMs = 0.0;
    const isDownward =
      features.hasFace &&
      (features.headPose.zone === PoseZone.DASHBOARD || features.headPose.relPitchDeg <= -9.0);
    if (isDownward) {
      if (this._downwardStartTsMs === null) {
        this._downwardStartTsMs = ts;
      }
      downwardDwellMs = Math.max(0.0, ts - this._downwardStartTsMs);
    } else {
      this._downwardStartTsMs = null;
    }

    // 7. Face loss tracking
    let faceLostDurationMs = 0.0;
    if (!features.hasFace) {
      if (this._faceLostStartTsMs === null) {
        this._faceLostStartTsMs = ts;
      }
      faceLostDurationMs = Math.max(0.0, ts - this._faceLostStartTsMs);
    } else {
      this._faceLostStartTsMs = null;
    }

    // 8. Nodding microsleep detection
    // Pitch drops rapidly by >= 15 deg compared to 1-2s ago while eyes are closed >= 800ms
    let isNoddingMicrosleep = false;
    if (activeClosureDurationMs >= this.noddingMinClosureMs && features.hasFace && this._pitchHistory.length >= 2) {
      const currentPitch = features.headPose.relPitchDeg;
      const recentWindowStart = ts - 2000.0;
      const priorPitches = this._pitchHistory.filter(([t]) => t >= recentWindowStart && t <= ts - 300.0);
      if (priorPitches.length > 0) {
        const baselinePriorPitch = Math.max(...priorPitches.map(([, p]) => p));
        const pitchDelta = currentPitch - baselinePriorPitch;
        if (pitchDelta <= this.noddingPitchDropDeg) {
          isNoddingMicrosleep = true;
        }
      }
    }

    // Context classifications
    const isMirrorCheckContext =
      features.headPose.zone === PoseZone.LEFT_MIRROR || features.headPose.zone === PoseZone.RIGHT_MIRROR;
    const isDashboardCheckContext = features.headPose.zone === PoseZone.DASHBOARD;

    // Multi-window aggregations
    const blinks60s = this._blinks;
    const blinks30s = this._blinks.filter((b) => b.endTsMs > cutoff30s);
    const yawns60s = this._yawns;
    const yawns30s = this._yawns.filter((y) => y.endTsMs > cutoff30s);

    const blinkCount60s = blinks60s.length;
    const blinkCount30s = blinks30s.length;

    let meanBlinkDurationMs60s = 0.0;
    let prolongedBlinkCount60s = 0;
    let severeBlinkCount60s = 0;
    let rcrSum = 0.0;
    let rcrCount = 0;

    for (const b of blinks60s) {
      meanBlinkDurationMs60s += b.durationMs;
      if (b.durationMs >= 500.0) prolongedBlinkCount60s++;
      if (b.durationMs >= 800.0) severeBlinkCount60s++;
      if (b.isKinematicsReliable && b.rcr > 0.0) {
        rcrSum += b.rcr;
        rcrCount++;
      }
    }
    if (blinkCount60s > 0) {
      meanBlinkDurationMs60s /= blinkCount60s;
    }
    const meanRcr60s = rcrCount > 0 ? rcrSum / rcrCount : 1.0;

    this._prevTsMs = ts;

    return {
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
      yawnCount60s: yawns60s.length,
      yawnCount30s: yawns30s.length,
      activeYawnDurationMs: features.mar.currentYawnDurationMs,
      perclosPct: features.perclos.perclosPct,
      perclosCoveragePct: features.perclos.coveragePct,
      perclosIsReliable: features.perclos.isReliable,
      effectivePerclosPct: features.perclos.effectivePerclosPct,
      hasFace: features.hasFace,
      faceLostDurationMs,
      isEyeClosed: features.hasFace ? features.blink.isEyeClosed : null,
      completedBlink: features.blink.completedBlink,
      downwardDwellMs,
    };
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
  }
}
