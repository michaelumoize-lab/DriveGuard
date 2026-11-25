/**
 * DriveGuard — Mouth Aspect Ratio (MAR) & Yawn Detection Module in TypeScript.
 *
 * Direct, verified port of backend/app/features/mar.py.
 * Computes:
 * - Mouth Aspect Ratio (MAR) from 3 inner-lip vertical pairs normalized by outer-lip width
 * - Schmitt trigger hysteresis (open threshold 0.55, hold threshold 0.45)
 * - 200 ms dropout grace period for transient tracking loss
 * - 250 ms maximum sampling gap before resetting in-flight candidates
 * - Monotonic timestamp enforcement and resting baseline tracking
 */

import { MouthAspectRatioResult, NormalizedLandmark, YawnEvent } from "./types";
import { Point2D } from "./ear";

export const MOUTH_OUTER_CORNERS = [61, 291] as const;
export const MOUTH_VERTICAL_PAIRS = [
  [13, 14],
  [81, 178],
  [311, 402],
] as const;

export const DEFAULT_MAR_OPEN_THRESHOLD = 0.55;
export const DEFAULT_MAR_HOLD_THRESHOLD = 0.45;
export const DEFAULT_MIN_YAWN_DURATION_MS = 1500.0;
export const DEFAULT_MAX_YAWN_DURATION_MS = 8000.0;
export const DEFAULT_DROPOUT_GRACE_MS = 200.0;
export const DEFAULT_MAX_SAMPLING_GAP_MS = 400.0;
export const DEFAULT_CLOSED_BASELINE_THRESHOLD = 0.28;
export const DEFAULT_RESTING_BASELINE_MAR = 0.15;
export const DEFAULT_BASELINE_ALPHA = 0.02;

const ALL_MOUTH_INDICES = [61, 291, 13, 14, 81, 178, 311, 402] as const;

/**
 * Compute Mouth Aspect Ratio (MAR) and evaluate geometric landmark validity.
 *
 * MAR = (mean vertical inner-lip distance) / (outer corner horizontal width + eps)
 */
export function calculateMarWithValidity(
  landmarks: ArrayLike<NormalizedLandmark | Point2D>,
  aspectRatio: number = 16.0 / 9.0,
  isNormalized?: boolean | null,
  eps: number = 1e-6
): [number, boolean] {
  const maxIdx = Math.max(...ALL_MOUTH_INDICES);
  if (!landmarks || landmarks.length <= maxIdx) {
    return [0.0, false];
  }

  const coords: Point2D[] = [];
  for (let i = 0; i < ALL_MOUTH_INDICES.length; i++) {
    const pt = landmarks[ALL_MOUTH_INDICES[i]];
    if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
      return [0.0, false];
    }
    coords.push({ x: pt.x, y: pt.y });
  }

  // Coordinate space determination
  let normMode: boolean;
  if (isNormalized !== undefined && isNormalized !== null) {
    normMode = isNormalized;
  } else {
    let maxCoord = 0;
    for (let i = 0; i < coords.length; i++) {
      maxCoord = Math.max(maxCoord, Math.abs(coords[i].x), Math.abs(coords[i].y));
    }
    normMode = maxCoord <= 2.0;
  }

  const arFactor = normMode ? aspectRatio : 1.0;

  // Horizontal mouth width (between outer lip corners 61 and 291)
  const pLeft = coords[0];
  const pRight = coords[1];
  const horiz = Math.hypot((pLeft.x - pRight.x) * arFactor, pLeft.y - pRight.y);
  if (horiz < eps) {
    return [0.0, false];
  }

  // Vertical inner-lip distances (3 canonical pairs: 13-14, 81-178, 311-402)
  let vertSum = 0.0;
  for (let i = 0; i < MOUTH_VERTICAL_PAIRS.length; i++) {
    const pTop = coords[2 + 2 * i];
    const pBot = coords[3 + 2 * i];
    vertSum += Math.hypot((pTop.x - pBot.x) * arFactor, pTop.y - pBot.y);
  }

  const meanVert = vertSum / MOUTH_VERTICAL_PAIRS.length;
  const mar = meanVert / (horiz + eps);

  if (mar < 0.0 || mar > 2.5) {
    return [0.0, false];
  }

  return [mar, true];
}

/**
 * Compute scalar Mouth Aspect Ratio (MAR) helper.
 */
export function calculateMar(
  landmarks: ArrayLike<NormalizedLandmark | Point2D>,
  aspectRatio: number = 16.0 / 9.0,
  isNormalized?: boolean | null,
  eps: number = 1e-6
): number {
  const [mar] = calculateMarWithValidity(landmarks, aspectRatio, isNormalized, eps);
  return mar;
}

export class YawnDetector {
  public marOpenThreshold: number;
  public minYawnDurationMs: number;
  public maxYawnDurationMs: number;
  public marHoldThreshold: number;
  public dropoutGraceMs: number;
  public maxSamplingGapMs: number;
  public closedBaselineThreshold: number;
  public baselineAlpha: number;

  private _isOpening: boolean = false;
  private _openingExceededMax: boolean = false;
  private _openingStartTs: number | null = null;
  private _lastOpenValidTs: number | null = null;
  private _observedOpenMs: number = 0.0;
  private _openMarSamples: number[] = [];
  private _peakMar: number = 0.0;
  private _totalYawns: number = 0;

  private _lastValidTs: number | null = null;
  private _dropoutStartTs: number | null = null;
  private _baselineMar: number = DEFAULT_RESTING_BASELINE_MAR;

  constructor(options?: {
    marOpenThreshold?: number;
    onsetThreshold?: number;
    minYawnDurationMs?: number;
    maxYawnDurationMs?: number;
    marHoldThreshold?: number;
    releaseThreshold?: number;
    dropoutGraceMs?: number;
    maxSamplingGapMs?: number;
    closedBaselineThreshold?: number;
    baselineAlpha?: number;
  }) {
    this.marOpenThreshold = options?.marOpenThreshold ?? options?.onsetThreshold ?? DEFAULT_MAR_OPEN_THRESHOLD;
    this.minYawnDurationMs = options?.minYawnDurationMs ?? DEFAULT_MIN_YAWN_DURATION_MS;
    this.maxYawnDurationMs = options?.maxYawnDurationMs ?? DEFAULT_MAX_YAWN_DURATION_MS;
    this.marHoldThreshold = options?.marHoldThreshold ?? options?.releaseThreshold ?? DEFAULT_MAR_HOLD_THRESHOLD;
    this.dropoutGraceMs = options?.dropoutGraceMs ?? DEFAULT_DROPOUT_GRACE_MS;
    this.maxSamplingGapMs = options?.maxSamplingGapMs ?? DEFAULT_MAX_SAMPLING_GAP_MS;
    this.closedBaselineThreshold = options?.closedBaselineThreshold ?? DEFAULT_CLOSED_BASELINE_THRESHOLD;
    this.baselineAlpha = options?.baselineAlpha ?? DEFAULT_BASELINE_ALPHA;
  }

  public get baselineMar(): number {
    return this._baselineMar;
  }

  public update(
    landmarks: ArrayLike<NormalizedLandmark | Point2D>,
    timestampMs: number,
    aspectRatio: number = 16.0 / 9.0,
    hasFace: boolean = true,
    isNormalized?: boolean | null,
    isLookingAway?: boolean | null
  ): MouthAspectRatioResult {
    const ts = Number(timestampMs);
    if (!Number.isFinite(ts) || ts < 0.0) {
      return this._buildResult(0.0, false, false, this._lastValidTs || 0.0, null);
    }

    if (this._lastValidTs !== null) {
      const dt = ts - this._lastValidTs;
      if (dt <= 0.0) {
        return this._buildResult(0.0, false, false, this._lastValidTs, null);
      }
      if (dt > this.maxSamplingGapMs) {
        this._abortOpening();
        this._dropoutStartTs = null;
      }
    }

    let mar = 0.0;
    let isValid = false;
    if (hasFace && landmarks && landmarks.length > 0) {
      [mar, isValid] = calculateMarWithValidity(landmarks, aspectRatio, isNormalized);
    }

    if (!isValid) {
      if (this._isOpening) {
        if (this._dropoutStartTs === null) {
          this._dropoutStartTs = ts;
        }
        const dropoutElapsed = ts - this._dropoutStartTs;
        if (dropoutElapsed <= this.dropoutGraceMs) {
          this._lastValidTs = ts;
          return this._buildResult(0.0, false, false, ts, null);
        } else {
          this._abortOpening();
          this._dropoutStartTs = null;
        }
      }
      this._lastValidTs = ts;
      return this._buildResult(0.0, false, false, ts, null);
    }

    let wasInDropout = false;
    if (this._dropoutStartTs !== null) {
      if (ts - this._dropoutStartTs > this.dropoutGraceMs) {
        this._abortOpening();
      } else {
        wasInDropout = true;
      }
      this._dropoutStartTs = null;
    }

    let completedYawn: YawnEvent | null = null;
    let isMouthOpen = false;

    if (!this._isOpening) {
      if (mar >= this.marOpenThreshold) {
        this._isOpening = true;
        this._openingStartTs = ts;
        this._lastOpenValidTs = ts;
        this._observedOpenMs = 0.0;
        this._openMarSamples = [mar];
        this._peakMar = mar;
        isMouthOpen = true;
      } else {
        isMouthOpen = false;
        if (mar <= this.closedBaselineThreshold && isLookingAway !== true) {
          this._baselineMar = (1.0 - this.baselineAlpha) * this._baselineMar + this.baselineAlpha * mar;
          this._baselineMar = Math.max(0.05, Math.min(0.35, this._baselineMar));
        }
      }
    } else {
      if (mar >= this.marHoldThreshold) {
        isMouthOpen = true;
        if (this._lastOpenValidTs !== null && !wasInDropout) {
          const stepOpenDt = ts - this._lastOpenValidTs;
          this._observedOpenMs += stepOpenDt;
        }
        this._lastOpenValidTs = ts;
        this._peakMar = Math.max(this._peakMar, mar);
        this._openMarSamples.push(mar);
        if (this._observedOpenMs > this.maxYawnDurationMs) {
          this._openingExceededMax = true;
        }
      } else {
        isMouthOpen = false;
        if (this._openingStartTs !== null) {
          const wallClockDuration = ts - this._openingStartTs;
          const observedDuration = this._observedOpenMs;

          if (this.minYawnDurationMs <= observedDuration && observedDuration <= this.maxYawnDurationMs) {
            const sum = this._openMarSamples.reduce((a, b) => a + b, 0);
            const meanMar = this._openMarSamples.length > 0 ? sum / this._openMarSamples.length : this._peakMar;
            completedYawn = {
              startTsMs: this._openingStartTs,
              endTsMs: ts,
              durationMs: observedDuration,
              peakMar: this._peakMar,
              observedDurationMs: observedDuration,
              wallClockDurationMs: wallClockDuration,
              meanMar,
              openingRatio: this._peakMar / Math.max(this._baselineMar, 1e-6),
            };
            this._totalYawns += 1;
          }
        }
        this._abortOpening();
      }
    }

    this._lastValidTs = ts;

    return this._buildResult(mar, isMouthOpen, true, ts, completedYawn);
  }

  private _abortOpening(): void {
    this._isOpening = false;
    this._openingExceededMax = false;
    this._openingStartTs = null;
    this._lastOpenValidTs = null;
    this._observedOpenMs = 0.0;
    this._openMarSamples = [];
    this._peakMar = 0.0;
  }

  private _buildResult(
    mar: number,
    isMouthOpen: boolean,
    isValid: boolean,
    currentTs: number,
    completedYawn: YawnEvent | null
  ): MouthAspectRatioResult {
    let wallClockDuration = 0.0;
    let observedDuration = 0.0;

    if (
      isValid &&
      this._isOpening &&
      !this._openingExceededMax &&
      this._openingStartTs !== null
    ) {
      wallClockDuration = Math.max(0.0, currentTs - this._openingStartTs);
      observedDuration = this._observedOpenMs;
    }

    const activeWithinBounds =
      isValid &&
      this._isOpening &&
      !this._openingExceededMax &&
      observedDuration <= this.maxYawnDurationMs;

    // 750 ms onset gate (0.5 * minYawnDurationMs) matching Python mar.py
    const isYawnInProgress =
      activeWithinBounds && observedDuration >= this.minYawnDurationMs * 0.5;
    const isAuthentic =
      activeWithinBounds && observedDuration >= this.minYawnDurationMs;

    const openingRatio = mar / Math.max(this._baselineMar, 1e-6);

    return {
      mar,
      isMouthOpen,
      isYawnInProgress,
      currentYawnDurationMs: wallClockDuration,
      isAuthenticYawn: isAuthentic,
      completedYawn,
      totalYawnCount: this._totalYawns,
      isValid,
      baselineMar: this._baselineMar,
      openingRatio,
      peakMar: isValid && this._isOpening ? this._peakMar : mar,
      observedYawnDurationMs: observedDuration,
    };
  }

  public reset(): void {
    this._abortOpening();
    this._totalYawns = 0;
    this._lastValidTs = null;
    this._dropoutStartTs = null;
    this._baselineMar = DEFAULT_RESTING_BASELINE_MAR;
  }
}

export { YawnDetector as MarDetector };

