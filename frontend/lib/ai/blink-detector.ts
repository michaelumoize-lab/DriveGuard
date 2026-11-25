/**
 * DriveGuard — Temporal Blink Kinematics & Sub-Frame Event Detector in TypeScript.
 *
 * Direct, verified port of backend/app/features/blink_detector.py.
 * Computes:
 * - Deterministic eyelid state machine (OPEN, CLOSING, CLOSED, REOPENING, PROLONGED_CLOSURE, UNCERTAIN_GAP)
 * - Sub-frame linear crossing interpolation:
 *     t* = t_prev + (t_curr - t_prev) * (threshold - ear_prev) / (ear_curr - ear_prev)
 * - Exact physical kinematics: closing velocity, reopening velocity, and RCR (v_reopen / v_close)
 * - Left-censoring on recovery from gaps or face loss (sustained closure monitored without fabricating completed blinks)
 * - Schmitt-trigger hysteresis across closure_threshold and reopen_threshold
 */

import { BlinkClassification, BlinkEvent, BlinkSnapshot, BlinkState } from "./types";

export interface BlinkDetectorOptions {
  closureThreshold?: number;
  earClosureThreshold?: number;
  reopenThreshold?: number;
  minBlinkDurationMs?: number;
  maxBlinkDurationMs?: number;
  prolongedDurationMs?: number;
  microsleepDurationMs?: number;
  minAmplitude?: number;
  maxSamplingGapMs?: number;
}

export class BlinkKinematicsDetector {
  public static readonly CLOSING_NOISE_TOLERANCE: number = 0.002;
  public static readonly CLOSING_STALL_MS: number = 50.0;
  public static readonly REOPENING_TIMEOUT_MS: number = 1000.0;

  public closureThreshold: number;
  public reopenThreshold: number;
  public minBlinkDurationMs: number;
  public prolongedDurationMs: number;
  public microsleepDurationMs: number;
  public minAmplitude: number;
  public maxSamplingGapMs: number;

  private _state: BlinkState = BlinkState.OPEN;
  private _prevTs: number | null = null;
  private _prevEar: number | null = null;

  // Active in-flight blink tracking
  private _blinkStartTs: number | null = null;
  private _closingEntryTs: number | null = null;
  private _closureStartTs: number | null = null;
  private _reopeningEntryTs: number | null = null;
  private _closingOnsetEar: number = 1.0;
  private _closingMinEar: number = 1.0;
  private _closingLastProgressTs: number | null = null;
  private _closingReanchored: boolean = false;
  private _minEar: number = 1.0;
  private _minEarTs: number | null = null;
  private _openEarBefore: number = 0.30;
  private _lastOpenEar: number | null = null;
  private _wasLeftClosed: boolean = false;
  private _wasRightClosed: boolean = false;
  private _isLeftCensored: boolean = false;

  // Totals and historical metrics
  private _totalBlinks: number = 0;
  private _prolongedBlinks: number = 0;
  private _recentDurationMs: number | null = null;
  private _recentRcr: number | null = null;

  constructor(options?: BlinkDetectorOptions) {
    const closure = options?.closureThreshold ?? options?.earClosureThreshold ?? 0.205;
    this.closureThreshold = closure;
    this.reopenThreshold = options?.reopenThreshold ?? (closure + 0.035);
    this.minBlinkDurationMs = options?.minBlinkDurationMs ?? 40.0;
    this.prolongedDurationMs = options?.prolongedDurationMs ?? options?.maxBlinkDurationMs ?? 500.0;
    this.microsleepDurationMs = options?.microsleepDurationMs ?? 800.0;
    this.minAmplitude = options?.minAmplitude ?? 0.04;
    this.maxSamplingGapMs = options?.maxSamplingGapMs ?? 250.0;
  }

  public updateThresholds(closureThreshold: number, reopenThreshold: number): void {
    this.closureThreshold = closureThreshold;
    this.reopenThreshold = reopenThreshold;
    if (this._lastOpenEar !== null && this._lastOpenEar < this.reopenThreshold) {
      this._lastOpenEar = null;
    }
    if (this._state === BlinkState.CLOSING && this._prevEar !== null && this._prevEar >= this.reopenThreshold) {
      this._state = BlinkState.OPEN;
      this._blinkStartTs = null;
      this._closingEntryTs = null;
      this._closureStartTs = null;
      this._closingOnsetEar = 1.0;
      this._closingMinEar = 1.0;
      this._closingLastProgressTs = null;
    }
  }

  public get state(): BlinkState {
    return this._state;
  }

  private _isHoverBeforeFastStep(tCurr: number, earCurr: number): boolean {
    if (this._blinkStartTs === null || this._prevTs === null || this._prevEar === null) {
      return false;
    }
    const preDtMs = this._prevTs - this._blinkStartTs;
    const stepDtMs = tCurr - this._prevTs;
    if (preDtMs < 60.0 || stepDtMs <= 1e-6) {
      return false;
    }
    const preDrop = Math.max(0.0, this._closingOnsetEar - this._prevEar);
    const stepDrop = Math.max(0.0, this._prevEar - earCurr);
    const vPrePerS = (preDrop / preDtMs) * 1000.0;
    const vStepPerS = (stepDrop / stepDtMs) * 1000.0;
    if (vStepPerS > 6.0 * vPrePerS) {
      return true;
    }
    if (vPrePerS < 0.15 && vStepPerS > 2.5 * vPrePerS) {
      return true;
    }
    return false;
  }

  private _interpolateCrossing(
    tPrev: number,
    earPrev: number,
    tCurr: number,
    earCurr: number,
    threshold: number
  ): number {
    const denom = earCurr - earPrev;
    if (Math.abs(denom) < 1e-6) {
      return tCurr;
    }
    let ratio = (threshold - earPrev) / denom;
    ratio = Math.max(0.0, Math.min(1.0, ratio));
    return tPrev + (tCurr - tPrev) * ratio;
  }

  public update(
    leftEar: number | null,
    rightEar?: number | null,
    ear: number | null = null,
    timestampMs: number = 0.0,
    hasFace: boolean = true,
    compensatedEar: number | null = null,
    leftValid: boolean = true,
    rightValid: boolean = true
  ): BlinkSnapshot {
    // Support overloaded call: update(combinedEar, timestampMs, hasFace)
    if (typeof ear === "boolean") {
      hasFace = ear;
      timestampMs = typeof rightEar === "number" ? rightEar : 0.0;
      ear = leftEar;
      rightEar = leftEar;
    } else if (ear === null && typeof rightEar === "number") {
      timestampMs = rightEar;
      ear = leftEar;
      rightEar = leftEar;
    }

    const earVal = ear !== null ? ear : (compensatedEar !== null ? compensatedEar : 0.0);

    // 1. Input Validation: Guard against NaN, Inf, and negative values
    if (
      !Number.isFinite(earVal) ||
      earVal < 0.0 ||
      !Number.isFinite(timestampMs) ||
      timestampMs < 0.0 ||
      (leftEar !== null && !Number.isFinite(leftEar)) ||
      (rightEar !== null && rightEar !== undefined && !Number.isFinite(rightEar))
    ) {
      const isInClosure =
        this._state === BlinkState.CLOSED ||
        this._state === BlinkState.PROLONGED_CLOSURE ||
        this._state === BlinkState.REOPENING;
      const curDur =
        this._prevTs !== null && this._closureStartTs !== null && isInClosure
          ? Math.max(0.0, this._prevTs - this._closureStartTs)
          : 0.0;
      return {
        state: this._state,
        isEyeClosed: isInClosure,
        currentClosureDurationMs: curDur,
        completedBlink: null,
        recentBlinkDurationMs: this._recentDurationMs,
        recentBlinkRcr: this._recentRcr,
        totalBlinkCount: this._totalBlinks,
        prolongedBlinkCount: this._prolongedBlinks,
        samplingGapDetected: this._state === BlinkState.UNCERTAIN_GAP,
      };
    }

    // 2. Face Loss Handling
    if (!hasFace || (!leftValid && !rightValid)) {
      this._handleFaceLost();
      return {
        state: this._state,
        isEyeClosed: false,
        currentClosureDurationMs: 0.0,
        completedBlink: null,
        recentBlinkDurationMs: this._recentDurationMs,
        recentBlinkRcr: this._recentRcr,
        totalBlinkCount: this._totalBlinks,
        prolongedBlinkCount: this._prolongedBlinks,
        samplingGapDetected: this._state === BlinkState.UNCERTAIN_GAP,
      };
    }

    // 3. Timestamp Monotonicity & Sampling Gap Handling
    if (this._prevTs !== null) {
      const gapMs = timestampMs - this._prevTs;
      if (gapMs <= 0.0) {
        const isInClosure =
          this._state === BlinkState.CLOSED ||
          this._state === BlinkState.PROLONGED_CLOSURE ||
          this._state === BlinkState.REOPENING;
        const curDur =
          this._closureStartTs !== null && isInClosure
            ? Math.max(0.0, this._prevTs - this._closureStartTs)
            : 0.0;
        return {
          state: this._state,
          isEyeClosed: isInClosure,
          currentClosureDurationMs: curDur,
          completedBlink: null,
          recentBlinkDurationMs: this._recentDurationMs,
          recentBlinkRcr: this._recentRcr,
          totalBlinkCount: this._totalBlinks,
          prolongedBlinkCount: this._prolongedBlinks,
          samplingGapDetected: this._state === BlinkState.UNCERTAIN_GAP,
        };
      }

      if (gapMs > this.maxSamplingGapMs) {
        this._handleSamplingGap();
        this._prevTs = timestampMs;
        this._prevEar = earVal;
        if (earVal >= this.reopenThreshold) {
          this._lastOpenEar = earVal;
        }
        return {
          state: BlinkState.UNCERTAIN_GAP,
          isEyeClosed: false,
          currentClosureDurationMs: 0.0,
          completedBlink: null,
          recentBlinkDurationMs: this._recentDurationMs,
          recentBlinkRcr: this._recentRcr,
          totalBlinkCount: this._totalBlinks,
          prolongedBlinkCount: this._prolongedBlinks,
          samplingGapDetected: true,
        };
      }
    }

    let completedBlink: BlinkEvent | null = null;
    let currentClosureDurationMs = 0.0;

    const isBelowClosed = earVal < this.closureThreshold;
    const isAboveReopen = earVal >= this.reopenThreshold;

    const lVal = (leftEar !== null && leftEar !== undefined) ? leftEar : earVal;
    const rVal = (rightEar !== null && rightEar !== undefined) ? rightEar : earVal;
    const leftClosedNow = Boolean(leftValid && lVal < this.closureThreshold);
    const rightClosedNow = Boolean(rightValid && rVal < this.closureThreshold);
    const leftBelowReopen = Boolean(leftValid && lVal < this.reopenThreshold);
    const rightBelowReopen = Boolean(rightValid && rVal < this.reopenThreshold);

    let validEyeClosed = false;
    let validEyeBelowReopen = false;
    if (leftValid && rightValid) {
      validEyeClosed = leftClosedNow && rightClosedNow;
      validEyeBelowReopen = leftBelowReopen && rightBelowReopen;
    } else if (leftValid) {
      validEyeClosed = leftClosedNow;
      validEyeBelowReopen = leftBelowReopen;
    } else if (rightValid) {
      validEyeClosed = rightClosedNow;
      validEyeBelowReopen = rightBelowReopen;
    }

    if (
      isBelowClosed ||
      this._state === BlinkState.CLOSING ||
      this._state === BlinkState.CLOSED ||
      this._state === BlinkState.PROLONGED_CLOSURE
    ) {
      if (leftClosedNow && rightClosedNow) {
        this._wasLeftClosed = true;
        this._wasRightClosed = true;
      }
    }

    // 4. State Machine Evaluation
    if (this._state === BlinkState.UNCERTAIN_GAP) {
      if (isBelowClosed) {
        this._state = BlinkState.CLOSED;
        this._closureStartTs = timestampMs;
        this._blinkStartTs = null;
        this._isLeftCensored = true;
        this._minEar = earVal;
        this._minEarTs = timestampMs;
        if (leftClosedNow && rightClosedNow) {
          this._wasLeftClosed = true;
          this._wasRightClosed = true;
        }
      } else if (isAboveReopen) {
        this._state = BlinkState.OPEN;
        this._isLeftCensored = false;
        this._closureStartTs = null;
        this._blinkStartTs = null;
      } else {
        this._state = BlinkState.OPEN;
        this._isLeftCensored = false;
        this._closureStartTs = null;
        this._blinkStartTs = null;
      }
    } else if (this._prevEar !== null && this._prevTs !== null) {
      if (this._state === BlinkState.OPEN) {
        if (isBelowClosed) {
          const tCrossClose = this._interpolateCrossing(
            this._prevTs,
            this._prevEar,
            timestampMs,
            earVal,
            this.closureThreshold
          );
          let tCrossOpen = this._prevTs;
          if (this._prevEar >= this.reopenThreshold) {
            tCrossOpen = this._interpolateCrossing(
              this._prevTs,
              this._prevEar,
              timestampMs,
              earVal,
              this.reopenThreshold
            );
            this._openEarBefore = this._prevEar;
          } else {
            this._openEarBefore = this._lastOpenEar !== null ? this._lastOpenEar : this._prevEar;
          }
          this._blinkStartTs = tCrossOpen;
          this._closingEntryTs = null;
          this._closureStartTs = tCrossClose;
          this._reopeningEntryTs = null;
          this._closingOnsetEar = this.reopenThreshold;
          this._closingMinEar = earVal;
          this._closingLastProgressTs = timestampMs;
          this._closingReanchored = false;
          this._minEar = earVal;
          this._minEarTs = timestampMs;
          this._isLeftCensored = false;
          this._state = BlinkState.CLOSED;
        } else if (this._prevEar >= this.reopenThreshold && earVal < this.reopenThreshold) {
          const tCrossOpen = this._interpolateCrossing(
            this._prevTs,
            this._prevEar,
            timestampMs,
            earVal,
            this.reopenThreshold
          );
          this._openEarBefore = this._prevEar;
          this._blinkStartTs = tCrossOpen;
          this._closingEntryTs = tCrossOpen;
          this._reopeningEntryTs = null;
          this._closingOnsetEar = this.reopenThreshold;
          this._closingMinEar = earVal;
          this._closingLastProgressTs = timestampMs;
          this._closingReanchored = false;
          this._isLeftCensored = false;
          this._state = BlinkState.CLOSING;
        }
      } else if (this._state === BlinkState.CLOSING) {
        if (isBelowClosed) {
          const tCrossClose = this._interpolateCrossing(
            this._prevTs,
            this._prevEar,
            timestampMs,
            earVal,
            this.closureThreshold
          );
          if (
            this._blinkStartTs === null ||
            this._closingLastProgressTs === null ||
            this._isHoverBeforeFastStep(timestampMs, earVal)
          ) {
            this._blinkStartTs = this._prevTs;
            this._closingOnsetEar = this._prevEar;
            this._closingReanchored = true;
          }
          this._closingEntryTs = null;
          this._closureStartTs = tCrossClose;
          this._reopeningEntryTs = null;
          this._minEar = earVal;
          this._minEarTs = timestampMs;
          this._state = BlinkState.CLOSED;
        } else if (earVal >= this.reopenThreshold) {
          this._state = BlinkState.OPEN;
          this._blinkStartTs = null;
          this._closingEntryTs = null;
          this._closureStartTs = null;
          this._closingOnsetEar = 1.0;
          this._closingMinEar = 1.0;
          this._closingLastProgressTs = null;
          this._minEar = 1.0;
          this._minEarTs = null;
          this._wasLeftClosed = false;
          this._wasRightClosed = false;
        } else if (this._closingEntryTs !== null && timestampMs - this._closingEntryTs > 1000.0) {
          this._state = BlinkState.OPEN;
          this._blinkStartTs = null;
          this._closingEntryTs = null;
          this._closureStartTs = null;
          this._closingOnsetEar = 1.0;
          this._closingMinEar = 1.0;
          this._closingLastProgressTs = null;
          this._minEar = 1.0;
          this._minEarTs = null;
          this._lastOpenEar = null;
          this._wasLeftClosed = false;
          this._wasRightClosed = false;
        } else {
          if (this._closingLastProgressTs === null) {
            if (earVal < this._prevEar - 1e-4) {
              this._blinkStartTs = this._prevTs;
              this._closingOnsetEar = this._prevEar;
              this._closingMinEar = earVal;
              this._closingLastProgressTs = timestampMs;
              this._closingReanchored = true;
            } else {
              this._closingMinEar = earVal;
            }
          } else if (earVal < this._closingMinEar - 1e-4) {
            if (this._isHoverBeforeFastStep(timestampMs, earVal)) {
              this._blinkStartTs = this._prevTs;
              this._closingOnsetEar = this._prevEar;
              this._closingReanchored = true;
            }
            this._closingMinEar = earVal;
            this._closingLastProgressTs = timestampMs;
          } else {
            const stalledByRebound = earVal > this._closingMinEar + BlinkKinematicsDetector.CLOSING_NOISE_TOLERANCE;
            const stalledByTime =
              timestampMs - this._closingLastProgressTs >= BlinkKinematicsDetector.CLOSING_STALL_MS;
            if (stalledByRebound || stalledByTime) {
              this._closingLastProgressTs = null;
              this._closingMinEar = earVal;
            }
          }
        }
      } else if (this._state === BlinkState.CLOSED || this._state === BlinkState.PROLONGED_CLOSURE) {
        if (earVal < this._minEar) {
          this._minEar = earVal;
          this._minEarTs = timestampMs;
        }
        if (this._closureStartTs !== null) {
          currentClosureDurationMs = timestampMs - this._closureStartTs;
        }
        if (currentClosureDurationMs >= this.prolongedDurationMs) {
          this._state = BlinkState.PROLONGED_CLOSURE;
        }
        if (earVal > this.closureThreshold) {
          this._state = BlinkState.REOPENING;
          this._reopeningEntryTs = timestampMs;
        }
      }

      if (this._state === BlinkState.REOPENING) {
        if (isAboveReopen) {
          if (this._isLeftCensored) {
            this._state = BlinkState.OPEN;
            this._isLeftCensored = false;
            this._blinkStartTs = null;
            this._closingEntryTs = null;
            this._closureStartTs = null;
            this._reopeningEntryTs = null;
            this._closingOnsetEar = 1.0;
            this._closingMinEar = 1.0;
            this._closingLastProgressTs = null;
            this._closingReanchored = false;
            this._minEar = 1.0;
            this._minEarTs = null;
            this._wasLeftClosed = false;
            this._wasRightClosed = false;
          } else {
            const tEnd = this._interpolateCrossing(
              this._prevTs,
              this._prevEar,
              timestampMs,
              earVal,
              this.reopenThreshold
            );
            const startTs =
              this._blinkStartTs !== null
                ? this._blinkStartTs
                : this._closureStartTs !== null
                  ? this._closureStartTs
                  : this._prevTs;
            const minTs =
              this._minEarTs !== null ? this._minEarTs : startTs + (tEnd - startTs) * 0.5;

            const measuredDurationMs = tEnd - startTs;
            if (measuredDurationMs > 0.0) {
              const durationMs = measuredDurationMs;
              const amplitude = Math.max(0.0, this._openEarBefore - this._minEar);
              const closeDisplacement = this._closingReanchored
                ? Math.max(0.0, this._closingOnsetEar - this._minEar)
                : amplitude;

              const closeDtSec = (minTs - startTs) / 1000.0;
              const reopenDtSec = (tEnd - minTs) / 1000.0;

              let isKinematicsReliable = false;
              let vClose = 0.0;
              let vReopen = 0.0;
              let rcr = 0.0;

              if (closeDtSec > 0.005 && reopenDtSec > 0.005) {
                isKinematicsReliable = true;
                vClose = closeDisplacement / closeDtSec;
                vReopen = amplitude / reopenDtSec;
                rcr = vReopen / Math.max(1e-6, vClose);
              }

              const isBinocular = this._wasLeftClosed && this._wasRightClosed;

              let classification: BlinkClassification;
              if (durationMs < this.minBlinkDurationMs || amplitude < this.minAmplitude) {
                classification = BlinkClassification.UNCERTAIN;
              } else if (durationMs <= this.prolongedDurationMs) {
                classification = BlinkClassification.NORMAL;
              } else if (durationMs < this.microsleepDurationMs) {
                classification = BlinkClassification.PROLONGED;
              } else {
                classification = BlinkClassification.MICROSLEEP_CANDIDATE;
              }

              if (classification !== BlinkClassification.UNCERTAIN) {
                completedBlink = {
                  startTsMs: startTs,
                  minTsMs: minTs,
                  endTsMs: tEnd,
                  durationMs,
                  openEarBefore: this._openEarBefore,
                  minEar: this._minEar,
                  amplitude,
                  closingVelocityPerS: vClose,
                  reopeningVelocityPerS: vReopen,
                  rcr,
                  classification,
                  isBinocular,
                  isKinematicsReliable,
                };

                if (isBinocular) {
                  this._totalBlinks += 1;
                  if (
                    classification === BlinkClassification.PROLONGED ||
                    classification === BlinkClassification.MICROSLEEP_CANDIDATE
                  ) {
                    this._prolongedBlinks += 1;
                  }
                  this._recentDurationMs = durationMs;
                  if (isKinematicsReliable) {
                    this._recentRcr = rcr;
                  }
                }
              }
            }

            this._state = BlinkState.OPEN;
            this._blinkStartTs = null;
            this._closingEntryTs = null;
            this._closureStartTs = null;
            this._reopeningEntryTs = null;
            this._closingOnsetEar = 1.0;
            this._closingMinEar = 1.0;
            this._closingLastProgressTs = null;
            this._closingReanchored = false;
            this._minEar = 1.0;
            this._minEarTs = null;
            this._wasLeftClosed = false;
            this._wasRightClosed = false;
          }
        } else if (earVal < this.closureThreshold) {
          if (earVal < this._minEar) {
            this._minEar = earVal;
            this._minEarTs = timestampMs;
          }
          if (this._closureStartTs === null) {
            this._closureStartTs = this._interpolateCrossing(
              this._prevTs,
              this._prevEar,
              timestampMs,
              earVal,
              this.closureThreshold
            );
          }
          this._reopeningEntryTs = null;
          if (timestampMs - this._closureStartTs >= this.prolongedDurationMs) {
            this._state = BlinkState.PROLONGED_CLOSURE;
          } else {
            this._state = BlinkState.CLOSED;
          }
        } else if (
          this._reopeningEntryTs !== null &&
          timestampMs - this._reopeningEntryTs > BlinkKinematicsDetector.REOPENING_TIMEOUT_MS
        ) {
          this._state = BlinkState.OPEN;
          this._isLeftCensored = false;
          this._blinkStartTs = null;
          this._closingEntryTs = null;
          this._closureStartTs = null;
          this._reopeningEntryTs = null;
          this._closingOnsetEar = 1.0;
          this._closingMinEar = 1.0;
          this._closingLastProgressTs = null;
          this._closingReanchored = false;
          this._minEar = 1.0;
          this._minEarTs = null;
          this._lastOpenEar = null;
          this._wasLeftClosed = false;
          this._wasRightClosed = false;
        }
      }
    }

    if (isAboveReopen) {
      this._lastOpenEar = earVal;
    } else if (this._state === BlinkState.OPEN) {
      this._lastOpenEar = null;
    }
    this._prevTs = timestampMs;
    this._prevEar = earVal;

    const isInClosureState =
      this._state === BlinkState.CLOSED ||
      this._state === BlinkState.PROLONGED_CLOSURE ||
      this._state === BlinkState.REOPENING;
    let hystereticEyeClosed = false;
    if (this._state === BlinkState.REOPENING) {
      hystereticEyeClosed = validEyeBelowReopen;
    } else {
      hystereticEyeClosed = isInClosureState && validEyeClosed;
    }

    if (this._closureStartTs !== null && hystereticEyeClosed) {
      currentClosureDurationMs = Math.max(0.0, timestampMs - this._closureStartTs);
    } else {
      currentClosureDurationMs = 0.0;
    }

    return {
      state: this._state,
      isEyeClosed: hystereticEyeClosed,
      currentClosureDurationMs,
      completedBlink,
      recentBlinkDurationMs: this._recentDurationMs,
      recentBlinkRcr: this._recentRcr,
      totalBlinkCount: this._totalBlinks,
      prolongedBlinkCount: this._prolongedBlinks,
      samplingGapDetected: false,
    };
  }

  private _handleFaceLost(): void {
    this._state = BlinkState.UNCERTAIN_GAP;
    this._blinkStartTs = null;
    this._closingEntryTs = null;
    this._closureStartTs = null;
    this._reopeningEntryTs = null;
    this._closingOnsetEar = 1.0;
    this._closingMinEar = 1.0;
    this._closingLastProgressTs = null;
    this._closingReanchored = false;
    this._minEar = 1.0;
    this._minEarTs = null;
    this._lastOpenEar = null;
    this._wasLeftClosed = false;
    this._wasRightClosed = false;
    this._prevTs = null;
    this._prevEar = null;
    this._isLeftCensored = true;
  }

  private _handleSamplingGap(): void {
    this._state = BlinkState.UNCERTAIN_GAP;
    this._blinkStartTs = null;
    this._closingEntryTs = null;
    this._closureStartTs = null;
    this._reopeningEntryTs = null;
    this._closingOnsetEar = 1.0;
    this._closingMinEar = 1.0;
    this._closingLastProgressTs = null;
    this._closingReanchored = false;
    this._minEar = 1.0;
    this._minEarTs = null;
    this._lastOpenEar = null;
    this._wasLeftClosed = false;
    this._wasRightClosed = false;
    this._prevTs = null;
    this._prevEar = null;
    this._isLeftCensored = true;
  }

  public reset(): void {
    this._state = BlinkState.OPEN;
    this._prevTs = null;
    this._prevEar = null;
    this._blinkStartTs = null;
    this._closingEntryTs = null;
    this._closureStartTs = null;
    this._reopeningEntryTs = null;
    this._closingOnsetEar = 1.0;
    this._closingMinEar = 1.0;
    this._closingLastProgressTs = null;
    this._closingReanchored = false;
    this._minEar = 1.0;
    this._minEarTs = null;
    this._lastOpenEar = null;
    this._wasLeftClosed = false;
    this._wasRightClosed = false;
    this._isLeftCensored = false;
    this._totalBlinks = 0;
    this._prolongedBlinks = 0;
    this._recentDurationMs = null;
    this._recentRcr = null;
  }
}

export { BlinkKinematicsDetector as BlinkDetector };

