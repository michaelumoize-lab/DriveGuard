/**
 * DriveGuard — PERCLOS (Percentage of Eye Closure) Accumulator in TypeScript.
 *
 * Direct, verified port of backend/app/features/perclos.py.
 * Standardized P80 implementation over sliding time window (default: 60.0 seconds).
 *
 * Computes:
 * - Time-weighted exact elapsed time integration of eye closure
 * - Boundary clipping at window cutoff
 * - Active observation coverage tracking and reliability gating (min 50% coverage)
 * - Safe handling of sampling gaps (> 250ms treated as unobserved time)
 * - Monotonic timestamp validation
 */

import { PerclosResult } from "./types";

export const PERCLOS_VARIANT = "P80";
export const DEFAULT_PERCLOS_WINDOW_SEC = 60.0;
export const DEFAULT_MIN_COVERAGE_PCT = 50.0;
export const DEFAULT_MAX_VALID_DT_SEC = 0.25;

interface ObservationInterval {
  startTsMs: number;
  endTsMs: number;
  isClosed: boolean;
}

export class PerclosCalculator {
  public windowSec: number;
  public minCoveragePct: number;
  public maxValidDtSec: number;

  private _samples: ObservationInterval[] = [];
  private _prevTsMs: number | null = null;
  private _prevValid: boolean = false;

  constructor(
    windowSec: number = DEFAULT_PERCLOS_WINDOW_SEC,
    minCoveragePct: number = DEFAULT_MIN_COVERAGE_PCT,
    maxValidDtSec: number = DEFAULT_MAX_VALID_DT_SEC
  ) {
    if (!Number.isFinite(windowSec) || windowSec <= 0.0) {
      throw new Error("windowSec must be positive.");
    }
    if (!Number.isFinite(minCoveragePct) || minCoveragePct <= 0.0 || minCoveragePct > 100.0) {
      throw new Error("minCoveragePct must be in (0.0, 100.0]");
    }
    if (!Number.isFinite(maxValidDtSec) || maxValidDtSec <= 0.0) {
      throw new Error("maxValidDtSec must be positive.");
    }

    this.windowSec = windowSec;
    this.minCoveragePct = minCoveragePct;
    this.maxValidDtSec = maxValidDtSec;
  }

  public update(
    timestampMs: number,
    isClosed?: boolean | null,
    hasFace: boolean = true,
    isValid: boolean = true
  ): PerclosResult {
    const ts = Number(timestampMs);
    if (!Number.isFinite(ts) || ts < 0.0) {
      return this._evaluate(this._prevTsMs);
    }

    if (this._prevTsMs !== null) {
      if (ts < this._prevTsMs) {
        return this._evaluate(this._prevTsMs);
      }
      if (ts === this._prevTsMs) {
        return this._evaluate(this._prevTsMs);
      }
    }

    const dtSec = this._prevTsMs !== null ? (ts - this._prevTsMs) / 1000.0 : 0.0;
    const currentValid = Boolean(hasFace && isValid && isClosed !== null && isClosed !== undefined);

    const wasObserved = currentValid && this._prevValid;
    if (wasObserved && 0.0 < dtSec && dtSec <= this.maxValidDtSec && this._prevTsMs !== null) {
      this._samples.push({
        startTsMs: this._prevTsMs,
        endTsMs: ts,
        isClosed: Boolean(isClosed),
      });
    }

    this._prevTsMs = ts;
    this._prevValid = currentValid;

    return this._evaluate(ts);
  }

  private _evaluate(referenceTsMs: number | null): PerclosResult {
    if (referenceTsMs === null || referenceTsMs <= 0.0) {
      return {
        perclosPct: 0.0,
        effectivePerclosPct: null,
        coveragePct: 0.0,
        isReliable: false,
        closedDurationSec: 0.0,
        observedDurationSec: 0.0,
        windowSec: this.windowSec,
        unobservedDurationSec: this.windowSec,
      };
    }

    const windowMs = this.windowSec * 1000.0;
    const windowCutoffMs = referenceTsMs - windowMs;

    // Prune expired samples
    while (this._samples.length > 0 && this._samples[0].endTsMs <= windowCutoffMs) {
      this._samples.shift();
    }

    let totalObservedSec = 0.0;
    let totalClosedSec = 0.0;

    for (let i = 0; i < this._samples.length; i++) {
      const sample = this._samples[i];
      const effStart = Math.max(sample.startTsMs, windowCutoffMs);
      const effDt = Math.max(0.0, (sample.endTsMs - effStart) / 1000.0);
      totalObservedSec += effDt;
      if (sample.isClosed) {
        totalClosedSec += effDt;
      }
    }

    totalObservedSec = Math.min(this.windowSec, totalObservedSec);
    totalClosedSec = Math.min(totalObservedSec, totalClosedSec);

    const unobservedDurationSec = Math.max(0.0, this.windowSec - totalObservedSec);
    const coveragePct = Math.min(100.0, Math.max(0.0, (totalObservedSec / this.windowSec) * 100.0));
    const isReliable = coveragePct >= this.minCoveragePct;

    let perclosPct = 0.0;
    if (totalObservedSec > 1e-4) {
      perclosPct = Math.min(100.0, Math.max(0.0, (totalClosedSec / totalObservedSec) * 100.0));
    }

    return {
      perclosPct,
      effectivePerclosPct: isReliable ? perclosPct : null,
      coveragePct,
      isReliable,
      closedDurationSec: totalClosedSec,
      observedDurationSec: totalObservedSec,
      windowSec: this.windowSec,
      unobservedDurationSec,
    };
  }

  public reset(): void {
    this._samples = [];
    this._prevTsMs = null;
    this._prevValid = false;
  }
}

export { PerclosCalculator as RollingPerclosCalculator };

