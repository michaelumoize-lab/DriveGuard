/**
 * DriveGuard — Personalised Driver Behaviour Profile (PDBP) & 4-Zone Calibration Engine in TypeScript.
 *
 * Direct, hardened port of backend/app/ai/pdbp_calibration.py and compute_zone_thresholds:
 * - Robust 10% bilateral trimming + Median + Scaled MAD (1.4826 * MAD + eps)
 * - Blink-dip rejection (< 0.72 * runningMedian) & head-motion stability gating
 * - SO(3) geodesic medoid neutral rotation matrix (orthonormal with |det(R) - 1| < 1e-5)
 * - Mathematical threshold feasibility verification:
 *     span = ear_median - c >= 0.040, c < r, r - c >= 0.025, ear_median - r >= 0.015
 * - Crash-safe rejection handling, step skipping, and single-zone recalibration
 */

import { isValidRotationMatrix, matMul3x3, transpose3x3 } from "./head-pose";
import {
  CalibrationProgress,
  CalibrationTarget,
  DriverBehaviourProfileData,
  PoseZone,
  ZoneBaselineData,
} from "./types";

export function computeMedian(values: number[]): number {
  if (!values || values.length === 0) return 0.0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) {
    return (sorted[mid - 1] + sorted[mid]) / 2.0;
  }
  return sorted[mid];
}

/**
 * Remove the lowest and highest `trimFraction` tails (default 10% each side)
 * when sample count is sufficient, protecting baseline stats from transient artifacts.
 */
export function bilateralTrim(values: number[], trimFraction: number = 0.10): number[] {
  if (!values || values.length === 0) return [];
  const finite = values.filter((v) => Number.isFinite(v));
  if (finite.length < 5) return finite;
  const sorted = [...finite].sort((a, b) => a - b);
  const cut = Math.floor(sorted.length * Math.max(0, Math.min(0.25, trimFraction)));
  if (cut <= 0 || sorted.length - 2 * cut < 3) return sorted;
  return sorted.slice(cut, sorted.length - cut);
}

export function computeMedianAndScaledMad(
  values: number[],
  minFloor: number = 0.01,
  eps: number = 1e-6
): [number, number] {
  if (!values || values.length === 0) {
    return [0.0, minFloor];
  }
  const med = computeMedian(values);
  const deviations = values.map((v) => Math.abs(v - med));
  const mad = computeMedian(deviations);
  const scaledMad = Math.max(1.4826 * mad + eps, minFloor);
  return [med, scaledMad];
}

export function computeRobustZ(
  x: number,
  medianVal: number,
  scaledMad: number,
  eps: number = 1e-6
): number {
  return (x - medianVal) / (scaledMad + eps);
}

/**
 * Compute dynamic closure threshold, reopening threshold, and feasibility for an EAR baseline.
 * Mirrors backend/app/features/feature_extractor.py:compute_zone_thresholds() 1:1.
 */
export function computeZoneThresholds(
  earMedian: number,
  earMadScaled: number
): {
  closureThreshold: number;
  reopenThreshold: number;
  span: number;
  isFeasible: boolean;
} {
  if (!Number.isFinite(earMedian) || !Number.isFinite(earMadScaled) || earMedian <= 0) {
    return {
      closureThreshold: 0.20,
      reopenThreshold: 0.23,
      span: 0,
      isFeasible: false,
    };
  }
  const rawC = Math.min(earMedian * 0.76, earMedian - 2.3 * earMadScaled);
  const c = Math.max(0.12, Math.min(0.22, rawC));
  const span = earMedian - c;
  const r = Math.min(0.30, Math.max(c + 0.025, Math.min(earMedian * 0.82, earMedian - 0.015)));

  const isFeasible =
    span >= 0.040 - 1e-9 &&
    c < r &&
    r - c >= 0.025 - 1e-9 &&
    earMedian - r >= 0.015 - 1e-9;

  return {
    closureThreshold: c,
    reopenThreshold: r,
    span,
    isFeasible,
  };
}

/**
 * Compute the geodesic medoid on SO(3) from a list of valid 3x3 rotation matrices.
 * Guaranteed to remain an exact orthonormal SO(3) rotation matrix (|det(R) - 1| < 1e-5).
 */
export function computeSo3Medoid(matrices: number[][][]): number[][] | null {
  const valid = matrices.filter((m) => isValidRotationMatrix(m, 1e-4));
  if (valid.length === 0) return null;
  if (valid.length === 1) return valid[0].map((r) => [...r]);

  let bestIdx = 0;
  let bestDistSum = Number.POSITIVE_INFINITY;

  for (let i = 0; i < valid.length; i++) {
    let distSum = 0;
    const RiT = transpose3x3(valid[i]);
    for (let j = 0; j < valid.length; j++) {
      if (i === j) continue;
      const rel = matMul3x3(RiT, valid[j]);
      const trace = rel[0][0] + rel[1][1] + rel[2][2];
      const cosTheta = Math.max(-1.0, Math.min(1.0, (trace - 1.0) / 2.0));
      distSum += Math.acos(cosTheta);
    }
    if (distSum < bestDistSum) {
      bestDistSum = distSum;
      bestIdx = i;
    }
  }

  return valid[bestIdx].map((r) => [...r]);
}

function generateProfileUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `brp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export const CALIBRATION_STAGES: CalibrationTarget[] = [
  {
    zone: PoseZone.CENTER,
    label: "Center Reference",
    description: "Look straight ahead through the windshield in your normal driving posture.",
    targetYawDeg: 0.0,
    targetPitchDeg: 0.0,
    yawToleranceDeg: 12.0,
    pitchToleranceDeg: 10.0,
    durationMs: 4000,
  },
  {
    zone: PoseZone.LEFT_MIRROR,
    label: "Left Wing Mirror",
    description: "Glance at your driver-side left mirror.",
    targetYawDeg: 24.0,
    targetPitchDeg: -2.0,
    yawToleranceDeg: 14.0,
    pitchToleranceDeg: 12.0,
    durationMs: 3000,
  },
  {
    zone: PoseZone.RIGHT_MIRROR,
    label: "Right / Rearview Mirror",
    description: "Glance toward your right mirror or center rearview mirror.",
    targetYawDeg: -24.0,
    targetPitchDeg: -2.0,
    yawToleranceDeg: 14.0,
    pitchToleranceDeg: 12.0,
    durationMs: 3000,
  },
  {
    zone: PoseZone.DASHBOARD,
    label: "Dashboard / Cluster",
    description: "Glance down at your speedometer and instrument cluster.",
    targetYawDeg: 0.0,
    targetPitchDeg: -14.0,
    yawToleranceDeg: 12.0,
    pitchToleranceDeg: 10.0,
    durationMs: 3000,
  },
];

export interface GuidedCalibrationOptions {
  driverId?: string;
  isMirrored?: boolean;
  targetZone?: PoseZone | null;
  existingProfile?: DriverBehaviourProfileData | null;
  minSamplesPerZone?: number;
}

export class GuidedCalibrationEngine {
  private _driverId: string;
  private _stages: CalibrationTarget[];
  private _existingProfile: DriverBehaviourProfileData | null;
  private _minSamplesOverride: number | null;
  private _stageIndex: number = 0;
  private _stageStartTs: number | null = null;
  private _zoneEarSamples: Map<PoseZone, number[]> = new Map();
  private _zonePitchSamples: Map<PoseZone, number[]> = new Map();
  private _zoneYawSamples: Map<PoseZone, number[]> = new Map();
  private _centerRotMats: number[][][] = [];
  private _prevYawDeg: number | null = null;
  private _prevPitchDeg: number | null = null;
  private _isMirrored: boolean = false;
  private _isCompleted: boolean = false;
  private _rejectionReason: string | undefined = undefined;
  private _generatedProfile: DriverBehaviourProfileData | undefined = undefined;

  constructor(
    driverIdOrOptions: string | GuidedCalibrationOptions = "driver_default",
    isMirrored: boolean = false
  ) {
    if (typeof driverIdOrOptions === "string") {
      this._driverId = driverIdOrOptions;
      this._isMirrored = isMirrored;
      this._stages = [...CALIBRATION_STAGES];
      this._existingProfile = null;
      this._minSamplesOverride = null;
    } else {
      this._driverId = driverIdOrOptions.driverId ?? "driver_default";
      this._isMirrored = driverIdOrOptions.isMirrored ?? false;
      this._existingProfile = driverIdOrOptions.existingProfile ?? null;
      this._minSamplesOverride = driverIdOrOptions.minSamplesPerZone ?? null;
      if (driverIdOrOptions.targetZone) {
        const matched = CALIBRATION_STAGES.find((s) => s.zone === driverIdOrOptions.targetZone);
        this._stages = matched ? [matched] : [...CALIBRATION_STAGES];
      } else {
        this._stages = [...CALIBRATION_STAGES];
      }
    }
    this.reset();
  }

  public get isCompleted(): boolean {
    return this._isCompleted;
  }

  public get rejectionReason(): string | undefined {
    return this._rejectionReason;
  }

  public get generatedProfile(): DriverBehaviourProfileData | undefined {
    return this._generatedProfile;
  }

  public reset(): void {
    this._stageIndex = 0;
    this._stageStartTs = null;
    this._zoneEarSamples.clear();
    this._zonePitchSamples.clear();
    this._zoneYawSamples.clear();
    for (const stage of this._stages) {
      this._zoneEarSamples.set(stage.zone, []);
      this._zonePitchSamples.set(stage.zone, []);
      this._zoneYawSamples.set(stage.zone, []);
    }
    this._centerRotMats = [];
    this._prevYawDeg = null;
    this._prevPitchDeg = null;
    this._isCompleted = false;
    this._rejectionReason = undefined;
    this._generatedProfile = undefined;
  }

  private _minSamplesForStage(stage: CalibrationTarget): number {
    if (this._minSamplesOverride !== null && this._minSamplesOverride > 0) {
      return this._minSamplesOverride;
    }
    return Math.max(12, Math.round((stage.durationMs / 1000) * 10));
  }

  /**
   * Manually skip the current zone step (e.g., peripheral mirror zone) and advance.
   * Completes calibration if skipping the final stage.
   */
  public skipStep(): CalibrationProgress {
    if (this._isCompleted) {
      return this._buildTerminalProgress(0, 0);
    }

    const currentStage = this._stages[Math.min(this._stageIndex, this._stages.length - 1)];
    this._stageIndex++;
    this._stageStartTs = null;
    this._prevYawDeg = null;
    this._prevPitchDeg = null;

    if (this._stageIndex >= this._stages.length) {
      this._completeCalibration();
      return this._buildTerminalProgress(0, 0);
    }

    const nextStage = this._stages[this._stageIndex];
    const overallProgressPct = Math.min(
      100,
      Math.round((this._stageIndex / this._stages.length) * 100)
    );
    return {
      currentZone: nextStage.zone,
      stepIndex: this._stageIndex + 1,
      totalSteps: this._stages.length,
      zoneProgressPct: 0,
      overallProgressPct,
      inZone: false,
      guidanceText: `Skipped ${currentStage.label}. ${nextStage.description}`,
      currentYawDeg: 0,
      currentPitchDeg: 0,
      isCompleted: false,
    };
  }

  private _buildTerminalProgress(yawDeg: number, pitchDeg: number): CalibrationProgress {
    const lastStage = this._stages[this._stages.length - 1];
    const rejected = Boolean(this._rejectionReason);
    return {
      currentZone: lastStage.zone,
      stepIndex: this._stages.length,
      totalSteps: this._stages.length,
      zoneProgressPct: rejected ? 0 : 100,
      overallProgressPct: rejected ? 0 : 100,
      inZone: !rejected,
      guidanceText: rejected
        ? `Calibration rejected: ${this._rejectionReason}`
        : "Calibration completed successfully!",
      currentYawDeg: yawDeg,
      currentPitchDeg: pitchDeg,
      isCompleted: true,
      rejectedReason: this._rejectionReason,
      profile: this._generatedProfile,
    };
  }

  /**
   * Ingest one raw (uncompensated) EAR + head pose sample for calibration.
   */
  public update(
    timestampMs: number,
    rawWeightedEar: number,
    pitchDeg: number,
    yawDeg: number,
    hasFace: boolean,
    isEyeValid: boolean,
    rotMat?: number[][] | null
  ): CalibrationProgress {
    if (this._isCompleted || this._stageIndex >= this._stages.length) {
      return this._buildTerminalProgress(yawDeg, pitchDeg);
    }

    const stage = this._stages[this._stageIndex];
    const samples = this._zoneEarSamples.get(stage.zone) || [];
    const minSamplesNeeded = this._minSamplesForStage(stage);

    const yawDelta = Math.abs(yawDeg - stage.targetYawDeg);
    const pitchDelta = Math.abs(pitchDeg - stage.targetPitchDeg);
    const inZone =
      hasFace &&
      isEyeValid &&
      Number.isFinite(rawWeightedEar) &&
      rawWeightedEar > 0.0 &&
      rawWeightedEar <= 1.0 &&
      yawDelta <= stage.yawToleranceDeg &&
      pitchDelta <= stage.pitchToleranceDeg;

    let guidanceText = stage.description;
    let rejectedTransient = false;

    if (!hasFace) {
      guidanceText = "Face not detected. Position yourself inside the camera frame.";
    } else if (!inZone) {
      if (yawDeg < stage.targetYawDeg - stage.yawToleranceDeg) {
        guidanceText = `Turn more to your left towards ${stage.label}.`;
      } else if (yawDeg > stage.targetYawDeg + stage.yawToleranceDeg) {
        guidanceText = `Turn more to your right towards ${stage.label}.`;
      } else if (pitchDeg < stage.targetPitchDeg - stage.pitchToleranceDeg) {
        guidanceText = `Raise your head slightly towards ${stage.label}.`;
      } else {
        guidanceText = `Tilt your head down slightly towards ${stage.label}.`;
      }
    }

    if (inZone) {
      // Head-motion stability check (reject rapid transitions)
      const motionJump =
        this._prevYawDeg !== null &&
        this._prevPitchDeg !== null &&
        (Math.abs(yawDeg - this._prevYawDeg) > 6.5 ||
          Math.abs(pitchDeg - this._prevPitchDeg) > 5.5);

      // Blink-dip rejection (< 0.72 * running median once >= 5 baseline samples exist)
      const runningMedian = samples.length >= 5 ? computeMedian(samples) : null;
      const isBlinkDip =
        runningMedian !== null && rawWeightedEar < 0.72 * runningMedian;

      if (motionJump) {
        rejectedTransient = true;
        guidanceText = "Hold head steady for a moment while sampling…";
      } else if (isBlinkDip) {
        rejectedTransient = true;
        guidanceText = "Blink filtered — keep eyes open naturally…";
      } else {
        if (this._stageStartTs === null) {
          this._stageStartTs = timestampMs;
        }
        samples.push(rawWeightedEar);
        this._zonePitchSamples.get(stage.zone)?.push(pitchDeg);
        this._zoneYawSamples.get(stage.zone)?.push(yawDeg);

        if (stage.zone === PoseZone.CENTER && rotMat && isValidRotationMatrix(rotMat, 1e-4)) {
          this._centerRotMats.push(rotMat.map((r) => [...r]));
        }
      }
    } else {
      this._stageStartTs = null;
    }

    if (hasFace && Number.isFinite(yawDeg) && Number.isFinite(pitchDeg)) {
      this._prevYawDeg = yawDeg;
      this._prevPitchDeg = pitchDeg;
    }

    const zoneProgressPct = Math.min(
      100,
      Math.round((samples.length / Math.max(1, minSamplesNeeded)) * 100)
    );

    if (zoneProgressPct >= 100) {
      this._stageIndex++;
      this._stageStartTs = null;
      this._prevYawDeg = null;
      this._prevPitchDeg = null;
      if (this._stageIndex >= this._stages.length) {
        this._completeCalibration();
        return this._buildTerminalProgress(yawDeg, pitchDeg);
      }
    }

    const activeStage = this._stages[Math.min(this._stageIndex, this._stages.length - 1)];
    const activeSamples = this._zoneEarSamples.get(activeStage.zone) || [];
    const activeZonePct = Math.min(
      100,
      Math.round((activeSamples.length / Math.max(1, this._minSamplesForStage(activeStage))) * 100)
    );
    const overallProgressPct = Math.min(
      100,
      Math.round(
        ((this._stageIndex * 100 + activeZonePct) / (this._stages.length * 100)) * 100
      )
    );

    return {
      currentZone: activeStage.zone,
      stepIndex: this._stageIndex + 1,
      totalSteps: this._stages.length,
      zoneProgressPct: activeZonePct,
      overallProgressPct,
      inZone: inZone && !rejectedTransient,
      guidanceText: activeStage.zone === stage.zone ? guidanceText : activeStage.description,
      currentYawDeg: yawDeg,
      currentPitchDeg: pitchDeg,
      isCompleted: false,
    };
  }

  private _completeCalibration(): void {
    // Always mark completed (even on rejection) so subsequent update() calls never index out-of-bounds
    this._isCompleted = true;

    const rawCenterEars = this._zoneEarSamples.get(PoseZone.CENTER) || [];
    const trimmedCenter = bilateralTrim(rawCenterEars, 0.10);

    let centerMed: number;
    let centerMad: number;
    let centerPitch: number;
    let centerYaw: number;
    let neutralRotMat: number[][] | null;

    if (trimmedCenter.length >= 10) {
      [centerMed, centerMad] = computeMedianAndScaledMad(trimmedCenter, 0.015);
      centerPitch = computeMedian(
        bilateralTrim(this._zonePitchSamples.get(PoseZone.CENTER) || [], 0.10)
      );
      centerYaw = computeMedian(
        bilateralTrim(this._zoneYawSamples.get(PoseZone.CENTER) || [], 0.10)
      );
      neutralRotMat = computeSo3Medoid(this._centerRotMats);
    } else if (this._existingProfile && this._stages.length === 1 && this._stages[0].zone !== PoseZone.CENTER) {
      // Single peripheral-zone recalibration: preserve existing calibrated CENTER reference
      centerMed = this._existingProfile.centerEarMedian;
      centerMad = this._existingProfile.centerEarMadScaled;
      centerPitch = this._existingProfile.neutralPitchDeg;
      centerYaw = this._existingProfile.neutralYawDeg;
      neutralRotMat = this._existingProfile.neutralRotMat ?? null;
    } else {
      this._rejectionReason = "Insufficient stable open-eye samples collected during CENTER calibration.";
      this._generatedProfile = undefined;
      return;
    }

    const centerThresh = computeZoneThresholds(centerMed, centerMad);
    if (!centerThresh.isFeasible) {
      this._rejectionReason = `Baseline EAR (${centerMed.toFixed(3)}) or variance (${centerMad.toFixed(3)}) failed mathematical feasibility constraints.`;
      this._generatedProfile = undefined;
      return;
    }

    // Start from existing zones if single-zone recalibration, otherwise build full 4-zone map
    const zoneBaselines: Record<string, ZoneBaselineData> = this._existingProfile?.zoneBaselines
      ? { ...this._existingProfile.zoneBaselines }
      : {};

    for (const stage of CALIBRATION_STAGES) {
      const rawEars = this._zoneEarSamples.get(stage.zone) || [];
      const trimmedEars = bilateralTrim(rawEars, 0.10);
      const pitches = bilateralTrim(this._zonePitchSamples.get(stage.zone) || [], 0.10);
      const yaws = bilateralTrim(this._zoneYawSamples.get(stage.zone) || [], 0.10);

      if (trimmedEars.length >= 6) {
        const [zMed, zMad] = computeMedianAndScaledMad(trimmedEars, 0.015);
        const zThresh = computeZoneThresholds(zMed, zMad);
        if (zThresh.isFeasible) {
          zoneBaselines[stage.zone] = {
            zone: stage.zone,
            earMedian: Number(zMed.toFixed(4)),
            earMadScaled: Number(zMad.toFixed(4)),
            pitchMedianDeg: Number(computeMedian(pitches).toFixed(1)),
            yawMedianDeg: Number(computeMedian(yaws).toFixed(1)),
            sampleCount: trimmedEars.length,
          };
          continue;
        }
      }

      // Preserve existing valid zone during single-zone recalibration
      if (zoneBaselines[stage.zone]) {
        continue;
      }

      // Deterministic scaled fallback relative to feasible CENTER baseline
      let scale = 1.0;
      let pitchOffset = 0.0;
      let yawOffset = 0.0;
      if (stage.zone === PoseZone.LEFT_MIRROR) {
        scale = 0.92;
        yawOffset = 20.0;
      } else if (stage.zone === PoseZone.RIGHT_MIRROR) {
        scale = 0.92;
        yawOffset = -20.0;
      } else if (stage.zone === PoseZone.DASHBOARD) {
        scale = 0.88;
        pitchOffset = -12.0;
      }

      zoneBaselines[stage.zone] = {
        zone: stage.zone,
        earMedian: Number((centerMed * scale).toFixed(4)),
        earMadScaled: Number(Math.max(centerMad, 0.015).toFixed(4)),
        pitchMedianDeg: Number((centerPitch + pitchOffset).toFixed(1)),
        yawMedianDeg: Number((centerYaw + yawOffset).toFixed(1)),
        sampleCount: 0,
      };
    }

    const calibratedZonesCaptured = Object.values(zoneBaselines).filter(
      (z) => z.sampleCount > 0
    ).length;
    const calibrationMode =
      calibratedZonesCaptured >= 4
        ? "FOUR_ZONE"
        : calibratedZonesCaptured >= 3
          ? "THREE_ZONE"
          : "CENTER_ONLY";

    this._generatedProfile = {
      driverId: this._driverId,
      engineType: "BROWSER",
      profileVersion: (this._existingProfile?.profileVersion ?? 0) + 1,
      profileUuid: generateProfileUuid(),
      contentHash: null,
      calibrationMode,
      centerEarMedian: Number(centerMed.toFixed(4)),
      centerEarMadScaled: Number(centerMad.toFixed(4)),
      blinkDurationMedianMs: this._existingProfile?.blinkDurationMedianMs ?? 235.0,
      blinkDurationMadMs: this._existingProfile?.blinkDurationMadMs ?? 40.0,
      blinkRcrMedian: this._existingProfile?.blinkRcrMedian ?? 0.95,
      blinkRcrMad: this._existingProfile?.blinkRcrMad ?? 0.18,
      blinkFreqMedianPerMin: this._existingProfile?.blinkFreqMedianPerMin ?? 16.0,
      blinkFreqMadPerMin: this._existingProfile?.blinkFreqMadPerMin ?? 4.5,
      neutralPitchDeg: Number(centerPitch.toFixed(1)),
      neutralYawDeg: Number(centerYaw.toFixed(1)),
      calibrationReliability: calibratedZonesCaptured >= 4 ? 0.92 : 0.82,
      activeCenterEarMedian: Number(centerMed.toFixed(4)),
      baselinePerclosPct: this._existingProfile?.baselinePerclosPct ?? 3.5,
      blinkAmplitudeMedian: this._existingProfile?.blinkAmplitudeMedian ?? 0.14,
      temporalBlinksSampled: this._existingProfile?.temporalBlinksSampled ?? 0,
      temporalWindowSec: 120.0,
      sensitivityLevel: this._existingProfile?.sensitivityLevel ?? "STANDARD",
      version: 1,
      calibratedAt: new Date().toISOString(),
      neutralRotMat,
      calibrationIsMirrored: this._isMirrored,
      zoneBaselines,
    };
  }
}
