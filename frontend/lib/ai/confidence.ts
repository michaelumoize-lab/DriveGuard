/**
 * DriveGuard — Multi-Signal Reliability and Confidence Scoring in TypeScript.
 *
 * Direct port of backend/app/intelligence/confidence.py combining:
 * - OpticalFrameQuality (meanLuminance, blurVarianceScore, lightingCondition, blurCondition, overallQuality)
 * - MediaPipe face/eye/pose/mouth geometric validity
 * - Calibration operational flags (isCalibrated, isCalibrating)
 *
 * Enables genuine OperationalState.DEGRADED gating on severe underexposure/blur/face-loss
 * and SUSPECTED_HEAD_SLUMP detection when head pose is viable while eyes are occluded.
 */

import type { OpticalFrameQuality } from "@/types/camera";
import type { ConfidenceAssessment } from "./decision-engine";
import { BlinkState, DriverFeatures, PoseZone } from "./types";

export enum ConfidenceTier {
  HIGH = "HIGH",
  ACCEPTABLE = "ACCEPTABLE",
  DEGRADED = "DEGRADED",
  UNAVAILABLE = "UNAVAILABLE",
}

export interface SignalConfidenceAssessment extends ConfidenceAssessment {
  tier: ConfidenceTier;
  reasons: string[];
  isOpticalViable: boolean;
}

export interface ConfidenceEvaluatorOptions {
  minReliableConfidence?: number;
  minViableLuminance?: number;
  maxViableLuminance?: number;
  minViableBlurVariance?: number;
}

export class SignalConfidenceEvaluator {
  public readonly minReliableConfidence: number;
  public readonly minViableLuminance: number;
  public readonly maxViableLuminance: number;
  public readonly minViableBlurVariance: number;

  constructor(options: ConfidenceEvaluatorOptions = {}) {
    this.minReliableConfidence = options.minReliableConfidence ?? 0.50;
    this.minViableLuminance = options.minViableLuminance ?? 22.0;
    this.maxViableLuminance = options.maxViableLuminance ?? 242.0;
    this.minViableBlurVariance = options.minViableBlurVariance ?? 15.0;
  }

  public evaluate(
    features: DriverFeatures,
    quality?: OpticalFrameQuality | null,
    options?: {
      isCalibrated?: boolean;
      isCalibrating?: boolean;
      landmarkVisibility?: number | null;
    }
  ): SignalConfidenceAssessment {
    const reasons: string[] = [];
    const isCalibrated = options?.isCalibrated ?? true;
    const isCalibrating = options?.isCalibrating ?? false;

    // 1. Face presence check
    if (!features.hasFace) {
      reasons.push("No driver face detected");
      return {
        overallConfidence: 0.0,
        tier: ConfidenceTier.UNAVAILABLE,
        isReliable: false,
        ocularViable: false,
        headPoseViable: false,
        yawnViable: false,
        isOcularAvailable: false,
        isCalibrated,
        isCalibrating,
        isOpticalViable: quality ? this._isOpticalViable(quality) : true,
        reasons,
      };
    }

    let score = 1.0;

    // Optional landmark visibility penalty
    if (
      options?.landmarkVisibility !== undefined &&
      options.landmarkVisibility !== null &&
      Number.isFinite(options.landmarkVisibility)
    ) {
      if (options.landmarkVisibility < 0.45) {
        score *= 0.45;
        reasons.push(`Low landmark visibility (${options.landmarkVisibility.toFixed(2)})`);
      } else if (options.landmarkVisibility < 0.70) {
        score *= 0.75;
        reasons.push(`Marginal landmark visibility (${options.landmarkVisibility.toFixed(2)})`);
      }
    }

    // 2. Geometric validity checks
    const earGeomValid = Boolean(features.ear.isGeometricValid ?? features.ear.isValid);
    if (!earGeomValid) {
      score *= 0.60;
      reasons.push("Eye landmarks degenerate or occluded");
    }

    if (!features.headPose.isValid || features.headPose.zone === PoseZone.UNKNOWN) {
      score *= 0.70;
      reasons.push("Head pose estimation did not converge");
    }

    // 3. Optical frame quality checks
    let isOpticalViable = true;
    if (quality) {
      isOpticalViable = this._isOpticalViable(quality);
      if (!isOpticalViable) {
        score *= 0.25;
        reasons.push(
          `Optical quality unviable (${quality.overallQuality}: lum=${quality.meanLuminance.toFixed(0)}, blur=${quality.blurVarianceScore.toFixed(1)})`
        );
      } else {
        if (quality.lightingCondition !== "OPTIMAL") {
          score *= 0.75;
          reasons.push(`Suboptimal illumination: ${quality.lightingCondition}`);
        }
        if (quality.blurCondition === "BLURRY") {
          score *= 0.70;
          reasons.push(`Motion blur detected (var=${quality.blurVarianceScore.toFixed(1)})`);
        }
        if (quality.overallQuality === "POOR") {
          score *= 0.80;
          reasons.push("Poor overall optical quality");
        }
      }
    }

    score = Math.max(0.0, Math.min(1.0, score));

    let tier: ConfidenceTier;
    if (score >= 0.80) {
      tier = ConfidenceTier.HIGH;
    } else if (score >= 0.50) {
      tier = ConfidenceTier.ACCEPTABLE;
    } else if (score >= 0.25) {
      tier = ConfidenceTier.DEGRADED;
    } else {
      tier = ConfidenceTier.UNAVAILABLE;
    }

    const isReliable = isOpticalViable && score >= this.minReliableConfidence;

    // Independent channel viability evaluation
    const headPoseViable =
      features.hasFace &&
      isOpticalViable &&
      features.headPose.isValid &&
      features.headPose.zone !== PoseZone.UNKNOWN;

    const blinkGap =
      Boolean(features.blink.samplingGapDetected) ||
      features.blink.state === BlinkState.UNCERTAIN_GAP;

    const ocularViable =
      features.hasFace &&
      isOpticalViable &&
      features.ear.isValid &&
      isCalibrated &&
      !blinkGap;

    const yawnViable =
      features.hasFace &&
      isOpticalViable &&
      features.mar.isValid;

    return {
      overallConfidence: Number(score.toFixed(3)),
      tier,
      isReliable,
      ocularViable,
      headPoseViable,
      yawnViable,
      isOcularAvailable: ocularViable,
      isCalibrated,
      isCalibrating,
      isOpticalViable,
      reasons,
    };
  }

  private _isOpticalViable(quality: OpticalFrameQuality): boolean {
    if (quality.overallQuality === "SEVERELY_DEGRADED") {
      return false;
    }
    if (
      Number.isFinite(quality.meanLuminance) &&
      (quality.meanLuminance < this.minViableLuminance ||
        quality.meanLuminance > this.maxViableLuminance)
    ) {
      return false;
    }
    if (
      Number.isFinite(quality.blurVarianceScore) &&
      quality.blurVarianceScore < this.minViableBlurVariance &&
      quality.blurCondition === "BLURRY"
    ) {
      return false;
    }
    return true;
  }

  public reset(): void {
    // Stateless per-frame evaluator (kept for parity with Python SignalConfidenceEvaluator.reset())
  }
}
