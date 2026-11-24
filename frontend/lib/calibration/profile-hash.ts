import crypto from "crypto";
import type { EngineType } from "@prisma/client";

export interface CanonicalProfileInput {
  engineType: EngineType | "BROWSER" | "PYTHON";
  calibrationMode: string;
  isCalibrated: boolean;
  centerEarMedian: number;
  centerEarMadScaled: number;
  blinkDurationMedianMs: number;
  blinkDurationMadMs: number;
  blinkRcrMedian: number;
  blinkRcrMad: number;
  blinkFreqMedianPerMin: number;
  blinkFreqMadPerMin: number;
  baselinePerclosPct: number;
  blinkAmplitudeMedian: number;
  temporalBlinksSampled: number;
  temporalWindowSec: number;
  sensitivityLevel: string;
  neutralPitchDeg: number;
  neutralYawDeg: number;
  neutralRotMat?: number[][] | null;
  calibrationIsMirrored?: boolean | null;
  calibrationReliability: number;
  zoneBaselines?: Record<string, unknown> | null;
}

export interface ProfileValidationResult {
  valid: boolean;
  error?: string;
}

function round6(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return Math.round(value * 1_000_000) / 1_000_000;
}

export function isValidRotationMatrix(mat: unknown): mat is number[][] {
  if (!Array.isArray(mat) || mat.length !== 3) return false;
  for (const row of mat) {
    if (!Array.isArray(row) || row.length !== 3) return false;
    for (const item of row) {
      if (typeof item !== "number" || !Number.isFinite(item)) return false;
    }
  }
  return true;
}

/**
 * Extract and normalize spatial zones (excluding internal `_`-prefixed metadata blocks)
 * into a deterministic sorted structure supporting both snake_case (Python) and camelCase (Browser).
 */
export function canonicalizeSpatialZones(
  rawZones: Record<string, unknown> | null | undefined
): Record<string, Record<string, number | string | null>> {
  const result: Record<string, Record<string, number | string | null>> = {};
  if (!rawZones || typeof rawZones !== "object" || Array.isArray(rawZones)) {
    return result;
  }

  const zoneKeys = Object.keys(rawZones)
    .filter((k) => !k.startsWith("_"))
    .sort();

  for (const key of zoneKeys) {
    const rawVal = rawZones[key];
    if (!rawVal || typeof rawVal !== "object" || Array.isArray(rawVal)) {
      continue;
    }
    const z = rawVal as Record<string, unknown>;
    const earMedian = Number(z.ear_median ?? z.earMedian);
    const earMad = Number(z.ear_mad_scaled ?? z.earMadScaled);
    const pitch = Number(z.pitch_median_deg ?? z.pitch_deg_median ?? z.pitchMedianDeg ?? 0);
    const yaw = Number(z.yaw_median_deg ?? z.yaw_deg_median ?? z.yawMedianDeg ?? 0);
    const sampleCount = Math.max(0, Math.round(Number(z.sample_count ?? z.sampleCount ?? 0) || 0));
    const dynThresh = z.dynamic_closure_threshold ?? z.dynamicClosureThreshold;
    const leftEar = z.ear_left_median ?? z.earLeftMedian;
    const rightEar = z.ear_right_median ?? z.earRightMedian;

    result[key] = {
      zone: key,
      earMedian: Number.isFinite(earMedian) ? round6(earMedian) : 0,
      earMadScaled: Number.isFinite(earMad) ? round6(earMad) : 0,
      dynamicClosureThreshold:
        dynThresh !== undefined && dynThresh !== null && Number.isFinite(Number(dynThresh))
          ? round6(Number(dynThresh))
          : null,
      pitchMedianDeg: Number.isFinite(pitch) ? round6(pitch) : 0,
      yawMedianDeg: Number.isFinite(yaw) ? round6(yaw) : 0,
      sampleCount,
      earLeftMedian:
        leftEar !== undefined && leftEar !== null && Number.isFinite(Number(leftEar))
          ? round6(Number(leftEar))
          : null,
      earRightMedian:
        rightEar !== undefined && rightEar !== null && Number.isFinite(Number(rightEar))
          ? round6(Number(rightEar))
          : null,
    };
  }

  return result;
}

/**
 * Validate biometric plausibility of a calibration payload before persistence.
 */
export function validateCalibrationBiometrics(
  input: Partial<CanonicalProfileInput>
): ProfileValidationResult {
  const centerEar = Number(input.centerEarMedian);
  if (!Number.isFinite(centerEar) || centerEar <= 0.0 || centerEar > 1.0) {
    return {
      valid: false,
      error: "Invalid calibration payload: centerEarMedian must be a finite EAR value in (0, 1].",
    };
  }

  const reliability = Number(input.calibrationReliability ?? 0.85);
  if (!Number.isFinite(reliability) || reliability < 0.0 || reliability > 1.0) {
    return {
      valid: false,
      error: "Invalid calibration payload: calibrationReliability must be in [0, 1].",
    };
  }

  if (input.neutralRotMat !== undefined && input.neutralRotMat !== null) {
    if (!isValidRotationMatrix(input.neutralRotMat)) {
      return {
        valid: false,
        error: "Invalid calibration payload: neutralRotMat must be a finite 3x3 numeric matrix.",
      };
    }
  }

  if (input.zoneBaselines && typeof input.zoneBaselines === "object" && !Array.isArray(input.zoneBaselines)) {
    for (const [key, rawVal] of Object.entries(input.zoneBaselines)) {
      if (key.startsWith("_")) continue;
      if (!rawVal || typeof rawVal !== "object" || Array.isArray(rawVal)) {
        return {
          valid: false,
          error: `Invalid spatial zone entry for '${key}'.`,
        };
      }
      const z = rawVal as Record<string, unknown>;
      const ear = Number(z.ear_median ?? z.earMedian);
      if (!Number.isFinite(ear) || ear <= 0.0 || ear > 1.0) {
        return {
          valid: false,
          error: `Invalid EAR median for zone '${key}': must be in (0, 1].`,
        };
      }
    }
  }

  return { valid: true };
}

/**
 * Deterministic server-side SHA-256 content hash over canonical biometric profile fields.
 * Excludes volatile timestamps and server-assigned version/UUID fields so identical
 * biometric retries deduplicate reliably regardless of device clock skew.
 */
export function computeCanonicalProfileHash(input: CanonicalProfileInput): string {
  const canonicalPayload = {
    engineType: input.engineType,
    calibrationMode: String(input.calibrationMode),
    isCalibrated: Boolean(input.isCalibrated),
    centerEarMedian: round6(Number(input.centerEarMedian)),
    centerEarMadScaled: round6(Number(input.centerEarMadScaled)),
    blinkDurationMedianMs: round6(Number(input.blinkDurationMedianMs)),
    blinkDurationMadMs: round6(Number(input.blinkDurationMadMs)),
    blinkRcrMedian: round6(Number(input.blinkRcrMedian)),
    blinkRcrMad: round6(Number(input.blinkRcrMad)),
    blinkFreqMedianPerMin: round6(Number(input.blinkFreqMedianPerMin)),
    blinkFreqMadPerMin: round6(Number(input.blinkFreqMadPerMin)),
    baselinePerclosPct: round6(Number(input.baselinePerclosPct)),
    blinkAmplitudeMedian: round6(Number(input.blinkAmplitudeMedian)),
    temporalBlinksSampled: Math.max(0, Math.round(Number(input.temporalBlinksSampled) || 0)),
    temporalWindowSec: round6(Number(input.temporalWindowSec)),
    sensitivityLevel: String(input.sensitivityLevel).toUpperCase(),
    neutralPitchDeg: round6(Number(input.neutralPitchDeg)),
    neutralYawDeg: round6(Number(input.neutralYawDeg)),
    neutralRotMat: isValidRotationMatrix(input.neutralRotMat)
      ? input.neutralRotMat.map((row) => row.map((v) => round6(v)))
      : null,
    calibrationIsMirrored:
      typeof input.calibrationIsMirrored === "boolean" ? input.calibrationIsMirrored : null,
    calibrationReliability: round6(Number(input.calibrationReliability)),
    zoneBaselines: canonicalizeSpatialZones(input.zoneBaselines),
  };

  return crypto
    .createHash("sha256")
    .update(JSON.stringify(canonicalPayload))
    .digest("hex");
}
