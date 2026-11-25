/**
 * DriveGuard — Canonical Driver Profile Normalizer & Feasibility Validator.
 *
 * Bridges all profile representations across Browser AI, Prisma/Neon PostgreSQL,
 * IndexedDB, and FastAPI snake_case DTOs:
 * - Safely handles null/undefined profiles and zoneBaselines without throwing
 * - Normalizes both snake_case and camelCase spatial zone keys
 * - Unpacks neutralRotMat & calibrationIsMirrored from dedicated columns or legacy _poseMetadata
 * - Verifies finite EAR values and mathematical threshold feasibility (closure < reopen < openMedian)
 */

import { isValidRotationMatrix } from "./head-pose";
import { computeZoneThresholds } from "./pdbp-calibration";
import { DriverBehaviourProfileData, PoseZone, ZoneBaselineData } from "./types";

export interface ZoneThresholdPair {
  closureThreshold: number;
  reopenThreshold: number;
  earMedian: number;
  earMadScaled: number;
  isFeasible: boolean;
}

export interface NormalizedDriverProfile {
  isCalibrated: boolean;
  isFeasible: boolean;
  profile: DriverBehaviourProfileData | null;
  zoneThresholds: Map<PoseZone, ZoneThresholdPair>;
  rejectionReason: string | null;
}

const SPATIAL_ZONES: readonly PoseZone[] = [
  PoseZone.CENTER,
  PoseZone.LEFT_MIRROR,
  PoseZone.RIGHT_MIRROR,
  PoseZone.DASHBOARD,
] as const;

function finiteNum(val: unknown): number | null {
  if (val === null || val === undefined || typeof val === "boolean") return null;
  const n = Number(val);
  return Number.isFinite(n) ? n : null;
}

function finiteOrDefault(val: unknown, fallback: number): number {
  const n = finiteNum(val);
  return n !== null ? n : fallback;
}

function extractValidRotMat(candidate: unknown): number[][] | null {
  if (!Array.isArray(candidate) || candidate.length !== 3) return null;
  const parsed: number[][] = [];
  for (const row of candidate) {
    if (!Array.isArray(row) || row.length !== 3) return null;
    const nums: number[] = [];
    for (const item of row) {
      if (typeof item !== "number" || !Number.isFinite(item)) return null;
      nums.push(item);
    }
    parsed.push(nums);
  }
  return isValidRotationMatrix(parsed, 0.05) ? parsed : null;
}

function normalizeZoneEntry(zoneKey: string, rawVal: unknown): ZoneBaselineData | null {
  if (!rawVal || typeof rawVal !== "object" || Array.isArray(rawVal)) return null;
  const z = rawVal as Record<string, unknown>;
  const earMedian = finiteNum(z.earMedian ?? z.ear_median);
  if (earMedian === null || earMedian <= 0.0 || earMedian > 1.0) return null;

  const rawMad = finiteNum(z.earMadScaled ?? z.ear_mad_scaled);
  const earMadScaled = rawMad !== null && rawMad >= 0.0 ? Math.max(rawMad, 0.005) : 0.018;

  const entry: ZoneBaselineData = {
    zone: zoneKey,
    earMedian,
    earMadScaled,
    pitchMedianDeg: finiteOrDefault(z.pitchMedianDeg ?? z.pitch_median_deg ?? z.pitch_deg_median, 0.0),
    yawMedianDeg: finiteOrDefault(z.yawMedianDeg ?? z.yaw_median_deg ?? z.yaw_deg_median, 0.0),
    sampleCount: Math.max(0, Math.round(finiteOrDefault(z.sampleCount ?? z.sample_count, 0))),
  };

  const leftEar = finiteNum(z.earLeftMedian ?? z.ear_left_median);
  if (leftEar !== null) entry.earLeftMedian = leftEar;
  const rightEar = finiteNum(z.earRightMedian ?? z.ear_right_median);
  if (rightEar !== null) entry.earRightMedian = rightEar;

  return entry;
}

/**
 * Normalize and mathematically validate any raw driver profile object.
 * Never throws on malformed/null inputs; explicitly signals uncalibrated or infeasible states.
 */
export function normalizeDriverProfile(raw: unknown): NormalizedDriverProfile {
  const emptyResult = (rejectionReason: string): NormalizedDriverProfile => ({
    isCalibrated: false,
    isFeasible: false,
    profile: null,
    zoneThresholds: new Map(),
    rejectionReason,
  });

  if (!raw || typeof raw !== "object" || Array.isArray(raw)) {
    return emptyResult("NO_PROFILE");
  }

  const src = raw as Record<string, unknown>;

  const rawCalibrationMode = String(
    src.calibrationMode ?? src.calibration_mode ?? ""
  ).trim();
  const explicitIsCalibrated =
    src.isCalibrated !== undefined
      ? Boolean(src.isCalibrated)
      : src.is_calibrated !== undefined
        ? Boolean(src.is_calibrated)
        : undefined;

  // Reject explicitly uncalibrated profiles or default placeholders
  if (
    explicitIsCalibrated === false ||
    rawCalibrationMode.toUpperCase() === "UNCALIBRATED_DEFAULT"
  ) {
    return emptyResult("UNCALIBRATED_PROFILE");
  }

  // Extract raw zones object safely (may be null/undefined)
  const rawZonesObj = src.zoneBaselines ?? src.zone_baselines;
  const rawZonesRecord =
    rawZonesObj && typeof rawZonesObj === "object" && !Array.isArray(rawZonesObj)
      ? (rawZonesObj as Record<string, unknown>)
      : null;

  const poseMeta =
    rawZonesRecord &&
    rawZonesRecord._poseMetadata &&
    typeof rawZonesRecord._poseMetadata === "object" &&
    !Array.isArray(rawZonesRecord._poseMetadata)
      ? (rawZonesRecord._poseMetadata as Record<string, unknown>)
      : null;

  const zoneBaselines: Record<string, ZoneBaselineData> = {};
  if (rawZonesRecord) {
    for (const [key, val] of Object.entries(rawZonesRecord)) {
      if (key.startsWith("_")) continue;
      const normalized = normalizeZoneEntry(key, val);
      if (normalized) {
        zoneBaselines[key] = normalized;
      }
    }
  }

  // If explicitIsCalibrated was omitted (e.g. legacy DriverBehaviourProfileData),
  // require at least a valid CENTER zone or calibrated mode to treat as calibrated.
  const centerZone = zoneBaselines[PoseZone.CENTER];
  const centerEarMedian = finiteNum(
    src.centerEarMedian ?? src.center_ear_median ?? centerZone?.earMedian
  );
  if (centerEarMedian === null || centerEarMedian <= 0.0 || centerEarMedian > 1.0) {
    return emptyResult("INVALID_CENTER_EAR");
  }

  const rawCenterMad = finiteNum(
    src.centerEarMadScaled ?? src.center_ear_mad_scaled ?? centerZone?.earMadScaled
  );
  const centerEarMadScaled =
    rawCenterMad !== null && rawCenterMad >= 0.0 ? Math.max(rawCenterMad, 0.005) : 0.018;

  // Check mathematical feasibility of CENTER baseline
  const centerComputed = computeZoneThresholds(centerEarMedian, centerEarMadScaled);
  const centerFinite =
    Number.isFinite(centerComputed.closureThreshold) &&
    Number.isFinite(centerComputed.reopenThreshold) &&
    centerComputed.closureThreshold > 0.08 &&
    centerComputed.closureThreshold < centerComputed.reopenThreshold &&
    centerComputed.reopenThreshold < centerEarMedian;

  if (!centerComputed.isFeasible || !centerFinite) {
    return emptyResult("INFEASIBLE_CENTER_THRESHOLDS");
  }

  // If neither explicitIsCalibrated===true nor calibrated spatial zones exist, reject
  if (explicitIsCalibrated === undefined && Object.keys(zoneBaselines).length === 0) {
    return emptyResult("MISSING_SPATIAL_ZONES");
  }

  // Build validated threshold map for all 4 cockpit zones
  const zoneThresholds = new Map<PoseZone, ZoneThresholdPair>();
  for (const zone of SPATIAL_ZONES) {
    const zb = zoneBaselines[zone];
    if (zb) {
      const t = computeZoneThresholds(zb.earMedian, zb.earMadScaled);
      if (
        t.isFeasible &&
        Number.isFinite(t.closureThreshold) &&
        Number.isFinite(t.reopenThreshold) &&
        t.closureThreshold > 0.08 &&
        t.closureThreshold < t.reopenThreshold &&
        t.reopenThreshold < zb.earMedian
      ) {
        zoneThresholds.set(zone, {
          closureThreshold: t.closureThreshold,
          reopenThreshold: t.reopenThreshold,
          earMedian: zb.earMedian,
          earMadScaled: zb.earMadScaled,
          isFeasible: true,
        });
        continue;
      }
    }

    // Fallback scaled from feasible CENTER baseline for missing/infeasible peripheral zone
    const scale =
      zone === PoseZone.LEFT_MIRROR || zone === PoseZone.RIGHT_MIRROR
        ? 0.92
        : zone === PoseZone.DASHBOARD
          ? 0.88
          : 1.0;
    const scaledMedian = centerEarMedian * scale;
    const scaledMad = Math.max(centerEarMadScaled, 0.015);
    const fallbackT = computeZoneThresholds(scaledMedian, scaledMad);
    if (fallbackT.isFeasible && fallbackT.closureThreshold < fallbackT.reopenThreshold) {
      zoneThresholds.set(zone, {
        closureThreshold: fallbackT.closureThreshold,
        reopenThreshold: fallbackT.reopenThreshold,
        earMedian: scaledMedian,
        earMadScaled: scaledMad,
        isFeasible: true,
      });
    } else {
      zoneThresholds.set(zone, {
        closureThreshold: centerComputed.closureThreshold,
        reopenThreshold: centerComputed.reopenThreshold,
        earMedian: centerEarMedian,
        earMadScaled: centerEarMadScaled,
        isFeasible: true,
      });
    }
  }

  // Unpack neutralRotMat & calibrationIsMirrored from top-level columns or legacy _poseMetadata
  const neutralRotMat = extractValidRotMat(
    src.neutralRotMat ?? src.neutral_rot_mat ?? poseMeta?.neutral_rot_mat
  );
  const rawMirrored =
    src.calibrationIsMirrored ??
    src.calibration_is_mirrored ??
    poseMeta?.calibration_is_mirrored;
  const calibrationIsMirrored =
    typeof rawMirrored === "boolean" ? rawMirrored : null;

  const rawEngine = String(src.engineType ?? src.engine_type ?? "BROWSER").toUpperCase();
  const engineType = rawEngine === "PYTHON" ? "PYTHON" : "BROWSER";

  const rawCalAt = src.calibratedAt ?? src.calibrated_at;
  const calibratedAt =
    rawCalAt instanceof Date
      ? rawCalAt.toISOString()
      : typeof rawCalAt === "string" && rawCalAt.trim().length > 0
        ? rawCalAt
        : undefined;

  const normalizedProfile: DriverBehaviourProfileData = {
    driverId: String(src.driverId ?? src.driver_id ?? src.userId ?? "driver_default"),
    engineType,
    profileVersion: Math.max(
      1,
      Math.round(finiteOrDefault(src.profileVersion ?? src.profile_version, 1))
    ),
    profileUuid:
      typeof (src.profileUuid ?? src.profile_uuid) === "string"
        ? String(src.profileUuid ?? src.profile_uuid)
        : undefined,
    contentHash:
      typeof (src.contentHash ?? src.content_hash) === "string"
        ? String(src.contentHash ?? src.content_hash)
        : null,
    calibrationMode: rawCalibrationMode || "FOUR_ZONE",
    centerEarMedian,
    centerEarMadScaled,
    blinkDurationMedianMs: finiteOrDefault(
      src.blinkDurationMedianMs ?? src.blink_duration_median_ms,
      235.0
    ),
    blinkDurationMadMs: finiteOrDefault(
      src.blinkDurationMadMs ?? src.blink_duration_mad_ms,
      40.0
    ),
    blinkRcrMedian: finiteOrDefault(src.blinkRcrMedian ?? src.blink_rcr_median, 0.95),
    blinkRcrMad: finiteOrDefault(src.blinkRcrMad ?? src.blink_rcr_mad, 0.18),
    blinkFreqMedianPerMin: finiteOrDefault(
      src.blinkFreqMedianPerMin ?? src.blink_freq_median_per_min,
      16.0
    ),
    blinkFreqMadPerMin: finiteOrDefault(
      src.blinkFreqMadPerMin ?? src.blink_freq_mad_per_min,
      4.5
    ),
    baselinePerclosPct: finiteOrDefault(
      src.baselinePerclosPct ?? src.baseline_perclos_pct,
      3.5
    ),
    blinkAmplitudeMedian: finiteOrDefault(
      src.blinkAmplitudeMedian ?? src.blink_amplitude_median,
      0.14
    ),
    temporalBlinksSampled: Math.max(
      0,
      Math.round(finiteOrDefault(src.temporalBlinksSampled ?? src.temporal_blinks_sampled, 0))
    ),
    temporalWindowSec: finiteOrDefault(
      src.temporalWindowSec ?? src.temporal_window_sec,
      120.0
    ),
    sensitivityLevel: String(src.sensitivityLevel ?? src.sensitivity_level ?? "STANDARD"),
    neutralPitchDeg: finiteOrDefault(src.neutralPitchDeg ?? src.neutral_pitch_deg, 0.0),
    neutralYawDeg: finiteOrDefault(src.neutralYawDeg ?? src.neutral_yaw_deg, 0.0),
    neutralRotMat,
    calibrationIsMirrored,
    calibrationReliability: Math.max(
      0.0,
      Math.min(1.0, finiteOrDefault(src.calibrationReliability ?? src.calibration_reliability, 0.85))
    ),
    version: Math.max(1, Math.round(finiteOrDefault(src.version, 1))),
    calibratedAt,
    zoneBaselines,
  };

  return {
    isCalibrated: true,
    isFeasible: true,
    profile: normalizedProfile,
    zoneThresholds,
    rejectionReason: null,
  };
}
