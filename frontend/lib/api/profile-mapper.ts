/**
 * Canonical mapper between Next.js / Prisma driver profile records and FastAPI's
 * strict snake_case `DriverProfileDTO` (`backend/app/schemas/profile.py`).
 *
 * Eliminates duplicated profile-building code across API routes and React components,
 * preserves `neutral_rot_mat` & `calibration_is_mirrored` (HIGH-7), and preserves
 * backend-computed `dynamic_closure_threshold` without re-deriving thresholds in JS.
 */

import type { CalibratedProfilePayload, ZoneBaselineEntry } from "@/types/telemetry";

export interface DriverProfileSource {
  driverId?: string;
  driver_id?: string;
  version?: number;
  calibratedAt?: Date | string | null;
  calibrated_at?: string | null;
  isCalibrated?: boolean;
  is_calibrated?: boolean;
  calibrationMode?: string | null;
  calibration_mode?: string | null;
  centerEarMedian?: number;
  center_ear_median?: number;
  centerEarMadScaled?: number;
  center_ear_mad_scaled?: number;
  blinkDurationMedianMs?: number;
  blink_duration_median_ms?: number;
  blinkDurationMadMs?: number;
  blink_duration_mad_ms?: number;
  blinkRcrMedian?: number;
  blink_rcr_median?: number;
  blinkRcrMad?: number;
  blink_rcr_mad?: number;
  blinkFreqMedianPerMin?: number;
  blink_freq_median_per_min?: number;
  blinkFreqMadPerMin?: number;
  blink_freq_mad_per_min?: number;
  baselinePerclosPct?: number;
  baseline_perclos_pct?: number;
  blinkAmplitudeMedian?: number;
  blink_amplitude_median?: number;
  temporalBlinksSampled?: number;
  temporal_blinks_sampled?: number;
  temporalWindowSec?: number;
  temporal_window_sec?: number;
  sensitivityLevel?: string;
  sensitivity_level?: string;
  neutralPitchDeg?: number;
  neutral_pitch_deg?: number;
  neutralYawDeg?: number;
  neutral_yaw_deg?: number;
  neutralRotMat?: number[][] | null;
  neutral_rot_mat?: number[][] | null;
  calibrationIsMirrored?: boolean | null;
  calibration_is_mirrored?: boolean | null;
  calibrationReliability?: number;
  calibration_reliability?: number;
  zoneBaselines?: Record<string, unknown> | null;
  zone_baselines?: Record<string, unknown> | null;
}

function finiteOrDefault(val: unknown, fallback: number): number {
  const n = Number(val);
  return Number.isFinite(n) ? n : fallback;
}

function isValidRotMat(mat: unknown): mat is number[][] {
  if (!Array.isArray(mat) || mat.length !== 3) return false;
  for (const row of mat) {
    if (!Array.isArray(row) || row.length !== 3) return false;
    for (const v of row) {
      if (typeof v !== "number" || !Number.isFinite(v)) return false;
    }
  }
  return true;
}

/**
 * Extract pure spatial zones for FastAPI `DriverProfileDTO.zone_baselines`,
 * stripping internal `_`-prefixed JSONB metadata blocks (e.g. `_temporalBaseline`, `_poseMetadata`).
 */
export function extractSpatialZoneBaselines(
  rawZones: Record<string, unknown> | null | undefined,
  fallbackCenterEar: number,
  fallbackCenterMad: number
): Record<string, ZoneBaselineEntry> {
  const spatialZones: Record<string, ZoneBaselineEntry> = {};
  if (!rawZones || typeof rawZones !== "object" || Array.isArray(rawZones)) {
    return spatialZones;
  }

  for (const [zoneKey, rawVal] of Object.entries(rawZones)) {
    if (zoneKey.startsWith("_") || !rawVal || typeof rawVal !== "object" || Array.isArray(rawVal)) {
      continue;
    }
    const zv = rawVal as Record<string, unknown>;
    const earMedian = finiteOrDefault(zv.ear_median, fallbackCenterEar);
    const earMad = finiteOrDefault(zv.ear_mad_scaled, fallbackCenterMad);
    const storedThreshold = Number(zv.dynamic_closure_threshold);
    // Preserve the backend-computed dynamic_closure_threshold when valid; only fall back if absent
    const dynamicThreshold =
      Number.isFinite(storedThreshold) && storedThreshold >= 0.0 && storedThreshold <= 1.0
        ? storedThreshold
        : Math.min(0.22, Math.max(0.12, Math.min(earMedian * 0.76, earMedian - 2.3 * earMad)));

    const entry: ZoneBaselineEntry = {
      zone: zoneKey,
      ear_median: earMedian,
      ear_mad_scaled: earMad,
      dynamic_closure_threshold: dynamicThreshold,
      pitch_median_deg: finiteOrDefault(zv.pitch_median_deg ?? zv.pitch_deg_median, 0.0),
      yaw_median_deg: finiteOrDefault(zv.yaw_median_deg ?? zv.yaw_deg_median, 0.0),
      sample_count: Math.max(0, Math.round(finiteOrDefault(zv.sample_count, 0))),
    };

    if (zv.ear_left_median !== undefined && zv.ear_left_median !== null && Number.isFinite(Number(zv.ear_left_median))) {
      entry.ear_left_median = Number(zv.ear_left_median);
    }
    if (zv.ear_right_median !== undefined && zv.ear_right_median !== null && Number.isFinite(Number(zv.ear_right_median))) {
      entry.ear_right_median = Number(zv.ear_right_median);
    }
    if (zv.blendshape_open_median !== undefined && zv.blendshape_open_median !== null && Number.isFinite(Number(zv.blendshape_open_median))) {
      entry.blendshape_open_median = Number(zv.blendshape_open_median);
    }
    if (zv.confidence_median !== undefined && zv.confidence_median !== null && Number.isFinite(Number(zv.confidence_median))) {
      entry.confidence_median = Number(zv.confidence_median);
    }

    spatialZones[zoneKey] = entry;
  }

  return spatialZones;
}

/**
 * Build a strictly-conforming snake_case FastAPI `DriverProfileDTO` payload.
 */
export function buildDriverProfileDtoPayload(
  source: DriverProfileSource,
  driverIdOverride?: string
): CalibratedProfilePayload {
  const driverId = driverIdOverride || source.driver_id || source.driverId || "driver_default";
  const rawZones = (source.zone_baselines ?? source.zoneBaselines) as Record<string, unknown> | null | undefined;
  const poseMeta =
    rawZones && typeof rawZones === "object" && !Array.isArray(rawZones)
      ? (rawZones._poseMetadata as Record<string, unknown> | undefined)
      : undefined;

  const isCalibrated = Boolean(source.is_calibrated ?? source.isCalibrated ?? false);
  const rawMode = source.calibration_mode ?? source.calibrationMode;
  const calibrationMode = isCalibrated
    ? rawMode && rawMode !== "UNCALIBRATED_DEFAULT"
      ? String(rawMode)
      : "4ZONE"
    : "UNCALIBRATED_DEFAULT";

  const centerEarMedian = finiteOrDefault(source.center_ear_median ?? source.centerEarMedian, 0.28);
  const centerEarMad = finiteOrDefault(source.center_ear_mad_scaled ?? source.centerEarMadScaled, 0.018);

  const rawCalAt = source.calibrated_at ?? source.calibratedAt;
  const calibratedAt =
    rawCalAt instanceof Date
      ? rawCalAt.toISOString()
      : typeof rawCalAt === "string" && rawCalAt.trim().length > 0
        ? rawCalAt
        : isCalibrated
          ? new Date().toISOString()
          : null;

  const candidateRotMat = source.neutral_rot_mat ?? source.neutralRotMat ?? poseMeta?.neutral_rot_mat;
  const neutralRotMat = isValidRotMat(candidateRotMat) ? candidateRotMat : null;

  const candidateMirrored =
    source.calibration_is_mirrored ?? source.calibrationIsMirrored ?? poseMeta?.calibration_is_mirrored;
  const calibrationIsMirrored =
    typeof candidateMirrored === "boolean" ? candidateMirrored : null;

  return {
    version: Math.max(1, Math.round(finiteOrDefault(source.version, 1))),
    driver_id: driverId,
    calibrated_at: calibratedAt,
    is_calibrated: isCalibrated,
    calibration_mode: calibrationMode,
    center_ear_median: centerEarMedian,
    blink_duration_median_ms: finiteOrDefault(
      source.blink_duration_median_ms ?? source.blinkDurationMedianMs,
      235.0
    ),
    blink_duration_mad_ms: finiteOrDefault(
      source.blink_duration_mad_ms ?? source.blinkDurationMadMs,
      40.0
    ),
    blink_rcr_median: finiteOrDefault(source.blink_rcr_median ?? source.blinkRcrMedian, 0.95),
    blink_rcr_mad: finiteOrDefault(source.blink_rcr_mad ?? source.blinkRcrMad, 0.18),
    blink_freq_median_per_min: finiteOrDefault(
      source.blink_freq_median_per_min ?? source.blinkFreqMedianPerMin,
      16.0
    ),
    blink_freq_mad_per_min: finiteOrDefault(
      source.blink_freq_mad_per_min ?? source.blinkFreqMadPerMin,
      4.5
    ),
    baseline_perclos_pct: finiteOrDefault(
      source.baseline_perclos_pct ?? source.baselinePerclosPct,
      3.5
    ),
    blink_amplitude_median: finiteOrDefault(
      source.blink_amplitude_median ?? source.blinkAmplitudeMedian,
      0.14
    ),
    temporal_blinks_sampled: Math.max(
      0,
      Math.round(finiteOrDefault(source.temporal_blinks_sampled ?? source.temporalBlinksSampled, 0))
    ),
    temporal_window_sec: finiteOrDefault(
      source.temporal_window_sec ?? source.temporalWindowSec,
      120.0
    ),
    sensitivity_level: String(source.sensitivity_level ?? source.sensitivityLevel ?? "STANDARD"),
    neutral_pitch_deg: finiteOrDefault(source.neutral_pitch_deg ?? source.neutralPitchDeg, 0.0),
    neutral_yaw_deg: finiteOrDefault(source.neutral_yaw_deg ?? source.neutralYawDeg, 0.0),
    neutral_rot_mat: neutralRotMat,
    calibration_is_mirrored: calibrationIsMirrored,
    calibration_reliability: finiteOrDefault(
      source.calibration_reliability ?? source.calibrationReliability,
      isCalibrated ? 0.9 : 0.5
    ),
    zone_baselines: extractSpatialZoneBaselines(rawZones, centerEarMedian, centerEarMad),
  };
}
