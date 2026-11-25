import type { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";

export interface DriverProfileRow {
  engineType: string;
  isCalibrated: boolean;
  calibrationMode: string;
  centerEarMedian: number;
  centerEarMadScaled: number;
  blinkDurationMedianMs: number;
  blinkDurationMadMs: number;
  blinkRcrMedian: number;
  blinkRcrMad: number;
  neutralPitchDeg: number;
  neutralYawDeg: number;
  neutralRotMat: unknown;
  calibrationIsMirrored: boolean | null;
  calibrationReliability: number;
  profileVersion: number;
  contentHash: string | null;
  zoneBaselines: unknown;
}

/**
 * Maps a persisted Prisma `DriverProfile` row into the serialized client
 * initialization payload shared by the Live Monitor and Calibration pages.
 */
export function mapDriverProfileData(
  profile: DriverProfileRow | null | undefined
): InitialDriverProfileData | null {
  if (!profile) return null;

  return {
    engineType: profile.engineType,
    isCalibrated: profile.isCalibrated,
    calibrationMode: profile.calibrationMode,
    centerEarMedian: profile.centerEarMedian,
    centerEarMadScaled: profile.centerEarMadScaled,
    blinkDurationMedianMs: profile.blinkDurationMedianMs,
    blinkDurationMadMs: profile.blinkDurationMadMs,
    blinkRcrMedian: profile.blinkRcrMedian,
    blinkRcrMad: profile.blinkRcrMad,
    neutralPitchDeg: profile.neutralPitchDeg,
    neutralYawDeg: profile.neutralYawDeg,
    neutralRotMat: (profile.neutralRotMat as number[][] | null) ?? null,
    calibrationIsMirrored: profile.calibrationIsMirrored ?? null,
    calibrationReliability: profile.calibrationReliability,
    profileVersion: profile.profileVersion,
    profileHash: profile.contentHash ?? null,
    zoneBaselines: (profile.zoneBaselines as Record<string, unknown> | null) ?? null,
  };
}
