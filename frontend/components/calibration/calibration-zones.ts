import type { CalibratedProfilePayload } from "@/types/telemetry";

export type CalibrationZoneKey = "CENTER" | "LEFT_MIRROR" | "RIGHT_MIRROR" | "DASHBOARD";

export interface CalibrationZoneMeta {
  key: CalibrationZoneKey;
  stepNumber: number;
  title: string;
  shortTitle: string;
  subtitle: string;
  activePrompt: string;
  hint: string;
}

export const CALIBRATION_ZONES: readonly CalibrationZoneMeta[] = [
  {
    key: "CENTER",
    stepNumber: 1,
    title: "Road Ahead",
    shortTitle: "Road Ahead",
    subtitle: "Primary straight-ahead baseline",
    activePrompt: "Look straight ahead as if watching the road",
    hint: "Sit naturally, face forward, and keep your head steady.",
  },
  {
    key: "LEFT_MIRROR",
    stepNumber: 2,
    title: "Left Mirror",
    shortTitle: "Left Mirror",
    subtitle: "Driver-side mirror glance",
    activePrompt: "Turn your head slightly toward the left mirror",
    hint: "Hold a natural mirror-check glance until the bar fills.",
  },
  {
    key: "RIGHT_MIRROR",
    stepNumber: 3,
    title: "Right Mirror",
    shortTitle: "Right Mirror",
    subtitle: "Passenger / rearview glance",
    activePrompt: "Turn your head slightly toward the right mirror",
    hint: "Glance comfortably toward the right or center mirror.",
  },
  {
    key: "DASHBOARD",
    stepNumber: 4,
    title: "Dashboard",
    shortTitle: "Dashboard",
    subtitle: "Instrument cluster glance",
    activePrompt: "Glance slightly down toward the dashboard",
    hint: "Lower your gaze toward the speedometer without dropping out of frame.",
  },
] as const;

export function getZoneMeta(zoneKey?: string | null): CalibrationZoneMeta {
  const match = CALIBRATION_ZONES.find((z) => z.key === zoneKey);
  return match ?? CALIBRATION_ZONES[0];
}

export function formatZoneLabel(zoneKey?: string | null): string {
  const match = CALIBRATION_ZONES.find((z) => z.key === zoneKey);
  if (match) return match.title;
  return (zoneKey || "CENTER").replace(/_/g, " ");
}

export interface FriendlyCalibrationIssue {
  title: string;
  description: string;
  rawDetail?: string;
}

export function formatCalibrationRejection(reason: string): FriendlyCalibrationIssue {
  if (reason.includes("REJECTED_POOR_VISIBILITY")) {
    return {
      title: "Not enough clear eye tracking samples",
      description:
        "DriveGuard could not see your eyes clearly enough during the straight-ahead step. Improve room or cabin lighting, uncover your face, and keep your eyes visible to the camera.",
      rawDetail: reason,
    };
  }
  if (reason.includes("REJECTED_HEAD_POSE_INSTABILITY")) {
    return {
      title: "Head movement was too unsteady",
      description:
        "Your head shifted too much while establishing your straight-ahead reference. Sit comfortably, look steadily at one spot ahead, and retry.",
      rawDetail: reason,
    };
  }
  return {
    title: "Calibration could not be completed",
    description:
      "Please check your camera framing and lighting, then run calibration again.",
    rawDetail: reason,
  };
}

export function formatQualityWarning(warning: string): string {
  if (warning.includes("POOR_VISIBILITY")) {
    const zoneMatch = warning.match(/zone '([A-Z_]+)'/);
    const zoneName = zoneMatch ? formatZoneLabel(zoneMatch[1]) : "one zone";
    return `${zoneName} had limited visibility and was estimated safely from your Road Ahead baseline. You can recalibrate ${zoneName} individually below.`;
  }
  return warning;
}

export function validateBackendProfilePayload(
  profile: CalibratedProfilePayload | null | undefined
): profile is CalibratedProfilePayload {
  return Boolean(
    profile &&
      typeof profile.center_ear_median === "number" &&
      Number.isFinite(profile.center_ear_median) &&
      profile.center_ear_median > 0
  );
}
