import React from "react";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { GuidedCalibrationWizard } from "@/components/calibration/guided-calibration-wizard";
import { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";

export const metadata = {
  title: "Driver Calibration | DriveGuard",
  description:
    "2-step guided 4-zone calibration to personalize DriveGuard to your eyes and driving posture.",
};

export const dynamic = "force-dynamic";

export default async function CalibrationPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const driverProfiles = userId
    ? await prisma.driverProfile.findMany({
        where: { userId },
      })
    : [];

  const mapProfile = (
    p: (typeof driverProfiles)[number] | undefined
  ): InitialDriverProfileData | null =>
    p
      ? {
          engineType: p.engineType,
          isCalibrated: p.isCalibrated,
          calibrationMode: p.calibrationMode,
          centerEarMedian: p.centerEarMedian,
          centerEarMadScaled: p.centerEarMadScaled,
          blinkDurationMedianMs: p.blinkDurationMedianMs,
          blinkDurationMadMs: p.blinkDurationMadMs,
          blinkRcrMedian: p.blinkRcrMedian,
          blinkRcrMad: p.blinkRcrMad,
          neutralPitchDeg: p.neutralPitchDeg,
          neutralYawDeg: p.neutralYawDeg,
          neutralRotMat: (p.neutralRotMat as number[][] | null) ?? null,
          calibrationIsMirrored: p.calibrationIsMirrored ?? null,
          calibrationReliability: p.calibrationReliability,
          profileVersion: p.profileVersion,
          profileHash: p.contentHash ?? null,
          zoneBaselines: (p.zoneBaselines as Record<string, unknown> | null) ?? null,
        }
      : null;

  const initialBrowserProfile = mapProfile(
    driverProfiles.find((p) => p.engineType === "BROWSER")
  );
  const initialPythonProfile = mapProfile(
    driverProfiles.find((p) => p.engineType === "PYTHON")
  );

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground text-balance">
          Driver Calibration
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-2xl">
          Personalize DriveGuard to your eyes and driving posture across 4 views (Road Ahead,
          Left Mirror, Right Mirror, and Dashboard) in about 15–20 seconds. Runs locally in
          Browser AI by default with optional Python AI calibration.
        </p>
      </div>

      <GuidedCalibrationWizard
        driverId={userId || "driver_default"}
        initialDriverProfile={initialBrowserProfile}
        initialPythonProfile={initialPythonProfile}
      />
    </div>
  );
}
