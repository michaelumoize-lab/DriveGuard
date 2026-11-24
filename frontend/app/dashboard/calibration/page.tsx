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

  const driverProfile = userId
    ? await prisma.driverProfile.findUnique({
        where: { userId_engineType: { userId, engineType: "PYTHON" } },
      })
    : null;

  const initialDriverProfile: InitialDriverProfileData | null = driverProfile
    ? {
        isCalibrated: driverProfile.isCalibrated,
        calibrationMode: driverProfile.calibrationMode,
        centerEarMedian: driverProfile.centerEarMedian,
        centerEarMadScaled: driverProfile.centerEarMadScaled,
        blinkDurationMedianMs: driverProfile.blinkDurationMedianMs,
        blinkDurationMadMs: driverProfile.blinkDurationMadMs,
        blinkRcrMedian: driverProfile.blinkRcrMedian,
        blinkRcrMad: driverProfile.blinkRcrMad,
        neutralPitchDeg: driverProfile.neutralPitchDeg,
        neutralYawDeg: driverProfile.neutralYawDeg,
        calibrationReliability: driverProfile.calibrationReliability,
        zoneBaselines: (driverProfile.zoneBaselines as Record<string, unknown> | null) ?? null,
      }
    : null;

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground text-balance">
          Driver Calibration
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-2xl">
          Personalize DriveGuard to your eyes and driving posture across 4 views (Road Ahead,
          Left Mirror, Right Mirror, and Dashboard) in about 15–20 seconds.
        </p>
      </div>

      <GuidedCalibrationWizard
        driverId={userId || "driver_default"}
        initialDriverProfile={initialDriverProfile}
      />
    </div>
  );
}
