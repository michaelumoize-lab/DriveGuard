import React from "react";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { GuidedCalibrationWizard } from "@/components/calibration/guided-calibration-wizard";
import { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";

export const metadata = {
  title: "4-Zone Calibration Wizard | DriveGuard",
  description:
    "Guided 4-Zone Personalised Baseline Calibration with real-time target-acquisition spatial gating.",
};

export const dynamic = "force-dynamic";

export default async function CalibrationPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const driverProfile = userId
    ? await prisma.driverProfile.findUnique({ where: { userId } })
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
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          4-Zone Personal Baseline Calibration
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground">
          Calibrate your personal Eye Aspect Ratio (EAR) baselines across CENTER, LEFT MIRROR, RIGHT MIRROR, and DASHBOARD.
        </p>
      </div>

      <GuidedCalibrationWizard
        driverId={userId || "driver_default"}
        initialDriverProfile={initialDriverProfile}
      />
    </div>
  );
}
