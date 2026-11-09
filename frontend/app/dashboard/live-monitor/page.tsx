import React from "react";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { UnifiedLiveMonitor, InitialDriverProfileData, InitialUserPreferencesData } from "@/components/monitor/unified-live-monitor";

export const metadata = {
  title: "Live Driver Telemetry Monitor | DriveGuard",
  description:
    "Real-time WebRTC computer vision telemetry streaming via LiveKit to the DriveGuard Edge AI Cascade.",
};

export const dynamic = "force-dynamic";

export default async function LiveMonitorPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const [userProfile, driverProfile] = await Promise.all([
    userId ? getOrCreateUserProfile(userId) : null,
    userId ? prisma.driverProfile.findUnique({ where: { userId } }) : null,
  ]);

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

  const initialPreferences: InitialUserPreferencesData = {
    audioAlertsEnabled: userProfile?.audioAlertsEnabled ?? true,
    sensitivityLevel: userProfile?.sensitivityLevel ?? "MEDIUM",
    mirrorSuppression: userProfile?.mirrorSuppression ?? true,
    wearsGlasses: userProfile?.wearsGlasses ?? false,
  };

  return (
    <div className="p-3 sm:p-5 lg:p-6 max-w-[1600px] mx-auto w-full">
      <UnifiedLiveMonitor
        driverId={userId || "driver_default"}
        initialDriverProfile={initialDriverProfile}
        initialPreferences={initialPreferences}
      />
    </div>
  );
}
