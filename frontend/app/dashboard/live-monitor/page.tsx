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

export type InspectorTab = "overview" | "signals" | "testbench" | "setup";

interface LiveMonitorPageProps {
  searchParams?: Promise<Record<string, string | string[] | undefined>>;
}

function pickParam(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

export default async function LiveMonitorPage({ searchParams }: LiveMonitorPageProps) {
  const resolvedParams = searchParams ? await searchParams : {};
  const modeParam = pickParam(resolvedParams.mode);
  const tabParam = pickParam(resolvedParams.tab);
  const runtimeParam = pickParam(resolvedParams.runtime);
  const viewParam = pickParam(resolvedParams.view);

  const initialConsoleMode =
    modeParam === "inspector" || modeParam === "diagnostics"
      ? "diagnostics"
      : modeParam === "driver" || modeParam === "cockpit"
        ? "cockpit"
        : undefined;

  const initialInspectorTab: InspectorTab | undefined =
    tabParam === "overview" ||
    tabParam === "signals" ||
    tabParam === "testbench" ||
    tabParam === "setup"
      ? tabParam
      : undefined;

  const initialRuntimeMode =
    runtimeParam === "browser" || runtimeParam === "remote" ? runtimeParam : undefined;

  const initialOrientation =
    viewParam === "portrait" || viewParam === "landscape" ? viewParam : undefined;

  const session = await getServerSession();
  const userId = session?.user?.id;

  const targetEngineType = initialRuntimeMode === "remote" ? "PYTHON" : "BROWSER";

  const [userProfile, driverProfile] = await Promise.all([
    userId ? getOrCreateUserProfile(userId) : null,
    userId
      ? prisma.driverProfile.findUnique({
          where: {
            userId_engineType: {
              userId,
              engineType: targetEngineType,
            },
          },
        })
      : null,
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
    sensitivityLevel: userProfile?.sensitivityLevel ?? "STANDARD",
    mirrorSuppression: userProfile?.mirrorSuppression ?? true,
    wearsGlasses: userProfile?.wearsGlasses ?? false,
  };

  return (
    <div className="p-3 sm:p-5 lg:p-6 max-w-[1600px] mx-auto w-full">
      <UnifiedLiveMonitor
        driverId={userId || "driver_default"}
        initialDriverProfile={initialDriverProfile}
        initialPreferences={initialPreferences}
        initialConsoleMode={initialConsoleMode}
        initialInspectorTab={initialInspectorTab}
        initialRuntimeMode={initialRuntimeMode}
        initialOrientation={initialOrientation}
      />
    </div>
  );
}
