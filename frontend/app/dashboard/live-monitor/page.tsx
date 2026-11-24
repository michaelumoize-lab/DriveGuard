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

  const [userProfile, driverProfiles] = await Promise.all([
    userId ? getOrCreateUserProfile(userId) : null,
    userId
      ? prisma.driverProfile.findMany({
          where: { userId },
        })
      : [],
  ]);

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

  const browserProfile = mapProfile(
    driverProfiles.find((p) => p.engineType === "BROWSER")
  );
  const pythonProfile = mapProfile(
    driverProfiles.find((p) => p.engineType === "PYTHON")
  );
  const initialDriverProfile =
    targetEngineType === "PYTHON" ? pythonProfile : browserProfile;

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
        initialDriverProfile={browserProfile ?? initialDriverProfile}
        initialPythonProfile={pythonProfile}
        initialPreferences={initialPreferences}
        initialConsoleMode={initialConsoleMode}
        initialInspectorTab={initialInspectorTab}
        initialRuntimeMode={initialRuntimeMode}
        initialOrientation={initialOrientation}
      />
    </div>
  );
}
