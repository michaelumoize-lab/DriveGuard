import React from "react";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { mapDriverProfileData } from "@/lib/api/driver-profile-mapper";
import { UnifiedLiveMonitor, InitialUserPreferencesData } from "@/components/monitor/unified-live-monitor";

export const metadata = {
  title: "Live Driver Telemetry Monitor | DriveGuard",
  description:
    "Real-time geometric-authoritative driver drowsiness monitoring powered by independent Browser WASM and Local/Remote Python MediaPipe + 4-Zone PDBP engines.",
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
    runtimeParam === "browser" ||
    runtimeParam === "python-local" ||
    runtimeParam === "remote"
      ? runtimeParam
      : undefined;

  const initialOrientation =
    viewParam === "portrait" || viewParam === "landscape" ? viewParam : undefined;

  const session = await getServerSession();
  const userId = session?.user?.id;

  const targetEngineType =
    initialRuntimeMode === "remote" || initialRuntimeMode === "python-local"
      ? "PYTHON"
      : "BROWSER";

  const [userProfile, driverProfiles] = await Promise.all([
    userId ? getOrCreateUserProfile(userId) : null,
    userId
      ? prisma.driverProfile.findMany({
          where: { userId },
        })
      : [],
  ]);

  const browserProfile = mapDriverProfileData(
    driverProfiles.find((p) => p.engineType === "BROWSER")
  );
  const pythonProfile = mapDriverProfileData(
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
