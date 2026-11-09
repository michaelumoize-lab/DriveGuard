import { viewPaths } from "@better-auth-ui/core";
import { getServerSession } from "@/lib/get-session";
import { dehydrate, HydrationBoundary } from "@tanstack/react-query";
import { notFound, redirect } from "next/navigation";

import { Settings } from "@/components/auth/settings/settings";
import { DriverPreferencesForm } from "@/components/settings/driver-preferences-form";
import { getQueryClient } from "@/lib/query-client";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

export default async function SettingsPage({
  params,
}: {
  params: Promise<{
    path: string;
  }>;
}) {
  const { path } = await params;

  if (!Object.values(viewPaths.settings).includes(path)) {
    notFound();
  }

  const queryClient = getQueryClient();
  const session = await getServerSession();

  if (!session) {
    redirect(
      `/auth/sign-in?redirectTo=${encodeURIComponent(`/settings/${path}`)}`
    );
  }

  const [userProfile, driverProfile] = await Promise.all([
    getOrCreateUserProfile(session.user.id),
    prisma.driverProfile
      .findUnique({ where: { userId: session.user.id } })
      .catch(() => null),
  ]);

  return (
    <HydrationBoundary state={dehydrate(queryClient)}>
      <div className="w-full max-w-4xl mx-auto p-4 md:p-6 space-y-8">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-foreground">
            Account &amp; Driver Preferences
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground mt-1">
            Manage your personal safety monitoring profile, cabin calibration settings, and account security.
          </p>
        </div>

        <DriverPreferencesForm
          initialProfile={userProfile}
          isCalibrated={Boolean(driverProfile?.isCalibrated)}
        />

        <div className="space-y-3 pt-2 border-t border-border">
          <h2 className="text-base font-semibold text-foreground">
            Authentication &amp; Security Settings
          </h2>
          <Settings path={path} />
        </div>
      </div>
    </HydrationBoundary>
  );
}
