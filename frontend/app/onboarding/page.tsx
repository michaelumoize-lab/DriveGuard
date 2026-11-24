import React from "react";
import { redirect } from "next/navigation";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { OnboardingWizard } from "@/components/onboarding/onboarding-wizard";

export default async function OnboardingPage() {
  const session = await getServerSession();

  if (!session || !session.user) {
    redirect("/auth/sign-in?redirectTo=/onboarding");
  }

  const [profile, driverProfiles] = await Promise.all([
    getOrCreateUserProfile(session.user.id),
    prisma.driverProfile.findMany({
      where: { userId: session.user.id },
    }),
  ]);

  if (profile.onboardingCompleted) {
    redirect("/dashboard");
  }

  const mapProfile = (p: (typeof driverProfiles)[number] | undefined) =>
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
          zoneBaselines:
            (p.zoneBaselines as Record<string, unknown> | null) ?? null,
        }
      : null;

  const browserProfile = mapProfile(
    driverProfiles.find((p) => p.engineType === "BROWSER")
  );
  const pythonProfile = mapProfile(
    driverProfiles.find((p) => p.engineType === "PYTHON")
  );

  return (
    <OnboardingWizard
      user={{
        id: session.user.id,
        name: session.user.name,
        email: session.user.email,
      }}
      initialProfile={{
        userId: profile.userId,
        onboardingStep: profile.onboardingStep,
        completedSteps: profile.completedSteps,
        phoneNumber: profile.phoneNumber,
        licenseNumber: profile.licenseNumber,
        vehicleMake: profile.vehicleMake,
        vehicleModel: profile.vehicleModel,
        vehicleYear: profile.vehicleYear,
        licensePlate: profile.licensePlate,
        emergencyContactName: profile.emergencyContactName,
        emergencyContactPhone: profile.emergencyContactPhone,
        sensitivityLevel: profile.sensitivityLevel,
        audioAlertsEnabled: profile.audioAlertsEnabled,
        mirrorSuppression: profile.mirrorSuppression,
        wearsGlasses: profile.wearsGlasses,
      }}
      initialDriverProfile={browserProfile}
      initialPythonProfile={pythonProfile}
    />
  );
}
