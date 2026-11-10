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

  const [profile, driverProfile] = await Promise.all([
    getOrCreateUserProfile(session.user.id),
    prisma.driverProfile.findUnique({ where: { userId: session.user.id } }),
  ]);

  if (profile.onboardingCompleted) {
    redirect("/dashboard");
  }

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
      initialDriverProfile={
        driverProfile
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
              zoneBaselines:
                (driverProfile.zoneBaselines as Record<string, unknown> | null) ?? null,
            }
          : null
      }
    />
  );
}
