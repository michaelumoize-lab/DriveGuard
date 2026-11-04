import { prisma } from "@/lib/prisma";
import type { UserProfile, SensitivityLevel } from "@prisma/client";

export async function getUserProfile(userId: string): Promise<UserProfile | null> {
  try {
    return await prisma.userProfile.findUnique({
      where: { userId },
    });
  } catch (error) {
    console.error("Error fetching user profile:", error);
    return null;
  }
}

export async function getOrCreateUserProfile(userId: string): Promise<UserProfile> {
  const existing = await getUserProfile(userId);
  if (existing) {
    return existing;
  }

  return await prisma.userProfile.create({
    data: {
      userId,
      onboardingCompleted: false,
      onboardingStep: 1,
      completedSteps: [],
    },
  });
}

export interface OnboardingUpdatePayload {
  step: number;
  completedStepKey?: string;
  isFinalStep?: boolean;
  phoneNumber?: string;
  licenseNumber?: string;
  vehicleMake?: string;
  vehicleModel?: string;
  vehicleYear?: number;
  licensePlate?: string;
  emergencyContactName?: string;
  emergencyContactPhone?: string;
  sensitivityLevel?: SensitivityLevel;
  audioAlertsEnabled?: boolean;
  mirrorSuppression?: boolean;
  wearsGlasses?: boolean;
}

export async function saveOnboardingStep(
  userId: string,
  payload: OnboardingUpdatePayload
): Promise<UserProfile> {
  const currentProfile = await getOrCreateUserProfile(userId);

  const updatedCompletedSteps = new Set(currentProfile.completedSteps || []);
  if (payload.completedStepKey) {
    updatedCompletedSteps.add(payload.completedStepKey);
  }

  const nextStep = payload.isFinalStep
    ? 4
    : Math.max(currentProfile.onboardingStep, payload.step + 1);

  return await prisma.userProfile.update({
    where: { userId },
    data: {
      onboardingStep: nextStep,
      completedSteps: Array.from(updatedCompletedSteps),
      onboardingCompleted: payload.isFinalStep ? true : currentProfile.onboardingCompleted,
      ...(payload.phoneNumber !== undefined && { phoneNumber: payload.phoneNumber }),
      ...(payload.licenseNumber !== undefined && { licenseNumber: payload.licenseNumber }),
      ...(payload.vehicleMake !== undefined && { vehicleMake: payload.vehicleMake }),
      ...(payload.vehicleModel !== undefined && { vehicleModel: payload.vehicleModel }),
      ...(payload.vehicleYear !== undefined && { vehicleYear: payload.vehicleYear }),
      ...(payload.licensePlate !== undefined && { licensePlate: payload.licensePlate }),
      ...(payload.emergencyContactName !== undefined && { emergencyContactName: payload.emergencyContactName }),
      ...(payload.emergencyContactPhone !== undefined && { emergencyContactPhone: payload.emergencyContactPhone }),
      ...(payload.sensitivityLevel !== undefined && { sensitivityLevel: payload.sensitivityLevel }),
      ...(payload.audioAlertsEnabled !== undefined && { audioAlertsEnabled: payload.audioAlertsEnabled }),
      ...(payload.mirrorSuppression !== undefined && { mirrorSuppression: payload.mirrorSuppression }),
      ...(payload.wearsGlasses !== undefined && { wearsGlasses: payload.wearsGlasses }),
    },
  });
}
