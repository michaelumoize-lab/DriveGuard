import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile, saveOnboardingStep } from "@/lib/user-profile";

export async function GET() {
  const session = await getServerSession();
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  const profile = await getOrCreateUserProfile(session.user.id);
  return NextResponse.json({
    user: {
      id: session.user.id,
      name: session.user.name,
      email: session.user.email,
    },
    profile,
  });
}

export async function POST(req: NextRequest) {
  const session = await getServerSession();
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const body = await req.json();
    const { step, completedStepKey, isFinalStep, ...stepData } = body;

    if (typeof step !== "number" || step < 1 || step > 4) {
      return NextResponse.json({ error: "Invalid step number" }, { status: 400 });
    }

    const updatedProfile = await saveOnboardingStep(session.user.id, {
      step,
      completedStepKey,
      isFinalStep: Boolean(isFinalStep),
      ...stepData,
    });

    return NextResponse.json({
      success: true,
      profile: updatedProfile,
    });
  } catch (error) {
    console.error("Failed to save onboarding progress:", error);
    return NextResponse.json(
      { error: "Internal server error saving onboarding progress" },
      { status: 500 }
    );
  }
}
