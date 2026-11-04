import React from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { Logo } from "@/components/logo";
import { ThemeToggle } from "@/components/theme-toggle";

export default async function OnboardingLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerSession();

  if (!session || !session.user) {
    redirect("/auth/sign-in?redirectTo=/onboarding");
  }

  const profile = await getOrCreateUserProfile(session.user.id);
  // If the user has already completed onboarding, redirect straight to dashboard
  if (profile.onboardingCompleted) {
    redirect("/dashboard");
  }

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col">
      {/* Onboarding Top Navigation */}
      <header className="border-b border-border/60 bg-background/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-4xl mx-auto px-4 sm:px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Logo className="size-9 rounded-xl" />
            <div>
              <span className="font-bold tracking-tight text-base block text-foreground">IntelliGuard</span>
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider block">
                Driver Onboarding
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-xs text-muted-foreground hidden sm:inline-block">
              Signed in as <strong className="text-foreground">{session.user.email}</strong>
            </span>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Form Content */}
      <main className="flex-1 flex items-center justify-center p-4 sm:p-6 md:p-8">
        <div className="w-full max-w-3xl">
          {children}
        </div>
      </main>
    </div>
  );
}
