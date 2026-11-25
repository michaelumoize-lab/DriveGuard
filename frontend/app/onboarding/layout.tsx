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
    <div className="min-h-svh bg-background text-foreground flex flex-col">
      {/* Onboarding Top Navigation */}
      <header className="border-b border-border/60 bg-background/80 backdrop-blur-md sticky top-0 z-40 pt-safe">
        <div className="max-w-4xl mx-auto pl-[max(1rem,env(safe-area-inset-left,0px))] pr-[max(1rem,env(safe-area-inset-right,0px))] sm:pl-[max(1.5rem,env(safe-area-inset-left,0px))] sm:pr-[max(1.5rem,env(safe-area-inset-right,0px))] min-h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3 min-w-0">
            <Logo className="size-9 rounded-xl shrink-0" />
            <div className="min-w-0">
              <span className="font-bold tracking-tight text-base block text-foreground truncate">DriveGuard</span>
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider block truncate">
                Driver Onboarding
              </span>
            </div>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <span className="text-xs text-muted-foreground hidden sm:inline-block">
              Signed in as <strong className="text-foreground">{session.user.email}</strong>
            </span>
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main Form Content */}
      <main id="main-content" tabIndex={-1} className="flex-1 flex items-center justify-center p-4 sm:p-6 md:p-8 pb-[max(1rem,env(safe-area-inset-bottom,0px))] px-safe outline-none">
        <div className="w-full max-w-5xl">
          {children}
        </div>
      </main>
    </div>
  );
}
