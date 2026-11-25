import React from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { Sidebar } from "@/components/sidebar";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { CommandPalette } from "@/components/command-palette";
import { NavbarSystemHealthBadge } from "@/components/dashboard/navbar-health-badge";
import { DashboardHeaderActions } from "@/components/dashboard/dashboard-header-actions";
import { WorkspaceBreadcrumb } from "@/components/dashboard/workspace-breadcrumb";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerSession();

  if (!session || !session.user) {
    redirect("/auth/sign-in?redirectTo=/dashboard");
  }

  // Enforce Onboarding & fetch user-scoped critical drowsiness alerts count in parallel
  const [profile, criticalAlertsCount] = await Promise.all([
    getOrCreateUserProfile(session.user.id),
    prisma.drowsinessAlert.count({
      where: {
        severity: "CRITICAL",
        session: { userId: session.user.id },
      },
    }),
  ]);

  if (!profile.onboardingCompleted) {
    redirect("/onboarding");
  }

  const user = {
    id: session.user.id,
    email: session.user.email,
    name: session.user.name,
    image: session.user.image,
    role: session.user.role || "USER",
  };

  return (
    <TooltipProvider>
      <SidebarProvider defaultOpen={true}>
        <div className="flex h-svh w-full bg-background text-foreground overflow-hidden">
          <Sidebar user={user} />

          <SidebarInset className="flex flex-col flex-1 bg-background min-w-0 h-full overflow-hidden">
            {/* Top Navbar Header aligned seamlessly at h-14 with Sidebar Header */}
            <header className="shrink-0 z-30 flex min-h-14 pt-safe items-center justify-between gap-2 border-b border-border bg-background/95 pl-[max(0.5rem,env(safe-area-inset-left,0px))] pr-[max(0.5rem,env(safe-area-inset-right,0px))] sm:pl-[max(1rem,env(safe-area-inset-left,0px))] sm:pr-[max(1rem,env(safe-area-inset-right,0px))] md:pl-[max(1.5rem,env(safe-area-inset-left,0px))] md:pr-[max(1.5rem,env(safe-area-inset-right,0px))] backdrop-blur-md">
              {/* Left Section: Sidebar Toggle + Breadcrumb */}
              <div className="flex items-center gap-1.5 sm:gap-3 min-w-0">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <SidebarTrigger className="text-muted-foreground hover:text-foreground cursor-pointer size-11 md:size-9 shrink-0" aria-label="Toggle Navigation Sidebar" />
                  </TooltipTrigger>
                  <TooltipContent side="bottom">Toggle Sidebar</TooltipContent>
                </Tooltip>
                <WorkspaceBreadcrumb />
              </div>

              {/* Center Section: Health Badge */}
              <div className="flex items-center gap-2 shrink-0">
                <NavbarSystemHealthBadge />
              </div>

              {/* Right Section: Smart Actions (Adaptive Live Monitor + Alerts + Theme) */}
              <DashboardHeaderActions criticalAlertsCount={criticalAlertsCount} />
            </header>

            {/* Main Page Content */}
            <main id="main-content" tabIndex={-1} className="flex-1 overflow-y-auto min-h-0 pb-safe px-safe bg-background outline-none">
              {children}
            </main>

            {/* Global Cmd+K Command Palette */}
            <CommandPalette />
          </SidebarInset>
        </div>
      </SidebarProvider>
    </TooltipProvider>
  );
}
