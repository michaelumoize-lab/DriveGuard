import React from "react";
import Link from "next/link";
import { redirect } from "next/navigation";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { AdminSidebar } from "@/components/admin-sidebar";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { Tooltip, TooltipTrigger, TooltipContent, TooltipProvider } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { CommandPalette } from "@/components/command-palette";
import { NavbarSystemHealthBadge } from "@/components/dashboard/navbar-health-badge";
import { ShieldAlert, Bell, Activity } from "lucide-react";

export default async function DashboardLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerSession();

  if (!session || !session.user) {
    redirect("/auth/sign-in?redirectTo=/dashboard");
  }

  // Enforce Onboarding: if profile is not completed, redirect to /onboarding
  const profile = await getOrCreateUserProfile(session.user.id);
  if (!profile.onboardingCompleted) {
    redirect("/onboarding");
  }

  // Fetch recent critical drowsiness alerts count for notification bell
  const criticalAlertsCount = await prisma.drowsinessAlert.count({
    where: { severity: "CRITICAL" },
  });

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
        <div className="flex min-h-screen w-full bg-background text-foreground">
          <AdminSidebar user={user} />

          <SidebarInset className="flex flex-col flex-1 bg-background min-w-0">
            {/* Top Navbar Header aligned seamlessly at h-14 with Sidebar Header */}
            <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-background/80 px-2 sm:px-4 md:px-6 backdrop-blur-md">
              {/* Left Section: Sidebar Toggle + Breadcrumb */}
              <div className="flex items-center gap-1.5 sm:gap-3 min-w-0">
                <Tooltip>
                  <TooltipTrigger asChild>
                    <SidebarTrigger className="text-muted-foreground hover:text-foreground cursor-pointer h-9 w-9 shrink-0" aria-label="Toggle Navigation Sidebar" />
                  </TooltipTrigger>
                  <TooltipContent side="bottom">Toggle Sidebar</TooltipContent>
                </Tooltip>
                <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground truncate">
                  <span className="text-muted-foreground/70">IntelliGuard</span>
                  <span className="text-border">/</span>
                  <span className="text-foreground font-medium">Admin System</span>
                </div>
              </div>

              {/* Center Section: Health Badge */}
              <div className="flex items-center gap-2 shrink-0">
                <NavbarSystemHealthBadge />
              </div>

              {/* Right Section: Quick Live Scan Button + Alerts Bell + Theme Toggle */}
              {/* Right Section: System Health + Quick Actions */}
              <div className="flex items-center gap-1.5 sm:gap-2.5 shrink-0">
                {/* Live Driver Monitor Quick Button */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Link
                      href="/dashboard/live-monitor"
                      aria-label="Launch Live Driver Monitoring"
                      className="px-2.5 sm:px-3 py-1.5 bg-primary/10 hover:bg-primary/20 border border-primary/30 text-primary text-xs font-semibold rounded-xl transition-all flex items-center gap-1.5 shadow-xs shrink-0"
                    >
                      <Activity className="w-3.5 h-3.5" />
                      <span className="hidden xs:inline">Live Monitor</span>
                    </Link>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">Launch Live Driver Monitor</TooltipContent>
                </Tooltip>

                {/* Safety Alerts Notification Bell */}
                <Tooltip>
                  <TooltipTrigger asChild>
                    <Button
                      asChild
                      variant="ghost"
                      size="icon"
                      className="relative h-9 w-9 text-muted-foreground hover:text-foreground shrink-0"
                    >
                      <Link href="/dashboard/alerts" aria-label={`Safety Alerts (${criticalAlertsCount} critical)`}>
                        <Bell className="w-4 h-4" />
                        {criticalAlertsCount > 0 && (
                          <span className="absolute -top-1 -right-1 flex h-4 min-w-4 px-1 items-center justify-center rounded-full bg-rose-600 dark:bg-rose-500 text-[10px] font-bold font-mono text-white ring-2 ring-background leading-none shadow-xs pointer-events-none">
                            {criticalAlertsCount}
                          </span>
                        )}
                      </Link>
                    </Button>
                  </TooltipTrigger>
                  <TooltipContent side="bottom">
                    {criticalAlertsCount > 0
                      ? `Critical Fatigue Alerts (${criticalAlertsCount})`
                      : "Safety Alerts (All clear)"}
                  </TooltipContent>
                </Tooltip>

                {/* Theme Mode Toggle */}
                <ThemeToggle />
              </div>
            </header>

            {/* Main Page Content */}
            <main className="flex-1 overflow-y-auto">
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
