import React from "react";
import { redirect } from "next/navigation";
import { getServerSession } from "@/lib/get-session";
import { AdminSidebar } from "@/components/admin-sidebar";
import { SidebarProvider, SidebarInset, SidebarTrigger } from "@/components/ui/sidebar";
import { TooltipProvider } from "@/components/ui/tooltip";
import { NavbarSystemHealthBadge } from "@/components/dashboard/navbar-health-badge";
import { ThemeToggle } from "@/components/theme-toggle";
import { CommandPalette } from "@/components/command-palette";

export default async function SettingsLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  const session = await getServerSession();

  if (!session) {
    redirect("/auth/sign-in?redirectTo=/settings/account");
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
        <div className="flex min-h-screen w-full bg-background text-foreground">
          <AdminSidebar user={user} />

          <SidebarInset className="flex flex-col flex-1 bg-background min-w-0">
            {/* Top Navbar Header aligned seamlessly with Sidebar */}
            <header className="sticky top-0 z-30 flex h-14 items-center justify-between border-b border-border bg-background/80 px-2 sm:px-4 md:px-6 backdrop-blur-md">
              <div className="flex items-center gap-1.5 sm:gap-3 min-w-0">
                <SidebarTrigger className="text-muted-foreground hover:text-foreground cursor-pointer h-9 w-9 shrink-0" aria-label="Toggle Navigation Sidebar" />
                <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground truncate">
                  <span className="text-muted-foreground/70">IntelliGuard</span>
                  <span className="text-border">/</span>
                  <span className="text-foreground font-medium">Settings</span>
                </div>
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <NavbarSystemHealthBadge />
              </div>

              <div className="flex items-center gap-2 shrink-0">
                <ThemeToggle />
              </div>
            </header>

            <main className="flex-1 overflow-y-auto">
              {children}
            </main>

            <CommandPalette />
          </SidebarInset>
        </div>
      </SidebarProvider>
    </TooltipProvider>
  );
}
