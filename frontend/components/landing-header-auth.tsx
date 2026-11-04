"use client";

import Link from "next/link";
import { ArrowRight, LayoutDashboard, LogIn } from "lucide-react";
import { UserButton } from "@/components/auth/user/user-button";
import { ThemeToggle } from "@/components/theme-toggle";
import { useSession } from "@/lib/auth-client";

import type { User } from "better-auth";

export interface LandingHeaderUser {
  id: string;
  name?: string | null;
  email?: string | null;
  image?: string | null;
  username?: string | null;
  displayUsername?: string | null;
}

interface LandingHeaderAuthProps {
  user?: LandingHeaderUser | null;
}

export function LandingHeaderAuth({ user }: LandingHeaderAuthProps) {
  const { data: clientSession, isPending, error } = useSession();

  // Prefer server-passed user on initial mount/hydration to eliminate UI flicker.
  // When client session finishes loading, use it for live updates (e.g. sign out).
  const effectiveUser = isPending || error
    ? user
    : (clientSession?.user ?? null);

  const isAuthenticated = !!effectiveUser;

  return (
    <div className="flex items-center gap-2 sm:gap-2.5">
      <ThemeToggle />

      {isAuthenticated ? (
        <>
          <Link
            href="/dashboard"
            className="px-3.5 py-2 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-xs rounded-xl transition-all flex items-center gap-1.5 shadow-xs outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            <LayoutDashboard className="w-3.5 h-3.5" />
            <span className="hidden xs:inline">Command Center</span>
            <span className="xs:hidden">Dashboard</span>
            <ArrowRight className="w-3.5 h-3.5" />
          </Link>
          <UserButton size="icon" user={effectiveUser as User} />
        </>
      ) : (
        <>
          <Link
            href="/auth/sign-in"
            className="px-3.5 py-2 text-xs font-semibold text-foreground hover:bg-muted border border-border rounded-xl transition-colors flex items-center gap-1.5 outline-none focus-visible:ring-2 focus-visible:ring-primary"
          >
            <LogIn className="w-3.5 h-3.5" />
            <span>Sign In</span>
          </Link>
          <UserButton size="icon" user={null} />
        </>
      )}
    </div>
  );
}
