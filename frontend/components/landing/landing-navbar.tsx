"use client";

import React, { useState } from "react";
import Link from "next/link";
import { useMotionValueEvent, useScroll } from "motion/react";
import { Menu, X } from "lucide-react";
import { Logo } from "@/components/logo";
import {
  LandingHeaderAuth,
  type LandingHeaderUser,
} from "@/components/landing-header-auth";
import { cn } from "@/lib/utils";

interface LandingNavbarProps {
  user: LandingHeaderUser | null;
}

const NAV_LINKS = [
  { href: "#vision", label: "Vision" },
  { href: "#calibration", label: "Calibration" },
  { href: "#alerts", label: "Alerts" },
  { href: "#dashboard-preview", label: "Dashboard" },
  { href: "/help", label: "Documentation", isRoute: true },
] as const;

export function LandingNavbar({ user }: LandingNavbarProps) {
  const { scrollY } = useScroll();
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  useMotionValueEvent(scrollY, "change", (latest) => {
    const next = latest > 24;
    if (next !== isScrolled) {
      setIsScrolled(next);
    }
  });

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-[background-color,border-color,backdrop-filter] duration-300",
        isScrolled || mobileMenuOpen
          ? "bg-[#101211]/90 backdrop-blur-md border-b border-[#F5F3EC]/10"
          : "bg-transparent border-b border-transparent"
      )}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
        <Link
          href="/"
          className="flex items-center gap-3 rounded-lg outline-none focus-visible:ring-2 focus-visible:ring-[#E58B43]"
        >
          <Logo variant="black" className="size-9 rounded-xl border border-[#F5F3EC]/10" />
          <div>
            <span className="font-display font-bold tracking-tight text-base block text-[#F5F3EC]">
              DriveGuard
            </span>
            <span className="text-[10px] font-mono text-[#9EA39B] uppercase tracking-[0.14em] block">
              Driver Monitoring AI
            </span>
          </div>
        </Link>

        <nav
          aria-label="Primary Navigation"
          className="hidden md:flex items-center gap-7 text-xs font-medium text-[#9EA39B]"
        >
          {NAV_LINKS.map((item) =>
            "isRoute" in item && item.isRoute ? (
              <Link
                key={item.href}
                href={item.href}
                className="hover:text-[#F5F3EC] transition-colors py-1 outline-none focus-visible:text-[#F5F3EC] focus-visible:underline"
              >
                {item.label}
              </Link>
            ) : (
              <a
                key={item.href}
                href={item.href}
                className="hover:text-[#F5F3EC] transition-colors py-1 outline-none focus-visible:text-[#F5F3EC] focus-visible:underline"
              >
                {item.label}
              </a>
            )
          )}
        </nav>

        <div className="flex items-center gap-2">
          <LandingHeaderAuth user={user} variant="landing" />
          <button
            type="button"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-expanded={mobileMenuOpen}
            aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
            className="md:hidden inline-flex items-center justify-center size-9 rounded-xl border border-[#F5F3EC]/15 text-[#F5F3EC] hover:bg-[#191D1A] transition-colors outline-none focus-visible:ring-2 focus-visible:ring-[#E58B43]"
          >
            {mobileMenuOpen ? <X className="size-4" /> : <Menu className="size-4" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      {mobileMenuOpen && (
        <nav
          aria-label="Mobile Navigation"
          className="md:hidden border-t border-[#F5F3EC]/10 bg-[#101211]/95 backdrop-blur-md px-4 py-4 space-y-1"
        >
          {NAV_LINKS.map((item) =>
            "isRoute" in item && item.isRoute ? (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="block rounded-lg px-3 py-2.5 text-sm font-medium text-[#F5F3EC]/85 hover:bg-[#191D1A] hover:text-[#F5F3EC] transition-colors"
              >
                {item.label}
              </Link>
            ) : (
              <a
                key={item.href}
                href={item.href}
                onClick={() => setMobileMenuOpen(false)}
                className="block rounded-lg px-3 py-2.5 text-sm font-medium text-[#F5F3EC]/85 hover:bg-[#191D1A] hover:text-[#F5F3EC] transition-colors"
              >
                {item.label}
              </a>
            )
          )}
        </nav>
      )}
    </header>
  );
}
