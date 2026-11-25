"use client";

import React, { useEffect, useState } from "react";
import Link from "next/link";
import { AnimatePresence, motion, useMotionValueEvent, useReducedMotion, useScroll } from "motion/react";
import { ArrowUpRight, Menu, X } from "lucide-react";
import { Logo } from "@/components/logo";
import {
  LandingHeaderAuth,
  type LandingHeaderUser,
} from "@/components/landing/landing-header-auth";
import { scrollToSection } from "@/lib/smooth-scroll";
import { cn } from "@/lib/utils";

interface LandingNavbarProps {
  user: LandingHeaderUser | null;
}

const SECTION_LINKS = [
  { href: "#vision", id: "vision", label: "Vision" },
  { href: "#calibration", id: "calibration", label: "Calibration" },
  { href: "#alerts", id: "alerts", label: "Alerts" },
  { href: "#dashboard-preview", id: "dashboard-preview", label: "Dashboard" },
] as const;

export function LandingNavbar({ user }: LandingNavbarProps) {
  const { scrollY } = useScroll();
  const prefersReducedMotion = useReducedMotion();
  const [isScrolled, setIsScrolled] = useState(false);
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState<string>("vision");

  useMotionValueEvent(scrollY, "change", (latest) => {
    const next = latest > 24;
    if (next !== isScrolled) {
      setIsScrolled(next);
    }
  });

  useEffect(() => {
    const sections = SECTION_LINKS.map((item) => document.getElementById(item.id)).filter(
      (el): el is HTMLElement => el !== null
    );
    if (sections.length === 0) return;

    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            setActiveSection(entry.target.id);
          }
        }
      },
      { rootMargin: "-35% 0px -50% 0px", threshold: 0.05 }
    );

    sections.forEach((sec) => observer.observe(sec));
    return () => observer.disconnect();
  }, []);

  const handleAnchorClick = (
    e: React.MouseEvent<HTMLAnchorElement>,
    href: string,
    id: string,
    closeMobile = false
  ) => {
    setActiveSection(id);
    scrollToSection(
      e,
      href,
      closeMobile ? () => setMobileMenuOpen(false) : undefined
    );
  };

  return (
    <header
      className={cn(
        "fixed inset-x-0 top-0 z-50 transition-[padding,background-color,border-color,backdrop-filter] duration-300",
        isScrolled
          ? "pt-safe bg-background/85 backdrop-blur-xl border-b border-border/80 shadow-xs"
          : "pt-[max(0.625rem,env(safe-area-inset-top,0px))] sm:pt-[max(0.875rem,env(safe-area-inset-top,0px))] bg-transparent border-b border-transparent"
      )}
    >
      <div
        className={cn(
          "max-w-7xl mx-auto pl-[max(1rem,env(safe-area-inset-left,0px))] pr-[max(1rem,env(safe-area-inset-right,0px))] sm:pl-[max(1.5rem,env(safe-area-inset-left,0px))] sm:pr-[max(1.5rem,env(safe-area-inset-right,0px))] lg:pl-[max(2rem,env(safe-area-inset-left,0px))] lg:pr-[max(2rem,env(safe-area-inset-right,0px))] flex items-center justify-between gap-2 sm:gap-4 transition-[height] duration-300",
          isScrolled ? "h-16" : "h-16"
        )}
      >
        {/* Brand Identity */}
        <Link
          href="/"
          className="group flex items-center gap-3 rounded-xl py-1 pr-2 outline-none focus-visible:ring-2 focus-visible:ring-ring min-w-0"
        >
          <Logo className="size-9 rounded-xl border border-border/80 transition-transform duration-200 group-hover:scale-[1.03] shrink-0" />
          <div className="leading-tight min-w-0">
            <span className="font-display font-bold tracking-tight text-base block text-foreground whitespace-nowrap">
              DriveGuard
            </span>
            <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-[0.14em] hidden xs:block whitespace-nowrap">
              Driver Monitoring AI
            </span>
          </div>
        </Link>

        {/* Centered Architectural Pill Navigation Rail */}
        <nav
          aria-label="Primary Navigation"
          className={cn(
            "hidden md:flex items-center gap-1 rounded-full p-1 transition-[background-color,border-color,box-shadow] duration-300",
            isScrolled
              ? "bg-muted/70 border border-border/80"
              : "bg-card/80 border border-border/80 backdrop-blur-xl shadow-sm"
          )}
        >
          {SECTION_LINKS.map((item) => {
            const isActive = activeSection === item.id;
            return (
              <a
                key={item.href}
                href={item.href}
                onClick={(e) => handleAnchorClick(e, item.href, item.id)}
                className={cn(
                  "relative px-3.5 py-1.5 rounded-full text-xs font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                  isActive
                    ? "text-foreground"
                    : "text-muted-foreground hover:text-foreground"
                )}
              >
                {isActive && (
                  <motion.span
                    layoutId="landing-nav-pill"
                    transition={
                      prefersReducedMotion
                        ? { duration: 0 }
                        : { type: "spring", stiffness: 380, damping: 32 }
                    }
                    className="absolute inset-0 rounded-full bg-background border border-border/90 shadow-2xs"
                  />
                )}
                <span className="relative z-10 inline-flex items-center gap-1.5">
                  {isActive && (
                    <span
                      aria-hidden="true"
                      className="size-1.5 rounded-full bg-primary"
                    />
                  )}
                  <span>{item.label}</span>
                </span>
              </a>
            );
          })}

          <span aria-hidden="true" className="mx-1 h-3.5 w-px bg-border" />

          <Link
            href="/help"
            className="inline-flex items-center gap-1 px-3.5 py-1.5 rounded-full text-xs font-medium text-muted-foreground hover:text-foreground transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            <span>Docs</span>
            <ArrowUpRight className="size-3 opacity-70" />
          </Link>
        </nav>

        {/* Right Actions + Mobile Trigger */}
        <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
          <LandingHeaderAuth user={user} />
          <button
            type="button"
            onClick={() => setMobileMenuOpen((prev) => !prev)}
            aria-expanded={mobileMenuOpen}
            aria-label={mobileMenuOpen ? "Close navigation menu" : "Open navigation menu"}
            className="md:hidden inline-flex items-center justify-center size-11 rounded-xl border border-border bg-card/80 text-foreground hover:bg-accent transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {mobileMenuOpen ? <X className="size-4" aria-hidden="true" /> : <Menu className="size-4" aria-hidden="true" />}
          </button>
        </div>
      </div>

      {/* Mobile Navigation Drawer */}
      <AnimatePresence>
        {mobileMenuOpen && (
          <motion.nav
            aria-label="Mobile Navigation"
            initial={prefersReducedMotion ? false : { opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={prefersReducedMotion ? undefined : { opacity: 0, y: -8 }}
            transition={{ duration: 0.2, ease: [0.22, 1, 0.36, 1] }}
            className="md:hidden mt-1 border-y border-border bg-card/95 backdrop-blur-xl pl-[max(1rem,env(safe-area-inset-left,0px))] pr-[max(1rem,env(safe-area-inset-right,0px))] py-3 pb-[max(0.75rem,env(safe-area-inset-bottom,0px))] flex flex-col gap-1 shadow-lg overscroll-contain"
          >
            {SECTION_LINKS.map((item) => {
              const isActive = activeSection === item.id;
              return (
                <a
                  key={item.href}
                  href={item.href}
                  onClick={(e) => handleAnchorClick(e, item.href, item.id, true)}
                  className={cn(
                    "flex items-center justify-between rounded-xl px-3.5 min-h-11 py-2.5 text-sm font-medium transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    isActive
                      ? "bg-muted text-foreground"
                      : "text-muted-foreground hover:bg-accent hover:text-foreground"
                  )}
                >
                  <span>{item.label}</span>
                  {isActive && <span className="size-1.5 rounded-full bg-primary" aria-hidden="true" />}
                </a>
              );
            })}
            <Link
              href="/help"
              onClick={() => setMobileMenuOpen(false)}
              className="flex items-center justify-between rounded-xl px-3.5 min-h-11 py-2.5 text-sm font-medium text-muted-foreground hover:bg-accent hover:text-foreground transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
            >
              <span>Documentation</span>
              <ArrowUpRight className="size-4 opacity-70" aria-hidden="true" />
            </Link>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
