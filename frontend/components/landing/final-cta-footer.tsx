"use client";

import React from "react";
import Link from "next/link";
import { motion, useReducedMotion } from "motion/react";
import { ArrowRight, BookOpen, ShieldCheck } from "lucide-react";
import { Logo } from "@/components/logo";
import { scrollToSection } from "@/lib/smooth-scroll";

interface FinalCtaFooterProps {
  isAuthenticated: boolean;
}

export function FinalCtaFooter({ isAuthenticated }: FinalCtaFooterProps) {
  const prefersReducedMotion = useReducedMotion();

  return (
    <>
      {/* Section G: Final CTA */}
      <section className="py-20 md:py-28 border-t border-border">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <motion.div
            initial={prefersReducedMotion ? false : { opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.35 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="relative overflow-hidden rounded-3xl border border-border bg-card px-6 py-14 sm:px-12 sm:py-20 text-center shadow-xl"
          >
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-primary/60 to-transparent"
            />
            <div
              aria-hidden="true"
              className="pointer-events-none absolute -bottom-24 left-1/2 -translate-x-1/2 h-64 w-96 rounded-full bg-[radial-gradient(circle,color-mix(in_srgb,var(--primary)_14%,transparent),transparent_70%)] blur-2xl"
            />

            <div className="relative z-10 max-w-2xl mx-auto space-y-6">
              <span className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-primary-accent">
                Ready for the Road
              </span>

              <h2 className="font-display text-3xl sm:text-5xl font-bold tracking-[-0.03em] text-foreground leading-[1.08] text-balance">
                Every journey deserves an alert driver.
              </h2>

              <p className="text-sm sm:text-base text-muted-foreground leading-relaxed max-w-xl mx-auto">
                Calibrate your 4-zone baseline in seconds and experience real-time, personalized drowsiness protection.
              </p>

              <div className="flex flex-col sm:flex-row items-stretch sm:items-center justify-center gap-3.5 pt-2">
                <Link
                  href={isAuthenticated ? "/dashboard/live-monitor" : "/auth/sign-in"}
                  className="inline-flex items-center justify-center gap-2.5 px-7 py-3.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-sm transition-colors shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-card"
                >
                  <ShieldCheck className="size-4" />
                  <span>Start Monitoring</span>
                  <ArrowRight className="size-4" />
                </Link>

                <Link
                  href="/help"
                  className="inline-flex items-center justify-center gap-2 px-7 py-3.5 rounded-xl border border-border bg-background/80 hover:bg-accent text-foreground font-medium text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  <BookOpen className="size-4 text-primary-accent" />
                  <span>Explore the Technology</span>
                </Link>
              </div>
            </div>
          </motion.div>
        </div>
      </section>

      {/* Clean Footer */}
      <footer className="border-t border-border bg-background py-10 text-xs text-muted-foreground">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
          <div className="flex flex-col md:flex-row md:items-center justify-between gap-6">
            <div className="flex items-center gap-3">
              <Logo className="size-8 rounded-lg border border-border" />
              <div>
                <span className="font-display font-semibold text-sm text-foreground block">
                  DriveGuard
                </span>
                <span className="text-[11px] text-muted-foreground">
                  Personalized Adaptive AI Driver Drowsiness Detection
                </span>
              </div>
            </div>

            <nav aria-label="Footer Navigation" className="flex flex-wrap items-center gap-6 text-xs font-medium text-muted-foreground">
              <a
                href="#vision"
                onClick={(e) => scrollToSection(e, "#vision")}
                className="hover:text-foreground transition-colors"
              >
                About
              </a>
              <a
                href="#calibration"
                onClick={(e) => scrollToSection(e, "#calibration")}
                className="hover:text-foreground transition-colors"
              >
                Technology
              </a>
              <Link href="/help" className="hover:text-foreground transition-colors">
                Documentation
              </Link>
              <a
                href="https://github.com/michaelumoize-lab/DriveGuard"
                target="_blank"
                rel="noopener noreferrer"
                className="hover:text-foreground transition-colors"
              >
                GitHub
              </a>
            </nav>
          </div>

          <div className="pt-6 border-t border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 text-[11px] text-muted-foreground/80">
            <span>
              DriveGuard &copy; {new Date().getFullYear()} · Ephemeral in-memory video processing. Academic research prototype, not an ISO 26262 certified ADAS device.
            </span>

            <div className="flex flex-wrap items-center gap-4">
              <Link href="/dashboard/benchmarks" className="hover:text-foreground transition-colors">
                Ablation Benchmarks
              </Link>
              <Link href="/dashboard/hardware" className="hover:text-foreground transition-colors">
                Hardware Co-Processor
              </Link>
              <Link href="/dashboard/defense-guide" className="hover:text-foreground transition-colors">
                Defense Guide
              </Link>
            </div>
          </div>
        </div>
      </footer>
    </>
  );
}
