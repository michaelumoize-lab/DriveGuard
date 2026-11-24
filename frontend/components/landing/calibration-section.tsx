"use client";

import React, { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import {
  ArrowDown,
  ArrowLeft,
  ArrowRight,
  CheckCircle2,
  Compass,
  Pause,
  Play,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface CalibrationZoneStep {
  id: "CENTER" | "LEFT_MIRROR" | "RIGHT_MIRROR" | "DASHBOARD";
  stepNumber: string;
  label: string;
  icon: React.ComponentType<{ className?: string }>;
  medianEar: string;
  zoneThreshold: string;
  yawRange: string;
  detail: string;
}

const CALIBRATION_STEPS: CalibrationZoneStep[] = [
  {
    id: "CENTER",
    stepNumber: "01",
    label: "Center position",
    icon: Compass,
    medianEar: "0.295",
    zoneThreshold: "0.212",
    yawRange: "-10° to +10°",
    detail:
      "Captures your natural resting eyelid aperture and normal blink velocity while looking straight ahead at the roadway.",
  },
  {
    id: "LEFT_MIRROR",
    stepNumber: "02",
    label: "Left mirror",
    icon: ArrowLeft,
    medianEar: "0.256",
    zoneThreshold: "0.184",
    yawRange: "-32° to -12°",
    detail:
      "Learns geometric foreshortening when checking the driver-side mirror so brief glances under 3.0s never trigger false alarms.",
  },
  {
    id: "RIGHT_MIRROR",
    stepNumber: "03",
    label: "Right mirror",
    icon: ArrowRight,
    medianEar: "0.252",
    zoneThreshold: "0.181",
    yawRange: "+12° to +32°",
    detail:
      "Records your passenger-mirror head turn geometry and establishes an independent per-zone closure threshold.",
  },
  {
    id: "DASHBOARD",
    stepNumber: "04",
    label: "Dashboard",
    icon: ArrowDown,
    medianEar: "0.238",
    zoneThreshold: "0.171",
    yawRange: "-14° Pitch",
    detail:
      "Accounts for natural upper-eyelid droop during quick instrument-panel checks without mistaking glances for drowsiness.",
  },
];

export function CalibrationSection() {
  const prefersReducedMotion = useReducedMotion();
  const [activeIndex, setActiveIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    if (prefersReducedMotion || !isPlaying) return;
    const timer = window.setInterval(() => {
      setActiveIndex((prev) => (prev + 1) % CALIBRATION_STEPS.length);
    }, 3200);
    return () => window.clearInterval(timer);
  }, [prefersReducedMotion, isPlaying]);

  const currentStep = CALIBRATION_STEPS[activeIndex];
  const progressPct = ((activeIndex + 1) / CALIBRATION_STEPS.length) * 100;

  return (
    <section
      id="calibration"
      className="py-20 md:py-28 border-t border-border scroll-mt-24"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-12 items-center">
          {/* Left Column: Narrative Differentiation */}
          <motion.div
            initial={prefersReducedMotion ? false : { opacity: 0, y: 18 }}
            whileInView={{ opacity: 1, y: 0 }}
            viewport={{ once: true, amount: 0.25 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-5 space-y-6"
          >
            <span className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-primary-accent">
              <span className="size-1.5 rounded-full bg-primary" />
              Personalized Baseline Calibration
            </span>

            <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-foreground leading-[1.12] text-balance">
              Adapted to your anatomy, not a generic threshold.
            </h2>

            <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
              Every driver has distinct eyelid geometry, natural blink cadence, and seating posture. Static detectors (like fixed EAR &lt; 0.22) fire false alarms the moment you check a side mirror. DriveGuard calibrates a 4-zone personal baseline in 15 seconds.
            </p>

            {/* Baseline Comparison Highlight */}
            <div className="rounded-2xl border border-border bg-card p-5 space-y-3 shadow-xs">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-muted-foreground">INDIVIDUAL BASELINE MODEL</span>
                <span className="text-safe font-semibold">72% OF ZONE MEDIAN</span>
              </div>
              <p className="text-xs text-muted-foreground leading-relaxed">
                Instead of one rigid number, DriveGuard pairs your live 6DoF gaze zone with that zone&apos;s personal EAR median and MAD dispersion—suppressing routine mirror checks (&le; 3.0s) while keeping fatigue sensitivity sharp.
              </p>
            </div>
          </motion.div>

          {/* Right Column: Interactive 4-Step Calibration Sequence */}
          <div className="lg:col-span-7 rounded-3xl border border-border bg-card p-5 sm:p-7 space-y-6 shadow-xl">
            <div className="flex flex-wrap items-center justify-between gap-3 pb-4 border-b border-border">
              <div>
                <span className="text-xs font-mono uppercase tracking-[0.14em] text-muted-foreground block">
                  4-Zone Calibration Sequence
                </span>
                <span className="text-sm font-semibold text-foreground">
                  Establishing Individual Driver Baseline
                </span>
              </div>

              <button
                type="button"
                onClick={() => setIsPlaying((prev) => !prev)}
                className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-background px-3 py-1.5 text-xs font-medium text-foreground hover:border-primary/50 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                {isPlaying && !prefersReducedMotion ? (
                  <>
                    <Pause className="size-3.5 text-primary-accent" />
                    <span>Pause</span>
                  </>
                ) : (
                  <>
                    <Play className="size-3.5 text-safe" />
                    <span>Play</span>
                  </>
                )}
              </button>
            </div>

            {/* Animated Progress Indicator */}
            <div className="space-y-2">
              <div className="flex items-center justify-between text-xs font-mono">
                <span className="text-muted-foreground">
                  ZONE {currentStep.stepNumber} OF 04 ·{" "}
                  <strong className="text-foreground">{currentStep.label.toUpperCase()}</strong>
                </span>
                <span className="tabular-nums text-safe font-semibold">{progressPct}%</span>
              </div>
              <div className="h-1.5 w-full rounded-full bg-surface-panel overflow-hidden">
                <motion.div
                  className="h-full rounded-full bg-primary"
                  animate={{ width: `${progressPct}%` }}
                  transition={{ duration: 0.4, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            </div>

            {/* 4-Step Selectable Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
              {CALIBRATION_STEPS.map((step, idx) => {
                const Icon = step.icon;
                const isActive = idx === activeIndex;
                const isCompleted = idx < activeIndex;

                return (
                  <button
                    key={step.id}
                    type="button"
                    onClick={() => {
                      setActiveIndex(idx);
                      setIsPlaying(false);
                    }}
                    className={cn(
                      "text-left rounded-2xl border p-4 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                      isActive
                        ? "border-primary/70 bg-background shadow-xs"
                        : "border-border bg-surface-panel/70 hover:border-border/80 hover:bg-background/60"
                    )}
                  >
                    <div className="flex items-center justify-between mb-2.5">
                      <span
                        className={cn(
                          "inline-flex items-center justify-center size-8 rounded-lg border",
                          isActive
                            ? "border-primary/50 bg-primary/15 text-primary-accent"
                            : isCompleted
                              ? "border-safe/40 bg-safe/15 text-safe"
                              : "border-border bg-card text-muted-foreground"
                        )}
                      >
                        {isCompleted ? (
                          <CheckCircle2 className="size-4" />
                        ) : (
                          <Icon className="size-4" />
                        )}
                      </span>
                      <span className="text-[10px] font-mono text-muted-foreground">
                        {isCompleted
                          ? "BASELINE LOCKED"
                          : isActive
                            ? "CALIBRATING"
                            : `STEP ${step.stepNumber}`}
                      </span>
                    </div>

                    <div className="space-y-1">
                      <span className="block font-display text-sm font-semibold text-foreground">
                        {step.label}
                      </span>
                      <div className="flex items-center gap-3 text-[11px] font-mono tabular-nums text-muted-foreground">
                        <span>Median: {step.medianEar}</span>
                        <span>·</span>
                        <span className={isActive ? "text-safe font-semibold" : undefined}>
                          Floor: {step.zoneThreshold}
                        </span>
                      </div>
                    </div>
                  </button>
                );
              })}
            </div>

            {/* Active Zone Detailed Baseline Output */}
            <div className="rounded-2xl border border-border bg-background p-4 sm:p-5 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="flex items-center gap-2 text-xs font-mono text-safe font-semibold">
                  <span>PERSONALIZED PROFILE OUTPUT</span>
                  <span className="text-border">·</span>
                  <span className="text-muted-foreground font-normal">{currentStep.yawRange}</span>
                </div>
                <p className="text-xs text-muted-foreground leading-relaxed max-w-lg">
                  {currentStep.detail}
                </p>
              </div>
              <div className="shrink-0 rounded-xl border border-safe/35 bg-safe/10 px-3.5 py-2 text-right font-mono">
                <span className="block text-[10px] text-muted-foreground">ZONE THRESHOLD</span>
                <span className="text-base font-bold tabular-nums text-safe">
                  {currentStep.zoneThreshold}
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
