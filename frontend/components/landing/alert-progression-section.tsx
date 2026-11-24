"use client";

import React, { useEffect, useState } from "react";
import { motion, useReducedMotion } from "motion/react";
import {
  AlertTriangle,
  BellRing,
  Pause,
  Play,
  ShieldCheck,
  Volume2,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface AlertStage {
  id: "normal" | "warning" | "critical";
  label: string;
  headline: string;
  color: string;
  bgTint: string;
  borderTint: string;
  icon: React.ComponentType<{ className?: string }>;
  riskScore: number;
  responseAction: string;
  description: string;
}

const ALERT_STAGES: AlertStage[] = [
  {
    id: "normal",
    label: "Normal monitoring",
    headline: "Driver appears alert",
    color: "var(--safe)",
    bgTint: "color-mix(in srgb, var(--safe) 10%, transparent)",
    borderTint: "color-mix(in srgb, var(--safe) 36%, transparent)",
    icon: ShieldCheck,
    riskScore: 12,
    responseAction: "Silent passive telemetry · Mirror checks suppressed",
    description:
      "Eye closure duration and blink reopening velocity remain well within your 4-zone personalized baseline.",
  },
  {
    id: "warning",
    label: "Early warning",
    headline: "Potential signs of fatigue detected",
    color: "var(--warning)",
    bgTint: "color-mix(in srgb, var(--warning) 11%, transparent)",
    borderTint: "color-mix(in srgb, var(--warning) 44%, transparent)",
    icon: AlertTriangle,
    riskScore: 58,
    responseAction: "Visual HUD highlight · Gentle pre-alert chime",
    description:
      "Rising 60s rolling PERCLOS and sluggish blink recovery persist beyond transient noise, cueing an early non-startling warning.",
  },
  {
    id: "critical",
    label: "Critical alert",
    headline: "Escalated warning to the driver",
    color: "var(--critical)",
    bgTint: "color-mix(in srgb, var(--critical) 12%, transparent)",
    borderTint: "color-mix(in srgb, var(--critical) 48%, transparent)",
    icon: BellRing,
    riskScore: 92,
    responseAction: "Priority acoustic alarm · Reliable ACK & event log",
    description:
      "Prolonged eye closure (≥ 1.1s microsleep override) or sustained high fatigue triggers immediate multi-tier acoustic intervention.",
  },
];

export function AlertProgressionSection() {
  const prefersReducedMotion = useReducedMotion();
  const [stageIndex, setStageIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);

  useEffect(() => {
    if (prefersReducedMotion || !isPlaying) return;
    const timer = window.setInterval(() => {
      setStageIndex((prev) => (prev + 1) % ALERT_STAGES.length);
    }, 3400);
    return () => window.clearInterval(timer);
  }, [prefersReducedMotion, isPlaying]);

  const current = ALERT_STAGES[stageIndex];
  const ActiveIcon = current.icon;

  return (
    <section
      id="alerts"
      className="py-20 md:py-28 border-t border-border scroll-mt-24"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        {/* Section Header */}
        <div className="flex flex-col md:flex-row md:items-end justify-between gap-6">
          <div className="max-w-2xl space-y-3">
            <div className="flex items-center gap-3">
              <span className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-primary-accent">
                <Volume2 className="size-3.5" />
                <span>Intelligent Alert System</span>
              </span>
              <span className="rounded-md border border-border bg-card px-2 py-0.5 text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                Illustrative
              </span>
            </div>
            <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-foreground leading-[1.12] text-balance">
              Escalated protection only when fatigue persists.
            </h2>
            <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
              Hysteresis state timers prevent jarring alarms from single slow blinks while reacting decisively to microsleeps.
            </p>
          </div>

          <button
            type="button"
            onClick={() => setIsPlaying((prev) => !prev)}
            className="inline-flex items-center gap-1.5 self-start md:self-auto rounded-lg border border-border bg-card px-3.5 py-2 text-xs font-medium text-foreground hover:border-primary/50 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
          >
            {isPlaying && !prefersReducedMotion ? (
              <>
                <Pause className="size-3.5 text-primary-accent" />
                <span>Pause sequence</span>
              </>
            ) : (
              <>
                <Play className="size-3.5 text-safe" />
                <span>Play sequence</span>
              </>
            )}
          </button>
        </div>

        {/* Alert Progression Interactive Stage */}
        <div
          className="rounded-3xl border bg-card p-5 sm:p-8 space-y-7 shadow-xl transition-colors duration-500"
          style={{ borderColor: current.borderTint }}
        >
          {/* 3-Stage Selectable Stepper */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {ALERT_STAGES.map((stage, idx) => {
              const Icon = stage.icon;
              const isSelected = idx === stageIndex;
              return (
                <button
                  key={stage.id}
                  type="button"
                  onClick={() => {
                    setStageIndex(idx);
                    setIsPlaying(false);
                  }}
                  className={cn(
                    "relative text-left rounded-2xl border p-5 transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    isSelected
                      ? "bg-background shadow-xs"
                      : "border-border bg-surface-panel/75 hover:bg-background/60"
                  )}
                  style={{
                    borderColor: isSelected ? stage.borderTint : undefined,
                  }}
                >
                  <div className="flex items-center justify-between mb-3">
                    <span
                      className="inline-flex items-center justify-center size-9 rounded-xl border transition-colors"
                      style={{
                        color: stage.color,
                        backgroundColor: isSelected ? stage.bgTint : "transparent",
                        borderColor: isSelected ? stage.borderTint : "var(--border)",
                      }}
                    >
                      <Icon className="size-4" />
                    </span>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-muted-foreground">
                      STAGE 0{idx + 1}
                    </span>
                  </div>

                  <span
                    className="block font-display text-base font-semibold transition-colors"
                    style={{ color: isSelected ? stage.color : "var(--foreground)" }}
                  >
                    {stage.label}
                  </span>
                  <span className="mt-1 block text-xs text-muted-foreground">
                    {stage.headline}
                  </span>
                </button>
              );
            })}
          </div>

          {/* Live HUD State Transition Bar */}
          <div
            className="rounded-2xl border p-5 sm:p-6 transition-colors duration-500 flex flex-col lg:flex-row lg:items-center justify-between gap-6"
            style={{
              backgroundColor: current.bgTint,
              borderColor: current.borderTint,
            }}
          >
            <div className="flex items-start gap-4">
              <div
                className="shrink-0 size-11 rounded-2xl border flex items-center justify-center bg-background/90"
                style={{
                  color: current.color,
                  borderColor: current.borderTint,
                }}
              >
                <ActiveIcon className="size-5" />
              </div>
              <div className="space-y-1.5">
                <div className="flex flex-wrap items-center gap-2.5">
                  <span
                    className="text-xs font-mono font-bold uppercase tracking-wider"
                    style={{ color: current.color }}
                  >
                    {current.label}
                  </span>
                  <span className="text-border">·</span>
                  <span className="text-xs font-mono text-foreground/85">
                    {current.responseAction}
                  </span>
                </div>
                <p className="text-sm text-foreground/90 leading-relaxed max-w-2xl">
                  {current.description}
                </p>
              </div>
            </div>

            {/* Composite Fatigue Gauge */}
            <div className="shrink-0 w-full lg:w-56 space-y-2 rounded-xl border border-border bg-background/90 p-4">
              <div className="flex items-center justify-between text-[11px] font-mono">
                <span className="text-muted-foreground">RISK INDEX</span>
                <span
                  className="font-bold tabular-nums"
                  style={{ color: current.color }}
                >
                  {current.riskScore} / 100
                </span>
              </div>
              <div className="h-2 w-full rounded-full bg-surface-panel overflow-hidden">
                <motion.div
                  className="h-full rounded-full"
                  animate={{
                    width: `${current.riskScore}%`,
                    backgroundColor: current.color,
                  }}
                  transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
