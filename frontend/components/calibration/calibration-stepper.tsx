"use client";

import React from "react";
import { CheckCircle2, Camera, Compass, ShieldCheck } from "lucide-react";
import { CALIBRATION_ZONES } from "./calibration-zones";

export type CalibrationJourneyStage = "overview" | "positioning" | "calibrating" | "complete";

interface CalibrationStepperProps {
  journeyStage: CalibrationJourneyStage;
  currentZone?: string | null;
  isCalibrating: boolean;
  isSuccess: boolean;
}

export function CalibrationStepper({
  journeyStage,
  currentZone,
  isCalibrating,
  isSuccess,
}: CalibrationStepperProps) {
  const highLevelSteps: Array<{
    id: string;
    number: number;
    title: string;
    subtitle: string;
    icon: React.ComponentType<{ className?: string }>;
    isActive: boolean;
    isDone: boolean;
  }> = [
    {
      id: "positioning",
      number: 1,
      title: "Camera & Positioning",
      subtitle: "Verify framing & light",
      icon: Camera,
      isActive: journeyStage === "overview" || journeyStage === "positioning",
      isDone: journeyStage === "calibrating" || journeyStage === "complete",
    },
    {
      id: "capture",
      number: 2,
      title: "4-Zone Capture",
      subtitle: "Road, mirrors & dash",
      icon: Compass,
      isActive: journeyStage === "calibrating",
      isDone: journeyStage === "complete",
    },
    {
      id: "complete",
      number: 3,
      title: "Saved & Ready",
      subtitle: "Profile active",
      icon: ShieldCheck,
      isActive: journeyStage === "complete",
      isDone: journeyStage === "complete",
    },
  ];

  const activeZoneIndex = CALIBRATION_ZONES.findIndex((z) => z.key === currentZone);

  return (
    <div className="space-y-3">
      {/* 3-Step Journey Header */}
      <ol
        aria-label="Calibration workflow steps"
        className="grid grid-cols-1 sm:grid-cols-3 gap-2"
      >
        {highLevelSteps.map((step) => {
          const Icon = step.icon;
          return (
            <li
              key={step.id}
              aria-current={step.isActive ? "step" : undefined}
              className={`flex items-center gap-2.5 rounded-xl border px-3 py-2 text-xs transition-colors ${
                step.isActive
                  ? "border-primary bg-primary/10 text-foreground"
                  : step.isDone
                    ? "border-emerald-500/35 bg-emerald-500/5 text-foreground"
                    : "border-border/70 bg-muted/20 text-muted-foreground"
              }`}
            >
              <div
                className={`flex size-6 shrink-0 items-center justify-center rounded-full text-[11px] font-semibold ${
                  step.isDone
                    ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400"
                    : step.isActive
                      ? "bg-primary text-primary-foreground"
                      : "bg-muted text-muted-foreground"
                }`}
              >
                {step.isDone ? (
                  <CheckCircle2 className="size-3.5" aria-hidden="true" />
                ) : (
                  <Icon className="size-3.5" aria-hidden="true" />
                )}
              </div>
              <div className="min-w-0">
                <div className="font-semibold truncate">
                  <span className="sr-only">Step {step.number}: </span>
                  {step.title}
                </div>
                <div className="hidden sm:block text-[11px] text-muted-foreground truncate">
                  {step.subtitle}
                </div>
              </div>
            </li>
          );
        })}
      </ol>

      {/* 4-Zone Spatial Progress Strip */}
      <ol
        aria-label="4-zone gaze calibration progress"
        className="grid grid-cols-2 sm:grid-cols-4 gap-2"
      >
        {CALIBRATION_ZONES.map((zone, idx) => {
          const isCurrent = isCalibrating && currentZone === zone.key;
          const isCompletedZone =
            isSuccess || (isCalibrating && activeZoneIndex > idx);

          return (
            <li
              key={zone.key}
              aria-current={isCurrent ? "step" : undefined}
              className={`rounded-xl border px-3 py-2 text-xs transition-colors ${
                isCurrent
                  ? "border-primary bg-primary/10 text-foreground shadow-xs"
                  : isCompletedZone
                    ? "border-emerald-500/40 bg-emerald-500/5 text-emerald-600 dark:text-emerald-400"
                    : "border-border/70 bg-card/60 text-muted-foreground"
              }`}
            >
              <div className="flex items-center justify-between gap-1.5 font-semibold">
                <span className="truncate tabular-nums">
                  {zone.stepNumber}. {zone.shortTitle}
                </span>
                {isCompletedZone && (
                  <CheckCircle2
                    className="size-3.5 shrink-0 text-emerald-500"
                    aria-hidden="true"
                  />
                )}
              </div>
              <div className="text-[11px] opacity-80 truncate mt-0.5">
                {zone.subtitle}
              </div>
            </li>
          );
        })}
      </ol>
    </div>
  );
}
