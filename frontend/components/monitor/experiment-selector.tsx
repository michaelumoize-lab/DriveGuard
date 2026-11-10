"use client";

import React from "react";
import { ExperimentMode } from "@/types/telemetry";
import { Sparkles, Cpu, Eye, CheckCircle2 } from "lucide-react";

interface ExperimentSelectorProps {
  currentMode: ExperimentMode;
  onSelectMode: (mode: ExperimentMode) => void;
  disabled?: boolean;
}

export function ExperimentSelector({
  currentMode,
  onSelectMode,
  disabled = false,
}: ExperimentSelectorProps) {
  const experiments: {
    id: ExperimentMode;
    label: string;
    sublabel: string;
    tag: string;
    desc: string;
    icon: React.ReactNode;
  }[] = [
    {
      id: "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
      label: "Cascade (Adaptive)",
      sublabel: "EAR + MobileNetV3 CNN",
      tag: "Recommended",
      desc: "High accuracy & low latency: uses geometric fast-path and escalates to CNN only on borderline confidence.",
      icon: <Sparkles className="size-4 text-primary shrink-0" />,
    },
    {
      id: "EXPERIMENT_A_GEOMETRIC_ONLY",
      label: "Geometric Fast-Path",
      sublabel: "MediaPipe 3D EAR only",
      tag: "<2ms latency",
      desc: "Zero CNN inference: relies solely on 3D eye aspect ratio and facial landmarks.",
      icon: <Eye className="size-4 text-emerald-500 shrink-0" />,
    },
    {
      id: "EXPERIMENT_B_CNN_ALWAYS",
      label: "CNN Heavy Inference",
      sublabel: "MobileNetV3 per frame",
      tag: "Compute Intensive",
      desc: "Runs deep learning eye state classification on every frame regardless of confidence.",
      icon: <Cpu className="size-4 text-amber-500 shrink-0" />,
    },
  ];

  return (
    <div
      role="radiogroup"
      aria-label="Cascade AI Inference Experiment Mode"
      className="grid grid-cols-1 gap-2"
    >
      {experiments.map((exp) => {
        const isSelected = currentMode === exp.id;
        return (
          <button
            key={exp.id}
            type="button"
            role="radio"
            aria-checked={isSelected}
            disabled={disabled}
            onClick={() => onSelectMode(exp.id)}
            className={`w-full text-left p-2.5 rounded-xl border transition-all flex items-start justify-between gap-3 cursor-pointer outline-none focus-visible:ring-2 focus-visible:ring-primary ${
              isSelected
                ? "bg-primary/10 border-primary/40 shadow-xs"
                : "bg-card/50 hover:bg-card border-border/80 text-muted-foreground hover:text-foreground"
            } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
            title={exp.desc}
          >
            <div className="flex items-start gap-2.5 min-w-0">
              <div className="mt-0.5">{exp.icon}</div>
              <div className="space-y-0.5 min-w-0">
                <div className="flex items-center gap-1.5 flex-wrap">
                  <span
                    className={`text-xs font-semibold ${
                      isSelected ? "text-foreground" : "text-muted-foreground"
                    }`}
                  >
                    {exp.label}
                  </span>
                  <span className="text-[10px] font-mono px-1.5 py-0.2 rounded-md bg-muted text-muted-foreground">
                    {exp.tag}
                  </span>
                </div>
                <p className="text-[11px] text-muted-foreground leading-snug">
                  {exp.desc}
                </p>
              </div>
            </div>

            <div className="shrink-0 mt-0.5">
              <div
                className={`size-4 rounded-full border flex items-center justify-center transition-all ${
                  isSelected
                    ? "border-primary bg-primary text-primary-foreground"
                    : "border-border"
                }`}
              >
                {isSelected && <span className="size-1.5 rounded-full bg-white" />}
              </div>
            </div>
          </button>
        );
      })}
    </div>
  );
}
