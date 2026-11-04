"use client";

import React from "react";
import { ExperimentMode } from "@/types/telemetry";
import { Button } from "@/components/ui/button";
import { Sparkles, Cpu, Eye } from "lucide-react";

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
  const experiments: { id: ExperimentMode; label: string; desc: string; icon: React.ReactNode }[] = [
    {
      id: "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
      label: "Cascade (Adaptive)",
      desc: "Confidence-Gated EAR + MobileNetV3 (Default)",
      icon: <Sparkles className="size-3.5 text-primary" />,
    },
    {
      id: "EXPERIMENT_A_GEOMETRIC_ONLY",
      label: "Exp A: Geometric",
      desc: "Pure MediaPipe 3D EAR (Zero CNN)",
      icon: <Eye className="size-3.5 text-emerald-400" />,
    },
    {
      id: "EXPERIMENT_B_CNN_ALWAYS",
      label: "Exp B: CNN Always",
      desc: "MobileNetV3 on every single frame",
      icon: <Cpu className="size-3.5 text-amber-400" />,
    },
  ];

  return (
    <div className="flex flex-wrap items-center gap-1.5 sm:gap-2">
      {experiments.map((exp) => {
        const isSelected = currentMode === exp.id;
        return (
          <Button
            key={exp.id}
            size="sm"
            variant={isSelected ? "default" : "outline"}
            disabled={disabled}
            onClick={() => onSelectMode(exp.id)}
            className={`text-xs h-8 sm:h-9 px-3 rounded-xl gap-1.5 transition-all ${
              isSelected ? "font-semibold shadow-xs" : "border-border text-muted-foreground hover:text-foreground"
            }`}
            title={exp.desc}
          >
            {exp.icon}
            <span>{exp.label}</span>
          </Button>
        );
      })}
    </div>
  );
}
