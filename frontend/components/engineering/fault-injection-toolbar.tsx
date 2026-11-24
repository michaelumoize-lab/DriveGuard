"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Compass,
  Zap,
  EyeOff,
  RotateCcw,
  Sparkles,
  ShieldAlert,
} from "lucide-react";
import { toast } from "sonner";

export interface FaultInjectionProps {
  onInjectMirrorGlance: () => void;
  onInjectSluggishReopen: () => void;
  onInjectGlare: () => void;
  onInjectWatchdogTimeout: () => void;
  onResetFaults: () => void;
  activeFault: string | null;
}

export function FaultInjectionToolbar({
  onInjectMirrorGlance,
  onInjectSluggishReopen,
  onInjectGlare,
  onInjectWatchdogTimeout,
  onResetFaults,
  activeFault,
}: FaultInjectionProps) {
  return (
    <div className="p-4 rounded-2xl bg-card border border-border/80 shadow-xs space-y-3">
      <div className="flex items-center justify-between border-b border-border/60 pb-2 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Sparkles className="size-4 text-primary shrink-0" aria-hidden="true" />
          <span className="text-xs font-semibold text-foreground truncate">
            Simulated Edge-Case Scenarios
          </span>
        </div>
        <Badge
          variant="outline"
          className={`text-[10px] font-mono shrink-0 ${
            activeFault
              ? "border-amber-500/50 bg-amber-500/10 text-amber-500"
              : "border-border text-muted-foreground"
          }`}
        >
          {activeFault ? `ACTIVE: ${activeFault}` : "Nominal"}
        </Badge>
      </div>

      <p className="text-xs text-muted-foreground leading-relaxed">
        Inject synthetic edge-case conditions to verify false-alarm suppression, early kinematic fatigue warnings, and failsafe fallbacks:
      </p>

      <div className="grid grid-cols-2 gap-2 pt-0.5 text-xs">
        <Button
          type="button"
          size="sm"
          variant={activeFault === "MIRROR_GLANCE" ? "default" : "outline"}
          onClick={() => {
            onInjectMirrorGlance();
            toast.info("Simulating Left Mirror Glance (Yaw +25°). Verifying zero false alarms…");
          }}
          className="h-auto py-2.5 px-3 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-xs">
            <Compass className="size-3.5 text-sky-400 shrink-0" aria-hidden="true" />
            <span>Mirror Glance</span>
          </div>
          <span className="text-[11px] text-muted-foreground line-clamp-1">
            Tests 4-zone suppression
          </span>
        </Button>

        <Button
          type="button"
          size="sm"
          variant={activeFault === "SLUGGISH_REOPEN" ? "default" : "outline"}
          onClick={() => {
            onInjectSluggishReopen();
            toast.warning("Simulating Sluggish Blink Reopen (RCR = 0.65). Verifying early warning…");
          }}
          className="h-auto py-2.5 px-3 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-xs">
            <Zap className="size-3.5 text-amber-400 shrink-0" aria-hidden="true" />
            <span>Sluggish Blink</span>
          </div>
          <span className="text-[11px] text-muted-foreground line-clamp-1">
            Tests RCR kinematics
          </span>
        </Button>

        <Button
          type="button"
          size="sm"
          variant={activeFault === "GLARE_DEFOCUS" ? "default" : "outline"}
          onClick={() => {
            onInjectGlare();
            toast.info("Simulating Glasses Glare. Verifying geometry fallback…");
          }}
          className="h-auto py-2.5 px-3 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-xs">
            <EyeOff className="size-3.5 text-purple-400 shrink-0" aria-hidden="true" />
            <span>Glasses Glare</span>
          </div>
          <span className="text-[11px] text-muted-foreground line-clamp-1">
            Tests reliability arbitration
          </span>
        </Button>

        <Button
          type="button"
          size="sm"
          variant={activeFault === "WATCHDOG_TIMEOUT" ? "default" : "outline"}
          onClick={() => {
            onInjectWatchdogTimeout();
            toast.error("Simulating Serial Heartbeat Drop. Verifying ESP32 watchdog failsafe…");
          }}
          className="h-auto py-2.5 px-3 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-xs">
            <ShieldAlert className="size-3.5 text-rose-400 shrink-0" aria-hidden="true" />
            <span>Watchdog Drop</span>
          </div>
          <span className="text-[11px] text-muted-foreground line-clamp-1">
            Tests hardware failsafe
          </span>
        </Button>
      </div>

      {activeFault && (
        <div className="flex justify-end pt-1">
          <Button
            type="button"
            size="sm"
            variant="ghost"
            onClick={onResetFaults}
            className="h-7 text-xs text-muted-foreground hover:text-foreground"
          >
            <RotateCcw className="size-3 mr-1.5" aria-hidden="true" />
            Clear Simulated Fault
          </Button>
        </div>
      )}
    </div>
  );
}
