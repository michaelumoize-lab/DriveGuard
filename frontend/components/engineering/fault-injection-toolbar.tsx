"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  Compass,
  Zap,
  EyeOff,
  Flame,
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
    <div className="p-3 sm:p-4 rounded-2xl bg-card border border-border/80 shadow-xs space-y-3">
      <div className="flex items-center justify-between border-b border-border/60 pb-2">
        <div className="flex items-center gap-2">
          <Sparkles className="size-4 text-primary" />
          <span className="text-xs font-bold uppercase tracking-wider text-foreground">
            Academic Viva & Fault Injection Testbench
          </span>
        </div>
        <Badge
          variant="outline"
          className={`text-[10px] font-mono ${
            activeFault
              ? "border-amber-500/50 bg-amber-500/10 text-amber-500 animate-pulse"
              : "border-border text-muted-foreground"
          }`}
        >
          {activeFault ? `FAULT ACTIVE: ${activeFault}` : "READY (NOMINAL)"}
        </Badge>
      </div>

      <p className="text-xs text-muted-foreground leading-relaxed">
        Demonstrate DriveGuard&apos;s 5 academic innovations to examiners by injecting simulated edge-case conditions in real time:
      </p>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 pt-1 font-mono text-xs">
        <Button
          size="sm"
          variant={activeFault === "MIRROR_GLANCE" ? "default" : "outline"}
          onClick={() => {
            onInjectMirrorGlance();
            toast.info("Simulating Left Mirror Glance (Yaw +25°). Verifying 0 false alarms...");
          }}
          className="h-auto py-2 px-2.5 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-[11px]">
            <Compass className="size-3.5 text-sky-400" />
            <span>Mirror Glance</span>
          </div>
          <span className="text-[9px] text-muted-foreground line-clamp-1">
            Tests 4-Zone PDBP Suppression
          </span>
        </Button>

        <Button
          size="sm"
          variant={activeFault === "SLUGGISH_REOPEN" ? "default" : "outline"}
          onClick={() => {
            onInjectSluggishReopen();
            toast.warning("Simulating Neuromuscular Fatigue (RCR = 0.65). Verifying 40s advance warning...");
          }}
          className="h-auto py-2 px-2.5 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-[11px]">
            <Zap className="size-3.5 text-amber-400" />
            <span>Sluggish Blink</span>
          </div>
          <span className="text-[9px] text-muted-foreground line-clamp-1">
            Tests RCR Kinematics (40s Lead)
          </span>
        </Button>

        <Button
          size="sm"
          variant={activeFault === "GLARE_DEFOCUS" ? "default" : "outline"}
          onClick={() => {
            onInjectGlare();
            toast.info("Simulating Specular Glasses Glare. Verifying CNN-to-geometry arbitration fallback...");
          }}
          className="h-auto py-2 px-2.5 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-[11px]">
            <EyeOff className="size-3.5 text-purple-400" />
            <span>Glasses Glare</span>
          </div>
          <span className="text-[9px] text-muted-foreground line-clamp-1">
            Tests Reliability Arbitration
          </span>
        </Button>

        <Button
          size="sm"
          variant={activeFault === "WATCHDOG_TIMEOUT" ? "default" : "outline"}
          onClick={() => {
            onInjectWatchdogTimeout();
            toast.error("Simulating UART Serial Heartbeat Drop. Verifying ESP32 hardware watchdog failsafe...");
          }}
          className="h-auto py-2 px-2.5 flex flex-col items-start gap-1 text-left"
        >
          <div className="flex items-center gap-1.5 font-semibold text-[11px]">
            <ShieldAlert className="size-3.5 text-rose-400" />
            <span>Watchdog Drop</span>
          </div>
          <span className="text-[9px] text-muted-foreground line-clamp-1">
            Tests Cyber-Physical Failsafe
          </span>
        </Button>
      </div>

      {activeFault && (
        <div className="flex justify-end pt-1">
          <Button
            size="sm"
            variant="ghost"
            onClick={onResetFaults}
            className="h-7 text-xs text-muted-foreground hover:text-foreground"
          >
            <RotateCcw className="size-3 mr-1.5" />
            Clear Injected Fault
          </Button>
        </div>
      )}
    </div>
  );
}
