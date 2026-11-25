"use client";

import React, { useState } from "react";
import Link from "next/link";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { FaultInjectionToolbar } from "@/components/engineering/fault-injection-toolbar";
import {
  Activity,
  AlertTriangle,
  FlaskConical,
  ShieldCheck,
} from "lucide-react";

interface SyntheticScenarioPreview {
  title: string;
  operationalState: "MONITORING_ACTIVE" | "DEGRADED" | "ERROR";
  drowsinessState: "NORMAL" | "ATTENTION" | "WARNING" | "CRITICAL" | "INSUFFICIENT_DATA";
  poseZone: string;
  ear: number;
  closureThreshold: number;
  rcr: number;
  perclosPct: number;
  reasonCodes: string[];
  summary: string;
}

const SCENARIO_MAP: Record<string, SyntheticScenarioPreview> = {
  NOMINAL: {
    title: "Nominal Driving Baseline",
    operationalState: "MONITORING_ACTIVE",
    drowsinessState: "NORMAL",
    poseZone: "CENTER",
    ear: 0.312,
    closureThreshold: 0.215,
    rcr: 0.96,
    perclosPct: 3.2,
    reasonCodes: ["ALERT_NORMAL"],
    summary: "Calibrated driver looking straight ahead with crisp blink velocity and low PERCLOS.",
  },
  MIRROR_GLANCE: {
    title: "Left Wing Mirror Glance (Yaw +25°, Dwell 1.6s)",
    operationalState: "MONITORING_ACTIVE",
    drowsinessState: "NORMAL",
    poseZone: "LEFT_MIRROR",
    ear: 0.238,
    closureThreshold: 0.194,
    rcr: 0.94,
    perclosPct: 4.1,
    reasonCodes: ["MIRROR_CHECK_CONTEXT"],
    summary:
      "4-Zone PDBP adapts closure cutoff for LEFT_MIRROR and suppresses transient alarms for glances ≤ 3.0 s.",
  },
  SLUGGISH_REOPEN: {
    title: "Sluggish Eyelid Reopening (RCR = 0.65, Duration 580 ms)",
    operationalState: "MONITORING_ACTIVE",
    drowsinessState: "WARNING",
    poseZone: "CENTER",
    ear: 0.205,
    closureThreshold: 0.215,
    rcr: 0.65,
    perclosPct: 16.8,
    reasonCodes: ["PROLONGED_BLINK", "PERCLOS_ELEVATED"],
    summary:
      "Sub-frame blink velocity ratio (RCR < 0.72) combined with elevated rolling 60 s PERCLOS triggers early WARNING.",
  },
  GLARE_DEFOCUS: {
    title: "Severe Optical Glare / Eyewear Defocus",
    operationalState: "DEGRADED",
    drowsinessState: "INSUFFICIENT_DATA",
    poseZone: "CENTER",
    ear: 0.0,
    closureThreshold: 0.215,
    rcr: 0.0,
    perclosPct: 0.0,
    reasonCodes: ["SENSOR_DEGRADED"],
    summary:
      "SignalConfidenceEvaluator detects SEVERELY_DEGRADED optics and gates monitoring into DEGRADED instead of emitting a false NORMAL.",
  },
  WATCHDOG_TIMEOUT: {
    title: "Simulated Heartbeat / Sensor Stream Loss (> 3.5 s)",
    operationalState: "DEGRADED",
    drowsinessState: "INSUFFICIENT_DATA",
    poseZone: "UNKNOWN",
    ear: 0.0,
    closureThreshold: 0.215,
    rcr: 0.0,
    perclosPct: 0.0,
    reasonCodes: ["FACE_LOST", "SENSOR_DEGRADED"],
    summary:
      "Loss of live telemetry stream triggers explicit failsafe degradation without fabricating green NORMAL telemetry.",
  },
};

export default function SyntheticTestbenchPage() {
  const [activeFault, setActiveFault] = useState<string | null>(null);
  const scenario = SCENARIO_MAP[activeFault ?? "NOMINAL"] ?? SCENARIO_MAP.NOMINAL;

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-6xl mx-auto">
      {/* Prominent Non-Live Simulation Banner */}
      <div
        role="note"
        className="p-4 rounded-2xl border border-amber-500/40 bg-amber-500/10 flex flex-col sm:flex-row sm:items-center justify-between gap-3"
      >
        <div className="flex items-start gap-2.5">
          <AlertTriangle className="size-5 text-amber-500 shrink-0 mt-0.5" aria-hidden="true" />
          <div className="space-y-1">
            <div className="flex items-center gap-2 flex-wrap">
              <Badge
                variant="outline"
                className="border-amber-500/60 bg-amber-500/15 text-amber-600 dark:text-amber-300 font-mono text-[11px]"
              >
                SYNTHETIC SIMULATION TESTBENCH — NOT LIVE TELEMETRY
              </Badge>
            </div>
            <p className="text-xs text-muted-foreground">
              This engineering sandbox drives isolated mock telemetry purely in browser memory for viva demonstrations. It never touches live camera sessions, alerts, or PostgreSQL/SQLite persistence.
            </p>
          </div>
        </div>
        <Button asChild size="sm" variant="outline" className="shrink-0 h-8 text-xs">
          <Link href="/dashboard/live-monitor">
            <Activity className="size-3.5 mr-1.5" aria-hidden="true" />
            Return to Live Monitor
          </Link>
        </Button>
      </div>

      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2.5">
          <FlaskConical className="size-6 text-primary" aria-hidden="true" />
          <span>Synthetic Fault Simulation Testbench</span>
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-3xl">
          Inspect how DriveGuard&apos;s deterministic state machine separates operational readiness states from drowsiness severity under synthetic edge-case conditions.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        <div className="lg:col-span-5">
          <FaultInjectionToolbar
            activeFault={activeFault}
            onInjectMirrorGlance={() => setActiveFault("MIRROR_GLANCE")}
            onInjectSluggishReopen={() => setActiveFault("SLUGGISH_REOPEN")}
            onInjectGlare={() => setActiveFault("GLARE_DEFOCUS")}
            onInjectWatchdogTimeout={() => setActiveFault("WATCHDOG_TIMEOUT")}
            onResetFaults={() => setActiveFault(null)}
          />
        </div>

        <Card className="lg:col-span-7 border-border/80">
          <CardHeader className="pb-3 border-b border-border/60">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div>
                <CardTitle className="text-base flex items-center gap-2">
                  <ShieldCheck className="size-4 text-primary" aria-hidden="true" />
                  <span>{scenario.title}</span>
                </CardTitle>
                <CardDescription className="text-xs mt-0.5">
                  Isolated synthetic state machine output (zero production impact)
                </CardDescription>
              </div>
              <Badge variant="secondary" className="font-mono text-[10px]">
                MOCK PREVIEW
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="pt-4 space-y-4 text-xs">
            <div className="grid grid-cols-2 gap-3">
              <div className="p-3 rounded-xl border border-border bg-muted/20">
                <span className="text-[11px] text-muted-foreground block">Operational State</span>
                <span className="text-sm font-bold font-mono text-foreground mt-0.5 block">
                  {scenario.operationalState}
                </span>
              </div>
              <div className="p-3 rounded-xl border border-border bg-muted/20">
                <span className="text-[11px] text-muted-foreground block">Drowsiness State</span>
                <span
                  className={`text-sm font-bold font-mono mt-0.5 block ${
                    scenario.drowsinessState === "WARNING" || scenario.drowsinessState === "CRITICAL"
                      ? "text-amber-500"
                      : scenario.drowsinessState === "INSUFFICIENT_DATA"
                        ? "text-muted-foreground"
                        : "text-emerald-500"
                  }`}
                >
                  {scenario.drowsinessState}
                </span>
              </div>
            </div>

            <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5 font-mono tabular-nums">
              <div className="p-2.5 rounded-lg border border-border bg-background">
                <span className="text-[10px] font-sans text-muted-foreground block">Pose Zone</span>
                <span className="font-semibold text-foreground">{scenario.poseZone}</span>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-background">
                <span className="text-[10px] font-sans text-muted-foreground block">EAR / Cutoff</span>
                <span className="font-semibold text-foreground">
                  {scenario.ear.toFixed(3)} / {scenario.closureThreshold.toFixed(3)}
                </span>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-background">
                <span className="text-[10px] font-sans text-muted-foreground block">Blink RCR</span>
                <span className="font-semibold text-foreground">{scenario.rcr.toFixed(2)}</span>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-background">
                <span className="text-[10px] font-sans text-muted-foreground block">PERCLOS (60s)</span>
                <span className="font-semibold text-foreground">{scenario.perclosPct.toFixed(1)}%</span>
              </div>
            </div>

            <div className="p-3.5 rounded-xl border border-border/70 bg-muted/20 space-y-2">
              <div className="flex items-center gap-1.5 flex-wrap">
                <span className="text-muted-foreground font-medium">Reason Codes:</span>
                {scenario.reasonCodes.map((code) => (
                  <Badge key={code} variant="outline" className="font-mono text-[10px]">
                    {code}
                  </Badge>
                ))}
              </div>
              <p className="text-muted-foreground leading-relaxed">{scenario.summary}</p>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
