"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { CheckCircle2, ChevronDown, RotateCcw } from "lucide-react";
import type { InitialDriverProfileData } from "@/components/monitor/unified-live-monitor";
import type { ZoneBaselineEntry } from "@/types/telemetry";
import { CALIBRATION_ZONES } from "./calibration-zones";

interface CalibrationProfileInspectorProps {
  savedProfile: InitialDriverProfileData | null;
  isCalibrating: boolean;
  isSavingProfile: boolean;
  onRecalibrateZone: (zone: string) => void;
}

export function CalibrationProfileInspector({
  savedProfile,
  isCalibrating,
  isSavingProfile,
  onRecalibrateZone,
}: CalibrationProfileInspectorProps) {
  const isCalibrated = Boolean(savedProfile?.isCalibrated);
  const zbMap = (savedProfile?.zoneBaselines ?? null) as Record<
    string,
    ZoneBaselineEntry
  > | null;

  return (
    <Card className="border-border/80">
      <CardHeader className="pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
        <div>
          <CardTitle className="text-base font-semibold flex items-center gap-2">
            <CheckCircle2 className="size-4 text-primary shrink-0" aria-hidden="true" />
            <span>Saved 4-Zone Baseline</span>
          </CardTitle>
          <CardDescription className="text-xs">
            Recalibrate any single zone anytime without repeating the full 4-zone setup.
          </CardDescription>
        </div>

        <Badge
          variant="outline"
          className={
            isCalibrated
              ? "border-emerald-500/40 text-emerald-600 dark:text-emerald-400 bg-emerald-500/10 font-mono text-xs w-fit"
              : "border-amber-500/40 text-amber-600 dark:text-amber-400 bg-amber-500/10 font-mono text-xs w-fit"
          }
        >
          {isCalibrated ? "CALIBRATED PROFILE ACTIVE" : "NOT CALIBRATED YET"}
        </Badge>
      </CardHeader>

      <CardContent className="space-y-4">
        {/* 4 Spatial Zones Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {CALIBRATION_ZONES.map((zone) => {
            const zb = isCalibrated ? zbMap?.[zone.key] : undefined;
            const earMed =
              typeof zb?.ear_median === "number"
                ? zb.ear_median
                : zone.key === "CENTER" && isCalibrated && savedProfile
                  ? savedProfile.centerEarMedian
                  : null;
            const thresh =
              typeof zb?.dynamic_closure_threshold === "number"
                ? zb.dynamic_closure_threshold
                : null;
            const sampleCount =
              typeof zb?.sample_count === "number" ? zb.sample_count : null;
            const isEstimatedFallback = isCalibrated && sampleCount === 0;

            return (
              <div
                key={zone.key}
                className="p-3.5 rounded-xl border border-border bg-muted/20 flex flex-col justify-between gap-3"
              >
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-semibold text-foreground">
                      {zone.title}
                    </span>
                    <Badge variant="secondary" className="text-[10px] font-mono">
                      {earMed === null
                        ? "Uncalibrated"
                        : isEstimatedFallback
                          ? "Estimated"
                          : "Calibrated"}
                    </Badge>
                  </div>

                  <div className="pt-1 flex items-baseline justify-between font-mono tabular-nums">
                    <span className="text-[11px] font-sans text-muted-foreground">
                      Normal Eye Openness:
                    </span>
                    <span className="text-sm font-bold text-foreground">
                      {earMed !== null ? earMed.toFixed(3) : "—"}
                    </span>
                  </div>

                  <div className="flex items-baseline justify-between font-mono tabular-nums">
                    <span className="text-[11px] font-sans text-muted-foreground">
                      Closure Cutoff:
                    </span>
                    <span className="text-xs font-semibold text-emerald-600 dark:text-emerald-400">
                      {thresh !== null ? thresh.toFixed(3) : "—"}
                    </span>
                  </div>
                </div>

                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => onRecalibrateZone(zone.key)}
                  disabled={isCalibrating || isSavingProfile}
                  className="w-full h-9 text-xs"
                >
                  <RotateCcw className="size-3 mr-1.5" aria-hidden="true" />
                  Recalibrate {zone.shortTitle}
                </Button>
              </div>
            );
          })}
        </div>

        {/* Collapsible Secondary Technical Details */}
        <details className="group rounded-xl border border-border/70 bg-muted/15 px-3.5 py-2.5 text-xs">
          <summary className="flex cursor-pointer list-none items-center justify-between font-medium text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring rounded-md">
            <span>Technical calibration details</span>
            <ChevronDown
              className="size-4 transition-transform group-open:rotate-180"
              aria-hidden="true"
            />
          </summary>

          <div className="pt-3 mt-2.5 border-t border-border/60 grid grid-cols-2 sm:grid-cols-4 gap-3 font-mono tabular-nums">
            <div className="rounded-lg border border-border/60 bg-card/60 p-2.5">
              <span className="text-[10px] font-sans text-muted-foreground block">
                Profile Mode
              </span>
              <span className="font-semibold text-foreground">
                {savedProfile?.calibrationMode ?? "UNCALIBRATED"}
              </span>
            </div>

            <div className="rounded-lg border border-border/60 bg-card/60 p-2.5">
              <span className="text-[10px] font-sans text-muted-foreground block">
                Center MAD Spread
              </span>
              <span className="font-semibold text-foreground">
                {isCalibrated && savedProfile
                  ? `±${savedProfile.centerEarMadScaled.toFixed(3)}`
                  : "—"}
              </span>
            </div>

            <div className="rounded-lg border border-border/60 bg-card/60 p-2.5">
              <span className="text-[10px] font-sans text-muted-foreground block">
                Neutral Pose (Pitch / Yaw)
              </span>
              <span className="font-semibold text-foreground">
                {isCalibrated && savedProfile
                  ? `${savedProfile.neutralPitchDeg.toFixed(1)}° / ${savedProfile.neutralYawDeg.toFixed(1)}°`
                  : "—"}
              </span>
            </div>

            <div className="rounded-lg border border-border/60 bg-card/60 p-2.5">
              <span className="text-[10px] font-sans text-muted-foreground block">
                Calibration Reliability
              </span>
              <span className="font-semibold text-emerald-600 dark:text-emerald-400">
                {isCalibrated && savedProfile
                  ? `${Math.round(savedProfile.calibrationReliability * 100)}%`
                  : "—"}
              </span>
            </div>
          </div>
        </details>
      </CardContent>
    </Card>
  );
}
