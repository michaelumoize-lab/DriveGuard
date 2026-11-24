"use client";

import React, { useEffect, useMemo, useState } from "react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { FormattedTimestamp } from "@/components/ui/formatted-timestamp";
import { explainReasonCode } from "@/types/telemetry";
import {
  Activity,
  CheckCircle2,
  Clock,
  Compass,
  Cpu,
  Eye,
  Pause,
  Play,
  RotateCcw,
  ShieldAlert,
} from "lucide-react";

export interface ReplayAlertEvent {
  id: string;
  timestampMs: number;
  createdAt: string;
  severity: string;
  riskScore: number;
  earValue: number;
  perclosPct: number;
  poseZone: string;
  reasons: string[];
  cascadeDecision: string;
}

export interface DriverBaselineSnapshot {
  centerEarMedian: number;
  baselinePerclosPct: number;
  isCalibrated: boolean;
  calibrationMode: string;
  sensitivityLevel: string;
}

interface SessionReplayTimelineProps {
  sessionStartedAt: string;
  durationSeconds: number;
  alerts: ReplayAlertEvent[];
  baseline: DriverBaselineSnapshot;
}

function formatOffset(seconds: number): string {
  const safe = Math.max(0, Math.floor(seconds));
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  return `${String(mins).padStart(2, "0")}:${String(secs).padStart(2, "0")}`;
}

export function SessionReplayTimeline({
  sessionStartedAt,
  durationSeconds,
  alerts,
  baseline,
}: SessionReplayTimelineProps) {
  const startMs = useMemo(() => new Date(sessionStartedAt).getTime(), [sessionStartedAt]);

  const enrichedAlerts = useMemo(() => {
    return alerts
      .map((alert) => {
        const createdMs = new Date(alert.createdAt).getTime();
        const rawOffsetSec = Number.isFinite(createdMs) && Number.isFinite(startMs)
          ? Math.max(0, Math.round((createdMs - startMs) / 1000))
          : 0;
        const offsetSec =
          durationSeconds > 0 ? Math.min(rawOffsetSec, durationSeconds) : rawOffsetSec;
        return {
          ...alert,
          offsetSec,
          offsetLabel: `+${formatOffset(offsetSec)}`,
          riskPct: Math.round(alert.riskScore * 100),
        };
      })
      .sort((a, b) => a.offsetSec - b.offsetSec || a.timestampMs - b.timestampMs);
  }, [alerts, durationSeconds, startMs]);

  const effectiveDuration = useMemo(() => {
    if (durationSeconds > 0) return durationSeconds;
    const maxOffset = enrichedAlerts.reduce((acc, item) => Math.max(acc, item.offsetSec), 0);
    return Math.max(maxOffset, 60);
  }, [durationSeconds, enrichedAlerts]);

  const [selectedId, setSelectedId] = useState<string | null>(
    enrichedAlerts[0]?.id ?? null
  );
  const [playheadSec, setPlayheadSec] = useState<number>(
    enrichedAlerts[0]?.offsetSec ?? 0
  );
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState<1 | 2 | 4>(2);
  const [chartMode, setChartMode] = useState<"fatigue" | "ear">("fatigue");

  const selectedAlert = useMemo(
    () => enrichedAlerts.find((a) => a.id === selectedId) ?? enrichedAlerts[0] ?? null,
    [enrichedAlerts, selectedId]
  );

  // Advance playhead during replay and auto-select the most recent event passed
  useEffect(() => {
    if (!isPlaying) return;
    const interval = setInterval(() => {
      setPlayheadSec((prev) => {
        const next = prev + playbackSpeed;
        if (next >= effectiveDuration) {
          setIsPlaying(false);
          return effectiveDuration;
        }
        return next;
      });
    }, 500);
    return () => clearInterval(interval);
  }, [isPlaying, playbackSpeed, effectiveDuration]);

  useEffect(() => {
    if (!isPlaying || enrichedAlerts.length === 0) return;
    const passed = enrichedAlerts.filter((a) => a.offsetSec <= playheadSec);
    const latest = passed[passed.length - 1];
    if (latest && latest.id !== selectedId) {
      setSelectedId(latest.id);
    }
  }, [playheadSec, isPlaying, enrichedAlerts, selectedId]);

  const handleSelectEvent = (alertId: string, offsetSec: number) => {
    setIsPlaying(false);
    setSelectedId(alertId);
    setPlayheadSec(offsetSec);
  };

  const handleReset = () => {
    setIsPlaying(false);
    setPlayheadSec(0);
    setSelectedId(enrichedAlerts[0]?.id ?? null);
  };

  const chartData = useMemo(
    () =>
      enrichedAlerts.map((a) => ({
        time: a.offsetLabel,
        perclos: Number(a.perclosPct.toFixed(1)),
        riskPct: a.riskPct,
        ear: Number(a.earValue.toFixed(3)),
        baselineEar: Number(baseline.centerEarMedian.toFixed(3)),
      })),
    [enrichedAlerts, baseline.centerEarMedian]
  );

  if (enrichedAlerts.length === 0) {
    return (
      <Card className="border-emerald-500/30 bg-emerald-500/5">
        <CardHeader>
          <CardTitle className="text-base flex items-center gap-2 text-emerald-600 dark:text-emerald-400">
            <CheckCircle2 className="size-5" />
            Clean Trip — No Fatigue Events Detected
          </CardTitle>
          <CardDescription>
            Zero warning or critical drowsiness escalations were recorded during this{" "}
            {formatOffset(effectiveDuration)} monitoring session.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground flex flex-wrap gap-4">
          <span>
            Calibrated EAR Baseline:{" "}
            <strong className="font-mono text-foreground">
              {baseline.centerEarMedian.toFixed(3)}
            </strong>
          </span>
          <span>
            Baseline PERCLOS:{" "}
            <strong className="font-mono text-foreground">
              {baseline.baselinePerclosPct.toFixed(1)}%
            </strong>
          </span>
          <span>
            Sensitivity:{" "}
            <strong className="font-mono text-foreground">{baseline.sensitivityLevel}</strong>
          </span>
        </CardContent>
      </Card>
    );
  }

  return (
    <div className="space-y-6">
      {/* Interactive Timeline Scrubber */}
      <Card className="border-border/80">
        <CardHeader className="pb-3 flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div>
            <CardTitle className="text-base flex items-center gap-2">
              <Clock className="size-4 text-primary" />
              Session Replay &amp; Detection Timeline
            </CardTitle>
            <CardDescription className="text-xs">
              Scrub through the trip or click any detection marker to inspect why an alert fired.
            </CardDescription>
          </div>

          <div className="flex items-center gap-2">
            <Button
              type="button"
              size="sm"
              variant={isPlaying ? "secondary" : "default"}
              onClick={() => {
                if (playheadSec >= effectiveDuration) setPlayheadSec(0);
                setIsPlaying((p) => !p);
              }}
              className="h-8 text-xs"
            >
              {isPlaying ? (
                <>
                  <Pause className="size-3.5 mr-1.5" /> Pause
                </>
              ) : (
                <>
                  <Play className="size-3.5 mr-1.5" /> Replay
                </>
              )}
            </Button>

            <Button
              type="button"
              size="sm"
              variant="outline"
              onClick={handleReset}
              aria-label="Reset replay to start"
              className="h-8 px-2.5 text-xs"
            >
              <RotateCcw className="size-3.5" />
            </Button>

            <div className="inline-flex rounded-md border border-border bg-muted/40 p-0.5 text-[11px] font-mono">
              {([1, 2, 4] as const).map((speed) => (
                <button
                  key={speed}
                  type="button"
                  onClick={() => setPlaybackSpeed(speed)}
                  className={`px-2 py-0.5 rounded transition-colors cursor-pointer ${
                    playbackSpeed === speed
                      ? "bg-primary text-primary-foreground font-semibold"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {speed}x
                </button>
              ))}
            </div>
          </div>
        </CardHeader>

        <CardContent className="space-y-3">
          {/* Timeline Track */}
          <div className="relative pt-4 pb-2">
            <div className="h-2.5 w-full rounded-full bg-muted relative overflow-visible">
              {/* Progress fill */}
              <div
                className="h-full rounded-full bg-primary/35 transition-all duration-300"
                style={{
                  width: `${Math.min(100, (playheadSec / effectiveDuration) * 100)}%`,
                }}
              />
              {/* Playhead */}
              <div
                className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-3.5 h-3.5 rounded-full bg-primary border-2 border-background shadow transition-all duration-300 pointer-events-none"
                style={{
                  left: `${Math.min(100, (playheadSec / effectiveDuration) * 100)}%`,
                }}
              />
              {/* Alert Pins */}
              {enrichedAlerts.map((alert) => {
                const leftPct = Math.min(99, Math.max(1, (alert.offsetSec / effectiveDuration) * 100));
                const isSelected = selectedAlert?.id === alert.id;
                const isCritical = alert.severity === "CRITICAL";
                return (
                  <button
                    key={alert.id}
                    type="button"
                    onClick={() => handleSelectEvent(alert.id, alert.offsetSec)}
                    title={`${alert.offsetLabel} — ${alert.severity} (Risk ${alert.riskPct}%)`}
                    aria-label={`Jump to ${alert.severity} event at ${alert.offsetLabel}`}
                    style={{ left: `${leftPct}%` }}
                    className={`absolute -top-2 -translate-x-1/2 w-3.5 h-6 rounded-sm border transition-transform cursor-pointer ${
                      isCritical
                        ? "bg-red-500 border-red-200 dark:border-red-900"
                        : "bg-amber-500 border-amber-200 dark:border-amber-900"
                    } ${isSelected ? "scale-125 ring-2 ring-ring z-10" : "opacity-85 hover:opacity-100"}`}
                  />
                );
              })}
            </div>

            <div className="flex items-center justify-between text-[11px] font-mono text-muted-foreground mt-2">
              <span>00:00</span>
              <span className="text-foreground font-semibold">
                Playhead: {formatOffset(playheadSec)} / {formatOffset(effectiveDuration)}
              </span>
              <span>{formatOffset(effectiveDuration)}</span>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Physiological & Risk Trend Chart */}
      {chartData.length >= 2 && (
        <Card className="border-border/80">
          <CardHeader className="py-3 px-4 flex flex-row items-center justify-between gap-2 space-y-0 border-b border-border/60">
            <CardTitle className="text-sm font-semibold flex items-center gap-2">
              <Activity className="size-4 text-primary" />
              Detection Trend Across Events
            </CardTitle>
            <div className="inline-flex items-center rounded-lg border border-border bg-muted/40 p-0.5 text-[11px] font-medium">
              <button
                type="button"
                onClick={() => setChartMode("fatigue")}
                className={`px-2.5 py-0.5 rounded-md transition-colors cursor-pointer ${
                  chartMode === "fatigue"
                    ? "bg-primary text-primary-foreground font-semibold"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                PERCLOS &amp; Risk %
              </button>
              <button
                type="button"
                onClick={() => setChartMode("ear")}
                className={`px-2.5 py-0.5 rounded-md transition-colors cursor-pointer ${
                  chartMode === "ear"
                    ? "bg-primary text-primary-foreground font-semibold"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                EAR vs. Baseline
              </button>
            </div>
          </CardHeader>
          <CardContent className="p-4">
            <div className="h-[220px] w-full min-w-0">
              <ResponsiveContainer
                width="100%"
                height="100%"
                minWidth={0}
                minHeight={220}
                initialDimension={{ width: 480, height: 220 }}
              >
                {chartMode === "fatigue" ? (
                  <AreaChart data={chartData} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                    <defs>
                      <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#ef4444" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#ef4444" stopOpacity={0.02} />
                      </linearGradient>
                      <linearGradient id="perclosGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="#f59e0b" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        borderColor: "hsl(var(--border))",
                        fontSize: "12px",
                      }}
                    />
                    <ReferenceLine
                      y={baseline.baselinePerclosPct}
                      stroke="#10b981"
                      strokeDasharray="4 4"
                      label={{ value: "Baseline PERCLOS", fontSize: 10, fill: "#10b981" }}
                    />
                    <Area
                      type="monotone"
                      dataKey="riskPct"
                      name="Risk Score (%)"
                      stroke="#ef4444"
                      fill="url(#riskGrad)"
                      strokeWidth={2}
                    />
                    <Area
                      type="monotone"
                      dataKey="perclos"
                      name="PERCLOS (%)"
                      stroke="#f59e0b"
                      fill="url(#perclosGrad)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                ) : (
                  <AreaChart data={chartData} margin={{ top: 8, right: 12, left: -16, bottom: 0 }}>
                    <defs>
                      <linearGradient id="earSessionGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.35} />
                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" opacity={0.15} />
                    <XAxis dataKey="time" tick={{ fontSize: 11 }} />
                    <YAxis domain={[0, 0.5]} tick={{ fontSize: 11 }} />
                    <Tooltip
                      contentStyle={{
                        backgroundColor: "hsl(var(--card))",
                        borderColor: "hsl(var(--border))",
                        fontSize: "12px",
                      }}
                    />
                    <ReferenceLine
                      y={baseline.centerEarMedian}
                      stroke="#10b981"
                      strokeDasharray="4 4"
                      label={{
                        value: `Calibrated EAR (${baseline.centerEarMedian.toFixed(3)})`,
                        fontSize: 10,
                        fill: "#10b981",
                      }}
                    />
                    <Area
                      type="monotone"
                      dataKey="ear"
                      name="Observed EAR"
                      stroke="#3b82f6"
                      fill="url(#earSessionGrad)"
                      strokeWidth={2}
                    />
                  </AreaChart>
                )}
              </ResponsiveContainer>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Split View: Chronological Event List + Explainable Event Inspector */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Event List */}
        <Card className="lg:col-span-5 border-border/80">
          <CardHeader className="pb-3">
            <CardTitle className="text-sm flex items-center gap-2">
              <ShieldAlert className="size-4 text-primary" />
              Detection Events ({enrichedAlerts.length})
            </CardTitle>
            <CardDescription className="text-xs">
              Chronological fatigue escalations captured during this trip.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-2 max-h-[420px] overflow-y-auto pr-2">
            {enrichedAlerts.map((alert) => {
              const isSelected = selectedAlert?.id === alert.id;
              return (
                <button
                  key={alert.id}
                  type="button"
                  onClick={() => handleSelectEvent(alert.id, alert.offsetSec)}
                  className={`w-full text-left p-3 rounded-lg border transition-colors cursor-pointer ${
                    isSelected
                      ? "border-primary bg-primary/5"
                      : "border-border bg-muted/15 hover:bg-muted/30"
                  }`}
                >
                  <div className="flex items-center justify-between gap-2">
                    <div className="flex items-center gap-2">
                      <Badge
                        variant={alert.severity === "CRITICAL" ? "destructive" : "outline"}
                        className="text-[10px] font-mono"
                      >
                        {alert.severity}
                      </Badge>
                      <span className="font-mono text-xs font-semibold">{alert.offsetLabel}</span>
                    </div>
                    <span className="font-mono text-xs text-muted-foreground">
                      Risk {alert.riskPct}%
                    </span>
                  </div>
                  <div className="mt-1.5 flex items-center justify-between text-[11px] text-muted-foreground font-mono">
                    <span>
                      EAR {alert.earValue.toFixed(3)} • PERCLOS {alert.perclosPct.toFixed(1)}%
                    </span>
                    <span>{alert.poseZone}</span>
                  </div>
                </button>
              );
            })}
          </CardContent>
        </Card>

        {/* Right: Selected Event Explainability Inspector */}
        <Card className="lg:col-span-7 border-border/80">
          <CardHeader className="pb-3 border-b border-border/60">
            <div className="flex items-center justify-between gap-2">
              <CardTitle className="text-sm flex items-center gap-2">
                <Eye className="size-4 text-primary" />
                Explainable Event Breakdown
              </CardTitle>
              {selectedAlert && (
                <Badge
                  variant={selectedAlert.severity === "CRITICAL" ? "destructive" : "outline"}
                  className="font-mono text-[11px]"
                >
                  {selectedAlert.severity} at {selectedAlert.offsetLabel}
                </Badge>
              )}
            </div>
            {selectedAlert && (
              <CardDescription className="text-xs">
                Recorded at{" "}
                <FormattedTimestamp date={selectedAlert.createdAt} mode="datetime" showSeconds={true} />
              </CardDescription>
            )}
          </CardHeader>

          {selectedAlert && (
            <CardContent className="pt-4 space-y-5">
              {/* Metrics vs Baseline */}
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                <div className="p-3 rounded-lg border border-border bg-muted/20">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Risk Score
                  </span>
                  <div className="text-lg font-bold font-mono mt-0.5">
                    {selectedAlert.riskPct}%
                  </div>
                </div>

                <div className="p-3 rounded-lg border border-border bg-muted/20">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    EAR vs Baseline
                  </span>
                  <div className="text-lg font-bold font-mono mt-0.5">
                    {selectedAlert.earValue.toFixed(3)}
                  </div>
                  <div className="text-[10px] font-mono text-muted-foreground">
                    Base {baseline.centerEarMedian.toFixed(3)} (
                    {(selectedAlert.earValue - baseline.centerEarMedian).toFixed(3)})
                  </div>
                </div>

                <div className="p-3 rounded-lg border border-border bg-muted/20">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    PERCLOS
                  </span>
                  <div className="text-lg font-bold font-mono mt-0.5">
                    {selectedAlert.perclosPct.toFixed(1)}%
                  </div>
                  <div className="text-[10px] font-mono text-muted-foreground">
                    Base {baseline.baselinePerclosPct.toFixed(1)}%
                  </div>
                </div>

                <div className="p-3 rounded-lg border border-border bg-muted/20">
                  <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                    Pose &amp; Cascade
                  </span>
                  <div className="text-xs font-bold font-mono mt-1 flex items-center gap-1">
                    <Compass className="size-3 text-muted-foreground" />
                    {selectedAlert.poseZone}
                  </div>
                  <div className="text-[10px] font-mono text-muted-foreground flex items-center gap-1 mt-0.5">
                    <Cpu className="size-3" />
                    {selectedAlert.cascadeDecision}
                  </div>
                </div>
              </div>

              {/* Reason Codes */}
              <div className="space-y-2">
                <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground">
                  Why This Alert Triggered
                </h4>
                {selectedAlert.reasons.length === 0 ? (
                  <p className="text-xs text-muted-foreground">
                    No specific reason codes attached to this event.
                  </p>
                ) : (
                  <div className="space-y-2">
                    {selectedAlert.reasons.map((code) => {
                      const info = explainReasonCode(code);
                      return (
                        <div
                          key={code}
                          className="p-3 rounded-lg border border-border bg-background/60 space-y-1"
                        >
                          <div className="flex items-center justify-between gap-2">
                            <span className="text-xs font-semibold text-foreground">
                              {info.title}
                            </span>
                            <Badge variant="outline" className="font-mono text-[10px]">
                              {code}
                            </Badge>
                          </div>
                          <p className="text-xs text-muted-foreground">{info.explanation}</p>
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            </CardContent>
          )}
        </Card>
      </div>
    </div>
  );
}
