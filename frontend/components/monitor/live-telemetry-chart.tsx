"use client";

import React, { useState } from "react";
import {
  ResponsiveContainer,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
} from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Activity, Eye } from "lucide-react";

export interface TelemetryHistoryPoint {
  timeLabel: string;
  ear: number;
  threshold: number;
  perclos: number;
  riskPct: number;
}

interface LiveTelemetryChartProps {
  data: TelemetryHistoryPoint[];
}

export const LiveTelemetryChart = React.memo(function LiveTelemetryChart({
  data,
}: LiveTelemetryChartProps) {
  const [mode, setMode] = useState<"ear" | "fatigue">("ear");

  return (
    <Card className="border-border/80 bg-card/60">
      <CardHeader className="py-3 px-4 flex flex-row items-center justify-between gap-2 space-y-0 border-b border-border/60">
        <CardTitle className="text-xs font-mono uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
          <Activity className="size-3.5 text-primary" />
          Rolling 60s Telemetry Waveform
        </CardTitle>
        <div className="inline-flex items-center rounded-lg border border-border bg-muted/40 p-0.5 text-[11px] font-mono">
          <button
            type="button"
            onClick={() => setMode("ear")}
            className={`px-2 py-0.5 rounded-md transition-colors cursor-pointer ${
              mode === "ear"
                ? "bg-primary text-primary-foreground font-semibold"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            EAR vs Threshold
          </button>
          <button
            type="button"
            onClick={() => setMode("fatigue")}
            className={`px-2 py-0.5 rounded-md transition-colors cursor-pointer ${
              mode === "fatigue"
                ? "bg-primary text-primary-foreground font-semibold"
                : "text-muted-foreground hover:text-foreground"
            }`}
          >
            PERCLOS & Risk %
          </button>
        </div>
      </CardHeader>
      <CardContent className="p-3">
        {data.length < 2 ? (
          <div className="h-[150px] flex flex-col items-center justify-center text-xs text-muted-foreground gap-1.5">
            <Eye className="size-5 text-muted-foreground/50" />
            <span>Start live camera stream to plot real-time physiological waveforms.</span>
          </div>
        ) : (
          <div className="h-[150px] w-full" aria-label="Real-time telemetry chart">
            <ResponsiveContainer width="100%" height="100%">
              {mode === "ear" ? (
                <AreaChart data={data} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="earGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#10b981" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#10b981" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="timeLabel"
                    tick={{ fontSize: 10 }}
                    stroke="currentColor"
                    className="text-muted-foreground"
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    domain={[0.08, 0.42]}
                    tick={{ fontSize: 10 }}
                    stroke="currentColor"
                    className="text-muted-foreground"
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: "11px",
                      borderRadius: "8px",
                      backgroundColor: "rgba(15, 23, 42, 0.92)",
                      borderColor: "rgba(148, 163, 184, 0.25)",
                      color: "#f8fafc",
                    }}
                  />
                  <Area
                    type="monotone"
                    dataKey="ear"
                    name="Raw EAR"
                    stroke="#10b981"
                    strokeWidth={2}
                    fill="url(#earGrad)"
                    isAnimationActive={false}
                  />
                  <Area
                    type="stepAfter"
                    dataKey="threshold"
                    name="Zone Closure Threshold"
                    stroke="#f59e0b"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    fill="none"
                    isAnimationActive={false}
                  />
                </AreaChart>
              ) : (
                <AreaChart data={data} margin={{ top: 6, right: 8, left: -20, bottom: 0 }}>
                  <defs>
                    <linearGradient id="riskGrad" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.35} />
                      <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="timeLabel"
                    tick={{ fontSize: 10 }}
                    stroke="currentColor"
                    className="text-muted-foreground"
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    domain={[0, 100]}
                    tick={{ fontSize: 10 }}
                    stroke="currentColor"
                    className="text-muted-foreground"
                  />
                  <Tooltip
                    contentStyle={{
                      fontSize: "11px",
                      borderRadius: "8px",
                      backgroundColor: "rgba(15, 23, 42, 0.92)",
                      borderColor: "rgba(148, 163, 184, 0.25)",
                      color: "#f8fafc",
                    }}
                  />
                  <ReferenceLine y={52} stroke="#f59e0b" strokeDasharray="3 3" />
                  <ReferenceLine y={78} stroke="#f43f5e" strokeDasharray="3 3" />
                  <Area
                    type="monotone"
                    dataKey="riskPct"
                    name="Composite Risk (%)"
                    stroke="#f43f5e"
                    strokeWidth={2}
                    fill="url(#riskGrad)"
                    isAnimationActive={false}
                  />
                  <Area
                    type="monotone"
                    dataKey="perclos"
                    name="60s PERCLOS (%)"
                    stroke="#3b82f6"
                    strokeWidth={1.8}
                    fill="none"
                    isAnimationActive={false}
                  />
                </AreaChart>
              )}
            </ResponsiveContainer>
          </div>
        )}
      </CardContent>
    </Card>
  );
});
