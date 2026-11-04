"use client";

import React, { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { CheckCircle2, XCircle, ShieldCheck, Activity, BarChart2 } from "lucide-react";
import { Button } from "@/components/ui/button";

export interface HourlyAccessData {
  time: string;
  granted: number;
  denied: number;
}

interface AccessOverviewChartProps {
  total: number;
  granted: number;
  denied: number;
  todayTotal: number;
  todayGranted: number;
  todayDenied: number;
  hourlyData?: HourlyAccessData[];
}

export function AccessOverviewChart({
  total,
  granted,
  denied,
  todayTotal,
  todayGranted,
  todayDenied,
  hourlyData,
}: AccessOverviewChartProps) {
  const [viewMode, setViewMode] = useState<"chart" | "bar">("chart");
  const [isMounted, setIsMounted] = useState<boolean>(false);

  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  const grantedPct = total > 0 ? (granted / total) * 100 : 0;
  const deniedPct = total > 0 ? (denied / total) * 100 : 0;

  // Fallback zero-valued baseline buckets if no hourly data is available
  const chartData: HourlyAccessData[] =
    hourlyData && hourlyData.length > 0
      ? hourlyData
      : [
          { time: "00:00", granted: 0, denied: 0 },
          { time: "04:00", granted: 0, denied: 0 },
          { time: "08:00", granted: 0, denied: 0 },
          { time: "12:00", granted: 0, denied: 0 },
          { time: "16:00", granted: 0, denied: 0 },
          { time: "20:00", granted: 0, denied: 0 },
        ];

  return (
    <div className="bg-card text-card-foreground border border-border rounded-2xl p-4 sm:p-6 shadow-sm flex flex-col justify-between space-y-4 sm:space-y-5 h-full min-w-0">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-border/60">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-primary/10 border border-primary/20 text-primary rounded-xl">
            <ShieldCheck className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground tracking-tight">Access Decisions</h3>
            <p className="text-[11px] text-muted-foreground font-mono">Today vs. Historical Trends</p>
          </div>
        </div>

        {/* View mode toggle */}
        <div className="flex items-center gap-1 bg-muted/60 p-1 rounded-lg border border-border/60 text-xs">
          <Button
            type="button"
            variant={viewMode === "chart" ? "secondary" : "ghost"}
            size="sm"
            className="h-6 px-2 text-[11px] font-medium"
            onClick={() => setViewMode("chart")}
          >
            <Activity className="w-3 h-3 mr-1 text-emerald-500" /> Timeline
          </Button>
          <Button
            type="button"
            variant={viewMode === "bar" ? "secondary" : "ghost"}
            size="sm"
            className="h-6 px-2 text-[11px] font-medium"
            onClick={() => setViewMode("bar")}
          >
            <BarChart2 className="w-3 h-3 mr-1 text-primary" /> Ratio
          </Button>
        </div>
      </div>

      {/* Main Visualization Body */}
      {viewMode === "chart" ? (
        <div className="w-full h-44 sm:h-48 pt-2 min-w-0">
          {isMounted ? (
            <ResponsiveContainer
              width="100%"
              height="100%"
              minWidth={0}
              minHeight={0}
              initialDimension={{ width: 480, height: 180 }}
            >
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <defs>
                <linearGradient id="colorGranted" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#10b981" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#10b981" stopOpacity={0.0} />
                </linearGradient>
                <linearGradient id="colorDenied" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#ef4444" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#ef4444" stopOpacity={0.0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="var(--border)" strokeOpacity={0.6} />
              <XAxis
                dataKey="time"
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
              />
              <YAxis
                allowDecimals={false}
                tickLine={false}
                axisLine={false}
                tick={{ fontSize: 10, fill: "var(--muted-foreground)" }}
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-popover border border-border shadow-md rounded-xl p-2.5 text-xs text-popover-foreground space-y-1">
                        <p className="font-mono text-muted-foreground font-semibold">{label}</p>
                        <p className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-medium">
                          <CheckCircle2 className="w-3 h-3" /> Granted: {payload[0]?.value || 0}
                        </p>
                        <p className="flex items-center gap-1.5 text-destructive font-medium">
                          <XCircle className="w-3 h-3" /> Denied: {payload[1]?.value || 0}
                        </p>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Area
                type="monotone"
                dataKey="granted"
                stroke="#10b981"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorGranted)"
                name="Granted"
              />
              <Area
                type="monotone"
                dataKey="denied"
                stroke="#ef4444"
                strokeWidth={2}
                fillOpacity={1}
                fill="url(#colorDenied)"
                name="Denied"
              />
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground animate-pulse">
            Loading timeline...
          </div>
        )}
      </div>
    ) : (
        <div className="space-y-3 py-4">
          <div className="flex justify-between text-xs text-muted-foreground font-medium">
            <span className="flex items-center gap-1.5 text-emerald-600 dark:text-emerald-400 font-semibold">
              <CheckCircle2 className="w-3.5 h-3.5 shrink-0" /> Granted ({grantedPct.toFixed(1)}%)
            </span>
            <span className="flex items-center gap-1.5 text-destructive font-semibold">
              <XCircle className="w-3.5 h-3.5 shrink-0" /> Denied ({deniedPct.toFixed(1)}%)
            </span>
          </div>

          <div className="w-full h-4 bg-muted rounded-full overflow-hidden flex shadow-inner">
            <div
              style={{ width: `${grantedPct}%` }}
              className="h-full bg-emerald-500 transition-all duration-700"
            />
            <div
              style={{ width: `${deniedPct}%` }}
              className="h-full bg-destructive transition-all duration-700"
            />
          </div>
        </div>
      )}

      {/* Grid Summary Comparison */}
      <div className="grid grid-cols-2 gap-2.5 sm:gap-4 pt-1">
        <div className="p-3 bg-muted/30 hover:bg-muted/50 transition-colors rounded-xl border border-border/80">
          <p className="text-[10px] text-muted-foreground uppercase font-semibold tracking-wider">Today Granted</p>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-xl font-bold font-mono text-emerald-600 dark:text-emerald-400">
              {todayGranted}
            </span>
            <span className="text-xs text-muted-foreground font-mono">/ {todayTotal} total</span>
          </div>
        </div>

        <div className="p-3 bg-muted/30 hover:bg-muted/50 transition-colors rounded-xl border border-border/80">
          <p className="text-[10px] text-muted-foreground uppercase font-semibold tracking-wider">Today Denied</p>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-xl font-bold font-mono text-destructive">
              {todayDenied}
            </span>
            <span className="text-xs text-muted-foreground font-mono">/ {todayTotal} total</span>
          </div>
        </div>
      </div>

      {/* All-Time Historical Breakdown */}
      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/70 text-xs">
        <div className="flex items-center justify-between p-2 bg-muted/20 rounded-lg">
          <span className="text-muted-foreground text-[11px]">All-Time</span>
          <span className="font-mono font-bold text-foreground">{total}</span>
        </div>

        <div className="flex items-center justify-between p-2 bg-muted/20 rounded-lg">
          <span className="text-emerald-600 dark:text-emerald-400 text-[11px] font-semibold">Granted</span>
          <span className="font-mono font-bold text-foreground">{granted}</span>
        </div>

        <div className="flex items-center justify-between p-2 bg-muted/20 rounded-lg">
          <span className="text-destructive text-[11px] font-semibold">Denied</span>
          <span className="font-mono font-bold text-foreground">{denied}</span>
        </div>
      </div>
    </div>
  );
}
