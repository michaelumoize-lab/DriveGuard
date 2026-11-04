"use client";

import React from "react";
import { PieChart, Pie, Cell, ResponsiveContainer, Tooltip } from "recharts";
import { Cpu, CheckCircle, HelpCircle, AlertTriangle, Sparkles, Clock, Zap } from "lucide-react";

interface RecognitionBreakdownProps {
  matched: number;
  unknown: number;
  ambiguous: number;
  avgQuality?: number;
  avgSimilarity?: number;
  avgLatencySec?: number;
}

export function RecognitionBreakdown({
  matched,
  unknown,
  ambiguous,
  avgQuality = 0,
  avgSimilarity = 0,
  avgLatencySec = 0,
}: RecognitionBreakdownProps) {
  const [isMounted, setIsMounted] = React.useState<boolean>(false);

  React.useEffect(() => {
    setIsMounted(true);
  }, []);

  const total = matched + unknown + ambiguous;

  const matchedPct = total > 0 ? (matched / total) * 100 : 0;
  const unknownPct = total > 0 ? (unknown / total) * 100 : 0;
  const ambiguousPct = total > 0 ? (ambiguous / total) * 100 : 0;

  const donutData = [
    { name: "Matched", value: matched, color: "#10b981" },
    { name: "Unknown", value: unknown, color: "#64748b" },
    { name: "Ambiguous", value: ambiguous, color: "#f59e0b" },
  ].filter((item) => item.value > 0);

  // Fallback placeholder if no records
  const displayDonutData =
    donutData.length > 0
      ? donutData
      : [{ name: "No Data", value: 1, color: "var(--muted)" }];

  return (
    <div className="bg-card text-card-foreground border border-border rounded-2xl p-4 sm:p-6 shadow-sm flex flex-col justify-between space-y-4 sm:space-y-5 h-full min-w-0">
      {/* Header */}
      <div className="flex items-center justify-between flex-wrap gap-2 pb-2 border-b border-border/60">
        <div className="flex items-center gap-2.5">
          <div className="p-2 bg-primary/10 border border-primary/20 text-primary rounded-xl">
            <Cpu className="w-4 h-4" />
          </div>
          <div>
            <h3 className="text-sm font-semibold text-foreground tracking-tight">Recognition Distribution</h3>
            <p className="text-[11px] text-muted-foreground font-mono">Biometric Match Analytics</p>
          </div>
        </div>
        <span className="text-xs font-mono font-medium px-2 py-0.5 rounded-full bg-muted text-muted-foreground border border-border/60">
          {total} Attempts
        </span>
      </div>

      {/* Donut Chart & Distribution Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-12 gap-4 items-center">
        {/* Interactive Donut Chart */}
        <div className="sm:col-span-5 w-full h-36 relative flex items-center justify-center min-w-0">
          {isMounted ? (
            <ResponsiveContainer
              width="100%"
              height="100%"
              minWidth={0}
              minHeight={0}
              initialDimension={{ width: 160, height: 144 }}
            >
              <PieChart>
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const data = payload[0];
                      return (
                        <div className="bg-popover border border-border shadow-md rounded-xl px-2.5 py-1.5 text-xs text-popover-foreground">
                          <span className="font-semibold" style={{ color: data.payload.color }}>
                            {data.name}:
                          </span>{" "}
                          <span className="font-mono font-bold">{data.value}</span>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Pie
                  data={displayDonutData}
                  innerRadius={38}
                  outerRadius={56}
                  paddingAngle={4}
                  dataKey="value"
                  stroke="none"
                >
                  {displayDonutData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
              </PieChart>
            </ResponsiveContainer>
          ) : (
            <div className="w-full h-full flex items-center justify-center text-xs text-muted-foreground animate-pulse">
              Loading distribution...
            </div>
          )}

          {/* Center Donut Label */}
          <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
            <span className="text-lg font-bold font-mono tracking-tight text-foreground">{total}</span>
            <span className="text-[10px] text-muted-foreground font-medium uppercase tracking-wider">Total</span>
          </div>
        </div>

        {/* Legend Cards */}
        <div className="sm:col-span-7 flex flex-col gap-2">
          <div className="flex items-center justify-between p-2 rounded-xl bg-emerald-500/5 border border-emerald-500/15">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-emerald-600 dark:text-emerald-400">
              <CheckCircle className="w-3.5 h-3.5 shrink-0" /> Matched
            </span>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono font-bold text-sm text-foreground">{matched}</span>
              <span className="text-[10px] text-muted-foreground font-mono">({matchedPct.toFixed(1)}%)</span>
            </div>
          </div>

          <div className="flex items-center justify-between p-2 rounded-xl bg-muted/30 border border-border/60">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-muted-foreground">
              <HelpCircle className="w-3.5 h-3.5 shrink-0" /> Unknown
            </span>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono font-bold text-sm text-foreground">{unknown}</span>
              <span className="text-[10px] text-muted-foreground font-mono">({unknownPct.toFixed(1)}%)</span>
            </div>
          </div>

          <div className="flex items-center justify-between p-2 rounded-xl bg-amber-500/5 border border-amber-500/15">
            <span className="flex items-center gap-1.5 text-xs font-semibold text-amber-600 dark:text-amber-400">
              <AlertTriangle className="w-3.5 h-3.5 shrink-0" /> Ambiguous
            </span>
            <div className="flex items-baseline gap-1.5">
              <span className="font-mono font-bold text-sm text-foreground">{ambiguous}</span>
              <span className="text-[10px] text-muted-foreground font-mono">({ambiguousPct.toFixed(1)}%)</span>
            </div>
          </div>
        </div>
      </div>

      {/* Telemetry Radial Gauge Summaries */}
      <div className="grid grid-cols-3 gap-2 pt-2 border-t border-border/70">
        {/* Quality Gauge */}
        <div className="p-2 sm:p-2.5 bg-muted/20 hover:bg-muted/40 transition-colors rounded-xl border border-border/60 text-center flex flex-col items-center">
          <div className="flex items-center gap-1 text-[11px] text-muted-foreground mb-1 font-medium">
            <Sparkles className="w-3 h-3 text-primary shrink-0" /> Quality
          </div>
          <span className="text-base font-bold font-mono text-foreground">{avgQuality.toFixed(1)}%</span>
          <div className="w-full bg-muted h-1 rounded-full mt-1.5 overflow-hidden">
            <div
              style={{ width: `${Math.min(100, avgQuality)}%` }}
              className="bg-primary h-full transition-all duration-500"
            />
          </div>
        </div>

        {/* Similarity Gauge */}
        <div className="p-2 sm:p-2.5 bg-muted/20 hover:bg-muted/40 transition-colors rounded-xl border border-border/60 text-center flex flex-col items-center">
          <div className="flex items-center gap-1 text-[11px] text-muted-foreground mb-1 font-medium">
            <Zap className="w-3 h-3 text-emerald-500 shrink-0" /> Similarity
          </div>
          <span className="text-base font-bold font-mono text-emerald-600 dark:text-emerald-400">
            {avgSimilarity.toFixed(1)}%
          </span>
          <div className="w-full bg-muted h-1 rounded-full mt-1.5 overflow-hidden">
            <div
              style={{ width: `${Math.min(100, avgSimilarity)}%` }}
              className="bg-emerald-500 h-full transition-all duration-500"
            />
          </div>
        </div>

        {/* Latency Gauge */}
        <div className="p-2 sm:p-2.5 bg-muted/20 hover:bg-muted/40 transition-colors rounded-xl border border-border/60 text-center flex flex-col items-center">
          <div className="flex items-center gap-1 text-[11px] text-muted-foreground mb-1 font-medium">
            <Clock className="w-3 h-3 text-blue-500 shrink-0" /> Latency
          </div>
          <span className="text-base font-bold font-mono text-foreground">
            {avgLatencySec.toFixed(2)}s
          </span>
          <div className="w-full bg-muted h-1 rounded-full mt-1.5 overflow-hidden">
            <div
              style={{ width: `${Math.min(100, (avgLatencySec / 1.0) * 100)}%` }}
              className="bg-blue-500 h-full transition-all duration-500"
            />
          </div>
        </div>
      </div>
    </div>
  );
}
