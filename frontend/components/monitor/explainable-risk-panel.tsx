"use client";

import React from "react";
import { FrameTelemetry, LatencyDiagnostics, explainReasonCode } from "@/types/telemetry";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ShieldCheck, AlertTriangle, Flame, Clock, Cpu, Info } from "lucide-react";

interface ExplainableRiskPanelProps {
  telemetry: FrameTelemetry | null;
  latencyDiagnostics: LatencyDiagnostics | null;
}

export const ExplainableRiskPanel = React.memo(function ExplainableRiskPanel({
  telemetry,
  latencyDiagnostics,
}: ExplainableRiskPanelProps) {
  const reasons = telemetry?.reason_codes ?? [];
  const explained = reasons.map(explainReasonCode);
  const state = telemetry?.state ?? "NORMAL";

  // Separate primary fatigue/safety reasons from passive observation/quality notes when state is NORMAL
  const primaryReasons =
    state === "NORMAL"
      ? explained.filter((r) => r.subsystem !== "QUALITY" && r.severity === "CRITICAL")
      : explained.filter((r) => r.subsystem !== "QUALITY");
  const qualityNotes =
    state === "NORMAL"
      ? explained.filter((r) => !primaryReasons.includes(r))
      : explained.filter((r) => r.subsystem === "QUALITY");

  const queueMs =
    latencyDiagnostics?.queue_latency_ms ?? telemetry?.timing?.queue_latency_ms ?? 0;
  const inferMs =
    latencyDiagnostics?.infer_duration_ms ??
    telemetry?.timing?.infer_duration_ms ??
    telemetry?.inference_latency_ms ??
    0;
  const cnnMs = telemetry?.cascade?.cnn_latency_ms ?? 0;
  const rttMs =
    latencyDiagnostics?.datachannel_rtt_ms ?? telemetry?.network_roundtrip_ms ?? 0;

  return (
    <Card className="border-border/80 bg-card/60">
      <CardHeader className="py-3 px-4 border-b border-border/60 flex flex-row items-center justify-between space-y-0">
        <CardTitle className="text-xs font-mono uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
          <Cpu className="size-3.5 text-primary" />
          Explainable Risk Rationale &amp; Stage Timings
        </CardTitle>
        <Badge variant="outline" className="text-[10px] font-mono tabular-nums">
          Risk: {telemetry ? `${(telemetry.risk_score * 100).toFixed(0)}%` : "—"}
        </Badge>
      </CardHeader>
      <CardContent className="p-3.5 space-y-3">
        {/* Active Explainable Reasons */}
        {!telemetry ? (
          <div className="text-xs text-muted-foreground py-2">
            Awaiting live telemetry stream to evaluate physiological risk factors.
          </div>
        ) : primaryReasons.length === 0 ? (
          <div className="space-y-2">
            <div className="p-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/5 flex items-center gap-2.5 text-xs">
              <ShieldCheck className="size-4 text-emerald-500 shrink-0" />
              <div>
                <div className="font-semibold text-foreground">All Physiological Signals Nominal</div>
                <div className="text-muted-foreground text-[11px]">
                  EAR, blink reopening velocity (RCR), 60s PERCLOS, and head pitch are within safe bounds.
                </div>
              </div>
            </div>

            {qualityNotes.length > 0 && (
              <div className="px-2.5 py-1.5 rounded-lg bg-muted/40 border border-border/60 flex items-center gap-2 flex-wrap text-[11px] text-muted-foreground">
                <Info className="size-3.5 text-primary shrink-0" />
                <span className="font-medium text-foreground">Diagnostic Notes:</span>
                {qualityNotes.map((note) => (
                  <span
                    key={note.code}
                    title={note.description}
                    className="px-1.5 py-0.5 rounded bg-background border border-border text-[10px] font-mono"
                  >
                    {note.title}
                  </span>
                ))}
              </div>
            )}
          </div>
        ) : (
          <div className="space-y-2">
            {primaryReasons.map((item) => (
              <div
                key={item.code}
                className={`p-2.5 rounded-lg border text-xs flex items-start gap-2.5 ${
                  item.severity === "CRITICAL"
                    ? "border-rose-500/40 bg-rose-500/10"
                    : item.severity === "WARNING"
                      ? "border-amber-500/40 bg-amber-500/10"
                      : "border-primary/30 bg-primary/5"
                }`}
              >
                {item.severity === "CRITICAL" ? (
                  <Flame className="size-4 text-rose-500 shrink-0 mt-0.5" />
                ) : item.severity === "WARNING" ? (
                  <AlertTriangle className="size-4 text-amber-500 shrink-0 mt-0.5" />
                ) : (
                  <ShieldCheck className="size-4 text-primary shrink-0 mt-0.5" />
                )}
                <div className="min-w-0 flex-1 space-y-0.5">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold text-foreground">{item.title}</span>
                    <Badge variant="outline" className="text-[10px] font-mono shrink-0">
                      {item.subsystem}
                    </Badge>
                  </div>
                  <p className="text-[11px] text-muted-foreground leading-snug">
                    {item.description}
                  </p>
                </div>
              </div>
            ))}

            {qualityNotes.length > 0 && (
              <div className="px-2.5 py-1.5 rounded-lg bg-muted/40 border border-border/60 flex items-center gap-2 flex-wrap text-[11px] text-muted-foreground">
                <Info className="size-3.5 text-primary shrink-0" />
                {qualityNotes.map((note) => (
                  <span
                    key={note.code}
                    title={note.description}
                    className="px-1.5 py-0.5 rounded bg-background border border-border text-[10px] font-mono"
                  >
                    {note.title}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {/* Pipeline Stage Timing Breakdown */}
        <div className="pt-2 border-t border-border/60">
          <div className="flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground mb-2">
            <Clock className="size-3 text-primary" />
            <span>PIPELINE STAGE INSTRUMENTATION (MS)</span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono tabular-nums">
            <div className="p-2 rounded-lg bg-muted/40 border border-border/60">
              <span className="text-[10px] text-muted-foreground block">Ingest Queue</span>
              <span className="font-semibold text-foreground">{queueMs.toFixed(1)} ms</span>
            </div>
            <div className="p-2 rounded-lg bg-muted/40 border border-border/60">
              <span className="text-[10px] text-muted-foreground block">Total AI Infer</span>
              <span className="font-semibold text-foreground">{inferMs.toFixed(1)} ms</span>
            </div>
            <div className="p-2 rounded-lg bg-muted/40 border border-border/60">
              <span className="text-[10px] text-muted-foreground block">CNN Stage 2</span>
              <span className="font-semibold text-foreground">
                {telemetry?.cascade?.cnn_triggered ? `${cnnMs.toFixed(1)} ms` : "Bypassed"}
              </span>
            </div>
            <div className="p-2 rounded-lg bg-muted/40 border border-border/60">
              <span className="text-[10px] text-muted-foreground block">WebRTC RTT</span>
              <span className="font-semibold text-foreground">
                {rttMs > 0 ? `${rttMs} ms` : "—"}
              </span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
});
