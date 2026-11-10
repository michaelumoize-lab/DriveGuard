"use client";

import React, { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  Search,
  Eye,
  Clock,
  Activity,
  ShieldAlert,
  CheckCircle2,
  Gauge,
} from "lucide-react";
import { explainReasonCode } from "@/types/telemetry";

export interface SerializedAlertItem {
  id: string;
  createdAt: string;
  severity: string;
  riskScore: number;
  earValue: number;
  perclosPct: number;
  poseZone: string;
  reasons: string[];
  cascadeDecision: string;
}

export interface SerializedDrivingSession {
  id: string;
  sessionCode: string;
  startedAt: string;
  endedAt: string | null;
  durationSeconds: number;
  averageFps: number | null;
  averageLatencyMs: number | null;
  averagePerclosPct: number | null;
  maxPerclosPct: number | null;
  totalBlinks: number;
  averageBlinkRate: number | null;
  normalFramesCount: number;
  warningAlertsCount: number;
  criticalAlertsCount: number;
  mirrorSuppressions: number;
  alerts?: SerializedAlertItem[];
}

interface SessionsTableInteractiveProps {
  sessions: SerializedDrivingSession[];
}

export function SessionsTableInteractive({ sessions }: SessionsTableInteractiveProps) {
  const [searchQuery, setSearchQuery] = useState("");
  const [statusFilter, setStatusFilter] = useState<"ALL" | "CLEAN" | "WITH_ALERTS">("ALL");
  const [selectedSession, setSelectedSession] = useState<SerializedDrivingSession | null>(null);

  const filteredSessions = useMemo(() => {
    return sessions.filter((s) => {
      const matchesQuery =
        !searchQuery.trim() ||
        s.sessionCode.toLowerCase().includes(searchQuery.trim().toLowerCase());
      const hasAlerts = s.criticalAlertsCount > 0 || s.warningAlertsCount > 0;
      const matchesStatus =
        statusFilter === "ALL" ||
        (statusFilter === "CLEAN" && !hasAlerts) ||
        (statusFilter === "WITH_ALERTS" && hasAlerts);
      return matchesQuery && matchesStatus;
    });
  }, [sessions, searchQuery, statusFilter]);

  return (
    <div className="space-y-4">
      {/* Filter & Search Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="relative max-w-xs w-full">
          <Search className="size-3.5 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
          <Input
            type="search"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Filter by session code..."
            aria-label="Filter sessions by session code"
            className="pl-8 h-8 text-xs font-mono"
          />
        </div>

        <div className="flex items-center gap-1.5" role="group" aria-label="Filter sessions by alert status">
          {[
            { id: "ALL" as const, label: `All (${sessions.length})` },
            {
              id: "CLEAN" as const,
              label: `Clean (${sessions.filter((s) => s.criticalAlertsCount === 0 && s.warningAlertsCount === 0).length})`,
            },
            {
              id: "WITH_ALERTS" as const,
              label: `With Alerts (${sessions.filter((s) => s.criticalAlertsCount > 0 || s.warningAlertsCount > 0).length})`,
            },
          ].map((tab) => (
            <Button
              key={tab.id}
              type="button"
              variant={statusFilter === tab.id ? "default" : "outline"}
              size="sm"
              onClick={() => setStatusFilter(tab.id)}
              className="h-7 text-[11px] px-2.5"
            >
              {tab.label}
            </Button>
          ))}
        </div>
      </div>

      {filteredSessions.length === 0 ? (
        <div className="text-center py-10 text-muted-foreground text-xs border border-dashed border-border rounded-lg">
          No driving sessions match the active filter criteria.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border/60">
          <table className="w-full text-xs text-left">
            <caption className="sr-only">
              Historical log of DriveGuard driving sessions and fatigue telemetry summaries
            </caption>
            <thead className="border-b border-border bg-muted/30 text-muted-foreground">
              <tr>
                <th scope="col" className="py-2.5 px-3 font-medium">Session Code</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Date &amp; Time</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Duration</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Avg PERCLOS</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Max PERCLOS</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Blinks</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Safety Alerts</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Avg FPS</th>
                <th scope="col" className="py-2.5 px-3 font-medium text-right">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filteredSessions.map((s) => (
                <tr key={s.id} className="hover:bg-muted/30 transition-colors">
                  <td className="py-2.5 px-3 font-mono font-medium text-foreground">
                    {s.sessionCode}
                  </td>
                  <td className="py-2.5 px-3 text-muted-foreground whitespace-nowrap">
                    {new Date(s.startedAt).toLocaleDateString()}{" "}
                    <span className="font-mono text-[11px]">
                      {new Date(s.startedAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}
                    </span>
                  </td>
                  <td className="py-2.5 px-3 font-mono whitespace-nowrap">
                    {Math.floor(s.durationSeconds / 60)}m {s.durationSeconds % 60}s
                  </td>
                  <td className="py-2.5 px-3 font-mono">
                    {s.averagePerclosPct !== null && s.averagePerclosPct !== undefined
                      ? `${s.averagePerclosPct.toFixed(1)}%`
                      : "—"}
                  </td>
                  <td className="py-2.5 px-3 font-mono">
                    {s.maxPerclosPct !== null && s.maxPerclosPct !== undefined
                      ? `${s.maxPerclosPct.toFixed(1)}%`
                      : "—"}
                  </td>
                  <td className="py-2.5 px-3 font-mono">{s.totalBlinks}</td>
                  <td className="py-2.5 px-3">
                    <div className="flex items-center gap-1.5">
                      {s.criticalAlertsCount > 0 ? (
                        <Badge variant="destructive" className="text-[10px] font-mono">
                          {s.criticalAlertsCount} Crit
                        </Badge>
                      ) : null}
                      {s.warningAlertsCount > 0 ? (
                        <Badge
                          variant="outline"
                          className="text-[10px] font-mono text-amber-500 border-amber-500/30"
                        >
                          {s.warningAlertsCount} Warn
                        </Badge>
                      ) : null}
                      {s.criticalAlertsCount === 0 && s.warningAlertsCount === 0 && (
                        <Badge
                          variant="outline"
                          className="text-[10px] text-emerald-500 border-emerald-500/30"
                        >
                          0 (Clean)
                        </Badge>
                      )}
                    </div>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-muted-foreground">
                    {s.averageFps && s.averageFps > 0 ? `${s.averageFps.toFixed(1)} fps` : "—"}
                  </td>
                  <td className="py-2.5 px-3 text-right">
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      onClick={() => setSelectedSession(s)}
                      className="h-7 px-2 text-xs"
                    >
                      <Eye className="size-3.5 mr-1" />
                      Inspect
                    </Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Session Detail Modal / Drawer */}
      <Dialog open={Boolean(selectedSession)} onOpenChange={(open) => !open && setSelectedSession(null)}>
        <DialogContent className="max-w-2xl max-h-[85vh] overflow-y-auto overscroll-contain">
          {selectedSession && (
            <>
              <DialogHeader>
                <DialogTitle className="flex items-center gap-2 font-mono text-base">
                  <Activity className="size-4 text-primary" />
                  Session Report: {selectedSession.sessionCode}
                </DialogTitle>
                <DialogDescription className="text-xs">
                  Started {new Date(selectedSession.startedAt).toLocaleString()}
                  {selectedSession.endedAt
                    ? ` • Ended ${new Date(selectedSession.endedAt).toLocaleTimeString()}`
                    : " • Ended unexpectedly / active"}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-4 pt-2">
                {/* Summary KPI Grid */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2.5">
                  <div className="p-3 rounded-lg border border-border bg-muted/20">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                      <Clock className="size-3" /> Duration
                    </span>
                    <div className="text-sm font-bold font-mono mt-1">
                      {Math.floor(selectedSession.durationSeconds / 60)}m {selectedSession.durationSeconds % 60}s
                    </div>
                  </div>
                  <div className="p-3 rounded-lg border border-border bg-muted/20">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      Avg / Peak PERCLOS
                    </span>
                    <div className="text-sm font-bold font-mono mt-1">
                      {selectedSession.averagePerclosPct !== null
                        ? `${selectedSession.averagePerclosPct.toFixed(1)}%`
                        : "—"}{" "}
                      /{" "}
                      {selectedSession.maxPerclosPct !== null
                        ? `${selectedSession.maxPerclosPct.toFixed(1)}%`
                        : "—"}
                    </div>
                  </div>
                  <div className="p-3 rounded-lg border border-border bg-muted/20">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground">
                      Total Blinks (Rate)
                    </span>
                    <div className="text-sm font-bold font-mono mt-1">
                      {selectedSession.totalBlinks}{" "}
                      <span className="text-xs text-muted-foreground font-normal">
                        ({selectedSession.averageBlinkRate ? `${selectedSession.averageBlinkRate.toFixed(1)}/m` : "—"})
                      </span>
                    </div>
                  </div>
                  <div className="p-3 rounded-lg border border-border bg-muted/20">
                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground flex items-center gap-1">
                      <Gauge className="size-3" /> Transport Quality
                    </span>
                    <div className="text-sm font-bold font-mono mt-1">
                      {selectedSession.averageFps && selectedSession.averageFps > 0
                        ? `${selectedSession.averageFps.toFixed(1)} FPS`
                        : "—"}
                      {selectedSession.averageLatencyMs
                        ? ` (${selectedSession.averageLatencyMs.toFixed(0)}ms)`
                        : ""}
                    </div>
                  </div>
                </div>

                {/* Frame & Suppression Breakdown */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs p-3 rounded-lg border border-border bg-background">
                  <div>
                    <span className="text-muted-foreground block text-[11px]">Normal Frames</span>
                    <span className="font-mono font-semibold text-emerald-500">
                      {selectedSession.normalFramesCount}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[11px]">Warning Transitions</span>
                    <span className="font-mono font-semibold text-amber-500">
                      {selectedSession.warningAlertsCount}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[11px]">Critical Interventions</span>
                    <span className="font-mono font-semibold text-rose-500">
                      {selectedSession.criticalAlertsCount}
                    </span>
                  </div>
                  <div>
                    <span className="text-muted-foreground block text-[11px]">Mirror Suppressions</span>
                    <span className="font-mono font-semibold text-primary">
                      {selectedSession.mirrorSuppressions}
                    </span>
                  </div>
                </div>

                {/* Session Alerts Timeline */}
                <div className="space-y-2">
                  <h4 className="text-xs font-semibold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                    <ShieldAlert className="size-3.5 text-primary" />
                    Recorded Safety Events ({selectedSession.alerts?.length ?? 0})
                  </h4>

                  {!selectedSession.alerts || selectedSession.alerts.length === 0 ? (
                    <div className="p-4 rounded-lg border border-emerald-500/20 bg-emerald-500/5 text-xs text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
                      <CheckCircle2 className="size-4 shrink-0" />
                      Zero fatigue or drowsiness interventions occurred during this session.
                    </div>
                  ) : (
                    <div className="space-y-2 max-h-60 overflow-y-auto pr-1">
                      {selectedSession.alerts.map((alert) => (
                        <div
                          key={alert.id}
                          className="p-2.5 rounded-lg border border-border bg-muted/15 text-xs space-y-1.5"
                        >
                          <div className="flex items-center justify-between">
                            <div className="flex items-center gap-2">
                              <Badge
                                variant={alert.severity === "CRITICAL" ? "destructive" : "outline"}
                                className="text-[10px] font-mono"
                              >
                                {alert.severity}
                              </Badge>
                              <span className="font-mono text-[11px] text-muted-foreground">
                                {new Date(alert.createdAt).toLocaleTimeString()}
                              </span>
                              <Badge variant="outline" className="text-[10px] font-mono">
                                {alert.poseZone}
                              </Badge>
                            </div>
                            <span className="font-mono text-[11px]">
                              Risk {(alert.riskScore * 100).toFixed(0)}% • EAR {alert.earValue.toFixed(3)} • PERCLOS{" "}
                              {alert.perclosPct.toFixed(1)}%
                            </span>
                          </div>
                          {alert.reasons.length > 0 && (
                            <div className="flex flex-wrap gap-1 pt-0.5">
                              {alert.reasons.map((code) => {
                                const info = explainReasonCode(code);
                                return (
                                  <span
                                    key={code}
                                    title={info.explanation}
                                    className="px-1.5 py-0.5 rounded bg-background border border-border text-[10px] font-mono"
                                  >
                                    {info.title} ({code})
                                  </span>
                                );
                              })}
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
