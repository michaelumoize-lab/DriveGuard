"use client";

import React, { useMemo, useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Search } from "lucide-react";
import { explainReasonCode } from "@/types/telemetry";
import type { SerializedAlertItem } from "@/components/dashboard/sessions-table-interactive";

interface AlertsTableInteractiveProps {
  alerts: SerializedAlertItem[];
  initialSeverity?: string;
}

export function AlertsTableInteractive({
  alerts,
  initialSeverity,
}: AlertsTableInteractiveProps) {
  const [severityFilter, setSeverityFilter] = useState<"ALL" | "CRITICAL" | "WARNING">(
    initialSeverity === "CRITICAL" || initialSeverity === "WARNING" ? initialSeverity : "ALL"
  );
  const [zoneFilter, setZoneFilter] = useState<string>("ALL");
  const [searchQuery, setSearchQuery] = useState("");

  const filteredAlerts = useMemo(() => {
    return alerts.filter((a) => {
      const matchesSeverity =
        severityFilter === "ALL" || a.severity.toUpperCase() === severityFilter;
      const matchesZone = zoneFilter === "ALL" || a.poseZone === zoneFilter;
      const q = searchQuery.trim().toLowerCase();
      const matchesSearch =
        !q ||
        a.reasons.some(
          (r) =>
            r.toLowerCase().includes(q) ||
            explainReasonCode(r).title.toLowerCase().includes(q)
        ) ||
        a.cascadeDecision.toLowerCase().includes(q) ||
        a.poseZone.toLowerCase().includes(q);
      return matchesSeverity && matchesZone && matchesSearch;
    });
  }, [alerts, severityFilter, zoneFilter, searchQuery]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
        <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="Filter alerts by severity">
          {[
            { id: "ALL" as const, label: `All (${alerts.length})` },
            {
              id: "CRITICAL" as const,
              label: `Critical (${alerts.filter((a) => a.severity === "CRITICAL").length})`,
            },
            {
              id: "WARNING" as const,
              label: `Warning (${alerts.filter((a) => a.severity === "WARNING").length})`,
            },
          ].map((tab) => (
            <Button
              key={tab.id}
              type="button"
              variant={severityFilter === tab.id ? "default" : "outline"}
              size="sm"
              onClick={() => setSeverityFilter(tab.id)}
              className="h-7 text-[11px] px-2.5"
            >
              {tab.label}
            </Button>
          ))}
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            aria-label="Filter alerts by head-pose zone"
            value={zoneFilter}
            onChange={(e) => setZoneFilter(e.target.value)}
            className="h-8 rounded-md border border-input bg-background px-2.5 text-xs font-mono text-foreground"
          >
            <option value="ALL">All Gaze Zones</option>
            <option value="FRONT">FRONT</option>
            <option value="LEFT_MIRROR">LEFT_MIRROR</option>
            <option value="RIGHT_MIRROR">RIGHT_MIRROR</option>
            <option value="REARVIEW_MIRROR">REARVIEW_MIRROR</option>
          </select>

          <div className="relative w-56">
            <Search className="size-3.5 text-muted-foreground absolute left-2.5 top-1/2 -translate-y-1/2 pointer-events-none" />
            <Input
              type="search"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Filter reason or authority..."
              aria-label="Filter alerts by reason code or cascade authority"
              className="pl-8 h-8 text-xs"
            />
          </div>
        </div>
      </div>

      {filteredAlerts.length === 0 ? (
        <div className="text-center py-10 text-muted-foreground text-xs border border-dashed border-border rounded-lg">
          No safety alert events match the active filters.
        </div>
      ) : (
        <div className="overflow-x-auto rounded-lg border border-border/60">
          <table className="w-full text-xs text-left">
            <caption className="sr-only">
              Auditable log of DriveGuard fatigue and drowsiness alert interventions
            </caption>
            <thead className="border-b border-border bg-muted/30 text-muted-foreground">
              <tr>
                <th scope="col" className="py-2.5 px-3 font-medium">Timestamp</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Severity</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Risk Score</th>
                <th scope="col" className="py-2.5 px-3 font-medium">EAR</th>
                <th scope="col" className="py-2.5 px-3 font-medium">PERCLOS (60s)</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Gaze Zone</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Explainable Reasons</th>
                <th scope="col" className="py-2.5 px-3 font-medium">Cascade Authority</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-border">
              {filteredAlerts.map((a) => (
                <tr key={a.id} className="hover:bg-muted/30 transition-colors">
                  <td className="py-2.5 px-3 font-mono text-muted-foreground whitespace-nowrap">
                    {new Date(a.createdAt).toLocaleDateString()}{" "}
                    {new Date(a.createdAt).toLocaleTimeString()}
                  </td>
                  <td className="py-2.5 px-3">
                    <Badge
                      variant={a.severity === "CRITICAL" ? "destructive" : "outline"}
                      className={
                        a.severity === "CRITICAL"
                          ? "text-[10px] font-mono"
                          : "text-[10px] font-mono text-amber-500 border-amber-500/30"
                      }
                    >
                      {a.severity}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-3 font-mono font-semibold">
                    {(a.riskScore * 100).toFixed(0)}%
                  </td>
                  <td className="py-2.5 px-3 font-mono">{a.earValue.toFixed(3)}</td>
                  <td className="py-2.5 px-3 font-mono">{a.perclosPct.toFixed(1)}%</td>
                  <td className="py-2.5 px-3">
                    <Badge variant="outline" className="text-[10px] font-mono">
                      {a.poseZone}
                    </Badge>
                  </td>
                  <td className="py-2.5 px-3">
                    <div className="flex flex-wrap gap-1">
                      {a.reasons.length === 0 ? (
                        <span className="text-muted-foreground">—</span>
                      ) : (
                        a.reasons.map((code) => {
                          const info = explainReasonCode(code);
                          return (
                            <span
                              key={code}
                              title={info.explanation}
                              className="inline-flex items-center gap-1 rounded border border-border bg-muted/40 px-1.5 py-0.5 text-[10px] font-mono text-foreground"
                            >
                              {info.title}
                            </span>
                          );
                        })
                      )}
                    </div>
                  </td>
                  <td className="py-2.5 px-3 font-mono text-muted-foreground">
                    {a.cascadeDecision}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
