import React from "react";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ShieldAlert } from "lucide-react";
import { AlertsTableInteractive } from "@/components/dashboard/alerts-table-interactive";
import { DataExportButton } from "@/components/dashboard/data-export-button";
import type { SerializedAlertItem } from "@/components/dashboard/sessions-table-interactive";

export const dynamic = "force-dynamic";

interface AlertsPageProps {
  searchParams: Promise<{
    severity?: string;
  }>;
}

export default async function AlertsPage({ searchParams }: AlertsPageProps) {
  const session = await getServerSession();
  const userId = session?.user?.id;
  const params = await searchParams;
  const severityFilter = params.severity?.toUpperCase();

  const rawAlerts = userId
    ? await prisma.drowsinessAlert.findMany({
        where: {
          session: { userId },
        },
        orderBy: { createdAt: "desc" },
        take: 100,
      })
    : [];

  const alerts: SerializedAlertItem[] = rawAlerts.map((a) => ({
    id: a.id,
    createdAt: a.createdAt.toISOString(),
    severity: a.severity,
    riskScore: a.riskScore,
    earValue: a.earValue,
    perclosPct: a.perclosPct,
    poseZone: a.poseZone,
    reasons: a.reasons,
    cascadeDecision: a.cascadeDecision,
  }));

  const criticalCount = alerts.filter((a) => a.severity === "CRITICAL").length;
  const warningCount = alerts.filter((a) => a.severity === "WARNING").length;

  const exportRecords = alerts.map((a) => ({
    createdAt: a.createdAt,
    severity: a.severity,
    riskScorePct: (a.riskScore * 100).toFixed(1),
    earValue: a.earValue.toFixed(3),
    perclosPct: a.perclosPct.toFixed(1),
    poseZone: a.poseZone,
    reasons: a.reasons.join("; "),
    cascadeDecision: a.cascadeDecision,
  }));

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <ShieldAlert className="size-6 text-primary" />
            Fatigue &amp; Drowsiness Alerts
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Auditable log of explainable safety alerts, micro-sleep flags, and kinematic thresholds.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <Badge variant="outline" className="text-xs font-mono text-amber-500 border-amber-500/30">
            {warningCount} Warnings
          </Badge>
          <Badge variant="destructive" className="text-xs font-mono">
            {criticalCount} Critical
          </Badge>
          <DataExportButton
            data={exportRecords}
            filenamePrefix="driveguard_alerts"
          />
        </div>
      </div>

      <Card className="border-border/80">
        <CardHeader>
          <CardTitle className="text-lg">Recorded Alert Events</CardTitle>
          <CardDescription>
            Showing safety interventions detected during your driving sessions. Hover any reason badge for its kinematic threshold explanation.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {alerts.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-xs">
              No drowsiness alerts recorded. The driver has maintained alert state!
            </div>
          ) : (
            <AlertsTableInteractive
              alerts={alerts}
              initialSeverity={severityFilter}
            />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
