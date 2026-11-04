import React from "react";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ShieldAlert, AlertTriangle, Eye, Activity } from "lucide-react";

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

  const alerts = userId
    ? await prisma.drowsinessAlert.findMany({
        where: {
          session: { userId },
          ...(severityFilter && ["WARNING", "CRITICAL"].includes(severityFilter)
            ? { severity: severityFilter }
            : {}),
        },
        orderBy: { createdAt: "desc" },
        take: 50,
      })
    : [];

  const criticalCount = alerts.filter((a) => a.severity === "CRITICAL").length;
  const warningCount = alerts.filter((a) => a.severity === "WARNING").length;

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <ShieldAlert className="size-6 text-primary" />
            Fatigue & Drowsiness Alerts
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Auditable log of explainable safety alerts, micro-sleep flags, and kinematic thresholds.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-xs font-mono">
            {warningCount} Warnings
          </Badge>
          <Badge variant="destructive" className="text-xs font-mono">
            {criticalCount} Critical
          </Badge>
        </div>
      </div>

      <Card className="border-border/80">
        <CardHeader>
          <CardTitle className="text-lg">Recorded Alert Events</CardTitle>
          <CardDescription>
            Showing the latest safety interventions detected during your driving sessions.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {alerts.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-xs">
              No drowsiness alerts recorded. The driver has maintained alert state!
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="border-b border-border text-muted-foreground">
                  <tr>
                    <th className="py-2.5 px-3">Time</th>
                    <th className="py-2.5 px-3">Severity</th>
                    <th className="py-2.5 px-3">Risk</th>
                    <th className="py-2.5 px-3">EAR</th>
                    <th className="py-2.5 px-3">PERCLOS (60s)</th>
                    <th className="py-2.5 px-3">Gaze Zone</th>
                    <th className="py-2.5 px-3">Reasons</th>
                    <th className="py-2.5 px-3">Cascade Authority</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {alerts.map((a) => (
                    <tr key={a.id} className="hover:bg-muted/30">
                      <td className="py-2.5 px-3 font-mono text-muted-foreground">
                        {new Date(a.createdAt).toLocaleTimeString()}
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge
                          variant={a.severity === "CRITICAL" ? "destructive" : "outline"}
                          className="text-[10px]"
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
                      <td className="py-2.5 px-3 text-muted-foreground">
                        {a.reasons.join(", ")}
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
        </CardContent>
      </Card>
    </div>
  );
}
