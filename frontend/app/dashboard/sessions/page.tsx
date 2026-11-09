import React from "react";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { ClipboardList, Activity } from "lucide-react";
import {
  SessionsTableInteractive,
  type SerializedDrivingSession,
} from "@/components/dashboard/sessions-table-interactive";
import { DataExportButton } from "@/components/dashboard/data-export-button";

export const dynamic = "force-dynamic";

export default async function DrivingSessionsPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const rawSessions = userId
    ? await prisma.drivingSession.findMany({
        where: { userId },
        include: {
          alerts: {
            orderBy: { createdAt: "desc" },
            take: 25,
          },
        },
        orderBy: { startedAt: "desc" },
        take: 50,
      })
    : [];

  const sessions: SerializedDrivingSession[] = rawSessions.map((s) => ({
    id: s.id,
    sessionCode: s.sessionCode,
    startedAt: s.startedAt.toISOString(),
    endedAt: s.endedAt ? s.endedAt.toISOString() : null,
    durationSeconds: s.durationSeconds,
    averageFps: s.averageFps,
    averageLatencyMs: s.averageLatencyMs,
    averagePerclosPct: s.averagePerclosPct,
    maxPerclosPct: s.maxPerclosPct,
    totalBlinks: s.totalBlinks,
    averageBlinkRate: s.averageBlinkRate,
    normalFramesCount: s.normalFramesCount,
    warningAlertsCount: s.warningAlertsCount,
    criticalAlertsCount: s.criticalAlertsCount,
    mirrorSuppressions: s.mirrorSuppressions,
    alerts: s.alerts.map((a) => ({
      id: a.id,
      createdAt: a.createdAt.toISOString(),
      severity: a.severity,
      riskScore: a.riskScore,
      earValue: a.earValue,
      perclosPct: a.perclosPct,
      poseZone: a.poseZone,
      reasons: a.reasons,
      cascadeDecision: a.cascadeDecision,
    })),
  }));

  const exportRecords = sessions.map((s) => ({
    sessionCode: s.sessionCode,
    startedAt: s.startedAt,
    endedAt: s.endedAt ?? "",
    durationSeconds: s.durationSeconds,
    averagePerclosPct: s.averagePerclosPct ?? "",
    maxPerclosPct: s.maxPerclosPct ?? "",
    totalBlinks: s.totalBlinks,
    averageBlinkRate: s.averageBlinkRate ?? "",
    normalFramesCount: s.normalFramesCount,
    warningAlertsCount: s.warningAlertsCount,
    criticalAlertsCount: s.criticalAlertsCount,
    mirrorSuppressions: s.mirrorSuppressions,
    averageFps: s.averageFps ?? "",
    averageLatencyMs: s.averageLatencyMs ?? "",
  }));

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <ClipboardList className="size-6 text-primary" />
            Driving Sessions History
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Audit history of driving trips, duration, blink kinematics, and fatigue prevention telemetry.
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <DataExportButton
            data={exportRecords}
            filenamePrefix="driveguard_sessions"
          />
          <Button asChild size="sm">
            <Link href="/dashboard/live-monitor">
              <Activity className="size-4 mr-1.5" />
              Start New Session
            </Link>
          </Button>
        </div>
      </div>

      <Card className="border-border/80">
        <CardHeader>
          <CardTitle className="text-lg">Trip Log</CardTitle>
          <CardDescription>
            All tracked driving sessions recorded by DriveGuard. Click Inspect on any session for a full breakdown.
          </CardDescription>
        </CardHeader>
        <CardContent>
          {sessions.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-xs">
              No driving sessions recorded yet. Launch live monitoring to track your first session!
            </div>
          ) : (
            <SessionsTableInteractive sessions={sessions} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}
