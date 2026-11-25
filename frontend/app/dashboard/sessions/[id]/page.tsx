import React from "react";
import Link from "next/link";
import { notFound, redirect } from "next/navigation";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { FormattedTimestamp } from "@/components/ui/formatted-timestamp";
import { DataExportButton } from "@/components/dashboard/data-export-button";
import {
  SessionReplayTimeline,
  type DriverBaselineSnapshot,
  type ReplayAlertEvent,
} from "@/components/dashboard/session-replay-timeline";
import {
  Activity,
  ArrowLeft,
  CheckCircle2,
  Clock,
  Eye,
  Gauge,
  ShieldAlert,
} from "lucide-react";

export const dynamic = "force-dynamic";

interface SessionDetailPageProps {
  params: Promise<{ id: string }>;
}

export default async function SessionDetailPage({ params }: SessionDetailPageProps) {
  const { id } = await params;
  const session = await getServerSession();

  if (!session?.user?.id) {
    redirect(`/auth/sign-in?redirectTo=/dashboard/sessions/${encodeURIComponent(id)}`);
  }

  const [drivingSession, driverProfiles] = await Promise.all([
    prisma.drivingSession.findFirst({
      where: {
        id,
        userId: session.user.id,
      },
      include: {
        alerts: {
          orderBy: [{ createdAt: "asc" }, { timestampMs: "asc" }],
        },
      },
    }),
    prisma.driverProfile.findMany({
      where: { userId: session.user.id },
    }),
  ]);

  if (!drivingSession) {
    notFound();
  }

  const driverProfile =
    (drivingSession.engineType
      ? driverProfiles.find((p) => p.engineType === drivingSession.engineType)
      : undefined) ??
    driverProfiles.find((p) => p.engineType === "PYTHON" && p.isCalibrated) ??
    driverProfiles.find((p) => p.isCalibrated) ??
    driverProfiles[0] ??
    null;

  const alerts: ReplayAlertEvent[] = drivingSession.alerts.map((a) => ({
    id: a.id,
    timestampMs: a.timestampMs,
    createdAt: a.createdAt.toISOString(),
    severity: a.severity,
    riskScore: a.riskScore,
    earValue: a.earValue,
    perclosPct: a.perclosPct,
    poseZone: a.poseZone,
    reasons: a.reasons,
    cascadeDecision: a.cascadeDecision,
  }));

  const baseline: DriverBaselineSnapshot = {
    centerEarMedian: driverProfile?.centerEarMedian ?? 0.29,
    baselinePerclosPct: driverProfile?.baselinePerclosPct ?? 3.5,
    isCalibrated: driverProfile?.isCalibrated ?? false,
    calibrationMode: driverProfile?.calibrationMode ?? "UNIVARIATE_CENTER",
    sensitivityLevel: driverProfile?.sensitivityLevel ?? "STANDARD",
  };

  const hasAlerts =
    drivingSession.criticalAlertsCount > 0 || drivingSession.warningAlertsCount > 0;

  const durationMins = Math.floor(drivingSession.durationSeconds / 60);
  const durationSecs = drivingSession.durationSeconds % 60;

  const exportRows = alerts.map((a) => ({
    sessionCode: drivingSession.sessionCode,
    timestamp: a.createdAt,
    timestampMs: a.timestampMs,
    severity: a.severity,
    riskScore: a.riskScore,
    earValue: a.earValue,
    perclosPct: a.perclosPct,
    poseZone: a.poseZone,
    cascadeDecision: a.cascadeDecision,
    reasons: a.reasons.join("; "),
  }));

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Top Breadcrumb & Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div className="space-y-1.5">
          <Button asChild variant="ghost" size="sm" className="-ml-2 h-7 px-2 text-xs text-muted-foreground">
            <Link href="/dashboard/sessions">
              <ArrowLeft className="size-3.5 mr-1" />
              Back to Driving Sessions
            </Link>
          </Button>

          <div className="flex flex-wrap items-center gap-2.5">
            <h1 className="text-2xl sm:text-3xl font-bold tracking-tight font-mono text-foreground flex items-center gap-2">
              <Activity className="size-6 text-primary" />
              {drivingSession.sessionCode}
            </h1>
            {hasAlerts ? (
              <Badge variant="destructive" className="font-mono text-xs">
                {drivingSession.criticalAlertsCount} Critical / {drivingSession.warningAlertsCount} Warning
              </Badge>
            ) : (
              <Badge
                variant="outline"
                className="font-mono text-xs text-emerald-500 border-emerald-500/40"
              >
                <CheckCircle2 className="size-3 mr-1" />
                Clean Session
              </Badge>
            )}
          </div>

          <p className="text-xs text-muted-foreground">
            Started{" "}
            <FormattedTimestamp date={drivingSession.startedAt.toISOString()} mode="datetime" />
            {drivingSession.endedAt ? (
              <>
                {" • Ended "}
                <FormattedTimestamp
                  date={drivingSession.endedAt.toISOString()}
                  mode="time"
                  showSeconds={true}
                />
              </>
            ) : (
              " • Active or ended unexpectedly"
            )}
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          {exportRows.length > 0 && (
            <DataExportButton
              data={exportRows}
              filenamePrefix={`driveguard_session_${drivingSession.sessionCode}`}
            />
          )}
          <Button asChild size="sm">
            <Link href="/dashboard/live-monitor">
              <Activity className="size-4 mr-1.5" />
              New Monitoring Session
            </Link>
          </Button>
        </div>
      </div>

      {/* Session KPI Summary Grid */}
      <div className="grid grid-cols-2 lg:grid-cols-5 gap-3">
        <Card className="border-border/80">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Clock className="size-3.5" /> Duration
            </CardTitle>
          </CardHeader>
          <CardContent className="pb-3 px-4">
            <div className="text-xl font-bold font-mono">
              {durationMins}m {durationSecs}s
            </div>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
              {drivingSession.normalFramesCount.toLocaleString()} normal frames
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Eye className="size-3.5" /> Avg / Peak PERCLOS
            </CardTitle>
          </CardHeader>
          <CardContent className="pb-3 px-4">
            <div className="text-xl font-bold font-mono">
              {drivingSession.averagePerclosPct !== null
                ? `${drivingSession.averagePerclosPct.toFixed(1)}%`
                : "—"}{" "}
              <span className="text-sm text-muted-foreground">
                /{" "}
                {drivingSession.maxPerclosPct !== null
                  ? `${drivingSession.maxPerclosPct.toFixed(1)}%`
                  : "—"}
              </span>
            </div>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
              Baseline: {baseline.baselinePerclosPct.toFixed(1)}%
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground">
              Total Blinks
            </CardTitle>
          </CardHeader>
          <CardContent className="pb-3 px-4">
            <div className="text-xl font-bold font-mono">{drivingSession.totalBlinks}</div>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
              {drivingSession.averageBlinkRate !== null
                ? `${drivingSession.averageBlinkRate.toFixed(1)} blinks/min`
                : "Rate unavailable"}
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <ShieldAlert className="size-3.5" /> Mirror Suppressions
            </CardTitle>
          </CardHeader>
          <CardContent className="pb-3 px-4">
            <div className="text-xl font-bold font-mono">{drivingSession.mirrorSuppressions}</div>
            <p className="text-[11px] text-muted-foreground mt-0.5">
              False-positive checks prevented
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80 col-span-2 lg:col-span-1">
          <CardHeader className="pb-1 pt-3 px-4">
            <CardTitle className="text-[11px] uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
              <Gauge className="size-3.5" /> Inference Rate
            </CardTitle>
          </CardHeader>
          <CardContent className="pb-3 px-4">
            <div className="text-xl font-bold font-mono">
              {drivingSession.averageFps ? `${drivingSession.averageFps.toFixed(1)} fps` : "—"}
            </div>
            <p className="text-[11px] text-muted-foreground font-mono mt-0.5">
              {drivingSession.averageLatencyMs
                ? `${drivingSession.averageLatencyMs.toFixed(1)} ms avg`
                : "Latency unavailable"}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Interactive Replay & Detection Timeline */}
      <SessionReplayTimeline
        sessionStartedAt={drivingSession.startedAt.toISOString()}
        durationSeconds={drivingSession.durationSeconds}
        alerts={alerts}
        baseline={baseline}
      />
    </div>
  );
}
