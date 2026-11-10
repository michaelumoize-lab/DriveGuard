import React from "react";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/dashboard/stat-card";
import { FormattedTimestamp } from "@/components/ui/formatted-timestamp";
import {
  Activity,
  ShieldCheck,
  AlertTriangle,
  Clock,
  Car,
  Eye,
  Sparkles,
  ArrowRight,
  TrendingDown,
  ShieldAlert,
  Compass,
  Settings,
} from "lucide-react";
import { ZoneBaselineEntry } from "@/types/telemetry";

export const dynamic = "force-dynamic";

export default async function AdminDashboardPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const [
    profile,
    driverProfile,
    totalSessions,
    sessionAggregates,
    sessionsList,
    totalCriticalAlerts,
    totalWarningAlerts,
    recentAlerts,
  ] = await Promise.all([
    userId ? getOrCreateUserProfile(userId) : null,
    userId ? prisma.driverProfile.findUnique({ where: { userId } }) : null,
    userId ? prisma.drivingSession.count({ where: { userId } }) : 0,
    userId
      ? prisma.drivingSession.aggregate({
          where: { userId },
          _sum: { durationSeconds: true },
          _avg: { averagePerclosPct: true },
        })
      : { _sum: { durationSeconds: null }, _avg: { averagePerclosPct: null } },
    userId
      ? prisma.drivingSession.findMany({
          where: { userId },
          orderBy: { startedAt: "desc" },
          take: 5,
        })
      : [],
    userId
      ? prisma.drowsinessAlert.count({
          where: { session: { userId }, severity: "CRITICAL" },
        })
      : 0,
    userId
      ? prisma.drowsinessAlert.count({
          where: { session: { userId }, severity: "WARNING" },
        })
      : 0,
    userId
      ? prisma.drowsinessAlert.findMany({
          where: { session: { userId } },
          orderBy: { createdAt: "desc" },
          take: 5,
        })
      : [],
  ]);

  const totalDurationSec = sessionAggregates._sum.durationSeconds ?? 0;
  const totalHours = Math.floor(totalDurationSec / 3600);
  const totalMinutes = Math.floor((totalDurationSec % 3600) / 60);
  const totalSecondsRem = totalDurationSec % 60;

  const formattedDrivingTime =
    totalDurationSec === 0
      ? "0m"
      : totalHours > 0
        ? `${totalHours}h ${totalMinutes}m`
        : totalMinutes > 0
          ? `${totalMinutes}m ${totalSecondsRem}s`
          : `${totalSecondsRem}s`;

  const avgPerclosVal = sessionAggregates._avg.averagePerclosPct;
  const formattedAvgPerclos =
    avgPerclosVal !== null && avgPerclosVal !== undefined
      ? `${avgPerclosVal.toFixed(1)}%`
      : "—";

  const zoneBaselines = (driverProfile?.zoneBaselines ?? null) as Record<
    string,
    ZoneBaselineEntry
  > | null;

  const zoneKeys = ["CENTER", "LEFT_MIRROR", "RIGHT_MIRROR", "DASHBOARD"] as const;

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      {/* Driver & Vehicle Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-primary/20 bg-gradient-to-r from-primary/10 via-primary/5 to-transparent p-6 shadow-sm">
        <div className="relative z-10 flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="space-y-1.5">
            <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-primary/20 text-primary border border-primary/30">
              <Sparkles className="size-3" />
              <span>DriveGuard Active Driver Safety</span>
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
              Welcome back, {session?.user?.name || "Driver"}
            </h1>
            <p className="text-xs sm:text-sm text-muted-foreground flex items-center gap-2 flex-wrap">
              <span className="flex items-center gap-1 text-foreground">
                <Car className="size-3.5 text-primary" />
                {profile?.vehicleMake && profile?.vehicleModel
                  ? `${profile.vehicleMake} ${profile.vehicleModel}${profile.vehicleYear ? ` (${profile.vehicleYear})` : ""}`
                  : "Vehicle Profile Not Set"}
              </span>
              <span>•</span>
              <span>
                Sensitivity: <strong className="text-foreground">{profile?.sensitivityLevel || "MEDIUM"}</strong>
              </span>
              <span>•</span>
              <span>
                Baseline:{" "}
                {driverProfile?.isCalibrated ? (
                  <strong className="text-emerald-500 font-mono tabular-nums">
                    4-Zone Calibrated (Center EAR: {driverProfile.centerEarMedian.toFixed(3)})
                  </strong>
                ) : (
                  <span className="text-amber-500 font-medium">Uncalibrated (Using Fallback Defaults)</span>
                )}
              </span>
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-2.5 shrink-0">
            <Button asChild size="sm" variant="outline" className="h-9">
              <Link href="/dashboard/calibration">
                <Eye className="size-4 mr-1.5" />
                {driverProfile?.isCalibrated ? "Inspect / Recalibrate Zones" : "Calibrate 4-Zones"}
              </Link>
            </Button>
            <Button asChild size="sm" className="h-9 bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/dashboard/live-monitor">
                <Activity className="size-4 mr-1.5" />
                Start Live Monitor
              </Link>
            </Button>
          </div>
        </div>
      </div>

      {/* KPI Cards (Authentic Aggregated Database Values) */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Monitored Driving Time"
          value={formattedDrivingTime}
          description={`Cumulative across ${totalSessions} recorded ${totalSessions === 1 ? "session" : "sessions"}`}
          icon={<Clock className="size-4 text-primary" />}
        />
        <StatCard
          title="Average PERCLOS (60s)"
          value={formattedAvgPerclos}
          description={
            avgPerclosVal !== null && avgPerclosVal !== undefined
              ? "Measured rolling eye closure (<12% normal)"
              : "Complete a monitored session to compute"
          }
          icon={<TrendingDown className="size-4 text-primary" />}
        />
        <StatCard
          title="Driving Trips Monitored"
          value={totalSessions.toString()}
          description="Total logged monitoring sessions"
          icon={<ShieldCheck className="size-4 text-primary" />}
        />
        <StatCard
          title="Fatigue Interventions"
          value={(totalCriticalAlerts + totalWarningAlerts).toString()}
          description={`${totalWarningAlerts} warning / ${totalCriticalAlerts} critical alerts`}
          icon={<AlertTriangle className="size-4 text-primary" />}
        />
      </div>

      {/* Middle Row: 4-Zone Calibration Status + Quick Launch + Active Safeguards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 4-Zone Personal Baseline Summary Card */}
        <Card className="lg:col-span-2 border-border/80">
          <CardHeader className="pb-3">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <div className="space-y-1">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Compass className="size-5 text-primary" />
                  4-Zone Pose-Conditioned Driver Baseline (PDBP)
                </CardTitle>
                <CardDescription>
                  Personal Eye Aspect Ratio (EAR) medians and closure thresholds across cabin gaze zones.
                </CardDescription>
              </div>
              {driverProfile?.isCalibrated ? (
                <Badge variant="outline" className="border-emerald-500/40 text-emerald-500 bg-emerald-500/10 font-mono">
                  CALIBRATED ({driverProfile.calibrationMode})
                </Badge>
              ) : (
                <Badge variant="outline" className="border-amber-500/40 text-amber-500 bg-amber-500/10 font-mono">
                  UNCALIBRATED
                </Badge>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            {!driverProfile?.isCalibrated ? (
              <div className="p-4 rounded-xl border border-amber-500/30 bg-amber-500/5 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div className="space-y-1">
                  <div className="font-semibold text-sm text-foreground">Personal 4-Zone Calibration Recommended</div>
                  <p className="text-xs text-muted-foreground">
                    You have not completed the 4-zone calibration wizard yet. The AI engine will use generic uncalibrated defaults until you capture your personal CENTER, LEFT_MIRROR, RIGHT_MIRROR, and DASHBOARD baselines.
                  </p>
                </div>
                <Button asChild size="sm" className="shrink-0">
                  <Link href="/dashboard/calibration">
                    Run 20s Wizard <ArrowRight className="size-4 ml-1.5" />
                  </Link>
                </Button>
              </div>
            ) : (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {zoneKeys.map((zone) => {
                  const zb = zoneBaselines?.[zone];
                  const earMed =
                    zb?.ear_median ??
                    (zone === "CENTER" ? driverProfile.centerEarMedian : null);
                  const thresh =
                    zb?.dynamic_closure_threshold ??
                    (earMed ? Number((earMed * 0.72).toFixed(3)) : null);
                  return (
                    <div
                      key={zone}
                      className="p-3 rounded-xl border border-border bg-muted/30 space-y-1"
                    >
                      <span className="text-[11px] font-mono text-muted-foreground block">
                        {zone.replace("_", " ")}
                      </span>
                      <div className="text-base font-bold font-mono tabular-nums text-foreground">
                        {earMed !== null ? earMed.toFixed(3) : "—"}
                      </div>
                      <span className="text-[11px] font-mono tabular-nums text-emerald-500 block">
                        Thresh: {thresh !== null ? thresh.toFixed(3) : "—"}
                      </span>
                    </div>
                  );
                })}
              </div>
            )}

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 pt-1">
              <div className="p-3.5 rounded-xl bg-muted/40 border border-border flex items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <div className="font-semibold text-xs text-foreground">Real-Time Monitoring Workspace</div>
                  <div className="text-xs text-muted-foreground">
                    LiveKit WebRTC video + 15 Hz telemetry + synthesized audio alerts
                  </div>
                </div>
                <Button asChild size="sm" className="shrink-0 h-8 text-xs">
                  <Link href="/dashboard/live-monitor">
                    Launch <ArrowRight className="size-3.5 ml-1" />
                  </Link>
                </Button>
              </div>

              <div className="p-3.5 rounded-xl bg-muted/40 border border-border flex items-center justify-between gap-3">
                <div className="space-y-0.5">
                  <div className="font-semibold text-xs text-foreground">Driver & Alert Preferences</div>
                  <div className="text-xs text-muted-foreground">
                    Configure vehicle details, sensitivity, mirror suppression & audio
                  </div>
                </div>
                <Button asChild variant="outline" size="sm" className="shrink-0 h-8 text-xs">
                  <Link href="/settings/account">
                    <Settings className="size-3.5 mr-1" /> Configure
                  </Link>
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Safety & Alert Overview Card */}
        <Card className="border-border/80">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <ShieldAlert className="size-5 text-primary" />
              Active Safeguards
            </CardTitle>
            <CardDescription>Configured fatigue prevention parameters</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Mirror Glance Suppression:</span>
              <span className="font-semibold text-foreground">
                {profile?.mirrorSuppression !== false ? "Enabled (<= 3.0s)" : "Disabled"}
              </span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Micro-Sleep Override:</span>
              <span className="font-semibold font-mono tabular-nums text-rose-500">
                &ge; {driverProfile?.isCalibrated ? Math.max(1.1, (driverProfile.blinkDurationMedianMs * 3.5) / 1000).toFixed(2) : "1.10"}s Closure
              </span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Eyewear Profile:</span>
              <span className="font-semibold text-foreground">
                {profile?.wearsGlasses ? "Prescription / Glasses" : "No Eyewear"}
              </span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Audible Cabin Siren:</span>
              <span className="font-semibold text-foreground">
                {profile?.audioAlertsEnabled !== false ? "Enabled (4-Tier)" : "Muted"}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Bottom Tables: Recent Driving Sessions & Recent Fatigue Interventions */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Recent Sessions Table */}
        <Card className="border-border/80">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-lg">Recent Driving Sessions</CardTitle>
                <CardDescription>Latest 5 monitored sessions for your account.</CardDescription>
              </div>
              <Button asChild variant="ghost" size="sm" className="text-xs">
                <Link href="/dashboard/sessions">View All Sessions</Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {sessionsList.length === 0 ? (
              <div className="text-center py-8 text-xs text-muted-foreground">
                No driving sessions recorded yet. Launch the Live Monitor to record your first session.
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs text-left">
                  <caption className="sr-only">Recent 5 driving sessions</caption>
                  <thead className="border-b border-border text-muted-foreground">
                    <tr>
                      <th scope="col" className="py-2 px-3">Session</th>
                      <th scope="col" className="py-2 px-3">Duration</th>
                      <th scope="col" className="py-2 px-3">Avg PERCLOS</th>
                      <th scope="col" className="py-2 px-3">Alerts</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {sessionsList.map((s) => (
                      <tr key={s.id} className="hover:bg-muted/30">
                        <td className="py-2.5 px-3 font-mono text-foreground">
                          {s.sessionCode}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums text-muted-foreground">
                          {s.durationSeconds > 0
                            ? `${Math.floor(s.durationSeconds / 60)}m ${s.durationSeconds % 60}s`
                            : "—"}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums">
                          {s.averagePerclosPct !== null && s.averagePerclosPct !== undefined
                            ? `${s.averagePerclosPct.toFixed(1)}%`
                            : "—"}
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums">
                          <span className={s.criticalAlertsCount > 0 ? "text-rose-500 font-semibold" : "text-muted-foreground"}>
                            {s.criticalAlertsCount} crit
                          </span>{" "}
                          / {s.warningAlertsCount} warn
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </CardContent>
        </Card>

        {/* Recent Alerts Table */}
        <Card className="border-border/80">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-lg">Recent Fatigue Interventions</CardTitle>
                <CardDescription>Latest drowsiness state transitions and reasons.</CardDescription>
              </div>
              <Button asChild variant="ghost" size="sm" className="text-xs">
                <Link href="/dashboard/alerts">View All Alerts</Link>
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            {recentAlerts.length === 0 ? (
              <div className="text-center py-8 text-xs text-muted-foreground">
                No fatigue alerts recorded yet. Start a driving session to begin monitoring!
              </div>
            ) : (
              <div
                className="overflow-x-auto rounded-md focus-visible:outline-hidden focus-visible:ring-1 focus-visible:ring-ring"
                tabIndex={0}
                role="region"
                aria-label="Recent fatigue intervention alerts"
              >
                <table className="w-full text-xs text-left">
                  <caption className="sr-only">Recent 5 fatigue intervention alerts</caption>
                  <thead className="sticky top-0 bg-background/95 backdrop-blur-xs border-b border-border text-muted-foreground z-10">
                    <tr>
                      <th scope="col" className="py-2 px-3">Time</th>
                      <th scope="col" className="py-2 px-3">Severity</th>
                      <th scope="col" className="py-2 px-3">Risk</th>
                      <th scope="col" className="py-2 px-3">Zone</th>
                      <th scope="col" className="py-2 px-3">Reasons</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-border">
                    {recentAlerts.map((alert) => (
                      <tr key={alert.id} className="hover:bg-muted/30">
                        <td className="py-2.5 px-3 font-mono tabular-nums text-muted-foreground whitespace-nowrap">
                          <FormattedTimestamp date={alert.createdAt} mode="time" showSeconds={true} />
                        </td>
                        <td className="py-2.5 px-3">
                          <Badge
                            variant={alert.severity === "CRITICAL" ? "destructive" : "outline"}
                            className={
                              alert.severity === "CRITICAL"
                                ? "text-[10px]"
                                : "text-[10px] border-amber-500/40 text-amber-500"
                            }
                          >
                            {alert.severity}
                          </Badge>
                        </td>
                        <td className="py-2.5 px-3 font-mono tabular-nums font-medium">
                          {(alert.riskScore * 100).toFixed(0)}%
                        </td>
                        <td className="py-2.5 px-3">
                          <Badge variant="outline" className="text-[10px] font-mono">
                            {alert.poseZone}
                          </Badge>
                        </td>
                        <td className="py-2.5 px-3 text-muted-foreground truncate max-w-[180px]">
                          {alert.reasons.join(", ")}
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
    </div>
  );
}
