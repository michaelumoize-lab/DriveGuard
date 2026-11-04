import React from "react";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";
import { getOrCreateUserProfile } from "@/lib/user-profile";
import { prisma } from "@/lib/prisma";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { StatCard } from "@/components/dashboard/stat-card";
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
} from "lucide-react";

export const dynamic = "force-dynamic";

export default async function AdminDashboardPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const profile = userId ? await getOrCreateUserProfile(userId) : null;
  const driverProfile = userId
    ? await prisma.driverProfile.findUnique({ where: { userId } })
    : null;

  // Query driving sessions and alerts in parallel
  const [
    totalSessions,
    sessionsList,
    totalCriticalAlerts,
    totalWarningAlerts,
    recentAlerts,
  ] = await Promise.all([
    userId ? prisma.drivingSession.count({ where: { userId } }) : 0,
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

  const totalDurationSec = sessionsList.reduce((acc, s) => acc + (s.durationSeconds || 0), 0);
  const totalHours = Math.floor(totalDurationSec / 3600);
  const totalMinutes = Math.floor((totalDurationSec % 3600) / 60);

  const avgPerclos =
    sessionsList.length > 0
      ? (
          sessionsList.reduce((acc, s) => acc + (s.averagePerclosPct || 3.5), 0) /
          sessionsList.length
        ).toFixed(1)
      : "3.2";

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
                  ? `${profile.vehicleMake} ${profile.vehicleModel} (${profile.vehicleYear || ""})`
                  : "Vehicle Profile Ready"}
              </span>
              <span>•</span>
              <span>Sensitivity: <strong className="text-foreground">{profile?.sensitivityLevel || "STANDARD"}</strong></span>
              <span>•</span>
              <span>
                Baseline:{" "}
                {driverProfile?.isCalibrated ? (
                  <strong className="text-emerald-500">4-Zone Calibrated (EAR: {driverProfile.centerEarMedian})</strong>
                ) : (
                  <span className="text-amber-500">Default (Uncalibrated)</span>
                )}
              </span>
            </p>
          </div>

          <div className="flex items-center gap-2.5 shrink-0">
            <Button asChild size="sm" variant="outline" className="h-9">
              <Link href="/dashboard/calibration">
                <Eye className="size-4 mr-1.5" />
                Calibrate 4-Zones
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

      {/* KPI Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Monitored Driving Time"
          value={`${totalHours}h ${totalMinutes}m`}
          description="Total active safety tracking duration"
          icon={<Clock className="size-4 text-primary" />}
        />
        <StatCard
          title="Average PERCLOS (60s)"
          value={`${avgPerclos}%`}
          description="Baseline eye closure rate (<12% alert)"
          icon={<TrendingDown className="size-4 text-primary" />}
        />
        <StatCard
          title="Driving Trips Monitored"
          value={totalSessions.toString()}
          description="Recorded driving sessions"
          icon={<ShieldCheck className="size-4 text-primary" />}
        />
        <StatCard
          title="Fatigue Interventions"
          value={totalCriticalAlerts.toString()}
          description={`${totalWarningAlerts} warnings / ${totalCriticalAlerts} critical`}
          icon={<AlertTriangle className="size-4 text-primary" />}
        />
      </div>

      {/* Quick Launch & Monitoring Status Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Live Monitoring Portal Card */}
        <Card className="lg:col-span-2 border-border/80">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div className="space-y-1">
                <CardTitle className="text-lg flex items-center gap-2">
                  <Activity className="size-5 text-primary" />
                  Real-Time Driver Monitoring
                </CardTitle>
                <CardDescription>
                  Direct WebSocket feed connected to the DriveGuard FastAPI Edge Engine.
                </CardDescription>
              </div>
              <Badge variant="outline" className="border-emerald-500/30 text-emerald-500 bg-emerald-500/10">
                AI Ready
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="p-4 rounded-xl bg-muted/40 border border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="font-semibold text-sm text-foreground">Launch Webcam Driver Telemetry</div>
                <div className="text-xs text-muted-foreground">
                  Stream local camera frames at 20+ FPS to measure sub-frame blink velocity, head pose, and rolling PERCLOS.
                </div>
              </div>
              <Button asChild size="sm" className="shrink-0">
                <Link href="/dashboard/live-monitor">
                  Open Live Monitor
                  <ArrowRight className="size-4 ml-1.5" />
                </Link>
              </Button>
            </div>

            <div className="p-4 rounded-xl bg-muted/40 border border-border flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
              <div className="space-y-1">
                <div className="font-semibold text-sm text-foreground">Guided 4-Zone Spatial Calibration</div>
                <div className="text-xs text-muted-foreground">
                  Calibrate your eye aspect ratio baselines across CENTER, LEFT MIRROR, RIGHT MIRROR, and DASHBOARD.
                </div>
              </div>
              <Button asChild variant="outline" size="sm" className="shrink-0">
                <Link href="/dashboard/calibration">
                  Begin Calibration
                  <ArrowRight className="size-4 ml-1.5" />
                </Link>
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Safety & Alert Overview Card */}
        <Card className="border-border/80">
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <ShieldAlert className="size-5 text-primary" />
              Safety Safeguards
            </CardTitle>
            <CardDescription>Active fatigue prevention mechanisms</CardDescription>
          </CardHeader>
          <CardContent className="space-y-3">
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Mirror Glance Tolerance:</span>
              <span className="font-semibold text-foreground">3.0s Suppression</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Micro-Sleep Override:</span>
              <span className="font-semibold text-rose-500">Instant Critical (1.1s)</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Cascade Stage 2:</span>
              <span className="font-semibold text-primary">MobileNetV3-Small</span>
            </div>
            <div className="flex items-center justify-between p-2.5 rounded-lg border border-border text-xs">
              <span className="text-muted-foreground">Audible Alerts:</span>
              <span className="font-semibold text-foreground">
                {profile?.audioAlertsEnabled ? "Enabled" : "Muted"}
              </span>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Recent Alerts Table */}
      <Card className="border-border/80">
        <CardHeader>
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-lg">Recent Fatigue Interventions</CardTitle>
              <CardDescription>Audit log of drowsiness detection alerts from recent sessions.</CardDescription>
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
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="border-b border-border text-muted-foreground">
                  <tr>
                    <th className="py-2 px-3">Timestamp</th>
                    <th className="py-2 px-3">Severity</th>
                    <th className="py-2 px-3">Risk Score</th>
                    <th className="py-2 px-3">EAR</th>
                    <th className="py-2 px-3">Zone</th>
                    <th className="py-2 px-3">Reasons</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {recentAlerts.map((alert) => (
                    <tr key={alert.id} className="hover:bg-muted/30">
                      <td className="py-2.5 px-3 font-mono text-muted-foreground">
                        {new Date(alert.createdAt).toLocaleTimeString()}
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge
                          variant={alert.severity === "CRITICAL" ? "destructive" : "outline"}
                          className="text-[10px]"
                        >
                          {alert.severity}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 font-mono font-medium">
                        {(alert.riskScore * 100).toFixed(0)}%
                      </td>
                      <td className="py-2.5 px-3 font-mono">
                        {alert.earValue.toFixed(3)}
                      </td>
                      <td className="py-2.5 px-3">
                        <Badge variant="outline" className="text-[10px] font-mono">
                          {alert.poseZone}
                        </Badge>
                      </td>
                      <td className="py-2.5 px-3 text-muted-foreground">
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
  );
}
