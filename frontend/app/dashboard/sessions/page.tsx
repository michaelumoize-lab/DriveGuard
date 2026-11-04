import React from "react";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { ClipboardList, Clock, ShieldCheck, Activity } from "lucide-react";

export const dynamic = "force-dynamic";

export default async function DrivingSessionsPage() {
  const session = await getServerSession();
  const userId = session?.user?.id;

  const sessions = userId
    ? await prisma.drivingSession.findMany({
        where: { userId },
        orderBy: { startedAt: "desc" },
        take: 30,
      })
    : [];

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground flex items-center gap-2">
            <ClipboardList className="size-6 text-primary" />
            Driving Sessions History
          </h1>
          <p className="text-xs sm:text-sm text-muted-foreground">
            Audit history of driving trips, duration, blink rates, and fatigue prevention statistics.
          </p>
        </div>

        <Button asChild size="sm">
          <Link href="/dashboard/live-monitor">
            <Activity className="size-4 mr-1.5" />
            Start New Session
          </Link>
        </Button>
      </div>

      <Card className="border-border/80">
        <CardHeader>
          <CardTitle className="text-lg">Trip Log</CardTitle>
          <CardDescription>All tracked driving sessions recorded by DriveGuard.</CardDescription>
        </CardHeader>
        <CardContent>
          {sessions.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-xs">
              No driving sessions recorded yet. Launch live monitoring to track your first session!
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-xs text-left">
                <thead className="border-b border-border text-muted-foreground">
                  <tr>
                    <th className="py-2.5 px-3">Session Code</th>
                    <th className="py-2.5 px-3">Date</th>
                    <th className="py-2.5 px-3">Duration</th>
                    <th className="py-2.5 px-3">Avg PERCLOS</th>
                    <th className="py-2.5 px-3">Max PERCLOS</th>
                    <th className="py-2.5 px-3">Blinks</th>
                    <th className="py-2.5 px-3">Critical Alerts</th>
                    <th className="py-2.5 px-3">FPS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {sessions.map((s) => (
                    <tr key={s.id} className="hover:bg-muted/30">
                      <td className="py-2.5 px-3 font-mono font-medium text-foreground">
                        {s.sessionCode}
                      </td>
                      <td className="py-2.5 px-3 text-muted-foreground">
                        {new Date(s.startedAt).toLocaleDateString()} {new Date(s.startedAt).toLocaleTimeString()}
                      </td>
                      <td className="py-2.5 px-3 font-mono">
                        {Math.floor(s.durationSeconds / 60)}m {s.durationSeconds % 60}s
                      </td>
                      <td className="py-2.5 px-3 font-mono">
                        {s.averagePerclosPct ? `${s.averagePerclosPct.toFixed(1)}%` : "N/A"}
                      </td>
                      <td className="py-2.5 px-3 font-mono">
                        {s.maxPerclosPct ? `${s.maxPerclosPct.toFixed(1)}%` : "N/A"}
                      </td>
                      <td className="py-2.5 px-3 font-mono">{s.totalBlinks}</td>
                      <td className="py-2.5 px-3">
                        {s.criticalAlertsCount > 0 ? (
                          <Badge variant="destructive" className="text-[10px]">
                            {s.criticalAlertsCount}
                          </Badge>
                        ) : (
                          <Badge variant="outline" className="text-[10px] text-emerald-500 border-emerald-500/30">
                            0 (Clean)
                          </Badge>
                        )}
                      </td>
                      <td className="py-2.5 px-3 font-mono text-muted-foreground">
                        {s.averageFps ? s.averageFps.toFixed(1) : "20+"}
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
