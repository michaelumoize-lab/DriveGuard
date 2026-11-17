import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

function sanitizeNum(val: unknown, fallback: number, min?: number, max?: number): number {
  const n = Number(val);
  if (!Number.isFinite(n)) return fallback;
  if (min !== undefined && n < min) return min;
  if (max !== undefined && n > max) return max;
  return n;
}


export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      sessionCode,
      durationSeconds,
      durationSec,
      averageFps = 0,
      averageLatencyMs = 0,
      averagePerclosPct = 0,
      maxPerclosPct,
      maxPerclos,
      totalBlinks = 0,
      averageBlinkRate = 0,
      normalFramesCount,
      totalFrames,
      warningAlertsCount = 0,
      criticalAlertsCount,
      criticalAlerts,
      mirrorSuppressions = 0,
    } = body;

    if (!sessionCode) {
      return NextResponse.json({ error: "sessionCode is required" }, { status: 400 });
    }

    const existing = await prisma.drivingSession.findUnique({
      where: { sessionCode },
    });

    if (!existing || existing.userId !== userId) {
      return NextResponse.json({ error: "Session not found or forbidden" }, { status: 404 });
    }

    const resolvedDuration = Math.max(0, sanitizeNum(durationSeconds ?? durationSec, 0));
    const resolvedMaxPerclos = sanitizeNum(maxPerclosPct ?? maxPerclos, 0, 0, 100);
    const resolvedNormalFrames = Math.max(0, Math.round(sanitizeNum(normalFramesCount ?? totalFrames, 0)));
    const resolvedCriticalAlerts = Math.max(0, Math.round(sanitizeNum(criticalAlertsCount ?? criticalAlerts, 0)));
    const resolvedAvgFps = Math.max(0, sanitizeNum(averageFps, 0));
    const resolvedAvgLatency = Math.max(0, sanitizeNum(averageLatencyMs, 0));
    const resolvedAvgPerclos = sanitizeNum(averagePerclosPct, 0, 0, 100);
    const resolvedTotalBlinks = Math.max(0, Math.round(sanitizeNum(totalBlinks, 0)));
    const resolvedAvgBlinkRate = Math.max(0, sanitizeNum(averageBlinkRate, 0));
    const resolvedWarningAlerts = Math.max(0, Math.round(sanitizeNum(warningAlertsCount, 0)));
    const resolvedMirrorSuppressions = Math.max(0, Math.round(sanitizeNum(mirrorSuppressions, 0)));

    const updated = await prisma.drivingSession.update({
      where: { sessionCode },
      data: {
        endedAt: new Date(),
        durationSeconds: Math.max(existing.durationSeconds, Math.round(resolvedDuration)),
        averageFps: resolvedAvgFps > 0 ? resolvedAvgFps : existing.averageFps,
        averageLatencyMs: resolvedAvgLatency > 0 ? resolvedAvgLatency : existing.averageLatencyMs,
        averagePerclosPct: resolvedAvgPerclos > 0 ? resolvedAvgPerclos : existing.averagePerclosPct,
        maxPerclosPct: Math.max(existing.maxPerclosPct || 0, resolvedMaxPerclos),
        totalBlinks: resolvedTotalBlinks > 0 ? resolvedTotalBlinks : existing.totalBlinks,
        averageBlinkRate: resolvedAvgBlinkRate > 0 ? resolvedAvgBlinkRate : existing.averageBlinkRate,
        normalFramesCount: Math.max(existing.normalFramesCount, Math.round(resolvedNormalFrames)),
        warningAlertsCount: Math.max(existing.warningAlertsCount, Math.round(resolvedWarningAlerts)),
        criticalAlertsCount: Math.max(existing.criticalAlertsCount, Math.round(resolvedCriticalAlerts)),
        mirrorSuppressions: Math.max(existing.mirrorSuppressions, Math.round(resolvedMirrorSuppressions)),
      },
    });

    return NextResponse.json({
      success: true,
      session: updated,
    });
  } catch (error) {
    console.error("Failed to end driving session:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
