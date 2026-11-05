import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    let userId = session?.user?.id;
    if (!userId) {
      const fallbackUser = await prisma.user.findFirst();
      if (fallbackUser) {
        userId = fallbackUser.id;
      } else {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
    }

    const body = await req.json();
    const {
      sessionCode,
      durationSeconds = 0,
      averageFps = 0,
      averageLatencyMs = 0,
      averagePerclosPct = 0,
      maxPerclosPct = 0,
      totalBlinks = 0,
      averageBlinkRate = 0,
      normalFramesCount = 0,
      warningAlertsCount = 0,
      criticalAlertsCount = 0,
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

    const updated = await prisma.drivingSession.update({
      where: { sessionCode },
      data: {
        endedAt: new Date(),
        durationSeconds: Math.max(existing.durationSeconds, durationSeconds),
        averageFps: averageFps > 0 ? averageFps : existing.averageFps,
        averageLatencyMs: averageLatencyMs > 0 ? averageLatencyMs : existing.averageLatencyMs,
        averagePerclosPct: averagePerclosPct > 0 ? averagePerclosPct : existing.averagePerclosPct,
        maxPerclosPct: Math.max(existing.maxPerclosPct || 0, maxPerclosPct),
        totalBlinks: totalBlinks > 0 ? totalBlinks : existing.totalBlinks,
        averageBlinkRate: averageBlinkRate > 0 ? averageBlinkRate : existing.averageBlinkRate,
        normalFramesCount: Math.max(existing.normalFramesCount, normalFramesCount),
        warningAlertsCount: Math.max(existing.warningAlertsCount, warningAlertsCount),
        criticalAlertsCount: Math.max(existing.criticalAlertsCount, criticalAlertsCount),
        mirrorSuppressions: Math.max(existing.mirrorSuppressions, mirrorSuppressions),
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
