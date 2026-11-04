import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      sessionCode,
      severity = "WARNING",
      riskScore = 0.5,
      earValue = 0.2,
      perclosPct = 15.0,
      poseZone = "CENTER",
      reasons = [],
      cascadeDecision = "GEOMETRIC",
      timestampMs = Date.now(),
    } = body;

    if (!sessionCode) {
      return NextResponse.json({ error: "sessionCode is required" }, { status: 400 });
    }

    const drivingSession = await prisma.drivingSession.findUnique({
      where: { sessionCode },
    });

    if (!drivingSession || drivingSession.userId !== session.user.id) {
      return NextResponse.json({ error: "Session not found or forbidden" }, { status: 404 });
    }

    const alert = await prisma.drowsinessAlert.create({
      data: {
        sessionId: drivingSession.id,
        timestampMs,
        severity: severity.toUpperCase() === "CRITICAL" ? "CRITICAL" : "WARNING",
        riskScore: Number(riskScore),
        earValue: Number(earValue),
        perclosPct: Number(perclosPct),
        poseZone: String(poseZone),
        reasons: Array.isArray(reasons) ? reasons : [String(reasons)],
        cascadeDecision: String(cascadeDecision),
      },
    });

    // Increment corresponding alert counter on the session
    if (severity.toUpperCase() === "CRITICAL") {
      await prisma.drivingSession.update({
        where: { id: drivingSession.id },
        data: { criticalAlertsCount: { increment: 1 } },
      });
    } else {
      await prisma.drivingSession.update({
        where: { id: drivingSession.id },
        data: { warningAlertsCount: { increment: 1 } },
      });
    }

    return NextResponse.json({ success: true, alertId: alert.id });
  } catch (error) {
    console.error("Failed to log drowsiness alert:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
