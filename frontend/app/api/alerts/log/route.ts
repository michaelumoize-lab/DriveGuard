import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

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
      severity = "WARNING",
      riskScore = 0.5,
      earValue = 0.2,
      perclosPct = 15.0,
      poseZone = "CENTER",
      reasons = [],
      cascadeDecision = "GEOMETRIC",
      timestampMs = Date.now(),
    } = body;

    if (!sessionCode || typeof sessionCode !== "string") {
      return NextResponse.json({ error: "sessionCode is required and must be a string" }, { status: 400 });
    }

    const drivingSession = await prisma.drivingSession.findUnique({
      where: { sessionCode: sessionCode.trim() },
    });

    if (!drivingSession || drivingSession.userId !== userId) {
      return NextResponse.json({ error: "Session not found or forbidden" }, { status: 404 });
    }

    // Input validation & clamping
    const parsedRisk = Math.max(0.0, Math.min(1.0, Number.isFinite(Number(riskScore)) ? Number(riskScore) : 0.5));
    const parsedEar = Math.max(0.0, Number.isFinite(Number(earValue)) ? Number(earValue) : 0.2);
    const parsedPerclos = Math.max(0.0, Math.min(100.0, Number.isFinite(Number(perclosPct)) ? Number(perclosPct) : 15.0));
    const parsedTimestamp = Number.isFinite(Number(timestampMs)) && Number(timestampMs) > 0 ? Number(timestampMs) : Date.now();
    const normalizedSeverity = String(severity).toUpperCase() === "CRITICAL" ? "CRITICAL" : "WARNING";

    // DG-17: Atomic Transaction: Alert creation and session counter increment must commit together
    const [alert] = await prisma.$transaction([
      prisma.drowsinessAlert.create({
        data: {
          sessionId: drivingSession.id,
          timestampMs: parsedTimestamp,
          severity: normalizedSeverity,
          riskScore: parsedRisk,
          earValue: parsedEar,
          perclosPct: parsedPerclos,
          poseZone: String(poseZone || "CENTER"),
          reasons: Array.isArray(reasons) ? reasons.map(String) : [String(reasons)],
          cascadeDecision: String(cascadeDecision || "GEOMETRIC"),
        },
      }),
      prisma.drivingSession.update({
        where: { id: drivingSession.id },
        data: normalizedSeverity === "CRITICAL"
          ? { criticalAlertsCount: { increment: 1 } }
          : { warningAlertsCount: { increment: 1 } },
      }),
    ]);

    return NextResponse.json({ success: true, alertId: alert.id });
  } catch (error) {
    console.error("Failed to log drowsiness alert:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
