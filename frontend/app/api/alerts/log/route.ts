import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

interface InsertedAlertRow {
  id: string;
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
      clientSessionId,
      eventId,
      severity = "WARNING",
      riskScore = 0.5,
      earValue = 0.2,
      perclosPct = 15.0,
      poseZone = "CENTER",
      reasons = [],
      cascadeDecision = "GEOMETRIC",
      timestampMs = Date.now(),
    } = body;

    const trimmedSessionCode =
      typeof sessionCode === "string" && sessionCode.trim().length > 0
        ? sessionCode.trim()
        : null;
    const trimmedClientSessionId =
      typeof clientSessionId === "string" && clientSessionId.trim().length > 0
        ? clientSessionId.trim()
        : null;
    const trimmedEventId =
      typeof eventId === "string" && eventId.trim().length > 0
        ? eventId.trim().slice(0, 128)
        : null;

    if (!trimmedSessionCode && !trimmedClientSessionId) {
      return NextResponse.json(
        { error: "sessionCode or clientSessionId is required" },
        { status: 400 }
      );
    }

    const drivingSession = trimmedClientSessionId
      ? await prisma.drivingSession.findUnique({
          where: { clientSessionId: trimmedClientSessionId },
        })
      : await prisma.drivingSession.findUnique({
          where: { sessionCode: trimmedSessionCode! },
        });

    if (!drivingSession || drivingSession.userId !== userId) {
      return NextResponse.json({ error: "Session not found or forbidden" }, { status: 404 });
    }

    // Input validation & clamping
    const parsedRisk = Math.max(
      0.0,
      Math.min(1.0, Number.isFinite(Number(riskScore)) ? Number(riskScore) : 0.5)
    );
    const parsedEar = Math.max(
      0.0,
      Number.isFinite(Number(earValue)) ? Number(earValue) : 0.2
    );
    const parsedPerclos = Math.max(
      0.0,
      Math.min(100.0, Number.isFinite(Number(perclosPct)) ? Number(perclosPct) : 15.0)
    );
    const parsedTimestamp =
      Number.isFinite(Number(timestampMs)) && Number(timestampMs) > 0
        ? Number(timestampMs)
        : Date.now();
    const normalizedSeverity =
      String(severity).toUpperCase() === "CRITICAL" ? "CRITICAL" : "WARNING";
    const normalizedPoseZone = String(poseZone || "CENTER");
    const normalizedReasons = Array.isArray(reasons)
      ? reasons.map(String)
      : reasons != null
        ? [String(reasons)]
        : [];
    const normalizedCascade = String(cascadeDecision || "GEOMETRIC");

    // Atomic transaction: deduplicate by eventId via ON CONFLICT DO NOTHING,
    // and increment session alert counters ONLY when a new alert row is actually inserted.
    const result = await prisma.$transaction(async (tx) => {
      if (trimmedEventId) {
        const generatedAlertId = `alrt_${crypto.randomUUID().replace(/-/g, "")}`;
        const inserted = await tx.$queryRaw<InsertedAlertRow[]>`
          INSERT INTO public.drowsiness_alerts (
            id,
            "sessionId",
            "eventId",
            "timestampMs",
            severity,
            "riskScore",
            "earValue",
            "perclosPct",
            "poseZone",
            reasons,
            "cascadeDecision",
            "createdAt"
          )
          VALUES (
            ${generatedAlertId},
            ${drivingSession.id},
            ${trimmedEventId},
            ${parsedTimestamp},
            ${normalizedSeverity},
            ${parsedRisk},
            ${parsedEar},
            ${parsedPerclos},
            ${normalizedPoseZone},
            ${normalizedReasons}::text[],
            ${normalizedCascade},
            NOW()
          )
          ON CONFLICT ("eventId") DO NOTHING
          RETURNING id
        `;

        if (inserted.length === 1) {
          await tx.drivingSession.update({
            where: { id: drivingSession.id },
            data:
              normalizedSeverity === "CRITICAL"
                ? { criticalAlertsCount: { increment: 1 } }
                : { warningAlertsCount: { increment: 1 } },
          });

          return {
            status: 200 as const,
            alertId: inserted[0].id,
            deduplicated: false,
          };
        }

        // Duplicate eventId (including concurrent race): verify ownership without incrementing counters
        const existingAlert = await tx.drowsinessAlert.findUnique({
          where: { eventId: trimmedEventId },
          include: { session: { select: { userId: true } } },
        });

        if (!existingAlert || existingAlert.session.userId !== userId) {
          return {
            status: 409 as const,
            error: "Conflicting eventId belongs to another session",
          };
        }

        return {
          status: 200 as const,
          alertId: existingAlert.id,
          deduplicated: true,
        };
      }

      // Legacy fallback when caller does not supply eventId
      const created = await tx.drowsinessAlert.create({
        data: {
          sessionId: drivingSession.id,
          timestampMs: parsedTimestamp,
          severity: normalizedSeverity,
          riskScore: parsedRisk,
          earValue: parsedEar,
          perclosPct: parsedPerclos,
          poseZone: normalizedPoseZone,
          reasons: normalizedReasons,
          cascadeDecision: normalizedCascade,
        },
      });

      await tx.drivingSession.update({
        where: { id: drivingSession.id },
        data:
          normalizedSeverity === "CRITICAL"
            ? { criticalAlertsCount: { increment: 1 } }
            : { warningAlertsCount: { increment: 1 } },
      });

      return {
        status: 200 as const,
        alertId: created.id,
        deduplicated: false,
      };
    });

    if (result.status === 409) {
      return NextResponse.json({ error: result.error }, { status: 409 });
    }

    return NextResponse.json({
      success: true,
      alertId: result.alertId,
      deduplicated: result.deduplicated,
    });
  } catch (error) {
    console.error("Failed to log drowsiness alert:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
