import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

interface LockedSessionRow {
  id: string;
  userId: string;
  sessionCode: string;
  status: string;
  finalizeToken: string | null;
  durationSeconds: number;
  averageFps: number | null;
  averageLatencyMs: number | null;
  averagePerclosPct: number | null;
  maxPerclosPct: number | null;
  totalBlinks: number;
  averageBlinkRate: number | null;
  normalFramesCount: number;
  warningAlertsCount: number;
  criticalAlertsCount: number;
  mirrorSuppressions: number;
}

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
      clientSessionId,
      finalizeToken,
      endedAt,
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

    const trimmedSessionCode =
      typeof sessionCode === "string" && sessionCode.trim().length > 0
        ? sessionCode.trim()
        : null;
    const trimmedClientSessionId =
      typeof clientSessionId === "string" && clientSessionId.trim().length > 0
        ? clientSessionId.trim()
        : null;
    const normalizedFinalizeToken =
      typeof finalizeToken === "string" && finalizeToken.trim().length > 0
        ? finalizeToken.trim().slice(0, 128)
        : null;

    if (!trimmedSessionCode && !trimmedClientSessionId) {
      return NextResponse.json(
        { error: "sessionCode or clientSessionId is required" },
        { status: 400 }
      );
    }

    const resolvedDuration = Math.max(0, sanitizeNum(durationSeconds ?? durationSec, 0));
    const resolvedMaxPerclos = sanitizeNum(maxPerclosPct ?? maxPerclos, 0, 0, 100);
    const resolvedNormalFrames = Math.max(
      0,
      Math.round(sanitizeNum(normalFramesCount ?? totalFrames, 0))
    );
    const resolvedCriticalAlerts = Math.max(
      0,
      Math.round(sanitizeNum(criticalAlertsCount ?? criticalAlerts, 0))
    );
    const resolvedAvgFps = Math.max(0, sanitizeNum(averageFps, 0));
    const resolvedAvgLatency = Math.max(0, sanitizeNum(averageLatencyMs, 0));
    const resolvedAvgPerclos = sanitizeNum(averagePerclosPct, 0, 0, 100);
    const resolvedTotalBlinks = Math.max(0, Math.round(sanitizeNum(totalBlinks, 0)));
    const resolvedAvgBlinkRate = Math.max(0, sanitizeNum(averageBlinkRate, 0));
    const resolvedWarningAlerts = Math.max(
      0,
      Math.round(sanitizeNum(warningAlertsCount, 0))
    );
    const resolvedMirrorSuppressions = Math.max(
      0,
      Math.round(sanitizeNum(mirrorSuppressions, 0))
    );

    let resolvedEndedAt = new Date();
    if (typeof endedAt === "string" && endedAt.trim().length > 0) {
      const parsed = new Date(endedAt.trim());
      if (!Number.isNaN(parsed.getTime())) resolvedEndedAt = parsed;
    } else if (typeof endedAt === "number" && Number.isFinite(endedAt) && endedAt > 0) {
      const parsed = new Date(endedAt);
      if (!Number.isNaN(parsed.getTime())) resolvedEndedAt = parsed;
    }

    // Lock the driving_sessions row inside a transaction so concurrent or retried
    // /api/sessions/end calls finalize strictly once without corrupting counters.
    const outcome = await prisma.$transaction(async (tx) => {
      const rows = trimmedClientSessionId
        ? await tx.$queryRaw<LockedSessionRow[]>`
            SELECT
              id,
              "userId",
              "sessionCode",
              status::text AS status,
              "finalizeToken",
              "durationSeconds",
              "averageFps",
              "averageLatencyMs",
              "averagePerclosPct",
              "maxPerclosPct",
              "totalBlinks",
              "averageBlinkRate",
              "normalFramesCount",
              "warningAlertsCount",
              "criticalAlertsCount",
              "mirrorSuppressions"
            FROM public.driving_sessions
            WHERE "clientSessionId" = ${trimmedClientSessionId}
            LIMIT 1
            FOR UPDATE
          `
        : await tx.$queryRaw<LockedSessionRow[]>`
            SELECT
              id,
              "userId",
              "sessionCode",
              status::text AS status,
              "finalizeToken",
              "durationSeconds",
              "averageFps",
              "averageLatencyMs",
              "averagePerclosPct",
              "maxPerclosPct",
              "totalBlinks",
              "averageBlinkRate",
              "normalFramesCount",
              "warningAlertsCount",
              "criticalAlertsCount",
              "mirrorSuppressions"
            FROM public.driving_sessions
            WHERE "sessionCode" = ${trimmedSessionCode!}
            LIMIT 1
            FOR UPDATE
          `;

      const existing = rows[0];
      if (!existing || existing.userId !== userId) {
        return { found: false as const };
      }

      // Already finalized: return existing record idempotently without mutating stats
      if (existing.status === "COMPLETED") {
        const finalizedSession = await tx.drivingSession.findUnique({
          where: { id: existing.id },
        });
        return {
          found: true as const,
          deduplicated: true,
          session: finalizedSession,
        };
      }

      const updated = await tx.drivingSession.update({
        where: { id: existing.id },
        data: {
          status: "COMPLETED",
          finalizeToken: normalizedFinalizeToken ?? existing.finalizeToken,
          endedAt: resolvedEndedAt,
          durationSeconds: Math.max(existing.durationSeconds, Math.round(resolvedDuration)),
          averageFps: resolvedAvgFps > 0 ? resolvedAvgFps : existing.averageFps,
          averageLatencyMs: resolvedAvgLatency > 0 ? resolvedAvgLatency : existing.averageLatencyMs,
          averagePerclosPct:
            resolvedAvgPerclos > 0 ? resolvedAvgPerclos : existing.averagePerclosPct,
          maxPerclosPct: Math.max(existing.maxPerclosPct || 0, resolvedMaxPerclos),
          totalBlinks: resolvedTotalBlinks > 0 ? resolvedTotalBlinks : existing.totalBlinks,
          averageBlinkRate:
            resolvedAvgBlinkRate > 0 ? resolvedAvgBlinkRate : existing.averageBlinkRate,
          normalFramesCount: Math.max(
            existing.normalFramesCount,
            Math.round(resolvedNormalFrames)
          ),
          warningAlertsCount: Math.max(
            existing.warningAlertsCount,
            Math.round(resolvedWarningAlerts)
          ),
          criticalAlertsCount: Math.max(
            existing.criticalAlertsCount,
            Math.round(resolvedCriticalAlerts)
          ),
          mirrorSuppressions: Math.max(
            existing.mirrorSuppressions,
            Math.round(resolvedMirrorSuppressions)
          ),
        },
      });

      return {
        found: true as const,
        deduplicated: false,
        session: updated,
      };
    });

    if (!outcome.found) {
      return NextResponse.json({ error: "Session not found or forbidden" }, { status: 404 });
    }

    return NextResponse.json({
      success: true,
      deduplicated: outcome.deduplicated,
      session: outcome.session,
    });
  } catch (error) {
    console.error("Failed to end driving session:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
