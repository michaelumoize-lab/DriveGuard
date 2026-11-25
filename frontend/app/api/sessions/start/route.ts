import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";
import type { EngineType } from "@prisma/client";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

interface InsertedSessionRow {
  id: string;
  sessionCode: string;
  clientSessionId: string | null;
  engineType: string | null;
  status: string;
}

function generateCollisionResistantCode(): string {
  // Format: DRV-<timestamp_base36>-<4_random_bytes_hex>
  // High-entropy, sortable, collision-resistant code
  const timePart = Date.now().toString(36).toUpperCase();
  const randPart = crypto.randomBytes(4).toString("hex").toUpperCase();
  return `DRV-${timePart}-${randPart}`;
}

function parseOptionalEngineType(raw: unknown): { valid: boolean; value: EngineType | null } {
  if (raw === undefined || raw === null || raw === "") {
    return { valid: true, value: null };
  }
  if (typeof raw !== "string") {
    return { valid: false, value: null };
  }
  const upper = raw.trim().toUpperCase();
  if (upper === "BROWSER" || upper === "PYTHON") {
    return { valid: true, value: upper as EngineType };
  }
  return { valid: false, value: null };
}

function parseOptionalStartedAt(raw: unknown): Date {
  if (typeof raw === "number" && Number.isFinite(raw) && raw > 0) {
    const dt = new Date(raw);
    if (!Number.isNaN(dt.getTime())) return dt;
  }
  if (typeof raw === "string" && raw.trim().length > 0) {
    const dt = new Date(raw.trim());
    if (!Number.isNaN(dt.getTime())) return dt;
  }
  return new Date();
}

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json().catch(() => ({}));
    const customCode =
      typeof body.sessionCode === "string" && body.sessionCode.trim().length > 0
        ? body.sessionCode.trim()
        : null;
    const clientSessionId =
      typeof body.clientSessionId === "string" && body.clientSessionId.trim().length > 0
        ? body.clientSessionId.trim().slice(0, 128)
        : null;

    const parsedEngine = parseOptionalEngineType(body.engineType);
    if (!parsedEngine.valid) {
      return NextResponse.json(
        { error: "Invalid engineType: must be 'BROWSER' or 'PYTHON'" },
        { status: 400 }
      );
    }

    const startedAt = parseOptionalStartedAt(body.startedAt);

    // Idempotent provisional-session mapping via clientSessionId
    if (clientSessionId) {
      let attempts = 0;
      const maxAttempts = 3;

      while (attempts < maxAttempts) {
        attempts++;
        const codeToUse = customCode || generateCollisionResistantCode();
        const generatedId = `ses_${crypto.randomUUID().replace(/-/g, "")}`;

        try {
          const outcome = await prisma.$transaction(async (tx) => {
            const inserted = await tx.$queryRaw<InsertedSessionRow[]>`
              INSERT INTO public.driving_sessions (
                id,
                "userId",
                "engineType",
                "sessionCode",
                "clientSessionId",
                status,
                "startedAt",
                "createdAt",
                "updatedAt"
              )
              VALUES (
                ${generatedId},
                ${userId},
                ${parsedEngine.value}::public."EngineType",
                ${codeToUse},
                ${clientSessionId},
                'ACTIVE'::public."SessionStatus",
                ${startedAt},
                NOW(),
                NOW()
              )
              ON CONFLICT ("clientSessionId") DO NOTHING
              RETURNING id, "sessionCode", "clientSessionId", "engineType"::text AS "engineType", status::text AS status
            `;

            if (inserted.length === 1) {
              return {
                status: 200 as const,
                deduplicated: false,
                row: inserted[0],
              };
            }

            const existing = await tx.drivingSession.findUnique({
              where: { clientSessionId },
            });

            if (!existing) {
              return {
                status: 500 as const,
                error: "Unable to resolve existing driving session",
              };
            }

            if (existing.userId !== userId) {
              return {
                status: 403 as const,
                error: "Forbidden: clientSessionId belongs to another user",
              };
            }

            return {
              status: 200 as const,
              deduplicated: true,
              row: {
                id: existing.id,
                sessionCode: existing.sessionCode,
                clientSessionId: existing.clientSessionId,
                engineType: existing.engineType,
                status: existing.status,
              },
            };
          });

          if (outcome.status === 403 || outcome.status === 500) {
            return NextResponse.json({ error: outcome.error }, { status: outcome.status });
          }

          return NextResponse.json({
            success: true,
            sessionCode: outcome.row.sessionCode,
            sessionId: outcome.row.id,
            clientSessionId: outcome.row.clientSessionId,
            engineType: outcome.row.engineType,
            status: outcome.row.status,
            deduplicated: outcome.deduplicated,
          });
        } catch (err: unknown) {
          const pgCode =
            (err as { code?: string; meta?: { code?: string } })?.code ||
            (err as { meta?: { code?: string } })?.meta?.code;
          if ((pgCode === "P2002" || pgCode === "23505") && !customCode && attempts < maxAttempts) {
            continue;
          }
          throw err;
        }
      }

      return NextResponse.json(
        { error: "Failed to generate unique session code after multiple attempts" },
        { status: 500 }
      );
    }

    // Legacy path (no clientSessionId supplied)
    let drivingSession = null;
    let attempts = 0;
    const maxAttempts = 3;

    while (attempts < maxAttempts && !drivingSession) {
      attempts++;
      const codeToUse = customCode || generateCollisionResistantCode();
      try {
        drivingSession = await prisma.drivingSession.create({
          data: {
            userId,
            engineType: parsedEngine.value,
            sessionCode: codeToUse,
            status: "ACTIVE",
            startedAt,
          },
        });
      } catch (err: unknown) {
        const prismaCode = (err as { code?: string })?.code;
        if (prismaCode === "P2002" && !customCode && attempts < maxAttempts) {
          continue;
        }
        throw err;
      }
    }

    if (!drivingSession) {
      return NextResponse.json(
        { error: "Failed to generate unique session code after multiple attempts" },
        { status: 500 }
      );
    }

    return NextResponse.json({
      success: true,
      sessionCode: drivingSession.sessionCode,
      sessionId: drivingSession.id,
      clientSessionId: drivingSession.clientSessionId,
      engineType: drivingSession.engineType,
      status: drivingSession.status,
      deduplicated: false,
    });
  } catch (error) {
    console.error("Failed to start driving session:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
