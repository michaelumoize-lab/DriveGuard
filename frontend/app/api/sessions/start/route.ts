import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

function generateCollisionResistantCode(): string {
  // Format: DRV-<timestamp_base36>-<4_random_bytes_hex>
  // High-entropy, sortable, collision-resistant code
  const timePart = Date.now().toString(36).toUpperCase();
  const randPart = crypto.randomBytes(4).toString("hex").toUpperCase();
  return `DRV-${timePart}-${randPart}`;
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
            sessionCode: codeToUse,
            startedAt: new Date(),
          },
        });
      } catch (err: any) {
        // If unique constraint collision on auto-generated code, retry with a fresh random code
        if (err?.code === "P2002" && !customCode && attempts < maxAttempts) {
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
    });
  } catch (error) {
    console.error("Failed to start driving session:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
