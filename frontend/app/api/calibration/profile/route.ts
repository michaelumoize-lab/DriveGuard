import { NextRequest, NextResponse } from "next/server";
import type { EngineType } from "@prisma/client";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const rawEngine = req.nextUrl.searchParams.get("engineType");
    const normalized = rawEngine ? rawEngine.trim().toUpperCase() : "ALL";

    if (normalized !== "ALL" && normalized !== "BROWSER" && normalized !== "PYTHON") {
      return NextResponse.json(
        { error: "Invalid engineType query parameter: expected BROWSER, PYTHON, or ALL" },
        { status: 400 }
      );
    }

    if (normalized === "ALL") {
      const [profiles, pendingConflicts] = await Promise.all([
        prisma.driverProfile.findMany({ where: { userId } }),
        prisma.driverProfileHistory.findMany({
          where: {
            userId,
            archiveReason: "CONFLICT_PENDING_REVIEW",
          },
          orderBy: { serverRecordedAt: "desc" },
          take: 20,
        }),
      ]);

      return NextResponse.json({
        success: true,
        browser: profiles.find((p) => p.engineType === "BROWSER") ?? null,
        python: profiles.find((p) => p.engineType === "PYTHON") ?? null,
        pendingConflicts: {
          browser: pendingConflicts.filter((c) => c.engineType === "BROWSER"),
          python: pendingConflicts.filter((c) => c.engineType === "PYTHON"),
        },
      });
    }

    const engineType = normalized as EngineType;
    const [profile, pendingConflicts] = await Promise.all([
      prisma.driverProfile.findUnique({
        where: { userId_engineType: { userId, engineType } },
      }),
      prisma.driverProfileHistory.findMany({
        where: {
          userId,
          engineType,
          archiveReason: "CONFLICT_PENDING_REVIEW",
        },
        orderBy: { serverRecordedAt: "desc" },
        take: 10,
      }),
    ]);

    return NextResponse.json({
      success: true,
      engineType,
      profile,
      pendingConflicts,
    });
  } catch (error) {
    console.error("Failed to fetch calibration profile:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
