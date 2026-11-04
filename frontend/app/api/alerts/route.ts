import { NextRequest, NextResponse } from "next/server";
import { prisma } from "@/lib/prisma";
import { getServerSession } from "@/lib/get-session";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const session = await getServerSession();
  if (!session || !session.user) {
    return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
  }

  try {
    const { searchParams } = new URL(req.url);
    const severity = searchParams.get("severity")?.toUpperCase();
    const limit = Math.min(100, parseInt(searchParams.get("limit") || "50", 10));

    const alerts = await prisma.drowsinessAlert.findMany({
      where: {
        session: { userId: session.user.id },
        ...(severity && ["WARNING", "CRITICAL"].includes(severity) ? { severity } : {}),
      },
      orderBy: { createdAt: "desc" },
      take: limit,
      include: {
        session: {
          select: {
            sessionCode: true,
            startedAt: true,
          },
        },
      },
    });

    return NextResponse.json({
      success: true,
      alerts,
    });
  } catch (error) {
    console.error("Failed to fetch drowsiness alerts:", error);
    return NextResponse.json(
      { error: "Internal server error fetching alerts" },
      { status: 500 }
    );
  }
}
