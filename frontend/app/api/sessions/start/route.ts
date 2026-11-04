import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    if (!session?.user?.id) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json().catch(() => ({}));
    const customCode = body.sessionCode || `DRV-${Date.now().toString().slice(-6)}`;

    const drivingSession = await prisma.drivingSession.create({
      data: {
        userId: session.user.id,
        sessionCode: customCode,
        startedAt: new Date(),
      },
    });

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
