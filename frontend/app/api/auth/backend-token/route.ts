import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { createBackendAccessToken } from "@/lib/backend-auth";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json(
        { error: "Unauthorized: Active driver session required" },
        { status: 401 }
      );
    }

    const role = (session?.user as any)?.role || "USER";
    const token = createBackendAccessToken({ userId, role }, 300); // 5 min TTL

    return NextResponse.json({
      token,
      userId,
      expiresIn: 300,
    });
  } catch (error) {
    return NextResponse.json(
      { error: "Failed to generate backend token" },
      { status: 500 }
    );
  }
}
