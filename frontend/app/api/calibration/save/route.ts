import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    let userId = session?.user?.id;
    if (!userId) {
      const fallbackUser = await prisma.user.findFirst();
      if (fallbackUser) {
        userId = fallbackUser.id;
      } else {
        return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
      }
    }

    const body = await req.json();
    const {
      calibrationMode = "3_ZONE",
      centerEarMedian = 0.285,
      centerEarMadScaled = 0.018,
      blinkDurationMedianMs = 185.0,
      blinkDurationMadMs = 35.0,
      blinkRcrMedian = 0.92,
      blinkRcrMad = 0.14,
      neutralPitchDeg = 0.0,
      neutralYawDeg = 0.0,
      calibrationReliability = 0.85,
      zoneBaselines = null,
    } = body;

    const profile = await prisma.driverProfile.upsert({
      where: { userId },
      create: {
        userId,
        calibrationMode,
        centerEarMedian: Number(centerEarMedian),
        centerEarMadScaled: Number(centerEarMadScaled),
        blinkDurationMedianMs: Number(blinkDurationMedianMs),
        blinkDurationMadMs: Number(blinkDurationMadMs),
        blinkRcrMedian: Number(blinkRcrMedian),
        blinkRcrMad: Number(blinkRcrMad),
        neutralPitchDeg: Number(neutralPitchDeg),
        neutralYawDeg: Number(neutralYawDeg),
        calibrationReliability: Number(calibrationReliability),
        zoneBaselines: zoneBaselines || undefined,
        isCalibrated: true,
        calibratedAt: new Date(),
      },
      update: {
        calibrationMode,
        centerEarMedian: Number(centerEarMedian),
        centerEarMadScaled: Number(centerEarMadScaled),
        blinkDurationMedianMs: Number(blinkDurationMedianMs),
        blinkDurationMadMs: Number(blinkDurationMadMs),
        blinkRcrMedian: Number(blinkRcrMedian),
        blinkRcrMad: Number(blinkRcrMad),
        neutralPitchDeg: Number(neutralPitchDeg),
        neutralYawDeg: Number(neutralYawDeg),
        calibrationReliability: Number(calibrationReliability),
        zoneBaselines: zoneBaselines || undefined,
        isCalibrated: true,
        calibratedAt: new Date(),
      },
    });

    return NextResponse.json({ success: true, profile });
  } catch (error) {
    console.error("Failed to save calibration profile:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
