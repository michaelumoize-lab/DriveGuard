import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { resolveCloudFastApiUrl } from "@/lib/fastapi-url";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      calibrationMode = "4_ZONE",
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
        calibrationMode: String(calibrationMode),
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
        calibrationMode: String(calibrationMode),
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

    // Best-effort sync to FastAPI worker so the live session uses the latest profile immediately
    const apiBaseUrl = resolveCloudFastApiUrl();

    fetch(`${apiBaseUrl}/api/v1/profiles`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        driver_id: userId,
        calibration_mode: profile.calibrationMode,
        ear_median: profile.centerEarMedian,
        ear_mad_scaled: profile.centerEarMadScaled,
        blink_duration_median_ms: profile.blinkDurationMedianMs,
        blink_duration_mad_ms: profile.blinkDurationMadMs,
        blink_rcr_median: profile.blinkRcrMedian,
        blink_rcr_mad: profile.blinkRcrMad,
        neutral_pitch_deg: profile.neutralPitchDeg,
        neutral_yaw_deg: profile.neutralYawDeg,
        calibration_reliability: profile.calibrationReliability,
        zone_baselines: profile.zoneBaselines ?? {},
      }),
    }).catch(() => {
      // Non-fatal if worker is offline during profile save
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
