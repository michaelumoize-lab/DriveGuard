import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { resolveCloudFastApiUrl } from "@/lib/fastapi-url";
import { createBackendAccessToken } from "@/lib/backend-auth";
import { buildDriverProfileDtoPayload } from "@/lib/api/profile-mapper";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      calibrationMode = "FOUR_ZONE",
      centerEarMedian,
      centerEarMadScaled = 0.018,
      blinkDurationMedianMs = 235.0,
      blinkDurationMadMs = 40.0,
      blinkRcrMedian = 0.95,
      blinkRcrMad = 0.18,
      blinkFreqMedianPerMin = 16.0,
      blinkFreqMadPerMin = 4.5,
      baselinePerclosPct = 3.5,
      blinkAmplitudeMedian = 0.14,
      temporalBlinksSampled = 0,
      temporalWindowSec = 120.0,
      sensitivityLevel = "STANDARD",
      neutralPitchDeg = 0.0,
      neutralYawDeg = 0.0,
      neutralRotMat = null,
      calibrationIsMirrored = null,
      calibrationReliability = 0.85,
      zoneBaselines = null,
    } = body;

    const numericCenterEar = Number(centerEarMedian);
    if (!Number.isFinite(numericCenterEar) || numericCenterEar <= 0.0 || numericCenterEar > 1.0) {
      return NextResponse.json(
        { error: "Invalid calibration payload: centerEarMedian must be a valid EAR value in (0, 1]." },
        { status: 400 }
      );
    }

    const baseZones =
      zoneBaselines && typeof zoneBaselines === "object" && !Array.isArray(zoneBaselines)
        ? { ...(zoneBaselines as Record<string, unknown>) }
        : {};

    const calibratedAtNow = new Date();
    const enrichedZoneBaselines: Record<string, unknown> = {
      ...baseZones,
      _temporalBaseline: {
        blinkFreqMedianPerMin: Number(blinkFreqMedianPerMin),
        blinkFreqMadPerMin: Number(blinkFreqMadPerMin),
        blinkDurationMedianMs: Number(blinkDurationMedianMs),
        blinkDurationMadMs: Number(blinkDurationMadMs),
        blinkRcrMedian: Number(blinkRcrMedian),
        blinkRcrMad: Number(blinkRcrMad),
        baselinePerclosPct: Number(baselinePerclosPct),
        blinkAmplitudeMedian: Number(blinkAmplitudeMedian),
        temporalBlinksSampled: Math.round(Number(temporalBlinksSampled)),
        temporalWindowSec: Number(temporalWindowSec),
        calibratedAt: calibratedAtNow.toISOString(),
      },
      _poseMetadata: {
        neutral_rot_mat: neutralRotMat ?? null,
        calibration_is_mirrored:
          typeof calibrationIsMirrored === "boolean" ? calibrationIsMirrored : null,
      },
    };

    const profileData = {
      calibrationMode: String(calibrationMode),
      centerEarMedian: numericCenterEar,
      centerEarMadScaled: Number(centerEarMadScaled),
      blinkDurationMedianMs: Number(blinkDurationMedianMs),
      blinkDurationMadMs: Number(blinkDurationMadMs),
      blinkRcrMedian: Number(blinkRcrMedian),
      blinkRcrMad: Number(blinkRcrMad),
      blinkFreqMedianPerMin: Number(blinkFreqMedianPerMin),
      blinkFreqMadPerMin: Number(blinkFreqMadPerMin),
      baselinePerclosPct: Number(baselinePerclosPct),
      blinkAmplitudeMedian: Number(blinkAmplitudeMedian),
      temporalBlinksSampled: Math.round(Number(temporalBlinksSampled)),
      temporalWindowSec: Number(temporalWindowSec),
      sensitivityLevel: String(sensitivityLevel),
      neutralPitchDeg: Number(neutralPitchDeg),
      neutralYawDeg: Number(neutralYawDeg),
      calibrationReliability: Number(calibrationReliability),
      zoneBaselines: enrichedZoneBaselines as any,
      isCalibrated: true,
      calibratedAt: calibratedAtNow,
    };

    const profile = await prisma.driverProfile.upsert({
      where: { userId },
      create: {
        userId,
        ...profileData,
      },
      update: {
        ...profileData,
      },
    });

    // Build strict snake_case FastAPI DriverProfileDTO using canonical shared mapper
    const fastApiProfileDto = buildDriverProfileDtoPayload(
      {
        ...profile,
        neutralRotMat,
        calibrationIsMirrored,
        zoneBaselines: enrichedZoneBaselines,
        isCalibrated: true,
        calibratedAt: calibratedAtNow,
      },
      userId
    );

    // Authenticated best-effort sync to FastAPI worker so the live session uses the latest profile immediately
    const apiBaseUrl = resolveCloudFastApiUrl();
    const role = (session.user as any)?.role || "USER";
    const backendToken = createBackendAccessToken({ userId, role });

    fetch(`${apiBaseUrl}/api/v1/profiles`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${backendToken}`,
      },
      body: JSON.stringify(fastApiProfileDto),
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
