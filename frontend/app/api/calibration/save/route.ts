import { NextRequest, NextResponse } from "next/server";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { resolveCloudFastApiUrl } from "@/lib/fastapi-url";
import { createBackendAccessToken } from "@/lib/backend-auth";

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      calibrationMode = "FOUR_ZONE_PLUS_BLINK",
      centerEarMedian = 0.285,
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
      calibrationReliability = 0.85,
      zoneBaselines = null,
    } = body;

    const baseZones =
      zoneBaselines && typeof zoneBaselines === "object" && !Array.isArray(zoneBaselines)
        ? { ...(zoneBaselines as Record<string, any>) }
        : {};

    const enrichedZoneBaselines = {
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
        calibratedAt: new Date().toISOString(),
      },
    };

    const profileData = {
      calibrationMode: String(calibrationMode),
      centerEarMedian: Number(centerEarMedian),
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
      zoneBaselines: enrichedZoneBaselines,
      isCalibrated: true,
      calibratedAt: new Date(),
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

    // Extract pure spatial zones for FastAPI DriverProfileDTO validation
    const spatialZonesForWorker: Record<string, any> = {};
    for (const [k, v] of Object.entries(baseZones)) {
      if (!k.startsWith("_") && v && typeof v === "object") {
        const zv = v as Record<string, any>;
        const earMed = Number(zv.ear_median ?? centerEarMedian);
        const earMad = Number(zv.ear_mad_scaled ?? centerEarMadScaled);
        spatialZonesForWorker[k] = {
          zone: k,
          ear_median: earMed,
          ear_mad_scaled: earMad,
          dynamic_closure_threshold: Number(
            zv.dynamic_closure_threshold ?? Math.max(0.12, Math.min(0.22, earMed - 2.3 * earMad))
          ),
          pitch_median_deg: Number(zv.pitch_median_deg ?? zv.pitch_deg_median ?? 0.0),
          yaw_median_deg: Number(zv.yaw_median_deg ?? zv.yaw_deg_median ?? 0.0),
          sample_count: Number(zv.sample_count ?? 35),
        };
      }
    }

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
      body: JSON.stringify({
        version: 1,
        driver_id: userId,
        calibrated_at: new Date().toISOString(),
        is_calibrated: true,
        calibration_mode: profile.calibrationMode,
        center_ear_median: profile.centerEarMedian,
        blink_duration_median_ms: profile.blinkDurationMedianMs,
        blink_duration_mad_ms: profile.blinkDurationMadMs,
        blinkRcrMedian: profile.blinkRcrMedian,
        blinkRcrMad: profile.blinkRcrMad,
        blink_freq_median_per_min: Number(profile.blinkFreqMedianPerMin),
        blink_freq_mad_per_min: Number(profile.blinkFreqMadPerMin),
        baseline_perclos_pct: Number(profile.baselinePerclosPct),
        blink_amplitude_median: Number(profile.blinkAmplitudeMedian),
        temporal_blinks_sampled: Number(profile.temporalBlinksSampled),
        temporal_window_sec: Number(profile.temporalWindowSec),
        sensitivity_level: profile.sensitivityLevel,
        neutral_pitch_deg: profile.neutralPitchDeg,
        neutral_yaw_deg: profile.neutralYawDeg,
        calibration_reliability: profile.calibrationReliability,
        zone_baselines: spatialZonesForWorker,
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
