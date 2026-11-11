import { NextRequest, NextResponse } from "next/server";
import { PRODUCTION_FASTAPI_URL, resolveCloudFastApiUrl } from "@/lib/fastapi-url";
import { getServerSession } from "@/lib/get-session";
import { createBackendAccessToken } from "@/lib/backend-auth";
import { prisma } from "@/lib/prisma";

export const dynamic = "force-dynamic";

async function proxyToFastApi(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
): Promise<NextResponse> {
  const { path } = await context.params;
  const subPath = Array.isArray(path) ? path.join("/") : "";
  const search = req.nextUrl.search || "";

  // Public operational paths that do not require driver session
  const isPublicHealth = subPath === "health" || subPath === "" || subPath === "health/liveness";

  let backendToken: string | null = null;
  const primaryBase = resolveCloudFastApiUrl();

  if (!isPublicHealth) {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json(
        { error: "Unauthorized: Active driver session required" },
        { status: 401 }
      );
    }
    const role = (session?.user as any)?.role || "USER";
    backendToken = createBackendAccessToken({ userId, role });

    // DG-04: Prioritize PostgreSQL calibration records across backend restarts.
    // When initializing a session or requesting a LiveKit token, synchronize the authoritative
    // profile from PostgreSQL to FastAPI so cold-started workers immediately have calibrated baselines.
    if (subPath === "livekit/token" || subPath === "sessions") {
      try {
        const dbProfile = await prisma.driverProfile.findUnique({ where: { userId } });
        if (dbProfile && dbProfile.isCalibrated) {
          const syncUrl = `${primaryBase}/api/v1/profiles`;
          const baseZones =
            dbProfile.zoneBaselines &&
            typeof dbProfile.zoneBaselines === "object" &&
            !Array.isArray(dbProfile.zoneBaselines)
              ? (dbProfile.zoneBaselines as Record<string, any>)
              : {};
          const spatialZones: Record<string, any> = {};
          for (const [k, v] of Object.entries(baseZones)) {
            if (!k.startsWith("_") && v && typeof v === "object") {
              const zv = v as Record<string, any>;
              const earMed = Number(zv.ear_median ?? dbProfile.centerEarMedian);
              const earMad = Number(zv.ear_mad_scaled ?? dbProfile.centerEarMadScaled);
              spatialZones[k] = {
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

          fetch(syncUrl, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${backendToken}`,
            },
            body: JSON.stringify({
              version: 1,
              driver_id: userId,
              calibrated_at: dbProfile.calibratedAt?.toISOString() || new Date().toISOString(),
              is_calibrated: true,
              calibration_mode: dbProfile.calibrationMode,
              center_ear_median: dbProfile.centerEarMedian,
              blink_duration_median_ms: dbProfile.blinkDurationMedianMs,
              blink_duration_mad_ms: dbProfile.blinkDurationMadMs,
              blink_rcr_median: dbProfile.blinkRcrMedian,
              blink_rcr_mad: dbProfile.blinkRcrMad,
              blink_freq_median_per_min: dbProfile.blinkFreqMedianPerMin,
              blink_freq_mad_per_min: dbProfile.blinkFreqMadPerMin,
              baseline_perclos_pct: dbProfile.baselinePerclosPct,
              blink_amplitude_median: dbProfile.blinkAmplitudeMedian,
              temporal_blinks_sampled: dbProfile.temporalBlinksSampled,
              temporal_window_sec: dbProfile.temporalWindowSec,
              sensitivity_level: dbProfile.sensitivityLevel,
              neutral_pitch_deg: dbProfile.neutralPitchDeg,
              neutral_yaw_deg: dbProfile.neutralYawDeg,
              calibration_reliability: dbProfile.calibrationReliability,
              zone_baselines: spatialZones,
            }),
            signal: AbortSignal.timeout(3000),
          }).catch((syncErr) => {
            console.warn("[API Proxy] Background profile sync warning:", syncErr);
          });
        }
      } catch (dbErr) {
        console.warn("[API Proxy] Postgres profile lookup warning:", dbErr);
      }
    }
  }

  const targetUrl = `${primaryBase}/api/v1/${subPath}${search}`;

  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) {
    headers.set("Content-Type", contentType);
  }
  headers.set("Accept", req.headers.get("accept") || "application/json");
  if (backendToken) {
    headers.set("Authorization", `Bearer ${backendToken}`);
  }

  const hasBody = !["GET", "HEAD", "OPTIONS"].includes(req.method.toUpperCase());
  const bodyBuffer = hasBody ? await req.arrayBuffer() : undefined;

  const executeUpstream = async (url: string) => {
    return fetch(url, {
      method: req.method,
      headers,
      body: bodyBuffer && bodyBuffer.byteLength > 0 ? bodyBuffer : undefined,
      signal: AbortSignal.timeout(25000),
    });
  };

  try {
    let upstreamRes = await executeUpstream(targetUrl);

    // If a custom env URL points to an inactive Render instance (`x-render-routing: no-server`),
    // automatically retry against the canonical production Render service.
    if (
      upstreamRes.status === 404 &&
      upstreamRes.headers.get("x-render-routing") === "no-server" &&
      primaryBase !== PRODUCTION_FASTAPI_URL
    ) {
      const fallbackUrl = `${PRODUCTION_FASTAPI_URL}/api/v1/${subPath}${search}`;
      upstreamRes = await executeUpstream(fallbackUrl);
    }

    const responseText = await upstreamRes.text();
    const resContentType = upstreamRes.headers.get("content-type") || "application/json";

    return new NextResponse(responseText, {
      status: upstreamRes.status,
      headers: {
        "Content-Type": resContentType,
        "Cache-Control": "no-store, no-cache, must-revalidate",
      },
    });
  } catch (error) {
    console.error(`[DriveGuard API v1 Proxy] Failed to reach ${targetUrl}:`, error);
    return NextResponse.json(
      {
        status: "ERROR",
        detail:
          error instanceof Error
            ? `AI backend unreachable: ${error.message}`
            : "AI backend unreachable",
      },
      { status: 502 }
    );
  }
}

export async function GET(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToFastApi(req, context);
}

export async function POST(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToFastApi(req, context);
}

export async function PUT(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToFastApi(req, context);
}

export async function DELETE(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToFastApi(req, context);
}

export async function PATCH(
  req: NextRequest,
  context: { params: Promise<{ path: string[] }> }
) {
  return proxyToFastApi(req, context);
}
