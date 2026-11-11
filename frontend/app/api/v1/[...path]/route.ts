import { NextRequest, NextResponse } from "next/server";
import { PRODUCTION_FASTAPI_URL, resolveCloudFastApiUrl } from "@/lib/fastapi-url";
import { getServerSession } from "@/lib/get-session";
import { createBackendAccessToken } from "@/lib/backend-auth";

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
  }

  const primaryBase = resolveCloudFastApiUrl();
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
