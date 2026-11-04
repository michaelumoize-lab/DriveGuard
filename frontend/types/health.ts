export interface HealthCheckResponse {
  success: boolean;
  services: {
    nextjs: { status: "online"; latencyMs: number };
    fastapi: { status: "online" | "offline" | "error"; latencyMs?: number; message?: string };
    postgresql: { status: "connected" | "disconnected"; latencyMs?: number };
    pgvector?: { status: "available" | "unavailable"; message?: string };
    imagekit?: { status: "configured" | "misconfigured"; message?: string };
  };
}
