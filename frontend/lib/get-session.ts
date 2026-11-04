import { headers } from "next/headers";
import { auth } from "./auth";

export type ServerSessionResult = Awaited<ReturnType<typeof auth.api.getSession>>;

let getServerSessionMock: (() => Promise<ServerSessionResult | null>) | null = null;

export function __setGetServerSessionMock(
  mockFn: (() => Promise<ServerSessionResult | null>) | null
) {
  getServerSessionMock = mockFn;
}

export async function getServerSession(): Promise<ServerSessionResult | null> {
  if (getServerSessionMock) {
    return await getServerSessionMock();
  }
  try {
    const reqHeaders = await headers();
    return await auth.api.getSession({ headers: reqHeaders });
  } catch (error) {
    if (
      error &&
      typeof error === "object" &&
      "digest" in error &&
      (error as { digest?: string }).digest === "DYNAMIC_SERVER_USAGE"
    ) {
      throw error;
    }
    if (
      error instanceof Error &&
      error.message.includes("outside a request scope")
    ) {
      try {
        return await auth.api.getSession({ headers: new Headers() });
      } catch {
        return null;
      }
    }
    return null;
  }
}
