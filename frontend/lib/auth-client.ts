import { createAuthClient } from "better-auth/react";
import type { auth } from "@/lib/auth";
import {
  inferAdditionalFields,
  adminClient,
  multiSessionClient,
} from "better-auth/client/plugins";
import { lastLoginMethodClient } from "better-auth/client/plugins"

export const authClient = createAuthClient({
  plugins: [
    inferAdditionalFields<typeof auth>(),
    adminClient(),
    multiSessionClient(),
    lastLoginMethodClient(),
  ],
});

export type AuthClient = typeof authClient;

export const { signIn, signUp, signOut, useSession } = authClient;
export type SessionData = Awaited<ReturnType<typeof useSession>>["data"];
