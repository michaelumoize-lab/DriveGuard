import { betterAuth } from "better-auth";
import { prismaAdapter } from "better-auth/adapters/prisma";
import { prisma } from "@/lib/prisma";
import { admin, multiSession } from "better-auth/plugins";
import { nextCookies } from "better-auth/next-js";
import { lastLoginMethod } from "better-auth/plugins";
import { passkey } from "@better-auth/passkey";

const BETTER_AUTH_SECRET = process.env.BETTER_AUTH_SECRET;
const APP_URL = process.env.BETTER_AUTH_URL ?? "http://localhost:3000";

if (!BETTER_AUTH_SECRET) {
  throw new Error("BETTER_AUTH_SECRET environment variable is required");
}

const getHostFromUrl = (urlStr?: string): string[] => {
  if (!urlStr) return [];
  try {
    const parsed = new URL(urlStr.includes("://") ? urlStr : `https://${urlStr}`);
    return [parsed.host];
  } catch {
    return [];
  }
};

const appHosts = [
  ...getHostFromUrl(process.env.BETTER_AUTH_URL),
  ...getHostFromUrl(process.env.NEXT_PUBLIC_APP_URL),
];

const allowedHosts = Array.from(
  new Set(
    [
      "localhost:3000",
      "127.0.0.1:3000",
      ...(process.env.VERCEL_PROJECT_PRODUCTION_URL ? [process.env.VERCEL_PROJECT_PRODUCTION_URL] : []),
      ...(process.env.VERCEL_URL ? [process.env.VERCEL_URL] : []),
      ...appHosts,
    ].filter(Boolean) as string[],
  ),
);

export const auth = betterAuth({
  database: prismaAdapter(prisma, {
    provider: "postgresql",
  }),
  secret: BETTER_AUTH_SECRET,
  baseURL: {
    allowedHosts,
    protocol: process.env.NODE_ENV === "production" ? "https" : "auto",
    fallback: APP_URL,
  },

  plugins: [
    admin({
      defaultRole: "USER",
      adminRole: "ADMIN",
    }),
    multiSession(),
    lastLoginMethod(),
    passkey({
      rpName: "DriveGuard In-Cabin Biometrics",
    }),
    nextCookies(),
  ],

  rateLimit: {
    enabled: true,
    window: 10,
    max: 20,
  },

  trustedOrigins: [
    APP_URL,
    "*.vercel.app",
    "https://*.vercel.app",
  ],

  emailAndPassword: {
    enabled: true,
  },

  user: {
    deleteUser: {
      enabled: true,
    },
  },

  databaseHooks: {
    user: {
      create: {
        after: async (user) => {
          try {
            await prisma.userProfile.upsert({
              where: { userId: user.id },
              update: {},
              create: {
                userId: user.id,
                onboardingCompleted: false,
                onboardingStep: 1,
                completedSteps: [],
              },
            });
          } catch (err) {
            console.error(
              "Failed to initialize UserProfile on registration:",
              err,
            );
          }
        },
      },
    },
  },

  socialProviders: {
    ...(process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET
      ? {
          google: {
            clientId: process.env.GOOGLE_CLIENT_ID,
            clientSecret: process.env.GOOGLE_CLIENT_SECRET,
          },
        }
      : {}),
  },

  advanced: {
    disableErrorPage: true,
    defaultCookieAttributes: {
      sameSite: "lax",
      secure: process.env.NODE_ENV === "production",
      httpOnly: true,
    },
    onAPIError: {
      disableErrorPage: true,
    },
  },
});

// TypeScript exports for frontend and backend use
export type Auth = typeof auth;
export type Session = typeof auth.$Infer.Session;
export type User = typeof auth.$Infer.Session.user;
