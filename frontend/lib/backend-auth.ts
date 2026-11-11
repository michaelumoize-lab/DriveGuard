/**
 * DriveGuard — Next.js Server to FastAPI Backend Authentication Token Minting.
 *
 * Mints short-lived, cryptographically signed HS256 JWTs using BETTER_AUTH_SECRET.
 * FastAPI independently verifies the signature, issuer, audience, and expiration.
 */

import crypto from "crypto";

export interface BackendTokenPayload {
  sub: string;
  role?: string;
  iss: string;
  aud: string;
  exp: number;
  iat: number;
  scope: string[];
}

export function createBackendAccessToken(
  payload: { userId: string; role?: string; scope?: string[] },
  expiresInSec: number = 300
): string {
  const secret =
    process.env.BETTER_AUTH_SECRET ||
    process.env.AUTH_SECRET ||
    "driveguard-dev-secret-key-32-bytes-minimum-length-strictly-enforced";

  const now = Math.floor(Date.now() / 1000);
  const header = { alg: "HS256", typ: "JWT" };
  const claims: BackendTokenPayload = {
    sub: payload.userId,
    role: payload.role || "USER",
    iss: "driveguard-frontend",
    aud: "driveguard-backend",
    iat: now,
    exp: now + expiresInSec,
    scope: payload.scope || ["*"],
  };

  const b64Header = Buffer.from(JSON.stringify(header)).toString("base64url");
  const b64Payload = Buffer.from(JSON.stringify(claims)).toString("base64url");
  const data = `${b64Header}.${b64Payload}`;
  const signature = crypto
    .createHmac("sha256", secret)
    .update(data)
    .digest("base64url");

  return `${data}.${signature}`;
}
