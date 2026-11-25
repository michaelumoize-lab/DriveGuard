/**
 * DriveGuard — Phase 4B Verification Suite:
 * LiveKit Responsibility Audit & Cloud Profile Sync Hardening.
 *
 * Verifies:
 * 1. docs/LIVEKIT_RESPONSIBILITY_AUDIT.md exists and documents all LiveKit files, routes,
 *    DataChannel message types, and migration risks.
 * 2. BFF profile sync awaits POST /api/v1/profiles BEFORE upstream GET /api/v1/livekit/token
 *    or POST /api/v1/sessions executes.
 * 3. BFF profile sync strictly ignores BROWSER profiles and uncalibrated profiles.
 * 4. BFF profile sync enforces bounded timeout and non-fatal fallback if FastAPI is unreachable.
 * 5. resolveCloudFastApiUrl() defaults local-first (http://127.0.0.1:8000) in NODE_ENV=development
 *    and uses Render production URL strictly in NODE_ENV=production.
 */

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { syncAuthoritativePythonProfileBeforeForward } from "../lib/api/bff-profile-sync";
import {
  LOCAL_FASTAPI_URL,
  LOCAL_FASTAPI_WS_URL,
  PRODUCTION_FASTAPI_URL,
  PRODUCTION_FASTAPI_WS_URL,
  resolveCloudFastApiUrl,
  resolveCloudFastApiWsUrl,
} from "../lib/fastapi-url";

let passed = 0;
let failed = 0;

async function runTest(name: string, fn: () => void | Promise<void>) {
  try {
    await fn();
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.error(`  FAIL  ${name}`);
    console.error(err);
    failed++;
  }
}

async function main() {
  console.log("\n=== Phase 4B: LiveKit Audit & Cloud Profile Sync Verification ===");

  await runTest("LIVEKIT_RESPONSIBILITY_AUDIT.md inventories all LiveKit files, routes, messages, and migration risks", () => {
    const auditPath = path.resolve(process.cwd(), "..", "docs", "LIVEKIT_RESPONSIBILITY_AUDIT.md");
    assert.ok(fs.existsSync(auditPath), `Expected audit doc at ${auditPath}`);
    const content = fs.readFileSync(auditPath, "utf-8");

    const requiredTerms = [
      "livekit_worker.py",
      "livekit_token.py",
      "livekit-session-client.ts",
      "use-livekit-telemetry-stream.ts",
      "/api/v1/livekit/token",
      "/api/v1/livekit/stop-worker",
      "TELEMETRY",
      "SAFETY_EVENT",
      "EVENT_ACK",
      "STATE_SYNC",
      "START_CALIBRATION",
      "RECALIBRATE_ZONE",
      "Migration Criteria",
    ];

    for (const term of requiredTerms) {
      assert.ok(content.includes(term), `LIVEKIT_RESPONSIBILITY_AUDIT.md missing required section/term: ${term}`);
    }
  });

  await runTest("BFF sync awaits POST /api/v1/profiles COMPLETELY before GET /api/v1/livekit/token forwards upstream", async () => {
    const order: string[] = [];
    let syncedPayload: Record<string, unknown> | null = null;

    const mockFetch = (async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input);
      if (url.endsWith("/api/v1/profiles")) {
        order.push("PROFILE_SYNC_START");
        await new Promise((r) => setTimeout(r, 60));
        syncedPayload = JSON.parse(String(init?.body ?? "{}"));
        order.push("PROFILE_SYNC_DONE");
        return new Response(JSON.stringify({ status: "OK" }), { status: 200 });
      }
      order.push("LIVEKIT_TOKEN_FORWARD");
      return new Response(JSON.stringify({ token: "lk_tok" }), { status: 200 });
    }) as typeof fetch;

    const syncRes = await syncAuthoritativePythonProfileBeforeForward({
      subPath: "livekit/token",
      userId: "user_python_1",
      primaryBase: "http://127.0.0.1:8000",
      backendToken: "jwt_test_123",
      fetchImpl: mockFetch,
      timeoutMs: 2500,
      loadPythonProfile: async () => ({
        engineType: "PYTHON",
        isCalibrated: true,
        calibrationMode: "FOUR_ZONE",
        centerEarMedian: 0.31,
        centerEarMadScaled: 0.015,
        neutralPitchDeg: 0,
        neutralYawDeg: 0,
        calibrationReliability: 0.95,
        zoneBaselines: {
          CENTER: {
            zone: "CENTER",
            ear_median: 0.31,
            ear_mad_scaled: 0.015,
            dynamic_closure_threshold: 0.21,
            sample_count: 40,
          },
        },
      }),
    });

    // Simulate proxy forwarding upstream immediately after awaiting sync
    await mockFetch("http://127.0.0.1:8000/api/v1/livekit/token?room_name=r1");

    assert.equal(syncRes.attempted, true);
    assert.equal(syncRes.synced, true);
    assert.deepEqual(order, ["PROFILE_SYNC_START", "PROFILE_SYNC_DONE", "LIVEKIT_TOKEN_FORWARD"]);
    assert.ok(syncedPayload);
    assert.equal((syncedPayload as Record<string, unknown>).engine_type, "PYTHON");
    assert.equal((syncedPayload as Record<string, unknown>).driver_id, "user_python_1");
  });

  await runTest("BFF sync strictly ignores BROWSER profiles and uncalibrated profiles", async () => {
    let fetchCalls = 0;
    const mockFetch = (async () => {
      fetchCalls++;
      return new Response("{}", { status: 200 });
    }) as typeof fetch;

    const browserRes = await syncAuthoritativePythonProfileBeforeForward({
      subPath: "livekit/token",
      userId: "user_browser_only",
      primaryBase: "http://127.0.0.1:8000",
      backendToken: "jwt_test",
      fetchImpl: mockFetch,
      loadPythonProfile: async () => ({
        engineType: "BROWSER",
        isCalibrated: true,
        centerEarMedian: 0.32,
      }),
    });
    assert.equal(browserRes.attempted, false);
    assert.equal(browserRes.skippedReason, "NON_PYTHON_ENGINE_IGNORED");
    assert.equal(fetchCalls, 0, "BROWSER profile must NEVER be synced to Python FastAPI");

    const uncalRes = await syncAuthoritativePythonProfileBeforeForward({
      subPath: "sessions",
      userId: "user_uncal",
      primaryBase: "http://127.0.0.1:8000",
      backendToken: "jwt_test",
      fetchImpl: mockFetch,
      loadPythonProfile: async () => ({
        engineType: "PYTHON",
        isCalibrated: false,
        centerEarMedian: 0.28,
      }),
    });
    assert.equal(uncalRes.attempted, false);
    assert.equal(uncalRes.skippedReason, "PROFILE_NOT_CALIBRATED");
    assert.equal(fetchCalls, 0);
  });

  await runTest("BFF sync degrades non-fatally within bounded timeout when FastAPI is unreachable", async () => {
    const startMs = Date.now();
    const res = await syncAuthoritativePythonProfileBeforeForward({
      subPath: "livekit/token",
      userId: "user_timeout",
      primaryBase: "http://127.0.0.1:8000",
      backendToken: "jwt_test",
      timeoutMs: 120,
      fetchImpl: (async (_url: RequestInfo | URL, init?: RequestInit) => {
        return new Promise((_resolve, reject) => {
          init?.signal?.addEventListener("abort", () => {
            reject(new DOMException("Aborted", "TimeoutError"));
          });
        });
      }) as typeof fetch,
      loadPythonProfile: async () => ({
        engineType: "PYTHON",
        isCalibrated: true,
        centerEarMedian: 0.30,
      }),
    });
    const elapsedMs = Date.now() - startMs;
    assert.equal(res.attempted, true);
    assert.equal(res.synced, false);
    assert.ok(elapsedMs < 1500, `Bounded timeout exceeded: ${elapsedMs}ms`);
  });

  await runTest("resolveCloudFastApiUrl defaults local-first in development and Render-only in production", () => {
    const savedEnv = {
      NODE_ENV: process.env.NODE_ENV,
      FASTAPI_URL: process.env.FASTAPI_URL,
      NEXT_PUBLIC_FASTAPI_URL: process.env.NEXT_PUBLIC_FASTAPI_URL,
      NEXT_PUBLIC_API_BASE_URL: process.env.NEXT_PUBLIC_API_BASE_URL,
      NEXT_PUBLIC_FASTAPI_WS_URL: process.env.NEXT_PUBLIC_FASTAPI_WS_URL,
    };

    try {
      delete process.env.FASTAPI_URL;
      delete process.env.NEXT_PUBLIC_FASTAPI_URL;
      delete process.env.NEXT_PUBLIC_API_BASE_URL;
      delete process.env.NEXT_PUBLIC_FASTAPI_WS_URL;

      (process.env as Record<string, string | undefined>).NODE_ENV = "development";
      assert.equal(resolveCloudFastApiUrl(), LOCAL_FASTAPI_URL);
      assert.equal(resolveCloudFastApiWsUrl(), LOCAL_FASTAPI_WS_URL);

      (process.env as Record<string, string | undefined>).NODE_ENV = "production";
      assert.equal(resolveCloudFastApiUrl(), PRODUCTION_FASTAPI_URL);
      assert.equal(resolveCloudFastApiWsUrl(), PRODUCTION_FASTAPI_WS_URL);

      // Production strips accidental localhost or stale PR preview URLs
      process.env.FASTAPI_URL = "http://127.0.0.1:8000";
      assert.equal(resolveCloudFastApiUrl(), PRODUCTION_FASTAPI_URL);

      process.env.FASTAPI_URL = "https://driveguard-pr-42.onrender.com";
      assert.equal(resolveCloudFastApiUrl(), PRODUCTION_FASTAPI_URL);
    } finally {
      for (const [k, v] of Object.entries(savedEnv)) {
        if (v === undefined) {
          delete (process.env as Record<string, string | undefined>)[k];
        } else {
          (process.env as Record<string, string | undefined>)[k] = v;
        }
      }
    }
  });

  console.log(`\nPhase 4B Verification Result: ${passed} passed, ${failed} failed\n`);
  if (failed > 0) {
    process.exit(1);
  }
}

main();
