/**
 * Unit & Contract Test Suite for DriveGuard Security Guards & Telemetry Contracts
 * ===============================================================================
 * Verifies:
 * 1. Production guard on __setGetServerSessionMock
 * 2. Unauthenticated 401 enforcement on all telemetry/calibration API routes:
 *    - POST /api/calibration/save
 *    - POST /api/sessions/start
 *    - POST /api/sessions/end
 *    - POST /api/alerts/log
 * 3. Explainable AI Reason Code Catalog (explainReasonCode)
 */

import "dotenv/config";
import { NextRequest } from "next/server";
import { __setGetServerSessionMock } from "../lib/get-session";
import { POST as saveCalibrationPost } from "../app/api/calibration/save/route";
import { POST as startSessionPost } from "../app/api/sessions/start/route";
import { POST as endSessionPost } from "../app/api/sessions/end/route";
import { POST as logAlertPost } from "../app/api/alerts/log/route";
import { explainReasonCode, REASON_CODE_CATALOG } from "../types/telemetry";
import { sanitizeFastApiUrl, PRODUCTION_FASTAPI_URL } from "../lib/fastapi-url";

let passed = 0;
let total = 0;

async function it(desc: string, fn: () => void | Promise<void>) {
  total++;
  try {
    await fn();
    passed++;
    console.log(`  [PASS] ${desc}`);
  } catch (err) {
    console.error(`  [FAIL] ${desc}`);
    console.error(err);
    process.exitCode = 1;
  }
}

function expect(val: unknown) {
  return {
    toBe(expected: unknown) {
      if (val !== expected) {
        throw new Error(`Expected ${String(val)} to be ${String(expected)}`);
      }
    },
    toBeTruthy() {
      if (!val) {
        throw new Error(`Expected ${String(val)} to be truthy`);
      }
    },
  };
}

async function runTests() {
  console.log("======================================================================");
  console.log("   [DriveGuard] Security Guards & Telemetry Contract Unit Tests");
  console.log("======================================================================\n");

  // Force unauthenticated session mock
  __setGetServerSessionMock(async () => null);

  await it("1. POST /api/calibration/save returns 401 Unauthorized when unauthenticated", async () => {
    const req = new NextRequest("http://localhost:3000/api/calibration/save", {
      method: "POST",
      body: JSON.stringify({ centerEarMedian: 0.28 }),
    });
    const res = await saveCalibrationPost(req);
    expect(res.status).toBe(401);
  });

  await it("2. POST /api/sessions/start returns 401 Unauthorized when unauthenticated", async () => {
    const req = new NextRequest("http://localhost:3000/api/sessions/start", {
      method: "POST",
      body: JSON.stringify({}),
    });
    const res = await startSessionPost(req);
    expect(res.status).toBe(401);
  });

  await it("3. POST /api/sessions/end returns 401 Unauthorized when unauthenticated", async () => {
    const req = new NextRequest("http://localhost:3000/api/sessions/end", {
      method: "POST",
      body: JSON.stringify({
        sessionCode: "DG-TEST01",
        durationSeconds: 60,
        maxPerclosPct: 12.4,
      }),
    });
    const res = await endSessionPost(req);
    expect(res.status).toBe(401);
  });

  await it("4. POST /api/alerts/log returns 401 Unauthorized when unauthenticated", async () => {
    const req = new NextRequest("http://localhost:3000/api/alerts/log", {
      method: "POST",
      body: JSON.stringify({
        sessionCode: "DG-TEST01",
        severity: "CRITICAL",
        riskScore: 0.91,
      }),
    });
    const res = await logAlertPost(req);
    expect(res.status).toBe(401);
  });

  await it("5. explainReasonCode resolves all catalog reason codes with human-readable explanations", () => {
    const catalogKeys = Object.keys(REASON_CODE_CATALOG);
    expect(catalogKeys.length >= 9).toBeTruthy();

    for (const key of catalogKeys) {
      const info = explainReasonCode(key);
      expect(info.code).toBe(key);
      expect(Boolean(info.title && info.title.length > 3)).toBeTruthy();
      expect(Boolean(info.explanation && info.explanation.length > 10)).toBeTruthy();
    }
  });

  await it("6. explainReasonCode formats unknown reason codes gracefully", () => {
    const fallback = explainReasonCode("CUSTOM_DRIVER_FATIGUE_SIGNAL");
    expect(fallback.code).toBe("CUSTOM_DRIVER_FATIGUE_SIGNAL");
    expect(fallback.title).toBe("Custom Driver Fatigue Signal");
    expect(fallback.severity).toBe("WARNING");
  });

  await it("7. sanitizeFastApiUrl rewrites stale Render PR preview URLs (driveguard-pr-1.onrender.com) to production", () => {
    const sanitized = sanitizeFastApiUrl(
      "https://driveguard-pr-1.onrender.com/",
      PRODUCTION_FASTAPI_URL
    );
    expect(sanitized).toBe("https://driveguard-x048.onrender.com");
  });

  // Restore mock
  __setGetServerSessionMock(null);

  console.log("\n======================================================================");
  console.log(`   [DriveGuard Security & Contract Tests] ${passed}/${total} Passed!`);
  console.log("======================================================================\n");

  if (passed !== total) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error("Security test suite failed:", err);
  process.exit(1);
});
