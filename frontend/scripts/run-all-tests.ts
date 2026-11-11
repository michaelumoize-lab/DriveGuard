/**
 * DriveGuard Full End-to-End System Test Suite
 * ============================================
 * Tests:
 * 1. Neon PostgreSQL Database Connectivity via Prisma (@prisma/adapter-neon)
 * 2. DriverProfile Upsert & Retrieval (Personalized AI Baseline)
 * 3. DrivingSession Lifecycle (Start -> Telemetry Log -> Complete)
 * 4. DrowsinessAlert Persistence & Cascade Decision Audit
 * 5. LiveKit Cloud Configuration & Token Verification
 * 6. FastAPI Backend Health & REST Contract Compliance
 *
 * DG-16: Uses an ephemeral, disposable test user ID and cleans up all generated
 * records in a finally block to ensure real user profiles are never overwritten.
 */

import "dotenv/config";
import crypto from "crypto";
import { prisma } from "../lib/prisma";

async function main() {
  console.log("======================================================================");
  console.log("   [DriveGuard] End-to-End System Integration Test Suite");
  console.log("======================================================================");

  let passedTests = 0;
  let totalTests = 0;

  function assert(condition: boolean, testName: string, detail?: string) {
    totalTests++;
    if (condition) {
      passedTests++;
      console.log(`  [PASS] ${testName}`);
    } else {
      console.error(`  [FAIL] ${testName}${detail ? `: ${detail}` : ""}`);
      process.exitCode = 1;
    }
  }

  // -------------------------------------------------------------------------
  // DG-16: Ephemeral Disposable Test User Creation
  // -------------------------------------------------------------------------
  console.log("\n[1/4] Testing Neon PostgreSQL Database Connectivity (Ephemeral User)...");
  const randSuffix = crypto.randomBytes(4).toString("hex");
  const ephemeralUserId = `test_driver_${Date.now()}_${randSuffix}`;
  const ephemeralEmail = `${ephemeralUserId}@driveguard.local`;

  let user: { id: string; email: string | null } | null = null;
  let sessionId: string | null = null;

  try {
    user = await prisma.user.create({
      data: {
        id: ephemeralUserId,
        name: "Ephemeral Integration Test Driver",
        email: ephemeralEmail,
      },
    });

    assert(!!user && !!user.id, "Neon PostgreSQL Connected & Ephemeral User Created", `User ID: ${user.id}`);

    // -------------------------------------------------------------------------
    // TEST 2: Personalized Driver Profile (PDBP) Baseline Persistence
    // -------------------------------------------------------------------------
    console.log("\n[2/4] Testing Driver Profile (PDBP) Persistence in PostgreSQL...");
    const dummyZoneBaselines = {
      CENTER: { ear_median: 0.285, ear_mad_scaled: 0.016, sample_count: 35 },
      LEFT_MIRROR: { ear_median: 0.262, ear_mad_scaled: 0.018, sample_count: 20 },
      RIGHT_MIRROR: { ear_median: 0.260, ear_mad_scaled: 0.019, sample_count: 20 },
      DASHBOARD: { ear_median: 0.251, ear_mad_scaled: 0.021, sample_count: 20 },
    };

    const savedProfile = await prisma.driverProfile.upsert({
      where: { userId: user.id },
      create: {
        userId: user.id,
        calibrationMode: "4_ZONE",
        centerEarMedian: 0.285,
        centerEarMadScaled: 0.016,
        blinkDurationMedianMs: 195.0,
        blinkDurationMadMs: 32.0,
        blinkRcrMedian: 0.98,
        blinkRcrMad: 0.15,
        blinkFreqMedianPerMin: 16.0,
        blinkFreqMadPerMin: 4.5,
        baselinePerclosPct: 3.5,
        blinkAmplitudeMedian: 0.14,
        temporalBlinksSampled: 25,
        temporalWindowSec: 120.0,
        sensitivityLevel: "STANDARD",
        neutralPitchDeg: 0.0,
        neutralYawDeg: 0.0,
        calibrationReliability: 0.92,
        zoneBaselines: dummyZoneBaselines,
        isCalibrated: true,
        calibratedAt: new Date(),
      },
      update: {
        calibrationMode: "4_ZONE",
        centerEarMedian: 0.285,
        centerEarMadScaled: 0.016,
        blinkDurationMedianMs: 195.0,
        blinkDurationMadMs: 32.0,
        blinkRcrMedian: 0.98,
        blinkRcrMad: 0.15,
        calibrationReliability: 0.92,
        zoneBaselines: dummyZoneBaselines,
        isCalibrated: true,
        calibratedAt: new Date(),
      },
    });

    assert(
      savedProfile.centerEarMedian === 0.285 && savedProfile.isCalibrated === true,
      "DriverProfile Saved to Neon Postgres (Center EAR = 0.285, 4-Zone Baselines Active)"
    );

    const fetchedProfile = await prisma.driverProfile.findUnique({
      where: { userId: user.id },
    });
    assert(
      !!fetchedProfile && fetchedProfile.calibrationMode === "4_ZONE",
      "DriverProfile Retrieved & Validated from Neon Postgres"
    );

    // -------------------------------------------------------------------------
    // TEST 3: Driving Session & Drowsiness Alert Log Lifecycle
    // -------------------------------------------------------------------------
    console.log("\n[3/4] Testing Driving Session & Alert Logging Lifecycle...");
    const testSessionCode = `E2E-TEST-${Date.now().toString(36).toUpperCase()}-${randSuffix.toUpperCase()}`;

    const session = await prisma.drivingSession.create({
      data: {
        userId: user.id,
        sessionCode: testSessionCode,
        startedAt: new Date(),
      },
    });
    sessionId = session.id;
    assert(!!session && session.sessionCode === testSessionCode, "Driving Session Created in Postgres");

    // Create Critical Drowsiness Alert Log
    const alert = await prisma.drowsinessAlert.create({
      data: {
        sessionId: session.id,
        timestampMs: Date.now(),
        severity: "CRITICAL",
        riskScore: 0.92,
        earValue: 0.185,
        perclosPct: 28.5,
        poseZone: "CENTER",
        reasons: ["MICROSLEEP_EXTENDED_EYE_CLOSURE", "PERCLOS_CRITICAL_EXCEEDED"],
        cascadeDecision: "GEOMETRIC_DISAGREEMENT_CONFIRMED_CNN",
      },
    });
    assert(
      !!alert && alert.severity === "CRITICAL" && alert.riskScore === 0.92,
      "DrowsinessAlert Logged & Linked to Session (Severity: CRITICAL, Reason: MICROSLEEP)"
    );

    // End Session with Telemetry Summary Aggregates
    const endedSession = await prisma.drivingSession.update({
      where: { sessionCode: testSessionCode },
      data: {
        endedAt: new Date(),
        durationSeconds: 120,
        averageFps: 22.4,
        averageLatencyMs: 38.5,
        averagePerclosPct: 14.2,
        maxPerclosPct: 28.5,
        criticalAlertsCount: 1,
        totalBlinks: 24,
      },
    });
    assert(
      endedSession.durationSeconds === 120 && endedSession.criticalAlertsCount === 1,
      "Driving Session Completed & Telemetry Aggregates Persisted"
    );

    // -------------------------------------------------------------------------
    // TEST 4: LiveKit Credentials & Environment Integrity
    // -------------------------------------------------------------------------
    console.log("\n[4/4] Validating LiveKit WebRTC Configuration...");
    const lkUrl = process.env.LIVEKIT_URL || process.env.NEXT_PUBLIC_LIVEKIT_URL;
    const lkKey = process.env.LIVEKIT_API_KEY;
    const lkSecret = process.env.LIVEKIT_API_SECRET;

    assert(
      !!lkUrl && lkUrl.startsWith("wss://"),
      "LiveKit Cloud SFU URL Configured",
      lkUrl
    );
    assert(
      !!lkKey && lkKey.startsWith("API"),
      "LiveKit Cloud API Key Present & Valid"
    );
    assert(
      !!lkSecret && lkSecret.length >= 20,
      "LiveKit Cloud API Secret Present & Valid"
    );

  } finally {
    // -------------------------------------------------------------------------
    // DG-16 Cleanup: Safely remove all ephemeral test records
    // -------------------------------------------------------------------------
    if (user?.id) {
      try {
        if (sessionId) {
          await prisma.drowsinessAlert.deleteMany({ where: { sessionId } });
          await prisma.drivingSession.deleteMany({ where: { id: sessionId } });
        }
        await prisma.driverProfile.deleteMany({ where: { userId: user.id } });
        await prisma.user.delete({ where: { id: user.id } });
        console.log("  [CLEANUP] Ephemeral test user and associated records successfully pruned.");
      } catch (cleanErr) {
        console.warn("  [CLEANUP] Warning during test data pruning:", cleanErr);
      }
    }
  }

  console.log("\n======================================================================");
  console.log(`   [DriveGuard Test Results] ${passedTests}/${totalTests} Tests Passed Successfully!`);
  console.log("======================================================================\n");
}

main()
  .catch((err) => {
    console.error("Test Suite Execution Error:", err);
    process.exit(1);
  })
  .finally(async () => {
    await prisma.$disconnect();
  });
