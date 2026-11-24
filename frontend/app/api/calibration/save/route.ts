import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";
import { Prisma, type EngineType } from "@prisma/client";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import { resolveCloudFastApiUrl } from "@/lib/fastapi-url";
import { createBackendAccessToken } from "@/lib/backend-auth";
import { buildDriverProfileDtoPayload } from "@/lib/api/profile-mapper";
import {
  computeCanonicalProfileHash,
  isValidRotationMatrix,
  validateCalibrationBiometrics,
} from "@/lib/calibration/profile-hash";
import { normalizeSensitivityLevel } from "@/lib/user-profile";

interface LockedDriverProfileRow {
  id: string;
  profileVersion: number;
  profileUuid: string;
  contentHash: string | null;
  isCalibrated: boolean;
}

function parseEngineType(raw: unknown): { valid: boolean; value: EngineType } {
  // Pre-Phase-3 GuidedCalibrationWizard omits engineType and targets Python/LiveKit calibration
  if (raw === undefined || raw === null || raw === "") {
    return { valid: true, value: "PYTHON" };
  }
  if (typeof raw !== "string") {
    return { valid: false, value: "PYTHON" };
  }
  const upper = raw.trim().toUpperCase();
  if (upper === "BROWSER" || upper === "PYTHON") {
    return { valid: true, value: upper as EngineType };
  }
  return { valid: false, value: "PYTHON" };
}

function parseOptionalClientDate(raw: unknown): Date | null {
  if (typeof raw === "string" && raw.trim().length > 0) {
    const dt = new Date(raw.trim());
    if (!Number.isNaN(dt.getTime())) return dt;
  } else if (typeof raw === "number" && Number.isFinite(raw) && raw > 0) {
    const dt = new Date(raw);
    if (!Number.isNaN(dt.getTime())) return dt;
  }
  return null;
}

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const {
      engineType: rawEngineType,
      baseServerVersion,
      clientCalibratedAt,
      isCalibrated: rawIsCalibrated = true,
      calibrationMode = "FOUR_ZONE",
      centerEarMedian,
      centerEarMadScaled = 0.018,
      blinkDurationMedianMs = 235.0,
      blinkDurationMadMs = 40.0,
      blinkRcrMedian = 0.95,
      blinkRcrMad = 0.18,
      blinkFreqMedianPerMin = 16.0,
      blinkFreqMadPerMin = 4.5,
      baselinePerclosPct = 3.5,
      blinkAmplitudeMedian = 0.14,
      temporalBlinksSampled = 0,
      temporalWindowSec = 120.0,
      sensitivityLevel = "STANDARD",
      neutralPitchDeg = 0.0,
      neutralYawDeg = 0.0,
      neutralRotMat = null,
      calibrationIsMirrored = null,
      calibrationReliability = 0.85,
      zoneBaselines = null,
    } = body;

    const parsedEngine = parseEngineType(rawEngineType);
    if (!parsedEngine.valid) {
      return NextResponse.json(
        { error: "Invalid engineType: must be 'BROWSER' or 'PYTHON'" },
        { status: 400 }
      );
    }
    const engineType = parsedEngine.value;

    let parsedBaseVersion: number | undefined;
    if (baseServerVersion !== undefined && baseServerVersion !== null) {
      const n = Number(baseServerVersion);
      if (!Number.isInteger(n) || n < 0) {
        return NextResponse.json(
          { error: "Invalid baseServerVersion: must be a non-negative integer" },
          { status: 400 }
        );
      }
      parsedBaseVersion = n;
    }

    const normalizedSensitivity =
      normalizeSensitivityLevel(String(sensitivityLevel || "STANDARD")) ?? "STANDARD";
    const incomingIsCalibrated = Boolean(rawIsCalibrated);
    const normalizedRotMat = isValidRotationMatrix(neutralRotMat) ? neutralRotMat : null;
    const normalizedMirrored =
      typeof calibrationIsMirrored === "boolean" ? calibrationIsMirrored : null;

    const baseZones =
      zoneBaselines && typeof zoneBaselines === "object" && !Array.isArray(zoneBaselines)
        ? { ...(zoneBaselines as Record<string, unknown>) }
        : {};

    const validation = validateCalibrationBiometrics({
      centerEarMedian: Number(centerEarMedian),
      calibrationReliability: Number(calibrationReliability),
      neutralRotMat: neutralRotMat === null ? null : (neutralRotMat as number[][]),
      zoneBaselines: baseZones,
    });
    if (!validation.valid) {
      return NextResponse.json({ error: validation.error }, { status: 400 });
    }

    const numericCenterEar = Number(centerEarMedian);
    const calibratedAtNow = new Date();
    const parsedClientCalibratedAt = parseOptionalClientDate(clientCalibratedAt) ?? calibratedAtNow;

    // Phase 0 additive compatibility: persist neutralRotMat & calibrationIsMirrored in
    // dedicated columns while retaining legacy _temporalBaseline & _poseMetadata in JSONB
    // until all existing readers are migrated in later phases.
    const enrichedZoneBaselines: Record<string, unknown> = {
      ...baseZones,
      _temporalBaseline: {
        blinkFreqMedianPerMin: Number(blinkFreqMedianPerMin),
        blinkFreqMadPerMin: Number(blinkFreqMadPerMin),
        blinkDurationMedianMs: Number(blinkDurationMedianMs),
        blinkDurationMadMs: Number(blinkDurationMadMs),
        blinkRcrMedian: Number(blinkRcrMedian),
        blinkRcrMad: Number(blinkRcrMad),
        baselinePerclosPct: Number(baselinePerclosPct),
        blinkAmplitudeMedian: Number(blinkAmplitudeMedian),
        temporalBlinksSampled: Math.round(Number(temporalBlinksSampled)),
        temporalWindowSec: Number(temporalWindowSec),
        calibratedAt: calibratedAtNow.toISOString(),
      },
      _poseMetadata: {
        neutral_rot_mat: normalizedRotMat,
        calibration_is_mirrored: normalizedMirrored,
      },
    };

    // Server-computed canonical SHA-256 hash (never trusts client-provided hash or clock)
    const serverContentHash = computeCanonicalProfileHash({
      engineType,
      calibrationMode: String(calibrationMode),
      isCalibrated: incomingIsCalibrated,
      centerEarMedian: numericCenterEar,
      centerEarMadScaled: Number(centerEarMadScaled),
      blinkDurationMedianMs: Number(blinkDurationMedianMs),
      blinkDurationMadMs: Number(blinkDurationMadMs),
      blinkRcrMedian: Number(blinkRcrMedian),
      blinkRcrMad: Number(blinkRcrMad),
      blinkFreqMedianPerMin: Number(blinkFreqMedianPerMin),
      blinkFreqMadPerMin: Number(blinkFreqMadPerMin),
      baselinePerclosPct: Number(baselinePerclosPct),
      blinkAmplitudeMedian: Number(blinkAmplitudeMedian),
      temporalBlinksSampled: Math.round(Number(temporalBlinksSampled)),
      temporalWindowSec: Number(temporalWindowSec),
      sensitivityLevel: normalizedSensitivity,
      neutralPitchDeg: Number(neutralPitchDeg),
      neutralYawDeg: Number(neutralYawDeg),
      neutralRotMat: normalizedRotMat,
      calibrationIsMirrored: normalizedMirrored,
      calibrationReliability: Number(calibrationReliability),
      zoneBaselines: baseZones,
    });

    const profileFields = {
      calibrationMode: String(calibrationMode),
      centerEarMedian: numericCenterEar,
      centerEarMadScaled: Number(centerEarMadScaled),
      blinkDurationMedianMs: Number(blinkDurationMedianMs),
      blinkDurationMadMs: Number(blinkDurationMadMs),
      blinkRcrMedian: Number(blinkRcrMedian),
      blinkRcrMad: Number(blinkRcrMad),
      blinkFreqMedianPerMin: Number(blinkFreqMedianPerMin),
      blinkFreqMadPerMin: Number(blinkFreqMadPerMin),
      baselinePerclosPct: Number(baselinePerclosPct),
      blinkAmplitudeMedian: Number(blinkAmplitudeMedian),
      temporalBlinksSampled: Math.round(Number(temporalBlinksSampled)),
      temporalWindowSec: Number(temporalWindowSec),
      sensitivityLevel: normalizedSensitivity,
      neutralPitchDeg: Number(neutralPitchDeg),
      neutralYawDeg: Number(neutralYawDeg),
      neutralRotMat: normalizedRotMat
        ? (normalizedRotMat as Prisma.InputJsonValue)
        : Prisma.JsonNull,
      calibrationIsMirrored: normalizedMirrored,
      calibrationReliability: Number(calibrationReliability),
      zoneBaselines: enrichedZoneBaselines as Prisma.InputJsonValue,
      isCalibrated: incomingIsCalibrated,
      calibratedAt: calibratedAtNow,
      contentHash: serverContentHash,
    };

    const txOutcome = await prisma.$transaction(async (tx) => {
      const locked = await tx.$queryRaw<LockedDriverProfileRow[]>`
        SELECT
          id,
          "profileVersion",
          "profileUuid",
          "contentHash",
          "isCalibrated"
        FROM public.driver_profiles
        WHERE "userId" = ${userId}
          AND "engineType" = ${engineType}::public."EngineType"
        LIMIT 1
        FOR UPDATE
      `;

      const existingRow = locked[0] ?? null;

      // 1. Monotonic validity guard: never downgrade an existing calibrated profile
      if (existingRow && existingRow.isCalibrated && !incomingIsCalibrated) {
        return {
          kind: "UNPROCESSABLE" as const,
          error: "Cannot overwrite an existing calibrated profile with an uncalibrated payload.",
        };
      }

      if (existingRow) {
        const fullExisting = await tx.driverProfile.findUniqueOrThrow({
          where: { id: existingRow.id },
        });

        // 2. Content-hash idempotency: identical biometric payload is a no-op
        if (existingRow.contentHash && existingRow.contentHash === serverContentHash) {
          return {
            kind: "UNCHANGED" as const,
            profile: fullExisting,
          };
        }

        // 3. Server-authoritative version check (immune to client clock skew / timestamp ties)
        if (
          parsedBaseVersion !== undefined &&
          parsedBaseVersion !== existingRow.profileVersion
        ) {
          const candidateUuid = crypto.randomUUID();
          const conflictSnapshot = {
            userId,
            engineType,
            baseServerVersion: parsedBaseVersion,
            profileUuid: candidateUuid,
            ...profileFields,
            neutralRotMat: normalizedRotMat,
            calibratedAt: parsedClientCalibratedAt.toISOString(),
          };

          const conflictHistory = await tx.driverProfileHistory.create({
            data: {
              driverProfileId: existingRow.id,
              userId,
              engineType,
              profileVersion: parsedBaseVersion,
              profileUuid: candidateUuid,
              contentHash: serverContentHash,
              snapshot: conflictSnapshot as Prisma.InputJsonValue,
              clientCalibratedAt: parsedClientCalibratedAt,
              archiveReason: "CONFLICT_PENDING_REVIEW",
            },
          });

          return {
            kind: "CONFLICT" as const,
            serverProfile: fullExisting,
            conflictHistoryId: conflictHistory.id,
          };
        }

        // 4. Fast-forward update: archive prior version before mutating active profile
        await tx.driverProfileHistory.create({
          data: {
            driverProfileId: fullExisting.id,
            userId,
            engineType,
            profileVersion: fullExisting.profileVersion,
            profileUuid: fullExisting.profileUuid,
            contentHash: fullExisting.contentHash,
            snapshot: JSON.parse(JSON.stringify(fullExisting)) as Prisma.InputJsonValue,
            clientCalibratedAt: fullExisting.calibratedAt,
            archiveReason: "RECALIBRATION",
          },
        });

        const updated = await tx.driverProfile.update({
          where: { id: fullExisting.id },
          data: {
            ...profileFields,
            profileVersion: fullExisting.profileVersion + 1,
            profileUuid: crypto.randomUUID(),
          },
        });

        return {
          kind: "SYNCED" as const,
          profile: updated,
        };
      }

      // 5. First calibration for this (userId, engineType)
      const created = await tx.driverProfile.create({
        data: {
          userId,
          engineType,
          profileVersion: 1,
          profileUuid: crypto.randomUUID(),
          ...profileFields,
        },
      });

      await tx.driverProfileHistory.create({
        data: {
          driverProfileId: created.id,
          userId,
          engineType,
          profileVersion: created.profileVersion,
          profileUuid: created.profileUuid,
          contentHash: created.contentHash,
          snapshot: JSON.parse(JSON.stringify(created)) as Prisma.InputJsonValue,
          clientCalibratedAt: parsedClientCalibratedAt,
          archiveReason: "INITIAL_CALIBRATION",
        },
      });

      return {
        kind: "SYNCED" as const,
        profile: created,
      };
    });

    if (txOutcome.kind === "UNPROCESSABLE") {
      return NextResponse.json({ error: txOutcome.error }, { status: 422 });
    }

    if (txOutcome.kind === "CONFLICT") {
      return NextResponse.json(
        {
          success: false,
          status: "CONFLICT_SERVER_AUTHORITATIVE",
          serverProfile: txOutcome.serverProfile,
          conflictHistoryId: txOutcome.conflictHistoryId,
          contentHash: serverContentHash,
        },
        { status: 409 }
      );
    }

    if (txOutcome.kind === "UNCHANGED") {
      return NextResponse.json({
        success: true,
        status: "UNCHANGED",
        profile: txOutcome.profile,
      });
    }

    // Non-blocking best-effort FastAPI sync ONLY for PYTHON engine profiles.
    // BROWSER profiles remain strictly isolated from FastAPI / Python Local.
    if (engineType === "PYTHON") {
      const fastApiProfileDto = buildDriverProfileDtoPayload(
        {
          ...txOutcome.profile,
          neutralRotMat: normalizedRotMat,
          calibrationIsMirrored: normalizedMirrored,
          zoneBaselines: enrichedZoneBaselines,
          isCalibrated: txOutcome.profile.isCalibrated,
          calibratedAt: txOutcome.profile.calibratedAt,
        },
        userId
      );

      const apiBaseUrl = resolveCloudFastApiUrl();
      const role = (session.user as { role?: string })?.role || "USER";
      const backendToken = createBackendAccessToken({ userId, role });

      fetch(`${apiBaseUrl}/api/v1/profiles`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${backendToken}`,
        },
        body: JSON.stringify(fastApiProfileDto),
        signal: AbortSignal.timeout(3000),
      }).catch(() => {
        // Non-fatal if Python worker is offline during profile save
      });
    }

    return NextResponse.json({
      success: true,
      status: "SYNCED",
      profile: txOutcome.profile,
    });
  } catch (error) {
    console.error("Failed to save calibration profile:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
