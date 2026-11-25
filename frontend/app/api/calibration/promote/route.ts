import { NextRequest, NextResponse } from "next/server";
import crypto from "crypto";
import { Prisma } from "@prisma/client";
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
}

export async function POST(req: NextRequest) {
  try {
    const session = await getServerSession();
    const userId = session?.user?.id;
    if (!userId) {
      return NextResponse.json({ error: "Unauthorized" }, { status: 401 });
    }

    const body = await req.json();
    const conflictHistoryId =
      typeof body.conflictHistoryId === "string" && body.conflictHistoryId.trim().length > 0
        ? body.conflictHistoryId.trim()
        : null;
    const expectedServerVersion = Number(body.expectedServerVersion);

    if (!conflictHistoryId) {
      return NextResponse.json(
        { error: "conflictHistoryId is required" },
        { status: 400 }
      );
    }

    if (!Number.isInteger(expectedServerVersion) || expectedServerVersion < 1) {
      return NextResponse.json(
        { error: "expectedServerVersion must be a positive integer" },
        { status: 400 }
      );
    }

    const outcome = await prisma.$transaction(async (tx) => {
      const historyRow = await tx.driverProfileHistory.findUnique({
        where: { id: conflictHistoryId },
      });

      if (!historyRow || historyRow.userId !== userId) {
        return { kind: "NOT_FOUND" as const, error: "Conflict history record not found" };
      }

      if (historyRow.archiveReason !== "CONFLICT_PENDING_REVIEW") {
        return {
          kind: "CONFLICT" as const,
          status: "CONFLICT_ALREADY_RESOLVED",
          error: "This calibration conflict has already been resolved.",
        };
      }

      const locked = await tx.$queryRaw<LockedDriverProfileRow[]>`
        SELECT id, "profileVersion"
        FROM public.driver_profiles
        WHERE "userId" = ${userId}
          AND "engineType" = ${historyRow.engineType}::public."EngineType"
        LIMIT 1
        FOR UPDATE
      `;

      const activeMeta = locked[0];
      if (!activeMeta) {
        return { kind: "NOT_FOUND" as const, error: "Active driver profile not found" };
      }

      if (activeMeta.profileVersion !== expectedServerVersion) {
        return {
          kind: "STALE_VERSION" as const,
          currentServerVersion: activeMeta.profileVersion,
        };
      }

      const fullActive = await tx.driverProfile.findUniqueOrThrow({
        where: { id: activeMeta.id },
      });

      const snap =
        historyRow.snapshot &&
        typeof historyRow.snapshot === "object" &&
        !Array.isArray(historyRow.snapshot)
          ? (historyRow.snapshot as Record<string, unknown>)
          : null;

      if (!snap) {
        return { kind: "INVALID" as const, error: "Archived conflict snapshot is malformed" };
      }

      const snapZones =
        snap.zoneBaselines &&
        typeof snap.zoneBaselines === "object" &&
        !Array.isArray(snap.zoneBaselines)
          ? (snap.zoneBaselines as Record<string, unknown>)
          : {};
      const snapRotMat = isValidRotationMatrix(snap.neutralRotMat) ? snap.neutralRotMat : null;
      const snapMirrored =
        typeof snap.calibrationIsMirrored === "boolean" ? snap.calibrationIsMirrored : null;
      const snapSensitivity =
        normalizeSensitivityLevel(String(snap.sensitivityLevel ?? "STANDARD")) ?? "STANDARD";

      const validation = validateCalibrationBiometrics({
        centerEarMedian: Number(snap.centerEarMedian),
        calibrationReliability: Number(snap.calibrationReliability ?? 0.85),
        neutralRotMat: snapRotMat,
        zoneBaselines: snapZones,
      });
      if (!validation.valid) {
        return { kind: "INVALID" as const, error: validation.error || "Invalid snapshot biometrics" };
      }

      const serverHash = computeCanonicalProfileHash({
        engineType: historyRow.engineType,
        calibrationMode: String(snap.calibrationMode ?? fullActive.calibrationMode),
        isCalibrated: true,
        centerEarMedian: Number(snap.centerEarMedian),
        centerEarMadScaled: Number(snap.centerEarMadScaled ?? fullActive.centerEarMadScaled),
        blinkDurationMedianMs: Number(
          snap.blinkDurationMedianMs ?? fullActive.blinkDurationMedianMs
        ),
        blinkDurationMadMs: Number(snap.blinkDurationMadMs ?? fullActive.blinkDurationMadMs),
        blinkRcrMedian: Number(snap.blinkRcrMedian ?? fullActive.blinkRcrMedian),
        blinkRcrMad: Number(snap.blinkRcrMad ?? fullActive.blinkRcrMad),
        blinkFreqMedianPerMin: Number(
          snap.blinkFreqMedianPerMin ?? fullActive.blinkFreqMedianPerMin
        ),
        blinkFreqMadPerMin: Number(snap.blinkFreqMadPerMin ?? fullActive.blinkFreqMadPerMin),
        baselinePerclosPct: Number(snap.baselinePerclosPct ?? fullActive.baselinePerclosPct),
        blinkAmplitudeMedian: Number(
          snap.blinkAmplitudeMedian ?? fullActive.blinkAmplitudeMedian
        ),
        temporalBlinksSampled: Math.round(
          Number(snap.temporalBlinksSampled ?? fullActive.temporalBlinksSampled)
        ),
        temporalWindowSec: Number(snap.temporalWindowSec ?? fullActive.temporalWindowSec),
        sensitivityLevel: snapSensitivity,
        neutralPitchDeg: Number(snap.neutralPitchDeg ?? fullActive.neutralPitchDeg),
        neutralYawDeg: Number(snap.neutralYawDeg ?? fullActive.neutralYawDeg),
        neutralRotMat: snapRotMat,
        calibrationIsMirrored: snapMirrored,
        calibrationReliability: Number(
          snap.calibrationReliability ?? fullActive.calibrationReliability
        ),
        zoneBaselines: snapZones,
      });

      // Archive the currently active server version before promoting the user's chosen conflict
      await tx.driverProfileHistory.create({
        data: {
          driverProfileId: fullActive.id,
          userId,
          engineType: fullActive.engineType,
          profileVersion: fullActive.profileVersion,
          profileUuid: fullActive.profileUuid,
          contentHash: fullActive.contentHash,
          snapshot: JSON.parse(JSON.stringify(fullActive)) as Prisma.InputJsonValue,
          clientCalibratedAt: fullActive.calibratedAt,
          archiveReason: "SUPERSEDED_BY_CONFLICT_PROMOTION",
        },
      });

      const promoted = await tx.driverProfile.update({
        where: { id: fullActive.id },
        data: {
          profileVersion: fullActive.profileVersion + 1,
          profileUuid: crypto.randomUUID(),
          contentHash: serverHash,
          calibrationMode: String(snap.calibrationMode ?? fullActive.calibrationMode),
          centerEarMedian: Number(snap.centerEarMedian),
          centerEarMadScaled: Number(snap.centerEarMadScaled ?? fullActive.centerEarMadScaled),
          blinkDurationMedianMs: Number(
            snap.blinkDurationMedianMs ?? fullActive.blinkDurationMedianMs
          ),
          blinkDurationMadMs: Number(snap.blinkDurationMadMs ?? fullActive.blinkDurationMadMs),
          blinkRcrMedian: Number(snap.blinkRcrMedian ?? fullActive.blinkRcrMedian),
          blinkRcrMad: Number(snap.blinkRcrMad ?? fullActive.blinkRcrMad),
          blinkFreqMedianPerMin: Number(
            snap.blinkFreqMedianPerMin ?? fullActive.blinkFreqMedianPerMin
          ),
          blinkFreqMadPerMin: Number(snap.blinkFreqMadPerMin ?? fullActive.blinkFreqMadPerMin),
          baselinePerclosPct: Number(snap.baselinePerclosPct ?? fullActive.baselinePerclosPct),
          blinkAmplitudeMedian: Number(
            snap.blinkAmplitudeMedian ?? fullActive.blinkAmplitudeMedian
          ),
          temporalBlinksSampled: Math.round(
            Number(snap.temporalBlinksSampled ?? fullActive.temporalBlinksSampled)
          ),
          temporalWindowSec: Number(snap.temporalWindowSec ?? fullActive.temporalWindowSec),
          sensitivityLevel: snapSensitivity,
          neutralPitchDeg: Number(snap.neutralPitchDeg ?? fullActive.neutralPitchDeg),
          neutralYawDeg: Number(snap.neutralYawDeg ?? fullActive.neutralYawDeg),
          neutralRotMat: snapRotMat
            ? (snapRotMat as Prisma.InputJsonValue)
            : Prisma.JsonNull,
          calibrationIsMirrored: snapMirrored,
          calibrationReliability: Number(
            snap.calibrationReliability ?? fullActive.calibrationReliability
          ),
          zoneBaselines: snapZones as Prisma.InputJsonValue,
          isCalibrated: true,
          calibratedAt: new Date(),
        },
      });

      // Mark the promoted conflict history entry so it cannot be re-promoted
      await tx.driverProfileHistory.update({
        where: { id: historyRow.id },
        data: { archiveReason: "PROMOTED_BY_USER" },
      });

      return {
        kind: "PROMOTED" as const,
        profile: promoted,
      };
    });

    if (outcome.kind === "NOT_FOUND") {
      return NextResponse.json({ error: outcome.error }, { status: 404 });
    }
    if (outcome.kind === "INVALID") {
      return NextResponse.json({ error: outcome.error }, { status: 422 });
    }
    if (outcome.kind === "STALE_VERSION") {
      return NextResponse.json(
        {
          success: false,
          status: "STALE_EXPECTED_VERSION",
          currentServerVersion: outcome.currentServerVersion,
        },
        { status: 409 }
      );
    }
    if (outcome.kind === "CONFLICT") {
      return NextResponse.json(
        { success: false, status: outcome.status, error: outcome.error },
        { status: 409 }
      );
    }

    // Best-effort non-blocking FastAPI sync ONLY when promoting a PYTHON profile
    if (outcome.profile.engineType === "PYTHON") {
      const fastApiProfileDto = buildDriverProfileDtoPayload(
        {
          ...outcome.profile,
          neutralRotMat: isValidRotationMatrix(outcome.profile.neutralRotMat)
            ? outcome.profile.neutralRotMat
            : null,
          zoneBaselines:
            outcome.profile.zoneBaselines &&
            typeof outcome.profile.zoneBaselines === "object" &&
            !Array.isArray(outcome.profile.zoneBaselines)
              ? (outcome.profile.zoneBaselines as Record<string, unknown>)
              : null,
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
        // Non-fatal if Python worker is offline during profile promotion
      });
    }

    return NextResponse.json({
      success: true,
      status: "PROMOTED",
      profile: outcome.profile,
    });
  } catch (error) {
    console.error("Failed to promote calibration conflict:", error);
    return NextResponse.json(
      { error: "Internal Server Error" },
      { status: 500 }
    );
  }
}
