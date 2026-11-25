/**
 * DriveGuard — Awaited, Strictly PYTHON-Scoped BFF Profile Sync Helper (Phase 4B).
 *
 * Eliminates the cold-start race where `GET /api/v1/livekit/token` or `POST /api/v1/sessions`
 * spawned a Python AI worker before PostgreSQL's calibrated `PYTHON` profile finished syncing
 * to FastAPI (`POST /api/v1/profiles`).
 *
 * Safety invariants:
 * - Strictly syncs ONLY calibrated profiles where `engineType === "PYTHON"` (never `BROWSER`).
 * - Always `await`ed before upstream request forwarding, bounded by a 2500ms timeout.
 * - Non-fatal fallback if FastAPI is unreachable or times out during wake-up.
 */

import { buildDriverProfileDtoPayload } from "@/lib/api/profile-mapper";

export interface SyncableDbDriverProfile {
  engineType?: string | null;
  isCalibrated: boolean;
  calibrationMode?: string | null;
  centerEarMedian: number;
  centerEarMadScaled?: number | null;
  blinkDurationMedianMs?: number | null;
  blinkDurationMadMs?: number | null;
  blinkRcrMedian?: number | null;
  blinkRcrMad?: number | null;
  blinkFreqMedianPerMin?: number | null;
  blinkFreqMadPerMin?: number | null;
  baselinePerclosPct?: number | null;
  blinkAmplitudeMedian?: number | null;
  temporalBlinksSampled?: number | null;
  temporalWindowSec?: number | null;
  sensitivityLevel?: string | null;
  neutralPitchDeg?: number | null;
  neutralYawDeg?: number | null;
  neutralRotMat?: unknown;
  calibrationIsMirrored?: boolean | null;
  calibrationReliability?: number | null;
  calibratedAt?: Date | string | null;
  profileVersion?: number | null;
  profileUuid?: string | null;
  contentHash?: string | null;
  zoneBaselines?: unknown;
}

export interface BffProfileSyncResult {
  attempted: boolean;
  synced: boolean;
  statusCode?: number;
  skippedReason?: string;
  error?: string;
}

export interface SyncPythonProfileParams {
  subPath: string;
  userId: string;
  primaryBase: string;
  backendToken: string | null;
  loadPythonProfile: (userId: string) => Promise<SyncableDbDriverProfile | null>;
  fetchImpl?: typeof fetch;
  timeoutMs?: number;
}

const SYNC_TRIGGER_SUBPATHS = new Set(["livekit/token", "sessions"]);

export async function syncAuthoritativePythonProfileBeforeForward(
  params: SyncPythonProfileParams
): Promise<BffProfileSyncResult> {
  const {
    subPath,
    userId,
    primaryBase,
    backendToken,
    loadPythonProfile,
    fetchImpl = fetch,
    timeoutMs = 2500,
  } = params;

  if (!SYNC_TRIGGER_SUBPATHS.has(subPath)) {
    return { attempted: false, synced: false, skippedReason: "NON_SYNC_ROUTE" };
  }

  let dbProfile: SyncableDbDriverProfile | null = null;
  try {
    dbProfile = await loadPythonProfile(userId);
  } catch (dbErr) {
    const msg = dbErr instanceof Error ? dbErr.message : String(dbErr);
    console.warn("[API Proxy] Postgres PYTHON profile lookup warning:", dbErr);
    return { attempted: false, synced: false, error: msg };
  }

  if (!dbProfile) {
    return { attempted: false, synced: false, skippedReason: "NO_PYTHON_PROFILE" };
  }

  // Strict dual-engine boundary: NEVER sync a BROWSER profile to Python FastAPI
  if ((dbProfile.engineType || "").toUpperCase() !== "PYTHON") {
    return { attempted: false, synced: false, skippedReason: "NON_PYTHON_ENGINE_IGNORED" };
  }

  if (!dbProfile.isCalibrated) {
    return { attempted: false, synced: false, skippedReason: "PROFILE_NOT_CALIBRATED" };
  }

  const syncUrl = `${primaryBase.replace(/\/+$/, "")}/api/v1/profiles`;
  const fastApiProfileDto = buildDriverProfileDtoPayload(
    {
      engineType: "PYTHON",
      isCalibrated: dbProfile.isCalibrated,
      calibrationMode: dbProfile.calibrationMode ?? undefined,
      centerEarMedian: dbProfile.centerEarMedian,
      centerEarMadScaled: dbProfile.centerEarMadScaled ?? undefined,
      blinkDurationMedianMs: dbProfile.blinkDurationMedianMs ?? undefined,
      blinkDurationMadMs: dbProfile.blinkDurationMadMs ?? undefined,
      blinkRcrMedian: dbProfile.blinkRcrMedian ?? undefined,
      blinkRcrMad: dbProfile.blinkRcrMad ?? undefined,
      blinkFreqMedianPerMin: dbProfile.blinkFreqMedianPerMin ?? undefined,
      blinkFreqMadPerMin: dbProfile.blinkFreqMadPerMin ?? undefined,
      baselinePerclosPct: dbProfile.baselinePerclosPct ?? undefined,
      blinkAmplitudeMedian: dbProfile.blinkAmplitudeMedian ?? undefined,
      temporalBlinksSampled: dbProfile.temporalBlinksSampled ?? undefined,
      temporalWindowSec: dbProfile.temporalWindowSec ?? undefined,
      sensitivityLevel: dbProfile.sensitivityLevel ?? undefined,
      neutralPitchDeg: dbProfile.neutralPitchDeg ?? undefined,
      neutralYawDeg: dbProfile.neutralYawDeg ?? undefined,
      neutralRotMat: Array.isArray(dbProfile.neutralRotMat)
        ? (dbProfile.neutralRotMat as number[][])
        : null,
      calibrationIsMirrored: dbProfile.calibrationIsMirrored ?? null,
      calibrationReliability: dbProfile.calibrationReliability ?? undefined,
      calibratedAt: dbProfile.calibratedAt ?? null,
      profileVersion: dbProfile.profileVersion ?? undefined,
      profileUuid: dbProfile.profileUuid ?? undefined,
      contentHash: dbProfile.contentHash ?? null,
      zoneBaselines:
        dbProfile.zoneBaselines &&
        typeof dbProfile.zoneBaselines === "object" &&
        !Array.isArray(dbProfile.zoneBaselines)
          ? (dbProfile.zoneBaselines as Record<string, unknown>)
          : null,
    },
    userId
  );

  const controller = new AbortController();
  const timer = setTimeout(() => {
    controller.abort(new Error(`PYTHON profile sync timed out after ${timeoutMs}ms`));
  }, timeoutMs);

  try {
    const res = await fetchImpl(syncUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        ...(backendToken ? { Authorization: `Bearer ${backendToken}` } : {}),
      },
      body: JSON.stringify(fastApiProfileDto),
      signal: controller.signal,
    });

    if (!res.ok) {
      console.warn(`[API Proxy] PYTHON profile sync returned HTTP ${res.status}`);
      return { attempted: true, synced: false, statusCode: res.status };
    }

    return { attempted: true, synced: true, statusCode: res.status };
  } catch (syncErr) {
    const msg = syncErr instanceof Error ? syncErr.message : String(syncErr);
    console.warn("[API Proxy] Awaited PYTHON profile sync fallback (non-fatal):", syncErr);
    return { attempted: true, synced: false, error: msg };
  } finally {
    clearTimeout(timer);
  }
}
