/**
 * DriveGuard — Browser AI IndexedDB Profile Store & Conflict-Aware PostgreSQL Sync.
 *
 * Provides full Browser AI operational independence:
 * - Stores the driver's BROWSER calibration profile locally in IndexedDB (`key: ${userId}:BROWSER`)
 *   with graceful in-memory fallback (`LOCAL_STORAGE_DEGRADED`) in private browsing or SSR/Node.
 * - Hydrates immediately from local storage with zero network blocking (works 100% offline).
 * - Reconciles asynchronously with PostgreSQL via `/api/calibration/profile?engineType=BROWSER`
 *   and `/api/calibration/save` (engineType="BROWSER"), preserving server-authoritative versions
 *   and surfacing `409 CONFLICT_SERVER_AUTHORITATIVE` without silent overwrites.
 */

import { normalizeDriverProfile } from "./profile-normalizer";
import type { DriverBehaviourProfileData } from "./types";

export type BrowserProfileSyncState =
  | "SYNCED"
  | "PENDING_UPLOAD"
  | "CONFLICT_PENDING_REVIEW"
  | "LOCAL_ONLY";

export type BrowserProfileStorageMode = "INDEXED_DB" | "LOCAL_STORAGE_DEGRADED";

export interface StoredBrowserProfileRecord {
  key: string;
  userId: string;
  engineType: "BROWSER";
  profile: DriverBehaviourProfileData;
  profileUuid: string;
  profileVersion: number;
  baseServerVersion: number | null;
  contentHash: string | null;
  calibratedAt: string;
  syncState: BrowserProfileSyncState;
  pendingConflictId: string | null;
  updatedAt: string;
}

export interface ProfileSyncOutcome {
  status:
    | "SYNCED"
    | "UNCHANGED"
    | "CONFLICT_SERVER_AUTHORITATIVE"
    | "OFFLINE_DEFERRED"
    | "NO_LOCAL_PROFILE"
    | "ERROR";
  record: StoredBrowserProfileRecord | null;
  conflictHistoryId?: string | null;
  serverProfile?: Record<string, unknown> | null;
  error?: string;
}

const DB_NAME = "driveguard_browser_ai_v2";
const DB_VERSION = 2;
const STORE_PROFILES = "driver_profiles";
const STORE_SESSION_STARTS = "session_starts";

function makeProfileKey(userId: string): string {
  return `${userId}:BROWSER`;
}

function generateUuid(): string {
  if (typeof crypto !== "undefined" && typeof crypto.randomUUID === "function") {
    return crypto.randomUUID();
  }
  return `brp_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 10)}`;
}

export class BrowserProfileStore {
  private _dbPromise: Promise<IDBDatabase | null> | null = null;
  private _memoryFallback: Map<string, StoredBrowserProfileRecord> = new Map();
  private _storageMode: BrowserProfileStorageMode =
    typeof indexedDB !== "undefined" ? "INDEXED_DB" : "LOCAL_STORAGE_DEGRADED";

  public get storageMode(): BrowserProfileStorageMode {
    return this._storageMode;
  }

  private _openDb(): Promise<IDBDatabase | null> {
    if (typeof indexedDB === "undefined") {
      this._storageMode = "LOCAL_STORAGE_DEGRADED";
      return Promise.resolve(null);
    }
    if (this._dbPromise) return this._dbPromise;

    this._dbPromise = new Promise((resolve) => {
      try {
        const req = indexedDB.open(DB_NAME, DB_VERSION);
        req.onupgradeneeded = () => {
          const db = req.result;
          if (!db.objectStoreNames.contains(STORE_PROFILES)) {
            db.createObjectStore(STORE_PROFILES, { keyPath: "key" });
          }
          if (!db.objectStoreNames.contains(STORE_SESSION_STARTS)) {
            db.createObjectStore(STORE_SESSION_STARTS, { keyPath: "clientSessionId" });
          }
        };
        req.onsuccess = () => {
          this._storageMode = "INDEXED_DB";
          resolve(req.result);
        };
        req.onerror = () => {
          this._storageMode = "LOCAL_STORAGE_DEGRADED";
          resolve(null);
        };
      } catch {
        this._storageMode = "LOCAL_STORAGE_DEGRADED";
        resolve(null);
      }
    });

    return this._dbPromise;
  }

  public async getLocalProfile(userId: string): Promise<StoredBrowserProfileRecord | null> {
    const key = makeProfileKey(userId);
    const db = await this._openDb();
    if (!db) {
      return this._memoryFallback.get(key) ?? null;
    }

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_PROFILES, "readonly");
        const store = tx.objectStore(STORE_PROFILES);
        const req = store.get(key);
        req.onsuccess = () => {
          resolve((req.result as StoredBrowserProfileRecord | undefined) ?? null);
        };
        req.onerror = () => {
          resolve(this._memoryFallback.get(key) ?? null);
        };
      } catch {
        resolve(this._memoryFallback.get(key) ?? null);
      }
    });
  }

  public async saveLocalProfile(
    userId: string,
    profile: DriverBehaviourProfileData,
    meta?: {
      profileUuid?: string;
      profileVersion?: number;
      baseServerVersion?: number | null;
      contentHash?: string | null;
      syncState?: BrowserProfileSyncState;
      pendingConflictId?: string | null;
    }
  ): Promise<StoredBrowserProfileRecord> {
    const existing = await this.getLocalProfile(userId);
    const nowIso = new Date().toISOString();
    const record: StoredBrowserProfileRecord = {
      key: makeProfileKey(userId),
      userId,
      engineType: "BROWSER",
      profile: {
        ...profile,
        driverId: userId,
        engineType: "BROWSER",
      },
      profileUuid: meta?.profileUuid ?? profile.profileUuid ?? generateUuid(),
      profileVersion:
        meta?.profileVersion ??
        profile.profileVersion ??
        (existing ? existing.profileVersion + 1 : 1),
      baseServerVersion:
        meta?.baseServerVersion !== undefined
          ? meta.baseServerVersion
          : existing?.baseServerVersion ?? null,
      contentHash:
        meta?.contentHash !== undefined ? meta.contentHash : profile.contentHash ?? null,
      calibratedAt: profile.calibratedAt || nowIso,
      syncState: meta?.syncState ?? "PENDING_UPLOAD",
      pendingConflictId:
        meta?.pendingConflictId !== undefined ? meta.pendingConflictId : null,
      updatedAt: nowIso,
    };

    this._memoryFallback.set(record.key, record);
    const db = await this._openDb();
    if (!db) {
      return record;
    }

    return new Promise((resolve) => {
      try {
        const tx = db.transaction(STORE_PROFILES, "readwrite");
        const store = tx.objectStore(STORE_PROFILES);
        const req = store.put(record);
        req.onsuccess = () => resolve(record);
        req.onerror = () => resolve(record);
      } catch {
        resolve(record);
      }
    });
  }

  public async clearLocalProfile(userId: string): Promise<void> {
    const key = makeProfileKey(userId);
    this._memoryFallback.delete(key);
    const db = await this._openDb();
    if (!db) return;

    await new Promise<void>((resolve) => {
      try {
        const tx = db.transaction(STORE_PROFILES, "readwrite");
        const req = tx.objectStore(STORE_PROFILES).delete(key);
        req.onsuccess = () => resolve();
        req.onerror = () => resolve();
      } catch {
        resolve();
      }
    });
  }

  /**
   * Non-blocking upload of local BROWSER profile to PostgreSQL `/api/calibration/save`.
   * Strictly sets `engineType: "BROWSER"` so PYTHON profiles are never touched.
   */
  public async syncToServer(userId: string): Promise<ProfileSyncOutcome> {
    const local = await this.getLocalProfile(userId);
    if (!local) {
      return { status: "NO_LOCAL_PROFILE", record: null };
    }

    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      return { status: "OFFLINE_DEFERRED", record: local };
    }

    const p = local.profile;
    try {
      const res = await fetch("/api/calibration/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          engineType: "BROWSER",
          ...(local.baseServerVersion !== null
            ? { baseServerVersion: local.baseServerVersion }
            : {}),
          clientCalibratedAt: local.calibratedAt,
          isCalibrated: true,
          calibrationMode: p.calibrationMode || "FOUR_ZONE",
          centerEarMedian: p.centerEarMedian,
          centerEarMadScaled: p.centerEarMadScaled,
          blinkDurationMedianMs: p.blinkDurationMedianMs,
          blinkDurationMadMs: p.blinkDurationMadMs,
          blinkRcrMedian: p.blinkRcrMedian,
          blinkRcrMad: p.blinkRcrMad,
          blinkFreqMedianPerMin: p.blinkFreqMedianPerMin,
          blinkFreqMadPerMin: p.blinkFreqMadPerMin,
          baselinePerclosPct: p.baselinePerclosPct,
          blinkAmplitudeMedian: p.blinkAmplitudeMedian,
          temporalBlinksSampled: p.temporalBlinksSampled,
          temporalWindowSec: p.temporalWindowSec,
          sensitivityLevel: p.sensitivityLevel || "STANDARD",
          neutralPitchDeg: p.neutralPitchDeg,
          neutralYawDeg: p.neutralYawDeg,
          neutralRotMat: p.neutralRotMat ?? null,
          calibrationIsMirrored: p.calibrationIsMirrored ?? false,
          calibrationReliability: p.calibrationReliability,
          zoneBaselines: p.zoneBaselines,
        }),
        signal: AbortSignal.timeout(5000),
      });

      if (res.status === 409) {
        const body = await res.json().catch(() => ({}));
        const updated = await this.saveLocalProfile(userId, local.profile, {
          profileUuid: local.profileUuid,
          profileVersion: local.profileVersion,
          baseServerVersion: local.baseServerVersion,
          contentHash: body.contentHash ?? local.contentHash,
          syncState: "CONFLICT_PENDING_REVIEW",
          pendingConflictId: body.conflictHistoryId ?? null,
        });
        return {
          status: "CONFLICT_SERVER_AUTHORITATIVE",
          record: updated,
          conflictHistoryId: body.conflictHistoryId ?? null,
          serverProfile: body.serverProfile ?? null,
        };
      }

      if (!res.ok) {
        return {
          status: res.status >= 500 ? "OFFLINE_DEFERRED" : "ERROR",
          record: local,
          error: `HTTP ${res.status}`,
        };
      }

      const data = await res.json();
      const serverProfile = data.profile ?? {};
      const updated = await this.saveLocalProfile(userId, local.profile, {
        profileUuid: serverProfile.profileUuid ?? local.profileUuid,
        profileVersion: Number(serverProfile.profileVersion ?? local.profileVersion),
        baseServerVersion: Number(serverProfile.profileVersion ?? local.profileVersion),
        contentHash: serverProfile.contentHash ?? local.contentHash,
        syncState: "SYNCED",
        pendingConflictId: null,
      });

      return {
        status: data.status === "UNCHANGED" ? "UNCHANGED" : "SYNCED",
        record: updated,
      };
    } catch {
      return { status: "OFFLINE_DEFERRED", record: local };
    }
  }

  /**
   * Bidirectional non-blocking reconciliation between local IndexedDB BROWSER profile
   * and PostgreSQL `/api/calibration/profile?engineType=BROWSER`.
   */
  public async reconcileWithServer(userId: string): Promise<{
    record: StoredBrowserProfileRecord | null;
    syncOutcome: ProfileSyncOutcome["status"];
  }> {
    const local = await this.getLocalProfile(userId);

    if (typeof navigator !== "undefined" && navigator.onLine === false) {
      return { record: local, syncOutcome: "OFFLINE_DEFERRED" };
    }

    // If local has an un-uploaded calibration, attempt server sync first (handles fast-forward or 409 safely)
    if (local && local.syncState === "PENDING_UPLOAD") {
      const outcome = await this.syncToServer(userId);
      return { record: outcome.record, syncOutcome: outcome.status };
    }

    // Otherwise pull authoritative server BROWSER profile if newer
    try {
      const res = await fetch("/api/calibration/profile?engineType=BROWSER", {
        method: "GET",
        headers: { "Cache-Control": "no-cache" },
        signal: AbortSignal.timeout(4000),
      });
      if (!res.ok) {
        return { record: local, syncOutcome: "OFFLINE_DEFERRED" };
      }

      const data = await res.json();
      const serverRaw = data.profile;
      if (!serverRaw) {
        return { record: local, syncOutcome: local ? "UNCHANGED" : "NO_LOCAL_PROFILE" };
      }

      const normalized = normalizeDriverProfile(serverRaw);
      if (!normalized.isCalibrated || !normalized.profile) {
        return { record: local, syncOutcome: "UNCHANGED" };
      }

      const serverVersion = Number(serverRaw.profileVersion ?? 1);
      if (
        !local ||
        serverVersion > local.profileVersion ||
        (local.syncState === "SYNCED" && serverRaw.contentHash !== local.contentHash)
      ) {
        const hydrated = await this.saveLocalProfile(userId, normalized.profile, {
          profileUuid: serverRaw.profileUuid,
          profileVersion: serverVersion,
          baseServerVersion: serverVersion,
          contentHash: serverRaw.contentHash ?? null,
          syncState: "SYNCED",
          pendingConflictId: null,
        });
        return { record: hydrated, syncOutcome: "SYNCED" };
      }

      return { record: local, syncOutcome: "UNCHANGED" };
    } catch {
      return { record: local, syncOutcome: "OFFLINE_DEFERRED" };
    }
  }
}

export const browserProfileStore = new BrowserProfileStore();
