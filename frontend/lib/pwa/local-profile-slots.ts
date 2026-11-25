import {
  BrowserProfileStore,
  type ProfileSyncOutcome,
  type StoredBrowserProfileRecord,
} from "@/lib/ai";

export const LOCAL_SLOT_PREFIX = "local_slot:";
export const DEFAULT_LOCAL_SLOT_ID = `${LOCAL_SLOT_PREFIX}default`;
const SLOTS_STORAGE_KEY = "driveguard_local_driver_slots_v1";
const ACTIVE_SLOT_STORAGE_KEY = "driveguard_local_active_slot_v1";

export interface LocalDriverSlot {
  id: string;
  label: string;
  createdAt: string;
}

const DEFAULT_SLOT: LocalDriverSlot = {
  id: DEFAULT_LOCAL_SLOT_ID,
  label: "Local Driver 1",
  createdAt: "2026-01-01T00:00:00.000Z",
};

/**
 * Isolated local-only BrowserProfileStore subclass for `/local-monitor`.
 * Persists calibrations in IndexedDB (`driveguard_browser_ai_v2`) under explicit
 * `local_slot:*` keys without calling `/api/calibration/*` even when online,
 * ensuring shared-device isolation across authenticated and offline sessions.
 */
export class LocalOnlyProfileStore extends BrowserProfileStore {
  public override async syncToServer(userId: string): Promise<ProfileSyncOutcome> {
    const local = await this.getLocalProfile(userId);
    if (!local) {
      return { status: "NO_LOCAL_PROFILE", record: null };
    }
    if (local.syncState !== "LOCAL_ONLY") {
      const updated = await this.saveLocalProfile(userId, local.profile, {
        profileUuid: local.profileUuid,
        profileVersion: local.profileVersion,
        baseServerVersion: null,
        contentHash: local.contentHash,
        syncState: "LOCAL_ONLY",
        pendingConflictId: null,
      });
      return { status: "OFFLINE_DEFERRED", record: updated };
    }
    return { status: "OFFLINE_DEFERRED", record: local };
  }

  public override async reconcileWithServer(userId: string): Promise<{
    record: StoredBrowserProfileRecord | null;
    syncOutcome: ProfileSyncOutcome["status"];
  }> {
    const local = await this.getLocalProfile(userId);
    return {
      record: local,
      syncOutcome: local ? "UNCHANGED" : "NO_LOCAL_PROFILE",
    };
  }
}

export const localOnlyProfileStore = new LocalOnlyProfileStore();

export function isValidLocalSlotId(id: string): boolean {
  return typeof id === "string" && id.startsWith(LOCAL_SLOT_PREFIX) && id.length > LOCAL_SLOT_PREFIX.length;
}

export function listLocalDriverSlots(): LocalDriverSlot[] {
  if (typeof window === "undefined") return [DEFAULT_SLOT];
  try {
    const raw = window.localStorage.getItem(SLOTS_STORAGE_KEY);
    if (!raw) return [DEFAULT_SLOT];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [DEFAULT_SLOT];
    const valid = parsed.filter(
      (s): s is LocalDriverSlot =>
        Boolean(s) &&
        typeof s.id === "string" &&
        isValidLocalSlotId(s.id) &&
        typeof s.label === "string" &&
        s.label.trim().length > 0
    );
    if (valid.length === 0) return [DEFAULT_SLOT];
    return valid;
  } catch {
    return [DEFAULT_SLOT];
  }
}

function saveSlots(slots: LocalDriverSlot[]): void {
  if (typeof window === "undefined") return;
  try {
    window.localStorage.setItem(SLOTS_STORAGE_KEY, JSON.stringify(slots));
  } catch {
    // Ignore storage quota / private mode failures gracefully
  }
}

export function getActiveLocalSlotId(slots: LocalDriverSlot[]): string {
  if (typeof window === "undefined") return slots[0]?.id ?? DEFAULT_LOCAL_SLOT_ID;
  try {
    const stored = window.localStorage.getItem(ACTIVE_SLOT_STORAGE_KEY);
    if (stored && slots.some((s) => s.id === stored)) {
      return stored;
    }
  } catch {
    // Ignore storage errors
  }
  return slots[0]?.id ?? DEFAULT_LOCAL_SLOT_ID;
}

export function setActiveLocalSlotId(slotId: string): void {
  if (typeof window === "undefined" || !isValidLocalSlotId(slotId)) return;
  try {
    window.localStorage.setItem(ACTIVE_SLOT_STORAGE_KEY, slotId);
  } catch {
    // Ignore storage errors
  }
}

export function createLocalDriverSlot(labelInput?: string): {
  slots: LocalDriverSlot[];
  created: LocalDriverSlot;
} {
  const existing = listLocalDriverSlots();
  const trimmed = (labelInput ?? "").trim().slice(0, 32);
  const label = trimmed || `Local Driver ${existing.length + 1}`;
  const slug = `${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`;
  const created: LocalDriverSlot = {
    id: `${LOCAL_SLOT_PREFIX}${slug}`,
    label,
    createdAt: new Date().toISOString(),
  };
  const next = [...existing, created];
  saveSlots(next);
  setActiveLocalSlotId(created.id);
  return { slots: next, created };
}

export async function clearAndRemoveLocalSlot(slotId: string): Promise<{
  slots: LocalDriverSlot[];
  activeId: string;
}> {
  await localOnlyProfileStore.clearLocalProfile(slotId);
  const existing = listLocalDriverSlots();
  if (slotId === DEFAULT_LOCAL_SLOT_ID && existing.length <= 1) {
    return { slots: existing, activeId: DEFAULT_LOCAL_SLOT_ID };
  }
  const filtered = existing.filter((s) => s.id !== slotId);
  const next = filtered.length > 0 ? filtered : [DEFAULT_SLOT];
  saveSlots(next);
  const activeId = next[0].id;
  setActiveLocalSlotId(activeId);
  return { slots: next, activeId };
}
