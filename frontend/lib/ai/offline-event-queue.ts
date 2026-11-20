/**
 * DriveGuard — Offline-Resilient Detection Event Queue & Persistence Sync in TypeScript.
 *
 * Implements Section 11 Offline & Network Failure Handling:
 * - Local buffering of safety events and session aggregates when network drops
 * - Deduplication via client-generated unique event UUIDs
 * - Safe asynchronous flush to Next.js API endpoints (/api/alerts/log and /api/sessions/end)
 * - Honors HTTP 401 expired authentication sessions
 * - Distinguishes local detection state from confirmed persistence state
 */

export interface BufferedAlertEvent {
  eventId: string;
  sessionCode: string;
  severity: "WARNING" | "CRITICAL";
  riskScore: number;
  earValue: number;
  perclosPct: number;
  poseZone: string;
  reasons: string[];
  cascadeDecision: string;
  timestampMs: number;
  attempts: number;
}

export interface BufferedSessionSummary {
  sessionCode: string;
  durationSeconds: number;
  averageFps: number;
  averageLatencyMs: number;
  averagePerclosPct: number;
  maxPerclosPct: number;
  totalBlinks: number;
  averageBlinkRate: number;
  normalFramesCount: number;
  warningAlertsCount: number;
  criticalAlertsCount: number;
  mirrorSuppressions: number;
}

export type PersistenceSyncStatus = "synced" | "pending" | "offline";

export class OfflineEventQueue {
  private static readonly STORAGE_KEY = "driveguard_buffered_events_v1";
  private static readonly SESSION_STORAGE_KEY = "driveguard_buffered_session_v1";

  private _alertQueue: BufferedAlertEvent[] = [];
  private _pendingSessionEnd: BufferedSessionSummary | null = null;
  private _isFlushing: boolean = false;
  private _isOnline: boolean = typeof navigator !== "undefined" ? navigator.onLine : true;
  private _authExpired: boolean = false;
  private _listeners: Set<(status: PersistenceSyncStatus, pendingCount: number) => void> = new Set();
  private _flushTimer: ReturnType<typeof setInterval> | null = null;

  constructor() {
    this._loadFromStorage();
    if (typeof window !== "undefined") {
      window.addEventListener("online", () => {
        this._isOnline = true;
        this._notifyListeners();
        this.flush();
      });
      window.addEventListener("offline", () => {
        this._isOnline = false;
        this._notifyListeners();
      });

      // Background retry interval (every 8 seconds when online and has items)
      this._flushTimer = setInterval(() => {
        if (this._isOnline && !this._authExpired && (this._alertQueue.length > 0 || this._pendingSessionEnd)) {
          this.flush();
        }
      }, 8000);
    }
  }

  public get status(): PersistenceSyncStatus {
    if (!this._isOnline) return "offline";
    if (this._alertQueue.length > 0 || this._pendingSessionEnd !== null) return "pending";
    return "synced";
  }

  public get pendingCount(): number {
    return this._alertQueue.length + (this._pendingSessionEnd ? 1 : 0);
  }

  public get isAuthExpired(): boolean {
    return this._authExpired;
  }

  public subscribe(listener: (status: PersistenceSyncStatus, pendingCount: number) => void): () => void {
    this._listeners.add(listener);
    listener(this.status, this.pendingCount);
    return () => {
      this._listeners.delete(listener);
    };
  }

  private _notifyListeners(): void {
    const s = this.status;
    const c = this.pendingCount;
    for (const listener of this._listeners) {
      try {
        listener(s, c);
      } catch (err) {
        console.warn("[OfflineQueue] Listener error:", err);
      }
    }
  }

  public enqueueAlert(event: Omit<BufferedAlertEvent, "attempts">): void {
    const existing = this._alertQueue.some((e) => e.eventId === event.eventId);
    if (existing) return;

    this._alertQueue.push({
      ...event,
      attempts: 0,
    });
    this._saveToStorage();
    this._notifyListeners();

    if (this._isOnline && !this._authExpired) {
      this.flush();
    }
  }

  public enqueueSessionEnd(summary: BufferedSessionSummary): void {
    this._pendingSessionEnd = { ...summary };
    this._saveToStorage();
    this._notifyListeners();

    if (this._isOnline && !this._authExpired) {
      this.flush();
    }
  }

  public async flush(): Promise<void> {
    if (this._isFlushing || !this._isOnline || this._authExpired) return;
    this._isFlushing = true;

    try {
      // 1. Flush buffered alerts in FIFO order
      while (this._alertQueue.length > 0) {
        const item = this._alertQueue[0];
        try {
          const res = await fetch("/api/alerts/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              sessionCode: item.sessionCode,
              severity: item.severity,
              riskScore: item.riskScore,
              earValue: item.earValue,
              perclosPct: item.perclosPct,
              poseZone: item.poseZone,
              reasons: item.reasons,
              cascadeDecision: item.cascadeDecision,
              timestampMs: item.timestampMs,
            }),
          });

          if (res.status === 401) {
            this._authExpired = true;
            console.warn("[OfflineQueue] Auth session expired while flushing alert logs.");
            break;
          }

          if (res.ok) {
            this._alertQueue.shift();
            this._saveToStorage();
            this._notifyListeners();
          } else if (res.status >= 400 && res.status < 500) {
            // Client error (e.g. session not found): drop to avoid blocking queue
            console.warn(`[OfflineQueue] Alert rejected by server (${res.status}), dropping:`, item.eventId);
            this._alertQueue.shift();
            this._saveToStorage();
            this._notifyListeners();
          } else {
            // Server error / network error: retry later
            item.attempts++;
            if (item.attempts > 10) {
              this._alertQueue.shift(); // Drop after 10 retries
              this._saveToStorage();
            }
            break;
          }
        } catch (fetchErr) {
          // Network connection failed during request
          this._isOnline = false;
          break;
        }
      }

      // 2. Flush pending session completion
      if (this._pendingSessionEnd && !this._authExpired && this._isOnline) {
        try {
          const res = await fetch("/api/sessions/end", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(this._pendingSessionEnd),
          });

          if (res.status === 401) {
            this._authExpired = true;
          } else if (res.ok || (res.status >= 400 && res.status < 500)) {
            this._pendingSessionEnd = null;
            this._saveToStorage();
            this._notifyListeners();
          }
        } catch (sessionErr) {
          this._isOnline = false;
        }
      }
    } finally {
      this._isFlushing = false;
      this._notifyListeners();
    }
  }

  private _saveToStorage(): void {
    if (typeof localStorage === "undefined") return;
    try {
      localStorage.setItem(OfflineEventQueue.STORAGE_KEY, JSON.stringify(this._alertQueue.slice(-200)));
      if (this._pendingSessionEnd) {
        localStorage.setItem(OfflineEventQueue.SESSION_STORAGE_KEY, JSON.stringify(this._pendingSessionEnd));
      } else {
        localStorage.removeItem(OfflineEventQueue.SESSION_STORAGE_KEY);
      }
    } catch {
      // Storage quota or private mode fallback
    }
  }

  private _loadFromStorage(): void {
    if (typeof localStorage === "undefined") return;
    try {
      const raw = localStorage.getItem(OfflineEventQueue.STORAGE_KEY);
      if (raw) {
        this._alertQueue = JSON.parse(raw);
      }
      const rawSession = localStorage.getItem(OfflineEventQueue.SESSION_STORAGE_KEY);
      if (rawSession) {
        this._pendingSessionEnd = JSON.parse(rawSession);
      }
    } catch {
      this._alertQueue = [];
      this._pendingSessionEnd = null;
    }
  }

  public dispose(): void {
    if (this._flushTimer) {
      clearInterval(this._flushTimer);
      this._flushTimer = null;
    }
    this._listeners.clear();
  }
}
