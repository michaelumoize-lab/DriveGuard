/**
 * DriveGuard — Offline-Resilient Causal Event Queue & Persistence Sync in TypeScript.
 *
 * Implements Phase 3 in-tab offline reliability:
 * - Strict causal flush ordering: SESSION_START -> ALERTS -> SESSION_END
 * - Supports provisional offline session creation (`DRV-...` + `clientSessionId`)
 * - Idempotent deduplication via unique `eventId`, `clientSessionId`, and `finalizeToken`
 * - Bounded exponential backoff retry (never permanently freezes on transient 502 / fetch error)
 * - Clean removal of bound `window` online/offline listeners in `dispose()` (zero leaks)
 * - Honors HTTP 401 expired authentication sessions without losing queued data
 */

export interface BufferedSessionStart {
  clientSessionId: string;
  sessionCode: string;
  engineType: "BROWSER" | "PYTHON";
  startedAt: string;
  attempts: number;
}

export interface BufferedAlertEvent {
  eventId: string;
  sessionCode: string;
  clientSessionId?: string;
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
  clientSessionId?: string;
  finalizeToken?: string;
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
  attempts?: number;
}

export type PersistenceSyncStatus = "synced" | "pending" | "offline";

export function createProvisionalOfflineSession(): {
  clientSessionId: string;
  sessionCode: string;
  startedAt: string;
} {
  const timePart = Date.now().toString(36).toUpperCase();
  const randPart = Math.random().toString(36).slice(2, 10).toUpperCase();
  const uuid =
    typeof crypto !== "undefined" && typeof crypto.randomUUID === "function"
      ? crypto.randomUUID()
      : `cses_${timePart}_${randPart}`;
  return {
    clientSessionId: uuid,
    sessionCode: `DRV-${timePart}-${randPart.slice(0, 8)}`,
    startedAt: new Date().toISOString(),
  };
}

export interface OfflineEventQueueOptions {
  baseRetryDelayMs?: number;
  maxRetryDelayMs?: number;
  maxAttempts?: number;
  autoFlushIntervalMs?: number;
}

export class OfflineEventQueue {
  private static readonly STARTS_STORAGE_KEY = "driveguard_buffered_starts_v2";
  private static readonly ALERTS_STORAGE_KEY = "driveguard_buffered_events_v1";
  private static readonly ENDS_STORAGE_KEY = "driveguard_buffered_session_v1";

  private _sessionStartQueue: BufferedSessionStart[] = [];
  private _alertQueue: BufferedAlertEvent[] = [];
  private _sessionEndQueue: BufferedSessionSummary[] = [];

  private _isFlushing: boolean = false;
  private _activeFlushPromise: Promise<void> | null = null;
  private _pendingFlushRequested: boolean = false;
  private _isDisposed: boolean = false;
  private _isOnline: boolean =
    typeof navigator !== "undefined" && typeof navigator.onLine === "boolean"
      ? navigator.onLine
      : true;
  private _authExpired: boolean = false;
  private _consecutiveFailures: number = 0;

  private readonly _baseRetryDelayMs: number;
  private readonly _maxRetryDelayMs: number;
  private readonly _maxAttempts: number;

  private _listeners: Set<(status: PersistenceSyncStatus, pendingCount: number) => void> = new Set();
  private _flushTimer: ReturnType<typeof setInterval> | null = null;
  private _retryTimer: ReturnType<typeof setTimeout> | null = null;

  private readonly _boundOnOnline: () => void;
  private readonly _boundOnOffline: () => void;

  constructor(options: OfflineEventQueueOptions = {}) {
    this._baseRetryDelayMs = options.baseRetryDelayMs ?? 2000;
    this._maxRetryDelayMs = options.maxRetryDelayMs ?? 30000;
    this._maxAttempts = options.maxAttempts ?? 10;

    this._loadFromStorage();

    this._boundOnOnline = () => {
      if (this._isDisposed) return;
      this._isOnline = true;
      this._consecutiveFailures = 0;
      this._clearRetryTimer();
      this._notifyListeners();
      void this.flush();
    };

    this._boundOnOffline = () => {
      if (this._isDisposed) return;
      this._isOnline = false;
      this._clearRetryTimer();
      this._notifyListeners();
    };

    if (typeof window !== "undefined") {
      window.addEventListener("online", this._boundOnOnline);
      window.addEventListener("offline", this._boundOnOffline);

      const pollMs = options.autoFlushIntervalMs ?? 8000;
      if (pollMs > 0) {
        this._flushTimer = setInterval(() => {
          if (
            !this._isDisposed &&
            this._isOnline &&
            !this._authExpired &&
            this._retryTimer === null &&
            this.pendingCount > 0
          ) {
            void this.flush();
          }
        }, pollMs);
      }
    }
  }

  public get status(): PersistenceSyncStatus {
    if (!this._isOnline) return "offline";
    if (this.pendingCount > 0) return "pending";
    return "synced";
  }

  public get pendingCount(): number {
    return (
      this._sessionStartQueue.length +
      this._alertQueue.length +
      this._sessionEndQueue.length
    );
  }

  public get isAuthExpired(): boolean {
    return this._authExpired;
  }

  public get consecutiveFailures(): number {
    return this._consecutiveFailures;
  }

  public subscribe(
    listener: (status: PersistenceSyncStatus, pendingCount: number) => void
  ): () => void {
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

  private _clearRetryTimer(): void {
    if (this._retryTimer !== null) {
      clearTimeout(this._retryTimer);
      this._retryTimer = null;
    }
  }

  private _scheduleBackoffRetry(): void {
    if (this._isDisposed || this._authExpired || !this._isOnline || this.pendingCount === 0) {
      return;
    }
    this._clearRetryTimer();
    const exponent = Math.max(0, this._consecutiveFailures - 1);
    const delayMs = Math.min(
      this._maxRetryDelayMs,
      this._baseRetryDelayMs * Math.pow(2, exponent)
    );
    this._retryTimer = setTimeout(() => {
      this._retryTimer = null;
      if (!this._isDisposed && this._isOnline && !this._authExpired) {
        void this.flush();
      }
    }, delayMs);
  }

  public enqueueSessionStart(start: Omit<BufferedSessionStart, "attempts">): void {
    if (this._isDisposed) return;
    const exists = this._sessionStartQueue.some(
      (s) => s.clientSessionId === start.clientSessionId
    );
    if (exists) return;

    this._sessionStartQueue.push({ ...start, attempts: 0 });
    this._saveToStorage();
    this._notifyListeners();

    if (this._isOnline && !this._authExpired) {
      void this.flush();
    }
  }

  public enqueueAlert(event: Omit<BufferedAlertEvent, "attempts">): void {
    if (this._isDisposed) return;
    const existing = this._alertQueue.some((e) => e.eventId === event.eventId);
    if (existing) return;

    this._alertQueue.push({
      ...event,
      attempts: 0,
    });
    this._saveToStorage();
    this._notifyListeners();

    if (this._isOnline && !this._authExpired) {
      void this.flush();
    }
  }

  public enqueueSessionEnd(summary: BufferedSessionSummary): void {
    if (this._isDisposed) return;
    const existing = this._sessionEndQueue.some(
      (s) =>
        s.sessionCode === summary.sessionCode ||
        (summary.finalizeToken && s.finalizeToken === summary.finalizeToken)
    );
    if (existing) return;

    this._sessionEndQueue.push({ ...summary, attempts: 0 });
    this._saveToStorage();
    this._notifyListeners();

    if (this._isOnline && !this._authExpired) {
      void this.flush();
    }
  }

  /**
   * Flush queued items in strict causal order:
   * 1. SESSION_START (register provisional offline sessions before alerts reference them)
   * 2. ALERTS (idempotent via eventId)
   * 3. SESSION_END (idempotent via finalizeToken)
   */
  public async flush(): Promise<void> {
    if (this._isDisposed || !this._isOnline || this._authExpired) {
      return;
    }
    if (this._isFlushing && this._activeFlushPromise) {
      this._pendingFlushRequested = true;
      await this._activeFlushPromise;
      return;
    }

    const runPass = async (): Promise<void> => {
      do {
        this._pendingFlushRequested = false;
        await this._flushOnce();
      } while (
        this._pendingFlushRequested &&
        !this._isDisposed &&
        this._isOnline &&
        !this._authExpired &&
        this.pendingCount > 0
      );
    };

    this._activeFlushPromise = runPass().finally(() => {
      this._activeFlushPromise = null;
    });
    await this._activeFlushPromise;
  }

  private async _flushOnce(): Promise<void> {
    if (this._isDisposed || this._isFlushing || !this._isOnline || this._authExpired) {
      return;
    }
    this._isFlushing = true;
    let transientFailure = false;

    try {
      // 1. Flush SESSION_START items first
      while (this._sessionStartQueue.length > 0) {
        const item = this._sessionStartQueue[0];
        try {
          const res = await fetch("/api/sessions/start", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              clientSessionId: item.clientSessionId,
              sessionCode: item.sessionCode,
              engineType: item.engineType,
              startedAt: item.startedAt,
            }),
          });

          if (res.status === 401) {
            this._authExpired = true;
            break;
          }

          if (res.ok || (res.status >= 400 && res.status < 500)) {
            this._sessionStartQueue.shift();
            this._consecutiveFailures = 0;
            this._saveToStorage();
            this._notifyListeners();
          } else {
            item.attempts++;
            this._consecutiveFailures++;
            transientFailure = true;
            if (item.attempts > this._maxAttempts) {
              this._sessionStartQueue.shift();
              this._saveToStorage();
            }
            break;
          }
        } catch {
          this._consecutiveFailures++;
          transientFailure = true;
          break;
        }
      }

      // Causal barrier: do not flush alerts/ends while any SESSION_START failed transiently
      if (transientFailure || this._authExpired) {
        return;
      }

      // 2. Flush buffered alerts in FIFO order
      while (this._alertQueue.length > 0) {
        const item = this._alertQueue[0];
        try {
          const res = await fetch("/api/alerts/log", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              eventId: item.eventId,
              sessionCode: item.sessionCode,
              ...(item.clientSessionId ? { clientSessionId: item.clientSessionId } : {}),
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
            break;
          }

          if (res.ok) {
            this._alertQueue.shift();
            this._consecutiveFailures = 0;
            this._saveToStorage();
            this._notifyListeners();
          } else if (res.status >= 400 && res.status < 500) {
            console.warn(
              `[OfflineQueue] Alert rejected by server (${res.status}), dropping:`,
              item.eventId
            );
            this._alertQueue.shift();
            this._saveToStorage();
            this._notifyListeners();
          } else {
            item.attempts++;
            this._consecutiveFailures++;
            transientFailure = true;
            if (item.attempts > this._maxAttempts) {
              this._alertQueue.shift();
              this._saveToStorage();
            }
            break;
          }
        } catch {
          this._consecutiveFailures++;
          transientFailure = true;
          break;
        }
      }

      if (transientFailure || this._authExpired) {
        return;
      }

      // 3. Flush pending session completions in FIFO order
      while (this._sessionEndQueue.length > 0) {
        const item = this._sessionEndQueue[0];
        try {
          const { attempts: _attempts, ...payload } = item;
          const res = await fetch("/api/sessions/end", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });

          if (res.status === 401) {
            this._authExpired = true;
            break;
          }

          if (res.ok || (res.status >= 400 && res.status < 500)) {
            this._sessionEndQueue.shift();
            this._consecutiveFailures = 0;
            this._saveToStorage();
            this._notifyListeners();
          } else {
            item.attempts = (item.attempts ?? 0) + 1;
            this._consecutiveFailures++;
            transientFailure = true;
            if (item.attempts > this._maxAttempts) {
              this._sessionEndQueue.shift();
              this._saveToStorage();
            }
            break;
          }
        } catch {
          this._consecutiveFailures++;
          transientFailure = true;
          break;
        }
      }
    } finally {
      this._isFlushing = false;
      if (transientFailure) {
        this._scheduleBackoffRetry();
      }
      this._notifyListeners();
    }
  }

  private _saveToStorage(): void {
    if (typeof localStorage === "undefined") return;
    try {
      localStorage.setItem(
        OfflineEventQueue.STARTS_STORAGE_KEY,
        JSON.stringify(this._sessionStartQueue.slice(-50))
      );
      localStorage.setItem(
        OfflineEventQueue.ALERTS_STORAGE_KEY,
        JSON.stringify(this._alertQueue.slice(-200))
      );
      localStorage.setItem(
        OfflineEventQueue.ENDS_STORAGE_KEY,
        JSON.stringify(this._sessionEndQueue.slice(-50))
      );
    } catch {
      // Storage quota or private mode fallback
    }
  }

  private _loadFromStorage(): void {
    if (typeof localStorage === "undefined") return;
    try {
      const rawStarts = localStorage.getItem(OfflineEventQueue.STARTS_STORAGE_KEY);
      if (rawStarts) {
        const parsed = JSON.parse(rawStarts);
        if (Array.isArray(parsed)) this._sessionStartQueue = parsed;
      }

      const rawAlerts = localStorage.getItem(OfflineEventQueue.ALERTS_STORAGE_KEY);
      if (rawAlerts) {
        const parsed = JSON.parse(rawAlerts);
        if (Array.isArray(parsed)) this._alertQueue = parsed;
      }

      const rawEnds = localStorage.getItem(OfflineEventQueue.ENDS_STORAGE_KEY);
      if (rawEnds) {
        const parsed = JSON.parse(rawEnds);
        if (Array.isArray(parsed)) {
          this._sessionEndQueue = parsed;
        } else if (parsed && typeof parsed === "object") {
          // Backward-compatible migration from single-object v1 format
          this._sessionEndQueue = [parsed as BufferedSessionSummary];
        }
      }
    } catch {
      this._sessionStartQueue = [];
      this._alertQueue = [];
      this._sessionEndQueue = [];
    }
  }

  public dispose(): void {
    this._isDisposed = true;
    if (typeof window !== "undefined") {
      window.removeEventListener("online", this._boundOnOnline);
      window.removeEventListener("offline", this._boundOnOffline);
    }
    if (this._flushTimer !== null) {
      clearInterval(this._flushTimer);
      this._flushTimer = null;
    }
    this._clearRetryTimer();
    this._listeners.clear();
  }
}
