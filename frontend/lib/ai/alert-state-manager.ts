/**
 * DriveGuard — Alert State Manager in TypeScript.
 *
 * Direct port of backend/app/alerts/alert_state_manager.py with bounded repeat
 * cooldowns and unique eventId generation per emitted safety episode.
 *
 * Enforces:
 * 1. Priority-preserving transitions (CRITICAL > WARNING > ATTENTION > NORMAL).
 * 2. Alert continuity: active unresolved alerts are never downgraded prematurely.
 * 3. Safe handling of INSUFFICIENT_DATA / face-loss / uncalibrated states (never downgrades to NORMAL).
 * 4. Elimination of per-frame alert storms: emits ONLY on state transitions, verified
 *    recovery, or bounded repeat cooldowns while a WARNING/CRITICAL alert remains unresolved.
 * 5. Unique idempotent eventId per emitted alert episode for `/api/alerts/log`.
 */

import {
  DecisionResult,
  DrowsinessState,
  ExplainableReason,
  ReasonCode,
  TemporalMetrics,
} from "./types";

export enum ManagedAlertState {
  NORMAL = "NORMAL",
  ATTENTION = "ATTENTION",
  WARNING = "WARNING",
  CRITICAL = "CRITICAL",
}

const ALERT_SEVERITY: Record<ManagedAlertState, number> = {
  [ManagedAlertState.NORMAL]: 0,
  [ManagedAlertState.ATTENTION]: 1,
  [ManagedAlertState.WARNING]: 2,
  [ManagedAlertState.CRITICAL]: 3,
};

export interface AlertTransition {
  eventId: string;
  timestampMs: number;
  previousState: ManagedAlertState;
  newState: ManagedAlertState;
  severity: number;
  reason: ExplainableReason | null;
  reasonCode: string | null;
  isRecovery: boolean;
  isRepeatReminder: boolean;
  alertActive: boolean;
  unresolvedAlertState: ManagedAlertState | null;
}

export interface AlertStateManagerOptions {
  maxHistorySize?: number;
  criticalRepeatCooldownMs?: number;
  warningRepeatCooldownMs?: number;
}

export function mapDrowsinessStateToAlertState(
  state: DrowsinessState
): ManagedAlertState | null {
  switch (state) {
    case DrowsinessState.NORMAL:
      return ManagedAlertState.NORMAL;
    case DrowsinessState.ATTENTION:
      return ManagedAlertState.ATTENTION;
    case DrowsinessState.DROWSINESS_WARNING:
      return ManagedAlertState.WARNING;
    case DrowsinessState.CRITICAL_WARNING:
      return ManagedAlertState.CRITICAL;
    case DrowsinessState.INSUFFICIENT_DATA:
      return null;
    default:
      return null;
  }
}

export class AlertStateManager {
  public readonly maxHistorySize: number;
  public readonly criticalRepeatCooldownMs: number;
  public readonly warningRepeatCooldownMs: number;

  private _currentState: ManagedAlertState = ManagedAlertState.NORMAL;
  private _activeAlertLevel: ManagedAlertState | null = null;
  private _isRecovering: boolean = false;
  private _lastReason: ExplainableReason | null = null;
  private _lastAssessmentTsMs: number | null = null;
  private _lastTransitionTsMs: number | null = null;
  private _lastSafetyEmitTsMs: number | null = null;
  private _trackingAvailable: boolean = true;
  private _eventSeq: number = 0;
  private _history: AlertTransition[] = [];

  constructor(options: AlertStateManagerOptions = {}) {
    this.maxHistorySize = options.maxHistorySize ?? 1000;
    this.criticalRepeatCooldownMs = options.criticalRepeatCooldownMs ?? 8000.0;
    this.warningRepeatCooldownMs = options.warningRepeatCooldownMs ?? 15000.0;
  }

  public get currentState(): ManagedAlertState {
    return this._currentState;
  }

  public get severity(): number {
    return ALERT_SEVERITY[this._currentState];
  }

  public get isRecovering(): boolean {
    return this._isRecovering;
  }

  public get isAlertActive(): boolean {
    return (
      this._activeAlertLevel !== null ||
      this._currentState === ManagedAlertState.WARNING ||
      this._currentState === ManagedAlertState.CRITICAL
    );
  }

  public get trackingAvailable(): boolean {
    return this._trackingAvailable;
  }

  public get activeReason(): ExplainableReason | null {
    return this._lastReason;
  }

  public get history(): AlertTransition[] {
    return [...this._history];
  }

  private _nextEventId(tsMs: number): string {
    this._eventSeq++;
    const rand = Math.random().toString(36).slice(2, 8);
    return `evt_br_${Math.round(tsMs)}_${this._eventSeq}_${rand}`;
  }

  public update(
    decision: DecisionResult | null,
    temporal?: TemporalMetrics | null
  ): AlertTransition | null {
    if (!decision) return null;

    const tsMs = decision.timestampMs;
    if (!Number.isFinite(tsMs) || tsMs < 0.0) return null;

    // Monotonicity check: reject backward timestamps
    if (this._lastAssessmentTsMs !== null && tsMs < this._lastAssessmentTsMs) {
      return null;
    }
    this._lastAssessmentTsMs = tsMs;

    // Sensor loss / uncalibrated / insufficient data: preserve continuity, never downgrade to NORMAL
    if (decision.state === DrowsinessState.INSUFFICIENT_DATA) {
      this._trackingAvailable = false;
      if (this._activeAlertLevel !== null) {
        this._isRecovering = false;
      }
      return null;
    }

    this._trackingAvailable = true;

    const candidateAlert =
      mapDrowsinessStateToAlertState(decision.state) ?? ManagedAlertState.NORMAL;
    const unresolvedMapped =
      decision.unresolvedAlertState !== null
        ? mapDrowsinessStateToAlertState(decision.unresolvedAlertState)
        : null;

    const prevState = this._currentState;
    let targetState = this._currentState;
    let isRecoveryEvent = false;

    if (decision.alertActive) {
      let activeCandidate = candidateAlert;
      if (
        unresolvedMapped !== null &&
        ALERT_SEVERITY[unresolvedMapped] > ALERT_SEVERITY[activeCandidate]
      ) {
        activeCandidate = unresolvedMapped;
      }

      // Priority guard: never downgrade an active unresolved alert
      if (this._activeAlertLevel !== null) {
        if (ALERT_SEVERITY[activeCandidate] > ALERT_SEVERITY[this._activeAlertLevel]) {
          this._activeAlertLevel = activeCandidate;
        }
      } else {
        this._activeAlertLevel = activeCandidate;
      }

      targetState = this._activeAlertLevel;

      if (decision.isAlertRecovering !== undefined) {
        this._isRecovering = Boolean(decision.isAlertRecovering);
      } else {
        const eyeClosed = temporal ? temporal.isEyeClosed : null;
        this._isRecovering =
          candidateAlert === ManagedAlertState.NORMAL && eyeClosed !== true;
      }

      if (
        decision.primaryReason !== null &&
        ALERT_SEVERITY[candidateAlert] >= ALERT_SEVERITY[targetState]
      ) {
        this._lastReason = decision.primaryReason;
      } else if (decision.historicalReason !== null) {
        this._lastReason = decision.historicalReason;
      } else if (decision.primaryReason !== null && this._lastReason === null) {
        this._lastReason = decision.primaryReason;
      }
    } else {
      if (this._activeAlertLevel !== null) {
        isRecoveryEvent = true;
        this._activeAlertLevel = null;
        this._isRecovering = false;
        targetState = candidateAlert;
        this._lastReason = decision.primaryReason;
      } else {
        if (
          prevState === ManagedAlertState.ATTENTION &&
          candidateAlert === ManagedAlertState.NORMAL
        ) {
          isRecoveryEvent = true;
          this._isRecovering = false;
        } else if (ALERT_SEVERITY[candidateAlert] > ALERT_SEVERITY[prevState]) {
          this._isRecovering = false;
        }
        targetState = candidateAlert;
        this._lastReason = decision.primaryReason;
      }
    }

    // 1. State transition or recovery event
    if (targetState !== prevState || isRecoveryEvent) {
      const transition: AlertTransition = {
        eventId: this._nextEventId(tsMs),
        timestampMs: tsMs,
        previousState: prevState,
        newState: targetState,
        severity: ALERT_SEVERITY[targetState],
        reason: this._lastReason,
        reasonCode: this._lastReason ? this._lastReason.code : null,
        isRecovery: isRecoveryEvent,
        isRepeatReminder: false,
        alertActive: decision.alertActive,
        unresolvedAlertState: this._activeAlertLevel,
      };

      this._currentState = targetState;
      this._lastTransitionTsMs = tsMs;
      if (
        targetState === ManagedAlertState.WARNING ||
        targetState === ManagedAlertState.CRITICAL
      ) {
        this._lastSafetyEmitTsMs = tsMs;
      } else if (isRecoveryEvent) {
        this._lastSafetyEmitTsMs = null;
      }
      this._recordTransition(transition);
      return transition;
    }

    // 2. Bounded repeat reminder while WARNING or CRITICAL persists unresolved (not recovering)
    if (
      (targetState === ManagedAlertState.CRITICAL ||
        targetState === ManagedAlertState.WARNING) &&
      !this._isRecovering
    ) {
      const cooldown =
        targetState === ManagedAlertState.CRITICAL
          ? this.criticalRepeatCooldownMs
          : this.warningRepeatCooldownMs;
      if (
        this._lastSafetyEmitTsMs !== null &&
        tsMs - this._lastSafetyEmitTsMs >= cooldown
      ) {
        this._lastSafetyEmitTsMs = tsMs;
        const repeatTransition: AlertTransition = {
          eventId: this._nextEventId(tsMs),
          timestampMs: tsMs,
          previousState: prevState,
          newState: targetState,
          severity: ALERT_SEVERITY[targetState],
          reason: this._lastReason,
          reasonCode: this._lastReason ? this._lastReason.code : null,
          isRecovery: false,
          isRepeatReminder: true,
          alertActive: decision.alertActive,
          unresolvedAlertState: this._activeAlertLevel,
        };
        this._recordTransition(repeatTransition);
        return repeatTransition;
      }
    }

    // Identical state inside cooldown: suppress duplicate frame storm
    return null;
  }

  public forceRecovery(timestampMs: number): AlertTransition | null {
    if (
      this._currentState === ManagedAlertState.NORMAL &&
      this._activeAlertLevel === null
    ) {
      return null;
    }

    const prevState = this._currentState;
    this._currentState = ManagedAlertState.NORMAL;
    this._activeAlertLevel = null;
    this._isRecovering = false;
    this._lastSafetyEmitTsMs = null;
    this._lastReason = {
      code: ReasonCode.ALERT_NORMAL,
      message: "Alert status returned to normal by explicit recovery.",
    };

    const transition: AlertTransition = {
      eventId: this._nextEventId(timestampMs),
      timestampMs,
      previousState: prevState,
      newState: ManagedAlertState.NORMAL,
      severity: 0,
      reason: this._lastReason,
      reasonCode: "MANUAL_RECOVERY",
      isRecovery: true,
      isRepeatReminder: false,
      alertActive: false,
      unresolvedAlertState: null,
    };
    this._recordTransition(transition);
    return transition;
  }

  public reset(): void {
    this._currentState = ManagedAlertState.NORMAL;
    this._activeAlertLevel = null;
    this._isRecovering = false;
    this._lastReason = null;
    this._lastAssessmentTsMs = null;
    this._lastTransitionTsMs = null;
    this._lastSafetyEmitTsMs = null;
    this._trackingAvailable = true;
    this._history = [];
  }

  private _recordTransition(transition: AlertTransition): void {
    this._history.push(transition);
    if (this._history.length > this.maxHistorySize) {
      this._history.shift();
    }
  }
}
