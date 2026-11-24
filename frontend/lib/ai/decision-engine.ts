/**
 * DriveGuard — Deterministic Multi-Indicator Decision Engine with Hysteresis in TypeScript.
 *
 * Direct, verified port of backend/app/intelligence/decision_engine.py.
 * Computes:
 * - 5-level deterministic state machine:
 *     NORMAL (0) -> ATTENTION (1) -> DROWSINESS_WARNING (2) -> CRITICAL_WARNING (3) / INSUFFICIENT_DATA (-1)
 * - Strict safety-critical urgency prioritization
 * - Fully explainable reason codes with metric values and safety thresholds
 * - Asymmetric recovery cooldown hysteresis (escalations are immediate; de-escalations require sustained recovery)
 * - Two-timer alert lifecycle: state machine timer and alert resolution cooldown timer
 */

import {
  BlinkClassification,
  DecisionResult,
  DrowsinessState,
  ExplainableReason,
  OperationalState,
  ReasonCode,
  TemporalMetrics,
} from "./types";

export interface ConfidenceAssessment {
  overallConfidence: number;
  ocularViable: boolean;
  headPoseViable: boolean;
  yawnViable: boolean;
  isReliable: boolean;
  isOcularAvailable?: boolean | null;
  isCalibrated?: boolean;
  isCalibrating?: boolean;
}

export class DecisionEngine {
  public static readonly OCULAR_REASON_CODES = new Set<ReasonCode>([
    ReasonCode.SUSTAINED_EYE_CLOSURE,
    ReasonCode.NODDING_MICROSLEEP,
    ReasonCode.SEVERE_PERCLOS_CRISIS,
    ReasonCode.PERCLOS_ELEVATED,
    ReasonCode.PROLONGED_BLINK,
    ReasonCode.EXTENDED_CLOSURE,
    ReasonCode.MILD_PERCLOS,
  ]);

  public static readonly SEVERITY_MAP: Record<DrowsinessState, number> = {
    [DrowsinessState.NORMAL]: 0,
    [DrowsinessState.ATTENTION]: 1,
    [DrowsinessState.DROWSINESS_WARNING]: 2,
    [DrowsinessState.CRITICAL_WARNING]: 3,
    [DrowsinessState.INSUFFICIENT_DATA]: -1,
  };

  public sustainedClosureThresholdMs: number;
  public extendedClosureThresholdMs: number;
  public perclosWarningThresholdPct: number;
  public perclosAttentionThresholdPct: number;
  public severePerclosThresholdPct: number;
  public prolongedBlinkThresholdMs: number;
  public offRoadDistractionLimitMs: number;

  public recoveryCriticalMs: number;
  public recoveryWarningMs: number;
  public recoveryAttentionMs: number;
  public faceLossGraceMs: number;
  public sensorUnreliableGraceMs: number;
  public maxSamplingGapMs: number;

  // Active FSM State
  public currentState: DrowsinessState = DrowsinessState.NORMAL;
  public alertActive: boolean = false;

  private _recoveryStartTsMs: number | null = null;
  private _recoveryCandidateSev: number | null = null;
  private _stableFaceStartTsMs: number | null = null;
  private _sensorUnreliableStartTsMs: number | null = null;
  private _prevTsMs: number | null = null;

  // Alert Continuity
  private _unresolvedAlertState: DrowsinessState | null = null;
  private _historicalReason: ExplainableReason | null = null;
  private _lastElevatedReason: ExplainableReason | null = null;
  private _alertRecoveryStartTsMs: number | null = null;
  private _ocularElevatedActive: boolean = false;

  constructor(options?: {
    sustainedClosureThresholdMs?: number;
    extendedClosureThresholdMs?: number;
    perclosWarningThresholdPct?: number;
    perclosAttentionThresholdPct?: number;
    severePerclosThresholdPct?: number;
    prolongedBlinkThresholdMs?: number;
    offRoadDistractionLimitMs?: number;
    recoveryCriticalMs?: number;
    recoveryWarningMs?: number;
    recoveryAttentionMs?: number;
    faceLossGraceMs?: number;
    sensorUnreliableGraceMs?: number;
    maxSamplingGapMs?: number;
  }) {
    this.sustainedClosureThresholdMs = options?.sustainedClosureThresholdMs ?? 1500.0;
    this.extendedClosureThresholdMs = options?.extendedClosureThresholdMs ?? 500.0;
    this.perclosWarningThresholdPct = options?.perclosWarningThresholdPct ?? 15.0;
    this.perclosAttentionThresholdPct = options?.perclosAttentionThresholdPct ?? 8.0;
    this.severePerclosThresholdPct = options?.severePerclosThresholdPct ?? 30.0;
    this.prolongedBlinkThresholdMs = options?.prolongedBlinkThresholdMs ?? 500.0;
    this.offRoadDistractionLimitMs = options?.offRoadDistractionLimitMs ?? 3500.0;

    this.recoveryCriticalMs = options?.recoveryCriticalMs ?? 2000.0;
    this.recoveryWarningMs = options?.recoveryWarningMs ?? 1500.0;
    this.recoveryAttentionMs = options?.recoveryAttentionMs ?? 1000.0;
    this.faceLossGraceMs = options?.faceLossGraceMs ?? 1000.0;
    this.sensorUnreliableGraceMs = options?.sensorUnreliableGraceMs ?? 1000.0;
    this.maxSamplingGapMs = options?.maxSamplingGapMs ?? 250.0;
  }

  public evaluate(
    metrics: TemporalMetrics,
    confidence?: ConfidenceAssessment
  ): DecisionResult {
    const tsMs = metrics.timestampMs;
    const reasons: ExplainableReason[] = [];

    const conf: ConfidenceAssessment = confidence ?? {
      overallConfidence: 0.90,
      ocularViable: true,
      headPoseViable: true,
      yawnViable: true,
      isReliable: true,
      isOcularAvailable: true,
      isCalibrated: true,
      isCalibrating: false,
    };

    // Monotonic timestamp validation
    if (!Number.isFinite(tsMs) || tsMs < 0.0) {
      return this._buildResult(
        this._prevTsMs || 0.0,
        this.currentState,
        [],
        null,
        false,
        false,
        conf.isCalibrated === false
          ? OperationalState.CALIBRATION_REQUIRED
          : OperationalState.MONITORING_ACTIVE
      );
    }

    if (this._prevTsMs !== null) {
      const dtMs = tsMs - this._prevTsMs;
      if (dtMs <= 0.0) {
        return this._buildResult(
          this._prevTsMs,
          this.currentState,
          [],
          null,
          false,
          false,
          conf.isCalibrated === false
            ? OperationalState.CALIBRATION_REQUIRED
            : OperationalState.MONITORING_ACTIVE
        );
      }
      if (dtMs > this.maxSamplingGapMs) {
        this._recoveryStartTsMs = null;
        this._recoveryCandidateSev = null;
        this._alertRecoveryStartTsMs = null;
      }
    }

    this._prevTsMs = tsMs;

    // STEP 0: Strict Operational Calibration Gate (Decision 2)
    // Never emit NORMAL when calibration is in progress or missing/infeasible.
    if (conf.isCalibrating === true) {
      this._recoveryStartTsMs = null;
      this._recoveryCandidateSev = null;
      this._alertRecoveryStartTsMs = null;
      const primary: ExplainableReason = {
        code: ReasonCode.CALIBRATION_IN_PROGRESS,
        message: "Guided 4-Zone baseline calibration is in progress",
      };
      this.currentState = DrowsinessState.INSUFFICIENT_DATA;
      return this._buildResult(
        tsMs,
        DrowsinessState.INSUFFICIENT_DATA,
        [primary],
        primary,
        false,
        false,
        OperationalState.CALIBRATING
      );
    }

    if (conf.isCalibrated === false) {
      this._stableFaceStartTsMs = null;
      this._recoveryStartTsMs = null;
      this._recoveryCandidateSev = null;
      this._alertRecoveryStartTsMs = null;
      this.alertActive = false;
      this._unresolvedAlertState = null;
      this._historicalReason = null;
      const primary: ExplainableReason = {
        code: ReasonCode.CALIBRATION_REQUIRED,
        message: "Personalised 4-Zone calibration is required before drowsiness monitoring activates",
      };
      this.currentState = DrowsinessState.INSUFFICIENT_DATA;
      return this._buildResult(
        tsMs,
        DrowsinessState.INSUFFICIENT_DATA,
        [primary],
        primary,
        false,
        false,
        OperationalState.CALIBRATION_REQUIRED
      );
    }

    // STEP 1: Sensor & Face Viability Evaluation
    const isFaceLost = !metrics.hasFace;
    const isProlongedFaceLoss = isFaceLost && metrics.faceLostDurationMs >= this.faceLossGraceMs;

    const isPrimaryViable = conf.headPoseViable || conf.ocularViable;
    let isProlongedSensorFault = false;
    let isSensorUnreliable = false;
    let sensorUnreliableDurationMs = 0.0;

    if (!isFaceLost && !isPrimaryViable) {
      if (this._sensorUnreliableStartTsMs === null) {
        this._sensorUnreliableStartTsMs = tsMs;
      }
      sensorUnreliableDurationMs = tsMs - this._sensorUnreliableStartTsMs;
      isProlongedSensorFault = sensorUnreliableDurationMs >= this.sensorUnreliableGraceMs;
      isSensorUnreliable = true;
    } else {
      this._sensorUnreliableStartTsMs = null;
    }

    if (isProlongedFaceLoss || isProlongedSensorFault) {
      this._stableFaceStartTsMs = null;
      this._recoveryStartTsMs = null;
      this._recoveryCandidateSev = null;
      this._alertRecoveryStartTsMs = null;
      const candidateState = DrowsinessState.INSUFFICIENT_DATA;
      const primary: ExplainableReason = {
        code: isProlongedFaceLoss ? ReasonCode.FACE_LOST : ReasonCode.SENSOR_DEGRADED,
        message: isProlongedFaceLoss
          ? `Driver face not detected for ${metrics.faceLostDurationMs.toFixed(0)}ms`
          : `Sensor reliability degraded for ${sensorUnreliableDurationMs.toFixed(0)}ms`,
        metricValue: isProlongedFaceLoss ? metrics.faceLostDurationMs : conf.overallConfidence,
        threshold: isProlongedFaceLoss ? this.faceLossGraceMs : 0.50,
      };
      reasons.push(primary);
      this.currentState = candidateState;
      return this._buildResult(
        tsMs,
        candidateState,
        reasons,
        primary,
        false,
        false,
        OperationalState.DEGRADED
      );
    }

    if (isFaceLost || isSensorUnreliable) {
      this._recoveryStartTsMs = null;
      this._recoveryCandidateSev = null;
      this._alertRecoveryStartTsMs = null;
      const code = isFaceLost ? ReasonCode.FACE_LOST : ReasonCode.SENSOR_DEGRADED;
      const dur = isFaceLost ? metrics.faceLostDurationMs : sensorUnreliableDurationMs;
      const limit = isFaceLost ? this.faceLossGraceMs : this.sensorUnreliableGraceMs;
      const primary: ExplainableReason = {
        code,
        message: isFaceLost
          ? `Face loss grace period active (${dur.toFixed(0)}ms)`
          : `Sensor degradation grace period active (${dur.toFixed(0)}ms)`,
        metricValue: dur,
        threshold: limit,
      };
      return this._buildResult(
        tsMs,
        this.currentState,
        [primary],
        primary,
        false,
        false,
        OperationalState.DEGRADED
      );
    }

    if (this._stableFaceStartTsMs === null) {
      this._stableFaceStartTsMs = tsMs;
    }

    // STEP 2: Candidate State Arbitration (Highest Severity First)
    let candidateState = DrowsinessState.NORMAL;

    // Tier 3: CRITICAL_WARNING
    if (metrics.activeClosureDurationMs >= this.sustainedClosureThresholdMs) {
      candidateState = DrowsinessState.CRITICAL_WARNING;
      reasons.push({
        code: ReasonCode.SUSTAINED_EYE_CLOSURE,
        message: `Continuous eye closure of ${metrics.activeClosureDurationMs.toFixed(0)}ms exceeds safety limit`,
        metricValue: metrics.activeClosureDurationMs,
        threshold: this.sustainedClosureThresholdMs,
      });
    }

    if (metrics.isNoddingMicrosleep) {
      candidateState = DrowsinessState.CRITICAL_WARNING;
      const noddingThresholdMs = metrics.activeClosureDurationMs < 800.0 ? 500.0 : 800.0;
      reasons.push({
        code: ReasonCode.NODDING_MICROSLEEP,
        message: "Head nodding slump detected during active eye closure",
        metricValue: metrics.activeClosureDurationMs,
        threshold: noddingThresholdMs,
      });
    }

    if (metrics.perclosIsReliable && metrics.perclosPct >= this.severePerclosThresholdPct) {
      candidateState = DrowsinessState.CRITICAL_WARNING;
      reasons.push({
        code: ReasonCode.SEVERE_PERCLOS_CRISIS,
        message: `Critical PERCLOS accumulation of ${metrics.perclosPct.toFixed(1)}% exceeds emergency limit`,
        metricValue: metrics.perclosPct,
        threshold: this.severePerclosThresholdPct,
      });
    }

    // Tier 2: DROWSINESS_WARNING
    if (candidateState !== DrowsinessState.CRITICAL_WARNING) {
      if (metrics.perclosIsReliable && metrics.perclosPct >= this.perclosWarningThresholdPct) {
        candidateState = DrowsinessState.DROWSINESS_WARNING;
        reasons.push({
          code: ReasonCode.PERCLOS_ELEVATED,
          message: `Rolling 60s PERCLOS of ${metrics.perclosPct.toFixed(1)}% indicates driver drowsiness`,
          metricValue: metrics.perclosPct,
          threshold: this.perclosWarningThresholdPct,
        });
      }

      if (
        metrics.prolongedBlinkCount60s >= 3 ||
        (metrics.meanBlinkDurationMs60s >= this.prolongedBlinkThresholdMs && metrics.blinkCount60s >= 3)
      ) {
        candidateState = DrowsinessState.DROWSINESS_WARNING;
        reasons.push({
          code: ReasonCode.PROLONGED_BLINK,
          message: `Sluggish blink kinematics (${metrics.prolongedBlinkCount60s} prolonged blinks, mean ${metrics.meanBlinkDurationMs60s.toFixed(0)}ms)`,
          metricValue: metrics.meanBlinkDurationMs60s,
          threshold: this.prolongedBlinkThresholdMs,
        });
      }

      if (
        metrics.yawnCount60s >= 2 &&
        ((metrics.perclosIsReliable && metrics.perclosPct >= 8.0) || metrics.prolongedBlinkCount60s >= 1)
      ) {
        candidateState = DrowsinessState.DROWSINESS_WARNING;
        reasons.push({
          code: ReasonCode.FREQUENT_YAWNING,
          message: `Repeated yawns (${metrics.yawnCount60s} in 60s) combined with fatigue kinematics`,
          metricValue: metrics.yawnCount60s,
          threshold: 2.0,
        });
      }

      if (
        metrics.activeOffRoadDwellMs >= this.offRoadDistractionLimitMs &&
        !metrics.isMirrorCheckContext
      ) {
        candidateState = DrowsinessState.DROWSINESS_WARNING;
        reasons.push({
          code: ReasonCode.PERSISTENT_DISTRACTION,
          message: `Off-road head orientation dwell of ${(metrics.activeOffRoadDwellMs / 1000.0).toFixed(1)}s exceeds safety threshold`,
          metricValue: metrics.activeOffRoadDwellMs,
          threshold: this.offRoadDistractionLimitMs,
        });
      }
    }

    // Tier 1: ATTENTION
    if (candidateState !== DrowsinessState.CRITICAL_WARNING && candidateState !== DrowsinessState.DROWSINESS_WARNING) {
      if (conf.headPoseViable && !conf.ocularViable && metrics.downwardDwellMs > 2000.0) {
        candidateState = DrowsinessState.ATTENTION;
        reasons.push({
          code: ReasonCode.SUSPECTED_HEAD_SLUMP,
          message: `Sustained downward head slump (${metrics.downwardDwellMs.toFixed(0)}ms) with eyes unobservable`,
          metricValue: metrics.downwardDwellMs,
          threshold: 2000.0,
        });
      }

      if (metrics.activeClosureDurationMs >= this.extendedClosureThresholdMs) {
        candidateState = DrowsinessState.ATTENTION;
        reasons.push({
          code: ReasonCode.EXTENDED_CLOSURE,
          message: `Extended eye closure of ${metrics.activeClosureDurationMs.toFixed(0)}ms`,
          metricValue: metrics.activeClosureDurationMs,
          threshold: this.extendedClosureThresholdMs,
        });
      }

      const hasCompletedYawn = metrics.yawnCount30s >= 1;
      const hasActiveYawn = conf.yawnViable && metrics.activeYawnDurationMs >= 1500.0;
      if (hasCompletedYawn || hasActiveYawn) {
        candidateState = DrowsinessState.ATTENTION;
        reasons.push({
          code: ReasonCode.AUTHENTIC_YAWN,
          message: hasCompletedYawn
            ? "Authentic yawn detected"
            : `Sustained mouth opening (candidate yawn: ${metrics.activeYawnDurationMs.toFixed(0)}ms)`,
          metricValue: Math.max(metrics.activeYawnDurationMs, 1500.0),
          threshold: 1500.0,
        });
      }

      if (metrics.perclosIsReliable && metrics.perclosPct >= this.perclosAttentionThresholdPct) {
        candidateState = DrowsinessState.ATTENTION;
        reasons.push({
          code: ReasonCode.MILD_PERCLOS,
          message: `Mildly elevated PERCLOS of ${metrics.perclosPct.toFixed(1)}% indicates emerging fatigue`,
          metricValue: metrics.perclosPct,
          threshold: this.perclosAttentionThresholdPct,
        });
      }

      if (metrics.isMirrorCheckContext) {
        reasons.push({
          code: ReasonCode.MIRROR_CHECK_CONTEXT,
          message: "Normal driving mirror check",
          metricValue: metrics.activeOffRoadDwellMs,
          threshold: 3000.0,
        });
      } else if (metrics.isDashboardCheckContext) {
        reasons.push({
          code: ReasonCode.DASHBOARD_CHECK_CONTEXT,
          message: "Instrument cluster glance",
          metricValue: metrics.activeOffRoadDwellMs,
          threshold: 2000.0,
        });
      }
    }

    // Tier 0: NORMAL
    if (candidateState === DrowsinessState.NORMAL && reasons.length === 0) {
      reasons.push({
        code: ReasonCode.ALERT_NORMAL,
        message: "Driver alert; head orientation and eye metrics within nominal ranges",
      });
    }

    // STEP 3: Hysteresis and Persistence State Machine
    if (
      this.currentState === DrowsinessState.INSUFFICIENT_DATA &&
      this.alertActive &&
      this._unresolvedAlertState !== null
    ) {
      this.currentState = this._unresolvedAlertState;
    }

    const currentSev = DecisionEngine.SEVERITY_MAP[this.currentState];
    const candidateSev = DecisionEngine.SEVERITY_MAP[candidateState];
    const candidatePrimaryReason = reasons[0] || null;
    const candidateHasOcularReason = reasons.some((r) => DecisionEngine.OCULAR_REASON_CODES.has(r.code));

    if (candidateSev > 0 && candidatePrimaryReason !== null) {
      this._lastElevatedReason = candidatePrimaryReason;
    }

    // Alert Lifecycle initiation / escalation
    if (candidateSev >= 2) {
      if (!this.alertActive) {
        this.alertActive = true;
        this._unresolvedAlertState = candidateState;
        this._historicalReason = candidatePrimaryReason;
        this._alertRecoveryStartTsMs = null;
      } else {
        if (
          candidateState === DrowsinessState.CRITICAL_WARNING &&
          this._unresolvedAlertState === DrowsinessState.DROWSINESS_WARNING
        ) {
          this._unresolvedAlertState = DrowsinessState.CRITICAL_WARNING;
          this._historicalReason = candidatePrimaryReason;
        }
        this._alertRecoveryStartTsMs = null;
      }
    }

    // Case A: Escalation (Immediate)
    if (candidateSev > currentSev) {
      this.currentState = candidateState;
      this._recoveryStartTsMs = null;
      this._recoveryCandidateSev = null;
      if (candidateSev > 0 && candidateHasOcularReason) {
        this._ocularElevatedActive = true;
      } else if (candidateSev === 0) {
        this._ocularElevatedActive = false;
      }
    }
    // Case B: De-escalation (Requires Continuous Recovery Cooldown)
    else if (candidateSev < currentSev) {
      const ocularUnobservable =
        !conf.ocularViable || (conf.isOcularAvailable !== null && metrics.isEyeClosed === null);
      if (this._ocularElevatedActive && ocularUnobservable) {
        this._recoveryStartTsMs = null;
        this._recoveryCandidateSev = null;
      } else {
        let requiredCooldown = this.recoveryAttentionMs;
        if (currentSev === 3) {
          requiredCooldown = this.recoveryCriticalMs;
        } else if (currentSev === 2) {
          requiredCooldown = this.recoveryWarningMs;
        }

        if (
          this._recoveryStartTsMs === null ||
          this._recoveryCandidateSev === null ||
          candidateSev > this._recoveryCandidateSev
        ) {
          this._recoveryStartTsMs = tsMs;
          this._recoveryCandidateSev = candidateSev;
        } else if (candidateSev < this._recoveryCandidateSev) {
          this._recoveryCandidateSev = candidateSev;
        }

        const elapsedRecovery = tsMs - this._recoveryStartTsMs;
        if (elapsedRecovery >= requiredCooldown) {
          this.currentState = candidateState;
          this._recoveryStartTsMs = null;
          this._recoveryCandidateSev = null;
          if (candidateSev === 0) {
            this._ocularElevatedActive = false;
          } else if (candidateHasOcularReason) {
            this._ocularElevatedActive = true;
          }
        }
      }
    }
    // Case C: State matches candidate
    else {
      this._recoveryStartTsMs = null;
      this._recoveryCandidateSev = null;
      if (candidateSev > 0 && candidateHasOcularReason) {
        this._ocularElevatedActive = true;
      } else if (candidateSev === 0) {
        this._ocularElevatedActive = false;
      }
    }

    // Timer 2: Alert Resolution Cooldown
    if (this.alertActive) {
      const requiredAlertCooldown =
        this._unresolvedAlertState === DrowsinessState.CRITICAL_WARNING
          ? this.recoveryCriticalMs
          : this.recoveryWarningMs;

      if (this._alertRecoveryStartTsMs === null) {
        if (
          candidateState === DrowsinessState.NORMAL &&
          metrics.isEyeClosed === false &&
          conf.isReliable &&
          metrics.hasFace
        ) {
          this._alertRecoveryStartTsMs = tsMs;
        }
      } else {
        const completedBlink = metrics.completedBlink;
        const hasCompletedProlongedBlink =
          completedBlink !== null &&
          completedBlink.isBinocular &&
          (completedBlink.classification === BlinkClassification.PROLONGED ||
            completedBlink.classification === BlinkClassification.MICROSLEEP_CANDIDATE ||
            completedBlink.durationMs > this.prolongedBlinkThresholdMs);

        if (
          candidateState === DrowsinessState.NORMAL &&
          metrics.isEyeClosed !== null &&
          metrics.activeClosureDurationMs < this.prolongedBlinkThresholdMs &&
          !hasCompletedProlongedBlink &&
          conf.isReliable &&
          metrics.hasFace
        ) {
          // Continue recovery
        } else {
          this._alertRecoveryStartTsMs = null;
        }
      }

      if (this._alertRecoveryStartTsMs !== null) {
        const elapsedAlertRecovery = tsMs - this._alertRecoveryStartTsMs;
        if (elapsedAlertRecovery >= requiredAlertCooldown) {
          this.alertActive = false;
          this._unresolvedAlertState = null;
          this._historicalReason = null;
          this._alertRecoveryStartTsMs = null;
        }
      }
    }

    if (this.currentState === DrowsinessState.NORMAL && !this.alertActive) {
      this._lastElevatedReason = null;
    }

    let primaryReason = reasons[0] || null;
    if (
      DecisionEngine.SEVERITY_MAP[this.currentState] > candidateSev &&
      (primaryReason === null || primaryReason.code === ReasonCode.ALERT_NORMAL)
    ) {
      const heldReason = this._historicalReason || this._lastElevatedReason;
      if (heldReason !== null) {
        primaryReason = heldReason;
      }
    }

    const isAlertRecovering = Boolean(this.alertActive && this._alertRecoveryStartTsMs !== null);

    return this._buildResult(
      tsMs,
      this.currentState,
      reasons,
      primaryReason,
      isAlertRecovering,
      metrics.isMirrorCheckContext,
      OperationalState.MONITORING_ACTIVE
    );
  }

  private _buildResult(
    timestampMs: number,
    state: DrowsinessState,
    reasons: ExplainableReason[],
    primaryReason: ExplainableReason | null,
    isAlertRecovering: boolean,
    isMirrorCheckContext: boolean = false,
    operationalState: OperationalState = OperationalState.MONITORING_ACTIVE
  ): DecisionResult {
    const sev = DecisionEngine.SEVERITY_MAP[state];
    const codeStr = primaryReason ? primaryReason.code : "NONE";
    const summaryText = `[${operationalState} | ${state} (Sev ${sev}) | ${codeStr}]`;

    return {
      timestampMs,
      operationalState,
      state,
      severity: sev,
      reasons,
      primaryReason,
      isMirrorCheckContext,
      alertActive: this.alertActive,
      unresolvedAlertState: this._unresolvedAlertState,
      historicalReason: this._historicalReason,
      isAlertRecovering,
      summaryText,
    };
  }

  public reset(): void {
    this.currentState = DrowsinessState.NORMAL;
    this.alertActive = false;
    this._recoveryStartTsMs = null;
    this._recoveryCandidateSev = null;
    this._stableFaceStartTsMs = null;
    this._sensorUnreliableStartTsMs = null;
    this._prevTsMs = null;
    this._unresolvedAlertState = null;
    this._historicalReason = null;
    this._lastElevatedReason = null;
    this._alertRecoveryStartTsMs = null;
    this._ocularElevatedActive = false;
  }
}
