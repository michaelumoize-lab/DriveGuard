/**
 * DriveGuard Browser AI Runtime — Shared TypeScript Interfaces and Types.
 *
 * Provides strict type definitions for:
 * - 3D Facial Landmarks
 * - Eye Aspect Ratio (EAR) metrics and closure states
 * - Blink states, classifications, and kinematics
 * - 6-DoF Head Pose, Euler angles, and Cockpit Zones
 * - Mouth Aspect Ratio (MAR) and Yawn candidates
 * - Cumulative PERCLOS (P80) sliding-window evaluation
 * - Temporal multi-window metrics
 * - Personalised Driver Behaviour Profile (PDBP) baselines and calibration
 * - Multi-indicator Decision Engine alert states and explainable reason codes
 * - Runtime telemetry, performance metrics, and offline sync contracts
 */

export interface NormalizedLandmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface EyeAspectRatioResult {
  leftEar: number;
  rightEar: number;
  weightedEar: number;
  compensatedEar: number;
  closureThreshold: number;
  isLeftClosed: boolean;
  isRightClosed: boolean;
  isClosed: boolean;
  isValid: boolean;
  rawEar: number;
  isLeftValid: boolean;
  isRightValid: boolean;
  isGeometricValid?: boolean;
}

export enum BlinkState {
  OPEN = "OPEN",
  CLOSING = "CLOSING",
  CLOSED = "CLOSED",
  REOPENING = "REOPENING",
  PROLONGED_CLOSURE = "PROLONGED_CLOSURE",
  UNCERTAIN_GAP = "UNCERTAIN_GAP",
}

export enum BlinkClassification {
  NORMAL = "NORMAL",
  PROLONGED = "PROLONGED",
  MICROSLEEP_CANDIDATE = "MICROSLEEP",
  UNCERTAIN = "UNCERTAIN",
}

export interface BlinkEvent {
  startTsMs: number;
  minTsMs: number;
  endTsMs: number;
  durationMs: number;
  openEarBefore: number;
  minEar: number;
  amplitude: number;
  closingVelocityPerS: number;
  reopeningVelocityPerS: number;
  rcr: number;
  classification: BlinkClassification;
  isBinocular: boolean;
  isKinematicsReliable: boolean;
}

export interface BlinkSnapshot {
  state: BlinkState;
  isEyeClosed: boolean;
  currentClosureDurationMs: number;
  completedBlink: BlinkEvent | null;
  recentBlinkDurationMs: number | null;
  recentBlinkRcr: number | null;
  totalBlinkCount: number;
  prolongedBlinkCount: number;
  samplingGapDetected: boolean;
}

export enum PoseZone {
  CENTER = "CENTER",
  LEFT_MIRROR = "LEFT_MIRROR",
  RIGHT_MIRROR = "RIGHT_MIRROR",
  DASHBOARD = "DASHBOARD",
  OUT_OF_BOUNDS = "OUT_OF_BOUNDS",
  UNKNOWN = "UNKNOWN",
}

export interface HeadPoseResult {
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
  zone: PoseZone;
  isLookingAway: boolean;
  isValid: boolean;
  relPitchDeg: number;
  relYawDeg: number;
  relRollDeg: number;
  rotMat: number[][] | null;
}

export interface YawnEvent {
  startTsMs: number;
  endTsMs: number;
  durationMs: number;
  peakMar: number;
  observedDurationMs: number;
  wallClockDurationMs: number;
  meanMar: number;
  openingRatio: number;
}

export interface MouthAspectRatioResult {
  mar: number;
  isMouthOpen: boolean;
  isYawnInProgress: boolean;
  currentYawnDurationMs: number;
  isAuthenticYawn: boolean;
  completedYawn: YawnEvent | null;
  totalYawnCount: number;
  isValid: boolean;
  baselineMar: number;
  openingRatio: number;
  peakMar: number;
  observedYawnDurationMs: number;
}

export interface PerclosResult {
  perclosPct: number;
  effectivePerclosPct: number | null;
  coveragePct: number;
  isReliable: boolean;
  closedDurationSec: number;
  observedDurationSec: number;
  windowSec: number;
  unobservedDurationSec: number;
}

export interface DriverFeatures {
  timestampMs: number;
  hasFace: boolean;
  ear: EyeAspectRatioResult;
  headPose: HeadPoseResult;
  mar: MouthAspectRatioResult;
  blink: BlinkSnapshot;
  perclos: PerclosResult;
  summaryText: string;
}

export type EngineType = "BROWSER" | "PYTHON";

/**
 * Operational readiness state for the AI runtime, strictly separated from
 * drowsiness severity (DrowsinessState).
 */
export enum OperationalState {
  CALIBRATION_REQUIRED = "CALIBRATION_REQUIRED",
  CALIBRATING = "CALIBRATING",
  MONITORING_ACTIVE = "MONITORING_ACTIVE",
  DEGRADED = "DEGRADED",
  ERROR = "ERROR",
}

export enum DrowsinessState {
  NORMAL = "NORMAL",
  ATTENTION = "ATTENTION",
  DROWSINESS_WARNING = "DROWSINESS_WARNING",
  CRITICAL_WARNING = "CRITICAL_WARNING",
  INSUFFICIENT_DATA = "INSUFFICIENT_DATA",
}

export enum ReasonCode {
  SUSTAINED_EYE_CLOSURE = "SUSTAINED_EYE_CLOSURE",
  NODDING_MICROSLEEP = "NODDING_MICROSLEEP",
  SEVERE_PERCLOS_CRISIS = "SEVERE_PERCLOS_CRISIS",
  PERCLOS_ELEVATED = "PERCLOS_ELEVATED",
  PROLONGED_BLINK = "PROLONGED_BLINK",
  FREQUENT_YAWNING = "FREQUENT_YAWNING",
  PERSISTENT_DISTRACTION = "PERSISTENT_DISTRACTION",
  AUTHENTIC_YAWN = "AUTHENTIC_YAWN",
  EXTENDED_CLOSURE = "EXTENDED_CLOSURE",
  SUSPECTED_HEAD_SLUMP = "SUSPECTED_HEAD_SLUMP",
  MILD_PERCLOS = "MILD_PERCLOS",
  MIRROR_CHECK_CONTEXT = "MIRROR_CHECK_CONTEXT",
  DASHBOARD_CHECK_CONTEXT = "DASHBOARD_CHECK_CONTEXT",
  ALERT_NORMAL = "ALERT_NORMAL",
  FACE_LOST = "FACE_LOST",
  SENSOR_DEGRADED = "SENSOR_DEGRADED",
}

export interface ExplainableReason {
  code: ReasonCode;
  message: string;
  metricValue?: number;
  threshold?: number;
}

export interface DecisionResult {
  timestampMs: number;
  state: DrowsinessState;
  severity: number;
  reasons: ExplainableReason[];
  primaryReason: ExplainableReason | null;
  isMirrorCheckContext: boolean;
  alertActive: boolean;
  unresolvedAlertState: DrowsinessState | null;
  historicalReason: ExplainableReason | null;
  isAlertRecovering?: boolean;
  summaryText: string;
}

export interface TemporalMetrics {
  timestampMs: number;
  activeClosureDurationMs: number;
  activeOffRoadDwellMs: number;
  isMirrorCheckContext: boolean;
  isDashboardCheckContext: boolean;
  isNoddingMicrosleep: boolean;
  blinkCount60s: number;
  blinkCount30s: number;
  meanBlinkDurationMs60s: number;
  prolongedBlinkCount60s: number;
  severeBlinkCount60s: number;
  meanRcr60s: number;
  yawnCount60s: number;
  yawnCount30s: number;
  activeYawnDurationMs: number;
  perclosPct: number;
  perclosCoveragePct: number;
  perclosIsReliable: boolean;
  effectivePerclosPct: number | null;
  hasFace: boolean;
  faceLostDurationMs: number;
  isEyeClosed: boolean | null;
  completedBlink: BlinkEvent | null;
  downwardDwellMs: number;
}

export interface ZoneBaselineData {
  zone: string;
  earMedian: number;
  earMadScaled: number;
  pitchMedianDeg: number;
  yawMedianDeg: number;
  sampleCount: number;
  earLeftMedian?: number;
  earRightMedian?: number;
}

export interface DriverBehaviourProfileData {
  driverId: string;
  engineType?: EngineType;
  profileVersion?: number;
  profileUuid?: string;
  contentHash?: string | null;
  calibrationMode: string;
  centerEarMedian: number;
  centerEarMadScaled: number;
  blinkDurationMedianMs: number;
  blinkDurationMadMs: number;
  blinkRcrMedian: number;
  blinkRcrMad: number;
  blinkFreqMedianPerMin: number;
  blinkFreqMadPerMin: number;
  neutralPitchDeg: number;
  neutralYawDeg: number;
  calibrationReliability: number;
  activeCenterEarMedian?: number;
  baselinePerclosPct: number;
  blinkAmplitudeMedian: number;
  temporalBlinksSampled: number;
  temporalWindowSec: number;
  sensitivityLevel: string;
  version: number;
  calibratedAt?: string;
  neutralRotMat?: number[][] | null;
  calibrationIsMirrored?: boolean | null;
  zoneBaselines: Record<string, ZoneBaselineData>;
}

export interface CalibrationTarget {
  zone: PoseZone;
  label: string;
  description: string;
  targetYawDeg: number;
  targetPitchDeg: number;
  yawToleranceDeg: number;
  pitchToleranceDeg: number;
  durationMs: number;
}

export interface CalibrationProgress {
  currentZone: PoseZone;
  stepIndex: number;
  totalSteps: number;
  zoneProgressPct: number;
  overallProgressPct: number;
  inZone: boolean;
  guidanceText: string;
  currentYawDeg: number;
  currentPitchDeg: number;
  isCompleted: boolean;
  rejectedReason?: string;
  profile?: DriverBehaviourProfileData;
}

export interface DetectionTelemetry {
  frameId: number;
  timestampMs: number;
  hasFace: boolean;
  operationalState?: OperationalState;
  engineType?: EngineType;
  alertState: DrowsinessState;
  severity: number;
  primaryReason: string;
  reasons: string[];
  ear: number;
  rawEar: number;
  leftEar: number;
  rightEar: number;
  earClosureThreshold?: number;
  perclosPct: number;
  effectivePerclosPct: number | null;
  perclosCoveragePct?: number;
  perclosReliable?: boolean;
  isClosed: boolean;
  closureDurationMs: number;
  blinkCount: number;
  blinkCount60s?: number;
  totalBlinkCount?: number;
  recentBlinkDurationMs: number | null;
  recentBlinkRcr: number | null;
  mar: number;
  isYawning: boolean;
  yawnCount: number;
  pitchDeg: number;
  yawDeg: number;
  rollDeg: number;
  poseZone: PoseZone;
  isLookingAway: boolean;
  inferenceFps: number;
  inferenceLatencyMs: number;
  runtimeMode: "browser" | "remote";
  persistenceSyncStatus: "synced" | "pending" | "offline";
}

export interface RuntimePerformanceMetrics {
  inferenceFps: number;
  cameraFps: number;
  averageInferenceLatencyMs: number;
  minInferenceLatencyMs: number;
  maxInferenceLatencyMs: number;
  totalFramesProcessed: number;
  droppedFrames: number;
  memoryUsageMb?: number;
}
