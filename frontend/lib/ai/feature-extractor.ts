/**
 * DriveGuard — Unified Driver Feature Extractor with PDBP Personalization in TypeScript.
 *
 * Direct, verified port of backend/app/features/feature_extractor.py.
 * Coordinates:
 * - EAR extraction (yaw-weighted and relative-pitch-compensated)
 * - Head pose estimation (Euler angles and cockpit zone with 1.5° hysteresis)
 * - Mouth Aspect Ratio & Yawn detection
 * - Blink kinematics state machine with sub-frame interpolation
 * - Rolling PERCLOS (P80) calculation using hysteretic closure
 * - Strict PDBP profile normalization & feasibility gating (disables ocular classification when uncalibrated)
 * - Dynamic PDBP zone baseline threshold latching (prevents threshold jumps during in-flight blinks)
 */

import { BlinkKinematicsDetector } from "./blink-detector";
import { extractEyeAspectRatio } from "./ear";
import { estimateHeadPose } from "./head-pose";
import { YawnDetector } from "./mar";
import { PerclosCalculator } from "./perclos";
import { normalizeDriverProfile, ZoneThresholdPair } from "./profile-normalizer";
import {
  BlinkState,
  DriverBehaviourProfileData,
  DriverFeatures,
  NormalizedLandmark,
  PoseZone,
} from "./types";

export class DriverFeatureExtractor {
  public blinkDetector: BlinkKinematicsDetector;
  public perclosCalculator: PerclosCalculator;
  public yawnDetector: YawnDetector;

  private _profile: DriverBehaviourProfileData | null = null;
  private _zoneThresholds: Map<PoseZone, ZoneThresholdPair> = new Map();
  private _latchedThresholds: { closureThreshold: number; reopenThreshold: number } | null = null;
  private _currentActiveThresholds: { closureThreshold: number; reopenThreshold: number } = {
    closureThreshold: 0.20,
    reopenThreshold: 0.23,
  };
  private _isProfileCalibrated: boolean = false;
  private _isProfileFeasible: boolean = false;
  private _profileRejectionReason: string | null = "NO_PROFILE";
  private _isMirrored: boolean = false;
  private _lastZone: PoseZone | null = null;

  constructor(options?: {
    profile?: DriverBehaviourProfileData | unknown | null;
    isMirrored?: boolean;
    closureThreshold?: number;
    reopenThreshold?: number;
  }) {
    this._isMirrored = options?.isMirrored ?? false;
    const defaultClosure = options?.closureThreshold ?? 0.20;
    const defaultReopen = options?.reopenThreshold ?? 0.23;

    this.blinkDetector = new BlinkKinematicsDetector({
      closureThreshold: defaultClosure,
      reopenThreshold: defaultReopen,
    });
    this.perclosCalculator = new PerclosCalculator();
    this.yawnDetector = new YawnDetector();

    if (options?.profile !== undefined) {
      this.setProfile(options.profile);
    }
  }

  public get isProfileCalibrated(): boolean {
    return this._isProfileCalibrated;
  }

  public get isProfileFeasible(): boolean {
    return this._isProfileFeasible;
  }

  public get profileRejectionReason(): string | null {
    return this._profileRejectionReason;
  }

  public get activeClosureThreshold(): number {
    return this._latchedThresholds
      ? this._latchedThresholds.closureThreshold
      : this._currentActiveThresholds.closureThreshold;
  }

  public get activeReopenThreshold(): number {
    return this._latchedThresholds
      ? this._latchedThresholds.reopenThreshold
      : this._currentActiveThresholds.reopenThreshold;
  }

  public get normalizedProfile(): DriverBehaviourProfileData | null {
    return this._profile;
  }

  public setProfile(rawProfile: DriverBehaviourProfileData | unknown | null): void {
    const normalized = normalizeDriverProfile(rawProfile);
    this._profile = normalized.profile;
    this._zoneThresholds = normalized.zoneThresholds;
    this._isProfileCalibrated = normalized.isCalibrated;
    this._isProfileFeasible = normalized.isFeasible;
    this._profileRejectionReason = normalized.rejectionReason;
    this._latchedThresholds = null;

    if (!normalized.isCalibrated || !normalized.isFeasible) {
      this._currentActiveThresholds = { closureThreshold: 0.20, reopenThreshold: 0.23 };
      this.blinkDetector.updateThresholds(0.20, 0.23);
      return;
    }

    const centerZoneT = this._zoneThresholds.get(PoseZone.CENTER);
    if (centerZoneT) {
      this._currentActiveThresholds = {
        closureThreshold: centerZoneT.closureThreshold,
        reopenThreshold: centerZoneT.reopenThreshold,
      };
      this.blinkDetector.updateThresholds(
        centerZoneT.closureThreshold,
        centerZoneT.reopenThreshold
      );
    }
  }

  public setIsMirrored(mirrored: boolean): void {
    this._isMirrored = mirrored;
  }

  public extract(
    landmarks: ArrayLike<NormalizedLandmark> | null,
    timestampMs: number,
    frameWidth: number = 640,
    frameHeight: number = 480,
    facialTransformMatrix?: ArrayLike<number> | number[][] | null
  ): DriverFeatures {
    const hasFace = Boolean(landmarks && landmarks.length >= 468);
    const ts = Number(timestampMs);
    const aspectRatio = frameWidth > 0 && frameHeight > 0 ? frameWidth / frameHeight : 16.0 / 9.0;

    // 1. Head Pose Estimation (passes _lastZone to enable 1.5° zone hysteresis)
    let headPose = hasFace
      ? estimateHeadPose(
          landmarks!,
          frameWidth,
          frameHeight,
          this._profile?.neutralPitchDeg ?? 0.0,
          this._profile?.neutralYawDeg ?? 0.0,
          this._profile?.neutralRotMat ?? null,
          this._isMirrored,
          this._lastZone,
          facialTransformMatrix,
          this._profile?.calibrationIsMirrored ?? null
        )
      : {
          pitchDeg: 0.0,
          yawDeg: 0.0,
          rollDeg: 0.0,
          zone: PoseZone.UNKNOWN,
          isLookingAway: false,
          isValid: false,
          relPitchDeg: 0.0,
          relYawDeg: 0.0,
          relRollDeg: 0.0,
          rotMat: null,
        };

    this._lastZone = hasFace && headPose.isValid ? headPose.zone : null;

    // 2. Resolve Thresholds with Blink Latching
    const ocularClassificationAllowed = this._isProfileCalibrated && this._isProfileFeasible;
    const targetZone = headPose.isValid ? headPose.zone : PoseZone.CENTER;
    const zoneT =
      this._zoneThresholds.get(targetZone) ||
      this._zoneThresholds.get(PoseZone.CENTER) ||
      this._currentActiveThresholds;

    // Latching logic: do not alter thresholds during an in-flight blink
    const isBlinkInFlight =
      this.blinkDetector.state !== BlinkState.OPEN &&
      this.blinkDetector.state !== BlinkState.UNCERTAIN_GAP;

    if (ocularClassificationAllowed) {
      if (isBlinkInFlight) {
        if (!this._latchedThresholds) {
          this._latchedThresholds = { ...this._currentActiveThresholds };
        }
      } else {
        this._latchedThresholds = null;
        if (
          this._currentActiveThresholds.closureThreshold !== zoneT.closureThreshold ||
          this._currentActiveThresholds.reopenThreshold !== zoneT.reopenThreshold
        ) {
          this._currentActiveThresholds = {
            closureThreshold: zoneT.closureThreshold,
            reopenThreshold: zoneT.reopenThreshold,
          };
          this.blinkDetector.updateThresholds(zoneT.closureThreshold, zoneT.reopenThreshold);
        }
      }
    } else {
      this._latchedThresholds = null;
    }

    const effectiveClosureThreshold = this._latchedThresholds
      ? this._latchedThresholds.closureThreshold
      : this._currentActiveThresholds.closureThreshold;

    // 3. EAR Feature Extraction (uses relative pitch headPose.relPitchDeg for pitch compensation)
    const rawEarResult =
      hasFace && landmarks
        ? extractEyeAspectRatio(
            landmarks,
            headPose.relYawDeg,
            headPose.relPitchDeg,
            aspectRatio,
            effectiveClosureThreshold,
            true,
            this._isMirrored
          )
        : {
            leftEar: 0.0,
            rightEar: 0.0,
            weightedEar: 0.0,
            compensatedEar: 0.0,
            closureThreshold: effectiveClosureThreshold,
            isLeftClosed: false,
            isRightClosed: false,
            isClosed: false,
            isValid: false,
            rawEar: 0.0,
            isLeftValid: false,
            isRightValid: false,
            isGeometricValid: false,
          };

    // Strict calibrated feasibility gate: when uncalibrated, preserve raw EAR numbers
    // for calibration capture / diagnostics, but disable ocular closure classification.
    const ear = ocularClassificationAllowed
      ? rawEarResult
      : {
          ...rawEarResult,
          isLeftClosed: false,
          isRightClosed: false,
          isClosed: false,
          isValid: false,
        };

    // 4. Blink Kinematics Detection (disabled when uncalibrated)
    const blink = this.blinkDetector.update(
      ear.leftEar,
      ear.rightEar,
      ear.weightedEar,
      ts,
      hasFace && ear.isValid && ocularClassificationAllowed,
      ear.compensatedEar,
      ear.isLeftValid && ocularClassificationAllowed,
      ear.isRightValid && ocularClassificationAllowed
    );

    // 5. PERCLOS Accumulation (uses hysteretic closure: ear.isClosed || blink.isEyeClosed)
    const hystereticClosed = ear.isClosed || blink.isEyeClosed;
    const perclos = this.perclosCalculator.update(
      ts,
      hasFace && ear.isValid && ocularClassificationAllowed ? hystereticClosed : null,
      hasFace && ocularClassificationAllowed,
      ear.isValid && ocularClassificationAllowed
    );

    // 6. Mouth Aspect Ratio & Yawn Detection
    const mar = this.yawnDetector.update(
      landmarks || [],
      ts,
      aspectRatio,
      hasFace,
      true,
      headPose.isLookingAway
    );

    const summaryText = !hasFace
      ? "FACE: LOST"
      : !ocularClassificationAllowed
        ? `CALIBRATION REQUIRED | Raw EAR: ${rawEarResult.weightedEar.toFixed(2)} | Pose: ${headPose.zone}`
        : `EAR: ${ear.compensatedEar.toFixed(2)} | PERCLOS: ${perclos.perclosPct.toFixed(1)}% | Blinks: ${blink.totalBlinkCount} | Pose: ${headPose.zone} | MAR: ${mar.mar.toFixed(2)}`;

    return {
      timestampMs: ts,
      hasFace,
      ear,
      headPose,
      mar,
      blink,
      perclos,
      summaryText,
    };
  }

  public reset(): void {
    this.blinkDetector.reset();
    this.perclosCalculator.reset();
    this.yawnDetector.reset();
    this._latchedThresholds = null;
    this._lastZone = null;
  }
}
