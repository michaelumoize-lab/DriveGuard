/**
 * DriveGuard — Unified Driver Feature Extractor with PDBP Personalization in TypeScript.
 *
 * Direct, verified port of backend/app/features/feature_extractor.py.
 * Coordinates:
 * - EAR extraction (yaw-weighted and pitch-compensated)
 * - Head pose estimation (Euler angles and cockpit zone)
 * - Mouth Aspect Ratio & Yawn detection
 * - Blink kinematics state machine with sub-frame interpolation
 * - Rolling PERCLOS (P80) calculation
 * - Dynamic PDBP zone baseline threshold latching (prevents threshold jumps during in-flight blinks)
 */

import { BlinkKinematicsDetector } from "./blink-detector";
import { extractEyeAspectRatio } from "./ear";
import { estimateHeadPose } from "./head-pose";
import { YawnDetector } from "./mar";
import { computeZoneThresholds } from "./pdbp-calibration";
import { PerclosCalculator } from "./perclos";
import {
  BlinkState,
  DriverBehaviourProfileData,
  DriverFeatures,
  NormalizedLandmark,
  PoseZone,
  ZoneBaselineData,
} from "./types";

export class DriverFeatureExtractor {
  public blinkDetector: BlinkKinematicsDetector;
  public perclosCalculator: PerclosCalculator;
  public yawnDetector: YawnDetector;

  private _profile: DriverBehaviourProfileData | null = null;
  private _zoneThresholds: Map<PoseZone, { closureThreshold: number; reopenThreshold: number }> = new Map();
  private _latchedThresholds: { closureThreshold: number; reopenThreshold: number } | null = null;
  private _currentActiveThresholds: { closureThreshold: number; reopenThreshold: number } = {
    closureThreshold: 0.20,
    reopenThreshold: 0.23,
  };
  private _isProfileFeasible: boolean = false;
  private _isMirrored: boolean = false;

  constructor(options?: {
    profile?: DriverBehaviourProfileData | null;
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

    if (options?.profile) {
      this.setProfile(options.profile);
    }
  }

  public setProfile(profile: DriverBehaviourProfileData | null): void {
    this._profile = profile;
    this._zoneThresholds.clear();

    if (!profile) {
      this._isProfileFeasible = false;
      this._currentActiveThresholds = { closureThreshold: 0.20, reopenThreshold: 0.23 };
      this.blinkDetector.updateThresholds(0.20, 0.23);
      return;
    }

    // Evaluate feasibility of CENTER baseline
    const centerT = computeZoneThresholds(profile.centerEarMedian, profile.centerEarMadScaled);
    this._isProfileFeasible = centerT.isFeasible;

    // Cache per-zone thresholds
    for (const zoneKey of [PoseZone.CENTER, PoseZone.LEFT_MIRROR, PoseZone.RIGHT_MIRROR, PoseZone.DASHBOARD]) {
      const zb: ZoneBaselineData | undefined = profile.zoneBaselines[zoneKey];
      if (zb) {
        const t = computeZoneThresholds(zb.earMedian, zb.earMadScaled);
        this._zoneThresholds.set(zoneKey, {
          closureThreshold: t.closureThreshold,
          reopenThreshold: t.reopenThreshold,
        });
      } else {
        // Fallback relative to center
        let scale = 1.0;
        if (zoneKey === PoseZone.LEFT_MIRROR || zoneKey === PoseZone.RIGHT_MIRROR) scale = 0.92;
        else if (zoneKey === PoseZone.DASHBOARD) scale = 0.88;
        const t = computeZoneThresholds(profile.centerEarMedian * scale, Math.max(profile.centerEarMadScaled, 0.015));
        this._zoneThresholds.set(zoneKey, {
          closureThreshold: t.closureThreshold,
          reopenThreshold: t.reopenThreshold,
        });
      }
    }

    if (this._isProfileFeasible) {
      const centerZoneT = this._zoneThresholds.get(PoseZone.CENTER) || centerT;
      this._currentActiveThresholds = { ...centerZoneT };
      this.blinkDetector.updateThresholds(centerZoneT.closureThreshold, centerZoneT.reopenThreshold);
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
    facialTransformMatrix?: number[] | number[][] | null
  ): DriverFeatures {
    const hasFace = Boolean(landmarks && landmarks.length >= 468);
    const ts = Number(timestampMs);
    const aspectRatio = frameWidth > 0 && frameHeight > 0 ? frameWidth / frameHeight : 16.0 / 9.0;

    // 1. Head Pose Estimation
    let headPose = estimateHeadPose(
      landmarks || [],
      frameWidth,
      frameHeight,
      this._profile?.neutralPitchDeg ?? 0.0,
      this._profile?.neutralYawDeg ?? 0.0,
      this._profile?.neutralRotMat ?? null,
      this._isMirrored,
      null,
      facialTransformMatrix,
      this._profile?.calibrationIsMirrored ?? null
    );

    if (!hasFace) {
      headPose = {
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
    }

    // 2. Resolve Thresholds with Blink Latching
    // Determine target thresholds for the current zone
    const targetZone = headPose.isValid ? headPose.zone : PoseZone.CENTER;
    const zoneT =
      this._zoneThresholds.get(targetZone) ||
      this._zoneThresholds.get(PoseZone.CENTER) ||
      this._currentActiveThresholds;

    // Latching logic: do not alter thresholds during an in-flight blink (CLOSING, CLOSED, PROLONGED_CLOSURE, REOPENING)
    const isBlinkInFlight =
      this.blinkDetector.state !== BlinkState.OPEN &&
      this.blinkDetector.state !== BlinkState.UNCERTAIN_GAP;

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
        this._currentActiveThresholds = { ...zoneT };
        this.blinkDetector.updateThresholds(zoneT.closureThreshold, zoneT.reopenThreshold);
      }
    }

    const effectiveClosureThreshold = this._latchedThresholds
      ? this._latchedThresholds.closureThreshold
      : this._currentActiveThresholds.closureThreshold;

    // 3. EAR Feature Extraction
    const ear = hasFace && landmarks
      ? extractEyeAspectRatio(
          landmarks,
          headPose.yawDeg,
          headPose.pitchDeg,
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

    // 4. Blink Kinematics Detection
    const blink = this.blinkDetector.update(
      ear.leftEar,
      ear.rightEar,
      ear.weightedEar,
      ts,
      hasFace && ear.isValid,
      ear.compensatedEar,
      ear.isLeftValid,
      ear.isRightValid
    );

    // 5. PERCLOS Accumulation
    const perclos = this.perclosCalculator.update(
      ts,
      hasFace && ear.isValid ? ear.isClosed : null,
      hasFace,
      ear.isValid
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

    const summaryText = hasFace
      ? `EAR: ${ear.compensatedEar.toFixed(2)} | PERCLOS: ${perclos.perclosPct.toFixed(1)}% | Blinks: ${blink.totalBlinkCount} | Pose: ${headPose.zone} | MAR: ${mar.mar.toFixed(2)}`
      : "FACE: LOST";

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
  }
}
