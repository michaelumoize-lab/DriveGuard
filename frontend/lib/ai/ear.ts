/**
 * DriveGuard — Eye Aspect Ratio (EAR) Feature Extraction Module in TypeScript.
 *
 * Direct, verified port of backend/app/features/ear.py.
 * Computes:
 * - Independent left and right EAR from canonical MediaPipe eye contours
 * - C1-continuous Hermite smoothstep perspective-weighted EAR conditioned on head yaw
 * - Safe downward pitch compensation (strictly active for open eyes above threshold)
 * - Raw uncompensated arithmetic mean and independent eye validity states
 */

import { EyeAspectRatioResult, NormalizedLandmark } from "./types";

export const LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144] as const;
export const RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380] as const;

export const DEFAULT_EAR_CLOSURE_THRESHOLD = 0.20;

export interface Point2D {
  x: number;
  y: number;
}

/**
 * Compute Eye Aspect Ratio for a single 6-point eye contour and evaluate landmark validity.
 *
 * EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4|| + eps)
 */
export function calculateSingleEyeEarWithValidity(
  landmarks: ArrayLike<NormalizedLandmark | Point2D>,
  indices: readonly [number, number, number, number, number, number],
  aspectRatio: number = 16.0 / 9.0,
  isNormalized?: boolean | null,
  eps: number = 1e-6
): [number, boolean] {
  const maxIdx = Math.max(...indices);
  if (!landmarks || landmarks.length <= maxIdx) {
    return [0.0, false];
  }

  const p: Point2D[] = [];
  for (let i = 0; i < 6; i++) {
    const pt = landmarks[indices[i]];
    if (!pt || !Number.isFinite(pt.x) || !Number.isFinite(pt.y)) {
      return [0.0, false];
    }
    p.push({ x: pt.x, y: pt.y });
  }

  // Coordinate space determination
  let normMode: boolean;
  if (isNormalized !== undefined && isNormalized !== null) {
    normMode = isNormalized;
  } else {
    let maxCoord = 0;
    for (let i = 0; i < 6; i++) {
      maxCoord = Math.max(maxCoord, Math.abs(p[i].x), Math.abs(p[i].y));
    }
    normMode = maxCoord <= 2.0;
  }

  const arFactor = normMode ? aspectRatio : 1.0;

  const [p1, p2, p3, p4, p5, p6] = p;

  // Horizontal eye width (between outer and inner corners p1 and p4)
  const horiz = Math.hypot((p1.x - p4.x) * arFactor, p1.y - p4.y);
  if (horiz < eps) {
    return [0.0, false];
  }

  // Vertical eyelid spans
  const v1 = Math.hypot((p2.x - p6.x) * arFactor, p2.y - p6.y);
  const v2 = Math.hypot((p3.x - p5.x) * arFactor, p3.y - p5.y);

  const ear = (v1 + v2) / (2.0 * horiz + eps);

  if (ear < 0.0 || ear > 1.5) {
    return [0.0, false];
  }

  return [ear, true];
}

/**
 * Backward-compatible single eye EAR scalar helper.
 */
export function calculateSingleEyeEar(
  landmarks: ArrayLike<NormalizedLandmark | Point2D>,
  indices: readonly [number, number, number, number, number, number],
  aspectRatio: number = 16.0 / 9.0,
  isNormalized?: boolean | null,
  eps: number = 1e-6
): number {
  const [ear] = calculateSingleEyeEarWithValidity(landmarks, indices, aspectRatio, isNormalized, eps);
  return ear;
}

/**
 * Combine left and right EAR with smoothstep (C1-continuous) yaw perspective weighting.
 *
 * Perspective Geometry & Coordinate Convention:
 * - Yaw = 0.0°: 50% left_ear, 50% right_ear
 * - Positive yaw (driver turned left): left_ear (MediaPipe idx 33 / camera image left / driver right eye)
 *   faces camera with full openness; weight smoothly ramps toward left_ear, reaching 80% at +30.0°
 * - Negative yaw (driver turned right): right_ear (MediaPipe idx 362 / camera image right / driver left eye)
 *   faces camera; weight smoothly ramps toward right_ear, reaching 80% at -30.0°
 * - Mirrored mode: effective_yaw is inverted (-yawDeg)
 * - Cubic Hermite smoothstep: S(u) = 3u^2 - 2u^3
 */
export function calculateYawWeightedEar(
  leftEar: number,
  rightEar: number,
  yawDeg: number,
  leftValid: boolean = true,
  rightValid: boolean = true,
  isMirrored: boolean = false
): number {
  if (!leftValid && !rightValid) return 0.0;
  if (leftValid && !rightValid) return leftEar;
  if (rightValid && !leftValid) return rightEar;

  const effectiveYaw = isMirrored ? -yawDeg : yawDeg;
  const yawClamped = Math.max(-30.0, Math.min(30.0, effectiveYaw));
  const u = (yawClamped + 30.0) / 60.0;

  const sU = u * u * (3.0 - 2.0 * u);
  const wLeft = 0.20 + 0.60 * sU;
  const wRight = 1.0 - wLeft;

  return wLeft * leftEar + wRight * rightEar;
}

/**
 * Pitch compensation for open eyes looking downward toward dashboard instruments.
 *
 * Safety contract:
 * Pitch compensation NEVER boosts an EAR that is at or below the closure threshold.
 */
export function calculatePitchCompensatedEar(
  baseEar: number,
  pitchDeg: number,
  neutralPitchDeg: number = 0.0,
  closureThreshold: number = DEFAULT_EAR_CLOSURE_THRESHOLD
): number {
  if (baseEar <= 0.0 || baseEar <= closureThreshold) {
    return baseEar;
  }

  const relPitch = pitchDeg - neutralPitchDeg;
  if (relPitch < -4.0) {
    const pitchBoost = 1.0 + Math.min(0.18, Math.max(0.0, (-relPitch - 4.0) * 0.012));
    return baseEar * pitchBoost;
  }
  return baseEar;
}

/**
 * Complete EAR pipeline extracting left, right, raw, yaw-weighted, and pitch-compensated EAR.
 */
export function extractEyeAspectRatio(
  landmarks: ArrayLike<NormalizedLandmark | Point2D>,
  yawDeg: number = 0.0,
  pitchDeg: number = 0.0,
  aspectRatio: number = 16.0 / 9.0,
  closureThreshold: number = DEFAULT_EAR_CLOSURE_THRESHOLD,
  isNormalized?: boolean | null,
  isMirrored: boolean = false
): EyeAspectRatioResult {
  const [leftEar, leftValid] = calculateSingleEyeEarWithValidity(
    landmarks,
    LEFT_EYE_INDICES,
    aspectRatio,
    isNormalized
  );
  const [rightEar, rightValid] = calculateSingleEyeEarWithValidity(
    landmarks,
    RIGHT_EYE_INDICES,
    aspectRatio,
    isNormalized
  );

  const isValid = leftValid || rightValid;

  let rawEar = 0.0;
  if (leftValid && rightValid) {
    rawEar = (leftEar + rightEar) / 2.0;
  } else if (leftValid) {
    rawEar = leftEar;
  } else if (rightValid) {
    rawEar = rightEar;
  }

  const weighted = calculateYawWeightedEar(leftEar, rightEar, yawDeg, leftValid, rightValid, isMirrored);

  const compensated = calculatePitchCompensatedEar(weighted, pitchDeg, 0.0, closureThreshold);

  const isLeftClosed = Boolean(leftValid && leftEar < closureThreshold);
  const isRightClosed = Boolean(rightValid && rightEar < closureThreshold);

  let isClosed = false;
  if (leftValid && rightValid) {
    isClosed = isLeftClosed && isRightClosed;
  } else if (leftValid) {
    isClosed = isLeftClosed;
  } else if (rightValid) {
    isClosed = isRightClosed;
  }

  const isGeometricValid = Boolean(leftValid && rightValid);

  return {
    leftEar,
    rightEar,
    weightedEar: weighted,
    compensatedEar: compensated,
    closureThreshold,
    isLeftClosed,
    isRightClosed,
    isClosed,
    isValid,
    rawEar,
    isLeftValid: leftValid,
    isRightValid: rightValid,
    isGeometricValid,
  };
}
