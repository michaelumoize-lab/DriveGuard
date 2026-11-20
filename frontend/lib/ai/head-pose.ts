/**
 * DriveGuard — Head Pose Estimation & Cockpit Zone Classifier in TypeScript.
 *
 * Direct, verified port of backend/app/features/head_pose.py.
 * Computes:
 * - 6-DoF Head Pose (Pitch, Yaw, Roll in degrees) from rigid cranial landmarks
 * - Closed-form intrinsic Y-X-Z Cardan/Tait-Bryan Euler angle decomposition
 * - Neutral baseline reconstruction & relative rotation: R_rel = R_neutral.T @ R
 * - Driver-centric coordinate convention:
 *     Positive yaw = turned to driver's left, Negative yaw = turned to right
 *     Negative pitch = nodding down, Positive pitch = looking up
 *     Positive roll = tilting right
 * - Cockpit Zone classification with transition hysteresis deadband (1.5°):
 *     CENTER, LEFT_MIRROR, RIGHT_MIRROR, DASHBOARD, OUT_OF_BOUNDS, UNKNOWN
 * - 2D Profile Fallback when extreme yaw occurs
 */

import { HeadPoseResult, NormalizedLandmark, PoseZone } from "./types";
import { Point2D } from "./ear";

export const SOLVEPNP_LANDMARK_INDICES = [1, 152, 33, 263, 61, 291, 6, 10] as const;
export const RIGID_SOLVEPNP_INDICES = [0, 2, 3, 6, 7] as const; // indices into CANONICAL_3D_FACE_POINTS (1, 33, 263, 6, 10)

export const CANONICAL_3D_FACE_POINTS: readonly [number, number, number][] = [
  [0.0, 0.0, 0.0], // 0: Nose tip (idx 1)
  [0.0, 63.6, 12.5], // 1: Chin (idx 152)
  [-43.3, -32.7, 26.0], // 2: Right eye outer corner (idx 33)
  [43.3, -32.7, 26.0], // 3: Left eye outer corner (idx 263)
  [-28.9, 28.9, 24.1], // 4: Right mouth corner (idx 61)
  [28.9, 28.9, 24.1], // 5: Left mouth corner (idx 291)
  [0.0, -31.0, 22.0], // 6: Sellion / nose bridge (idx 6)
  [0.0, -68.0, 35.0], // 7: Glabella / forehead (idx 10)
];

const CANONICAL_EYE_WIDTH_MM = CANONICAL_3D_FACE_POINTS[3][0] - CANONICAL_3D_FACE_POINTS[2][0]; // 86.6 mm
const FALLBACK_K_EYE_NOSE = (CANONICAL_3D_FACE_POINTS[2][2] - CANONICAL_3D_FACE_POINTS[0][2]) / CANONICAL_EYE_WIDTH_MM;
const FALLBACK_K_SELLION_NOSE = (CANONICAL_3D_FACE_POINTS[6][2] - CANONICAL_3D_FACE_POINTS[0][2]) / CANONICAL_EYE_WIDTH_MM;

const FALLBACK_MIN_EYE_SPAN_PX = 10.0;
const FALLBACK_MIN_FACE_HEIGHT_PX = 15.0;
const FALLBACK_MIN_EYE_TO_HEIGHT_RATIO = 0.12;
const FALLBACK_NOSE_RATIO_TRIGGER_LOW = 0.12;
const FALLBACK_NOSE_RATIO_TRIGGER_HIGH = 0.88;
const FALLBACK_NOSE_RATIO_MIN = -0.85;
const FALLBACK_NOSE_RATIO_MAX = 1.85;
const FALLBACK_SELLION_RATIO_MIN = -0.35;
const FALLBACK_SELLION_RATIO_MAX = 1.35;
const FALLBACK_MAX_CUE_DISAGREEMENT_DEG = 30.0;
const FALLBACK_MIN_ABS_YAW_DEG = 25.0;
const FALLBACK_MAX_ABS_YAW_DEG = 75.0;

/**
 * Validate that rotMat is finite, 3x3, and orthogonal with det(R) ~ 1.0.
 */
export function isValidRotationMatrix(rotMat: number[][], tol: number = 1e-2): boolean {
  if (!rotMat || rotMat.length !== 3) return false;
  for (let i = 0; i < 3; i++) {
    if (!rotMat[i] || rotMat[i].length !== 3) return false;
    for (let j = 0; j < 3; j++) {
      if (!Number.isFinite(rotMat[i][j])) return false;
    }
  }

  // Check R^T @ R ~ I
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += rotMat[k][i] * rotMat[k][j];
      }
      const expected = i === j ? 1.0 : 0.0;
      if (Math.abs(sum - expected) > tol) {
        return false;
      }
    }
  }

  // Determinant check
  const det =
    rotMat[0][0] * (rotMat[1][1] * rotMat[2][2] - rotMat[1][2] * rotMat[2][1]) -
    rotMat[0][1] * (rotMat[1][0] * rotMat[2][2] - rotMat[1][2] * rotMat[2][0]) +
    rotMat[0][2] * (rotMat[1][0] * rotMat[2][1] - rotMat[1][1] * rotMat[2][0]);

  if (Math.abs(det - 1.0) > tol) {
    return false;
  }

  return true;
}

/**
 * Multiply two 3x3 matrices: A @ B
 */
export function matMul3x3(A: number[][], B: number[][]): number[][] {
  const result: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      let sum = 0;
      for (let k = 0; k < 3; k++) {
        sum += A[i][k] * B[k][j];
      }
      result[i][j] = sum;
    }
  }
  return result;
}

/**
 * Transpose 3x3 matrix: A^T
 */
export function transpose3x3(A: number[][]): number[][] {
  return [
    [A[0][0], A[1][0], A[2][0]],
    [A[0][1], A[1][1], A[2][1]],
    [A[0][2], A[1][2], A[2][2]],
  ];
}

/**
 * Extract intrinsic Y-X-Z Cardan/Tait-Bryan Euler angles in degrees from 3x3 rotation matrix.
 *
 * R = R_y(yaw) @ R_x(pitch) @ R_z(roll)
 */
export function extractEulerYXZ(rotMat: number[][]): [number, number, number] {
  const sinPitch = Math.max(-1.0, Math.min(1.0, -rotMat[1][2]));
  const pitchRad = Math.asin(sinPitch);

  const cosPitch = Math.hypot(rotMat[1][0], rotMat[1][1]);
  let yawRad = 0;
  let rollRad = 0;

  if (cosPitch > 1e-4) {
    yawRad = Math.atan2(rotMat[0][2], rotMat[2][2]);
    rollRad = Math.atan2(rotMat[1][0], rotMat[1][1]);
  } else {
    yawRad = Math.atan2(-rotMat[2][0], rotMat[0][0]);
    rollRad = 0.0;
  }

  return [(yawRad * 180.0) / Math.PI, (pitchRad * 180.0) / Math.PI, (rollRad * 180.0) / Math.PI];
}

/**
 * Construct an orthonormal 3x3 neutral rotation matrix from calibrated angles or matrix.
 */
export function buildNeutralMatrix(
  neutralPitchDeg: number = 0.0,
  neutralYawDeg: number = 0.0,
  neutralRotMat?: number[][] | null,
  isMirrored: boolean = false,
  neutralIsMirrored?: boolean | null
): number[][] {
  if (neutralRotMat && isValidRotationMatrix(neutralRotMat)) {
    let mat = neutralRotMat.map((r) => [...r]);
    if (neutralIsMirrored !== undefined && neutralIsMirrored !== null && Boolean(neutralIsMirrored) !== Boolean(isMirrored)) {
      // Conjugation by M = diag(-1, 1, 1): R' = M @ R @ M
      const M = [
        [-1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
      ];
      mat = matMul3x3(matMul3x3(M, mat), M);
    }
    return mat;
  }

  const camYaw = isMirrored ? neutralYawDeg : -neutralYawDeg;
  const camPitch = -neutralPitchDeg;
  const psi = (camYaw * Math.PI) / 180.0;
  const theta = (camPitch * Math.PI) / 180.0;

  const Ry = [
    [Math.cos(psi), 0.0, Math.sin(psi)],
    [0.0, 1.0, 0.0],
    [-Math.sin(psi), 0.0, Math.cos(psi)],
  ];
  const Rx = [
    [1.0, 0.0, 0.0],
    [0.0, Math.cos(theta), -Math.sin(theta)],
    [0.0, Math.sin(theta), Math.cos(theta)],
  ];
  return matMul3x3(Ry, Rx);
}

/**
 * Classify continuous 3D Euler angles into discrete cockpit zones relative to driver neutral posture.
 */
export function classifyPoseZone(
  pitchDeg: number,
  yawDeg: number,
  rollDeg: number = 0.0,
  neutralPitchDeg: number = 0.0,
  neutralYawDeg: number = 0.0,
  currentZone?: PoseZone | null,
  hysteresisDeg: number = 1.5
): PoseZone {
  if (
    !Number.isFinite(pitchDeg) ||
    !Number.isFinite(yawDeg) ||
    !Number.isFinite(rollDeg) ||
    !Number.isFinite(neutralPitchDeg) ||
    !Number.isFinite(neutralYawDeg)
  ) {
    return PoseZone.UNKNOWN;
  }

  const relPitch = pitchDeg - neutralPitchDeg;
  const relYaw = yawDeg - neutralYawDeg;

  // 1. OUT_OF_BOUNDS
  const oobH = currentZone === PoseZone.OUT_OF_BOUNDS ? hysteresisDeg : 0.0;
  if (relYaw > 50.0 - oobH || relYaw < -70.0 + oobH || Math.abs(relPitch) > 38.0 - oobH) {
    return PoseZone.OUT_OF_BOUNDS;
  }

  // 2. LEFT_MIRROR
  const leftH = currentZone === PoseZone.LEFT_MIRROR ? hysteresisDeg : 0.0;
  if (relYaw >= 14.0 - leftH) {
    return PoseZone.LEFT_MIRROR;
  }

  // 3. RIGHT_MIRROR
  const rightH = currentZone === PoseZone.RIGHT_MIRROR ? hysteresisDeg : 0.0;
  if (relYaw <= -14.0 + rightH) {
    return PoseZone.RIGHT_MIRROR;
  }

  // 4. DASHBOARD (constrained to forward cone |relYaw| < 16.0°)
  const dashH = currentZone === PoseZone.DASHBOARD ? hysteresisDeg : 0.0;
  if (relPitch <= -6.0 + dashH && Math.abs(relYaw) < 16.0) {
    return PoseZone.DASHBOARD;
  }

  // 5. CENTER
  return PoseZone.CENTER;
}

/**
 * 2D Landmark Profile Fallback for severe profile yaw when 3D PnP cannot converge.
 */
export function estimate2DProfileFallbackYaw(
  imagePts: Point2D[],
  frameWidth: number,
  isMirrored: boolean
): number | null {
  if (!imagePts || imagePts.length < SOLVEPNP_LANDMARK_INDICES.length) {
    return null;
  }

  const nosePt = imagePts[0]; // idx 1: nose tip
  const chinPt = imagePts[1]; // idx 152: chin
  const leftEyePt = imagePts[2]; // idx 33: left outer corner
  const rightEyePt = imagePts[3]; // idx 263: right outer corner
  const sellionPt = imagePts[6]; // idx 6: sellion / nose bridge

  if (
    !Number.isFinite(nosePt.x) ||
    !Number.isFinite(nosePt.y) ||
    !Number.isFinite(chinPt.x) ||
    !Number.isFinite(chinPt.y) ||
    !Number.isFinite(leftEyePt.x) ||
    !Number.isFinite(leftEyePt.y) ||
    !Number.isFinite(rightEyePt.x) ||
    !Number.isFinite(rightEyePt.y) ||
    !Number.isFinite(sellionPt.x) ||
    !Number.isFinite(sellionPt.y)
  ) {
    return null;
  }

  const faceHeightY = chinPt.y - sellionPt.y;
  if (!(sellionPt.y < nosePt.y && nosePt.y < chinPt.y) || faceHeightY < FALLBACK_MIN_FACE_HEIGHT_PX) {
    return null;
  }

  const noseSellionDx = sellionPt.x - nosePt.x;
  if (Math.abs(noseSellionDx) >= 0.15 * frameWidth || Math.abs(noseSellionDx) >= 1.2 * faceHeightY) {
    return null;
  }

  const eyeSpanX = rightEyePt.x - leftEyePt.x;
  if (eyeSpanX <= FALLBACK_MIN_EYE_SPAN_PX) {
    return null;
  }
  if (eyeSpanX / faceHeightY < FALLBACK_MIN_EYE_TO_HEIGHT_RATIO) {
    return null;
  }

  const noseRatio = (nosePt.x - leftEyePt.x) / eyeSpanX;
  const sellionRatio = (sellionPt.x - leftEyePt.x) / eyeSpanX;

  if (noseRatio < FALLBACK_NOSE_RATIO_MIN || noseRatio > FALLBACK_NOSE_RATIO_MAX) return null;
  if (sellionRatio < FALLBACK_SELLION_RATIO_MIN || sellionRatio > FALLBACK_SELLION_RATIO_MAX) return null;

  if (noseRatio >= FALLBACK_NOSE_RATIO_TRIGGER_LOW && noseRatio <= FALLBACK_NOSE_RATIO_TRIGGER_HIGH) {
    return null;
  }

  const sign = isMirrored ? 1.0 : -1.0;
  const cueNoseEye = sign * (0.5 - noseRatio);
  const cueNoseSellion = sign * (noseSellionDx / eyeSpanX);

  const yawEyeDeg = (Math.atan2(cueNoseEye, FALLBACK_K_EYE_NOSE) * 180.0) / Math.PI;
  const yawSellionDeg = (Math.atan2(cueNoseSellion, FALLBACK_K_SELLION_NOSE) * 180.0) / Math.PI;

  if (yawEyeDeg * yawSellionDeg <= 0.0) return null;
  if (Math.abs(yawEyeDeg - yawSellionDeg) > FALLBACK_MAX_CUE_DISAGREEMENT_DEG) return null;

  const fusedYawDeg = 0.5 * (yawEyeDeg + yawSellionDeg);
  if (!Number.isFinite(fusedYawDeg) || Math.abs(fusedYawDeg) < FALLBACK_MIN_ABS_YAW_DEG) {
    return null;
  }

  return Math.max(-FALLBACK_MAX_ABS_YAW_DEG, Math.min(FALLBACK_MAX_ABS_YAW_DEG, fusedYawDeg));
}

/**
 * Closed-form Orthogonal Procrustes / Kabsch rotation alignment from 3D reference model to landmark coordinates.
 *
 * Aligns CANONICAL_3D_FACE_POINTS (rigid indices: nose, eye corners, sellion, glabella)
 * to observed 3D landmarks, solving for optimal 3x3 orthonormal rotation matrix R.
 */
function solveRigidRotationFrom3DLandmarks(
  landmarks: ArrayLike<NormalizedLandmark>,
  frameWidth: number,
  frameHeight: number
): number[][] | null {
  // Extract 3D points for rigid landmarks (1, 33, 263, 6, 10)
  const P: [number, number, number][] = [];
  const Q: [number, number, number][] = [];

  for (let i = 0; i < RIGID_SOLVEPNP_INDICES.length; i++) {
    const ptIdx = RIGID_SOLVEPNP_INDICES[i];
    const lmIdx = SOLVEPNP_LANDMARK_INDICES[ptIdx];
    const lm = landmarks[lmIdx];
    if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y) || !Number.isFinite(lm.z)) {
      return null;
    }

    // Model point P (in mm)
    P.push([
      CANONICAL_3D_FACE_POINTS[ptIdx][0],
      CANONICAL_3D_FACE_POINTS[ptIdx][1],
      CANONICAL_3D_FACE_POINTS[ptIdx][2],
    ]);

    // Observed point Q (scaled to aspect ratio / frame dimension)
    // MediaPipe landmark z is roughly on the same scale as x (normalized width)
    Q.push([
      (lm.x - 0.5) * frameWidth,
      (lm.y - 0.5) * frameHeight,
      lm.z * frameWidth,
    ]);
  }

  // Compute centroids
  const n = P.length;
  let pCentroid: [number, number, number] = [0, 0, 0];
  let qCentroid: [number, number, number] = [0, 0, 0];
  for (let i = 0; i < n; i++) {
    pCentroid[0] += P[i][0] / n;
    pCentroid[1] += P[i][1] / n;
    pCentroid[2] += P[i][2] / n;
    qCentroid[0] += Q[i][0] / n;
    qCentroid[1] += Q[i][1] / n;
    qCentroid[2] += Q[i][2] / n;
  }

  // Center points
  const pCentered: [number, number, number][] = P.map((p) => [
    p[0] - pCentroid[0],
    p[1] - pCentroid[1],
    p[2] - pCentroid[2],
  ]);
  const qCentered: [number, number, number][] = Q.map((q) => [
    q[0] - qCentroid[0],
    q[1] - qCentroid[1],
    q[2] - qCentroid[2],
  ]);

  // Covariance matrix H = P^T @ Q (3x3)
  const H: number[][] = [
    [0, 0, 0],
    [0, 0, 0],
    [0, 0, 0],
  ];
  for (let i = 0; i < n; i++) {
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        H[r][c] += pCentered[i][r] * qCentered[i][c];
      }
    }
  }

  // Direct polar decomposition / SVD approximation of 3x3 matrix H = R @ S
  // Compute R = H @ (H^T H)^(-1/2) using iterative matrix square root (Denman-Beavers / Schulz)
  let R = H.map((row) => [...row]);
  // Normalize initial guess
  let normH = 0;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      normH += H[r][c] * H[r][c];
    }
  }
  normH = Math.sqrt(normH) || 1.0;
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      R[r][c] /= normH;
    }
  }

  // 10 iterations of Schulz-Newton polar orthogonalization: R_{k+1} = 0.5 * R_k * (3I - R_k^T R_k)
  for (let iter = 0; iter < 12; iter++) {
    const RtR = matMul3x3(transpose3x3(R), R);
    const threeI_minus_RtR = [
      [3.0 - RtR[0][0], -RtR[0][1], -RtR[0][2]],
      [-RtR[1][0], 3.0 - RtR[1][1], -RtR[1][2]],
      [-RtR[2][0], -RtR[2][1], 3.0 - RtR[2][2]],
    ];
    const nextR = matMul3x3(R, threeI_minus_RtR);
    for (let r = 0; r < 3; r++) {
      for (let c = 0; c < 3; c++) {
        R[r][c] = 0.5 * nextR[r][c];
      }
    }
  }

  // Ensure det(R) == +1 (Special Orthogonal SO(3))
  const det =
    R[0][0] * (R[1][1] * R[2][2] - R[1][2] * R[2][1]) -
    R[0][1] * (R[1][0] * R[2][2] - R[1][2] * R[2][0]) +
    R[0][2] * (R[1][0] * R[2][1] - R[1][1] * R[2][0]);

  if (det < 0) {
    // Reflection correction: flip third column
    R[0][2] = -R[0][2];
    R[1][2] = -R[1][2];
    R[2][2] = -R[2][2];
  }

  return isValidRotationMatrix(R, 0.05) ? R : null;
}

/**
 * 6-DoF Head Pose Estimation from MediaPipe Facial Landmarks.
 */
export function estimateHeadPose(
  landmarks: ArrayLike<NormalizedLandmark>,
  frameWidth: number = 640,
  frameHeight: number = 480,
  neutralPitchDeg: number = 0.0,
  neutralYawDeg: number = 0.0,
  neutralRotMat?: number[][] | null,
  isMirrored: boolean = false,
  currentZone?: PoseZone | null,
  facialTransformMatrix?: number[] | number[][] | null,
  neutralIsMirrored?: boolean | null
): HeadPoseResult {
  const invalidResult: HeadPoseResult = {
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

  if (!landmarks) return invalidResult;

  const minRequired = Math.max(...SOLVEPNP_LANDMARK_INDICES) + 1;
  if (landmarks.length < minRequired) return invalidResult;

  // Extract pixel coordinates for landmark points
  const imagePts: Point2D[] = [];
  for (let i = 0; i < SOLVEPNP_LANDMARK_INDICES.length; i++) {
    const idx = SOLVEPNP_LANDMARK_INDICES[i];
    const lm = landmarks[idx];
    if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) {
      return invalidResult;
    }
    imagePts.push({ x: lm.x * frameWidth, y: lm.y * frameHeight });
  }

  let rotMat: number[][] | null = null;

  // 1. If MediaPipe provided a 4x4 facial transformation matrix directly, extract 3x3 rotation
  if (facialTransformMatrix) {
    if (Array.isArray(facialTransformMatrix)) {
      if (facialTransformMatrix.length === 16 && typeof facialTransformMatrix[0] === "number") {
        const m = facialTransformMatrix as number[];
        // Column-major or row-major check
        rotMat = [
          [m[0], m[1], m[2]],
          [m[4], m[5], m[6]],
          [m[8], m[9], m[10]],
        ];
      } else if (facialTransformMatrix.length === 4 && Array.isArray(facialTransformMatrix[0])) {
        const m = facialTransformMatrix as number[][];
        rotMat = [
          [m[0][0], m[0][1], m[0][2]],
          [m[1][0], m[1][1], m[1][2]],
          [m[2][0], m[2][1], m[2][2]],
        ];
      }
    }
    if (rotMat && !isValidRotationMatrix(rotMat, 0.1)) {
      rotMat = null;
    }
  }

  // 2. Solve rigid cranial rotation using 3D Procrustes
  if (!rotMat) {
    rotMat = solveRigidRotationFrom3DLandmarks(landmarks, frameWidth, frameHeight);
  }

  // 3. Fallback: 2D profile yaw if 3D alignment fails on extreme profile glances
  if (!rotMat) {
    const approxAbsYaw = estimate2DProfileFallbackYaw(imagePts, frameWidth, isMirrored);
    if (approxAbsYaw !== null) {
      let effNeutralYaw = neutralYawDeg;
      let effNeutralPitch = neutralPitchDeg;
      if (neutralRotMat && isValidRotationMatrix(neutralRotMat)) {
        const rNeutFb = buildNeutralMatrix(
          neutralPitchDeg,
          neutralYawDeg,
          neutralRotMat,
          isMirrored,
          neutralIsMirrored
        );
        const [nRawYaw, nRawPitch] = extractEulerYXZ(rNeutFb);
        effNeutralYaw = isMirrored ? nRawYaw : -nRawYaw;
        effNeutralPitch = -nRawPitch;
      }

      const approxRelYaw = approxAbsYaw - effNeutralYaw;
      const fallbackZone = classifyPoseZone(
        0.0,
        approxRelYaw,
        0.0,
        0.0,
        0.0,
        currentZone
      );

      if (fallbackZone === PoseZone.CENTER) {
        return invalidResult;
      }

      return {
        pitchDeg: effNeutralPitch,
        yawDeg: approxAbsYaw,
        rollDeg: 0.0,
        zone: fallbackZone,
        isLookingAway: true,
        isValid: true,
        relPitchDeg: 0.0,
        relYawDeg: approxRelYaw,
        relRollDeg: 0.0,
        rotMat: null,
      };
    }
    return invalidResult;
  }

  // Build calibrated neutral matrix: R_neutral
  const R_neutral = buildNeutralMatrix(
    neutralPitchDeg,
    neutralYawDeg,
    neutralRotMat,
    isMirrored,
    neutralIsMirrored
  );

  // Compute 3D relative rotation matrix: R_relative = R_neutral.T @ R_pnp
  let R_relative = matMul3x3(transpose3x3(R_neutral), rotMat);
  if (!isValidRotationMatrix(R_relative, 0.1)) {
    R_relative = rotMat.map((r) => [...r]);
  }

  // Closed-form intrinsic Y-X-Z Euler angle extraction
  const [absRawYaw, absRawPitch, absRawRoll] = extractEulerYXZ(rotMat);
  const [relRawYaw, relRawPitch, relRawRoll] = extractEulerYXZ(R_relative);

  // Convert to driver-centric convention:
  // Turning to driver's LEFT is positive (+yaw), RIGHT is negative (-yaw).
  // Nodding DOWN is negative (-pitch), UP is positive (+pitch).
  // Tilting toward RIGHT shoulder is positive (+roll).
  let yaw: number;
  let pitch: number;
  let roll: number;
  let relYaw: number;
  let relPitch: number;
  let relRoll: number;

  if (isMirrored) {
    yaw = absRawYaw;
    pitch = -absRawPitch;
    roll = -absRawRoll;
    relYaw = relRawYaw;
    relPitch = -relRawPitch;
    relRoll = -relRawRoll;
  } else {
    yaw = -absRawYaw;
    pitch = -absRawPitch;
    roll = absRawRoll;
    relYaw = -relRawYaw;
    relPitch = -relRawPitch;
    relRoll = relRawRoll;
  }

  const zone = classifyPoseZone(
    relPitch,
    relYaw,
    relRoll,
    0.0,
    0.0,
    currentZone
  );
  const isLookingAway = zone !== PoseZone.CENTER;

  return {
    pitchDeg: pitch,
    yawDeg: yaw,
    rollDeg: roll,
    zone,
    isLookingAway,
    isValid: true,
    relPitchDeg: relPitch,
    relYawDeg: relYaw,
    relRollDeg: relRoll,
    rotMat,
  };
}

export { estimateHeadPose as estimateHeadPoseFromLandmarks };
export { classifyPoseZone as classifyCockpitZone };


