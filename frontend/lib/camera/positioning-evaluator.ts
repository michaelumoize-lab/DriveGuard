/**
 * DriveGuard — Client-Side Automotive Camera Positioning & Framing Evaluator.
 *
 * Implements browser-based parity with backend/app/camera/positioning_guide.py.
 * Evaluates driver framing, face distance ratio, oblique yaw mounting angles,
 * and reticle centering.
 */

import { PositioningGuidance } from "@/types/camera";

export function evaluateCameraPositioning(params: {
  hasFace: boolean;
  faceBboxNorm?: [number, number, number, number]; // [xMin, yMin, xMax, yMax] in [0, 1]
  yawDeg?: number;
  pitchDeg?: number;
  rollDeg?: number;
}): PositioningGuidance {
  const { hasFace, faceBboxNorm, yawDeg = 0, pitchDeg = 0, rollDeg = 0 } = params;

  if (!hasFace || !faceBboxNorm) {
    return {
      isOptimal: false,
      hasFace: false,
      distanceStatus: "UNKNOWN",
      distanceFeedback: "Position your face in front of the camera.",
      faceHeightRatio: 0,
      centeringStatus: "UNKNOWN",
      centeringFeedback: "Awaiting face detection.",
      angleStatus: "UNKNOWN",
      angleFeedback: "Look straight toward the road.",
      yawDeg,
      pitchDeg,
      rollDeg,
    };
  }

  const [xMin, yMin, xMax, yMax] = faceBboxNorm;
  const faceHeightRatio = Math.max(0, yMax - yMin);
  const cx = (xMin + xMax) / 2.0;
  const cy = (yMin + yMax) / 2.0;

  // 1. Distance evaluation
  let distanceStatus: "PERFECT" | "TOO_FAR" | "TOO_CLOSE" = "PERFECT";
  let distanceFeedback = "Distance is optimal for DMS tracking.";
  if (faceHeightRatio < 0.18) {
    distanceStatus = "TOO_FAR";
    distanceFeedback = "Move slightly closer to the camera or adjust mount.";
  } else if (faceHeightRatio > 0.52) {
    distanceStatus = "TOO_CLOSE";
    distanceFeedback = "Step back slightly; face fills too much of the frame.";
  }

  // 2. Centering evaluation
  let centeringStatus: "CENTERED" | "TOO_LEFT" | "TOO_RIGHT" | "TOO_HIGH" | "TOO_LOW" = "CENTERED";
  let centeringFeedback = "Driver head is well centered in reticle.";
  if (cx < 0.32) {
    centeringStatus = "TOO_LEFT";
    centeringFeedback = "Shift slightly right to center in the camera frame.";
  } else if (cx > 0.68) {
    centeringStatus = "TOO_RIGHT";
    centeringFeedback = "Shift slightly left to center in the camera frame.";
  } else if (cy < 0.22) {
    centeringStatus = "TOO_HIGH";
    centeringFeedback = "Lower camera tilt; head is too high in frame.";
  } else if (cy > 0.72) {
    centeringStatus = "TOO_LOW";
    centeringFeedback = "Raise camera tilt; head is too low in frame.";
  }

  // 3. Angular alignment evaluation
  let angleStatus: "OPTIMAL" | "TOO_OBLIQUE" | "TOO_STEEP" | "TILTED" = "OPTIMAL";
  let angleFeedback = "Road gaze angle is within automotive tolerances.";
  if (Math.abs(yawDeg) > 30.0) {
    angleStatus = "TOO_OBLIQUE";
    angleFeedback = "Camera is mounted at too steep a side angle (> 30° yaw).";
  } else if (pitchDeg < -24.0 || pitchDeg > 20.0) {
    angleStatus = "TOO_STEEP";
    angleFeedback = "Adjust mount vertical pitch towards natural road eye level.";
  } else if (Math.abs(rollDeg) > 18.0) {
    angleStatus = "TILTED";
    angleFeedback = "Straighten phone mount; horizon is tilted.";
  }

  const isOptimal =
    distanceStatus === "PERFECT" &&
    centeringStatus === "CENTERED" &&
    angleStatus === "OPTIMAL";

  return {
    isOptimal,
    hasFace: true,
    distanceStatus,
    distanceFeedback,
    faceHeightRatio: Number(faceHeightRatio.toFixed(3)),
    centeringStatus,
    centeringFeedback,
    angleStatus,
    angleFeedback,
    yawDeg: Number(yawDeg.toFixed(1)),
    pitchDeg: Number(pitchDeg.toFixed(1)),
    rollDeg: Number(rollDeg.toFixed(1)),
  };
}
