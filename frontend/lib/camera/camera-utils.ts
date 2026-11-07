import {
  CameraDevice,
  CameraError,
  CameraFacingMode,
  CameraResolution,
} from "@/types/camera";

/**
 * Checks if the current browser environment supports WebRTC MediaDevices and getUserMedia.
 */
export function checkMediaDevicesSupport(): {
  isSupported: boolean;
  isSecureContext: boolean;
  errorReason?: string;
} {
  if (typeof window === "undefined" || typeof navigator === "undefined") {
    return {
      isSupported: false,
      isSecureContext: false,
      errorReason: "Camera APIs are only accessible in browser client environments.",
    };
  }

  const isSecure = window.isSecureContext || window.location.hostname === "localhost" || window.location.hostname === "127.0.0.1";
  if (!isSecure) {
    return {
      isSupported: false,
      isSecureContext: false,
      errorReason: "Browser media access requires a secure context (HTTPS or localhost).",
    };
  }

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    return {
      isSupported: false,
      isSecureContext: true,
      errorReason: "Your browser does not support the WebRTC getUserMedia API.",
    };
  }

  return { isSupported: true, isSecureContext: true };
}

/**
 * Builds recommended MediaStream constraints for automotive driver monitoring.
 * Phase 1 Target: 640x480 at 25 FPS, user-facing, audio disabled.
 */
export function getRecommendedConstraints(
  deviceId?: string,
  facingMode: "user" | "environment" = "user",
  idealWidth: number = 640,
  idealHeight: number = 480,
  idealFps: number = 25
): MediaStreamConstraints {
  const maxWidth = Math.max(1280, idealWidth);
  const maxHeight = Math.max(720, idealHeight);
  const maxFps = Math.max(30, idealFps);

  if (deviceId) {
    return {
      video: {
        deviceId: { exact: deviceId },
        width: { ideal: idealWidth, max: maxWidth },
        height: { ideal: idealHeight, max: maxHeight },
        frameRate: { ideal: idealFps, max: maxFps },
      },
      audio: false,
    };
  }

  return {
    video: {
      facingMode: { ideal: facingMode },
      width: { ideal: idealWidth, max: maxWidth },
      height: { ideal: idealHeight, max: maxHeight },
      frameRate: { ideal: idealFps, max: maxFps },
    },
    audio: false,
  };
}

/**
 * Fallback constraints for devices that reject specific resolution constraints.
 */
export function getFallbackConstraints(
  facingMode: "user" | "environment" = "user"
): MediaStreamConstraints {
  return {
    video: {
      facingMode: { ideal: facingMode },
    },
    audio: false,
  };
}

/**
 * Minimal fallback constraints when exact/ideal facing mode fails.
 */
export function getBasicConstraints(): MediaStreamConstraints {
  return {
    video: true,
    audio: false,
  };
}

/**
 * Inspects MediaDeviceInfo array and infers facing mode from device label keywords.
 */
export function parseCameraDevices(rawDevices: MediaDeviceInfo[]): CameraDevice[] {
  const videoDevices = rawDevices.filter((d) => d.kind === "videoinput");

  return videoDevices.map((device, idx) => {
    const label = device.label || `Camera ${idx + 1}`;
    const lower = label.toLowerCase();

    let facingMode: CameraFacingMode = "unknown";
    if (
      lower.includes("front") ||
      lower.includes("user") ||
      lower.includes("facetime") ||
      lower.includes("selfie") ||
      lower.includes("true_depth")
    ) {
      facingMode = "user";
    } else if (
      lower.includes("back") ||
      lower.includes("rear") ||
      lower.includes("environment") ||
      lower.includes("ultra wide") ||
      lower.includes("telephoto")
    ) {
      facingMode = "environment";
    }

    return {
      deviceId: device.deviceId,
      label,
      groupId: device.groupId,
      facingMode,
      isDefault: idx === 0,
    };
  });
}

/**
 * Extracts resolution and framerate from an active HTMLVideoElement and MediaStreamTrack.
 */
export function extractCameraResolution(
  video: HTMLVideoElement,
  track?: MediaStreamTrack
): CameraResolution | null {
  let w = video.videoWidth;
  let h = video.videoHeight;
  let frameRate: number | undefined;

  if (track && track.getSettings) {
    const settings = track.getSettings();
    if (settings.width && settings.height) {
      w = settings.width;
      h = settings.height;
    }
    if (settings.frameRate) {
      frameRate = Math.round(settings.frameRate);
    }
  }

  if (!w || !h) return null;

  return {
    width: w,
    height: h,
    aspectRatio: parseFloat((w / h).toFixed(2)),
    frameRate,
  };
}

/**
 * Standardizes browser DOMExceptions into structured CameraError objects with recovery actions.
 */
export function mapMediaErrorToCameraError(err: unknown): CameraError {
  const error = err as Error & { name?: string; constraint?: string };
  const name = error.name || "Error";
  const rawMessage = error.message || "";

  switch (name) {
    case "NotAllowedError":
    case "PermissionDeniedError":
      return {
        code: "permission-denied",
        name,
        message: "Camera permission was denied in your browser settings.",
        recoveryAction:
          "Click the camera or lock icon in your browser address bar, set Camera to 'Allow', and refresh this page.",
        technicalDetails: rawMessage,
      };

    case "NotFoundError":
    case "DevicesNotFoundError":
      return {
        code: "no-camera-available",
        name,
        message: "No camera hardware detected on this device.",
        recoveryAction:
          "Please connect a USB webcam or ensure your device's built-in camera is enabled in device settings.",
        technicalDetails: rawMessage,
      };

    case "NotReadableError":
    case "TrackStartError":
      return {
        code: "camera-already-in-use",
        name,
        message: "Camera is currently locked by another application or tab.",
        recoveryAction:
          "Close other video apps (Zoom, Microsoft Teams, Google Meet, FaceTime, or other browser tabs) and click 'Retry Camera'.",
        technicalDetails: rawMessage,
      };

    case "OverconstrainedError":
    case "ConstraintNotSatisfiedError":
      return {
        code: "connection-error",
        name,
        message: `Your camera does not support the requested resolution or framerate (${error.constraint || "resolution"}).`,
        recoveryAction:
          "DriveGuard will automatically fall back to standard hardware resolution. Click 'Retry Camera'.",
        technicalDetails: rawMessage,
      };

    case "SecurityError":
      return {
        code: "unsupported",
        name,
        message: "Camera access is blocked due to security or protocol restrictions.",
        recoveryAction:
          "Ensure you are accessing DriveGuard over HTTPS or on localhost.",
        technicalDetails: rawMessage,
      };

    case "AbortError":
      return {
        code: "connection-error",
        name,
        message: "Hardware stream initialization was aborted.",
        recoveryAction: "Click 'Retry Camera' to restart the video pipeline.",
        technicalDetails: rawMessage,
      };

    default:
      return {
        code: "connection-error",
        name,
        message: rawMessage || "An unexpected error occurred while accessing the webcam.",
        recoveryAction: "Ensure your camera is connected and click 'Retry Camera'.",
        technicalDetails: rawMessage,
      };
  }
}
