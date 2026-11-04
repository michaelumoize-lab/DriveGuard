/**
 * Core type definitions for the DriveGuard Webcam Simulation & Monitoring Architecture.
 * Strictly adheres to standard WebRTC browser MediaDevices APIs.
 */

export type CameraState =
  | "not-initialized"
  | "requesting-permission"
  | "permission-granted"
  | "active"
  | "paused"
  | "stopped"
  | "permission-denied"
  | "no-camera-available"
  | "camera-already-in-use"
  | "unsupported"
  | "connection-error";

export type CameraFacingMode = "user" | "environment" | "unknown";

export interface CameraDevice {
  deviceId: string;
  label: string;
  groupId?: string;
  facingMode: CameraFacingMode;
  isDefault?: boolean;
}

export interface CameraResolution {
  width: number;
  height: number;
  aspectRatio: number;
  frameRate?: number;
}

export interface FrameCaptureOptions {
  /** Output image mime format, defaults to image/jpeg */
  format?: "image/jpeg" | "image/png" | "image/webp";
  /** Compression quality between 0.0 and 1.0 (default 0.92) */
  quality?: number;
  /** Custom downscale width for network-efficient transmission */
  targetWidth?: number;
  /** Custom downscale height */
  targetHeight?: number;
  /** Whether to mirror the captured frame horizontally (useful for front-facing selfie view) */
  mirror?: boolean;
  /** Optional aspect ratio to center-crop before saving (e.g. 9/16 for portrait mobile HUD) */
  cropToAspectRatio?: number;
}

export interface CapturedFrame {
  blob: Blob;
  file: File;
  dataUrl: string;
  width: number;
  height: number;
  format: string;
  sizeBytes: number;
  timestamp: number;
}

export interface CameraError {
  code: CameraState;
  name: string;
  message: string;
  recoveryAction: string;
  technicalDetails?: string;
}

export interface UseCameraOptions {
  autoStart?: boolean;
  preferredDeviceId?: string;
  defaultFacingMode?: "user" | "environment";
  mirrored?: boolean;
  idealWidth?: number;
  idealHeight?: number;
  idealFps?: number;
  onStateChange?: (state: CameraState) => void;
  onError?: (error: CameraError) => void;
}

export interface FrameConsumer {
  id: string;
  name: string;
  onFrame: (frame: CapturedFrame) => Promise<void> | void;
}
