"use client";

import { useState, useRef, useEffect, useCallback } from "react";
import {
  CameraDevice,
  CameraError,
  CameraFacingMode,
  CameraResolution,
  CameraState,
  CapturedFrame,
  FrameCaptureOptions,
  UseCameraOptions,
} from "@/types/camera";
import {
  checkMediaDevicesSupport,
  extractCameraResolution,
  getBasicConstraints,
  getFallbackConstraints,
  getRecommendedConstraints,
  mapMediaErrorToCameraError,
  parseCameraDevices,
} from "@/lib/camera/camera-utils";
import { captureVideoFrame } from "@/lib/camera/capture-frame";

export function useCamera(options: UseCameraOptions = {}) {
  const {
    autoStart = false,
    preferredDeviceId,
    defaultFacingMode = "user",
    mirrored = true,
    idealWidth = 1280,
    idealHeight = 720,
    idealFps = 30,
    onStateChange,
    onError,
  } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const wakeLockRef = useRef<WakeLockSentinel | null>(null);
  const runIdRef = useRef<number>(0);

  // States
  const [cameraState, setCameraState] = useState<CameraState>("not-initialized");
  const [error, setError] = useState<CameraError | null>(null);
  const [devices, setDevices] = useState<CameraDevice[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(preferredDeviceId || "");
  const [facingMode, setFacingMode] = useState<"user" | "environment">(defaultFacingMode);
  const [resolution, setResolution] = useState<CameraResolution | null>(null);
  const [isMirrored, setIsMirrored] = useState<boolean>(mirrored);
  const [streamDurationSec, setStreamDurationSec] = useState<number>(0);

  // Synchronized callback refs
  const onStateChangeRef = useRef(onStateChange);
  useEffect(() => {
    onStateChangeRef.current = onStateChange;
  }, [onStateChange]);

  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  const updateState = useCallback((newState: CameraState) => {
    setCameraState(newState);
    if (onStateChangeRef.current) {
      onStateChangeRef.current(newState);
    }
  }, []);

  // Screen Wake Lock API management
  const acquireWakeLock = useCallback(async () => {
    try {
      if (typeof navigator !== "undefined" && "wakeLock" in navigator) {
        wakeLockRef.current = await navigator.wakeLock.request("screen");
      }
    } catch {
      // Non-fatal if wakeLock is denied or unsupported
    }
  }, []);

  const releaseWakeLock = useCallback(async () => {
    if (wakeLockRef.current) {
      try {
        await wakeLockRef.current.release();
      } catch {
        // ignore
      } finally {
        wakeLockRef.current = null;
      }
    }
  }, []);

  // Refresh available camera devices
  const refreshDevices = useCallback(async (): Promise<CameraDevice[]> => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
      return [];
    }

    try {
      const rawDevices = await navigator.mediaDevices.enumerateDevices();
      const parsed = parseCameraDevices(rawDevices);
      setDevices(parsed);
      return parsed;
    } catch (err) {
      console.warn("Unable to enumerate camera devices:", err);
      return [];
    }
  }, []);

  // Release all hardware tracks and clear video source
  const releaseStream = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => {
        try {
          track.stop();
        } catch (_) {}
      });
      streamRef.current = null;
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    releaseWakeLock();
    setResolution(null);
  }, [releaseWakeLock]);

  // Stop camera and transition to stopped
  const stopCamera = useCallback(() => {
    runIdRef.current++;
    releaseStream();
    updateState("stopped");
    setError(null);
  }, [releaseStream, updateState]);

  // Pause video feed without unmounting tracks (freeze frame)
  const pauseCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = false;
      });
      updateState("paused");
    }
  }, [updateState]);

  // Resume paused video feed
  const resumeCamera = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getVideoTracks().forEach((track) => {
        track.enabled = true;
      });
      updateState("active");
    }
  }, [updateState]);

  // Start the camera stream with constraints cascade and error recovery
  const startCamera = useCallback(
    async (deviceId?: string, targetFacingMode?: "user" | "environment") => {
      runIdRef.current++;
      const currentRunId = runIdRef.current;

      // Release previous hardware tracks
      releaseStream();

      // Check environment support
      const support = checkMediaDevicesSupport();
      if (!support.isSupported) {
        const customError: CameraError = {
          code: "unsupported",
          name: "UnsupportedEnvironment",
          message: support.errorReason || "Camera is not supported.",
          recoveryAction: "Open DriveGuard in a modern browser (Chrome, Edge, Safari) over HTTPS.",
        };
        setError(customError);
        updateState("unsupported");
        if (onErrorRef.current) onErrorRef.current(customError);
        return;
      }

      updateState("requesting-permission");
      setError(null);

      const activeFacing = targetFacingMode || facingMode;
      const activeDeviceId = deviceId || selectedDeviceId;

      try {
        let stream: MediaStream;

        // Try Stage 1: Recommended constraints (ideal 1280x720 @ 15-30fps)
        try {
          const constraints = getRecommendedConstraints(
            activeDeviceId || undefined,
            activeFacing,
            idealWidth,
            idealHeight,
            idealFps
          );
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (stage1Err: unknown) {
          // If canceled by a newer start/stop invocation, exit
          if (currentRunId !== runIdRef.current) return;

          console.warn("Recommended camera constraints rejected, falling back to basic facing mode:", stage1Err);

          // Try Stage 2: Facing mode fallback without strict resolution
          try {
            const fallbackConstraints = getFallbackConstraints(activeFacing);
            stream = await navigator.mediaDevices.getUserMedia(fallbackConstraints);
          } catch (stage2Err: unknown) {
            if (currentRunId !== runIdRef.current) return;

            console.warn("Facing mode fallback rejected, falling back to minimal video constraints:", stage2Err);

            // Try Stage 3: Minimal fallback
            const basicConstraints = getBasicConstraints();
            stream = await navigator.mediaDevices.getUserMedia(basicConstraints);
          }
        }

        // Abort if another action occurred during async getUserMedia
        if (currentRunId !== runIdRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        updateState("permission-granted");
        streamRef.current = stream;

        // Inspect actual track settings to confirm facing mode & device
        const videoTrack = stream.getVideoTracks()[0];
        if (videoTrack) {
          // Listen for sudden hardware detachment (e.g. unplugged USB camera)
          videoTrack.onended = () => {
            if (currentRunId === runIdRef.current) {
              const disconnectError: CameraError = {
                code: "connection-error",
                name: "DeviceDisconnected",
                message: "The camera device was disconnected or stopped unexpectedly.",
                recoveryAction: "Ensure the camera is connected and click 'Retry Camera'.",
              };
              setError(disconnectError);
              updateState("connection-error");
              releaseStream();
            }
          };

          const settings = videoTrack.getSettings ? videoTrack.getSettings() : {};
          if (settings.facingMode === "environment") {
            setFacingMode("environment");
            setIsMirrored(false);
          } else if (settings.facingMode === "user") {
            setFacingMode("user");
            setIsMirrored(true);
          }

          if (settings.deviceId) {
            setSelectedDeviceId(settings.deviceId);
          }
        }

        // Attach stream to HTMLVideoElement
        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          video.setAttribute("playsinline", "true");
          video.setAttribute("autoplay", "true");
          video.muted = true;

          try {
            await video.play();
          } catch (playErr) {
            console.warn("video.play() autoplay exception (will play on first interaction):", playErr);
          }

          // Measure resolution once metadata loads
          const updateRes = () => {
            const res = extractCameraResolution(video, videoTrack);
            if (res) setResolution(res);
          };

          if (video.readyState >= 1) {
            updateRes();
          } else {
            video.onloadedmetadata = updateRes;
          }
        }

        if (currentRunId === runIdRef.current) {
          updateState("active");
          acquireWakeLock();
          // Re-enumerate devices with newly granted permission to populate real device labels
          await refreshDevices();
        }
      } catch (err: unknown) {
        if (currentRunId !== runIdRef.current) return;

        releaseStream();
        const cameraError = mapMediaErrorToCameraError(err);
        setError(cameraError);
        updateState(cameraError.code);

        if (onErrorRef.current) {
          onErrorRef.current(cameraError);
        }
      }
    },
    [
      facingMode,
      selectedDeviceId,
      idealWidth,
      idealHeight,
      idealFps,
      releaseStream,
      updateState,
      acquireWakeLock,
      refreshDevices,
    ]
  );

  // Switch between front (user) and rear (environment) camera
  const switchFacingMode = useCallback(() => {
    const nextMode = facingMode === "user" ? "environment" : "user";
    setFacingMode(nextMode);
    setIsMirrored(nextMode === "user");
    setSelectedDeviceId(""); // Clear exact deviceId to allow facingMode constraint to resolve next device
    if (cameraState === "active" || cameraState === "requesting-permission") {
      startCamera(undefined, nextMode);
    }
  }, [facingMode, cameraState, startCamera]);

  // Select a specific device ID from dropdown
  const selectDevice = useCallback(
    (deviceId: string) => {
      setSelectedDeviceId(deviceId);
      if (cameraState === "active" || cameraState === "requesting-permission") {
        startCamera(deviceId);
      }
    },
    [cameraState, startCamera]
  );

  // Take a still frame snapshot
  const captureSnapshot = useCallback(
    async (options?: FrameCaptureOptions): Promise<CapturedFrame | null> => {
      const video = videoRef.current;
      if (!video || !streamRef.current) {
        return null;
      }

      const captureOptions: FrameCaptureOptions = {
        mirror: isMirrored,
        ...options,
      };

      return await captureVideoFrame(video, captureOptions);
    },
    [isMirrored]
  );

  // Listen to hardware device connection changes
  useEffect(() => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.addEventListener) {
      return;
    }

    const handleDeviceChange = () => {
      refreshDevices();
    };

    navigator.mediaDevices.addEventListener("devicechange", handleDeviceChange);
    refreshDevices();

    return () => {
      navigator.mediaDevices.removeEventListener("devicechange", handleDeviceChange);
    };
  }, [refreshDevices]);

  // Stream duration timer
  useEffect(() => {
    let timer: NodeJS.Timeout | null = null;
    if (cameraState === "active") {
      timer = setInterval(() => {
        setStreamDurationSec((prev) => prev + 1);
      }, 1000);
    } else if (cameraState === "stopped" || cameraState === "not-initialized") {
      setStreamDurationSec(0);
    }

    return () => {
      if (timer) clearInterval(timer);
    };
  }, [cameraState]);

  // Auto-start on mount if requested
  useEffect(() => {
    if (autoStart) {
      startCamera();
    }

    return () => {
      releaseStream();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    videoRef,
    cameraState,
    error,
    devices,
    selectedDeviceId,
    facingMode,
    resolution,
    isMirrored,
    setIsMirrored,
    streamDurationSec,
    isActive: cameraState === "active",
    isPaused: cameraState === "paused",
    isStopped: cameraState === "stopped" || cameraState === "not-initialized",
    isLoading: cameraState === "requesting-permission" || cameraState === "permission-granted",
    startCamera,
    stopCamera,
    pauseCamera,
    resumeCamera,
    switchFacingMode,
    selectDevice,
    captureSnapshot,
    refreshDevices,
  };
}
