"use client";

import { useState, useRef, useEffect, useCallback } from "react";

export type WebcamStatus = "idle" | "loading" | "ready" | "error";

export interface UseWebcamOptions {
  mirrored?: boolean;
  facingMode?: "user" | "environment";
  preferredDeviceId?: string;
  autoStart?: boolean;
  onStatusChange?: (ready: boolean) => void;
  onError?: (error: Error) => void;
}

export interface CaptureFrameOptions {
  mirror?: boolean;
  aspectRatio?: number; // e.g. 3/4 for portrait crop
  format?: string;
  quality?: number;
}

export function useWebcam(options: UseWebcamOptions = {}) {
  const {
    mirrored = true,
    facingMode = "user",
    preferredDeviceId,
    autoStart = false,
    onStatusChange,
    onError,
  } = options;

  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const [status, setStatus] = useState<WebcamStatus>("idle");
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [isPermissionDenied, setIsPermissionDenied] = useState<boolean>(false);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [selectedDeviceId, setSelectedDeviceId] = useState<string>(preferredDeviceId || "");
  const [currentFacingMode, setCurrentFacingMode] = useState<"user" | "environment">(facingMode);

  // Guards against race conditions and stale async operations
  const runIdRef = useRef<number>(0);
  const isCameraActiveRef = useRef<boolean>(false);

  // Keep callback refs stable to prevent unneeded effect re-runs
  const onStatusChangeRef = useRef(onStatusChange);
  useEffect(() => {
    onStatusChangeRef.current = onStatusChange;
  }, [onStatusChange]);

  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  // Enumerate connected video input devices
  const refreshDevices = useCallback(async () => {
    if (typeof navigator === "undefined" || !navigator.mediaDevices?.enumerateDevices) {
      return;
    }
    try {
      const allDevices = await navigator.mediaDevices.enumerateDevices();
      const videoInputs = allDevices.filter((d) => d.kind === "videoinput");
      setDevices(videoInputs);
    } catch (err) {
      console.warn("Unable to enumerate video input devices:", err);
    }
  }, []);

  // Stop active camera stream tracks
  const stopCamera = useCallback(() => {
    runIdRef.current++;
    isCameraActiveRef.current = false;

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

    setStatus("idle");
    setErrorMessage(null);
    setIsPermissionDenied(false);

    if (onStatusChangeRef.current) {
      onStatusChangeRef.current(false);
    }
  }, []);

  // Start the camera stream
  const startCamera = useCallback(
    async (deviceId?: string) => {
      runIdRef.current++;
      const currentRunId = runIdRef.current;
      isCameraActiveRef.current = true;

      // Clean up previous stream tracks
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => {
          try {
            t.stop();
          } catch (_) {}
        });
        streamRef.current = null;
      }
      if (videoRef.current) {
        videoRef.current.srcObject = null;
      }

      setStatus("loading");
      setErrorMessage(null);
      setIsPermissionDenied(false);

      if (onStatusChangeRef.current) {
        onStatusChangeRef.current(false);
      }

      if (typeof navigator === "undefined" || !navigator.mediaDevices?.getUserMedia) {
        if (currentRunId === runIdRef.current) {
          setStatus("error");
          setErrorMessage("Camera access is not supported by this browser. HTTPS or localhost is required.");
        }
        return;
      }

      try {
        const targetDeviceId = deviceId || selectedDeviceId;
        const constraints: MediaStreamConstraints = {
          video: targetDeviceId
            ? { deviceId: { exact: targetDeviceId } }
            : {
                facingMode: { ideal: currentFacingMode },
                width: { ideal: 1280 },
                height: { ideal: 720 },
              },
          audio: false,
        };

        let stream: MediaStream;
        try {
          stream = await navigator.mediaDevices.getUserMedia(constraints);
        } catch (initialErr: unknown) {
          console.warn("Preferred getUserMedia constraints failed, falling back to basic video:", initialErr);
          if (currentRunId !== runIdRef.current || !isCameraActiveRef.current) return;

          stream = await navigator.mediaDevices.getUserMedia({
            video: true,
            audio: false,
          });
        }

        // Ensure this invocation is still valid
        if (currentRunId !== runIdRef.current || !isCameraActiveRef.current) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }

        streamRef.current = stream;

        // Detect actual facing mode from track
        const track = stream.getVideoTracks()[0];
        if (track) {
          const settings = track.getSettings?.();
          if (settings?.facingMode === "environment") {
            setCurrentFacingMode("environment");
          } else if (settings?.facingMode === "user") {
            setCurrentFacingMode("user");
          }
        }

        const video = videoRef.current;
        if (video) {
          video.srcObject = stream;
          try {
            await video.play();
          } catch (playErr) {
            console.warn("Video play exception (may require user interaction):", playErr);
          }
        }

        if (currentRunId === runIdRef.current && isCameraActiveRef.current) {
          setStatus("ready");
          if (onStatusChangeRef.current) {
            onStatusChangeRef.current(true);
          }
          await refreshDevices();
        }
      } catch (err: unknown) {
        if (currentRunId !== runIdRef.current) return;

        setStatus("error");
        if (onStatusChangeRef.current) {
          onStatusChangeRef.current(false);
        }

        const error = err as Error & { name?: string };
        if (error.name === "NotAllowedError" || error.name === "PermissionDeniedError") {
          setIsPermissionDenied(true);
          setErrorMessage("Camera permission was denied. Please allow camera access in your browser settings.");
        } else if (error.name === "NotFoundError" || error.name === "DevicesNotFoundError") {
          setErrorMessage("No webcam device found on your computer or device.");
        } else if (error.name === "NotReadableError" || error.name === "TrackStartError") {
          setErrorMessage("Camera is currently in use by another application or tab.");
        } else {
          setErrorMessage(error.message || "Failed to initialize webcam stream.");
        }

        if (onErrorRef.current) {
          onErrorRef.current(error);
        }
      }
    },
    [selectedDeviceId, currentFacingMode, refreshDevices]
  );

  // Toggle camera power on or off
  const toggleCamera = useCallback(() => {
    if (status === "ready" || status === "loading") {
      stopCamera();
    } else {
      startCamera(selectedDeviceId || undefined);
    }
  }, [status, stopCamera, startCamera, selectedDeviceId]);

  // Switch to a different camera device
  const selectDevice = useCallback(
    (newDeviceId: string) => {
      setSelectedDeviceId(newDeviceId);
      if (status === "ready" || status === "loading") {
        startCamera(newDeviceId);
      }
    },
    [status, startCamera]
  );

  // Capture a still frame from the live video, supporting portrait crop matching the on-screen framing
  const captureFrame = useCallback(
    async (captureOptions: CaptureFrameOptions = {}): Promise<Blob | null> => {
      const video = videoRef.current;
      if (!video || !streamRef.current) {
        return null;
      }

      // If dimensions are still 0, wait briefly up to 300ms
      let vw = video.videoWidth;
      let vh = video.videoHeight;
      if (vw === 0 || vh === 0) {
        await new Promise<void>((res) => setTimeout(res, 200));
        vw = video.videoWidth;
        vh = video.videoHeight;
      }
      if (vw === 0 || vh === 0) return null;

      // Determine whether this capture should be mirrored
      const shouldMirror = captureOptions.mirror ?? (mirrored && currentFacingMode === "user");
      const targetRatio = captureOptions.aspectRatio ?? (3 / 4); // Default to portrait 3:4 crop
      const format = captureOptions.format || "image/jpeg";
      const quality = captureOptions.quality ?? 0.92;

      // Calculate center crop coordinates matching portrait framing
      let sourceX = 0;
      let sourceY = 0;
      let sourceWidth = vw;
      let sourceHeight = vh;

      if (targetRatio && targetRatio > 0) {
        const currentRatio = vw / vh;
        if (currentRatio > targetRatio) {
          // Video is wider than target (e.g. 16:9 vs 3:4) -> center crop width
          sourceWidth = Math.round(vh * targetRatio);
          sourceX = Math.round((vw - sourceWidth) / 2);
        } else if (currentRatio < targetRatio) {
          // Video is taller than target -> center crop height
          sourceHeight = Math.round(vw / targetRatio);
          sourceY = Math.round((vh - sourceHeight) / 2);
        }
      }

      let canvas = canvasRef.current;
      if (!canvas) {
        canvas = document.createElement("canvas");
      }

      canvas.width = sourceWidth;
      canvas.height = sourceHeight;

      const ctx = canvas.getContext("2d");
      if (!ctx) return null;

      ctx.save();
      if (shouldMirror) {
        // Horizontally flip the captured frame so it matches the mirrored preview
        ctx.translate(canvas.width, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(
        video,
        sourceX,
        sourceY,
        sourceWidth,
        sourceHeight,
        0,
        0,
        canvas.width,
        canvas.height
      );
      ctx.restore();

      return new Promise<Blob | null>((resolve) => {
        canvas?.toBlob((blob) => resolve(blob), format, quality);
      });
    },
    [mirrored, currentFacingMode]
  );

  // Start automatically if requested
  useEffect(() => {
    if (autoStart) {
      startCamera(selectedDeviceId || undefined);
    }

    return () => {
      stopCamera();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return {
    videoRef,
    canvasRef,
    status,
    isCameraOn: status === "ready" || status === "loading",
    isCameraReady: status === "ready",
    isLoading: status === "loading",
    isError: status === "error",
    errorMessage,
    isPermissionDenied,
    devices,
    selectedDeviceId,
    currentFacingMode,
    startCamera,
    stopCamera,
    toggleCamera,
    selectDevice,
    captureFrame,
    refreshDevices,
  };
}
