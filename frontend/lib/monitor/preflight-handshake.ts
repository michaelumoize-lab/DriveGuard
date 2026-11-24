/**
 * DriveGuard — Automotive Pre-Flight Readiness Interlock Gate.
 *
 * Enforces ISO 26262 functional safety and truthful subsystem readiness:
 * - In Browser AI mode ("browser"), only the local Camera Sensor and MediaPipe
 *   3D FaceLandmarker mesh are hard blockers. Backend / Neon DB connectivity is
 *   advisory: if offline or unreachable, the monitor starts cleanly in
 *   OFFLINE_QUEUED persistence mode with zero Python or LiveKit dependency.
 * - In Remote AI mode ("remote"), backend telemetry connectivity is required.
 */

import { OpticalFrameQuality } from "@/types/camera";

export type PreFlightSubsystemId =
  | "camera"
  | "mediapipe"
  | "cascade"
  | "backend"
  | "audio";

export type PreFlightStatus = "pending" | "checking" | "ready" | "failed" | "warning";

export type PreFlightPersistenceMode = "ONLINE" | "OFFLINE_QUEUED";

export interface PreFlightCheckItem {
  id: PreFlightSubsystemId;
  label: string;
  status: PreFlightStatus;
  message: string;
  latencyMs?: number;
}

export interface PreFlightResult {
  isAllReady: boolean;
  canStart: boolean;
  persistenceMode: PreFlightPersistenceMode;
  items: PreFlightCheckItem[];
  failedSubsystems: string[];
}

export interface PreFlightCheckParams {
  runtimeMode?: "browser" | "remote";
  videoElement: HTMLVideoElement | null;
  opticalQuality?: OpticalFrameQuality | null;
  isMediaPipeReady: boolean;
  isCascadeReady?: boolean;
  isAudioUnlocked?: boolean;
}

export async function runPreFlightHandshake(
  params: PreFlightCheckParams
): Promise<PreFlightResult> {
  const {
    runtimeMode = "browser",
    videoElement,
    opticalQuality,
    isMediaPipeReady,
    isCascadeReady = true,
    isAudioUnlocked = true,
  } = params;

  const isBrowserMode = runtimeMode === "browser";
  const items: PreFlightCheckItem[] = [];
  let persistenceMode: PreFlightPersistenceMode = "ONLINE";

  // 1. Camera Hardware & Optical Feed Check
  const camStart = performance.now();
  if (
    videoElement &&
    videoElement.readyState >= 2 &&
    videoElement.videoWidth > 0 &&
    videoElement.videoHeight > 0
  ) {
    if (opticalQuality?.overallQuality === "SEVERELY_DEGRADED") {
      items.push({
        id: "camera",
        label: "Camera Sensor & Optics",
        status: "warning",
        message: `Video active (${videoElement.videoWidth}x${videoElement.videoHeight}), but environment degraded: ${opticalQuality.reasonCodes.join(", ")}`,
        latencyMs: Math.round(performance.now() - camStart),
      });
    } else {
      items.push({
        id: "camera",
        label: "Camera Sensor & Optics",
        status: "ready",
        message: `Optical stream locked: ${videoElement.videoWidth}x${videoElement.videoHeight} @ ${opticalQuality?.blurCondition || "SHARP"}`,
        latencyMs: Math.round(performance.now() - camStart),
      });
    }
  } else {
    items.push({
      id: "camera",
      label: "Camera Sensor & Optics",
      status: "failed",
      message: "Camera stream not initialized or no optical frames received.",
      latencyMs: Math.round(performance.now() - camStart),
    });
  }

  // 2. Landmark / Perception Engine Check
  if (isBrowserMode) {
    if (isMediaPipeReady) {
      items.push({
        id: "mediapipe",
        label: "3D FaceLandmarker Mesh",
        status: "ready",
        message: "Browser MediaPipe 478-landmark WASM engine primed locally.",
      });
    } else {
      items.push({
        id: "mediapipe",
        label: "3D FaceLandmarker Mesh",
        status: "failed",
        message: "3D landmark detection engine is initializing or not ready.",
      });
    }
  } else {
    if (isMediaPipeReady) {
      items.push({
        id: "mediapipe",
        label: "Remote AI Telemetry Link",
        status: "ready",
        message: "LiveKit WebRTC / Python AI worker telemetry link connected.",
      });
    } else {
      items.push({
        id: "mediapipe",
        label: "Remote AI Telemetry Link",
        status: "failed",
        message: "Remote Python AI worker is disconnected or not ready.",
      });
    }
  }

  // 3. Authoritative Geometric & Temporal Pipeline Check (Truthful — no ONNX claims)
  if (isBrowserMode) {
    items.push({
      id: "cascade",
      label: "Geometric & Temporal AI Pipeline",
      status: isCascadeReady ? "ready" : "warning",
      message: isCascadeReady
        ? "Browser MediaPipe + TypeScript PDBP geometric engine ready (zero backend required)."
        : "Browser geometric pipeline initializing; standing by for landmarks.",
    });
  } else {
    items.push({
      id: "cascade",
      label: "Python Geometric AI Pipeline",
      status: isCascadeReady ? "ready" : "warning",
      message: isCascadeReady
        ? "Authoritative Python PDBP geometric pipeline ready."
        : "Remote Python geometric pipeline warming up.",
    });
  }

  // 4. Persistence & Telemetry Sync Check
  const netStart = performance.now();
  const browserExplicitlyOffline =
    typeof navigator !== "undefined" &&
    typeof navigator.onLine === "boolean" &&
    !navigator.onLine;

  if (isBrowserMode && browserExplicitlyOffline) {
    persistenceMode = "OFFLINE_QUEUED";
    items.push({
      id: "backend",
      label: "Persistence & Telemetry Sync",
      status: "warning",
      message: "Offline mode active — telemetry & alerts will queue locally (OFFLINE_QUEUED).",
      latencyMs: 0,
    });
  } else {
    try {
      const timeoutMs = isBrowserMode ? 2500 : 5000;
      const res = await fetch("/api/dashboard/health", {
        method: "GET",
        headers: { "Cache-Control": "no-cache" },
        signal: AbortSignal.timeout(timeoutMs),
      });
      const netLatency = Math.round(performance.now() - netStart);

      if (res.ok) {
        const data = await res.json();
        const isDbOk = data.database?.status === "online";
        persistenceMode = "ONLINE";
        items.push({
          id: "backend",
          label: isBrowserMode ? "Persistence & Telemetry Sync" : "Backend Telemetry & DB",
          status: isDbOk ? "ready" : "warning",
          message: isDbOk
            ? `Persistence services connected (${netLatency} ms, Neon DB synced).`
            : `API reachable (${netLatency} ms), database sync pending.`,
          latencyMs: netLatency,
        });
      } else if (isBrowserMode) {
        persistenceMode = "OFFLINE_QUEUED";
        items.push({
          id: "backend",
          label: "Persistence & Telemetry Sync",
          status: "warning",
          message: `API returned HTTP ${res.status} — offline queue active (OFFLINE_QUEUED).`,
          latencyMs: netLatency,
        });
      } else {
        persistenceMode = "OFFLINE_QUEUED";
        items.push({
          id: "backend",
          label: "Backend Telemetry & DB",
          status: "failed",
          message: `Backend health check returned HTTP ${res.status}.`,
          latencyMs: netLatency,
        });
      }
    } catch {
      const netLatency = Math.round(performance.now() - netStart);
      persistenceMode = "OFFLINE_QUEUED";
      if (isBrowserMode) {
        items.push({
          id: "backend",
          label: "Persistence & Telemetry Sync",
          status: "warning",
          message: "Backend unreachable — offline mode active (OFFLINE_QUEUED).",
          latencyMs: netLatency,
        });
      } else {
        items.push({
          id: "backend",
          label: "Backend Telemetry & DB",
          status: "failed",
          message: "Cannot reach backend API. Check local network or server status.",
          latencyMs: netLatency,
        });
      }
    }
  }

  // 5. Auditory Warning Synthesizer Check
  if (isAudioUnlocked) {
    items.push({
      id: "audio",
      label: "Acoustic HMI Warning Subsystem",
      status: "ready",
      message: "Web Audio synthesizer active for ISO 15007 safety sweeps.",
    });
  } else {
    items.push({
      id: "audio",
      label: "Acoustic HMI Warning Subsystem",
      status: "warning",
      message: "Audio context suspended. User click required to enable chimes.",
    });
  }

  const failedSubsystems = items
    .filter((item) => item.status === "failed")
    .map((item) => item.label);

  const canStart = failedSubsystems.length === 0;
  const isAllReady = items.every((item) => item.status === "ready");

  return {
    isAllReady,
    canStart,
    persistenceMode,
    items,
    failedSubsystems,
  };
}
