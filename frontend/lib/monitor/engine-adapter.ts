/**
 * DriveGuard — Normalized Monitoring Engine Adapter, Two-Phase Exclusive Camera Handoff
 * & Bidirectional Python Profile Reconciliation (Phase 4C).
 *
 * Provides:
 * 1. `MonitoringEngineAdapter`: unified interface consumed by Live Monitor & Calibration Wizard
 *    across all three engine runtimes (`BROWSER`, `PYTHON_LOCAL`, `PYTHON_REMOTE`).
 * 2. `executeTwoPhaseCameraHandoff`: OS-safe release-before-acquire protocol guaranteeing
 *    browser `MediaStreamTrack.readyState === "ended"` or Python `/api/v1/local/stop`
 *    `camera_released === true`, plus a 200ms OS handle settle delay before the next engine starts.
 * 3. `reconcilePythonProfileWithServer`: non-blocking bidirectional sync between Python's local
 *    `ProfileStore` (`/api/v1/local/profile*`) and Next.js PostgreSQL (`engineType="PYTHON"`),
 *    strictly isolating `PYTHON` from `BROWSER` profiles.
 * 4. `normalizeLocalTelemetryPayload`: maps Python loopback WebSocket telemetry to canonical
 *    `FrameTelemetry` for dashboard HUDs without altering authoritative backend decisions.
 */

import { OperationalState } from "@/lib/ai";
import { LOCAL_FASTAPI_URL } from "@/lib/fastapi-url";
import { buildDriverProfileDtoPayload } from "@/lib/api/profile-mapper";
import type {
  AlertState,
  CalibrationPhase,
  FrameTelemetry,
} from "@/types/telemetry";

export type EngineAdapterId = "BROWSER" | "PYTHON_LOCAL" | "PYTHON_REMOTE";
export type EngineOwnerType = "BROWSER" | "PYTHON";
export type CameraOwnerKind = "BROWSER_WEBRTC" | "PYTHON_LOCAL_OPENCV";

export interface EngineProfileMeta {
  engineType: EngineOwnerType;
  isCalibrated: boolean;
  centerEarMedian?: number;
  profileVersion?: number;
  profileUuid?: string | null;
  syncState?: string;
}

export interface MonitoringEngineAdapter {
  readonly engineId: EngineAdapterId;
  readonly engineType: EngineOwnerType;
  readonly cameraOwner: CameraOwnerKind;
  readonly operationalState: OperationalState | "IDLE";
  readonly drowsinessState: AlertState;
  readonly isRunning: boolean;
  readonly isCalibrated: boolean;
  readonly calibrationStatus: CalibrationPhase | null;
  readonly telemetry: FrameTelemetry | null;
  readonly profileMeta: EngineProfileMeta;
  start: () => Promise<boolean>;
  stop: () => Promise<boolean>;
  releaseCamera: () => Promise<boolean>;
  startCalibration?: () => Promise<void> | void;
  skipCalibrationStep?: () => Promise<void> | void;
  cancelCalibration?: () => Promise<void> | void;
  recalibrateZone?: (zone: string) => Promise<void> | void;
}

export interface CameraHandoffOptions {
  /** Milliseconds to wait after hardware release for Windows DirectShow/MSMF handle settle. */
  settleDelayMs?: number;
  /** Whether to immediately invoke `toAdapter.start()` after release + settle. */
  autoStartTarget?: boolean;
  /** Optional custom sleep implementation for deterministic unit testing. */
  sleepImpl?: (ms: number) => Promise<void>;
}

export interface CameraHandoffResult {
  fromEngine: EngineAdapterId;
  toEngine: EngineAdapterId;
  previousReleased: boolean;
  settleDelayMs: number;
  targetStarted: boolean;
  error?: string;
}

const defaultSleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Stop every track on a browser MediaStream and confirm all tracks reached `readyState === "ended"`.
 */
export function stopAndVerifyBrowserMediaStream(
  stream: MediaStream | null | undefined
): boolean {
  if (!stream) return true;
  const tracks = typeof stream.getTracks === "function" ? stream.getTracks() : [];
  for (const track of tracks) {
    try {
      track.stop();
    } catch {
      // ignore already-stopped track errors
    }
  }
  return tracks.every((t) => t.readyState === "ended");
}

/**
 * Two-Phase Exclusive Camera Handoff (Decision B1 / Mandatory Test D):
 *
 * Phase 1: Stop previous engine and explicitly confirm physical camera release
 *          (browser MediaStreamTrack ended OR Python `/api/v1/local/stop` `camera_released=true`).
 * Phase 2: Wait `settleDelayMs` (default 200ms) for OS driver release (DirectShow / MSMF / V4L2),
 *          then optionally start the target engine.
 *
 * Never attempts to start the target engine if the previous camera handle failed to release.
 */
export async function executeTwoPhaseCameraHandoff(
  fromAdapter: Pick<MonitoringEngineAdapter, "engineId" | "stop" | "releaseCamera">,
  toAdapter: Pick<MonitoringEngineAdapter, "engineId" | "start">,
  options: CameraHandoffOptions = {}
): Promise<CameraHandoffResult> {
  const {
    settleDelayMs = 200,
    autoStartTarget = false,
    sleepImpl = defaultSleep,
  } = options;

  if (fromAdapter.engineId === toAdapter.engineId) {
    return {
      fromEngine: fromAdapter.engineId,
      toEngine: toAdapter.engineId,
      previousReleased: true,
      settleDelayMs: 0,
      targetStarted: false,
    };
  }

  let previousReleased = false;
  try {
    await fromAdapter.stop();
    previousReleased = await fromAdapter.releaseCamera();
  } catch (err) {
    return {
      fromEngine: fromAdapter.engineId,
      toEngine: toAdapter.engineId,
      previousReleased: false,
      settleDelayMs: 0,
      targetStarted: false,
      error: err instanceof Error ? err.message : String(err),
    };
  }

  if (!previousReleased) {
    return {
      fromEngine: fromAdapter.engineId,
      toEngine: toAdapter.engineId,
      previousReleased: false,
      settleDelayMs: 0,
      targetStarted: false,
      error: `Failed to confirm camera release from ${fromAdapter.engineId} before switching to ${toAdapter.engineId}.`,
    };
  }

  if (settleDelayMs > 0) {
    await sleepImpl(settleDelayMs);
  }

  let targetStarted = false;
  if (autoStartTarget) {
    try {
      targetStarted = await toAdapter.start();
    } catch (err) {
      return {
        fromEngine: fromAdapter.engineId,
        toEngine: toAdapter.engineId,
        previousReleased: true,
        settleDelayMs,
        targetStarted: false,
        error: err instanceof Error ? err.message : String(err),
      };
    }
  }

  return {
    fromEngine: fromAdapter.engineId,
    toEngine: toAdapter.engineId,
    previousReleased: true,
    settleDelayMs,
    targetStarted,
  };
}

export interface PythonReconcileOutcome {
  status:
    | "SYNCED"
    | "ADOPTED_SERVER"
    | "UPLOADED_LOCAL"
    | "CONFLICT_PENDING_REVIEW"
    | "OFFLINE_LOCAL_ONLY"
    | "NO_PROFILE"
    | "ERROR";
  localProfile?: Record<string, unknown> | null;
  serverProfile?: Record<string, unknown> | null;
  conflictHistoryId?: string | null;
  error?: string;
}

/**
 * Non-blocking bidirectional reconciliation between local Python ProfileStore
 * (`http://127.0.0.1:8000/api/v1/local/profile*`) and Next.js PostgreSQL
 * (`/api/calibration/profile?engineType=PYTHON` & `/api/calibration/save`).
 *
 * Strictly enforces `engineType === "PYTHON"` on both ends — NEVER crosses with `BROWSER`.
 */
export async function reconcilePythonProfileWithServer(params: {
  driverId: string;
  localBaseUrl?: string;
  fetchImpl?: typeof fetch;
}): Promise<PythonReconcileOutcome> {
  const {
    driverId,
    localBaseUrl = LOCAL_FASTAPI_URL,
    fetchImpl = fetch,
  } = params;

  const base = localBaseUrl.replace(/\/+$/, "");

  // 1. Read local Python JSON ProfileStore
  let localProfile: Record<string, any> | null = null;
  try {
    const localRes = await fetchImpl(
      `${base}/api/v1/local/profile?driver_id=${encodeURIComponent(driverId)}`,
      { method: "GET", headers: { Accept: "application/json" } }
    );
    if (localRes.ok) {
      const localJson = await localRes.json();
      if (localJson?.found && localJson?.profile) {
        localProfile = localJson.profile;
      }
    }
  } catch (err) {
    return {
      status: "ERROR",
      error: err instanceof Error ? err.message : String(err),
    };
  }

  // 2. Fetch PostgreSQL PYTHON profile (non-fatal if offline or unauthenticated)
  let serverProfile: Record<string, any> | null = null;
  try {
    const serverRes = await fetchImpl("/api/calibration/profile?engineType=PYTHON", {
      method: "GET",
      headers: { Accept: "application/json" },
    });
    if (!serverRes.ok) {
      return {
        status: localProfile ? "OFFLINE_LOCAL_ONLY" : "NO_PROFILE",
        localProfile,
      };
    }
    const serverJson = await serverRes.json();
    const candidate = serverJson?.profile ?? null;
    if (candidate && (candidate.engineType ?? "PYTHON").toUpperCase() === "PYTHON") {
      serverProfile = candidate;
    }
  } catch {
    return {
      status: localProfile ? "OFFLINE_LOCAL_ONLY" : "NO_PROFILE",
      localProfile,
    };
  }

  if (!localProfile && !serverProfile) {
    return { status: "NO_PROFILE", localProfile: null, serverProfile: null };
  }

  // 3. Server has PYTHON profile, local does not (or server is newer) -> reconcile into local Python store
  if (serverProfile && serverProfile.isCalibrated) {
    const remoteDto = buildDriverProfileDtoPayload(
      {
        ...serverProfile,
        engineType: "PYTHON",
      },
      driverId
    );
    try {
      const recRes = await fetchImpl(`${base}/api/v1/local/profile/reconcile`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          driver_id: driverId,
          remote_profile: remoteDto,
        }),
      });
      if (recRes.ok) {
        const recJson = await recRes.json();
        const action = String(recJson?.action ?? "");
        if (action === "ADOPTED_REMOTE" || action === "CONFLICT_RESOLVED_REMOTE_NEWER") {
          return {
            status: "ADOPTED_SERVER",
            localProfile: recJson.profile ?? localProfile,
            serverProfile,
          };
        }
        if (action === "UNCHANGED") {
          return {
            status: "SYNCED",
            localProfile: recJson.profile ?? localProfile,
            serverProfile,
          };
        }
        // Local is ahead or newer -> fall through to upload local PYTHON profile to PostgreSQL
        localProfile = recJson.profile ?? localProfile;
      }
    } catch {
      // Non-fatal: continue to local upload attempt if localProfile exists
    }
  }

  // 4. Local has calibrated PYTHON profile ahead of server -> save strictly under engineType="PYTHON"
  if (localProfile && localProfile.is_calibrated) {
    const payload = {
      engineType: "PYTHON" as const,
      profileUuid: localProfile.profile_uuid ?? undefined,
      baseServerVersion:
        typeof localProfile.base_server_version === "number"
          ? localProfile.base_server_version
          : undefined,
      calibrationMode: localProfile.calibration_mode || "FOUR_ZONE",
      centerEarMedian: localProfile.center_ear_median,
      centerEarMadScaled:
        localProfile.zone_baselines?.CENTER?.ear_mad_scaled ?? 0.018,
      blinkDurationMedianMs: localProfile.blink_duration_median_ms ?? 235,
      blinkDurationMadMs: localProfile.blink_duration_mad_ms ?? 40,
      blinkRcrMedian: localProfile.blink_rcr_median ?? 0.95,
      blinkRcrMad: localProfile.blink_rcr_mad ?? 0.18,
      blinkFreqMedianPerMin: localProfile.blink_freq_median_per_min ?? 16,
      blinkFreqMadPerMin: localProfile.blink_freq_mad_per_min ?? 4.5,
      baselinePerclosPct: localProfile.baseline_perclos_pct ?? 3.5,
      blinkAmplitudeMedian: localProfile.blink_amplitude_median ?? 0.14,
      temporalBlinksSampled: localProfile.temporal_blinks_sampled ?? 0,
      temporalWindowSec: localProfile.temporal_window_sec ?? 0,
      sensitivityLevel: localProfile.sensitivity_level ?? "STANDARD",
      neutralPitchDeg: localProfile.neutral_pitch_deg ?? 0,
      neutralYawDeg: localProfile.neutral_yaw_deg ?? 0,
      neutralRotMat: localProfile.neutral_rot_mat ?? null,
      calibrationIsMirrored: localProfile.calibration_is_mirrored ?? null,
      calibrationReliability: localProfile.calibration_reliability ?? 0.9,
      zoneBaselines: localProfile.zone_baselines ?? {},
    };

    try {
      const saveRes = await fetchImpl("/api/calibration/save", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (saveRes.status === 409) {
        const conflictJson = await saveRes.json().catch(() => ({}));
        return {
          status: "CONFLICT_PENDING_REVIEW",
          localProfile,
          serverProfile: conflictJson?.serverProfile ?? serverProfile,
          conflictHistoryId: conflictJson?.conflictHistoryId ?? null,
        };
      }
      if (saveRes.ok) {
        const savedJson = await saveRes.json().catch(() => ({}));
        return {
          status: "UPLOADED_LOCAL",
          localProfile,
          serverProfile: savedJson?.profile ?? serverProfile,
        };
      }
    } catch {
      return { status: "OFFLINE_LOCAL_ONLY", localProfile };
    }
  }

  return { status: "SYNCED", localProfile, serverProfile };
}

/**
 * Convert a loopback `/ws/local/telemetry` message from Python `LocalCameraController`
 * into the canonical `FrameTelemetry` consumed by `DriverCockpitHUD` & `InspectorWorkspace`.
 */
export function normalizeLocalTelemetryPayload(raw: Record<string, any>): FrameTelemetry {
  const eye = raw.eye_metrics ?? {};
  const pose = raw.head_pose ?? {};
  const mouth = raw.mouth_metrics ?? {};
  const opt = raw.optical_quality ?? {};
  const cascade = raw.cascade ?? {};

  const opState = (raw.operational_state ?? "MONITORING_ACTIVE") as FrameTelemetry["operational_state"];
  const rawState = String(raw.state ?? "MONITORING_UNAVAILABLE") as AlertState;

  return {
    session_id: String(raw.session_id ?? raw.session_code ?? "python_local"),
    driver_id: String(raw.driver_id ?? "driver_default"),
    engine_mode: "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE",
    frame_index: Number(raw.frame_id ?? 0),
    timestamp_ms: Number(raw.timestamp_ms ?? Date.now()),
    fps: Number(raw.fps ?? 0),
    inference_latency_ms: Number(raw.inference_latency_ms ?? 0),
    face_detected: Boolean(raw.face_detected),
    operational_state: opState,
    engine_type: "PYTHON",
    state: rawState,
    monitoring_status: String(raw.monitoring_status ?? opState),
    risk_score: Number(raw.risk_score ?? 0),
    pose_zone: (raw.pose_zone ?? pose.zone ?? "CENTER") as FrameTelemetry["pose_zone"],
    reason_codes: Array.isArray(raw.reason_codes) ? raw.reason_codes : [],
    eye_metrics: {
      ear_left: Number(eye.ear_left ?? 0),
      ear_right: Number(eye.ear_right ?? 0),
      ear_raw: Number(eye.ear_raw ?? 0),
      ear_normalized: Number(eye.ear_normalized ?? eye.ear_raw ?? 0),
      ear_closure_threshold: Number(eye.ear_closure_threshold ?? 0.21),
      is_eye_closed: Boolean(eye.is_eye_closed ?? false),
      perclos_pct: Number(eye.perclos_pct ?? 0),
      perclos_reliable: Boolean(eye.perclos_reliable ?? false),
      perclos_coverage_pct: Number(eye.perclos_coverage_pct ?? 0),
      blink_rate_per_min: Number(eye.blink_rate_per_min ?? 0),
      blink_duration_ms:
        typeof eye.blink_duration_ms === "number" ? eye.blink_duration_ms : null,
      blink_rcr: typeof eye.blink_rcr === "number" ? eye.blink_rcr : null,
      total_blink_count: Number(eye.total_blink_count ?? raw.total_blinks ?? 0),
      mar_yawn: Number(eye.mar_yawn ?? mouth.mar ?? 0),
      total_yawn_count: Number(eye.total_yawn_count ?? mouth.total_yawns ?? raw.total_yawns ?? 0),
      is_yawning: Boolean(eye.is_yawning ?? mouth.is_yawning ?? raw.is_yawning ?? false),
      eye_closure_duration_ms: Number(eye.eye_closure_duration_ms ?? 0),
    },
    head_pose: {
      pitch_deg: Number(pose.pitch_deg ?? 0),
      yaw_deg: Number(pose.yaw_deg ?? 0),
      roll_deg: Number(pose.roll_deg ?? 0),
      zone: (pose.zone ?? raw.pose_zone ?? "CENTER") as FrameTelemetry["pose_zone"],
    },
    mouth_metrics: {
      mar: Number(mouth.mar ?? 0),
      is_yawning: Boolean(mouth.is_yawning ?? false),
    },
    cascade: {
      cnn_execution_mode: String(cascade.cnn_execution_mode ?? "SHADOW_DISABLED"),
      arbitration_winner: String(cascade.arbitration_winner ?? "GEOMETRIC_AUTHORITATIVE"),
      onnx_inference_executed: Boolean(cascade.onnx_inference_executed ?? false),
      cnn_eye_prob: null,
    },
    optical_quality: {
      mean_luminance: Number(opt.luminance ?? 0),
      blur_variance: Number(opt.blur_variance ?? 0),
      status: String(opt.status ?? "OPTIMAL"),
    },
    runtime_mode: "remote",
  } as unknown as FrameTelemetry;
}
