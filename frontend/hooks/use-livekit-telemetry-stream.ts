"use client";

import { useState, useRef, useEffect, useCallback, useMemo } from "react";
import {
  Room,
  RoomEvent,
  LocalVideoTrack,
  RemoteParticipant,
  ConnectionState,
} from "livekit-client";
import {
  ExperimentMode,
  FrameTelemetry,
  CalibrationPhase,
  SafetyEvent,
  LatencyDiagnostics,
  AlertState,
  BackendProcessingStatus,
} from "@/types/telemetry";
import { LOCAL_FASTAPI_URL, resolveCloudFastApiUrl } from "@/lib/fastapi-url";
import {
  fetchBackendAccessToken,
  requestLiveKitToken,
  stopLiveKitWorker,
} from "@/lib/api/driveguard-api";
import {
  computeBackendProcessingStatus,
  createProtocolSequenceTracker,
  processDataChannelMessage,
  resetProtocolSequenceTracker,
  teardownLiveKitSession,
} from "@/lib/livekit/livekit-session-client";

export interface UseLivekitTelemetryStreamOptions {
  sessionId?: string;
  driverId?: string;
  defaultBackend?: "cloud" | "local";
  onTelemetry?: (telemetry: FrameTelemetry) => void;
  onSafetyEvent?: (event: SafetyEvent) => void;
  onCalibration?: (calibration: CalibrationPhase) => void;
  onError?: (err: Error) => void;
}

export function useLivekitTelemetryStream(options: UseLivekitTelemetryStreamOptions = {}) {
  const {
    sessionId: initialSessionId,
    driverId = "driver_default",
    defaultBackend,
    onTelemetry,
    onSafetyEvent,
    onCalibration,
    onError,
  } = options;

  // Default AI backend to Cloud ("cloud") unless explicitly overridden
  const [backendType, setBackendType] = useState<"cloud" | "local">(defaultBackend || "cloud");

  const stableSessionId = useMemo(
    () => initialSessionId || `dg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
    [initialSessionId]
  );

  const directCloudApiBase = resolveCloudFastApiUrl();
  const localApiBase = LOCAL_FASTAPI_URL;

  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  >("disconnected");
  const [backendProcessingStatus, setBackendProcessingStatus] =
    useState<BackendProcessingStatus>("idle");
  const [workerError, setWorkerError] = useState<string | null>(null);
  const [telemetry, setTelemetry] = useState<FrameTelemetry | null>(null);
  const [latencyDiagnostics, setLatencyDiagnostics] = useState<LatencyDiagnostics | null>(null);
  const [activeSafetyEvent, setActiveSafetyEvent] = useState<SafetyEvent | null>(null);
  const [calibration, setCalibration] = useState<CalibrationPhase | null>(null);
  const [roundtripLatencyMs, setRoundtripLatencyMs] = useState<number>(0);
  const [isStale, setIsStale] = useState<boolean>(false);
  const [lastTelemetryAt, setLastTelemetryAt] = useState<number | null>(null);
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>(
    "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
  );

  const roomRef = useRef<Room | null>(null);
  const publishedTrackRef = useRef<LocalVideoTrack | null>(null);
  const publishedMediaTrackIdRef = useRef<string | null>(null);
  const isPublishingTrackRef = useRef<boolean>(false);
  const lastVideoElementRef = useRef<HTMLVideoElement | undefined>(undefined);
  const isConnectingRef = useRef<boolean>(false);
  const isLoopActiveRef = useRef<boolean>(false);
  const pendingCommandsRef = useRef<Array<Record<string, unknown>>>([]);
  const authTokenRef = useRef<string | null>(null);
  const connectedAtRef = useRef<number | null>(null);
  const lastTelemetryReceivedAtRef = useRef<number | null>(null);
  const calibrationCompletionEmittedRef = useRef<boolean>(false);

  // Protocol sequence tracker, state guards, and latency tracking refs
  const seqTrackerRef = useRef(createProtocolSequenceTracker());
  const activeStateRef = useRef<AlertState>("NORMAL");
  const lastRenderedStateRef = useRef<AlertState>("NORMAL");
  const rttSamplesRef = useRef<number[]>([]);
  const latestRttRef = useRef<number>(0);
  const pingSeqRef = useRef<number>(0);
  const pingIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const lastReactUpdateRef = useRef<number>(0);
  const latestTimingRef = useRef<{ queueMs: number; inferMs: number; serverTotalMs: number }>({
    queueMs: 0,
    inferMs: 0,
    serverTotalMs: 0,
  });

  // Synchronized callback refs
  const onTelemetryRef = useRef(onTelemetry);
  useEffect(() => {
    onTelemetryRef.current = onTelemetry;
  }, [onTelemetry]);

  const onSafetyEventRef = useRef(onSafetyEvent);
  useEffect(() => {
    onSafetyEventRef.current = onSafetyEvent;
  }, [onSafetyEvent]);

  const onCalibrationRef = useRef(onCalibration);
  useEffect(() => {
    onCalibrationRef.current = onCalibration;
  }, [onCalibration]);

  const onErrorRef = useRef(onError);
  useEffect(() => {
    onErrorRef.current = onError;
  }, [onError]);

  // Publish camera track once with mutex guard against duplicate publications
  const publishVideoTrackIfNeeded = useCallback(
    async (room: Room, videoElement?: HTMLVideoElement) => {
      if (!videoElement?.srcObject) return;
      if (room.state !== ConnectionState.Connected) return;
      if (isPublishingTrackRef.current) return;

      const mediaStream = videoElement.srcObject as MediaStream;
      const videoTrack = mediaStream.getVideoTracks()[0];
      if (!videoTrack || videoTrack.readyState === "ended") return;

      // Already published this exact live track
      if (publishedTrackRef.current && publishedMediaTrackIdRef.current === videoTrack.id) {
        return;
      }

      isPublishingTrackRef.current = true;
      try {
        // Unpublish stale track if camera device switched
        if (publishedTrackRef.current) {
          try {
            await room.localParticipant.unpublishTrack(publishedTrackRef.current, false);
          } catch {
            // ignore
          }
          publishedTrackRef.current = null;
          publishedMediaTrackIdRef.current = null;
        }

        const pub = await room.localParticipant.publishTrack(videoTrack, {
          name: "driver-camera",
          simulcast: false,
          videoEncoding: {
            maxBitrate: 650000,
            maxFramerate: 25,
          },
        });
        publishedTrackRef.current = (pub?.track as LocalVideoTrack) ?? null;
        publishedMediaTrackIdRef.current = videoTrack.id;
      } catch (err) {
        console.warn("[DriveGuard LiveKit] Track publish error:", err);
      } finally {
        isPublishingTrackRef.current = false;
      }
    },
    []
  );

  // Stale telemetry & initial-frame watchdog
  useEffect(() => {
    const syncStatus = () => {
      const next = computeBackendProcessingStatus({
        connectionStatus,
        connectedAtMs: connectedAtRef.current,
        lastTelemetryAtMs: lastTelemetryReceivedAtRef.current,
        nowMs: Date.now(),
      });
      setIsStale(next.isStale);
      setBackendProcessingStatus(next.status);
    };

    syncStatus();
    if (connectionStatus !== "connected") {
      return;
    }

    const timer = setInterval(syncStatus, 1000);
    return () => clearInterval(timer);
  }, [connectionStatus]);

  // Disconnect & cleanup LiveKit room, unpublish camera track, and stop Python worker with auth
  const disconnectRoom = useCallback(async () => {
    isConnectingRef.current = false;
    isLoopActiveRef.current = false;
    isPublishingTrackRef.current = false;
    resetProtocolSequenceTracker(seqTrackerRef.current);
    connectedAtRef.current = null;
    lastTelemetryReceivedAtRef.current = null;
    calibrationCompletionEmittedRef.current = false;
    activeStateRef.current = "NORMAL";
    lastRenderedStateRef.current = "NORMAL";
    latestRttRef.current = 0;
    rttSamplesRef.current = [];
    setIsStale(false);
    setLastTelemetryAt(null);
    setTelemetry(null);
    setActiveSafetyEvent(null);
    setCalibration(null);
    setLatencyDiagnostics(null);
    setRoundtripLatencyMs(0);
    setWorkerError(null);

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    const currentRoom = roomRef.current;
    const currentPublishedTrack = publishedTrackRef.current;
    roomRef.current = null;
    publishedTrackRef.current = null;
    publishedMediaTrackIdRef.current = null;

    if (currentRoom) {
      await teardownLiveKitSession({
        room: currentRoom as unknown as Parameters<typeof teardownLiveKitSession>[0]["room"],
        publishedTrack: currentPublishedTrack,
        roomName: stableSessionId,
        backendMode: backendType,
        localBaseUrl: localApiBase,
        bearerToken: authTokenRef.current,
        keepalive: true,
      });
    }

    setConnectionStatus("disconnected");
    setBackendProcessingStatus("idle");
  }, [backendType, localApiBase, stableSessionId]);

  // Connect to LiveKit SFU via authenticated FastAPI token endpoint
  const connectLivekit = useCallback(
    async (videoElement?: HTMLVideoElement) => {
      if (videoElement) {
        lastVideoElementRef.current = videoElement;
      }

      // If already connected, ensure video track is published once
      if (roomRef.current && roomRef.current.state === ConnectionState.Connected) {
        await publishVideoTrackIfNeeded(roomRef.current, videoElement ?? lastVideoElementRef.current);
        return;
      }

      if (isConnectingRef.current) return;

      // Cleanly tear down any prior room instance before re-minting connection
      if (roomRef.current) {
        try {
          if (publishedTrackRef.current) {
            await roomRef.current.localParticipant.unpublishTrack(publishedTrackRef.current, false);
          }
          await roomRef.current.disconnect(false);
        } catch {
          // ignore
        }
        roomRef.current = null;
        publishedTrackRef.current = null;
        publishedMediaTrackIdRef.current = null;
      }

      resetProtocolSequenceTracker(seqTrackerRef.current);
      connectedAtRef.current = null;
      lastTelemetryReceivedAtRef.current = null;
      setLastTelemetryAt(null);
      setWorkerError(null);
      isConnectingRef.current = true;
      setConnectionStatus("connecting");
      setBackendProcessingStatus("connecting");

      try {
        // Acquire and cache short-lived backend Bearer token so both token minting and teardown are authenticated
        let bearerToken: string | null = null;
        try {
          bearerToken = await fetchBackendAccessToken();
          authTokenRef.current = bearerToken;
        } catch (authErr) {
          if (backendType === "local") {
            throw authErr;
          }
        }

        let data;
        try {
          data = await requestLiveKitToken(
            {
              roomName: stableSessionId,
              driverId,
              experimentMode,
              autoStartWorker: true,
            },
            {
              mode: backendType,
              localBaseUrl: localApiBase,
              bearerToken,
            }
          );
        } catch (tokenErr) {
          if (backendType === "cloud" && bearerToken && directCloudApiBase) {
            data = await requestLiveKitToken(
              {
                roomName: stableSessionId,
                driverId,
                experimentMode,
                autoStartWorker: true,
              },
              {
                mode: "local",
                localBaseUrl: directCloudApiBase,
                bearerToken,
              }
            );
          } else {
            throw tokenErr;
          }
        }

        // Instantiate WebRTC Room
        const room = new Room({
          adaptiveStream: false,
          dynacast: false,
        });
        roomRef.current = room;

        room.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
          if (state === ConnectionState.Connected) {
            setConnectionStatus("connected");
          } else if (state === ConnectionState.Reconnecting) {
            setConnectionStatus("reconnecting");
            setBackendProcessingStatus("connecting");
          } else if (state === ConnectionState.Disconnected) {
            setConnectionStatus("disconnected");
            setBackendProcessingStatus("idle");
          }
        });

        room.on(RoomEvent.Reconnected, () => {
          setConnectionStatus("connected");
          connectedAtRef.current = Date.now();
          // Ensure camera track is still published without duplicating it
          publishVideoTrackIfNeeded(room, lastVideoElementRef.current).catch(() => {});
          try {
            const payload = new TextEncoder().encode(JSON.stringify({ action: "SYNC_STATE" }));
            room.localParticipant.publishData(payload, { reliable: true }).catch(() => {});
          } catch {
            // ignore
          }
        });

        const updateRttSample = (
          rttMs: number,
          queueMs?: number,
          inferMs?: number,
          serverTotalMs?: number
        ) => {
          if (rttMs <= 0 || Number.isNaN(rttMs)) return;
          const roundedRtt = Math.round(rttMs);
          latestRttRef.current = roundedRtt;
          setRoundtripLatencyMs(roundedRtt);

          const samples = rttSamplesRef.current;
          samples.push(rttMs);
          if (samples.length > 40) samples.shift();

          const sorted = [...samples].sort((a, b) => a - b);
          const p50 = sorted[Math.floor(sorted.length * 0.5)];
          const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];

          setLatencyDiagnostics((prev) => {
            const effectiveQueue = queueMs !== undefined ? queueMs : (prev?.queue_latency_ms ?? 0);
            const effectiveInfer = inferMs !== undefined ? inferMs : (prev?.infer_duration_ms ?? 0);
            const effectiveServerTotal =
              serverTotalMs !== undefined && serverTotalMs > 0
                ? serverTotalMs
                : effectiveQueue + effectiveInfer;

            return {
              datachannel_rtt_ms: roundedRtt,
              queue_latency_ms: Math.round(effectiveQueue * 10) / 10,
              infer_duration_ms: Math.round(effectiveInfer * 10) / 10,
              server_total_ms: Math.round(effectiveServerTotal * 10) / 10,
              rtt_p50: Math.round(p50),
              rtt_p95: Math.round(p95),
              last_measured_at: Date.now(),
            };
          });
        };

        const updateServerTimings = (
          queueMs: number,
          inferMs: number,
          serverTotalMs?: number
        ) => {
          setLatencyDiagnostics((prev) => {
            const effectiveQueue = Math.round(queueMs * 10) / 10;
            const effectiveInfer = Math.round(inferMs * 10) / 10;
            const effectiveServerTotal =
              serverTotalMs !== undefined && serverTotalMs > 0
                ? serverTotalMs
                : effectiveQueue + effectiveInfer;

            const measuredRtt = latestRttRef.current;
            const p50 = prev?.rtt_p50 ?? (measuredRtt > 0 ? measuredRtt : 0);
            const p95 = prev?.rtt_p95 ?? (measuredRtt > 0 ? measuredRtt : 0);

            return {
              datachannel_rtt_ms: measuredRtt,
              queue_latency_ms: effectiveQueue,
              infer_duration_ms: effectiveInfer,
              server_total_ms: Math.round(effectiveServerTotal * 10) / 10,
              rtt_p50: Math.round(p50),
              rtt_p95: Math.round(p95),
              last_measured_at: prev?.last_measured_at ?? Date.now(),
            };
          });
        };

        // Real-time telemetry & control responses from Python AI worker over LiveKit DataChannel
        room.on(RoomEvent.DataReceived, (payload: Uint8Array, _participant?: RemoteParticipant) => {
          try {
            const rawText = new TextDecoder().decode(payload);
            const msg = JSON.parse(rawText);

            const parsed = processDataChannelMessage(msg, seqTrackerRef.current, {
              latestRttMs: latestRttRef.current,
              nowMs: performance.now(),
            });

            // Always send EVENT_ACK when protocol handler produced one (even for deduped events)
            if (parsed.ackToSend && roomRef.current && roomRef.current.state === ConnectionState.Connected) {
              try {
                const ackBytes = new TextEncoder().encode(JSON.stringify(parsed.ackToSend));
                roomRef.current.localParticipant
                  .publishData(ackBytes, { reliable: true })
                  .catch((ackErr) => {
                    console.warn("[DriveGuard LiveKit] Failed to send EVENT_ACK:", ackErr);
                  });
              } catch (ackErr) {
                console.warn("[DriveGuard LiveKit] Failed to send EVENT_ACK:", ackErr);
              }
            }

            if (parsed.kind === "TELEMETRY" && parsed.telemetry) {
              const nowWall = Date.now();
              lastTelemetryReceivedAtRef.current = nowWall;
              setLastTelemetryAt(nowWall);
              setIsStale(false);
              setBackendProcessingStatus("processing");
              setWorkerError(null);

              const enriched = parsed.telemetry;
              activeStateRef.current = enriched.state;

              const qLat = enriched.timing?.queue_latency_ms ?? 0;
              const infLat = enriched.timing?.infer_duration_ms ?? (enriched.inference_latency_ms ?? 0);
              const sTotal = enriched.timing?.server_total_ms ?? qLat + infLat;
              latestTimingRef.current = { queueMs: qLat, inferMs: infLat, serverTotalMs: sTotal };

              if (onTelemetryRef.current) {
                onTelemetryRef.current(enriched);
              }

              if (parsed.calibration) {
                const calFrame = parsed.calibration;
                if (!calFrame.is_completed) {
                  calibrationCompletionEmittedRef.current = false;
                  if (onCalibrationRef.current) onCalibrationRef.current(calFrame);
                } else if (!calibrationCompletionEmittedRef.current) {
                  calibrationCompletionEmittedRef.current = true;
                  if (onCalibrationRef.current) onCalibrationRef.current(calFrame);
                }
              }

              const isSafetyTransition = enriched.state !== lastRenderedStateRef.current;
              const nowPerf = performance.now();
              const shouldRender = isSafetyTransition || nowPerf - lastReactUpdateRef.current >= 33;

              if (shouldRender) {
                lastReactUpdateRef.current = nowPerf;
                lastRenderedStateRef.current = enriched.state;
                setTelemetry(enriched);
                updateServerTimings(qLat, infLat, sTotal);
                if (parsed.calibration) {
                  setCalibration(parsed.calibration);
                }
              } else if (parsed.calibration?.is_completed) {
                setCalibration(parsed.calibration);
              }
            } else if (parsed.kind === "SAFETY_EVENT" && parsed.safetyEvent) {
              const safetyEvent = parsed.safetyEvent;
              activeStateRef.current = safetyEvent.new_state;
              lastRenderedStateRef.current = safetyEvent.new_state;
              setActiveSafetyEvent(safetyEvent);

              setTelemetry((prev) =>
                prev
                  ? {
                      ...prev,
                      state: safetyEvent.new_state,
                      drowsiness_state:
                        safetyEvent.metrics?.drowsiness_state ?? prev.drowsiness_state,
                      alert_state: safetyEvent.metrics?.alert_state ?? prev.alert_state,
                      is_recovering:
                        typeof safetyEvent.metrics?.is_recovering === "boolean"
                          ? safetyEvent.metrics.is_recovering
                          : prev.is_recovering,
                      reason_codes: Array.isArray(safetyEvent.metrics?.reasons)
                        ? safetyEvent.metrics.reasons
                        : prev.reason_codes,
                    }
                  : prev
              );

              if (onSafetyEventRef.current) {
                onSafetyEventRef.current(safetyEvent);
              }
            } else if (parsed.kind === "STATE_SYNC" && parsed.stateSync) {
              if (parsed.stateSync.latestSafetyEvent) {
                setActiveSafetyEvent(parsed.stateSync.latestSafetyEvent);
              }
              if (parsed.stateSync.currentState) {
                activeStateRef.current = parsed.stateSync.currentState;
                lastRenderedStateRef.current = parsed.stateSync.currentState;
                setTelemetry((prev) =>
                  prev ? { ...prev, state: parsed.stateSync!.currentState! } : prev
                );
              }
            } else if (parsed.kind === "PONG" && parsed.pongClientTs !== undefined) {
              const rtt = Math.max(0, performance.now() - parsed.pongClientTs);
              const timing = latestTimingRef.current;
              updateRttSample(rtt, timing.queueMs, timing.inferMs, timing.serverTotalMs);
            } else if (parsed.kind === "CALIBRATION_EVENT") {
              if (parsed.calibrationEventType === "CALIBRATION_STARTED") {
                calibrationCompletionEmittedRef.current = false;
              }
              if (parsed.calibration) {
                const cal = parsed.calibration;
                setCalibration(cal);
                if (!cal.is_completed) {
                  calibrationCompletionEmittedRef.current = false;
                  if (onCalibrationRef.current) onCalibrationRef.current(cal);
                } else if (!calibrationCompletionEmittedRef.current) {
                  calibrationCompletionEmittedRef.current = true;
                  if (onCalibrationRef.current) onCalibrationRef.current(cal);
                }
              }
            } else if (parsed.kind === "CALIBRATION_CANCELLED") {
              calibrationCompletionEmittedRef.current = false;
              setCalibration(null);
            } else if (parsed.kind === "EXPERIMENT_SWITCHED" && parsed.experimentMode) {
              setExperimentMode(parsed.experimentMode);
            } else if (parsed.kind === "ERROR" && parsed.errorMessage) {
              setWorkerError(parsed.errorMessage);
              if (onErrorRef.current) {
                onErrorRef.current(new Error(`[${parsed.action ?? "WORKER"}] ${parsed.errorMessage}`));
              }
            }
          } catch (err) {
            console.warn("[DriveGuard LiveKit] Error parsing data packet:", err);
          }
        });

        room.on(RoomEvent.Disconnected, () => {
          resetProtocolSequenceTracker(seqTrackerRef.current);
          connectedAtRef.current = null;
          latestRttRef.current = 0;
          publishedTrackRef.current = null;
          publishedMediaTrackIdRef.current = null;
          if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
          }
          setConnectionStatus("disconnected");
          setBackendProcessingStatus("idle");
        });

        await room.connect(data.url, data.token);
        connectedAtRef.current = Date.now();
        setConnectionStatus("connected");
        setBackendProcessingStatus("awaiting_telemetry");

        // Periodic RTT ping probe (1s)
        if (pingIntervalRef.current) {
          clearInterval(pingIntervalRef.current);
        }
        pingIntervalRef.current = setInterval(() => {
          if (room.state !== ConnectionState.Connected) return;
          pingSeqRef.current += 1;
          const pingMsg = {
            action: "PING",
            client_ts: performance.now(),
            ping_id: pingSeqRef.current,
          };
          try {
            const payload = new TextEncoder().encode(JSON.stringify(pingMsg));
            room.localParticipant.publishData(payload, { reliable: false }).catch(() => {});
          } catch {
            // ignore
          }
        }, 1000);

        // Request initial state sync
        try {
          const payload = new TextEncoder().encode(JSON.stringify({ action: "SYNC_STATE" }));
          room.localParticipant.publishData(payload, { reliable: true }).catch(() => {});
        } catch {
          // ignore
        }

        // Flush queued commands
        while (pendingCommandsRef.current.length > 0) {
          const queuedCmd = pendingCommandsRef.current.shift();
          if (queuedCmd) {
            try {
              const payload = new TextEncoder().encode(JSON.stringify(queuedCmd));
              await room.localParticipant.publishData(payload, { reliable: true });
            } catch (err) {
              console.error("[DriveGuard LiveKit] Failed to publish queued command:", err);
            }
          }
        }

        // Publish camera video track once
        await publishVideoTrackIfNeeded(room, videoElement ?? lastVideoElementRef.current);
      } catch (err: unknown) {
        const errorObj = err instanceof Error ? err : new Error(String(err));
        console.error("[DriveGuard LiveKit] Connection failure:", errorObj);
        setConnectionStatus("error");
        setBackendProcessingStatus("error");
        setWorkerError(errorObj.message);
        if (onErrorRef.current) {
          onErrorRef.current(errorObj);
        }
      } finally {
        isConnectingRef.current = false;
      }
    },
    [
      backendType,
      directCloudApiBase,
      driverId,
      experimentMode,
      localApiBase,
      publishVideoTrackIfNeeded,
      stableSessionId,
    ]
  );

  // Send control actions over LiveKit reliable DataChannel
  const sendCommand = useCallback(async (cmd: Record<string, unknown>) => {
    const room = roomRef.current;
    if (room && room.state === ConnectionState.Connected) {
      try {
        const payload = new TextEncoder().encode(JSON.stringify(cmd));
        await room.localParticipant.publishData(payload, { reliable: true });
      } catch (err) {
        console.error("[DriveGuard LiveKit] Failed to publish control data:", err);
      }
    } else {
      pendingCommandsRef.current.push(cmd);
    }
  }, []);

  const startStreamingLoop = useCallback(
    (videoElement: HTMLVideoElement, _isMirrored = true) => {
      isLoopActiveRef.current = true;
      lastVideoElementRef.current = videoElement;
      if (!roomRef.current || roomRef.current.state === ConnectionState.Disconnected) {
        connectLivekit(videoElement);
      } else if (roomRef.current.state === ConnectionState.Connected) {
        publishVideoTrackIfNeeded(roomRef.current, videoElement).catch(() => {});
      }
    },
    [connectLivekit, publishVideoTrackIfNeeded]
  );

  const stopStreamingLoop = useCallback(() => {
    isLoopActiveRef.current = false;
    disconnectRoom();
  }, [disconnectRoom]);

  // Controls
  const switchExperiment = useCallback(
    (mode: ExperimentMode) => {
      setExperimentMode(mode);
      sendCommand({ action: "SWITCH_EXPERIMENT", mode });
    },
    [sendCommand]
  );

  const startCalibration = useCallback(
    (fourZone = true, durationSec = 4.0, includeBlinkBaseline = false, blinkDurationSec = 120.0) => {
      calibrationCompletionEmittedRef.current = false;
      sendCommand({
        action: "START_CALIBRATION",
        four_zone: fourZone,
        zone_duration_sec: durationSec,
        include_blink_baseline: includeBlinkBaseline,
        blink_baseline_duration_sec: blinkDurationSec,
      });
    },
    [sendCommand]
  );

  const cancelCalibration = useCallback(() => {
    calibrationCompletionEmittedRef.current = false;
    sendCommand({ action: "CANCEL_CALIBRATION" });
  }, [sendCommand]);

  const skipCalibrationStep = useCallback(() => {
    sendCommand({ action: "SKIP_CALIBRATION_STEP" });
  }, [sendCommand]);

  const recalibrateZone = useCallback(
    (zone: string, durationSec = 4.0, blinkDurationSec = 120.0) => {
      calibrationCompletionEmittedRef.current = false;
      sendCommand({
        action: "RECALIBRATE_ZONE",
        zone,
        zone_duration_sec: durationSec,
        blink_baseline_duration_sec: blinkDurationSec,
      });
    },
    [sendCommand]
  );

  const saveProfile = useCallback(
    (driverName = "Driver", vehicleId = "DriveGuard-01") => {
      sendCommand({
        action: "SAVE_PROFILE",
        driver_name: driverName,
        vehicle_id: vehicleId,
      });
    },
    [sendCommand]
  );

  const switchBackend = useCallback(
    (type: "cloud" | "local") => {
      setBackendType(type);
      disconnectRoom();
    },
    [disconnectRoom]
  );

  // Page unload cleanup so closing tab/window does not orphan backend LiveKit worker
  useEffect(() => {
    if (typeof window === "undefined") return;
    const handleUnload = () => {
      if (roomRef.current) {
        stopLiveKitWorker(
          stableSessionId,
          {
            mode: backendType,
            localBaseUrl: localApiBase,
            bearerToken: authTokenRef.current,
          },
          { keepalive: true }
        ).catch(() => {});
      }
    };
    window.addEventListener("pagehide", handleUnload);
    return () => {
      window.removeEventListener("pagehide", handleUnload);
    };
  }, [backendType, localApiBase, stableSessionId]);

  // Unmount cleanup
  useEffect(() => {
    return () => {
      disconnectRoom();
    };
  }, [disconnectRoom]);

  return {
    isConnected: connectionStatus === "connected",
    connectionStatus,
    backendProcessingStatus,
    workerError,
    isStale,
    lastTelemetryAt,
    roomName: stableSessionId,
    telemetry,
    latencyDiagnostics,
    activeSafetyEvent,
    calibration,
    roundtripLatencyMs,
    experimentMode,
    backendType,
    switchBackend,
    startStreamingLoop,
    stopStreamingLoop,
    sendCommand,
    switchExperiment,
    startCalibration,
    recalibrateZone,
    cancelCalibration,
    skipCalibrationStep,
    saveProfile,
    reconnect: connectLivekit,
  };
}
