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
  EventAck,
} from "@/types/telemetry";

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

  // Consistent initial state on both SSR and client to prevent hydration mismatch
  const [backendType, setBackendType] = useState<"cloud" | "local">(defaultBackend || "local");

  // Sync client hostname strictly after mount
  useEffect(() => {
    if (!defaultBackend && typeof window !== "undefined") {
      const hostname = window.location.hostname;
      const isLocal = hostname === "localhost" || hostname === "127.0.0.1" || hostname === "::1";
      setBackendType(isLocal ? "local" : "cloud");
    }
  }, [defaultBackend]);

  const stableSessionId = useMemo(
    () => initialSessionId || `dg_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 6)}`,
    [initialSessionId]
  );

  const cloudApiBase =
    process.env.NEXT_PUBLIC_FASTAPI_URL || "https://driveguard-x048.onrender.com";
  const localApiBase = "http://127.0.0.1:8000";
  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  >("disconnected");
  const [telemetry, setTelemetry] = useState<FrameTelemetry | null>(null);
  const [latencyDiagnostics, setLatencyDiagnostics] = useState<LatencyDiagnostics | null>(null);
  const [activeSafetyEvent, setActiveSafetyEvent] = useState<SafetyEvent | null>(null);
  const [calibration, setCalibration] = useState<CalibrationPhase | null>(null);
  const [roundtripLatencyMs, setRoundtripLatencyMs] = useState<number>(0);
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>(
    "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
  );

  const roomRef = useRef<Room | null>(null);
  const publishedTrackRef = useRef<LocalVideoTrack | null>(null);
  const isConnectingRef = useRef<boolean>(false);
  const isLoopActiveRef = useRef<boolean>(false);
  const pendingCommandsRef = useRef<Array<Record<string, any>>>([]);

  // Telemetry deduplication, state guard, and latency tracking refs
  const lastTelemetrySeqRef = useRef<number>(-1);
  const activeStateRef = useRef<AlertState>("NORMAL");
  const processedEventIdsRef = useRef<Set<string>>(new Set());
  const rttSamplesRef = useRef<number[]>([]);
  const latestRttRef = useRef<number>(0);
  const pingSeqRef = useRef<number>(0);
  const pingIntervalRef = useRef<NodeJS.Timeout | null>(null);
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

  // Disconnect & cleanup LiveKit room
  const disconnectRoom = useCallback(async () => {
    isConnectingRef.current = false;
    isLoopActiveRef.current = false;
    lastTelemetrySeqRef.current = -1;
    latestRttRef.current = 0;

    if (pingIntervalRef.current) {
      clearInterval(pingIntervalRef.current);
      pingIntervalRef.current = null;
    }

    const currentRoom = roomRef.current;
    roomRef.current = null;
    publishedTrackRef.current = null;

    if (currentRoom) {
      try {
        if (currentRoom.state !== ConnectionState.Disconnected) {
          await currentRoom.disconnect();
        }
      } catch (_) {}
    }

    setConnectionStatus("disconnected");
  }, []);

  // Connect to LiveKit Cloud via signed token
  const connectLivekit = useCallback(
    async (videoElement?: HTMLVideoElement) => {
      // If already connected, ensure video track is published if provided
      if (roomRef.current && roomRef.current.state === ConnectionState.Connected) {
        if (videoElement && videoElement.srcObject && !publishedTrackRef.current) {
          const mediaStream = videoElement.srcObject as MediaStream;
          const videoTrack = mediaStream.getVideoTracks()[0];
          if (videoTrack) {
            try {
              console.log(`[DriveGuard LiveKit] Attaching camera track (${videoTrack.label})...`);
              const pub = await roomRef.current.localParticipant.publishTrack(videoTrack, {
                name: "driver-camera",
                simulcast: false,
              });
              publishedTrackRef.current = pub.track as LocalVideoTrack;
            } catch (err) {
              console.warn("[DriveGuard LiveKit] Track publish error:", err);
            }
          }
        }
        return;
      }

      if (isConnectingRef.current) return;

      // Cleanly teardown any prior room instance before re-minting connection
      if (roomRef.current) {
        try {
          await roomRef.current.disconnect();
        } catch (_) {}
        roomRef.current = null;
        publishedTrackRef.current = null;
      }

      lastTelemetrySeqRef.current = -1;
      isConnectingRef.current = true;
      setConnectionStatus("connecting");

      try {
        const base = backendType === "cloud" ? cloudApiBase : localApiBase;
        const tokenUrl = `${base}/api/v1/livekit/token?room_name=${stableSessionId}&driver_id=${driverId}&experiment_mode=${experimentMode}&auto_start_worker=true`;

        console.log(`[DriveGuard LiveKit] Requesting token from: ${tokenUrl}`);
        const tokenRes = await fetch(tokenUrl);
        if (!tokenRes.ok) {
          throw new Error(`Failed to acquire LiveKit token (${tokenRes.status}): ${await tokenRes.text()}`);
        }

        const data: { token: string; url: string; room_name: string; participant_identity: string } =
          await tokenRes.json();

        // Instantiate WebRTC Room
        const room = new Room({
          adaptiveStream: false,
          dynacast: false,
        });
        roomRef.current = room;

        // Wire Room Events
        room.on(RoomEvent.ConnectionStateChanged, (state: ConnectionState) => {
          if (state === ConnectionState.Connected) {
            setConnectionStatus("connected");
          } else if (state === ConnectionState.Reconnecting) {
            setConnectionStatus("reconnecting");
          } else if (state === ConnectionState.Disconnected) {
            setConnectionStatus("disconnected");
          }
        });

        // Helper to update DataChannel RTT diagnostics and percentiles
        const updateRttSample = (
          rttMs: number,
          queueMs?: number,
          inferMs?: number,
          serverTotalMs?: number
        ) => {
          if (rttMs <= 0 || isNaN(rttMs)) return;
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
                : (effectiveQueue + effectiveInfer);

            // Estimated glass-to-decision latency: (RTT / 2) + Server Queue + Server Inference + (RTT / 2)
            // = RTT + Server Processing Total
            const estimatedE2E = (p50 > 0 ? p50 : rttMs) + effectiveServerTotal;

            return {
              datachannel_rtt_ms: roundedRtt,
              queue_latency_ms: Math.round(effectiveQueue * 10) / 10,
              infer_duration_ms: Math.round(effectiveInfer * 10) / 10,
              estimated_e2e_latency_ms: Math.round(estimatedE2E),
              rtt_p50: Math.round(p50),
              rtt_p95: Math.round(p95),
              last_measured_at: Date.now(),
            };
          });
        };

        // Listen for real-time telemetry from Python AI worker over WebRTC Data Channel
        room.on(RoomEvent.DataReceived, (payload: Uint8Array, participant?: RemoteParticipant) => {
          try {
            const rawText = new TextDecoder().decode(payload);
            const msg = JSON.parse(rawText);

            if (msg.type === "TELEMETRY" && msg.telemetry) {
              const seq = msg.seq ?? (msg.telemetry.frame_index ?? 0);
              // Stale packet rejection: discard packets with older or equal sequence numbers
              if (seq <= lastTelemetrySeqRef.current) {
                return;
              }
              lastTelemetrySeqRef.current = seq;

              const tel = msg.telemetry as FrameTelemetry;
              activeStateRef.current = tel.state;

              const qLat = tel.timing?.queue_latency_ms ?? 0;
              const infLat = tel.timing?.infer_duration_ms ?? (tel.inference_latency_ms ?? 0);
              const sTotal = tel.timing?.server_total_ms ?? (qLat + infLat);
              latestTimingRef.current = { queueMs: qLat, inferMs: infLat, serverTotalMs: sTotal };

              const enriched: FrameTelemetry = {
                ...tel,
                network_roundtrip_ms:
                  latestRttRef.current > 0 ? latestRttRef.current : tel.network_roundtrip_ms,
              };

              // Immediately invoke callback for external synchronous listeners
              if (onTelemetryRef.current) {
                onTelemetryRef.current(enriched);
              }

              // Throttle React state updates to ~20 FPS (max 1 update per 48ms) to prevent UI overload
              const now = performance.now();
              if (now - lastReactUpdateRef.current >= 48) {
                lastReactUpdateRef.current = now;
                setTelemetry(enriched);
              }

              if (msg.calibration) {
                const cal = msg.calibration as CalibrationPhase;
                setCalibration(cal);
                if (onCalibrationRef.current) {
                  onCalibrationRef.current(cal);
                }
              }
            } else if (msg.type === "SAFETY_EVENT") {
              const safetyEvent = msg as SafetyEvent;
              console.log(
                `[DriveGuard LiveKit] Safety event received: ${safetyEvent.previous_state} -> ${safetyEvent.new_state} (ID: ${safetyEvent.event_id})`
              );

              // 1. Immediately acknowledge event back to Python worker over reliable DataChannel
              if (roomRef.current && roomRef.current.state === ConnectionState.Connected) {
                const ackPayload: EventAck = {
                  action: "EVENT_ACK",
                  event_id: safetyEvent.event_id,
                  event_seq: safetyEvent.event_seq,
                  client_timestamp: performance.now(),
                };
                try {
                  const ackBytes = new TextEncoder().encode(JSON.stringify(ackPayload));
                  roomRef.current.localParticipant
                    .publishData(ackBytes, { reliable: true })
                    .catch((ackErr) => {
                      console.warn("[DriveGuard LiveKit] Failed to send EVENT_ACK:", ackErr);
                    });
                } catch (ackErr) {
                  console.warn("[DriveGuard LiveKit] Failed to send EVENT_ACK:", ackErr);
                }
              }

              // 2. Idempotent deduplication (prevent re-triggering alert audio on retransmitted packets)
              if (!processedEventIdsRef.current.has(safetyEvent.event_id)) {
                processedEventIdsRef.current.add(safetyEvent.event_id);
                if (processedEventIdsRef.current.size > 200) {
                  const oldest = processedEventIdsRef.current.values().next().value;
                  if (oldest) processedEventIdsRef.current.delete(oldest);
                }

                activeStateRef.current = safetyEvent.new_state;
                setActiveSafetyEvent(safetyEvent);

                if (onSafetyEventRef.current) {
                  onSafetyEventRef.current(safetyEvent);
                }
              }
            } else if (msg.type === "STATE_SYNC") {
              console.log(
                `[DriveGuard LiveKit] State synchronization received from server. Current state: ${msg.current_state}`
              );
              if (msg.current_state) {
                activeStateRef.current = msg.current_state;
              }
              if (msg.latest_safety_event) {
                setActiveSafetyEvent(msg.latest_safety_event as SafetyEvent);
              }
            } else if (msg.type === "PONG") {
              if (msg.client_ts !== undefined && msg.client_ts !== null) {
                const rtt = Math.max(0, performance.now() - msg.client_ts);
                const timing = latestTimingRef.current;
                updateRttSample(rtt, timing.queueMs, timing.inferMs, timing.serverTotalMs);
              }
            } else if (
              msg.type === "CALIBRATION_STARTED" ||
              msg.type === "CALIBRATION_STEP_SKIPPED" ||
              msg.type === "CALIBRATION_COMPLETED"
            ) {
              if (msg.calibration) {
                const cal = msg.calibration as CalibrationPhase & { profile?: unknown };
                if (msg.profile) {
                  cal.profile = msg.profile;
                }
                setCalibration(cal);
                if (onCalibrationRef.current) {
                  onCalibrationRef.current(cal);
                }
              }
            } else if (msg.type === "EXPERIMENT_SWITCHED" && msg.mode) {
              setExperimentMode(msg.mode as ExperimentMode);
            }
          } catch (err) {
            console.warn("[DriveGuard LiveKit] Error parsing data packet:", err);
          }
        });

        room.on(RoomEvent.Disconnected, () => {
          console.log("[DriveGuard LiveKit] Room disconnected.");
          lastTelemetrySeqRef.current = -1;
          latestRttRef.current = 0;
          if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
          }
          setConnectionStatus("disconnected");
        });

        // Connect WebRTC to LiveKit Cloud
        console.log(`[DriveGuard LiveKit] Connecting to SFU: ${data.url}`);
        lastTelemetrySeqRef.current = -1;
        await room.connect(data.url, data.token);
        console.log(`[DriveGuard LiveKit] Successfully connected to room '${data.room_name}'!`);
        setConnectionStatus("connected");

        // Start periodic RTT ping probe (every 1s) to measure exact DataChannel latency
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
            room.localParticipant
              .publishData(payload, { reliable: false })
              .catch((pingErr) => {
                console.debug("[DriveGuard LiveKit] PING publish dropped:", pingErr);
              });
          } catch (_) {}
        }, 1000);

        // Request initial state synchronization from server upon connection/reconnection
        try {
          const syncMsg = { action: "SYNC_STATE" };
          const payload = new TextEncoder().encode(JSON.stringify(syncMsg));
          room.localParticipant
            .publishData(payload, { reliable: true })
            .catch((syncErr) => {
              console.warn("[DriveGuard LiveKit] SYNC_STATE publish failed:", syncErr);
            });
        } catch (_) {}

        // Flush any queued commands (e.g. START_CALIBRATION requested before handshake)
        while (pendingCommandsRef.current.length > 0) {
          const queuedCmd = pendingCommandsRef.current.shift();
          if (queuedCmd) {
            try {
              console.log(`[DriveGuard LiveKit] Dispatching queued command: ${queuedCmd.action}`);
              const payload = new TextEncoder().encode(JSON.stringify(queuedCmd));
              await room.localParticipant.publishData(payload, { reliable: true });
            } catch (err) {
              console.error("[DriveGuard LiveKit] Failed to publish queued command:", err);
            }
          }
        }

        // Publish video track from camera stream
        if (videoElement && videoElement.srcObject) {
          const mediaStream = videoElement.srcObject as MediaStream;
          const videoTrack = mediaStream.getVideoTracks()[0];

          if (videoTrack) {
            const trackSettings = videoTrack.getSettings ? videoTrack.getSettings() : {};
            console.log(`[DriveGuard LiveKit] Publishing camera track (${videoTrack.label}):`, {
              actualWidth: trackSettings.width,
              actualHeight: trackSettings.height,
              actualFps: trackSettings.frameRate,
            });
            const pub = await room.localParticipant.publishTrack(videoTrack, {
              name: "driver-camera",
              simulcast: false,
              videoEncoding: {
                maxBitrate: 650000,
                maxFramerate: 25,
              },
            });
            publishedTrackRef.current = pub.track as LocalVideoTrack;
            console.log("[DriveGuard LiveKit] Camera track published successfully with 640x480@25fps encoding profile!");
          }
        }
      } catch (err: any) {
        console.error("[DriveGuard LiveKit] Connection failure:", err);
        setConnectionStatus("error");
        if (onErrorRef.current) {
          onErrorRef.current(err instanceof Error ? err : new Error(String(err)));
        }
      } finally {
        isConnectingRef.current = false;
      }
    },
    [backendType, cloudApiBase, localApiBase, stableSessionId, driverId, experimentMode]
  );

  // Send control actions over WebRTC Data Channel (reliable mode)
  const sendCommand = useCallback(async (cmd: Record<string, any>) => {
    const room = roomRef.current;
    if (room && room.state === ConnectionState.Connected) {
      try {
        const payload = new TextEncoder().encode(JSON.stringify(cmd));
        await room.localParticipant.publishData(payload, { reliable: true });
        console.log("[DriveGuard LiveKit] Published command immediately:", cmd.action);
      } catch (err) {
        console.error("[DriveGuard LiveKit] Failed to publish control data:", err);
      }
    } else {
      console.log("[DriveGuard LiveKit] Room not connected yet. Queueing command for dispatch upon connection:", cmd.action);
      pendingCommandsRef.current.push(cmd);
    }
  }, []);

  // Start LiveKit streaming loop
  const startStreamingLoop = useCallback(
    (videoElement: HTMLVideoElement, isMirrored = true) => {
      isLoopActiveRef.current = true;
      if (!roomRef.current || roomRef.current.state === ConnectionState.Disconnected) {
        connectLivekit(videoElement);
      } else if (roomRef.current.state === ConnectionState.Connected && !publishedTrackRef.current) {
        if (videoElement.srcObject) {
          const mediaStream = videoElement.srcObject as MediaStream;
          const videoTrack = mediaStream.getVideoTracks()[0];
          if (videoTrack) {
            const trackSettings = videoTrack.getSettings ? videoTrack.getSettings() : {};
            console.log(`[DriveGuard LiveKit] Publishing camera track (${videoTrack.label}):`, {
              actualWidth: trackSettings.width,
              actualHeight: trackSettings.height,
              actualFps: trackSettings.frameRate,
            });
            roomRef.current.localParticipant
              .publishTrack(videoTrack, {
                name: "driver-camera",
                simulcast: false,
                videoEncoding: {
                  maxBitrate: 650000,
                  maxFramerate: 25,
                },
              })
              .then((pub) => {
                publishedTrackRef.current = pub.track as LocalVideoTrack;
                console.log("[DriveGuard LiveKit] Camera track published successfully with 640x480@25fps encoding profile!");
              })
              .catch((err) => console.warn("[DriveGuard LiveKit] Track publish error:", err));
          }
        }
      }
    },
    [connectLivekit]
  );

  // Stop LiveKit streaming loop
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
    (fourZone = true, durationSec = 4.0) => {
      sendCommand({
        action: "START_CALIBRATION",
        four_zone: fourZone,
        zone_duration_sec: durationSec,
      });
    },
    [sendCommand]
  );

  const cancelCalibration = useCallback(() => {
    sendCommand({ action: "CANCEL_CALIBRATION" });
  }, [sendCommand]);

  const skipCalibrationStep = useCallback(() => {
    sendCommand({ action: "SKIP_CALIBRATION_STEP" });
  }, [sendCommand]);

  const recalibrateZone = useCallback(
    (zone: string, durationSec = 4.0) => {
      sendCommand({
        action: "RECALIBRATE_ZONE",
        zone,
        zone_duration_sec: durationSec,
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
      disconnectRoom().then(() => {
        // Will reconnect on next frame or explicit reconnect
      });
    },
    [disconnectRoom]
  );

  // Unmount cleanup
  useEffect(() => {
    return () => {
      disconnectRoom();
    };
  }, [disconnectRoom]);

  return {
    isConnected: connectionStatus === "connected",
    connectionStatus,
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
