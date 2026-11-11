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
import { LOCAL_FASTAPI_URL, resolveCloudFastApiUrl } from "@/lib/fastapi-url";

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

  // Use same-origin `/api/v1/*` proxy in cloud mode to avoid browser CORS preflight issues,
  // with direct sanitized Render URL fallback.
  const cloudApiBase = "";
  const directCloudApiBase = resolveCloudFastApiUrl();
  const localApiBase = LOCAL_FASTAPI_URL;
  const [connectionStatus, setConnectionStatus] = useState<
    "disconnected" | "connecting" | "connected" | "reconnecting" | "error"
  >("disconnected");
  const [telemetry, setTelemetry] = useState<FrameTelemetry | null>(null);
  const [latencyDiagnostics, setLatencyDiagnostics] = useState<LatencyDiagnostics | null>(null);
  const [activeSafetyEvent, setActiveSafetyEvent] = useState<SafetyEvent | null>(null);
  const [calibration, setCalibration] = useState<CalibrationPhase | null>(null);
  const [roundtripLatencyMs, setRoundtripLatencyMs] = useState<number>(0);
  const [isStale, setIsStale] = useState<boolean>(false);
  const [experimentMode, setExperimentMode] = useState<ExperimentMode>(
    "EXPERIMENT_C_CONFIDENCE_GATED_CASCADE"
  );

  const roomRef = useRef<Room | null>(null);
  const publishedTrackRef = useRef<LocalVideoTrack | null>(null);
  const lastVideoElementRef = useRef<HTMLVideoElement | undefined>(undefined);
  const isConnectingRef = useRef<boolean>(false);
  const isLoopActiveRef = useRef<boolean>(false);
  const pendingCommandsRef = useRef<Array<Record<string, any>>>([]);
  const lastTelemetryReceivedAtRef = useRef<number>(0);
  const calibrationCompletionEmittedRef = useRef<boolean>(false);

  // Telemetry deduplication, state guard, and latency tracking refs
  const lastTelemetrySeqRef = useRef<bigint>(BigInt(-1));
  const lastSafetyEventSeqRef = useRef<bigint>(BigInt(-1));
  const activeStateRef = useRef<AlertState>("NORMAL");
  const lastRenderedStateRef = useRef<AlertState>("NORMAL");
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

  // Stale telemetry watchdog (triggers if connected and no telemetry arrives for > 3.5s after initial frame)
  useEffect(() => {
    if (connectionStatus !== "connected") {
      setIsStale(false);
      return;
    }
    const timer = setInterval(() => {
      if (
        lastTelemetryReceivedAtRef.current > 0 &&
        Date.now() - lastTelemetryReceivedAtRef.current > 3500
      ) {
        setIsStale(true);
      }
    }, 1000);
    return () => clearInterval(timer);
  }, [connectionStatus]);

  // Disconnect & cleanup LiveKit room and reset all cockpit telemetry state
  const disconnectRoom = useCallback(async () => {
    isConnectingRef.current = false;
    isLoopActiveRef.current = false;
    lastTelemetrySeqRef.current = BigInt(-1);
    lastTelemetryReceivedAtRef.current = 0;
    calibrationCompletionEmittedRef.current = false;
    activeStateRef.current = "NORMAL";
    lastRenderedStateRef.current = "NORMAL";
    latestRttRef.current = 0;
    rttSamplesRef.current = [];
    setIsStale(false);
    setTelemetry(null);
    setActiveSafetyEvent(null);
    setCalibration(null);
    setLatencyDiagnostics(null);
    setRoundtripLatencyMs(0);

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
          // Pass stopTracks=false so LiveKit does not kill the browser's MediaStreamTrack behind useCamera
          await currentRoom.disconnect(false);
        }
      } catch (_) {}
      const base = backendType === "cloud" ? cloudApiBase : localApiBase;
      fetch(`${base}/api/v1/livekit/stop-worker?room_name=${encodeURIComponent(stableSessionId)}`, {
        method: "POST",
        keepalive: true,
      }).catch(() => {});
    }

    setConnectionStatus("disconnected");
  }, [backendType, cloudApiBase, localApiBase, stableSessionId]);

  // Connect to LiveKit Cloud via signed token
  const connectLivekit = useCallback(
    async (videoElement?: HTMLVideoElement) => {
      if (videoElement) {
        lastVideoElementRef.current = videoElement;
      }
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

      lastTelemetrySeqRef.current = BigInt(-1);
      isConnectingRef.current = true;
      setConnectionStatus("connecting");

      try {
        const base = backendType === "cloud" ? cloudApiBase : localApiBase;
        const query = `room_name=${encodeURIComponent(stableSessionId)}&driver_id=${encodeURIComponent(driverId)}&experiment_mode=${encodeURIComponent(experimentMode)}&auto_start_worker=true`;
        const tokenUrl = `${base}/api/v1/livekit/token?${query}`;

        console.log(`[DriveGuard LiveKit] Requesting token from: ${tokenUrl}`);
        let tokenRes = await fetch(tokenUrl);
        if (!tokenRes.ok && backendType === "cloud") {
          const fallbackUrl = `${directCloudApiBase}/api/v1/livekit/token?${query}`;
          console.warn(`[DriveGuard LiveKit] Proxy returned ${tokenRes.status}, retrying direct cloud URL: ${fallbackUrl}`);
          tokenRes = await fetch(fallbackUrl);
        }
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

        room.on(RoomEvent.Reconnected, () => {
          setConnectionStatus("connected");
          try {
            const payload = new TextEncoder().encode(JSON.stringify({ action: "SYNC_STATE" }));
            room.localParticipant.publishData(payload, { reliable: true }).catch(() => {});
          } catch (_) {}
        });

        // Helper to update DataChannel RTT diagnostics and percentiles (invoked only from PING/PONG)
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

        // Helper to update server timing diagnostics from telemetry frames without recording an RTT sample
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
                : (effectiveQueue + effectiveInfer);

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

        // Listen for real-time telemetry from Python AI worker over WebRTC Data Channel
        room.on(RoomEvent.DataReceived, (payload: Uint8Array, participant?: RemoteParticipant) => {
          try {
            const rawText = new TextDecoder().decode(payload);
            const msg = JSON.parse(rawText);

            if (msg.type === "TELEMETRY" && msg.telemetry) {
              const rawSeq = msg.frame_seq ?? msg.seq ?? (msg.telemetry.frame_index ?? 0);
              let currentSeq: bigint;
              try {
                currentSeq = BigInt(String(rawSeq));
              } catch {
                currentSeq = BigInt(0);
              }
              // Stale packet rejection: discard packets with older or equal sequence numbers
              if (currentSeq <= lastTelemetrySeqRef.current) {
                return;
              }
              lastTelemetrySeqRef.current = currentSeq;
              lastTelemetryReceivedAtRef.current = Date.now();
              setIsStale(false);

              const tel = msg.telemetry as FrameTelemetry;
              activeStateRef.current = tel.state;

              const qLat = tel.timing?.queue_latency_ms ?? 0;
              const infLat = tel.timing?.infer_duration_ms ?? (tel.inference_latency_ms ?? 0);
              const sTotal = tel.timing?.server_total_ms ?? (qLat + infLat);
              latestTimingRef.current = { queueMs: qLat, inferMs: infLat, serverTotalMs: sTotal };

              // Reuse the last measured RTT when enriching telemetry.
              // Ensure no RTT is inferred from server processing time before a ping/pong match.
              const measuredRtt =
                latestRttRef.current > 0 ? latestRttRef.current : undefined;

              // Guard against normal-blink false positives (< 310ms while driver is in NORMAL state)
              const rawReasons = Array.isArray(tel.reason_codes) ? tel.reason_codes : [];
              const filteredReasons =
                tel.state === "NORMAL" &&
                (tel.eye_metrics?.eye_closure_duration_ms ?? 0) < 310 &&
                (tel.eye_metrics?.blink_duration_ms ?? 0) < 310
                  ? rawReasons.filter((r) => r !== "BLINK_DURATION_ABOVE_PERSONAL_BASELINE")
                  : rawReasons;

              const enriched: FrameTelemetry = {
                ...tel,
                reason_codes: filteredReasons,
                network_roundtrip_ms: measuredRtt,
              };

              // Immediately invoke callbacks for external synchronous listeners
              if (onTelemetryRef.current) {
                onTelemetryRef.current(enriched);
              }
              if (msg.calibration) {
                const calFrame = msg.calibration as CalibrationPhase;
                if (!calFrame.is_completed) {
                  calibrationCompletionEmittedRef.current = false;
                  if (onCalibrationRef.current) {
                    onCalibrationRef.current(calFrame);
                  }
                } else if (!calibrationCompletionEmittedRef.current) {
                  calibrationCompletionEmittedRef.current = true;
                  if (onCalibrationRef.current) {
                    onCalibrationRef.current(calFrame);
                  }
                }
              }

              // EXP-3A Event-Tiered Frontend Updates:
              // Tier 1: Safety state transitions (e.g. NORMAL -> WARNING/CRITICAL) bypass
              // the continuous telemetry throttle and dispatch immediately to React state.
              // Tier 2: Continuous needle/gauge telemetry throttles to ~30 FPS (33ms) for smooth rendering.
              const isSafetyTransition = tel.state !== lastRenderedStateRef.current;
              const now = performance.now();
              const shouldRender = isSafetyTransition || (now - lastReactUpdateRef.current >= 33);

              if (shouldRender) {
                lastReactUpdateRef.current = now;
                lastRenderedStateRef.current = tel.state;
                setTelemetry(enriched);
                updateServerTimings(qLat, infLat, sTotal);
                if (msg.calibration) {
                  setCalibration(msg.calibration as CalibrationPhase);
                }
              } else if (msg.calibration && (msg.calibration as CalibrationPhase).is_completed) {
                // Ensure calibration completion is immediately reflected in React state
                setCalibration(msg.calibration as CalibrationPhase);
              }
            } else if (msg.type === "SAFETY_EVENT") {
              const safetyEvent = msg as SafetyEvent;
              let currentEventSeq: bigint;
              try {
                currentEventSeq = BigInt(String(safetyEvent.event_seq ?? 0));
              } catch {
                currentEventSeq = BigInt(0);
              }
              console.log(
                `[DriveGuard LiveKit] Safety event received: seq=${currentEventSeq} ${safetyEvent.previous_state} -> ${safetyEvent.new_state} (ID: ${safetyEvent.event_id})`
              );

              // 1. Immediately acknowledge event back to Python worker over reliable DataChannel
              // (always ACK, even for duplicates or out-of-order packets, so server clears retry task)
              if (roomRef.current && roomRef.current.state === ConnectionState.Connected) {
                const ackPayload: EventAck = {
                  action: "EVENT_ACK",
                  event_id: safetyEvent.event_id,
                  event_seq: String(safetyEvent.event_seq),
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

              // 2. Out-of-order guard: prevent delayed older safety events from overwriting newer states
              if (currentEventSeq <= lastSafetyEventSeqRef.current) {
                console.warn(
                  `[DriveGuard LiveKit] Discarding delayed out-of-order safety event seq=${currentEventSeq} (latest applied seq=${lastSafetyEventSeqRef.current})`
                );
                return;
              }

              // 3. Idempotent deduplication (prevent re-triggering alert audio on retransmitted packets)
              if (!processedEventIdsRef.current.has(safetyEvent.event_id)) {
                processedEventIdsRef.current.add(safetyEvent.event_id);
                lastSafetyEventSeqRef.current = currentEventSeq;
                if (processedEventIdsRef.current.size > 200) {
                  const oldest = processedEventIdsRef.current.values().next().value;
                  if (oldest) processedEventIdsRef.current.delete(oldest);
                }

                activeStateRef.current = safetyEvent.new_state;
                lastRenderedStateRef.current = safetyEvent.new_state;
                setActiveSafetyEvent(safetyEvent);

                // EXP-3A: Synchronously update active telemetry state so visual HUD alerts update immediately
                setTelemetry((prev) => (prev ? { ...prev, state: safetyEvent.new_state } : prev));

                if (onSafetyEventRef.current) {
                  onSafetyEventRef.current(safetyEvent);
                }
              }
            } else if (msg.type === "STATE_SYNC") {
              console.log(
                `[DriveGuard LiveKit] State synchronization received from server. Current state: ${msg.current_state}`
              );
              if (msg.latest_safety_event) {
                const syncEvt = msg.latest_safety_event as SafetyEvent;
                try {
                  const syncSeq = BigInt(String(syncEvt.event_seq ?? 0));
                  if (syncSeq > lastSafetyEventSeqRef.current) {
                    lastSafetyEventSeqRef.current = syncSeq;
                  }
                } catch {
                  // Ignore parse error
                }
                setActiveSafetyEvent(syncEvt);
              }
              if (msg.current_state) {
                activeStateRef.current = msg.current_state;
                lastRenderedStateRef.current = msg.current_state;
                setTelemetry((prev) => (prev ? { ...prev, state: msg.current_state } : prev));
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
              msg.type === "CALIBRATION_COMPLETED" ||
              msg.type === "CALIBRATION_REJECTED"
            ) {
              if (msg.type === "CALIBRATION_STARTED") {
                calibrationCompletionEmittedRef.current = false;
              }
              if (msg.calibration) {
                const cal = msg.calibration as CalibrationPhase & { profile?: unknown };
                if (msg.profile) {
                  cal.profile = msg.profile;
                }
                if (msg.rejection_reason && !cal.rejection_reason) {
                  cal.rejection_reason = msg.rejection_reason;
                }
                setCalibration(cal);
                if (!cal.is_completed) {
                  calibrationCompletionEmittedRef.current = false;
                  if (onCalibrationRef.current) {
                    onCalibrationRef.current(cal);
                  }
                } else if (!calibrationCompletionEmittedRef.current) {
                  calibrationCompletionEmittedRef.current = true;
                  if (onCalibrationRef.current) {
                    onCalibrationRef.current(cal);
                  }
                }
              }
            } else if (msg.type === "CALIBRATION_CANCELLED") {
              calibrationCompletionEmittedRef.current = false;
              setCalibration(null);
            } else if (msg.type === "EXPERIMENT_SWITCHED" && msg.mode) {
              setExperimentMode(msg.mode as ExperimentMode);
            }
          } catch (err) {
            console.warn("[DriveGuard LiveKit] Error parsing data packet:", err);
          }
        });

        room.on(RoomEvent.Disconnected, () => {
          console.log("[DriveGuard LiveKit] Room disconnected.");
          lastTelemetrySeqRef.current = BigInt(-1);
          latestRttRef.current = 0;
          if (pingIntervalRef.current) {
            clearInterval(pingIntervalRef.current);
            pingIntervalRef.current = null;
          }
          setConnectionStatus("disconnected");
        });

        // Connect WebRTC to LiveKit Cloud
        console.log(`[DriveGuard LiveKit] Connecting to SFU: ${data.url}`);
        lastTelemetrySeqRef.current = BigInt(-1);
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

  // Unmount cleanup
  useEffect(() => {
    return () => {
      disconnectRoom();
    };
  }, [disconnectRoom]);

  return {
    isConnected: connectionStatus === "connected",
    connectionStatus,
    isStale,
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
