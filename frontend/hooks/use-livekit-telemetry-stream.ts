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
} from "@/types/telemetry";

export interface UseLivekitTelemetryStreamOptions {
  sessionId?: string;
  driverId?: string;
  defaultBackend?: "cloud" | "local";
  onTelemetry?: (telemetry: FrameTelemetry) => void;
  onCalibration?: (calibration: CalibrationPhase) => void;
  onError?: (err: Error) => void;
}

export function useLivekitTelemetryStream(options: UseLivekitTelemetryStreamOptions = {}) {
  const {
    sessionId: initialSessionId,
    driverId = "driver_default",
    defaultBackend,
    onTelemetry,
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

  // Synchronized callback refs
  const onTelemetryRef = useRef(onTelemetry);
  useEffect(() => {
    onTelemetryRef.current = onTelemetry;
  }, [onTelemetry]);

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

        // Listen for real-time telemetry from Python AI worker over WebRTC Data Channel
        room.on(RoomEvent.DataReceived, (payload: Uint8Array, participant?: RemoteParticipant) => {
          try {
            const rawText = new TextDecoder().decode(payload);
            const msg = JSON.parse(rawText);

            if (msg.type === "TELEMETRY" && msg.telemetry) {
              const tel = msg.telemetry as FrameTelemetry;
              const rtt = msg.network_roundtrip_ms ?? tel.inference_latency_ms;
              setRoundtripLatencyMs(Math.round(rtt));

              const enriched: FrameTelemetry = {
                ...tel,
                network_roundtrip_ms: rtt,
              };

              setTelemetry(enriched);
              if (onTelemetryRef.current) {
                onTelemetryRef.current(enriched);
              }

              if (msg.calibration) {
                const cal = msg.calibration as CalibrationPhase;
                setCalibration(cal);
                if (onCalibrationRef.current) {
                  onCalibrationRef.current(cal);
                }
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
          setConnectionStatus("disconnected");
        });

        // Connect WebRTC to LiveKit Cloud
        console.log(`[DriveGuard LiveKit] Connecting to SFU: ${data.url}`);
        await room.connect(data.url, data.token);
        console.log(`[DriveGuard LiveKit] Successfully connected to room '${data.room_name}'!`);
        setConnectionStatus("connected");

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
            console.log(`[DriveGuard LiveKit] Publishing camera track (${videoTrack.label})...`);
            const pub = await room.localParticipant.publishTrack(videoTrack, {
              name: "driver-camera",
              simulcast: false,
            });
            publishedTrackRef.current = pub.track as LocalVideoTrack;
            console.log("[DriveGuard LiveKit] Camera track published successfully!");
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
            roomRef.current.localParticipant
              .publishTrack(videoTrack, {
                name: "driver-camera",
                simulcast: false,
              })
              .then((pub) => {
                publishedTrackRef.current = pub.track as LocalVideoTrack;
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
    cancelCalibration,
    skipCalibrationStep,
    saveProfile,
    reconnect: connectLivekit,
  };
}
