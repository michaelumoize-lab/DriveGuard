"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import {
  DriveGuardAudioEngine,
  AudioEngineConfig,
} from "@/lib/audio/audio-engine";

export type AudioDisplayStatus = "ready" | "muted" | "blocked" | "unavailable";

export interface UseDriveGuardAudioOptions extends AudioEngineConfig {
  autoUnlockOnGesture?: boolean;
}

export interface UseDriveGuardAudioReturn {
  isAudioUnlocked: boolean;
  isMuted: boolean;
  volume: number;
  audioStatus: AudioDisplayStatus;
  toggleMute: () => void;
  setVolume: (volume: number) => void;
  unlockAudio: () => Promise<boolean>;
  getAudioLatencyEstimateMs: () => number | null;
  engine: DriveGuardAudioEngine | null;
}

// Module-level shared singleton instance & subscriber reference counter
let sharedAudioEngine: DriveGuardAudioEngine | null = null;
let activeHookSubscribers = 0;

export function useDriveGuardAudio(
  options?: UseDriveGuardAudioOptions
): UseDriveGuardAudioReturn {
  const engineRef = useRef<DriveGuardAudioEngine | null>(null);

  // Synchronized React state for UI rendering
  const [isAudioUnlocked, setIsAudioUnlocked] = useState<boolean>(false);
  const [isMuted, setIsMuted] = useState<boolean>(Boolean(options?.muted));
  const [volume, setVolumeState] = useState<number>(options?.masterVolume ?? 0.75);
  const [audioStatus, setAudioStatus] = useState<AudioDisplayStatus>("blocked");

  // Shared singleton accessor ensuring single AudioContext across components
  const getEngine = useCallback((): DriveGuardAudioEngine => {
    if (!sharedAudioEngine) {
      sharedAudioEngine = new DriveGuardAudioEngine({
        masterVolume: options?.masterVolume ?? 0.75,
        muted: options?.muted ?? false,
        warningCooldownMs: options?.warningCooldownMs,
        drowsyCooldownMs: options?.drowsyCooldownMs,
        criticalRepeatIntervalMs: options?.criticalRepeatIntervalMs,
      });
    }
    engineRef.current = sharedAudioEngine;
    return sharedAudioEngine;
  }, [
    options?.masterVolume,
    options?.muted,
    options?.warningCooldownMs,
    options?.drowsyCooldownMs,
    options?.criticalRepeatIntervalMs,
  ]);

  // Eager ref population on render
  if (!engineRef.current) {
    getEngine();
  }

  const updateDisplayStatus = useCallback(() => {
    const engine = engineRef.current;
    if (!engine) {
      setAudioStatus("blocked");
      return;
    }

    const st = engine.getStatus();
    if (st === "unsupported" || st === "error") {
      setAudioStatus("unavailable");
      setIsAudioUnlocked(false);
    } else if (engine.isMuted()) {
      setAudioStatus("muted");
      setIsMuted(true);
    } else if (st === "running") {
      setAudioStatus("ready");
      setIsAudioUnlocked(true);
      setIsMuted(false);
    } else {
      // "suspended" or "uninitialized" (waiting for user interaction)
      setAudioStatus("blocked");
      setIsAudioUnlocked(false);
    }
  }, []);

  // Explicit user unlock action (e.g. from UI button or camera start)
  const unlockAudio = useCallback(async (): Promise<boolean> => {
    const engine = getEngine();
    const success = await engine.unlock();
    if (success) {
      setIsAudioUnlocked(true);
    }
    updateDisplayStatus();
    return success;
  }, [getEngine, updateDisplayStatus]);

  // Volume & Mute actions
  const setVolume = useCallback(
    (newVolume: number) => {
      const clamped = Math.max(0.0, Math.min(1.0, newVolume));
      const engine = getEngine();
      engine.setVolume(clamped);
      setVolumeState(clamped);
      updateDisplayStatus();
    },
    [getEngine, updateDisplayStatus]
  );

  const toggleMute = useCallback(() => {
    const engine = getEngine();
    const newMuted = !engine.isMuted();
    engine.setMuted(newMuted);
    setIsMuted(newMuted);
    updateDisplayStatus();
  }, [getEngine, updateDisplayStatus]);

  // Auto-unlock on first user interaction gesture
  useEffect(() => {
    const engine = getEngine();
    engine.initialize();
    updateDisplayStatus();

    if (options?.autoUnlockOnGesture === false) {
      return;
    }

    const handleUserGesture = () => {
      engine.unlock().then((unlocked) => {
        if (unlocked) {
          setIsAudioUnlocked(true);
          updateDisplayStatus();
        }
      });
    };

    const events = ["click", "touchstart", "keydown"];
    events.forEach((ev) => {
      window.addEventListener(ev, handleUserGesture, { once: true, passive: true });
    });

    return () => {
      events.forEach((ev) => {
        window.removeEventListener(ev, handleUserGesture);
      });
    };
  }, [getEngine, options?.autoUnlockOnGesture, updateDisplayStatus]);

  // Teardown and resource disposal on unmount with subscriber reference counting
  useEffect(() => {
    activeHookSubscribers++;
    return () => {
      activeHookSubscribers = Math.max(0, activeHookSubscribers - 1);
      if (activeHookSubscribers === 0 && sharedAudioEngine) {
        sharedAudioEngine.dispose();
        sharedAudioEngine = null;
      }
      engineRef.current = null;
    };
  }, []);

  const getAudioLatencyEstimateMs = useCallback((): number | null => {
    return engineRef.current?.getOutputLatencyEstimateMs() ?? null;
  }, []);

  return {
    isAudioUnlocked,
    isMuted,
    volume,
    audioStatus,
    toggleMute,
    setVolume,
    unlockAudio,
    getAudioLatencyEstimateMs,
    engine: engineRef.current,
  };
}
