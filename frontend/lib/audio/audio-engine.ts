/**
 * DriveGuard Web Audio Alert Engine
 * =================================
 * Native browser audio synthesizer for real-time driver drowsiness safety alerts.
 * Uses Web Audio API (OscillatorNode, GainNode, BiquadFilterNode) to generate
 * mathematically synthesized audio alerts without external audio assets.
 *
 * Implements:
 * 1. WARNING: Single 440 Hz sine chime (180ms), 8.0s cooldown.
 * 2. DROWSY: Dual harmonic 587 Hz (sine) & 740 Hz (triangle) double beep (120ms on, 80ms off, 120ms on), 5.0s cooldown.
 * 3. CRITICAL: Bandpass-filtered 880 Hz -> 1046 Hz sweep triple pulse (80ms on, 40ms off x 3),
 *    repeated every 1.5s until recovery or stop. Preempts lower priority alerts immediately.
 * 4. RECOVERY: Gentle ascending 523 Hz -> 659 Hz two-tone chime (~150ms), played once on transition to NORMAL.
 * 5. Full lifecycle management: mute, master volume, autoplay unlock, and node cleanup.
 */

export type AlertTier = "WARNING" | "DROWSY" | "CRITICAL" | "RECOVERY";

export type AudioEngineStatus =
  | "uninitialized"
  | "suspended"
  | "running"
  | "muted"
  | "unsupported"
  | "error";

export interface AudioEngineConfig {
  masterVolume?: number; // 0.0 to 1.0 (default 0.75)
  muted?: boolean;
  warningCooldownMs?: number; // default 8000
  drowsyCooldownMs?: number; // default 5000
  criticalRepeatIntervalMs?: number; // default 1500
}

interface TonePattern {
  tier: AlertTier;
  durationMs: number;
}

export class DriveGuardAudioEngine {
  private ctx: AudioContext | null = null;
  private masterGain: GainNode | null = null;

  // Configuration
  private _masterVolume: number = 0.75;
  private _isMuted: boolean = false;
  private readonly warningCooldownMs: number;
  private readonly drowsyCooldownMs: number;
  private readonly criticalRepeatIntervalMs: number;

  // Active Alert & Priority State
  private currentRiskState: string = "NORMAL";
  private currentTier: AlertTier | null = null;
  private criticalTimerId: ReturnType<typeof setTimeout> | null = null;
  private isCriticalLoopActive: boolean = false;

  // Cooldown & Deduplication Tracking
  private readonly lastAlertTimestamps: Map<AlertTier, number> = new Map();
  private readonly handledEventIds: Set<string> = new Set();
  private readonly activeNodes: Set<AudioNode> = new Set();
  private readonly cleanupTimerIds: Set<ReturnType<typeof setTimeout>> = new Set();
  private lastStateTimestampMs: number | null = null;

  // Status & Telemetry
  private status: AudioEngineStatus = "uninitialized";
  private isUnlocked: boolean = false;

  constructor(config?: AudioEngineConfig) {
    this._masterVolume = Math.max(0.0, Math.min(1.0, config?.masterVolume ?? 0.75));
    this._isMuted = Boolean(config?.muted);
    this.warningCooldownMs = config?.warningCooldownMs ?? 8000;
    this.drowsyCooldownMs = config?.drowsyCooldownMs ?? 5000;
    this.criticalRepeatIntervalMs = config?.criticalRepeatIntervalMs ?? 1500;
  }

  // ---------------------------------------------------------------------------
  // Lifecycle & Autoplay Unlock
  // ---------------------------------------------------------------------------

  /**
   * Initializes the AudioContext if not already created.
   * Safe to call multiple times. Handles browser support gracefully.
   */
  public initialize(): boolean {
    if (this.ctx && this.ctx.state !== "closed") {
      return true;
    }

    try {
      const AudioCtxClass =
        typeof window !== "undefined"
          ? window.AudioContext ||
            (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext
          : null;

      if (!AudioCtxClass) {
        this.status = "unsupported";
        return false;
      }

      this.ctx = new AudioCtxClass();
      this.masterGain = this.ctx.createGain();
      this.updateMasterGainValue();
      this.masterGain.connect(this.ctx.destination);

      if (this.ctx.state === "running") {
        this.isUnlocked = true;
        this.status = this._isMuted ? "muted" : "running";
      } else {
        this.status = "suspended";
      }

      return true;
    } catch (err) {
      console.warn("[DriveGuardAudioEngine] Failed to initialize AudioContext:", err);
      this.status = "error";
      return false;
    }
  }

  /**
   * Attempt to unlock or resume a suspended AudioContext after a user gesture.
   */
  public async unlock(): Promise<boolean> {
    if (!this.initialize()) {
      return false;
    }

    if (!this.ctx) return false;

    if (this.ctx.state === "suspended") {
      try {
        await this.ctx.resume();
        this.isUnlocked = true;
        this.status = this._isMuted ? "muted" : "running";
        return true;
      } catch (err) {
        console.warn("[DriveGuardAudioEngine] AudioContext resume failed:", err);
        return false;
      }
    } else if (this.ctx.state === "running") {
      this.isUnlocked = true;
      this.status = this._isMuted ? "muted" : "running";
      return true;
    }

    return false;
  }

  /**
   * Suspend audio context to conserve power when idle.
   */
  public async suspend(): Promise<void> {
    this.stopCurrentAlert();
    if (this.ctx && this.ctx.state === "running") {
      try {
        await this.ctx.suspend();
        this.status = "suspended";
      } catch (_) {}
    }
  }

  /**
   * Complete teardown of all audio resources, timers, and contexts.
   */
  public dispose(): void {
    this.stopCurrentAlert();
    this.stopActiveNodes();
    this.clearCleanupTimers();

    if (this.masterGain) {
      try {
        this.masterGain.disconnect();
      } catch (_) {}
      this.masterGain = null;
    }

    if (this.ctx && this.ctx.state !== "closed") {
      try {
        this.ctx.close();
      } catch (_) {}
      this.ctx = null;
    }

    this.handledEventIds.clear();
    this.lastAlertTimestamps.clear();
    this.lastStateTimestampMs = null;
    this.currentRiskState = "NORMAL";
    this.status = "uninitialized";
    this.isUnlocked = false;
  }

  // ---------------------------------------------------------------------------
  // Volume & Mute Controls
  // ---------------------------------------------------------------------------

  public setVolume(volume: number): void {
    this._masterVolume = Math.max(0.0, Math.min(1.0, volume));
    this.updateMasterGainValue();
  }

  public getVolume(): number {
    return this._masterVolume;
  }

  public setMuted(muted: boolean): void {
    this._isMuted = muted;
    this.updateMasterGainValue();
    if (this.status !== "unsupported" && this.status !== "error") {
      this.status = muted ? "muted" : this.ctx?.state === "running" ? "running" : "suspended";
    }
    if (muted) {
      this.stopCurrentAlert();
    } else if (this.currentRiskState === "CRITICAL" && !this.isCriticalLoopActive) {
      // Unmuted while CRITICAL remains active: resume repeating alarm immediately
      this.isCriticalLoopActive = true;
      this.currentTier = "CRITICAL";
      this.playCriticalSequence();
    }
  }

  public isMuted(): boolean {
    return this._isMuted;
  }

  public getStatus(): AudioEngineStatus {
    if (this.status === "unsupported" || this.status === "error") return this.status;
    if (this._isMuted) return "muted";
    if (this.ctx?.state === "running") return "running";
    if (this.ctx?.state === "suspended") return "suspended";
    return this.status;
  }

  public isReady(): boolean {
    return Boolean(this.ctx && this.ctx.state === "running" && !this._isMuted);
  }

  private updateMasterGainValue(): void {
    if (!this.masterGain || !this.ctx) return;
    const targetGain = this._isMuted ? 0.0 : this._masterVolume;
    try {
      this.masterGain.gain.setValueAtTime(targetGain, this.ctx.currentTime);
    } catch (_) {}
  }

  // ---------------------------------------------------------------------------
  // State Transitions & Preemption
  // ---------------------------------------------------------------------------

  /**
   * Sets the authoritative backend safety state, managing priority transitions,
   * alert preemption, deduplication, and recovery chimes.
   */
  public setRiskState(
    newState: string,
    eventId?: string,
    timestampMs?: number
  ): boolean {
    // 0. Stale frame rejection: if timestamp is provided and older than our latest authoritative state transition, reject
    if (typeof timestampMs === "number" && !isNaN(timestampMs)) {
      if (this.lastStateTimestampMs !== null && timestampMs < this.lastStateTimestampMs) {
        return false;
      }
      this.lastStateTimestampMs = timestampMs;
    }

    // 1. Deduplicate by event ID if supplied
    if (eventId) {
      if (this.handledEventIds.has(eventId)) {
        return false;
      }
      this.handledEventIds.add(eventId);
      if (this.handledEventIds.size > 200) {
        const [first] = this.handledEventIds;
        this.handledEventIds.delete(first);
      }
    }

    const prevState = this.currentRiskState;
    this.currentRiskState = newState;

    // 2. CRITICAL State Handling (Highest Priority)
    if (newState === "CRITICAL") {
      if (!this.isCriticalLoopActive) {
        // Genuine transition into CRITICAL: immediately preempt lower sounds & start loop
        this.stopCurrentAlert(); // Preempts WARNING or DROWSY
        this.isCriticalLoopActive = true;
        this.currentTier = "CRITICAL";
        this.playCriticalSequence();
        return true;
      }
      // If already in critical loop, do not restart pulses on duplicate telemetry updates
      return false;
    }

    // If leaving CRITICAL, stop the repeating critical alarm immediately
    if (prevState === "CRITICAL" && newState !== "CRITICAL") {
      this.stopCriticalLoop();
    }

    // 3. Recovery Handling (Transition from any alert state back to NORMAL)
    if (newState === "NORMAL") {
      if (prevState === "CRITICAL" || prevState === "DROWSY" || prevState === "WARNING") {
        this.playAlert("RECOVERY");
        return true;
      }
      return false;
    }

    // 4. DROWSY State Handling
    if (newState === "DROWSY") {
      return this.playAlert("DROWSY");
    }

    // 5. WARNING State Handling
    if (newState === "WARNING") {
      return this.playAlert("WARNING");
    }

    return false;
  }

  /**
   * Plays a synthesized alert pattern if permitted by priority, mute state, and cooldowns.
   */
  public playAlert(tier: AlertTier): boolean {
    if (this._isMuted) return false;
    if (!this.initialize() || !this.ctx || !this.masterGain) return false;

    // CRITICAL is active: lower-priority alerts cannot interrupt it
    if (this.isCriticalLoopActive && tier !== "CRITICAL") {
      return false;
    }

    const now = Date.now();

    // Check cooldowns
    if (tier === "WARNING") {
      const last = this.lastAlertTimestamps.get("WARNING") ?? 0;
      if (now - last < this.warningCooldownMs) {
        return false;
      }
      this.lastAlertTimestamps.set("WARNING", now);
    } else if (tier === "DROWSY") {
      const last = this.lastAlertTimestamps.get("DROWSY") ?? 0;
      if (now - last < this.drowsyCooldownMs) {
        return false;
      }
      this.lastAlertTimestamps.set("DROWSY", now);
    }

    this.currentTier = tier;

    // Dispatch synthesis
    switch (tier) {
      case "WARNING":
        this.synthesizeWarning();
        break;
      case "DROWSY":
        this.synthesizeDrowsy();
        break;
      case "CRITICAL":
        this.synthesizeCriticalPulse();
        break;
      case "RECOVERY":
        this.synthesizeRecovery();
        break;
    }

    return true;
  }

  /**
   * Stops any currently sounding tones and cancels critical repetition.
   */
  public stopCurrentAlert(): void {
    this.stopCriticalLoop();
    this.stopActiveNodes();
    this.currentTier = null;
  }

  private stopCriticalLoop(): void {
    this.isCriticalLoopActive = false;
    if (this.criticalTimerId !== null) {
      clearTimeout(this.criticalTimerId);
      this.criticalTimerId = null;
    }
  }

  private stopActiveNodes(): void {
    this.clearCleanupTimers();
    for (const node of this.activeNodes) {
      try {
        if ("stop" in node && typeof (node as { stop: () => void }).stop === "function") {
          (node as { stop: () => void }).stop();
        }
        node.disconnect();
      } catch (_) {}
    }
    this.activeNodes.clear();
  }

  // ---------------------------------------------------------------------------
  // Sound Synthesis Implementations
  // ---------------------------------------------------------------------------

  /**
   * WARNING: Soft 440 Hz (A4) Sine chime (180 ms).
   */
  private synthesizeWarning(): TonePattern {
    if (!this.ctx || !this.masterGain) return { tier: "WARNING", durationMs: 180 };

    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sine";
    osc.frequency.setValueAtTime(440, t0);

    // Soft attack (20ms), sustain (80ms), gentle decay (80ms)
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.linearRampToValueAtTime(0.18, t0 + 0.02);
    gain.gain.setValueAtTime(0.18, t0 + 0.10);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.18);

    osc.connect(gain);
    gain.connect(this.masterGain);

    this.registerActiveNode(osc);
    this.registerActiveNode(gain);

    osc.start(t0);
    osc.stop(t0 + 0.185);

    this.scheduleNodeCleanup(osc, gain, 200);
    return { tier: "WARNING", durationMs: 180 };
  }

  /**
   * DROWSY: Dual harmonic 587 Hz (sine) + 740 Hz (triangle) double beep.
   * Timing: 120ms ON, 80ms OFF, 120ms ON. (Total 320ms)
   */
  private synthesizeDrowsy(): TonePattern {
    if (!this.ctx || !this.masterGain) return { tier: "DROWSY", durationMs: 320 };

    const t0 = this.ctx.currentTime;

    const osc1 = this.ctx.createOscillator();
    const osc2 = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc1.type = "sine";
    osc1.frequency.setValueAtTime(587, t0); // D5

    osc2.type = "triangle";
    osc2.frequency.setValueAtTime(740, t0); // F#5

    // Beep 1: 0.00 -> 0.12s
    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.linearRampToValueAtTime(0.20, t0 + 0.02);
    gain.gain.setValueAtTime(0.20, t0 + 0.10);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.12);

    // Gap: 0.12 -> 0.20s
    gain.gain.setValueAtTime(0.0001, t0 + 0.20);

    // Beep 2: 0.20 -> 0.32s
    gain.gain.linearRampToValueAtTime(0.20, t0 + 0.22);
    gain.gain.setValueAtTime(0.20, t0 + 0.30);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.32);

    osc1.connect(gain);
    osc2.connect(gain);
    gain.connect(this.masterGain);

    this.registerActiveNode(osc1);
    this.registerActiveNode(osc2);
    this.registerActiveNode(gain);

    osc1.start(t0);
    osc2.start(t0);
    osc1.stop(t0 + 0.33);
    osc2.stop(t0 + 0.33);

    this.scheduleNodeCleanup(osc1, gain, 350);
    this.scheduleNodeCleanup(osc2, null, 350);
    return { tier: "DROWSY", durationMs: 320 };
  }

  /**
   * CRITICAL Sequence: Repeats every 1.5s while isCriticalLoopActive remains true.
   */
  private playCriticalSequence(): void {
    if (!this.isCriticalLoopActive) return;

    this.synthesizeCriticalPulse();

    // Schedule next repetition after criticalRepeatIntervalMs (1500ms)
    this.criticalTimerId = setTimeout(() => {
      if (this.isCriticalLoopActive && this.currentRiskState === "CRITICAL") {
        this.playCriticalSequence();
      } else {
        this.isCriticalLoopActive = false;
      }
    }, this.criticalRepeatIntervalMs);
  }

  /**
   * CRITICAL: Staccato triple pulse (80ms ON, 40ms OFF x 3).
   * Filtered sawtooth/square sweep from 880 Hz to 1046 Hz.
   */
  private synthesizeCriticalPulse(): TonePattern {
    if (!this.ctx || !this.masterGain) return { tier: "CRITICAL", durationMs: 320 };

    const t0 = this.ctx.currentTime;

    const oscSaw = this.ctx.createOscillator();
    const oscSquare = this.ctx.createOscillator();
    const filter = this.ctx.createBiquadFilter();
    const gain = this.ctx.createGain();

    oscSaw.type = "sawtooth";
    oscSquare.type = "square";

    filter.type = "lowpass";
    filter.frequency.setValueAtTime(2200, t0); // Bandlimit to remove harsh digital sizzle
    filter.Q.setValueAtTime(1.5, t0);

    // Apply frequency sweeps on each of the 3 pulses
    // Pulse 1: 0.00 -> 0.08s
    oscSaw.frequency.setValueAtTime(880, t0);
    oscSaw.frequency.exponentialRampToValueAtTime(1046, t0 + 0.08);
    oscSquare.frequency.setValueAtTime(880, t0);
    oscSquare.frequency.exponentialRampToValueAtTime(1046, t0 + 0.08);

    // Pulse 2: 0.12 -> 0.20s
    oscSaw.frequency.setValueAtTime(880, t0 + 0.12);
    oscSaw.frequency.exponentialRampToValueAtTime(1046, t0 + 0.20);
    oscSquare.frequency.setValueAtTime(880, t0 + 0.12);
    oscSquare.frequency.exponentialRampToValueAtTime(1046, t0 + 0.20);

    // Pulse 3: 0.24 -> 0.32s
    oscSaw.frequency.setValueAtTime(880, t0 + 0.24);
    oscSaw.frequency.exponentialRampToValueAtTime(1046, t0 + 0.32);
    oscSquare.frequency.setValueAtTime(880, t0 + 0.24);
    oscSquare.frequency.exponentialRampToValueAtTime(1046, t0 + 0.32);

    // Gain envelope for 3 staccato pulses
    gain.gain.setValueAtTime(0.0001, t0);

    // Pulse 1
    gain.gain.linearRampToValueAtTime(0.24, t0 + 0.015);
    gain.gain.setValueAtTime(0.24, t0 + 0.065);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.08);

    // Gap 1: 0.08 -> 0.12s
    gain.gain.setValueAtTime(0.0001, t0 + 0.12);

    // Pulse 2
    gain.gain.linearRampToValueAtTime(0.24, t0 + 0.135);
    gain.gain.setValueAtTime(0.24, t0 + 0.185);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.20);

    // Gap 2: 0.20 -> 0.24s
    gain.gain.setValueAtTime(0.0001, t0 + 0.24);

    // Pulse 3
    gain.gain.linearRampToValueAtTime(0.24, t0 + 0.255);
    gain.gain.setValueAtTime(0.24, t0 + 0.305);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.32);

    // Connect graph: Oscillators -> Filter -> Gain -> Master
    oscSaw.connect(filter);
    oscSquare.connect(filter);
    filter.connect(gain);
    gain.connect(this.masterGain);

    this.registerActiveNode(oscSaw);
    this.registerActiveNode(oscSquare);
    this.registerActiveNode(filter);
    this.registerActiveNode(gain);

    oscSaw.start(t0);
    oscSquare.start(t0);
    oscSaw.stop(t0 + 0.33);
    oscSquare.stop(t0 + 0.33);

    this.scheduleNodeCleanup(oscSaw, gain, 350);
    this.scheduleNodeCleanup(oscSquare, filter, 350);
    return { tier: "CRITICAL", durationMs: 320 };
  }

  /**
   * RECOVERY: Ascending gentle two-tone chime 523 Hz (C5) -> 659 Hz (E5) (~150ms).
   */
  private synthesizeRecovery(): TonePattern {
    if (!this.ctx || !this.masterGain) return { tier: "RECOVERY", durationMs: 150 };

    const t0 = this.ctx.currentTime;
    const osc = this.ctx.createOscillator();
    const gain = this.ctx.createGain();

    osc.type = "sine";

    // Tone 1: 523 Hz (0 to 75ms)
    osc.frequency.setValueAtTime(523, t0);
    // Tone 2: 659 Hz (75ms to 150ms)
    osc.frequency.setValueAtTime(659, t0 + 0.075);

    gain.gain.setValueAtTime(0.0001, t0);
    gain.gain.linearRampToValueAtTime(0.16, t0 + 0.02);
    gain.gain.setValueAtTime(0.16, t0 + 0.06);
    gain.gain.linearRampToValueAtTime(0.14, t0 + 0.075);
    gain.gain.linearRampToValueAtTime(0.18, t0 + 0.095);
    gain.gain.setValueAtTime(0.18, t0 + 0.125);
    gain.gain.linearRampToValueAtTime(0.0001, t0 + 0.15);

    osc.connect(gain);
    gain.connect(this.masterGain);

    this.registerActiveNode(osc);
    this.registerActiveNode(gain);

    osc.start(t0);
    osc.stop(t0 + 0.155);

    this.scheduleNodeCleanup(osc, gain, 180);
    return { tier: "RECOVERY", durationMs: 150 };
  }

  public getOutputLatencyEstimateMs(): number {
    if (!this.ctx) return 0;
    const base = typeof this.ctx.baseLatency === "number" ? this.ctx.baseLatency : 0;
    const output = typeof (this.ctx as any).outputLatency === "number" ? (this.ctx as any).outputLatency : 0;
    return Math.round((base + output) * 1000 * 10) / 10;
  }

  // ---------------------------------------------------------------------------
  // Resource Tracking & Cleanup Helpers
  // ---------------------------------------------------------------------------

  private registerActiveNode(node: AudioNode): void {
    this.activeNodes.add(node);
  }

  private scheduleNodeCleanup(
    osc: AudioNode,
    gain: AudioNode | null,
    delayMs: number
  ): void {
    const timerId = setTimeout(() => {
      this.cleanupTimerIds.delete(timerId);
      this.activeNodes.delete(osc);
      try {
        osc.disconnect();
      } catch (_) {}

      if (gain) {
        this.activeNodes.delete(gain);
        try {
          gain.disconnect();
        } catch (_) {}
      }
    }, delayMs);
    this.cleanupTimerIds.add(timerId);
  }

  private clearCleanupTimers(): void {
    for (const tid of this.cleanupTimerIds) {
      clearTimeout(tid);
    }
    this.cleanupTimerIds.clear();
  }
}
