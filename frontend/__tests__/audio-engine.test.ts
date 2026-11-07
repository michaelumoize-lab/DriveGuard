/**
 * Unit Test Suite for DriveGuardAudioEngine (Phase 6)
 * ==================================================
 * Tests the Web Audio API synthesized alert engine:
 * 1. Lifecycle, autoplay unlock & status reporting
 * 2. Volume control & mute behavior
 * 3. 4-tier alert synthesis (WARNING, DROWSY, CRITICAL, RECOVERY)
 * 4. Priority preemption (CRITICAL preempts WARNING/DROWSY)
 * 5. 1.5s repeating loop for CRITICAL alerts until recovery
 * 6. Recovery chime playback (once on return to NORMAL)
 * 7. Deduplication by state and event_id
 * 8. Cooldown enforcement (8s for WARNING, 5s for DROWSY)
 * 9. Resource cleanup and disposal
 */

import { DriveGuardAudioEngine, AlertTier } from "../lib/audio/audio-engine";

// ---------------------------------------------------------------------------
// Mock Web Audio API
// ---------------------------------------------------------------------------

class MockAudioParam {
  value: number;
  calls: Array<{ method: string; args: any[] }> = [];

  constructor(defaultValue: number = 0) {
    this.value = defaultValue;
  }

  setValueAtTime(value: number, startTime: number) {
    this.value = value;
    this.calls.push({ method: "setValueAtTime", args: [value, startTime] });
  }

  linearRampToValueAtTime(value: number, endTime: number) {
    this.value = value;
    this.calls.push({ method: "linearRampToValueAtTime", args: [value, endTime] });
  }

  exponentialRampToValueAtTime(value: number, endTime: number) {
    this.value = value;
    this.calls.push({ method: "exponentialRampToValueAtTime", args: [value, endTime] });
  }
}

class MockAudioNode {
  connectedTo: any[] = [];
  disconnected: boolean = false;

  connect(dest: any) {
    this.connectedTo.push(dest);
    return dest;
  }

  disconnect() {
    this.disconnected = true;
    this.connectedTo = [];
  }
}

class MockGainNode extends MockAudioNode {
  gain = new MockAudioParam(1.0);
}

class MockOscillatorNode extends MockAudioNode {
  type: OscillatorType = "sine";
  frequency = new MockAudioParam(440);
  started: boolean = false;
  stopped: boolean = false;
  startTime: number = 0;
  stopTime: number = 0;
  onended: (() => void) | null = null;

  start(time?: number) {
    this.started = true;
    this.startTime = time ?? 0;
  }

  stop(time?: number) {
    this.stopped = true;
    this.stopTime = time ?? 0;
  }
}

class MockBiquadFilterNode extends MockAudioNode {
  type: BiquadFilterType = "bandpass";
  frequency = new MockAudioParam(1000);
  Q = new MockAudioParam(1);
}

class MockAudioContext {
  state: AudioContextState = "suspended";
  currentTime: number = 0.0;
  destination = new MockAudioNode();
  createdOscillators: MockOscillatorNode[] = [];
  createdGains: MockGainNode[] = [];
  createdFilters: MockBiquadFilterNode[] = [];

  createGain(): MockGainNode {
    const gain = new MockGainNode();
    this.createdGains.push(gain);
    return gain;
  }

  createOscillator(): MockOscillatorNode {
    const osc = new MockOscillatorNode();
    this.createdOscillators.push(osc);
    return osc;
  }

  createBiquadFilter(): MockBiquadFilterNode {
    const filter = new MockBiquadFilterNode();
    this.createdFilters.push(filter);
    return filter;
  }

  async resume(): Promise<void> {
    this.state = "running";
  }

  async suspend(): Promise<void> {
    this.state = "suspended";
  }

  async close(): Promise<void> {
    this.state = "closed";
  }
}

// Global window mock setup
let mockCtxInstance: MockAudioContext | null = null;
function setupMockWindow() {
  mockCtxInstance = null;
  (global as any).window = {
    AudioContext: function () {
      mockCtxInstance = new MockAudioContext();
      return mockCtxInstance;
    },
  };
}

// ---------------------------------------------------------------------------
// Test Runner Harness
// ---------------------------------------------------------------------------

let passed = 0;
let total = 0;

function it(desc: string, fn: () => void | Promise<void>) {
  total++;
  return Promise.resolve()
    .then(fn)
    .then(() => {
      passed++;
      console.log(`  [PASS] ${desc}`);
    })
    .catch((err) => {
      console.error(`  [FAIL] ${desc}`);
      console.error(err);
      process.exitCode = 1;
    });
}

function expect(val: any) {
  return {
    toBe(expected: any) {
      if (val !== expected) throw new Error(`Expected ${val} to be ${expected}`);
    },
    toEqual(expected: any) {
      if (JSON.stringify(val) !== JSON.stringify(expected)) {
        throw new Error(`Expected ${JSON.stringify(val)} to equal ${JSON.stringify(expected)}`);
      }
    },
    toBeGreaterThan(expected: number) {
      if (val <= expected) throw new Error(`Expected ${val} > ${expected}`);
    },
    toBeTruthy() {
      if (!val) throw new Error(`Expected ${val} to be truthy`);
    },
    toBeFalsy() {
      if (val) throw new Error(`Expected ${val} to be falsy`);
    },
  };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

async function runTests() {
  console.log("======================================================================");
  console.log("   [DriveGuard] Web Audio API Synthesizer Subsystem Unit Tests");
  console.log("======================================================================\n");

  setupMockWindow();

  // 1. Initialization and Lifecycle
  await it("1. AudioEngine initializes in suspended state prior to gesture unlock", () => {
    const engine = new DriveGuardAudioEngine();
    expect(engine.getStatus()).toBe("uninitialized");

    const ok = engine.initialize();
    expect(ok).toBeTruthy();
    expect(engine.getStatus()).toBe("suspended");
    expect(engine.isReady()).toBeFalsy();
    engine.dispose();
  });

  await it("2. Autoplay unlock transitions AudioContext from suspended to running", async () => {
    const engine = new DriveGuardAudioEngine();
    const unlocked = await engine.unlock();
    expect(unlocked).toBeTruthy();
    expect(engine.getStatus()).toBe("running");
    expect(engine.isReady()).toBeTruthy();
    engine.dispose();
  });

  // 2. Volume and Mute
  await it("3. Volume clamping and master gain scaling", async () => {
    const engine = new DriveGuardAudioEngine({ masterVolume: 0.5 });
    await engine.unlock();
    expect(engine.getVolume()).toBe(0.5);

    engine.setVolume(1.5); // Should clamp to 1.0
    expect(engine.getVolume()).toBe(1.0);

    engine.setVolume(-0.2); // Should clamp to 0.0
    expect(engine.getVolume()).toBe(0.0);

    engine.setVolume(0.8);
    expect(engine.getVolume()).toBe(0.8);
    engine.dispose();
  });

  await it("4. Mute immediately silences alerts and updates status", async () => {
    const engine = new DriveGuardAudioEngine();
    await engine.unlock();
    expect(engine.isMuted()).toBeFalsy();

    engine.setMuted(true);
    expect(engine.isMuted()).toBeTruthy();
    expect(engine.getStatus()).toBe("muted");
    expect(engine.isReady()).toBeFalsy();

    // Muted engine should reject playAlert
    const played = engine.playAlert("WARNING");
    expect(played).toBeFalsy();

    engine.setMuted(false);
    expect(engine.isMuted()).toBeFalsy();
    expect(engine.getStatus()).toBe("running");
    engine.dispose();
  });

  // 3. Synthesis: WARNING
  await it("5. Synthesizes WARNING alert: single 440 Hz sine chime (~180ms)", async () => {
    const engine = new DriveGuardAudioEngine({ warningCooldownMs: 100 });
    await engine.unlock();
    const initialOscCount = mockCtxInstance!.createdOscillators.length;

    const played = engine.setRiskState("WARNING");
    expect(played).toBeTruthy();

    const newOscs = mockCtxInstance!.createdOscillators.slice(initialOscCount);
    expect(newOscs.length).toBe(1);
    expect(newOscs[0].type).toBe("sine");
    expect(newOscs[0].frequency.value).toBe(440);
    expect(newOscs[0].started).toBeTruthy();
    expect(newOscs[0].stopped).toBeTruthy();
    // Stop time should include ramp completion grace period (~0.185s)
    expect(newOscs[0].stopTime).toBe(0.185);
    engine.dispose();
  });

  // 4. Synthesis: DROWSY
  await it("6. Synthesizes DROWSY alert: dual harmonic 587 Hz + 740 Hz double pulse", async () => {
    const engine = new DriveGuardAudioEngine({ drowsyCooldownMs: 100 });
    await engine.unlock();
    const initialOscCount = mockCtxInstance!.createdOscillators.length;

    const played = engine.setRiskState("DROWSY");
    expect(played).toBeTruthy();

    const newOscs = mockCtxInstance!.createdOscillators.slice(initialOscCount);
    // Should create 2 oscillators (587 Hz sine + 740 Hz triangle)
    expect(newOscs.length).toBe(2);
    const types = newOscs.map((o) => o.type).sort();
    expect(types).toEqual(["sine", "triangle"]);

    const freqs = newOscs.map((o) => o.frequency.value).sort();
    expect(freqs).toEqual([587, 740]);
    engine.dispose();
  });

  // 5. Synthesis: CRITICAL & Preemption
  await it("7. Synthesizes CRITICAL alert with lowpass filter & frequency sweep, preempting lower alerts", async () => {
    const engine = new DriveGuardAudioEngine({ criticalRepeatIntervalMs: 50 });
    await engine.unlock();

    // Start with WARNING
    engine.setRiskState("WARNING");

    const oscCountBefore = mockCtxInstance!.createdOscillators.length;
    const filterCountBefore = mockCtxInstance!.createdFilters.length;

    // Transition to CRITICAL (should preempt WARNING and start repeating loop)
    const transitioned = engine.setRiskState("CRITICAL", "crit_evt_1");
    expect(transitioned).toBeTruthy();

    // Verify filter was created to bandlimit harsh harmonics
    const newFilters = mockCtxInstance!.createdFilters.slice(filterCountBefore);
    expect(newFilters.length).toBeGreaterThan(0);
    expect(newFilters[0].type).toBe("lowpass");

    // Verify saw/square oscillators were created for the pulse
    const newOscs = mockCtxInstance!.createdOscillators.slice(oscCountBefore);
    expect(newOscs.length).toBe(2); // Saw + Square
    expect(newOscs[0].type).toBe("sawtooth");
    expect(newOscs[1].type).toBe("square");

    // Frequency sweep should start at 880 and ramp towards 1046
    const fCalls = newOscs[0].frequency.calls;
    expect(fCalls.some((c) => c.method === "setValueAtTime" && c.args[0] === 880)).toBeTruthy();
    expect(
      fCalls.some((c) => c.method === "exponentialRampToValueAtTime" && c.args[0] === 1046)
    ).toBeTruthy();

    // While CRITICAL is active, WARNING should be rejected (preempted)
    const warningRejected = engine.playAlert("WARNING");
    expect(warningRejected).toBeFalsy();

    engine.dispose();
  });

  // 6. Repeating CRITICAL alert loop & Recovery Chime
  await it("8. CRITICAL repeats periodically and plays RECOVERY chime on transition to NORMAL", async () => {
    const engine = new DriveGuardAudioEngine({ criticalRepeatIntervalMs: 40 });
    await engine.unlock();

    engine.setRiskState("CRITICAL", "evt_repeat_1");

    // Wait for at least one repetition trigger
    await new Promise((r) => setTimeout(r, 60));

    // Oscillators should have been created for multiple pulses across loop ticks
    expect(mockCtxInstance!.createdOscillators.length).toBeGreaterThan(2);

    // Transition to NORMAL -> should stop critical loop and play ascending recovery chime
    const initialOscsBeforeRecovery = mockCtxInstance!.createdOscillators.length;
    const recoveryPlayed = engine.setRiskState("NORMAL");
    expect(recoveryPlayed).toBeTruthy();

    // Recovery sequences two ascending frequency steps (523 Hz -> 659 Hz)
    const recoveryOscs = mockCtxInstance!.createdOscillators.slice(initialOscsBeforeRecovery);
    expect(recoveryOscs.length).toBe(1);
    const fCalls = recoveryOscs[0].frequency.calls;
    expect(fCalls.some((c) => c.method === "setValueAtTime" && c.args[0] === 523)).toBeTruthy();
    expect(fCalls.some((c) => c.method === "setValueAtTime" && c.args[0] === 659)).toBeTruthy();

    // Another NORMAL call without an alert state should NOT trigger a second recovery chime
    const duplicateNormal = engine.setRiskState("NORMAL");
    expect(duplicateNormal).toBeFalsy();

    engine.dispose();
  });

  // 7. Deduplication by event_id & Cooldowns
  await it("9. Deduplication of safety event_id prevents redundant alert triggers", async () => {
    const engine = new DriveGuardAudioEngine({ warningCooldownMs: 0 });
    await engine.unlock();

    const first = engine.setRiskState("WARNING", "unique_evt_99");
    expect(first).toBeTruthy();

    // Same event_id must be ignored
    const duplicate = engine.setRiskState("WARNING", "unique_evt_99");
    expect(duplicate).toBeFalsy();

    engine.dispose();
  });

  await it("10. Cooldown enforcement for WARNING and DROWSY alerts", async () => {
    const engine = new DriveGuardAudioEngine({
      warningCooldownMs: 1000,
      drowsyCooldownMs: 1000,
    });
    await engine.unlock();

    // First WARNING succeeds
    const w1 = engine.playAlert("WARNING");
    expect(w1).toBeTruthy();

    // Immediate second WARNING fails cooldown check
    const w2 = engine.playAlert("WARNING");
    expect(w2).toBeFalsy();

    // First DROWSY succeeds (different cooldown bucket)
    const d1 = engine.playAlert("DROWSY");
    expect(d1).toBeTruthy();

    // Immediate second DROWSY fails cooldown check
    const d2 = engine.playAlert("DROWSY");
    expect(d2).toBeFalsy();

    engine.dispose();
  });

  // 8. Disposal & Resource Teardown
  await it("11. Disposing the engine disconnects master gain and closes context", async () => {
    const engine = new DriveGuardAudioEngine();
    await engine.unlock();

    const ctx = mockCtxInstance!;
    expect(ctx.state).toBe("running");

    engine.dispose();
    expect(engine.getStatus()).toBe("uninitialized");
    expect(ctx.state).toBe("closed");
  });

  console.log("\n======================================================================");
  console.log(`   [DriveGuard Audio Tests] ${passed}/${total} Tests Passed Successfully!`);
  console.log("======================================================================\n");

  if (passed !== total) {
    process.exit(1);
  }
}

runTests().catch((err) => {
  console.error("Audio test run failed:", err);
  process.exit(1);
});
