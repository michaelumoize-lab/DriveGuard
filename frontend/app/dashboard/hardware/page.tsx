"use client";

import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Cpu,
  Radio,
  Zap,
  Volume2,
  Sliders,
  CheckCircle2,
  Clock,
  ShieldAlert,
  Flame,
  Activity,
  Play,
  RotateCcw,
  Sparkles,
} from "lucide-react";
import { toast } from "sonner";

export default function HardwareConsolePage() {
  const [hapticMode, setHapticMode] = useState<"TRANSITION" | "PULSE" | "BUZZ">("TRANSITION");
  const [isHapticActive, setIsHapticActive] = useState(false);
  const [reactionTimerMs, setReactionTimerMs] = useState<number | null>(482);
  const [simulatedFrameId, setSimulatedFrameId] = useState(1402);
  const [busCurrentMa, setBusCurrentMa] = useState(42.5);

  useEffect(() => {
    const timer = setInterval(() => {
      setSimulatedFrameId((prev) => (prev + 1) % 65535);
    }, 500);
    return () => clearInterval(timer);
  }, []);

  const triggerHapticTest = () => {
    setIsHapticActive(true);
    setBusCurrentMa(165.0); // Current surge during motor activation
    toast.info(`DRV2605L Triggered: Waveform ${hapticMode} (Address 0x5A)`);
    setTimeout(() => {
      setIsHapticActive(false);
      setBusCurrentMa(42.5);
    }, 1200);
  };

  const simulateDriverAck = () => {
    const measuredTime = Math.floor(Math.random() * 400) + 320;
    setReactionTimerMs(measuredTime);
    toast.success(`HITL Button Pressed: Driver Reaction Time t = ${measuredTime} ms [ALERT SILENCED]`);
  };

  return (
    <div className="p-3 sm:p-5 lg:p-6 max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border/80 pb-5">
        <div>
          <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-mono mb-2">
            <Cpu className="size-3" />
            <span>Computer Engineering Deliverable 5</span>
          </div>
          <h1 className="text-2xl sm:text-3xl font-extrabold text-foreground tracking-tight">
            Cyber-Physical Co-Processor &amp; Hardware HMI Console
          </h1>
          <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
            Real-time microcontroller supervision (ESP32/STM32) over an isolated UART serial link.
            Manages deterministic tri-modal actuation and closed-loop Human-in-the-Loop (HITL) reaction chronometry.
          </p>
        </div>

        <div className="flex items-center gap-2.5">
          <Badge variant="outline" className="text-xs font-mono text-emerald-500 border-emerald-500/30">
            UART LINK: 115200 BAUD [ACTIVE]
          </Badge>
        </div>
      </div>

      {/* Top Telemetry Gauges */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
        <Card className="border-border/80 bg-card/60 shadow-xs">
          <CardContent className="p-4 space-y-1">
            <div className="flex items-center justify-between text-muted-foreground text-xs font-mono">
              <span>WATCHDOG SUPERVISOR</span>
              <ShieldAlert className="size-4 text-emerald-500" />
            </div>
            <div className="text-2xl font-black text-emerald-500 font-mono">
              498 <span className="text-sm font-semibold text-foreground">/ 500 ms</span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Hardware countdown timer; trips failsafe if video engine freezes.
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80 bg-card/60 shadow-xs">
          <CardContent className="p-4 space-y-1">
            <div className="flex items-center justify-between text-muted-foreground text-xs font-mono">
              <span>HITL REACTION CHRONO</span>
              <Clock className="size-4 text-sky-400" />
            </div>
            <div className="text-2xl font-black text-sky-400 font-mono">
              {reactionTimerMs} <span className="text-sm font-semibold text-foreground">ms</span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              Driver physical steering acknowledgment latency (t &lt; 1500 ms).
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80 bg-card/60 shadow-xs">
          <CardContent className="p-4 space-y-1">
            <div className="flex items-center justify-between text-muted-foreground text-xs font-mono">
              <span>INA219 LOAD CURRENT</span>
              <Zap className="size-4 text-amber-500" />
            </div>
            <div className="text-2xl font-black text-amber-500 font-mono">
              {busCurrentMa.toFixed(1)} <span className="text-sm font-semibold text-foreground">mA</span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              I2C sensor address 0x40 (V_bus = 3.3V, P = {(3.3 * busCurrentMa).toFixed(0)} mW).
            </p>
          </CardContent>
        </Card>

        <Card className="border-border/80 bg-card/60 shadow-xs">
          <CardContent className="p-4 space-y-1">
            <div className="flex items-center justify-between text-muted-foreground text-xs font-mono">
              <span>I2C MULTI-DROP BUS</span>
              <Radio className="size-4 text-purple-400" />
            </div>
            <div className="text-2xl font-black text-foreground font-mono">
              3 <span className="text-sm font-semibold text-purple-400">Slaves Locked</span>
            </div>
            <p className="text-[11px] text-muted-foreground">
              DRV2605L (0x5A), SSD1306 (0x3C), INA219 (0x40).
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Hardware Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        {/* Left Column: UART Frame Inspector & OLED Simulator */}
        <div className="lg:col-span-7 space-y-5">
          {/* UART Binary Protocol Inspector */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 border-b border-border/60">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Radio className="size-4 text-primary" />
                  <CardTitle className="text-sm font-bold text-foreground">
                    Live 8-Byte Packed Binary UART Frame Inspector
                  </CardTitle>
                </div>
                <Badge variant="secondary" className="text-[10px] font-mono">
                  FULL-DUPLEX 115200 BAUD
                </Badge>
              </div>
              <CardDescription className="text-xs">
                Guarantees deterministic sub-millisecond telemetry transmission without JSON serialization overhead.
              </CardDescription>
            </CardHeader>
            <CardContent className="p-4 space-y-3 font-mono text-xs">
              <div className="grid grid-cols-4 sm:grid-cols-8 gap-2 text-center">
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">MAGIC 1</span>
                  <span className="font-bold text-sky-400">0xAA</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">MAGIC 2</span>
                  <span className="font-bold text-sky-400">0x55</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">FRAME ID</span>
                  <span className="font-bold text-foreground">0x{simulatedFrameId.toString(16).toUpperCase()}</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">ZONE</span>
                  <span className="font-bold text-emerald-400">0x00</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">EAR (Q8)</span>
                  <span className="font-bold text-cyan-400">0x48</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">RISK (Q8)</span>
                  <span className="font-bold text-amber-400">0x12</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">STATE</span>
                  <span className="font-bold text-emerald-500">0x01</span>
                </div>
                <div className="p-2 rounded bg-muted/40 border border-border">
                  <span className="text-[9px] text-muted-foreground block">CRC-8</span>
                  <span className="font-bold text-purple-400">0x9F</span>
                </div>
              </div>
              <p className="text-[11px] text-muted-foreground font-sans pt-1">
                Parsed by FreeRTOS ISR ring buffer on ESP32 in &lt; 85 microseconds.
              </p>
            </CardContent>
          </Card>

          {/* SSD1306 0.96" OLED HUD Screen Simulator */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 border-b border-border/60">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Activity className="size-4 text-primary" />
                  <CardTitle className="text-sm font-bold text-foreground">
                    SSD1306 0.96&quot; Monochrome OLED HUD Simulator (I2C: 0x3C)
                  </CardTitle>
                </div>
                <Badge variant="outline" className="text-[10px] font-mono text-zinc-400 border-zinc-700">
                  128 x 64 PIXELS
                </Badge>
              </div>
            </CardHeader>
            <CardContent className="p-4 flex flex-col sm:flex-row items-center gap-4">
              {/* Monochrome Screen Frame */}
              <div className="w-56 h-28 bg-black rounded-lg border-4 border-zinc-800 p-2 text-sky-300 font-mono text-[10px] flex flex-col justify-between shadow-2xl shrink-0">
                <div className="flex justify-between border-b border-sky-900 pb-1">
                  <span>DRIVEGUARD HUD</span>
                  <span className="text-emerald-400">NOMINAL</span>
                </div>
                <div className="text-center py-1">
                  <div className="text-xs font-bold text-white tracking-widest">STATE: NORMAL</div>
                  <div className="text-[9px] text-sky-400">EAR: 0.284 · ZONE: CENTER</div>
                </div>
                <div className="flex justify-between border-t border-sky-900 pt-1 text-[8px] text-zinc-400">
                  <span>PERCLOS: 3.2%</span>
                  <span>WDT: 498ms</span>
                </div>
              </div>

              <div className="space-y-2 text-xs text-muted-foreground">
                <p>
                  Line-of-sight windshield micro-display powered via hardware SPI/I2C. Displays glanceable warnings even if the in-cabin infotainment head unit experiences an OS failure.
                </p>
              </div>
            </CardContent>
          </Card>
        </div>

        {/* Right Column: Actuator Controls & Closed-Loop HITL */}
        <div className="lg:col-span-5 space-y-5">
          {/* DRV2605L Haptics Controller */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 border-b border-border/60">
              <div className="flex items-center gap-2">
                <Zap className="size-4 text-amber-500" />
                <CardTitle className="text-sm font-bold text-foreground">
                  TI DRV2605L Somatosensory Haptics (I2C: 0x5A)
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="p-4 space-y-3">
              <p className="text-xs text-muted-foreground">
                Linear Resonant Actuator (LRA) mounted in steering wheel and driver seat cushion:
              </p>

              <div className="flex items-center gap-2 font-mono text-xs">
                {(["TRANSITION", "PULSE", "BUZZ"] as const).map((m) => (
                  <Button
                    key={m}
                    size="sm"
                    variant={hapticMode === m ? "default" : "outline"}
                    onClick={() => setHapticMode(m)}
                    className="flex-1 h-8 text-xs font-mono"
                  >
                    {m}
                  </Button>
                ))}
              </div>

              <Button
                size="sm"
                onClick={triggerHapticTest}
                className={`w-full font-mono text-xs ${
                  isHapticActive ? "bg-amber-500 text-black animate-pulse" : "bg-primary text-primary-foreground"
                }`}
              >
                <Zap className="size-3.5 mr-1.5" />
                {isHapticActive ? "Vibrating Steering LRA..." : "Test Haptic Pulse (DRV2605L)"}
              </Button>
            </CardContent>
          </Card>

          {/* Closed-Loop Human-in-the-Loop (HITL) */}
          <Card className="border-border/80 shadow-xs">
            <CardHeader className="p-4 border-b border-border/60">
              <div className="flex items-center gap-2">
                <Clock className="size-4 text-sky-400" />
                <CardTitle className="text-sm font-bold text-foreground">
                  Closed-Loop HITL Reaction Chronometer
                </CardTitle>
              </div>
            </CardHeader>
            <CardContent className="p-4 space-y-3 text-xs">
              <p className="text-muted-foreground">
                When a warning sounds, the driver touches the steering acknowledgment button. Microsecond reaction time is measured to evaluate cognitive impairment:
              </p>

              <div className="p-3 rounded-xl border border-sky-500/30 bg-sky-500/5 font-mono flex items-center justify-between">
                <div>
                  <span className="text-[10px] text-muted-foreground block">LAST REACTION TIME</span>
                  <span className="text-xl font-bold text-sky-400">{reactionTimerMs} ms</span>
                </div>
                <Badge variant="outline" className="text-xs text-emerald-500 border-emerald-500/30">
                  OPTIMAL (&lt; 600 ms)
                </Badge>
              </div>

              <Button
                size="sm"
                variant="outline"
                onClick={simulateDriverAck}
                className="w-full text-xs font-mono"
              >
                <CheckCircle2 className="size-3.5 mr-1.5 text-emerald-500" />
                Simulate Steering Wheel Button Press
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
