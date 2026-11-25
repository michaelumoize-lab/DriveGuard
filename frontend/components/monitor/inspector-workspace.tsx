"use client";

import React, { useState } from "react";
import Link from "next/link";
import {
  Camera,
  Cpu,
  Monitor,
  Radio,
  ShieldCheck,
  SlidersHorizontal,
  Smartphone,
  Sparkles,
  Volume2,
  Activity,
  Download,
  FlaskConical,
  LayoutGrid,
  ArrowRight,
} from "lucide-react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Slider } from "@/components/ui/slider";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { LiveTelemetryHUD } from "./live-telemetry-hud";
import { ExplainableRiskPanel } from "./explainable-risk-panel";
import { LiveTelemetryChart, TelemetryHistoryPoint } from "./live-telemetry-chart";
import { CameraSelector } from "@/components/camera/camera-selector";
import { KinematicsOscilloscope } from "@/components/engineering/kinematics-oscilloscope";
import { EulerHeadGyro } from "@/components/engineering/euler-head-gyro";
import type {
  FrameTelemetry,
  LatencyDiagnostics,
} from "@/types/telemetry";
import type { CameraDevice, CameraState } from "@/types/camera";

export type InspectorTab = "overview" | "signals" | "testbench" | "setup";

export interface InspectorWorkspaceProps {
  activeTab: InspectorTab;
  onTabChange: (tab: InspectorTab) => void;
  telemetry: FrameTelemetry | null;
  cameraState: CameraState;
  runtimeMode: "browser" | "python-local" | "remote";
  onSwitchRuntimeMode: (mode: "browser" | "python-local" | "remote") => void;
  isSwitchingEngine?: boolean;
  backendType: "cloud" | "local";
  onSwitchBackend: (backend: "cloud" | "local") => void;
  latencyDiagnostics: LatencyDiagnostics | null;
  isStale: boolean;
  isAudioUnlocked: boolean;
  isMuted: boolean;
  volume: number;
  audioStatus: "ready" | "muted" | "blocked" | "unavailable";
  onToggleMute: () => void;
  onSetVolume: (volume: number) => void;
  onUnlockAudio: () => Promise<boolean>;
  onTestAudioTier: (tier: "WARNING" | "DROWSY" | "CRITICAL" | "RECOVERY") => void;
  onSwitchToCockpit: () => void;
  chartHistory: TelemetryHistoryPoint[];
  devices: CameraDevice[];
  selectedDeviceId: string;
  onSelectDevice: (deviceId: string) => void;
  onRefreshDevices: () => void;
  viewOrientation: "portrait" | "landscape";
  onOrientationChange: (orientation: "portrait" | "landscape") => void;
  hasActiveSession: boolean;
  browserRecordedFrames?: number;
  onExportBrowserTelemetry?: () => void;
}

export function InspectorWorkspace({
  activeTab,
  onTabChange,
  telemetry,
  cameraState,
  runtimeMode,
  onSwitchRuntimeMode,
  isSwitchingEngine = false,
  backendType,
  onSwitchBackend,
  latencyDiagnostics,
  isStale,
  isAudioUnlocked,
  isMuted,
  volume,
  audioStatus,
  onToggleMute,
  onSetVolume,
  onUnlockAudio,
  onTestAudioTier,
  onSwitchToCockpit,
  chartHistory,
  devices,
  selectedDeviceId,
  onSelectDevice,
  onRefreshDevices,
  viewOrientation,
  onOrientationChange,
  hasActiveSession,
  browserRecordedFrames = 0,
  onExportBrowserTelemetry,
}: InspectorWorkspaceProps) {
  const [pendingSwitch, setPendingSwitch] = useState<
    | { type: "runtime"; target: "browser" | "python-local" | "remote" }
    | { type: "backend"; target: "cloud" | "local" }
    | null
  >(null);

  const requestRuntimeSwitch = (target: "browser" | "python-local" | "remote") => {
    if (isSwitchingEngine || target === runtimeMode) return;
    if (hasActiveSession || cameraState === "active") {
      setPendingSwitch({ type: "runtime", target });
    } else {
      onSwitchRuntimeMode(target);
    }
  };

  const requestBackendSwitch = (target: "cloud" | "local") => {
    if (target === backendType) return;
    if (hasActiveSession || cameraState === "active") {
      setPendingSwitch({ type: "backend", target });
    } else {
      onSwitchBackend(target);
    }
  };

  const confirmPendingSwitch = () => {
    if (!pendingSwitch) return;
    if (pendingSwitch.type === "runtime") {
      onSwitchRuntimeMode(pendingSwitch.target);
    } else {
      onSwitchBackend(pendingSwitch.target);
    }
    setPendingSwitch(null);
  };

  return (
    <>
      <Tabs
        value={activeTab}
        onValueChange={(v) => onTabChange(v as InspectorTab)}
        className="w-full space-y-3"
      >
        <TabsList className="grid grid-cols-2 xs:grid-cols-4 w-full h-auto gap-1 p-1 rounded-xl bg-muted/60 border border-border/80">
          <TabsTrigger
            value="overview"
            className="min-h-11 sm:min-h-8 text-xs font-medium rounded-lg gap-1.5 cursor-pointer"
          >
            <LayoutGrid className="size-3.5 shrink-0" aria-hidden="true" />
            <span>Overview</span>
          </TabsTrigger>
          <TabsTrigger
            value="signals"
            className="min-h-11 sm:min-h-8 text-xs font-medium rounded-lg gap-1.5 cursor-pointer"
          >
            <Activity className="size-3.5 shrink-0" aria-hidden="true" />
            <span>Signals</span>
          </TabsTrigger>
          <TabsTrigger
            value="testbench"
            className="min-h-11 sm:min-h-8 text-xs font-medium rounded-lg gap-1.5 cursor-pointer"
          >
            <FlaskConical className="size-3.5 shrink-0" aria-hidden="true" />
            <span>Testbench</span>
          </TabsTrigger>
          <TabsTrigger
            value="setup"
            className="min-h-11 sm:min-h-8 text-xs font-medium rounded-lg gap-1.5 cursor-pointer"
          >
            <SlidersHorizontal className="size-3.5 shrink-0" aria-hidden="true" />
            <span>Setup</span>
          </TabsTrigger>
        </TabsList>

        {/* Tab 1: Overview (Operational State, Drowsiness Severity, Core Metrics, Explainable Reasons) */}
        <TabsContent value="overview" className="mt-0 flex flex-col gap-3.5 focus-visible:outline-none">
          <div className="grid grid-cols-1 xs:grid-cols-2 gap-2.5 text-xs">
            <div className="p-2.5 rounded-xl border border-border/80 bg-card flex items-center justify-between gap-2 min-w-0">
              <span className="text-muted-foreground truncate">Operational State</span>
              <Badge variant="outline" className="font-mono text-[10px] shrink-0">
                {telemetry?.operational_state ?? (cameraState === "active" ? "CALIBRATION_REQUIRED" : "IDLE")}
              </Badge>
            </div>
            <div className="p-2.5 rounded-xl border border-border/80 bg-card flex items-center justify-between gap-2 min-w-0">
              <span className="text-muted-foreground truncate">Drowsiness State</span>
              <Badge variant="outline" className="font-mono text-[10px] shrink-0">
                {telemetry?.state ?? "INSUFFICIENT_DATA"}
              </Badge>
            </div>
          </div>

          <LiveTelemetryHUD
            telemetry={telemetry}
            cameraState={cameraState}
            runtimeMode={runtimeMode}
            backendType={backendType}
            latencyDiagnostics={latencyDiagnostics}
            isStale={runtimeMode === "remote" ? isStale : false}
            isAudioUnlocked={isAudioUnlocked}
            isMuted={isMuted}
            volume={volume}
            audioStatus={audioStatus}
            onToggleMute={onToggleMute}
            onSetVolume={onSetVolume}
            onUnlockAudio={onUnlockAudio}
            onTestAudioTier={onTestAudioTier}
            onSwitchToCockpit={onSwitchToCockpit}
            showEngineStrip={false}
            showComparisonStrip={false}
            showAudioTestBar={false}
          />

          <ExplainableRiskPanel
            telemetry={telemetry}
            latencyDiagnostics={latencyDiagnostics}
            isStale={runtimeMode === "remote" ? isStale : false}
          />
        </TabsContent>

        {/* Tab 2: Signals (60s Chart, 60fps Oscilloscope, 3D Head Gyroscope) */}
        <TabsContent value="signals" className="mt-0 space-y-3.5 focus-visible:outline-none">
          <LiveTelemetryChart data={chartHistory} />

          <KinematicsOscilloscope
            currentEar={telemetry ? (telemetry.eye_metrics?.ear_raw ?? null) : null}
            sampleTimestampMs={telemetry?.timestamp_ms ?? null}
            closureThreshold={
              telemetry?.comparison?.personalised_threshold ??
              telemetry?.eye_metrics?.ear_closure_threshold ??
              0.2
            }
            rcr={telemetry?.eye_metrics?.blink_rcr ?? undefined}
            isActive={cameraState === "active" && activeTab === "signals"}
          />

          <EulerHeadGyro
            pitchDeg={telemetry?.head_pose?.pitch_deg ?? 0}
            yawDeg={telemetry?.head_pose?.yaw_deg ?? 0}
            rollDeg={telemetry?.head_pose?.roll_deg ?? 0}
            zone={telemetry?.comparison?.personalised_zone ?? telemetry?.pose_zone ?? "CENTER"}
            hasTelemetry={Boolean(telemetry)}
          />

          {runtimeMode === "browser" && onExportBrowserTelemetry && (
            <Card className="border-border/80 bg-muted/20">
              <CardContent className="p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
                <div className="space-y-0.5">
                  <div className="font-semibold text-foreground flex items-center gap-1.5">
                    <Download className="size-3.5 text-primary shrink-0" aria-hidden="true" />
                    <span>Browser AI Telemetry Recording</span>
                    <Badge variant="outline" className="font-mono text-[10px]">
                      {browserRecordedFrames.toLocaleString()} frames
                    </Badge>
                  </div>
                  <p className="text-muted-foreground">
                    Export captured Browser AI frames as canonical JSONL to audit with{" "}
                    <code className="font-mono text-[11px]">npm run analyze:frontend</code>.
                  </p>
                </div>
                <Button
                  size="sm"
                  variant="outline"
                  onClick={onExportBrowserTelemetry}
                  disabled={browserRecordedFrames === 0}
                  className="min-h-11 sm:h-8 sm:min-h-0 text-xs gap-1.5 shrink-0"
                >
                  <Download className="size-3.5" aria-hidden="true" />
                  Export .jsonl
                </Button>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        {/* Tab 3: Testbench (Live PDBP Comparison, Audio Tier Test & Link to Isolated Synthetic Testbench) */}
        <TabsContent value="testbench" className="mt-0 flex flex-col gap-3.5 focus-visible:outline-none">
          <Card className="border-border/80 bg-muted/20">
            <CardContent className="p-3.5 flex flex-col sm:flex-row sm:items-center justify-between gap-3 text-xs">
              <div className="space-y-0.5">
                <div className="font-semibold text-foreground flex items-center gap-1.5">
                  <FlaskConical className="size-3.5 text-primary shrink-0" aria-hidden="true" />
                  <span>Synthetic Fault Simulations Moved to Engineering Lab</span>
                </div>
                <p className="text-muted-foreground">
                  To keep production live monitoring 100% authentic, synthetic fault scenarios run in an isolated simulation workspace.
                </p>
              </div>
              <Button asChild size="sm" variant="outline" className="min-h-11 sm:h-8 sm:min-h-0 text-xs shrink-0">
                <Link href="/dashboard/synthetic-testbench">
                  Open Synthetic Testbench <ArrowRight className="size-3.5 ml-1" aria-hidden="true" />
                </Link>
              </Button>
            </CardContent>
          </Card>

          {/* PDBP vs Fixed 0.22 Rule Benchmark */}
          <Card className="border-border/80">
            <CardHeader className="p-3.5 pb-2 border-b border-border/60">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <CardTitle className="text-xs font-semibold text-foreground flex items-center gap-2">
                  <ShieldCheck className="size-4 text-emerald-500" aria-hidden="true" />
                  <span>Personalised 4-Zone vs Fixed 0.22 Rule</span>
                </CardTitle>
                {telemetry?.comparison?.fixed_false_alarm ? (
                  <Badge className="bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 text-[10px]">
                    False alarm prevented
                  </Badge>
                ) : telemetry?.comparison?.is_mirror_check_suppressed ? (
                  <Badge variant="outline" className="border-primary/40 text-primary text-[10px]">
                    Mirror check suppressed
                  </Badge>
                ) : null}
              </div>
            </CardHeader>
            <CardContent className="p-3.5 text-xs">
              {telemetry?.comparison ? (
                <div className="grid grid-cols-1 xs:grid-cols-3 gap-2 font-mono tabular-nums">
                  <div className="p-2.5 rounded-xl bg-muted/40 border border-border/60">
                    <div className="text-[11px] font-sans text-muted-foreground">Zone threshold</div>
                    <div className="font-bold text-foreground mt-0.5">
                      {telemetry.comparison.personalised_threshold.toFixed(3)}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-muted/40 border border-border/60">
                    <div className="text-[11px] font-sans text-muted-foreground">Static cutoff</div>
                    <div className="font-bold text-foreground mt-0.5">
                      {telemetry.comparison.fixed_cutoff_ear.toFixed(2)}
                    </div>
                  </div>
                  <div className="p-2.5 rounded-xl bg-muted/40 border border-border/60">
                    <div className="text-[11px] font-sans text-muted-foreground">Fixed rule state</div>
                    <div className="font-bold text-foreground mt-0.5 truncate">
                      {telemetry.comparison.fixed_alert_state}
                    </div>
                  </div>
                </div>
              ) : (
                <p className="text-muted-foreground">
                  Start the camera to compare your 4-zone personalised eyelid cutoff against a conventional fixed 0.22 threshold.
                </p>
              )}
            </CardContent>
          </Card>

          {/* 4-Tier Audio Alert Synthesizer Tester */}
          <Card className="border-border/80">
            <CardHeader className="p-3.5 pb-2 border-b border-border/60">
              <div className="flex flex-wrap items-center justify-between gap-2">
                <CardTitle className="text-xs font-semibold text-foreground flex items-center gap-2">
                  <Volume2 className="size-4 text-primary" aria-hidden="true" />
                  <span>Audio Alert Synthesizer Test</span>
                </CardTitle>
                <div className="flex items-center gap-2.5 min-h-11 sm:min-h-0">
                  <label
                    htmlFor="testbench-alert-volume"
                    className="text-[11px] font-mono tabular-nums text-muted-foreground"
                  >
                    Vol {Math.round(volume * 100)}%
                  </label>
                  <Slider
                    id="testbench-alert-volume"
                    min={0}
                    max={1}
                    step={0.05}
                    value={[volume]}
                    aria-label="Alert volume"
                    onValueChange={(vals) => onSetVolume(vals[0] ?? volume)}
                    className="w-24 sm:w-20 cursor-pointer py-2"
                  />
                </div>
              </div>
            </CardHeader>
            <CardContent className="p-3.5">
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs">
                <button
                  type="button"
                  onClick={() => onTestAudioTier("WARNING")}
                  className="min-h-11 sm:min-h-9 py-2 px-2.5 rounded-xl border border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-400 font-medium hover:bg-amber-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Warn Beep
                </button>
                <button
                  type="button"
                  onClick={() => onTestAudioTier("DROWSY")}
                  className="min-h-11 sm:min-h-9 py-2 px-2.5 rounded-xl border border-orange-500/40 bg-orange-500/10 text-orange-600 dark:text-orange-400 font-medium hover:bg-orange-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Drowsy Pulse
                </button>
                <button
                  type="button"
                  onClick={() => onTestAudioTier("CRITICAL")}
                  className="min-h-11 sm:min-h-9 py-2 px-2.5 rounded-xl border border-rose-500/40 bg-rose-500/10 text-rose-600 dark:text-rose-400 font-medium hover:bg-rose-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Critical Siren
                </button>
                <button
                  type="button"
                  onClick={() => onTestAudioTier("RECOVERY")}
                  className="min-h-11 sm:min-h-9 py-2 px-2.5 rounded-xl border border-emerald-500/40 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 font-medium hover:bg-emerald-500/20 transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  Recovery Tone
                </button>
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Tab 4: Setup (Runtime, Cascade, Camera, Endpoint, Layout) */}
        <TabsContent value="setup" className="mt-0 space-y-3.5 focus-visible:outline-none">
          {/* Runtime Mode */}
          <Card className="border-border/80">
            <CardHeader className="p-3.5 pb-2 border-b border-border/60">
              <CardTitle className="text-xs font-semibold text-foreground flex items-center gap-2">
                <Sparkles className="size-4 text-primary" aria-hidden="true" />
                <span>AI Processing Runtime</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3.5 space-y-2.5">
              <div className="grid grid-cols-1 gap-2">
                <Button
                  type="button"
                  disabled={isSwitchingEngine}
                  variant={runtimeMode === "browser" ? "default" : "outline"}
                  onClick={() => requestRuntimeSwitch("browser")}
                  className="h-auto py-2.5 px-3 rounded-xl justify-start text-left gap-2.5"
                >
                  <Sparkles className="size-4 shrink-0" aria-hidden="true" />
                  <div className="min-w-0">
                    <div className="text-xs font-semibold">On-Device Browser AI (Default)</div>
                    <div className="text-[11px] opacity-80 truncate">
                      MediaPipe WASM · Works offline · Zero backend required
                    </div>
                  </div>
                </Button>

                <Button
                  type="button"
                  disabled={isSwitchingEngine}
                  variant={runtimeMode === "python-local" ? "default" : "outline"}
                  onClick={() => requestRuntimeSwitch("python-local")}
                  className="h-auto py-2.5 px-3 rounded-xl justify-start text-left gap-2.5"
                >
                  <Cpu className="size-4 shrink-0" aria-hidden="true" />
                  <div className="min-w-0">
                    <div className="text-xs font-semibold">Local Python AI (Direct Camera)</div>
                    <div className="text-[11px] opacity-80 truncate">
                      127.0.0.1:8000 OpenCV + SQLite · Zero LiveKit or cloud
                    </div>
                  </div>
                </Button>

                <Button
                  type="button"
                  disabled={isSwitchingEngine}
                  variant={runtimeMode === "remote" ? "default" : "outline"}
                  onClick={() => requestRuntimeSwitch("remote")}
                  className="h-auto py-2.5 px-3 rounded-xl justify-start text-left gap-2.5"
                >
                  <Radio className="size-4 shrink-0" aria-hidden="true" />
                  <div className="min-w-0">
                    <div className="text-xs font-semibold">Remote LiveKit AI (Cloud)</div>
                    <div className="text-[11px] opacity-80 truncate">
                      LiveKit WebRTC + Remote Python FastAPI
                    </div>
                  </div>
                </Button>
              </div>

              {runtimeMode === "remote" && (
                <div className="pt-2 border-t border-border/60 space-y-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-muted-foreground">Remote server target</span>
                    <span className="font-mono font-semibold text-foreground" suppressHydrationWarning>
                      {backendType === "cloud" ? "LiveKit Cloud (Render)" : "127.0.0.1:8000"}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 gap-2">
                    <Button
                      type="button"
                      size="sm"
                      variant={backendType === "cloud" ? "default" : "outline"}
                      onClick={() => requestBackendSwitch("cloud")}
                      className="text-xs min-h-11 sm:h-9 sm:min-h-0 rounded-xl"
                    >
                      LiveKit Cloud
                    </Button>
                    <Button
                      type="button"
                      size="sm"
                      variant={backendType === "local" ? "default" : "outline"}
                      onClick={() => requestBackendSwitch("local")}
                      className="text-xs min-h-11 sm:h-9 sm:min-h-0 rounded-xl"
                    >
                      Localhost (8000)
                    </Button>
                  </div>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Authoritative Production Pipeline Mode */}
          <Card className="border-border/80">
            <CardHeader className="p-3.5 pb-2 border-b border-border/60">
              <CardTitle className="text-xs font-semibold text-foreground flex items-center gap-2">
                <Cpu className="size-4 text-primary" aria-hidden="true" />
                <span>Authoritative Detection Pipeline</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3.5 space-y-2 text-xs">
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Production Decision Engine</span>
                <Badge variant="outline" className="font-mono text-[10px] border-emerald-500/40 text-emerald-500 bg-emerald-500/10">
                  GEOMETRIC_AUTHORITATIVE
                </Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-muted-foreground">Passive Eye CNN Shadow Observer</span>
                <Badge variant="outline" className="font-mono text-[10px]">
                  SHADOW_DISABLED
                </Badge>
              </div>
              <p className="text-[11px] text-muted-foreground leading-relaxed pt-1">
                Production alerts are driven deterministically by 4-zone calibrated EAR, sub-frame blink kinematics, rolling PERCLOS, MAR yawns, and 6-DoF head pose. Offline historical ablation experiments live under{" "}
                <Link href="/dashboard/benchmarks" className="text-primary hover:underline">
                  Ablation Benchmarks
                </Link>
                .
              </p>
            </CardContent>
          </Card>

          {/* Camera & Preview Orientation */}
          <Card className="border-border/80">
            <CardHeader className="p-3.5 pb-2 border-b border-border/60">
              <CardTitle className="text-xs font-semibold text-foreground flex items-center gap-2">
                <Camera className="size-4 text-primary" aria-hidden="true" />
                <span>Camera Device &amp; Viewport Framing</span>
              </CardTitle>
            </CardHeader>
            <CardContent className="p-3.5 space-y-3">
              <CameraSelector
                devices={devices}
                selectedDeviceId={selectedDeviceId}
                onSelectDevice={onSelectDevice}
                onRefreshDevices={onRefreshDevices}
              />

              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-border/60">
                <span className="text-xs text-muted-foreground">Preview aspect ratio</span>
                <div
                  className="inline-flex items-center rounded-xl bg-muted/60 p-1 border border-border/80"
                  role="group"
                  aria-label="Viewport orientation"
                >
                  <button
                    type="button"
                    onClick={() => onOrientationChange("landscape")}
                    aria-pressed={viewOrientation === "landscape"}
                    className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 min-h-10 sm:min-h-7 py-1 text-xs font-medium transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                      viewOrientation === "landscape"
                        ? "bg-background text-foreground shadow-xs"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Monitor className="size-3.5" aria-hidden="true" />
                    <span>Landscape</span>
                  </button>
                  <button
                    type="button"
                    onClick={() => onOrientationChange("portrait")}
                    aria-pressed={viewOrientation === "portrait"}
                    className={`inline-flex items-center gap-1.5 rounded-lg px-2.5 min-h-10 sm:min-h-7 py-1 text-xs font-medium transition-colors cursor-pointer focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring ${
                      viewOrientation === "portrait"
                        ? "bg-background text-foreground shadow-xs"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    <Smartphone className="size-3.5" aria-hidden="true" />
                    <span>Portrait</span>
                  </button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>

      {/* Confirmation Dialog when switching AI engine during an active session */}
      <AlertDialog
        open={Boolean(pendingSwitch)}
        onOpenChange={(open) => {
          if (!open) setPendingSwitch(null);
        }}
      >
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Switch AI engine mid-session?</AlertDialogTitle>
            <AlertDialogDescription>
              {pendingSwitch?.type === "backend"
                ? "Switching server targets will end and save your current monitoring session before reconnecting."
                : "Switching AI runtimes will briefly pause telemetry analysis while the new engine initializes."}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Keep current</AlertDialogCancel>
            <AlertDialogAction onClick={confirmPendingSwitch}>
              Switch engine
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
