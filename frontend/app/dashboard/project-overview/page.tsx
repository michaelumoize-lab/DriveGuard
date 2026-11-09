import React from "react";
import { redirect } from "next/navigation";
import Link from "next/link";
import { getServerSession } from "@/lib/get-session";
import { prisma } from "@/lib/prisma";
import {
  Sparkles,
  ShieldCheck,
  Cpu,
  Database,
  Layers,
  Camera,
  Server,
  ArrowRight,
  Terminal,
  Activity,
  HardDrive,
  Users,
  CheckCircle2,
  AlertTriangle,
  Radio,
  FileCode,
  ShieldAlert,
  Sliders,
  ExternalLink,
  Eye,
  Car,
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export const dynamic = "force-dynamic";

export default async function ProjectOverviewPage() {
  const session = await getServerSession();

  if (!session?.user) {
    redirect("/auth/sign-in?redirectTo=/dashboard/project-overview");
  }

  const userId = session.user.id;

  // Fetch user-scoped summary counts for the dashboard overview
  const [
    totalSessionsCount,
    driverProfile,
    warningAlertsCount,
    criticalAlertsCount,
  ] = await Promise.all([
    prisma.drivingSession.count({ where: { userId } }),
    prisma.driverProfile.findUnique({ where: { userId } }),
    prisma.drowsinessAlert.count({ where: { severity: "WARNING", session: { userId } } }),
    prisma.drowsinessAlert.count({ where: { severity: "CRITICAL", session: { userId } } }),
  ]);

  const calibratedZonesCount = driverProfile?.zoneBaselines
    ? Object.keys(driverProfile.zoneBaselines as Record<string, unknown>).length
    : driverProfile?.isCalibrated
      ? 4
      : 0;

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-8 max-w-7xl mx-auto">
      {/* 1. Project Title & Mission Banner */}
      <div className="relative overflow-hidden rounded-2xl border border-primary/20 bg-gradient-to-br from-primary/10 via-primary/5 to-transparent p-6 sm:p-8 shadow-sm">
        <div className="max-w-3xl space-y-3">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono border-primary/40 text-primary bg-primary/10">
              Final-Year Engineering Project
            </Badge>
            <Badge variant="secondary" className="text-xs font-mono">
              Computer Vision & AI
            </Badge>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-foreground leading-[1.15]">
            DriveGuard: Personalised Adaptive AI Driver Drowsiness Detection System
          </h1>

          <p className="text-base sm:text-lg text-muted-foreground leading-relaxed">
            A personalised automotive safety platform combining 4-zone pose-conditioned calibration, sub-frame blink velocity kinematics, rolling time-weighted PERCLOS, and a confidence-gated MobileNetV3-Small deep learning cascade to deliver reliable, explainable driver drowsiness detection.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Button asChild className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/dashboard/live-monitor">
                <Activity className="w-4 h-4" /> Launch Driver Monitor
              </Link>
            </Button>
            <Button asChild variant="outline" className="gap-2 border-border hover:bg-muted">
              <Link href="/dashboard/calibration">
                <Eye className="w-4 h-4 text-primary" /> 4-Zone Calibration
              </Link>
            </Button>
          </div>
        </div>

        {/* Executive Metrics Overview */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 pt-8 mt-6 border-t border-border/60">
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-xs uppercase font-mono text-muted-foreground">Your Sessions</span>
            <p className="text-xl sm:text-2xl font-bold font-mono tabular-nums text-foreground mt-0.5">{totalSessionsCount}</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-xs uppercase font-mono text-muted-foreground">Baseline Status</span>
            <p className="text-xl sm:text-2xl font-bold font-mono tabular-nums text-emerald-500 mt-0.5">
              {driverProfile?.isCalibrated ? "Calibrated" : "Pending"}
            </p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-xs uppercase font-mono text-muted-foreground">Active Spatial Zones</span>
            <p className="text-xl sm:text-2xl font-bold font-mono tabular-nums text-foreground mt-0.5">{calibratedZonesCount} / 4</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-xs uppercase font-mono text-muted-foreground">Warning Alerts</span>
            <p className="text-xl sm:text-2xl font-bold font-mono tabular-nums text-amber-500 mt-0.5">{warningAlertsCount}</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80 col-span-2 sm:col-span-1">
            <span className="text-xs uppercase font-mono text-muted-foreground">Critical Interventions</span>
            <p className="text-xl sm:text-2xl font-bold font-mono tabular-nums text-rose-500 mt-0.5">{criticalAlertsCount}</p>
          </div>
        </div>
      </div>

      {/* 2. Three Pillars of DriveGuard Architecture */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Architecture</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Three Pillars of DriveGuard Technology</h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-3xl">
            DriveGuard solves the three critical failure modes of conventional drowsiness detectors: false alarms during mirror checks, slow response to micro-sleeps, and high CPU power consumption.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <Card className="border border-border/80 bg-card/60">
            <CardHeader className="pb-3">
              <div className="p-2.5 rounded-xl bg-primary/10 text-primary w-fit mb-2">
                <Eye className="size-5" />
              </div>
              <CardTitle className="text-base font-semibold">1. 4-Zone PDBP Calibration</CardTitle>
              <CardDescription className="text-xs">Pose-Conditioned Dynamic Baselines</CardDescription>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground space-y-2">
              <p>
                Eliminates false alarms when glancing at mirrors or dashboard. The system calibrates separate EAR thresholds for CENTER, LEFT MIRROR, RIGHT MIRROR, and DASHBOARD.
              </p>
              <div className="p-2 rounded bg-muted/40 font-mono text-[11px]">
                Adaptive: EAR_th(zone) = med(zone) - 2.3 * MAD(zone)
              </div>
            </CardContent>
          </Card>

          <Card className="border border-border/80 bg-card/60">
            <CardHeader className="pb-3">
              <div className="p-2.5 rounded-xl bg-primary/10 text-primary w-fit mb-2">
                <Cpu className="size-5" />
              </div>
              <CardTitle className="text-base font-semibold">2. Confidence-Gated Cascade</CardTitle>
              <CardDescription className="text-xs">Edge Efficiency with Deep Verifier</CardDescription>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground space-y-2">
              <p>
                Runs fast geometric analysis on every frame (~2.5ms). Only invokes the MobileNetV3-Small ONNX classifier when geometric confidence is low or when near the closure boundary, reducing inference power by ~67%.
              </p>
              <div className="p-2 rounded bg-muted/40 font-mono text-[11px]">
                Arbitration: Reliability-Weighted Consensus
              </div>
            </CardContent>
          </Card>

          <Card className="border border-border/80 bg-card/60">
            <CardHeader className="pb-3">
              <div className="p-2.5 rounded-xl bg-primary/10 text-primary w-fit mb-2">
                <Activity className="size-5" />
              </div>
              <CardTitle className="text-base font-semibold">3. Sub-Frame Kinematics</CardTitle>
              <CardDescription className="text-xs">Micro-Sleep Velocity Tracking</CardDescription>
            </CardHeader>
            <CardContent className="text-xs text-muted-foreground space-y-2">
              <p>
                Captures parabolic blink trajectory, opening-to-closing velocity ratios (RCR), rolling 60s PERCLOS, and head nodding pitch to detect micro-sleeps up to 1.8 seconds faster than static rules.
              </p>
              <div className="p-2 rounded bg-muted/40 font-mono text-[11px]">
                FSM States: NORMAL &rarr; WARNING &rarr; CRITICAL
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 3. System Tech Stack Summary */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Tech Stack</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Production Technical Stack</h2>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl border border-border/80 bg-card/60 space-y-2">
            <div className="flex items-center gap-2 font-semibold text-sm text-foreground">
              <Server className="size-4 text-primary" /> Fast AI Backend
            </div>
            <p className="text-xs text-muted-foreground">
              Python 3.11, FastAPI, MediaPipe Tasks 478 Mesh, ONNX Runtime MobileNetV3-Small, OpenCV Head Pose solvePnP.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-border/80 bg-card/60 space-y-2">
            <div className="flex items-center gap-2 font-semibold text-sm text-foreground">
              <Activity className="size-4 text-primary" /> Real-Time WebRTC Transport
            </div>
            <p className="text-xs text-muted-foreground">
              LiveKit Cloud SFU video streaming with dual-channel DataChannel v2 telemetry (lossy per-frame UDP + reliable ACKed safety events).
            </p>
          </div>

          <div className="p-4 rounded-xl border border-border/80 bg-card/60 space-y-2">
            <div className="flex items-center gap-2 font-semibold text-sm text-foreground">
              <Database className="size-4 text-primary" /> Relational Storage
            </div>
            <p className="text-xs text-muted-foreground">
              PostgreSQL on Neon Cloud, Prisma ORM, storing 4-zone driver profiles, driving sessions, and explainable fatigue alert timelines.
            </p>
          </div>

          <div className="p-4 rounded-xl border border-border/80 bg-card/60 space-y-2">
            <div className="flex items-center gap-2 font-semibold text-sm text-foreground">
              <Layers className="size-4 text-primary" /> Modern Frontend & Audio
            </div>
            <p className="text-xs text-muted-foreground">
              Next.js 16 (App Router), React 19, Better Auth, Tailwind CSS v4, synthesized 4-tier Web Audio API alert engine.
            </p>
          </div>
        </div>
      </section>

      {/* 4. Presentation & Defense Callout */}
      <Card className="border border-primary/30 bg-primary/5">
        <CardContent className="p-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center md:text-left">
            <h4 className="text-base font-bold text-foreground">Project Defense & Demonstration Ready</h4>
            <p className="text-xs text-muted-foreground">
              DriveGuard integrates real-time computer vision, adaptive statistical baselines, and modern web systems for academic presentation.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/dashboard/live-monitor">
                <Camera className="w-4 h-4 mr-1.5" /> Start Live Defense Demo
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
