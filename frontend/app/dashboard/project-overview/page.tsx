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
  Lock,
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
} from "lucide-react";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  InteractiveArchitectureViewer,
  BiometricPipelineVisualizer,
  DecisionFlowchart,
} from "@/components/dashboard/project-overview-interactive";

export const dynamic = "force-dynamic";

export default async function ProjectOverviewPage() {
  const session = await getServerSession();

  // Strict server-side authorization: unauthenticated callers redirect to sign-in
  if (!session?.user) {
    redirect("/auth/sign-in?redirectTo=/dashboard/project-overview");
  }

  // Strict server-side authorization: non-admin callers redirect to regular dashboard
  const userRole = session.user.role;
  if (userRole !== "ADMIN" && userRole !== "admin") {
    redirect("/dashboard");
  }

  // Retrieve live statistics for executive metrics panel
  const [
    totalSessionsCount,
    calibratedDriversCount,
    criticalAlertsCount,
    warningAlertsCount,
    totalProfilesCount,
  ] = await Promise.all([
    prisma.drivingSession.count(),
    prisma.driverProfile.count({ where: { isCalibrated: true } }),
    prisma.drowsinessAlert.count({ where: { severity: "CRITICAL" } }),
    prisma.drowsinessAlert.count({ where: { severity: "WARNING" } }),
    prisma.userProfile.count(),
  ]);

  return (
    <div className="max-w-7xl mx-auto p-4 sm:p-6 md:p-10 space-y-8 sm:space-y-10 bg-background text-foreground">
      {/* 1. Hero / Executive Engineering Showcase Banner */}
      <div className="relative overflow-hidden rounded-3xl bg-gradient-to-br from-primary/20 via-background to-card border border-primary/30 p-6 sm:p-8 md:p-12 shadow-md">
        <div className="relative z-10 max-w-4xl space-y-4">
          <div className="flex flex-wrap items-center gap-2">
            <Badge variant="outline" className="bg-primary/10 text-primary border-primary/30 px-3 py-1 text-xs font-semibold">
              <Sparkles className="w-3.5 h-3.5 mr-1" /> Final-Year Computer Engineering Project
            </Badge>
            <Badge variant="secondary" className="text-xs font-mono">
              Academic Thesis Edition
            </Badge>
          </div>

          <h1 className="text-3xl sm:text-4xl md:text-5xl font-extrabold tracking-tight text-foreground leading-[1.15]">
            IntelliGuard: Personalised Adaptive AI Driver Drowsiness Detection & Early Warning System
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
            <span className="text-[10px] uppercase font-mono text-muted-foreground">Driving Sessions</span>
            <p className="text-xl sm:text-2xl font-bold font-mono text-foreground mt-0.5">{totalSessionsCount}</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-[10px] uppercase font-mono text-muted-foreground">Calibrated Baselines</span>
            <p className="text-xl sm:text-2xl font-bold font-mono text-emerald-500 mt-0.5">{calibratedDriversCount}</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-[10px] uppercase font-mono text-muted-foreground">Configured Drivers</span>
            <p className="text-xl sm:text-2xl font-bold font-mono text-foreground mt-0.5">{totalProfilesCount}</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80">
            <span className="text-[10px] uppercase font-mono text-muted-foreground">Warning Alerts</span>
            <p className="text-xl sm:text-2xl font-bold font-mono text-amber-500 mt-0.5">{warningAlertsCount}</p>
          </div>
          <div className="p-3 bg-card/60 backdrop-blur-md rounded-xl border border-border/80 col-span-2 sm:col-span-1">
            <span className="text-[10px] uppercase font-mono text-muted-foreground">Critical Interventions</span>
            <p className="text-xl sm:text-2xl font-bold font-mono text-rose-500 mt-0.5">{criticalAlertsCount}</p>
          </div>
        </div>
      </div>

      {/* 2. Interactive End-to-End System Architecture */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Architecture</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Interactive System Architecture</h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-3xl">
            Explore the decoupled multi-service architecture of IntelliGuard. Click any component in the visual topology to inspect communication protocols, computational roles, and security guarantees.
          </p>
        </div>
        <InteractiveArchitectureViewer />
      </section>

      {/* 3. Biometric Pipeline Walkthrough */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Computer Vision</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Facial Recognition & Vectorization Pipeline</h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-3xl">
            InsightFace utilizes deep convolutional neural networks with ArcFace loss to transform raw pixel arrays into 512-dimensional hyperspherical vectors for instant cosine distance search in PostgreSQL.
          </p>
        </div>
        <BiometricPipelineVisualizer />
      </section>

      {/* 4. Deterministic Access Decision Matrix */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Access Engine</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Access-Control Decision Logic</h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-3xl">
            IntelliGuard strictly isolates recognition match status from organizational authorization rules, guaranteeing fail-closed facility defense.
          </p>
        </div>
        <DecisionFlowchart />
      </section>

      {/* 5. Comprehensive Technology Stack Grid */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Technologies</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Engineering Technology Stack</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            A production-ready stack chosen for performance, safety, and strict separation of concerns.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Frontend Layer */}
          <Card className="border border-border/80 bg-card">
            <CardHeader className="pb-3 border-b border-border/60">
              <CardTitle className="text-sm font-bold text-foreground flex items-center gap-2">
                <Sliders className="w-4 h-4 text-primary" /> Frontend & UI Layer
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-2 text-xs">
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">Next.js 16 (App Router)</span>
                <p className="text-muted-foreground mt-0.5">Server Components, dynamic API routes, and optimized streaming.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">React 19 & TypeScript</span>
                <p className="text-muted-foreground mt-0.5">Strict compile-time type safety with zero unchecked `any` leakage.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">Tailwind CSS & Shadcn UI</span>
                <p className="text-muted-foreground mt-0.5">Accessible UI primitives built with Radix UI and Lucide icons.</p>
              </div>
            </CardContent>
          </Card>

          {/* AI Inference Layer */}
          <Card className="border border-border/80 bg-card">
            <CardHeader className="pb-3 border-b border-border/60">
              <CardTitle className="text-sm font-bold text-foreground flex items-center gap-2">
                <Cpu className="w-4 h-4 text-amber-500" /> AI & Computer Vision
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-2 text-xs">
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">FastAPI & Python 3.11</span>
                <p className="text-muted-foreground mt-0.5">High-throughput asynchronous ML service with Pydantic validation.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">InsightFace (Buffalo_S)</span>
                <p className="text-muted-foreground mt-0.5">512D ArcFace embeddings with sub-millimeter 5-point alignment.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">ONNX Runtime & OpenCV</span>
                <p className="text-muted-foreground mt-0.5">Optimized hardware execution and image safety bound validation.</p>
              </div>
            </CardContent>
          </Card>

          {/* Database & Storage */}
          <Card className="border border-border/80 bg-card">
            <CardHeader className="pb-3 border-b border-border/60">
              <CardTitle className="text-sm font-bold text-foreground flex items-center gap-2">
                <Database className="w-4 h-4 text-emerald-500" /> Database & Storage
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-2 text-xs">
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">PostgreSQL 16 + pgvector</span>
                <p className="text-muted-foreground mt-0.5">Exact cosine similarity queries (cosine distance &lt;=&gt;) over 512D vector columns.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">Prisma ORM</span>
                <p className="text-muted-foreground mt-0.5">Type-safe relational mapping with atomic transaction support.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">ImageKit Storage</span>
                <p className="text-muted-foreground mt-0.5">Encrypted portrait storage with private authenticated delivery URLs.</p>
              </div>
            </CardContent>
          </Card>

          {/* Security & Authentication */}
          <Card className="border border-border/80 bg-card">
            <CardHeader className="pb-3 border-b border-border/60">
              <CardTitle className="text-sm font-bold text-foreground flex items-center gap-2">
                <Lock className="w-4 h-4 text-rose-500" /> Security & Testing
              </CardTitle>
            </CardHeader>
            <CardContent className="p-4 space-y-2 text-xs">
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">Better-Auth</span>
                <p className="text-muted-foreground mt-0.5">Role-based session authorization for administrative portals.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">Cryptographic Device Auth</span>
                <p className="text-muted-foreground mt-0.5">Constant-time HMAC key checks for physical ESP32 endpoints.</p>
              </div>
              <div className="p-2 rounded bg-muted/40 border border-border">
                <span className="font-semibold text-foreground">Automated Test Suites</span>
                <p className="text-muted-foreground mt-0.5">End-to-end regression tests covering fail-safe rules and P0 security.</p>
              </div>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 6. Database Entity-Relationship Mapping */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Database Schema</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Relational Schema & Biometric Storage</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            Prisma relational structure ensuring referential integrity and foreign-key cascade safety.
          </p>
        </div>

        <Card className="border border-border/80 bg-card p-4 sm:p-6">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 text-xs font-mono">
            <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2">
              <div className="flex items-center justify-between border-b border-border pb-1.5 font-bold text-foreground">
                <span>Model: Person</span>
                <Badge variant="outline" className="text-[10px]">1-to-Many</Badge>
              </div>
              <ul className="text-muted-foreground space-y-1">
                <li>&bull; id: Int (PK)</li>
                <li>&bull; personCode: String (Unique)</li>
                <li>&bull; firstName, lastName: String</li>
                <li>&bull; status: active | inactive | suspended</li>
                <li>&bull; faceImageUrl: String?</li>
                <li>&bull; embeddings: FaceEmbedding[]</li>
              </ul>
            </div>

            <div className="p-3.5 rounded-xl border border-primary/40 bg-primary/5 space-y-2">
              <div className="flex items-center justify-between border-b border-primary/20 pb-1.5 font-bold text-foreground">
                <span>Model: FaceEmbedding</span>
                <Badge className="bg-primary text-primary-foreground text-[10px]">pgvector</Badge>
              </div>
              <ul className="text-muted-foreground space-y-1">
                <li>&bull; id: Int (PK)</li>
                <li>&bull; personId: Int (FK &rarr; Person)</li>
                <li>&bull; embedding: Unsupported("vector(512)")</li>
                <li>&bull; model: String (buffalo_s)</li>
                <li>&bull; qualityScore: Float?</li>
                <li>&bull; isActive: Boolean (Default: true)</li>
              </ul>
            </div>

            <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2">
              <div className="flex items-center justify-between border-b border-border pb-1.5 font-bold text-foreground">
                <span>Model: Device</span>
                <Badge variant="outline" className="text-[10px]">IoT Registry</Badge>
              </div>
              <ul className="text-muted-foreground space-y-1">
                <li>&bull; id: Int (PK)</li>
                <li>&bull; serialNumber: String (Unique)</li>
                <li>&bull; apiKeyHash: String</li>
                <li>&bull; status: online | offline | maintenance</li>
                <li>&bull; lastHeartbeat: DateTime?</li>
                <li>&bull; accessLogs: AccessLog[]</li>
              </ul>
            </div>

            <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2">
              <div className="flex items-center justify-between border-b border-border pb-1.5 font-bold text-foreground">
                <span>Model: AccessLog</span>
                <Badge variant="outline" className="text-[10px]">Audit Trail</Badge>
              </div>
              <ul className="text-muted-foreground space-y-1">
                <li>&bull; id: Int (PK)</li>
                <li>&bull; matchStatus: matched | unknown | ambiguous</li>
                <li>&bull; accessStatus: granted | denied</li>
                <li>&bull; reason: face_match | unauthorized | ...</li>
                <li>&bull; doorAction: unlock | lock | no_action</li>
                <li>&bull; confidenceScore: Float?</li>
              </ul>
            </div>

            <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2">
              <div className="flex items-center justify-between border-b border-border pb-1.5 font-bold text-foreground">
                <span>Model: Alert</span>
                <Badge variant="outline" className="text-[10px]">Security Anomaly</Badge>
              </div>
              <ul className="text-muted-foreground space-y-1">
                <li>&bull; id: Int (PK)</li>
                <li>&bull; alertType: system_error | unknown_face | ...</li>
                <li>&bull; severity: critical | high | medium | low</li>
                <li>&bull; resolved: Boolean</li>
                <li>&bull; resolvedBy: String? (FK &rarr; User)</li>
                <li>&bull; resolvedAt: DateTime?</li>
              </ul>
            </div>

            <div className="p-3.5 rounded-xl border border-border bg-muted/20 space-y-2">
              <div className="flex items-center justify-between border-b border-border pb-1.5 font-bold text-foreground">
                <span>Model: User & Session</span>
                <Badge variant="outline" className="text-[10px]">Better-Auth</Badge>
              </div>
              <ul className="text-muted-foreground space-y-1">
                <li>&bull; id: String (PK)</li>
                <li>&bull; email: String (Unique)</li>
                <li>&bull; role: ADMIN | USER</li>
                <li>&bull; sessions: Session[]</li>
                <li>&bull; resolvedAlerts: Alert[]</li>
              </ul>
            </div>
          </div>
        </Card>
      </section>

      {/* 7. Planned Physical IoT Prototype Architecture */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono border-amber-500/30 text-amber-500">
              Hardware Roadmap
            </Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">
              Planned Physical IoT Hardware Integration
            </h2>
          </div>
          <p className="text-sm text-muted-foreground max-w-3xl">
            In the complete physical prototype, an ESP32-S3 microcontroller will replace browser webcam captures. The hardware architecture specifically isolates capture and user feedback on the edge while centralizing inference on the server.
          </p>
        </div>

        <Card className="border border-amber-500/30 bg-amber-500/5">
          <CardHeader className="pb-3 border-b border-amber-500/20">
            <div className="flex items-center justify-between">
              <CardTitle className="text-base font-bold text-foreground flex items-center gap-2">
                <Radio className="w-4 h-4 text-amber-500" /> Physical Peripheral Blueprint
              </CardTitle>
              <Badge variant="outline" className="text-amber-500 border-amber-500/30 text-xs">
                Status: In Development (Planned Prototype)
              </Badge>
            </div>
          </CardHeader>
          <CardContent className="p-4 sm:p-6 space-y-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
              <div className="p-3 rounded-lg bg-card border border-border space-y-1">
                <span className="text-xs font-bold text-foreground">1. ESP32-S3-CAM + OV5640</span>
                <p className="text-xs text-muted-foreground">
                  Dual-core Xtensa LX7 processor running FreeRTOS with 8MB PSRAM for optical frame buffering.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-card border border-border space-y-1">
                <span className="text-xs font-bold text-foreground">2. HC-SR501 PIR Motion Sensor</span>
                <p className="text-xs text-muted-foreground">
                  Triggers hardware wake-up interrupt on physical approach to conserve power and avoid continuous camera streaming.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-card border border-border space-y-1">
                <span className="text-xs font-bold text-foreground">3. 0.96" I2C OLED Display (SSD1306)</span>
                <p className="text-xs text-muted-foreground">
                  Displays live visitor feedback: "Scanning...", "Welcome, [Name]", or "Access Denied".
                </p>
              </div>
              <div className="p-3 rounded-lg bg-card border border-border space-y-1">
                <span className="text-xs font-bold text-foreground">4. KY-016 RGB LED</span>
                <p className="text-xs text-muted-foreground">
                  Instant visual signaling: Green (Access Granted), Red (Access Denied / Alert), Blue (System Scanning).
                </p>
              </div>
              <div className="p-3 rounded-lg bg-card border border-border space-y-1">
                <span className="text-xs font-bold text-foreground">5. KY-012 Active Buzzer</span>
                <p className="text-xs text-muted-foreground">
                  Audible confirmation chime on door unlock and rapid pulsating security alarm tone on unauthorized attempts.
                </p>
              </div>
              <div className="p-3 rounded-lg bg-card border border-border space-y-1">
                <span className="text-xs font-bold text-foreground">6. Breadboard & Jumper Harness</span>
                <p className="text-xs text-muted-foreground">
                  Clean 3.3V / 5V power regulation and shared I2C bus wiring designed for laboratory defense demonstration.
                </p>
              </div>
            </div>

            <div className="p-3 bg-card rounded-lg border border-border/80 text-xs text-muted-foreground">
              <strong className="text-foreground">Architectural Design Choice:</strong> The ESP32-S3 is deliberately not tasked with onboard neural network face inference. By offloading image bytes to the FastAPI/InsightFace service, IntelliGuard maintains 100% full-resolution ArcFace accuracy without microcontroller memory exhaustion.
            </div>
          </CardContent>
        </Card>
      </section>

      {/* 8. Implementation Status & Engineering Roadmap */}
      <section className="space-y-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Badge variant="outline" className="text-xs uppercase font-mono">Milestones</Badge>
            <h2 className="text-2xl font-bold tracking-tight text-foreground">Engineering Implementation Status</h2>
          </div>
          <p className="text-sm text-muted-foreground">
            Audit of verified codebase capabilities versus planned academic deliverables.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <Card className="border border-emerald-500/30 bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-2">
                <CheckCircle2 className="w-4 h-4" /> Fully Implemented
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs text-muted-foreground">
              <p>&bull; Next.js 16 Web Dashboard & Navigation</p>
              <p>&bull; FastAPI InsightFace 512D ArcFace Service</p>
              <p>&bull; PostgreSQL + pgvector Cosine Search</p>
              <p>&bull; Biometric Person Registration Flow</p>
              <p>&bull; Duplicate Face Registration Prevention</p>
              <p>&bull; Active, Inactive & Suspended Status Checks</p>
              <p>&bull; Fail-Safe Lock Error Handling</p>
              <p>&bull; Security Anomaly Alerts & Batch Resolve</p>
              <p>&bull; Webcam Simulation with Audio HUD</p>
              <p>&bull; Automated End-to-End Test Suites</p>
            </CardContent>
          </Card>

          <Card className="border border-amber-500/30 bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-bold text-amber-500 flex items-center gap-2">
                <Activity className="w-4 h-4" /> In Development
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs text-muted-foreground">
              <p>&bull; ESP32-S3-CAM Firmware Client</p>
              <p>&bull; Hardware WiFi REST Connection Script</p>
              <p>&bull; Device API Key Provisioning UI</p>
              <p>&bull; Real-time Heartbeat Telemetry Logging</p>
              <p>&bull; Offline Device Anomaly Detection</p>
            </CardContent>
          </Card>

          <Card className="border border-border/80 bg-card">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm font-bold text-muted-foreground flex items-center gap-2">
                <Layers className="w-4 h-4" /> Planned Improvements
              </CardTitle>
            </CardHeader>
            <CardContent className="space-y-2 text-xs text-muted-foreground">
              <p>&bull; Physical Breadboard Hardware Prototype</p>
              <p>&bull; Physical Sensor-backed Liveness Detection</p>
              <p>&bull; Variable Cosine Threshold Testing Matrix</p>
              <p>&bull; Push Notification Alerts via WebSockets</p>
              <p>&bull; Door Access Schedule Time Restriction Rules</p>
            </CardContent>
          </Card>
        </div>
      </section>

      {/* 9. Presentation Footer Bar */}
      <Card className="border border-primary/30 bg-primary/5">
        <CardContent className="p-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center md:text-left">
            <h4 className="text-base font-bold text-foreground">Project Defense & Demonstration Ready</h4>
            <p className="text-xs text-muted-foreground">
              IntelliGuard integrates modern web engineering, deep learning biometrics, and embedded IoT systems for academic presentation.
            </p>
          </div>
          <div className="flex items-center gap-3">
            <Button asChild className="bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/dashboard/simulation">
                <Camera className="w-4 h-4 mr-1.5" /> Start Live Defense Demo
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
