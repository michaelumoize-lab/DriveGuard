"use client";

import React, { useState, useMemo } from "react";
import Link from "next/link";
import {
  Search,
  Sparkles,
  Server,
  Camera,
  ExternalLink,
  CheckCircle2,
  AlertTriangle,
  ShieldCheck,
  ShieldAlert,
  Activity,
  Cpu,
  Layers,
  ArrowRight,
  HelpCircle,
  Eye,
  Sliders,
  Terminal,
  Zap,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface HelpTopic {
  id: string;
  category: "getting-started" | "calibration" | "monitoring" | "ai-cascade" | "troubleshooting";
  title: string;
  summary: string;
  badge?: string;
  content: React.ReactNode;
}

export default function HelpPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState<string>("all");

  const topics: HelpTopic[] = [
    {
      id: "what-is-driveguard",
      category: "getting-started",
      title: "What is DriveGuard?",
      summary: "Overview of DriveGuard's Personalised Adaptive AI Driver Drowsiness Early Warning System.",
      badge: "Core Architecture",
      content: (
        <div className="space-y-4 text-sm text-muted-foreground leading-relaxed">
          <p>
            <strong className="text-foreground">DriveGuard</strong> is a final-year Computer Engineering project
            delivering a personalized, adaptive driver fatigue monitoring and early warning system.
          </p>
          <p>
            Unlike conventional systems that rely on a single, brittle static threshold (such as EAR &lt; 0.22),
            DriveGuard tailors drowsiness detection to each driver&apos;s individual facial geometry, spatial head pose,
            and natural blink dynamics across 4 distinct cabin zones.
          </p>
          <div className="p-3 rounded-xl border border-primary/20 bg-primary/5 space-y-2">
            <div className="font-semibold text-foreground text-xs uppercase tracking-wider flex items-center gap-2">
              <Zap className="size-3.5 text-primary" /> Core Subsystems
            </div>
            <ul className="list-disc pl-5 space-y-1 text-xs">
              <li><strong>MediaPipe Tasks 478 3D Mesh:</strong> Real-time 6DoF head pose estimation and iris tracking.</li>
              <li><strong>4-Zone Pose-Conditioned Baselines:</strong> CENTER, LEFT MIRROR, RIGHT MIRROR, DASHBOARD.</li>
              <li><strong>Dual-Engine Geometric-Authoritative AI:</strong> Independent Browser WASM and Local/Remote Python MediaPipe + PDBP pipelines (passive neural shadow observer disabled in production).</li>
              <li><strong>Sub-Frame Blink Kinematics:</strong> Duration, Reopening Velocity (RCR), and rolling 60s PERCLOS.</li>
              <li><strong>Explainable Hysteresis FSM:</strong> NORMAL, WARNING, CRITICAL states with human-readable reason codes.</li>
            </ul>
          </div>
        </div>
      ),
    },
    {
      id: "calibration-guide",
      category: "calibration",
      title: "Streamlined 15-Second 4-Zone Calibration",
      summary: "How to establish your personal behavioral baselines across all 4 spatial zones with zero blinking delay.",
      badge: "Calibration",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            Head pose changes facial perspective. When glancing at a side mirror or dashboard, the Eye Aspect Ratio (EAR)
            naturally narrows even when the driver is fully awake. Conventional systems produce false alarms during these glances.
          </p>
          <div className="p-3 rounded-xl bg-emerald-500/5 border border-emerald-500/20 text-xs">
            <strong className="text-foreground block mb-1">Streamlined Calibration Optimization:</strong>
            The redundant 2-minute blinking test has been removed. Research confirms that staring at a camera for 2 minutes causes conscious observer bias (unnatural blinking). Natural blink duration (235 ms) and reopening velocity ratio (RCR = 0.95) are seeded from established clinical norms and continuously adapted online via our dual-anchor drift filter.
          </div>
          <p>
            DriveGuard&apos;s guided calibration captures your unique baseline across 4 zones in just ~15 seconds:
          </p>
          <ol className="list-decimal pl-5 space-y-1 text-xs">
            <li><strong>CENTER (4.0 s):</strong> Looking straight ahead at the roadway. Establishes your primary anchor baseline (B₀) and normal median EAR.</li>
            <li><strong>LEFT MIRROR (3.0 s):</strong> Glancing toward your driver-side mirror (Yaw ≈ +24°).</li>
            <li><strong>RIGHT MIRROR (3.0 s):</strong> Glancing toward your passenger-side mirror (Yaw ≈ -24°).</li>
            <li><strong>DASHBOARD (3.0 s):</strong> Glancing down toward the instrument cluster (Pitch ≈ -14°).</li>
          </ol>
          <p className="text-xs">
            The calibration wizard uses real-time <strong>target-acquisition spatial gating</strong>: samples are only gathered
            when your head orientation is properly aligned with the active target zone!
          </p>
        </div>
      ),
    },
    {
      id: "camera-and-positioning",
      category: "getting-started",
      title: "Camera Mounting & Optical Diagnostics",
      summary: "Guidelines for in-cabin phone mounting, reticle alignment, and optical environment badges.",
      badge: "Camera & Optics",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            DriveGuard is designed for real automotive cockpits with obliquely mounted smartphone cameras (e.g., windshield suction mount, air vent clip, or dashboard mount).
          </p>
          <div className="space-y-2 text-xs">
            <div>
              <strong className="text-foreground">Mounting &amp; Framing Rule:</strong> Look naturally forward down the road, NOT directly at the phone lens. The camera should capture your head with a face-height ratio between 18% and 50% of the frame.
            </div>
            <div>
              <strong className="text-foreground">Optical Quality Monitoring:</strong> The camera feed continuously samples mean luminance (40–210) to flag dark cabins or glare, and computes the Laplacian edge variance (≥ 75) to detect lens defocus or motion blur.
            </div>
            <div>
              <strong className="text-foreground">Screen Wake Lock:</strong> While monitoring is active, the browser automatically acquires a Screen Wake Lock to prevent your phone screen from dimming or locking while driving.
            </div>
          </div>
        </div>
      ),
    },
    {
      id: "preflight-interlock-gate",
      category: "monitoring",
      title: "Automotive Pre-Flight Readiness Interlock Gate",
      summary: "Why monitoring sessions require a 5-point subsystem readiness handshake before recording begins.",
      badge: "Safety Architecture",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            In accordance with ISO 26262 functional safety, DriveGuard will not begin recording a driving session until all 5 critical subsystems pass their readiness handshake:
          </p>
          <ol className="list-decimal pl-5 space-y-1 text-xs">
            <li><strong>Camera Feed:</strong> Optical video stream is active, negotiated, and not severely degraded.</li>
            <li><strong>MediaPipe WASM / Vision Runtime:</strong> 478-landmark 3D face mesh perception engine is loaded in memory and primed.</li>
            <li><strong>Geometric &amp; Temporal AI Pipeline:</strong> Calibrated 4-zone PDBP baselines, head pose estimator, and temporal PERCLOS/blink kinematics are verified in runtime memory.</li>
            <li><strong>Persistence &amp; Backend Sync:</strong> Checks API/database connectivity; in Browser AI mode, sessions remain operational offline via the local <code className="font-mono">OFFLINE_QUEUED</code> store and sync automatically once reconnected.</li>
            <li><strong>Audio Synthesizer:</strong> Web Audio context is initialized for acoustic warning sweeps.</li>
          </ol>
          <p className="text-xs">
            This prevents orphaned sessions from polluting the database before computer vision and backend telemetry links are fully operational.
          </p>
        </div>
      ),
    },
    {
      id: "academic-suite-guide",
      category: "ai-cascade",
      title: "Academic Research Suite & Viva Voce Guide",
      summary: "How to use the Ablation Lab, Hardware Co-Processor console, and Defense Companion.",
      badge: "Academic Suite",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            DriveGuard includes dedicated academic and engineering portals built for external examiner evaluations and capstone defenses—kept strictly separate from live production monitoring:
          </p>
          <ul className="list-disc pl-5 space-y-1.5 text-xs">
            <li>
              <strong>Ablation Benchmarks (`/dashboard/benchmarks`):</strong> Displays offline historical quantitative evaluation tables for RQ1/RQ2/RQ3, false alarm reductions, and an interactive sensitivity slider with one-click LaTeX table exports for your thesis.
            </li>
            <li>
              <strong>Synthetic Testbench (`/dashboard/synthetic-testbench`):</strong> Isolated in-memory fault-injection sandbox for previewing mirror glance, sluggish reopen, glare defocus, and watchdog scenarios without touching live sessions or the database.
            </li>
            <li>
              <strong>Simulated Hardware Co-Processor (`/dashboard/hardware`):</strong> Browser-simulated protocol console demonstrating 8-byte packed binary UART frames, DRV2605L haptic actuator pulses, SSD1306 OLED HUD emulation, and INA219 power profiles (no physical MCU required).
            </li>
            <li>
              <strong>Defense Companion (`/dashboard/defense-guide`):</strong> Provides detailed academic answers to the 5 classic examiner questions (why not 3D CNNs, handling narrow eye shapes, dataset leakage prevention).
            </li>
          </ul>
        </div>
      ),
    },
    {
      id: "ai-cascade-modes",
      category: "ai-cascade",
      title: "Offline Research Cascade Modes (Exp A, B, C)",
      summary: "Understand the three experimental ablation modes evaluated in the thesis research study.",
      badge: "Research Ablations",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            In our offline ablation study, we evaluated three experimental arbitration modes to measure CPU power efficiency versus detection accuracy. <strong>Live production monitoring uses Geometric Authoritative mode</strong> (4-zone PDBP + temporal kinematics, with passive shadow disabled):
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div className="p-3 rounded-lg border border-primary/30 bg-primary/5">
              <div className="font-semibold text-primary text-xs">Exp A: Geometric Only (Production Basis)</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Deterministic MediaPipe 478-landmark 3D geometry, 4-zone PDBP, and temporal kinematics. Lowest CPU footprint (~2.5 ms).
              </p>
            </div>
            <div className="p-3 rounded-lg border border-border bg-card">
              <div className="font-semibold text-foreground text-xs">Exp B: CNN Always (Offline Research)</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Runs MobileNetV3-Small on every frame during offline dataset evaluation. Higher compute cost on edge CPUs.
              </p>
            </div>
            <div className="p-3 rounded-lg border border-border bg-card">
              <div className="font-semibold text-foreground text-xs">Exp C: Adaptive Cascade (Offline Research)</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Evaluates geometric analysis first and gates MobileNetV3 on ambiguous frames, reducing neural invocations by 61.1% in offline benchmarks.
              </p>
            </div>
          </div>
        </div>
      ),
    },
    {
      id: "metrics-explained",
      category: "monitoring",
      title: "Understanding Telemetry Metrics (EAR & PERCLOS)",
      summary: "Definitions of Eye Aspect Ratio, PERCLOS, Blink Rate, and Fatigue Risk Score.",
      badge: "Diagnostics",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <div className="space-y-2 text-xs">
            <div>
              <strong className="text-foreground">Eye Aspect Ratio (EAR):</strong> Ratio of vertical eyelid distances to horizontal eye width. Fully open eyes typically produce an EAR of 0.28 - 0.35; closed eyes drop below 0.18.
            </div>
            <div>
              <strong className="text-foreground">PERCLOS (Percentage of Eye Closure):</strong> The gold-standard metric for driver drowsiness, measured over a rolling 60-second window. Values above 15% indicate drowsiness; values above 25% indicate severe fatigue.
            </div>
            <div>
              <strong className="text-foreground">Blink Duration:</strong> Normal blinks last 150 - 250 ms. Micro-sleeps or drowsy slow eyelid closures exceed 400 - 800 ms.
            </div>
            <div>
              <strong className="text-foreground">Composite Risk Score:</strong> A fused score between 0.0 and 1.0 combining PERCLOS, prolonged closure, head nodding pitch, and blink kinematics.
            </div>
          </div>
        </div>
      ),
    },
    {
      id: "backend-troubleshooting",
      category: "troubleshooting",
      title: "Connecting Local vs Cloud AI Backend",
      summary: "Resolve connection issues between Next.js and the FastAPI computer vision engine.",
      badge: "Troubleshooting",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            DriveGuard supports independent <strong>Browser AI (WASM)</strong>, <strong>Local Python AI (OpenCV/MediaPipe)</strong>, and <strong>Remote Cloud Python</strong> modes:
          </p>
          <ul className="list-disc pl-5 space-y-1.5 text-xs">
            <li>
              <strong>Browser AI (Default, Zero-Backend Required):</strong> Runs MediaPipe Tasks WASM directly in your browser—works even completely offline without starting Python.
            </li>
            <li>
              <strong>Local Python Engine (Recommended for Native OpenCV Evaluation):</strong> Start the FastAPI engine locally from the repository root with local camera access enabled:
              <pre className="mt-1 p-2 rounded bg-muted font-mono text-xs overflow-x-auto">LOCAL_CAMERA_API_ENABLED=true python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000</pre>
              <span className="block mt-1 text-[11px]">
                On Windows PowerShell: <code className="font-mono">$env:LOCAL_CAMERA_API_ENABLED=&quot;true&quot;; python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000</code>
              </span>
              Then select <strong>Python AI (Local)</strong> in the Live Monitor engine selector.
            </li>
            <li>
              <strong>Cloud Engine (Render + LiveKit Cloud SFU):</strong> Render free-tier instances spin down after 15 minutes of inactivity. When connecting remotely for the first time, allow 30–60 seconds for the cloud worker to wake up, or stay on Browser AI for instant local monitoring.
            </li>
          </ul>
        </div>
      ),
    },
    {
      id: "privacy-and-safety-disclaimer",
      category: "getting-started",
      title: "Privacy Architecture & Safety Prototype Disclaimer",
      summary: "How camera frames are processed in memory, what telemetry is stored, and prototype safety notices.",
      badge: "Privacy & Safety",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <div className="space-y-2 text-xs">
            <div>
              <strong className="text-foreground">In-Memory Ephemeral Video Processing:</strong> WebRTC video frames streamed via LiveKit are processed strictly in volatile RAM by the Python AI worker (`maxsize=1` frame buffer) to extract 478 facial landmarks and eye crops. Raw video recordings and camera images are never written to disk or stored in PostgreSQL.
            </div>
            <div>
              <strong className="text-foreground">Stored Telemetry & User Control:</strong> Only numeric 4-zone calibration baselines (`DriverProfile`), aggregate session statistics (`DrivingSession`), and state-transition safety alert metadata (`DrowsinessAlert`) are stored in Neon PostgreSQL under your authenticated account.
            </div>
            <div className="p-3 rounded-lg border border-amber-500/30 bg-amber-500/10 text-amber-600 dark:text-amber-300">
              <strong>Important Safety Disclaimer:</strong> DriveGuard is an academic computer engineering research platform and prototype driver monitoring system. It is not a certified automotive ADAS medical or safety device (e.g., ISO 26262 / Euro NCAP). Never rely on DriveGuard to continue driving while fatigued.
            </div>
          </div>
        </div>
      ),
    },
  ];

  const filteredTopics = useMemo(() => {
    return topics.filter((t) => {
      const matchCat = selectedCategory === "all" || t.category === selectedCategory;
      const matchSearch =
        searchQuery.trim() === "" ||
        t.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        t.summary.toLowerCase().includes(searchQuery.toLowerCase());
      return matchCat && matchSearch;
    });
  }, [topics, selectedCategory, searchQuery]);

  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-5xl mx-auto">
      {/* Header */}
      <div className="space-y-2">
        <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full text-xs font-semibold bg-primary/10 text-primary border border-primary/20">
          <HelpCircle className="size-3.5" />
          <span>DriveGuard Help & Documentation</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
          System Guide & Knowledge Base
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground max-w-2xl">
          Everything you need to know about DriveGuard&apos;s computer vision pipeline, 4-zone calibration, AI cascades, and real-time telemetry.
        </p>
      </div>

      {/* Search Input */}
      <div className="relative">
        <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 size-4 text-muted-foreground" />
        <Input
          type="text"
          placeholder="Search topics, metrics, or troubleshooting guides..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          className="pl-10 h-11 bg-card/60 backdrop-blur-sm"
        />
      </div>

      {/* Category Tabs */}
      <Tabs value={selectedCategory} onValueChange={setSelectedCategory} className="w-full">
        <TabsList className="grid grid-cols-2 sm:grid-cols-5 h-auto p-1 gap-1">
          <TabsTrigger value="all" className="text-xs py-1.5">All Topics</TabsTrigger>
          <TabsTrigger value="getting-started" className="text-xs py-1.5">Overview</TabsTrigger>
          <TabsTrigger value="calibration" className="text-xs py-1.5">Calibration</TabsTrigger>
          <TabsTrigger value="monitoring" className="text-xs py-1.5">Metrics</TabsTrigger>
          <TabsTrigger value="troubleshooting" className="text-xs py-1.5">Support</TabsTrigger>
        </TabsList>
      </Tabs>

      {/* Topics Accordion */}
      <div className="space-y-4">
        {filteredTopics.length === 0 ? (
          <Card className="p-8 text-center text-muted-foreground">
            <HelpCircle className="size-8 mx-auto mb-2 text-muted-foreground/60" />
            <p className="text-sm font-medium">No matching documentation topics found.</p>
            <p className="text-xs mt-1">Try searching for &quot;calibration&quot;, &quot;EAR&quot;, or &quot;cascade&quot;.</p>
          </Card>
        ) : (
          <Accordion type="single" collapsible defaultValue={filteredTopics[0]?.id} className="space-y-3">
            {filteredTopics.map((topic) => (
              <AccordionItem
                key={topic.id}
                value={topic.id}
                className="border border-border/80 rounded-xl px-4 py-1 bg-card/60 backdrop-blur-xs shadow-xs"
              >
                <AccordionTrigger className="hover:no-underline py-3">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-left w-full pr-4">
                    <span className="font-semibold text-foreground text-sm">{topic.title}</span>
                    {topic.badge && (
                      <Badge variant="outline" className="text-[10px] self-start sm:self-auto font-mono">
                        {topic.badge}
                      </Badge>
                    )}
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pt-2 pb-4 border-t border-border/40">
                  {topic.content}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </div>

      {/* Quick API / Live Links Card */}
      <Card className="border-border/80 bg-muted/20">
        <CardContent className="p-4 flex flex-col sm:flex-row items-center justify-between gap-4 text-xs">
          <div className="flex items-center gap-3">
            <div className="p-2.5 rounded-xl bg-primary/10 text-primary">
              <Terminal className="size-5" />
            </div>
            <div>
              <div className="font-semibold text-foreground">Interactive FastAPI Swagger Documentation</div>
              <div className="text-muted-foreground">Inspect raw DTO schemas, REST endpoints, and WebSocket specifications.</div>
            </div>
          </div>
          <Button asChild size="sm" variant="outline" className="h-8 gap-1.5 shrink-0">
            <a href="https://driveguard-x048.onrender.com/docs" target="_blank" rel="noopener noreferrer">
              Open /docs <ExternalLink className="size-3.5" />
            </a>
          </Button>
        </CardContent>
      </Card>
    </div>
  );
}
