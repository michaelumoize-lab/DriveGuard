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
              <li><strong>Confidence-Gated Cascade:</strong> Fast geometric analysis with on-demand MobileNetV3-Small ONNX verification.</li>
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
      title: "Guided 4-Zone Calibration",
      summary: "How to establish your personal behavioral baselines across all 4 spatial zones.",
      badge: "Calibration",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            Head pose changes facial perspective. When glancing at a side mirror or dashboard, the Eye Aspect Ratio (EAR)
            naturally narrows even when the driver is fully awake. Conventional systems produce false alarms during these glances.
          </p>
          <p>
            DriveGuard&apos;s guided calibration captures your unique baseline across 4 zones:
          </p>
          <ol className="list-decimal pl-5 space-y-1 text-xs">
            <li><strong>CENTER:</strong> Looking straight ahead at the roadway. Establishes your primary anchor baseline ($B_0$).</li>
            <li><strong>LEFT MIRROR:</strong> Looking toward your driver-side mirror.</li>
            <li><strong>RIGHT MIRROR:</strong> Looking toward your passenger-side mirror.</li>
            <li><strong>DASHBOARD:</strong> Looking slightly down toward the instrument cluster.</li>
          </ol>
          <p className="text-xs">
            The calibration wizard uses real-time <strong>target-acquisition spatial gating</strong>: samples are only gathered
            when your head orientation is properly aligned with the active target zone!
          </p>
        </div>
      ),
    },
    {
      id: "ai-cascade-modes",
      category: "ai-cascade",
      title: "Confidence-Gated Cascade (Exp A, B, C)",
      summary: "Understand the three experimental cascade modes evaluated in the research study.",
      badge: "Inference Modes",
      content: (
        <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
          <p>
            DriveGuard implements three distinct arbitration modes to balance CPU power efficiency and detection accuracy:
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 pt-1">
            <div className="p-3 rounded-lg border border-border bg-card">
              <div className="font-semibold text-foreground text-xs">Exp A: Geometric Only</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Fast first stage using MediaPipe landmarks and EAR calculation. Lowest CPU footprint (~2.5ms).
              </p>
            </div>
            <div className="p-3 rounded-lg border border-border bg-card">
              <div className="font-semibold text-foreground text-xs">Exp B: CNN Always</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Runs MobileNetV3-Small ONNX on every frame. High accuracy under sunglasses/shadows, higher compute cost.
              </p>
            </div>
            <div className="p-3 rounded-lg border border-primary/30 bg-primary/5">
              <div className="font-semibold text-primary text-xs">Exp C: Adaptive Cascade</div>
              <p className="text-[11px] text-muted-foreground mt-1">
                Runs geometric analysis first; invokes MobileNetV3 only on low confidence or borderline closure. Saves ~67% CPU power!
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
            DriveGuard supports both <strong>Local</strong> and <strong>Cloud</strong> inference backends:
          </p>
          <ul className="list-disc pl-5 space-y-1.5 text-xs">
            <li>
              <strong>Local Engine (Recommended for Low-Latency Evaluation):</strong> Start the backend locally with:
              <pre className="mt-1 p-2 rounded bg-muted font-mono text-xs">python run.py --port 8000</pre>
              Select <strong>Local</strong> in the Live Monitor top toolbar.
            </li>
            <li>
              <strong>Cloud Engine (Render + LiveKit Cloud SFU):</strong> Render free-tier instances spin down after 15 minutes of inactivity. When connecting for the first time, allow 30–60 seconds for the cloud worker to wake up, or use the built-in Connection Recovery Assistant.
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
