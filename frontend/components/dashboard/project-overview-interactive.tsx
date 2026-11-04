"use client";

import React, { useState } from "react";
import {
  Server,
  Cpu,
  Database,
  Camera,
  ShieldCheck,
  ShieldAlert,
  Sliders,
  ArrowRight,
  CheckCircle2,
  AlertTriangle,
  Lock,
  Layers,
  Sparkles,
  Zap,
  Radio,
  Share2,
  Terminal,
  Activity,
  HardDrive,
  UserCheck,
  Eye,
  Info,
  ChevronRight,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface ArchNode {
  id: string;
  name: string;
  category: "Client / Edge" | "Application Layer" | "AI Inference" | "Database & Storage" | "Access Decision";
  technology: string;
  role: string;
  details: string[];
  communication: string;
  status: "Implemented" | "In Development" | "Planned";
}

const ARCHITECTURE_NODES: Record<string, ArchNode> = {
  "webcam-client": {
    id: "webcam-client",
    name: "Webcam Simulation Client",
    category: "Client / Edge",
    technology: "WebRTC / HTML5 Canvas / Next.js",
    role: "Simulates optical video capture for local software testing and enrollment.",
    details: [
      "Captures raw video frames at 3/4 aspect ratio",
      "Executes in-flight request throttling to prevent API congestion",
      "Renders biometric alignment oval with real-time HUD overlays",
      "Does not transmit device serial/key headers (treats request as user-facing simulation)",
    ],
    communication: "POST /api/access with base64 image payload over HTTPS/HTTP",
    status: "Implemented",
  },
  "esp32-cam": {
    id: "esp32-cam",
    name: "ESP32-S3-CAM Edge Node",
    category: "Client / Edge",
    technology: "ESP32-S3 / OV5640 / FreeRTOS",
    role: "Physical IoT edge device responsible for motion detection, image capture, and peripheral feedback.",
    details: [
      "Signals image capture on HC-SR501 PIR motion sensor interrupt",
      "Captures JPEG frame via OV5640 5MP camera sensor",
      "Authenticates via x-device-serial and x-device-key custom headers",
      "Receives unlock/lock instruction and drives 0.96-inch OLED, RGB LED, and buzzer",
      "Reports periodic WiFi, heap, and RSSI telemetry heartbeats",
    ],
    communication: "Authenticated REST over WiFi (HTTP/HTTPS) to Next.js API gateway",
    status: "In Development",
  },
  "nextjs-gateway": {
    id: "nextjs-gateway",
    name: "Next.js Application Gateway",
    category: "Application Layer",
    technology: "Next.js 16 (App Router) / Node.js 22 / TypeScript",
    role: "Central administrative hub, device authenticator, and request orchestrator.",
    details: [
      "Strict dual authentication: Better-Auth web sessions vs. device API keys",
      "Fail-safe device isolation: Malformed device headers return 401 without session fallback",
      "Forwards captured frame to FastAPI AI inference service under strict timeout bounds",
      "Admin dashboard management: Person profiles, device registry, and audit logs",
    ],
    communication: "HTTP proxy to FastAPI (:8000), Prisma ORM to PostgreSQL (:5432)",
    status: "Implemented",
  },
  "fastapi-ai": {
    id: "fastapi-ai",
    name: "FastAPI AI Biometric Service",
    category: "AI Inference",
    technology: "FastAPI / Python 3.11 / InsightFace / ONNX Runtime / OpenCV",
    role: "Dedicated machine learning microservice for facial detection, quality gating, and ArcFace feature extraction.",
    details: [
      "Image validation: Decompression bomb guards and OpenCV resolution bounds",
      "Face detection & landmark alignment via RetinaFace/Buffalo_S ONNX model",
      "Biometric quality gate: Evaluates pose, blur, and lighting before vector extraction",
      "Feature extraction: Generates normalized 512-dimensional ArcFace embedding float array",
      "Concurrency control: Inference semaphore protects GPU/CPU from resource exhaustion",
    ],
    communication: "JSON REST API (:8000) with Next.js application backend",
    status: "Implemented",
  },
  "pgvector-db": {
    id: "pgvector-db",
    name: "PostgreSQL & pgvector Engine",
    category: "Database & Storage",
    technology: "PostgreSQL 16 / pgvector extension / Prisma ORM",
    role: "Relational persistence store and high-dimensional vector similarity search engine.",
    details: [
      "Stores registered persons, face embeddings, IoT devices, logs, and security alerts",
      "Executes exact cosine distance vector similarity queries (embedding <=> target_vector)",
      "Strict foreign key cascading and relational audit integrity",
      "Guarantees that raw 512D float arrays remain securely encapsulated in database columns",
    ],
    communication: "PostgreSQL wire protocol via Prisma Client with pooling",
    status: "Implemented",
  },
  "decision-engine": {
    id: "decision-engine",
    name: "Access Control Decision Engine",
    category: "Access Decision",
    technology: "TypeScript / Deterministic Rule Engine",
    role: "Evaluates biometric similarity matches against organizational access policies.",
    details: [
      "Confidence thresholding: Requires >=0.50 (50%) cosine similarity for face_match",
      "Ambiguity guard: If top 2 candidates are within 0.05 similarity, marks ambiguous and denies access",
      "Personnel state evaluation: Denies access with unauthorized for inactive or suspended users",
      "Fail-Safe default: Network timeouts, 500s, or DB errors force door_action: lock and alert creation",
    ],
    communication: "Internal functional pipeline within Next.js API runtime",
    status: "Implemented",
  },
};

export function InteractiveArchitectureViewer() {
  const [selectedNodeId, setSelectedNodeId] = useState<string>("nextjs-gateway");
  const selectedNode = ARCHITECTURE_NODES[selectedNodeId] || ARCHITECTURE_NODES["nextjs-gateway"];

  return (
    <div className="space-y-6">
      {/* Visual System Flow Graphic */}
      <Card className="border border-border/80 bg-gradient-to-b from-card/80 to-card/40 backdrop-blur-md overflow-hidden shadow-sm">
        <CardHeader className="pb-3 border-b border-border/60">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div>
              <CardTitle className="text-base font-bold text-foreground flex items-center gap-2">
                <Share2 className="w-4 h-4 text-primary" /> End-to-End System Topology
              </CardTitle>
              <CardDescription className="text-xs">
                Click any component node below to inspect its technical role, protocols, and implementation status.
              </CardDescription>
            </div>
            <div className="flex items-center gap-2">
              <span className="flex items-center gap-1.5 text-[11px] font-mono text-muted-foreground">
                <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
                Live Architecture Map
              </span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-4 sm:p-6">
          {/* Architecture Node Grid */}
          <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-6 gap-3">
            {Object.values(ARCHITECTURE_NODES).map((node) => {
              const isSelected = selectedNodeId === node.id;
              return (
                <button
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  className={`p-3.5 rounded-xl border text-left transition-all cursor-pointer relative flex flex-col justify-between min-h-[130px] ${
                    isSelected
                      ? "border-primary bg-primary/10 shadow-[0_0_15px_rgba(var(--primary),0.15)] ring-1 ring-primary/40"
                      : "border-border/70 bg-card hover:bg-muted/40 hover:border-muted-foreground/30"
                  }`}
                >
                  <div>
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] uppercase font-mono tracking-wider text-muted-foreground">
                        {node.category.split(" ")[0]}
                      </span>
                      <span
                        className={`w-2 h-2 rounded-full ${
                          node.status === "Implemented"
                            ? "bg-emerald-500"
                            : node.status === "In Development"
                            ? "bg-amber-500 animate-pulse"
                            : "bg-slate-400"
                        }`}
                      />
                    </div>
                    <h4 className="text-xs font-bold text-foreground leading-tight">{node.name}</h4>
                  </div>
                  <div className="pt-2 border-t border-border/50">
                    <p className="text-[10px] font-mono text-muted-foreground truncate">{node.technology}</p>
                  </div>
                </button>
              );
            })}
          </div>

          {/* Flow Direction Indicator */}
          <div className="hidden lg:flex items-center justify-between px-6 pt-4 text-muted-foreground/60">
            <span className="text-[10px] font-mono flex items-center gap-1">
              Capture <ChevronRight className="w-3 h-3" />
            </span>
            <span className="text-[10px] font-mono flex items-center gap-1">
              Gateway <ChevronRight className="w-3 h-3" />
            </span>
            <span className="text-[10px] font-mono flex items-center gap-1">
              Inference <ChevronRight className="w-3 h-3" />
            </span>
            <span className="text-[10px] font-mono flex items-center gap-1">
              Vector Search <ChevronRight className="w-3 h-3" />
            </span>
            <span className="text-[10px] font-mono flex items-center gap-1">
              Decision <ChevronRight className="w-3 h-3" />
            </span>
            <span className="text-[10px] font-mono">Actuator Action</span>
          </div>
        </CardContent>
      </Card>

      {/* Selected Node Technical Deep-Dive Inspector */}
      <Card className="border border-border shadow-sm bg-card">
        <CardHeader className="pb-3 border-b border-border/60">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2">
                <Badge variant="outline" className="text-xs font-semibold px-2.5 py-0.5">
                  {selectedNode.category}
                </Badge>
                <Badge
                  variant={
                    selectedNode.status === "Implemented"
                      ? "default"
                      : selectedNode.status === "In Development"
                      ? "secondary"
                      : "outline"
                  }
                  className="text-xs font-semibold"
                >
                  {selectedNode.status}
                </Badge>
              </div>
              <h3 className="text-lg font-bold text-foreground tracking-tight">{selectedNode.name}</h3>
            </div>
            <div className="text-xs font-mono text-muted-foreground bg-muted/60 px-3 py-1.5 rounded-lg border border-border">
              Technology: <span className="text-foreground font-semibold">{selectedNode.technology}</span>
            </div>
          </div>
        </CardHeader>
        <CardContent className="p-4 sm:p-6 space-y-4">
          <div>
            <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-1.5">
              Core Architectural Responsibility
            </h4>
            <p className="text-sm text-foreground font-medium">{selectedNode.role}</p>
          </div>

          <div>
            <h4 className="text-xs font-semibold uppercase text-muted-foreground tracking-wider mb-2">
              Implementation Details & Guarantees
            </h4>
            <ul className="grid grid-cols-1 md:grid-cols-2 gap-2">
              {selectedNode.details.map((detail, idx) => (
                <li
                  key={idx}
                  className="p-2.5 rounded-lg border border-border/70 bg-muted/30 text-xs text-muted-foreground flex items-start gap-2"
                >
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500 shrink-0 mt-0.5" />
                  <span className="leading-snug">{detail}</span>
                </li>
              ))}
            </ul>
          </div>

          <div className="pt-2 border-t border-border/60 flex flex-col sm:flex-row sm:items-center justify-between gap-2">
            <div className="flex items-center gap-2 text-xs text-muted-foreground">
              <Radio className="w-3.5 h-3.5 text-primary" />
              <span>
                Communication Channel: <code className="text-xs font-mono text-foreground">{selectedNode.communication}</code>
              </span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}

export function BiometricPipelineVisualizer() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: "1. Capture & Normalization",
      subtitle: "Webcam / ESP32-S3 OV5640",
      description: "Image captured in standard 3/4 aspect ratio and encoded as base64 JPEG payload.",
      tech: "HTML5 MediaDevices / OV5640 Sensor",
      guarantees: "In-flight concurrency throttling; malformed base64 strings rejected with HTTP 400 before decoding.",
    },
    {
      title: "2. Preprocessing & Quality Gate",
      subtitle: "FastAPI Image Safety Gate",
      description: "Verifies decompression bomb limits (Pillow bounds), OpenCV dimension constraints, and image sharpness.",
      tech: "Pillow / OpenCV (cv2)",
      guarantees: "Rejects oversized or corrupt buffers before allocation to prevent memory-exhaustion DoS.",
    },
    {
      title: "3. Face Detection & Alignment",
      subtitle: "RetinaFace Landmark Localization",
      description: "Detects primary face bounding box and extracts 5 facial landmarks (eyes, nose tip, mouth corners).",
      tech: "InsightFace Buffalo_S / ONNX Runtime",
      guarantees: "Must detect exactly 1 face for enrollment; returns 'person_no_face' or 'face_low_quality' if unaligned.",
    },
    {
      title: "4. ArcFace 512D Vectorization",
      subtitle: "Deep Metric Embedding Generation",
      description: "Transforms aligned 112x112 face crop into a high-dimensional 512-float hypersphere vector representation.",
      tech: "ArcFace ResNet-50 ONNX Model",
      guarantees: "Inference semaphore guarantees thread safety; embeddings are L2-normalized so dot product equals cosine similarity.",
    },
    {
      title: "5. pgvector Cosine Search",
      subtitle: "PostgreSQL Vector Distance (<=>)",
      description: "Executes native vector similarity query against stored embeddings in PostgreSQL.",
      tech: "PostgreSQL 16 + pgvector",
      guarantees: "Evaluates cosine distance in milliseconds; queries top 2 matches for ambiguity evaluation.",
    },
    {
      title: "6. Authorization & Actuator Action",
      subtitle: "Fail-Safe Decision Engine",
      description: "Evaluates match similarity (>=0.50), ambiguity gap, and person status (active/suspended) to issue door command.",
      tech: "TypeScript Rule Engine",
      guarantees: "Fail-closed lock; unhandled errors and unknown faces strictly keep facility secured.",
    },
  ];

  return (
    <Card className="border border-border shadow-sm bg-card overflow-hidden">
      <CardHeader className="pb-4 border-b border-border">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2">
          <div>
            <CardTitle className="text-base font-bold text-foreground flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-500" /> Biometric Vector Transformation Pipeline
            </CardTitle>
            <CardDescription className="text-xs">
              Step-by-step walkthrough of how an optical face capture is transformed into an authorized physical door action.
            </CardDescription>
          </div>
          <Badge variant="outline" className="w-fit text-[11px] font-mono">
            InsightFace Buffalo_S (512D)
          </Badge>
        </div>
      </CardHeader>
      <CardContent className="p-4 sm:p-6 space-y-6">
        {/* Step Indicator Pills */}
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-2">
          {steps.map((st, idx) => (
            <button
              key={idx}
              onClick={() => setActiveStep(idx)}
              className={`p-2.5 rounded-lg border text-left transition-all cursor-pointer ${
                activeStep === idx
                  ? "bg-primary/10 border-primary text-primary"
                  : "bg-muted/30 border-border text-muted-foreground hover:bg-muted"
              }`}
            >
              <span className="text-[10px] font-mono block opacity-75">Step 0{idx + 1}</span>
              <span className="text-xs font-semibold block truncate text-foreground">{st.title.split(". ")[1]}</span>
            </button>
          ))}
        </div>

        {/* Active Step Showcase */}
        <div className="p-4 sm:p-5 rounded-xl border border-border/80 bg-muted/20 space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-3 border-b border-border/60">
            <div>
              <span className="text-xs font-mono text-primary font-semibold uppercase tracking-wider">
                Pipeline Stage 0{activeStep + 1}
              </span>
              <h4 className="text-base font-bold text-foreground">{steps[activeStep].title}</h4>
            </div>
            <Badge variant="secondary" className="w-fit font-mono text-xs">
              {steps[activeStep].tech}
            </Badge>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div className="space-y-1.5">
              <p className="text-xs uppercase font-mono text-muted-foreground font-semibold">Stage Objective</p>
              <p className="text-sm text-foreground leading-relaxed">{steps[activeStep].description}</p>
            </div>
            <div className="space-y-1.5">
              <p className="text-xs uppercase font-mono text-muted-foreground font-semibold">Security & Quality Guarantee</p>
              <p className="text-xs text-muted-foreground leading-relaxed bg-card p-3 rounded-lg border border-border">
                {steps[activeStep].guarantees}
              </p>
            </div>
          </div>

          {/* Synthetic Vector Visualization for Step 4 & 5 */}
          {(activeStep === 3 || activeStep === 4) && (
            <div className="p-3 bg-black/90 rounded-lg border border-border/80 font-mono text-[11px] space-y-1.5">
              <div className="flex items-center justify-between text-muted-foreground">
                <span>[Synthetic 512D ArcFace Normalized Tensor Snapshot]</span>
                <span className="text-emerald-400">||v|| = 1.0000 (Unit Sphere)</span>
              </div>
              <p className="text-emerald-400/90 truncate">
                [-0.0412, 0.0891, -0.0125, 0.1542, -0.0632, 0.0041, 0.1128, -0.0984, 0.0215, 0.0763, -0.1421, 0.0382, ... +500 floats]
              </p>
              <p className="text-slate-400 text-[10px]">
                Cosine Distance metric: 1.0 - (&sum; a_i &times; b_i). Matches evaluated at &le; 0.50 distance threshold.
              </p>
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}

export function DecisionFlowchart() {
  return (
    <Card className="border border-border shadow-sm bg-card">
      <CardHeader className="pb-3 border-b border-border">
        <CardTitle className="text-base font-bold text-foreground flex items-center gap-2">
          <ShieldCheck className="w-4 h-4 text-emerald-500" /> Deterministic Access Decision Matrix
        </CardTitle>
        <CardDescription className="text-xs">
          Comprehensive logic tree executed by evaluateAccess() in frontend/lib/access/access-control.ts.
        </CardDescription>
      </CardHeader>
      <CardContent className="p-4 sm:p-6 space-y-3">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          <div className="p-3.5 rounded-xl border border-emerald-500/30 bg-emerald-500/5 space-y-1.5">
            <div className="flex items-center justify-between">
              <Badge className="bg-emerald-500 text-white text-[10px] uppercase font-mono">GRANTED &bull; UNLOCK</Badge>
              <span className="text-[10px] font-mono text-emerald-600 dark:text-emerald-400 font-bold">Code: face_match</span>
            </div>
            <h5 className="text-xs font-bold text-foreground">Active Person Verified</h5>
            <p className="text-xs text-muted-foreground">
              Similarity &ge; 0.50, ambiguity gap &gt; 0.05, and person status is <code>active</code>. Door actuator receives command: <code>unlock</code> (60s).
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-rose-500/30 bg-rose-500/5 space-y-1.5">
            <div className="flex items-center justify-between">
              <Badge className="bg-rose-500 text-white text-[10px] uppercase font-mono">DENIED &bull; LOCK</Badge>
              <span className="text-[10px] font-mono text-rose-600 dark:text-rose-400 font-bold">Code: face_no_match</span>
            </div>
            <h5 className="text-xs font-bold text-foreground">Unknown Individual</h5>
            <p className="text-xs text-muted-foreground">
              Face detected cleanly, but highest similarity score &lt; 0.50 against all enrolled profiles. Door remains locked.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-amber-500/30 bg-amber-500/5 space-y-1.5">
            <div className="flex items-center justify-between">
              <Badge className="bg-amber-500 text-white text-[10px] uppercase font-mono">DENIED &bull; LOCK</Badge>
              <span className="text-[10px] font-mono text-amber-600 dark:text-amber-400 font-bold">Code: insufficient_confidence</span>
            </div>
            <h5 className="text-xs font-bold text-foreground">Ambiguous Recognition</h5>
            <p className="text-xs text-muted-foreground">
              Two candidates matched within 0.05 similarity margin. System safely rejects both to prevent misidentification.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-destructive/30 bg-destructive/5 space-y-1.5">
            <div className="flex items-center justify-between">
              <Badge className="bg-destructive text-destructive-foreground text-[10px] uppercase font-mono">DENIED &bull; LOCK</Badge>
              <span className="text-[10px] font-mono text-destructive font-bold">Code: unauthorized</span>
            </div>
            <h5 className="text-xs font-bold text-foreground">Inactive or Suspended Person</h5>
            <p className="text-xs text-muted-foreground">
              Face matched with high confidence, but account status is <code>inactive</code> or <code>suspended</code>. Door locked; anomaly alert flagged.
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-slate-500/30 bg-slate-500/5 space-y-1.5">
            <div className="flex items-center justify-between">
              <Badge variant="outline" className="text-[10px] uppercase font-mono">NO ACTION</Badge>
              <span className="text-[10px] font-mono text-slate-500 font-bold">Code: poor_quality_face</span>
            </div>
            <h5 className="text-xs font-bold text-foreground">Empty Frame or Low Quality</h5>
            <p className="text-xs text-muted-foreground">
              No person in view, face obstructed, or extreme blur. Actuator takes no action (avoids unnecessary solenoid chatter).
            </p>
          </div>

          <div className="p-3.5 rounded-xl border border-destructive/40 bg-destructive/10 space-y-1.5">
            <div className="flex items-center justify-between">
              <Badge className="bg-destructive text-destructive-foreground text-[10px] uppercase font-mono">DENIED &bull; FAIL-SAFE</Badge>
              <span className="text-[10px] font-mono text-destructive font-bold">Code: system_error</span>
            </div>
            <h5 className="text-xs font-bold text-foreground">Technical / Database Error</h5>
            <p className="text-xs text-muted-foreground">
              pgvector search failure or AI service 502/timeout. Fails closed with <code>door_action: lock</code>; high-severity alert created.
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
