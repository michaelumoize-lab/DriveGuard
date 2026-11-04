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
  UserCheck,
  UserPlus,
  RefreshCw,
  Cpu,
  Lock,
  Layers,
  FileText,
  Key,
  Copy,
  Check,
  ChevronRight,
  Database,
  ArrowRight,
  HelpCircle,
  AlertCircle,
  Eye,
  Sliders,
  Terminal,
} from "lucide-react";

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface HelpTopic {
  id: string;
  category: "getting-started" | "registration" | "simulation" | "security" | "devices" | "troubleshooting";
  title: string;
  summary: string;
  badge?: string;
  content: React.ReactNode;
}

export default function HelpPage() {
  const [searchQuery, setSearchQuery] = useState("");
  const [activeTab, setActiveTab] = useState<string>("all");
  const [copiedSection, setCopiedSection] = useState<string | null>(null);

  const copyToClipboard = (text: string, label: string) => {
    navigator.clipboard.writeText(text);
    setCopiedSection(label);
    setTimeout(() => setCopiedSection(null), 2000);
  };

  const helpTopics: HelpTopic[] = useMemo(
    () => [
      {
        id: "gs-overview",
        category: "getting-started",
        title: "System Overview & Architecture",
        summary: "Understand how the Next.js web application, FastAPI AI service, and pgvector database coordinate.",
        badge: "Architecture",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              IntelliGuard is an AI-powered access control and security monitoring system. The system decouples high-performance biometric vector inference from administrative access management:
            </p>
            <ul className="list-disc pl-5 space-y-1.5">
              <li>
                <strong className="text-foreground">Next.js 16 Web Application:</strong> Manages administrative workflows, role-based authentication (Better-Auth), person profiles, access decision logic, real-time logging, and security alerts.
              </li>
              <li>
                <strong className="text-foreground">FastAPI AI Service:</strong> A dedicated Python service leveraging InsightFace (<code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">buffalo_s</code> ArcFace model) to detect faces, evaluate biometric quality, and generate 512-dimensional facial embeddings.
              </li>
              <li>
                <strong className="text-foreground">PostgreSQL + pgvector:</strong> Stores employee profiles alongside 512D floating-point vector embeddings, using cosine similarity queries to match incoming scans in milliseconds.
              </li>
              <li>
                <strong className="text-foreground">Access Decision Engine:</strong> Evaluates recognition scores against strict organizational policies (e.g. employee status, confidence thresholds) and issues fail-safe door lock/unlock instructions.
              </li>
            </ul>
            <div className="flex items-center gap-2 pt-2">
              <Button asChild size="sm" variant="outline" className="gap-1.5 text-xs">
                <Link href="/dashboard/project-overview">
                  <Sparkles className="w-3.5 h-3.5 text-primary" />
                  View Interactive Project Overview
                </Link>
              </Button>
            </div>
          </div>
        ),
      },
      {
        id: "gs-navigation",
        category: "getting-started",
        title: "Navigating the Administrative Dashboard",
        summary: "Quick tour of key dashboard panels, statistics, activity charts, and system health status.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>The main dashboard provides real-time visibility into the security perimeter:</p>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 pt-1">
              <div className="p-3 rounded-lg border border-border bg-card/50">
                <h4 className="text-xs font-semibold text-foreground flex items-center gap-1.5 mb-1">
                  <Sliders className="w-3.5 h-3.5 text-primary" /> Access Analytics
                </h4>
                <p className="text-xs">
                  Review today's total access attempts, door unlocks, rejections, and hourly traffic distribution.
                </p>
              </div>
              <div className="p-3 rounded-lg border border-border bg-card/50">
                <h4 className="text-xs font-semibold text-foreground flex items-center gap-1.5 mb-1">
                  <Database className="w-3.5 h-3.5 text-emerald-500" /> Biometric Health
                </h4>
                <p className="text-xs">
                  Verify PostgreSQL connectivity, pgvector extension readiness, and average biometric quality scores.
                </p>
              </div>
              <div className="p-3 rounded-lg border border-border bg-card/50">
                <h4 className="text-xs font-semibold text-foreground flex items-center gap-1.5 mb-1">
                  <ShieldAlert className="w-3.5 h-3.5 text-amber-500" /> Security Anomaly Panel
                </h4>
                <p className="text-xs">
                  Track unresolved alerts categorized by severity (Critical, High, Medium, Low) with one-click resolution.
                </p>
              </div>
              <div className="p-3 rounded-lg border border-border bg-card/50">
                <h4 className="text-xs font-semibold text-foreground flex items-center gap-1.5 mb-1">
                  <FileText className="w-3.5 h-3.5 text-cyan-500" /> Live Audit Trail
                </h4>
                <p className="text-xs">
                  Inspect the latest access verification events with timestamp, confidence, and door actuator actions.
                </p>
              </div>
            </div>
          </div>
        ),
      },
      {
        id: "reg-workflow",
        category: "registration",
        title: "Registering a New Person & Face",
        summary: "Step-by-step instructions for enrolling personnel with biometric photos and access permissions.",
        badge: "Core Workflow",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              To register an authorized employee or visitor, navigate to{" "}
              <Link href="/dashboard/persons/register" className="text-primary font-medium hover:underline">
                Person Management &rarr; Register Person
              </Link>
              :
            </p>
            <ol className="list-decimal pl-5 space-y-2">
              <li>
                <strong className="text-foreground">Personal Details:</strong> Provide the person's full name, email, department, role, and unique employee ID code.
              </li>
              <li>
                <strong className="text-foreground">Access Permissions:</strong> Assign status (<code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">active</code>, <code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">inactive</code>, or <code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">suspended</code>) and designated security clearance level.
              </li>
              <li>
                <strong className="text-foreground">Face Image Capture or Upload:</strong>
                <ul className="list-disc pl-5 mt-1 space-y-1">
                  <li>Use the live webcam capture tool with oval alignment reticle, OR</li>
                  <li>Upload a crisp portrait photo in JPEG/PNG format (max 5MB).</li>
                </ul>
              </li>
              <li>
                <strong className="text-foreground">Automatic Precheck Validation:</strong> The system sends the photo to the FastAPI AI service to verify that exactly one usable face is detected and that image quality exceeds acceptable thresholds.
              </li>
              <li>
                <strong className="text-foreground">Duplicate Detection Precheck:</strong> The AI extracts a 512D embedding and searches existing database records. If a matching face already exists, registration is prevented to avoid duplicate identities.
              </li>
              <li>
                <strong className="text-foreground">Enrollment Completion:</strong> Once validated, the person record is saved, the optimized photo is securely stored, and the active face embedding is persisted in pgvector.
              </li>
            </ol>
          </div>
        ),
      },
      {
        id: "reg-requirements",
        category: "registration",
        title: "Face Registration Quality Requirements",
        summary: "Guidelines to ensure high biometric enrollment quality and avoid registration rejections.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              InsightFace uses deep convolutional neural networks to extract facial landmarks. For best enrollment accuracy, follow these requirements:
            </p>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <div className="p-3 rounded-lg border border-emerald-500/30 bg-emerald-500/5">
                <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400 flex items-center gap-1.5 mb-1.5">
                  <CheckCircle2 className="w-3.5 h-3.5" /> What to Ensure
                </span>
                <ul className="text-xs space-y-1 list-disc pl-4">
                  <li>Direct frontal gaze toward camera</li>
                  <li>Neutral or gentle facial expression</li>
                  <li>Even lighting without harsh backlighting or heavy shadows</li>
                  <li>Face occupies 40%–70% of the frame</li>
                  <li>Sharp focus with minimal motion blur</li>
                </ul>
              </div>
              <div className="p-3 rounded-lg border border-destructive/30 bg-destructive/5">
                <span className="text-xs font-bold text-destructive flex items-center gap-1.5 mb-1.5">
                  <AlertCircle className="w-3.5 h-3.5" /> What to Avoid
                </span>
                <ul className="text-xs space-y-1 list-disc pl-4">
                  <li>Sunglasses, tinted eyewear, or wide-brimmed hats</li>
                  <li>Face coverings obscuring nose or mouth</li>
                  <li>Extreme side angles or tilted head positions</li>
                  <li>Multiple individuals visible in the capture frame</li>
                  <li>Severely compressed or grainy low-resolution photos</li>
                </ul>
              </div>
            </div>
          </div>
        ),
      },
      {
        id: "reg-duplicate",
        category: "registration",
        title: "Duplicate Face Prevention",
        summary: "How IntelliGuard detects and prevents registering the same person under multiple identities.",
        badge: "P0 Protection",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              To maintain data integrity and prevent identity fraud, IntelliGuard incorporates server-enforced duplicate face detection:
            </p>
            <ul className="list-disc pl-5 space-y-1.5">
              <li>
                During registration, after generating the 512D ArcFace embedding, the system executes a vector cosine similarity search across all active registered embeddings.
              </li>
              <li>
                If the similarity score exceeds the threshold of <strong className="text-foreground">0.50 (50%)</strong>, the registration API immediately rejects the request with HTTP <code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">409 Conflict</code>.
              </li>
              <li>
                The administrator is informed of the existing registered person's name and employee ID, preventing accidental duplicate profiles.
              </li>
            </ul>
          </div>
        ),
      },
      {
        id: "reg-status",
        category: "registration",
        title: "Managing Person Status: Active, Inactive & Suspended",
        summary: "Understanding how organizational status impacts biometric authorization decisions.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>Each person profile maintains an administrative access status:</p>
            <div className="space-y-2 pt-1">
              <div className="p-2.5 rounded-lg border border-emerald-500/30 bg-emerald-500/5">
                <span className="text-xs font-bold text-emerald-600 dark:text-emerald-400">ACTIVE:</span>
                <p className="text-xs mt-0.5">
                  Full facility access authorized. When recognized with sufficient biometric similarity (&ge;0.50), the door lock is triggered to unlock for 60 seconds.
                </p>
              </div>
              <div className="p-2.5 rounded-lg border border-amber-500/30 bg-amber-500/5">
                <span className="text-xs font-bold text-amber-500">INACTIVE:</span>
                <p className="text-xs mt-0.5">
                  Access is denied (<code className="text-xs font-mono">reason: "unauthorized"</code>). Identity is recognized for audit logging, but the door actuator maintains a safe locked state.
                </p>
              </div>
              <div className="p-2.5 rounded-lg border border-destructive/30 bg-destructive/5">
                <span className="text-xs font-bold text-destructive">SUSPENDED:</span>
                <p className="text-xs mt-0.5">
                  Access strictly blocked. The system immediately records an access rejection event and flags an anomaly alert for administrator review.
                </p>
              </div>
            </div>
            <p className="text-xs pt-1">
              Administrators can toggle person status at any time from the{" "}
              <Link href="/dashboard/persons" className="text-primary hover:underline font-medium">
                Person Management
              </Link>{" "}
              table.
            </p>
          </div>
        ),
      },
      {
        id: "reg-replace",
        category: "registration",
        title: "Replacing an Existing Face Biometric",
        summary: "How to update or re-enroll a person's photo when appearance changes or image quality degrades.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              When an employee changes appearance or requests a clearer photo, administrators can replace their facial biometric without recreating their profile:
            </p>
            <ol className="list-decimal pl-5 space-y-1.5">
              <li>Navigate to <Link href="/dashboard/persons" className="text-primary hover:underline font-medium">Person Management</Link> and select the person.</li>
              <li>In the profile modal or detail view, click <strong className="text-foreground">"Update Face Photo"</strong>.</li>
              <li>Capture or upload the new portrait photo.</li>
              <li>
                The system checks the new photo for single-face validity and runs duplicate detection against <em>other</em> registered individuals (exempting the current person).
              </li>
              <li>
                Upon confirmation, the previous embedding is marked inactive, and the new 512D embedding is activated seamlessly.
              </li>
            </ol>
          </div>
        ),
      },
      {
        id: "sim-webcam",
        category: "simulation",
        title: "Using Webcam Simulation for Testing",
        summary: "Simulate physical camera hardware from your desktop browser to test recognition and door logic.",
        badge: "Live Testing",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              The{" "}
              <Link href="/dashboard/simulation" className="text-primary font-medium hover:underline">
                Webcam Simulation
              </Link>{" "}
              page allows complete end-to-end testing of the IntelliGuard access control system without physical ESP32 hardware:
            </p>
            <ul className="list-disc pl-5 space-y-1.5">
              <li>
                <strong className="text-foreground">Live Feed & Reticle:</strong> Uses your browser's webcam with biometric oval guidelines and mirroring options.
              </li>
              <li>
                <strong className="text-foreground">Auto-Scan Loop:</strong> Automatically analyzes frames every 3 seconds, or allows manual one-click scan triggers.
              </li>
              <li>
                <strong className="text-foreground">Simulation Safeguards:</strong> In-flight request throttling prevents server flooding, and the request explicitly omits physical device headers to test pure browser access flows.
              </li>
              <li>
                <strong className="text-foreground">Audible Feedback:</strong> Provides optional audio chimes for granted unlocks, rejections, and error states.
              </li>
              <li>
                <strong className="text-foreground">Recent Scan History:</strong> Maintains the 10 most recent scans in a session reel for instant debugging.
              </li>
            </ul>
          </div>
        ),
      },
      {
        id: "sim-results",
        category: "simulation",
        title: "Understanding Recognition & Access Decisions",
        summary: "Deciphering match statuses, similarity percentages, door actions, and refusal reasons.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>Every scan returns a structured decision result evaluated by the backend engine:</p>
            <div className="space-y-2 pt-1">
              <div className="p-2.5 rounded-lg border border-border bg-card/60">
                <span className="text-xs font-mono font-bold text-emerald-500 uppercase">ACCESS GRANTED &bull; UNLOCK</span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Face matched an active employee profile with similarity &ge;50% (<code className="text-xs font-mono">match_status: "matched"</code>). Actuator triggers a temporary 60-second door unlock.
                </p>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-card/60">
                <span className="text-xs font-mono font-bold text-rose-500 uppercase">ACCESS DENIED &bull; UNKNOWN INDIVIDUAL</span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Face detected cleanly, but cosine similarity to all registered embeddings was below 50% (<code className="text-xs font-mono">match_status: "unknown"</code>). Door stays locked.
                </p>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-card/60">
                <span className="text-xs font-mono font-bold text-amber-500 uppercase">ACCESS DENIED &bull; AMBIGUOUS MATCH</span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Top two candidate matches were within 0.05 (5%) similarity of each other (<code className="text-xs font-mono">match_status: "ambiguous"</code>). Access is denied fail-closed with security logging.
                </p>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-card/60">
                <span className="text-xs font-mono font-bold text-slate-400 uppercase">NO ACTION &bull; FRAME EMPTY / LOW QUALITY</span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  No person detected or face quality too low for reliable matching. Actuator takes no action (<code className="text-xs font-mono">door_action: "no_action"</code>).
                </p>
              </div>
              <div className="p-2.5 rounded-lg border border-border bg-card/60">
                <span className="text-xs font-mono font-bold text-destructive uppercase">ACCESS DENIED &bull; SYSTEM / TECHNICAL ERROR</span>
                <p className="text-xs text-muted-foreground mt-0.5">
                  Database query failure, AI service timeout, or network drop. Fail-safe rule strictly denies access with an explicit lock command (<code className="text-xs font-mono">door_action: "lock"</code>).
                </p>
              </div>
            </div>
          </div>
        ),
      },
      {
        id: "sec-monitoring",
        category: "security",
        title: "Access Logs & Security Anomaly Monitoring",
        summary: "How to inspect audit logs, filter event histories, and resolve security alerts.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>IntelliGuard maintains comprehensive audit trails for physical security compliance:</p>
            <ul className="list-disc pl-5 space-y-1.5">
              <li>
                <strong className="text-foreground">Access Logs (<Link href="/dashboard/access-logs" className="text-primary hover:underline font-medium">/dashboard/access-logs</Link>):</strong> Comprehensive log of every biometric evaluation including timestamp, person identity (if recognized), similarity score, door action, and processing latency. Zero raw 512D vector floats are ever exposed in client responses.
              </li>
              <li>
                <strong className="text-foreground">Security Alerts (<Link href="/dashboard/alerts" className="text-primary hover:underline font-medium">/dashboard/alerts</Link>):</strong> Real-time anomalies triggered by unauthorized access attempts, repeated unknown scans, device offline signals, or system vector search failures.
              </li>
              <li>
                <strong className="text-foreground">Single & Batch Resolution:</strong> Administrators can mark individual alerts as resolved or use the <strong className="text-foreground">"Resolve All"</strong> button to clear all active alerts in a single authenticated operation. Each resolution records the admin's user ID for accountability.
              </li>
            </ul>
          </div>
        ),
      },
      {
        id: "sec-failsafe",
        category: "security",
        title: "Fail-Safe Engineering & Privacy Guarantees",
        summary: "Architectural rules that guarantee the facility remains locked during system errors.",
        badge: "Security Principle",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>IntelliGuard is engineered under the principle of <em>fail-closed security</em>:</p>
            <ul className="list-disc pl-5 space-y-1.5">
              <li>
                <strong className="text-foreground">Never Fall Back to Web Session:</strong> If a request contains physical device authentication headers (<code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">x-device-serial</code> or <code className="text-xs bg-muted px-1 py-0.5 rounded font-mono">x-device-key</code>) that are malformed or invalid, the server immediately responds with 401 Unauthorized without falling back to a browser session.
              </li>
              <li>
                <strong className="text-foreground">Technical Errors Are Never Unknowns:</strong> Database connectivity drops, pgvector index errors, or AI service 500s are strictly normalized as <code className="text-xs font-mono">status: "recognition_error"</code> with an explicit <code className="text-xs font-mono">lock</code> actuator command. No technical error can ever trigger an unlock.
              </li>
              <li>
                <strong className="text-foreground">Biometric Privacy:</strong> Raw 512D vector embeddings are stored in PostgreSQL pgvector columns and never returned over public client APIs. Facial photos are accessible only via signed ImageKit URLs with strict admin session authorization.
              </li>
            </ul>
          </div>
        ),
      },
      {
        id: "dev-management",
        category: "devices",
        title: "Managing IoT Devices & Provisioning API Keys",
        summary: "Registering ESP32-S3 devices, generating secret device keys, and tracking telemetry.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              Physical entry points are controlled by IoT microcontrollers registered under{" "}
              <Link href="/dashboard/devices" className="text-primary font-medium hover:underline">
                Device Management
              </Link>
              :
            </p>
            <ol className="list-decimal pl-5 space-y-1.5">
              <li>
                <strong className="text-foreground">Adding a Device:</strong> Provide a unique device name, serial number (e.g. <code className="text-xs font-mono">ESP32-CAM-01</code>), and physical location description.
              </li>
              <li>
                <strong className="text-foreground">Device Secret Generation:</strong> The system generates a cryptographic secret key (<code className="text-xs font-mono">dev_live_...</code>). This secret is revealed exactly once and must be provisioned onto the ESP32 microcontroller firmware.
              </li>
              <li>
                <strong className="text-foreground">Device Authentication:</strong> When transmitting camera captures, the ESP32 sends its serial number and hashed API key in custom headers.
              </li>
              <li>
                <strong className="text-foreground">Heartbeat & Health Sync:</strong> Devices report their telemetry (firmware version, WiFi RSSI, heap memory, IP address) via periodic heartbeats. Devices with no contact for over 5 minutes are automatically flagged as offline.
              </li>
            </ol>
          </div>
        ),
      },
      {
        id: "tb-service",
        category: "troubleshooting",
        title: "FastAPI AI Service Connection & Startup",
        summary: "Troubleshooting local uvicorn AI service errors, model downloads, and port availability.",
        badge: "Troubleshooting",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              If the dashboard displays an AI service offline badge or simulation scans return connection errors:
            </p>
            <div className="space-y-2">
              <div className="p-3 bg-muted/60 border border-border rounded-lg">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-mono font-semibold text-foreground flex items-center gap-1.5">
                    <Terminal className="w-3.5 h-3.5 text-primary" /> Start AI Service Locally
                  </span>
                  <button
                    onClick={() =>
                      copyToClipboard(
                        "cd ai-service\n.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload",
                        "cmd-ai-start"
                      )
                    }
                    className="text-xs text-muted-foreground hover:text-foreground flex items-center gap-1"
                  >
                    {copiedSection === "cmd-ai-start" ? <Check className="w-3 h-3 text-emerald-500" /> : <Copy className="w-3 h-3" />}
                    <span>{copiedSection === "cmd-ai-start" ? "Copied" : "Copy"}</span>
                  </button>
                </div>
                <pre className="text-xs font-mono text-muted-foreground overflow-x-auto p-2 bg-background/80 rounded border">
{`cd ai-service
.\\.venv\\Scripts\\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`}
                </pre>
              </div>
              <p className="text-xs">
                Check that <code className="text-xs font-mono">http://localhost:8000/health</code> returns <code className="text-xs font-mono">{"{\"status\": \"healthy\"}"}</code>. If the model fails to load, verify your Internet connection for the initial automatic InsightFace <code className="text-xs font-mono">buffalo_s</code> model weights download.
              </p>
            </div>
          </div>
        ),
      },
      {
        id: "tb-camera",
        category: "troubleshooting",
        title: "Webcam Permissions & Browser Media Errors",
        summary: "Resolving 'Camera Access Denied' or black preview screens during simulation and registration.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>If the camera preview fails to load or displays a black canvas:</p>
            <ul className="list-disc pl-5 space-y-1.5">
              <li>
                <strong className="text-foreground">Browser Permissions:</strong> Click the camera/lock icon in your browser URL address bar and ensure Camera access is set to "Allow".
              </li>
              <li>
                <strong className="text-foreground">Secure Context (HTTPS / localhost):</strong> WebRTC camera APIs are restricted to secure contexts (<code className="text-xs font-mono">http://localhost</code> or <code className="text-xs font-mono">https://</code>).
              </li>
              <li>
                <strong className="text-foreground">Device Contention:</strong> Ensure no other application (e.g. Zoom, Teams, OBS Studio) has exclusive locks on your webcam hardware.
              </li>
            </ul>
          </div>
        ),
      },
      {
        id: "tb-pgvector",
        category: "troubleshooting",
        title: "PostgreSQL & pgvector Extension Verification",
        summary: "Diagnosing vector similarity query errors and database connectivity issues.",
        content: (
          <div className="space-y-3 text-sm text-muted-foreground leading-relaxed">
            <p>
              IntelliGuard requires the native <code className="text-xs font-mono">vector</code> PostgreSQL extension. If vector similarity searches return 500 errors:
            </p>
            <ol className="list-decimal pl-5 space-y-1.5">
              <li>
                Verify that your database supports pgvector (e.g. Neon, Supabase, or PostgreSQL 16 with pgvector installed).
              </li>
              <li>
                Ensure the extension is enabled in your database:
                <pre className="text-xs font-mono bg-muted p-2 rounded mt-1 overflow-x-auto">
CREATE EXTENSION IF NOT EXISTS vector;
                </pre>
              </li>
              <li>
                Check that your <code className="text-xs font-mono">DATABASE_URL</code> in <code className="text-xs font-mono">.env</code> is properly configured and reachable.
              </li>
            </ol>
          </div>
        ),
      },
    ],
    [copiedSection]
  );

  const filteredTopics = useMemo(() => {
    return helpTopics.filter((topic) => {
      const matchesSearch =
        searchQuery.trim() === "" ||
        topic.title.toLowerCase().includes(searchQuery.toLowerCase()) ||
        topic.summary.toLowerCase().includes(searchQuery.toLowerCase());
      const matchesCategory = activeTab === "all" || topic.category === activeTab;
      return matchesSearch && matchesCategory;
    });
  }, [helpTopics, searchQuery, activeTab]);

  return (
    <div className="container max-w-6xl mx-auto p-4 sm:p-6 md:p-8 space-y-8 bg-background text-foreground">
      {/* Header Banner */}
      <div className="relative overflow-hidden rounded-2xl bg-gradient-to-r from-primary/15 via-primary/5 to-background border border-primary/20 p-6 md:p-8 shadow-sm">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 relative z-10">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <Badge variant="outline" className="bg-primary/10 text-primary border-primary/30 px-3 py-1 text-xs font-semibold">
                <Sparkles className="w-3.5 h-3.5 mr-1" /> Help & Knowledge Center
              </Badge>
            </div>
            <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-foreground">
              IntelliGuard Help Center
            </h1>
            <p className="text-sm md:text-base text-muted-foreground max-w-2xl">
              Complete administrator reference for facial recognition workflows, person enrollment, access control policies, device management, and troubleshooting.
            </p>
          </div>

          <div className="flex flex-col sm:flex-row gap-3 shrink-0">
            <Button asChild variant="outline" size="sm" className="gap-2 border-border hover:bg-accent">
              <a href="https://intelliguard-ai.onrender.com/docs" target="_blank" rel="noopener noreferrer">
                <Server className="w-4 h-4 text-primary" />
                Swagger Docs (Render Cloud)
                <ExternalLink className="w-3.5 h-3.5 text-muted-foreground" />
              </a>
            </Button>
            <Button asChild size="sm" className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/dashboard/simulation">
                <Camera className="w-4 h-4" />
                Open Live Simulation
              </Link>
            </Button>
          </div>
        </div>
      </div>

      {/* Search Bar & Quick Categories */}
      <div className="space-y-4">
        <div className="relative">
          <Search className="absolute left-3.5 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search help topics (e.g., registration, duplicate check, simulation, fail-safe)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10 h-11 bg-card border-border shadow-sm text-sm"
          />
        </div>

        {/* Category Tabs */}
        <Tabs value={activeTab} onValueChange={setActiveTab} className="w-full">
          <div className="w-full overflow-x-auto pb-2 pt-0.5">
            <TabsList className="inline-flex w-auto min-w-full sm:min-w-0 p-1.5 bg-muted/60 border border-border/80 rounded-xl justify-start gap-1.5 shadow-xs">
              <TabsTrigger value="all" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                All Topics ({helpTopics.length})
              </TabsTrigger>
              <TabsTrigger value="getting-started" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                Getting Started
              </TabsTrigger>
              <TabsTrigger value="registration" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                Person Registration
              </TabsTrigger>
              <TabsTrigger value="simulation" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                Simulation &amp; Decisions
              </TabsTrigger>
              <TabsTrigger value="security" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                Security &amp; Monitoring
              </TabsTrigger>
              <TabsTrigger value="devices" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                Device Management
              </TabsTrigger>
              <TabsTrigger value="troubleshooting" className="text-xs px-3.5 py-1.5 rounded-lg shrink-0 font-medium transition-all">
                Troubleshooting
              </TabsTrigger>
            </TabsList>
          </div>
        </Tabs>
      </div>

      {/* Help Topics List */}
      <div className="space-y-4">
        {filteredTopics.length === 0 ? (
          <Card className="border-border text-center py-12">
            <CardContent className="space-y-3">
              <HelpCircle className="w-10 h-10 text-muted-foreground/60 mx-auto" />
              <h3 className="text-base font-semibold text-foreground">No matching topics found</h3>
              <p className="text-xs text-muted-foreground max-w-sm mx-auto">
                No help documentation matched "{searchQuery}". Try searching for terms like "registration", "vector", "camera", or "fail-safe".
              </p>
              <Button size="sm" variant="outline" onClick={() => { setSearchQuery(""); setActiveTab("all"); }}>
                Reset Filters
              </Button>
            </CardContent>
          </Card>
        ) : (
          <Accordion type="single" collapsible className="space-y-3">
            {filteredTopics.map((topic) => (
              <AccordionItem
                key={topic.id}
                value={topic.id}
                className="bg-card border border-border rounded-xl px-4 sm:px-6 shadow-sm overflow-hidden"
              >
                <AccordionTrigger className="hover:no-underline py-4">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 text-left w-full pr-3">
                    <div>
                      <div className="flex items-center gap-2">
                        <span className="font-semibold text-sm sm:text-base text-foreground tracking-tight">
                          {topic.title}
                        </span>
                        {topic.badge && (
                          <Badge variant="outline" className="text-[10px] uppercase font-mono px-2 py-0 border-primary/30 text-primary">
                            {topic.badge}
                          </Badge>
                        )}
                      </div>
                      <p className="text-xs text-muted-foreground font-normal mt-0.5 line-clamp-1 sm:line-clamp-none">
                        {topic.summary}
                      </p>
                    </div>
                  </div>
                </AccordionTrigger>
                <AccordionContent className="pb-5 pt-1 border-t border-border/60">
                  {topic.content}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        )}
      </div>

      {/* Quick Links Footer Bar */}
      <Card className="border border-border bg-muted/30">
        <CardContent className="p-4 sm:p-6 flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="space-y-1 text-center md:text-left">
            <h4 className="text-sm font-semibold text-foreground">Need in-depth technical architecture details?</h4>
            <p className="text-xs text-muted-foreground">
              Review the complete interactive software and IoT architecture diagram prepared for graduation project presentations.
            </p>
          </div>
          <div className="flex items-center gap-2.5">
            <Button asChild size="sm" className="bg-primary text-primary-foreground hover:bg-primary/90">
              <Link href="/dashboard/project-overview">
                Explore Project Overview <ArrowRight className="w-3.5 h-3.5 ml-1" />
              </Link>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
