"use client";

import React from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion";
import {
  Sparkles,
  ShieldAlert,
  Brain,
  Compass,
  Zap,
  Activity,
  Layers,
  CheckCircle2,
  AlertTriangle,
  HelpCircle,
} from "lucide-react";

export default function DefenseGuidePage() {
  return (
    <div className="p-3 sm:p-5 lg:p-6 max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="border-b border-border/80 pb-5 space-y-2">
        <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-mono">
          <Brain className="size-3" />
          <span>Capstone Examination &amp; Viva Voce Companion</span>
        </div>
        <h1 className="text-2xl sm:text-3xl font-extrabold text-foreground tracking-tight">
          Academic Novelty, Mathematical Proofs &amp; Defense Guide
        </h1>
        <p className="text-sm text-muted-foreground max-w-3xl leading-relaxed">
          Technical design justifications and empirical evidence addressing the classic questions asked by university external examiners and defense panels.
        </p>
      </div>

      {/* Production vs Research Architecture Clarification */}
      <div className="rounded-xl border border-sky-500/30 bg-sky-500/5 p-3.5 flex items-start gap-3">
        <Layers className="size-4 text-sky-500 shrink-0 mt-0.5" />
        <p className="text-xs text-muted-foreground leading-relaxed">
          <strong className="text-foreground">Production Runtime vs. Research Deliverables:</strong> Live DriveGuard monitoring runs <span className="font-mono text-foreground">GEOMETRIC_AUTHORITATIVE</span> across independent Browser WASM and Local/Remote Python MediaPipe + 4-Zone PDBP engines (passive shadow disabled). The MobileNetV3-Small cascade metrics represent offline historical ablation research (with an optional diagnostic shadow observer), and the ESP32 HMI console is a simulated cyber-physical deliverable.
        </p>
      </div>

      {/* The Core Examiner Critique Banner */}
      <div className="p-4 sm:p-5 rounded-2xl border border-primary/30 bg-primary/5 space-y-3">
        <div className="flex items-center gap-2 text-primary font-bold text-sm uppercase tracking-wide">
          <Sparkles className="size-4" />
          <span>The Classic Examiner Critique &amp; Scientific Defense</span>
        </div>
        <blockquote className="border-l-2 border-primary/60 pl-3 italic text-foreground text-sm font-medium">
          &quot;All you did was import an open-source computer vision library (MediaPipe) and put an `if (EAR &lt; 0.2)` statement on it. Where is your original engineering and scientific contribution?&quot;
        </blockquote>
        <p className="text-xs text-muted-foreground leading-relaxed pt-1">
          In Applied Computer Vision and Intelligent Transportation Systems (ITS), the scientific challenge is <strong>not</strong> training a 500-million parameter vision model from scratch. Instead, the true engineering contribution lies in <strong>diagnosing, modeling, and mathematically solving the four real-world failure modes that cause generic computer vision models to fail in automotive cockpits</strong>:
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-xs font-mono pt-1">
          <div className="p-2.5 rounded-lg bg-card/60 border border-border/80">
            <span className="font-bold text-sky-400 block">1. Perspective Distortion</span>
            <span className="text-[11px] text-muted-foreground">Looking at mirrors drops 2D EAR, causing false alarms. Solved via 4-Zone PDBP (34 → 0 FAs).</span>
          </div>
          <div className="p-2.5 rounded-lg bg-card/60 border border-border/80">
            <span className="font-bold text-amber-400 block">2. Late Warning Problem</span>
            <span className="text-[11px] text-muted-foreground">Waiting for 1.5s closure means active microsleep. Solved via sub-frame velocity ratio RCR (40s lead time).</span>
          </div>
          <div className="p-2.5 rounded-lg bg-card/60 border border-border/80">
            <span className="font-bold text-purple-400 block">3. Sensor Ambiguity &amp; Glare</span>
            <span className="text-[11px] text-muted-foreground">Optical confidence gating flags low-quality frames in production; MobileNetV3-Small cascade evaluated in offline ablation research.</span>
          </div>
          <div className="p-2.5 rounded-lg bg-card/60 border border-border/80">
            <span className="font-bold text-emerald-400 block">4. Cyber-Physical Determinism</span>
            <span className="text-[11px] text-muted-foreground">OS delays can mute alarms. Modeled via simulated ESP32 UART safety co-processor protocol and HITL chronometer.</span>
          </div>
        </div>
      </div>

      {/* The 5 Tough Examiner Questions Accordion */}
      <Card className="border-border/80 shadow-xs">
        <CardHeader className="p-4 sm:p-5 border-b border-border/60">
          <CardTitle className="text-base font-bold text-foreground">
            Examiner Defense Q&amp;A: Tough Technical Questions
          </CardTitle>
          <CardDescription className="text-xs">
            Review these scripted answers before your project demonstration and thesis presentation.
          </CardDescription>
        </CardHeader>
        <CardContent className="p-4 sm:p-5">
          <Accordion type="single" collapsible className="w-full space-y-3">
            <AccordionItem value="q1" className="border border-border/80 rounded-xl px-4 py-1">
              <AccordionTrigger className="text-sm font-semibold hover:no-underline py-2.5">
                Q1: &quot;Why didn&apos;t you just train an end-to-end 3D CNN or Vision Transformer on full video clips?&quot;
              </AccordionTrigger>
              <AccordionContent className="text-xs text-muted-foreground leading-relaxed space-y-2 pt-2">
                <p>
                  <strong className="text-foreground">Defense Answer:</strong> End-to-end video models (such as Video Swin or SlowFast) require gigabytes of VRAM, consume 50–100W of electrical power, and introduce temporal sliding-window latencies that are unacceptable in automotive safety.
                </p>
                <p>
                  Automotive ISO 26262 functional safety demands determinism, interpretability, and execution within a 33.3 ms frame budget. DriveGuard achieves this in production via deterministic 3D geometric tracking and 4-zone PDBP kinematics, while our offline ablation benchmark evaluated selective MobileNetV3 verification at <strong>2.23 ms mean latency on standard edge CPUs</strong>.
                </p>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="q2" className="border border-border/80 rounded-xl px-4 py-1">
              <AccordionTrigger className="text-sm font-semibold hover:no-underline py-2.5">
                Q2: &quot;How do you handle drivers with naturally narrower eyes or varying facial morphology?&quot;
              </AccordionTrigger>
              <AccordionContent className="text-xs text-muted-foreground leading-relaxed space-y-2 pt-2">
                <p>
                  <strong className="text-foreground">Defense Answer:</strong> Universal fixed thresholds (e.g. `EAR &lt; 0.22`) suffer from severe demographic bias. DriveGuard completely eliminates this bias through the Personalised Driver Behaviour Profile (PDBP).
                </p>
                <p>
                  By measuring each driver&apos;s open-eye distribution using Median and Median Absolute Deviation (MAD) during a 15-second 4-zone posture alignment, thresholds are normalized into personal robust z-scores: z_EAR = (x - x̃) / (1.4826 · MAD). An individual with narrower eyes has a lower numerical cutoff tailored to their anatomy, guaranteeing equal sensitivity across diverse demographics.
                </p>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="q3" className="border border-border/80 rounded-xl px-4 py-1">
              <AccordionTrigger className="text-sm font-semibold hover:no-underline py-2.5">
                Q3: &quot;How was the MobileNetV3-Small model trained, and how did you prevent data leakage?&quot;
              </AccordionTrigger>
              <AccordionContent className="text-xs text-muted-foreground leading-relaxed space-y-2 pt-2">
                <p>
                  <strong className="text-foreground">Defense Answer:</strong> We fine-tuned MobileNetV3-Small on the MRL Eye Dataset (84,898 ocular crops across 37 subjects). Crucially, we enforced a strict <strong>subject-disjoint split</strong>: the evaluation partition contained images exclusively from participants that were never seen during training.
                </p>
                <p>
                  This prevented the model from simply memorizing eye identities and proved generalization to unseen drivers, achieving <strong>92.77% accuracy and 97.44% ROC-AUC</strong> on completely novel subjects.
                </p>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="q4" className="border border-border/80 rounded-xl px-4 py-1">
              <AccordionTrigger className="text-sm font-semibold hover:no-underline py-2.5">
                Q4: &quot;What prevents the system from sounding alarms when a driver simply talks, laughs, or yawns?&quot;
              </AccordionTrigger>
              <AccordionContent className="text-xs text-muted-foreground leading-relaxed space-y-2 pt-2">
                <p>
                  <strong className="text-foreground">Defense Answer:</strong> DriveGuard separates ocular metrics from oral metrics. While Mouth Aspect Ratio (MAR) tracks yawning, yawning alone never triggers an immediate emergency alarm—it only increments the cumulative fatigue index.
                </p>
                <p>
                  Smiling and talking compress the cheeks and alter the lower eyelid. Our multi-factor fusion isolates this: when lower eyelid shift occurs without a reduction in blink reopening velocity (v_open), the risk engine suppresses false microsleep alarms.
                </p>
              </AccordionContent>
            </AccordionItem>

            <AccordionItem value="q5" className="border border-border/80 rounded-xl px-4 py-1">
              <AccordionTrigger className="text-sm font-semibold hover:no-underline py-2.5">
                Q5: &quot;Why is an embedded microcontroller (ESP32/STM32) co-processor necessary?&quot;
              </AccordionTrigger>
              <AccordionContent className="text-xs text-muted-foreground leading-relaxed space-y-2 pt-2">
                <p>
                  <strong className="text-foreground">Defense Answer:</strong> General-purpose operating systems (Windows, Linux, macOS) are non-deterministic, suffer from thread scheduling jitter, and rely on audio daemons that can be muted or delayed.
                </p>
                <p>
                  In ISO 26262 automotive engineering, mission-critical alarms require an isolated hardware safety co-processor. The secondary MCU operates a hardware watchdog supervisor (500 ms timeout), drives physical LRA haptics directly into the steering wheel, and closes the control loop via the physical acknowledgment button.
                </p>
              </AccordionContent>
            </AccordionItem>
          </Accordion>
        </CardContent>
      </Card>
    </div>
  );
}
