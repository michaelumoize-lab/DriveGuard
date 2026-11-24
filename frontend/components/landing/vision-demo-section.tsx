"use client";

import React, { useEffect, useId, useRef, useState } from "react";
import { animate, motion, useReducedMotion } from "motion/react";
import { Eye, ScanFace } from "lucide-react";
import { LandingMediaSlot } from "@/components/landing/landing-media-slot";
import { cn } from "@/lib/utils";

type VisionStateKey = "alert" | "distracted" | "drowsy";

interface VisionStateProfile {
  key: VisionStateKey;
  label: string;
  /** Theme-aware token for UI panels */
  themeColor: string;
  /** Optical camera HUD color (always over dark sensor feed) */
  hudColor: string;
  badgeBg: string;
  badgeBorder: string;
  eyeState: string;
  headPose: string;
  blinkAnalysis: string;
  attention: string;
  ear: number;
  perclos: number;
  blinkMs: number;
  mar: number;
  eyeOpenFactor: number;
  headOffsetX: number;
  headOffsetY: number;
  mouthOpen: number;
  yawDeg: number;
  pitchDeg: number;
  summary: string;
}

const VISION_STATES: Record<VisionStateKey, VisionStateProfile> = {
  alert: {
    key: "alert",
    label: "Alert",
    themeColor: "var(--safe)",
    hudColor: "#79B99A",
    badgeBg: "rgba(121, 185, 154, 0.14)",
    badgeBorder: "rgba(121, 185, 154, 0.38)",
    eyeState: "Open",
    headPose: "Center",
    blinkAnalysis: "Normal",
    attention: "Focused",
    ear: 0.296,
    perclos: 3.4,
    blinkMs: 142,
    mar: 0.18,
    eyeOpenFactor: 1,
    headOffsetX: 0,
    headOffsetY: 0,
    mouthOpen: 0.08,
    yawDeg: 1,
    pitchDeg: 0,
    summary:
      "Nominal ocular kinematics. Eyelid reopening velocity and forward road gaze remain within personal baseline.",
  },
  distracted: {
    key: "distracted",
    label: "Distracted",
    themeColor: "var(--warning)",
    hudColor: "#E8A04B",
    badgeBg: "rgba(232, 160, 75, 0.14)",
    badgeBorder: "rgba(232, 160, 75, 0.4)",
    eyeState: "Shifted Gaze",
    headPose: "Off-Axis (-24°)",
    blinkAnalysis: "Elevated",
    attention: "Diverted",
    ear: 0.244,
    perclos: 8.6,
    blinkMs: 178,
    mar: 0.22,
    eyeOpenFactor: 0.82,
    headOffsetX: -14,
    headOffsetY: 3,
    mouthOpen: 0.12,
    yawDeg: -24,
    pitchDeg: -5,
    summary:
      "Sustained off-axis head yaw exceeding the 3.0s mirror-glance window triggers an early attention advisory.",
  },
  drowsy: {
    key: "drowsy",
    label: "Drowsy",
    themeColor: "var(--critical)",
    hudColor: "#E45E52",
    badgeBg: "rgba(228, 94, 82, 0.15)",
    badgeBorder: "rgba(228, 94, 82, 0.44)",
    eyeState: "Prolonged Closure",
    headPose: "Forward Nod (-16°)",
    blinkAnalysis: "Sluggish RCR",
    attention: "Impaired",
    ear: 0.148,
    perclos: 26.8,
    blinkMs: 420,
    mar: 0.78,
    eyeOpenFactor: 0.24,
    headOffsetX: 2,
    headOffsetY: 11,
    mouthOpen: 0.92,
    yawDeg: 3,
    pitchDeg: -16,
    summary:
      "EAR drops below the personalized closure floor alongside yawns and slow eyelid recovery—escalating warning state.",
  },
};

const STATE_ORDER: VisionStateKey[] = ["alert", "distracted", "drowsy"];

function AnimatedNumber({
  value,
  decimals = 0,
  suffix = "",
}: {
  value: number;
  decimals?: number;
  suffix?: string;
}) {
  const prefersReducedMotion = useReducedMotion();
  const [display, setDisplay] = useState(value);
  const prevRef = useRef(value);

  useEffect(() => {
    if (prefersReducedMotion) {
      prevRef.current = value;
      return;
    }

    const controls = animate(prevRef.current, value, {
      duration: 0.45,
      ease: [0.22, 1, 0.36, 1],
      onUpdate: (latest) => setDisplay(latest),
      onComplete: () => {
        prevRef.current = value;
      },
    });

    return () => controls.stop();
  }, [value, prefersReducedMotion]);

  const shownValue = prefersReducedMotion ? value : display;

  return (
    <span className="tabular-nums">
      {shownValue.toFixed(decimals)}
      {suffix}
    </span>
  );
}

export function VisionDemoSection() {
  const [activeKey, setActiveKey] = useState<VisionStateKey>("alert");
  const prefersReducedMotion = useReducedMotion();
  const tablistId = useId();
  const tabRefs = useRef<Array<HTMLButtonElement | null>>([]);
  const active = VISION_STATES[activeKey];

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIdx = index;
    if (e.key === "ArrowRight") {
      nextIdx = (index + 1) % STATE_ORDER.length;
    } else if (e.key === "ArrowLeft") {
      nextIdx = (index - 1 + STATE_ORDER.length) % STATE_ORDER.length;
    } else if (e.key === "Home") {
      nextIdx = 0;
    } else if (e.key === "End") {
      nextIdx = STATE_ORDER.length - 1;
    } else {
      return;
    }
    e.preventDefault();
    const nextKey = STATE_ORDER[nextIdx];
    setActiveKey(nextKey);
    tabRefs.current[nextIdx]?.focus();
  };

  const cx = 160 + active.headOffsetX;
  const cy = 152 + active.headOffsetY;
  const eyeRy = 8.5 * active.eyeOpenFactor;
  const mouthRy = 2.5 + active.mouthOpen * 13;

  return (
    <section
      id="vision"
      className="py-20 md:py-28 border-t border-border scroll-mt-24"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-12">
        {/* Section Header + Interactive State Switcher */}
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div className="max-w-2xl space-y-3">
            <span className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-primary-accent">
              <ScanFace className="size-3.5" />
              <span>Computer Vision Architecture</span>
            </span>
            <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-foreground leading-[1.12] text-balance">
              How DriveGuard sees what others miss
            </h2>
            <p className="text-sm sm:text-base text-muted-foreground leading-relaxed">
              Switch between driver states below to inspect how DriveGuard extracts Eye Aspect Ratio (EAR), blink duration, 6DoF head pose, and yawning kinematics in real time.
            </p>
          </div>

          {/* Accessible Single-Row Segmented Control */}
          <div
            role="tablist"
            aria-label="Simulated driver monitoring states"
            className="inline-flex flex-nowrap items-center gap-1 rounded-2xl border border-border bg-card p-1.5 self-start lg:self-auto shadow-xs"
          >
            {STATE_ORDER.map((key, idx) => {
              const stateItem = VISION_STATES[key];
              const selected = activeKey === key;
              return (
                <button
                  key={key}
                  ref={(el) => {
                    tabRefs.current[idx] = el;
                  }}
                  id={`${tablistId}-tab-${key}`}
                  type="button"
                  role="tab"
                  aria-selected={selected}
                  tabIndex={selected ? 0 : -1}
                  onClick={() => setActiveKey(key)}
                  onKeyDown={(e) => handleKeyDown(e, idx)}
                  className={cn(
                    "relative shrink-0 px-4 py-2 rounded-xl text-xs font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring",
                    selected
                      ? "text-foreground"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  {selected && (
                    <motion.span
                      layoutId="vision-state-pill"
                      transition={{ type: "spring", stiffness: 360, damping: 32 }}
                      className="absolute inset-0 rounded-xl bg-background border border-border shadow-2xs"
                    />
                  )}
                  <span className="relative z-10 inline-flex items-center gap-2">
                    <span
                      className="size-2 rounded-full"
                      style={{ backgroundColor: stateItem.themeColor }}
                    />
                    <span>{stateItem.label}</span>
                  </span>
                  {selected && (
                    <span
                      aria-hidden="true"
                      className="absolute inset-x-3 bottom-0.5 h-[2px] rounded-full z-10"
                      style={{ backgroundColor: stateItem.themeColor }}
                    />
                  )}
                </button>
              );
            })}
          </div>
        </div>

        {/* Main Interactive Demonstration Stage */}
        <div
          role="tabpanel"
          aria-labelledby={`${tablistId}-tab-${activeKey}`}
          tabIndex={0}
          className="grid grid-cols-1 lg:grid-cols-12 gap-6 rounded-3xl border border-border bg-card p-4 sm:p-6 lg:p-8 shadow-xl outline-none focus-visible:ring-2 focus-visible:ring-ring"
        >
          {/* Left: Optical Facial Landmark & Eye Contour Rig */}
          <div className="lg:col-span-7">
            <LandingMediaSlot
              slot="visionFace"
              sizes="(max-width: 1024px) 100vw, 55vw"
              className="aspect-[5/4] sm:aspect-[16/12] w-full rounded-2xl border border-border/80"
              overlayClassName="bg-gradient-to-b from-[#101211]/35 via-transparent to-[#101211]/80"
            >
              <div className="absolute top-3.5 left-3.5 z-10 flex items-center gap-2 rounded-lg border border-white/12 bg-[#101211]/85 px-2.5 py-1.5 text-[10px] font-mono text-[#9EA39B] backdrop-blur-sm">
                <Eye className="size-3 text-[#E58B43]" />
                <span>MESH: 478-PT 3D</span>
              </div>

              <div
                className="absolute top-3.5 right-3.5 z-10 rounded-lg px-2.5 py-1 text-[10px] font-mono font-semibold uppercase tracking-wider backdrop-blur-sm border transition-colors duration-300"
                style={{
                  color: active.hudColor,
                  backgroundColor: active.badgeBg,
                  borderColor: active.badgeBorder,
                }}
              >
                STATE: {active.label}
              </div>

              <svg
                viewBox="0 0 320 280"
                aria-hidden="true"
                className="pointer-events-none relative z-[5] h-full w-full"
              >
                <motion.rect
                  animate={{
                    x: cx - 72,
                    y: cy - 92,
                  }}
                  transition={{ type: "spring", stiffness: 180, damping: 24 }}
                  width="144"
                  height="184"
                  rx="14"
                  fill="none"
                  stroke={active.hudColor}
                  strokeOpacity="0.45"
                  strokeWidth="1"
                  strokeDasharray="4 4"
                />

                <motion.g
                  animate={{ x: active.headOffsetX, y: active.headOffsetY }}
                  transition={{ type: "spring", stiffness: 180, damping: 24 }}
                >
                  <path
                    d="M 102 126 C 102 74, 218 74, 218 126 C 218 168, 198 220, 160 226 C 122 220, 102 168, 102 126 Z"
                    fill="rgba(16,18,17,0.35)"
                    stroke="rgba(245,243,236,0.22)"
                    strokeWidth="1.2"
                  />

                  <g stroke="rgba(245,243,236,0.14)" strokeWidth="0.75">
                    <line x1="116" y1="114" x2="140" y2="128" />
                    <line x1="204" y1="114" x2="180" y2="128" />
                    <line x1="140" y1="128" x2="160" y2="160" />
                    <line x1="180" y1="128" x2="160" y2="160" />
                    <line x1="114" y1="158" x2="160" y2="160" />
                    <line x1="206" y1="158" x2="160" y2="160" />
                    <line x1="138" y1="194" x2="160" y2="160" />
                    <line x1="182" y1="194" x2="160" y2="160" />
                    <line x1="118" y1="192" x2="160" y2="224" />
                    <line x1="202" y1="192" x2="160" y2="224" />
                  </g>

                  <path
                    d="M 118 113 Q 134 106 149 113"
                    fill="none"
                    stroke="rgba(245,243,236,0.45)"
                    strokeWidth="1.3"
                    strokeLinecap="round"
                  />
                  <path
                    d="M 171 113 Q 186 106 202 113"
                    fill="none"
                    stroke="rgba(245,243,236,0.45)"
                    strokeWidth="1.3"
                    strokeLinecap="round"
                  />

                  <motion.ellipse
                    cx="134"
                    cy="128"
                    rx="15"
                    animate={{ ry: Math.max(1.8, eyeRy) }}
                    transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                    fill={active.badgeBg}
                    stroke={active.hudColor}
                    strokeWidth="1.6"
                  />
                  <motion.circle
                    cx={activeKey === "distracted" ? 128 : 134}
                    cy="128"
                    animate={{ r: active.eyeOpenFactor > 0.35 ? 3 : 1.2, opacity: active.eyeOpenFactor }}
                    fill={active.hudColor}
                  />
                  <line x1="134" y1="117" x2="134" y2="139" stroke={active.hudColor} strokeOpacity="0.45" strokeWidth="0.9" strokeDasharray="2 2" />

                  <motion.ellipse
                    cx="186"
                    cy="128"
                    rx="15"
                    animate={{ ry: Math.max(1.8, eyeRy) }}
                    transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                    fill={active.badgeBg}
                    stroke={active.hudColor}
                    strokeWidth="1.6"
                  />
                  <motion.circle
                    cx={activeKey === "distracted" ? 180 : 186}
                    cy="128"
                    animate={{ r: active.eyeOpenFactor > 0.35 ? 3 : 1.2, opacity: active.eyeOpenFactor }}
                    fill={active.hudColor}
                  />
                  <line x1="186" y1="117" x2="186" y2="139" stroke={active.hudColor} strokeOpacity="0.45" strokeWidth="0.9" strokeDasharray="2 2" />

                  <g transform="translate(160, 158)">
                    <motion.line
                      x1="0"
                      y1="0"
                      animate={{ x2: active.yawDeg * 0.9, y2: -4 }}
                      stroke="#E58B43"
                      strokeWidth="1.6"
                      strokeLinecap="round"
                    />
                    <motion.line
                      x1="0"
                      y1="0"
                      animate={{ x2: 0, y2: -20 + active.pitchDeg * -0.8 }}
                      stroke="#79B99A"
                      strokeWidth="1.4"
                      strokeLinecap="round"
                    />
                    <circle cx="0" cy="0" r="2.6" fill="#F5F3EC" />
                  </g>

                  <motion.ellipse
                    cx="160"
                    cy="192"
                    rx="17"
                    animate={{ ry: mouthRy }}
                    transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
                    fill={activeKey === "drowsy" ? "rgba(228, 94, 82, 0.2)" : "rgba(245, 243, 236, 0.06)"}
                    stroke={activeKey === "drowsy" ? "#E45E52" : "rgba(245, 243, 236, 0.45)"}
                    strokeWidth="1.35"
                  />

                  {[
                    [118, 113],
                    [149, 113],
                    [171, 113],
                    [202, 113],
                    [119, 128],
                    [149, 128],
                    [171, 128],
                    [201, 128],
                    [143, 192],
                    [177, 192],
                    [160, 226],
                  ].map(([lx, ly], idx) => (
                    <circle
                      key={idx}
                      cx={lx}
                      cy={ly}
                      r="1.8"
                      fill={active.hudColor}
                      fillOpacity="0.85"
                    />
                  ))}
                </motion.g>
              </svg>

              {!prefersReducedMotion && (
                <motion.div
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-x-[22%] top-[18%] h-px bg-gradient-to-r from-transparent via-[#E58B43]/65 to-transparent"
                  animate={{ y: [0, 185, 0] }}
                  transition={{ duration: 5.2, repeat: Infinity, ease: "easeInOut" }}
                />
              )}

              <div className="absolute bottom-3.5 inset-x-3.5 z-10 flex flex-wrap items-center justify-between gap-2 rounded-xl border border-white/10 bg-[#101211]/85 px-3.5 py-2 text-[11px] font-mono backdrop-blur-md">
                <span className="text-[#9EA39B]">
                  EAR:{" "}
                  <strong className="text-[#F5F3EC]">
                    <AnimatedNumber value={active.ear} decimals={3} />
                  </strong>
                </span>
                <span className="text-[#9EA39B]">
                  YAW:{" "}
                  <strong className="text-[#F5F3EC]">
                    <AnimatedNumber value={active.yawDeg} decimals={0} suffix="°" />
                  </strong>
                </span>
                <span className="text-[#9EA39B]">
                  PITCH:{" "}
                  <strong className="text-[#F5F3EC]">
                    <AnimatedNumber value={active.pitchDeg} decimals={0} suffix="°" />
                  </strong>
                </span>
                <span className="text-[#9EA39B]">
                  MAR:{" "}
                  <strong className="text-[#F5F3EC]">
                    <AnimatedNumber value={active.mar} decimals={2} />
                  </strong>
                </span>
              </div>
            </LandingMediaSlot>
          </div>

          {/* Right: DriveGuard Vision Theme-Aware Metrics Panel */}
          <div className="lg:col-span-5 flex flex-col justify-between gap-5 rounded-2xl border border-border bg-background p-5 sm:p-6">
            <div className="space-y-5">
              <div className="flex items-center justify-between border-b border-border pb-4">
                <div>
                  <span className="text-xs font-semibold text-foreground block">
                    DriveGuard Vision
                  </span>
                  <span className="text-[10px] font-mono uppercase tracking-[0.16em] text-muted-foreground">
                    Facial Analysis
                  </span>
                </div>
                <span className="inline-flex items-center gap-1.5 rounded-md border border-primary/40 bg-primary/15 px-2.5 py-1 text-[10px] font-mono uppercase tracking-wider text-primary-accent">
                  Demo Mode
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: "Eye state", value: active.eyeState },
                  { label: "Head pose", value: active.headPose },
                  { label: "Blink analysis", value: active.blinkAnalysis },
                  { label: "Attention", value: active.attention },
                ].map((metric) => (
                  <div
                    key={metric.label}
                    className="rounded-xl border border-border bg-card p-3.5 space-y-1"
                  >
                    <span className="block text-[11px] text-muted-foreground">
                      {metric.label}
                    </span>
                    <span
                      className="block text-sm font-semibold transition-colors duration-300"
                      style={{ color: active.themeColor }}
                    >
                      {metric.value}
                    </span>
                  </div>
                ))}
              </div>

              <div className="rounded-xl border border-border bg-surface-panel p-4 space-y-3">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-muted-foreground">60s Rolling PERCLOS</span>
                  <span className="font-mono font-semibold text-foreground">
                    <AnimatedNumber value={active.perclos} decimals={1} suffix="%" />
                  </span>
                </div>
                <div className="h-1.5 w-full rounded-full bg-border overflow-hidden">
                  <motion.div
                    className="h-full rounded-full"
                    animate={{
                      width: `${Math.min(100, (active.perclos / 35) * 100)}%`,
                      backgroundColor: active.themeColor,
                    }}
                    transition={{ duration: 0.45, ease: [0.22, 1, 0.36, 1] }}
                  />
                </div>

                <div className="flex items-center justify-between text-xs pt-1">
                  <span className="text-muted-foreground">Blink Duration</span>
                  <span className="font-mono font-semibold text-foreground">
                    <AnimatedNumber value={active.blinkMs} decimals={0} suffix=" ms" />
                  </span>
                </div>
              </div>

              <p className="text-xs text-muted-foreground leading-relaxed">
                {active.summary}
              </p>
            </div>

            <div className="pt-3 border-t border-border flex items-center justify-between text-[11px] text-muted-foreground">
              <span>Illustrative interface concept, not actual live model output.</span>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
