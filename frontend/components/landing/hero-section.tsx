"use client";

import React, { useRef } from "react";
import Link from "next/link";
import {
  motion,
  useMotionValue,
  useReducedMotion,
  useSpring,
  useTransform,
} from "motion/react";
import { ArrowRight, Eye, ShieldCheck, Sparkles } from "lucide-react";
import { LandingMediaSlot } from "@/components/landing/landing-media-slot";
import { scrollToSection } from "@/lib/smooth-scroll";

interface HeroSectionProps {
  isAuthenticated: boolean;
}

const TRUST_INDICATORS = [
  "Real-time monitoring",
  "Personalized calibration",
  "Intelligent alerts",
];

const HERO_LANDMARKS = [
  { x: 185, y: 122 },
  { x: 215, y: 122 },
  { x: 170, y: 145 },
  { x: 230, y: 145 },
  { x: 200, y: 158 },
  { x: 184, y: 176 },
  { x: 216, y: 176 },
  { x: 200, y: 196 },
];

export function HeroSection({ isAuthenticated }: HeroSectionProps) {
  const prefersReducedMotion = useReducedMotion();
  const cardRef = useRef<HTMLDivElement>(null);

  const mouseX = useMotionValue(0);
  const mouseY = useMotionValue(0);
  const smoothX = useSpring(mouseX, { stiffness: 140, damping: 20 });
  const smoothY = useSpring(mouseY, { stiffness: 140, damping: 20 });

  const rotateY = useTransform(smoothX, [-0.5, 0.5], [-3.5, 3.5]);
  const rotateX = useTransform(smoothY, [-0.5, 0.5], [3, -3]);

  const handlePointerMove = (e: React.PointerEvent<HTMLDivElement>) => {
    if (prefersReducedMotion || e.pointerType === "touch") return;
    const rect = cardRef.current?.getBoundingClientRect();
    if (!rect) return;
    mouseX.set((e.clientX - rect.left) / rect.width - 0.5);
    mouseY.set((e.clientY - rect.top) / rect.height - 0.5);
  };

  const handlePointerLeave = () => {
    mouseX.set(0);
    mouseY.set(0);
  };

  return (
    <section className="relative pt-28 pb-20 md:pt-36 md:pb-28 overflow-hidden">
      {/* Subtle ambient radial glow — restrained, theme-aware */}
      <div
        aria-hidden="true"
        className="pointer-events-none absolute -top-24 right-[-10%] h-[520px] w-[520px] rounded-full bg-[radial-gradient(circle,color-mix(in_srgb,var(--primary)_12%,transparent),transparent_65%)] blur-2xl"
      />
      <div
        aria-hidden="true"
        className="pointer-events-none absolute bottom-0 left-[-8%] h-[420px] w-[420px] rounded-full bg-[radial-gradient(circle,color-mix(in_srgb,var(--safe)_9%,transparent),transparent_70%)] blur-2xl"
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-10 items-center">
          {/* Left Column: Brand & Product Positioning */}
          <motion.div
            initial={prefersReducedMotion ? false : { opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-6 space-y-7"
          >
            <div className="inline-flex items-center gap-2.5 px-3 py-1.5 rounded-full border border-border bg-card/85 shadow-2xs">
              <span className="size-1.5 rounded-full bg-primary" />
              <span className="text-[11px] font-mono uppercase tracking-[0.18em] text-foreground/85">
                Intelligent Driver Monitoring
              </span>
            </div>

            <div className="space-y-4">
              <h1 className="font-display text-4xl sm:text-5xl xl:text-[3.75rem] font-bold tracking-[-0.03em] text-foreground leading-[1.06] text-balance">
                Stay Alert.{" "}
                <span className="text-foreground/65">Arrive Safe.</span>
              </h1>
              <p className="text-base sm:text-lg text-muted-foreground leading-relaxed max-w-xl text-pretty">
                DriveGuard pairs 478-point facial kinematics with personalized 4-zone baseline calibration to detect early drowsiness, prolonged eye closure, and diverted attention—without false alarms during mirror checks.
              </p>
            </div>

            <div className="flex flex-col sm:flex-row items-stretch sm:items-center gap-3.5 pt-1">
              <Link
                href={isAuthenticated ? "/dashboard/live-monitor" : "/auth/sign-in"}
                className="inline-flex items-center justify-center gap-2.5 px-6 py-3.5 rounded-xl bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-sm transition-colors shadow-sm outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 focus-visible:ring-offset-background"
              >
                <ShieldCheck className="size-4" />
                <span>{isAuthenticated ? "Launch Live Monitor" : "Explore DriveGuard"}</span>
                <ArrowRight className="size-4" />
              </Link>

              <a
                href="#vision"
                onClick={(e) => scrollToSection(e, "#vision")}
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 rounded-xl border border-border bg-card/80 hover:bg-accent text-foreground font-medium text-sm transition-colors outline-none focus-visible:ring-2 focus-visible:ring-ring"
              >
                <Eye className="size-4 text-primary-accent" />
                <span>See How It Works</span>
              </a>
            </div>

            {/* Trust Indicators */}
            <div className="pt-3 border-t border-border flex flex-wrap items-center gap-x-5 gap-y-2 text-xs text-muted-foreground">
              {TRUST_INDICATORS.map((item, idx) => (
                <React.Fragment key={item}>
                  <span className="inline-flex items-center gap-2">
                    <span className="size-1.5 rounded-full bg-safe" />
                    <span>{item}</span>
                  </span>
                  {idx < TRUST_INDICATORS.length - 1 && (
                    <span aria-hidden="true" className="hidden sm:inline text-border">
                      ·
                    </span>
                  )}
                </React.Fragment>
              ))}
            </div>
          </motion.div>

          {/* Right Column: DriveGuard Optical Monitoring Interface */}
          <motion.div
            initial={prefersReducedMotion ? false : { opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.65, delay: 0.1, ease: [0.22, 1, 0.36, 1] }}
            className="lg:col-span-6 perspective-[1100px]"
          >
            <motion.div
              ref={cardRef}
              onPointerMove={handlePointerMove}
              onPointerLeave={handlePointerLeave}
              style={prefersReducedMotion ? undefined : { rotateX, rotateY }}
              className="relative rounded-2xl border border-border bg-card p-2.5 sm:p-3 shadow-2xl"
            >
              {/* Optical Monitor Header Bar */}
              <div className="flex items-center justify-between px-3 py-2 border-b border-border/80 mb-2.5 text-[11px] font-mono text-muted-foreground">
                <div className="flex items-center gap-2">
                  <span className="size-2 rounded-full bg-safe" />
                  <span className="text-foreground/90 tracking-wider">DG-OPTICAL // CABIN CAM 01</span>
                </div>
                <span className="text-muted-foreground/85">60 FPS · 478 LANDMARKS</span>
              </div>

              {/* Driver Visual Viewport (Blank Drop-In Slot + CV Overlay) */}
              <LandingMediaSlot
                slot="heroDriver"
                isLcp
                sizes="(max-width: 1024px) 100vw, 50vw"
                className="aspect-[16/11] w-full rounded-xl border border-border/80"
                overlayClassName="bg-gradient-to-t from-[#101211]/85 via-[#101211]/20 to-[#101211]/45"
              >
                {/* Steering Wheel & Driver Silhouette Depth Hint when image slot is blank */}
                <svg
                  viewBox="0 0 400 275"
                  aria-hidden="true"
                  className="pointer-events-none absolute inset-0 h-full w-full"
                >
                  <path
                    d="M 118 275 A 90 90 0 0 1 282 275"
                    fill="none"
                    stroke="rgba(245,243,236,0.12)"
                    strokeWidth="12"
                    strokeLinecap="round"
                  />
                  <path
                    d="M 136 275 A 72 72 0 0 1 264 275"
                    fill="none"
                    stroke="rgba(245,243,236,0.06)"
                    strokeWidth="1.5"
                  />

                  <ellipse
                    cx="200"
                    cy="132"
                    rx="46"
                    ry="58"
                    fill="rgba(25,29,26,0.45)"
                    stroke="rgba(245,243,236,0.16)"
                    strokeWidth="1"
                  />
                  <path
                    d="M 122 252 C 134 212 164 202 200 202 C 236 202 266 212 278 252"
                    fill="rgba(25,29,26,0.38)"
                    stroke="rgba(245,243,236,0.11)"
                    strokeWidth="1"
                  />

                  {/* Corner Targeting Brackets */}
                  <g stroke="#E58B43" strokeOpacity="0.7" strokeWidth="1.4" fill="none">
                    <path d="M 142 78 L 142 66 L 154 66" />
                    <path d="M 246 66 L 258 66 L 258 78" />
                    <path d="M 142 186 L 142 198 L 154 198" />
                    <path d="M 246 198 L 258 198 L 258 186" />
                  </g>

                  {/* Eye Tracking Contours & Reticle */}
                  <g stroke="#79B99A" strokeWidth="1.25" fill="rgba(121,185,154,0.08)">
                    <path d="M 171 122 Q 182 114 193 122 Q 182 130 171 122 Z" />
                    <path d="M 207 122 Q 218 114 229 122 Q 218 130 207 122 Z" />
                  </g>
                  <circle cx="182" cy="122" r="2.2" fill="#79B99A" />
                  <circle cx="218" cy="122" r="2.2" fill="#79B99A" />

                  {/* Subtle Facial Landmark Mesh Connections */}
                  <g stroke="rgba(245,243,236,0.18)" strokeWidth="0.75">
                    <line x1="185" y1="122" x2="200" y2="158" />
                    <line x1="215" y1="122" x2="200" y2="158" />
                    <line x1="170" y1="145" x2="200" y2="158" />
                    <line x1="230" y1="145" x2="200" y2="158" />
                    <line x1="184" y1="176" x2="216" y2="176" />
                    <line x1="184" y1="176" x2="200" y2="196" />
                    <line x1="216" y1="176" x2="200" y2="196" />
                  </g>

                  {HERO_LANDMARKS.map((pt, i) => (
                    <motion.circle
                      key={`${pt.x}-${pt.y}`}
                      cx={pt.x}
                      cy={pt.y}
                      r="1.9"
                      fill="#F5F3EC"
                      initial={{ opacity: 0.45 }}
                      animate={
                        prefersReducedMotion
                          ? { opacity: 0.75 }
                          : { opacity: [0.4, 0.95, 0.4] }
                      }
                      transition={
                        prefersReducedMotion
                          ? undefined
                          : {
                              duration: 2.8,
                              repeat: Infinity,
                              delay: i * 0.22,
                              ease: "easeInOut",
                            }
                      }
                    />
                  ))}
                </svg>

                {/* Subtle GPU-Composited Vertical Scan Beam */}
                {!prefersReducedMotion && (
                  <motion.div
                    aria-hidden="true"
                    className="pointer-events-none absolute inset-x-[28%] top-[20%] h-px bg-gradient-to-r from-transparent via-[#E58B43]/75 to-transparent shadow-[0_0_12px_rgba(229,139,67,0.45)]"
                    animate={{ y: [0, 128, 0] }}
                    transition={{ duration: 4.5, repeat: Infinity, ease: "easeInOut" }}
                  />
                )}

                {/* Top-Right Floating Status Card */}
                <div className="absolute top-3 right-3 sm:top-4 sm:right-4 z-10 rounded-xl border border-[#79B99A]/35 bg-[#101211]/88 px-3 py-2 backdrop-blur-md shadow-lg">
                  <div className="flex items-center gap-2">
                    <motion.span
                      className="size-2 rounded-full bg-[#79B99A]"
                      animate={
                        prefersReducedMotion ? undefined : { opacity: [1, 0.45, 1] }
                      }
                      transition={
                        prefersReducedMotion
                          ? undefined
                          : { duration: 2.2, repeat: Infinity, ease: "easeInOut" }
                      }
                    />
                    <span className="text-[10px] font-mono uppercase tracking-[0.12em] text-[#9EA39B]">
                      Driver Status
                    </span>
                  </div>
                  <p className="mt-0.5 text-xs font-mono font-semibold tracking-wide text-[#79B99A]">
                    ALERT · CENTER ZONE
                  </p>
                </div>

                {/* Bottom-Left Attention Score Visualization */}
                <div className="absolute bottom-3 left-3 sm:bottom-4 sm:left-4 z-10 flex items-center gap-3 rounded-xl border border-white/12 bg-[#101211]/88 px-3.5 py-2.5 backdrop-blur-md shadow-lg">
                  <div className="relative size-11 shrink-0">
                    <svg viewBox="0 0 44 44" className="size-full -rotate-90">
                      <circle
                        cx="22"
                        cy="22"
                        r="18"
                        fill="none"
                        stroke="rgba(245,243,236,0.12)"
                        strokeWidth="3.5"
                      />
                      <circle
                        cx="22"
                        cy="22"
                        r="18"
                        fill="none"
                        stroke="#79B99A"
                        strokeWidth="3.5"
                        strokeLinecap="round"
                        strokeDasharray={113.1}
                        strokeDashoffset={113.1 * 0.04}
                      />
                    </svg>
                    <span className="absolute inset-0 flex items-center justify-center text-[11px] font-mono font-bold tabular-nums text-[#F5F3EC]">
                      96
                    </span>
                  </div>
                  <div className="space-y-0.5">
                    <span className="block text-[10px] font-mono uppercase tracking-[0.12em] text-[#9EA39B]">
                      Attention Score
                    </span>
                    <div className="flex items-center gap-2 text-xs font-mono tabular-nums text-[#F5F3EC]">
                      <span>EAR 0.294</span>
                      <span className="text-[#F5F3EC]/30">·</span>
                      <span className="text-[#79B99A]">PERCLOS 3.2%</span>
                    </div>
                  </div>
                </div>
              </LandingMediaSlot>

              {/* Subtle Bottom Telemetry Strip */}
              <div className="mt-2.5 flex flex-wrap items-center justify-between gap-2 px-2 py-1 text-[11px] font-mono text-muted-foreground">
                <span className="inline-flex items-center gap-1.5">
                  <Sparkles className="size-3 text-primary-accent" />
                  <span>4-Zone Personalized Baseline Active</span>
                </span>
                <span className="text-muted-foreground/75">Illustrative HUD concept</span>
              </div>
            </motion.div>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
