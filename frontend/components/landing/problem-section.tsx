"use client";

import React from "react";
import { motion, useReducedMotion } from "motion/react";
import { AlertCircle, Compass, EyeOff } from "lucide-react";
import { LandingMediaSlot } from "@/components/landing/landing-media-slot";

const FATIGUE_SIGNALS = [
  {
    icon: EyeOff,
    code: "SIGNAL 01 // EAR + PERCLOS",
    title: "Prolonged eye closure",
    description:
      "Micro-sleeps begin milliseconds before the driver notices—measured through eyelid aperture collapse and rolling 60-second PERCLOS.",
  },
  {
    icon: AlertCircle,
    code: "SIGNAL 02 // MAR ENVELOPE",
    title: "Repeated yawning",
    description:
      "Sustained jaw-opening cycles reveal rising physiological fatigue long before lane drift or steering corrections appear.",
  },
  {
    icon: Compass,
    code: "SIGNAL 03 // 6-DOF HEAD POSE",
    title: "Head movement and reduced attention",
    description:
      "Forward pitch nods and prolonged off-road gaze shifts signal declining situational awareness on monotonous highways.",
  },
];

export function ProblemSection() {
  const prefersReducedMotion = useReducedMotion();

  return (
    <section className="py-16 md:py-24 border-t border-border">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="relative overflow-hidden rounded-3xl border border-border bg-card shadow-xl">
          {/* Cinematic Night Driving Media Container (Dark overlay ensures legible text in both Light & Dark modes) */}
          <LandingMediaSlot
            slot="nightFatigue"
            sizes="(max-width: 1280px) 100vw, 1280px"
            className="min-h-[440px] sm:min-h-[480px] flex flex-col justify-between p-6 sm:p-10 md:p-14"
            overlayClassName="bg-gradient-to-r from-[#101211]/95 via-[#101211]/82 to-[#101211]/65"
          >
            <div
              aria-hidden="true"
              className="pointer-events-none absolute inset-x-0 top-1/2 h-px bg-gradient-to-r from-transparent via-[#E58B43]/25 to-transparent"
            />

            <motion.div
              initial={prefersReducedMotion ? false : { opacity: 0, y: 18 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, amount: 0.3 }}
              transition={{ duration: 0.55, ease: [0.22, 1, 0.36, 1] }}
              className="relative z-10 max-w-2xl space-y-4"
            >
              <span className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-[#E58B43]">
                <span className="size-1.5 rounded-full bg-[#E58B43]" />
                The Reality Behind Highway Fatigue
              </span>
              <h2 className="font-display text-3xl sm:text-4xl md:text-5xl font-bold tracking-[-0.025em] text-[#F5F3EC] leading-[1.1] text-balance">
                Fatigue doesn&apos;t always announce itself.
              </h2>
              <p className="text-sm sm:text-base text-[#9EA39B] leading-relaxed max-w-xl">
                By the time a tired driver realizes their focus is slipping, reaction windows have already narrowed. DriveGuard continuously monitors subtle ocular and kinematic precursors before a lapse becomes critical.
              </p>
            </motion.div>

            {/* Three Scroll-Reveal Indicators */}
            <div className="relative z-10 mt-10 grid grid-cols-1 md:grid-cols-3 gap-4">
              {FATIGUE_SIGNALS.map((signal, index) => {
                const Icon = signal.icon;
                return (
                  <motion.div
                    key={signal.title}
                    initial={prefersReducedMotion ? false : { opacity: 0, y: 18 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true, amount: 0.25 }}
                    transition={{
                      duration: 0.45,
                      delay: prefersReducedMotion ? 0 : index * 0.1,
                      ease: [0.22, 1, 0.36, 1],
                    }}
                    className="rounded-2xl border border-white/12 bg-[#101211]/82 backdrop-blur-md p-5 space-y-3 hover:border-[#E58B43]/40 transition-colors"
                  >
                    <div className="flex items-center justify-between">
                      <span className="inline-flex items-center justify-center size-9 rounded-xl border border-white/12 bg-[#191D1A] text-[#E58B43]">
                        <Icon className="size-4" />
                      </span>
                      <span className="text-[10px] font-mono tracking-wider text-[#9EA39B]">
                        {signal.code}
                      </span>
                    </div>
                    <div className="space-y-1.5">
                      <h3 className="font-display text-base font-semibold text-[#F5F3EC]">
                        {signal.title}
                      </h3>
                      <p className="text-xs text-[#9EA39B] leading-relaxed">
                        {signal.description}
                      </p>
                    </div>
                  </motion.div>
                );
              })}
            </div>
          </LandingMediaSlot>
        </div>
      </div>
    </section>
  );
}
