"use client";

import React, { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { motion, useReducedMotion, useScroll, useTransform } from "motion/react";
import {
  Activity,
  ArrowUpRight,
  Compass,
  History,
  LayoutDashboard,
  ShieldCheck,
} from "lucide-react";
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { useIsHydrated } from "@/components/auth/use-is-hydrated";
import { LANDING_MEDIA, type LandingMediaSlotKey } from "@/lib/landing-media";
import { LandingMediaSlot } from "@/components/landing/landing-media-slot";
import { cn } from "@/lib/utils";

type PreviewTab = "overview" | "live";

const TREND_DATA = [
  { time: "00:00", attention: 96, perclos: 2.8 },
  { time: "05:00", attention: 95, perclos: 3.1 },
  { time: "10:00", attention: 94, perclos: 3.6 },
  { time: "15:00", attention: 91, perclos: 4.9 },
  { time: "20:00", attention: 84, perclos: 8.4 },
  { time: "25:00", attention: 89, perclos: 5.5 },
  { time: "30:00", attention: 95, perclos: 3.2 },
];

const SESSION_HISTORY = [
  {
    route: "Highway Corridor — Night",
    duration: "42m 18s",
    avgPerclos: "3.4%",
    events: "0 Critical · 1 Early",
    status: "NORMAL",
    color: "#79B99A",
  },
  {
    route: "Ring Road Commute",
    duration: "28m 05s",
    avgPerclos: "4.1%",
    events: "0 Critical · 0 Early",
    status: "NORMAL",
    color: "#79B99A",
  },
  {
    route: "Late Evening Transfer",
    duration: "54m 40s",
    avgPerclos: "9.8%",
    events: "1 Critical · 2 Early",
    status: "WARNING",
    color: "#E8A04B",
  },
];

const CALIBRATION_ZONES = [
  { zone: "CENTER", median: "0.295", thresh: "0.212" },
  { zone: "LEFT_MIRROR", median: "0.256", thresh: "0.184" },
  { zone: "RIGHT_MIRROR", median: "0.252", thresh: "0.181" },
  { zone: "DASHBOARD", median: "0.238", thresh: "0.171" },
];

export function DashboardPreviewSection() {
  const prefersReducedMotion = useReducedMotion();
  const [activeTab, setActiveTab] = useState<PreviewTab>("overview");
  const mounted = useIsHydrated();
  const frameRef = useRef<HTMLDivElement>(null);

  const { scrollYProgress } = useScroll({
    target: frameRef,
    offset: ["start end", "end start"],
  });

  const frameY = useTransform(scrollYProgress, [0, 1], [18, -18]);

  const activeSlot: LandingMediaSlotKey =
    activeTab === "overview" ? "dashboardOverview" : "dashboardLiveMonitor";
  const hasRealScreenshot = Boolean(LANDING_MEDIA[activeSlot].src);

  return (
    <section
      id="dashboard-preview"
      className="py-20 md:py-28 border-t border-[#F5F3EC]/8 scroll-mt-20"
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
        {/* Section Heading */}
        <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
          <div className="max-w-2xl space-y-3">
            <span className="inline-flex items-center gap-2 text-[11px] font-mono uppercase tracking-[0.18em] text-[#E58B43]">
              <LayoutDashboard className="size-3.5" />
              <span>Command Center &amp; Session Analytics</span>
            </span>
            <h2 className="font-display text-3xl sm:text-4xl font-bold tracking-[-0.025em] text-[#F5F3EC] leading-[1.12] text-balance">
              Complete visibility across every journey.
            </h2>
            <p className="text-sm sm:text-base text-[#9EA39B] leading-relaxed">
              Review monitoring session history, drowsiness events, attention trends, and your personalized 4-zone calibration profile in one unified cockpit.
            </p>
          </div>

          {/* View Switcher */}
          <div
            role="tablist"
            aria-label="Dashboard preview views"
            className="inline-flex flex-nowrap items-center gap-1 rounded-2xl border border-[#F5F3EC]/12 bg-[#191D1A] p-1.5 self-start lg:self-auto"
          >
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "overview"}
              onClick={() => setActiveTab("overview")}
              className={cn(
                "relative px-4 py-2 rounded-xl text-xs font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-[#E58B43]",
                activeTab === "overview"
                  ? "bg-[#101211] text-[#F5F3EC] border border-[#F5F3EC]/14"
                  : "text-[#9EA39B] hover:text-[#F5F3EC]"
              )}
            >
              Session Analytics
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={activeTab === "live"}
              onClick={() => setActiveTab("live")}
              className={cn(
                "relative px-4 py-2 rounded-xl text-xs font-semibold transition-colors outline-none focus-visible:ring-2 focus-visible:ring-[#E58B43]",
                activeTab === "live"
                  ? "bg-[#101211] text-[#F5F3EC] border border-[#F5F3EC]/14"
                  : "text-[#9EA39B] hover:text-[#F5F3EC]"
              )}
            >
              Live Cockpit View
            </button>
          </div>
        </div>

        {/* Large Unified Application Frame */}
        <motion.div
          ref={frameRef}
          style={prefersReducedMotion ? undefined : { y: frameY }}
          className="rounded-3xl border border-[#F5F3EC]/14 bg-[#191D1A] p-2.5 sm:p-4 shadow-[0_40px_100px_-30px_rgba(0,0,0,0.9)]"
        >
          {/* Application Window Chrome */}
          <div className="flex flex-wrap items-center justify-between gap-3 px-3 py-2.5 border-b border-[#F5F3EC]/10 mb-3">
            <div className="flex items-center gap-2">
              <span className="size-2.5 rounded-full bg-[#F5F3EC]/20" />
              <span className="size-2.5 rounded-full bg-[#F5F3EC]/20" />
              <span className="size-2.5 rounded-full bg-[#F5F3EC]/20" />
              <span className="ml-2 text-xs font-mono text-[#9EA39B]">
                driveguard.app/dashboard
                {activeTab === "live" ? "/live-monitor" : ""}
              </span>
            </div>

            <Link
              href={activeTab === "live" ? "/dashboard/live-monitor" : "/dashboard"}
              className="inline-flex items-center gap-1 text-[11px] font-mono text-[#E58B43] hover:underline"
            >
              <span>Open App</span>
              <ArrowUpRight className="size-3.5" />
            </Link>
          </div>

          {/* If real application screenshot is provided in LANDING_MEDIA, display it directly */}
          {hasRealScreenshot ? (
            <LandingMediaSlot
              slot={activeSlot}
              sizes="(max-width: 1280px) 100vw, 1200px"
              className="aspect-[16/10] w-full rounded-2xl border border-[#F5F3EC]/10"
              showSlotHint={false}
            />
          ) : (
            /* Otherwise render the authentic structured dark-mode application preview with drop-in slot hint */
            <div className="relative rounded-2xl border border-[#F5F3EC]/10 bg-[#101211] p-4 sm:p-6 space-y-6 overflow-hidden">
              {/* Top App Summary Strip */}
              <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 pb-5 border-b border-[#F5F3EC]/8">
                <div className="space-y-1">
                  <div className="inline-flex items-center gap-2 text-[11px] font-mono text-[#79B99A]">
                    <ShieldCheck className="size-3.5" />
                    <span>CALIBRATION ACTIVE · 4-ZONE PROFILE</span>
                  </div>
                  <h3 className="font-display text-lg sm:text-xl font-semibold text-[#F5F3EC]">
                    {activeTab === "overview"
                      ? "Driver Safety Overview & Session History"
                      : "Live WebRTC Optical Cockpit & Telemetry"}
                  </h3>
                </div>

                <div className="grid grid-cols-3 gap-4 sm:gap-6 text-left md:text-right">
                  <div>
                    <span className="block text-[10px] font-mono uppercase text-[#9EA39B]">
                      Monitored Time
                    </span>
                    <span className="font-mono text-base font-bold tabular-nums text-[#F5F3EC]">
                      14h 22m
                    </span>
                  </div>
                  <div>
                    <span className="block text-[10px] font-mono uppercase text-[#9EA39B]">
                      Avg PERCLOS
                    </span>
                    <span className="font-mono text-base font-bold tabular-nums text-[#79B99A]">
                      3.8%
                    </span>
                  </div>
                  <div>
                    <span className="block text-[10px] font-mono uppercase text-[#9EA39B]">
                      Critical Events
                    </span>
                    <span className="font-mono text-base font-bold tabular-nums text-[#F5F3EC]">
                      1
                    </span>
                  </div>
                </div>
              </div>

              {activeTab === "overview" ? (
                <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
                  {/* Left: Attention & PERCLOS Trend Chart */}
                  <div className="lg:col-span-7 rounded-2xl border border-[#F5F3EC]/10 bg-[#141715] p-4 sm:p-5 space-y-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <span className="text-xs font-semibold text-[#F5F3EC] block">
                          Driver Attention &amp; PERCLOS Trend
                        </span>
                        <span className="text-[11px] text-[#9EA39B]">
                          30-minute rolling session window (illustrative sample)
                        </span>
                      </div>
                      <Activity className="size-4 text-[#E58B43]" />
                    </div>

                    <div className="h-56 w-full">
                      {mounted ? (
                        <ResponsiveContainer width="100%" height="100%">
                          <AreaChart
                            data={TREND_DATA}
                            margin={{ top: 8, right: 8, left: -24, bottom: 0 }}
                          >
                            <defs>
                              <linearGradient id="dgAttentionFill" x1="0" y1="0" x2="0" y2="1">
                                <stop offset="5%" stopColor="#79B99A" stopOpacity={0.35} />
                                <stop offset="95%" stopColor="#79B99A" stopOpacity={0} />
                              </linearGradient>
                            </defs>
                            <CartesianGrid
                              stroke="rgba(245,243,236,0.06)"
                              strokeDasharray="3 3"
                              vertical={false}
                            />
                            <XAxis
                              dataKey="time"
                              stroke="#6E746C"
                              fontSize={11}
                              tickLine={false}
                              axisLine={false}
                            />
                            <YAxis
                              stroke="#6E746C"
                              fontSize={11}
                              tickLine={false}
                              axisLine={false}
                              domain={[60, 100]}
                            />
                            <Tooltip
                              contentStyle={{
                                backgroundColor: "#101211",
                                border: "1px solid rgba(245,243,236,0.14)",
                                borderRadius: "10px",
                                fontSize: "12px",
                                color: "#F5F3EC",
                              }}
                            />
                            <Area
                              type="monotone"
                              dataKey="attention"
                              name="Attention Score"
                              stroke="#79B99A"
                              strokeWidth={2}
                              fillOpacity={1}
                              fill="url(#dgAttentionFill)"
                              isAnimationActive={!prefersReducedMotion}
                            />
                          </AreaChart>
                        </ResponsiveContainer>
                      ) : (
                        <div className="h-full w-full rounded-xl bg-[#191D1A]/50" />
                      )}
                    </div>
                  </div>

                  {/* Right: Calibration Profile & Recent Sessions */}
                  <div className="lg:col-span-5 space-y-4">
                    {/* Calibration Profile Summary */}
                    <div className="rounded-2xl border border-[#F5F3EC]/10 bg-[#141715] p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-[#F5F3EC] inline-flex items-center gap-1.5">
                          <Compass className="size-3.5 text-[#E58B43]" />
                          4-Zone Calibration Profile
                        </span>
                        <span className="text-[10px] font-mono text-[#79B99A]">VERIFIED</span>
                      </div>
                      <div className="grid grid-cols-2 gap-2">
                        {CALIBRATION_ZONES.map((z) => (
                          <div
                            key={z.zone}
                            className="rounded-xl border border-[#F5F3EC]/8 bg-[#101211] px-3 py-2"
                          >
                            <span className="block text-[10px] font-mono text-[#9EA39B]">
                              {z.zone}
                            </span>
                            <span className="text-xs font-mono font-semibold tabular-nums text-[#F5F3EC]">
                              Thresh {z.thresh}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>

                    {/* Monitoring Session History & Drowsiness Events */}
                    <div className="rounded-2xl border border-[#F5F3EC]/10 bg-[#141715] p-4 space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="text-xs font-semibold text-[#F5F3EC] inline-flex items-center gap-1.5">
                          <History className="size-3.5 text-[#9EA39B]" />
                          Recent Sessions &amp; Events
                        </span>
                      </div>
                      <div className="space-y-2">
                        {SESSION_HISTORY.map((item) => (
                          <div
                            key={item.route}
                            className="flex items-center justify-between gap-2 rounded-xl border border-[#F5F3EC]/8 bg-[#101211] px-3 py-2.5 text-xs"
                          >
                            <div className="min-w-0">
                              <span className="block font-medium text-[#F5F3EC] truncate">
                                {item.route}
                              </span>
                              <span className="block text-[11px] font-mono text-[#9EA39B]">
                                {item.duration} · {item.events}
                              </span>
                            </div>
                            <span
                              className="shrink-0 rounded-md px-2 py-0.5 text-[10px] font-mono font-semibold"
                              style={{
                                color: item.color,
                                backgroundColor: "rgba(245,243,236,0.04)",
                                border: `1px solid ${item.color}44`,
                              }}
                            >
                              {item.status}
                            </span>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </div>
              ) : (
                /* Live Cockpit Blank Screenshot Slot */
                <LandingMediaSlot
                  slot="dashboardLiveMonitor"
                  isHiddenPanel
                  sizes="(max-width: 1280px) 100vw, 1140px"
                  className="aspect-[16/9] w-full rounded-2xl border border-[#F5F3EC]/10 flex items-center justify-center p-6"
                >
                  <div className="relative z-10 text-center max-w-md space-y-2">
                    <span className="inline-flex items-center gap-2 rounded-full border border-[#E58B43]/35 bg-[#E58B43]/10 px-3 py-1 text-[11px] font-mono text-[#E58B43]">
                      LIVE MONITOR SCREENSHOT SLOT
                    </span>
                    <p className="text-sm font-medium text-[#F5F3EC]">
                      Capture your real `/dashboard/live-monitor` view and save as{" "}
                      <code className="font-mono text-xs text-[#E58B43]">
                        public/images/landing/dashboard-live-monitor.png
                      </code>
                    </p>
                    <p className="text-xs text-[#9EA39B]">
                      Update <code className="font-mono">lib/landing-media.ts</code> to display your live application screenshot here.
                    </p>
                  </div>
                </LandingMediaSlot>
              )}

              {/* Subtle drop-in hint for overview screenshot */}
              <div className="flex flex-wrap items-center justify-between gap-2 pt-2 border-t border-[#F5F3EC]/8 text-[11px] font-mono text-[#6E746C]">
                <span>Illustrative application preview · Replace anytime via lib/landing-media.ts</span>
                <span>Target slot: media/{LANDING_MEDIA[activeSlot].placeholderFilename}</span>
              </div>
            </div>
          )}
        </motion.div>
      </div>
    </section>
  );
}
