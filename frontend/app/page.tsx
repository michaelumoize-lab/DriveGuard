import Link from "next/link";
import {
  ShieldCheck,
  ArrowRight,
  LayoutDashboard,
  Sparkles,
  Activity,
  Eye,
  CheckCircle2,
  Clock,
  Car,
  TrendingDown,
  ShieldAlert,
} from "lucide-react";
import { getServerSession } from "@/lib/get-session";
import { LandingHeaderAuth } from "@/components/landing-header-auth";
import { Logo } from "@/components/logo";

export default async function Home() {
  const session = await getServerSession();
  const user = session?.user
    ? {
        id: session.user.id,
        name: session.user.name,
        email: session.user.email,
        image: session.user.image,
      }
    : null;

  return (
    <div className="min-h-screen bg-background text-foreground flex flex-col selection:bg-primary/20 selection:text-primary">
      {/* Top Portal Navigation */}
      <header className="border-b border-border/60 bg-background/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Logo className="size-9 rounded-xl" />
            <div>
              <span className="font-bold tracking-tight text-base block text-foreground">IntelliGuard</span>
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider block">
                Driver Safety System
              </span>
            </div>
          </div>

          <LandingHeaderAuth user={user} />
        </div>
      </header>

      {/* Hero Section */}
      <main className="flex-1">
        <section className="relative pt-16 pb-20 md:pt-24 md:pb-32 overflow-hidden">
          {/* Subtle Grid Accent */}
          <div className="absolute inset-0 -z-10 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto space-y-6">
              {/* Badge */}
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-medium">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Personalised Adaptive Driver Monitoring System</span>
              </div>

              {/* Title */}
              <h1 className="text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-foreground leading-[1.15]">
                Intelligent Adaptive AI <br className="hidden sm:inline" />
                <span className="bg-gradient-to-r from-primary via-blue-500 to-indigo-500 bg-clip-text text-transparent">
                  Driver Drowsiness Early Warning
                </span>
              </h1>

              {/* Description */}
              <p className="text-sm sm:text-base md:text-lg text-muted-foreground leading-relaxed max-w-2xl mx-auto">
                Next-generation automotive safety framework combining 4-zone pose-conditioned calibration,
                sub-frame blink kinematics, rolling time-weighted PERCLOS, and confidence-gated edge deep learning.
              </p>

              {/* Action Buttons */}
              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
                <Link
                  href="/dashboard"
                  className="w-full sm:w-auto px-6 py-3 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-sm rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                >
                  <LayoutDashboard className="w-4 h-4" /> Enter Safety Dashboard
                </Link>

                <Link
                  href="/dashboard/live-monitor"
                  className="w-full sm:w-auto px-6 py-3 bg-background hover:bg-muted text-foreground font-semibold text-sm rounded-xl border border-border transition-all flex items-center justify-center gap-2"
                >
                  <Activity className="w-4 h-4 text-primary" />
                  <span>Launch Live Monitor</span>
                  <ArrowRight className="w-4 h-4 text-muted-foreground" />
                </Link>
              </div>
            </div>

            {/* Dashboard Mockup Preview */}
            <div className="mt-12 sm:mt-16 max-w-5xl mx-auto">
              <div className="rounded-2xl border border-border bg-card/80 backdrop-blur-xl p-4 sm:p-6 shadow-2xl relative overflow-hidden space-y-5">
                <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

                {/* Top Controls */}
                <div className="flex items-center justify-between pb-3 border-b border-border/60">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-destructive/60 inline-block" />
                    <span className="w-3 h-3 rounded-full bg-amber-500/60 inline-block" />
                    <span className="w-3 h-3 rounded-full bg-emerald-500/60 inline-block" />
                    <span className="text-xs font-mono text-muted-foreground ml-2 hidden sm:inline">
                      intelliguard://driver-telemetry
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs font-mono text-emerald-500">
                    <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                    <span className="font-semibold">AI CASCADE • ACTIVE</span>
                  </div>
                </div>

                {/* Mini System Health Panel */}
                <div className="p-3 rounded-xl border border-border bg-background/50 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-muted-foreground">AI Cascade Engine:</span>
                    <span className="font-mono font-semibold text-foreground">Online (24.2 ms)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-muted-foreground">MobileNetV3 ONNX:</span>
                    <span className="font-mono font-semibold text-foreground">Operational</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-emerald-500" />
                    <span className="text-muted-foreground">4-Zone Baselines:</span>
                    <span className="font-mono font-semibold text-foreground">Calibrated</span>
                  </div>
                </div>

                {/* Mini Metric Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-[11px] text-muted-foreground block">Active Gaze Zone</span>
                    <span className="text-base font-bold text-foreground">CENTER</span>
                  </div>
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-[11px] text-muted-foreground block">Eye Aspect Ratio</span>
                    <span className="text-base font-mono font-bold text-emerald-500">0.285 (Open)</span>
                  </div>
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-[11px] text-muted-foreground block">Rolling PERCLOS (60s)</span>
                    <span className="text-base font-mono font-bold text-foreground">3.8% (Alert)</span>
                  </div>
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-[11px] text-muted-foreground block">Cascade Mode</span>
                    <span className="text-base font-bold text-primary">BYPASS (Power Save)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/60 py-6 text-center text-xs text-muted-foreground">
        IntelliGuard &copy; {new Date().getFullYear()} — Computer Engineering Final-Year Project.
      </footer>
    </div>
  );
}
