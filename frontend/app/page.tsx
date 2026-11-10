import Link from "next/link";
import {
  ShieldCheck,
  ArrowRight,
  LayoutDashboard,
  Sparkles,
  Activity,
  Eye,
  CheckCircle2,
  Cpu,
  Volume2,
  Compass,
  Sliders,
  Lock,
  AlertTriangle,
  CircleHelp,
} from "lucide-react";
import { getServerSession } from "@/lib/get-session";
import { LandingHeaderAuth } from "@/components/landing-header-auth";
import { Logo } from "@/components/logo";
import { Badge } from "@/components/ui/badge";

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
    <div className="min-h-dvh bg-background text-foreground flex flex-col selection:bg-primary/20 selection:text-primary">
      {/* Top Portal Navigation */}
      <header className="border-b border-border/60 bg-background/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-3">
          <div className="flex items-center gap-3">
            <Logo className="size-9 rounded-xl" />
            <div>
              <span className="font-bold tracking-tight text-base block text-foreground">DriveGuard</span>
              <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider block">
                Driver Safety System
              </span>
            </div>
          </div>

          <nav aria-label="Public Navigation" className="hidden md:flex items-center gap-6 text-xs font-medium text-muted-foreground">
            <a href="#pipeline" className="hover:text-foreground transition-colors">How It Works</a>
            <a href="#personalization" className="hover:text-foreground transition-colors">4-Zone Calibration</a>
            <a href="#safety-privacy" className="hover:text-foreground transition-colors">Privacy & Safety</a>
            <Link href="/help" className="hover:text-foreground transition-colors inline-flex items-center gap-1">
              <CircleHelp className="size-3.5" /> System Guide
            </Link>
          </nav>

          <LandingHeaderAuth user={user} />
        </div>
      </header>

      {/* Main Content */}
      <main id="main-content" className="flex-1">
        {/* Hero Section */}
        <section className="relative pt-14 pb-16 md:pt-20 md:pb-24 overflow-hidden">
          <div className="absolute inset-0 -z-10 bg-[linear-gradient(to_right,#8080800a_1px,transparent_1px),linear-gradient(to_bottom,#8080800a_1px,transparent_1px)] bg-[size:24px_24px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="text-center max-w-3xl mx-auto space-y-6">
              <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full border border-primary/20 bg-primary/5 text-primary text-xs font-medium">
                <Sparkles className="w-3.5 h-3.5" />
                <span>Personalised Adaptive AI Driver Monitoring Platform</span>
              </div>

              <h1 className="text-3xl sm:text-5xl md:text-6xl font-extrabold tracking-tight text-foreground leading-[1.12] text-balance">
                Pose-Conditioned AI <br className="hidden sm:inline" />
                <span className="bg-gradient-to-r from-primary via-blue-500 to-indigo-500 bg-clip-text text-transparent">
                  Driver Drowsiness Early Warning
                </span>
              </h1>

              <p className="text-sm sm:text-base md:text-lg text-muted-foreground leading-relaxed max-w-2xl mx-auto text-pretty">
                Eliminating mirror-check false alarms through 4-zone personalized baselines, sub-frame blink velocity kinematics, rolling 60-second PERCLOS, and a confidence-gated MobileNetV3-Small ONNX cascade over LiveKit WebRTC.
              </p>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
                {user ? (
                  <>
                    <Link
                      href="/dashboard/live-monitor"
                      className="w-full sm:w-auto px-6 py-3 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-sm rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                    >
                      <Activity className="w-4 h-4" /> Launch Live Monitor
                    </Link>
                    <Link
                      href="/dashboard"
                      className="w-full sm:w-auto px-6 py-3 bg-background hover:bg-muted text-foreground font-semibold text-sm rounded-xl border border-border transition-all flex items-center justify-center gap-2"
                    >
                      <LayoutDashboard className="w-4 h-4 text-primary" />
                      <span>Open Safety Dashboard</span>
                    </Link>
                  </>
                ) : (
                  <>
                    <Link
                      href="/auth/sign-in"
                      className="w-full sm:w-auto px-6 py-3 bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-sm rounded-xl transition-all flex items-center justify-center gap-2 shadow-sm"
                    >
                      <ShieldCheck className="w-4 h-4" /> Sign In / Get Started
                    </Link>
                    <Link
                      href="/help"
                      className="w-full sm:w-auto px-6 py-3 bg-background hover:bg-muted text-foreground font-semibold text-sm rounded-xl border border-border transition-all flex items-center justify-center gap-2"
                    >
                      <CircleHelp className="w-4 h-4 text-primary" />
                      <span>Explore System Architecture</span>
                      <ArrowRight className="w-4 h-4 text-muted-foreground" />
                    </Link>
                  </>
                )}
              </div>
            </div>

            {/* Explicitly Labeled Illustrative Telemetry Preview */}
            <div className="mt-12 sm:mt-16 max-w-5xl mx-auto">
              <div className="rounded-2xl border border-border bg-card/80 backdrop-blur-xl p-4 sm:p-6 shadow-2xl relative overflow-hidden space-y-5">
                <div className="absolute top-0 right-0 w-96 h-96 bg-primary/5 rounded-full blur-3xl pointer-events-none" />

                {/* Top Controls with Explicit Demo Simulation Label */}
                <div className="flex flex-wrap items-center justify-between gap-2 pb-3 border-b border-border/60">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded-full bg-destructive/60 inline-block" />
                    <span className="w-3 h-3 rounded-full bg-amber-500/60 inline-block" />
                    <span className="w-3 h-3 rounded-full bg-emerald-500/60 inline-block" />
                    <span className="text-xs font-mono text-muted-foreground ml-2 hidden sm:inline">
                      driveguard://telemetry-hud-preview
                    </span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-[11px] font-mono border-amber-500/40 bg-amber-500/10 text-amber-600 dark:text-amber-300">
                      ILLUSTRATIVE TELEMETRY PREVIEW • SAMPLE DATA
                    </Badge>
                  </div>
                </div>

                {/* Illustrative Subsystem Architecture Row */}
                <div className="p-3 rounded-xl border border-border bg-background/50 flex flex-wrap items-center justify-between gap-3 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary" />
                    <span className="text-muted-foreground">Stage 1 Geometry:</span>
                    <span className="font-mono tabular-nums font-semibold text-foreground">478 3D Landmarks (~2.5 ms)</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary" />
                    <span className="text-muted-foreground">Stage 2 Verifier:</span>
                    <span className="font-mono font-semibold text-foreground">MobileNetV3-Small ONNX</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary" />
                    <span className="text-muted-foreground">Spatial Context:</span>
                    <span className="font-mono font-semibold text-foreground">4-Zone PDBP Baselines</span>
                  </div>
                </div>

                {/* Sample Metric Cards */}
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-xs text-muted-foreground block">Sample Gaze Zone</span>
                    <span className="text-base font-bold font-mono text-foreground">LEFT_MIRROR</span>
                    <span className="text-[11px] text-emerald-500 block mt-0.5">Glance &le; 3.0s Suppressed</span>
                  </div>
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-xs text-muted-foreground block">Sample Zone EAR</span>
                    <span className="text-base font-mono tabular-nums font-bold text-emerald-500">0.248 (Open)</span>
                    <span className="text-[11px] text-muted-foreground font-mono block mt-0.5">Zone Thresh: 0.189</span>
                  </div>
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-xs text-muted-foreground block">Sample Rolling PERCLOS</span>
                    <span className="text-base font-mono tabular-nums font-bold text-foreground">4.1% (Normal)</span>
                    <span className="text-[11px] text-muted-foreground block mt-0.5">60s Time-Weighted</span>
                  </div>
                  <div className="p-3 rounded-xl border border-border/80 bg-background/40">
                    <span className="text-xs text-muted-foreground block">Sample Cascade Arbitration</span>
                    <span className="text-base font-bold text-primary">GEOMETRY</span>
                    <span className="text-[11px] text-muted-foreground block mt-0.5">CNN Bypassed (High Conf)</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 2: How DriveGuard Works (6-Stage Pipeline) */}
        <section id="pipeline" className="py-16 border-t border-border/60 bg-muted/20 scroll-mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-10">
            <div className="text-center max-w-3xl mx-auto space-y-2">
              <Badge variant="outline" className="font-mono text-xs uppercase">End-to-End AI Pipeline</Badge>
              <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
                From WebRTC Camera Stream to Explainable Risk State
              </h2>
              <p className="text-sm text-muted-foreground">
                Every video frame traverses a deterministic, low-latency computer vision and physiological modeling pipeline.
              </p>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5">
              <div className="p-5 rounded-xl border border-border bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <span className="p-2 rounded-lg bg-primary/10 text-primary"><Eye className="size-5" /></span>
                  <span className="text-xs font-mono text-muted-foreground">STAGE 01</span>
                </div>
                <h3 className="font-bold text-base text-foreground">3D Landmarks & Head Pose</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  MediaPipe Face Landmarker extracts 478 3D coordinates and solves 6DoF Euler head orientation (Pitch, Yaw, Roll) alongside yaw-weighted Eye Aspect Ratio (EAR).
                </p>
              </div>

              <div className="p-5 rounded-xl border border-border bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <span className="p-2 rounded-lg bg-primary/10 text-primary"><Compass className="size-5" /></span>
                  <span className="text-xs font-mono text-muted-foreground">STAGE 02</span>
                </div>
                <h3 className="font-bold text-base text-foreground">4-Zone PDBP Calibration</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Maps driver gaze into CENTER, LEFT_MIRROR, RIGHT_MIRROR, or DASHBOARD and applies the driver&apos;s personal median and MAD closure threshold for that specific zone.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-border bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <span className="p-2 rounded-lg bg-primary/10 text-primary"><Cpu className="size-5" /></span>
                  <span className="text-xs font-mono text-muted-foreground">STAGE 03</span>
                </div>
                <h3 className="font-bold text-base text-foreground">Confidence-Gated CNN Cascade</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Uses fast geometric probability when landmark reliability is high, and selectively triggers MobileNetV3-Small ONNX inference on ambiguous or glare-affected frames.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-border bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <span className="p-2 rounded-lg bg-primary/10 text-primary"><Activity className="size-5" /></span>
                  <span className="text-xs font-mono text-muted-foreground">STAGE 04</span>
                </div>
                <h3 className="font-bold text-base text-foreground">Blink Kinematics & 60s PERCLOS</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Tracks continuous closure duration, blink reopening velocity ratio (RCR), abnormal blink clusters, and observation-coverage-aware 60-second rolling PERCLOS.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-border bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <span className="p-2 rounded-lg bg-primary/10 text-primary"><Sliders className="size-5" /></span>
                  <span className="text-xs font-mono text-muted-foreground">STAGE 05</span>
                </div>
                <h3 className="font-bold text-base text-foreground">Hysteresis Risk State Machine</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Combines directional robust z-scores into a composite fatigue score with persistence timers, microsleep overrides (&ge; 1.1s), and explainable machine reason codes.
                </p>
              </div>

              <div className="p-5 rounded-xl border border-border bg-card space-y-2">
                <div className="flex items-center justify-between">
                  <span className="p-2 rounded-lg bg-primary/10 text-primary"><Volume2 className="size-5" /></span>
                  <span className="text-xs font-mono text-muted-foreground">STAGE 06</span>
                </div>
                <h3 className="font-bold text-base text-foreground">Reliable ACK & Web Audio Siren</h3>
                <p className="text-xs text-muted-foreground leading-relaxed">
                  Dispatches state transitions over reliable WebRTC DataChannels with client ACK/retry and triggers multi-tier synthesized Web Audio API acoustic warnings.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Section 3: Why 4-Zone Personalization Matters */}
        <section id="personalization" className="py-16 border-t border-border/60 scroll-mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8 items-center">
              <div className="space-y-4">
                <Badge variant="outline" className="font-mono text-xs uppercase">Personalised vs Static Thresholds</Badge>
                <h2 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-foreground">
                  Why Fixed EAR &lt; 0.22 Fails During Real Driving
                </h2>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  When a driver checks their side mirrors or glances down at the instrument panel, perspective foreshortening and downward eyelid posture naturally reduce raw Eye Aspect Ratio below 0.22—even when the driver is wide awake.
                </p>
                <ul className="space-y-2.5 text-xs text-muted-foreground">
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="size-4 text-emerald-500 shrink-0 mt-0.5" />
                    <span><strong className="text-foreground">Zone-Specific Closure Thresholds:</strong> Calibrates independent EAR medians and dynamic thresholds (72% of zone median) for CENTER, LEFT_MIRROR, RIGHT_MIRROR, and DASHBOARD.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="size-4 text-emerald-500 shrink-0 mt-0.5" />
                    <span><strong className="text-foreground">3.0s Mirror-Glance Suppression:</strong> Normal transient mirror checks (&le; 3.0s) use the mirror zone&apos;s baseline and dampen false head-pose penalties.</span>
                  </li>
                  <li className="flex items-start gap-2">
                    <CheckCircle2 className="size-4 text-emerald-500 shrink-0 mt-0.5" />
                    <span><strong className="text-foreground">Live Side-by-Side Comparison:</strong> Every frame is simultaneously evaluated by both DriveGuard and a conventional fixed-0.22 detector so researchers can observe false-alarm suppression in real time.</span>
                  </li>
                </ul>
              </div>

              <div className="p-6 rounded-2xl border border-border bg-card space-y-4">
                <div className="text-xs font-mono uppercase text-muted-foreground">Live Comparison Scenario: Driver Glances at Left Mirror</div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  <div className="p-4 rounded-xl border border-rose-500/30 bg-rose-500/5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground">Conventional Static Rule</span>
                      <Badge variant="destructive" className="text-[10px] font-mono">FALSE ALARM</Badge>
                    </div>
                    <p className="text-xs font-mono text-muted-foreground">Rule: Fixed EAR &lt; 0.220</p>
                    <p className="text-xs text-muted-foreground">
                      Mirror glance drops apparent EAR to <strong className="font-mono text-foreground">0.205</strong> &rarr; falsely triggers a drowsiness warning during a safe driving check.
                    </p>
                  </div>

                  <div className="p-4 rounded-xl border border-emerald-500/30 bg-emerald-500/5 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-bold text-foreground">DriveGuard 4-Zone PDBP</span>
                      <Badge variant="outline" className="text-[10px] font-mono border-emerald-500/40 text-emerald-500">SUPPRESSED</Badge>
                    </div>
                    <p className="text-xs font-mono text-muted-foreground">Zone: LEFT_MIRROR (Thresh: 0.184)</p>
                    <p className="text-xs text-muted-foreground">
                      Recognizes <strong className="font-mono text-foreground">LEFT_MIRROR</strong> context (&le; 3.0s) and compares 0.205 against personal zone threshold 0.184 &rarr; remains <strong className="text-emerald-500">NORMAL</strong>.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Section 4: Privacy & Safety Prototype Disclaimer */}
        <section id="safety-privacy" className="py-12 border-t border-border/60 bg-muted/10 scroll-mt-16">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div className="p-5 rounded-xl border border-border bg-card flex items-start gap-3.5">
                <Lock className="size-5 text-primary shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-foreground">Ephemeral In-Memory Video Processing</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    Camera video streamed via LiveKit WebRTC is processed strictly in volatile memory (`maxsize=1` latest-frame buffer) to extract facial landmarks. Raw video feeds and facial images are never recorded or stored in the database—only numeric calibration baselines, session summaries, and alert timestamps are persisted.
                  </p>
                </div>
              </div>

              <div className="p-5 rounded-xl border border-amber-500/30 bg-amber-500/5 flex items-start gap-3.5">
                <AlertTriangle className="size-5 text-amber-500 shrink-0 mt-0.5" />
                <div className="space-y-1">
                  <h3 className="text-sm font-bold text-foreground">Research Prototype Safety Notice</h3>
                  <p className="text-xs text-muted-foreground leading-relaxed">
                    DriveGuard is an academic computer engineering research and demonstration platform. It is not a certified automotive safety system (ISO 26262) or medical device. Drivers remain solely responsible for safe vehicle operation and must never rely on prototype software to drive while fatigued.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </section>
      </main>

      {/* Footer */}
      <footer className="border-t border-border/60 py-6 text-xs text-muted-foreground">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-center justify-between gap-3">
          <span>DriveGuard &copy; {new Date().getFullYear()} — Computer Engineering Final-Year Project.</span>
          <div className="flex items-center gap-4">
            <Link href="/help" className="hover:text-foreground transition-colors">Documentation & FAQ</Link>
            <Link href="/auth/sign-in" className="hover:text-foreground transition-colors">Sign In</Link>
          </div>
        </div>
      </footer>
    </div>
  );
}
