import { getServerSession } from "@/lib/get-session";
import { LandingMotionProvider } from "@/components/landing/landing-motion-provider";
import { LandingNavbar } from "@/components/landing/landing-navbar";
import { HeroSection } from "@/components/landing/hero-section";
import { ProblemSection } from "@/components/landing/problem-section";
import { VisionDemoSection } from "@/components/landing/vision-demo-section";
import { CalibrationSection } from "@/components/landing/calibration-section";
import { AlertProgressionSection } from "@/components/landing/alert-progression-section";
import { DashboardPreviewSection } from "@/components/landing/dashboard-preview-section";
import { FinalCtaFooter } from "@/components/landing/final-cta-footer";

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

  const isAuthenticated = Boolean(user);

  return (
    <LandingMotionProvider>
      <div className="driveguard-landing min-h-dvh bg-[#101211] text-[#F5F3EC] flex flex-col selection:bg-[#E58B43]/25 selection:text-[#F5F3EC]">
        <LandingNavbar user={user} />

        <main id="main-content" className="flex-1">
          <HeroSection isAuthenticated={isAuthenticated} />
          <ProblemSection />
          <VisionDemoSection />
          <CalibrationSection />
          <AlertProgressionSection />
          <DashboardPreviewSection />
        </main>

        <FinalCtaFooter isAuthenticated={isAuthenticated} />
      </div>
    </LandingMotionProvider>
  );
}
