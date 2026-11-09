import Link from "next/link";
import { ArrowLeft, ShieldCheck } from "lucide-react";
import { getServerSession } from "@/lib/get-session";
import { LandingHeaderAuth } from "@/components/landing-header-auth";
import { Logo } from "@/components/logo";
import HelpPage from "@/app/dashboard/help/page";

export default async function PublicHelpPage() {
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
    <div className="min-h-dvh bg-background text-foreground flex flex-col">
      <header className="border-b border-border/60 bg-background/80 backdrop-blur-md sticky top-0 z-40">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Link href="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
              <Logo className="size-9 rounded-xl" />
              <div>
                <span className="font-bold tracking-tight text-base block text-foreground">DriveGuard</span>
                <span className="text-[10px] font-mono text-muted-foreground uppercase tracking-wider block">
                  System Documentation
                </span>
              </div>
            </Link>
          </div>

          <div className="flex items-center gap-3">
            <Link
              href="/"
              className="text-xs font-medium text-muted-foreground hover:text-foreground inline-flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg hover:bg-muted transition-colors"
            >
              <ArrowLeft className="size-3.5" />
              <span>Home</span>
            </Link>
            <LandingHeaderAuth user={user} />
          </div>
        </div>
      </header>

      <main id="main-content" className="flex-1">
        <HelpPage />
      </main>

      <footer className="border-t border-border/60 py-6 text-center text-xs text-muted-foreground">
        <div className="max-w-5xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
          <span>DriveGuard &copy; {new Date().getFullYear()} — Personalised Adaptive AI Driver Drowsiness Detection.</span>
          <span className="inline-flex items-center gap-1 text-muted-foreground">
            <ShieldCheck className="size-3.5 text-primary" /> Academic Research Prototype — Not a certified medical/ADAS device.
          </span>
        </div>
      </footer>
    </div>
  );
}
