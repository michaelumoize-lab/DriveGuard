import type { Metadata, Viewport } from "next";
import { DM_Sans, Geist, Geist_Mono, Manrope } from "next/font/google";
import "./globals.css";
import type { ReactNode } from "react";

import { ThemeProvider } from "@/components/theme-provider";
import { Providers } from "@/components/providers";
import { cn } from "@/lib/utils";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });
const geistMono = Geist_Mono({ subsets: ["latin"], variable: "--font-geist-mono" });
const manrope = Manrope({ subsets: ["latin"], variable: "--font-manrope" });
const dmSans = DM_Sans({ subsets: ["latin"], variable: "--font-dm-sans" });

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 5,
  viewportFit: "cover",
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f6f4ee" },
    { media: "(prefers-color-scheme: dark)", color: "#101211" },
  ],
};

export const metadata: Metadata = {
  applicationName: "DriveGuard",
  other: {
    "color-scheme": "light dark",
  },
  title: {
    default: "DriveGuard - Personalised AI Driver Drowsiness Detection System",
    template: "%s | DriveGuard",
  },
  description:
    "Personalised Adaptive AI Driver Drowsiness Detection & Early Warning System using Computer Vision, 4-Zone Pose-Conditioned Calibration, and Confidence-Gated Cascades.",
  appleWebApp: {
    capable: true,
    statusBarStyle: "black-translucent",
    title: "DriveGuard",
  },
  formatDetection: {
    telephone: false,
  },
  keywords: [
    "DriveGuard",
    "Driver Safety",
    "Drowsiness Detection",
    "Driver Monitoring System",
    "Computer Vision",
    "MediaPipe",
    "MobileNetV3",
    "Edge AI",
    "Next.js",
    "FastAPI",
    "PostgreSQL",
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: ReactNode;
}>) {
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={cn(
        "font-sans",
        geist.variable,
        geistMono.variable,
        manrope.variable,
        dmSans.variable
      )}
    >
      <body className="antialiased min-h-svh flex flex-col">
        <a
          href="#main-content"
          className="sr-only focus:not-sr-only focus:fixed focus:top-3 focus:left-3 focus:z-[100] focus:rounded-md focus:bg-primary focus:px-4 focus:py-2 focus:text-xs focus:font-semibold focus:text-primary-foreground focus:shadow-lg"
        >
          Skip to main content
        </a>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <Providers>{children}</Providers>
        </ThemeProvider>
      </body>
    </html>
  );
}
