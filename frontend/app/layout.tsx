import type { Metadata } from "next";
import { Geist } from "next/font/google";
import "./globals.css";
import type { ReactNode } from "react";

import { ThemeProvider } from "next-themes";
import { Providers } from "@/components/providers";
import { cn } from "@/lib/utils";

const geist = Geist({ subsets: ["latin"], variable: "--font-sans" });

export const metadata: Metadata = {
  title: {
    default: "DriveGuard - Personalised AI Driver Drowsiness Detection System",
    template: "%s | DriveGuard",
  },
  description:
    "Personalised Adaptive AI Driver Drowsiness Detection & Early Warning System using Computer Vision, 4-Zone Pose-Conditioned Calibration, and Confidence-Gated Cascades.",
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
      className={cn("font-sans", geist.variable)}
    >
      <body className="antialiased min-h-svh flex flex-col">
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
