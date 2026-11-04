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
    default: "IntelliGuard - AI-Powered Access Control & Security Monitoring",
    template: "%s | IntelliGuard",
  },
  description:
    "AI-powered intelligent access control and security monitoring system with real-time biometric verification and edge device management.",
  keywords: [
    "IntelliGuard",
    "Access Control",
    "Security Monitoring",
    "Facial Recognition",
    "ESP32-S3 CAM",
    "Biometrics",
    "Next.js",
    "FastAPI",
    "PostgreSQL",
    "Vector Search",
    "IoT",
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
