"use client";

import * as React from "react";
import { ThemeProvider as NextThemesProvider } from "next-themes";

// Filter out React 19 false-positive development warning for next-themes inline script
// and TensorFlow Lite / MediaPipe WASM informational stderr logs (which Emscripten directs to console.error)
if (typeof window !== "undefined") {
  const originalConsoleError = console.error;
  console.error = (...args: unknown[]) => {
    const fullMessage = args
      .map((a) => (typeof a === "string" ? a : typeof a === "object" && a !== null ? (a as Error).message || "" : ""))
      .join(" ");

    if (fullMessage.includes("Encountered a script tag while rendering React component")) {
      return;
    }

    // MediaPipe Vision WASM & TFLite C++ route normal startup diagnostics (e.g. XNNPACK delegate creation)
    // to stderr, which Emscripten forwards to console.error. In Next.js dev mode, console.error triggers
    // the Turbopack runtime error modal. We reroute these informational lines to console.info.
    if (
      fullMessage.startsWith("INFO:") ||
      fullMessage.includes("Created TensorFlow Lite XNNPACK delegate") ||
      fullMessage.includes("TensorFlow Lite") ||
      fullMessage.includes("Logging before InitGoogleLogging")
    ) {
      console.info(...args);
      return;
    }

    originalConsoleError.apply(console, args);
  };
}

export function ThemeProvider({
  children,
  ...props
}: React.ComponentProps<typeof NextThemesProvider>) {
  return <NextThemesProvider {...props}>{children}</NextThemesProvider>;
}
