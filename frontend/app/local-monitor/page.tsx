import type { Metadata } from "next";
import { LocalMonitorShell } from "@/components/pwa/local-monitor-shell";

export const metadata: Metadata = {
  title: "Offline Local Monitor",
  description:
    "Device-isolated, offline-first DriveGuard drowsiness detection using on-device MediaPipe Browser AI and IndexedDB calibration storage.",
};

export default function LocalMonitorPage() {
  return (
    <main
      id="main-content"
      className="min-h-svh bg-background text-foreground pt-safe pb-safe px-safe"
    >
      <LocalMonitorShell />
    </main>
  );
}
