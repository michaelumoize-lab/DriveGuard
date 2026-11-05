import React from "react";
import { UnifiedLiveMonitor } from "@/components/monitor/unified-live-monitor";

export const metadata = {
  title: "Live Driver Telemetry Monitor | DriveGuard",
  description:
    "Real-time webcam computer vision telemetry streaming via WebSocket to the DriveGuard Edge AI Cascade.",
};

export default function LiveMonitorPage() {
  return (
    <div className="p-3 sm:p-5 lg:p-6 max-w-[1600px] mx-auto w-full">
      <UnifiedLiveMonitor />
    </div>
  );
}
