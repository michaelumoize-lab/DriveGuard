import React from "react";
import { UnifiedLiveMonitor } from "@/components/monitor/unified-live-monitor";

export const metadata = {
  title: "Live Driver Telemetry Monitor | DriveGuard",
  description:
    "Real-time webcam computer vision telemetry streaming via WebSocket to the DriveGuard Edge AI Cascade.",
};

export default function LiveMonitorPage() {
  return (
    <div className="p-2 sm:p-6 md:p-8 space-y-3 sm:space-y-6 max-w-7xl mx-auto">
      <div className="hidden sm:block">
        <h1 className="text-xl sm:text-3xl font-bold tracking-tight text-foreground">
          Live Driver Monitoring Console
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground">
          Real-time responsive cabin monitoring powered by DriveGuard adaptive confidence cascade, 3D gaze tracking, and sub-frame blink kinematics.
        </p>
      </div>

      <UnifiedLiveMonitor />
    </div>
  );
}
