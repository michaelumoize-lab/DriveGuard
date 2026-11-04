import React from "react";
import { LiveDriverMonitor } from "@/components/monitor/live-driver-monitor";

export const metadata = {
  title: "Live Driver Telemetry Monitor | IntelliGuard",
  description: "Real-time webcam computer vision telemetry streaming via WebSocket to the DriveGuard Edge AI Cascade.",
};

export default function LiveMonitorPage() {
  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          Live Driver Monitoring Console
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground">
          Stream real-time webcam video to the DriveGuard Cascade Engine to assess sub-frame blink kinematics, 3D gaze zones, and fatigue risks.
        </p>
      </div>

      <LiveDriverMonitor />
    </div>
  );
}
