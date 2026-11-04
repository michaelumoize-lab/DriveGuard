import React from "react";
import { CameraSimulationHUD } from "@/components/camera/camera-simulation-hud";

export const metadata = {
  title: "Responsive Webcam Simulation | DriveGuard",
  description:
    "Production-grade browser webcam monitoring interface with mobile portrait support, multi-device switching, frame capture, and robust error recovery.",
};

export default function CameraSimulationPage() {
  return (
    <div className="p-3 sm:p-6 md:p-8 space-y-4 max-w-7xl mx-auto">
      <div className="hidden sm:block">
        <h1 className="text-xl sm:text-3xl font-bold tracking-tight text-foreground">
          Webcam Monitoring Simulation
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground">
          Access camera hardware, switch devices, capture frames, and verify mobile responsiveness without backend network dependencies.
        </p>
      </div>

      <CameraSimulationHUD />
    </div>
  );
}
