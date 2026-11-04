import React from "react";
import { GuidedCalibrationWizard } from "@/components/calibration/guided-calibration-wizard";

export const metadata = {
  title: "4-Zone Calibration Wizard | IntelliGuard",
  description: "Guided 4-Zone Personalised Baseline Calibration with real-time target-acquisition spatial gating.",
};

export default function CalibrationPage() {
  return (
    <div className="p-4 sm:p-6 md:p-8 space-y-6 max-w-7xl mx-auto">
      <div>
        <h1 className="text-2xl sm:text-3xl font-bold tracking-tight text-foreground">
          4-Zone Personal Baseline Calibration
        </h1>
        <p className="text-xs sm:text-sm text-muted-foreground">
          Calibrate your personal Eye Aspect Ratio (EAR) baselines across CENTER, LEFT MIRROR, RIGHT MIRROR, and DASHBOARD.
        </p>
      </div>

      <GuidedCalibrationWizard />
    </div>
  );
}
