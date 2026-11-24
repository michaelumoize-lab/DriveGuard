"use client";

import React from "react";
import { usePathname } from "next/navigation";

const ROUTE_SEGMENTS: Record<string, string[]> = {
  "/dashboard": ["Safety Dashboard"],
  "/dashboard/live-monitor": ["Live Monitoring"],
  "/dashboard/calibration": ["Driver Calibration"],
  "/dashboard/sessions": ["Driving Sessions"],
  "/dashboard/alerts": ["Fatigue Alerts"],
  "/dashboard/project-overview": ["Research & System", "Project Overview"],
  "/dashboard/benchmarks": ["Research & System", "Ablation Benchmarks"],
  "/dashboard/hardware": ["Research & System", "Hardware Co-Processor"],
  "/dashboard/camera-simulation": ["Research & System", "Camera Simulation"],
  "/dashboard/defense-guide": ["Research & System", "Defense Companion"],
  "/dashboard/help": ["Help"],
};

function resolveSegments(pathname: string): string[] {
  if (ROUTE_SEGMENTS[pathname]) {
    return ROUTE_SEGMENTS[pathname];
  }
  if (pathname.startsWith("/dashboard/sessions/")) {
    return ["Driving Sessions", "Session Replay"];
  }
  if (pathname.startsWith("/settings")) {
    return ["Settings"];
  }
  return ["Driver Safety Center"];
}

export function WorkspaceBreadcrumb() {
  const pathname = usePathname();
  const segments = resolveSegments(pathname);

  return (
    <div className="hidden sm:flex items-center gap-2 text-xs font-mono text-muted-foreground truncate">
      <span className="text-muted-foreground/70">DriveGuard</span>
      {segments.map((segment, index) => {
        const isLast = index === segments.length - 1;
        return (
          <React.Fragment key={segment}>
            <span className="text-border">/</span>
            <span className={isLast ? "text-foreground font-medium" : "text-muted-foreground/80"}>
              {segment}
            </span>
          </React.Fragment>
        );
      })}
    </div>
  );
}
