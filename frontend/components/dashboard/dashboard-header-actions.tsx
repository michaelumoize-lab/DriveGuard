"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { Button } from "@/components/ui/button";
import { ThemeToggle } from "@/components/theme-toggle";
import { Activity, Bell, Radio } from "lucide-react";

interface DashboardHeaderActionsProps {
  criticalAlertsCount: number;
}

export function DashboardHeaderActions({
  criticalAlertsCount,
}: DashboardHeaderActionsProps) {
  const pathname = usePathname();
  const isLiveMonitor = pathname === "/dashboard/live-monitor" || pathname === "/dashboard/camera-simulation";

  const formattedAlertCount =
    criticalAlertsCount > 99 ? "99+" : criticalAlertsCount.toString();

  return (
    <div className="flex items-center gap-1.5 sm:gap-2.5 shrink-0">
      {/* Live Driver Monitor Quick Button / Active Indicator */}
      {isLiveMonitor ? (
        <div
          aria-label="Live Driver Monitoring is active on this page"
          className="px-2.5 py-1 bg-emerald-500/10 border border-emerald-500/30 text-emerald-600 dark:text-emerald-400 text-xs font-semibold rounded-xl flex items-center gap-1.5 shadow-xs shrink-0 select-none"
        >
          <Radio className="w-3.5 h-3.5 text-emerald-500 animate-pulse" />
          <span className="hidden xs:inline font-mono text-[11px]">LIVE CONSOLE</span>
        </div>
      ) : (
        <Tooltip>
          <TooltipTrigger asChild>
            <Link
              href="/dashboard/live-monitor"
              aria-label="Launch Live Driver Monitoring"
              className="px-2.5 sm:px-3 py-1.5 bg-primary/10 hover:bg-primary/20 border border-primary/30 text-primary text-xs font-semibold rounded-xl transition-all flex items-center gap-1.5 shadow-xs shrink-0"
            >
              <Activity className="w-3.5 h-3.5" />
              <span className="hidden xs:inline">Live Monitor</span>
            </Link>
          </TooltipTrigger>
          <TooltipContent side="bottom">Launch Live Driver Monitor</TooltipContent>
        </Tooltip>
      )}

      {/* Safety Alerts Notification Bell */}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            asChild
            variant="ghost"
            size="icon"
            className="relative h-9 w-9 text-muted-foreground hover:text-foreground shrink-0 rounded-xl"
          >
            <Link
              href="/dashboard/alerts"
              aria-label={`Safety Alerts (${criticalAlertsCount} critical)`}
            >
              <Bell className="w-4 h-4" />
              {criticalAlertsCount > 0 && (
                <span className="absolute -top-1 -right-1 flex h-4 min-w-4 px-1 items-center justify-center rounded-full bg-rose-600 dark:bg-rose-500 text-[10px] font-bold font-mono text-white ring-2 ring-background leading-none shadow-xs pointer-events-none">
                  {formattedAlertCount}
                </span>
              )}
            </Link>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          {criticalAlertsCount > 0
            ? `Critical Fatigue Alerts (${criticalAlertsCount})`
            : "Safety Alerts (All clear)"}
        </TooltipContent>
      </Tooltip>

      {/* Theme Mode Toggle */}
      <ThemeToggle />
    </div>
  );
}
