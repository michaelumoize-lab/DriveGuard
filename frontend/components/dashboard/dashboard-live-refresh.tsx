"use client";

import React, { useEffect, useState, useTransition, useCallback } from "react";
import { useRouter } from "next/navigation";
import { RotateCw, Activity, Pause, Play, Check } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";

interface DashboardLiveRefreshProps {
  initialTimestamp?: string;
}

const INTERVAL_OPTIONS = [
  { label: "10s", seconds: 10 },
  { label: "30s", seconds: 30 },
  { label: "60s", seconds: 60 },
  { label: "Off", seconds: 0 },
];

export function DashboardLiveRefresh({ initialTimestamp }: DashboardLiveRefreshProps) {
  const router = useRouter();
  const [isPending, startTransition] = useTransition();
  const [selectedInterval, setSelectedInterval] = useState<number>(10);
  const [secondsRemaining, setSecondsRemaining] = useState<number>(10);
  const [lastRefreshedAt, setLastRefreshedAt] = useState<string>("--:--");

  useEffect(() => {
    setLastRefreshedAt(
      new Date(initialTimestamp || Date.now()).toLocaleTimeString([], {
        hour: "2-digit",
        minute: "2-digit",
        hour12: false,
      })
    );
  }, [initialTimestamp]);

  const remainingRef = React.useRef(selectedInterval);

  const triggerRefresh = useCallback(() => {
    startTransition(() => {
      router.refresh();
      setLastRefreshedAt(
        new Date().toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          hour12: false,
        })
      );
    });
  }, [router]);

  // Countdown timer effect cleanly decoupled from state updater
  useEffect(() => {
    remainingRef.current = selectedInterval;
    setSecondsRemaining(selectedInterval);

    if (selectedInterval === 0) return;

    const timer = setInterval(() => {
      remainingRef.current -= 1;
      if (remainingRef.current <= 0) {
        remainingRef.current = selectedInterval;
        setSecondsRemaining(selectedInterval);
        triggerRefresh();
      } else {
        setSecondsRemaining(remainingRef.current);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [selectedInterval, triggerRefresh]);

  const handleIntervalChange = (secs: number) => {
    remainingRef.current = secs;
    setSelectedInterval(secs);
    setSecondsRemaining(secs);
  };

  const isLive = selectedInterval > 0;

  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      {/* Live Badge & Interval Selector */}
      <div className="flex items-center rounded-lg border border-border bg-muted/30 p-1 backdrop-blur-sm">
        <div className="flex items-center gap-1.5 px-2 py-1">
          <span className="relative flex h-2 w-2">
            {isLive ? (
              <>
                <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
              </>
            ) : (
              <span className="relative inline-flex rounded-full h-2 w-2 bg-muted-foreground/40"></span>
            )}
          </span>
          <span className="font-semibold text-[11px] tracking-wide text-foreground">
            {isLive ? "LIVE TELEMETRY" : "PAUSED"}
          </span>
          {isLive && (
            <span className="font-mono text-[10px] text-muted-foreground ml-0.5">
              ({secondsRemaining}s)
            </span>
          )}
        </div>

        <div className="h-4 w-px bg-border mx-1" />

        {/* Interval Chips */}
        <div className="flex items-center gap-0.5">
          {INTERVAL_OPTIONS.map((opt) => (
            <button
              key={opt.label}
              onClick={() => handleIntervalChange(opt.seconds)}
              className={`px-2 py-0.5 rounded text-[11px] font-medium transition-colors ${
                selectedInterval === opt.seconds
                  ? "bg-primary text-primary-foreground shadow-xs"
                  : "text-muted-foreground hover:text-foreground hover:bg-muted"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Manual Refresh Button */}
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="outline"
            size="sm"
            onClick={() => {
              triggerRefresh();
              if (selectedInterval > 0) {
                remainingRef.current = selectedInterval;
                setSecondsRemaining(selectedInterval);
              }
            }}
            disabled={isPending}
            className="h-8 px-2.5 text-xs gap-1.5 border-border bg-background hover:bg-muted"
          >
            <RotateCw className={`h-3.5 w-3.5 text-muted-foreground ${isPending ? "animate-spin text-primary" : ""}`} />
            <span className="hidden sm:inline">Refresh</span>
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">Trigger manual telemetry refresh</TooltipContent>
      </Tooltip>

      {/* Last Updated Timestamp */}
      <div className="hidden lg:flex items-center text-[11px] text-muted-foreground font-mono pl-1">
        Updated: {lastRefreshedAt}
      </div>
    </div>
  );
}
