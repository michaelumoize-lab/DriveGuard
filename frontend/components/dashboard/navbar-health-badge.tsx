"use client";

import React from "react";
import { RefreshCw } from "lucide-react";
import { useSystemHealth } from "@/hooks/use-system-health";

export function NavbarSystemHealthBadge() {
  const { health, isLoading, isRefreshing, refresh } = useSystemHealth();

  const fastApiStatus = health?.services?.fastapi?.status;

  let badgeContent: React.ReactElement | null = null;

  if (isLoading && !health) {
    badgeContent = (
      <button
        type="button"
        onClick={() => refresh()}
        aria-label="Checking AI Engine status. Click to refresh."
        className="relative flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 sm:py-1 bg-muted/80 hover:bg-muted border border-border rounded-full text-xs font-mono text-muted-foreground transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-ring outline-none after:content-[''] after:absolute after:-inset-y-2 after:inset-x-0 pointer-fine:after:hidden"
      >
        <span className="size-2 rounded-full bg-muted-foreground animate-pulse motion-reduce:animate-none" aria-hidden="true" />
        <span className="hidden sm:inline">Checking AI Status…</span>
        <span className="inline sm:hidden text-[10px]">AI…</span>
      </button>
    );
  } else if (fastApiStatus === "online") {
    badgeContent = (
      <button
        type="button"
        onClick={() => refresh()}
        aria-label={`AI Engine Online${health?.services?.fastapi?.latencyMs ? `, Latency ${health.services.fastapi.latencyMs}ms` : ""}. Click to refresh.`}
        className="relative flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 sm:py-1 bg-muted/80 hover:bg-muted border border-emerald-500/20 hover:border-emerald-500/40 rounded-full text-xs font-mono text-emerald-600 dark:text-emerald-400 transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-ring outline-none after:content-[''] after:absolute after:-inset-y-2 after:inset-x-0 pointer-fine:after:hidden"
      >
        {isRefreshing ? (
          <RefreshCw className="size-3 animate-spin motion-reduce:animate-none text-emerald-500" aria-hidden="true" />
        ) : (
          <span className="size-2 rounded-full bg-emerald-500 animate-pulse motion-reduce:animate-none" aria-hidden="true" />
        )}
        <span className="hidden sm:inline">AI Engine: Online</span>
        <span className="inline sm:hidden text-[10px] font-semibold">AI OK</span>
      </button>
    );
  } else if (fastApiStatus === "error") {
    badgeContent = (
      <button
        type="button"
        onClick={() => refresh()}
        aria-label={`AI Engine Degraded: ${health?.services?.fastapi?.message || "Service Degraded"}. Click to retry.`}
        className="relative flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 sm:py-1 bg-amber-500/10 hover:bg-amber-500/20 border border-amber-500/30 rounded-full text-xs font-mono text-amber-600 dark:text-amber-400 transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-ring outline-none after:content-[''] after:absolute after:-inset-y-2 after:inset-x-0 pointer-fine:after:hidden"
      >
        {isRefreshing ? (
          <RefreshCw className="size-3 animate-spin motion-reduce:animate-none text-amber-500" aria-hidden="true" />
        ) : (
          <span className="size-2 rounded-full bg-amber-500 animate-pulse motion-reduce:animate-none" aria-hidden="true" />
        )}
        <span className="hidden sm:inline">AI Engine: Degraded</span>
        <span className="inline sm:hidden text-[10px] font-semibold">AI WARN</span>
      </button>
    );
  } else {
    badgeContent = (
      <button
        type="button"
        onClick={() => refresh()}
        aria-label={`AI Engine Offline: ${health?.services?.fastapi?.message || "Service unreachable"}. Click to retry.`}
        className="relative flex items-center gap-1.5 sm:gap-2 px-2.5 sm:px-3 py-1.5 sm:py-1 bg-destructive/10 hover:bg-destructive/20 border border-destructive/30 rounded-full text-xs font-mono text-destructive transition-colors cursor-pointer focus-visible:ring-2 focus-visible:ring-ring outline-none after:content-[''] after:absolute after:-inset-y-2 after:inset-x-0 pointer-fine:after:hidden"
      >
        {isRefreshing ? (
          <RefreshCw className="size-3 animate-spin motion-reduce:animate-none text-destructive" aria-hidden="true" />
        ) : (
          <span className="size-2 rounded-full bg-destructive" aria-hidden="true" />
        )}
        <span className="hidden sm:inline">AI Engine: Offline</span>
        <span className="inline sm:hidden text-[10px] font-semibold">AI OFF</span>
      </button>
    );
  }

  return badgeContent;
}
