"use client";

import { useQuery, useQueryClient, useIsFetching } from "@tanstack/react-query";
import type { HealthCheckResponse } from "@/types/health";

export const SYSTEM_HEALTH_QUERY_KEY = ["system-health"] as const;

export async function fetchSystemHealth(force = false): Promise<HealthCheckResponse> {
  const url = force ? "/api/dashboard/health?force=true" : "/api/dashboard/health";
  const res = await fetch(url, {
    cache: "no-store",
    headers: {
      Accept: "application/json",
    },
  });

  if (!res.ok) {
    throw new Error(`Health check probe returned HTTP ${res.status}`);
  }

  return res.json();
}

/**
 * Shared hook providing a synchronized, real-time single source of truth
 * for backend service health (FastAPI AI Engine, PostgreSQL, pgvector, Next.js, ImageKit).
 *
 * Ensures all UI components (navbar badges, dashboard panels, telemetry cards)
 * always stay in 100% sync, whether updated via interval polling, focus refetch,
 * or manual refresh buttons.
 */
export function useSystemHealth() {
  const queryClient = useQueryClient();

  const query = useQuery({
    queryKey: SYSTEM_HEALTH_QUERY_KEY,
    queryFn: () => fetchSystemHealth(false),
    staleTime: 10_000, // 10 seconds stale time
    refetchInterval: 30_000, // Poll every 30s to keep badge and dashboard in sync
    refetchIntervalInBackground: false,
    refetchOnWindowFocus: true,
  });

  const isFetchingCount = useIsFetching({ queryKey: SYSTEM_HEALTH_QUERY_KEY });
  const isRefreshing = isFetchingCount > 0;

  const refresh = async () => {
    try {
      return await queryClient.fetchQuery({
        queryKey: SYSTEM_HEALTH_QUERY_KEY,
        queryFn: () => fetchSystemHealth(true),
        staleTime: 0,
      });
    } catch {
      return null;
    }
  };

  return {
    health: query.data ?? null,
    isLoading: query.isLoading,
    isFetching: query.isFetching,
    isRefreshing,
    isError: query.isError,
    error: query.error,
    dataUpdatedAt: query.dataUpdatedAt,
    refetch: query.refetch,
    refresh,
  };
}
