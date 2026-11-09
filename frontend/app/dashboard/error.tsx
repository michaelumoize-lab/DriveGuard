"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { AlertTriangle, RotateCcw, Shield } from "lucide-react";

export default function DashboardError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Dashboard error boundary caught:", error);
  }, [error]);

  return (
    <div className="p-6 md:p-12 flex items-center justify-center">
      <Card className="max-w-lg w-full border-destructive/40 shadow-md">
        <CardHeader>
          <div className="size-10 rounded-xl bg-destructive/10 border border-destructive/20 flex items-center justify-center text-destructive mb-2">
            <AlertTriangle className="size-5" />
          </div>
          <CardTitle className="text-lg">Workspace Load Failure</CardTitle>
          <CardDescription className="text-xs">
            Could not load telemetry or session records for this dashboard view.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="p-3 rounded-lg bg-muted/50 border border-border font-mono text-[11px] text-muted-foreground break-words">
            {error.message || "Database or server connection error"}
          </div>
        </CardContent>
        <CardFooter className="flex items-center justify-between gap-2 border-t border-border pt-4">
          <Button variant="outline" size="sm" asChild>
            <Link href="/dashboard/live-monitor">
              <Shield className="size-3.5 mr-1.5" />
              Open Live Monitor
            </Link>
          </Button>
          <Button size="sm" onClick={() => reset()}>
            <RotateCcw className="size-3.5 mr-1.5" />
            Retry
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
