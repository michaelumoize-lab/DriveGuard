"use client";

import React, { useEffect } from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { AlertTriangle, RotateCcw, Home } from "lucide-react";

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("DriveGuard unhandled route error:", error);
  }, [error]);

  return (
    <div className="min-h-svh flex items-center justify-center p-4 bg-background text-foreground">
      <Card className="max-w-md w-full border-destructive/40 shadow-lg">
        <CardHeader>
          <div className="size-10 rounded-xl bg-destructive/10 border border-destructive/20 flex items-center justify-center text-destructive mb-2">
            <AlertTriangle className="size-5" />
          </div>
          <CardTitle className="text-lg">Something went wrong</CardTitle>
          <CardDescription className="text-xs">
            DriveGuard encountered an unexpected error while rendering this view.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="p-3 rounded-lg bg-muted/50 border border-border font-mono text-[11px] text-muted-foreground break-words">
            {error.message || "Unexpected runtime error"}
            {error.digest && (
              <span className="block mt-1 text-[10px] opacity-70">Digest: {error.digest}</span>
            )}
          </div>
        </CardContent>
        <CardFooter className="flex items-center justify-between gap-2 border-t border-border pt-4">
          <Button variant="outline" size="sm" asChild>
            <Link href="/">
              <Home className="size-3.5 mr-1.5" />
              Return Home
            </Link>
          </Button>
          <Button size="sm" onClick={() => reset()}>
            <RotateCcw className="size-3.5 mr-1.5" />
            Try Again
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
