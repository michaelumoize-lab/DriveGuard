import React from "react";
import Link from "next/link";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle, CardDescription, CardFooter } from "@/components/ui/card";
import { ShieldAlert, ArrowLeft, Activity } from "lucide-react";

export default function NotFoundPage() {
  return (
    <div className="min-h-svh flex items-center justify-center p-4 bg-background text-foreground">
      <Card className="max-w-md w-full border-border/80 shadow-lg">
        <CardHeader>
          <div className="size-10 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary mb-2">
            <ShieldAlert className="size-5" />
          </div>
          <CardTitle className="text-xl font-bold">404 — Route Not Found</CardTitle>
          <CardDescription className="text-xs">
            The DriveGuard page or telemetry workspace you requested does not exist or has moved.
          </CardDescription>
        </CardHeader>
        <CardContent className="text-xs text-muted-foreground">
          Use the links below to return to the Command Overview or launch the Live Monitoring Console.
        </CardContent>
        <CardFooter className="flex items-center justify-between gap-2 border-t border-border pt-4">
          <Button variant="outline" size="sm" asChild>
            <Link href="/dashboard">
              <ArrowLeft className="size-3.5 mr-1.5" />
              Dashboard
            </Link>
          </Button>
          <Button size="sm" asChild>
            <Link href="/dashboard/live-monitor">
              <Activity className="size-3.5 mr-1.5" />
              Live Monitor
            </Link>
          </Button>
        </CardFooter>
      </Card>
    </div>
  );
}
