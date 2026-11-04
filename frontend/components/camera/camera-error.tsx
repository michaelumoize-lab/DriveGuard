"use client";

import React from "react";
import { CameraError } from "@/types/camera";
import { AlertCircle, Lock, RefreshCw, Smartphone, VideoOff, WifiOff } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CameraErrorDisplayProps {
  error: CameraError | null;
  onRetry: () => void;
  onDismiss?: () => void;
}

export function CameraErrorDisplay({
  error,
  onRetry,
  onDismiss,
}: CameraErrorDisplayProps) {
  if (!error) return null;

  const getErrorIcon = () => {
    switch (error.code) {
      case "permission-denied":
        return <Lock className="size-6 text-rose-500" />;
      case "no-camera-available":
        return <VideoOff className="size-6 text-amber-500" />;
      case "camera-already-in-use":
        return <AlertCircle className="size-6 text-amber-500" />;
      case "unsupported":
        return <Smartphone className="size-6 text-blue-500" />;
      default:
        return <WifiOff className="size-6 text-rose-500" />;
    }
  };

  return (
    <div className="p-4 sm:p-5 rounded-2xl bg-destructive/10 border border-destructive/30 shadow-lg text-foreground space-y-3 animate-in fade-in slide-in-from-top-2 duration-300">
      <div className="flex items-start gap-3.5">
        <div className="p-2.5 rounded-xl bg-destructive/15 shrink-0 mt-0.5">
          {getErrorIcon()}
        </div>
        <div className="space-y-1 flex-1">
          <div className="flex items-center justify-between">
            <h4 className="font-semibold text-sm sm:text-base text-foreground">
              {error.message}
            </h4>
          </div>
          <p className="text-xs sm:text-sm text-muted-foreground leading-relaxed">
            {error.recoveryAction}
          </p>

          {error.technicalDetails && (
            <p className="text-[10px] font-mono text-muted-foreground/60 pt-1">
              Details: {error.technicalDetails}
            </p>
          )}
        </div>
      </div>

      <div className="flex items-center justify-end gap-2 pt-2 border-t border-destructive/20">
        {onDismiss && (
          <Button
            size="sm"
            variant="ghost"
            onClick={onDismiss}
            className="text-xs h-8 text-muted-foreground hover:text-foreground"
          >
            Dismiss
          </Button>
        )}
        <Button
          size="sm"
          onClick={onRetry}
          className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold text-xs h-8 px-4 rounded-xl gap-1.5 shadow-sm"
        >
          <RefreshCw className="size-3.5" />
          <span>Retry Camera Access</span>
        </Button>
      </div>
    </div>
  );
}
