"use client";

import React from "react";
import { CapturedFrame } from "@/types/camera";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogDescription,
  DialogFooter,
} from "@/components/ui/dialog";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { FormattedTimestamp } from "@/components/ui/formatted-timestamp";
import { Download, FileImage, Sparkles, Trash2 } from "lucide-react";
import Image from "next/image";

interface CameraSnapshotPreviewProps {
  frame: CapturedFrame | null;
  isOpen: boolean;
  onClose: () => void;
}

export function CameraSnapshotPreview({
  frame,
  isOpen,
  onClose,
}: CameraSnapshotPreviewProps) {
  if (!frame) return null;

  const handleDownload = () => {
    const url = URL.createObjectURL(frame.blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = frame.file.name;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  const fileSizeKb = (frame.sizeBytes / 1024).toFixed(1);

  return (
    <Dialog open={isOpen} onOpenChange={(open) => !open && onClose()}>
      <DialogContent className="max-w-lg sm:max-w-xl p-4 sm:p-6 rounded-2xl bg-card border border-border">
        <DialogHeader className="space-y-1">
          <DialogTitle className="text-base sm:text-lg font-bold flex items-center gap-2 text-foreground">
            <FileImage className="size-5 text-primary" />
            <span>Captured Video Frame</span>
          </DialogTitle>
          <DialogDescription className="text-xs text-muted-foreground">
            Still frame captured from active stream. Prepared for future AI inference.
          </DialogDescription>
        </DialogHeader>

        {/* Snapshot Image Preview */}
        <div className="relative aspect-video w-full rounded-xl overflow-hidden bg-black/90 border border-border/80 flex items-center justify-center my-2 shadow-inner">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img
            src={frame.dataUrl}
            alt="Captured video frame"
            className="w-full h-full object-contain"
          />
        </div>

        {/* Frame Telemetry Metadata */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-2 text-xs font-mono">
          <div className="p-2.5 rounded-xl bg-muted/40 border border-border">
            <div className="text-[10px] text-muted-foreground uppercase">Resolution</div>
            <div className="font-bold text-foreground mt-0.5">
              {frame.width} × {frame.height}
            </div>
          </div>

          <div className="p-2.5 rounded-xl bg-muted/40 border border-border">
            <div className="text-[10px] text-muted-foreground uppercase">File Size</div>
            <div className="font-bold text-foreground mt-0.5">{fileSizeKb} KB</div>
          </div>

          <div className="p-2.5 rounded-xl bg-muted/40 border border-border">
            <div className="text-[10px] text-muted-foreground uppercase">MIME Type</div>
            <div className="font-bold text-foreground mt-0.5">{frame.format}</div>
          </div>

          <div className="p-2.5 rounded-xl bg-muted/40 border border-border">
            <div className="text-[10px] text-muted-foreground uppercase">Timestamp</div>
            <div className="font-bold text-foreground mt-0.5">
              <FormattedTimestamp date={frame.timestamp} mode="time" showSeconds={true} />
            </div>
          </div>
        </div>

        <DialogFooter className="flex flex-row items-center justify-end gap-2 pt-3">
          <Button
            variant="outline"
            size="sm"
            onClick={onClose}
            className="rounded-xl text-xs h-9"
          >
            Close
          </Button>
          <Button
            size="sm"
            onClick={handleDownload}
            className="bg-primary hover:bg-primary/90 text-primary-foreground font-semibold rounded-xl text-xs h-9 gap-1.5 shadow-sm"
          >
            <Download className="size-3.5" />
            <span>Download JPEG ({fileSizeKb} KB)</span>
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}
