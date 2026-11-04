"use client";

import React, { useState, useRef } from "react";
import { toast } from "sonner";
import {
  Card,
  CardContent,
  CardHeader,
  CardTitle,
  CardDescription,
  CardFooter,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import {
  Eye,
  Camera,
  Play,
  RotateCcw,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  ArrowLeft,
  ArrowDown,
  Navigation,
} from "lucide-react";

interface CalibrationState {
  current_zone: string;
  phase_index: number;
  total_phases: number;
  zone_progress_pct: number;
  total_progress_pct: number;
  stable_samples_count: number;
  target_samples_count: number;
  instruction: string;
  is_head_in_target_zone: boolean;
  stability_feedback: string;
  is_completed: boolean;
}

export function GuidedCalibrationWizard() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  const [isCalibrating, setIsCalibrating] = useState(false);
  const [calState, setCalState] = useState<CalibrationState | null>(null);
  const [isSuccess, setIsSuccess] = useState(false);

  const wsBaseUrl =
    process.env.NEXT_PUBLIC_FASTAPI_WS_URL || "wss://driveguard-x048.onrender.com";

  const startCalibration = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: 640, height: 480, facingMode: "user" },
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      const sessionId = `cal_${Date.now()}`;
      const wsUrl = `${wsBaseUrl}/ws/v1/session/${sessionId}`;
      const ws = new WebSocket(wsUrl);
      wsRef.current = ws;

      ws.onopen = () => {
        setIsCalibrating(true);
        setIsSuccess(false);

        // Send start calibration command
        ws.send(JSON.stringify({ command: "START_4ZONE_CALIBRATION" }));
        toast.info("Starting Guided 4-Zone Calibration...");

        // Stream frames every 50ms (20 FPS)
        intervalRef.current = setInterval(() => {
          if (!videoRef.current || !canvasRef.current || ws.readyState !== WebSocket.OPEN) {
            return;
          }

          const video = videoRef.current;
          const canvas = canvasRef.current;
          const ctx = canvas.getContext("2d");
          if (!ctx) return;

          canvas.width = 320;
          canvas.height = 240;
          ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

          canvas.toBlob((blob) => {
            if (blob && ws.readyState === WebSocket.OPEN) {
              blob.arrayBuffer().then((buf) => ws.send(buf));
            }
          }, "image/jpeg", 0.65);
        }, 50);
      };

      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          if (data.type === "CALIBRATION_PHASE" && data.phase) {
            setCalState(data.phase);
            if (data.phase.is_completed) {
              setIsSuccess(true);
              toast.success("4-Zone Calibration Completed Successfully!");
            }
          }
        } catch {
          // ignore non-json messages
        }
      };

      ws.onerror = () => {
        toast.error("WebSocket connection failed. Verify the AI backend is reachable.");
      };

      ws.onclose = () => {
        setIsCalibrating(false);
      };
    } catch (err) {
      console.error(err);
      toast.error("Unable to access camera. Please allow webcam permissions.");
    }
  };

  const stopCalibration = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const stream = videoRef.current.srcObject as MediaStream;
      stream.getTracks().forEach((t) => t.stop());
      videoRef.current.srcObject = null;
    }
    setIsCalibrating(false);
  };

  const zoneDirections: Record<string, { label: string; icon: React.ReactNode }> = {
    CENTER: { label: "LOOK STRAIGHT AHEAD (CENTER)", icon: <Navigation className="size-6 text-primary rotate-0" /> },
    LEFT_MIRROR: { label: "TURN HEAD LEFT (LEFT MIRROR)", icon: <ArrowLeft className="size-6 text-primary" /> },
    RIGHT_MIRROR: { label: "TURN HEAD RIGHT (RIGHT MIRROR)", icon: <ArrowRight className="size-6 text-primary" /> },
    DASHBOARD: { label: "LOOK DOWN (DASHBOARD)", icon: <ArrowDown className="size-6 text-primary" /> },
  };

  const currentZoneInfo = calState?.current_zone
    ? zoneDirections[calState.current_zone] || { label: calState.current_zone, icon: <Eye className="size-6 text-primary" /> }
    : { label: "CENTER", icon: <Navigation className="size-6 text-primary" /> };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Video Viewport */}
      <Card className="lg:col-span-2 overflow-hidden border-border/80">
        <CardHeader className="py-3 px-4 flex flex-row items-center justify-between border-b border-border">
          <div className="flex items-center gap-2">
            <Camera className="size-4 text-primary" />
            <CardTitle className="text-sm font-semibold">Webcam Spatial Alignment</CardTitle>
          </div>
          <Badge variant="outline" className="text-xs font-mono">
            {isCalibrating ? "CALIBRATING" : "STANDBY"}
          </Badge>
        </CardHeader>

        <CardContent className="p-0 relative bg-black aspect-video flex items-center justify-center">
          <video
            ref={videoRef}
            autoPlay
            playsInline
            muted
            className="w-full h-full object-cover -scale-x-100"
          />
          <canvas ref={canvasRef} className="hidden" />

          {!isCalibrating && !isSuccess && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/90 backdrop-blur-xs text-center p-6 space-y-3">
              <div className="p-3 rounded-full bg-primary/10 text-primary">
                <Eye className="size-8" />
              </div>
              <div className="font-semibold text-foreground text-sm">
                4-Zone Personalised Baseline Calibration
              </div>
              <p className="text-xs text-muted-foreground max-w-sm">
                Align your head pose with CENTER, LEFT MIRROR, RIGHT MIRROR, and DASHBOARD. Progress only advances when your head is properly positioned!
              </p>
              <Button size="sm" onClick={startCalibration} className="bg-primary text-primary-foreground">
                <Play className="size-4 mr-1.5" />
                Start Calibration
              </Button>
            </div>
          )}

          {isSuccess && (
            <div className="absolute inset-0 flex flex-col items-center justify-center bg-background/95 backdrop-blur-xs text-center p-6 space-y-3">
              <div className="p-3 rounded-full bg-emerald-500/10 text-emerald-500">
                <CheckCircle2 className="size-10" />
              </div>
              <div className="font-bold text-foreground text-base">
                Calibration Successfully Completed!
              </div>
              <p className="text-xs text-muted-foreground max-w-sm">
                Your personalised baselines across all 4 spatial zones have been calculated and activated on the AI engine.
              </p>
              <Button size="sm" onClick={startCalibration} variant="outline">
                <RotateCcw className="size-4 mr-1.5" />
                Recalibrate
              </Button>
            </div>
          )}
        </CardContent>

        <CardFooter className="py-3 px-4 flex items-center justify-between border-t border-border bg-muted/20">
          {isCalibrating ? (
            <Button size="sm" variant="destructive" onClick={stopCalibration}>
              Cancel Calibration
            </Button>
          ) : (
            <Button size="sm" onClick={startCalibration} className="bg-primary text-primary-foreground">
              <Play className="size-3.5 mr-1.5" />
              Start 4-Zone Calibration
            </Button>
          )}
          <span className="text-xs text-muted-foreground font-mono">
            {calState ? `Phase ${calState.phase_index} of ${calState.total_phases}` : "Ready"}
          </span>
        </CardFooter>
      </Card>

      {/* Guidance & Target Alignment HUD */}
      <Card className="border-border/80">
        <CardHeader>
          <CardTitle className="text-sm font-semibold flex items-center gap-2">
            <Eye className="size-4 text-primary" />
            Guidance & Target Gating
          </CardTitle>
          <CardDescription className="text-xs">
            Follow the active directional instructions
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4 text-xs">
          {/* Active Target Banner */}
          <div className="p-3.5 rounded-xl border border-primary/20 bg-primary/5 flex items-center gap-3">
            <div className="p-2 rounded-lg bg-primary/10 shrink-0">
              {currentZoneInfo.icon}
            </div>
            <div>
              <span className="text-[10px] text-muted-foreground uppercase font-mono block">
                Target Gaze Zone
              </span>
              <span className="font-bold text-foreground text-sm">
                {currentZoneInfo.label}
              </span>
            </div>
          </div>

          {/* Spatial Alignment Status Indicator */}
          <div className="space-y-1.5">
            <div className="flex justify-between items-center">
              <span className="text-muted-foreground">Spatial Alignment:</span>
              <Badge
                variant={calState?.is_head_in_target_zone ? "outline" : "destructive"}
                className={`text-[10px] ${
                  calState?.is_head_in_target_zone
                    ? "border-emerald-500/40 text-emerald-500 bg-emerald-500/10"
                    : ""
                }`}
              >
                {calState?.is_head_in_target_zone ? "HEAD ALIGNED [OK]" : "PLEASE TURN HEAD"}
              </Badge>
            </div>
            <p className="text-[11px] text-muted-foreground">
              {calState?.instruction || "Waiting to begin calibration..."}
            </p>
          </div>

          {/* Active Zone Progress */}
          <div className="space-y-1.5 pt-2 border-t border-border">
            <div className="flex justify-between font-mono">
              <span className="text-muted-foreground">Zone Sample Acquisition:</span>
              <span className="font-semibold text-foreground">
                {calState ? `${Math.round(calState.zone_progress_pct)}%` : "0%"}
              </span>
            </div>
            <Progress value={calState?.zone_progress_pct || 0} className="h-2" />
            <div className="text-[10px] text-muted-foreground flex justify-between">
              <span>0% (Paused if Looking Away)</span>
              <span>100% (Locked)</span>
            </div>
          </div>

          {/* Overall 4-Zone Total Progress */}
          <div className="space-y-1.5 pt-2 border-t border-border">
            <div className="flex justify-between font-mono">
              <span className="text-muted-foreground">Total Calibration:</span>
              <span className="font-semibold text-foreground">
                {calState ? `${Math.round(calState.total_progress_pct)}%` : "0%"}
              </span>
            </div>
            <Progress value={calState?.total_progress_pct || 0} className="h-1.5" />
          </div>
        </CardContent>
      </Card>
    </div>
  );
}
