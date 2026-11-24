"use client";

import React, { useRef, useEffect, useCallback } from "react";
import { Badge } from "@/components/ui/badge";
import { Activity } from "lucide-react";

interface KinematicsOscilloscopeProps {
  currentEar: number | null;
  closureThreshold?: number;
  dEarDt?: number;
  rcr?: number;
  isActive?: boolean;
  className?: string;
}

export function KinematicsOscilloscope({
  currentEar,
  closureThreshold = 0.20,
  dEarDt = 0,
  rcr,
  isActive = true,
  className = "",
}: KinematicsOscilloscopeProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const earHistoryRef = useRef<number[]>([]);
  const derivHistoryRef = useRef<number[]>([]);
  const maxPoints = 120;

  const drawFrame = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = "#0B0F19";
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = "rgba(255, 255, 255, 0.06)";
    ctx.lineWidth = 1;
    const gridStepsY = 4;
    for (let i = 1; i < gridStepsY; i++) {
      const y = (h / gridStepsY) * i;
      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(w, y);
      ctx.stroke();
    }

    if (closureThreshold) {
      const normThreshY = h - ((closureThreshold - 0.1) / 0.3) * h;
      ctx.strokeStyle = "rgba(244, 63, 94, 0.65)";
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(0, normThreshY);
      ctx.lineTo(w, normThreshY);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const earPoints = earHistoryRef.current;
    if (earPoints.length > 1) {
      ctx.strokeStyle = "#00F0FF";
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (let i = 0; i < earPoints.length; i++) {
        const x = (i / (maxPoints - 1)) * w;
        const y = h - ((earPoints[i] - 0.1) / 0.3) * h;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }

    const derivPoints = derivHistoryRef.current;
    if (derivPoints.length > 1) {
      ctx.strokeStyle = "#F59E0B";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      for (let i = 0; i < derivPoints.length; i++) {
        const x = (i / (maxPoints - 1)) * w;
        const y = h / 2 - (derivPoints[i] / 1.5) * (h / 2);
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
    }
  }, [closureThreshold]);

  useEffect(() => {
    if (currentEar === null) {
      earHistoryRef.current = [];
      derivHistoryRef.current = [];
      drawFrame();
      return;
    }

    earHistoryRef.current.push(currentEar);
    if (earHistoryRef.current.length > maxPoints) {
      earHistoryRef.current.shift();
    }

    derivHistoryRef.current.push(dEarDt);
    if (derivHistoryRef.current.length > maxPoints) {
      derivHistoryRef.current.shift();
    }

    drawFrame();
  }, [currentEar, dEarDt, drawFrame]);

  useEffect(() => {
    drawFrame();

    const prefersReducedMotion =
      typeof window !== "undefined" &&
      window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!isActive || prefersReducedMotion) {
      return;
    }

    let animId: number;
    const loop = () => {
      drawFrame();
      animId = requestAnimationFrame(loop);
    };
    animId = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(animId);
  }, [drawFrame, isActive]);

  return (
    <div className={`p-4 rounded-2xl bg-card border border-border/80 shadow-xs space-y-3 ${className}`}>
      <div className="flex items-center justify-between border-b border-border/60 pb-2 gap-2">
        <div className="flex items-center gap-2 min-w-0">
          <Activity className="size-4 text-primary shrink-0" aria-hidden="true" />
          <span className="text-xs font-semibold text-foreground truncate">
            Blink Kinematics Oscilloscope
          </span>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <Badge variant="outline" className="text-[11px] font-mono tabular-nums text-cyan-400 border-cyan-400/30">
            EAR {currentEar !== null ? currentEar.toFixed(3) : "—"}
          </Badge>
          <Badge
            variant="outline"
            className={`text-[11px] font-mono tabular-nums ${
              currentEar === null
                ? "text-muted-foreground border-border"
                : rcr !== undefined && rcr < 0.85
                  ? "text-amber-400 border-amber-400/40 bg-amber-400/10"
                  : "text-emerald-400 border-emerald-400/30"
            }`}
          >
            RCR {currentEar === null ? "—" : rcr !== undefined ? rcr.toFixed(2) : "1.00"}
          </Badge>
        </div>
      </div>

      <div className="relative rounded-xl overflow-hidden border border-border/80">
        <canvas
          ref={canvasRef}
          width={480}
          height={140}
          aria-label="Live eye aspect ratio and velocity oscilloscope"
          className="w-full h-32 block bg-black"
        />
        <div className="absolute bottom-2 left-2 flex items-center gap-3 text-[10px] font-mono bg-black/70 backdrop-blur-xs px-2 py-0.5 rounded border border-white/10">
          <span className="flex items-center gap-1 text-cyan-400">
            <span className="w-2 h-0.5 bg-cyan-400" aria-hidden="true" /> EAR
          </span>
          <span className="flex items-center gap-1 text-amber-400">
            <span className="w-2 h-0.5 bg-amber-400" aria-hidden="true" /> d(EAR)/dt
          </span>
          <span className="flex items-center gap-1 text-rose-400">
            <span className="w-2 h-0.5 border-b border-dashed border-rose-400" aria-hidden="true" /> Threshold
          </span>
        </div>
      </div>
    </div>
  );
}
