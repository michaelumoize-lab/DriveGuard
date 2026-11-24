"use client";

import React, { useRef, useEffect } from "react";
import { Badge } from "@/components/ui/badge";
import { Activity } from "lucide-react";

interface KinematicsOscilloscopeProps {
  currentEar: number;
  closureThreshold?: number;
  dEarDt?: number;
  rcr?: number;
  className?: string;
}

export function KinematicsOscilloscope({
  currentEar,
  closureThreshold = 0.20,
  dEarDt = 0,
  rcr,
  className = "",
}: KinematicsOscilloscopeProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const earHistoryRef = useRef<number[]>([]);
  const derivHistoryRef = useRef<number[]>([]);
  const maxPoints = 120; // 2 seconds @ 60 FPS

  useEffect(() => {
    earHistoryRef.current.push(currentEar);
    if (earHistoryRef.current.length > maxPoints) {
      earHistoryRef.current.shift();
    }

    derivHistoryRef.current.push(dEarDt);
    if (derivHistoryRef.current.length > maxPoints) {
      derivHistoryRef.current.shift();
    }
  }, [currentEar, dEarDt]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    let animId: number;

    const render = () => {
      const w = canvas.width;
      const h = canvas.height;

      // Dark oscilloscope background
      ctx.fillStyle = "#0B0F19";
      ctx.fillRect(0, 0, w, h);

      // Grid lines
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      const gridStepsY = 4;
      for (let i = 1; i < gridStepsY; i++) {
        const y = (h / gridStepsY) * i;
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Draw dynamic closure threshold line (Dashed Rose)
      if (closureThreshold) {
        // EAR range 0.10 to 0.40 mapped to [h, 0]
        const normThreshY = h - ((closureThreshold - 0.1) / 0.3) * h;
        ctx.strokeStyle = "rgba(244, 63, 94, 0.6)";
        ctx.setLineDash([4, 4]);
        ctx.beginPath();
        ctx.moveTo(0, normThreshY);
        ctx.lineTo(w, normThreshY);
        ctx.stroke();
        ctx.setLineDash([]);
      }

      // Draw EAR Trace (Cyan)
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

      // Draw dEAR/dt Velocity Derivative Trace (Amber)
      const derivPoints = derivHistoryRef.current;
      if (derivPoints.length > 1) {
        ctx.strokeStyle = "#F59E0B";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        for (let i = 0; i < derivPoints.length; i++) {
          const x = (i / (maxPoints - 1)) * w;
          // Derivative mapped centered around h/2 (-1.0 to +1.0)
          const y = h / 2 - (derivPoints[i] / 1.5) * (h / 2);
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.stroke();
      }

      animId = requestAnimationFrame(render);
    };

    animId = requestAnimationFrame(render);
    return () => cancelAnimationFrame(animId);
  }, [closureThreshold]);

  return (
    <div className={`p-4 rounded-2xl bg-card border border-border/80 shadow-xs space-y-3 ${className}`}>
      <div className="flex items-center justify-between border-b border-border/60 pb-2">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-primary" />
          <span className="text-xs font-bold uppercase tracking-wider text-foreground">
            Blink Kinematics Oscilloscope (60 FPS)
          </span>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-[10px] font-mono text-cyan-400 border-cyan-400/30">
            EAR: {currentEar.toFixed(3)}
          </Badge>
          <Badge
            variant="outline"
            className={`text-[10px] font-mono ${
              rcr && rcr < 0.85
                ? "text-amber-400 border-amber-400/40 bg-amber-400/10"
                : "text-emerald-400 border-emerald-400/30"
            }`}
          >
            RCR: {rcr !== undefined ? rcr.toFixed(2) : "1.00"}
          </Badge>
        </div>
      </div>

      <div className="relative rounded-xl overflow-hidden border border-border/80">
        <canvas
          ref={canvasRef}
          width={480}
          height={140}
          className="w-full h-32 block bg-black"
        />
        <div className="absolute bottom-2 left-2 flex items-center gap-3 text-[10px] font-mono bg-black/60 backdrop-blur-xs px-2 py-0.5 rounded border border-white/10">
          <span className="flex items-center gap-1 text-cyan-400">
            <span className="w-2 h-0.5 bg-cyan-400" /> EAR Trajectory
          </span>
          <span className="flex items-center gap-1 text-amber-400">
            <span className="w-2 h-0.5 bg-amber-400" /> d(EAR)/dt Velocity
          </span>
          <span className="flex items-center gap-1 text-rose-400">
            <span className="w-2 h-0.5 border-b border-dashed border-rose-400" /> Threshold (Tz)
          </span>
        </div>
      </div>
    </div>
  );
}
