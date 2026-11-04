"use client";

import React from "react";
import { cn } from "@/lib/utils";

interface StatCardProps {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  description?: string;
  trend?: {
    value: string;
    isPositive?: boolean;
  };
  variant?: "default" | "emerald" | "destructive" | "amber" | "primary" | "purple";
  active?: boolean;
  onClick?: () => void;
  className?: string;
}

export function StatCard({
  title,
  value,
  icon,
  description,
  trend,
  variant = "default",
  active = false,
  onClick,
  className,
}: StatCardProps) {
  const getVariantStyles = () => {
    switch (variant) {
      case "emerald":
        return {
          border: "hover:border-emerald-500/40",
          iconBg: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20",
          glow: "from-emerald-500/10 to-transparent",
        };
      case "destructive":
        return {
          border: "hover:border-destructive/40",
          iconBg: "bg-destructive/10 text-destructive border-destructive/20",
          glow: "from-destructive/10 to-transparent",
        };
      case "amber":
        return {
          border: "hover:border-amber-500/40 border-amber-500/30",
          iconBg: "bg-amber-500/10 text-amber-500 border-amber-500/20",
          glow: "from-amber-500/10 to-transparent",
        };
      case "purple":
        return {
          border: "hover:border-purple-500/40",
          iconBg: "bg-purple-500/10 text-purple-500 border-purple-500/20",
          glow: "from-purple-500/10 to-transparent",
        };
      case "primary":
        return {
          border: "hover:border-primary/40",
          iconBg: "bg-primary/10 text-primary border-primary/20",
          glow: "from-primary/15 to-transparent",
        };
      default:
        return {
          border: "hover:border-border",
          iconBg: "bg-muted/80 text-foreground border-border",
          glow: "from-muted/20 to-transparent",
        };
    }
  };

  const vStyles = getVariantStyles();

  return (
    <div
      onClick={onClick}
      role={onClick ? "button" : undefined}
      tabIndex={onClick ? 0 : undefined}
      onKeyDown={
        onClick
          ? (e) => {
              if (e.key === "Enter" || e.key === " ") {
                e.preventDefault();
                onClick();
              }
            }
          : undefined
      }
      className={cn(
        "relative overflow-hidden bg-card text-card-foreground border border-border/80 rounded-2xl p-4 sm:p-5 shadow-xs hover:shadow-md transition-all duration-300 flex flex-col justify-between group outline-none",
        onClick ? "cursor-pointer focus-visible:ring-2 focus-visible:ring-primary" : "",
        active ? "border-primary ring-2 ring-primary/30 shadow-md" : vStyles.border,
        className
      )}
    >
      {/* Background Gradient Accent Glow */}
      <div
        className={cn(
          "absolute -right-8 -top-8 w-24 h-24 bg-gradient-to-br rounded-full blur-xl pointer-events-none opacity-40 group-hover:opacity-80 transition-opacity duration-300",
          active ? "from-primary/25 to-transparent opacity-80" : vStyles.glow
        )}
      />

      {/* Header Row: Full Title & Icon */}
      <div className="flex items-start justify-between gap-2 mb-3">
        <span className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider leading-snug">
          {title}
        </span>
        <div className={cn("p-1.5 rounded-lg border shrink-0 transition-transform duration-300 group-hover:scale-105", vStyles.iconBg)}>
          {icon}
        </div>
      </div>

      {/* Value & Description */}
      <div>
        <div className="flex items-baseline gap-2">
          <h3 className="text-3xl font-bold font-mono tracking-tight text-foreground">{value}</h3>
          {trend && (
            <span
              className={cn(
                "text-[10px] font-semibold px-1.5 py-0.5 rounded-full border font-mono",
                trend.isPositive
                  ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border-emerald-500/20"
                  : "bg-destructive/10 text-destructive border-destructive/20"
              )}
            >
              {trend.value}
            </span>
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground/80 mt-1 font-medium leading-normal">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}
