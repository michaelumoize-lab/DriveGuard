"use client";

import * as React from "react";
import { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { Moon, Sun, Monitor, Check } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

export interface ThemeToggleProps {
  variant?: "dropdown" | "cycle";
  size?: "default" | "sm" | "lg" | "icon" | "icon-sm" | "icon-lg";
  align?: "start" | "center" | "end";
  className?: string;
  showLabel?: boolean;
}

export function ThemeToggle({
  variant = "dropdown",
  size = "icon",
  align = "end",
  className,
  showLabel = false,
}: ThemeToggleProps) {
  const { theme, setTheme, resolvedTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  if (!mounted) {
    return (
      <Button
        variant="ghost"
        size={size}
        className={cn("text-muted-foreground shrink-0", className)}
        aria-label="Toggle theme"
        disabled
      >
        <Sun className="h-4 w-4 opacity-40" />
        <span className="sr-only">Toggle theme</span>
        {showLabel && <span className="ml-2 text-xs">Theme</span>}
      </Button>
    );
  }

  // Quick-cycle mode: direct toggle between light and dark
  if (variant === "cycle") {
    const handleCycle = () => {
      setTheme(resolvedTheme === "dark" ? "light" : "dark");
    };

    return (
      <Tooltip>
        <TooltipTrigger asChild>
          <Button
            variant="ghost"
            size={size}
            onClick={handleCycle}
            className={cn(
              "cursor-pointer text-muted-foreground hover:text-foreground relative shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-primary",
              className
            )}
            aria-label={`Current theme is ${theme}. Click to switch to ${
              resolvedTheme === "dark" ? "light" : "dark"
            } mode.`}
          >
            <Sun className="h-4 w-4 rotate-0 scale-100 transition-all duration-200 dark:-rotate-90 dark:scale-0 text-amber-500" />
            <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all duration-200 dark:rotate-0 dark:scale-100 text-blue-400" />
            <span className="sr-only">Toggle theme</span>
            {showLabel && (
              <span className="ml-2 text-xs capitalize text-foreground">{theme}</span>
            )}
          </Button>
        </TooltipTrigger>
        <TooltipContent side="bottom">
          Switch to {resolvedTheme === "dark" ? "light" : "dark"} mode
        </TooltipContent>
      </Tooltip>
    );
  }

  // Dropdown menu mode: choose between Light, Dark, or System
  return (
    <DropdownMenu>
      <Tooltip>
        <TooltipTrigger asChild>
          <DropdownMenuTrigger asChild>
            <Button
              variant="ghost"
              size={size}
              className={cn(
                "cursor-pointer text-muted-foreground hover:text-foreground relative shrink-0 outline-none focus-visible:ring-2 focus-visible:ring-primary",
                className
              )}
              aria-label="Select color theme (Light, Dark, System)"
            >
              <Sun className="h-4 w-4 rotate-0 scale-100 transition-all duration-200 dark:-rotate-90 dark:scale-0 text-amber-500" />
              <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all duration-200 dark:rotate-0 dark:scale-100 text-blue-400" />
              <span className="sr-only">Toggle theme</span>
              {showLabel && (
                <span className="ml-2 text-xs capitalize text-foreground">{theme}</span>
              )}
            </Button>
          </DropdownMenuTrigger>
        </TooltipTrigger>
        <TooltipContent side="bottom">Theme settings</TooltipContent>
      </Tooltip>
      <DropdownMenuContent align={align} className="min-w-36 rounded-xl border border-border shadow-md">
        <DropdownMenuItem
          onClick={() => setTheme("light")}
          className={cn(
            "flex items-center justify-between cursor-pointer px-2.5 py-1.5 text-xs rounded-lg",
            theme === "light" && "font-semibold text-primary bg-primary/10"
          )}
        >
          <div className="flex items-center gap-2">
            <Sun className="h-3.5 w-3.5 text-amber-500" />
            <span>Light</span>
          </div>
          {theme === "light" && <Check className="h-3.5 w-3.5 text-primary" />}
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={() => setTheme("dark")}
          className={cn(
            "flex items-center justify-between cursor-pointer px-2.5 py-1.5 text-xs rounded-lg",
            theme === "dark" && "font-semibold text-primary bg-primary/10"
          )}
        >
          <div className="flex items-center gap-2">
            <Moon className="h-3.5 w-3.5 text-blue-400" />
            <span>Dark</span>
          </div>
          {theme === "dark" && <Check className="h-3.5 w-3.5 text-primary" />}
        </DropdownMenuItem>

        <DropdownMenuItem
          onClick={() => setTheme("system")}
          className={cn(
            "flex items-center justify-between cursor-pointer px-2.5 py-1.5 text-xs rounded-lg",
            theme === "system" && "font-semibold text-primary bg-primary/10"
          )}
        >
          <div className="flex items-center gap-2">
            <Monitor className="h-3.5 w-3.5 text-muted-foreground" />
            <span>System</span>
          </div>
          {theme === "system" && <Check className="h-3.5 w-3.5 text-primary" />}
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
