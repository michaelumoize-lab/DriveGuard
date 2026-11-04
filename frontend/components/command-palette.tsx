"use client";

import React, { useEffect, useState, useCallback } from "react";
import { useRouter } from "next/navigation";
import { useTheme } from "next-themes";
import {
  CommandDialog,
  CommandInput,
  CommandList,
  CommandEmpty,
  CommandGroup,
  CommandItem,
  CommandSeparator,
  CommandShortcut,
} from "@/components/ui/command";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";
import {
  LayoutDashboard,
  Camera,
  Users,
  UserPlus,
  History,
  Cpu,
  ShieldAlert,
  Settings,
  Sun,
  Moon,
  Laptop,
  Search,
  ExternalLink,
  Check,
} from "lucide-react";

export function CommandPalette() {
  const [open, setOpen] = useState(false);
  const router = useRouter();
  const { theme, setTheme } = useTheme();

  useEffect(() => {
    const down = (e: KeyboardEvent) => {
      if ((e.key === "k" || e.key === "K") && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
    };

    document.addEventListener("keydown", down);
    return () => document.removeEventListener("keydown", down);
  }, []);

  const runCommand = useCallback((command: () => void) => {
    setOpen(false);
    command();
  }, []);

  return (
    <>
      <CommandDialog open={open} onOpenChange={setOpen}>
        <CommandInput placeholder="Type a command, page, or search term..." />
        <CommandList className="max-h-80">
          <CommandEmpty>No matching command or destination found.</CommandEmpty>

          {/* Quick Actions */}
          <CommandGroup heading="Quick Actions">
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard/simulation"))}
              className="gap-2.5 cursor-pointer"
            >
              <Camera className="w-4 h-4 text-primary" />
              <span>Launch Live Camera Scan</span>
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard/persons/register"))}
              className="gap-2.5 cursor-pointer"
            >
              <UserPlus className="w-4 h-4 text-emerald-500" />
              <span>Enroll New Biometric Identity</span>
            </CommandItem>
          </CommandGroup>

          <CommandSeparator />

          {/* Navigation */}
          <CommandGroup heading="Navigation">
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard"))}
              className="gap-2.5 cursor-pointer"
            >
              <LayoutDashboard className="w-4 h-4 text-muted-foreground" />
              <span>Security Overview Dashboard</span>
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard/persons"))}
              className="gap-2.5 cursor-pointer"
            >
              <Users className="w-4 h-4 text-muted-foreground" />
              <span>Personnel Directory</span>
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard/access-logs"))}
              className="gap-2.5 cursor-pointer"
            >
              <History className="w-4 h-4 text-muted-foreground" />
              <span>Access Event Logs</span>
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard/devices"))}
              className="gap-2.5 cursor-pointer"
            >
              <Cpu className="w-4 h-4 text-muted-foreground" />
              <span>Edge Devices & Gates</span>
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => router.push("/dashboard/alerts"))}
              className="gap-2.5 cursor-pointer"
            >
              <ShieldAlert className="w-4 h-4 text-muted-foreground" />
              <span>Security Alerts & Anomalies</span>
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => router.push("/settings/account"))}
              className="gap-2.5 cursor-pointer"
            >
              <Settings className="w-4 h-4 text-muted-foreground" />
              <span>Account & System Settings</span>
            </CommandItem>
          </CommandGroup>

          <CommandSeparator />

          {/* Theme Preferences */}
          <CommandGroup heading="Theme Preferences">
            <CommandItem
              onSelect={() => runCommand(() => setTheme("light"))}
              className="gap-2.5 cursor-pointer flex items-center justify-between"
            >
              <div className="flex items-center gap-2.5">
                <Sun className="w-4 h-4 text-amber-500" />
                <span>Light Mode</span>
              </div>
              {theme === "light" && <Check className="w-3.5 h-3.5 text-primary" />}
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => setTheme("dark"))}
              className="gap-2.5 cursor-pointer flex items-center justify-between"
            >
              <div className="flex items-center gap-2.5">
                <Moon className="w-4 h-4 text-blue-400" />
                <span>Dark Mode</span>
              </div>
              {theme === "dark" && <Check className="w-3.5 h-3.5 text-primary" />}
            </CommandItem>
            <CommandItem
              onSelect={() => runCommand(() => setTheme("system"))}
              className="gap-2.5 cursor-pointer flex items-center justify-between"
            >
              <div className="flex items-center gap-2.5">
                <Laptop className="w-4 h-4 text-muted-foreground" />
                <span>System Default</span>
              </div>
              {theme === "system" && <Check className="w-3.5 h-3.5 text-primary" />}
            </CommandItem>
          </CommandGroup>
        </CommandList>
      </CommandDialog>
    </>
  );
}

export function CommandPaletteTrigger() {
  const [mounted, setMounted] = useState(false);
  const [isMac, setIsMac] = useState(false);

  useEffect(() => {
    setMounted(true);
    setIsMac(navigator?.platform?.toUpperCase()?.indexOf("MAC") >= 0);
  }, []);

  const handleClick = () => {
    const event = new KeyboardEvent("keydown", {
      key: "k",
      metaKey: true,
      ctrlKey: true,
      bubbles: true,
    });
    document.dispatchEvent(event);
  };

  return (
    <Tooltip>
      <TooltipTrigger asChild>
        <button
          onClick={handleClick}
          type="button"
          className="hidden md:flex items-center gap-2 h-8 px-3 rounded-xl border border-input bg-muted/40 hover:bg-muted text-muted-foreground hover:text-foreground text-xs transition-colors shadow-2xs cursor-pointer"
        >
          <Search className="w-3.5 h-3.5 text-muted-foreground" />
          <span className="font-normal">Search or Jump to...</span>
          <kbd className="pointer-events-none inline-flex h-4 select-none items-center gap-0.5 rounded border border-border bg-background px-1.5 font-mono text-[10px] font-medium text-muted-foreground ml-2">
            {mounted && isMac ? "⌘K" : "Ctrl+K"}
          </kbd>
        </button>
      </TooltipTrigger>
      <TooltipContent side="bottom">Search or execute commands (Ctrl+K)</TooltipContent>
    </Tooltip>
  );
}
