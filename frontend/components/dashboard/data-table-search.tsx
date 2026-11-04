"use client";

import React, { useState, useEffect, useTransition } from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Search, X, Loader2 } from "lucide-react";
import { Tooltip, TooltipTrigger, TooltipContent } from "@/components/ui/tooltip";

interface DataTableSearchProps {
  placeholder?: string;
  paramName?: string;
  className?: string;
}

export function DataTableSearch({
  placeholder = "Search...",
  paramName = "search",
  className = "",
}: DataTableSearchProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const initialSearch = searchParams.get(paramName) || "";
  const [searchTerm, setSearchTerm] = useState<string>(initialSearch);
  const [isPending, startTransition] = useTransition();
  const lastPushedRef = React.useRef<string | null>(null);

  // Sync internal input state when URL searchParams change externally
  useEffect(() => {
    const urlVal = searchParams.get(paramName) || "";
    if (lastPushedRef.current !== null) {
      if (urlVal === lastPushedRef.current) {
        lastPushedRef.current = null;
        return;
      }
      lastPushedRef.current = null;
    }
    setSearchTerm(urlVal);
  }, [searchParams, paramName]);

  // Debounced URL Search Parameter update (300ms)
  useEffect(() => {
    const currentParam = searchParams.get(paramName) || "";
    if (searchTerm === currentParam) return;

    const timer = setTimeout(() => {
      const params = new URLSearchParams(searchParams.toString());
      const trimmed = searchTerm.trim();
      if (trimmed) {
        params.set(paramName, trimmed);
      } else {
        params.delete(paramName);
      }
      // Reset page to 1 whenever search query changes
      params.set("page", "1");

      lastPushedRef.current = trimmed;
      startTransition(() => {
        router.push(`${pathname}?${params.toString()}`);
      });
    }, 300);

    return () => clearTimeout(timer);
  }, [searchTerm, paramName, pathname, router, searchParams]);

  const handleClear = () => {
    setSearchTerm("");
    const params = new URLSearchParams(searchParams.toString());
    params.delete(paramName);
    params.set("page", "1");
    lastPushedRef.current = "";
    startTransition(() => {
      router.push(`${pathname}?${params.toString()}`);
    });
  };

  return (
    <div className={`relative flex items-center w-full max-w-sm ${className}`}>
      <Search className="w-4 h-4 text-muted-foreground absolute left-3 top-1/2 -translate-y-1/2 pointer-events-none" />
      <input
        type="text"
        value={searchTerm}
        onChange={(e) => setSearchTerm(e.target.value)}
        placeholder={placeholder}
        aria-label={placeholder}
        className="w-full h-9 pl-9 pr-8 bg-background border border-input rounded-xl text-xs text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-ring transition-all"
      />
      <div className="absolute right-2.5 top-1/2 -translate-y-1/2 flex items-center gap-1">
        {isPending ? (
          <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />
        ) : searchTerm ? (
          <Tooltip>
            <TooltipTrigger asChild>
              <button
                type="button"
                onClick={handleClear}
                className="text-muted-foreground hover:text-foreground transition-colors p-0.5 rounded cursor-pointer"
                aria-label="Clear search"
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </TooltipTrigger>
            <TooltipContent side="bottom">Clear search</TooltipContent>
          </Tooltip>
        ) : null}
      </div>
    </div>
  );
}
