"use client";

import React from "react";
import { useRouter, usePathname, useSearchParams } from "next/navigation";
import { Button } from "@/components/ui/button";
import { ChevronLeft, ChevronRight } from "lucide-react";

interface DataTablePaginationProps {
  total: number;
  page: number;
  limit: number;
  totalPages: number;
  itemLabel?: string;
}

export function DataTablePagination({
  total,
  page,
  limit,
  totalPages,
  itemLabel = "records",
}: DataTablePaginationProps) {
  const router = useRouter();
  const pathname = usePathname();
  const searchParams = useSearchParams();

  const navigateToPage = (newPage: number) => {
    const params = new URLSearchParams(searchParams.toString());
    params.set("page", newPage.toString());
    router.push(`${pathname}?${params.toString()}`);
  };

  if (total === 0) return null;

  const startCount = (page - 1) * limit + 1;
  const endCount = Math.min(page * limit, total);

  return (
    <div className="flex flex-col sm:flex-row items-center justify-between gap-3 px-4 py-3 border-t border-border text-xs text-muted-foreground bg-muted/20">
      <span>
        Showing <strong className="font-semibold text-foreground">{startCount}</strong> to{" "}
        <strong className="font-semibold text-foreground">{endCount}</strong> of{" "}
        <strong className="font-semibold text-foreground">{total}</strong> {itemLabel}
      </span>

      <div className="flex items-center gap-2">
        <Button
          variant="outline"
          size="sm"
          disabled={page <= 1}
          onClick={() => navigateToPage(page - 1)}
          aria-label="Go to previous page"
          className="h-8 px-2.5 text-xs flex items-center gap-1 cursor-pointer"
        >
          <ChevronLeft className="w-3.5 h-3.5" />
          <span>Previous</span>
        </Button>

        <span className="font-mono text-xs font-medium px-1">
          Page {page} of {totalPages}
        </span>

        <Button
          variant="outline"
          size="sm"
          disabled={page >= totalPages}
          onClick={() => navigateToPage(page + 1)}
          aria-label="Go to next page"
          className="h-8 px-2.5 text-xs flex items-center gap-1 cursor-pointer"
        >
          <span>Next</span>
          <ChevronRight className="w-3.5 h-3.5" />
        </Button>
      </div>
    </div>
  );
}
