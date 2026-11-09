"use client";

import React from "react";
import { Button } from "@/components/ui/button";
import { Download, FileSpreadsheet, FileJson } from "lucide-react";
import { toast } from "sonner";

interface DataExportButtonProps {
  data: Record<string, unknown>[];
  filenamePrefix: string;
}

function escapeCsvCell(value: unknown): string {
  if (value === null || value === undefined) return "";
  const raw = Array.isArray(value)
    ? value.join("; ")
    : typeof value === "object"
    ? JSON.stringify(value)
    : String(value);
  if (raw.includes(",") || raw.includes('"') || raw.includes("\n")) {
    return `"${raw.replace(/"/g, '""')}"`;
  }
  return raw;
}

export function DataExportButton({ data, filenamePrefix }: DataExportButtonProps) {
  const handleExportCsv = () => {
    if (!data || data.length === 0) {
      toast.info("No records available to export.");
      return;
    }

    const headers = Object.keys(data[0]);
    const csvRows = [
      headers.join(","),
      ...data.map((row) => headers.map((h) => escapeCsvCell(row[h])).join(",")),
    ];
    const blob = new Blob([csvRows.join("\n")], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${filenamePrefix}_${new Date().toISOString().slice(0, 10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success(`Exported ${data.length} records as CSV.`);
  };

  const handleExportJson = () => {
    if (!data || data.length === 0) {
      toast.info("No records available to export.");
      return;
    }

    const blob = new Blob([JSON.stringify(data, null, 2)], {
      type: "application/json;charset=utf-8;",
    });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `${filenamePrefix}_${new Date().toISOString().slice(0, 10)}.json`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
    toast.success(`Exported ${data.length} records as JSON.`);
  };

  return (
    <div className="flex items-center gap-1.5">
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={handleExportCsv}
        disabled={data.length === 0}
        className="h-8 text-xs"
      >
        <FileSpreadsheet className="size-3.5 mr-1.5" />
        Export CSV
      </Button>
      <Button
        type="button"
        variant="outline"
        size="sm"
        onClick={handleExportJson}
        disabled={data.length === 0}
        className="h-8 text-xs"
      >
        <FileJson className="size-3.5 mr-1.5" />
        Export JSON
      </Button>
    </div>
  );
}
