"use client";

import React, { useState } from "react";
import { Button } from "@/components/ui/button";
import { Download, FileSpreadsheet, Loader2 } from "lucide-react";
import { toast } from "sonner";

interface ExportLogsButtonProps {
  logs: {
    id: string | number;
    createdAt: string | Date;
    personName?: string | null;
    personCode?: string | null;
    accessStatus: string;
    matchStatus: string;
    reason: string;
    doorAction: string;
    confidenceScore?: number | null;
    processingTimeMs?: number | null;
  }[];
}

export function ExportLogsButton({ logs }: ExportLogsButtonProps) {
  const [isExporting, setIsExporting] = useState(false);

  const handleExportCSV = () => {
    if (logs.length === 0) {
      toast.error("No access logs available to export.");
      return;
    }
    setIsExporting(true);

    try {
      const headers = [
        "Log ID",
        "Timestamp",
        "Personnel Name",
        "Employee Code",
        "Access Status",
        "Match Status",
        "Reason",
        "Door Action",
        "Similarity Score",
        "Processing Latency (ms)",
      ];

      const rows = logs.map((log) => [
        log.id,
        new Date(log.createdAt).toISOString(),
        `"${(log.personName || "Unknown").replace(/"/g, '""')}"`,
        `"${(log.personCode || "N/A").replace(/"/g, '""')}"`,
        log.accessStatus.toUpperCase(),
        log.matchStatus.toUpperCase(),
        log.reason,
        log.doorAction.toUpperCase(),
        log.confidenceScore != null ? (log.confidenceScore * 100).toFixed(2) + "%" : "N/A",
        log.processingTimeMs ?? "N/A",
      ]);

      const csvContent = [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");
      const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.setAttribute("href", url);
      link.setAttribute("download", `driveguard_sessions_${Date.now()}.csv`);
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);

      toast.success(`Exported ${logs.length} access logs to CSV.`);
    } catch (err) {
      console.error("Export CSV failed:", err);
      toast.error("Failed to export logs.");
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <Button
      variant="outline"
      size="sm"
      onClick={handleExportCSV}
      disabled={isExporting}
      className="gap-2 text-xs font-semibold cursor-pointer border-border hover:bg-muted"
    >
      {isExporting ? (
        <Loader2 className="w-3.5 h-3.5 animate-spin" />
      ) : (
        <Download className="w-3.5 h-3.5 text-primary" />
      )}
      <span>Export CSV ({logs.length})</span>
    </Button>
  );
}
