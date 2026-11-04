"use client";

import React, { useSyncExternalStore } from "react";
import { cn } from "@/lib/utils";

const emptySubscribe = () => () => {};

const SHORT_MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
] as const;

export interface FormatTimestampOptions {
  /**
   * Controls which parts of the timestamp to render:
   * - "datetime": "03 Nov 2026, 14:35" (default)
   * - "date":     "03 Nov 2026"
   * - "time":     "14:35"
   */
  mode?: "datetime" | "date" | "time";
  /**
   * Whether to append the local timezone abbreviation. Defaults to false.
   */
  showTimezone?: boolean;
  /**
   * Whether to include seconds in time output. Defaults to false.
   */
  showSeconds?: boolean;
  /**
   * Internal flag for deterministic SSR fallback before client mount.
   */
  useUtc?: boolean;
}

/**
 * Formats a Date or date-like value into a compact, human-readable string.
 * Default format ("datetime"): `DD MMM YYYY, HH:mm` (e.g. `03 Nov 2026, 14:35`)
 * Default "date" format:       `DD MMM YYYY`        (e.g. `03 Nov 2026`)
 * Default "time" format:       `HH:mm`              (e.g. `14:35`)
 */
export function formatCompactTimestamp(
  dateInput: string | Date | number | null | undefined,
  options: FormatTimestampOptions = {}
): string | null {
  if (!dateInput) return null;
  const parsedDate = dateInput instanceof Date ? dateInput : new Date(dateInput);
  if (Number.isNaN(parsedDate.getTime())) return null;

  const {
    mode = "datetime",
    showTimezone = false,
    showSeconds = false,
    useUtc = false,
  } = options;

  const day = String(useUtc ? parsedDate.getUTCDate() : parsedDate.getDate()).padStart(2, "0");
  const monthIdx = useUtc ? parsedDate.getUTCMonth() : parsedDate.getMonth();
  const month = SHORT_MONTHS[monthIdx];
  const year = useUtc ? parsedDate.getUTCFullYear() : parsedDate.getFullYear();
  const datePart = `${day} ${month} ${year}`;

  const hours = String(useUtc ? parsedDate.getUTCHours() : parsedDate.getHours()).padStart(2, "0");
  const minutes = String(useUtc ? parsedDate.getUTCMinutes() : parsedDate.getMinutes()).padStart(2, "0");
  const seconds = String(useUtc ? parsedDate.getUTCSeconds() : parsedDate.getSeconds()).padStart(2, "0");
  const timePart = showSeconds ? `${hours}:${minutes}:${seconds}` : `${hours}:${minutes}`;

  let tzSuffix = "";
  if (showTimezone && !useUtc) {
    try {
      const tzPart = new Intl.DateTimeFormat(undefined, { timeZoneName: "short" })
        .formatToParts(parsedDate)
        .find((p) => p.type === "timeZoneName")?.value;
      if (tzPart) {
        tzSuffix = ` ${tzPart}`;
      }
    } catch {
      tzSuffix = "";
    }
  }

  if (mode === "date") {
    return datePart;
  }

  if (mode === "time") {
    return `${timePart}${tzSuffix}`;
  }

  return `${datePart}, ${timePart}${tzSuffix}`;
}

export interface FormattedTimestampProps extends React.HTMLAttributes<HTMLSpanElement> {
  date: string | Date | number | null | undefined;
  /**
   * Controls which parts of the timestamp to render:
   * - "datetime": "03 Nov 2026, 14:35" (default)
   * - "date":     "03 Nov 2026"
   * - "time":     "14:35"
   */
  mode?: "datetime" | "date" | "time";
  /**
   * Whether to append the local timezone abbreviation. Defaults to false.
   */
  showTimezone?: boolean;
  /**
   * Whether to include seconds in time output. Defaults to false.
   */
  showSeconds?: boolean;
  fallback?: string;
}

export function FormattedTimestamp({
  date,
  mode = "datetime",
  showTimezone = false,
  showSeconds = false,
  fallback = "N/A",
  className,
  ...props
}: FormattedTimestampProps) {
  const mounted = useSyncExternalStore(
    emptySubscribe,
    () => true,
    () => false
  );

  if (!date) {
    return (
      <span className={cn("text-muted-foreground", className)} {...props}>
        {fallback}
      </span>
    );
  }

  const parsedDate = date instanceof Date ? date : new Date(date);
  if (Number.isNaN(parsedDate.getTime())) {
    return (
      <span className={cn("text-muted-foreground", className)} {...props}>
        {fallback}
      </span>
    );
  }

  const displayValue =
    formatCompactTimestamp(parsedDate, {
      mode,
      showTimezone,
      showSeconds,
      useUtc: !mounted,
    }) ?? fallback;

  const tooltipValue =
    formatCompactTimestamp(parsedDate, {
      mode: "datetime",
      showTimezone: false,
      showSeconds: false,
      useUtc: !mounted,
    }) ?? undefined;

  return (
    <span
      title={tooltipValue}
      className={cn("tabular-nums", className)}
      suppressHydrationWarning
      {...props}
    >
      {displayValue}
    </span>
  );
}
