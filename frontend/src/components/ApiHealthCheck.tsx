"use client";

import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import { getCellarStats } from "@/lib/api";
import { formatCurrency } from "@/lib/utils";
import { ApiError } from "@/lib/api";

/**
 * Step 0.12 — End-to-end connection verification.
 *
 * Fetches GET /api/cellar/stats, logs the response to the browser console,
 * and renders a visual status panel so the connection can be confirmed
 * without opening DevTools.
 *
 * Remove or replace this component once Phase 1 pages are built.
 */
export default function ApiHealthCheck() {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["cellar-stats-healthcheck"],
    queryFn: getCellarStats,
    retry: 1,
  });

  // Acceptance criterion: result visible in browser console.
  useEffect(() => {
    if (data) {
      console.log("[ApiHealthCheck] GET /api/cellar/stats →", data);
    }
  }, [data]);

  useEffect(() => {
    if (isError) {
      console.error("[ApiHealthCheck] GET /api/cellar/stats failed →", error);
    }
  }, [isError, error]);

  const statusDot = isLoading
    ? "bg-yellow-400 animate-pulse"
    : isError
      ? "bg-red-500"
      : "bg-green-500";

  const statusText = isLoading
    ? "Connecting to API…"
    : isError
      ? `API unreachable — ${error instanceof ApiError ? `HTTP ${error.status}: ${error.message}` : String(error)}`
      : "API connected";

  return (
    <div className="w-full max-w-md rounded-lg border border-border bg-card p-4 shadow-sm text-sm">
      {/* Connection status row */}
      <div className="flex items-center gap-2 mb-3">
        <span className={`h-2.5 w-2.5 rounded-full shrink-0 ${statusDot}`} />
        <span className={isError ? "text-red-600" : "text-foreground"}>{statusText}</span>
      </div>

      {/* Stats summary once loaded */}
      {data && (
        <div className="grid grid-cols-2 gap-x-6 gap-y-1 text-muted-foreground">
          <span>Total bottles</span>
          <span className="text-foreground font-medium tabular-nums">
            {data.overview.total_bottles.toLocaleString()}
          </span>

          <span>Unique wines</span>
          <span className="text-foreground font-medium tabular-nums">
            {data.overview.unique_wines.toLocaleString()}
          </span>

          <span>Ready to drink</span>
          <span className="text-foreground font-medium tabular-nums">
            {data.drinking_stats.ready_to_drink.toLocaleString()}
          </span>

          {data.value_stats.by_currency[0] && (
            <>
              <span>Cellar value</span>
              <span className="text-foreground font-medium tabular-nums">
                {formatCurrency(
                  data.value_stats.by_currency[0].total_value,
                  data.value_stats.by_currency[0].currency,
                )}
              </span>
            </>
          )}
        </div>
      )}

      <p className="mt-3 text-xs text-muted-foreground">
        Check the browser console for the full JSON response.
      </p>
    </div>
  );
}

