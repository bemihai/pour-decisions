"use client";

/**
 * CellarSyncButton component.
 *
 * Triggers a CellarTracker sync via POST /api/cellar/sync and provides
 * non-blocking feedback via an inline result banner.
 *
 * On success:
 *   - Invalidates the TanStack Query ["inventory"] cache so CellarInventory
 *     auto-refetches without a full page reload.
 *   - Calls router.refresh() so the Next.js Server Component re-fetches
 *     cellar stats and chart data.
 *
 * Replaces st.button("Sync") -> st.rerun() from src/ui/pages/cellar.py.
 */

import { useEffect, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { CheckCircle2, RefreshCw, XCircle } from "lucide-react";

import type { SyncResponse } from "@/lib/types";
import { syncCellarTracker, ApiError } from "@/lib/api";
import { Button } from "@/components/ui/button";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type BannerState =
  | { kind: "idle" }
  | { kind: "success"; response: SyncResponse }
  | { kind: "error"; message: string };

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CellarSyncButton() {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [banner, setBanner] = useState<BannerState>({ kind: "idle" });

  const { mutate, isPending } = useMutation<SyncResponse, Error>({
    mutationFn: syncCellarTracker,
    onSuccess: (data) => {
      if (!data.success && data.error_message) {
        setBanner({ kind: "error", message: data.error_message });
        return;
      }
      setBanner({ kind: "success", response: data });
      // Invalidate client-side inventory cache so the list auto-refetches.
      queryClient.invalidateQueries({ queryKey: ["inventory"] });
      // Re-run Server Component data fetching (stats, charts, filter options).
      router.refresh();
    },
    onError: (err) => {
      const message =
        err instanceof ApiError
          ? err.message
          : (err as Error).message ?? "Sync failed. Please try again.";
      setBanner({ kind: "error", message });
    },
  });

  // Auto-dismiss the result banner after 5 seconds.
  useEffect(() => {
    if (banner.kind === "idle") return;
    const timer = setTimeout(() => setBanner({ kind: "idle" }), 5000);
    return () => clearTimeout(timer);
  }, [banner.kind]);

  function handleSync() {
    setBanner({ kind: "idle" });
    mutate();
  }

  return (
    <div className="flex flex-col items-end gap-2">
      <Button
        onClick={handleSync}
        disabled={isPending}
        variant="outline"
        className="gap-2 border-purple-300 text-purple-700 hover:bg-purple-50 hover:text-purple-800 dark:border-purple-700 dark:text-purple-400 dark:hover:bg-purple-950"
      >
        <RefreshCw className={cn("size-4", isPending && "animate-spin")} />
        {isPending ? "Syncing..." : "Sync CellarTracker"}
      </Button>

      {banner.kind !== "idle" && (
        <div
          role="status"
          aria-live="polite"
          className={cn(
            "flex items-start gap-2 rounded-lg border px-3 py-2 text-sm max-w-xs",
            banner.kind === "success"
              ? "border-emerald-200 bg-emerald-50 text-emerald-800 dark:border-emerald-800 dark:bg-emerald-950/40 dark:text-emerald-300"
              : "border-red-200 bg-red-50 text-red-800 dark:border-red-800 dark:bg-red-950/40 dark:text-red-300",
          )}
        >
          {banner.kind === "success" ? (
            <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-emerald-600 dark:text-emerald-400" />
          ) : (
            <XCircle className="mt-0.5 size-4 shrink-0 text-red-600 dark:text-red-400" />
          )}

          <div className="flex-1 min-w-0">
            <p className="font-medium">
              {banner.kind === "success" ? "Sync complete" : "Sync failed"}
            </p>
            <p className="text-xs text-muted-foreground truncate">
              {banner.kind === "success"
                ? `${banner.response.wines_imported} wines · ${banner.response.bottles_imported} bottles imported${
                    banner.response.errors.length > 0
                      ? ` (${banner.response.errors.length} warning${banner.response.errors.length > 1 ? "s" : ""})`
                      : ""
                  }`
                : banner.message}
            </p>
          </div>

          <TooltipProvider>
            <Tooltip>
              <TooltipTrigger asChild>
                <button
                  onClick={() => setBanner({ kind: "idle" })}
                  aria-label="Dismiss notification"
                  className="ml-1 shrink-0 rounded text-muted-foreground hover:text-foreground focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring"
                >
                  &times;
                </button>
              </TooltipTrigger>
              <TooltipContent side="top">Dismiss</TooltipContent>
            </Tooltip>
          </TooltipProvider>
        </div>
      )}
    </div>
  );
}

