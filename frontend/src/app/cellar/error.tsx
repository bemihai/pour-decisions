"use client";

/**
 * Cellar page error boundary.
 *
 * Next.js renders this file when the Server Component (page.tsx) throws —
 * most commonly when the FastAPI backend is unreachable. Must be a Client
 * Component so it can use the reset() callback.
 */

import { useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";

import { Button } from "@/components/ui/button";

interface CellarErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function CellarError({ error, reset }: CellarErrorProps) {
  useEffect(() => {
    // Log to the browser console so errors are visible during development.
    console.error("[CellarPage] data fetch error:", error);
  }, [error]);

  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      <div className="flex flex-col items-center justify-center gap-4 rounded-xl border border-destructive/30 bg-destructive/5 px-8 py-16 text-center">
        <AlertCircle className="size-10 text-destructive/70" />

        <div className="flex flex-col gap-1">
          <h2 className="text-lg font-semibold text-destructive">
            Unable to load cellar data
          </h2>
          <p className="max-w-sm text-sm text-muted-foreground">
            {error.message.includes("fetch")
              ? "The API server appears to be offline. Start it with make api and try again."
              : error.message}
          </p>
        </div>

        <Button
          variant="outline"
          onClick={reset}
          className="gap-2 border-destructive/40 text-destructive hover:bg-destructive/10"
        >
          <RefreshCw className="size-4" />
          Retry
        </Button>
      </div>
    </div>
  );
}

