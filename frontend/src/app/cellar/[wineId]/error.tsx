"use client";

/**
 * Wine detail page error boundary.
 *
 * Shown when page.tsx throws (e.g. API unreachable, wine not found falls
 * through as a 500 rather than a 404 handled by notFound()).
 */

import { useEffect } from "react";
import Link from "next/link";
import { AlertCircle, ArrowLeft, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface WineDetailErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function WineDetailError({ error, reset }: WineDetailErrorProps) {
  useEffect(() => {
    console.error("[WineDetailPage] error:", error);
  }, [error]);

  return (
    <div className="container mx-auto flex max-w-7xl items-center justify-center px-4 py-24">
      <div className="flex max-w-md flex-col items-center gap-4 rounded-xl border border-destructive/30 bg-destructive/5 px-8 py-12 text-center">
        <AlertCircle className="size-10 text-destructive/70" />
        <div className="flex flex-col gap-1">
          <h2 className="type-section-title text-destructive">Failed to load wine</h2>
          <p className="type-body text-muted-foreground">
            {error.message.includes("fetch")
              ? "The API server appears to be offline. Start it with make api and try again."
              : error.message}
          </p>
        </div>
        <div className="flex gap-2">
          <Button
            variant="outline"
            onClick={reset}
            className="gap-2 border-destructive/40 text-destructive hover:bg-destructive/10"
          >
            <RefreshCw className="size-4" />
            Try again
          </Button>
          <Button asChild variant="outline">
            <Link href="/cellar">
              <ArrowLeft className="mr-1.5 size-4" />
              Back to Cellar
            </Link>
          </Button>
        </div>
      </div>
    </div>
  );
}

