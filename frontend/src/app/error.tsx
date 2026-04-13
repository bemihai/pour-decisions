"use client";

/**
 * Root route (chatbot) error boundary.
 *
 * Rendered by Next.js when app/page.tsx throws. The chat page is a Client
 * Component, so errors typically come from API calls inside ChatInterface.
 * This boundary catches any unhandled throws from Server Components in the
 * root route segment.
 */

import { useEffect } from "react";
import { AlertCircle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ChatErrorProps {
  error: Error & { digest?: string };
  reset: () => void;
}

export default function ChatError({ error, reset }: ChatErrorProps) {
  useEffect(() => {
    console.error("[ChatbotPage] error:", error);
  }, [error]);

  return (
    <div className="flex flex-1 items-center justify-center p-8">
      <div className="flex flex-col items-center gap-4 rounded-xl border border-destructive/30 bg-destructive/5 px-8 py-16 text-center max-w-md">
        <AlertCircle className="size-10 text-destructive/70" />
        <div className="flex flex-col gap-1">
          <h2 className="type-section-title text-destructive">Something went wrong</h2>
          <p className="type-body text-muted-foreground">
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
          Try again
        </Button>
      </div>
    </div>
  );
}

