"use client";

/**
 * WineDescriptionGenerator — isolated client island for the wine detail page.
 *
 * Renders a "Generate Description" button and manages the async state for
 * POST /api/wines/:id/description. On success it replaces itself with the
 * generated description text so the detail page stays server-rendered otherwise.
 */

import { useState } from "react";
import { Loader2, Sparkles } from "lucide-react";

import { generateWineDescription } from "@/lib/api";
import { Button } from "@/components/ui/button";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

interface WineDescriptionGeneratorProps {
  wineId: number;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function WineDescriptionGenerator({ wineId }: WineDescriptionGeneratorProps) {
  const [description, setDescription] = useState<string | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  if (description) {
    return <p className="text-sm leading-relaxed text-muted-foreground">{description}</p>;
  }

  async function handleGenerate() {
    setIsGenerating(true);
    setError(null);
    try {
      const result = await generateWineDescription(wineId);
      if (result.success && result.description) {
        setDescription(result.description);
      } else {
        setError("Generation returned no description.");
      }
    } catch {
      setError("Failed to generate description. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <p className="text-sm text-muted-foreground">No description available yet.</p>
      <Button
        variant="outline"
        size="sm"
        onClick={handleGenerate}
        disabled={isGenerating}
        className="w-fit"
      >
        {isGenerating ? (
          <Loader2 className="mr-1.5 size-3.5 animate-spin" aria-hidden="true" />
        ) : (
          <Sparkles className="mr-1.5 size-3.5" aria-hidden="true" />
        )}
        {isGenerating ? "Generating…" : "Generate Description"}
      </Button>
      {error && <p className="type-caption text-destructive">{error}</p>}
    </div>
  );
}

