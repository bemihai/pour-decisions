"use client";

import { useRef, useState } from "react";
import { useRouter } from "next/navigation";
import { Loader2, Sparkles } from "lucide-react";

import { Button } from "@/components/ui/button";
import { generateGrapeDescription } from "@/lib/api";
import type { GrapeDetail } from "@/lib/types";

function GrapeDescription({ grape }: { grape: GrapeDetail }) {
  const router = useRouter();
  const pending = useRef(false);
  const [description, setDescription] = useState(grape.description);
  const [isGenerating, setIsGenerating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function handleGenerate() {
    if (pending.current) return;
    pending.current = true;
    setIsGenerating(true);
    setError(null);
    try {
      const result = await generateGrapeDescription(grape.id);
      if (!result.success || !result.description) {
        throw new Error("No description returned");
      }
      setDescription(result.description);
      router.refresh();
    } catch {
      setError("Could not generate a grape description. Any previous description has been kept. Please retry.");
    } finally {
      pending.current = false;
      setIsGenerating(false);
    }
  }

  return (
    <div className="flex flex-col gap-2">
      <h3 className="text-sm font-medium capitalize">{grape.name}</h3>
      <p className="text-sm leading-relaxed text-muted-foreground">
        {description || "No grape description available yet."}
      </p>
      <Button
        variant="outline"
        size="sm"
        className="w-fit"
        onClick={handleGenerate}
        disabled={isGenerating}
        aria-label={`${description ? "Regenerate" : "Generate"} ${grape.name} description`}
      >
        {isGenerating ? (
          <Loader2 className="mr-1.5 size-3.5 animate-spin" aria-hidden="true" />
        ) : (
          <Sparkles className="mr-1.5 size-3.5" aria-hidden="true" />
        )}
        {isGenerating ? "Generating…" : description ? "Regenerate Description" : "Generate Description"}
      </Button>
      {error && <p role="alert" className="type-caption text-destructive">{error}</p>}
    </div>
  );
}

export default function GrapeDescriptions({ grapes }: { grapes: GrapeDetail[] }) {
  if (grapes.length === 0) return null;
  return (
    <section aria-label="About the Grapes" className="flex flex-col gap-4">
      <h2 className="type-label font-semibold uppercase tracking-wide text-muted-foreground">About the Grapes</h2>
      {grapes.map((grape) => (
        <GrapeDescription key={`${grape.id}:${grape.description ?? ""}`} grape={grape} />
      ))}
    </section>
  );
}
