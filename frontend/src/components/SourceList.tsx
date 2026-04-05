/**
 * SourceList component.
 *
 * Renders citation sources below an AI chat message.
 * Replaces the sources/web_sources HTML blocks inside format_assistant_message()
 * from src/ui/helper/display.py (~60 lines of HTML string generation).
 *
 * Usage:
 *   <SourceList sources={ragSources} />
 *   <SourceList sources={webSources} isWeb />
 */
import { BookOpen, Globe } from "lucide-react";

import { cn } from "@/lib/utils";
import type { Source, WebSource } from "@/lib/types";

// ---------------------------------------------------------------------------
// Relevance indicator
// ---------------------------------------------------------------------------

interface RelevanceStyle {
  color: string;
  label: string;
  /** Short visible text shown beside the dot (D2: not color-only). */
  text: string;
}

/**
 * Maps a 0-1 relevance score to a Tailwind background color, accessible label,
 * and a short visible text label. Mirrors get_relevance_indicator() from
 * src/ui/helper/display.py with added text for color-blind accessibility (D2).
 */
function getRelevanceIndicator(score: number | null): RelevanceStyle {
  if (score === null) return { color: "bg-gray-300", label: "Unknown relevance", text: "" };
  if (score >= 0.8) return { color: "bg-green-500", label: "Excellent relevance", text: "High" };
  if (score >= 0.6) return { color: "bg-yellow-400", label: "Good relevance", text: "Good" };
  if (score >= 0.4) return { color: "bg-orange-400", label: "Fair relevance", text: "Fair" };
  return { color: "bg-red-500", label: "Low relevance", text: "Low" };
}

// ---------------------------------------------------------------------------
// Sub-renderers
// ---------------------------------------------------------------------------

function RagSourceItem({ source }: { source: Source }) {
  const { color, label, text } = getRelevanceIndicator(source.relevance);
  return (
    <div className="flex items-start gap-1.5 text-xs opacity-80">
      <span
        className={cn("mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full", color)}
        title={label}
        aria-label={label}
      />
      {text && <span className="text-muted-foreground shrink-0">{text}</span>}
      <span className="font-medium break-words min-w-0">{source.name}</span>
      {source.page !== null && (
        <span className="opacity-60 ml-auto pl-2 shrink-0">p.&nbsp;{source.page}</span>
      )}
    </div>
  );
}

function WebSourceItem({ source }: { source: WebSource }) {
  return (
    <a
      href={source.url}
      target="_blank"
      rel="noopener noreferrer"
      className="flex items-start gap-1.5 text-xs opacity-80 hover:opacity-100 underline underline-offset-2 break-all"
    >
      <Globe className="h-3 w-3 mt-0.5 shrink-0" aria-hidden />
      {source.title || source.url}
    </a>
  );
}

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

interface SourceListProps {
  sources: Source[] | WebSource[];
  isWeb?: boolean;
  className?: string;
}

export default function SourceList({ sources, isWeb = false, className }: SourceListProps) {
  if (sources.length === 0) return null;

  return (
    <div className={cn("mt-3 pt-3 border-t border-black/10", className)}>
      <div className="flex items-center gap-1.5 mb-2 text-xs font-semibold uppercase tracking-wide opacity-70">
        {isWeb ? (
          <Globe className="h-3 w-3" aria-hidden />
        ) : (
          <BookOpen className="h-3 w-3" aria-hidden />
        )}
        {isWeb ? "Web Sources" : "Sources"}
      </div>

      <div className="flex flex-col gap-1.5">
        {isWeb
          ? (sources as WebSource[]).map((src, i) => <WebSourceItem key={i} source={src} />)
          : (sources as Source[]).map((src, i) => <RagSourceItem key={i} source={src} />)}
      </div>
    </div>
  );
}

