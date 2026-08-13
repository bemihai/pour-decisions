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
 * and a short visible text label. Uses brand colors from design-tokens.
 */
function getRelevanceIndicator(score: number | null): RelevanceStyle {
  if (score === null) return { color: "bg-muted", label: "Unknown relevance", text: "" };
  if (score >= 0.8) return { color: "bg-green-500", label: "High relevance", text: "High" };
  if (score >= 0.4) return { color: "bg-yellow-500", label: "Medium relevance", text: "Medium" };
  return { color: "bg-red-500", label: "Low relevance", text: "Low" };
}

interface GroupedRagSource {
  name: string;
  pages: number[];
  relevance: number | null;
}

interface RagSourceAccumulator {
  name: string;
  pages: Set<number>;
  relevance: number | null;
}

/** Normalize only differences that do not change the source's visible identity. */
function normalizeSourceName(name: string): string {
  return name.trim().replace(/\s+/g, " ") || "Unknown";
}

/**
 * Collapse chunk-level citations into one row per displayed source name.
 *
 * The API intentionally retains one source per cited context chunk so inline
 * citation numbers continue to map to the generated context. Grouping belongs
 * here, at the display boundary, where it also fixes messages restored from
 * localStorage before invalid page sentinels were normalized by the API.
 */
function groupRagSources(sources: Source[]): GroupedRagSource[] {
  const grouped = new Map<string, RagSourceAccumulator>();

  for (const source of sources) {
    const name = normalizeSourceName(source.name);
    const key = name.toLowerCase();
    const existing = grouped.get(key);
    const group = existing ?? { name, pages: new Set<number>(), relevance: null };

    if (source.page !== null && Number.isInteger(source.page) && source.page > 0) {
      group.pages.add(source.page);
    }
    if (
      source.relevance !== null &&
      (group.relevance === null || source.relevance > group.relevance)
    ) {
      group.relevance = source.relevance;
    }

    if (!existing) grouped.set(key, group);
  }

  return Array.from(grouped.values(), (source) => ({
    name: source.name,
    pages: Array.from(source.pages).sort((left, right) => left - right),
    relevance: source.relevance,
  }));
}

// ---------------------------------------------------------------------------
// Sub-renderers
// ---------------------------------------------------------------------------

function RagSourceItem({ source }: { source: GroupedRagSource }) {
  const { color, label, text } = getRelevanceIndicator(source.relevance);
  const pageLabel =
    source.pages.length === 1
      ? `p.\u00a0${source.pages[0]}`
      : source.pages.length > 1
        ? `pp.\u00a0${source.pages.join(", ")}`
        : null;

  return (
    <div className="flex items-start gap-1.5 type-caption opacity-80">
      <span
        className={cn("mt-0.5 h-2.5 w-2.5 shrink-0 rounded-full", color)}
        title={label}
        aria-label={label}
      />
      {text && <span className="text-muted-foreground shrink-0">{text}</span>}
      <span className="font-medium break-words min-w-0">{source.name}</span>
      {pageLabel !== null && (
        <span className="opacity-60 ml-auto pl-2 shrink-0">{pageLabel}</span>
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
      className="flex items-start gap-1.5 type-caption opacity-80 hover:opacity-100 underline underline-offset-2 break-all"
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
  const ragSources = isWeb ? [] : groupRagSources(sources as Source[]);

  return (
    <div className={cn("mt-3 pt-3 border-t border-black/10", className)}>
      <div className="flex items-center gap-1.5 mb-2 type-label font-semibold uppercase tracking-wide opacity-70">
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
          : ragSources.map((src) => <RagSourceItem key={src.name} source={src} />)}
      </div>
    </div>
  );
}
