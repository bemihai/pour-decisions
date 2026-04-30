/**
 * TastingNote component.
 *
 * Renders a user's personal tasting note with a journal-like visual treatment:
 * warm parchment background tint, brand-gold left-border accent, and a decorative
 * opening-quote icon.  Used in ConsumedWineCard (TasteHistory) to make
 * user-generated tasting content a first-class visual element.
 *
 * Usage:
 *   <TastingNote notes="Rich cherry and dark plum with velvety tannins." />
 */

import { Quote } from "lucide-react";

import { cn } from "@/lib/utils";

const NOTE_DATE_PREFIX = /^\[(\d{4}-\d{2}-\d{2})]\s*/;

function dedupeNotes(items: string[]): string[] {
  const seen = new Set<string>();
  const unique: string[] = [];

  for (const item of items) {
    const key = item.replace(/\s+/g, " ").trim().toLowerCase();
    if (!key || seen.has(key)) {
      continue;
    }
    seen.add(key);
    unique.push(item);
  }

  return unique;
}

function parseTastingNotes(notes: string): string[] {
  const chunks = notes
    .split(/\r?\n+/)
    .map((part) => part.trim())
    .filter(Boolean)
    .flatMap((part) => part.split(/(?=\[\d{4}-\d{2}-\d{2}]\s*)/g))
    .map((part) => part.trim())
    .filter(Boolean)
    .map((part) => part.replace(NOTE_DATE_PREFIX, "").trim())
    .filter(Boolean);

  const parsed = chunks.length > 0 ? chunks : [notes.trim()];
  return dedupeNotes(parsed);
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TastingNoteProps {
  notes: string;
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TastingNote({ notes, className }: TastingNoteProps) {
  const items = parseTastingNotes(notes);
  const hasMultiple = items.length > 1;

  return (
    <div
      className={cn(
        "relative overflow-hidden rounded-md border-l-[3px] border-brand-gold bg-brand-gold-light/30 px-4 pb-3 pt-3",
        className,
      )}
    >
      {/* Decorative background quote mark */}
      <Quote
        className="pointer-events-none absolute -right-1 -top-1 size-10 rotate-180 text-brand-gold/20 select-none"
        aria-hidden="true"
      />

      {/* The note text */}
      {hasMultiple ? (
        <ul className="relative list-disc space-y-1 pl-5 pr-6 text-foreground/80">
          {items.map((item, index) => (
            <li key={`${item}-${index}`} className="type-body leading-relaxed italic">
              {item}
            </li>
          ))}
        </ul>
      ) : (
        <p className="relative type-body pr-6 leading-relaxed italic text-foreground/80">{items[0]}</p>
      )}
    </div>
  );
}

