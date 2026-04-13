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
      <p className="relative type-body pr-6 leading-relaxed italic text-foreground/80">{notes}</p>
    </div>
  );
}

