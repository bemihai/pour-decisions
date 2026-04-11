/**
 * Rating component.
 *
 * Displays a wine rating (0-100 scale) in one of three visual variants:
 *
 *   compact — coloured number only; used in card headers and table rows.
 *   full    — number + quality label + star icon; used in ConsumedWineCard.
 *   gauge   — thin fill bar + number; used as an inline indicator.
 *
 * Color reflects the quality tier (emerald → green → lime → yellow → orange → red).
 */

import { Star } from "lucide-react";
import { cn, formatRating, getRatingLabel, ratingColor } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Hex color matching each Tailwind ratingColor() tier. */
function ratingHex(rating: number): string {
  if (rating >= 94) return "#059669";
  if (rating >= 90) return "#16a34a";
  if (rating >= 86) return "#65a30d";
  if (rating >= 80) return "#ca8a04";
  if (rating >= 70) return "#f97316";
  return "#ef4444";
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface RatingProps {
  rating: number | null | undefined;
  /**
   * compact — coloured integer (default)
   * full    — "87/100" + label + star
   * gauge   — thin fill bar + number
   */
  variant?: "compact" | "full" | "gauge";
  className?: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function Rating({ rating, variant = "compact", className }: RatingProps) {
  if (rating == null) {
    if (variant === "full") {
      return (
        <span className={cn("type-caption text-muted-foreground", className)}>Not rated</span>
      );
    }
    return null;
  }

  const rounded = Math.round(rating);

  if (variant === "compact") {
    return (
      <span
        className={cn("text-sm font-semibold tabular-nums", ratingColor(rating), className)}
        aria-label={`Rating: ${rounded}/100 — ${getRatingLabel(rating)}`}
      >
        {rounded}
      </span>
    );
  }

  if (variant === "full") {
    return (
      <div className={cn("flex flex-col items-end gap-0.5", className)}>
        <span className={cn("font-bold leading-none", ratingColor(rating))}>
          {formatRating(rating)}
        </span>
        <span className="flex items-center gap-1 type-caption text-muted-foreground">
          <Star className="size-3 fill-amber-400 text-amber-400" aria-hidden="true" />
          {getRatingLabel(rating)}
        </span>
      </div>
    );
  }

  // gauge variant
  return (
    <div className={cn("flex items-center gap-2", className)}>
      <div
        className="relative h-1.5 w-16 overflow-hidden rounded-full bg-muted"
        role="meter"
        aria-valuenow={rounded}
        aria-valuemin={0}
        aria-valuemax={100}
        aria-label={`Rating ${rounded}/100`}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-full"
          style={{ width: `${rounded}%`, backgroundColor: ratingHex(rating) }}
        />
      </div>
      <span className={cn("type-caption font-medium tabular-nums", ratingColor(rating))}>
        {rounded}
      </span>
    </div>
  );
}

