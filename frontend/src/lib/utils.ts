import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// ---------------------------------------------------------------------------
// Rating helpers
// Mirrors src/etl/utils.py :: get_rating_description
// ---------------------------------------------------------------------------

/** Quality tiers matching the Python get_rating_description function. */
export function getRatingLabel(rating: number | null | undefined): string {
  if (rating == null) return "Not Rated";
  if (rating < 70) return "Below Average";
  if (rating < 80) return "Average";
  if (rating < 86) return "Good";
  if (rating < 90) return "Very Good";
  if (rating < 94) return "Excellent";
  if (rating < 98) return "Outstanding";
  return "Exceptional";
}

/** Format a 0-100 rating as "87/100" or "N/A". */
export function formatRating(rating: number | null | undefined): string {
  if (rating == null) return "N/A";
  return `${Math.round(rating)}/100`;
}

/** Return a Tailwind text-color class based on rating tier. */
export function ratingColor(rating: number | null | undefined): string {
  if (rating == null) return "text-muted-foreground";
  if (rating >= 94) return "text-emerald-600";
  if (rating >= 90) return "text-green-600";
  if (rating >= 86) return "text-lime-600";
  if (rating >= 80) return "text-yellow-600";
  if (rating >= 70) return "text-orange-500";
  return "text-red-500";
}

// ---------------------------------------------------------------------------
// Vintage helpers
// ---------------------------------------------------------------------------

/** Display vintage as a string, "NV" for non-vintage wines. */
export function formatVintage(vintage: number | null | undefined): string {
  return vintage != null ? String(vintage) : "NV";
}

// ---------------------------------------------------------------------------
// Currency / value helpers
// ---------------------------------------------------------------------------

/** Format a monetary value, e.g. "RON 1,234" or "EUR 567.89". */
export function formatCurrency(
  value: number | null | undefined,
  currency = "RON"
): string {
  if (value == null) return "N/A";
  return `${currency} ${value.toLocaleString("en-US", {
    minimumFractionDigits: 0,
    maximumFractionDigits: 0,
  })}`;
}

// ---------------------------------------------------------------------------
// Drinking-window / drink-index helpers
// Mirrors src/ui/helper/display.py :: get_drinking_status
// ---------------------------------------------------------------------------

export interface DrinkingStatus {
  label: string;
  /** Tailwind text-color class */
  colorClass: string;
  /** Hex color for Plotly / non-Tailwind usage */
  hex: string;
  /** Normalised 0–100 position within the collection */
  normalised: number;
}

/**
 * Derive a drinking status for a single wine given its drink_index and the
 * full collection of indices (for percentile normalisation).
 */
export function getDrinkingStatus(
  drinkIndex: number | null | undefined,
  allIndices: number[]
): DrinkingStatus {
  let normalised = 50;

  if (drinkIndex != null && allIndices.length > 0) {
    const sorted = [...allIndices].sort((a, b) => a - b);
    const p5 = sorted[Math.max(0, Math.floor(sorted.length * 0.05))];
    const p95 = sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95))];
    const lo = p5 !== p95 ? p5 : sorted[0];
    const hi = p5 !== p95 ? p95 : sorted[sorted.length - 1];
    normalised =
      hi === lo ? 50 : Math.max(0, Math.min(100, ((drinkIndex - lo) / (hi - lo)) * 100));
  }

  if (normalised >= 75)
    return { label: "Peak Drinking", colorClass: "text-green-600", hex: "#4CAF50", normalised };
  if (normalised >= 50)
    return { label: "Ready to Drink", colorClass: "text-yellow-500", hex: "#FFC107", normalised };
  if (normalised >= 25)
    return { label: "Approaching", colorClass: "text-orange-500", hex: "#FF9800", normalised };
  return { label: "Hold", colorClass: "text-red-500", hex: "#F44336", normalised };
}
