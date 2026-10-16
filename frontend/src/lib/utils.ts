import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

import type { DrinkStatus } from "@/lib/types";

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
  if (rating <= 70) return "Avoid";
  if (rating < 80) return "Below Average";
  if (rating < 86) return "Good";
  if (rating < 90) return "Very Good";
  if (rating < 94) return "Excellent";
  if (rating < 98) return "Outstanding";
  return "Extraordinary";
}

/** Format a 0-100 rating as "87/100" or "N/A". */
export function formatRating(rating: number | null | undefined): string {
  if (rating == null) return "N/A";
  return `${Math.round(rating)}/100`;
}

/** Return a Tailwind text-color class based on rating tier. */
export function ratingColor(rating: number | null | undefined): string {
  if (rating == null) return "text-muted-foreground";
  if (rating >= 98) return "text-emerald-600";
  if (rating >= 94) return "text-green-600";
  if (rating >= 90) return "text-lime-600";
  if (rating >= 86) return "text-yellow-600";
  if (rating >= 80) return "text-amber-600";
  if (rating >= 71) return "text-orange-500";
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
}

/** Map the backend-owned drinking phase to presentation-only tokens. */
export function getDrinkingStatus(drinkStatus: DrinkStatus): DrinkingStatus {
  if (drinkStatus === "ready")
    return { label: "Ready to Drink", colorClass: "text-green-600", hex: "#4CAF50" };
  if (drinkStatus === "past_peak")
    return { label: "Past Peak", colorClass: "text-orange-500", hex: "#FF9800" };
  if (drinkStatus === "hold")
    return { label: "Hold", colorClass: "text-red-500", hex: "#F44336" };
  return { label: "Unknown", colorClass: "text-muted-foreground", hex: "#6B7280" };
}
