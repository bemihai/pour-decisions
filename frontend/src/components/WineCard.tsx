"use client";

/**
 * WineCard component.
 *
 * Expandable card for a single InventoryItem. Shows the essential summary
 * in the collapsed state and full wine details when expanded.
 *
 * Replaces the st.expander + inline HTML table in show_cellar_inventory()
 * from src/ui/helper/cellar_stats.py.
 */

import { useState } from "react";
import { ChevronDown, Sparkles, Loader2 } from "lucide-react";

import type { InventoryItem } from "@/lib/types";
import { generateWineDescription } from "@/lib/api";
import { cn, formatCurrency, formatRating, ratingColor } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import DrinkingIndex from "@/components/DrinkingIndex";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Tailwind classes for each wine type badge. */
const WINE_TYPE_BADGE: Record<string, string> = {
  Red: "border-red-300 bg-red-50 text-red-700 dark:border-red-700 dark:bg-red-950 dark:text-red-300",
  White: "border-amber-300 bg-amber-50 text-amber-700 dark:border-amber-700 dark:bg-amber-950 dark:text-amber-300",
  "Rosé": "border-pink-300 bg-pink-50 text-pink-700 dark:border-pink-700 dark:bg-pink-950 dark:text-pink-300",
  Rose: "border-pink-300 bg-pink-50 text-pink-700 dark:border-pink-700 dark:bg-pink-950 dark:text-pink-300",
  Sparkling: "border-blue-300 bg-blue-50 text-blue-700 dark:border-blue-700 dark:bg-blue-950 dark:text-blue-300",
  Dessert: "border-orange-300 bg-orange-50 text-orange-700 dark:border-orange-700 dark:bg-orange-950 dark:text-orange-300",
  Fortified: "border-amber-400 bg-amber-100 text-amber-800 dark:border-amber-600 dark:bg-amber-900 dark:text-amber-200",
};
const WINE_TYPE_BADGE_DEFAULT =
  "border-purple-300 bg-purple-50 text-purple-700 dark:border-purple-700 dark:bg-purple-950 dark:text-purple-300";

/** Human-readable labels for drink_window_source values. */
const DRINK_SOURCE_LABEL: Record<string, string> = {
  cellar_tracker: "CT",
  manual: "Manual",
  llm: "AI (est.)",
  heuristic: "Estimated",
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

function wineTypeBadgeClass(wineType: string | null): string {
  return WINE_TYPE_BADGE[wineType ?? ""] ?? WINE_TYPE_BADGE_DEFAULT;
}

function formatDrinkWindow(wine: InventoryItem): string {
  if (!wine.drink_from_year && !wine.drink_to_year) return "—";
  const from = wine.drink_from_year ?? "Now";
  const to = wine.drink_to_year ?? "∞";
  return `${from}–${to}`;
}

function drinkWindowSourceLabel(source: string | null): string | null {
  if (!source) return null;
  return DRINK_SOURCE_LABEL[source] ?? null;
}

/** Inline detail row — label + value. */
function DetailRow({ label, value }: { label: string; value: string | number | null | undefined }) {
  if (value == null || value === "") return null;
  return (
    <div className="flex gap-2 text-sm">
      <span className="min-w-[90px] shrink-0 text-muted-foreground">{label}</span>
      <span className="font-medium">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component props
// ---------------------------------------------------------------------------

export interface WineCardProps {
  wine: InventoryItem;
  /** All drink_index values in the current inventory, used for p5/p95 normalisation. */
  allDrinkIndices: number[];
}

// ---------------------------------------------------------------------------
// WineCard
// ---------------------------------------------------------------------------

export default function WineCard({ wine, allDrinkIndices }: WineCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [localDescription, setLocalDescription] = useState<string | null>(wine.description);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  // The community cellar percentages.
  const communityPctHeld =
    wine.q_purchased > 0 ? Math.min((wine.q_quantity / wine.q_purchased) * 100, 100) : null;
  const communityPctConsumed =
    wine.q_purchased > 0 ? Math.min((wine.q_consumed / wine.q_purchased) * 100, 100) : null;

  async function handleGenerateDescription() {
    setIsGenerating(true);
    setGenerateError(null);
    try {
      const result = await generateWineDescription(wine.wine_id);
      if (result.success && result.description) {
        setLocalDescription(result.description);
      } else {
        setGenerateError("Generation returned no description.");
      }
    } catch {
      setGenerateError("Failed to generate description. Please try again.");
    } finally {
      setIsGenerating(false);
    }
  }

  const drinkSourceLabel = drinkWindowSourceLabel(wine.drink_window_source);

  return (
    <Card className="overflow-hidden transition-shadow hover:shadow-md">
      {/* ------------------------------------------------------------------ */}
      {/* Collapsed header — always visible, acts as the toggle trigger       */}
      {/* ------------------------------------------------------------------ */}
      <CardContent className="py-3 px-4">
        <button
          onClick={() => setIsExpanded((v) => !v)}
          className="flex w-full flex-wrap items-center gap-2 text-left sm:flex-nowrap sm:gap-3"
          aria-expanded={isExpanded}
        >
          {/* Wine type badge */}
          <Badge
            variant="outline"
            className={cn("shrink-0 text-xs font-semibold", wineTypeBadgeClass(wine.wine_type))}
          >
            {wine.wine_type ?? "Wine"}
          </Badge>

          {/* Producer · Wine Name · Vintage */}
          <div className="min-w-0 flex-1">
            <span className="block truncate font-semibold leading-snug">
              {wine.producer_name
                ? `${wine.producer_name} · ${wine.wine_name}`
                : wine.wine_name}
            </span>
            <span className="text-sm text-muted-foreground">
              {wine.vintage ?? "NV"}
            </span>
          </div>

          {/* Quantity */}
          <Badge variant="secondary" className="shrink-0 text-xs">
            {wine.quantity} btl
          </Badge>

          {/* Personal rating */}
          {wine.personal_rating != null && (
            <span className={cn("shrink-0 text-sm font-semibold tabular-nums", ratingColor(wine.personal_rating))}>
              {wine.personal_rating}/100
            </span>
          )}

          {/* Expand chevron */}
          <ChevronDown
            className={cn(
              "ml-auto size-4 shrink-0 text-muted-foreground transition-transform duration-200",
              isExpanded && "rotate-180",
            )}
          />
        </button>

        {/* ---------------------------------------------------------------- */}
        {/* Expandable body — CSS grid trick for smooth height animation      */}
        {/* ---------------------------------------------------------------- */}
        <div
          className={cn(
            "grid transition-all duration-300 ease-in-out",
            isExpanded ? "grid-rows-[1fr] pt-4" : "grid-rows-[0fr]",
          )}
        >
          <div className="overflow-hidden">
            <div className="flex flex-col gap-5">
              {/* ---------------------------------------------------------- */}
              {/* Row 1: Wine Details + Descriptions (2 columns)              */}
              {/* ---------------------------------------------------------- */}
              <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                {/* Left: wine metadata */}
                <div className="flex flex-col gap-1.5">
                  <p className="text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                    Wine Details
                  </p>
                  <DetailRow label="Producer" value={wine.producer_name} />
                  <DetailRow label="Wine" value={wine.wine_name} />
                  <DetailRow label="Varietal" value={wine.varietal} />
                  <DetailRow label="Country" value={wine.country} />
                  <DetailRow label="Region" value={wine.region_name} />
                  <DetailRow label="Location" value={wine.location} />
                  <DetailRow label="Bin" value={wine.bin} />
                  <DetailRow label="Bottle Note" value={wine.bottle_note} />
                  <DetailRow label="Last Tasted" value={wine.last_tasted_date} />
                  <div className="flex gap-2 text-sm">
                    <span className="min-w-[90px] shrink-0 text-muted-foreground">Drink Window</span>
                    <span className="font-medium">
                      {formatDrinkWindow(wine)}
                      {drinkSourceLabel && (
                        <span className="ml-1 text-xs text-muted-foreground">
                          ({drinkSourceLabel})
                        </span>
                      )}
                    </span>
                  </div>
                </div>

                {/* Right: descriptions */}
                <div className="flex flex-col gap-4">
                  {/* Producer description */}
                  <div>
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      About the Producer
                    </p>
                    {wine.producer_description ? (
                      <p className="text-sm italic text-muted-foreground leading-relaxed">
                        {wine.producer_description}
                      </p>
                    ) : (
                      <p className="text-sm text-muted-foreground">No producer description available.</p>
                    )}
                  </div>

                  {/* Wine description */}
                  <div>
                    <p className="mb-1 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
                      About this Wine
                    </p>
                    {localDescription ? (
                      <p className="text-sm italic text-muted-foreground leading-relaxed">
                        {localDescription}
                      </p>
                    ) : (
                      <div className="flex flex-col gap-2">
                        <p className="text-sm text-muted-foreground">No description yet.</p>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleGenerateDescription();
                          }}
                          disabled={isGenerating}
                          className="w-fit"
                        >
                          {isGenerating ? (
                            <Loader2 className="mr-1.5 size-3.5 animate-spin" />
                          ) : (
                            <Sparkles className="mr-1.5 size-3.5" />
                          )}
                          {isGenerating ? "Generating..." : "Generate Description"}
                        </Button>
                        {generateError && (
                          <p className="text-xs text-destructive">{generateError}</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* ---------------------------------------------------------- */}
              {/* Row 2: Drinking readiness (only when drink_index is known)  */}
              {/* ---------------------------------------------------------- */}
              {wine.drink_index != null && (
                <>
                  <Separator />
                  <DrinkingIndex drinkIndex={wine.drink_index} allIndices={allDrinkIndices} />
                </>
              )}

              {/* ---------------------------------------------------------- */}
              {/* Row 3: Stats strip — price · community rating · cellar bar  */}
              {/* ---------------------------------------------------------- */}
              <Separator />
              <div className="grid grid-cols-2 gap-4 text-sm md:grid-cols-3">
                {/* Purchase price */}
                <div>
                  <p className="text-xs text-muted-foreground">Purchase Price</p>
                  <p className="font-medium">
                    {wine.purchase_price != null
                      ? formatCurrency(wine.purchase_price, wine.currency ?? "RON")
                      : "—"}
                  </p>
                </div>

                {/* Community rating + likes */}
                <div>
                  <p className="text-xs text-muted-foreground">Community Rating</p>
                  <p className={cn("font-medium", ratingColor(wine.community_rating))}>
                    {wine.community_rating != null ? formatRating(wine.community_rating) : "—"}
                    {wine.like_percentage != null && wine.like_votes != null && (
                      <span className="ml-2 text-xs text-muted-foreground font-normal">
                        {wine.like_percentage.toFixed(0)}% liked ({wine.like_votes})
                      </span>
                    )}
                  </p>
                </div>

                {/* Community cellar bar */}
                {communityPctHeld != null && communityPctConsumed != null && (
                  <div className="col-span-2 md:col-span-1">
                    <p className="mb-1 text-xs text-muted-foreground">
                      Community Cellar (held / consumed)
                    </p>
                    <div className="flex items-center gap-2">
                      {/* Stacked mini progress bar */}
                      <div className="relative h-3.5 w-24 overflow-hidden rounded-full bg-muted">
                        <div
                          className="absolute inset-y-0 left-0 bg-purple-400"
                          style={{ width: `${communityPctHeld}%` }}
                        />
                        <div
                          className="absolute inset-y-0 bg-green-400"
                          style={{
                            left: `${communityPctHeld}%`,
                            width: `${communityPctConsumed}%`,
                          }}
                        />
                      </div>
                      <span className="text-xs text-muted-foreground">
                        {communityPctHeld.toFixed(0)}% / {communityPctConsumed.toFixed(0)}%
                      </span>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}

