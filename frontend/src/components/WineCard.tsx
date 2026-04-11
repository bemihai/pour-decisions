"use client";

/**
 * WineCard component — redesigned (Phase 4C).
 *
 * Collapsed state: left wine-type colour stripe + bottle illustration + structured
 *   producer / wine name / origin columns + vintage / quantity / rating on the right.
 *
 * Expanded state: four distinct visual zones —
 *   1. Details — icon-annotated key-value list + producer description.
 *   2. Description — tinted background, AI-generated wine description.
 *   3. Drinking Readiness — visual timeline (DrinkingIndex).
 *   4. Stats — purchase price, community rating, community cellar bar.
 *
 * Wrapped in React.memo so parent re-renders triggered by visibleCount changes
 * do not cascade to every visible card.
 */

import React, { useId, useState } from "react";
import Link from "next/link";
import {
  ArrowUpRight,
  Calendar,
  ChevronDown,
  Clock,
  Loader2,
  MapPin,
  Sparkles,
  User,
  Wine,
} from "lucide-react";

import type { InventoryItem } from "@/lib/types";
import { generateWineDescription } from "@/lib/api";
import { cn, formatCurrency, ratingColor } from "@/lib/utils";
import { getWineBottleIllustration, getWineTypeColors } from "@/lib/design-tokens";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import DrinkingIndex from "@/components/DrinkingIndex";
import Rating from "@/components/Rating";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DRINK_SOURCE_LABEL: Record<string, string> = {
  cellar_tracker: "CT",
  manual: "Manual",
  llm: "AI est.",
  heuristic: "Estimated",
};

// ---------------------------------------------------------------------------
// Small helpers
// ---------------------------------------------------------------------------

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

// ---------------------------------------------------------------------------
// DetailRow — icon + text, used in the expanded Details zone
// ---------------------------------------------------------------------------

interface DetailRowIconProps {
  icon: React.ComponentType<{ className?: string }>;
  label: string | null | undefined;
  extra?: string | null;
}

function DetailRowIcon({ icon: Icon, label, extra }: DetailRowIconProps) {
  if (!label) return null;
  return (
    <div className="flex items-start gap-2 text-sm">
      <Icon className="mt-0.5 size-3.5 shrink-0 text-muted-foreground" aria-hidden="true" />
      <span className="text-foreground">
        {label}
        {extra && (
          <span className="ml-1.5 type-caption text-muted-foreground">({extra})</span>
        )}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface WineCardProps {
  wine: InventoryItem;
  /** All drink_index values in the current inventory, for p5/p95 normalisation. */
  allDrinkIndices: number[];
}

// ---------------------------------------------------------------------------
// WineCard (inner, un-memoised)
// ---------------------------------------------------------------------------

function WineCard({ wine, allDrinkIndices }: WineCardProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const [localDescription, setLocalDescription] = useState<string | null>(wine.description);
  const [isGenerating, setIsGenerating] = useState(false);
  const [generateError, setGenerateError] = useState<string | null>(null);

  const expandedId = useId();

  const colors = getWineTypeColors(wine.wine_type);
  const bottleIllustration = getWineBottleIllustration(wine.wine_type);
  const drinkSourceLabel = drinkWindowSourceLabel(wine.drink_window_source);

  const communityPctHeld =
    wine.q_purchased > 0 ? Math.min((wine.q_quantity / wine.q_purchased) * 100, 100) : null;
  const communityPctConsumed =
    wine.q_purchased > 0 ? Math.min((wine.q_consumed / wine.q_purchased) * 100, 100) : null;

  const origin = [wine.region_name, wine.country].filter(Boolean).join(" · ");

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

  return (
    <Card className="overflow-hidden p-0 gap-0 transition-all duration-200 hover:shadow-md hover:-translate-y-px">
      <div className="flex">
        {/* ---------------------------------------------------------------- */}
        {/* Wine-type left-border colour stripe                              */}
        {/* ---------------------------------------------------------------- */}
        <div
          className="w-1 shrink-0 self-stretch rounded-l-xl"
          style={{ backgroundColor: colors.hex }}
          aria-hidden="true"
        />

        <div className="min-w-0 flex-1">
          {/* -------------------------------------------------------------- */}
          {/* Collapsed header — always visible, acts as the expand toggle   */}
          {/* -------------------------------------------------------------- */}
          <button
            onClick={() => setIsExpanded((v) => !v)}
            className="flex w-full items-center gap-3 px-4 py-3 text-left"
            aria-expanded={isExpanded}
            aria-controls={expandedId}
          >
            {/* Bottle illustration */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={bottleIllustration}
              alt=""
              aria-hidden="true"
              className="hidden sm:block h-9 w-auto shrink-0 select-none"
            />

            {/* Center: producer · wine name · origin */}
            <div className="min-w-0 flex-1">
              {wine.producer_name && (
                <p className="type-label text-muted-foreground truncate leading-tight">
                  {wine.producer_name}
                </p>
              )}
              <p className="font-semibold leading-snug truncate text-foreground">
                {wine.wine_name}
              </p>
              {origin && (
                <p className="type-caption text-muted-foreground truncate leading-tight">
                  {origin}
                </p>
              )}

            </div>

            {/* Right: vintage · quantity · rating · chevron */}
            <div className="ml-2 flex shrink-0 items-center gap-2">
              {wine.vintage != null && (
                <span className="text-sm font-semibold tabular-nums text-foreground">
                  {wine.vintage}
                </span>
              )}
              <Badge variant="secondary" className="shrink-0 text-xs">
                {wine.quantity} btl
              </Badge>
              {wine.personal_rating != null && (
                <Rating rating={wine.personal_rating} variant="compact" />
              )}
              <ChevronDown
                className={cn(
                  "size-4 shrink-0 text-muted-foreground transition-transform duration-200",
                  isExpanded && "rotate-180",
                )}
                aria-hidden="true"
              />
            </div>
          </button>

          {/* -------------------------------------------------------------- */}
          {/* Expanded body — CSS grid height animation                       */}
          {/* -------------------------------------------------------------- */}
          <div
            id={expandedId}
            className={cn(
              "grid transition-all duration-300 ease-in-out",
              isExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]",
            )}
          >
            <div className="overflow-hidden">
              <Separator />

              {/* 3-column expanded layout — fluid, no dividers between columns
                  Col 1: Wine details (icon rows)
                  Col 2: Drink status + stats (stacked)
                  Col 3: About the Producer + About this Wine (stacked, compact)
                  On mobile all columns stack vertically. */}
              <div className="grid grid-cols-1 gap-0 md:grid-cols-3">

                {/* -------------------------------------------------------- */}
                {/* Col 1: Wine Details                                       */}
                {/* -------------------------------------------------------- */}
                <div className="flex flex-col gap-3 px-4 py-4 md:border-b-0">
                  <p className="type-label font-semibold uppercase tracking-wide text-muted-foreground">
                    Wine Details
                  </p>
                  <div className="flex flex-col gap-1.5">
                    <DetailRowIcon icon={User} label={wine.producer_name} />
                    <DetailRowIcon icon={Wine} label={wine.varietal} />
                    <DetailRowIcon icon={MapPin} label={origin || null} />
                    <DetailRowIcon
                      icon={MapPin}
                      label={
                        wine.location
                          ? `${wine.location}${wine.bin ? ` · Bin ${wine.bin}` : ""}`
                          : null
                      }
                    />
                    <DetailRowIcon
                      icon={Calendar}
                      label={formatDrinkWindow(wine)}
                      extra={drinkSourceLabel}
                    />
                    {wine.last_tasted_date && (
                      <DetailRowIcon
                        icon={Clock}
                        label={`Last tasted: ${wine.last_tasted_date}`}
                      />
                    )}
                    {wine.bottle_note && (
                      <DetailRowIcon icon={Wine} label={wine.bottle_note} />
                    )}
                  </div>
                </div>

                {/* -------------------------------------------------------- */}
                {/* Col 2: Drink status + stats                               */}
                {/* -------------------------------------------------------- */}
                <div className="flex flex-col gap-4 px-4 py-4">

                  {/* Drink status */}
                  {wine.drink_index != null && (
                    <div className="flex flex-col gap-1.5">
                      <p className="type-label font-semibold uppercase tracking-wide text-muted-foreground">
                        Drinking Readiness
                      </p>
                      <DrinkingIndex
                        drinkIndex={wine.drink_index}
                        allIndices={allDrinkIndices}
                      />
                    </div>
                  )}

                  {/* Purchase price */}
                  <div>
                    <p className="type-caption text-muted-foreground">Purchase Price</p>
                    <p className="text-sm font-medium">
                      {wine.purchase_price != null
                        ? formatCurrency(wine.purchase_price, wine.currency ?? "RON")
                        : "—"}
                    </p>
                  </div>

                  {/* Community rating */}
                  <div>
                    <p className="type-caption text-muted-foreground">Community Rating</p>
                    <p className={cn("text-sm font-medium", ratingColor(wine.community_rating))}>
                      {wine.community_rating != null
                        ? `${Math.round(wine.community_rating)}/100`
                        : "—"}
                    </p>
                    {wine.like_percentage != null && wine.like_votes != null && (
                      <p className="type-caption text-muted-foreground">
                        {wine.like_percentage.toFixed(0)}% liked ({wine.like_votes} votes)
                      </p>
                    )}
                  </div>

                  {/* Community cellar */}
                  {communityPctHeld != null && communityPctConsumed != null && (
                    <div>
                      <p className="type-caption text-muted-foreground mb-1.5">
                        Community Cellar
                      </p>
                      <div className="flex items-center gap-2">
                        <div className="relative h-3 w-full max-w-[96px] overflow-hidden rounded-full bg-muted">
                          <div
                            className="absolute inset-y-0 left-0 bg-brand-burgundy"
                            style={{ width: `${communityPctHeld}%` }}
                          />
                          <div
                            className="absolute inset-y-0 bg-brand-gold"
                            style={{
                              left: `${communityPctHeld}%`,
                              width: `${communityPctConsumed}%`,
                            }}
                          />
                        </div>
                        <span className="type-caption text-muted-foreground">
                          {communityPctHeld.toFixed(0)}% held
                        </span>
                      </div>
                    </div>
                  )}

                </div>

                {/* -------------------------------------------------------- */}
                {/* Col 3: About the Producer + About this Wine               */}
                {/* -------------------------------------------------------- */}
                <div className="flex flex-col gap-4 bg-secondary/30 px-4 py-4">
                  {/* Producer description */}
                  <div className="flex flex-col gap-1.5">
                    <p className="type-label font-semibold uppercase tracking-wide text-muted-foreground">
                      About the Producer
                    </p>
                    {wine.producer_description ? (
                      <p className="text-xs leading-relaxed italic text-muted-foreground">
                        {wine.producer_description}
                      </p>
                    ) : (
                      <p className="text-xs text-muted-foreground/60">No producer description available.</p>
                    )}
                  </div>

                  <Separator />

                  {/* Wine description */}
                  <div className="flex flex-col gap-1.5">
                    <p className="type-label font-semibold uppercase tracking-wide text-muted-foreground">
                      About this Wine
                    </p>
                    {localDescription ? (
                      <p className="text-xs leading-relaxed italic text-muted-foreground">
                        {localDescription}
                      </p>
                    ) : (
                      <div className="flex flex-col gap-2">
                        <p className="text-xs text-muted-foreground">No description yet.</p>
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
                            <Loader2
                              className="mr-1.5 size-3.5 animate-spin"
                              aria-hidden="true"
                            />
                          ) : (
                            <Sparkles className="mr-1.5 size-3.5" aria-hidden="true" />
                          )}
                          {isGenerating ? "Generating…" : "Generate Description"}
                        </Button>
                        {generateError && (
                          <p className="type-caption text-destructive">{generateError}</p>
                        )}
                      </div>
                    )}
                  </div>
                </div>

              </div>

              {/* View full details link */}
              <div className="flex justify-end border-t border-border px-4 py-2">
                <Link
                  href={`/cellar/${wine.wine_id}`}
                  onClick={(e) => e.stopPropagation()}
                  className="flex items-center gap-1 type-caption text-brand-burgundy hover:underline"
                >
                  View full details
                  <ArrowUpRight className="size-3.5" aria-hidden="true" />
                </Link>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Card>
  );
}

export default React.memo(WineCard, (prev, next) =>
  prev.wine.wine_id === next.wine.wine_id &&
  prev.allDrinkIndices === next.allDrinkIndices,
);
