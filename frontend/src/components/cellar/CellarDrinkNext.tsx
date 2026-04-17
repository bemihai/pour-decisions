"use client";

/**
 * CellarDrinkNext component — displays wines ready to drink now.
 *
 * Shows wines grouped by type (Red, White, Rosé, Sparkling, etc.)
 * sorted by drink_index descending (highest = drink soonest).
 * Each wine type gets its own table with key information.
 * Pagination: shows 5 wines per type by default with "Show More" button.
 */

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Wine, AlertCircle, MapPin, Star, ChevronDown } from "lucide-react";
import Link from "next/link";

import type { DrinkNextResponse } from "@/lib/types";
import { getDrinkNext } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import Section from "@/components/Section";
import EmptyState from "@/components/EmptyState";
import DrinkingIndex from "@/components/DrinkingIndex";

// Wine type colors for section headers
const WINE_TYPE_COLORS: Record<string, string> = {
  Red: "text-red-600",
  White: "text-amber-600",
  "Rosé": "text-pink-600",
  Sparkling: "text-blue-600",
  Dessert: "text-orange-600",
  Fortified: "text-purple-600",
};

const WINES_PER_PAGE = 5;

export default function CellarDrinkNext() {
  const [expandedTypes, setExpandedTypes] = useState<Set<string>>(new Set());

  const { data, isLoading, error } = useQuery<DrinkNextResponse>({
    queryKey: ["drink-next"],
    queryFn: () => getDrinkNext(50),
    staleTime: 60_000, // 1 minute
  });

  const toggleExpanded = (wineType: string) => {
    setExpandedTypes((prev) => {
      const next = new Set(prev);
      if (next.has(wineType)) {
        next.delete(wineType);
      } else {
        next.add(wineType);
      }
      return next;
    });
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        {[1, 2, 3].map((i) => (
          <div key={i} className="rounded-lg border border-border bg-card p-6">
            <div className="mb-4 h-6 w-32 animate-pulse rounded bg-muted" />
            <div className="space-y-3">
              {[1, 2, 3].map((j) => (
                <div key={j} className="h-16 animate-pulse rounded bg-muted" />
              ))}
            </div>
          </div>
        ))}
      </div>
    );
  }

  if (error) {
    return (
      <Section>
        <EmptyState
          icon={AlertCircle}
          title="Failed to load recommendations"
          description="Could not fetch drink-next data from the server."
        />
      </Section>
    );
  }

  if (!data || data.total_wines === 0) {
    return (
      <Section>
        <EmptyState
          icon={Wine}
          title="No wines ready to drink"
          description="Either all your wines need more aging, or drinking window data is missing."
        />
      </Section>
    );
  }

  // Sort wine types: Red, White, Rosé, Sparkling, then others alphabetically
  const preferredOrder = ["Red", "White", "Rosé", "Sparkling"];
  const sortedTypes = Object.keys(data.by_type).sort((a, b) => {
    const aIdx = preferredOrder.indexOf(a);
    const bIdx = preferredOrder.indexOf(b);
    if (aIdx !== -1 && bIdx !== -1) return aIdx - bIdx;
    if (aIdx !== -1) return -1;
    if (bIdx !== -1) return 1;
    return a.localeCompare(b);
  });

  return (
    <div className="space-y-6">
      {/* Summary header */}
      <div className="rounded-lg border border-border bg-muted/30 px-4 py-3">
        <p className="text-sm text-muted-foreground">
          Found <strong className="text-foreground">{data.total_wines}</strong> wine
          {data.total_wines !== 1 ? "s" : ""} ready to drink across{" "}
          <strong className="text-foreground">{sortedTypes.length}</strong> type
          {sortedTypes.length !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Wine type sections */}
      {sortedTypes.map((wineType) => {
        const allWines = data.by_type[wineType];
        const isExpanded = expandedTypes.has(wineType);
        const displayWines = isExpanded ? allWines : allWines.slice(0, WINES_PER_PAGE);
        const hasMore = allWines.length > WINES_PER_PAGE;
        const colorClass = WINE_TYPE_COLORS[wineType] || "text-foreground";

        return (
          <div key={wineType} className="rounded-lg border border-border bg-card">
            {/* Section header */}
            <div className="border-b border-border px-6 py-4">
              <h2 className={cn("text-lg font-semibold", colorClass)}>
                {wineType}
                <Badge variant="secondary" className="ml-2 text-xs">
                  {allWines.length}
                </Badge>
              </h2>
            </div>

            {/* Table */}
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-border bg-muted/30 text-xs uppercase tracking-wide text-muted-foreground">
                    <th className="px-4 py-3 text-left font-medium">Wine</th>
                    <th className="px-4 py-3 text-left font-medium">Producer</th>
                    <th className="px-4 py-3 text-center font-medium">Vintage</th>
                    <th className="px-4 py-3 text-left font-medium">Region</th>
                    <th className="px-4 py-3 text-center font-medium">Qty</th>
                    <th className="px-4 py-3 text-center font-medium">Status</th>
                    <th className="px-4 py-3 text-center font-medium">Rating</th>
                    <th className="px-4 py-3 text-left font-medium">Location</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-border">
                  {displayWines.map((wine) => {

                    return (
                      <tr
                        key={wine.wine_id}
                        className="transition-colors hover:bg-muted/50"
                      >
                        {/* Wine name */}
                        <td className="px-4 py-3">
                          <Link
                            href={`/cellar/${wine.wine_id}`}
                            className="font-medium text-foreground hover:text-brand-burgundy hover:underline"
                          >
                            {wine.wine_name}
                          </Link>
                          {wine.varietal && (
                            <p className="text-xs text-muted-foreground">{wine.varietal}</p>
                          )}
                        </td>

                        {/* Producer */}
                        <td className="px-4 py-3 text-sm text-muted-foreground">
                          {wine.producer_name || "—"}
                        </td>

                        {/* Vintage */}
                        <td className="px-4 py-3 text-center text-sm tabular-nums">
                          {wine.vintage || "NV"}
                        </td>

                        {/* Region */}
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                            {wine.region_name || wine.country || "—"}
                          </div>
                        </td>

                        {/* Quantity */}
                        <td className="px-4 py-3 text-center">
                          <Badge variant="outline" className="tabular-nums">
                            {wine.quantity}
                          </Badge>
                        </td>

                        {/* Drinking status */}
                        <td className="px-4 py-3 text-center">
                          <DrinkingIndex drinkIndex={wine.drink_index} allIndices={[]} />
                        </td>

                        {/* Rating */}
                        <td className="px-4 py-3 text-center">
                          {wine.personal_rating != null ? (
                            <div className="flex items-center justify-center gap-1 text-sm">
                              <Star className="size-3.5 fill-amber-400 text-amber-400" />
                              <span className="tabular-nums">{wine.personal_rating}</span>
                            </div>
                          ) : wine.community_rating != null ? (
                            <span className="text-xs text-muted-foreground">
                              {wine.community_rating.toFixed(1)}
                            </span>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>

                        {/* Location */}
                        <td className="px-4 py-3">
                          {wine.location ? (
                            <div className="flex items-center gap-1.5 text-sm text-muted-foreground">
                              <MapPin className="size-3.5 shrink-0" />
                              <span className="truncate">{wine.location}</span>
                            </div>
                          ) : (
                            <span className="text-muted-foreground">—</span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>

            {/* Show More/Less button */}
            {hasMore && (
              <div className="border-t border-border px-6 py-3 text-center">
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => toggleExpanded(wineType)}
                  className="text-sm font-medium text-muted-foreground hover:text-foreground"
                >
                  {isExpanded ? (
                    <>
                      Show Less
                      <ChevronDown className="ml-1.5 size-4 rotate-180 transition-transform" />
                    </>
                  ) : (
                    <>
                      Show {allWines.length - WINES_PER_PAGE} More
                      <ChevronDown className="ml-1.5 size-4 transition-transform" />
                    </>
                  )}
                </Button>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

