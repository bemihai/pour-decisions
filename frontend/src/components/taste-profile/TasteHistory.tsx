"use client";

/**
 * TasteHistory component — Step 3.4.
 *
 * Consumed wines list with FilterPanel integration and TanStack Query caching.
 * Replaces show_consumed_wines_inventory() from src/ui/helper/taste_profile_stats.py.
 *
 * Data flow:
 *   FilterPanel -> filters state -> useQuery(["consumed-wines", filters]) -> card list
 *
 * Reuses FilterPanel from Phase 2 (showLocation=false, custom sortOptions).
 */

import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Calendar, Loader2, Star, Wine } from "lucide-react";

import type {
  ConsumedFilterOptions,
  ConsumedWineItem,
  ConsumedWinesFilters,
  FilterOptions,
  InventoryFilters,
} from "@/lib/types";
import { getConsumedWines } from "@/lib/api";
import { cn, formatRating, formatVintage, getRatingLabel, ratingColor } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import FilterPanel, { type SortOption } from "@/components/FilterPanel";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Sort options matching _CONSUMED_SQL_SORT in src/database/repository/stats.py. */
const CONSUMED_SORT_OPTIONS: SortOption[] = [
  { label: "Consumed Date (Recent)", value: "consumed_date_desc" },
  { label: "Consumed Date (Oldest)", value: "consumed_date_asc" },
  { label: "Rating (High to Low)",  value: "rating_desc" },
  { label: "Rating (Low to High)",  value: "rating_asc" },
  { label: "Producer",              value: "producer" },
  { label: "Wine Name",             value: "wine_name" },
];

const WINE_TYPE_COLORS: Record<string, string> = {
  Red:       "bg-red-900 text-white",
  White:     "bg-yellow-100 text-yellow-900",
  "Rosé":    "bg-pink-200 text-pink-900",
  Rose:      "bg-pink-200 text-pink-900",
  Sparkling: "bg-yellow-300 text-yellow-900",
  Dessert:   "bg-amber-200 text-amber-900",
  Fortified: "bg-amber-900 text-white",
};

const PAGE_SIZE = 20;

// ---------------------------------------------------------------------------
// ConsumedWineCard — private sub-component
// ---------------------------------------------------------------------------

interface ConsumedWineCardProps {
  wine: ConsumedWineItem;
}

function ConsumedWineCard({ wine }: ConsumedWineCardProps) {
  const typeColorClass =
    WINE_TYPE_COLORS[wine.wine_type ?? ""] ?? "bg-muted text-muted-foreground";

  return (
    <Card className="overflow-hidden transition-shadow hover:shadow-md">
      <CardContent className="p-4">
        <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
          {/* Left: wine info */}
          <div className="flex flex-col gap-1">
            <div className="flex flex-wrap items-center gap-2">
              {wine.wine_type && (
                <Badge className={cn("text-xs font-medium", typeColorClass)}>
                  {wine.wine_type}
                </Badge>
              )}
              <span className="font-semibold">
                {wine.producer_name
                  ? `${wine.producer_name}, ${wine.wine_name} (${formatVintage(wine.vintage)})`
                  : `${wine.wine_name} (${formatVintage(wine.vintage)})`}
              </span>
            </div>

            <div className="flex flex-wrap gap-3 text-sm text-muted-foreground">
              {wine.country && (
                <span className="flex items-center gap-1">
                  <Wine className="size-3.5" />
                  {wine.region_name ? `${wine.region_name}, ${wine.country}` : wine.country}
                </span>
              )}
              {wine.varietal && <span>{wine.varietal}</span>}
              {wine.consumed_date && (
                <span className="flex items-center gap-1">
                  <Calendar className="size-3.5" />
                  {wine.consumed_date}
                </span>
              )}
            </div>
          </div>

          {/* Right: rating */}
          <div className="shrink-0 text-right">
            {wine.personal_rating != null ? (
              <div className="flex flex-col items-end gap-0.5">
                <span className={cn("text-lg font-bold", ratingColor(wine.personal_rating))}>
                  {formatRating(wine.personal_rating)}
                </span>
                <span className="flex items-center gap-1 text-xs text-muted-foreground">
                  <Star className="size-3 fill-amber-400 text-amber-400" />
                  {getRatingLabel(wine.personal_rating)}
                </span>
              </div>
            ) : (
              <span className="text-sm text-muted-foreground">Not rated</span>
            )}
          </div>
        </div>

        {/* Tasting notes */}
        {wine.tasting_notes && (
          <p className="mt-2 border-t pt-2 text-sm italic text-muted-foreground">
            {wine.tasting_notes}
          </p>
        )}
      </CardContent>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TasteHistoryProps {
  /** Initial filter options from GET /api/taste-profile/consumed. */
  initialFilterOptions: ConsumedFilterOptions;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TasteHistory({ initialFilterOptions }: TasteHistoryProps) {
  const [filters, setFilters] = useState<ConsumedWinesFilters>({
    sort_by: "consumed_date_desc",
  });
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  // ---------------------------------------------------------------------------
  // Adapt ConsumedFilterOptions to the FilterPanel's FilterOptions shape.
  // ---------------------------------------------------------------------------
  const adaptedOptions: FilterOptions = useMemo(
    () => ({
      wine_types: initialFilterOptions.wine_types,
      countries: initialFilterOptions.countries,
      producers: initialFilterOptions.producers,
      locations: [],
      min_vintage: initialFilterOptions.min_vintage,
      max_vintage: initialFilterOptions.max_vintage,
    }),
    [initialFilterOptions],
  );

  // ---------------------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------------------
  const { data, isLoading, isError } = useQuery({
    queryKey: ["consumed-wines", filters],
    queryFn: () => getConsumedWines(filters),
    staleTime: 60_000,
  });

  // Prefer live filter_options from the response so dropdowns stay accurate.
  const liveFilterOptions: FilterOptions = useMemo(() => {
    if (!data?.filter_options) return adaptedOptions;
    const fo = data.filter_options;
    return {
      wine_types: fo.wine_types,
      countries: fo.countries,
      producers: fo.producers,
      locations: [],
      min_vintage: fo.min_vintage,
      max_vintage: fo.max_vintage,
    };
  }, [data, adaptedOptions]);

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  // Map InventoryFilters (FilterPanel output) to ConsumedWinesFilters.
  const handleFilterChange = useCallback((f: InventoryFilters) => {
    setVisibleCount(PAGE_SIZE);
    setFilters({
      wine_type: f.wine_type,
      country: f.country,
      producer: f.producer,
      rating_filter: f.rating_filter,
      search: f.search,
      sort_by: f.sort_by,
    });
  }, []);

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------
  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const visibleItems = items.slice(0, visibleCount);
  const hasMore = items.length > visibleCount;

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col gap-4">
      <FilterPanel
        options={liveFilterOptions}
        onChange={handleFilterChange}
        showLocation={false}
        sortOptions={CONSUMED_SORT_OPTIONS}
        defaultSort="consumed_date_desc"
      />

      {/* Results header */}
      {!isLoading && !isError && (
        <p className="text-sm text-muted-foreground">
          {total > 0
            ? `Showing ${visibleItems.length} of ${total} consumed wine${total !== 1 ? "s" : ""}`
            : "No consumed wines match the selected filters."}
        </p>
      )}

      {/* Loading state */}
      {isLoading && (
        <div className="flex items-center justify-center py-12">
          <Loader2 className="size-6 animate-spin text-purple-600" />
        </div>
      )}

      {/* Error state */}
      {isError && (
        <p className="text-sm text-destructive">
          Failed to load consumed wines. Please try again.
        </p>
      )}

      {/* Wine cards */}
      {!isLoading && !isError && (
        <div className="flex flex-col gap-3">
          {visibleItems.map((wine, idx) => (
            <ConsumedWineCard key={`${wine.wine_id}-${idx}`} wine={wine} />
          ))}
        </div>
      )}

      {/* Load more */}
      {hasMore && (
        <Button
          variant="outline"
          onClick={() => setVisibleCount((n) => n + PAGE_SIZE)}
          className="self-center"
        >
          Load more ({items.length - visibleCount} remaining)
        </Button>
      )}
    </div>
  );
}

