"use client";

/**
 * TasteHistory component — Phase 4E redesign.
 *
 * Consumed wines list with:
 *  - Redesigned ConsumedWineCard: wine-type left-border stripe, bottle illustration,
 *    TastingNote component, exceptional (94+) gold ring accent.
 *  - Timeline layout option: vertical date markers on the left, cards on the right.
 *  - FilterPanel integration and TanStack Query caching.
 */

import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Calendar, CalendarDays, LayoutList, Loader2, Wine } from "lucide-react";
import Image from "next/image";

import type {
  ConsumedFilterOptions,
  ConsumedWineItem,
  ConsumedWinesFilters,
  FilterOptions,
  InventoryFilters,
} from "@/lib/types";
import { getConsumedWines } from "@/lib/api";
import { cn } from "@/lib/utils";
import { getWineBottleIllustration, getWineTypeColors } from "@/lib/design-tokens";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import EmptyState from "@/components/EmptyState";
import FilterPanel, { type SortOption } from "@/components/FilterPanel";
import Rating from "@/components/Rating";
import TastingNote from "@/components/TastingNote";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const CONSUMED_SORT_OPTIONS: SortOption[] = [
  { label: "Consumed Date (Recent)", value: "consumed_date_desc" },
  { label: "Consumed Date (Oldest)", value: "consumed_date_asc" },
  { label: "Rating (High to Low)",   value: "rating_desc" },
  { label: "Rating (Low to High)",   value: "rating_asc" },
  { label: "Producer",               value: "producer" },
  { label: "Wine Name",              value: "wine_name" },
];

const PAGE_SIZE = 20;

// ---------------------------------------------------------------------------
// ConsumedWineCard — private sub-component (4E.1 redesign)
// ---------------------------------------------------------------------------

interface ConsumedWineCardProps {
  wine: ConsumedWineItem;
}

function ConsumedWineCard({ wine }: ConsumedWineCardProps) {
  const typeColors = getWineTypeColors(wine.wine_type);
  const bottleIllustration = getWineBottleIllustration(wine.wine_type);
  const isExceptional = wine.personal_rating != null && wine.personal_rating >= 94;

  return (
    <Card
      className={cn(
        "overflow-hidden p-0 gap-0 transition-all duration-200 hover:shadow-md hover:-translate-y-px",
        // 4E.3: Gold ring accent for exceptional wines (94+)
        isExceptional && "ring-2 ring-brand-gold/60",
      )}
    >
      <div className="flex">
        {/* Wine-type left-border colour stripe */}
        <div
          className="w-1 shrink-0 self-stretch rounded-l-xl"
          style={{ backgroundColor: typeColors.hex }}
          aria-hidden="true"
        />

        <CardContent className="flex-1 p-4">
          {/* Top row: bottle + info + rating */}
          <div className="flex items-start gap-3">
            {/* Bottle illustration */}
            <Image
              src={bottleIllustration}
              alt=""
              aria-hidden="true"
              width={28}
              height={72}
              className="hidden sm:block shrink-0 select-none object-contain"
            />

            {/* Center: wine identity */}
            <div className="flex-1 min-w-0">
              <div className="flex flex-wrap items-center gap-2 mb-1">
                {wine.wine_type && (
                  <Badge className={cn("text-xs font-medium shrink-0", typeColors.tailwind)}>
                    {wine.wine_type}
                  </Badge>
                )}
                {isExceptional && (
                  <Badge className="text-xs font-medium bg-brand-gold/20 text-brand-gold border border-brand-gold/40 shrink-0">
                    Outstanding
                  </Badge>
                )}
              </div>

              {wine.producer_name && (
                <p className="type-label text-muted-foreground leading-tight truncate">
                  {wine.producer_name}
                </p>
              )}
              <p className="font-semibold leading-snug text-foreground">
                {wine.wine_name}
                {wine.vintage != null && (
                  <span className="ml-1.5 text-sm font-normal text-muted-foreground">
                    {wine.vintage}
                  </span>
                )}
              </p>

              <div className="flex flex-wrap gap-3 type-caption text-muted-foreground mt-1">
                {wine.country && (
                  <span className="flex items-center gap-1">
                    <Wine className="size-3.5" aria-hidden="true" />
                    {wine.region_name
                      ? `${wine.region_name}, ${wine.country}`
                      : wine.country}
                  </span>
                )}
                {wine.varietal && <span>{wine.varietal}</span>}
                {wine.consumed_date && (
                  <span className="flex items-center gap-1">
                    <Calendar className="size-3.5" aria-hidden="true" />
                    {wine.consumed_date}
                  </span>
                )}
              </div>
            </div>

            {/* Right: rating */}
            <div className="shrink-0 text-right">
              <Rating rating={wine.personal_rating} variant="full" />
            </div>
          </div>

          {/* 4E.2: TastingNote component for personal tasting notes */}
          {wine.tasting_notes && (
            <TastingNote notes={wine.tasting_notes} className="mt-3" />
          )}
        </CardContent>
      </div>
    </Card>
  );
}

// ---------------------------------------------------------------------------
// 4E.4: Timeline item — date marker on left, card on right
// ---------------------------------------------------------------------------

function TimelineItem({
  wine,
  isLast,
}: {
  wine: ConsumedWineItem;
  isLast: boolean;
}) {
  const date = wine.consumed_date ? new Date(wine.consumed_date) : null;
  const monthLabel = date
    ? date.toLocaleDateString("en-US", { month: "short" })
    : "?";
  const yearLabel = date ? String(date.getFullYear()) : "";

  return (
    <div className="flex gap-4">
      {/* Left: date bubble + connecting line */}
      <div className="flex flex-col items-center shrink-0 w-14">
        <div className="flex flex-col items-center justify-center h-12 w-12 rounded-full bg-brand-gold/15 border border-brand-gold/30 text-center">
          <span className="text-[10px] font-semibold uppercase text-brand-gold leading-none">
            {monthLabel}
          </span>
          <span className="text-xs font-bold text-brand-gold leading-tight">{yearLabel}</span>
        </div>
        {!isLast && (
          <div className="w-px flex-1 bg-border mt-2 mb-0 min-h-[16px]" aria-hidden="true" />
        )}
      </div>

      {/* Right: card */}
      <div className="flex-1 pb-4">
        <ConsumedWineCard wine={wine} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TasteHistoryProps {
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
  const [viewMode, setViewMode] = useState<"cards" | "timeline">("cards");

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

  const { data, isLoading, isError } = useQuery({
    queryKey: ["consumed-wines", filters],
    queryFn: () => getConsumedWines(filters),
    staleTime: 60_000,
  });

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

  const items = data?.items ?? [];
  const total = data?.total ?? 0;
  const visibleItems = items.slice(0, visibleCount);
  const hasMore = items.length > visibleCount;

  return (
    <div className="flex flex-col gap-4">
      {/* Filters + view toggle row */}
      <div className="flex items-start gap-2">
        <div className="flex-1">
          <FilterPanel
            options={liveFilterOptions}
            onChange={handleFilterChange}
            showLocation={false}
            sortOptions={CONSUMED_SORT_OPTIONS}
            defaultSort="consumed_date_desc"
          />
        </div>
        {/* 4E.4: View toggle */}
        <div className="flex shrink-0 items-center rounded-lg border border-border bg-muted/30 p-1 gap-1 mt-0.5">
          <button
            onClick={() => setViewMode("cards")}
            aria-label="Card view"
            aria-pressed={viewMode === "cards"}
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-md transition-colors",
              viewMode === "cards"
                ? "bg-background text-brand-burgundy shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <LayoutList className="size-4" />
          </button>
          <button
            onClick={() => setViewMode("timeline")}
            aria-label="Timeline view"
            aria-pressed={viewMode === "timeline"}
            className={cn(
              "flex h-8 w-8 items-center justify-center rounded-md transition-colors",
              viewMode === "timeline"
                ? "bg-background text-brand-burgundy shadow-sm"
                : "text-muted-foreground hover:text-foreground",
            )}
          >
            <CalendarDays className="size-4" />
          </button>
        </div>
      </div>

      {/* Results count */}
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
          <Loader2 className="size-6 animate-spin text-brand-burgundy" />
        </div>
      )}

      {/* Error state */}
      {isError && (
        <p className="type-body text-destructive">
          Failed to load consumed wines. Please try again.
        </p>
      )}

      {/* Wine list */}
      {!isLoading && !isError && (
        <>
          {visibleItems.length === 0 ? (
            <EmptyState
              icon={Wine}
              title="No tasting history yet"
              description="Wines you mark as consumed will appear here with your tasting notes."
            />
          ) : viewMode === "timeline" ? (
            /* Timeline layout */
            <div className="flex flex-col">
              {visibleItems.map((wine, i) => (
                <TimelineItem
                  key={wine.wine_id ?? i}
                  wine={wine}
                  isLast={i === visibleItems.length - 1}
                />
              ))}
            </div>
          ) : (
            /* Card layout */
            <div className="flex flex-col gap-3">
              {visibleItems.map((wine, i) => (
                <ConsumedWineCard key={wine.wine_id ?? i} wine={wine} />
              ))}
            </div>
          )}
        </>
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

