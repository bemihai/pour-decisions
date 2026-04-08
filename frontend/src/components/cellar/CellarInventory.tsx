"use client";

/**
 * CellarInventory component.
 *
 * Filterable, sortable wine inventory with "load more" pagination.
 * Replaces show_cellar_inventory() (~400 lines) from src/ui/helper/cellar_stats.py.
 *
 * Data flow:
 *   FilterPanel -> filters state -> useQuery(["inventory", filters]) -> WineCard list
 *
 * filterOptions is passed from the parent (Server Component) as a pre-fetched
 * prop, then refreshed with the live filter_options included in every
 * InventoryResponse so dropdowns stay accurate after a CellarTracker sync.
 */

import { useCallback, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Loader2, Wine } from "lucide-react";

import type { FilterOptions, InventoryFilters } from "@/lib/types";
import { getInventory } from "@/lib/api";
import { Button } from "@/components/ui/button";
import FilterPanel from "@/components/FilterPanel";
import WineCard from "@/components/WineCard";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/** Number of WineCards shown initially and added on each "Load more" click. */
const PAGE_SIZE = 20;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CellarInventoryProps {
  /**
   * Available filter values pre-fetched by the parent Server Component via
   * GET /api/cellar/filters.  Refreshed automatically from the inventory
   * response on every successful fetch.
   */
  filterOptions: FilterOptions;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CellarInventory({ filterOptions }: CellarInventoryProps) {
  const [filters, setFilters] = useState<InventoryFilters>({});
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);

  // ---------------------------------------------------------------------------
  // Data fetching
  // ---------------------------------------------------------------------------

  const {
    data,
    isLoading,
    isFetching,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: ["inventory", filters],
    queryFn: () => getInventory(filters),
  });

  // Prefer live filter_options from the latest inventory response so the
  // dropdowns reflect any inventory changes (e.g. after a sync).
  const liveFilterOptions: FilterOptions = data?.filter_options ?? filterOptions;

  // ---------------------------------------------------------------------------
  // Derived state
  // ---------------------------------------------------------------------------

  // Collect all drink_index values from the FULL result set (not just the
  // visible slice) so DrinkingIndex normalisation is stable as the user
  // pages through the list.
  const allDrinkIndices = useMemo(
    () =>
      (data?.items ?? [])
        .map((item) => item.drink_index)
        .filter((i): i is number => i != null),
    [data],
  );

  const items = data?.items ?? [];
  const visibleItems = items.slice(0, visibleCount);
  const hasMore = items.length > visibleCount;
  const remaining = items.length - visibleCount;

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  // Stable callback — does not change between renders so FilterPanel's internal
  // refs never go stale.
  const handleFilterChange = useCallback((newFilters: InventoryFilters) => {
    setFilters(newFilters);
    setVisibleCount(PAGE_SIZE); // reset to first page on every new filter
  }, []);

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex flex-col gap-4">
      {/* ------------------------------------------------------------------ */}
      {/* Filter controls                                                      */}
      {/* ------------------------------------------------------------------ */}
      <FilterPanel options={liveFilterOptions} onChange={handleFilterChange} />

      {/* ------------------------------------------------------------------ */}
      {/* Results summary + background-fetch indicator                        */}
      {/* ------------------------------------------------------------------ */}
      <div className="flex min-h-[1.5rem] items-center justify-between">
        {data ? (
          <p className="text-sm text-muted-foreground">
            <span className="font-medium text-foreground">{data.total_wines}</span>{" "}
            {data.total_wines === 1 ? "wine" : "wines"},{" "}
            <span className="font-medium text-foreground">{data.total_bottles}</span>{" "}
            {data.total_bottles === 1 ? "bottle" : "bottles"}
          </p>
        ) : (
          <span /> // placeholder to keep the row height stable
        )}

        {/* Show a subtle spinner while TQ refetches in the background */}
        {isFetching && !isLoading && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" />
            Updating…
          </span>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Initial loading — skeleton cards                                     */}
      {/* ------------------------------------------------------------------ */}
      {isLoading && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Error state                                                          */}
      {/* ------------------------------------------------------------------ */}
      {isError && (
        <div className="flex flex-col items-center gap-3 rounded-xl border border-destructive/20 bg-destructive/5 p-10 text-center">
          <p className="text-sm font-medium text-destructive">
            {(error as Error)?.message ?? "Failed to load inventory."}
          </p>
          <Button variant="outline" size="sm" onClick={() => refetch()}>
            Try again
          </Button>
        </div>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Wine card list                                                       */}
      {/* ------------------------------------------------------------------ */}
      {!isLoading && !isError && (
        <>
          {visibleItems.length === 0 ? (
            <EmptyState />
          ) : (
            <div className="flex flex-col gap-3">
              {visibleItems.map((wine) => (
                <WineCard
                  key={wine.wine_id}
                  wine={wine}
                  allDrinkIndices={allDrinkIndices}
                />
              ))}
            </div>
          )}

          {/* "Load more" button — client-side slicing of the full API result */}
          {hasMore && (
            <div className="flex justify-center pt-2">
              <Button
                variant="outline"
                onClick={() => setVisibleCount((n) => n + PAGE_SIZE)}
              >
                Load {Math.min(PAGE_SIZE, remaining)} more
                <span className="ml-1 text-muted-foreground">
                  ({remaining} remaining)
                </span>
              </Button>
            </div>
          )}
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

/** Animated placeholder shown while the first fetch is in flight. */
function SkeletonCard() {
  return (
    <div className="h-14 animate-pulse rounded-xl bg-muted" />
  );
}

/** Shown when the filtered result set is empty. */
function EmptyState() {
  return (
    <div className="flex flex-col items-center gap-3 rounded-xl border border-dashed py-16 text-center">
      <Wine className="size-8 text-muted-foreground/50" />
      <div>
        <p className="text-sm font-medium">No wines found</p>
        <p className="mt-1 text-sm text-muted-foreground">
          Try adjusting the filters or clearing your search.
        </p>
      </div>
    </div>
  );
}

