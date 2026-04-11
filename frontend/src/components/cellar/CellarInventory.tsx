"use client";

/**
 * CellarInventory component.
 *
 * Filterable, sortable wine inventory with window-based virtual scrolling.
 * Replaces show_cellar_inventory() (~400 lines) from src/ui/helper/cellar_stats.py.
 *
 * Data flow:
 *   FilterPanel -> URL search params -> useQuery(["inventory", filters]) -> virtualizer -> WineCard list
 *
 * filterOptions is passed from the parent (Server Component) as a pre-fetched
 * prop, then refreshed with the live filter_options included in every
 * InventoryResponse so dropdowns stay accurate after a CellarTracker sync.
 *
 * Virtualization: all grouped wines are rendered via useWindowVirtualizer with
 * dynamic height measurement, so even large cellars (500+ unique wines) stay
 * performant without pagination.
 */

import { Suspense, useCallback, useMemo, useRef } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { useQuery } from "@tanstack/react-query";
import { useWindowVirtualizer } from "@tanstack/react-virtual";
import { Loader2, Wine } from "lucide-react";

import type { FilterOptions, InventoryFilters } from "@/lib/types";
import { getInventory } from "@/lib/api";
import { Button } from "@/components/ui/button";
import EmptyState from "@/components/EmptyState";
import FilterPanel from "@/components/FilterPanel";
import WineGroup, { groupWinesByIdentity } from "@/components/WineGroup";


// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Read a single search param, returning undefined when absent or empty. */
function spVal(params: URLSearchParams, key: string): string | undefined {
  const v = params.get(key);
  return v && v !== "" ? v : undefined;
}

/** Build InventoryFilters from URLSearchParams. */
function filtersFromParams(params: URLSearchParams): InventoryFilters {
  return {
    wine_type: spVal(params, "wine_type"),
    country: spVal(params, "country"),
    producer: spVal(params, "producer"),
    location: spVal(params, "location"),
    min_vintage: spVal(params, "min_vintage") ? Number(params.get("min_vintage")) : undefined,
    max_vintage: spVal(params, "max_vintage") ? Number(params.get("max_vintage")) : undefined,
    rating_filter: spVal(params, "rating_filter"),
    search: spVal(params, "search"),
    sort_by: spVal(params, "sort_by"),
  };
}

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
// CellarInventoryInner (uses useSearchParams — must be inside Suspense)
// ---------------------------------------------------------------------------

function CellarInventoryInner({ filterOptions }: CellarInventoryProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  // Derive initial filters from URL params; use them as the TanStack Query key
  // so navigating back with browser history instantly shows the cached result.
  const filters = useMemo(() => filtersFromParams(searchParams), [searchParams]);

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

  const items = data?.items ?? [];

  // Group ALL items by wine identity so the virtualizer sizes over groups.
  const groupedWines = useMemo(() => groupWinesByIdentity(items), [items]);

  // Collect all drink_index values from the full result set for p5/p95
  // normalisation in DrinkingIndex — stable as the user scrolls the list.
  const allDrinkIndices = useMemo(
    () =>
      items
        .map((item) => item.drink_index)
        .filter((i): i is number => i != null),
    [items],
  );

  // ---------------------------------------------------------------------------
  // Handlers
  // ---------------------------------------------------------------------------

  // Stable callback — updates URL search params (preserving unrelated params
  // such as ?tab=...) on every new filter application.
  const handleFilterChange = useCallback(
    (newFilters: InventoryFilters) => {
      const next = new URLSearchParams(searchParams.toString());
      // Clear all filter keys then re-apply the new values
      for (const key of [
        "wine_type", "country", "producer", "location",
        "min_vintage", "max_vintage", "rating_filter", "search", "sort_by",
      ]) {
        next.delete(key);
      }
      for (const [k, v] of Object.entries(newFilters)) {
        if (v !== undefined && v !== null && v !== "") {
          next.set(k, String(v));
        }
      }
      router.replace(`${pathname}?${next.toString()}`, { scroll: false });
    },
    [searchParams, router, pathname],
  );

  // ---------------------------------------------------------------------------
  // Virtualizer
  // ---------------------------------------------------------------------------

  // `listRef` marks the start of the virtual list so the virtualizer can
  // compute each item's distance from the window's scroll origin.
  const listRef = useRef<HTMLDivElement>(null);

  const virtualizer = useWindowVirtualizer({
    count: groupedWines.length,
    // Estimated collapsed WineCard height (colour stripe + one-line header).
    // measureElement provides exact sizes after first paint.
    estimateSize: () => 68,
    overscan: 5,
    // Distance from top of the document to the start of the list container.
    scrollMargin: listRef.current?.offsetTop ?? 0,
  });

  // ---------------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------------

  return (
    <div className="flex flex-col gap-4">
      {/* ------------------------------------------------------------------ */}
      {/* Filter controls                                                      */}
      {/* ------------------------------------------------------------------ */}
      <FilterPanel options={liveFilterOptions} onChange={handleFilterChange} defaultFilters={filters} />

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
          <span />
        )}

        {isFetching && !isLoading && (
          <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
            <Loader2 className="size-3.5 animate-spin" />
            Updating…
          </span>
        )}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Initial loading — structured skeleton cards                         */}
      {/* ------------------------------------------------------------------ */}
      {isLoading && (
        <div className="flex flex-col gap-3">
          {Array.from({ length: 8 }).map((_, i) => (
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
      {/* Virtual wine card list                                               */}
      {/* ------------------------------------------------------------------ */}
      {!isLoading && !isError && (
        groupedWines.length === 0 ? (
          <EmptyState
            icon={Wine}
            title="No wines found"
            description="Try adjusting the filters or clearing your search."
          />
        ) : (
          <div ref={listRef}>
            <div
              style={{
                height: `${virtualizer.getTotalSize()}px`,
                position: "relative",
              }}
            >
              {virtualizer.getVirtualItems().map((vRow) => (
                <div
                  key={vRow.key}
                  data-index={vRow.index}
                  ref={virtualizer.measureElement}
                  style={{
                    position: "absolute",
                    top: 0,
                    left: 0,
                    width: "100%",
                    transform: `translateY(${vRow.start - virtualizer.options.scrollMargin}px)`,
                    paddingBottom: "0.75rem", // gap-3 equivalent between cards
                  }}
                >
                  <WineGroup
                    wines={groupedWines[vRow.index].wines}
                    allDrinkIndices={allDrinkIndices}
                  />
                </div>
              ))}
            </div>
          </div>
        )
      )}
    </div>
  );
}

/**
 * Structured skeleton that mirrors the WineCard collapsed layout:
 * colour stripe | bottle illustration | producer + name + origin | vintage + qty + rating + chevron
 */
function SkeletonCard() {
  return (
    <div className="flex overflow-hidden rounded-xl border border-border bg-card animate-pulse">
      {/* Colour stripe */}
      <div className="w-1 shrink-0 bg-muted-foreground/20" />
      <div className="flex flex-1 items-center gap-3 px-4 py-3">
        {/* Bottle illustration placeholder */}
        <div className="hidden sm:block h-9 w-5 shrink-0 rounded bg-muted" />
        {/* Text area */}
        <div className="flex flex-1 flex-col gap-1.5 min-w-0">
          <div className="h-3 w-24 rounded bg-muted" />
          <div className="h-4 w-48 max-w-[60%] rounded bg-muted" />
          <div className="h-3 w-32 rounded bg-muted" />
        </div>
        {/* Right chips */}
        <div className="ml-2 flex shrink-0 items-center gap-2">
          <div className="h-4 w-10 rounded bg-muted" />
          <div className="h-5 w-12 rounded-md bg-muted" />
          <div className="h-4 w-6 rounded bg-muted" />
          <div className="h-4 w-4 rounded bg-muted" />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Public export — wraps CellarInventoryInner in a Suspense boundary so that
// useSearchParams() does not break static rendering of parent segments.
// ---------------------------------------------------------------------------

export default function CellarInventory(props: CellarInventoryProps) {
  return (
    <Suspense
      fallback={
        <div className="flex flex-col gap-3">
          {Array.from({ length: 8 }).map((_, i) => <SkeletonCard key={i} />)}
        </div>
      }
    >
      <CellarInventoryInner {...props} />
    </Suspense>
  );
}


