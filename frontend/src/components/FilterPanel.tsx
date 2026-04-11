"use client";

/**
 * FilterPanel component.
 *
 * Reusable filter controls for the cellar inventory and taste-profile pages.
 * Replaces duplicated filter logic in cellar_stats.py and taste_profile_stats.py.
 *
 * - Select filters (wine type, country, producer, location) trigger onChange immediately.
 * - The search input is debounced by 300 ms to avoid excessive refetches while typing.
 * - All props except `options` and `onChange` are optional; sections that are not
 *   needed can be hidden via `showLocation`, `showRating`, `showSort`.
 */

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Search, X } from "lucide-react";

import type { FilterOptions, InventoryFilters } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

// ---------------------------------------------------------------------------
// Public types — exported so consumers (CellarInventory, TasteProfile) can
// reference them without redeclaring.
// ---------------------------------------------------------------------------

export interface SortOption {
  label: string;
  value: string;
}

export interface FilterPanelProps {
  /** Available filter values returned by GET /api/cellar/inventory or /api/cellar/filters. */
  options: FilterOptions;
  /** Called with the current InventoryFilters whenever a filter changes. */
  onChange: (filters: InventoryFilters) => void;
  /**
   * Initial filter values to populate the dropdowns (e.g. read from URL search
   * params).  Only applied on mount — subsequent external changes are ignored.
   */
  defaultFilters?: InventoryFilters;
  /** Show the location dropdown (default: true). */
  showLocation?: boolean;
  /** Show the rating filter dropdown (default: true). */
  showRating?: boolean;
  /** Show the sort-by dropdown (default: true). */
  showSort?: boolean;
  /** Override the default sort options list. */
  sortOptions?: SortOption[];
  /** Override the initial sort_by value. Defaults to "created_at_desc". */
  defaultSort?: string;
  className?: string;
}

// ---------------------------------------------------------------------------
// Default constants
// ---------------------------------------------------------------------------

/**
 * Sentinel value that means "no filter applied" inside the component's
 * internal state.  Mapped to `undefined` (omitted) before calling onChange.
 */
const FILTER_ALL = "__all__";

/** Default sort options matching _SORT_KEYS in src/api/routes/cellar.py. */
export const DEFAULT_SORT_OPTIONS: SortOption[] = [
  { label: "Added (Newest)", value: "created_at_desc" },
  { label: "Producer", value: "producer" },
  { label: "Wine Name", value: "wine_name" },
  { label: "Vintage (New to Old)", value: "vintage_desc" },
  { label: "Vintage (Old to New)", value: "vintage_asc" },
  { label: "Rating (High to Low)", value: "rating_desc" },
  { label: "Rating (Low to High)", value: "rating_asc" },
  { label: "Drink Sooner", value: "drink_desc" },
  { label: "Drink Later", value: "drink_asc" },
];

const RATING_OPTIONS: Array<{ label: string; value: string }> = [
  { label: "All Ratings", value: FILTER_ALL },
  { label: "Rated Only", value: "rated" },
  { label: "Unrated", value: "unrated" },
  { label: "90+", value: "90+" },
  { label: "80+", value: "80+" },
  { label: "70+", value: "70+" },
];

// ---------------------------------------------------------------------------
// Internal state shape
// ---------------------------------------------------------------------------

interface SelectFilters {
  wine_type: string;
  country: string;
  producer: string;
  location: string;
  rating_filter: string;
  sort_by: string;
}

const INITIAL_SELECT_FILTERS: SelectFilters = {
  wine_type: FILTER_ALL,
  country: FILTER_ALL,
  producer: FILTER_ALL,
  location: FILTER_ALL,
  rating_filter: FILTER_ALL,
  sort_by: "created_at_desc",
};

/** Convert internal state + search string to the InventoryFilters API shape. */
function buildFilters(selects: SelectFilters, search: string): InventoryFilters {
  const out: InventoryFilters = {};
  if (selects.wine_type !== FILTER_ALL) out.wine_type = selects.wine_type;
  if (selects.country !== FILTER_ALL) out.country = selects.country;
  if (selects.producer !== FILTER_ALL) out.producer = selects.producer;
  if (selects.location !== FILTER_ALL) out.location = selects.location;
  if (selects.rating_filter !== FILTER_ALL) out.rating_filter = selects.rating_filter;
  if (selects.sort_by) out.sort_by = selects.sort_by;
  const trimmed = search.trim();
  if (trimmed) out.search = trimmed;
  return out;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function FilterPanel({
  options,
  onChange,
  defaultFilters,
  showLocation = true,
  showRating = true,
  showSort = true,
  sortOptions = DEFAULT_SORT_OPTIONS,
  defaultSort,
  className,
}: FilterPanelProps) {
  const initialSort = defaultFilters?.sort_by ?? defaultSort ?? "created_at_desc";
  const [selects, setSelects] = useState<SelectFilters>({
    wine_type: defaultFilters?.wine_type ?? FILTER_ALL,
    country: defaultFilters?.country ?? FILTER_ALL,
    producer: defaultFilters?.producer ?? FILTER_ALL,
    location: defaultFilters?.location ?? FILTER_ALL,
    rating_filter: defaultFilters?.rating_filter ?? FILTER_ALL,
    sort_by: initialSort,
  });
  const [searchInput, setSearchInput] = useState(defaultFilters?.search ?? "");

  // Stable refs so callbacks and effects never go stale without needing to be
  // listed as dependencies — avoids recreating functions on every render.
  const onChangeRef = useRef(onChange);
  useEffect(() => { onChangeRef.current = onChange; }, [onChange]);

  const selectsRef = useRef(selects);
  useEffect(() => { selectsRef.current = selects; }, [selects]);

  const searchInputRef = useRef(searchInput);
  useEffect(() => { searchInputRef.current = searchInput; }, [searchInput]);

  // True when any filter deviates from its default value.
  const isFiltered = useMemo(() => (
    selects.wine_type !== FILTER_ALL ||
    selects.country !== FILTER_ALL ||
    selects.producer !== FILTER_ALL ||
    selects.location !== FILTER_ALL ||
    selects.rating_filter !== FILTER_ALL ||
    selects.sort_by !== initialSort ||
    searchInput.trim() !== ""
  ), [selects, searchInput, initialSort]);

  function clearAll() {
    const resetSelects = { ...INITIAL_SELECT_FILTERS, sort_by: initialSort };
    setSelects(resetSelects);
    setSearchInput("");
    onChangeRef.current(buildFilters(resetSelects, ""));
  }

  // Immediate update for all select/dropdown filters.
  const updateSelect = useCallback((key: keyof SelectFilters, value: string) => {
    setSelects((prev) => {
      const next = { ...prev, [key]: value };
      onChangeRef.current(buildFilters(next, searchInputRef.current));
      return next;
    });
  }, []);

  // Debounced update for the search input — fires 300 ms after typing stops.
  useEffect(() => {
    const timer = setTimeout(() => {
      onChangeRef.current(buildFilters(selectsRef.current, searchInput));
    }, 300);
    return () => clearTimeout(timer);
  }, [searchInput]);

  return (
    <div className={cn("flex flex-col gap-3", className)}>
      {/* Row 1: type / country / producer / location
           2 cols until lg (1024px), then 4 — avoids cramped dropdowns at
           768-1023px (e.g. tablets and narrow desktops). */}
      <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
        <SelectFilter
          placeholder="All Types"
          value={selects.wine_type}
          onValueChange={(v) => updateSelect("wine_type", v)}
          items={options.wine_types}
        />
        <SelectFilter
          placeholder="All Countries"
          value={selects.country}
          onValueChange={(v) => updateSelect("country", v)}
          items={options.countries}
        />
        <SelectFilter
          placeholder="All Producers"
          value={selects.producer}
          onValueChange={(v) => updateSelect("producer", v)}
          items={options.producers}
        />
        {showLocation && (
          <SelectFilter
            placeholder="All Locations"
            value={selects.location}
            onValueChange={(v) => updateSelect("location", v)}
            items={options.locations}
          />
        )}
      </div>

      {/* Row 2: search / rating / sort / clear */}
      <div className="flex flex-wrap items-center gap-2">
        {/* Search — grows to fill available space */}
        <div className="relative min-w-[200px] flex-1">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            type="search"
            placeholder="Search by name, producer, varietal..."
            value={searchInput}
            onChange={(e) => setSearchInput(e.target.value)}
            className="pl-8"
          />
        </div>

        {showRating && (
          <Select
            value={selects.rating_filter}
            onValueChange={(v) => updateSelect("rating_filter", v)}
          >
            <SelectTrigger className="w-36">
              <SelectValue placeholder="All Ratings" />
            </SelectTrigger>
            <SelectContent>
              {RATING_OPTIONS.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {showSort && (
          <Select
            value={selects.sort_by}
            onValueChange={(v) => updateSelect("sort_by", v)}
          >
            <SelectTrigger className="w-44">
              <SelectValue placeholder="Sort by..." />
            </SelectTrigger>
            <SelectContent>
              {sortOptions.map((opt) => (
                <SelectItem key={opt.value} value={opt.value}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}

        {/* Clear all — only shown when at least one filter is active */}
        {isFiltered && (
          <Button
            variant="ghost"
            size="sm"
            onClick={clearAll}
            className="gap-1.5 text-muted-foreground hover:text-foreground"
          >
            <X className="size-3.5" />
            Clear all
          </Button>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// SelectFilter — private helper to keep the JSX above concise
// ---------------------------------------------------------------------------

interface SelectFilterProps {
  placeholder: string;
  value: string;
  onValueChange: (v: string) => void;
  items: string[];
}

function SelectFilter({ placeholder, value, onValueChange, items }: SelectFilterProps) {
  return (
    <Select value={value} onValueChange={onValueChange}>
      <SelectTrigger className="w-full">
        <SelectValue placeholder={placeholder} />
      </SelectTrigger>
      <SelectContent>
        {/* First item resets the filter */}
        <SelectItem value={FILTER_ALL}>{placeholder}</SelectItem>
        {items.map((item) => (
          <SelectItem key={item} value={item}>
            {item}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}

