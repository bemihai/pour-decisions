"use client";

/**
 * TasteProfileContent component — Phase 4G update.
 *
 * Underline-style tabs with brand-burgundy active state.
 * Analytics tab shows total wines tasted badge.
 * Icon-only tab triggers on mobile get Tooltip wrappers.
 * Tab panels fade in on activation.
 * Active tab is persisted in the URL (?tab=analytics|history|favorites|preferences) so that
 * browser back/forward and page shares preserve the selected view.
 */

import { type KeyboardEvent, Suspense, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { BarChart2, Heart, History, SlidersHorizontal } from "lucide-react";

import type {
  AppellationsResponse,
  ConsumedFilterOptions,
  CountriesResponse,
  ProducersResponse,
  RatingDistributionResponse,
  RatingTrendsResponse,
  RegionsResponse,
  VarietalsResponse,
  VintagesResponse,
  WineTypesResponse,
} from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import TasteAnalytics from "@/components/taste-profile/TasteAnalytics";
import TasteFavorites from "@/components/taste-profile/TasteFavorites";
import TasteHistory from "@/components/taste-profile/TasteHistory";
import TastePreferences from "@/components/taste-profile/TastePreferences";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId = "analytics" | "history" | "favorites" | "preferences";

const VALID_TAB_IDS: ReadonlyArray<TabId> = ["analytics", "history", "favorites", "preferences"];

function isValidTabId(s: string | null): s is TabId {
  return s !== null && (VALID_TAB_IDS as string[]).includes(s);
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TasteProfileContentProps {
  ratingDistribution: RatingDistributionResponse;
  wineTypes: WineTypesResponse;
  varietals: VarietalsResponse;
  ratingTrends: RatingTrendsResponse;
  producers: ProducersResponse;
  regions: RegionsResponse;
  countries: CountriesResponse;
  vintages: VintagesResponse;
  appellations: AppellationsResponse;
  initialConsumedFilterOptions: ConsumedFilterOptions;
}

// ---------------------------------------------------------------------------
// Component (inner — uses useSearchParams, must be inside Suspense)
// ---------------------------------------------------------------------------

function TasteProfileContentInner({
  ratingDistribution,
  wineTypes,
  varietals,
  ratingTrends,
  producers,
  regions,
  countries,
  vintages,
  appellations,
  initialConsumedFilterOptions,
}: TasteProfileContentProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const rawTab = searchParams.get("tab");
  const activeTab: TabId = isValidTabId(rawTab) ? rawTab : "analytics";

  // Lazy-mount each panel on first visit and keep it mounted afterwards to
  // preserve chart state when switching tabs.
  const [visited, setVisited] = useState<Record<TabId, boolean>>(() => ({
    analytics: true,
    history:   activeTab === "history",
    favorites: activeTab === "favorites",
    preferences: activeTab === "preferences",
  }));

  // Total wines tasted badge for the Analytics tab.
  const totalTasted = wineTypes.types.reduce((acc, t) => acc + t.wines_tasted, 0);

  function handleTabChange(tab: TabId) {
    setVisited((prev) => ({ ...prev, [tab]: true }));
    const next = new URLSearchParams(searchParams.toString());
    if (tab === "analytics") {
      next.delete("tab"); // analytics is the default — keep URLs clean
    } else {
      next.set("tab", tab);
    }
    const query = next.toString();
    router.replace(query ? `${pathname}?${query}` : pathname, { scroll: false });
  }

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, currentIndex: number) {
    let nextIndex: number | null = null;
    if (event.key === "ArrowRight") nextIndex = (currentIndex + 1) % VALID_TAB_IDS.length;
    if (event.key === "ArrowLeft") nextIndex = (currentIndex - 1 + VALID_TAB_IDS.length) % VALID_TAB_IDS.length;
    if (event.key === "Home") nextIndex = 0;
    if (event.key === "End") nextIndex = VALID_TAB_IDS.length - 1;
    if (nextIndex === null) return;

    event.preventDefault();
    const nextTab = VALID_TAB_IDS[nextIndex];
    handleTabChange(nextTab);
    document.getElementById(`tp-tab-${nextTab}`)?.focus();
  }

  const tabs = [
    { id: "analytics" as TabId, label: "Analytics",       Icon: BarChart2, badge: totalTasted > 0 ? totalTasted : null },
    { id: "history"   as TabId, label: "Tasting History", Icon: History,   badge: null },
    { id: "favorites" as TabId, label: "Favorites",       Icon: Heart,     badge: null },
    { id: "preferences" as TabId, label: "Preferences",   Icon: SlidersHorizontal, badge: null },
  ];

  return (
    <div className="flex flex-col gap-5">
      {/* Underline tab bar */}
      <div
        role="tablist"
        aria-label="Taste profile views"
        className="flex border-b border-border"
      >
        {tabs.map(({ id, label, Icon, badge }, index) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tp-panel-${id}`}
              id={`tp-tab-${id}`}
              tabIndex={isActive ? 0 : -1}
              onClick={() => handleTabChange(id)}
              onKeyDown={(event) => handleTabKeyDown(event, index)}
              className={cn(
                "-mb-px flex shrink-0 items-center gap-2 border-b-2 px-4 py-3 text-sm font-medium transition-colors duration-150",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2",
                isActive
                  ? "border-brand-burgundy text-brand-burgundy dark:border-brand-burgundy-light dark:text-brand-burgundy-light"
                  : "border-transparent text-muted-foreground hover:border-border hover:text-foreground",
              )}
            >
              <Icon className="size-4 shrink-0" aria-hidden="true" />
              <span>{label}</span>
              {badge != null && (
                <Badge variant="secondary" className="ml-0.5 text-xs tabular-nums">
                  {badge}
                </Badge>
              )}
            </button>
          );
        })}
      </div>

      {/* Analytics panel — always mounted */}
      <div
        id="tp-panel-analytics"
        role="tabpanel"
        aria-labelledby="tp-tab-analytics"
        className={cn(
          "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
          activeTab !== "analytics" && "hidden",
        )}
      >
        <TasteAnalytics
          ratingDistribution={ratingDistribution}
          wineTypes={wineTypes}
          varietals={varietals}
          ratingTrends={ratingTrends}
        />
      </div>

      {/* Tasting History panel — lazy-mounted on first visit */}
      {visited.history && (
        <div
          id="tp-panel-history"
          role="tabpanel"
          aria-labelledby="tp-tab-history"
          className={cn(
            "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
            activeTab !== "history" && "hidden",
          )}
        >
          <TasteHistory initialFilterOptions={initialConsumedFilterOptions} />
        </div>
      )}

      {/* Favorites panel — lazy-mounted on first visit */}
      {visited.favorites && (
        <div
          id="tp-panel-favorites"
          role="tabpanel"
          aria-labelledby="tp-tab-favorites"
          className={cn(
            "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
            activeTab !== "favorites" && "hidden",
          )}
        >
          <TasteFavorites
            producers={producers}
            regions={regions}
            countries={countries}
            vintages={vintages}
            appellations={appellations}
          />
        </div>
      )}

      {/* Preferences panel — lazy-mounted on first visit */}
      {visited.preferences && (
        <div
          id="tp-panel-preferences"
          role="tabpanel"
          aria-labelledby="tp-tab-preferences"
          className={cn(
            "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
            activeTab !== "preferences" && "hidden",
          )}
        >
          <TastePreferences />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Public export — Suspense boundary for useSearchParams.
// ---------------------------------------------------------------------------

export default function TasteProfileContent(props: TasteProfileContentProps) {
  return (
    <Suspense fallback={<div className="h-10 animate-pulse rounded-lg bg-muted" />}>
      <TasteProfileContentInner {...props} />
    </Suspense>
  );
}
