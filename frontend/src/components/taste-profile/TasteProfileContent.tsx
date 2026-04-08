"use client";

/**
 * TasteProfileContent component — Steps 3.2–3.5 wrapper.
 *
 * Three-tab switcher: Analytics / Tasting History / Favorites.
 * Uses the same lazy-mount strategy as CellarTabs so Plotly charts are not
 * rendered into a hidden container and TanStack Query cache stays populated
 * when switching tabs.
 *
 * All analytics and favorites data is pre-fetched by the parent Server
 * Component; only the Tasting History tab fetches client-side (because it
 * needs live filtering).
 */

import { useState } from "react";
import { BarChart2, Heart, History } from "lucide-react";

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
import TasteAnalytics from "@/components/taste-profile/TasteAnalytics";
import TasteFavorites from "@/components/taste-profile/TasteFavorites";
import TasteHistory from "@/components/taste-profile/TasteHistory";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId = "analytics" | "history" | "favorites";

const TABS = [
  { id: "analytics" as TabId,  label: "Analytics",       Icon: BarChart2 },
  { id: "history"   as TabId,  label: "Tasting History", Icon: History   },
  { id: "favorites" as TabId,  label: "Favorites",       Icon: Heart     },
] as const;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TasteProfileContentProps {
  // Analytics charts
  ratingDistribution: RatingDistributionResponse;
  wineTypes: WineTypesResponse;
  varietals: VarietalsResponse;
  ratingTrends: RatingTrendsResponse;
  // Favorites
  producers: ProducersResponse;
  regions: RegionsResponse;
  countries: CountriesResponse;
  vintages: VintagesResponse;
  appellations: AppellationsResponse;
  // Tasting history
  initialConsumedFilterOptions: ConsumedFilterOptions;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TasteProfileContent({
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
  const [activeTab, setActiveTab] = useState<TabId>("analytics");

  // Lazy-mount flags — once a tab has been visited its panel stays in the DOM
  // (hidden with CSS) so state is preserved and Plotly doesn't lose its canvas.
  const [visited, setVisited] = useState<Record<TabId, boolean>>({
    analytics: true, // analytics is the default tab, always mounted
    history:   false,
    favorites: false,
  });

  function handleTabChange(tab: TabId) {
    setActiveTab(tab);
    setVisited((prev) => ({ ...prev, [tab]: true }));
    if (tab === "analytics") {
      // Allow Plotly to recalculate chart widths when the tab becomes visible.
      requestAnimationFrame(() => window.dispatchEvent(new Event("resize")));
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Tab bar */}
      <div
        role="tablist"
        aria-label="Taste profile views"
        className="flex gap-2 rounded-xl border border-border bg-muted/40 p-1"
      >
        {TABS.map(({ id, label, Icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tp-panel-${id}`}
              id={`tp-tab-${id}`}
              onClick={() => handleTabChange(id)}
              className={cn(
                "flex flex-1 items-center justify-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all",
                isActive
                  ? "bg-purple-600 text-white shadow-sm dark:bg-purple-700"
                  : "text-muted-foreground hover:bg-background hover:text-foreground",
              )}
            >
              <Icon className="size-4 shrink-0" />
              <span className="hidden sm:inline">{label}</span>
            </button>
          );
        })}
      </div>

      {/* Analytics panel — always mounted */}
      <div
        id="tp-panel-analytics"
        role="tabpanel"
        aria-labelledby="tp-tab-analytics"
        className={cn(activeTab !== "analytics" && "hidden")}
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
          className={cn(activeTab !== "history" && "hidden")}
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
          className={cn(activeTab !== "favorites" && "hidden")}
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
    </div>
  );
}

