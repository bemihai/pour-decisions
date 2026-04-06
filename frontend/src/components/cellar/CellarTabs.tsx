"use client";

/**
 * CellarTabs component.
 *
 * Tab switcher between the Cellar Inventory and the Statistics & Charts views.
 * Replaces TABS_DISPLAY (40 lines of CSS) + st.tabs() from src/ui/pages/cellar.py.
 *
 * State management strategy:
 *   - CellarInventory is always mounted so its filter/pagination state
 *     is preserved when the user switches to Statistics and back.
 *   - CellarStatistics is lazy-mounted on first visit (avoids rendering
 *     Plotly charts into a hidden container) and stays mounted thereafter,
 *     hidden with CSS so Plotly's responsive listener keeps the charts alive.
 */

import { useState } from "react";
import { BarChart2, Wine } from "lucide-react";

import type { ChartDataResponse, FilterOptions } from "@/lib/types";
import { cn } from "@/lib/utils";
import CellarInventory from "@/components/cellar/CellarInventory";
import CellarStatistics from "@/components/cellar/CellarStatistics";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId = "inventory" | "statistics";

const TABS = [
  { id: "inventory" as TabId,  label: "Cellar Inventory",    Icon: Wine       },
  { id: "statistics" as TabId, label: "Statistics & Charts", Icon: BarChart2  },
] as const;

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CellarTabsProps {
  /** Pre-fetched by the parent Server Component via GET /api/cellar/filters. */
  filterOptions: FilterOptions;
  /** Pre-fetched by the parent Server Component via GET /api/cellar/charts. */
  chartData: ChartDataResponse;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CellarTabs({ filterOptions, chartData }: CellarTabsProps) {
  const [activeTab, setActiveTab] = useState<TabId>("inventory");
  // Track whether the Statistics tab has ever been visited so we can
  // lazy-mount CellarStatistics (avoids Plotly rendering into a hidden div).
  const [hasViewedStats, setHasViewedStats] = useState(false);

  function handleTabChange(tab: TabId) {
    setActiveTab(tab);
    if (tab === "statistics") setHasViewedStats(true);
  }

  return (
    <div className="flex flex-col gap-4">
      {/* ------------------------------------------------------------------ */}
      {/* Tab bar                                                              */}
      {/* ------------------------------------------------------------------ */}
      <div
        role="tablist"
        aria-label="Cellar views"
        className="flex gap-2 rounded-xl border border-border bg-muted/40 p-1"
      >
        {TABS.map(({ id, label, Icon }) => {
          const isActive = activeTab === id;
          return (
            <button
              key={id}
              role="tab"
              aria-selected={isActive}
              aria-controls={`tabpanel-${id}`}
              id={`tab-${id}`}
              onClick={() => handleTabChange(id)}
              className={cn(
                "flex flex-1 items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-all",
                isActive
                  ? "bg-purple-600 text-white shadow-sm dark:bg-purple-700"
                  : "text-muted-foreground hover:bg-background hover:text-foreground",
              )}
            >
              <Icon className="size-4 shrink-0" />
              <span>{label}</span>
            </button>
          );
        })}
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Tab panels                                                           */}
      {/* ------------------------------------------------------------------ */}

      {/* Inventory — always mounted to preserve filter / pagination state */}
      <div
        id="tabpanel-inventory"
        role="tabpanel"
        aria-labelledby="tab-inventory"
        className={cn(activeTab !== "inventory" && "hidden")}
      >
        <CellarInventory filterOptions={filterOptions} />
      </div>

      {/* Statistics — lazy-mounted on first visit, then kept in DOM so
          Plotly's responsive listener can resize charts on re-activation */}
      {hasViewedStats && (
        <div
          id="tabpanel-statistics"
          role="tabpanel"
          aria-labelledby="tab-statistics"
          className={cn(activeTab !== "statistics" && "hidden")}
        >
          <CellarStatistics data={chartData} />
        </div>
      )}
    </div>
  );
}

