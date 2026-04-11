"use client";

/**
 * CellarTabs component — Phase 4F redesign.
 *
 * Underline-style tab bar with brand-burgundy active state.
 * Inventory tab shows total wine count badge.
 * Tab panels fade in on activation.
 * Active tab is persisted in the URL (?tab=inventory|statistics) so that
 * browser back/forward and page shares preserve the selected view.
 * The Plotly resize workaround is removed (no longer needed with Recharts).
 */

import { Suspense, useMemo, useState } from "react";
import { usePathname, useRouter, useSearchParams } from "next/navigation";
import { BarChart2, Wine } from "lucide-react";

import type { ChartDataResponse, FilterOptions } from "@/lib/types";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import CellarInventory from "@/components/cellar/CellarInventory";
import CellarStatistics from "@/components/cellar/CellarStatistics";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type TabId = "inventory" | "statistics";

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
// CellarTabsInner (uses useSearchParams — wrapped in Suspense by the export)
// ---------------------------------------------------------------------------

function CellarTabsInner({ filterOptions, chartData }: CellarTabsProps) {
  const searchParams = useSearchParams();
  const router = useRouter();
  const pathname = usePathname();

  const activeTab = (searchParams.get("tab") as TabId | null) ?? "inventory";
  // Lazy-mount the statistics panel on first visit and keep it mounted to
  // preserve chart state when switching back to inventory.
  const [hasViewedStats, setHasViewedStats] = useState(() => activeTab === "statistics");

  // Total unique wines across all types for the inventory badge.
  const totalWines = useMemo(
    () =>
      chartData.wine_type_distribution.reduce(
        (acc, d) =>
          acc +
          (typeof d.unique_wines === "number"
            ? d.unique_wines
            : Number(d.unique_wines) || 0),
        0,
      ),
    [chartData],
  );

  function handleTabChange(tab: TabId) {
    if (tab === "statistics") setHasViewedStats(true);
    const next = new URLSearchParams(searchParams.toString());
    if (tab === "inventory") {
      next.delete("tab"); // inventory is the default — keep URLs clean
    } else {
      next.set("tab", tab);
    }
    router.replace(`${pathname}?${next.toString()}`, { scroll: false });
  }

  const tabs = [
    { id: "inventory"  as TabId, label: "Cellar Inventory",    Icon: Wine,      badge: totalWines > 0 ? totalWines : null },
    { id: "statistics" as TabId, label: "Statistics & Charts", Icon: BarChart2,  badge: null },
  ];

  return (
    <div className="flex flex-col gap-4">
      {/* Underline tab bar */}
      <div
        role="tablist"
        aria-label="Cellar views"
        className="flex border-b border-border"
      >
        {tabs.map(({ id, label, Icon, badge }) => {
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
                "flex items-center gap-2 px-4 py-3 text-sm font-medium",
                "border-b-2 -mb-px transition-colors duration-150",
                isActive
                  ? "border-brand-burgundy text-brand-burgundy"
                  : "border-transparent text-muted-foreground hover:text-foreground hover:border-border",
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

      {/* Inventory panel — always mounted */}
      <div
        id="tabpanel-inventory"
        role="tabpanel"
        aria-labelledby="tab-inventory"
        className={cn(
          "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
          activeTab !== "inventory" && "hidden",
        )}
      >
        <CellarInventory filterOptions={filterOptions} />
      </div>

      {/* Statistics panel — lazy-mounted on first visit */}
      {hasViewedStats && (
        <div
          id="tabpanel-statistics"
          role="tabpanel"
          aria-labelledby="tab-statistics"
          className={cn(
            "motion-safe:animate-in motion-safe:fade-in-0 motion-safe:duration-200",
            activeTab !== "statistics" && "hidden",
          )}
        >
          <CellarStatistics data={chartData} />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Public export — Suspense boundary for useSearchParams.
// ---------------------------------------------------------------------------

export default function CellarTabs(props: CellarTabsProps) {
  return (
    <Suspense fallback={<div className="h-10 animate-pulse rounded-lg bg-muted" />}>
      <CellarTabsInner {...props} />
    </Suspense>
  );
}

