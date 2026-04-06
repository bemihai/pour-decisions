/**
 * Cellar page — Server Component.
 *
 * Parallel-fetches cellar stats, filter options, and chart data at request
 * time so CellarOverview and CellarTabs are pre-populated with real data.
 * Implements Step 2.12 of the React migration plan.
 */

import { getCellarStats, getCellarCharts, getFilterOptions } from "@/lib/api";
import PageHeader from "@/components/PageHeader";
import CellarOverview from "@/components/cellar/CellarOverview";
import CellarSyncButton from "@/components/cellar/CellarSyncButton";
import CellarTabs from "@/components/cellar/CellarTabs";

export default async function CellarPage() {
  const [stats, filterOptions, chartData] = await Promise.all([
    getCellarStats(),
    getFilterOptions(),
    getCellarCharts(),
  ]);

  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      {/* Header row: page title + sync button */}
      <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <PageHeader
          title="Wine Cellar"
          subtitle="Your personal collection"
          compact
        />
        <CellarSyncButton />
      </div>

      {/* Key metrics */}
      <CellarOverview stats={stats} className="mb-6" />

      {/* Tabbed inventory and statistics */}
      <CellarTabs filterOptions={filterOptions} chartData={chartData} />
    </div>
  );
}
