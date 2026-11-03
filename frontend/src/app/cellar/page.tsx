/**
 * Cellar page — Server Component.
 *
 * Parallel-fetches cellar stats, filter options, and chart data at request
 * time so CellarOverview and CellarTabs are pre-populated with real data.
 *
 * force-dynamic: cellar inventory changes after every sync, so Next.js must
 * never serve a cached HTML response for this route.
 */

import type { Metadata } from "next";

// Opt out of Next.js static generation — always fetch fresh data.
export const dynamic = "force-dynamic";

export const metadata: Metadata = {
  title: "Wine Cellar",
  description: "Browse your wine inventory, statistics, and drinking windows.",
};

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
    <div className="container mx-auto max-w-[1424px] px-4 py-6">
      {/* Header row: page title + sync button */}
      <div className="mb-5 flex flex-wrap items-start justify-between gap-4">
        <PageHeader
          title="Wine Cellar"
          subtitle="Choose what to open, track what you own, and plan what comes next."
          eyebrow="Your collection"
          compact
          className="mb-0"
        />
        <CellarSyncButton />
      </div>

      {/* Key metrics */}
      <CellarOverview stats={stats} className="mb-5" />

      {/* Tabbed inventory and statistics */}
      <CellarTabs filterOptions={filterOptions} chartData={chartData} />
    </div>
  );
}
