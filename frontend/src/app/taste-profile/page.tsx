/**
 * Taste Profile page — Server Component (Step 3.6).
 *
 * Parallel-fetches all taste profile data at request time so the client
 * receives pre-populated props.  Only the Tasting History tab fetches
 * client-side (because it supports interactive filtering).
 *
 * force-dynamic: profile analytics change whenever tastings are recorded,
 * so Next.js must never serve a stale cached response.
 */

export const dynamic = "force-dynamic";

import {
  getAppellations,
  getConsumedWines,
  getCountries,
  getProducers,
  getRatingDistribution,
  getRatingTrends,
  getRegions,
  getTasteOverview,
  getVarietals,
  getVintages,
  getWineTypes,
} from "@/lib/api";
import PageHeader from "@/components/PageHeader";
import TasteOverview from "@/components/taste-profile/TasteOverview";
import TasteProfileContent from "@/components/taste-profile/TasteProfileContent";

export default async function TasteProfilePage() {
  const [
    overview,
    ratingDistribution,
    wineTypes,
    varietals,
    ratingTrends,
    producers,
    regions,
    countries,
    vintages,
    appellations,
    consumedInitial,
  ] = await Promise.all([
    getTasteOverview(),
    getRatingDistribution(),
    getWineTypes(),
    getVarietals(10),
    getRatingTrends(),
    getProducers(5),
    getRegions(5),
    getCountries(5),
    getVintages(5),
    getAppellations(5),
    // Fetch initial consumed wines only to obtain the filter options; the
    // TasteHistory tab re-fetches client-side when filters change.
    getConsumedWines({ limit: 20 }),
  ]);

  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      <PageHeader
        title="Taste Profile"
        subtitle="Your palate, analysed"
        compact
        className="mb-6"
      />

      {/* Key metrics */}
      <TasteOverview stats={overview} className="mb-6" />

      {/* Tabbed content: Analytics / Tasting History / Favorites */}
      <TasteProfileContent
        ratingDistribution={ratingDistribution}
        wineTypes={wineTypes}
        varietals={varietals}
        ratingTrends={ratingTrends}
        producers={producers}
        regions={regions}
        countries={countries}
        vintages={vintages}
        appellations={appellations}
        initialConsumedFilterOptions={consumedInitial.filter_options}
      />
    </div>
  );
}
