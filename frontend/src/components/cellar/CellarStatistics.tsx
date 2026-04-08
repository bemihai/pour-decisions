"use client";

/**
 * CellarStatistics component — 9-chart layout (Steps 2.8 + 2.9).
 *
 * Row 1: Wine Type Distribution · Country Distribution · Vintage Distribution
 * Row 2: Rating Distribution    · Drinking Window       · Wine Age Analysis
 * Row 3: Top Varietals          · Top Regions           · Cellar Size Over Time
 * Row 4: Top Rated Wines (full-width, conditional)
 *
 * Mirrors the 9-chart layout from show_cellar_statistics() in
 * src/ui/helper/cellar_stats.py.
 */

import type * as PlotlyType from "plotly.js";
import type { ChartDataResponse } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import PlotlyChart from "@/components/PlotlyChart";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WINE_TYPE_COLORS: Record<string, string> = {
  Red:       "rgba(139, 26, 26, 0.85)",
  White:     "rgba(244, 229, 161, 0.85)",
  "Rosé":    "rgba(255, 182, 193, 0.85)",
  Rose:      "rgba(255, 182, 193, 0.85)",
  Sparkling: "rgba(255, 215, 0, 0.85)",
  Dessert:   "rgba(221, 161, 94, 0.85)",
  Fortified: "rgba(160, 82, 45, 0.85)",
};

const PURPLE = "rgba(123, 31, 162, 0.85)";
const GREEN  = "rgba(67, 160, 71, 0.85)";
const BROWN  = "rgba(139, 69, 19, 0.85)";

const CHART_H = 300;
const MARGIN  = { t: 10, b: 10, l: 10, r: 10 };

// Rating tier colors — high (green) → low (orange), matching the Python version.
const RATING_TIER_COLORS = [
  "rgba(46, 125, 50, 0.85)",
  "rgba(67, 160, 71, 0.85)",
  "rgba(124, 179, 66, 0.85)",
  "rgba(253, 216, 53, 0.85)",
  "rgba(255, 179, 0, 0.85)",
  "rgba(245, 124, 0, 0.85)",
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function str(v: unknown, fallback = "Unknown"): string {
  return v != null && v !== "" ? String(v) : fallback;
}

function num(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function sumBottles(arr: unknown): number {
  if (!Array.isArray(arr)) return 0;
  return (arr as Record<string, unknown>[]).reduce((acc, item) => acc + num(item.bottles), 0);
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CellarStatisticsProps {
  data: ChartDataResponse;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CellarStatistics({ data }: CellarStatisticsProps) {

  // ── Row 1: Wine Type Distribution ────────────────────────────────────────
  const typeLabels = data.wine_type_distribution.map((d) => str(d.wine_type));
  const typeValues = data.wine_type_distribution.map((d) => num(d.bottles));
  const typeColors = typeLabels.map((l) => WINE_TYPE_COLORS[l] ?? PURPLE);

  const wineTypePie: PlotlyType.Data = {
    type: "pie", labels: typeLabels, values: typeValues,
    marker: { colors: typeColors }, hole: 0.4,
    textinfo: "label+percent", textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 1: Country Distribution (top 8) ──────────────────────────────────
  const countrySlice   = data.country_distribution.slice(0, 8);
  const countryNames   = countrySlice.map((d) => str(d.country));
  const countryBottles = countrySlice.map((d) => num(d.bottles));

  const countryBar: PlotlyType.Data = {
    type: "bar", x: countryNames, y: countryBottles,
    marker: { color: PURPLE }, text: countryBottles.map(String), textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 1: Vintage Distribution ───────────────────────────────────────────
  const vintages       = data.vintage_distribution.map((d) => num(d.vintage));
  const vintageBottles = data.vintage_distribution.map((d) => num(d.bottles));
  // Gradient from pale-red (young) to deep-red (old) — mirrors cellar_stats.py.
  const vintageColors  = vintages.map((_, i, arr) => {
    const t = arr.length > 1 ? i / (arr.length - 1) : 0;
    const r = Math.round(139 + t * (220 - 139));
    const g = Math.round(26  + t * (130 - 26));
    const b = Math.round(26  + t * (100 - 26));
    return `rgba(${r}, ${g}, ${b}, 0.85)`;
  });

  const vintageBar: PlotlyType.Data = {
    type: "bar", x: vintages, y: vintageBottles,
    marker: { color: vintageColors },
    text: vintageBottles.map(String), textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 2: Rating Distribution ────────────────────────────────────────────
  const ratingTiers   = data.rating_distribution.map((d) => str(d.tier));
  const ratingCounts  = data.rating_distribution.map((d) => num(d.wines));
  const ratingColors  = ratingCounts.map((_, i) => RATING_TIER_COLORS[i] ?? PURPLE);

  const ratingBar: PlotlyType.Data = {
    type: "bar", y: ratingTiers, x: ratingCounts,
    orientation: "h", marker: { color: ratingColors },
    text: ratingCounts.map(String), textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 2: Drinking Window Status ─────────────────────────────────────────
  const dw         = data.drinking_window_wines as Record<string, unknown>;
  const readyCount = sumBottles(dw.ready_now);
  const soonCount  = sumBottles(dw.drink_soon);
  const agingCount = sumBottles(dw.for_aging);
  const dwEmpty    = readyCount + soonCount + agingCount === 0;

  const dwPie: PlotlyType.Data = {
    type: "pie",
    labels: ["Ready Now", "Drink Soon (1-2 yrs)", "For Aging (3+ yrs)"],
    values: [readyCount, soonCount, agingCount],
    marker: { colors: ["rgba(67,160,71,0.85)", "rgba(255,167,38,0.85)", "rgba(139,26,26,0.85)"] },
    hole: 0.4, textinfo: "label+percent", textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 2: Wine Age Analysis ──────────────────────────────────────────────
  const ageRanges   = data.wine_age_distribution.map((d) => str(d.range));
  const ageBottles  = data.wine_age_distribution.map((d) => num(d.bottles));
  const ageColors   = [
    "rgba(255,224,130,0.85)", "rgba(255,183,77,0.85)", "rgba(255,152,0,0.85)",
    "rgba(245,124,0,0.85)",   "rgba(191,54,12,0.85)",
  ];

  const ageBar: PlotlyType.Data = {
    type: "bar", x: ageRanges, y: ageBottles,
    marker: { color: ageColors.slice(0, ageRanges.length) },
    text: ageBottles.map(String), textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 3: Top Varietals ──────────────────────────────────────────────────
  const varietalNames   = data.varietal_distribution.map((d) => str(d.varietal));
  const varietalBottles = data.varietal_distribution.map((d) => num(d.bottles));

  const varietalBar: PlotlyType.Data = {
    type: "bar", y: varietalNames, x: varietalBottles,
    orientation: "h", marker: { color: PURPLE },
    text: varietalBottles.map(String), textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 3: Top Regions ────────────────────────────────────────────────────
  const regionLabels = data.region_distribution.map((d) => {
    const r = str(d.region, ""); const c = str(d.country, "");
    return r && c ? `${r}, ${c}` : r || c || "Unknown";
  });
  const regionBottles = data.region_distribution.map((d) => num(d.bottles));

  const regionBar: PlotlyType.Data = {
    type: "bar", y: regionLabels, x: regionBottles,
    orientation: "h", marker: { color: GREEN },
    text: regionBottles.map(String), textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 3: Cellar Size Over Time ──────────────────────────────────────────
  const timeline        = data.cellar_size_over_time;
  const timelineMonths  = timeline.map((d) => str(d.month_display ?? d.month));
  const timelineBottles = timeline.map((d) => num(d.cumulative_bottles));
  const tickStep = Math.max(1, Math.floor(timelineMonths.length / 6));
  const tickVals = timelineMonths.filter((_, i) => i % tickStep === 0);

  const timelineBar: PlotlyType.Data = {
    type: "bar", x: timelineMonths, y: timelineBottles,
    marker: { color: BROWN }, text: timelineBottles.map(String),
    textposition: "auto", name: "Total Bottles",
  } as PlotlyType.Data;

  // ── Row 4: Top Rated Wines ────────────────────────────────────────────────
  const topRatedLabels = data.top_rated.map((d) => {
    const producer = str(d.producer, "");
    const wine     = str(d.wine_name, "");
    const vintage  = d.vintage != null ? ` (${d.vintage})` : "";
    return producer ? `${producer} · ${wine}${vintage}` : `${wine}${vintage}`;
  });
  const topRatedValues = data.top_rated.map((d) => num(d.personal_rating));
  const topRatedColors = topRatedValues.map((r) => {
    if (r >= 94) return "rgba(46,125,50,0.85)";
    if (r >= 90) return "rgba(67,160,71,0.85)";
    if (r >= 86) return "rgba(124,179,66,0.85)";
    if (r >= 80) return "rgba(253,216,53,0.85)";
    return "rgba(255,179,0,0.85)";
  });

  const topRatedBar: PlotlyType.Data = {
    type: "bar", y: topRatedLabels, x: topRatedValues,
    orientation: "h", marker: { color: topRatedColors },
    text: topRatedValues.map((v) => `${v}/100`), textposition: "auto",
  } as PlotlyType.Data;

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-6">
      {/* -------------------------------------------------------------------- */}
      {/* Row 1: Wine Type · Country · Vintage                                  */}
      {/* -------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ChartCard title="Wine Type Distribution" isEmpty={typeLabels.length === 0} emptyMessage="No wine type data available.">
          <PlotlyChart data={[wineTypePie]} layout={{ showlegend: true, height: CHART_H, margin: MARGIN }} />
        </ChartCard>

        <ChartCard title="Country Distribution" isEmpty={countryNames.length === 0} emptyMessage="No country data available.">
          <PlotlyChart
            data={[countryBar]}
            layout={{ xaxis: { title: { text: "Country" } }, yaxis: { title: { text: "Bottles" } }, showlegend: false, height: CHART_H, margin: MARGIN }}
          />
        </ChartCard>

        <ChartCard title="Vintage Distribution" isEmpty={vintages.length === 0} emptyMessage="No vintage data available.">
          <PlotlyChart
            data={[vintageBar]}
            layout={{ xaxis: { title: { text: "Vintage Year" }, type: "category" }, yaxis: { title: { text: "Bottles" } }, showlegend: false, height: CHART_H, margin: MARGIN }}
          />
        </ChartCard>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* Row 2: Rating Distribution · Drinking Window · Wine Age Analysis      */}
      {/* -------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ChartCard title="Rating Distribution" isEmpty={ratingTiers.length === 0} emptyMessage="No rated wines in cellar.">
          <PlotlyChart
            data={[ratingBar]}
            layout={{ xaxis: { title: { text: "Wines" } }, showlegend: false, height: CHART_H, margin: { t: 10, b: 10, l: 160, r: 10 } }}
          />
        </ChartCard>

        <ChartCard title="Drinking Window Status" isEmpty={dwEmpty} emptyMessage="No drinking window data available.">
          <PlotlyChart data={[dwPie]} layout={{ showlegend: true, height: CHART_H, margin: MARGIN }} />
        </ChartCard>

        <ChartCard title="Wine Age Analysis" isEmpty={ageRanges.length === 0} emptyMessage="No vintage data for age analysis.">
          <PlotlyChart
            data={[ageBar]}
            layout={{ xaxis: { title: { text: "Age Range" } }, yaxis: { title: { text: "Bottles" } }, showlegend: false, height: CHART_H, margin: MARGIN }}
          />
        </ChartCard>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* Row 3: Top Varietals · Top Regions · Cellar Size Over Time            */}
      {/* -------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ChartCard title="Top Varietals" isEmpty={varietalNames.length === 0} emptyMessage="No varietal data available.">
          <PlotlyChart
            data={[varietalBar]}
            layout={{ xaxis: { title: { text: "Bottles" } }, showlegend: false, height: CHART_H, margin: { t: 10, b: 10, l: 120, r: 10 } }}
          />
        </ChartCard>

        <ChartCard title="Top Regions" isEmpty={regionLabels.length === 0} emptyMessage="No region data available.">
          <PlotlyChart
            data={[regionBar]}
            layout={{ xaxis: { title: { text: "Bottles" } }, showlegend: false, height: CHART_H, margin: { t: 10, b: 10, l: 140, r: 10 } }}
          />
        </ChartCard>

        <ChartCard title="Cellar Size Over Time" isEmpty={timelineMonths.length === 0} emptyMessage="No CellarTracker purchase data available.">
          <PlotlyChart
            data={[timelineBar]}
            layout={{
              xaxis: { title: { text: "Month" }, tickangle: 45, tickmode: "array", tickvals: tickVals, ticktext: tickVals, type: "category" },
              yaxis: { title: { text: "Bottles" } },
              showlegend: false, height: CHART_H, margin: { t: 10, b: 50, l: 10, r: 10 },
            }}
          />
        </ChartCard>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* Row 4: Top Rated Wines (full-width, only when data is present)        */}
      {/* -------------------------------------------------------------------- */}
      {data.top_rated.length > 0 && (
        <div className="grid grid-cols-1 gap-4">
          <ChartCard title="Top Rated Wines" isEmpty={topRatedLabels.length === 0} emptyMessage="No rated wines found.">
            <PlotlyChart
              data={[topRatedBar]}
              layout={{
                xaxis: { title: { text: "Rating" }, range: [70, 100] },
                showlegend: false,
                height: Math.max(CHART_H, topRatedLabels.length * 32 + 40),
                margin: { t: 10, b: 10, l: 200, r: 60 },
              }}
            />
          </ChartCard>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChartCard — shared card wrapper
// ---------------------------------------------------------------------------

interface ChartCardProps {
  title: string;
  isEmpty: boolean;
  emptyMessage: string;
  children: React.ReactNode;
}

function ChartCard({ title, isEmpty, emptyMessage, children }: ChartCardProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm">{title}</CardTitle>
      </CardHeader>
      <CardContent className="p-0 pb-2">
        {isEmpty ? (
          <p className="py-10 text-center text-sm text-muted-foreground">{emptyMessage}</p>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}
