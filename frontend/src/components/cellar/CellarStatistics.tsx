/**
 * CellarStatistics component — all chart rows (Steps 2.8 + 2.9).
 *
 * Row 1 (Step 2.8):
 *   1. Wine Type Distribution  — donut (hole: 0.4), wine-type colour map
 *   2. Country Distribution    — vertical bar, purple, top 8 countries
 *   3. Top Varietals           — horizontal bar, purple
 *
 * Row 2 (Step 2.9):
 *   4. Top Regions             — horizontal bar, green
 *   5. Drinking Window Status  — donut pie (Ready / Soon / Aging)
 *   6. Cellar Size Over Time   — bar chart, wine-brown, monthly cumulative
 *
 * Row 3 (Step 2.9):
 *   7. Top Rated Wines         — horizontal bar, purple-to-green gradient
 *
 * All data comes from GET /api/cellar/charts (ChartDataResponse).
 * Conversion pattern: Python go.Figure() → <PlotlyChart data={[{...}]} layout={{...}} />
 */

import type * as PlotlyType from "plotly.js";
import type { ChartDataResponse } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import PlotlyChart from "@/components/PlotlyChart";

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const WINE_TYPE_COLORS: Record<string, string> = {
  Red: "rgba(139, 26, 26, 0.85)",
  White: "rgba(244, 229, 161, 0.85)",
  "Rosé": "rgba(255, 182, 193, 0.85)",
  Rose: "rgba(255, 182, 193, 0.85)",
  Sparkling: "rgba(255, 215, 0, 0.85)",
  Dessert: "rgba(221, 161, 94, 0.85)",
  Fortified: "rgba(160, 82, 45, 0.85)",
};

const PURPLE = "rgba(123, 31, 162, 0.85)";
const GREEN  = "rgba(67, 160, 71, 0.85)";
const BROWN  = "rgba(139, 69, 19, 0.85)";

const CHART_H = 300;
const MARGIN  = { t: 10, b: 10, l: 10, r: 10 };

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

/** Sum the `bottles` field across an array that came from a Record<string, unknown>. */
function sumBottles(arr: unknown): number {
  if (!Array.isArray(arr)) return 0;
  return (arr as Record<string, unknown>[]).reduce(
    (acc, item) => acc + num(item.bottles),
    0,
  );
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
  const typeLabels  = data.wine_type_distribution.map((d) => str(d.wine_type));
  const typeValues  = data.wine_type_distribution.map((d) => num(d.bottles));
  const typeColors  = typeLabels.map((l) => WINE_TYPE_COLORS[l] ?? PURPLE);

  const wineTypePie: PlotlyType.Data = {
    type: "pie",
    labels: typeLabels,
    values: typeValues,
    marker: { colors: typeColors },
    hole: 0.4,
    textinfo: "label+percent",
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 1: Country Distribution (top 8) ──────────────────────────────────
  const countrySlice   = data.country_distribution.slice(0, 8);
  const countryNames   = countrySlice.map((d) => str(d.country));
  const countryBottles = countrySlice.map((d) => num(d.bottles));

  const countryBar: PlotlyType.Data = {
    type: "bar",
    x: countryNames,
    y: countryBottles,
    marker: { color: PURPLE },
    text: countryBottles.map(String),
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 1: Varietal Distribution (horizontal bar) ────────────────────────
  const varietalNames   = data.varietal_distribution.map((d) => str(d.varietal));
  const varietalBottles = data.varietal_distribution.map((d) => num(d.bottles));

  const varietalBar: PlotlyType.Data = {
    type: "bar",
    y: varietalNames,
    x: varietalBottles,
    orientation: "h",
    marker: { color: PURPLE },
    text: varietalBottles.map(String),
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 2: Top Regions (horizontal bar, green) ───────────────────────────
  const regionLabels  = data.region_distribution.map((d) => {
    const r = str(d.region, "");
    const c = str(d.country, "");
    if (r && c) return `${r}, ${c}`;
    return r || c || "Unknown";
  });
  const regionBottles = data.region_distribution.map((d) => num(d.bottles));

  const regionBar: PlotlyType.Data = {
    type: "bar",
    y: regionLabels,
    x: regionBottles,
    orientation: "h",
    marker: { color: GREEN },
    text: regionBottles.map(String),
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 2: Drinking Window Status (donut pie) ─────────────────────────────
  const dw = data.drinking_window_wines as Record<string, unknown>;
  const readyCount = sumBottles(dw.ready_now);
  const soonCount  = sumBottles(dw.drink_soon);
  const agingCount = sumBottles(dw.for_aging);
  const dwEmpty    = readyCount + soonCount + agingCount === 0;

  const dwPie: PlotlyType.Data = {
    type: "pie",
    labels: ["Ready Now", "Drink Soon (1-2 yrs)", "For Aging (3+ yrs)"],
    values: [readyCount, soonCount, agingCount],
    marker: {
      colors: [
        "rgba(67, 160, 71, 0.85)",
        "rgba(255, 167, 38, 0.85)",
        "rgba(139, 26, 26, 0.85)",
      ],
    },
    hole: 0.4,
    textinfo: "label+percent",
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Row 2: Cellar Size Over Time (bar chart, wine-brown) ──────────────────
  const timeline        = data.cellar_size_over_time;
  const timelineMonths  = timeline.map((d) => str(d.month_display ?? d.month));
  const timelineBottles = timeline.map((d) => num(d.cumulative_bottles));

  // Downsample tick labels to max 6 — mirrors cellar_stats.py logic.
  const tickStep = Math.max(1, Math.floor(timelineMonths.length / 6));
  const tickVals = timelineMonths.filter((_, i) => i % tickStep === 0);

  const timelineBar: PlotlyType.Data = {
    type: "bar",
    x: timelineMonths,
    y: timelineBottles,
    marker: { color: BROWN },
    text: timelineBottles.map(String),
    textposition: "auto",
    name: "Total Bottles",
  } as PlotlyType.Data;

  const timelineLayout: Partial<PlotlyType.Layout> = {
    xaxis: {
      title: { text: "Month" },
      tickangle: 45,
      tickmode: "array",
      tickvals: tickVals,
      ticktext: tickVals,
      type: "category",
    },
    yaxis: { title: { text: "Bottles" } },
    showlegend: false,
    height: CHART_H,
    margin: { t: 10, b: 50, l: 10, r: 10 }, // extra bottom margin for rotated tick labels
  };

  // ── Row 3: Top Rated Wines (horizontal bar, purple) ───────────────────────
  // Label format: "Producer · Wine Name (Vintage)" — mirrors Streamlit expander titles.
  const topRatedLabels = data.top_rated.map((d) => {
    const producer = str(d.producer, "");
    const wine     = str(d.wine_name, "");
    const vintage  = d.vintage != null ? ` (${d.vintage})` : "";
    return producer ? `${producer} · ${wine}${vintage}` : `${wine}${vintage}`;
  });
  const topRatedValues = data.top_rated.map((d) => num(d.personal_rating));

  // Colour each bar by its rating tier (green → gold → orange, high → low).
  const topRatedColors = topRatedValues.map((r) => {
    if (r >= 94) return "rgba(46, 125, 50, 0.85)";   // Exceptional
    if (r >= 90) return "rgba(67, 160, 71, 0.85)";   // Excellent
    if (r >= 86) return "rgba(124, 179, 66, 0.85)";  // Very Good
    if (r >= 80) return "rgba(253, 216, 53, 0.85)";  // Good
    return "rgba(255, 179, 0, 0.85)";                // Average
  });

  const topRatedBar: PlotlyType.Data = {
    type: "bar",
    y: topRatedLabels,
    x: topRatedValues,
    orientation: "h",
    marker: { color: topRatedColors },
    text: topRatedValues.map((v) => `${v}/100`),
    textposition: "auto",
  } as PlotlyType.Data;

  const topRatedLayout: Partial<PlotlyType.Layout> = {
    xaxis: { title: { text: "Rating" }, range: [70, 100] },
    showlegend: false,
    height: Math.max(CHART_H, topRatedLabels.length * 32 + 40), // scale with item count
    margin: { t: 10, b: 10, l: 200, r: 60 }, // wide left margin for long wine names
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-6">
      {/* -------------------------------------------------------------------- */}
      {/* Row 1: Wine Type · Country · Varietals                               */}
      {/* -------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ChartCard
          title="Wine Type Distribution"
          isEmpty={typeLabels.length === 0}
          emptyMessage="No wine type data available."
        >
          <PlotlyChart
            data={[wineTypePie]}
            layout={{ showlegend: true, height: CHART_H, margin: MARGIN }}
          />
        </ChartCard>

        <ChartCard
          title="Country Distribution"
          isEmpty={countryNames.length === 0}
          emptyMessage="No country data available."
        >
          <PlotlyChart
            data={[countryBar]}
            layout={{
              xaxis: { title: { text: "Country" } },
              yaxis: { title: { text: "Bottles" } },
              showlegend: false,
              height: CHART_H,
              margin: MARGIN,
            }}
          />
        </ChartCard>

        <ChartCard
          title="Top Varietals"
          isEmpty={varietalNames.length === 0}
          emptyMessage="No varietal data available."
        >
          <PlotlyChart
            data={[varietalBar]}
            layout={{
              xaxis: { title: { text: "Bottles" } },
              showlegend: false,
              height: CHART_H,
              margin: { t: 10, b: 10, l: 120, r: 10 },
            }}
          />
        </ChartCard>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* Row 2: Regions · Drinking Window · Cellar Timeline                   */}
      {/* -------------------------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ChartCard
          title="Top Regions"
          isEmpty={regionLabels.length === 0}
          emptyMessage="No region data available."
        >
          <PlotlyChart
            data={[regionBar]}
            layout={{
              xaxis: { title: { text: "Bottles" } },
              showlegend: false,
              height: CHART_H,
              margin: { t: 10, b: 10, l: 140, r: 10 },
            }}
          />
        </ChartCard>

        <ChartCard
          title="Drinking Window Status"
          isEmpty={dwEmpty}
          emptyMessage="No drinking window data available."
        >
          <PlotlyChart
            data={[dwPie]}
            layout={{ showlegend: true, height: CHART_H, margin: MARGIN }}
          />
        </ChartCard>

        <ChartCard
          title="Cellar Size Over Time"
          isEmpty={timelineMonths.length === 0}
          emptyMessage="No CellarTracker purchase data available."
        >
          <PlotlyChart data={[timelineBar]} layout={timelineLayout} />
        </ChartCard>
      </div>

      {/* -------------------------------------------------------------------- */}
      {/* Row 3: Top Rated Wines                                               */}
      {/* -------------------------------------------------------------------- */}
      {data.top_rated.length > 0 && (
        <div className="grid grid-cols-1 gap-4">
          <ChartCard
            title="Top Rated Wines"
            isEmpty={topRatedLabels.length === 0}
            emptyMessage="No rated wines found."
          >
            <PlotlyChart data={[topRatedBar]} layout={topRatedLayout} />
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
          <p className="py-10 text-center text-sm text-muted-foreground">
            {emptyMessage}
          </p>
        ) : (
          children
        )}
      </CardContent>
    </Card>
  );
}



