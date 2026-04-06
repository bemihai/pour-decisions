/**
 * CellarStatistics component — Row 1 (Step 2.8).
 *
 * Renders the first row of 3 cellar analytics charts:
 *   1. Wine Type Distribution  — donut (hole: 0.4), wine-type colour map
 *   2. Country Distribution    — vertical bar, purple bars, top 8 countries
 *   3. Top Varietals           — horizontal bar, purple bars
 *
 * Props receive the pre-fetched ChartDataResponse from GET /api/cellar/charts.
 * Each chart is wrapped in a shadcn Card to match the existing design language.
 *
 * Conversion pattern:  Python go.Figure() → <PlotlyChart data={[{...}]} layout={{...}} />
 * The Plotly JSON schema is identical between Python and JavaScript.
 */

import type * as PlotlyType from "plotly.js";
import type { ChartDataResponse } from "@/lib/types";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import PlotlyChart from "@/components/PlotlyChart";

// ---------------------------------------------------------------------------
// Constants — match cellar_stats.py colour values exactly
// ---------------------------------------------------------------------------

/** Per-type fill colours matching show_cellar_statistics() in cellar_stats.py. */
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

/** Shared chart height — keeps all three cards the same visual height. */
const CHART_H = 300;

/** Compact shared margin so the chart fills the card without excessive padding. */
const MARGIN = { t: 10, b: 10, l: 10, r: 10 };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Safely coerce an `unknown` value to string. */
function str(v: unknown, fallback = "Unknown"): string {
  return v != null && v !== "" ? String(v) : fallback;
}

/** Safely coerce an `unknown` value to number. */
function num(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface CellarStatisticsProps {
  /** Pre-fetched from GET /api/cellar/charts. */
  data: ChartDataResponse;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function CellarStatistics({ data }: CellarStatisticsProps) {
  // ── Row 1: Wine Type Distribution ────────────────────────────────────────
  const typeLabels = data.wine_type_distribution.map((d) => str(d.wine_type));
  const typeValues = data.wine_type_distribution.map((d) => num(d.bottles));
  const typeColors = typeLabels.map(
    (l) => WINE_TYPE_COLORS[l] ?? PURPLE,
  );

  const wineTypePie: PlotlyType.Data = {
    type: "pie",
    labels: typeLabels,
    values: typeValues,
    marker: { colors: typeColors },
    hole: 0.4,
    textinfo: "label+percent",
    textposition: "auto",
  } as PlotlyType.Data;

  const wineTypeLayout: Partial<PlotlyType.Layout> = {
    showlegend: true,
    height: CHART_H,
    margin: MARGIN,
  };

  // ── Row 1: Country Distribution (top 8) ──────────────────────────────────
  const countrySlice = data.country_distribution.slice(0, 8);
  const countryNames = countrySlice.map((d) => str(d.country));
  const countryBottles = countrySlice.map((d) => num(d.bottles));

  const countryBar: PlotlyType.Data = {
    type: "bar",
    x: countryNames,
    y: countryBottles,
    marker: { color: PURPLE },
    text: countryBottles.map(String),
    textposition: "auto",
  } as PlotlyType.Data;

  const countryLayout: Partial<PlotlyType.Layout> = {
    xaxis: { title: { text: "Country" } },
    yaxis: { title: { text: "Bottles" } },
    showlegend: false,
    height: CHART_H,
    margin: MARGIN,
  };

  // ── Row 1: Varietal Distribution (horizontal bar) ────────────────────────
  const varietalNames = data.varietal_distribution.map((d) => str(d.varietal));
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

  const varietalLayout: Partial<PlotlyType.Layout> = {
    xaxis: { title: { text: "Bottles" } },
    showlegend: false,
    height: CHART_H,
    margin: { t: 10, b: 10, l: 120, r: 10 }, // wider left margin for long varietal names
  };

  // ── Render ────────────────────────────────────────────────────────────────
  return (
    <div className="flex flex-col gap-6">
      {/* Row 1 ---------------------------------------------------------------- */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        <ChartCard
          title="Wine Type Distribution"
          isEmpty={typeLabels.length === 0}
          emptyMessage="No wine type data available."
        >
          <PlotlyChart data={[wineTypePie]} layout={wineTypeLayout} />
        </ChartCard>

        <ChartCard
          title="Country Distribution"
          isEmpty={countryNames.length === 0}
          emptyMessage="No country data available."
        >
          <PlotlyChart data={[countryBar]} layout={countryLayout} />
        </ChartCard>

        <ChartCard
          title="Top Varietals"
          isEmpty={varietalNames.length === 0}
          emptyMessage="No varietal data available."
        >
          <PlotlyChart data={[varietalBar]} layout={varietalLayout} />
        </ChartCard>
      </div>

      {/* Rows 2 & 3 are added in Step 2.9 */}
    </div>
  );
}

// ---------------------------------------------------------------------------
// ChartCard — shared card wrapper for every chart in this component
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

