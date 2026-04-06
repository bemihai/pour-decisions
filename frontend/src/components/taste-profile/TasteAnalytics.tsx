"use client";

/**
 * TasteAnalytics component — Steps 3.2 + 3.3.
 *
 * Analytics tab with 5 Plotly charts:
 *   Row 1: Rating Distribution (donut) · Wine Type Distribution (pie)
 *   Row 2: Performance by Type (bar)   · Rating Trends (line+bar combo)
 *   Row 3: Varietal Analysis (dual-axis bar+line, full-width)
 *
 * Replaces show_rating_distribution(), show_wine_type_distribution(),
 * show_wine_type_performance(), show_rating_trends(), show_varietal_analysis()
 * from src/ui/helper/taste_profile_stats.py.
 */

import type * as PlotlyType from "plotly.js";

import type {
  RatingDistributionResponse,
  RatingTrendsResponse,
  VarietalsResponse,
  WineTypesResponse,
} from "@/lib/types";
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

const PURPLE     = "rgba(123, 31, 162, 0.85)";
const PURPLE_DIM = "rgba(123, 31, 162, 0.3)";
const AMBER      = "#FFC107";
const CHART_H    = 360;
const BASE_LAYOUT: Partial<PlotlyType.Layout> = {
  paper_bgcolor: "transparent",
  plot_bgcolor:  "transparent",
  font: { family: "Poppins, sans-serif", size: 12 },
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Red → yellow → green gradient matching the Python show_rating_distribution. */
function ratingBucketColor(index: number, total: number): string {
  const ratio = total > 1 ? index / (total - 1) : 0;
  if (ratio < 0.33) {
    return `rgb(244, ${Math.round(67 + (193 - 67) * (ratio / 0.33))}, ${Math.round(54 + (7 - 54) * (ratio / 0.33))})`;
  }
  if (ratio < 0.67) {
    const t = (ratio - 0.33) / 0.34;
    return `rgb(${Math.round(255 - (255 - 139) * t)}, ${Math.round(193 + (195 - 193) * t)}, ${Math.round(7 + (74 - 7) * t)})`;
  }
  const t = (ratio - 0.67) / 0.33;
  return `rgb(${Math.round(139 - 63 * t)}, ${Math.round(195 - 20 * t)}, ${Math.round(74 + 6 * t)})`;
}

function EmptyChart({ message }: { message: string }) {
  return (
    <div className="flex h-[200px] items-center justify-center text-sm text-muted-foreground">
      {message}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TasteAnalyticsProps {
  ratingDistribution: RatingDistributionResponse;
  wineTypes: WineTypesResponse;
  varietals: VarietalsResponse;
  ratingTrends: RatingTrendsResponse;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default function TasteAnalytics({
  ratingDistribution,
  wineTypes,
  varietals,
  ratingTrends,
}: TasteAnalyticsProps) {

  // ── Rating distribution donut ─────────────────────────────────────────────
  const ratingLabels = ratingDistribution.buckets.map((b) => b.range);
  const ratingValues = ratingDistribution.buckets.map((b) => b.count);
  const ratingColors = ratingLabels.map((_, i) =>
    ratingBucketColor(i, ratingLabels.length),
  );

  const ratingDonut: PlotlyType.Data = {
    type: "pie",
    labels: ratingLabels,
    values: ratingValues,
    hole: 0.4,
    marker: { colors: ratingColors },
    textinfo: "label+percent",
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Wine type distribution pie ────────────────────────────────────────────
  const typeLabels = wineTypes.types.map((t) => t.wine_type);
  const typeCounts = wineTypes.types.map((t) => t.wines_tasted);
  const typePieColors = typeLabels.map(
    (l) => WINE_TYPE_COLORS[l] ?? PURPLE,
  );

  const wineTypePie: PlotlyType.Data = {
    type: "pie",
    labels: typeLabels,
    values: typeCounts,
    hole: 0.4,
    marker: { colors: typePieColors },
    textinfo: "label+percent",
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Wine type performance (horizontal bar — avg rating) ───────────────────
  const ratedTypes = wineTypes.types.filter((t) => t.avg_rating != null);
  const perfNames  = ratedTypes.map((t) => t.wine_type);
  const perfRatings = ratedTypes.map((t) => t.avg_rating as number);
  const perfColors  = perfNames.map((n) => WINE_TYPE_COLORS[n] ?? PURPLE);

  const perfBar: PlotlyType.Data = {
    type: "bar",
    y: perfNames,
    x: perfRatings,
    orientation: "h",
    marker: { color: perfColors },
    text: perfRatings.map((r) => `${r.toFixed(1)}`),
    textposition: "auto",
  } as PlotlyType.Data;

  // ── Rating trends (line + bar combo) ─────────────────────────────────────
  const trendMonths  = ratingTrends.points.map((p) => p.month);
  const trendRatings = ratingTrends.points.map((p) => p.avg_rating);
  const trendCounts  = ratingTrends.points.map((p) => p.wines_count);

  const trendLine: PlotlyType.Data = {
    name: "Average Rating",
    type: "scatter",
    x: trendMonths,
    y: trendRatings,
    mode: "lines+markers",
    marker: { color: "#7b1fa2", size: 8 },
    line: { color: "#7b1fa2", width: 3 },
    yaxis: "y",
  } as PlotlyType.Data;

  const trendBar: PlotlyType.Data = {
    name: "Wines Tasted",
    type: "bar",
    x: trendMonths,
    y: trendCounts,
    marker: { color: PURPLE_DIM },
    yaxis: "y2",
  } as PlotlyType.Data;

  const trendLayout: Partial<PlotlyType.Layout> = {
    ...BASE_LAYOUT,
    height: CHART_H,
    margin: { t: 24, b: 40, l: 48, r: 48 },
    xaxis: { title: { text: "Month" } },
    yaxis: { title: { text: "Average Rating" }, side: "left", range: [0, 100] },
    yaxis2: { title: { text: "Wines Tasted" }, side: "right", overlaying: "y" } as unknown as PlotlyType.LayoutAxis,
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
    hovermode: "x unified",
  };

  // ── Varietal analysis (bar count + line avg rating, dual-axis) ────────────
  const varNames   = varietals.varietals.map((v) => v.varietal);
  const varCounts  = varietals.varietals.map((v) => v.wines_tasted);
  const varRatings = varietals.varietals.map((v) => v.avg_rating ?? 0);

  const varBar: PlotlyType.Data = {
    name: "Wines Tasted",
    type: "bar",
    x: varNames,
    y: varCounts,
    marker: { color: PURPLE },
    yaxis: "y",
    offsetgroup: 1,
  } as PlotlyType.Data;

  const varLine: PlotlyType.Data = {
    name: "Avg Rating",
    type: "scatter",
    x: varNames,
    y: varRatings,
    mode: "lines+markers",
    marker: { color: AMBER, size: 8 },
    line: { color: AMBER, width: 2 },
    yaxis: "y2",
  } as PlotlyType.Data;

  const varLayout: Partial<PlotlyType.Layout> = {
    ...BASE_LAYOUT,
    height: CHART_H,
    margin: { t: 24, b: 80, l: 48, r: 48 },
    xaxis: { tickangle: -45 },
    yaxis: { title: { text: "Wines Tasted" }, side: "left" },
    yaxis2: { title: { text: "Average Rating" }, side: "right", overlaying: "y", range: [0, 100] } as unknown as PlotlyType.LayoutAxis,
    legend: { orientation: "h", yanchor: "bottom", y: 1.02, xanchor: "right", x: 1 },
    hovermode: "x unified",
  };

  // Compact layout shared by the two donut/pie charts.
  const pieLayout: Partial<PlotlyType.Layout> = {
    ...BASE_LAYOUT,
    height: CHART_H,
    margin: { t: 24, b: 24, l: 16, r: 16 },
    showlegend: true,
    legend: { orientation: "v", yanchor: "middle", y: 0.5 },
  };

  const perfLayout: Partial<PlotlyType.Layout> = {
    ...BASE_LAYOUT,
    height: CHART_H,
    margin: { t: 24, b: 40, l: 100, r: 48 },
    xaxis: { title: { text: "Average Rating" }, range: [0, 100] },
  };

  return (
    <div className="flex flex-col gap-6">
      {/* Row 1: Rating Distribution + Wine Type Distribution */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Rating Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {ratingDistribution.buckets.length === 0 ? (
              <EmptyChart message="No rating data available yet." />
            ) : (
              <PlotlyChart data={[ratingDonut]} layout={pieLayout} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Wine Type Distribution</CardTitle>
          </CardHeader>
          <CardContent>
            {wineTypes.types.length === 0 ? (
              <EmptyChart message="No wine type data available yet." />
            ) : (
              <PlotlyChart data={[wineTypePie]} layout={pieLayout} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Row 2: Performance by Type + Rating Trends */}
      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Performance by Type</CardTitle>
          </CardHeader>
          <CardContent>
            {perfNames.length === 0 ? (
              <EmptyChart message="No rated wine types yet." />
            ) : (
              <PlotlyChart data={[perfBar]} layout={perfLayout} />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">
              Rating Trends
              {ratingTrends.trend && (
                <span className="ml-2 text-sm font-normal capitalize text-muted-foreground">
                  ({ratingTrends.trend})
                </span>
              )}
            </CardTitle>
          </CardHeader>
          <CardContent>
            {ratingTrends.points.length < 2 ? (
              <EmptyChart message="Not enough data to show rating trends. Keep tasting wines!" />
            ) : (
              <PlotlyChart data={[trendLine, trendBar]} layout={trendLayout} />
            )}
          </CardContent>
        </Card>
      </div>

      {/* Row 3: Varietal Analysis (full-width) */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Varietal Analysis</CardTitle>
        </CardHeader>
        <CardContent>
          {varietals.varietals.length === 0 ? (
            <EmptyChart message="No varietal data available yet." />
          ) : (
            <PlotlyChart data={[varBar, varLine]} layout={varLayout} />
          )}
        </CardContent>
      </Card>
    </div>
  );
}




