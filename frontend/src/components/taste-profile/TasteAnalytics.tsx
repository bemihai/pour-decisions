"use client";

/**
 * TasteAnalytics component — Phase 4F Recharts migration.
 *
 * 5 charts in a 2-column grid (with full-width bottom row):
 *   Row 1: Rating Distribution (donut) · Wine Type Distribution (pie)
 *   Row 2: Performance by Type (h-bar)  · Rating Trends (line+bar)
 *   Row 3: Varietal Analysis (bar+line, full width)
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Legend,
  Line,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type {
  RatingDistributionResponse,
  RatingTrendsResponse,
  VarietalsResponse,
  WineTypesResponse,
} from "@/lib/types";
import { CHART_ACCENT, CHART_ACCENT_DIM, wineTypeRgba } from "@/lib/chart-config";
import ChartCard from "@/components/charts/ChartCard";
import ChartTooltip from "@/components/charts/ChartTooltip";

// ---------------------------------------------------------------------------
// Shared constants
// ---------------------------------------------------------------------------

const TICK = { fontSize: 11, fill: "#8a7f77" };
const GRID = "#e8e2db";
const H = 300;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Red → yellow → green gradient, matching Python show_rating_distribution. */
function ratingBucketFill(index: number, total: number): string {
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
    <div className="flex items-center justify-center py-16 text-sm text-muted-foreground">
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

  // ── Rating Distribution donut ─────────────────────────────────────────────
  const ratingPieData = ratingDistribution.buckets.map((b, i, arr) => ({
    name: b.range,
    value: b.count,
    fill: ratingBucketFill(i, arr.length),
  }));

  // ── Wine Type Distribution pie ────────────────────────────────────────────
  const wineTypePieData = wineTypes.types.map((t) => ({
    name: t.wine_type,
    value: t.wines_tasted,
    fill: wineTypeRgba(t.wine_type),
  }));

  // ── Performance by Type (horizontal bar) ──────────────────────────────────
  const perfData = wineTypes.types
    .filter((t) => t.avg_rating != null)
    .map((t) => ({
      name: t.wine_type,
      value: parseFloat((t.avg_rating as number).toFixed(1)),
      fill: wineTypeRgba(t.wine_type),
    }));

  const perfYW = Math.min(perfData.reduce((m, d) => Math.max(m, d.name.length), 0) * 7 + 8, 120);

  // ── Rating Trends (ComposedChart — bar count + line avg rating) ───────────
  const trendData = ratingTrends.points.map((p) => ({
    month: p.month,
    avgRating: p.avg_rating,
    winesTasted: p.wines_count,
  }));

  // ── Varietal Analysis (ComposedChart — bar count + line avg rating) ───────
  const varData = varietals.varietals.map((v) => ({
    name: v.varietal,
    winesTasted: v.wines_tasted,
    avgRating: v.avg_rating ?? 0,
  }));

  return (
    <div className="flex flex-col gap-6">

      {/* Row 1: Rating Distribution + Wine Type Distribution */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Rating Distribution"
          description="How your tasting scores cluster across quality brackets."
        >
          {ratingPieData.length === 0 ? (
            <EmptyChart message="No rating data available yet." />
          ) : (
            <ResponsiveContainer width="100%" height={H}>
              <PieChart>
                <Pie
                  data={ratingPieData}
                  cx="50%" cy="50%"
                  innerRadius="35%" outerRadius="60%"
                  dataKey="value"
                  paddingAngle={2}
                >
                  {ratingPieData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
                </Pie>
                <Tooltip content={<ChartTooltip hideName />} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        <ChartCard
          title="Wine Type Distribution"
          description="Share of wines tasted by style — red, white, rosé, and more."
        >
          {wineTypePieData.length === 0 ? (
            <EmptyChart message="No wine type data available yet." />
          ) : (
            <ResponsiveContainer width="100%" height={H}>
              <PieChart>
                <Pie
                  data={wineTypePieData}
                  cx="50%" cy="50%"
                  innerRadius="35%" outerRadius="60%"
                  dataKey="value"
                  paddingAngle={2}
                >
                  {wineTypePieData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
                </Pie>
                <Tooltip content={<ChartTooltip hideName />} />
                <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
              </PieChart>
            </ResponsiveContainer>
          )}
        </ChartCard>
      </div>

      {/* Row 2: Performance by Type + Rating Trends */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Performance by Type"
          description="Average personal rating per wine style — which type do you rate highest?"
        >
          {perfData.length === 0 ? (
            <EmptyChart message="No rated wine types yet." />
          ) : (
            <ResponsiveContainer width="100%" height={H}>
              <BarChart data={perfData} layout="vertical" margin={{ top: 8, right: 48, bottom: 8, left: perfYW }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
                <XAxis type="number" domain={[0, 100]} tick={TICK} />
                <YAxis type="category" dataKey="name" tick={TICK} width={perfYW - 2} />
                <Tooltip content={<ChartTooltip formatter={(v) => `${v}/100`} hideName />} />
                <Bar dataKey="value" name="Avg Rating" radius={[0, 3, 3, 0]}>
                  {perfData.map((e, i) => <Cell key={i} fill={e.fill} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          )}
        </ChartCard>

        <ChartCard
          title={`Rating Trends${ratingTrends.trend ? ` (${ratingTrends.trend})` : ""}`}
          description="Monthly average rating (line) overlaid on wines tasted per month (bars)."
        >
          {trendData.length < 2 ? (
            <EmptyChart message="Not enough data to show trends yet. Keep tasting wines!" />
          ) : (
            <ResponsiveContainer width="100%" height={H}>
              <ComposedChart data={trendData} margin={{ top: 16, right: 48, bottom: 32, left: 48 }}>
                <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
                <XAxis dataKey="month" tick={{ ...TICK, fontSize: 10 }} angle={-30} textAnchor="end" interval={Math.max(0, Math.floor(trendData.length / 6) - 1)} />
                <YAxis yAxisId="left" domain={[0, 100]} tick={TICK} label={{ value: "Avg Rating", angle: -90, position: "insideLeft", style: TICK, dx: -4 }} />
                <YAxis yAxisId="right" orientation="right" tick={TICK} label={{ value: "Wines", angle: 90, position: "insideRight", style: TICK, dx: 4 }} />
                <Tooltip content={<ChartTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar yAxisId="right" dataKey="winesTasted" name="Wines Tasted" fill={CHART_ACCENT_DIM.tertiary} radius={[2, 2, 0, 0]} />
                <Line yAxisId="left" type="monotone" dataKey="avgRating" name="Avg Rating" stroke={CHART_ACCENT.primary} strokeWidth={2} dot={{ r: 3, fill: CHART_ACCENT.primary }} />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </ChartCard>
      </div>

      {/* Row 3: Varietal Analysis (full width) */}
      <ChartCard
        title="Varietal Analysis"
        description="Wines tasted per grape variety (bars) and average rating per varietal (line)."
      >
        {varData.length === 0 ? (
          <EmptyChart message="No varietal data available yet." />
        ) : (
          <ResponsiveContainer width="100%" height={H}>
            <ComposedChart data={varData} margin={{ top: 16, right: 48, bottom: 56, left: 48 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-40} textAnchor="end" interval={0} />
              <YAxis yAxisId="left" tick={TICK} label={{ value: "Wines Tasted", angle: -90, position: "insideLeft", style: TICK, dx: -4 }} />
              <YAxis yAxisId="right" orientation="right" domain={[0, 100]} tick={TICK} label={{ value: "Avg Rating", angle: 90, position: "insideRight", style: TICK, dx: 4 }} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar yAxisId="left" dataKey="winesTasted" name="Wines Tasted" fill={CHART_ACCENT.tertiary} radius={[2, 2, 0, 0]} />
              <Line yAxisId="right" type="monotone" dataKey="avgRating" name="Avg Rating" stroke={CHART_ACCENT.secondary} strokeWidth={2} dot={{ r: 3, fill: CHART_ACCENT.secondary }} />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </ChartCard>
    </div>
  );
}
