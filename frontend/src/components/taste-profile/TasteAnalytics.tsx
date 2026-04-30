"use client";

/**
 * TasteAnalytics component — consistency-aligned redesign.
 *
 * Layout:
 *   Row 1: Wine Type Distribution (donut) · Rating Distribution (ordered bar)
 *   Row 2: Rating Trends (line + muted bars) · Varietal Analysis (h-bar)
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
const H = 260;
const H_HBAR = (items: { name: string }[], perItem = 32) =>
  Math.min(Math.max(H, items.length * perItem + 48), 380);

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Red → yellow → green gradient for ordered rating buckets. */
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

function VarietalTooltip({
  active,
  payload,
  label,
}: {
  active?: boolean;
  payload?: Array<{ payload?: { winesTasted?: number; avgRating?: number } }>;
  label?: string | number;
}) {
  if (!active || !payload?.length) return null;

  const row = payload[0]?.payload;
  const winesTasted = row?.winesTasted ?? 0;
  const avgRating = row?.avgRating ?? 0;

  return (
    <div className="rounded-md border border-border bg-background px-3 py-2 shadow-lg text-xs">
      {label != null && <p className="font-medium text-foreground mb-1">{label}</p>}
      <div className="flex items-center gap-1.5">
        <span className="inline-block h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: CHART_ACCENT.tertiary }} />
        <span className="text-muted-foreground">Wines Tasted:</span>
        <span className="font-medium text-foreground">{winesTasted}</span>
      </div>
      <div className="flex items-center gap-1.5">
        <span className="inline-block h-2 w-2 rounded-full shrink-0" style={{ backgroundColor: CHART_ACCENT.primary }} />
        <span className="text-muted-foreground">Avg Rating:</span>
        <span className="font-medium text-foreground">{Number(avgRating).toFixed(1)}/100</span>
      </div>
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
  // Wine type donut
  const wineTypePieData = wineTypes.types.map((t) => ({
    name: t.wine_type,
    value: t.wines_tasted,
    fill: wineTypeRgba(t.wine_type),
  }));

  // Rating distribution (ordered bars; buckets are ordinal)
  const ratingBarData = ratingDistribution.buckets.map((b, i, arr) => ({
    name: b.range,
    value: b.count,
    fill: ratingBucketFill(i, arr.length),
  }));

  // Rating trends (line-first with muted volume bars)
  const trendData = ratingTrends.points.map((p) => ({
    month: p.month,
    avgRating: p.avg_rating,
    winesTasted: p.wines_count,
  }));
  const trendInterval = Math.max(0, Math.floor(trendData.length / 6) - 1);

  // Varietal analysis (h-bar ranking with avg rating in tooltip)
  const varData = varietals.varietals
    .map((v) => ({
      name: v.varietal,
      winesTasted: v.wines_tasted,
      avgRating: v.avg_rating ?? 0,
    }))
    .sort((a, b) => b.winesTasted - a.winesTasted);

  const varYW = Math.min(varData.reduce((m, d) => Math.max(m, d.name.length), 0) * 6 + 8, 180);

  return (
    <div className="flex flex-col gap-4">
      {/* Row 1: Wine Type Distribution + Rating Distribution */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Wine Type Distribution"
          description="Share of wines tasted by style."
          isEmpty={wineTypePieData.length === 0}
          emptyMessage="No wine type data available yet."
        >
          <ResponsiveContainer width="100%" height={H}>
            <PieChart>
              <Pie
                data={wineTypePieData}
                cx="50%"
                cy="50%"
                innerRadius="38%"
                outerRadius="62%"
                dataKey="value"
                paddingAngle={3}
              >
                {wineTypePieData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
              </Pie>
              <Tooltip content={<ChartTooltip hideName />} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Rating Distribution"
          description="How your scores spread across ordered rating buckets."
          isEmpty={ratingBarData.length === 0}
          emptyMessage="No rating data available yet."
        >
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={ratingBarData} margin={{ top: 8, right: 12, bottom: 32, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-25} textAnchor="end" interval={0} />
              <YAxis tick={TICK} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Wines" radius={[3, 3, 0, 0]}>
                {ratingBarData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 2: Rating Trends + Varietal Analysis */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title={`Rating Trends${ratingTrends.trend ? ` (${ratingTrends.trend})` : ""}`}
          description="Monthly average rating (line) with wines tasted (muted bars)."
          isEmpty={trendData.length < 2}
          emptyMessage="Not enough data to show trends yet. Keep tasting wines!"
        >
          <ResponsiveContainer width="100%" height={H}>
            <ComposedChart data={trendData} margin={{ top: 12, right: 32, bottom: 30, left: 32 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} />
              <XAxis
                dataKey="month"
                tick={{ ...TICK, fontSize: 10 }}
                angle={-30}
                textAnchor="end"
                interval={trendInterval}
              />
              <YAxis yAxisId="left" domain={[0, 100]} tick={TICK} />
              <YAxis yAxisId="right" orientation="right" tick={TICK} />
              <Tooltip content={<ChartTooltip />} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar
                yAxisId="right"
                dataKey="winesTasted"
                name="Wines Tasted"
                fill={CHART_ACCENT_DIM.tertiary}
                radius={[2, 2, 0, 0]}
              />
              <Line
                yAxisId="left"
                type="monotone"
                dataKey="avgRating"
                name="Avg Rating"
                stroke={CHART_ACCENT.primary}
                strokeWidth={2}
                dot={{ r: 3, fill: CHART_ACCENT.primary }}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Varietal Analysis"
          description="Most tasted varietals ranked by volume (avg rating in tooltip)."
          isEmpty={varData.length === 0}
          emptyMessage="No varietal data available yet."
        >
          <ResponsiveContainer width="100%" height={H_HBAR(varData)}>
            <BarChart data={varData} layout="vertical" margin={{ top: 8, right: 24, bottom: 8, left: varYW }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={varYW - 2} />
              <Tooltip content={<VarietalTooltip />} />
              <Bar dataKey="winesTasted" name="Wines Tasted" fill={CHART_ACCENT.tertiary} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>
    </div>
  );
}
