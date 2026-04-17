"use client";

/**
 * CellarStatistics component — redesigned for consistency and better chart types.
 *
 * Layout (2-column grid, all paired charts share H=260):
 *   Row 1: Wine Type Distribution (donut)  · Drinking Window Status (donut)
 *   Row 2: Vintage Distribution (bar)      · Country Distribution (h-bar)
 *   Row 3: Wine Age Breakdown (stat card)  · Rating Breakdown (stat card)
 *   Row 4: Top Varietals (h-bar)           · Top Regions (h-bar)
 *   Row 5: Cellar Size Over Time (area)    — full width, compact
 */

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Legend,
  Pie,
  PieChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { ChartDataResponse } from "@/lib/types";
import { CHART_ACCENT, RATING_TIER_COLORS, wineTypeRgba } from "@/lib/chart-config";
import ChartCard from "@/components/charts/ChartCard";
import ChartTooltip from "@/components/charts/ChartTooltip";

// ---------------------------------------------------------------------------
// Shared constants — all paired charts share the same height for consistency
// ---------------------------------------------------------------------------

const TICK = { fontSize: 11, fill: "#8a7f77" };
const GRID = "#e8e2db";
const H = 260;         // unified chart height for all paired cards
const H_HBAR = (items: { name: string }[], perItem = 34) =>
  Math.min(Math.max(H, items.length * perItem + 48), 380);

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
  return (arr as Record<string, unknown>[]).reduce((a, item) => a + num(item.bottles), 0);
}
const yW = (items: { name: string }[], cap = 160) =>
  Math.min(items.reduce((m, d) => Math.max(m, d.name.length), 0) * 6 + 8, cap);

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

  // Wine Type donut
  const wineTypePieData = data.wine_type_distribution.map((d) => ({
    name: str(d.wine_type), value: num(d.bottles), fill: wineTypeRgba(str(d.wine_type)),
  }));

  // Country — horizontal bar (more readable than angled vertical labels)
  const countryData = data.country_distribution.slice(0, 8).map((d) => ({
    name: str(d.country), value: num(d.bottles),
  }));

  // Vintage bar
  const vintageData = data.vintage_distribution.map((d, i, arr) => {
    const t = arr.length > 1 ? i / (arr.length - 1) : 0;
    return {
      name: String(num(d.vintage)),
      value: num(d.bottles),
      fill: `rgba(${Math.round(139 + t * 81)}, ${Math.round(26 + t * 104)}, ${Math.round(26 + t * 74)}, 0.85)`,
    };
  });

  // Drinking window donut
  const dw = data.drinking_window_wines as Record<string, unknown>;
  const dwReady  = sumBottles(dw.ready_now);
  const dwSoon   = sumBottles(dw.drink_soon);
  const dwAging  = sumBottles(dw.for_aging);
  const dwTotal  = dwReady + dwSoon + dwAging;
  const dwData = [
    { name: "Ready Now",           value: dwReady, fill: "rgba(67,160,71,0.85)" },
    { name: "Drink Soon (1-2 yr)", value: dwSoon,  fill: "rgba(255,167,38,0.85)" },
    { name: "For Aging (3+ yr)",   value: dwAging, fill: "rgba(139,26,26,0.85)" },
  ].filter((d) => d.value > 0);

  // Wine age snapshot — raw numbers for stat card (2 buckets is too few for a bar chart)
  const ageTotal = data.wine_age_distribution.reduce((s, d) => s + num(d.bottles), 0);
  const ageItems = data.wine_age_distribution.map((d, i) => ({
    label: str(d.range),
    value: num(d.bottles),
    pct: ageTotal > 0 ? Math.round((num(d.bottles) / ageTotal) * 100) : 0,
    color: ["rgba(255,183,77,0.9)", "rgba(255,152,0,0.9)", "rgba(245,124,0,0.9)",
            "rgba(239,83,80,0.9)", "rgba(191,54,12,0.9)"][i] ?? CHART_ACCENT.secondary,
  }));

  // Rating snapshot — stat card (often only 2-3 tiers, too few for a standalone chart)
  const ratingItems = data.rating_distribution.map((d, i) => ({
    label: str(d.tier),
    value: num(d.wines),
    fill: RATING_TIER_COLORS[i] ?? CHART_ACCENT.primary,
  }));
  const ratedTotal = ratingItems.reduce((s, d) => s + d.value, 0);
  const totalWines = data.wine_type_distribution.reduce((s, d) => s + num(d.unique_wines), 0);

  // Top varietals h-bar
  const varietalData = data.varietal_distribution.slice(0, 10).map((d) => ({
    name: str(d.varietal), value: num(d.bottles),
  }));

  // Top regions h-bar
  const regionData = data.region_distribution.slice(0, 10).map((d) => {
    const r = str(d.region, ""); const c = str(d.country, "");
    return { name: r && c ? `${r}, ${c}` : r || c || "Unknown", value: num(d.bottles) };
  });

  // Cellar size over time — AreaChart (correct for cumulative time series)
  const cellarData = data.cellar_size_over_time.map((d) => ({
    name: str(d.month_display ?? d.month), value: num(d.cumulative_bottles),
  }));
  const cellarInterval = Math.max(0, Math.floor(cellarData.length / 6) - 1);

  return (
    <div className="flex flex-col gap-4">

      {/* Row 1: Wine Type + Drinking Window */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Wine Type Distribution"
          description="Bottle count by wine style."
          isEmpty={wineTypePieData.length === 0}
          emptyMessage="No wine type data available."
        >
          <ResponsiveContainer width="100%" height={H}>
            <PieChart>
              <Pie data={wineTypePieData} cx="50%" cy="50%" innerRadius="38%" outerRadius="62%" dataKey="value" paddingAngle={3}>
                {wineTypePieData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
              </Pie>
              <Tooltip content={<ChartTooltip hideName />} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Drinking Window Status"
          description={dwTotal > 0 ? `${dwReady} bottles ready to open now out of ${dwTotal} total.` : "Which wines are ready to open?"}
          isEmpty={dwData.length === 0}
          emptyMessage="No drinking window data available."
        >
          <ResponsiveContainer width="100%" height={H}>
            <PieChart>
              <Pie data={dwData} cx="50%" cy="50%" innerRadius="38%" outerRadius="62%" dataKey="value" paddingAngle={3}>
                {dwData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
              </Pie>
              <Tooltip content={<ChartTooltip hideName />} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 2: Vintage + Country */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Vintage Distribution"
          description="Bottles per vintage year."
          isEmpty={vintageData.length === 0}
          emptyMessage="No vintage data available."
        >
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={vintageData} margin={{ top: 8, right: 8, bottom: 32, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-40} textAnchor="end" interval={Math.max(0, Math.floor(vintageData.length / 8) - 1)} />
              <YAxis tick={TICK} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" radius={[3, 3, 0, 0]}>
                {vintageData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Country Distribution"
          description="Top countries by bottle count."
          isEmpty={countryData.length === 0}
          emptyMessage="No country data available."
        >
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={countryData} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: yW(countryData, 90) }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={yW(countryData, 88)} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" fill={CHART_ACCENT.tertiary} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 3: Wine Age Breakdown + Rating Breakdown (stat cards) */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Wine Age Breakdown"
          description="How old your bottles are, by vintage."
          isEmpty={ageItems.length === 0}
          emptyMessage="No vintage data for age analysis."
        >
          <div className="flex flex-col gap-4 px-6 pb-6 pt-3">
            {ageItems.map((item) => (
              <div key={item.label} className="space-y-1.5">
                <div className="flex items-center justify-between text-sm">
                  <span className="font-medium text-foreground">{item.label}</span>
                  <span className="tabular-nums text-muted-foreground">
                    {item.value} btl · {item.pct}%
                  </span>
                </div>
                <div className="h-2.5 w-full overflow-hidden rounded-full bg-muted">
                  <div
                    className="h-full rounded-full transition-all"
                    style={{ width: `${item.pct}%`, backgroundColor: item.color }}
                  />
                </div>
              </div>
            ))}
          </div>
        </ChartCard>

        <ChartCard
          title="Rating Breakdown"
          description={
            ratedTotal > 0
              ? `${ratedTotal} of ${totalWines} wines have a personal rating.`
              : "No wines have been rated yet."
          }
          isEmpty={ratingItems.length === 0}
          emptyMessage="No rated wines in your cellar."
        >
          <div className="flex flex-col gap-3 px-6 pb-6 pt-3">
            {ratedTotal > 0 && (
              <div className="mb-1 h-2.5 w-full overflow-hidden rounded-full bg-muted flex">
                {ratingItems.map((item, i) => (
                  <div
                    key={i}
                    className="h-full first:rounded-l-full last:rounded-r-full"
                    style={{ width: `${(item.value / ratedTotal) * 100}%`, backgroundColor: item.fill }}
                  />
                ))}
              </div>
            )}
            {ratingItems.map((item, i) => (
              <div key={i} className="flex items-center gap-3 text-sm">
                <span className="h-2.5 w-2.5 shrink-0 rounded-full" style={{ backgroundColor: item.fill }} />
                <span className="flex-1 text-foreground">{item.label}</span>
                <span className="tabular-nums font-medium text-foreground">{item.value}</span>
                <span className="w-8 text-right tabular-nums text-muted-foreground text-xs">
                  {ratedTotal > 0 ? `${Math.round((item.value / ratedTotal) * 100)}%` : "—"}
                </span>
              </div>
            ))}
            {ratedTotal === 0 && (
              <p className="text-sm text-muted-foreground text-center py-4">
                Rate your wines in CellarTracker to see breakdown.
              </p>
            )}
          </div>
        </ChartCard>
      </div>

      {/* Row 4: Top Varietals + Top Regions */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard
          title="Top Varietals"
          description="Your most collected grape varieties."
          isEmpty={varietalData.length === 0}
          emptyMessage="No varietal data available."
        >
          <ResponsiveContainer width="100%" height={H_HBAR(varietalData)}>
            <BarChart data={varietalData} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: yW(varietalData) }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={yW(varietalData) - 2} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" fill={CHART_ACCENT.tertiary} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Top Regions"
          description="Your most collected wine regions."
          isEmpty={regionData.length === 0}
          emptyMessage="No region data available."
        >
          <ResponsiveContainer width="100%" height={H_HBAR(regionData, 32)}>
            <BarChart data={regionData} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: yW(regionData, 180) }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={yW(regionData, 178)} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" fill={CHART_ACCENT.green} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 5: Cellar Size Over Time — compact area chart */}
      <ChartCard
        title="Cellar Size Over Time"
        description="Cumulative bottle count per month based on purchase history."
        isEmpty={cellarData.length === 0}
        emptyMessage="No purchase history available."
      >
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={cellarData} margin={{ top: 8, right: 16, bottom: 36, left: 16 }}>
            <defs>
              <linearGradient id="cellarGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={CHART_ACCENT.secondary} stopOpacity={0.3} />
                <stop offset="95%" stopColor={CHART_ACCENT.secondary} stopOpacity={0.03} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
            <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-40} textAnchor="end" interval={cellarInterval} />
            <YAxis tick={TICK} width={32} />
            <Tooltip content={<ChartTooltip hideName />} />
            <Area
              type="monotone"
              dataKey="value"
              name="Total Bottles"
              stroke={CHART_ACCENT.secondary}
              strokeWidth={2}
              fill="url(#cellarGradient)"
              dot={{ r: 3, fill: CHART_ACCENT.secondary, strokeWidth: 0 }}
              activeDot={{ r: 5 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartCard>

    </div>
  );
}
