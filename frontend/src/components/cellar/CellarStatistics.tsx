"use client";

/**
 * CellarStatistics component — Phase 4F Recharts migration.
 *
 * 2-column grid layout (was 3-column Plotly).  All charts use Recharts
 * ResponsiveContainer so they size themselves to the card width.
 *
 * Rows:
 *   Row 1: Wine Type Distribution (pie)       · Country Distribution (bar)
 *   Row 2: Vintage Distribution (bar)          · Rating Distribution (h-bar)
 *   Row 3: Drinking Window Status (pie)        · Wine Age Analysis (bar)
 *   Row 4: Top Varietals (h-bar)               · Top Regions (h-bar)
 *   Row 5: Cellar Size Over Time (bar)         — full width
 *   Row 6: Top Rated Wines (h-bar)             — full width, conditional
 */

import {
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
// Shared axis / grid styling
// ---------------------------------------------------------------------------

const TICK = { fontSize: 11, fill: "#8a7f77" };
const GRID = "#e8e2db";
const H = 280;

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

  // Wine Type pie
  const wineTypePieData = data.wine_type_distribution.map((d) => ({
    name: str(d.wine_type), value: num(d.bottles), fill: wineTypeRgba(str(d.wine_type)),
  }));

  // Country bar (top 8)
  const countryData = data.country_distribution.slice(0, 8).map((d) => ({
    name: str(d.country), value: num(d.bottles),
  }));

  // Vintage bar (gradient red → deep red per age)
  const vintageData = data.vintage_distribution.map((d, i, arr) => {
    const t = arr.length > 1 ? i / (arr.length - 1) : 0;
    return {
      name: String(num(d.vintage)),
      value: num(d.bottles),
      fill: `rgba(${Math.round(139 + t * 81)}, ${Math.round(26 + t * 104)}, ${Math.round(26 + t * 74)}, 0.85)`,
    };
  });

  // Rating distribution horizontal bar
  const ratingData = data.rating_distribution.map((d, i) => ({
    name: str(d.tier), value: num(d.wines),
    fill: RATING_TIER_COLORS[i] ?? CHART_ACCENT.primary,
  }));

  // Drinking window pie
  const dw = data.drinking_window_wines as Record<string, unknown>;
  const dwData = [
    { name: "Ready Now",            value: sumBottles(dw.ready_now),  fill: "rgba(67,160,71,0.85)" },
    { name: "Drink Soon (1-2 yrs)", value: sumBottles(dw.drink_soon), fill: "rgba(255,167,38,0.85)" },
    { name: "For Aging (3+ yrs)",   value: sumBottles(dw.for_aging),  fill: "rgba(139,26,26,0.85)" },
  ].filter((d) => d.value > 0);

  // Wine age bar
  const AGE_FILLS = [
    "rgba(255,224,130,0.85)", "rgba(255,183,77,0.85)", "rgba(255,152,0,0.85)",
    "rgba(245,124,0,0.85)",   "rgba(191,54,12,0.85)",
  ];
  const ageData = data.wine_age_distribution.map((d, i) => ({
    name: str(d.range), value: num(d.bottles), fill: AGE_FILLS[i] ?? CHART_ACCENT.secondary,
  }));

  // Top varietals horizontal bar (top 10)
  const varietalData = data.varietal_distribution.slice(0, 10).map((d) => ({
    name: str(d.varietal), value: num(d.bottles),
  }));

  // Top regions horizontal bar (top 10)
  const regionData = data.region_distribution.slice(0, 10).map((d) => {
    const r = str(d.region, ""); const c = str(d.country, "");
    return { name: r && c ? `${r}, ${c}` : r || c || "Unknown", value: num(d.bottles) };
  });

  // Cellar size over time bar
  const cellarData = data.cellar_size_over_time.map((d) => ({
    name: str(d.month_display ?? d.month), value: num(d.cumulative_bottles),
  }));
  const cellarInterval = Math.max(0, Math.floor(cellarData.length / 6) - 1);

  // Top rated horizontal bar (conditional)
  const topRated = data.top_rated.map((d) => {
    const r = num(d.personal_rating);
    return {
      name: `${str(d.producer, "") ? str(d.producer, "") + " · " : ""}${str(d.wine_name)}${d.vintage != null ? ` (${d.vintage})` : ""}`,
      value: r,
      fill: r >= 94 ? "rgba(46,125,50,0.85)" : r >= 90 ? "rgba(67,160,71,0.85)"
          : r >= 86 ? "rgba(124,179,66,0.85)" : r >= 80 ? "rgba(253,216,53,0.85)"
          : "rgba(255,179,0,0.85)",
    };
  });

  // Y-axis widths for horizontal bar charts
  const yW = (items: { name: string }[], cap = 160) =>
    Math.min(items.reduce((m, d) => Math.max(m, d.name.length), 0) * 6 + 8, cap);

  return (
    <div className="flex flex-col gap-6">

      {/* Row 1 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Wine Type Distribution" description="Bottle count by wine style in your cellar." isEmpty={wineTypePieData.length === 0} emptyMessage="No wine type data available.">
          <ResponsiveContainer width="100%" height={H}>
            <PieChart>
              <Pie data={wineTypePieData} cx="50%" cy="50%" innerRadius="35%" outerRadius="60%" dataKey="value" paddingAngle={2}>
                {wineTypePieData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
              </Pie>
              <Tooltip content={<ChartTooltip hideName />} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Country Distribution" description="Top 8 countries by bottle count." isEmpty={countryData.length === 0} emptyMessage="No country data available.">
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={countryData} margin={{ top: 8, right: 16, bottom: 36, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-30} textAnchor="end" interval={0} />
              <YAxis tick={TICK} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" fill={CHART_ACCENT.tertiary} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 2 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Vintage Distribution" description="Bottles per vintage year — colours deepen with age." isEmpty={vintageData.length === 0} emptyMessage="No vintage data available.">
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={vintageData} margin={{ top: 8, right: 8, bottom: 36, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-45} textAnchor="end" interval={Math.max(0, Math.floor(vintageData.length / 8) - 1)} />
              <YAxis tick={TICK} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" radius={[3, 3, 0, 0]}>
                {vintageData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Rating Distribution" description="How your rated wines spread across quality tiers." isEmpty={ratingData.length === 0} emptyMessage="No rated wines in your cellar.">
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={ratingData} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: yW(ratingData, 120) }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={yW(ratingData, 118)} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Wines" radius={[0, 3, 3, 0]}>
                {ratingData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 3 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Drinking Window Status" description="Which of your wines are ready to open now?" isEmpty={dwData.length === 0} emptyMessage="No drinking window data available.">
          <ResponsiveContainer width="100%" height={H}>
            <PieChart>
              <Pie data={dwData} cx="50%" cy="50%" innerRadius="35%" outerRadius="60%" dataKey="value" paddingAngle={2}>
                {dwData.map((e, i) => <Cell key={i} fill={e.fill} stroke="none" />)}
              </Pie>
              <Tooltip content={<ChartTooltip hideName />} />
              <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Wine Age Analysis" description="Bottle count grouped by how many years old each wine is." isEmpty={ageData.length === 0} emptyMessage="No vintage data for age analysis.">
          <ResponsiveContainer width="100%" height={H}>
            <BarChart data={ageData} margin={{ top: 8, right: 8, bottom: 32, left: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
              <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-20} textAnchor="end" />
              <YAxis tick={TICK} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" radius={[3, 3, 0, 0]}>
                {ageData.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </div>

      {/* Row 4 */}
      <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
        <ChartCard title="Top Varietals" description="Your most collected grape varieties." isEmpty={varietalData.length === 0} emptyMessage="No varietal data available.">
          <ResponsiveContainer width="100%" height={Math.max(H, varietalData.length * 28 + 48)}>
            <BarChart data={varietalData} layout="vertical" margin={{ top: 8, right: 40, bottom: 8, left: yW(varietalData) }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={yW(varietalData) - 2} />
              <Tooltip content={<ChartTooltip hideName />} />
              <Bar dataKey="value" name="Bottles" fill={CHART_ACCENT.tertiary} radius={[0, 3, 3, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard title="Top Regions" description="Your most collected wine regions." isEmpty={regionData.length === 0} emptyMessage="No region data available.">
          <ResponsiveContainer width="100%" height={Math.max(H, regionData.length * 28 + 48)}>
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

      {/* Row 5: Cellar Size Over Time (full width) */}
      <ChartCard title="Cellar Size Over Time" description="Cumulative bottle count per month based on CellarTracker purchase history." isEmpty={cellarData.length === 0} emptyMessage="No CellarTracker purchase data available.">
        <ResponsiveContainer width="100%" height={H}>
          <BarChart data={cellarData} margin={{ top: 8, right: 16, bottom: 40, left: 16 }}>
            <CartesianGrid strokeDasharray="3 3" stroke={GRID} vertical={false} />
            <XAxis dataKey="name" tick={{ ...TICK, fontSize: 10 }} angle={-45} textAnchor="end" interval={cellarInterval} />
            <YAxis tick={TICK} />
            <Tooltip content={<ChartTooltip hideName />} />
            <Bar dataKey="value" name="Total Bottles" fill={CHART_ACCENT.secondary} radius={[3, 3, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </ChartCard>

      {/* Row 6: Top Rated Wines (full width, conditional) */}
      {topRated.length > 0 && (
        <ChartCard title="Top Rated Wines" description="Your highest-rated wines by personal score.">
          <ResponsiveContainer width="100%" height={Math.max(H, topRated.length * 30 + 48)}>
            <BarChart data={topRated} layout="vertical" margin={{ top: 8, right: 56, bottom: 8, left: yW(topRated, 240) }}>
              <CartesianGrid strokeDasharray="3 3" stroke={GRID} horizontal={false} />
              <XAxis type="number" domain={[70, 100]} tick={TICK} />
              <YAxis type="category" dataKey="name" tick={TICK} width={yW(topRated, 238)} />
              <Tooltip content={<ChartTooltip formatter={(v) => `${v}/100`} hideName />} />
              <Bar dataKey="value" name="Rating" radius={[0, 3, 3, 0]}>
                {topRated.map((e, i) => <Cell key={i} fill={e.fill} />)}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      )}
    </div>
  );
}
