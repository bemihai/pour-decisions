/**
 * CellarOverview component.
 *
 * Renders the five key cellar KPIs as MetricCards in a responsive grid.
 * Replaces show_cellar_metrics() from src/ui/helper/cellar_stats.py.
 *
 * Metrics displayed:
 *   1. Total Bottles      — with "N unique wines" delta
 *   2. Top Wine Type      — highest-bottle type as a percentage, with bottle count delta
 *   3. Ready to Drink     — from drinking_stats
 *   4. To Hold            — from drinking_stats
 *   5. Cellar Value       — primary currency total, with wines-tracked delta
 */

import type { CellarStatsResponse } from "@/lib/types";
import { cn, formatCurrency } from "@/lib/utils";
import MetricCard from "@/components/MetricCard";

interface CellarOverviewProps {
  stats: CellarStatsResponse;
  className?: string;
}

export default function CellarOverview({ stats, className }: CellarOverviewProps) {
  const { overview, drinking_stats, value_stats } = stats;

  // Top wine type — mirrors show_cellar_metrics() col2 logic.
  const topType = overview.by_type[0] ?? null;
  const topTypePct =
    topType && overview.total_bottles > 0
      ? `${((topType.bottles / overview.total_bottles) * 100).toFixed(1)}%`
      : "—";

  // Primary currency — first entry (highest value) from value_stats.by_currency.
  const primaryCurrency = value_stats.by_currency[0] ?? null;
  const cellarValue = primaryCurrency
    ? formatCurrency(primaryCurrency.total_value, primaryCurrency.currency)
    : "N/A";
  const cellarValueDelta = primaryCurrency
    ? `${primaryCurrency.wines_with_price} wines tracked`
    : undefined;

  return (
    <div className={cn("grid grid-cols-2 gap-4 md:grid-cols-5", className)}>
      <MetricCard
        label="Total Bottles"
        value={overview.total_bottles.toLocaleString()}
        delta={`${overview.unique_wines} unique wines`}
      />

      <MetricCard
        label={topType?.wine_type ?? "Top Type"}
        value={topTypePct}
        delta={topType ? `${topType.bottles} bottles` : undefined}
      />

      <MetricCard
        label="Ready to Drink"
        value={drinking_stats.ready_to_drink.toLocaleString()}
      />

      <MetricCard
        label="To Hold"
        value={drinking_stats.to_hold.toLocaleString()}
      />

      {/* On a 2-col mobile grid, span the last card across both columns so it
          doesn't sit awkwardly half-width. Resets to auto width at md+. */}
      <MetricCard
        label="Cellar Value"
        value={cellarValue}
        delta={cellarValueDelta}
        className="col-span-2 md:col-span-1"
      />
    </div>
  );
}

