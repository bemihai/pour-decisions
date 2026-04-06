/**
 * TasteOverview component — Step 3.1.
 *
 * Key insight metrics for the taste profile page: average rating, wines tasted,
 * favorite wine type, and percentage of highly-rated (90+) wines.
 * Replaces show_taste_profile_overview() from src/ui/helper/taste_profile_stats.py.
 */
import type { TasteOverviewResponse } from "@/lib/types";
import { cn } from "@/lib/utils";
import MetricCard from "@/components/MetricCard";

interface TasteOverviewProps {
  stats: TasteOverviewResponse;
  className?: string;
}

export default function TasteOverview({ stats, className }: TasteOverviewProps) {
  const avgRatingDisplay =
    stats.avg_rating != null ? `${stats.avg_rating.toFixed(1)}/100` : "N/A";

  return (
    <div className={cn("grid grid-cols-2 gap-4 md:grid-cols-4", className)}>
      <MetricCard label="Average Rating" value={avgRatingDisplay} />
      <MetricCard label="Wines Tasted" value={stats.wines_rated.toLocaleString()} />
      <MetricCard label="Favorite Type" value={stats.favorite_type} />
      {/* Span 2 cols on mobile so the last card does not sit orphaned. */}
      <MetricCard
        label="Highly Rated"
        value={`${stats.highly_rated_pct.toFixed(0)}%`}
        delta={`${stats.highly_rated_count} wines rated 90+`}
        className="col-span-2 md:col-span-1"
      />
    </div>
  );
}

