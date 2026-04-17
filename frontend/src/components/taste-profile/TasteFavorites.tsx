/**
 * TasteFavorites component — Step 3.5.
 *
 * Displays ranked lists of top producers, regions, countries, vintages, and
 * appellations from consumed wines.  Pure display component (no hooks needed).
 * Replaces show_producer_loyalty(), show_favorite_regions(),
 * show_favorite_countries(), show_favorite_vintages(), and
 * show_favorite_appellations() from src/ui/helper/taste_profile_stats.py.
 */
import Link from "next/link";
import { Award, Star, Wine } from "lucide-react";

import type {
  AppellationsResponse,
  CountriesResponse,
  ProducersResponse,
  RegionsResponse,
  VintagesResponse,
} from "@/lib/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import EmptyState from "@/components/EmptyState";
import Rating from "@/components/Rating";

// ---------------------------------------------------------------------------
// Props
// ---------------------------------------------------------------------------

export interface TasteFavoritesProps {
  producers: ProducersResponse;
  regions: RegionsResponse;
  countries: CountriesResponse;
  vintages: VintagesResponse;
  appellations: AppellationsResponse;
}

// ---------------------------------------------------------------------------
// RankItem — private row component used by every ranked list
// ---------------------------------------------------------------------------

interface RankItemProps {
  rank: number;
  name: string;
  subtitle?: string | null;
  winesTasted: number;
  avgRating?: number | null;
  highestRating?: number | null;
  bestWineId?: number | null;
}

function RankItem({
  rank,
  name,
  subtitle,
  winesTasted,
  avgRating,
  highestRating,
  bestWineId,
}: RankItemProps) {
  return (
    <div className="group rounded-xl border border-border/70 bg-card/60 p-3 transition-all hover:border-brand-burgundy/30 hover:bg-card hover:shadow-sm">
      <div className="mb-2.5 flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <div className="mb-1.5 flex items-center gap-2">
            <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-brand-burgundy/10 text-xs font-bold text-brand-burgundy">
              {rank}
            </div>
            <p className="truncate text-sm font-semibold text-foreground">{name}</p>
          </div>
          {subtitle && (
            <p className="truncate text-xs text-muted-foreground">{subtitle}</p>
          )}
        </div>

        {avgRating != null ? (
          <div className="shrink-0 text-right">
            <Rating rating={avgRating} variant="compact" className="text-base" />
          </div>
        ) : (
          <Badge variant="outline" className="shrink-0 text-[11px] text-muted-foreground">
            Unrated
          </Badge>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-2">
        <Badge variant="secondary" className="h-6 gap-1.5 text-[11px] font-medium">
          <Wine className="size-3" />
          {winesTasted} wine{winesTasted !== 1 ? "s" : ""}
        </Badge>
        {highestRating != null && (
          bestWineId ? (
            <Link href={`/cellar/${bestWineId}`} className="inline-flex">
              <Badge variant="outline" className="h-6 gap-1.5 text-[11px] hover:border-brand-burgundy/40 hover:text-brand-burgundy transition-colors cursor-pointer">
                <Award className="size-3 text-amber-500" />
                Best {Math.round(highestRating)}
              </Badge>
            </Link>
          ) : (
            <Badge variant="outline" className="h-6 gap-1.5 text-[11px]">
              <Award className="size-3 text-amber-500" />
              Best {Math.round(highestRating)}
            </Badge>
          )
        )}
        {avgRating != null && (
          <Badge variant="outline" className="h-6 gap-1.5 text-[11px]">
            <Star className="size-3 text-brand-burgundy" />
            Avg {avgRating.toFixed(1)}
          </Badge>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// EmptyList — shown when a category has no data
// ---------------------------------------------------------------------------

function EmptyList() {
  return (
    <EmptyState title="No data available yet." className="py-8 border-0" />
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

const FAVORITE_CARD_CLASS = "h-[460px]";
const FAVORITE_CONTENT_CLASS = "space-y-2 overflow-y-auto pr-1";

export default function TasteFavorites({
  producers,
  regions,
  countries,
  vintages,
  appellations,
}: TasteFavoritesProps) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
      {/* Producers */}
      <Card className={FAVORITE_CARD_CLASS}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Producers</CardTitle>
        </CardHeader>
        <CardContent className={FAVORITE_CONTENT_CLASS}>
          {producers.producers.length === 0 ? (
            <EmptyList />
          ) : (
            producers.producers.map((p, i) => (
              <RankItem
                key={p.producer_name}
                rank={i + 1}
                name={p.producer_name}
                subtitle={p.country ?? undefined}
                winesTasted={p.wines_tasted}
                avgRating={p.avg_rating}
                highestRating={p.highest_rating}
                bestWineId={p.best_wine_id}
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Regions */}
      <Card className={FAVORITE_CARD_CLASS}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Regions</CardTitle>
        </CardHeader>
        <CardContent className={FAVORITE_CONTENT_CLASS}>
          {regions.regions.length === 0 ? (
            <EmptyList />
          ) : (
            regions.regions.map((r, i) => (
              <RankItem
                key={r.region_name}
                rank={i + 1}
                name={r.region_name}
                subtitle={r.country ?? undefined}
                winesTasted={r.wines_tasted}
                avgRating={r.avg_rating}
                highestRating={r.highest_rating}
                bestWineId={r.best_wine_id}
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Countries */}
      <Card className={FAVORITE_CARD_CLASS}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Countries</CardTitle>
        </CardHeader>
        <CardContent className={FAVORITE_CONTENT_CLASS}>
          {countries.countries.length === 0 ? (
            <EmptyList />
          ) : (
            countries.countries.map((c, i) => (
              <RankItem
                key={c.country}
                rank={i + 1}
                name={c.country}
                winesTasted={c.wines_tasted}
                avgRating={c.avg_rating}
                highestRating={c.highest_rating}
                bestWineId={c.best_wine_id}
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Vintages */}
      <Card className={FAVORITE_CARD_CLASS}>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Top Vintages</CardTitle>
        </CardHeader>
        <CardContent className={FAVORITE_CONTENT_CLASS}>
          {vintages.vintages.length === 0 ? (
            <EmptyList />
          ) : (
            vintages.vintages.map((v, i) => (
              <RankItem
                key={v.vintage}
                rank={i + 1}
                name={String(v.vintage)}
                winesTasted={v.wines_tasted}
                avgRating={v.avg_rating}
                highestRating={v.highest_rating}
                bestWineId={v.best_wine_id}
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Appellations */}
      {appellations.appellations.length > 0 && (
        <Card className={FAVORITE_CARD_CLASS}>
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Appellations</CardTitle>
          </CardHeader>
          <CardContent className={FAVORITE_CONTENT_CLASS}>
            {appellations.appellations.map((a, i) => (
              <RankItem
                key={a.appellation}
                rank={i + 1}
                name={a.appellation}
                subtitle={a.country ?? undefined}
                winesTasted={a.wines_tasted}
                avgRating={a.avg_rating}
                highestRating={a.highest_rating}
                bestWineId={a.best_wine_id}
              />
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}
