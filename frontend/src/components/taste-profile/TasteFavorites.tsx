/**
 * TasteFavorites component — Step 3.5.
 *
 * Displays ranked lists of top producers, regions, countries, vintages, and
 * appellations from consumed wines.  Pure display component (no hooks needed).
 * Replaces show_producer_loyalty(), show_favorite_regions(),
 * show_favorite_countries(), show_favorite_vintages(), and
 * show_favorite_appellations() from src/ui/helper/taste_profile_stats.py.
 */
import type {
  AppellationsResponse,
  CountriesResponse,
  ProducersResponse,
  RegionsResponse,
  VintagesResponse,
} from "@/lib/types";
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
}

function RankItem({
  rank,
  name,
  subtitle,
  winesTasted,
  avgRating,
  highestRating,
}: RankItemProps) {
  return (
    <div className="flex items-start gap-3 border-b py-3 last:border-0 last:pb-0">
      {/* Rank badge — branded burgundy */}
      <div className="flex size-7 shrink-0 items-center justify-center rounded-full bg-brand-burgundy/10 type-caption font-bold text-brand-burgundy">
        {rank}
      </div>

      {/* Name + subtitle */}
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium">{name}</p>
        {subtitle && (
          <p className="type-caption text-muted-foreground">{subtitle}</p>
        )}
        <p className="type-caption text-muted-foreground">
          {winesTasted} wine{winesTasted !== 1 ? "s" : ""}
        </p>
      </div>

      {/* Rating */}
      {avgRating != null ? (
        <div className="flex shrink-0 flex-col items-end gap-0.5 text-right">
          <Rating rating={avgRating} variant="compact" className="text-base" />
          {highestRating != null && (
            <span className="type-caption text-muted-foreground">
              best {Math.round(highestRating)}
            </span>
          )}
        </div>
      ) : (
        <span className="shrink-0 type-caption text-muted-foreground">Unrated</span>
      )}
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

export default function TasteFavorites({
  producers,
  regions,
  countries,
  vintages,
  appellations,
}: TasteFavoritesProps) {
  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
      {/* Producers */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Favorite Producers</CardTitle>
        </CardHeader>
        <CardContent>
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
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Regions */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Favorite Regions</CardTitle>
        </CardHeader>
        <CardContent>
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
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Countries */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Favorite Countries</CardTitle>
        </CardHeader>
        <CardContent>
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
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Vintages */}
      <Card>
        <CardHeader className="pb-2">
          <CardTitle className="text-base">Top Vintages</CardTitle>
        </CardHeader>
        <CardContent>
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
              />
            ))
          )}
        </CardContent>
      </Card>

      {/* Appellations — spans both columns if non-empty */}
      {appellations.appellations.length > 0 && (
        <Card className="md:col-span-2">
          <CardHeader className="pb-2">
            <CardTitle className="text-base">Favorite Appellations</CardTitle>
          </CardHeader>
          <CardContent className="md:columns-2 md:gap-8">
            {appellations.appellations.map((a, i) => (
              <div key={a.appellation} className="break-inside-avoid">
                <RankItem
                  rank={i + 1}
                  name={a.appellation}
                  subtitle={a.country ?? undefined}
                  winesTasted={a.wines_tasted}
                  avgRating={a.avg_rating}
                  highestRating={a.highest_rating}
                />
              </div>
            ))}
          </CardContent>
        </Card>
      )}
    </div>
  );
}

