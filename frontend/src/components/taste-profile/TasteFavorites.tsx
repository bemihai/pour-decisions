"use client";

import { useId, useState } from "react";
import Link from "next/link";
import { Check, ChevronDown, Heart, Wine } from "lucide-react";

import type {
  AppellationsResponse,
  CountriesResponse,
  ProducerStats,
  ProducersResponse,
  RegionsResponse,
  VintagesResponse,
} from "@/lib/types";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

export interface TasteFavoritesProps {
  producers: ProducersResponse;
  regions: RegionsResponse;
  countries: CountriesResponse;
  vintages: VintagesResponse;
  appellations: AppellationsResponse;
}

type Favorite = Pick<ProducerStats,
  "wines_tasted" | "avg_rating" | "highest_rating" | "best_wine_id" |
  "best_wine_name" | "best_wine_vintage" | "best_wine_producer_name"
>;
type Category = "Producers" | "Regions" | "Countries" | "Vintages" | "Appellations";
const categories: Category[] = ["Producers", "Regions", "Countries", "Vintages", "Appellations"];
const guidance: Record<Category, string> = {
  Producers: "Taste wines with a recorded producer to discover your most explored producers.",
  Regions: "Taste wines with a recorded region to discover the places you return to.",
  Countries: "Taste wines with a recorded country to explore your country favorites.",
  Vintages: "Taste at least two wines from the same recorded vintage to compare your vintage favorites.",
  Appellations: "Taste wines with a recorded appellation to discover your appellation favorites.",
};

function BestWine({ favorite }: { favorite: Favorite }) {
  const label = favorite.best_wine_name
    ? `${favorite.best_wine_name}${favorite.best_wine_vintage != null ? ` (${favorite.best_wine_vintage})` : ""}`
    : "Wine details unavailable";
  return favorite.best_wine_id != null ? (
    <Link className="underline decoration-brand-burgundy/40 underline-offset-4 hover:text-brand-burgundy" href={`/cellar/${favorite.best_wine_id}`}>
      {label}
    </Link>
  ) : <span>{label}</span>;
}

export default function TasteFavorites({ producers, regions, countries, vintages, appellations }: TasteFavoritesProps) {
  const [category, setCategory] = useState<Category>("Producers");
  const [expanded, setExpanded] = useState(false);
  const [showCalculation, setShowCalculation] = useState(false);
  const id = useId();
  const producer = producers.producers[0];
  const region = regions.regions[0];
  const country = countries.countries[0];
  const vintage = vintages.vintages[0];
  const summary: string[] = [];
  if (producer) summary.push(`Your most explored producer is ${producer.producer_name}.`);
  if (region) summary.push(`You return most often to wines from ${region.region_name}.`);
  if (vintage?.avg_rating != null) {
    summary.push(`Your strongest repeated vintage is ${vintage.vintage}, averaging ${vintage.avg_rating.toFixed(1)} across ${vintage.wines_tasted} wines tasted.`);
  }
  const highlights = [
    { label: "Top producer", value: producer?.producer_name },
    { label: "Top region", value: region?.region_name },
    { label: "Top country", value: country?.country },
  ].filter((highlight) => highlight.value);
  const rows: (Favorite & { name: string; location?: string | null })[] = {
    Producers: producers.producers.map((item) => ({ ...item, name: item.producer_name, location: item.country })),
    Regions: regions.regions.map((item) => ({ ...item, name: item.region_name, location: item.country })),
    Countries: countries.countries.map((item) => ({ ...item, name: item.country })),
    Vintages: vintages.vintages.map((item) => ({ ...item, name: String(item.vintage) })),
    Appellations: appellations.appellations.map((item) => ({ ...item, name: item.appellation, location: item.country })),
  }[category].slice(0, 5);
  const featured = producer?.highest_rating != null && producer.best_wine_name ? producer : null;

  return (
    <div className="space-y-6">
      <Card className="border-brand-burgundy/10 bg-brand-burgundy/5">
        <CardContent className="grid gap-8 p-6 lg:grid-cols-2 lg:items-center">
          <div className="flex items-start gap-4">
            <Heart aria-hidden="true" className="mt-1 size-6 shrink-0 text-brand-burgundy" />
            <div className="min-w-0 space-y-3">
              <h2 className="font-heading text-xl font-semibold">Your taste at a glance</h2>
              <p className="break-words leading-relaxed">{summary.join(" ") || "Your taste story will grow as you record consumed wines and personal ratings."}</p>
              <p className="text-sm text-muted-foreground">Based on wines marked as consumed and your personal ratings.</p>
            </div>
          </div>
          {highlights.length > 0 && (
            <dl className="grid gap-6 sm:grid-cols-3">
              {highlights.map(({ label, value }) => (
                <div key={label} className="min-w-0 space-y-2 lg:border-l lg:border-brand-burgundy/15 lg:pl-5">
                  <dt className="text-sm text-muted-foreground">{label}</dt>
                  <dd className="break-words font-heading text-xl font-semibold">{value}</dd>
                </div>
              ))}
            </dl>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardContent className="flex flex-col gap-6 p-6 sm:flex-row sm:items-center">
          <Wine aria-hidden="true" className="size-8 shrink-0 text-brand-burgundy" />
          <div className="min-w-0 flex-1 space-y-2">
            <h2 className="text-sm font-semibold text-brand-burgundy">Featured favorite</h2>
            {featured ? (
              <>
                <p className="break-words font-heading text-2xl font-semibold"><BestWine favorite={featured} /></p>
                <p className="break-words text-muted-foreground">{featured.best_wine_producer_name ?? featured.producer_name}</p>
                <p className="text-sm text-muted-foreground">Your highest-rated wine from your most explored producer.</p>
              </>
            ) : (
              <p className="text-muted-foreground">No rated favorite to feature yet. A rated wine with available details from your top producer will appear here.</p>
            )}
          </div>
          {featured && (
            <div className="space-y-1 sm:border-l sm:pl-6">
              <p className="text-sm text-muted-foreground">Personal rating</p>
              <p className="text-3xl font-semibold tabular-nums text-brand-burgundy">{featured.highest_rating}<span className="text-sm font-normal text-muted-foreground"> / 100</span></p>
            </div>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader className="gap-5">
          <CardTitle className="font-heading text-xl">Browse your favorites</CardTitle>
          <div role="group" aria-label="Favorite categories" className="flex flex-wrap gap-2">
            {categories.map((choice) => (
              <Button key={choice} type="button" variant={category === choice ? "default" : "outline"}
                className="min-h-10 px-4" aria-pressed={category === choice} aria-controls={`${id}-rankings`}
                onClick={() => { setCategory(choice); setExpanded(false); }}>
                {category === choice && <Check aria-hidden="true" />}{choice}
              </Button>
            ))}
          </div>
        </CardHeader>
        <CardContent className="space-y-5">
          <section id={`${id}-rankings`} aria-label={`${category} ranking`}>
            {rows.length === 0 ? (
              <div className="space-y-2 py-6">
                <h3 className="font-semibold">No {category.toLowerCase()} yet</h3>
                <p className="text-muted-foreground">{guidance[category]}</p>
              </div>
            ) : (
              <ol className="space-y-4">
                {rows.slice(0, expanded ? 5 : 3).map((row, index) => (
                  <li key={row.name} className="rounded-xl border border-border/70 p-5">
                    <div className="flex items-start gap-4">
                      <span aria-hidden="true" className="font-heading text-2xl text-brand-burgundy">{index + 1}</span>
                      <div className="grid min-w-0 flex-1 gap-5 xl:grid-cols-[1fr_1.4fr_1.2fr] xl:items-start">
                        <div>
                          <h3 className="break-words font-heading text-lg font-semibold">{row.name}</h3>
                          {row.location && <p className="break-words text-sm text-muted-foreground">{row.location}</p>}
                        </div>
                        <dl className="grid gap-4 sm:grid-cols-3">
                          <div><dt className="text-sm text-muted-foreground">Wines tasted</dt><dd className="mt-1 font-medium">{row.wines_tasted}</dd></div>
                          <div><dt className="text-sm text-muted-foreground">Average rating</dt><dd className="mt-1 font-medium">{row.avg_rating != null ? row.avg_rating.toFixed(1) : "Unrated"}</dd></div>
                          <div><dt className="text-sm text-muted-foreground">Highest rating</dt><dd className="mt-1 font-medium">{row.highest_rating ?? "Unrated"}</dd></div>
                        </dl>
                        {(row.best_wine_name || row.best_wine_id != null) && (
                          <div className="break-words text-sm">
                            <p className="mb-1 text-muted-foreground">Best wine</p>
                            <BestWine favorite={row} />
                            {row.best_wine_producer_name && <p className="mt-1 text-muted-foreground">{row.best_wine_producer_name}</p>}
                          </div>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ol>
            )}
          </section>
          {rows.length > 3 && (
            <Button type="button" variant="link" aria-expanded={expanded} aria-controls={`${id}-rankings`} onClick={() => setExpanded(!expanded)}>
              {expanded ? "Show fewer" : `View all ${category.toLowerCase()}`}
            </Button>
          )}
        </CardContent>
      </Card>

      <div className="rounded-xl border border-border">
        <button type="button" className="flex w-full items-center justify-between gap-4 rounded-xl p-5 text-left font-medium focus-visible:outline-2 focus-visible:outline-ring"
          aria-expanded={showCalculation} aria-controls={`${id}-calculation`} onClick={() => setShowCalculation(!showCalculation)}>
          How favorites are calculated
          <ChevronDown aria-hidden="true" className={`size-5 shrink-0 transition-transform ${showCalculation ? "rotate-180" : ""}`} />
        </button>
        <div id={`${id}-calculation`} hidden={!showCalculation} className="space-y-3 px-5 pb-5 text-sm leading-relaxed text-muted-foreground">
          <p>Producers, regions, countries, and appellations rank by wines tasted, then average personal rating.</p>
          <p>Vintages rank by average personal rating, then wines tasted, and require at least two consumed wines.</p>
          <p>Only consumed wines contribute. Unrated wines count toward wines tasted, but not average or highest ratings. Best wines are selected by personal rating, then most recent consumption, then wine ID to break ties.</p>
        </div>
      </div>
    </div>
  );
}
