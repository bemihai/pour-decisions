/**
 * Wine Detail page — /cellar/[wineId]
 *
 * Server Component. Fetches full wine data from GET /api/wines/:id and renders
 * a comprehensive detail view with: wine identity header, details, drinking
 * readiness, descriptions, tasting notes, community stats, and bottle inventory.
 *
 * force-dynamic: wine data changes whenever bottles are synced or descriptions
 * are generated, so Next.js must never cache this route.
 */

import type { Metadata } from "next";
import { notFound } from "next/navigation";
import Link from "next/link";
import { ArrowLeft, MapPin } from "lucide-react";

import type { WineDetailResponse } from "@/lib/types";
import { getWine } from "@/lib/api";
import { cn, formatCurrency, ratingColor } from "@/lib/utils";
import { getWineBottleIllustration, getWineTypeColors } from "@/lib/design-tokens";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import Breadcrumbs from "@/components/Breadcrumbs";
import DrinkingIndex from "@/components/DrinkingIndex";
import Rating from "@/components/Rating";
import TastingNote from "@/components/TastingNote";
import WineDescriptionGenerator from "@/components/cellar/WineDescriptionGenerator";

export const dynamic = "force-dynamic";

// ---------------------------------------------------------------------------
// Metadata
// ---------------------------------------------------------------------------

interface PageProps {
  params: Promise<{ wineId: string }>;
}

export async function generateMetadata({ params }: PageProps): Promise<Metadata> {
  try {
    const { wineId } = await params;
    const wine = await getWine(parseInt(wineId));
    const title = wine.vintage ? `${wine.wine_name} ${wine.vintage}` : wine.wine_name;
    return {
      title: `${title} | Wine Cellar | Pour Decisions`,
      description:
        wine.description ??
        `${wine.wine_type} wine${wine.producer_name ? ` by ${wine.producer_name}` : ""}`,
    };
  } catch {
    return { title: "Wine Detail | Pour Decisions" };
  }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const DRINK_SOURCE_LABEL: Record<string, string> = {
  cellar_tracker: "CellarTracker",
  manual: "Manual",
  llm: "AI estimate",
  heuristic: "Estimate",
};

/** Single labelled field in the Details card. */
function DetailField({
  label,
  value,
}: {
  label: string;
  value: string | null | undefined;
}) {
  if (!value) return null;
  return (
    <div className="flex flex-col gap-0.5">
      <span className="type-caption text-muted-foreground">{label}</span>
      <span className="text-sm text-foreground">{value}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

export default async function WineDetailPage({ params }: PageProps) {
  const { wineId } = await params;
  const id = parseInt(wineId);
  if (isNaN(id)) notFound();

  let wine: WineDetailResponse;
  try {
    wine = await getWine(id);
  } catch {
    notFound();
  }

  const colors = getWineTypeColors(wine.wine_type);
  const bottleIllustration = getWineBottleIllustration(wine.wine_type);
  const origin = [wine.region_name, wine.country].filter(Boolean).join(", ");

  const breadcrumbs = [
    { label: "Cellar", href: "/cellar" },
    ...(wine.producer_name
      ? [
          {
            label: wine.producer_name,
            href: `/cellar?producer=${encodeURIComponent(wine.producer_name)}`,
          },
        ]
      : []),
    {
      label: wine.vintage ? `${wine.wine_name} ${wine.vintage}` : wine.wine_name,
    },
  ];

  // DrinkingIndex accepts allIndices for percentile normalisation. On the
  // detail page only one drink_index is available, so the label reflects
  // the absolute score rather than a relative percentile rank.
  const allIndices = wine.drink_index != null ? [wine.drink_index] : [];

  const drinkWindowSource = wine.drink_window_source
    ? (DRINK_SOURCE_LABEL[wine.drink_window_source] ?? wine.drink_window_source)
    : null;

  // Community cellar percentages
  const communityPctHeld =
    wine.q_purchased > 0
      ? Math.min((wine.q_quantity / wine.q_purchased) * 100, 100)
      : null;
  const communityPctConsumed =
    wine.q_purchased > 0
      ? Math.min((wine.q_consumed / wine.q_purchased) * 100, 100)
      : null;

  return (
    <div className="container mx-auto max-w-7xl px-4 py-6">
      {/* Breadcrumbs */}
      <Breadcrumbs items={breadcrumbs} className="mb-4" />

      {/* ------------------------------------------------------------------ */}
      {/* Wine header                                                          */}
      {/* ------------------------------------------------------------------ */}
      <div className="mb-8 overflow-hidden rounded-xl border border-border bg-card shadow-sm">
        <div className="flex">
          {/* Wine-type colour stripe */}
          <div
            className="w-2 shrink-0 self-stretch rounded-l-xl"
            style={{ backgroundColor: colors.hex }}
            aria-hidden="true"
          />

          <div className="flex flex-1 flex-wrap items-start gap-6 p-6">
            {/* Bottle illustration */}
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img
              src={bottleIllustration}
              alt=""
              aria-hidden="true"
              className="hidden sm:block h-28 w-auto shrink-0 select-none object-contain"
            />

            {/* Wine identity */}
            <div className="flex-1 min-w-0">
              {wine.producer_name && (
                <p className="type-label text-muted-foreground leading-tight">{wine.producer_name}</p>
              )}
              <h1 className="type-page-title text-foreground leading-tight">
                {wine.wine_name}
                {wine.vintage && (
                  <span className="ml-3 text-2xl font-normal text-muted-foreground">
                    {wine.vintage}
                  </span>
                )}
              </h1>
              {origin && <p className="type-body text-muted-foreground mt-1">{origin}</p>}
              <div className="mt-3 flex flex-wrap items-center gap-2">
                {wine.wine_type && (
                  <Badge className={cn("font-medium", colors.badge)}>{wine.wine_type}</Badge>
                )}
                {wine.varietal && <Badge variant="outline">{wine.varietal}</Badge>}
                {wine.appellation && wine.appellation !== wine.region_name && (
                  <Badge variant="outline">{wine.appellation}</Badge>
                )}
                {wine.owned_quantity > 0 && (
                  <Badge variant="secondary">
                    {wine.owned_quantity} bottle{wine.owned_quantity !== 1 ? "s" : ""} in cellar
                  </Badge>
                )}
              </div>
            </div>

            {/* Rating + back link */}
            <div className="flex shrink-0 flex-col items-end gap-4">
              <Rating rating={wine.personal_rating} variant="full" />
              <Button asChild variant="outline" size="sm">
                <Link href="/cellar">
                  <ArrowLeft className="mr-1.5 size-4" aria-hidden="true" />
                  Back to Cellar
                </Link>
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* ------------------------------------------------------------------ */}
      {/* Main content — 2-col on desktop (left 2/3, right 1/3)              */}
      {/* ------------------------------------------------------------------ */}
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        {/* ---------------------------------------------------------------- */}
        {/* Left column                                                       */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-6 lg:col-span-2">
          {/* Wine Details */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Wine Details</CardTitle>
            </CardHeader>
            <CardContent className="grid grid-cols-2 gap-x-8 gap-y-3 sm:grid-cols-3">
              <DetailField label="Type" value={wine.wine_type} />
              <DetailField label="Varietal" value={wine.varietal} />
              <DetailField
                label="Vintage"
                value={wine.vintage != null ? String(wine.vintage) : null}
              />
              <DetailField label="Region" value={wine.region_name} />
              <DetailField label="Country" value={wine.country} />
              <DetailField label="Appellation" value={wine.appellation} />
              <DetailField label="Designation" value={wine.designation} />
              <DetailField label="Vineyard" value={wine.vineyard} />
              <DetailField label="Bottle Size" value={wine.bottle_size !== "750ml" ? wine.bottle_size : null} />
              {(wine.drink_from_year || wine.drink_to_year) && (
                <DetailField
                  label={`Drink Window${drinkWindowSource ? ` (${drinkWindowSource})` : ""}`}
                  value={`${wine.drink_from_year ?? "Now"} – ${wine.drink_to_year ?? "∞"}`}
                />
              )}
              {wine.last_tasted_date && (
                <DetailField label="Last Tasted" value={wine.last_tasted_date} />
              )}
            </CardContent>
          </Card>

          {/* Drinking Readiness */}
          {wine.drink_index != null && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Drinking Readiness</CardTitle>
              </CardHeader>
              <CardContent>
                <DrinkingIndex drinkIndex={wine.drink_index} allIndices={allIndices} />
              </CardContent>
            </Card>
          )}

          {/* About this Wine */}
          {(wine.producer_description || true) && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">About this Wine</CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-4">
                {wine.producer_description && (
                  <>
                    <div>
                      <p className="type-label mb-1.5 font-semibold uppercase tracking-wide text-muted-foreground">
                        About the Producer
                      </p>
                      <p className="text-sm leading-relaxed italic text-muted-foreground">
                        {wine.producer_description}
                      </p>
                    </div>
                    <Separator />
                  </>
                )}
                <div>
                  <p className="type-label mb-1.5 font-semibold uppercase tracking-wide text-muted-foreground">
                    Wine Description
                  </p>
                  {wine.description ? (
                    <p className="text-sm leading-relaxed text-muted-foreground">{wine.description}</p>
                  ) : (
                    <WineDescriptionGenerator wineId={wine.id} />
                  )}
                </div>
              </CardContent>
            </Card>
          )}

          {/* Personal Tasting Notes */}
          {wine.tasting_notes && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="text-base">Personal Tasting Notes</CardTitle>
              </CardHeader>
              <CardContent>
                <TastingNote notes={wine.tasting_notes} />
                {wine.last_tasted_date && (
                  <p className="mt-2 type-caption text-muted-foreground">
                    Last tasted: {wine.last_tasted_date}
                  </p>
                )}
              </CardContent>
            </Card>
          )}
        </div>

        {/* ---------------------------------------------------------------- */}
        {/* Right column                                                      */}
        {/* ---------------------------------------------------------------- */}
        <div className="flex flex-col gap-6">
          {/* Community Stats */}
          <Card>
            <CardHeader className="pb-3">
              <CardTitle className="text-base">Community Stats</CardTitle>
            </CardHeader>
            <CardContent className="flex flex-col gap-4">
              <div>
                <p className="type-caption text-muted-foreground">Community Rating</p>
                <p className={cn("text-2xl font-bold tabular-nums", ratingColor(wine.community_rating))}>
                  {wine.community_rating != null ? `${Math.round(wine.community_rating)}/100` : "—"}
                </p>
              </div>

              {wine.q_purchased > 0 && communityPctHeld != null && communityPctConsumed != null && (
                <div>
                  <p className="type-caption text-muted-foreground mb-1.5">CellarTracker Community</p>
                  <div className="flex items-center gap-2 mb-1">
                    <div className="relative h-3 w-full max-w-[120px] overflow-hidden rounded-full bg-muted">
                      <div
                        className="absolute inset-y-0 left-0 bg-brand-burgundy"
                        style={{ width: `${communityPctHeld}%` }}
                      />
                      <div
                        className="absolute inset-y-0 bg-brand-gold"
                        style={{
                          left: `${communityPctHeld}%`,
                          width: `${communityPctConsumed}%`,
                        }}
                      />
                    </div>
                    <span className="type-caption text-muted-foreground">
                      {communityPctHeld.toFixed(0)}% held
                    </span>
                  </div>
                  <p className="type-caption text-muted-foreground">
                    {wine.q_purchased.toLocaleString()} purchased ·{" "}
                    {wine.q_quantity.toLocaleString()} in cellars ·{" "}
                    {wine.q_consumed.toLocaleString()} consumed
                  </p>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Your Bottles */}
          {wine.bottles.length > 0 && (
            <Card>
              <CardHeader className="pb-3">
                <CardTitle className="flex items-center gap-2 text-base">
                  Your Bottles
                  <Badge variant="secondary" className="tabular-nums">
                    {wine.owned_quantity}
                  </Badge>
                </CardTitle>
              </CardHeader>
              <CardContent className="flex flex-col gap-2">
                {wine.bottles.map((bottle) => (
                  <div
                    key={bottle.id}
                    className="flex flex-col gap-1 rounded-lg border border-border bg-muted/20 p-3"
                  >
                    <div className="flex items-center justify-between">
                      <Badge variant="outline" className="capitalize text-xs">
                        {bottle.status}
                      </Badge>
                      <span className="type-caption text-muted-foreground">
                        qty {bottle.quantity}
                      </span>
                    </div>
                    {(bottle.location || bottle.bin) && (
                      <p className="flex items-center gap-1 type-caption text-muted-foreground">
                        <MapPin className="size-3 shrink-0" aria-hidden="true" />
                        {[bottle.location, bottle.bin ? `Bin ${bottle.bin}` : null]
                          .filter(Boolean)
                          .join(" · ")}
                      </p>
                    )}
                    {bottle.purchase_price != null && (
                      <p className="type-caption text-muted-foreground">
                        {formatCurrency(bottle.purchase_price, bottle.currency)}
                        {bottle.store_name ? ` · ${bottle.store_name}` : ""}
                        {bottle.purchase_date ? ` · ${bottle.purchase_date}` : ""}
                      </p>
                    )}
                    {bottle.consumed_date && (
                      <p className="type-caption text-muted-foreground">
                        Consumed {bottle.consumed_date}
                      </p>
                    )}
                    {bottle.bottle_note && (
                      <p className="type-caption italic text-foreground/70">{bottle.bottle_note}</p>
                    )}
                  </div>
                ))}
              </CardContent>
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

