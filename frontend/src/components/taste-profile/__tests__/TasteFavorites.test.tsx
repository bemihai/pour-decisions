import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";

import TasteFavorites, { type TasteFavoritesProps } from "../TasteFavorites";

const stats = {
  wines_tasted: 4, avg_rating: 91.5, highest_rating: 96, best_wine_id: 42,
  best_wine_name: "Favorite cuvee", best_wine_vintage: 2018, best_wine_producer_name: "Antinori",
};
const empty: TasteFavoritesProps = {
  producers: { producers: [] }, regions: { regions: [] }, countries: { countries: [] },
  vintages: { vintages: [] }, appellations: { appellations: [] },
};
const populated: TasteFavoritesProps = {
  producers: { producers: ["Antinori", "Producer Two", "Producer Three", "Producer Four", "Producer Five"].map(
    (producer_name) => ({ ...stats, producer_name, country: "Italy" }),
  ) },
  regions: { regions: [{ ...stats, region_name: "Tuscany", country: "Italy" }] },
  countries: { countries: [{ ...stats, country: "Italy" }] },
  vintages: { vintages: [{ ...stats, vintage: 2018 }] },
  appellations: { appellations: [{ ...stats, appellation: "Chianti", country: "Italy" }] },
};

describe("TasteFavorites", () => {
  it("builds a deterministic summary and features the top producer's rated wine", () => {
    render(<TasteFavorites {...populated} />);
    expect(screen.getByText(/Your most explored producer is Antinori/)).toHaveTextContent(
      "Your most explored producer is Antinori. You return most often to wines from Tuscany. Your strongest repeated vintage is 2018, averaging 91.5 across 4 wines tasted.",
    );
    expect(screen.getByText("Based on wines marked as consumed and your personal ratings.")).toBeVisible();
    expect(screen.getByText("Your highest-rated wine from your most explored producer.")).toBeVisible();
    expect(screen.getByText("Personal rating")).toBeVisible();
    expect(screen.getAllByRole("link", { name: "Favorite cuvee (2018)" })).toHaveLength(4);
    for (const link of screen.getAllByRole("link", { name: "Favorite cuvee (2018)" })) {
      expect(link).toHaveAttribute("href", "/cellar/42");
    }
  });

  it("expands three entries to five and collapses without navigation", async () => {
    const user = userEvent.setup();
    render(<TasteFavorites {...populated} />);
    const ranking = screen.getByRole("region", { name: "Producers ranking" });
    expect(within(ranking).getAllByRole("listitem")).toHaveLength(3);
    const expand = screen.getByRole("button", { name: "View all producers" });
    expect(expand).toHaveAttribute("aria-expanded", "false");
    await user.click(expand);
    expect(within(ranking).getAllByRole("listitem")).toHaveLength(5);
    expect(screen.getByRole("button", { name: "Show fewer" })).toHaveAttribute("aria-expanded", "true");
    await user.click(screen.getByRole("button", { name: "Show fewer" }));
    expect(within(ranking).getAllByRole("listitem")).toHaveLength(3);
  });

  it("shows one category, exposes selection, and resets expansion when switching", async () => {
    const user = userEvent.setup();
    render(<TasteFavorites {...populated} />);
    await user.click(screen.getByRole("button", { name: "View all producers" }));
    for (const category of ["Regions", "Countries", "Vintages", "Appellations", "Producers"]) {
      await user.click(screen.getByRole("button", { name: category, exact: true }));
      expect(screen.getByRole("button", { name: category, exact: true })).toHaveAttribute("aria-pressed", "true");
      expect(screen.getAllByRole("region")).toHaveLength(1);
      expect(screen.getByRole("region", { name: `${category} ranking` })).toBeVisible();
    }
    expect(screen.getAllByRole("listitem")).toHaveLength(3);
    expect(screen.getByRole("button", { name: "Regions", exact: true })).toHaveAttribute("aria-pressed", "false");
  });

  it("supports keyboard category selection and calculation disclosure", async () => {
    const user = userEvent.setup();
    render(<TasteFavorites {...empty} />);
    await user.tab();
    expect(screen.getByRole("button", { name: "Producers", exact: true })).toHaveFocus();
    await user.tab();
    await user.keyboard("{Enter}");
    expect(screen.getByRole("button", { name: "Regions", exact: true })).toHaveAttribute("aria-pressed", "true");
    for (let i = 0; i < 4; i++) await user.tab();
    const disclosure = screen.getByRole("button", { name: "How favorites are calculated" });
    expect(disclosure).toHaveFocus();
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
    expect(screen.getByText(/Vintages rank by average/)).not.toBeVisible();
    await user.keyboard(" ");
    expect(disclosure).toHaveAttribute("aria-expanded", "true");
    expect(screen.getByText(/Vintages rank by average/)).toHaveTextContent("require at least two consumed wines");
    expect(screen.getByText(/Producers, regions, countries, and appellations rank/)).toBeVisible();
    await user.keyboard("{Enter}");
    expect(disclosure).toHaveAttribute("aria-expanded", "false");
  });

  it("provides category-specific empty guidance and no fabricated featured wine", async () => {
    const user = userEvent.setup();
    render(<TasteFavorites {...empty} />);
    expect(screen.getByText(/Your taste story will grow/)).toBeVisible();
    expect(screen.getByText(/No rated favorite to feature yet/)).toBeVisible();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    for (const category of ["Producers", "Regions", "Countries", "Vintages", "Appellations"]) {
      await user.click(screen.getByRole("button", { name: category, exact: true }));
      expect(screen.getByRole("heading", { name: `No ${category.toLowerCase()} yet` })).toBeVisible();
      expect(within(screen.getByRole("region")).getByText(/Taste /)).toBeVisible();
    }
  });

  it("summarizes only available facts and never calls an unrated vintage strongest", () => {
    render(<TasteFavorites {...empty} regions={populated.regions}
      vintages={{ vintages: [{ ...stats, vintage: 2020, avg_rating: null, highest_rating: null }] }} />);
    expect(screen.getByText("You return most often to wines from Tuscany.")).toBeVisible();
    expect(screen.queryByText(/Your most explored producer is/)).not.toBeInTheDocument();
    expect(screen.queryByText(/strongest repeated vintage/)).not.toBeInTheDocument();
  });

  it.each([null, "Known wine"])("does not feature unrated wine %s", (name) => {
    render(<TasteFavorites {...empty} producers={{ producers: [{ ...populated.producers.producers[0],
      avg_rating: null, highest_rating: null, best_wine_id: null, best_wine_name: name,
    }] }} />);
    expect(screen.getByText(/No rated favorite to feature yet/)).toBeVisible();
    expect(screen.getAllByText("Unrated")).toHaveLength(2);
  });

  it("renders known names without inventing a vintage or link", () => {
    render(<TasteFavorites {...empty} producers={{ producers: [{ ...populated.producers.producers[0],
      best_wine_id: null, best_wine_vintage: null, best_wine_producer_name: null,
    }] }} />);
    expect(screen.getAllByText("Favorite cuvee")).toHaveLength(2);
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
    expect(screen.queryByText(/Favorite cuvee \(/)).not.toBeInTheDocument();
  });

  it("keeps a bounded featured empty state when rated wine details are missing", () => {
    render(<TasteFavorites {...empty} producers={{ producers: [{ ...populated.producers.producers[0], best_wine_name: null }] }} />);
    expect(screen.getByText(/No rated favorite to feature yet/)).toBeVisible();
    expect(screen.getByRole("link", { name: "Wine details unavailable" })).toHaveAttribute("href", "/cellar/42");
  });
});
