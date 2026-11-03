import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import CellarOverview from "@/components/cellar/CellarOverview";
import type { CellarStatsResponse } from "@/lib/types";

const stats: CellarStatsResponse = {
  overview: {
    total_bottles: 72,
    unique_wines: 67,
    by_type: [{ wine_type: "Red", unique_wines: 59, bottles: 64 }],
    by_country: [],
  },
  drinking_stats: {
    ready_to_drink: 47,
    to_hold: 23,
    unknown: 2,
  },
  value_stats: {
    by_currency: [{ currency: "RON", total_value: 9374, wines_with_price: 67 }],
    bottles_without_price: 5,
  },
};

describe("CellarOverview", () => {
  it("renders every existing cellar metric in a labelled summary region", () => {
    render(<CellarOverview stats={stats} />);

    expect(screen.getByRole("region", { name: "Cellar summary" })).toBeInTheDocument();
    expect(screen.getByText("Total Bottles")).toBeInTheDocument();
    expect(screen.getByText("Red")).toBeInTheDocument();
    expect(screen.getByText("Ready to Drink")).toBeInTheDocument();
    expect(screen.getByText("To Hold")).toBeInTheDocument();
    expect(screen.getByText("Cellar Value")).toBeInTheDocument();
    expect(screen.getByText("65% of cellar")).toBeInTheDocument();
    expect(screen.getByText("For later")).toBeInTheDocument();
  });

  it("handles missing type and value data without inventing values", () => {
    render(
      <CellarOverview
        stats={{
          ...stats,
          overview: { ...stats.overview, total_bottles: 0, by_type: [] },
          value_stats: { by_currency: [], bottles_without_price: 0 },
        }}
      />,
    );

    expect(screen.getByText("Top Type")).toBeInTheDocument();
    expect(screen.getByText("N/A")).toBeInTheDocument();
    expect(screen.queryByText(/% of cellar/)).not.toBeInTheDocument();
  });
});
