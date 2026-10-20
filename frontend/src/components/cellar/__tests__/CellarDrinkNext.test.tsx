import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";

import CellarDrinkNext from "@/components/cellar/CellarDrinkNext";
import { getDrinkNext } from "@/lib/api";
import type { DrinkNextItem } from "@/lib/types";

vi.mock("@/lib/api", () => ({ getDrinkNext: vi.fn() }));

function wine(overrides: Partial<DrinkNextItem> = {}): DrinkNextItem {
  return {
    wine_id: 1, wine_name: "Test Cuvee", producer_name: "Test Estate",
    vintage: 2020, wine_type: "Red", varietal: "Pinot Noir",
    region_name: "Burgundy", country: "France", quantity: 3,
    q_quantity: 600, q_purchased: 1000, drink_index: 100,
    drink_status: "ready", drink_from_year: 2023, drink_to_year: 2030,
    personal_rating: 88, community_rating: 90.5, location: "Cellar",
    ...overrides,
  };
}

function renderWines(byType: Record<string, DrinkNextItem[]>) {
  vi.mocked(getDrinkNext).mockResolvedValue({
    by_type: byType,
    total_wines: Object.values(byType).flat().length,
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}><CellarDrinkNext /></QueryClientProvider>,
  );
}

afterEach(() => { cleanup(); vi.clearAllMocks(); });

describe("CellarDrinkNext community holdings", () => {
  it.each([
    [600, 1000, "60% · 600 / 1,000"],
    [1, 2, "50% · 1 / 2"],
    [0, 0, "— · 0 / 0"],
    [5, 0, "— · 5 / 0"],
    [0, 10, "0% · 0 / 10"],
    [1, 3, "33% · 1 / 3"],
    [2, 3, "67% · 2 / 3"],
    [1, 8, "13% · 1 / 8"],
    [12, 10, "100% · 12 / 10"],
  ])("renders %i held / %i purchased", async (held, purchased, expected) => {
    renderWines({ Red: [wine({ q_quantity: held, q_purchased: purchased })] });
    expect(await screen.findByRole("cell", { name: expected })).toBeInTheDocument();
    const row = screen.getByRole("link", { name: "Test Cuvee" }).closest("tr")!;
    expect(within(row).getAllByRole("cell")[3]).toHaveTextContent(/^3$/);
  });

  it.each([
    [undefined, undefined, "— · 0 / 0"],
    [undefined, 1000, "0% · 0 / 1,000"],
    [600, undefined, "— · 600 / 0"],
  ])("handles missing community fields (%s / %s)", async (held, purchased, expected) => {
    // JSON serialization omits undefined properties, like an older API response.
    const legacyWine = JSON.parse(JSON.stringify(wine({
      q_quantity: held,
      q_purchased: purchased,
    }))) as DrinkNextItem;
    renderWines({ Red: [legacyWine] });

    expect(await screen.findByRole("cell", { name: expected })).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Test Cuvee" })).toHaveAttribute("href", "/cellar/1");
    expect(screen.queryByText(/NaN/)).not.toBeInTheDocument();
  });

  it.each([
    [0, 100, "0%", "text-red-600"],
    [32, 100, "32%", "text-red-600"],
    [33, 100, "33%", "text-amber-600"],
    [66, 100, "66%", "text-amber-600"],
    [67, 100, "67%", "text-green-600"],
    [100, 100, "100%", "text-green-600"],
    [0, 0, "—", "text-muted-foreground"],
  ])("colors %i / %i holdings", async (held, purchased, percentage, color) => {
    renderWines({ Red: [wine({ q_quantity: held, q_purchased: purchased })] });
    expect(await screen.findByText(percentage)).toHaveClass(color);
  });

  it("preserves links, ratings, status, grouping and expansion with scrollable tables", async () => {
    const user = userEvent.setup();
    renderWines({
      White: [wine({ wine_id: 10, wine_name: "White wine", personal_rating: null })],
      Red: Array.from({ length: 6 }, (_, i) => wine({ wine_id: i + 1, wine_name: "Red " + (i + 1) })),
    });
    await screen.findByRole("link", { name: "Red 1" });
    expect(screen.getAllByRole("heading").map((heading) => heading.textContent)).toEqual(["Red6", "White1"]);
    expect(screen.queryByRole("columnheader", { name: "Region" })).not.toBeInTheDocument();
    expect(screen.queryByRole("columnheader", { name: "Location" })).not.toBeInTheDocument();
    const table = screen.getAllByRole("table")[0];
    expect(table.parentElement).toHaveClass("overflow-x-auto");
    const headers = within(table).getAllByRole("columnheader");
    expect(headers[3]).toHaveTextContent("Qty");
    expect(headers[4]).toHaveTextContent(/^Community holdings$/);
    expect(headers[4]).toHaveAccessibleDescription("Community held (including pending) / purchased bottles");
    expect(screen.getByRole("link", { name: "Red 1" })).toHaveAttribute("href", "/cellar/1");
    expect(within(table).getAllByText("88")).toHaveLength(5);
    expect(within(table).getAllByText("Ready to Drink")).toHaveLength(5);
    expect(screen.getByText("90.5")).toBeInTheDocument();
    expect(screen.queryByRole("link", { name: "Red 6" })).not.toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show 1 More" }));
    expect(screen.getByRole("link", { name: "Red 6" })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: "Show Less" }));
    expect(screen.queryByRole("link", { name: "Red 6" })).not.toBeInTheDocument();
    expect(screen.getByRole("link", { name: "White wine" })).toBeInTheDocument();
  });
});
