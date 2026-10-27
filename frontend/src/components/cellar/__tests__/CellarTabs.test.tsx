import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import CellarTabs from "@/components/cellar/CellarTabs";
import type { CellarTabsProps } from "@/components/cellar/CellarTabs";

const replace = vi.fn();
let currentSearchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  usePathname: () => "/cellar",
  useRouter: () => ({ replace }),
  useSearchParams: () => currentSearchParams,
}));

vi.mock("@/components/cellar/CellarInventory", () => ({ default: () => <div>Inventory panel</div> }));
vi.mock("@/components/cellar/CellarDrinkNext", () => ({ default: () => <div>Drink Next panel</div> }));
vi.mock("@/components/cellar/CellarShoppingList", () => ({ default: () => <div>Shopping List panel</div> }));
vi.mock("@/components/cellar/CellarStatistics", () => ({ default: () => <div>Statistics panel</div> }));

const props: CellarTabsProps = {
  filterOptions: {
    wine_types: [], countries: [], locations: [], producers: [], min_vintage: 1900, max_vintage: 2026,
  },
  chartData: {
    wine_type_distribution: [{ unique_wines: 12 }],
    country_distribution: [],
    varietal_distribution: [],
    region_distribution: [],
    drinking_window_wines: {},
    cellar_size_over_time: [],
    top_rated: [],
    vintage_distribution: [],
    rating_distribution: [],
    wine_age_distribution: [],
  },
};

describe("CellarTabs", () => {
  beforeEach(() => {
    replace.mockClear();
    currentSearchParams = new URLSearchParams();
  });

  it("opens the URL-selected Shopping List panel lazily", () => {
    currentSearchParams = new URLSearchParams("tab=shopping-list");
    render(<CellarTabs {...props} />);

    expect(screen.getByRole("tab", { name: "Shopping List" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: "Shopping List" })).toHaveTextContent("Shopping List panel");
    expect(screen.queryByText("Drink Next panel")).not.toBeInTheDocument();
    expect(screen.queryByText("Statistics panel")).not.toBeInTheDocument();
  });

  it("preserves unrelated query parameters and keeps the default URL clean", async () => {
    currentSearchParams = new URLSearchParams("sort=name");
    const user = userEvent.setup();
    const view = render(<CellarTabs {...props} />);

    await user.click(screen.getByRole("tab", { name: "Shopping List" }));
    expect(replace).toHaveBeenLastCalledWith("/cellar?sort=name&tab=shopping-list", { scroll: false });

    currentSearchParams = new URLSearchParams("sort=name&tab=shopping-list");
    view.rerender(<CellarTabs {...props} />);
    await user.click(screen.getByRole("tab", { name: /Inventory/ }));
    expect(replace).toHaveBeenLastCalledWith("/cellar?sort=name", { scroll: false });
  });

  it("supports arrow, Home, and End keyboard navigation with roving focus", async () => {
    const user = userEvent.setup();
    render(<CellarTabs {...props} />);
    const inventory = screen.getByRole("tab", { name: /Inventory/ });
    inventory.focus();

    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Drink Next" })).toHaveFocus();
    expect(replace).toHaveBeenLastCalledWith("/cellar?tab=drink-next", { scroll: false });

    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Stats" })).toHaveFocus();
    expect(replace).toHaveBeenLastCalledWith("/cellar?tab=statistics", { scroll: false });

    await user.keyboard("{Home}");
    expect(inventory).toHaveFocus();
    expect(replace).toHaveBeenLastCalledWith("/cellar", { scroll: false });
  });

  it("falls back to Inventory for an unknown tab value", () => {
    currentSearchParams = new URLSearchParams("tab=unknown");
    render(<CellarTabs {...props} />);

    expect(screen.getByRole("tab", { name: /Inventory/ })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: /Inventory/ })).toBeVisible();
    expect(screen.getByRole("tablist", { name: "Cellar views" })).not.toHaveClass("overflow-x-auto");
  });
});
