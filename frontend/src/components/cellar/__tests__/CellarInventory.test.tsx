import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import CellarInventory from "@/components/cellar/CellarInventory";
import { groupWinesByIdentity } from "@/components/WineGroup";
import type { InventoryItem } from "@/lib/types";

const state = vi.hoisted(() => ({ params: new URLSearchParams(), items: [] as InventoryItem[] }));
vi.mock("next/navigation", () => ({
  useSearchParams: () => state.params, usePathname: () => "/cellar", useRouter: () => ({ replace: vi.fn() }),
}));
vi.mock("@tanstack/react-query", () => ({ useQuery: () => ({ data: {
  items: state.items, total_wines: state.items.length, total_bottles: 9,
} }) }));
vi.mock("@tanstack/react-virtual", () => ({ useWindowVirtualizer: ({ count }: { count: number }) => ({
  getTotalSize: () => 500, measureElement: vi.fn(), options: { scrollMargin: 0 },
  getVirtualItems: () => Array.from({ length: count }, (_, index) => ({ index, key: index, start: index * 100 })),
}) }));
vi.mock("@/components/FilterPanel", () => ({ default: () => null }));
vi.mock("@/components/WineCard", () => ({ default: ({ wine }: { wine: InventoryItem }) =>
  <div data-testid="wine">{wine.wine_name} {wine.vintage}</div> }));

function wine(id: number, vintage: number | null, name = "Estate"): InventoryItem {
  return { wine_id: id, vintage, wine_name: name, producer_name: "Producer", region_name: "Region", quantity: 3 } as InventoryItem;
}
const old = wine(1, 2010);
const recent = wine(2, 2020);
const other = wine(3, 2015, "Other");
const options = { wine_types: [], countries: [], locations: [], producers: [], min_vintage: 2010, max_vintage: 2020 };
function setSort(sort: string, items: InventoryItem[]) {
  state.params = new URLSearchParams({ sort_by: sort });
  state.items = items;
}
afterEach(cleanup);

describe("inventory vintage ordering", () => {
  it("preserves ranked groups and members without losing wines or quantities", () => {
    const groups = groupWinesByIdentity([old, other, recent], true);
    expect(groups.map(group => group.wines.map(item => item.wine_id))).toEqual([[1, 2], [3]]);
    expect(groups.flatMap(group => group.wines).reduce((sum, item) => sum + item.quantity, 0)).toBe(9);
  });

  it("resets to the first-ranked vintage on switching drinking sorts and allows manual selection", async () => {
    const user = userEvent.setup();
    setSort("drink_desc", [old, other, recent]);
    const view = render(<CellarInventory filterOptions={options} />);
    expect(screen.getAllByTestId("wine").map(node => node.textContent)).toEqual(["Estate 2010", "Other 2015"]);
    await user.click(screen.getByRole("button", { name: "2020(3)" }));
    expect(screen.getAllByTestId("wine")[0]).toHaveTextContent("Estate 2020");
    setSort("drink_asc", [recent, other, old]);
    view.rerender(<CellarInventory filterOptions={options} />);
    expect(screen.getAllByTestId("wine")[0]).toHaveTextContent("Estate 2020");
    await user.click(screen.getByRole("button", { name: "2010(3)" }));
    setSort("drink_desc", [old, other, recent]);
    view.rerender(<CellarInventory filterOptions={options} />);
    expect(screen.getAllByTestId("wine")[0]).toHaveTextContent("Estate 2010");
  });

  it.each(["producer", "wine_name", "vintage_asc", "vintage_desc", "rating_asc", "rating_desc", "created_at_desc"])(
    "keeps newest-first vintage selection for %s", (sort) => {
      setSort(sort, [other, old, recent, wine(4, null)]);
      render(<CellarInventory filterOptions={options} />);
      expect(screen.getAllByTestId("wine").map(node => node.textContent)).toEqual(["Other 2015", "Estate 2020"]);
      expect(screen.getAllByRole("button").map(node => node.textContent)).toEqual(["2020(3)", "2010(3)", "NV(3)"]);
    },
  );
});
