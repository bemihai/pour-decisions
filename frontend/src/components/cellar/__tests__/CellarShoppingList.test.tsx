import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import CellarShoppingList from "@/components/cellar/CellarShoppingList";
import {
  ApiError,
  createShoppingListItem,
  deleteShoppingListItem,
  getShoppingList,
  getShoppingListCatalogueOptions,
  updateShoppingListItem,
} from "@/lib/api";
import type { ShoppingListItem, ShoppingListPage } from "@/lib/types";

vi.mock("sonner", () => ({ toast: { success: vi.fn() } }));

vi.mock("@/lib/api", async (importOriginal) => {
  const actual = await importOriginal<typeof import("@/lib/api")>();
  return {
    ...actual,
    createShoppingListItem: vi.fn(),
    deleteShoppingListItem: vi.fn(),
    getShoppingList: vi.fn(),
    getShoppingListCatalogueOptions: vi.fn(),
    updateShoppingListItem: vi.fn(),
  };
});

const item: ShoppingListItem = {
  id: 7,
  wine_id: null,
  identity_kind: "manual",
  catalogue_identity_id: null,
  link_available: false,
  name: "Weekend Barolo",
  vintage: 2019,
  quantity: 2,
  priority: "high",
  notes: "Try the merchant on Friday.",
  retailer_url: "https://retailer.example/wine",
  status: "wanted",
  version: 3,
  created_at: "2026-10-20T10:00:00Z",
  updated_at: "2026-10-20T10:00:00Z",
  purchased_at: null,
};

function page(items: ShoppingListItem[] = []): ShoppingListPage {
  return { items, total: items.length, limit: 25, offset: 0 };
}

function renderShoppingList() {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false }, mutations: { retry: false } },
  });
  const wrapper = ({ children }: { children: ReactNode }) => (
    <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
  );
  return render(<CellarShoppingList />, { wrapper });
}

describe("CellarShoppingList", () => {
  beforeEach(() => {
    vi.mocked(getShoppingList).mockResolvedValue(page());
    vi.mocked(getShoppingListCatalogueOptions).mockResolvedValue({
      items: [], has_more: false, limit: 20, offset: 0,
    });
    vi.mocked(createShoppingListItem).mockResolvedValue(item);
    vi.mocked(updateShoppingListItem).mockResolvedValue(item);
    vi.mocked(deleteShoppingListItem).mockResolvedValue({ id: 7, deleted_version: 3 });
  });

  afterEach(() => vi.clearAllMocks());

  it("shows explicit loading, error, and empty states", async () => {
    vi.mocked(getShoppingList).mockRejectedValueOnce(new ApiError(503, "Shopping list is busy."));
    const user = userEvent.setup();
    renderShoppingList();

    expect(await screen.findByRole("alert")).toHaveTextContent("Shopping list is busy.");
    await user.click(screen.getByRole("button", { name: "Retry" }));
    expect(await screen.findByText("Your shopping list is empty.")).toBeInTheDocument();
  });

  it("changes status and resets paging when searching", async () => {
    const user = userEvent.setup();
    renderShoppingList();
    await screen.findByText("Your shopping list is empty.");

    await user.click(screen.getByRole("button", { name: "Purchased" }));
    await user.type(screen.getByRole("searchbox", { name: "Search shopping list" }), "rioja");

    await waitFor(() => expect(getShoppingList).toHaveBeenLastCalledWith({
      status: "purchased", search: "rioja", limit: 25, offset: 0,
    }));
  });

  it("does not search the catalogue until submitted and records the selected identity", async () => {
    vi.mocked(getShoppingListCatalogueOptions).mockResolvedValue({
      items: [{ id: 19, name: "Local Riesling", vintage: 2022, producer: "Mosel House", bottle_size: "750ml" }],
      has_more: false, limit: 20, offset: 0,
    });
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Add wine" }));
    expect(getShoppingListCatalogueOptions).not.toHaveBeenCalled();
    await user.type(screen.getByRole("textbox", { name: "Catalogue wine name" }), "riesling");
    expect(getShoppingListCatalogueOptions).not.toHaveBeenCalled();
    await user.click(screen.getByRole("button", { name: "Search" }));
    await user.click(await screen.findByRole("radio", { name: /Local Riesling 2022/ }));
    await user.click(screen.getByRole("button", { name: "Add to list" }));

    expect(createShoppingListItem).toHaveBeenCalledWith({
      identity_kind: "catalogue", wine_id: 19, quantity: 1, priority: "normal",
      notes: null, retailer_url: null,
    });
  });

  it("preserves manual form values after a service failure", async () => {
    vi.mocked(createShoppingListItem).mockRejectedValue(new ApiError(503, "Save unavailable."));
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Add wine" }));
    await user.click(screen.getByRole("button", { name: "Enter manually" }));
    await user.type(screen.getByRole("textbox", { name: "Wine name" }), "Special Cuvée");
    await user.clear(screen.getByRole("spinbutton", { name: "Quantity" }));
    await user.type(screen.getByRole("spinbutton", { name: "Quantity" }), "4");
    await user.click(screen.getByRole("button", { name: "Add to list" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Save unavailable");
    expect(screen.getByRole("textbox", { name: "Wine name" })).toHaveValue("Special Cuvée");
    expect(screen.getByRole("spinbutton", { name: "Quantity" })).toHaveValue(4);
  });

  it("refreshes duplicate conflicts and offers the existing item", async () => {
    vi.mocked(createShoppingListItem).mockRejectedValue(
      new ApiError(409, "Already wanted.", "shopping_list_duplicate", { existing_item_id: 42 }),
    );
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Add wine" }));
    await user.click(screen.getByRole("button", { name: "Enter manually" }));
    await user.type(screen.getByRole("textbox", { name: "Wine name" }), "Duplicate Wine");
    await user.click(screen.getByRole("button", { name: "Add to list" }));

    const duplicateButton = await screen.findByRole("button", { name: "Show existing item #42" });
    expect(screen.getByRole("textbox", { name: "Wine name" })).toHaveValue("Duplicate Wine");
    expect(getShoppingList).toHaveBeenCalledTimes(2);
    await user.click(duplicateButton);
    expect(await screen.findByRole("searchbox", { name: "Search shopping list" })).toHaveValue("Duplicate Wine");
  });

  it("confirms purchase without implying an inventory change and uses optimistic versions", async () => {
    vi.mocked(getShoppingList).mockResolvedValue(page([item]));
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Mark purchased" }));
    const dialog = screen.getByRole("dialog", { name: "Mark this item purchased?" });
    expect(dialog).toHaveTextContent("will not add bottles to cellar inventory");
    await user.click(within(dialog).getByRole("button", { name: "Mark purchased" }));

    expect(updateShoppingListItem).toHaveBeenCalledWith(7, { expected_version: 3, status: "purchased" });
    expect(screen.getByRole("heading", { name: "Weekend Barolo 2019" }).closest("[data-slot='card-content']"))
      .toHaveClass("md:grid-cols-[minmax(0,1fr)_auto]");
  });

  it("keeps edit values after a version conflict and refreshes the list", async () => {
    vi.mocked(getShoppingList).mockResolvedValue(page([item]));
    vi.mocked(updateShoppingListItem).mockRejectedValue(
      new ApiError(409, "Item changed.", "shopping_list_version_conflict"),
    );
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Edit" }));
    const dialog = screen.getByRole("dialog", { name: "Edit shopping-list item" });
    const quantity = within(dialog).getByRole("spinbutton", { name: "Quantity" });
    await user.clear(quantity);
    await user.type(quantity, "5");
    await user.click(within(dialog).getByRole("button", { name: "Save changes" }));

    expect(updateShoppingListItem).toHaveBeenCalledWith(7, expect.objectContaining({
      expected_version: 3, quantity: 5,
    }));
    expect(await within(dialog).findByRole("alert")).toHaveTextContent("list has been refreshed");
    expect(quantity).toHaveValue(5);
    expect(getShoppingList).toHaveBeenCalledTimes(2);
  });

  it("shows restore failures inline and suppresses unsafe retailer schemes", async () => {
    vi.mocked(getShoppingList).mockResolvedValue(page([{
      ...item, status: "purchased", retailer_url: "javascript:alert(1)",
    }]));
    vi.mocked(updateShoppingListItem).mockRejectedValue(new ApiError(503, "Restore unavailable."));
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Move to Wanted" }));

    expect(await screen.findByRole("alert")).toHaveTextContent("Restore unavailable");
    expect(screen.queryByRole("link", { name: "Retailer" })).not.toBeInTheDocument();
    expect(updateShoppingListItem).toHaveBeenCalledWith(7, { expected_version: 3, status: "wanted" });
  });

  it("names destructive confirmation and sends the current version", async () => {
    vi.mocked(getShoppingList).mockResolvedValue(page([item]));
    const user = userEvent.setup();
    renderShoppingList();

    await user.click(await screen.findByRole("button", { name: "Delete Weekend Barolo 2019" }));
    const dialog = screen.getByRole("dialog", { name: "Delete Weekend Barolo 2019?" });
    await user.click(within(dialog).getByRole("button", { name: "Delete item" }));

    expect(deleteShoppingListItem).toHaveBeenCalledWith(7, 3);
  });
});
