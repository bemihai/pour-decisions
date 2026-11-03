import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import WineDetailPage from "@/app/cellar/[wineId]/page";
import {
  ApiError,
  generateGrapeDescription,
  generateProducerDescription,
  generateWineDescription,
  getWine,
} from "@/lib/api";
import type { WineDetailResponse } from "@/lib/types";

const { notFound, refresh } = vi.hoisted(() => ({
  notFound: vi.fn(() => { throw new Error("NEXT_NOT_FOUND"); }),
  refresh: vi.fn(),
}));
vi.mock("next/navigation", () => ({ notFound, useRouter: () => ({ refresh }) }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/api")>(),
  generateGrapeDescription: vi.fn(),
  generateProducerDescription: vi.fn(),
  generateWineDescription: vi.fn(),
  getWine: vi.fn(),
}));

beforeEach(() => vi.clearAllMocks());

const ownedWine = {
  id: 14,
  source: "cellar_tracker",
  external_id: "fixture-14",
  wine_name: "A Representative Cellar Wine",
  vintage: 2016,
  wine_type: "Red",
  varietal: "Red Blend",
  designation: "Reserve",
  appellation: null,
  vineyard: null,
  bottle_size: "750ml",
  drink_from_year: 2023,
  drink_to_year: 2029,
  drink_index: 100,
  drink_status: "ready",
  drink_window_source: "cellar_tracker",
  description: "A concise wine description.",
  producer_description: "Producer background.",
  grapes: [],
  producer_id: 9,
  producer_name: "Representative Producer",
  region_id: 4,
  region_name: "Representative Region",
  region_description: "Region background.",
  country: "Romania",
  personal_rating: 94,
  community_rating: 91,
  do_like: true,
  is_defective: false,
  tasting_notes: "[2026-08-17] Balanced and ready to drink.",
  last_tasted_date: "2026-08-17",
  q_purchased: 16,
  q_quantity: 12,
  q_consumed: 4,
  bottles: [
    {
      id: 1,
      quantity: 1,
      status: "in_cellar",
      location: "Cellar",
      bin: "1.3",
      purchase_date: "2025-01-02",
      purchase_price: 100,
      valuation_price: 110,
      currency: "RON",
      store_name: "Local retailer",
      consumed_date: null,
      bottle_note: null,
    },
    {
      id: 2,
      quantity: 1,
      status: "consumed",
      location: "Cellar",
      bin: "1.3",
      purchase_date: "2025-01-02",
      purchase_price: 100,
      valuation_price: 100,
      currency: "RON",
      store_name: "Local retailer",
      consumed_date: "2026-08-17",
      bottle_note: null,
    },
  ],
  owned_quantity: 1,
  created_at: "2025-01-02T00:00:00Z",
  updated_at: "2026-08-17T00:00:00Z",
} satisfies WineDetailResponse;

describe("wine detail error handling", () => {
  it("shows not found when the API confirms the wine is missing", async () => {
    vi.mocked(getWine).mockRejectedValue(new ApiError(404, "Wine not found"));
    await expect(WineDetailPage({ params: Promise.resolve({ wineId: "1078" }) })).rejects.toThrow("NEXT_NOT_FOUND");
    expect(notFound).toHaveBeenCalledOnce();
  });

  it.each([
    new ApiError(500, "Internal Server Error"),
    new TypeError("fetch failed"),
  ])("preserves operational errors for the error boundary: %s", async (error) => {
    vi.mocked(getWine).mockRejectedValue(error);
    await expect(WineDetailPage({ params: Promise.resolve({ wineId: "1078" }) })).rejects.toBe(error);
    expect(notFound).not.toHaveBeenCalled();
  });
});

describe("wine detail presentation", () => {
  it("prioritizes identity, personal experience, and practical cellar information", async () => {
    vi.mocked(getWine).mockResolvedValue(ownedWine);

    render(await WineDetailPage({ params: Promise.resolve({ wineId: "14" }) }));

    expect(screen.getByRole("heading", { level: 1, name: /A Representative Cellar Wine/ })).toBeInTheDocument();
    expect(screen.getByText("94/100")).toBeInTheDocument();
    expect(screen.getByText("91/100")).toBeInTheDocument();
    expect(screen.getByText("Ready to Drink")).toBeInTheDocument();
    expect(screen.getAllByText("Cellar · Bin 1.3").length).toBeGreaterThan(0);
    expect(screen.getByText("2 records · 2 bottles")).toBeInTheDocument();

    const about = screen.getByText("About this wine");
    const experience = screen.getByText("Your experience");
    const background = screen.getByText("Explore the wine");
    expect(about.compareDocumentPosition(experience) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
    expect(experience.compareDocumentPosition(background) & Node.DOCUMENT_POSITION_FOLLOWING).toBeTruthy();
  });

  it("shows intentional missing and unowned states", async () => {
    vi.mocked(getWine).mockResolvedValue({
      ...ownedWine,
      personal_rating: null,
      community_rating: null,
      do_like: null,
      tasting_notes: null,
      last_tasted_date: null,
      description: null,
      producer_id: null,
      producer_name: null,
      producer_description: null,
      region_description: null,
      drink_from_year: null,
      drink_to_year: null,
      drink_window_source: null,
      bottles: [{ ...ownedWine.bottles[1], id: 3 }],
      owned_quantity: 0,
    });

    render(await WineDetailPage({ params: Promise.resolve({ wineId: "15" }) }));

    expect(screen.getAllByText("Not rated")).toHaveLength(2);
    expect(screen.getByText("No longer in cellar")).toBeInTheDocument();
    expect(screen.getByText("No bottles owned")).toBeInTheDocument();
    expect(screen.getByText("You have no personal rating, preference, or tasting note for this wine.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Generate Description" })).toBeEnabled();
  });

  it("retains a long wine identity without truncating its accessible name", async () => {
    const longName = "A Very Long Wine Name That Must Remain Fully Available in the Detail Overview";
    vi.mocked(getWine).mockResolvedValue({ ...ownedWine, wine_name: longName });

    render(await WineDetailPage({ params: Promise.resolve({ wineId: "16" }) }));

    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent(longName);
  });

  it("reveals existing reference content without triggering generation", async () => {
    vi.mocked(getWine).mockResolvedValue({
      ...ownedWine,
      producer_description: null,
      grapes: [{ id: 1, name: "cabernet franc", description: null }],
    });

    render(await WineDetailPage({ params: Promise.resolve({ wineId: "17" }) }));

    fireEvent.click(screen.getByText("Read full description"));
    fireEvent.click(screen.getByText("About Representative Producer"));
    fireEvent.click(screen.getByText("Grapes"));
    fireEvent.click(screen.getByText("Technical details"));

    expect(screen.getAllByText("A concise wine description.")).toHaveLength(2);
    expect(screen.getByRole("button", { name: "Generate Description" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Generate cabernet franc description" })).toBeEnabled();
    expect(generateWineDescription).not.toHaveBeenCalled();
    expect(generateProducerDescription).not.toHaveBeenCalled();
    expect(generateGrapeDescription).not.toHaveBeenCalled();
  });

  it("separates current holdings from expandable historical records", async () => {
    vi.mocked(getWine).mockResolvedValue(ownedWine);

    render(await WineDetailPage({ params: Promise.resolve({ wineId: "18" }) }));

    expect(screen.getByText("Current holdings · 1 record · 1 bottle")).toBeInTheDocument();
    expect(screen.getByText("Historical records · 1 record · 1 bottle")).toBeInTheDocument();
    expect(screen.getAllByRole("columnheader", { name: "Quantity" })).toHaveLength(2);
    expect(screen.getByText("Record counts and bottle quantities are shown separately.")).toBeInTheDocument();
  });
});
