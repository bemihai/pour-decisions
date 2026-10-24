import { beforeEach, describe, expect, it, vi } from "vitest";

import WineDetailPage from "@/app/cellar/[wineId]/page";
import { ApiError, getWine } from "@/lib/api";

const { notFound } = vi.hoisted(() => ({
  notFound: vi.fn(() => { throw new Error("NEXT_NOT_FOUND"); }),
}));
vi.mock("next/navigation", () => ({ notFound }));
vi.mock("@/lib/api", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/lib/api")>(),
  getWine: vi.fn(),
}));

beforeEach(() => vi.clearAllMocks());

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
