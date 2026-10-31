import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

import WineDescriptionGenerator from "@/components/cellar/WineDescriptionGenerator";
import { generateWineDescription } from "@/lib/api";

vi.mock("@/lib/api", () => ({ generateWineDescription: vi.fn() }));
beforeEach(() => vi.clearAllMocks());

describe("wine detail description generation", () => {
  it.each([1079, 1082])("displays the fallback returned for wine %s", async (wineId) => {
    const description = "Based on general knowledge of the grape, this style may show fresh fruit character.";
    vi.mocked(generateWineDescription).mockResolvedValue({
      success: true, description, drink_from_year: 2026, drink_to_year: 2027,
    });
    render(<WineDescriptionGenerator wineId={wineId} />);
    fireEvent.click(screen.getByRole("button", { name: "Generate Description" }));
    expect(await screen.findByText(description)).toBeInTheDocument();
    expect(generateWineDescription).toHaveBeenCalledExactlyOnceWith(wineId);
    expect(screen.queryByRole("button")).not.toBeInTheDocument();
  });

  it("shows generation failure without automatically retrying", async () => {
    vi.mocked(generateWineDescription).mockRejectedValue(new Error("generation failed"));
    render(<WineDescriptionGenerator wineId={1079} />);
    fireEvent.click(screen.getByRole("button", { name: "Generate Description" }));
    expect(await screen.findByText("Failed to generate description. Please try again.")).toBeInTheDocument();
    expect(generateWineDescription).toHaveBeenCalledTimes(1);
    expect(screen.getByRole("button", { name: "Generate Description" })).toBeEnabled();
  });
});
