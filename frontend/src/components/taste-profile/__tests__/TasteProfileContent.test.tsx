import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import TasteProfileContent from "@/components/taste-profile/TasteProfileContent";
import type { TasteProfileContentProps } from "@/components/taste-profile/TasteProfileContent";

const replace = vi.fn();
let searchParams = new URLSearchParams();

vi.mock("next/navigation", () => ({
  usePathname: () => "/taste-profile",
  useRouter: () => ({ replace }),
  useSearchParams: () => searchParams,
}));

vi.mock("@/components/taste-profile/TasteAnalytics", () => ({ default: () => <div>Analytics panel</div> }));
vi.mock("@/components/taste-profile/TasteHistory", () => ({ default: () => <div>History panel</div> }));
vi.mock("@/components/taste-profile/TasteFavorites", () => ({ default: () => <div>Favorites panel</div> }));
vi.mock("@/components/taste-profile/TastePreferences", () => ({ default: () => <div>Preferences panel</div> }));
vi.mock("@/components/ui/tooltip", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipContent: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  TooltipTrigger: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

const props: TasteProfileContentProps = {
  ratingDistribution: { buckets: [], total: 0 },
  wineTypes: { types: [] },
  varietals: { varietals: [] },
  ratingTrends: { points: [], trend: null },
  producers: { producers: [] },
  regions: { regions: [] },
  countries: { countries: [] },
  vintages: { vintages: [] },
  appellations: { appellations: [] },
  initialConsumedFilterOptions: {
    wine_types: [],
    countries: [],
    producers: [],
    min_vintage: 2000,
    max_vintage: 2026,
  },
};

describe("TasteProfileContent preferences tab", () => {
  beforeEach(() => {
    replace.mockReset();
    searchParams = new URLSearchParams("tab=preferences");
  });

  it("selects and lazy-mounts Preferences from the URL after Favorites", () => {
    render(<TasteProfileContent {...props} />);

    const tabs = screen.getAllByRole("tab");
    expect(tabs.map((tab) => tab.textContent)).toEqual([
      "Analytics",
      "Tasting History",
      "Favorites",
      "Preferences",
    ]);
    expect(screen.getByRole("tab", { name: "Preferences" })).toHaveAttribute("aria-selected", "true");
    expect(screen.getByRole("tabpanel", { name: "Preferences" })).toHaveTextContent("Preferences panel");
  });

  it("supports arrow, Home, and End navigation with roving focus", async () => {
    searchParams = new URLSearchParams();
    const user = userEvent.setup();
    render(<TasteProfileContent {...props} />);

    const analyticsTab = screen.getByRole("tab", { name: "Analytics" });
    analyticsTab.focus();
    await user.keyboard("{ArrowRight}");
    expect(screen.getByRole("tab", { name: "Tasting History" })).toHaveFocus();
    expect(replace).toHaveBeenLastCalledWith("/taste-profile?tab=history", { scroll: false });

    await user.keyboard("{End}");
    expect(screen.getByRole("tab", { name: "Preferences" })).toHaveFocus();
    expect(replace).toHaveBeenLastCalledWith("/taste-profile?tab=preferences", { scroll: false });

    await user.keyboard("{Home}");
    expect(analyticsTab).toHaveFocus();
    expect(replace).toHaveBeenLastCalledWith("/taste-profile", { scroll: false });
  });

  it("keeps only the selected tab in the page tab order", () => {
    render(<TasteProfileContent {...props} />);
    expect(screen.getByRole("tab", { name: "Preferences" })).toHaveAttribute("tabindex", "0");
    expect(screen.getByRole("tab", { name: "Analytics" })).toHaveAttribute("tabindex", "-1");
  });
});
