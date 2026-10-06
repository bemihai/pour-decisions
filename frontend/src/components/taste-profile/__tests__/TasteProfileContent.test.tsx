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

  it("keeps the tab keyboard reachable and writes the approved URL", async () => {
    searchParams = new URLSearchParams();
    const user = userEvent.setup();
    render(<TasteProfileContent {...props} />);

    const preferencesTab = screen.getByRole("tab", { name: "Preferences" });
    await user.tab();
    await user.tab();
    await user.tab();
    await user.tab();
    expect(preferencesTab).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(replace).toHaveBeenCalledWith("/taste-profile?tab=preferences", { scroll: false });
  });
});
