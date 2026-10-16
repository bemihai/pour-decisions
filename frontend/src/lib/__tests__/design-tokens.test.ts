import { describe, expect, it } from "vitest";

import {
  getWineBottleIllustration,
  getWineTypeColors,
  WINE_TYPE_COLOR_DEFAULT,
} from "@/lib/design-tokens";

describe("wine type design tokens", () => {
  it("uses the established Sparkling styling and bottle illustration", () => {
    expect(getWineTypeColors("Sparkling")).not.toEqual(WINE_TYPE_COLOR_DEFAULT);
    expect(getWineBottleIllustration("Sparkling")).toBe(
      "/wine-illustrations/bottle-sparkling.svg",
    );
  });
});
