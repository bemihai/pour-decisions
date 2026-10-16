import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DrinkingIndex from "@/components/DrinkingIndex";
import type { DrinkStatus } from "@/lib/types";
import { getDrinkingStatus } from "@/lib/utils";

describe("getDrinkingStatus", () => {
  it.each<[DrinkStatus, string, string]>([
    ["hold", "Hold", "red"],
    ["ready", "Ready to Drink", "green"],
    ["past_peak", "Past Peak", "orange"],
    ["unknown", "Unknown", "muted"],
  ])("maps backend status %s to %s presentation", (status, label, color) => {
    const presentation = getDrinkingStatus(status);

    expect(presentation.label).toBe(label);
    expect(presentation.colorClass).toContain(color);
  });
});

describe("DrinkingIndex", () => {
  it.each<[DrinkStatus, string]>([
    ["hold", "Hold"],
    ["ready", "Ready to Drink"],
    ["past_peak", "Past Peak"],
    ["unknown", "Unknown"],
  ])("renders the backend-owned %s status", (status, label) => {
    render(<DrinkingIndex drinkStatus={status} />);

    expect(screen.getByText(label)).toBeInTheDocument();
  });

  it("does not render a percentile-dependent progress meter", () => {
    const { container } = render(<DrinkingIndex drinkStatus="ready" />);

    expect(container.querySelector('[role="meter"]')).toBeNull();
  });
});
