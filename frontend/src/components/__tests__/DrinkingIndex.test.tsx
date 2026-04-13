import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import DrinkingIndex from "@/components/DrinkingIndex";
import { getDrinkingStatus } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Pure function: getDrinkingStatus
// ---------------------------------------------------------------------------

describe("getDrinkingStatus", () => {
  const allIndices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

  it('returns "Peak Drinking" for high indices (normalised >= 75)', () => {
    // drinkIndex = 90 → near the top of the range → normalised ≈ 89
    const status = getDrinkingStatus(90, allIndices);
    expect(status.label).toBe("Peak Drinking");
    expect(status.colorClass).toContain("green");
  });

  it('returns "Ready to Drink" for mid-high indices (50 <= normalised < 75)', () => {
    // drinkIndex = 65 → normalised ≈ 61
    const status = getDrinkingStatus(65, allIndices);
    expect(status.label).toBe("Ready to Drink");
    expect(status.colorClass).toContain("yellow");
  });

  it('returns "Approaching" for mid-low indices (25 <= normalised < 50)', () => {
    // drinkIndex = 35 → normalised ≈ 28
    const status = getDrinkingStatus(35, allIndices);
    expect(status.label).toBe("Approaching");
    expect(status.colorClass).toContain("orange");
  });

  it('returns "Hold" for low indices (normalised < 25)', () => {
    // drinkIndex = 12 → normalised ≈ 2
    const status = getDrinkingStatus(12, allIndices);
    expect(status.label).toBe("Hold");
    expect(status.colorClass).toContain("red");
  });

  it("returns normalised = 50 when drinkIndex is null", () => {
    const status = getDrinkingStatus(null, allIndices);
    expect(status.normalised).toBe(50);
  });

  it("returns normalised = 50 when allIndices is empty", () => {
    const status = getDrinkingStatus(70, []);
    expect(status.normalised).toBe(50);
  });

  it("clamps normalised to [0, 100]", () => {
    const status = getDrinkingStatus(1000, [10, 20, 30]);
    expect(status.normalised).toBeLessThanOrEqual(100);
    expect(status.normalised).toBeGreaterThanOrEqual(0);
  });
});

// ---------------------------------------------------------------------------
// DrinkingIndex component
// ---------------------------------------------------------------------------

describe("DrinkingIndex", () => {
  const allIndices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

  it("renders null when drinkIndex is null", () => {
    const { container } = render(<DrinkingIndex drinkIndex={null} allIndices={allIndices} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders null when drinkIndex is undefined", () => {
    const { container } = render(<DrinkingIndex drinkIndex={undefined} allIndices={allIndices} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders the status badge label", () => {
    render(<DrinkingIndex drinkIndex={90} allIndices={allIndices} />);
    expect(screen.getByText("Peak Drinking")).toBeInTheDocument();
  });

  it("renders the progress meter with correct aria attributes", () => {
    render(<DrinkingIndex drinkIndex={90} allIndices={allIndices} />);
    const meter = screen.getByRole("meter");
    expect(meter).toBeInTheDocument();
    expect(meter).toHaveAttribute("aria-valuemin", "0");
    expect(meter).toHaveAttribute("aria-valuemax", "100");
    const now = Number(meter.getAttribute("aria-valuenow"));
    expect(now).toBeGreaterThanOrEqual(0);
    expect(now).toBeLessThanOrEqual(100);
  });

  it('shows "Drink Sooner" label when normalised >= 50', () => {
    render(<DrinkingIndex drinkIndex={90} allIndices={allIndices} />);
    expect(screen.getByText("Drink Sooner")).toBeInTheDocument();
  });

  it('shows "Drink Later" label when normalised < 50', () => {
    render(<DrinkingIndex drinkIndex={12} allIndices={allIndices} />);
    expect(screen.getByText("Drink Later")).toBeInTheDocument();
  });

  it("includes the accessible aria-label on the meter", () => {
    render(<DrinkingIndex drinkIndex={90} allIndices={allIndices} />);
    const meter = screen.getByRole("meter");
    expect(meter).toHaveAttribute("aria-label", expect.stringContaining("Drinking readiness"));
  });
});

