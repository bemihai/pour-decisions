import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import Rating from "@/components/Rating";
import { getRatingLabel, ratingColor } from "@/lib/utils";

const boundaries: Array<[number, string, string]> = [
  [0, "Avoid", "text-red-500"],
  [70, "Avoid", "text-red-500"],
  [71, "Below Average", "text-orange-500"],
  [79, "Below Average", "text-orange-500"],
  [80, "Good", "text-amber-600"],
  [85, "Good", "text-amber-600"],
  [86, "Very Good", "text-yellow-600"],
  [89, "Very Good", "text-yellow-600"],
  [90, "Excellent", "text-lime-600"],
  [93, "Excellent", "text-lime-600"],
  [94, "Outstanding", "text-green-600"],
  [97, "Outstanding", "text-green-600"],
  [98, "Extraordinary", "text-emerald-600"],
  [100, "Extraordinary", "text-emerald-600"],
];

describe("rating tiers", () => {
  it.each(boundaries)("maps %i to %s", (rating, label, color) => {
    expect(getRatingLabel(rating)).toBe(label);
    expect(ratingColor(rating)).toBe(color);
  });

  it("uses the approved label in compact accessibility text", () => {
    render(<Rating rating={98} />);
    expect(screen.getByLabelText("Rating: 98/100 — Extraordinary")).toBeInTheDocument();
  });

  it("uses the approved label in gauge accessibility text", () => {
    render(<Rating rating={70} variant="gauge" />);
    expect(screen.getByRole("meter", { name: "Rating: 70/100 — Avoid" })).toBeInTheDocument();
  });
});
