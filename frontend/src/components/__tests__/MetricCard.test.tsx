import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import MetricCard from "@/components/MetricCard";

describe("MetricCard", () => {
  it("renders the label", () => {
    render(<MetricCard label="Total Bottles" value={42} />);
    expect(screen.getByText("Total Bottles")).toBeInTheDocument();
  });

  it("renders a numeric value", () => {
    render(<MetricCard label="Total Bottles" value={42} />);
    expect(screen.getByText("42")).toBeInTheDocument();
  });

  it("renders a string value", () => {
    render(<MetricCard label="Avg Rating" value="91/100" />);
    expect(screen.getByText("91/100")).toBeInTheDocument();
  });

  it("does not render a delta element when delta is omitted", () => {
    const { container } = render(<MetricCard label="Total Bottles" value={42} />);
    // Only label + value spans — no third span
    const spans = container.querySelectorAll("span");
    expect(spans).toHaveLength(2);
  });

  it("renders the delta text when provided", () => {
    render(<MetricCard label="Total Bottles" value={42} delta="+3 this month" />);
    expect(screen.getByText("+3 this month")).toBeInTheDocument();
  });

  it("applies green color for a positive delta", () => {
    render(<MetricCard label="Total Bottles" value={42} delta="+3" />);
    const delta = screen.getByText("+3");
    expect(delta.className).toContain("text-emerald-600");
  });

  it("applies red color for a negative delta", () => {
    render(<MetricCard label="Total Bottles" value={42} delta="-2" />);
    const delta = screen.getByText("-2");
    expect(delta.className).toContain("text-red-500");
  });

  it("applies muted color for a neutral delta (no sign)", () => {
    render(<MetricCard label="Total Bottles" value={42} delta="no change" />);
    const delta = screen.getByText("no change");
    expect(delta.className).toContain("text-muted-foreground");
  });

  it("merges extra className onto the card", () => {
    const { container } = render(
      <MetricCard label="Total Bottles" value={42} className="my-custom" />,
    );
    // The outermost rendered element is the Card div
    expect(container.firstElementChild?.className).toContain("my-custom");
  });
});

