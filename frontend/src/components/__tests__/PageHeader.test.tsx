import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import PageHeader from "@/components/PageHeader";

describe("PageHeader", () => {
  it("renders the title", () => {
    render(<PageHeader title="Wine Cellar" />);
    expect(screen.getByRole("heading", { level: 1 })).toHaveTextContent("Wine Cellar");
  });

  it("renders the subtitle when provided", () => {
    render(<PageHeader title="Wine Cellar" subtitle="Manage your collection" />);
    expect(screen.getByText("Manage your collection")).toBeInTheDocument();
  });

  it("renders the optional eyebrow", () => {
    render(<PageHeader title="Wine Cellar" eyebrow="Your collection" />);
    expect(screen.getByText("Your collection")).toBeInTheDocument();
  });

  it("does not render supporting text when subtitle and eyebrow are omitted", () => {
    const { container } = render(<PageHeader title="Wine Cellar" />);
    expect(container.querySelector("p")).toBeNull();
  });

  it("uses the semantic page-title class in default mode", () => {
    render(<PageHeader title="Wine Cellar" />);
    const h1 = screen.getByRole("heading", { level: 1 });
    expect(h1.className).toContain("type-page-title");
  });

  it("uses smaller text class in compact mode", () => {
    render(<PageHeader title="Wine Cellar" compact />);
    const h1 = screen.getByRole("heading", { level: 1 });
    expect(h1.className).toContain("text-2xl");
  });

  it("applies compact margin wrapper in compact mode", () => {
    const { container } = render(<PageHeader title="Wine Cellar" compact />);
    expect(container.firstElementChild?.className).toContain("mb-4");
  });

  it("applies large margin wrapper in default mode", () => {
    const { container } = render(<PageHeader title="Wine Cellar" />);
    expect(container.firstElementChild?.className).toContain("mb-8");
  });

  it("merges extra className onto wrapper", () => {
    const { container } = render(<PageHeader title="Wine Cellar" className="custom-class" />);
    expect(container.firstElementChild?.className).toContain("custom-class");
  });
});

