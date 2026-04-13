import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";

import SourceList from "@/components/SourceList";
import type { Source, WebSource } from "@/lib/types";

const ragSources: Source[] = [
  { name: "The World Atlas of Wine", page: 42, relevance: 0.9 },
  { name: "Wine Folly Guide", page: null, relevance: 0.65 },
  { name: "Jancis Robinson's Guide", page: 10, relevance: 0.35 },
  { name: "Wine Spectator", page: null, relevance: null },
];

const webSources: WebSource[] = [
  { title: "Decanter: Bordeaux Guide", url: "https://decanter.com/bordeaux" },
  { title: "", url: "https://wineenthusiast.com" },
];

describe("SourceList — RAG sources", () => {
  it("renders nothing when sources array is empty", () => {
    const { container } = render(<SourceList sources={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a 'Sources' section header", () => {
    render(<SourceList sources={ragSources} />);
    expect(screen.getByText("Sources")).toBeInTheDocument();
  });

  it("renders all source names", () => {
    render(<SourceList sources={ragSources} />);
    expect(screen.getByText("The World Atlas of Wine")).toBeInTheDocument();
    expect(screen.getByText("Wine Folly Guide")).toBeInTheDocument();
  });

  it("renders page number when available", () => {
    render(<SourceList sources={ragSources} />);
    expect(screen.getByText(/p\..*42/)).toBeInTheDocument();
  });

  it("shows 'High' relevance label for score >= 0.8", () => {
    render(<SourceList sources={[ragSources[0]]} />);
    expect(screen.getByText("High")).toBeInTheDocument();
  });

  it("shows 'Good' relevance label for score in [0.6, 0.8)", () => {
    render(<SourceList sources={[ragSources[1]]} />);
    expect(screen.getByText("Good")).toBeInTheDocument();
  });

  it("shows 'Low' relevance label for score < 0.4", () => {
    render(<SourceList sources={[ragSources[2]]} />);
    expect(screen.getByText("Low")).toBeInTheDocument();
  });

  it("renders relevance dot with accessible label", () => {
    render(<SourceList sources={[ragSources[0]]} />);
    expect(screen.getByLabelText("Excellent relevance")).toBeInTheDocument();
  });
});

describe("SourceList — web sources", () => {
  it("renders nothing when web sources array is empty", () => {
    const { container } = render(<SourceList sources={[]} isWeb />);
    expect(container.firstChild).toBeNull();
  });

  it("renders a 'Web Sources' section header", () => {
    render(<SourceList sources={webSources} isWeb />);
    expect(screen.getByText("Web Sources")).toBeInTheDocument();
  });

  it("renders source titles as links", () => {
    render(<SourceList sources={webSources} isWeb />);
    const link = screen.getByRole("link", { name: "Decanter: Bordeaux Guide" });
    expect(link).toHaveAttribute("href", "https://decanter.com/bordeaux");
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });

  it("falls back to URL when title is empty", () => {
    render(<SourceList sources={webSources} isWeb />);
    expect(screen.getByRole("link", { name: "https://wineenthusiast.com" })).toBeInTheDocument();
  });
});

