import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatMessage from "@/components/ChatMessage";
import type { Source, WebSource } from "@/lib/types";

// Mock react-markdown to render its children as plain text so jsdom doesn't
// need to handle the remark/rehype plugin pipeline.
vi.mock("react-markdown", () => ({
  default: ({ children }: { children: string }) => <span data-testid="markdown">{children}</span>,
}));

// Mock LogoMark SVG so we don't need a DOM SVG environment.
vi.mock("@/components/LogoMark", () => ({
  default: ({ title }: { title?: string }) => <span data-testid="logo-mark">{title}</span>,
}));

const RAG_SOURCES: Source[] = [
  { name: "Wine Atlas", page: 10, relevance: 0.9 },
];

const WEB_SOURCES: WebSource[] = [
  { title: "Decanter", url: "https://decanter.com" },
];

describe("ChatMessage — human role", () => {
  it("renders the message content", () => {
    render(<ChatMessage role="human" content="What is Barolo [1]?" />);
    expect(screen.getByText("What is Barolo [1]?")).toBeInTheDocument();
  });

  it("renders the user avatar icon", () => {
    const { container } = render(<ChatMessage role="human" content="Hello" />);
    // User icon is an SVG from lucide-react inside the avatar div
    expect(container.querySelector('[aria-hidden]')).toBeInTheDocument();
  });

  it("aligns the bubble to the right (justify-end)", () => {
    const { container } = render(<ChatMessage role="human" content="Hello" />);
    expect(container.firstElementChild?.className).toContain("justify-end");
  });
});

describe("ChatMessage — AI role", () => {
  it("renders the message content via markdown", () => {
    render(<ChatMessage role="ai" content="Barolo is a red wine." />);
    expect(screen.getByTestId("markdown")).toHaveTextContent("Barolo is a red wine.");
  });

  it("renders the logo mark avatar", () => {
    render(<ChatMessage role="ai" content="Hello" />);
    expect(screen.getByTestId("logo-mark")).toBeInTheDocument();
  });

  it("renders RAG sources when provided", () => {
    render(<ChatMessage role="ai" content="Answer" sources={RAG_SOURCES} />);
    expect(screen.getByText("Wine Atlas")).toBeInTheDocument();
  });

  it("renders web sources when provided", () => {
    render(<ChatMessage role="ai" content="Answer" webSources={WEB_SOURCES} />);
    expect(screen.getByText("Web Sources")).toBeInTheDocument();
    expect(screen.getByRole("link", { name: "Decanter" })).toBeInTheDocument();
  });

  it("renders the agent mode label", () => {
    render(<ChatMessage role="ai" content="Answer" agentMode="intelligent" />);
    expect(screen.getByText("Intelligent Agent")).toBeInTheDocument();
  });

  it("renders 'RAG Only' label for rag_only mode", () => {
    render(<ChatMessage role="ai" content="Answer" agentMode="rag_only" />);
    expect(screen.getByText("RAG Only")).toBeInTheDocument();
  });

  it("does not render an agent mode label when agentMode is omitted", () => {
    render(<ChatMessage role="ai" content="Answer" />);
    expect(screen.queryByText("Intelligent Agent")).toBeNull();
  });
});

describe("ChatMessage — Intelligent citation cleanup", () => {
  it("removes single, grouped, and repeated numeric citations without punctuation gaps", () => {
    render(
      <ChatMessage
        role="ai"
        agentMode="intelligent"
        content="Barolo is made from Nebbiolo [1]. Compare [2, 3]  [4] Barbaresco!"
      />,
    );

    expect(screen.getByTestId("markdown")).toHaveTextContent(
      "Barolo is made from Nebbiolo. Compare Barbaresco!",
    );
  });

  it("preserves bracketed content that is not a numeric citation", () => {
    render(
      <ChatMessage
        role="ai"
        agentMode="intelligent"
        content="Keep [oak], [Vintage 2019], and [1a]."
      />,
    );

    expect(screen.getByTestId("markdown")).toHaveTextContent(
      "Keep [oak], [Vintage 2019], and [1a].",
    );
  });

  it("preserves numeric citations in RAG-only answers", () => {
    render(<ChatMessage role="ai" agentMode="rag_only" content="Nebbiolo [1]." />);
    expect(screen.getByTestId("markdown")).toHaveTextContent("Nebbiolo [1].");
  });
});

describe("ChatMessage — AI error", () => {
  it("renders the error bubble (uses isError)", () => {
    render(<ChatMessage role="ai" content="Something went wrong" isError />);
    // AlertTriangle icon replaces the logo mark in error state
    expect(screen.queryByTestId("logo-mark")).toBeNull();
  });

  it("offers an accessible retry action only when provided", async () => {
    const onRetry = vi.fn();
    const { rerender } = render(
      <ChatMessage role="ai" content="Known failure" isError onRetry={onRetry} />,
    );

    await userEvent.click(screen.getByRole("button", { name: "Retry response" }));
    expect(onRetry).toHaveBeenCalledOnce();

    rerender(<ChatMessage role="ai" content="Uncertain failure" isError />);
    expect(screen.queryByRole("button", { name: "Retry response" })).not.toBeInTheDocument();
  });

  it("does not alter Intelligent-mode error text", () => {
    render(<ChatMessage role="ai" agentMode="intelligent" content="Request failed [1]" isError />);
    expect(screen.getByTestId("markdown")).toHaveTextContent("Request failed [1]");
  });
});

describe("ChatMessage — copy button", () => {
  beforeEach(() => {
    Object.assign(navigator, {
      clipboard: { writeText: vi.fn().mockResolvedValue(undefined) },
    });
  });

  it("renders a copy button on AI messages", () => {
    render(<ChatMessage role="ai" content="Copy me" />);
    expect(screen.getByLabelText("Copy to clipboard")).toBeInTheDocument();
  });

  it("calls clipboard.writeText with the message content on click", async () => {
    render(<ChatMessage role="ai" content="Copy me" />);
    await userEvent.click(screen.getByLabelText("Copy to clipboard"));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("Copy me");
  });

  it("copies the cleaned Intelligent-mode message content", async () => {
    render(
      <ChatMessage
        role="ai"
        agentMode="intelligent"
        content="Barolo is made from Nebbiolo [1]."
      />,
    );
    await userEvent.click(screen.getByLabelText("Copy to clipboard"));
    expect(navigator.clipboard.writeText).toHaveBeenCalledWith("Barolo is made from Nebbiolo.");
  });
});
