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
    render(<ChatMessage role="human" content="What is Barolo?" />);
    expect(screen.getByText("What is Barolo?")).toBeInTheDocument();
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

  it("renders 'Keyword Agent' label for keyword mode", () => {
    render(<ChatMessage role="ai" content="Answer" agentMode="keyword" />);
    expect(screen.getByText("Keyword Agent")).toBeInTheDocument();
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

describe("ChatMessage — AI error", () => {
  it("renders the error bubble (uses isError)", () => {
    render(<ChatMessage role="ai" content="Something went wrong" isError />);
    // AlertTriangle icon replaces the logo mark in error state
    expect(screen.queryByTestId("logo-mark")).toBeNull();
  });

  it("does not render follow-up prompts on error bubbles", () => {
    const onFollowUp = vi.fn();
    render(
      <ChatMessage
        role="ai"
        content="Error"
        isError
        showFollowUps
        onFollowUp={onFollowUp}
      />,
    );
    expect(screen.queryByLabelText("Suggested follow-ups")).toBeNull();
  });
});

describe("ChatMessage — follow-up prompts", () => {
  it("renders follow-up pill buttons when showFollowUps is true", () => {
    render(
      <ChatMessage
        role="ai"
        content="Answer"
        showFollowUps
        onFollowUp={vi.fn()}
      />,
    );
    expect(screen.getByLabelText("Suggested follow-ups")).toBeInTheDocument();
    expect(screen.getAllByRole("button").length).toBeGreaterThan(0);
  });

  it("calls onFollowUp with the prompt text when a pill is clicked", async () => {
    const onFollowUp = vi.fn();
    render(
      <ChatMessage
        role="ai"
        content="Answer"
        showFollowUps
        onFollowUp={onFollowUp}
      />,
    );
    const pills = screen.getAllByRole("button");
    // Click the first follow-up pill (skip any action buttons before them)
    const followUpContainer = screen.getByLabelText("Suggested follow-ups");
    const firstPill = followUpContainer.querySelector("button")!;
    await userEvent.click(firstPill);
    expect(onFollowUp).toHaveBeenCalledOnce();
    expect(typeof onFollowUp.mock.calls[0][0]).toBe("string");
  });

  it("does not render follow-up prompts when showFollowUps is false", () => {
    render(<ChatMessage role="ai" content="Answer" onFollowUp={vi.fn()} />);
    expect(screen.queryByLabelText("Suggested follow-ups")).toBeNull();
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
});

