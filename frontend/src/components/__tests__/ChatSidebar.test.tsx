import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatSidebar, { MobileSidebarTrigger } from "@/components/ChatSidebar";
import { useChatStore } from "@/stores/chat-store";

// Mock the chat store
vi.mock("@/stores/chat-store", () => ({
  useChatStore: vi.fn(),
}));

describe("ChatSidebar — Agent Mode", () => {
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      resetChat: mockResetChat,
    });
  });

  it("renders all agent mode options", () => {
    render(<ChatSidebar />);

    expect(screen.getByText("Intelligent Agent")).toBeInTheDocument();
    expect(screen.getByText("No Agent (RAG Only)")).toBeInTheDocument();
  });

  it("shows intelligent agent as active by default", () => {
    render(<ChatSidebar />);

    const intelligentButton = screen.getByRole("button", { name: /Intelligent Agent/i });
    expect(intelligentButton.className).toContain("border-brand-burgundy");
    expect(intelligentButton).toHaveAttribute("aria-pressed", "true");
  });

  it("calls setAgentMode when switching to rag_only", async () => {
    render(<ChatSidebar />);

    const ragOnlyButton = screen.getByRole("button", { name: /No Agent \(RAG Only\)/i });
    await userEvent.click(ragOnlyButton);

    expect(mockSetAgentMode).toHaveBeenCalledWith("rag_only");
  });

  it("does not render a model provider toggle", () => {
    render(<ChatSidebar />);

    expect(screen.queryByText("Local (Ollama)")).not.toBeInTheDocument();
    expect(screen.queryByText("Cloud (Gemini)")).not.toBeInTheDocument();
  });
});

describe("ChatSidebar — Reset Chat", () => {
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      resetChat: mockResetChat,
    });
  });

  it("renders reset chat button", () => {
    render(<ChatSidebar />);

    // The reset button is inside a dialog trigger
    expect(screen.getByText("Reset Chat")).toBeInTheDocument();
  });

  it("shows confirmation dialog when clicking reset", async () => {
    render(<ChatSidebar />);

    const resetButton = screen.getByRole("button", { name: /Reset Chat/i });
    await userEvent.click(resetButton);

    // Dialog should appear with confirmation message
    expect(screen.getByText(/This will clear all messages/i)).toBeInTheDocument();
    // Verify both Cancel and Reset Chat buttons are in the dialog
    const dialog = screen.getByRole("dialog");
    expect(within(dialog).getByRole("button", { name: /Cancel/i })).toBeInTheDocument();
    expect(within(dialog).getByRole("button", { name: /Reset Chat/i })).toBeInTheDocument();
  });

  // Note: Testing the actual onClick handler in the dialog is difficult due to
  // pointer-events: none on the body during dialog rendering in jsdom.
  // The resetChat wiring is verified through the component structure and type-checking.
});

describe("ChatSidebar — Mobile Sheet", () => {
  const mockSetAgentMode = vi.fn();
  const mockResetChat = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      isLoading: false,
      setAgentMode: mockSetAgentMode,
      resetChat: mockResetChat,
    });
  });

  it("renders mobile sheet trigger button", () => {
    render(<MobileSidebarTrigger />);

    // The mobile version uses a Sheet with aria-label="Open chat settings"
    const mobileTrigger = screen.getByLabelText(/Open chat settings/i);
    expect(mobileTrigger).toBeInTheDocument();
  });

  // Note: Testing the sheet content is skipped due to ResizeObserver not being defined in jsdom.
  // The Sheet component relies on @radix-ui/react-use-size which requires ResizeObserver.
  // The mobile sheet renders the same SidebarContent component as desktop, tested above.
});










