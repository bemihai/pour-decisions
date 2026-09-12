import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatSidebar, { MobileSidebarTrigger } from "@/components/ChatSidebar";
import { deleteChatThread } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";

// Mock the chat store
vi.mock("@/stores/chat-store", () => ({
  useChatStore: vi.fn(),
}));
vi.mock("@/lib/api", () => ({
  deleteChatThread: vi.fn(),
}));

const mockDeleteChatThread = vi.mocked(deleteChatThread);
const THREAD_ID = "123e4567-e89b-42d3-a456-426614174000";

describe("ChatSidebar — Agent Mode", () => {
  const mockResetChat = vi.fn();
  const mockSetLoading = vi.fn();
  const mockSetConversationError = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockDeleteChatThread.mockResolvedValue();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      threadId: THREAD_ID,
      isLoading: false,
      conversationError: null,
      setLoading: mockSetLoading,
      setConversationError: mockSetConversationError,
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

  it("deletes the server thread before switching to rag_only", async () => {
    render(<ChatSidebar />);

    const ragOnlyButton = screen.getByRole("button", { name: /No Agent \(RAG Only\)/i });
    await userEvent.click(ragOnlyButton);

    await waitFor(() => expect(mockResetChat).toHaveBeenCalledWith("rag_only"));
    expect(mockDeleteChatThread).toHaveBeenCalledWith(THREAD_ID);
    expect(mockDeleteChatThread.mock.invocationCallOrder[0]).toBeLessThan(
      mockResetChat.mock.invocationCallOrder[0],
    );
  });

  it("preserves local state when mode-switch deletion fails", async () => {
    mockDeleteChatThread.mockRejectedValue(new Error("Delete unavailable"));
    render(<ChatSidebar />);

    await userEvent.click(screen.getByRole("button", { name: /No Agent \(RAG Only\)/i }));

    await waitFor(() =>
      expect(mockSetConversationError).toHaveBeenLastCalledWith(
        "Could not clear the current conversation: Delete unavailable",
      ),
    );
    expect(mockResetChat).not.toHaveBeenCalled();
  });

  it("renders lifecycle failures as an accessible alert", () => {
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      threadId: THREAD_ID,
      isLoading: false,
      conversationError: "Could not clear the current conversation",
      setLoading: mockSetLoading,
      setConversationError: mockSetConversationError,
      resetChat: mockResetChat,
    });

    render(<ChatSidebar />);

    expect(screen.getByRole("alert")).toHaveTextContent("Could not clear the current conversation");
  });

  it("does not render a model provider toggle", () => {
    render(<ChatSidebar />);

    expect(screen.queryByText("Local (Ollama)")).not.toBeInTheDocument();
    expect(screen.queryByText("Cloud (Gemini)")).not.toBeInTheDocument();
  });
});

describe("ChatSidebar — Reset Chat", () => {
  const mockResetChat = vi.fn();
  const mockSetLoading = vi.fn();
  const mockSetConversationError = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockDeleteChatThread.mockResolvedValue();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      threadId: THREAD_ID,
      isLoading: false,
      conversationError: null,
      setLoading: mockSetLoading,
      setConversationError: mockSetConversationError,
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

  it("deletes the server thread before resetting local state", async () => {
    render(<ChatSidebar />);
    await userEvent.click(screen.getByRole("button", { name: /Reset Chat/i }));

    const dialog = screen.getByRole("dialog");
    await userEvent.click(within(dialog).getByRole("button", { name: /Reset Chat/i }));

    await waitFor(() => expect(mockResetChat).toHaveBeenCalledWith(undefined));
    expect(mockDeleteChatThread).toHaveBeenCalledWith(THREAD_ID);
    expect(mockDeleteChatThread.mock.invocationCallOrder[0]).toBeLessThan(
      mockResetChat.mock.invocationCallOrder[0],
    );
  });
});

describe("ChatSidebar — Mobile Sheet", () => {
  const mockResetChat = vi.fn();
  const mockSetLoading = vi.fn();
  const mockSetConversationError = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    mockDeleteChatThread.mockResolvedValue();
    (useChatStore as unknown as ReturnType<typeof vi.fn>).mockReturnValue({
      agentMode: "intelligent",
      threadId: THREAD_ID,
      isLoading: false,
      conversationError: null,
      setLoading: mockSetLoading,
      setConversationError: mockSetConversationError,
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








