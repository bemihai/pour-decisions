import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatInterface from "@/components/ChatInterface";
import { ApiError, createConversation, getConversation, getConversationOperation, getConversationTurns, streamChatMessage } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";
import { useConversationStore } from "@/stores/conversation-store";
import type { ChatResponse, ConversationMetadata, ConversationTurnPage } from "@/lib/types";

vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return { ...original, createConversation: vi.fn(), getConversation: vi.fn(), getConversationTurns: vi.fn(), getConversationOperation: vi.fn(), streamChatMessage: vi.fn() };
});

const id = "123e4567-e89b-42d3-a456-426614174000";
const operationId = "223e4567-e89b-42d3-a456-426614174000";
const metadata: ConversationMetadata = { id, agent_mode: "intelligent", title: "Pairing help", version: 3, state: "active", created_at: "2026-10-28T10:00:00Z", updated_at: "2026-10-28T10:01:00Z" };
const emptyTurns: ConversationTurnPage = { items: [], next_before_sequence: null, conversation_version: 3 };
const response: ChatResponse = { answer: "Open the Barolo.", sources: [], web_sources: [], agent_mode: "intelligent", error: null, conversation_id: id, operation_id: operationId, turn_sequence: 1, conversation_version: 4 };

function renderChat(): void {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(<QueryClientProvider client={client}><ChatInterface /></QueryClientProvider>);
}

describe("saved conversation chat", () => {
  beforeEach(() => {
    vi.clearAllMocks(); localStorage.clear(); window.history.replaceState(null, "", "/chat");
    useChatStore.setState({ isLoading: false, conversationError: null });
    useConversationStore.setState({ selectedConversationId: null, selectedMode: "intelligent", isLegacySelected: false });
    vi.mocked(createConversation).mockResolvedValue({ ...metadata, version: 1 });
    vi.mocked(getConversation).mockResolvedValue(metadata);
    vi.mocked(getConversationTurns).mockResolvedValue(emptyTurns);
    vi.mocked(streamChatMessage).mockResolvedValue(response);
    vi.mocked(getConversationOperation).mockRejectedValue(new ApiError(404, "Not found"));
  });

  it("creates a saved conversation and sends display plus execution text", async () => {
    vi.spyOn(crypto, "randomUUID").mockReturnValueOnce(id).mockReturnValueOnce(operationId);
    renderChat();
    await userEvent.type(await screen.findByRole("textbox"), "/pair salmon");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(createConversation).toHaveBeenCalledWith({ id, agent_mode: "intelligent", first_message: "/pair salmon" }));
    expect(streamChatMessage).toHaveBeenCalledWith(expect.objectContaining({ message: expect.stringContaining("salmon"), display_message: "/pair salmon", conversation_id: id, thread_id: id, operation_id: operationId, expected_version: 1 }), expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(window.location.search).toBe("?conversation=" + id);
  });

  it("hydrates saved display text instead of the accepted execution input", async () => {
    window.history.replaceState(null, "", "/chat?conversation=" + id);
    vi.mocked(getConversationTurns).mockResolvedValue({ items: [{ sequence: 1, message: "Recommend wine pairings for salmon.", display_message: "/pair salmon", answer: "Try Chardonnay.", sources: [], web_sources: [], status: "completed", created_at: "2026-10-28T10:00:00Z", updated_at: "2026-10-28T10:01:00Z" }], next_before_sequence: null, conversation_version: 3 });
    renderChat();
    expect(await screen.findByText("/pair salmon")).toBeInTheDocument();
    expect(screen.getByText("Try Chardonnay.")).toBeInTheDocument();
    expect(screen.queryByText("Recommend wine pairings for salmon.")).not.toBeInTheDocument();
  });

  it("loads older turns explicitly", async () => {
    window.history.replaceState(null, "", "/chat?conversation=" + id);
    vi.mocked(getConversationTurns).mockResolvedValueOnce({ items: [], next_before_sequence: 4, conversation_version: 3 }).mockResolvedValueOnce(emptyTurns);
    renderChat();
    await userEvent.click(await screen.findByRole("button", { name: "Load older turns" }));
    await waitFor(() => expect(getConversationTurns).toHaveBeenLastCalledWith(id, 50, 4));
  });

  it("renders the legacy browser transcript as read-only", async () => {
    localStorage.setItem("pour-decisions-chat", JSON.stringify({ state: { messages: [{ role: "human", content: "Old question" }, { role: "ai", content: "Old answer" }], agentMode: "intelligent", threadId: id }, version: 1 }));
    useConversationStore.setState({ isLegacySelected: true }); renderChat();
    expect(await screen.findByText("Old question")).toBeInTheDocument();
    expect(screen.getByText(/Legacy browser chat — read only/)).toBeInTheDocument();
    expect(screen.queryByRole("textbox")).not.toBeInTheDocument();
  });

  it("regenerates from stored values and preserves the old answer on failure", async () => {
    window.history.replaceState(null, "", "/chat?conversation=" + id);
    vi.mocked(getConversationTurns).mockResolvedValue({ items: [{ sequence: 1, message: "Accepted execution", display_message: "Visible command", answer: "Original answer", sources: [], web_sources: [], status: "completed", created_at: "2026-10-28T10:00:00Z", updated_at: "2026-10-28T10:01:00Z" }], next_before_sequence: null, conversation_version: 3 });
    vi.mocked(streamChatMessage).mockRejectedValue(new ApiError(409, "No completed turn"));
    vi.spyOn(crypto, "randomUUID").mockReturnValue(operationId); renderChat();
    await userEvent.click(await screen.findByRole("button", { name: "Regenerate response" }));
    await waitFor(() => expect(streamChatMessage).toHaveBeenCalledWith(expect.objectContaining({ message: "Accepted execution", display_message: "Visible command", thread_action: "replace_last", expected_version: 3 }), expect.anything()));
    expect(screen.getByText("Original answer")).toBeInTheDocument();
    expect(await screen.findByRole("alert")).toHaveTextContent("outcome is uncertain");
  });

  it("reconciles an uncertain delivery using the same operation id", async () => {
    window.history.replaceState(null, "", "/chat?conversation=" + id);
    vi.mocked(streamChatMessage).mockRejectedValue(new TypeError("connection lost"));
    vi.mocked(getConversationOperation).mockResolvedValue({ operation_id: operationId, action: "append", turn_sequence: 2, status: "completed", response, error_code: null, conversation_version: 4 });
    vi.spyOn(crypto, "randomUUID").mockReturnValue(operationId); renderChat();
    await screen.findByText("This saved conversation has no turns yet.");
    await userEvent.type(await screen.findByRole("textbox"), "What should I open?");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(getConversationOperation).toHaveBeenCalledWith(id, operationId));
    expect(screen.queryByText(/outcome is uncertain/i)).not.toBeInTheDocument();
  });

  it("shows a distinct not-found URL state", async () => {
    window.history.replaceState(null, "", "/chat?conversation=" + id);
    vi.mocked(getConversation).mockRejectedValue(new ApiError(404, "Missing")); renderChat();
    expect(await screen.findByRole("alert")).toHaveTextContent("not found");
    expect(screen.getByRole("textbox")).toBeDisabled();
  });
});
