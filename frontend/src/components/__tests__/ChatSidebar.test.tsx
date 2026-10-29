import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatSidebar, { MobileSidebarTrigger } from "@/components/ChatSidebar";
import { deleteConversation, listConversations, renameConversation } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";
import { useConversationStore } from "@/stores/conversation-store";
import type { ConversationMetadata } from "@/lib/types";

vi.mock("@/lib/api", () => ({
  deleteConversation: vi.fn(),
  listConversations: vi.fn(),
  renameConversation: vi.fn(),
}));

const CONVERSATION: ConversationMetadata = {
  id: "123e4567-e89b-12d3-a456-426614174000",
  agent_mode: "intelligent",
  title: "Dinner wines",
  version: 4,
  state: "active",
  created_at: "2026-10-27T10:00:00Z",
  updated_at: "2026-10-28T10:00:00Z",
};

function renderSidebar() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <ChatSidebar />
    </QueryClientProvider>,
  );
}

describe("conversation library sidebar", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    window.history.replaceState(null, "", "/");
    vi.mocked(listConversations).mockResolvedValue({ items: [CONVERSATION], next_cursor: null });
    vi.mocked(renameConversation).mockResolvedValue({ ...CONVERSATION, title: "Renamed", version: 5 });
    vi.mocked(deleteConversation).mockResolvedValue();
    useChatStore.setState({ isLoading: false });
    useConversationStore.setState({
      selectedConversationId: null,
      selectedMode: "intelligent",
      isLegacySelected: false,
    });
  });

  it("shows New, recent conversations, and fixed-mode choices", async () => {
    renderSidebar();

    expect(screen.getByRole("button", { name: /New/i })).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: "Open Dinner wines" })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Intelligent Agent/i })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
    expect(screen.getByRole("button", { name: /RAG Only/i })).toBeInTheDocument();
  });

  it("opens a saved row and records its identity in the URL", async () => {
    renderSidebar();
    await userEvent.click(await screen.findByRole("button", { name: "Open Dinner wines" }));

    expect(useConversationStore.getState()).toMatchObject({
      selectedConversationId: CONVERSATION.id,
      selectedMode: "intelligent",
    });
    expect(new URL(window.location.href).searchParams.get("conversation")).toBe(CONVERSATION.id);
  });

  it("New and mode changes preserve saved conversations", async () => {
    renderSidebar();
    await userEvent.click(await screen.findByRole("button", { name: "Open Dinner wines" }));
    await userEvent.click(screen.getByRole("button", { name: /RAG Only/i }));

    expect(useConversationStore.getState()).toMatchObject({
      selectedConversationId: null,
      selectedMode: "rag_only",
    });
    expect(deleteConversation).not.toHaveBeenCalled();
    expect(new URL(window.location.href).searchParams.has("conversation")).toBe(false);
  });

  it("renames with the current version and refreshes the list", async () => {
    renderSidebar();
    await userEvent.click(await screen.findByRole("button", { name: "Rename Dinner wines" }));
    const dialog = screen.getByRole("dialog");
    const input = within(dialog).getByRole("textbox", { name: "Conversation title" });
    await userEvent.clear(input);
    await userEvent.type(input, "Renamed");
    await userEvent.click(within(dialog).getByRole("button", { name: "Save" }));

    await waitFor(() =>
      expect(renameConversation).toHaveBeenCalledWith(CONVERSATION.id, {
        title: "Renamed",
        expected_version: 4,
      }),
    );
  });

  it("confirms deletion and returns a selected conversation to a draft", async () => {
    useConversationStore.getState().selectConversation(CONVERSATION.id, "intelligent");
    renderSidebar();
    await userEvent.click(await screen.findByRole("button", { name: "Delete Dinner wines" }));
    await userEvent.click(within(screen.getByRole("dialog")).getByRole("button", { name: "Delete" }));

    await waitFor(() => expect(deleteConversation).toHaveBeenCalledWith(CONVERSATION.id, 4));
    expect(useConversationStore.getState().selectedConversationId).toBeNull();
  });

  it("disables New and mode changes while a delivery is active", async () => {
    useChatStore.setState({ isLoading: true });
    renderSidebar();

    expect(screen.getByRole("button", { name: /New/i })).toBeDisabled();
    expect(screen.getByRole("button", { name: /RAG Only/i })).toBeDisabled();
    expect(await screen.findByRole("button", { name: "Open Dinner wines" })).toBeDisabled();
  });
});

describe("mobile conversation library", () => {
  it("retains the existing right-side drawer entry point", () => {
    render(<MobileSidebarTrigger />);
    expect(screen.getByRole("button", { name: "Open conversation library" })).toBeInTheDocument();
  });
});
