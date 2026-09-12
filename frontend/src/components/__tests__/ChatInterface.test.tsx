import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatInterface from "@/components/ChatInterface";
import { ApiError, sendChatMessage } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";


vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return { ...original, sendChatMessage: vi.fn() };
});

const mockSendChatMessage = vi.mocked(sendChatMessage);


describe("ChatInterface thread synchronization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    useChatStore.setState({
      messages: [{ role: "ai", content: "Welcome" }],
      agentMode: "intelligent",
      threadId: "123e4567-e89b-42d3-a456-426614174000",
      isLoading: false,
      conversationError: null,
    });
  });

  it("sends history ending before the current human message", async () => {
    mockSendChatMessage.mockResolvedValue({
      answer: "A current answer",
      sources: [],
      web_sources: [],
      agent_mode: "intelligent",
      model_provider: "cloud",
      error: null,
      trace_id: null,
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
    });
    render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Current question");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(mockSendChatMessage).toHaveBeenCalledOnce());
    expect(mockSendChatMessage).toHaveBeenCalledWith({
      message: "Current question",
      agent_mode: "intelligent",
      message_history: [{ role: "ai", content: "Welcome" }],
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
      thread_action: "append",
    });
    expect(useChatStore.getState().messages.map((message) => message.content)).toEqual([
      "Welcome",
      "Current question",
      "A current answer",
    ]);
  });

  it("replaces only the final AI response after regeneration succeeds", async () => {
    useChatStore.setState({
      messages: [
        { role: "ai", content: "Welcome" },
        { role: "human", content: "Original question" },
        { role: "ai", content: "Original answer", agentMode: "intelligent" },
      ],
    });
    mockSendChatMessage.mockResolvedValue({
      answer: "Regenerated answer",
      sources: [],
      web_sources: [],
      agent_mode: "intelligent",
      model_provider: "cloud",
      error: null,
      trace_id: null,
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
    });
    render(<ChatInterface />);

    await userEvent.click(screen.getByRole("button", { name: "Regenerate response" }));

    await waitFor(() => expect(screen.getByText("Regenerated answer")).toBeInTheDocument());
    expect(mockSendChatMessage).toHaveBeenCalledWith({
      message: "Original question",
      agent_mode: "intelligent",
      message_history: [{ role: "ai", content: "Welcome" }],
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
      thread_action: "replace_last",
    });
    expect(useChatStore.getState().messages.map((message) => message.content)).toEqual([
      "Welcome",
      "Original question",
      "Regenerated answer",
    ]);
  });

  it("preserves the original answer when regeneration fails", async () => {
    useChatStore.setState({
      messages: [
        { role: "ai", content: "Welcome" },
        { role: "human", content: "Original question" },
        { role: "ai", content: "Original answer", agentMode: "intelligent" },
      ],
    });
    mockSendChatMessage.mockRejectedValue(new ApiError(409, "No completed turn"));
    render(<ChatInterface />);

    await userEvent.click(screen.getByRole("button", { name: "Regenerate response" }));

    await waitFor(() => expect(screen.getByRole("alert")).toHaveTextContent("No completed turn"));
    expect(screen.getByText("Original answer")).toBeInTheDocument();
    expect(useChatStore.getState().messages.map((message) => message.content)).toEqual([
      "Welcome",
      "Original question",
      "Original answer",
    ]);
  });

  it("retries a known failed append without duplicating its human message", async () => {
    useChatStore.setState({
      messages: [
        { role: "ai", content: "Welcome" },
        { role: "human", content: "Retry this" },
        {
          role: "ai",
          content: "Known failure",
          isError: true,
          isRetryable: true,
        },
      ],
    });
    mockSendChatMessage.mockResolvedValue({
      answer: "Retry succeeded",
      sources: [],
      web_sources: [],
      agent_mode: "intelligent",
      model_provider: "cloud",
      error: null,
      trace_id: null,
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
    });
    render(<ChatInterface />);

    await userEvent.click(screen.getByRole("button", { name: "Retry response" }));

    await waitFor(() => expect(screen.getByText("Retry succeeded")).toBeInTheDocument());
    expect(mockSendChatMessage).toHaveBeenCalledWith({
      message: "Retry this",
      agent_mode: "intelligent",
      message_history: [{ role: "ai", content: "Welcome" }],
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
      thread_action: "append",
    });
    expect(
      useChatStore.getState().messages.filter((message) => message.role === "human"),
    ).toHaveLength(1);
    expect(useChatStore.getState().messages.at(-1)).toMatchObject({
      content: "Retry succeeded",
      isError: false,
    });
  });

  it("marks unknown network outcomes as uncertain and non-retryable", async () => {
    mockSendChatMessage.mockRejectedValue(new TypeError("Network connection lost"));
    render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Uncertain question");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));

    await waitFor(() => expect(screen.getByText(/request outcome is uncertain/i)).toBeInTheDocument());
    expect(screen.queryByRole("button", { name: "Retry response" })).not.toBeInTheDocument();
    expect(useChatStore.getState().messages.at(-1)).toMatchObject({
      isError: true,
      isRetryable: false,
    });
  });

  it("offers a deliberate retry for a known failed append", async () => {
    mockSendChatMessage.mockResolvedValueOnce({
      answer: "Known provider failure",
      sources: [],
      web_sources: [],
      agent_mode: "intelligent",
      model_provider: "cloud",
      error: "Known provider failure",
      trace_id: null,
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
    });
    render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Retryable question");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(screen.getByRole("button", { name: "Retry response" })).toBeVisible());

    mockSendChatMessage.mockResolvedValueOnce({
      answer: "Recovered answer",
      sources: [],
      web_sources: [],
      agent_mode: "intelligent",
      model_provider: "cloud",
      error: null,
      trace_id: null,
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
    });
    await userEvent.click(screen.getByRole("button", { name: "Retry response" }));

    await waitFor(() => expect(screen.getByText("Recovered answer")).toBeInTheDocument());
    expect(useChatStore.getState().messages.map((message) => message.content)).toEqual([
      "Welcome",
      "Retryable question",
      "Recovered answer",
    ]);
  });

  it("renders a shared lifecycle failure as an accessible alert", () => {
    useChatStore.setState({ conversationError: "The current thread could not be deleted" });

    render(<ChatInterface />);

    expect(screen.getByRole("alert")).toHaveTextContent(
      "The current thread could not be deleted",
    );
  });
});
