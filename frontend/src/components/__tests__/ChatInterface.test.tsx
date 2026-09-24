import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeEach, describe, expect, it, vi } from "vitest";

import ChatInterface from "@/components/ChatInterface";
import { ApiError, sendChatMessage, streamChatMessage } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";
import type { ChatResponse, ToolProgressStreamEvent } from "@/lib/types";


vi.mock("@/lib/api", async (importOriginal) => {
  const original = await importOriginal<typeof import("@/lib/api")>();
  return { ...original, sendChatMessage: vi.fn(), streamChatMessage: vi.fn() };
});

const mockSendChatMessage = vi.mocked(sendChatMessage);
const mockStreamChatMessage = vi.mocked(streamChatMessage);


describe("ChatInterface thread synchronization", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    useChatStore.setState({
      messages: [{ role: "ai", content: "Welcome" }],
      agentMode: "intelligent",
      threadId: "123e4567-e89b-42d3-a456-426614174000",
      isLoading: false,
      conversationError: null,
    });
  });

  it("sends history ending before the current human message", async () => {
    mockStreamChatMessage.mockResolvedValue({
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

    await waitFor(() => expect(mockStreamChatMessage).toHaveBeenCalledOnce());
    expect(mockStreamChatMessage).toHaveBeenCalledWith({
      message: "Current question",
      agent_mode: "intelligent",
      message_history: [{ role: "ai", content: "Welcome" }],
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
      thread_action: "append",
    }, expect.objectContaining({ signal: expect.any(AbortSignal) }));
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
    mockStreamChatMessage.mockResolvedValue({
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
    expect(mockStreamChatMessage).toHaveBeenCalledWith({
      message: "Original question",
      agent_mode: "intelligent",
      message_history: [{ role: "ai", content: "Welcome" }],
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
      thread_action: "replace_last",
    }, expect.objectContaining({ signal: expect.any(AbortSignal) }));
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
    mockStreamChatMessage.mockRejectedValue(new ApiError(409, "No completed turn"));
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
    mockStreamChatMessage.mockResolvedValue({
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
    expect(mockStreamChatMessage).toHaveBeenCalledWith({
      message: "Retry this",
      agent_mode: "intelligent",
      message_history: [{ role: "ai", content: "Welcome" }],
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
      thread_action: "append",
    }, expect.objectContaining({ signal: expect.any(AbortSignal) }));
    expect(
      useChatStore.getState().messages.filter((message) => message.role === "human"),
    ).toHaveLength(1);
    expect(useChatStore.getState().messages.at(-1)).toMatchObject({
      content: "Retry succeeded",
      isError: false,
    });
  });

  it("marks unknown network outcomes as uncertain and non-retryable", async () => {
    mockStreamChatMessage.mockRejectedValue(new TypeError("Network connection lost"));
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

  it("shows request-local progress and clears it when the final answer arrives", async () => {
    let finish!: (response: ChatResponse) => void;
    mockStreamChatMessage.mockImplementation((_request, options) => {
      options?.onProgress?.({
        type: "tool_progress",
        invocation_id: 1,
        tool_key: "search_wine_price",
        status: "started",
      });
      return new Promise((resolve) => {
        finish = resolve;
      });
    });
    render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Check this price");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText("Checking current wine prices")).toBeInTheDocument();
    expect(screen.getByText("In progress")).toBeInTheDocument();

    await act(async () => {
      finish({
        answer: "Final price answer",
        sources: [],
        web_sources: [],
        agent_mode: "intelligent",
        model_provider: "cloud",
        error: null,
        trace_id: null,
        thread_id: "123e4567-e89b-42d3-a456-426614174000",
      });
    });

    expect(await screen.findByText("Final price answer")).toBeInTheDocument();
    expect(screen.queryByRole("status", { name: "Agent progress" })).not.toBeInTheDocument();
  });

  it("aborts an active stream when the component unmounts", async () => {
    let requestSignal: AbortSignal | undefined;
    mockStreamChatMessage.mockImplementation((_request, options) => {
      requestSignal = options?.signal;
      return new Promise((_resolve, reject) => {
        requestSignal?.addEventListener("abort", () =>
          reject(new DOMException("Aborted", "AbortError")),
        );
      });
    });
    const view = render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Keep working");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(mockStreamChatMessage).toHaveBeenCalledOnce());

    view.unmount();

    expect(requestSignal?.aborted).toBe(true);
  });

  it("ignores progress and completion from a request after the conversation resets", async () => {
    let onProgress: ((event: ToolProgressStreamEvent) => void) | undefined;
    let finish!: (response: ChatResponse) => void;
    mockStreamChatMessage.mockImplementation((_request, options) => {
      onProgress = options?.onProgress;
      return new Promise((resolve) => {
        finish = resolve;
      });
    });
    render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Old conversation request");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    await waitFor(() => expect(mockStreamChatMessage).toHaveBeenCalledOnce());

    act(() => useChatStore.getState().resetChat());
    act(() =>
      onProgress?.({
        type: "tool_progress",
        invocation_id: 9,
        tool_key: "search_web_for_wine",
        status: "started",
      }),
    );
    await act(async () => {
      finish({
        answer: "Late answer",
        sources: [],
        web_sources: [],
        agent_mode: "intelligent",
        model_provider: "cloud",
        error: null,
        trace_id: null,
        thread_id: "123e4567-e89b-42d3-a456-426614174000",
      });
    });

    expect(screen.queryByText("Searching current wine information")).not.toBeInTheDocument();
    expect(screen.queryByText("Late answer")).not.toBeInTheDocument();
    expect(useChatStore.getState().messages.map((message) => message.content)).toEqual([
      expect.stringContaining("Welcome to Pour Decisions"),
    ]);
  });

  it("keeps transient progress outside persisted chat state", async () => {
    mockStreamChatMessage.mockImplementation((_request, options) => {
      options?.onProgress?.({
        type: "tool_progress",
        invocation_id: 3,
        tool_key: "search_wine_reviews",
        status: "started",
      });
      return new Promise(() => undefined);
    });
    const view = render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Review this wine");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));
    expect(await screen.findByText("Checking wine reviews")).toBeInTheDocument();

    const persisted = localStorage.getItem("pour-decisions-chat") ?? "";
    expect(persisted).not.toContain("search_wine_reviews");
    expect(persisted).not.toContain("invocation_id");
    view.unmount();
  });

  it("keeps rag-only requests on the blocking endpoint", async () => {
    useChatStore.setState({ agentMode: "rag_only" });
    mockSendChatMessage.mockResolvedValue({
      answer: "RAG answer",
      sources: [],
      web_sources: [],
      agent_mode: "rag_only",
      model_provider: "cloud",
      error: null,
      trace_id: null,
      thread_id: "123e4567-e89b-42d3-a456-426614174000",
    });
    render(<ChatInterface />);

    await userEvent.type(screen.getByRole("textbox"), "Book question");
    await userEvent.click(screen.getByRole("button", { name: "Send message" }));

    expect(await screen.findByText("RAG answer")).toBeInTheDocument();
    expect(mockSendChatMessage).toHaveBeenCalledOnce();
    expect(mockStreamChatMessage).not.toHaveBeenCalled();
  });

  it("offers a deliberate retry for a known failed append", async () => {
    mockStreamChatMessage.mockResolvedValueOnce({
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

    mockStreamChatMessage.mockResolvedValueOnce({
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
