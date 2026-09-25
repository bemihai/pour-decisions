import { afterEach, describe, expect, it, vi } from "vitest";

import {
  ApiError,
  ChatStreamError,
  streamChatMessage,
} from "@/lib/api";
import type { ChatResponse } from "@/lib/types";


const RESPONSE: ChatResponse = {
  answer: "Try Grüner Veltliner 🍷",
  sources: [],
  web_sources: [],
  agent_mode: "intelligent",
  model_provider: "cloud",
  error: null,
  trace_id: null,
  thread_id: "thread-1",
};

function eventFrame(type: string, payload: unknown): string {
  return `event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`;
}

function responseFromChunks(chunks: Uint8Array[], onCancel?: () => void): Response {
  return new Response(
    new ReadableStream<Uint8Array>({
      start(controller) {
        for (const chunk of chunks) controller.enqueue(chunk);
        controller.close();
      },
      cancel: onCancel,
    }),
    { headers: { "Content-Type": "text/event-stream; charset=utf-8" } },
  );
}

function encodedChunks(value: string, splitAt: number[]): Uint8Array[] {
  const encoded = new TextEncoder().encode(value);
  const chunks: Uint8Array[] = [];
  let start = 0;
  for (const end of splitAt) {
    chunks.push(encoded.slice(start, end));
    start = end;
  }
  chunks.push(encoded.slice(start));
  return chunks;
}

describe("streamChatMessage", () => {
  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("decodes split UTF-8 and SSE frames and handles multiple frames per chunk", async () => {
    const progress = vi.fn();
    const body =
      eventFrame("tool_progress", {
        type: "tool_progress",
        invocation_id: 1,
        tool_key: "search_wine_knowledge",
        status: "started",
      }) +
      eventFrame("tool_progress", {
        type: "tool_progress",
        invocation_id: 1,
        tool_key: "search_wine_knowledge",
        status: "completed",
      }) +
      eventFrame("agent_done", { type: "agent_done", response: RESPONSE });
    const emojiIndex = new TextEncoder().encode(body.slice(0, body.indexOf("🍷"))).length + 1;
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        responseFromChunks(encodedChunks(body, [7, 31, emojiIndex, emojiIndex + 1])),
      ),
    );

    await expect(
      streamChatMessage(
        { message: "Pair this", agent_mode: "intelligent" },
        { onProgress: progress },
      ),
    ).resolves.toEqual(RESPONSE);
    expect(progress).toHaveBeenCalledTimes(2);
    expect(progress).toHaveBeenLastCalledWith(
      expect.objectContaining({ status: "completed", invocation_id: 1 }),
    );
  });

  it("rejects malformed payloads and cancels the owned reader", async () => {
    const cancelled = vi.fn();
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        new Response(
          new ReadableStream<Uint8Array>({
            start(controller) {
              controller.enqueue(
                new TextEncoder().encode("event: tool_progress\ndata: {bad json}\n\n"),
              );
            },
            cancel: cancelled,
          }),
          { headers: { "Content-Type": "text/event-stream" } },
        ),
      ),
    );

    await expect(streamChatMessage({ message: "Hello" })).rejects.toBeInstanceOf(
      ChatStreamError,
    );
    expect(cancelled).toHaveBeenCalledOnce();
  });

  it("rejects a terminal response that reports an unsupported local model", async () => {
    const localResponse = { ...RESPONSE, model_provider: "local" };
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        responseFromChunks([
          new TextEncoder().encode(
            eventFrame("agent_done", { type: "agent_done", response: localResponse }),
          ),
        ]),
      ),
    );

    await expect(streamChatMessage({ message: "Hello" })).rejects.toBeInstanceOf(ChatStreamError);
  });

  it("rejects EOF without a terminal event as uncertain", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        responseFromChunks([
          new TextEncoder().encode(
            eventFrame("tool_progress", {
              type: "tool_progress",
              invocation_id: 1,
              tool_key: "other",
              status: "started",
            }),
          ),
        ]),
      ),
    );

    await expect(streamChatMessage({ message: "Hello" })).rejects.toMatchObject({
      name: "ChatStreamError",
      outcome: "uncertain",
    });
  });

  it("surfaces the fixed terminal stream error as uncertain", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(
        responseFromChunks([
          new TextEncoder().encode(
            eventFrame("stream_error", {
              type: "stream_error",
              message:
                "The request outcome is uncertain because the streaming response could not be completed.",
              outcome: "uncertain",
            }),
          ),
        ]),
      ),
    );

    await expect(streamChatMessage({ message: "Hello" })).rejects.toMatchObject({
      message:
        "The request outcome is uncertain because the streaming response could not be completed.",
      outcome: "uncertain",
    });
  });

  it("falls back once only for the explicit disabled preflight response", async () => {
    const fetch = vi
      .fn()
      .mockResolvedValueOnce(
        new Response(
          JSON.stringify({
            detail: { code: "streaming_disabled", message: "Streaming is disabled" },
          }),
          { status: 404, headers: { "Content-Type": "application/json" } },
        ),
      )
      .mockResolvedValueOnce(
        new Response(JSON.stringify(RESPONSE), {
          status: 200,
          headers: { "Content-Type": "application/json" },
        }),
      );
    vi.stubGlobal("fetch", fetch);

    await expect(streamChatMessage({ message: "Hello" })).resolves.toEqual(RESPONSE);
    expect(fetch).toHaveBeenCalledTimes(2);
    expect(fetch.mock.calls[0][0]).toBe("http://localhost:8000/api/chat/stream");
    expect(fetch.mock.calls[1][0]).toBe("http://localhost:8000/api/chat");
  });

  it("never falls back for a generic preflight error", async () => {
    const fetch = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ detail: "Not found" }), {
        status: 404,
        headers: { "Content-Type": "application/json" },
      }),
    );
    vi.stubGlobal("fetch", fetch);

    await expect(streamChatMessage({ message: "Hello" })).rejects.toEqual(
      new ApiError(404, "Not found"),
    );
    expect(fetch).toHaveBeenCalledOnce();
  });

  it("passes abort through without retrying", async () => {
    const controller = new AbortController();
    controller.abort();
    const abortError = new DOMException("Aborted", "AbortError");
    const fetch = vi.fn().mockRejectedValue(abortError);
    vi.stubGlobal("fetch", fetch);

    await expect(
      streamChatMessage({ message: "Hello" }, { signal: controller.signal }),
    ).rejects.toBe(abortError);
    expect(fetch).toHaveBeenCalledOnce();
  });
});
