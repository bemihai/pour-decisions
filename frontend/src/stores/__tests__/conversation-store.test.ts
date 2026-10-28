import { beforeEach, describe, expect, it } from "vitest";

import {
  CONVERSATION_STORE_KEY,
  dismissLegacyBrowserConversation,
  readLegacyBrowserConversation,
  useConversationStore,
} from "@/stores/conversation-store";

const HUMAN_MESSAGE = { role: "human" as const, content: "/compare one two" };

describe("saved conversation browser state", () => {
  beforeEach(() => {
    localStorage.clear();
    useConversationStore.setState({
      selectedConversationId: null,
      selectedMode: "intelligent",
    });
  });

  it("persists only selected identity and fixed draft mode under its own key", () => {
    useConversationStore.getState().selectConversation("conversation-a", "rag_only");

    const persisted = JSON.parse(localStorage.getItem(CONVERSATION_STORE_KEY) ?? "{}");
    expect(persisted).toEqual({
      state: { selectedConversationId: "conversation-a", selectedMode: "rag_only" },
      version: 1,
    });
  });

  it("starts a draft without deleting or inventing a server conversation", () => {
    useConversationStore.getState().selectConversation("conversation-a", "intelligent");
    useConversationStore.getState().beginDraft("rag_only");

    expect(useConversationStore.getState()).toMatchObject({
      selectedConversationId: null,
      selectedMode: "rag_only",
    });
  });
});

describe("legacy browser transcript", () => {
  it("reads intact and truncated browser-only messages without requiring a thread", () => {
    const legacy = readLegacyBrowserConversation({
      getItem: () =>
        JSON.stringify({
          state: { messages: [HUMAN_MESSAGE], agentMode: "rag_only" },
          version: 1,
        }),
    });

    expect(legacy).toEqual({
      messages: [HUMAN_MESSAGE],
      agentMode: "rag_only",
      hasThreadId: false,
    });
  });

  it.each([
    ["malformed", "not json"],
    ["missing messages", JSON.stringify({ state: { agentMode: "intelligent" } })],
    ["empty", JSON.stringify({ state: { messages: [] } })],
    ["welcome only", JSON.stringify({ state: { messages: [{ role: "ai", content: "Welcome" }] } })],
  ])("does not invent a legacy view for %s state", (_label, raw) => {
    expect(readLegacyBrowserConversation({ getItem: () => raw })).toBeNull();
  });

  it("dismisses only the legacy storage key", () => {
    localStorage.setItem("pour-decisions-chat", "legacy");
    localStorage.setItem(CONVERSATION_STORE_KEY, "saved-selection");

    dismissLegacyBrowserConversation();

    expect(localStorage.getItem("pour-decisions-chat")).toBeNull();
    expect(localStorage.getItem(CONVERSATION_STORE_KEY)).toBe("saved-selection");
  });
});
