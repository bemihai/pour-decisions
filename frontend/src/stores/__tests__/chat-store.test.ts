import { beforeEach, describe, expect, it } from "vitest";

import { migrateChatState, useChatStore } from "@/stores/chat-store";


const LEGACY_MESSAGE = { role: "human" as const, content: "Legacy transcript" };


describe("chat store persistence", () => {
  beforeEach(() => {
    localStorage.clear();
    useChatStore.setState({
      messages: [{ role: "ai", content: "Welcome" }],
      agentMode: "intelligent",
      threadId: "existing-thread",
      isLoading: false,
    });
  });

  it("migrates the pre-M8 shape to a fresh thread and welcome transcript", () => {
    const migrated = migrateChatState(
      { messages: [LEGACY_MESSAGE], agentMode: "rag_only" },
      0,
      () => "new-migrated-thread",
    );

    expect(migrated.threadId).toBe("new-migrated-thread");
    expect(migrated.agentMode).toBe("rag_only");
    expect(migrated.messages).toHaveLength(1);
    expect(migrated.messages[0]).toMatchObject({ role: "ai" });
    expect(migrated.messages[0].content).not.toBe(LEGACY_MESSAGE.content);
  });

  it("preserves a valid current-version persisted shape", () => {
    const persisted = {
      messages: [LEGACY_MESSAGE],
      agentMode: "intelligent" as const,
      threadId: "persisted-thread",
    };

    expect(migrateChatState(persisted, 1)).toEqual(persisted);
  });

  it("creates a new UUID whenever the local conversation is reset", () => {
    const previousThreadId = useChatStore.getState().threadId;

    useChatStore.getState().resetChat();

    const state = useChatStore.getState();
    expect(state.threadId).not.toBe(previousThreadId);
    expect(state.threadId).toMatch(
      /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i,
    );
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0].role).toBe("ai");
  });

  it("rehydrates version 0 storage through the configured migration", async () => {
    localStorage.setItem(
      "pour-decisions-chat",
      JSON.stringify({
        state: { messages: [LEGACY_MESSAGE], agentMode: "rag_only" },
        version: 0,
      }),
    );

    await useChatStore.persist.rehydrate();

    const state = useChatStore.getState();
    expect(state.threadId).not.toBe("existing-thread");
    expect(state.agentMode).toBe("rag_only");
    expect(state.messages).toHaveLength(1);
    expect(state.messages[0].role).toBe("ai");
  });

  it("rehydrates current-version storage without changing its conversation", async () => {
    localStorage.setItem(
      "pour-decisions-chat",
      JSON.stringify({
        state: {
          messages: [LEGACY_MESSAGE],
          agentMode: "intelligent",
          threadId: "persisted-current-thread",
        },
        version: 1,
      }),
    );

    await useChatStore.persist.rehydrate();

    expect(useChatStore.getState()).toMatchObject({
      messages: [LEGACY_MESSAGE],
      agentMode: "intelligent",
      threadId: "persisted-current-thread",
    });
  });
});
