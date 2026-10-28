/** Browser identity and legacy-view state for server-owned saved conversations. */
import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { Message } from "@/stores/chat-store";
import type { AgentMode } from "@/lib/types";

export const CONVERSATION_STORE_KEY = "pour-decisions-conversations";
export const LEGACY_CHAT_STORE_KEY = "pour-decisions-chat";
const CONVERSATION_STORE_VERSION = 1;

interface ConversationBrowserState {
  selectedConversationId: string | null;
  selectedMode: AgentMode;
}

interface ConversationBrowserActions {
  beginDraft: (mode?: AgentMode) => void;
  selectConversation: (id: string, mode: AgentMode) => void;
  setSelectedMode: (mode: AgentMode) => void;
}

export interface LegacyBrowserConversation {
  messages: Message[];
  agentMode: AgentMode;
  hasThreadId: boolean;
}

function isAgentMode(value: unknown): value is AgentMode {
  return value === "intelligent" || value === "rag_only";
}

function isLegacyMessage(value: unknown): value is Message {
  if (typeof value !== "object" || value === null || Array.isArray(value)) return false;
  const candidate = value as Partial<Message>;
  return (
    (candidate.role === "human" || candidate.role === "ai") &&
    typeof candidate.content === "string" &&
    candidate.content.length > 0
  );
}

/** Read the old browser transcript without treating it as server-owned history. */
export function readLegacyBrowserConversation(
  storage: Pick<Storage, "getItem"> = localStorage,
): LegacyBrowserConversation | null {
  const raw = storage.getItem(LEGACY_CHAT_STORE_KEY);
  if (raw === null) return null;
  try {
    const parsed = JSON.parse(raw) as unknown;
    if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) return null;
    const state = (parsed as { state?: unknown }).state;
    if (typeof state !== "object" || state === null || Array.isArray(state)) return null;
    const candidate = state as { messages?: unknown; agentMode?: unknown; threadId?: unknown };
    if (!Array.isArray(candidate.messages) || !candidate.messages.every(isLegacyMessage)) return null;
    if (!candidate.messages.some((message) => message.role === "human")) return null;
    return {
      messages: candidate.messages,
      agentMode: isAgentMode(candidate.agentMode) ? candidate.agentMode : "intelligent",
      hasThreadId: typeof candidate.threadId === "string" && candidate.threadId.length > 0,
    };
  } catch {
    return null;
  }
}

/** Explicitly dismiss only the old browser transcript. */
export function dismissLegacyBrowserConversation(
  storage: Pick<Storage, "removeItem"> = localStorage,
): void {
  storage.removeItem(LEGACY_CHAT_STORE_KEY);
}

export const useConversationStore = create<ConversationBrowserState & ConversationBrowserActions>()(
  persist(
    (set) => ({
      selectedConversationId: null,
      selectedMode: "intelligent",
      beginDraft: (mode) =>
        set((state) => ({
          selectedConversationId: null,
          selectedMode: mode ?? state.selectedMode,
        })),
      selectConversation: (id, mode) =>
        set({ selectedConversationId: id, selectedMode: mode }),
      setSelectedMode: (mode) => set({ selectedMode: mode }),
    }),
    {
      name: CONVERSATION_STORE_KEY,
      version: CONVERSATION_STORE_VERSION,
      partialize: (state) => ({
        selectedConversationId: state.selectedConversationId,
        selectedMode: state.selectedMode,
      }),
    },
  ),
);
