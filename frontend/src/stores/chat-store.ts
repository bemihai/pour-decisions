/**
 * Zustand store for chat page state.
 *
 * Replaces st.session_state usage in src/ui/pages/chatbot.py.
 * Messages, agentMode, and threadId are persisted to localStorage; loading and
 * lifecycle errors are transient.
 *
 * Hydration: ChatInterface uses useChatStore.persist.onFinishHydration() to
 * wait for localStorage data before rendering, preventing SSR mismatches.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { AgentMode, Source, WebSource } from "@/lib/types";

/** Maximum number of messages persisted to localStorage. */
const MAX_MESSAGES = 200;
const CHAT_STORE_VERSION = 1;

export interface Message {
  role: "human" | "ai";
  content: string;
  sources?: Source[];
  webSources?: WebSource[];
  /** Which agent mode produced this message (shown as a badge on AI bubbles). */
  agentMode?: AgentMode;
  /** True when this message represents an error rather than an LLM response. */
  isError?: boolean;
  /** True when a known failed append can be resubmitted safely. */
  isRetryable?: boolean;
}

interface ChatState {
  messages: Message[];
  agentMode: AgentMode;
  threadId: string;
  isLoading: boolean;
  conversationError: string | null;
}

interface ChatActions {
  addMessage: (message: Message) => void;
  setLoading: (loading: boolean) => void;
  setConversationError: (error: string | null) => void;
  resetChat: (mode?: AgentMode) => void;
  /** Replace the final AI message atomically after regenerate or retry. */
  replaceLastAiMessage: (message: Message) => void;
}

const WELCOME_MESSAGE: Message = {
  role: "ai",
  content:
    "Welcome to Pour Decisions! I'm your wine assistant. Ask me anything about wine, your cellar, food pairings, or recommendations.",
};

interface PersistedChatState {
  messages: Message[];
  agentMode: AgentMode;
  threadId: string;
}

/** Generate the opaque identifier used to synchronize one browser conversation. */
export function createThreadId(): string {
  return globalThis.crypto.randomUUID();
}

function isAgentMode(value: unknown): value is AgentMode {
  return value === "intelligent" || value === "rag_only";
}

/** Migrate persisted state without displaying transcripts unknown to the server. */
export function migrateChatState(
  persistedState: unknown,
  version: number,
  makeThreadId: () => string = createThreadId,
): PersistedChatState {
  const persisted = (persistedState ?? {}) as Partial<PersistedChatState>;
  const agentMode = isAgentMode(persisted.agentMode) ? persisted.agentMode : "intelligent";

  if (version < CHAT_STORE_VERSION) {
    return {
      messages: [WELCOME_MESSAGE],
      agentMode,
      threadId: makeThreadId(),
    };
  }

  return {
    messages: Array.isArray(persisted.messages) ? persisted.messages : [WELCOME_MESSAGE],
    agentMode,
    threadId:
      typeof persisted.threadId === "string" && persisted.threadId.length > 0
        ? persisted.threadId
        : makeThreadId(),
  };
}

const initialState: ChatState = {
  messages: [WELCOME_MESSAGE],
  agentMode: "intelligent",
  threadId: createThreadId(),
  isLoading: false,
  conversationError: null,
};

export const useChatStore = create<ChatState & ChatActions>()(
  persist(
    (set) => ({
      ...initialState,
      addMessage: (message) =>
        set((state) => {
          const next = [...state.messages, message];
          // C3: trim oldest messages (keep welcome) when exceeding the cap.
          if (next.length > MAX_MESSAGES) {
            return { messages: [next[0], ...next.slice(next.length - MAX_MESSAGES + 1)] };
          }
          return { messages: next };
        }),
      setLoading: (loading) => set({ isLoading: loading }),
      setConversationError: (error) => set({ conversationError: error }),
      resetChat: (mode) =>
        set((state) => ({
          messages: [WELCOME_MESSAGE],
          agentMode: mode ?? state.agentMode,
          threadId: createThreadId(),
          isLoading: false,
          conversationError: null,
        })),
      replaceLastAiMessage: (message) =>
        set((state) => ({
          messages:
            state.messages.length > 0 &&
            state.messages[state.messages.length - 1].role === "ai"
              ? [...state.messages.slice(0, -1), message]
              : state.messages,
        })),
    }),
    {
      name: "pour-decisions-chat",
      version: CHAT_STORE_VERSION,
      migrate: migrateChatState,
      partialize: (state) => ({
        messages: state.messages,
        agentMode: state.agentMode,
        threadId: state.threadId,
      }),
    },
  ),
);
