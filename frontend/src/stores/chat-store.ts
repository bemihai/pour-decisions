/**
 * Zustand store for chat page state.
 *
 * Replaces st.session_state usage in src/ui/pages/chatbot.py.
 * Messages and agentMode are persisted to localStorage; isLoading is transient.
 *
 * Hydration: ChatInterface uses useChatStore.persist.onFinishHydration() to
 * wait for localStorage data before rendering, preventing SSR mismatches.
 */
import { create } from "zustand";
import { persist } from "zustand/middleware";

import type { AgentMode, ModelProvider, Source, WebSource } from "@/lib/types";

/** Maximum number of messages persisted to localStorage. */
const MAX_MESSAGES = 200;

export interface Message {
  role: "human" | "ai";
  content: string;
  sources?: Source[];
  webSources?: WebSource[];
  /** Which agent mode produced this message (shown as a badge on AI bubbles). */
  agentMode?: AgentMode;
  /** Which model provider produced this message (shown as a badge on AI bubbles). */
  modelProvider?: ModelProvider;
  /** True when this message represents an error rather than an LLM response. */
  isError?: boolean;
}

interface ChatState {
  messages: Message[];
  agentMode: AgentMode;
  modelProvider: ModelProvider;
  isLoading: boolean;
}

interface ChatActions {
  addMessage: (message: Message) => void;
  setAgentMode: (mode: AgentMode) => void;
  setModelProvider: (provider: ModelProvider) => void;
  setLoading: (loading: boolean) => void;
  resetChat: () => void;
  /** Removes the last message if it is an AI message. Used by the Regenerate action. */
  deleteLastAiMessage: () => void;
}

const WELCOME_MESSAGE: Message = {
  role: "ai",
  content:
    "Welcome to Pour Decisions! I'm your wine assistant. Ask me anything about wine, your cellar, food pairings, or recommendations.",
};

const initialState: ChatState = {
  messages: [WELCOME_MESSAGE],
  agentMode: "intelligent",
  modelProvider: "local",
  isLoading: false,
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
      setAgentMode: (mode) => set({ agentMode: mode }),
      setModelProvider: (provider) => set({ modelProvider: provider }),
      setLoading: (loading) => set({ isLoading: loading }),
      resetChat: () => set({ messages: [WELCOME_MESSAGE], isLoading: false }),
      deleteLastAiMessage: () =>
        set((state) => ({
          messages:
            state.messages.length > 0 &&
            state.messages[state.messages.length - 1].role === "ai"
              ? state.messages.slice(0, -1)
              : state.messages,
        })),
    }),
    {
      name: "pour-decisions-chat",
      partialize: (state) => ({
        messages: state.messages,
        agentMode: state.agentMode,
        modelProvider: state.modelProvider,
      }),
    },
  ),
);
