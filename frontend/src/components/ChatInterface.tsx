/**
 * ChatInterface component.
 *
 * Main chat interaction area: scrollable message list, loading indicator,
 * and sticky input bar. Replaces the entire message loop and input form from
 * src/ui/pages/chatbot.py (~250 lines of Streamlit session_state logic).
 *
 * Client Component — requires "use client" for hooks and event handlers.
 */
"use client";

import { useRef, useEffect, useState, useCallback, type FormEvent } from "react";
import { Send, Sparkles } from "lucide-react";

import ChatMessage from "@/components/ChatMessage";
import { sendChatMessage } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Starter prompts (B1)
// ---------------------------------------------------------------------------

const STARTER_PROMPTS = [
  "What wine pairs well with grilled salmon?",
  "Show me my cellar stats",
  "Recommend a Pinot Noir under $30",
  "What are the best wine regions in France?",
];

// ---------------------------------------------------------------------------
// Loading indicator
// ---------------------------------------------------------------------------

function ThinkingIndicator() {
  return (
    <div className="flex items-start justify-start gap-2 mb-4">
      <div
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-chat-ai text-xl select-none"
        aria-hidden
      >
        🍇
      </div>
      <div className="bg-chat-ai rounded-[20px_20px_20px_4px] px-4 py-3 shadow-sm">
        <div className="flex items-center gap-2">
          <div className="flex gap-1" role="status" aria-label="Thinking">
            <span className="h-2 w-2 rounded-full bg-brand-purple animate-bounce [animation-delay:0ms]" />
            <span className="h-2 w-2 rounded-full bg-brand-purple animate-bounce [animation-delay:150ms]" />
            <span className="h-2 w-2 rounded-full bg-brand-purple animate-bounce [animation-delay:300ms]" />
          </div>
          <span className="text-xs text-muted-foreground">Thinking...</span>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty-state starter prompts (B1)
// ---------------------------------------------------------------------------

function StarterPrompts({ onSelect }: { onSelect: (prompt: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-8">
      <Sparkles className="h-8 w-8 text-brand-purple opacity-50" />
      <p className="text-sm text-muted-foreground text-center max-w-md">
        Try one of these to get started:
      </p>
      <div className="flex flex-wrap justify-center gap-2 max-w-lg">
        {STARTER_PROMPTS.map((prompt) => (
          <button
            key={prompt}
            onClick={() => onSelect(prompt)}
            className={cn(
              "rounded-full border border-border bg-background px-4 py-2 text-xs",
              "text-muted-foreground hover:text-foreground hover:border-brand-purple/50",
              "hover:bg-muted/50 transition-colors",
            )}
          >
            {prompt}
          </button>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Hydration skeleton (A1)
// ---------------------------------------------------------------------------

function ChatSkeleton() {
  return (
    <div className="flex flex-col h-full items-center justify-center gap-3">
      <div className="h-10 w-10 rounded-full bg-muted animate-pulse" />
      <div className="h-4 w-48 rounded bg-muted animate-pulse" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Public component
// ---------------------------------------------------------------------------

export default function ChatInterface() {
  const {
    messages,
    agentMode,
    isLoading,
    addMessage,
    setLoading,
  } = useChatStore();
  const [input, setInput] = useState("");
  const [isHydrated, setIsHydrated] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // A1: Wait for Zustand persist to finish loading from localStorage.
  // This runs only on the client after mount, preventing SSR hydration mismatches.
  useEffect(() => {
    // Already done (e.g. fast localStorage read finished before mount).
    if (useChatStore.persist.hasHydrated()) {
      setIsHydrated(true);
      return;
    }
    // Otherwise subscribe and wait.
    const unsub = useChatStore.persist.onFinishHydration(() => {
      setIsHydrated(true);
    });
    return unsub;
  }, []);

  // Scroll to the latest message whenever messages change or loading toggles.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const submitMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      setInput("");
      addMessage({ role: "human", content: text.trim() });
      setLoading(true);

      try {
        // C2: read fresh history from the store (not the stale closure).
        const currentMessages = useChatStore.getState().messages;
        const history = currentMessages.map((m) => ({ role: m.role, content: m.content }));

        const response = await sendChatMessage({
          message: text.trim(),
          agent_mode: agentMode,
          message_history: history,
        });

        addMessage({
          role: "ai",
          content: response.answer,
          sources: response.sources,
          webSources: response.web_sources,
          agentMode: response.agent_mode,
        });
      } catch (err) {
        const detail = err instanceof Error ? err.message : "An unexpected error occurred.";
        addMessage({
          role: "ai",
          content: `Sorry, something went wrong: ${detail}`,
          isError: true,
        });
      } finally {
        setLoading(false);
        inputRef.current?.focus();
      }
    },
    [agentMode, isLoading, addMessage, setLoading],
  );

  function handleSubmit(e: FormEvent) {
    e.preventDefault();
    submitMessage(input);
  }

  // A1: show skeleton until localStorage has been loaded.
  if (!isHydrated) {
    return <ChatSkeleton />;
  }

  const showStarterPrompts = messages.length === 1 && !isLoading;

  return (
    <div className="flex flex-col h-full">
      {/* D1: Scrollable message list with ARIA log role */}
      <div
        className="flex-1 overflow-y-auto px-6 py-4"
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
      >
        <div className="max-w-5xl mx-auto">
          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              role={msg.role}
              content={msg.content}
              sources={msg.sources}
              webSources={msg.webSources}
              agentMode={msg.agentMode}
              isError={msg.isError}
            />
          ))}
          {showStarterPrompts && <StarterPrompts onSelect={submitMessage} />}
          {isLoading && <ThinkingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* Input bar — pinned to the bottom by the flex column layout */}
      <div className="shrink-0 border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6 py-3">
        <form
          onSubmit={handleSubmit}
          className="flex items-center gap-2 max-w-5xl mx-auto"
        >
          <input
            ref={inputRef}
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about wine, your cellar, pairings..."
            disabled={isLoading}
            autoFocus
            className={cn(
              "flex-1 rounded-full border border-input bg-background px-4 py-2.5 text-sm",
              "placeholder:text-muted-foreground",
              "focus:outline-none focus:ring-2 focus:ring-brand-purple focus:border-transparent",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "transition-shadow",
            )}
          />
          <button
            type="submit"
            disabled={isLoading || !input.trim()}
            aria-label="Send message"
            className={cn(
              "flex h-10 w-10 shrink-0 items-center justify-center rounded-full",
              "bg-brand-purple text-white",
              "hover:bg-brand-purple-dark transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "focus:outline-none focus:ring-2 focus:ring-brand-purple focus:ring-offset-2",
            )}
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
