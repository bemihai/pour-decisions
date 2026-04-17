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

import { useRef, useEffect, useState, useCallback, type KeyboardEvent } from "react";
import { ChevronDown, Send, Sparkles } from "lucide-react";

import ChatMessage from "@/components/ChatMessage";
import LogoMark from "@/components/LogoMark";
import { sendChatMessage } from "@/lib/api";
import { useChatStore } from "@/stores/chat-store";
import { cn } from "@/lib/utils";

// ---------------------------------------------------------------------------
// Starter prompts
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
        className="flex h-10 w-10 shrink-0 items-center justify-center rounded-full bg-brand-burgundy/10"
        aria-hidden
      >
        <LogoMark size={20} className="text-brand-burgundy" title="" />
      </div>
      <div className="flex flex-col gap-2 max-w-[70%]">
        <div className="bg-chat-ai rounded-[20px_20px_20px_4px] px-4 py-3 shadow-sm">
          <div className="flex items-center gap-2">
            <div className="flex gap-1" role="status" aria-label="Thinking">
              <span className="h-2 w-2 rounded-full bg-brand-burgundy animate-bounce [animation-delay:0ms]" />
              <span className="h-2 w-2 rounded-full bg-brand-burgundy animate-bounce [animation-delay:150ms]" />
              <span className="h-2 w-2 rounded-full bg-brand-burgundy animate-bounce [animation-delay:300ms]" />
            </div>
            <span className="text-xs text-muted-foreground">Thinking...</span>
          </div>
        </div>
        <div className="bg-chat-ai rounded-[20px_20px_20px_4px] px-4 py-3 shadow-sm space-y-2">
          <div className="h-3 w-full rounded bg-foreground/10 animate-pulse" />
          <div className="h-3 w-5/6 rounded bg-foreground/10 animate-pulse [animation-delay:100ms]" />
          <div className="h-3 w-3/4 rounded bg-foreground/10 animate-pulse [animation-delay:200ms]" />
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Empty-state starter prompts
// ---------------------------------------------------------------------------

function StarterPrompts({ onSelect }: { onSelect: (prompt: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center gap-4 py-8">
      <Sparkles className="h-8 w-8 text-brand-burgundy opacity-50" />
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
              "text-muted-foreground hover:text-foreground hover:border-brand-burgundy/50",
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
// Hydration skeleton
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
    deleteLastAiMessage,
  } = useChatStore();
  const [input, setInput] = useState("");
  const [isHydrated, setIsHydrated] = useState(false);
  const [showScrollBtn, setShowScrollBtn] = useState(false);

  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  // A1: Wait for Zustand persist to finish loading from localStorage.
  useEffect(() => {
    if (useChatStore.persist.hasHydrated()) {
      setIsHydrated(true);
      return;
    }
    return useChatStore.persist.onFinishHydration(() => setIsHydrated(true));
  }, []);

  // Auto-scroll to latest message.
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  // Auto-grow the textarea (1–4 rows).
  useEffect(() => {
    const ta = textareaRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = `${Math.min(ta.scrollHeight, 120)}px`;
  }, [input]);

  // Track scroll position to show/hide the scroll-to-bottom FAB.
  useEffect(() => {
    const el = scrollContainerRef.current;
    if (!el) return;
    const handleScroll = () => {
      const { scrollTop, scrollHeight, clientHeight } = el;
      setShowScrollBtn(scrollHeight - scrollTop - clientHeight > 120);
    };
    el.addEventListener("scroll", handleScroll, { passive: true });
    return () => el.removeEventListener("scroll", handleScroll);
  }, []);

  // 4D.6: "/" key focuses the textarea from anywhere on the page.
  useEffect(() => {
    function handleGlobalKeyDown(e: globalThis.KeyboardEvent) {
      if (
        e.key === "/" &&
        !e.ctrlKey &&
        !e.metaKey &&
        !(e.target instanceof HTMLInputElement) &&
        !(e.target instanceof HTMLTextAreaElement)
      ) {
        e.preventDefault();
        textareaRef.current?.focus();
      }
    }
    document.addEventListener("keydown", handleGlobalKeyDown);
    return () => document.removeEventListener("keydown", handleGlobalKeyDown);
  }, []);

  const submitMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || isLoading) return;

      setInput("");
      addMessage({ role: "human", content: text.trim() });
      setLoading(true);

      try {
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
        textareaRef.current?.focus();
      }
    },
    [agentMode, isLoading, addMessage, setLoading],
  );

  // 4D.2: Regenerate — remove last AI message and re-query without adding a new
  // human message so the transcript doesn't contain duplicate prompts.
  const handleRegenerate = useCallback(async () => {
    if (isLoading) return;
    const msgs = useChatStore.getState().messages;
    const lastHuman = [...msgs].reverse().find((m) => m.role === "human");
    if (!lastHuman) return;

    deleteLastAiMessage();
    setLoading(true);

    try {
      // Read messages after the AI message was removed.
      const currentMessages = useChatStore.getState().messages;
      const history = currentMessages.map((m) => ({ role: m.role, content: m.content }));

      const response = await sendChatMessage({
        message: lastHuman.content,
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
      textareaRef.current?.focus();
    }
  }, [agentMode, isLoading, addMessage, setLoading, deleteLastAiMessage]);

  // 4D.1: Enter to send, Shift+Enter for newline.
  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submitMessage(input);
    }
  }

  function scrollToBottom() {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }

  if (!isHydrated) {
    return <ChatSkeleton />;
  }

  const showStarterPrompts = messages.length === 1 && !isLoading;

  return (
    <div className="flex flex-col h-full relative">
      {/* Scrollable message list */}
      <div
        ref={scrollContainerRef}
        className="flex-1 overflow-y-auto px-6 py-4"
        role="log"
        aria-label="Chat messages"
        aria-live="polite"
      >
        <div className="max-w-5xl mx-auto">
          {messages.map((msg, i) => {
            const isLastMsg = i === messages.length - 1;
            const isLastAi = msg.role === "ai" && isLastMsg && !msg.isError;
            return (
              <ChatMessage
                key={i}
                role={msg.role}
                content={msg.content}
                sources={msg.sources}
                webSources={msg.webSources}
                agentMode={msg.agentMode}
                isError={msg.isError}
                onRegenerate={isLastAi && !isLoading ? handleRegenerate : undefined}
              />
            );
          })}
          {showStarterPrompts && <StarterPrompts onSelect={submitMessage} />}
          {isLoading && <ThinkingIndicator />}
          <div ref={bottomRef} />
        </div>
      </div>

      {/* 4D.3: Scroll-to-bottom FAB */}
      {showScrollBtn && (
        <button
          onClick={scrollToBottom}
          aria-label="Scroll to latest message"
          className={cn(
            "absolute bottom-20 right-8 z-10",
            "flex h-9 w-9 items-center justify-center rounded-full",
            "bg-brand-burgundy text-white shadow-lg",
            "hover:bg-brand-burgundy-dark transition-colors",
            "focus:outline-none focus:ring-2 focus:ring-brand-burgundy focus:ring-offset-2",
          )}
        >
          <ChevronDown className="h-4 w-4" />
        </button>
      )}

      {/* Input bar */}
      <div className="shrink-0 border-t border-border bg-background/95 backdrop-blur supports-[backdrop-filter]:bg-background/60 px-6 py-3">
        <form
          onSubmit={(e) => { e.preventDefault(); submitMessage(input); }}
          className="flex items-end gap-2 max-w-5xl mx-auto"
        >
          {/* 4D.1: Auto-growing textarea */}
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Ask about wine, your cellar, pairings… (Enter to send, Shift+Enter for new line)"
            disabled={isLoading}
            autoFocus
            rows={1}
            style={{ resize: "none", overflowY: "hidden" }}
            className={cn(
              "flex-1 rounded-2xl border border-input bg-background px-4 py-2.5 text-sm",
              "placeholder:text-muted-foreground leading-5",
              "focus:outline-none focus:ring-2 focus:ring-brand-burgundy focus:border-transparent",
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
              "bg-brand-burgundy text-white",
              "hover:bg-brand-burgundy-dark transition-colors",
              "disabled:opacity-50 disabled:cursor-not-allowed",
              "focus:outline-none focus:ring-2 focus:ring-brand-burgundy focus:ring-offset-2",
            )}
          >
            <Send className="h-4 w-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
