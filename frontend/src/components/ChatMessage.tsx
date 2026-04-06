/**
 * ChatMessage component.
 *
 * Renders a single chat bubble for either a human or AI turn.
 * Replaces format_user_message(), format_assistant_message(), display_message(),
 * and the CONTENT_STYLE CSS block from src/ui/helper/display.py.
 *
 * Human messages: right-aligned, green bubble, user avatar on the right.
 * AI messages:    left-aligned, purple bubble, grape avatar on the left.
 * Error messages: left-aligned, red-tinted bubble, warning icon.
 */
"use client";

import { memo } from "react";
import ReactMarkdown from "react-markdown";
import { AlertTriangle } from "lucide-react";

import SourceList from "@/components/SourceList";
import { cn } from "@/lib/utils";
import type { AgentMode, Source, WebSource } from "@/lib/types";

// ---------------------------------------------------------------------------
// Shared bubble config
// ---------------------------------------------------------------------------

const BUBBLE_BASE = "max-w-[70%] px-4 py-3 text-sm leading-relaxed break-words shadow-sm";

const USER_BUBBLE = cn(
  BUBBLE_BASE,
  "bg-chat-user text-foreground rounded-[20px_20px_4px_20px]",
);

const AI_BUBBLE = cn(
  BUBBLE_BASE,
  "bg-chat-ai text-foreground rounded-[20px_20px_20px_4px]",
);

const ERROR_BUBBLE = cn(
  BUBBLE_BASE,
  "bg-chat-error text-foreground rounded-[20px_20px_20px_4px] border border-destructive/30",
);

const AVATAR_BASE = "flex h-10 w-10 shrink-0 items-center justify-center rounded-full text-xl select-none";

// ---------------------------------------------------------------------------
// Agent mode label map
// ---------------------------------------------------------------------------

const AGENT_LABELS: Record<AgentMode, string> = {
  intelligent: "Intelligent Agent",
  keyword: "Keyword Agent",
  rag_only: "RAG Only",
};

// ---------------------------------------------------------------------------
// Markdown renderer for AI content
// ---------------------------------------------------------------------------

const markdownComponents: React.ComponentProps<typeof ReactMarkdown>["components"] = {
  p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
  ul: ({ children }) => <ul className="list-disc list-inside mb-2 space-y-0.5">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal list-inside mb-2 space-y-0.5">{children}</ol>,
  li: ({ children }) => <li>{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  em: ({ children }) => <em className="italic">{children}</em>,
  code: ({ children }) => (
    <code className="bg-black/10 rounded px-1 py-0.5 text-xs font-mono">{children}</code>
  ),
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noopener noreferrer"
      className="underline underline-offset-2 hover:opacity-80"
    >
      {children}
    </a>
  ),
};

// ---------------------------------------------------------------------------
// Sub-renderers
// ---------------------------------------------------------------------------

function UserMessage({ content }: { content: string }) {
  return (
    <div className="flex items-end justify-end gap-2 mb-4">
      <div className={USER_BUBBLE}>{content}</div>
      <div className={cn(AVATAR_BASE, "bg-chat-user")} aria-hidden>
        🧑‍💼
      </div>
    </div>
  );
}

function AIMessage({
  content,
  sources,
  webSources,
  agentMode,
  isError,
}: {
  content: string;
  sources?: Source[];
  webSources?: WebSource[];
  agentMode?: AgentMode;
  isError?: boolean;
}) {
  const bubble = isError ? ERROR_BUBBLE : AI_BUBBLE;
  const avatarBg = isError ? "bg-chat-error" : "bg-chat-ai";
  const avatar = isError ? <AlertTriangle className="h-5 w-5 text-destructive" /> : "🍇";

  return (
    <div className="flex items-start justify-start gap-2 mb-4">
      <div className={cn(AVATAR_BASE, avatarBg)} aria-hidden>
        {avatar}
      </div>
      <div className="flex flex-col gap-1 max-w-[70%]">
        <div className={bubble}>
          <ReactMarkdown components={markdownComponents}>{content}</ReactMarkdown>
          {sources && sources.length > 0 && <SourceList sources={sources} />}
          {webSources && webSources.length > 0 && <SourceList sources={webSources} isWeb />}
        </div>
        {agentMode && (
          <span className="text-[10px] text-muted-foreground ml-1">
            {AGENT_LABELS[agentMode]}
          </span>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Public component — wrapped in React.memo (C1)
// ---------------------------------------------------------------------------

export interface ChatMessageProps {
  role: "human" | "ai";
  content: string;
  sources?: Source[];
  webSources?: WebSource[];
  agentMode?: AgentMode;
  isError?: boolean;
}

function ChatMessageInner({ role, content, sources, webSources, agentMode, isError }: ChatMessageProps) {
  if (role === "human") {
    return <UserMessage content={content} />;
  }
  return (
    <AIMessage
      content={content}
      sources={sources}
      webSources={webSources}
      agentMode={agentMode}
      isError={isError}
    />
  );
}

const ChatMessage = memo(ChatMessageInner);
export default ChatMessage;
