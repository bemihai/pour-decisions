/**
 * ChatMessage component.
 *
 * Renders a single chat bubble for either a human or AI turn.
 * Replaces format_user_message(), format_assistant_message(), display_message(),
 * and the CONTENT_STYLE CSS block from src/ui/helper/display.py.
 *
 * Human messages: right-aligned, warm bubble, user avatar on the right.
 * AI messages:    left-aligned, muted bubble, logo mark avatar on the left.
 * Error messages: left-aligned, red-tinted bubble, warning icon.
 */
"use client";

import { memo, useState } from "react";
import ReactMarkdown from "react-markdown";
import { AlertTriangle, Check, Copy, RotateCcw, User } from "lucide-react";

import LogoMark from "@/components/LogoMark";
import SourceList from "@/components/SourceList";
import { cn } from "@/lib/utils";
import type { AgentMode, ModelProvider, Source, WebSource } from "@/lib/types";

// ---------------------------------------------------------------------------
// Shared bubble config
// ---------------------------------------------------------------------------

const BUBBLE_BASE = "max-w-[70%] px-4 py-3 shadow-sm";

const USER_BUBBLE = cn(
  BUBBLE_BASE,
  "type-body bg-chat-user text-foreground rounded-[20px_20px_4px_20px]",
);

const AI_BUBBLE = cn(
  BUBBLE_BASE,
  "type-body bg-chat-ai text-foreground rounded-[20px_20px_20px_4px]",
);

const ERROR_BUBBLE = cn(
  BUBBLE_BASE,
  "type-body bg-chat-error text-foreground rounded-[20px_20px_20px_4px] border border-destructive/30",
);

const AVATAR_BASE = "flex h-10 w-10 shrink-0 items-center justify-center rounded-full select-none";


// ---------------------------------------------------------------------------
// Agent mode label map
// ---------------------------------------------------------------------------

const AGENT_LABELS: Record<AgentMode, string> = {
  intelligent: "Intelligent Agent",
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
      <div
        className={cn(AVATAR_BASE, "bg-brand-gold/20")}
        aria-hidden
      >
        <User className="h-5 w-5 text-brand-gold" />
      </div>
    </div>
  );
}

function AIMessage({
  content,
  sources,
  webSources,
  agentMode,
  modelProvider,
  isError,
  onRegenerate,
}: {
  content: string;
  sources?: Source[];
  webSources?: WebSource[];
  agentMode?: AgentMode;
  modelProvider?: ModelProvider;
  isError?: boolean;
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);

  const bubble = isError ? ERROR_BUBBLE : AI_BUBBLE;

  async function handleCopy() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard API failed; ignore silently
    }
  }

  return (
    <div className="flex items-start justify-start gap-2 mb-4">
      {/* Avatar — LogoMark SVG replaces grape emoji */}
      <div
        className={cn(AVATAR_BASE, isError ? "bg-destructive/10" : "bg-brand-burgundy/10")}
        aria-hidden
      >
        {isError ? (
          <AlertTriangle className="h-5 w-5 text-destructive" />
        ) : (
          <LogoMark size={20} className="text-brand-burgundy" title="" />
        )}
      </div>

      <div className="flex flex-col gap-1 max-w-[70%]">
        <div className="relative group">
          <div className={bubble}>
            <ReactMarkdown components={markdownComponents}>{content}</ReactMarkdown>
            {sources && sources.length > 0 && <SourceList sources={sources} />}
            {webSources && webSources.length > 0 && <SourceList sources={webSources} isWeb />}
          </div>

          {/* Action buttons — visible on hover */}
          {!isError && (
            <div className="absolute top-2 right-2 flex gap-1 opacity-0 group-hover:opacity-100 transition-opacity">
              {/* Regenerate */}
              {onRegenerate && (
                <button
                  onClick={onRegenerate}
                  aria-label="Regenerate response"
                  className={cn(
                    "flex h-7 w-7 items-center justify-center rounded-md",
                    "bg-background/80 backdrop-blur-sm border border-border shadow-sm",
                    "hover:bg-muted focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-brand-burgundy",
                  )}
                >
                  <RotateCcw className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
                </button>
              )}
              {/* Copy */}
              <button
                onClick={handleCopy}
                aria-label={copied ? "Copied" : "Copy to clipboard"}
                className={cn(
                  "flex h-7 w-7 items-center justify-center rounded-md",
                  "bg-background/80 backdrop-blur-sm border border-border shadow-sm",
                  "hover:bg-muted focus:opacity-100 focus:outline-none focus:ring-2 focus:ring-brand-burgundy",
                )}
              >
                {copied ? (
                  <Check className="h-3.5 w-3.5 text-green-600" aria-hidden="true" />
                ) : (
                  <Copy className="h-3.5 w-3.5 text-muted-foreground" aria-hidden="true" />
                )}
              </button>
            </div>
          )}
        </div>

        {agentMode && (
          <span className="type-caption text-muted-foreground ml-1">
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
  modelProvider?: ModelProvider;
  isError?: boolean;
  /** Called when the user clicks Regenerate on an AI bubble. */
  onRegenerate?: () => void;
}

function ChatMessageInner({
  role,
  content,
  sources,
  webSources,
  agentMode,
  modelProvider,
  isError,
  onRegenerate,
}: ChatMessageProps) {
  if (role === "human") {
    return <UserMessage content={content} />;
  }
  return (
    <AIMessage
      content={content}
      sources={sources}
      webSources={webSources}
      agentMode={agentMode}
      modelProvider={modelProvider}
      isError={isError}
      onRegenerate={onRegenerate}
    />
  );
}

const ChatMessage = memo(ChatMessageInner);
export default ChatMessage;
